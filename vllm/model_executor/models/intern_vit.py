# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Iterable
from functools import partial
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .vision import run_dp_sharded_vision_model

logger = init_logger(__name__)

NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': nn.LayerNorm,
}


def build_reuse_mask_from_embeddings(
    embeddings: torch.Tensor,
    prev_embeddings: torch.Tensor,
    *,
    recompute_ratio: float,
    cls_index: int = 0,
) -> torch.Tensor:
    """
    Build a boolean reuse mask from input embeddings.
    True means "reuse/skip", False means "recompute". CLS is always recompute.
    """
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
        prev_embeddings = prev_embeddings.unsqueeze(0)

    if embeddings.shape != prev_embeddings.shape:
        raise ValueError("Embeddings shape mismatch for reuse mask.")

    batch_size, num_tokens, _ = embeddings.shape
    if num_tokens <= 1:
        return torch.zeros_like(embeddings[:, :, 0], dtype=torch.bool)

    if not (0.0 <= recompute_ratio <= 1.0):
        raise ValueError("recompute_ratio must be within [0, 1].")

    token_mask = torch.ones(num_tokens,
                            dtype=torch.bool,
                            device=embeddings.device)
    token_mask[cls_index] = False
    curr = embeddings[:, token_mask, :]
    prev = prev_embeddings[:, token_mask, :]
    curr = F.normalize(curr, dim=-1)
    prev = F.normalize(prev, dim=-1)
    similarity = (curr * prev).sum(dim=-1)  # [B, N-1]
    debug = os.environ.get("VLLM_INTERNVL_REUSE_DEBUG", "0") == "1"

    recompute_count = int(round((num_tokens - 1) * recompute_ratio))
    recompute_count = max(0, min(num_tokens - 1, recompute_count))

    reuse_mask = torch.ones((batch_size, num_tokens),
                            dtype=torch.bool,
                            device=embeddings.device)
    reuse_mask[:, cls_index] = False

    if debug:
        sim0 = similarity[0]
        logger.info(
            "InternVL reuse similarity: tokens=%s cls_index=%s "
            "recompute_count=%s sim[min/mean/max]=%.4f/%.4f/%.4f",
            num_tokens,
            cls_index,
            recompute_count,
            float(sim0.min().item()),
            float(sim0.mean().item()),
            float(sim0.max().item()),
        )

    if recompute_count == 0:
        return reuse_mask

    _, recompute_idx = torch.topk(similarity,
                                  k=recompute_count,
                                  dim=-1,
                                  largest=False)
    # Shift indices if CLS is at position 0 (common case).
    if cls_index == 0:
        recompute_idx = recompute_idx + 1

    for batch_idx in range(batch_size):
        reuse_mask[batch_idx, recompute_idx[batch_idx]] = False

    return reuse_mask


def apply_block_sparse_layer(
    layer: "InternVisionEncoderLayer",
    hidden_states: torch.Tensor,
    *,
    reuse_mask: torch.Tensor,
    cache: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a block-sparse attention+FFN path with token reuse.

    Tokens with reuse_mask=True are copied from cache; others are recomputed.
    This is an aggressive approximation for experimental use.
    """
    if cache is None:
        return layer(hidden_states)

    if reuse_mask.ndim == 1:
        reuse_mask = reuse_mask.unsqueeze(0).expand(hidden_states.size(0), -1)

    if reuse_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("reuse_mask shape must match hidden_states.")

    batch_size, num_tokens, _ = hidden_states.shape
    outputs = cache.clone()

    for batch_idx in range(batch_size):
        recompute_mask = ~reuse_mask[batch_idx]
        if not torch.any(recompute_mask):
            continue

        recompute_idx = recompute_mask.nonzero(as_tuple=False).squeeze(-1)
        x_small = hidden_states[batch_idx:batch_idx + 1, recompute_idx, :]

        attn_out = layer.attn(layer.norm1(x_small)) * layer.ls1
        x_small = x_small + attn_out
        mlp_out = layer.mlp(layer.norm2(x_small)) * layer.ls2
        x_small = x_small + mlp_out

        outputs[batch_idx, recompute_idx, :] = x_small.squeeze(0)

    return outputs


class InternVisionEmbeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size,
            self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed,
                                  size=(H, W),
                                  mode='bicubic',
                                  align_corners=False)
        return pos_embed.reshape(1, -1, H * W).permute(0, 2,
                                                       1).to(target_dtype)

    def _get_position_embedding(self, H: int, W: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if self.num_patches == H * W:
            return position_embedding

        return torch.cat(
            [
                position_embedding[:, :1, :],
                self._get_pos_embed(position_embedding[:, 1:, :], H, W),
            ],
            dim=1,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            target_dtype))  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1,
                                                   -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = self._get_position_embedding(height, width)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternVisionPatchModel(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
        *,
        reuse_mask: Optional[torch.Tensor] = None,
        reuse_cache: Optional[list[Optional[torch.Tensor]]] = None,
        reuse_start_layer: int = 2,
        reuse_end_layer: Optional[int] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                'You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f'wrong pixel_values size: {pixel_values.shape}')

        return hidden_states


class InternParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        self.tp_size = (1 if use_data_parallel else
                        get_tensor_model_parallel_world_size())
        self.tp_rank = (0 if use_data_parallel else
                        get_tensor_model_parallel_rank())

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
        self.num_heads_per_partition = divide(num_dummy_heads + self.num_heads,
                                              self.tp_size)

        self.scale = self.head_dim**-0.5
        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            num_dummy_heads + self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)
            self.k_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)

        self.proj = RowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        out = self.attn(q, k, v)
        out, _ = self.proj(out)
        return out


class InternMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1",
                                        disable_tp=use_data_parallel)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2",
                                     disable_tp=use_data_parallel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class InternVisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = self._init_attn(config,
                                    quant_config,
                                    num_dummy_heads=num_dummy_heads,
                                    prefix=f"{prefix}.attn",
                                    use_data_parallel=use_data_parallel)

        self.mlp = InternMLP(config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp",
                             use_data_parallel=use_data_parallel)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))

    def _init_attn(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        num_dummy_heads: int,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        # fallback to sdpa attention if tp unavailable
        tp_size = (1 if use_data_parallel else
                   get_tensor_model_parallel_world_size())
        num_heads = config.num_attention_heads

        # if the number of heads is not divisible by tp_size,
        # we also disable Attention's TP
        use_data_parallel = (use_data_parallel
                             or (num_heads + num_dummy_heads) % tp_size != 0)
        return InternParallelAttention(config,
                                       quant_config=quant_config,
                                       num_dummy_heads=num_dummy_heads,
                                       prefix=prefix,
                                       use_data_parallel=use_data_parallel)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states)) * self.ls1

        hidden_states = hidden_states + self.mlp(
            self.norm2(hidden_states)) * self.ls2

        return hidden_states


class InternVisionEncoder(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config,
                                     quant_config,
                                     num_dummy_heads=num_dummy_heads,
                                     prefix=f"{prefix}.layers.{layer_idx}",
                                     use_data_parallel=use_data_parallel)
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        reuse_mask: Optional[torch.Tensor] = None,
        reuse_cache: Optional[list[Optional[torch.Tensor]]] = None,
        reuse_start_layer: int = 2,
        reuse_end_layer: Optional[int] = None,
    ):
        hidden_states = inputs_embeds
        if reuse_mask is None or reuse_cache is None:
            for encoder_layer in self.layers:
                hidden_states = encoder_layer(hidden_states)
            return hidden_states

        num_layers = len(self.layers)
        if reuse_end_layer is None:
            reuse_end_layer = max(num_layers - 1, 0)
        if len(reuse_cache) != num_layers:
            raise ValueError("reuse_cache length must match encoder layers.")
        debug = os.environ.get("VLLM_INTERNVL_REUSE_DEBUG", "0") == "1"
        if debug:
            logger.info(
                "InternVL reuse layers: start=%s end=%s (exclusive) total=%s",
                reuse_start_layer,
                reuse_end_layer,
                num_layers,
            )

        for layer_idx, encoder_layer in enumerate(self.layers):
            if layer_idx < reuse_start_layer or layer_idx >= reuse_end_layer:
                hidden_states = encoder_layer(hidden_states)
            else:
                if debug:
                    cache_state = "hit" if reuse_cache[layer_idx] is not None else "miss"
                    logger.info("InternVL reuse layer=%s cache=%s", layer_idx,
                                cache_state)
                hidden_states = apply_block_sparse_layer(
                    encoder_layer,
                    hidden_states,
                    reuse_mask=reuse_mask,
                    cache=reuse_cache[layer_idx],
                )
            reuse_cache[layer_idx] = hidden_states

        return hidden_states


class InternVisionModel(nn.Module):

    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.use_data_parallel = use_data_parallel

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
            use_data_parallel=use_data_parallel,
        )

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
        *,
        reuse_mask: Optional[torch.Tensor] = None,
        reuse_cache: Optional[list[Optional[torch.Tensor]]] = None,
        reuse_start_layer: int = 2,
        reuse_end_layer: Optional[int] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                'You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f'wrong pixel_values size: {pixel_values.shape}')

        if self.use_data_parallel:
            if reuse_mask is not None:
                raise NotImplementedError(
                    "Reuse is not supported with data-parallel sharding.")
            encoder_outputs = run_dp_sharded_vision_model(
                hidden_states, self.encoder)
        else:
            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                reuse_mask=reuse_mask,
                reuse_cache=reuse_cache,
                reuse_start_layer=reuse_start_layer,
                reuse_end_layer=reuse_end_layer,
            )

        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
