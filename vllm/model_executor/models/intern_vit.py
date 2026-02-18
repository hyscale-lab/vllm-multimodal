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


def build_prune_mask_from_embeddings(
    embeddings: torch.Tensor,
    prev_embeddings: torch.Tensor,
    *,
    prune_ratio: float,
    cls_index: int = 0,
) -> torch.Tensor:
    """
    Build a boolean prune mask from input embeddings.
    True means "prune/skip", False means "recompute". CLS is always recompute.
    """
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
        prev_embeddings = prev_embeddings.unsqueeze(0)

    if embeddings.shape != prev_embeddings.shape:
        raise ValueError("Embeddings shape mismatch for prune mask.")

    batch_size, num_tokens, _ = embeddings.shape
    if num_tokens <= 1:
        return torch.zeros_like(embeddings[:, :, 0], dtype=torch.bool)

    if not (0.0 <= prune_ratio <= 1.0):
        raise ValueError("prune_ratio must be within [0, 1].")

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

    recompute_count = int(round((num_tokens - 1) * prune_ratio))
    recompute_count = max(0, min(num_tokens - 1, recompute_count))

    prune_mask = torch.ones((batch_size, num_tokens),
                            dtype=torch.bool,
                            device=embeddings.device)
    prune_mask[:, cls_index] = False

    if debug:
        sim0 = similarity[0]
        logger.info(
            "InternVL prune similarity: tokens=%s cls_index=%s "
            "recompute_count=%s sim[min/mean/max]=%.4f/%.4f/%.4f",
            num_tokens,
            cls_index,
            recompute_count,
            float(sim0.min().item()),
            float(sim0.mean().item()),
            float(sim0.max().item()),
        )

    if recompute_count == 0:
        return prune_mask

    _, recompute_idx = torch.topk(similarity,
                                  k=recompute_count,
                                  dim=-1,
                                  largest=False)
    # Shift indices if CLS is at position 0 (common case).
    if cls_index == 0:
        recompute_idx = recompute_idx + 1

    for batch_idx in range(batch_size):
        prune_mask[batch_idx, recompute_idx[batch_idx]] = False

    return prune_mask


def apply_pruned_layer(
    layer: "InternVisionEncoderLayer",
    hidden_states: torch.Tensor,
    *,
    prune_mask: torch.Tensor,
    cache: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a block-sparse attention+FFN path with pruning.

    Tokens with prune_mask=True are copied from cache; others are recomputed.
    This is an aggressive approximation for experimental use.
    """
    if cache is None:
        return layer(hidden_states)

    if prune_mask.ndim == 1:
        prune_mask = prune_mask.unsqueeze(0).expand(hidden_states.size(0), -1)

    if prune_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("prune_mask shape must match hidden_states.")

    batch_size, num_tokens, _ = hidden_states.shape
    outputs = cache.clone()

    for batch_idx in range(batch_size):
        recompute_mask = ~prune_mask[batch_idx]
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
        prune_mask: Optional[torch.Tensor] = None,
        prune_cache: Optional[list[Optional[torch.Tensor]]] = None,
        prune_update_cache: bool = True,
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

    def forward_pruned(
        self,
        normed_full_hidden: torch.Tensor,
        q_indices: torch.Tensor,
    ) -> torch.Tensor:
        if q_indices.numel() == 0:
            return normed_full_hidden[:, :0, :]

        qkv, _ = self.qkv(normed_full_hidden)
        q_full, k_full, v_full = qkv.chunk(3, dim=-1)

        if self.qk_normalization:
            q_full, k_full = self._apply_qk_norm(q_full, k_full)

        q_dyn = q_full.index_select(1, q_indices)
        out = self.attn(q_dyn, k_full, v_full)
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
        prune_mask: Optional[torch.Tensor] = None,
        prune_cache: Optional[list[Optional[torch.Tensor]]] = None,
        prune_update_cache: bool = True,
    ):
        hidden_states = inputs_embeds
        if prune_mask is None or prune_cache is None:
            for encoder_layer in self.layers:
                hidden_states = encoder_layer(hidden_states)
            return hidden_states

        num_layers = len(self.layers)
        if len(prune_cache) != num_layers:
            raise ValueError("prune_cache length must match encoder layers.")
        debug = os.environ.get("VLLM_INTERNVL_PRUNE_DEBUG", "0") == "1"
        if debug:
            logger.info(
                "InternVL prune layers: total=%s",
                num_layers,
            )

        if prune_mask.ndim == 1:
            prune_mask = prune_mask.unsqueeze(0)

        if prune_update_cache:
            for layer_idx, encoder_layer in enumerate(self.layers):
                if debug:
                    cache_state = "hit" if prune_cache[
                        layer_idx] is not None else "miss"
                    logger.info("InternVL prune layer=%s cache=%s", layer_idx,
                                cache_state)
                hidden_states = apply_pruned_layer(
                    encoder_layer,
                    hidden_states,
                    prune_mask=prune_mask,
                    cache=prune_cache[layer_idx],
                )
                prune_cache[layer_idx] = hidden_states
            return hidden_states

        if prune_mask.shape[0] > 1 and not torch.equal(
                prune_mask, prune_mask[0:1].expand_as(prune_mask)):
            raise NotImplementedError(
                "Per-example prune masks are not supported with compact "
                "hidden states.")

        dynamic_idx = (~prune_mask[0]).nonzero(as_tuple=False).squeeze(-1)
        if dynamic_idx.numel() == 0:
            return hidden_states[:, :0, :]

        hidden_states = hidden_states.index_select(1, dynamic_idx)
        for layer_idx, encoder_layer in enumerate(self.layers):
            cache = prune_cache[layer_idx]
            if cache is None:
                raise ValueError(
                    "Prune cache is missing for compact pruning.")

            full_hidden = cache.clone()
            full_hidden.index_copy_(1, dynamic_idx, hidden_states)
            normed_full = encoder_layer.norm1(full_hidden)
            attn_out = encoder_layer.attn.forward_pruned(
                normed_full, dynamic_idx) * encoder_layer.ls1
            hidden_states = hidden_states + attn_out
            mlp_out = encoder_layer.mlp(
                encoder_layer.norm2(hidden_states)) * encoder_layer.ls2
            hidden_states = hidden_states + mlp_out

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
        prune_mask: Optional[torch.Tensor] = None,
        prune_cache: Optional[list[Optional[torch.Tensor]]] = None,
        prune_update_cache: bool = True,
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
            if prune_mask is not None:
                raise NotImplementedError(
                    "Pruning is not supported with data-parallel sharding.")
            encoder_outputs = run_dp_sharded_vision_model(
                hidden_states, self.encoder)
        else:
            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                prune_mask=prune_mask,
                prune_cache=prune_cache,
                prune_update_cache=prune_update_cache,
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
