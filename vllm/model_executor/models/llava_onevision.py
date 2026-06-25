# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Final, Literal, Optional, Protocol, Union

import torch
import torch.nn as nn
from transformers import (BatchFeature, LlavaOnevisionConfig,
                          LlavaOnevisionProcessor)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape, unpad_image)

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   VideoEmbeddingItems, VideoProcessorItems)
from vllm.multimodal.processing import (MultiModalProcessingInfo,
                                        PromptReplacement, PromptUpdate)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import LlavaDummyInputsBuilder, init_vision_tower_for_llava
from .llava_next import (BaseLlavaNextMultiModalProcessor, LlavaNextLikeConfig,
                         LlavaNextProcessingInfo)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16

# Number of codec-selectable visual tokens per SINGLE-TILE image: 27x27 = 729
# SigLIP patches (no image-path pooling). The trailing image_newline token is
# always kept and is NOT part of the selectable proj grid, so the model emits
# kept_count + 1 tokens per single-tile image.
_LLAVA_NUM_PROJ_TOKENS = 729


def _get_llava_prune_settings() -> dict[str, object]:
    """Read VLLM_LLAVA_PRUNE env gate (mirror of InternVL's settings).

    Actual pruning triggers only when this is enabled AND codec data is
    present in the request (see ``has_codec`` checks below).
    """
    enabled = os.environ.get("VLLM_LLAVA_PRUNE", "0") == "1"
    debug = os.environ.get("VLLM_LLAVA_PRUNE_DEBUG", "0") == "1"
    return {
        "enabled": enabled,
        "debug": debug,
    }


class LlavaOnevisionVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of videos
        - f: Number of frames
        - c: Number of channels (3)
        - h: Height
        - w: Width

        Note that `num_videos` may be different for each batch, and 'num_frames'
        may be different for each video, in which case the data is passed as a
        list instead of a batched tensor.
    """
    type: Literal["pixel_values_videos"] = "pixel_values_videos"

    pixel_values_videos: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("bn", "f", 3, "h", "w", dynamic_dims={"f"}),
    ]


class LlavaOnevisionImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - np: Number of patches (1 + num_patches)
        - c: Number of channels (3)
        - h: Height
        - w: Width

        Note that `num_patches` may be different per batch and image,
        in which case the data is passed as a list instead of a batched tensor.
    """
    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("bn", "np", 3, "h", "w", dynamic_dims={"np"}),
    ]

    image_sizes: Annotated[Optional[torch.Tensor], TensorShape("bn", 2)]

    # Codec-guided pruning metadata (optional; present only when the request
    # carries codec_frame_info and VLLM_LLAVA_PRUNE is enabled). One row per
    # image: proj_mask over the 27x27 SigLIP grid + kept_count.
    codec_proj_mask: Optional[torch.Tensor] = None  # [num_images, 729]
    codec_kept_counts: Optional[torch.Tensor] = None  # [num_images]


class LlavaOnevisionImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - ifs: Image feature size
        - hs: Hidden size (must match language model backbone)
    """
    type: Literal["image_embeds"] = "image_embeds"

    data: Annotated[
        torch.Tensor,
        TensorShape("bn", "ifs", "hs"),
    ]


LlavaOnevisionImageInputs = Union[LlavaOnevisionImagePixelInputs,
                                  LlavaOnevisionImageEmbeddingInputs]

LlavaOnevisionMultiInputs = Union[LlavaOnevisionImageInputs,
                                  LlavaOnevisionVideoPixelInputs]


class LlavaOnevisionLikeConfig(LlavaNextLikeConfig, Protocol):
    video_token_index: Final[int]


class LlavaOnevisionProcessingInfo(LlavaNextProcessingInfo):

    def get_hf_config(self) -> LlavaOnevisionLikeConfig:
        return self.ctx.get_hf_config(LlavaOnevisionConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(LlavaOnevisionProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L86
    # with additional logic afterwards taken from LlavaOnevisionProcessor
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = int(
                round(original_height * (current_width / original_width), 7))
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = int(
                round(original_width * (current_height / original_height), 7))
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        ratio = math.sqrt(current_height * current_width / (9 * npatches**2))
        if ratio > 1.1:
            height_factor = int(current_height // ratio)
            width_factor = int(current_width // ratio)
            unpadded_features = height_factor * width_factor
            newline_features = height_factor

        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        # NOTE: This hardcoded value is found via processor tests
        return ImageSize(width=1153, height=944)

    def _get_num_frame_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        spatial_pool_stride = getattr(hf_config, "spatial_pool_stride", 2)

        vision_encoder_info = self.get_vision_encoder_info()
        patch_grid_length = vision_encoder_info.get_patch_grid_length()
        pooled_grid_length = math.ceil(patch_grid_length / spatial_pool_stride)

        return pooled_grid_length * pooled_grid_length

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
    ) -> int:
        num_frame_tokens = self._get_num_frame_tokens(
            image_width=image_width,
            image_height=image_height,
        )

        return num_frame_tokens * num_frames + 1  # Newline token

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_videos = mm_counts.get("video", 0)

        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
        )


class LlavaOnevisionDummyInputsBuilder(
        LlavaDummyInputsBuilder[LlavaOnevisionProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len,
                                                        mm_counts)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


class LlavaOnevisionMultiModalProcessor(
        BaseLlavaNextMultiModalProcessor[LlavaOnevisionProcessingInfo]):

    def _hash_mm_items(self, mm_items, hf_processor_mm_kwargs,
                       tokenization_kwargs, **kwargs):
        # Fold per-frame kept_count into the per-item cache key so that the
        # same pixels with different codec masks do not collide in the MM
        # cache (mirror of InternVL._hash_mm_items).
        codec_info = hf_processor_mm_kwargs.get("codec_frame_info")
        clean_kwargs = {k: v for k, v in hf_processor_mm_kwargs.items()
                        if k != "codec_frame_info"}
        if codec_info is None:
            return super()._hash_mm_items(
                mm_items, clean_kwargs, tokenization_kwargs, **kwargs)

        kept_counts = [f.get("kept_count") for f in codec_info]
        model_id = self.info.model_id
        hashes: dict[str, list[str]] = {}
        for modality, items in mm_items.items():
            modality_hashes: list[str] = []
            for i, item in enumerate(items):
                per_item_kw = dict(clean_kwargs)
                if modality == "image" and i < len(kept_counts):
                    # One codec entry per image (one frame per image).
                    per_item_kw["_codec_kept"] = kept_counts[i]
                modality_hashes.append(
                    MultiModalHasher.hash_kwargs(
                        model_id=model_id,
                        **{modality: item},
                        **per_item_kw,
                        **tokenization_kwargs))
            hashes[modality] = modality_hashes
        return hashes

    def _cached_apply_hf_processor(self, prompt, mm_data_items,
                                   hf_processor_mm_kwargs,
                                   tokenization_kwargs, **kwargs):
        cache = self.cache
        _, passthrough_data = self._get_hf_mm_data(mm_data_items)
        codec_frame_info = hf_processor_mm_kwargs.get("codec_frame_info")
        if cache is None or passthrough_data or codec_frame_info is None:
            return super()._cached_apply_hf_processor(
                prompt=prompt,
                mm_data_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                **kwargs)

        mm_hashes = self._hash_mm_items(
            mm_data_items, hf_processor_mm_kwargs,
            tokenization_kwargs, **kwargs)

        mm_is_cached = {
            modality: cache.is_cached(hashes)
            for modality, hashes in mm_hashes.items()
        }
        mm_missing_idxs: dict[str, list[int]] = {
            modality: [
                idx for idx, is_cached
                in enumerate(items_is_cached) if not is_cached
            ]
            for modality, items_is_cached in mm_is_cached.items()
        }
        mm_missing_data: dict[str, list] = {}
        for modality, idxs in mm_missing_idxs.items():
            mm_missing_data[modality] = [
                mm_data_items[modality][idx] for idx in idxs]
        mm_missing_data_items = self._to_mm_items(mm_missing_data)

        # Codec masks are per-image; re-filter to the cache-missing images so
        # they stay positionally aligned (mirror of InternVL).
        filtered_kwargs = dict(hf_processor_mm_kwargs)
        image_missing_idxs = mm_missing_idxs.get("image", [])
        filtered_codec = ([codec_frame_info[i] for i in image_missing_idxs]
                          if image_missing_idxs else None)
        if filtered_codec:
            filtered_kwargs["codec_frame_info"] = filtered_codec
        else:
            filtered_kwargs.pop("codec_frame_info", None)

        (
            prompt_ids,
            mm_missing_processed_data,
            is_update_applied,
        ) = self._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_missing_data_items,
            hf_processor_mm_kwargs=filtered_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=False,
        )

        mm_missing_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_missing_processed_data,
            self._get_mm_fields_config(mm_missing_processed_data,
                                       hf_processor_mm_kwargs),
        )
        mm_missing_prompt_updates = self._get_mm_prompt_updates(
            mm_missing_data_items,
            hf_processor_mm_kwargs,
            mm_missing_kwargs,
        )

        mm_kwargs, mm_prompt_updates = self._merge_mm_kwargs(
            cache,
            mm_hashes=mm_hashes,
            mm_missing_kwargs=mm_missing_kwargs,
            mm_missing_prompt_updates=mm_missing_prompt_updates,
        )

        mm_info = MultiModalProcessingInfo(
            kwargs=mm_kwargs,
            hashes=mm_hashes,
            prompt_updates=mm_prompt_updates,
        )

        if is_update_applied:
            return prompt_ids, mm_info, is_update_applied

        tokenizer = self.info.get_tokenizer()
        expanded_text = (prompt if isinstance(prompt, str)
                         else tokenizer.decode(prompt_ids))

        for modality in ("image", "video"):
            if modality not in mm_prompt_updates:
                continue
            for item_updates in mm_prompt_updates[modality]:
                if not item_updates:
                    continue
                update = item_updates[0]
                target = update.target
                repl = update.content.full
                if (not isinstance(target, str)
                        or not isinstance(repl, str)):
                    return prompt_ids, mm_info, is_update_applied
                if target not in expanded_text:
                    return prompt_ids, mm_info, is_update_applied
                expanded_text = expanded_text.replace(target, repl, 1)

        prompt_ids = tokenizer.encode(expanded_text,
                                      add_special_tokens=False)
        return prompt_ids, mm_info, True

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
        )
        if "codec_proj_mask" in hf_inputs:
            fields["codec_proj_mask"] = MultiModalFieldConfig.batched("image")
            fields["codec_kept_counts"] = \
                MultiModalFieldConfig.batched("image")
        return fields

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Strip codec metadata before the HF processor ever sees it
        # (mirror of InternVL._call_hf_processor).
        codec_frame_info = mm_kwargs.get("codec_frame_info")
        if codec_frame_info is not None:
            mm_kwargs = {k: v for k, v in mm_kwargs.items()
                         if k != "codec_frame_info"}

        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", [])
        assert isinstance(videos, list)

        if not videos:
            outputs = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
            # Frames arrive as individual images: attach per-IMAGE codec
            # masks (one frame per image) to the processor outputs.
            codec_outputs = self._build_image_codec_outputs(codec_frame_info)
            if codec_outputs:
                outputs = BatchFeature({**dict(outputs), **codec_outputs})
            return outputs

        # LLaVA-OneVision processor doesn't support multiple videos
        # with different sizes when converting back to tensors
        # So, we process each component separately
        # NOTE: No prompt replacement is applied in this case
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token

        text_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data={},
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        images = mm_data.pop("images", [])
        assert isinstance(images, list)
        if images:
            processor_outputs = super()._call_hf_processor(
                prompt=image_token * len(images),
                mm_data={"images": images},
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
            image_outputs = {
                k: v
                for k, v in processor_outputs.items()
                if k in ("pixel_values", "image_sizes")
            }
        else:
            image_outputs = {}

        pixel_values_videos = []
        for video in videos:
            item_outputs = super()._call_hf_processor(
                prompt=video_token,
                mm_data={"videos": video},
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )

            pixel_values_videos.append(item_outputs["pixel_values_videos"][0])

        video_outputs = {"pixel_values_videos": pixel_values_videos}

        # Codec masks apply to the IMAGE path (frames are sent as images).
        codec_outputs = self._build_image_codec_outputs(codec_frame_info)

        combined_outputs = dict(
            text_outputs,
            **image_outputs,
            **video_outputs,
            **codec_outputs,
        )
        return BatchFeature(combined_outputs)

    @staticmethod
    def _build_image_codec_outputs(
        codec_frame_info: Optional[list],
    ) -> dict[str, torch.Tensor]:
        """Build per-IMAGE codec tensors from the request codec_frame_info.

        Each frame is sent as one image, so there is exactly one entry per
        image: a length-729 proj_mask (27x27 SigLIP grid) and a kept_count.
        Mirrors InternVL._call_hf_processor's tensor construction.
        """
        if codec_frame_info is None:
            return {}
        num_images = len(codec_frame_info)
        num_proj = _LLAVA_NUM_PROJ_TOKENS  # 729
        proj_mask = torch.zeros(num_images, num_proj, dtype=torch.bool)
        kept_counts = torch.full((num_images, ), num_proj, dtype=torch.long)
        for i, f in enumerate(codec_frame_info):
            if f.get("proj_mask") is not None:
                proj_mask[i] = torch.tensor(f["proj_mask"], dtype=torch.bool)
            if f.get("kept_count") is not None:
                kept_counts[i] = f["kept_count"]
        return {
            "codec_proj_mask": proj_mask,
            "codec_kept_counts": kept_counts,
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        base_result = super()._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        # When codec pruning is active, the HF processor must NOT expand the
        # placeholders itself (it would emit the full per-image token count);
        # vLLM's codec-aware _get_prompt_updates handles expansion instead
        # (mirror of InternVL's enable_hf_prompt_update=False path).
        prune_enabled = bool(_get_llava_prune_settings()["enabled"])
        if prune_enabled and hf_processor_mm_kwargs.get(
                "codec_frame_info") is not None:
            return False

        return base_result and mm_items.get_count("video", strict=False) == 0

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_repls = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_index

        # Codec image path: each image emits (kept tokens) + 1 placeholders
        # (the +1 is the always-kept image_newline). The kept-token count is
        # derived from proj_mask.sum() -- the SAME quantity the forward uses
        # via proj_mask.nonzero() -- so the placeholder count and the produced
        # token count can never drift (mirror of InternVL's repl path).
        prune_enabled = bool(_get_llava_prune_settings()["enabled"])
        out_mm_data = out_mm_kwargs.get_data()
        img_codec_proj_t = (out_mm_data.get("codec_proj_mask")
                            if prune_enabled else None)
        if isinstance(img_codec_proj_t, torch.Tensor) and \
                img_codec_proj_t.numel() > 0:
            # Per-image kept-token count, clamped to >=1 to match the forward.
            per_image_kept = [
                max(1, int(img_codec_proj_t[i].sum().item()))
                for i in range(img_codec_proj_t.shape[0])
            ]
            image_token_id = hf_config.image_token_index

            def get_image_replacement_codec(item_idx: int):
                if item_idx < len(per_image_kept):
                    num_image_tokens = per_image_kept[item_idx] + 1
                else:
                    num_image_tokens = _LLAVA_NUM_PROJ_TOKENS + 1
                return [image_token_id] * num_image_tokens

            image_repls = [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement_codec,
                )
            ]

        def get_video_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (VideoEmbeddingItems, VideoProcessorItems))

            if isinstance(videos, VideoEmbeddingItems):
                num_video_tokens = videos.get_feature_size(item_idx)
            else:
                image_size = videos.get_frame_size(item_idx)
                num_video_tokens = self.info.get_num_video_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    num_frames=videos.get_num_frames(item_idx),
                )

            return [video_token_id] * num_video_tokens

        return [
            *image_repls,
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_video_replacement,
            ),
        ]


class LlavaOnevisionMultiModalProjector(nn.Module):

    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=config.multimodal_projector_bias)
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(config.text_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=config.multimodal_projector_bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevisionMultiModalProcessor,
    info=LlavaOnevisionProcessingInfo,
    dummy_inputs=LlavaOnevisionDummyInputsBuilder)
class LlavaOnevisionForConditionalGeneration(nn.Module, SupportsMultiModal,
                                             SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))

        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaOnevisionImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)
        codec_proj_mask = kwargs.pop("codec_proj_mask", None)
        codec_kept_counts = kwargs.pop("codec_kept_counts", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaOnevisionImagePixelInputs(
                type="pixel_values",
                pixel_values=flatten_bn(pixel_values),
                image_sizes=flatten_bn(image_sizes, concat=True),
                codec_proj_mask=codec_proj_mask,
                codec_kept_counts=codec_kept_counts,
                resolve_bindings={
                    "h": self.config.vision_config.image_size,
                    "w": self.config.vision_config.image_size
                })

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaOnevisionImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
            self,
            **kwargs: object) -> Optional[LlavaOnevisionVideoPixelInputs]:
        """
        A legal video input should have the following dimensions:
        {
            "pixel_values_videos" : 
                list[b, Tensor(nb_frames, nb_channels, height, width)]
        }
        """
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        if pixel_values_videos is None:
            return None

        if not isinstance(pixel_values_videos, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel_values_videos. "
                             f"Got type: {type(pixel_values_videos)}")

        return LlavaOnevisionVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=flatten_bn(pixel_values_videos),
            resolve_bindings={
                "h": self.config.vision_config.image_size,
                "w": self.config.vision_config.image_size
            })

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)

        return mm_input_by_modality

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)
        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(self,
                                      image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor,
                                      *,
                                      image_newline=None,
                                      vision_aspect_ratio="anyres_max_9",
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                # image_aspect_ratio == "anyres"
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches] \
                    .view(num_patch_height, num_patch_width, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     (orig_height, orig_width))
                    max_num_patches = int(
                        vision_aspect_ratio.removeprefix("anyres_max_"))
                    channels, curr_height, curr_width = other_patch_embeds.shape
                    ratio = math.sqrt(curr_height * curr_width /
                                      (max_num_patches * height**2))
                    if ratio > 1.1:
                        other_patch_embeds = other_patch_embeds[None]
                        other_patch_embeds = nn.functional.interpolate(
                            other_patch_embeds, [
                                int(curr_height // ratio),
                                int(curr_width // ratio)
                            ],
                            mode="bilinear")[0]
                    if image_newline is not None:
                        other_patch_embeds = torch.cat(
                            (
                                other_patch_embeds,
                                image_newline[:, None, None] \
                                .expand(*other_patch_embeds.shape[:-1], 1) \
                                .to(other_patch_embeds.device),
                            ),
                        dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaOnevisionImagePixelInputs,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return [
            self.multi_modal_projector(image_features) for image_features in
            torch.split(stacked_image_features, num_patches_per_batch)
        ]

    def _prune_image_with_codec(
        self,
        patch_features_batch: torch.Tensor,
        codec_proj_mask_i: torch.Tensor,
    ) -> torch.Tensor:
        """Single-tile codec prune: drop base-tile tokens, keep one newline.

        Args:
            patch_features_batch: [num_tiles, 729, d] projected patches. For the
                codec prototype we assume a SINGLE tile (num_tiles == 1).
            codec_proj_mask_i: [729] bool, True = keep this patch token.

        Returns:
            [kept + 1, d] = kept base tokens followed by the image_newline.
            Mirrors InternVL._prune_with_codec's proj_mask index_select.
        """
        base_patch_embeds = patch_features_batch[0]  # [729, d]
        device = base_patch_embeds.device
        # Flatten to 1-D over the 729 base-tile tokens (the batched MM field can
        # arrive as [1,729]); nonzero(as_tuple) then yields a clean 1-D index that
        # index_select requires.
        keep_idx = codec_proj_mask_i.to(device).reshape(-1).bool().nonzero(
            as_tuple=True)[0]
        if keep_idx.numel() == 0:
            keep_idx = torch.zeros(1, dtype=torch.long, device=device)
        kept = base_patch_embeds.index_select(0, keep_idx)
        newline = self.image_newline[None].to(device)
        return torch.cat((kept, newline), dim=0)

    def _process_image_input(
        self,
        image_input: LlavaOnevisionImageInputs,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        settings = _get_llava_prune_settings()
        codec_proj_mask = image_input.get("codec_proj_mask")
        if settings["enabled"] and codec_proj_mask is not None:
            # Single-tile codec prune (prototype): one frame per image.
            return [
                self._prune_image_with_codec(
                    patch_features_batch, codec_proj_mask[i])
                for i, patch_features_batch in enumerate(patch_embeddings)
            ]

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["pixel_values"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        return [
            self._merge_image_patch_embeddings(
                image_sizes[i],
                patch_features_batch,
                image_newline=self.image_newline,
                strategy="spatial_unpad")
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def _video_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        video_features = vision_tower(pixel_values)
        video_features = self._select_image_features(
            video_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        video_features = self.apply_pooling(video_features)
        return video_features

    def _process_video_pixels(self, inputs: LlavaOnevisionVideoPixelInputs):
        assert self.vision_tower is not None

        video_pixels = inputs["pixel_values_videos"]

        if isinstance(video_pixels, torch.Tensor):
            total_videos, frames, c, h, w = video_pixels.shape
            video_pixels_flat = video_pixels.view(total_videos * frames, c, h,
                                                  w)

            embeddings_flat = self._video_pixels_to_features(
                self.vision_tower, video_pixels_flat)

            embeddings_flat = embeddings_flat.reshape(
                total_videos, frames * embeddings_flat.shape[1], -1)

            image_newline = self.image_newline[None, None, :].expand(
                total_videos, -1, -1)
            return torch.cat((embeddings_flat, image_newline), dim=1)

        frames_per_video = [len(video) for video in video_pixels]
        video_pixels_flat = torch.cat(video_pixels)

        embeddings_flat = self._video_pixels_to_features(
            self.vision_tower, video_pixels_flat)

        image_newline = self.image_newline[None, None, :]

        return [
            torch.cat(
                (
                    embeds.reshape(1, num_frame * embeddings_flat.shape[1],
                                   -1),
                    image_newline,
                ),
                dim=1,
            ) for num_frame, embeds in zip(
                frames_per_video,
                torch.split(embeddings_flat, frames_per_video),
            )
        ]

    def apply_pooling(self, image_features: torch.Tensor, stride: int = 2):
        vision_config = self.config.vision_config
        height = width = vision_config.image_size // vision_config.patch_size
        batch_frames, _, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2)

        # TODO support other pooling types config
        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_features,
                                                  size=scaled_shape,
                                                  mode='bilinear')
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_frames, -1, dim)
        return image_feature

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return []
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(vision_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_pixels(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_index, self.config.video_token_index])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[LlavaOnevisionImagePixelInputs] = None,
        video_input: Optional[LlavaOnevisionVideoPixelInputs] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_index,
            )

        if video_input is not None:
            video_embeds = self._process_video_pixels(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_index,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-Onevision.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
