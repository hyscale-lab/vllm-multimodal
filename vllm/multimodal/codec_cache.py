# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.processing import MultiModalProcessingInfo

CODEC_SCHEMA_VERSION = 2


def _as_bool_list(value: Any, field: str) -> list[bool]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return [bool(item) for item in value]


def _as_shape(value: Any, field: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 2):
        raise TypeError(f"{field} must contain exactly two dimensions")
    shape = [int(dim) for dim in value]
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{field} dimensions must be positive")
    return shape


def normalize_codec_frame(
    frame: Mapping[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Convert codec metadata to schema v2, where every mask is True=keep."""
    if not isinstance(frame, Mapping):
        raise TypeError("codec_frame_info entries must be mappings")

    has_v2_fields = any(key in frame for key in (
        "patch_keep_mask", "token_keep_mask", "is_anchor", "token_shape"))
    version = frame.get("schema_version")
    if version is not None and int(version) != CODEC_SCHEMA_VERSION:
        raise ValueError(f"Unsupported codec schema_version: {version}")

    if has_v2_fields or version is not None:
        normalized: dict[str, Any] = {
            "schema_version": CODEC_SCHEMA_VERSION,
        }
        if "is_anchor" in frame:
            normalized["is_anchor"] = bool(frame["is_anchor"])
        if "anchor_idx" in frame:
            normalized["anchor_idx"] = int(frame["anchor_idx"])
        if "patch_keep_mask" in frame:
            normalized["patch_keep_mask"] = _as_bool_list(
                frame["patch_keep_mask"], "patch_keep_mask")
        if "token_keep_mask" in frame:
            normalized["token_keep_mask"] = _as_bool_list(
                frame["token_keep_mask"], "token_keep_mask")
        if "patch_shape" in frame:
            normalized["patch_shape"] = _as_shape(
                frame["patch_shape"], "patch_shape")
        if "token_shape" in frame:
            normalized["token_shape"] = _as_shape(
                frame["token_shape"], "token_shape")
    elif "mask" in frame or "proj_mask" in frame:
        normalized = {"schema_version": CODEC_SCHEMA_VERSION}
        legacy_patch = frame.get("mask")
        if legacy_patch is not None:
            legacy_patch = _as_bool_list(legacy_patch, "mask")
            if not legacy_patch:
                raise ValueError("legacy mask must include its CLS slot")
            normalized["patch_keep_mask"] = [
                not value for value in legacy_patch[1:]
            ]
            normalized["is_anchor"] = not any(legacy_patch)
        if "anchor_idx" in frame:
            normalized["anchor_idx"] = int(frame["anchor_idx"])
        if "proj_mask" in frame:
            normalized["token_keep_mask"] = _as_bool_list(
                frame["proj_mask"], "proj_mask")
        if "patch_shape" in frame:
            normalized["patch_shape"] = _as_shape(
                frame["patch_shape"], "patch_shape")
        if "proj_shape" in frame:
            normalized["token_shape"] = _as_shape(
                frame["proj_shape"], "proj_shape")
    else:
        return dict(frame)

    patch_keep = normalized.get("patch_keep_mask")
    patch_shape = normalized.get("patch_shape")
    token_keep = normalized.get("token_keep_mask")
    token_shape = normalized.get("token_shape")

    if patch_keep is not None and patch_shape is not None:
        if len(patch_keep) != patch_shape[0] * patch_shape[1]:
            raise ValueError("codec patch mask length does not match patch_shape")
    if token_keep is not None and token_shape is not None:
        if len(token_keep) != token_shape[0] * token_shape[1]:
            raise ValueError("codec token mask length does not match token_shape")

    if token_keep is not None:
        kept_count = sum(token_keep)
        if "kept_count" in frame and int(frame["kept_count"]) != kept_count:
            raise ValueError(
                "codec kept_count does not match token_keep_mask.sum()")
        normalized["kept_count"] = kept_count

    if require_complete:
        required = (
            "is_anchor", "anchor_idx", "patch_keep_mask", "token_keep_mask",
            "patch_shape", "token_shape", "kept_count",
        )
        missing = [key for key in required if key not in normalized]
        if missing:
            raise ValueError(
                "incomplete codec metadata; missing " + ", ".join(missing))
        if not normalized["patch_keep_mask"]:
            raise ValueError("patch_keep_mask cannot be empty")
        if not any(normalized["patch_keep_mask"]):
            raise ValueError("patch_keep_mask must keep at least one patch")
        if not any(normalized["token_keep_mask"]):
            raise ValueError("token_keep_mask must keep at least one token")
        if normalized["is_anchor"] and (
                not all(normalized["patch_keep_mask"])
                or not all(normalized["token_keep_mask"])):
            raise ValueError("anchor frames must keep every patch and token")

    return normalized


def normalize_codec_frame_info(
    codec_info: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    return [
        normalize_codec_frame(frame, require_complete=require_complete)
        for frame in codec_info
    ]


def _canonicalize_codec_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_codec_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_codec_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported codec metadata value: {type(value).__name__}")


def codec_frame_digest(frame: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonicalize_codec_value(normalize_codec_frame(frame)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def hash_mm_items_with_codec(
    processor: Any,
    mm_items: Any,
    hf_processor_mm_kwargs: Mapping[str, object],
    tokenization_kwargs: Mapping[str, object],
    *,
    mm_uuids: Mapping[str, object] | None = None,
) -> dict[str, list[str]]:
    codec_info = hf_processor_mm_kwargs.get("codec_frame_info")
    if not isinstance(codec_info, Sequence):
        raise TypeError("codec_frame_info must be a sequence")

    image_items = mm_items.get("image", ())
    if len(codec_info) != len(image_items):
        raise ValueError(
            "codec_frame_info/image count mismatch: "
            f"metadata={len(codec_info)}, images={len(image_items)}"
        )

    clean_kwargs = {
        key: value
        for key, value in hf_processor_mm_kwargs.items()
        if key != "codec_frame_info"
    }
    model_id = processor.info.model_id
    mm_uuids = mm_uuids or {}
    hashes: dict[str, list[str]] = {}
    for modality, items in mm_items.items():
        modality_uuids = mm_uuids.get(modality)
        if isinstance(modality_uuids, str):
            modality_uuids = [modality_uuids]
        if modality_uuids is not None and len(modality_uuids) != len(items):
            raise ValueError(
                f"{modality} UUID/item count mismatch: "
                f"uuids={len(modality_uuids)}, items={len(items)}"
            )

        modality_hashes: list[str] = []
        for index, item in enumerate(items):
            if modality_uuids is not None and modality_uuids[index] is not None:
                item = modality_uuids[index]
            per_item_kwargs = dict(clean_kwargs)
            if modality == "image":
                frame = codec_info[index]
                if not isinstance(frame, Mapping):
                    raise TypeError("codec_frame_info entries must be mappings")
                per_item_kwargs["_codec_frame_digest"] = codec_frame_digest(frame)
            modality_hashes.append(
                MultiModalHasher.hash_kwargs(
                    model_id=model_id,
                    **{modality: item},
                    **per_item_kwargs,
                    **tokenization_kwargs,
                )
            )
        hashes[modality] = modality_hashes
    return hashes


def apply_cached_prompt_updates(
    processor: Any,
    prompt: str | list[int],
    prompt_ids: list[int],
    mm_info: MultiModalProcessingInfo,
    is_update_applied: bool,
) -> tuple[list[int], MultiModalProcessingInfo, bool]:
    if is_update_applied:
        return prompt_ids, mm_info, True

    tokenizer = processor.info.get_tokenizer()
    expanded_text = prompt if isinstance(prompt, str) else tokenizer.decode(
        prompt_ids)
    for modality in ("image", "video"):
        for item_updates in mm_info.prompt_updates.get(modality, ()):
            if not item_updates:
                continue
            update = item_updates[0]
            target = update.target
            replacement = update.content.full
            if not isinstance(target, str) or not isinstance(replacement, str):
                return prompt_ids, mm_info, False
            if target not in expanded_text:
                return prompt_ids, mm_info, False
            expanded_text = expanded_text.replace(target, replacement, 1)

    return (
        tokenizer.encode(expanded_text, add_special_tokens=False),
        mm_info,
        True,
    )


def apply_hf_processor_with_codec_cache(
    processor: Any,
    prompt: str | list[int],
    mm_data_items: Any,
    hf_processor_mm_kwargs: Mapping[str, object],
    tokenization_kwargs: Mapping[str, object],
    **hash_kwargs: Any,
) -> tuple[list[int], MultiModalProcessingInfo, bool]:
    cache = processor.cache
    if cache is None:
        raise RuntimeError("codec cache helper requires an enabled processor cache")

    mm_hashes = processor._hash_mm_items(
        mm_data_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
        **hash_kwargs,
    )
    mm_missing_idxs = {
        modality: [
            index
            for index, is_cached in enumerate(cache.is_cached(hashes))
            if not is_cached
        ]
        for modality, hashes in mm_hashes.items()
    }
    mm_missing_data = {
        modality: [mm_data_items[modality][index] for index in indexes]
        for modality, indexes in mm_missing_idxs.items()
    }
    mm_missing_data_items = processor._to_mm_items(mm_missing_data)

    filtered_kwargs = dict(hf_processor_mm_kwargs)
    codec_info = hf_processor_mm_kwargs.get("codec_frame_info")
    if codec_info is not None:
        image_items = mm_data_items.get("image", ())
        if not isinstance(codec_info, Sequence):
            raise TypeError("codec_frame_info must be a sequence")
        if len(codec_info) != len(image_items):
            raise ValueError(
                "codec_frame_info/image count mismatch: "
                f"metadata={len(codec_info)}, images={len(image_items)}"
            )
        codec_info = normalize_codec_frame_info(codec_info)
        filtered_codec = [
            codec_info[index]
            for index in mm_missing_idxs.get("image", ())
        ]
        if filtered_codec:
            filtered_kwargs["codec_frame_info"] = filtered_codec
        else:
            filtered_kwargs.pop("codec_frame_info", None)

    prompt_ids, processed_data, is_update_applied = (
        processor._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_missing_data_items,
            hf_processor_mm_kwargs=filtered_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=False,
        )
    )
    missing_kwargs = MultiModalKwargsItems.from_hf_inputs(
        processed_data,
        processor._get_mm_fields_config(
            processed_data,
            hf_processor_mm_kwargs,
        ),
    )
    missing_updates = processor._get_mm_prompt_updates(
        mm_missing_data_items,
        hf_processor_mm_kwargs,
        missing_kwargs,
    )
    mm_kwargs, prompt_updates = processor._merge_mm_kwargs(
        cache,
        mm_hashes=mm_hashes,
        mm_missing_kwargs=missing_kwargs,
        mm_missing_prompt_updates=missing_updates,
    )
    mm_info = MultiModalProcessingInfo(
        kwargs=mm_kwargs,
        hashes=mm_hashes,
        prompt_updates=prompt_updates,
    )
    return apply_cached_prompt_updates(
        processor,
        prompt,
        prompt_ids,
        mm_info,
        is_update_applied,
    )
