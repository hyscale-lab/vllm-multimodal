# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vllm.multimodal.codec_cache as codec_cache
from vllm.multimodal.codec_cache import (
    apply_hf_processor_with_codec_cache,
    codec_frame_digest,
    hash_mm_items_with_codec,
    normalize_codec_frame,
    normalize_codec_frame_info,
)
from vllm.model_executor.models.interfaces import (
    SupportsCodecGuidedPruning,
    supports_codec_guided_pruning,
)


class _Info:
    model_id = "codec-cache-test"


class _Processor:
    info = _Info()


class _CodecModel(SupportsCodecGuidedPruning):
    pass


def test_codec_guided_pruning_protocol_marker() -> None:
    assert supports_codec_guided_pruning(_CodecModel)
    assert supports_codec_guided_pruning(_CodecModel())


def test_schema_v2_uses_true_as_keep() -> None:
    frame = normalize_codec_frame({
        "schema_version": 2,
        "is_anchor": False,
        "anchor_idx": 0,
        "patch_keep_mask": [True, False, True, False],
        "token_keep_mask": [True],
        "patch_shape": [2, 2],
        "token_shape": [1, 1],
        "kept_count": 1,
    }, require_complete=True)
    assert frame["patch_keep_mask"] == [True, False, True, False]
    assert frame["token_keep_mask"] == [True]


def test_legacy_schema_normalizes_to_equivalent_v2() -> None:
    legacy = {
        "anchor_idx": 0,
        "mask": [False, False, True, False, True],
        "proj_mask": [True],
        "patch_shape": [2, 2],
        "proj_shape": [1, 1],
        "kept_count": 1,
    }
    current = {
        "schema_version": 2,
        "is_anchor": False,
        "anchor_idx": 0,
        "patch_keep_mask": [True, False, True, False],
        "token_keep_mask": [True],
        "patch_shape": [2, 2],
        "token_shape": [1, 1],
        "kept_count": 1,
    }
    normalized = normalize_codec_frame_info(
        [current, legacy], require_complete=True)
    assert normalized[0] == normalized[1]
    assert codec_frame_digest(legacy) == codec_frame_digest(current)


def test_schema_rejects_inconsistent_kept_count() -> None:
    with pytest.raises(ValueError, match="kept_count"):
        normalize_codec_frame({
            "schema_version": 2,
            "is_anchor": False,
            "anchor_idx": 0,
            "patch_keep_mask": [True],
            "token_keep_mask": [True, False],
            "patch_shape": [1, 1],
            "token_shape": [1, 2],
            "kept_count": 2,
        }, require_complete=True)


def test_schema_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="patch mask length"):
        normalize_codec_frame({
            "schema_version": 2,
            "is_anchor": True,
            "anchor_idx": 0,
            "patch_keep_mask": [True],
            "token_keep_mask": [True],
            "patch_shape": [1, 2],
            "token_shape": [1, 1],
            "kept_count": 1,
        }, require_complete=True)


def test_codec_digest_uses_mask_content() -> None:
    left = {
        "kept_count": 2,
        "proj_shape": [2, 2],
        "proj_mask": [True, False, True, False],
    }
    right = {
        "kept_count": 2,
        "proj_shape": [2, 2],
        "proj_mask": [False, True, False, True],
    }
    assert codec_frame_digest(left) != codec_frame_digest(right)


def test_codec_digest_is_stable_for_mapping_order() -> None:
    left = {"anchor_idx": 0, "mask": np.array([True, False])}
    right = {"mask": np.array([True, False]), "anchor_idx": 0}
    assert codec_frame_digest(left) == codec_frame_digest(right)


def test_codec_hash_rejects_metadata_image_mismatch() -> None:
    with pytest.raises(ValueError, match="metadata=0, images=1"):
        hash_mm_items_with_codec(
            _Processor(),
            {"image": ["frame"]},
            {"codec_frame_info": []},
            {},
        )


def test_codec_hash_distinguishes_equal_size_masks() -> None:
    common = {"image": ["frame"]}
    left = hash_mm_items_with_codec(
        _Processor(),
        common,
        {
            "codec_frame_info": [{
                "kept_count": 2,
                "proj_mask": [True, False, True, False],
            }]
        },
        {},
    )
    right = hash_mm_items_with_codec(
        _Processor(),
        common,
        {
            "codec_frame_info": [{
                "kept_count": 2,
                "proj_mask": [False, True, False, True],
            }]
        },
        {},
    )
    assert left["image"] != right["image"]


def test_codec_hash_combines_uuid_with_mask() -> None:
    common = {"image": ["pixels-are-not-hashed"]}
    left = hash_mm_items_with_codec(
        _Processor(),
        common,
        {"codec_frame_info": [{"proj_mask": [True, False]}]},
        {},
        mm_uuids={"image": ["frame-uuid"]},
    )
    right = hash_mm_items_with_codec(
        _Processor(),
        common,
        {"codec_frame_info": [{"proj_mask": [False, True]}]},
        {},
        mm_uuids={"image": ["frame-uuid"]},
    )
    assert left["image"] != right["image"]


def test_cache_misses_filter_images_and_codec_together(monkeypatch) -> None:
    class Cache:
        def is_cached(self, hashes):
            return [value != "miss" for value in hashes]

    class Processor:
        cache = Cache()
        captured = None

        def _hash_mm_items(self, *args, **kwargs):
            return {"image": ["hit-a", "miss", "hit-b"]}

        def _to_mm_items(self, data):
            return data

        def _apply_hf_processor_main(self, **kwargs):
            self.captured = kwargs
            return [1], {}, True

        def _get_mm_fields_config(self, *args):
            return {}

        def _get_mm_prompt_updates(self, *args):
            return {}

        def _merge_mm_kwargs(self, *args, **kwargs):
            return {}, {}

    class KwargsItems:
        @staticmethod
        def from_hf_inputs(*args):
            return {}

    monkeypatch.setattr(codec_cache, "MultiModalKwargsItems", KwargsItems)
    processor = Processor()
    result = apply_hf_processor_with_codec_cache(
        processor,
        prompt="<image><image><image>",
        mm_data_items={"image": ["a", "b", "c"]},
        hf_processor_mm_kwargs={
            "codec_frame_info": [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        },
        tokenization_kwargs={},
    )

    assert result[2] is True
    assert processor.captured["mm_items"] == {"image": ["b"]}
    assert processor.captured["hf_processor_mm_kwargs"]["codec_frame_info"] == [
        {"id": "b"}
    ]
