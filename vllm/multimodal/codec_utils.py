"""Codec-aware pruning utilities for video multimodal processing.

Provides server-side motion-vector extraction and codec mask computation
so that clients can send encoded video directly and the server computes
``codec_frame_info`` metadata for InternVL3 / Qwen3-VL pruning paths.

The functions accept either:
  - PyAV structured MV arrays (``mvs_to_patch_grid``)
  - PyNvVideoCodec ``ParseDecodeStats()`` dicts (``nvdec_stats_to_patch_grid``)
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def mvs_to_patch_grid(
    mvs: np.ndarray | None,
    video_width: int,
    video_height: int,
    grid_size: int = 32,
) -> np.ndarray:
    """Map PyAV macroblock-level MVs to a patch grid of magnitudes.

    Returns ``(grid_size, grid_size)`` float32 array where each cell
    holds the maximum MV magnitude that falls within that spatial bin.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if mvs is None or len(mvs) == 0:
        return grid

    for mv in mvs:
        src_field = "dst_x" if "dst_x" in mvs.dtype.names else "src_x"
        dst_x = int(mv[src_field]) if src_field in mvs.dtype.names else 0
        dst_y_field = "dst_y" if "dst_y" in mvs.dtype.names else "src_y"
        dst_y = int(mv[dst_y_field]) if dst_y_field in mvs.dtype.names else 0
        mx = float(mv["motion_x"]) if "motion_x" in mvs.dtype.names else 0.0
        my = float(mv["motion_y"]) if "motion_y" in mvs.dtype.names else 0.0
        scale = float(mv["motion_scale"]) if "motion_scale" in mvs.dtype.names else 1.0
        if scale <= 0:
            scale = 1.0

        magnitude = math.sqrt(mx * mx + my * my) / scale

        gx = int(dst_x * grid_size / max(video_width, 1))
        gy = int(dst_y * grid_size / max(video_height, 1))
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[gy, gx] = max(grid[gy, gx], magnitude)

    return grid


def nvdec_stats_to_patch_grid(
    stats: dict[str, Any],
    video_width: int,
    video_height: int,
    grid_size: int = 32,
) -> np.ndarray:
    """Convert PyNvVideoCodec ``ParseDecodeStats()`` dict to a patch grid.

    NVDEC returns per-macroblock lists: ``mv0_x``, ``mv0_y`` (L0 motion
    vectors in quarter-pixel units), and ``cu_type`` (0=INTRA, 1=INTER,
    2=SKIP).  Macroblocks are laid out in raster order (row-major) with
    a fixed 16x16 macroblock size for H.264 (or variable CU for HEVC,
    but the stats are reported at macroblock granularity).

    Returns ``(grid_size, grid_size)`` float32 array of max MV magnitudes.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    mv0_x = stats.get("mv0_x", [])
    mv0_y = stats.get("mv0_y", [])
    if not mv0_x or not mv0_y:
        return grid

    mb_size = 16
    mb_cols = max((video_width + mb_size - 1) // mb_size, 1)

    for idx in range(min(len(mv0_x), len(mv0_y))):
        mx = mv0_x[idx] / 4.0
        my = mv0_y[idx] / 4.0
        magnitude = math.sqrt(mx * mx + my * my)

        mb_row = idx // mb_cols
        mb_col = idx % mb_cols
        cx = mb_col * mb_size + mb_size // 2
        cy = mb_row * mb_size + mb_size // 2

        gx = int(cx * grid_size / max(video_width, 1))
        gy = int(cy * grid_size / max(video_height, 1))
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[gy, gx] = max(grid[gy, gx], magnitude)

    return grid


def _downsample_mask_to_proj(
    dynamic_1024: np.ndarray,
    patch_grid: int = 32,
    downsample: int = 2,
) -> np.ndarray:
    """OR-pool a 1024-element dynamic mask to 256-element projected mask.

    ``pixel_shuffle(0.5)`` groups 2x2 patches into 1 projected token.
    A projected token is dynamic if ANY of its 4 source patches is dynamic.
    """
    mask_2d = dynamic_1024.reshape(patch_grid, patch_grid)
    proj_grid = patch_grid // downsample
    mask_2d = mask_2d.reshape(proj_grid, downsample, proj_grid, downsample)
    return mask_2d.any(axis=(1, 3)).flatten()


def compute_codec_masks(
    frame_types: Sequence[str],
    mv_grids: Sequence[np.ndarray],
    mv_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Build per-frame codec masks from pre-computed MV grids.

    This is the server-side equivalent of ``e2e_client.compute_codec_masks``
    but decoupled from the ``FrameData`` dataclass.

    Args:
        frame_types: Per-frame type string, ``"I"`` or ``"P"``.
        mv_grids: Per-frame ``(32, 32)`` MV magnitude grids (from either
            ``mvs_to_patch_grid`` or ``nvdec_stats_to_patch_grid``).
        mv_threshold: Magnitude threshold above which a patch is dynamic.

    Returns:
        Per-frame codec info dicts compatible with
        ``mm_processor_kwargs["codec_frame_info"]``.
    """
    num_patches = 1024
    num_proj_tokens = 256
    codec_info = []
    current_anchor = 0
    accum_dynamic = np.zeros(num_patches, dtype=bool)

    i_frame_mask = [False] * (1 + num_patches)
    i_frame_proj = [False] * num_proj_tokens

    for i, (ftype, mv_grid) in enumerate(zip(frame_types, mv_grids)):
        if ftype == "I" or i == 0:
            codec_info.append({
                "anchor_idx": i,
                "mask": i_frame_mask,
                "proj_mask": i_frame_proj,
                "kept_count": num_proj_tokens,
            })
            current_anchor = i
            accum_dynamic = np.zeros(num_patches, dtype=bool)
            continue

        frame_dynamic = mv_grid.flatten() >= mv_threshold
        accum_dynamic = accum_dynamic | frame_dynamic

        full_mask = [False] * (1 + num_patches)
        for j in range(num_patches):
            full_mask[1 + j] = not accum_dynamic[j]

        proj_dynamic = _downsample_mask_to_proj(accum_dynamic)
        kept_count = max(1, int(proj_dynamic.sum()))

        codec_info.append({
            "anchor_idx": current_anchor,
            "mask": full_mask,
            "proj_mask": proj_dynamic.tolist(),
            "kept_count": kept_count,
        })

    return codec_info


def compute_codec_masks_from_nvdec(
    decode_stats: Sequence[dict[str, Any]],
    frame_types: Sequence[str],
    video_width: int,
    video_height: int,
    mv_threshold: float = 1.0,
    grid_size: int = 32,
) -> list[dict[str, Any]]:
    """End-to-end: NVDEC decode stats -> codec_frame_info.

    Convenience wrapper that converts PyNvVideoCodec ``ParseDecodeStats()``
    outputs into the ``codec_frame_info`` format expected by InternVL3 and
    Qwen3-VL multimodal processors.
    """
    mv_grids = [
        nvdec_stats_to_patch_grid(stats, video_width, video_height, grid_size)
        for stats in decode_stats
    ]
    return compute_codec_masks(frame_types, mv_grids, mv_threshold)
