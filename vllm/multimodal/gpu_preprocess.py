"""Shared GPU preprocessing building blocks for multimodal models.

All GPU batch preprocessing for NVDEC-decoded frames lives here.
Model files (internvl.py, qwen3_vl.py) stay clean -- they contain
only the original CPU baseline preprocessing and dispatch here when
the ``_nvdec_decoded`` flag is set.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def frames_to_gpu_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert PIL images to a ``[N, C, H, W]`` float32 GPU tensor in [0, 1].

    Uses ``np.asarray`` (zero-copy when possible) and a single
    ``torch.from_numpy`` + ``permute`` + ``.cuda()`` to minimise
    host-side overhead.
    """
    np_images = [np.asarray(img.convert("RGB")) for img in images]
    batch_np = np.stack(np_images)
    batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    return batch_t.cuda().float().div_(255.0)


def gpu_resize(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Batch bicubic resize on GPU, clamped to [0, 1]."""
    if t.shape[2] == h and t.shape[3] == w:
        return t
    return F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False).clamp_(0.0, 1.0)


def gpu_normalize(
    t: torch.Tensor,
    mean: Union[tuple[float, ...], list[float]],
    std: Union[tuple[float, ...], list[float]],
) -> torch.Tensor:
    """Batch channel-wise normalisation on GPU."""
    m = torch.tensor(mean, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
    return t.sub_(m).div_(s)


def gpu_preprocess_internvl(
    images: list[Image.Image],
    input_size: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    min_num: int = 1,
    max_num: int = 1,
    use_thumbnail: bool = True,
) -> list[torch.Tensor]:
    """GPU batch equivalent of ``image_to_pixel_values_internvl``.

    Supports arbitrary ``max_dynamic_patch``.  Since all video frames
    share the same resolution the tiling ratio is computed once and
    applied to the whole batch on GPU.

    Returns a list of ``[num_tiles, 3, input_size, input_size]`` tensors
    (one per input image), same contract as the CPU per-frame path.
    """
    from vllm.model_executor.models.internvl import (
        find_closest_aspect_ratio,
        get_internvl_target_ratios,
    )

    t0 = time.perf_counter()
    t = frames_to_gpu_tensor(images)
    t1 = time.perf_counter()

    N, C, H, W = t.shape
    target_ratios = get_internvl_target_ratios(min_num, max_num)
    aspect_ratio = W / H
    cols, rows = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width=W, height=H, image_size=input_size
    )
    num_tiles = cols * rows
    target_w = input_size * cols
    target_h = input_size * rows

    t_resized = gpu_resize(t, target_h, target_w)
    tiles = t_resized.reshape(N, C, rows, input_size, cols, input_size).permute(
        0, 2, 4, 1, 3, 5
    ).reshape(N * num_tiles, C, input_size, input_size)
    tiles = gpu_normalize(tiles, image_mean, image_std)
    tiles = tiles.reshape(N, num_tiles, C, input_size, input_size)

    if use_thumbnail and num_tiles > 1:
        thumb = gpu_resize(t, input_size, input_size)
        thumb = gpu_normalize(thumb, image_mean, image_std).unsqueeze(1)
        tiles = torch.cat([tiles, thumb], dim=1)

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    logger.info(
        "gpu_preprocess_internvl: %d frames, %dx%d tiles, to_gpu=%.1fms, resize+norm+tile=%.1fms, total=%.1fms",
        N,
        cols,
        rows,
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
        (t2 - t0) * 1000,
    )

    return [tiles[i] for i in range(N)]


def gpu_preprocess_qwen3vl_images(
    images: list[Image.Image],
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU batch preprocessing for Qwen3-VL **image** inputs.

    Each image is treated as a single-frame video padded to
    ``temporal_patch_size`` frames (matching the HF processor contract).

    Returns ``(pixel_values, image_grid_thw)`` where:
    * ``pixel_values``: ``[N * grid_h * grid_w, C * temporal_patch_size * patch_size**2]``
    * ``image_grid_thw``: ``[N, 3]`` with each row ``(1, grid_h, grid_w)``
    """
    t = frames_to_gpu_tensor(images)
    N, C, H, W = t.shape
    factor = patch_size * merge_size
    h_bar = max(factor, round(H / factor) * factor)
    w_bar = max(factor, round(W / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(H * W / max_pixels)
        h_bar = max(factor, math.floor(H / beta / factor) * factor)
        w_bar = max(factor, math.floor(W / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (H * W))
        h_bar = max(factor, math.ceil(H * beta / factor) * factor)
        w_bar = max(factor, math.ceil(W * beta / factor) * factor)

    t = gpu_resize(t, h_bar, w_bar)
    t = gpu_normalize(t, image_mean, image_std)
    t = t.unsqueeze(1).expand(-1, temporal_patch_size, -1, -1, -1)

    grid_t = 1
    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size

    t = t.reshape(
        N,
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    t = t.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    patches = t.reshape(
        N,
        grid_t * grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size,
    )
    patches = patches.reshape(-1, C * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * N, dtype=torch.long)
    return patches, grid_thw
