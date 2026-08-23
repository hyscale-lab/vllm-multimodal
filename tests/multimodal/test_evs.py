# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.multimodal.evs import recompute_mrope_positions


def test_mrope_chunk_boundary_immediately_after_final_vision_start():
    """The next chunk belongs to the final media; there is no next start."""
    vision_start, image_token, video_token = 100, 101, 102
    input_ids = torch.tensor([10, vision_start, image_token, image_token,
                              image_token])
    initial = torch.arange(5).view(1, -1).expand(3, -1).clone()
    media_positions = [torch.tensor([
        [0, 1],
        [0, 0],
        [0, 1],
        [2, 2],
    ])]

    positions, _ = recompute_mrope_positions(
        input_ids,
        media_positions,
        initial,
        num_computed_tokens=2,  # exactly after the final vision-start token
        vision_start_token_id=vision_start,
        image_token_id=image_token,
        video_token_id=video_token,
    )

    assert positions.shape == initial.shape
    assert positions[:, 2:4].shape == (3, 2)
