from __future__ import annotations

import torch


DISTAL_JOINTS = (3, 6, 10, 13, 16)
LEFT_RIGHT = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))


def apply_frozen_source_corruption(
    inputs: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.5,
) -> torch.Tensor:
    """Apply the V1 nonzero, temporally correlated source corruption recipe."""

    if inputs.ndim != 5 or inputs.shape[-2:] != (17, 2):
        raise ValueError("inputs must have shape (B,P,T,17,2)")
    result = inputs.clone()
    batch, people, frames, _, _ = result.shape
    for b in range(batch):
        for p in range(people):
            if torch.rand((), generator=generator).item() >= probability:
                continue
            joint = DISTAL_JOINTS[int(torch.randint(len(DISTAL_JOINTS), (), generator=generator))]
            length = int(torch.randint(12, 61, (), generator=generator))
            start_max = max(frames - length + 1, 1)
            start = int(torch.randint(start_max, (), generator=generator))
            end = min(start + length, frames)
            actual = end - start
            magnitude = 0.03 + 0.12 * torch.rand((), generator=generator).item()
            direction = torch.randn((2,), generator=generator)
            direction = direction / direction.norm().clamp_min(1e-6)
            ramp = torch.sin(torch.linspace(0.0, torch.pi, actual, device=result.device))
            result[b, p, start:end, joint] += magnitude * ramp[:, None] * direction.to(result.device)

            spike_joint = int(torch.randint(17, (), generator=generator))
            spike_frame = int(torch.randint(frames, (), generator=generator))
            spike = (0.04 + 0.16 * torch.rand((), generator=generator).item())
            spike_direction = torch.randn((2,), generator=generator)
            spike_direction = spike_direction / spike_direction.norm().clamp_min(1e-6)
            result[b, p, spike_frame, spike_joint] += spike * spike_direction.to(result.device)

            if torch.rand((), generator=generator).item() < 0.35:
                left, right = LEFT_RIGHT[int(torch.randint(len(LEFT_RIGHT), (), generator=generator))]
                swap_length = int(torch.randint(4, 21, (), generator=generator))
                swap_start = int(torch.randint(max(frames - swap_length + 1, 1), (), generator=generator))
                swap_end = min(swap_start + swap_length, frames)
                saved = result[b, p, swap_start:swap_end, left].clone()
                result[b, p, swap_start:swap_end, left] = result[b, p, swap_start:swap_end, right]
                result[b, p, swap_start:swap_end, right] = saved
    return result

