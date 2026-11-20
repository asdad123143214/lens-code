"""The LENS score on a sequence of patch-token grids."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def local_descriptors(
    feature_map: torch.Tensor,
    kernel_size: int = 3,
    temperature: float = 0.1,
    include_center: bool = False,
) -> torch.Tensor:
    """Extract softmax-normalized local self-similarity descriptors.

    With include_center=False the centre bin (self-similarity, always 1/temperature) is masked like an
    out-of-grid neighbour, so each descriptor only describes how similarity spreads over the neighbours.
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must have shape [T, H, W, C], got {tuple(feature_map.shape)}"
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not include_center and kernel_size == 1:
        raise ValueError("include_center=False needs kernel_size >= 3")

    time_steps, height, width, channels = feature_map.shape
    normalized = F.normalize(feature_map.float(), dim=-1, eps=1e-6)
    normalized_nchw = normalized.permute(0, 3, 1, 2)
    padding = kernel_size // 2
    neighborhood_count = kernel_size * kernel_size
    neighborhoods = F.unfold(
        normalized_nchw,
        kernel_size=kernel_size,
        padding=padding,
    ).reshape(time_steps, channels, neighborhood_count, height * width)
    neighborhoods = neighborhoods.permute(0, 3, 2, 1)
    centers = normalized.reshape(time_steps, height * width, channels).unsqueeze(2)
    logits = (centers * neighborhoods).sum(dim=-1) / temperature

    valid = F.unfold(
        torch.ones(
            (time_steps, 1, height, width),
            dtype=normalized.dtype,
            device=normalized.device,
        ),
        kernel_size=kernel_size,
        padding=padding,
    ).reshape(time_steps, neighborhood_count, height * width)
    valid = valid.permute(0, 2, 1).bool()
    if not include_center:
        valid[..., neighborhood_count // 2] = False
    logits = logits.masked_fill(~valid, -torch.inf)
    descriptors = torch.softmax(logits, dim=-1)
    return descriptors.reshape(time_steps, height, width, neighborhood_count)


def best_neighbor_distance(
    descriptors: torch.Tensor,
    search_window_size: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match local descriptors in the next frame and return [T - 1, H, W]."""
    if descriptors.ndim != 4:
        raise ValueError(
            f"descriptors must have shape [T, H, W, D], got {tuple(descriptors.shape)}"
        )
    if descriptors.shape[0] < 2:
        raise ValueError("At least two frames are required")
    if search_window_size < 1 or search_window_size % 2 == 0:
        raise ValueError("search_window_size must be a positive odd integer")

    current = descriptors[:-1].float()
    following = descriptors[1:].float()
    _, height, width, _ = current.shape
    best = torch.full(
        current.shape[:-1],
        torch.inf,
        dtype=current.dtype,
        device=current.device,
    )
    radius = search_window_size // 2
    for delta_y in range(-radius, radius + 1):
        current_y0 = max(0, -delta_y)
        current_y1 = min(height, height - delta_y)
        next_y0 = current_y0 + delta_y
        next_y1 = current_y1 + delta_y
        for delta_x in range(-radius, radius + 1):
            current_x0 = max(0, -delta_x)
            current_x1 = min(width, width - delta_x)
            next_x0 = current_x0 + delta_x
            next_x1 = current_x1 + delta_x
            left = current[:, current_y0:current_y1, current_x0:current_x1]
            right = following[:, next_y0:next_y1, next_x0:next_x1]
            midpoint = 0.5 * (left + right)
            left_kl = left * (
                torch.log(left.clamp_min(eps))
                - torch.log(midpoint.clamp_min(eps))
            )
            right_kl = right * (
                torch.log(right.clamp_min(eps))
                - torch.log(midpoint.clamp_min(eps))
            )
            distance = 0.5 * (left_kl.sum(dim=-1) + right_kl.sum(dim=-1))
            region = best[:, current_y0:current_y1, current_x0:current_x1]
            torch.minimum(region, distance, out=region)
    return best.clamp_min(0.0)


def score_tokens(
    tokens: torch.Tensor,
    *,
    window_size: int = 5,
    kernel_size: int = 3,
    temperature: float = 0.1,
    include_center: bool = False,
) -> float:
    """Compute the LENS score from DINO patch tokens.

    Args:
        tokens: Tensor shaped [T, N, C], where T is frames, N is square patch count
            such as 14*14 for 224 input with ViT-L/16, and C is channel dim.
        window_size: spatial search window used by best-neighbor matching.
        kernel_size: local spatial structure kernel size.
        temperature: local spatial structure descriptor temperature.
        include_center: keep the centre (self-similarity) bin in the local descriptor.

    Returns:
        A scalar score. Higher means more fake-like under the project convention.
    """
    if tokens.ndim != 3 or tokens.shape[0] < 3:
        raise ValueError(f"Expected tokens [T,N,C] with T>=3, got {tuple(tokens.shape)}")

    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise ValueError(f"Patch count must be square, got N={tokens.shape[1]}")

    feature_map = tokens.reshape(tokens.shape[0], side, side, tokens.shape[-1])

    descriptors = local_descriptors(
        feature_map,
        kernel_size=kernel_size,
        temperature=temperature,
        include_center=include_center,
    )
    distances = best_neighbor_distance(
        descriptors,
        search_window_size=window_size,
    )

    sorted_all = torch.sort(distances.reshape(distances.shape[0], -1), dim=1).values
    all_w1 = (sorted_all[1:] - sorted_all[:-1]).abs().mean(dim=1)

    return float(-all_w1.mean().item())
