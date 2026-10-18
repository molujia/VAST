"""Bounded CNN execution over equal, complete native lengths only."""
from collections import defaultdict

import torch
from torch import Tensor, nn


def convolve_length_buckets(cnn: nn.Module, features: list, *, device,
                            batch_size: int, max_batch_samples: int) -> list[Tensor]:
    """Return one [channels, native positions] tensor per original input.

    The point limit counts native positions, not feature-channel scalars. A
    single sequence longer than the limit is processed intact and alone. No
    padding, splitting, deduplication, dtype change, or activation cache occurs.
    Limits bound individual CNN calls, not all activations retained by autograd.
    """
    buckets = defaultdict(list)
    for index, native in enumerate(features):
        buckets[len(native[0])].append(index)
    encoded: dict[int, Tensor] = {}
    for length, indices in buckets.items():
        capacity = max(1, min(batch_size, max_batch_samples // length))
        for offset in range(0, len(indices), capacity):
            chunk = indices[offset:offset + capacity]
            native = torch.tensor([features[index] for index in chunk], device=device, dtype=torch.float32)
            hidden = cnn(native)
            for index, output in zip(chunk, hidden.unbind(0)):
                encoded[index] = output
    return [encoded[index] for index in range(len(features))]
