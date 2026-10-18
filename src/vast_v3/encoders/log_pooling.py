"""Native-time, frequency-weighted log pooling with an ordered transpose.

Only already-encoded, visible records enter this module. Observed bins and
recovery-only closed time windows remain separate; no raw values are pooled.
"""
from __future__ import annotations

from math import floor

import torch
from torch import Tensor


class _OrderedSegmentSum(torch.autograd.Function):
    """Apply a fixed incidence matrix and its transpose without atomic scatter.

    Both directions gather into a fixed order and reduce disjoint segments.
    Calling this same linear operation with transposed metadata in backward
    also supports higher-order derivatives, without segment_reduce's native
    backward or a repeated-index index_select backward.
    """

    @staticmethod
    def forward(ctx, values, sources, lengths, reverse_sources, reverse_lengths):
        ctx.save_for_backward(sources, lengths, reverse_sources, reverse_lengths)
        selected = values.index_select(0, sources)
        # Metadata is constructed below: all lengths are nonnegative and sum
        # exactly to len(sources), including zero-length (missing) segments.
        return torch.segment_reduce(selected, 'sum', lengths=lengths, unsafe=True)

    @staticmethod
    def backward(ctx, gradient):
        sources, lengths, reverse_sources, reverse_lengths = ctx.saved_tensors
        transposed = _OrderedSegmentSum.apply(
            gradient, reverse_sources, reverse_lengths, sources, lengths)
        return transposed, None, None, None, None


def _closed_center_range(timestamp, centers, radius):
    """Binary-search the original abs(time-center) <= radius predicate.

    Searching rounded time +/- radius endpoints instead can change membership
    at one-ULP boundaries. Centers stay Python floats, as in NativeFrontend.
    """
    lo, hi = 0, len(centers)
    while lo < hi:
        mid = (lo+hi)//2
        if centers[mid] < timestamp and abs(timestamp-centers[mid]) > radius:
            lo = mid+1
        else:
            hi = mid
    first = lo
    hi = len(centers)
    while lo < hi:
        mid = (lo+hi)//2
        if centers[mid] > timestamp and abs(timestamp-centers[mid]) > radius:
            hi = mid
        else:
            lo = mid+1
    return range(first, lo)


def pool_encoded_logs(encoded: Tensor, owners, times, frequencies, *,
                      entity_count: int, bins: int, reference_start: float,
                      delta: float, radius_steps: float):
    """Return observed states/mask and recovery states/mask, in catalog order.

    Inputs preserve the visible record order and contain only the permitted
    case window. Frequencies are positive, following LogRecord's contract.
    Group membership is metadata only; gradients reach every encoded record.
    """
    shape = (entity_count, bins)
    if not len(times):
        states = encoded.new_zeros(*shape, encoded.shape[-1])
        mask = torch.zeros(shape, device=encoded.device, dtype=torch.bool)
        return states, mask, states.clone(), mask.clone()

    group_count = entity_count*bins
    groups = [[] for _ in range(2*group_count)]
    centers = [reference_start+(b+.5)*delta for b in range(bins)]
    radius = radius_steps*delta
    for index, (owner, timestamp) in enumerate(zip(owners, times)):
        b = floor((timestamp-reference_start)/delta)
        if 0 <= b < bins:
            groups[owner*bins+b].append(index)
        for b in _closed_center_range(timestamp, centers, radius):
            groups[group_count+owner*bins+b].append(index)

    observed_present = any(groups[:group_count])
    recovery_present = any(groups[group_count:])
    if not observed_present and not recovery_present:
        # Native float subtraction can round an in-window timestamp's bin up
        # to bins. Match legacy disconnected zeros, not zero MLP gradients:
        # Adam with existing momentum treats grad=None and grad=0 differently.
        states = encoded.new_zeros(*shape, encoded.shape[-1])
        mask = torch.zeros(shape, device=encoded.device, dtype=torch.bool)
        return states, mask, states.clone(), mask.clone()

    # Preserve original record order inside every destination group, and a
    # fixed destination order inside each source's transpose segment.
    reverse = [[] for _ in times]
    masses = []
    for destination, indices in enumerate(groups):
        masses.append(sum(frequencies[index] for index in indices) if indices else 1)
        for index in indices:
            reverse[index].append(destination)

    def index_tensor(values):
        return torch.tensor(values, device=encoded.device, dtype=torch.long)

    sources = index_tensor([index for group in groups for index in group])
    lengths = index_tensor([len(group) for group in groups])
    reverse_sources = index_tensor([index for group in reverse for index in group])
    reverse_lengths = index_tensor([len(group) for group in reverse])
    weighted = encoded*encoded.new_tensor(frequencies).unsqueeze(-1)
    sums = _OrderedSegmentSum.apply(weighted, sources, lengths, reverse_sources, reverse_lengths)
    # Python accumulation above retains the legacy denominator, including
    # large integer frequencies; only its final dtype conversion is tensorized.
    pooled = sums/encoded.new_tensor(masses).unsqueeze(-1)
    observed, recovery = pooled.reshape(2, *shape, encoded.shape[-1]).unbind(0)
    # A wholly empty branch must not gain an artificial MLP gradient path
    # through the other branch's shared packed reduction.
    if not observed_present:
        observed = encoded.new_zeros(*shape, encoded.shape[-1])
    if not recovery_present:
        recovery = encoded.new_zeros(*shape, encoded.shape[-1])
    observed_mask, recovery_mask = (lengths > 0).reshape(2, *shape).unbind(0)
    return observed, observed_mask, recovery, recovery_mask
