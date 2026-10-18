"""Pure v1.3 objectives; None denotes an ineligible component, never E=0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def eligible_mean(losses: Iterable[Tensor | None]) -> Tensor | None:
    """Average only eligible cases; an empty population has no computation."""
    present = [value for value in losses if value is not None]
    return torch.stack(present).mean(0) if present else None


def observation_loss(local: Tensor | None, process: Tensor | None,
                     invariance: Tensor | None) -> Tensor | None:
    """Keep each observation component's fixed 1/3, including missing ones."""
    present = [value for value in (local, process, invariance) if value is not None]
    return torch.stack(present).sum(0) / 3 if present else None


def class_normalized_bce(scores: Tensor, roots: Tensor) -> Tensor | None:
    """Equation (8), averaging present classes rather than catalog entries."""
    if scores.ndim != 1 or roots.shape != scores.shape or roots.dtype != torch.bool:
        raise ValueError("scores and boolean roots must be equally sized vectors")
    if not torch.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite")
    terms = []
    if roots.any():
        terms.append(F.softplus(-scores[roots]).mean())
    if (~roots).any():
        terms.append(F.softplus(scores[~roots]).mean())
    return eligible_mean(terms)


def hierarchical_recovery_loss(records: Iterable[Mapping[str, Any]]) -> Tensor | None:
    """Equation (11), equal modality/unit/entry mass and dimension-mean Huber.

    Each dict contains ``modality`` (hashable), ``unit`` (hashable node or
    directed owner ID), ``prediction`` [...,J,D], ``target`` [J,D], and
    ``eligible`` bool[J]. Hypothesis/batch prefix axes on predictions are
    preserved, but eligibility is always shared across source hypotheses.
    One record per (modality,unit) is required: combine its entries first.
    The caller marks only originally-observed, withheld, context-eligible
    targets. Nonfinite target coordinates are unavailable; an entry with no
    finite coordinates is omitted. Fixed targets are detached; nonfinite
    predictions at observed target coordinates raise rather than become zeros.
    """
    modalities = defaultdict(list)
    seen: set[tuple[Hashable, Hashable]] = set()
    prefix_shape = None
    for record in records:
        key = (record["modality"], record["unit"])
        if key in seen:
            raise ValueError("Duplicate modality/unit: each directed owner is counted once")
        seen.add(key)
        prediction, target = record["prediction"], record["target"]
        eligible = record["eligible"]
        if target.ndim != 2 or prediction.ndim < 2 or prediction.shape[-2:] != target.shape:
            raise ValueError("prediction [...,J,D] must match target [J,D]")
        if eligible.dtype != torch.bool or eligible.shape != target.shape[:1]:
            raise ValueError("eligible must be bool[J], shared across hypotheses")
        if target.shape[-1] == 0:
            raise ValueError("Target dimension must be nonzero")
        if prefix_shape is None:
            prefix_shape = prediction.shape[:-2]
        elif prefix_shape != prediction.shape[:-2]:
            raise ValueError("All records must have the same hypothesis prefix")
        finite = torch.isfinite(target)
        valid = eligible & finite.any(-1)
        if not valid.any():
            continue
        predicted = prediction[..., valid, :]
        coordinates = finite[valid]
        if not torch.isfinite(predicted[..., coordinates]).all():
            raise ValueError("Eligible predictions must be finite")
        # Select before Huber so masked NaNs cannot poison backward through a
        # multiplication by zero. The coordinate mask is hypothesis-independent.
        clean_prediction = torch.where(coordinates, predicted, torch.zeros_like(predicted))
        fixed = torch.where(coordinates, target.detach()[valid], 0).expand_as(predicted)
        per_coordinate = F.huber_loss(clean_prediction, fixed, reduction="none", delta=1.0)
        per_entry = per_coordinate.sum(-1) / coordinates.sum(-1)
        error = per_entry.mean(-1)
        modalities[key[0]].append(error)
    return eligible_mean(eligible_mean(units) for units in modalities.values())


def labeled_source_loss(true_error: Tensor | None,
                        wrong_errors: Tensor | None) -> Tensor | None:
    """Equation (12); no wrong alternatives leaves true completion only."""
    if true_error is None:
        return None
    if true_error.ndim != 0 or not torch.isfinite(true_error):
        raise ValueError("True completion error must be a finite scalar")
    if wrong_errors is None or wrong_errors.numel() == 0:
        return true_error
    if wrong_errors.ndim != 1 or not torch.isfinite(wrong_errors).all():
        raise ValueError("Wrong completion errors must be a finite vector")
    return true_error + F.softplus(true_error - wrong_errors).mean()


def unlabeled_source_loss(scores: Tensor, source_sets: Tensor, errors: Tensor | None,
                          *, T_E: float = 1.0, lambda_c: float = 0.5) -> Tensor | None:
    """Equation (13), retaining gradients through both q(scores) and errors.

    ``source_sets`` is discrete bool[H,N], ``scores`` [N], ``errors`` [H].
    Empty libraries or absent eligible targets return None explicitly; never
    give individual unreachable hypotheses a different eligibility mask.
    """
    if T_E <= 0 or not torch.isfinite(torch.tensor(T_E)):
        raise ValueError("T_E temperature must be finite and positive")
    if not torch.isfinite(torch.tensor(lambda_c)):
        raise ValueError("lambda_c cardinality penalty must be finite")
    if scores.ndim != 1 or source_sets.ndim != 2 or source_sets.shape[1] != scores.numel():
        raise ValueError("Expected scores[N] and source_sets[H,N]")
    if source_sets.dtype != torch.bool:
        raise ValueError("Source membership must be discrete boolean data")
    if errors is None:
        return None
    if errors.shape != source_sets.shape[:1]:
        raise ValueError("errors must contain one value per hypothesis")
    if source_sets.shape[0] == 0:
        return None
    if not source_sets.any(1).all():
        raise ValueError("Unlabeled hypotheses must be nonempty source sets")
    if not torch.isfinite(scores).all() or not torch.isfinite(errors).all():
        raise ValueError("Scores and completion errors must be finite")
    membership = source_sets.to(dtype=scores.dtype)
    logits = membership @ scores - lambda_c * membership.sum(-1)
    log_q = torch.log_softmax(logits, dim=0)
    return -T_E * torch.logsumexp(log_q - errors / T_E, dim=0)
