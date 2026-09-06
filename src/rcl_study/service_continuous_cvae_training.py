from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .service_continuous_cvae import (
    FactorizedConditionalVAE,
    masked_reconstruction_loss,
)


class CVAETrainingValidationError(ValueError):
    """Raised when VAE training attempts forbidden membership or label access."""


_BATCH_FIELDS = (
    "mechanism",
    "mechanism_mask",
    "propagation",
    "propagation_mask",
    "context",
    "context_mask",
    "target_context",
    "target_context_mask",
)


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ids(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CVAETrainingValidationError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise CVAETrainingValidationError(
            f"{context} must contain unique nonempty IDs"
        )
    return result


def _weights(
    value: Mapping[str, Any], required: set[str], context: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != required:
        raise CVAETrainingValidationError(f"{context} weight closure drift")
    result: dict[str, float] = {}
    for name in sorted(required):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise CVAETrainingValidationError(f"{context}.{name} must be numeric")
        weight = float(raw)
        if not math.isfinite(weight) or weight < 0.0:
            raise CVAETrainingValidationError(
                f"{context}.{name} must be finite and nonnegative"
            )
        result[name] = weight
    return result


def build_cvae_training_access_ledger(
    *,
    pretraining_case_ids: Sequence[Any],
    queried_labels: Mapping[str, Mapping[str, Any]],
    held_out_case_ids: Sequence[Any],
    query_budget: int = 30,
) -> dict[str, Any]:
    pretraining = _ids(pretraining_case_ids, "pretraining_case_ids")
    held_out = _ids(held_out_case_ids, "held_out_case_ids")
    if set(pretraining) & set(held_out):
        raise CVAETrainingValidationError(
            "held-out cases cannot enter VAE pretraining"
        )
    if (
        isinstance(query_budget, bool)
        or not isinstance(query_budget, int)
        or query_budget <= 0
    ):
        raise CVAETrainingValidationError("query_budget must be a positive integer")
    if not isinstance(queried_labels, Mapping) or len(queried_labels) != query_budget:
        raise CVAETrainingValidationError(
            "VAE supervision count must equal query_budget"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for raw_case_id, raw_label in queried_labels.items():
        case_id = str(raw_case_id).strip()
        if not case_id or not isinstance(raw_label, Mapping):
            raise CVAETrainingValidationError("queried label layout drift")
        label = deepcopy(dict(raw_label))
        if set(label) != {
            "root_cause",
            "fault_type",
            "label_source",
            "budget_cost",
        }:
            raise CVAETrainingValidationError("queried label closure drift")
        if (
            not str(label["root_cause"]).strip()
            or not str(label["fault_type"]).strip()
            or label["label_source"] != "queried_budget"
            or label["budget_cost"] != 1
        ):
            raise CVAETrainingValidationError(
                "only budget-owned queried labels may supervise the VAE"
            )
        normalized[case_id] = label
    label_ids = sorted(normalized)
    if not set(label_ids) <= set(pretraining):
        raise CVAETrainingValidationError(
            "queried label cases must belong to non-held-out pretraining membership"
        )
    observation_only = sorted(set(pretraining) - set(label_ids))
    identity = {
        "schema_version": "service-continuous-cvae-training-access-ledger-v1",
        "pretraining_case_ids": pretraining,
        "pretraining_case_count": len(pretraining),
        "held_out_case_ids": held_out,
        "held_out_overlap_count": 0,
        "label_case_ids": label_ids,
        "queried_label_case_count": len(label_ids),
        "queried_labels": {case_id: normalized[case_id] for case_id in label_ids},
        "observation_only_case_ids": observation_only,
        "unqueried_observation_case_count": len(observation_only),
        "query_budget_cost": sum(
            int(normalized[case_id]["budget_cost"]) for case_id in label_ids
        ),
    }
    return {**identity, "ledger_sha256": _semantic_hash(identity)}


def _validated_ledger(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CVAETrainingValidationError("training ledger must be a mapping")
    ledger = deepcopy(dict(value))
    supplied_hash = str(ledger.pop("ledger_sha256", ""))
    if supplied_hash != _semantic_hash(ledger):
        raise CVAETrainingValidationError("training ledger hash drift")
    if (
        ledger.get("schema_version")
        != "service-continuous-cvae-training-access-ledger-v1"
        or not isinstance(ledger.get("queried_label_case_count"), int)
        or ledger.get("queried_label_case_count") <= 0
        or ledger.get("query_budget_cost")
        != ledger.get("queried_label_case_count")
        or ledger.get("held_out_overlap_count") != 0
    ):
        raise CVAETrainingValidationError("training ledger protocol drift")
    return {**ledger, "ledger_sha256": supplied_hash}


def _batch(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(_BATCH_FIELDS):
        raise CVAETrainingValidationError("unlabeled batch field closure drift")
    result = {name: value[name] for name in _BATCH_FIELDS}
    if any(not isinstance(tensor, Tensor) for tensor in result.values()):
        raise CVAETrainingValidationError("unlabeled batch values must be tensors")
    return result


def _kl_loss(posterior: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
    losses = []
    for name in ("mechanism", "propagation", "context"):
        mu = posterior[name]["mu"]
        logvar = posterior[name]["logvar"]
        losses.append(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean())
    return torch.stack(losses).mean()


def compute_unlabeled_objectives(
    *,
    model: FactorizedConditionalVAE,
    batch: Mapping[str, Tensor],
    weights: Mapping[str, Any],
    sample_seed: int,
) -> dict[str, Any]:
    if not isinstance(model, FactorizedConditionalVAE):
        raise CVAETrainingValidationError("unlabeled objectives require the CVAE")
    tensors = _batch(batch)
    objective_weights = _weights(
        weights,
        {"masked_reconstruction", "kl", "target_context", "cycle_consistency"},
        "unlabeled objective",
    )
    first = model(**tensors, sample=True, sample_seed=int(sample_seed))
    reconstruction_losses = [
        masked_reconstruction_loss(
            first["reconstruction"][name], tensors[name], tensors[f"{name}_mask"]
        )
        for name in ("mechanism", "propagation", "context")
    ]
    reconstruction = torch.stack(reconstruction_losses).mean()
    kl = _kl_loss(first["posterior"])
    if first["reconstruction"]["context"].shape != tensors["target_context"].shape:
        raise CVAETrainingValidationError(
            "target-context objective requires matching context dimensions"
        )
    target_context = masked_reconstruction_loss(
        first["reconstruction"]["context"],
        tensors["target_context"],
        tensors["target_context_mask"],
    )
    cycle_batch = {
        "mechanism": first["reconstruction"]["mechanism"],
        "mechanism_mask": tensors["mechanism_mask"],
        "propagation": first["reconstruction"]["propagation"],
        "propagation_mask": tensors["propagation_mask"],
        "context": first["reconstruction"]["context"],
        "context_mask": tensors["context_mask"],
        "target_context": tensors["target_context"],
        "target_context_mask": tensors["target_context_mask"],
    }
    cycled = model(**cycle_batch, sample=False)
    cycle_losses = [
        masked_reconstruction_loss(
            cycled["reconstruction"][name],
            first["reconstruction"][name],
            tensors[f"{name}_mask"],
        )
        for name in ("mechanism", "propagation", "context")
    ]
    cycle = torch.stack(cycle_losses).mean()
    components = {
        "masked_reconstruction": reconstruction,
        "kl": kl,
        "target_context": target_context,
        "cycle_consistency": cycle,
    }
    total = sum(
        components[name] * objective_weights[name] for name in components
    )
    if not torch.isfinite(total):
        raise CVAETrainingValidationError("unlabeled objective must remain finite")
    return {
        "components": components,
        "total": total,
        "label_case_ids": [],
        "supervised_label_count": 0,
    }


def _tensor(value: Any, context: str, *, rank: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != rank:
        raise CVAETrainingValidationError(f"{context} must be rank-{rank} tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise CVAETrainingValidationError(f"{context} must be finite floating point")
    return value


def compute_queried_supervised_objectives(
    *,
    ledger: Mapping[str, Any],
    case_ids: Sequence[Any],
    rank_scores: Tensor,
    positive_indices: Tensor,
    source_mechanism: Tensor,
    generated_mechanism: Tensor,
    mechanism_mask: Tensor,
    weights: Mapping[str, Any],
) -> dict[str, Any]:
    access = _validated_ledger(ledger)
    cases = _ids(case_ids, "supervised case_ids")
    allowed = set(access["label_case_ids"])
    if not set(cases) <= allowed:
        raise CVAETrainingValidationError(
            "unqueried cases cannot contribute supervised VAE objectives"
        )
    objective_weights = _weights(
        weights,
        {"pairwise_ranking", "mechanism_preservation", "repeated_fault_compactness"},
        "supervised objective",
    )
    scores = _tensor(rank_scores, "rank_scores", rank=2)
    if (
        not isinstance(positive_indices, Tensor)
        or positive_indices.ndim != 1
        or positive_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise CVAETrainingValidationError(
            "positive_indices must be an integer rank-1 tensor"
        )
    batch_size, candidate_count = scores.shape
    if (
        batch_size != len(cases)
        or positive_indices.shape[0] != batch_size
        or candidate_count < 2
        or bool(torch.any(positive_indices < 0))
        or bool(torch.any(positive_indices >= candidate_count))
    ):
        raise CVAETrainingValidationError("pairwise ranking layout drift")
    positive = scores.gather(1, positive_indices.to(scores.device).view(-1, 1))
    negative_mask = torch.ones_like(scores, dtype=torch.bool)
    negative_mask.scatter_(1, positive_indices.to(scores.device).view(-1, 1), False)
    margins = positive.expand_as(scores)[negative_mask] - scores[negative_mask]
    pairwise = F.softplus(-margins).mean()

    source = _tensor(source_mechanism, "source_mechanism", rank=2)
    generated = _tensor(generated_mechanism, "generated_mechanism", rank=2)
    mask = _tensor(mechanism_mask, "mechanism_mask", rank=2)
    if source.shape != generated.shape or source.shape != mask.shape or source.shape[0] != batch_size:
        raise CVAETrainingValidationError("mechanism preservation layout drift")
    preservation = masked_reconstruction_loss(generated, source, mask)

    labels = access["queried_labels"]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, case_id in enumerate(cases):
        groups[str(labels[case_id]["fault_type"])].append(index)
    repeated = [indices for indices in groups.values() if len(indices) > 1]
    compactness_terms = []
    for indices in repeated:
        for left_position, left in enumerate(indices[:-1]):
            for right in indices[left_position + 1 :]:
                joint_mask = mask[left] * mask[right]
                compactness_terms.append(
                    masked_reconstruction_loss(
                        generated[left : left + 1],
                        generated[right : right + 1],
                        joint_mask.unsqueeze(0),
                    )
                )
    compactness = (
        torch.stack(compactness_terms).mean()
        if compactness_terms
        else scores.new_zeros(())
    )
    components = {
        "pairwise_ranking": pairwise,
        "mechanism_preservation": preservation,
        "repeated_fault_compactness": compactness,
    }
    total = sum(
        components[name] * objective_weights[name] for name in components
    )
    if not torch.isfinite(total):
        raise CVAETrainingValidationError("supervised objective must remain finite")
    return {
        "components": components,
        "total": total,
        "label_case_ids": cases,
        "supervised_label_count": len(cases),
        "repeated_fault_group_count": len(repeated),
    }


__all__ = [
    "CVAETrainingValidationError",
    "build_cvae_training_access_ledger",
    "compute_queried_supervised_objectives",
    "compute_unlabeled_objectives",
]
