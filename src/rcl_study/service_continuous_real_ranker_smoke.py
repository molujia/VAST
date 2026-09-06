from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .conservative_lofo_base_bridge import score_base_score_bridge
from .service_continuous_explicit import (
    build_explicit_arm_registry,
    generate_explicit_service_continuity,
)
from .service_continuous_pairwise_bridge import (
    fit_service_continuity_pairwise_bridge,
)
from .service_continuous_screen_gates import evaluate_ordinary_safety_gate
from .service_continuous_screen_scoring import (
    build_count_authoritative_score,
    build_ordinary_count_score,
    validate_ordinary_score,
    validate_type_score,
)
from .service_continuous_synthetic_admission import (
    build_synthetic_admission_ledger,
)
from .service_continuous_synthetic_training import (
    build_synthetic_training_ledger,
)


class RealRankerSmokeValidationError(ValueError):
    """Raised when the materialized real-fold smoke violates its frozen contract."""


_ARMS = (
    "baseline",
    "explicit_arbitrary",
    "explicit_compatible",
    "cvae_arbitrary",
    "cvae_compatible",
)
_OBSERVABLE_FIELDS = (
    "metric_direction",
    "metric_magnitude",
    "metric_duration",
    "metric_sparsity",
    "log_intensity",
    "log_template_change",
    "log_relative_time",
    "trace_latency",
    "trace_error",
    "trace_earliest_anomaly",
    "trace_hop_lag",
    "topology_depth",
    "topology_width",
    "topology_direction_consistency",
    "relative_onset",
    "relative_peak",
    "relative_recovery",
    "candidate_is_service",
    "candidate_reachability",
    "candidate_source_earliness",
    "candidate_explanation_coverage",
)
_PRESENCE_FIELDS = (
    "has_metric_signal",
    "has_log_signal",
    "has_trace_signal",
    "has_topology_signal",
    "has_time_signal",
    "has_candidate_signal",
)
_MECHANISM_FIELDS = _OBSERVABLE_FIELDS[:11]
_PROPAGATION_FIELDS = _OBSERVABLE_FIELDS[11:17]
_CONTEXT_FIELDS = _OBSERVABLE_FIELDS[17:] + _PRESENCE_FIELDS
_IDENTITY_CONTEXT_INDICES = tuple(range(0, 8)) + tuple(range(61, 66))
_SYMPTOM_INDICES = tuple(range(8, 61)) + tuple(range(66, 122))
_FROZEN_SMOKE_PROFILE = {
    "profile_id": "conservative_small",
    "hidden_width": 96,
    "mechanism_latent_width": 8,
    "propagation_latent_width": 8,
    "context_latent_width": 8,
    "kl_weight": 0.001,
    "sampling_radius": 0.35,
    "masked_reconstruction_weight": 1.0,
    "target_context_weight": 0.5,
    "cycle_consistency_weight": 0.5,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 5.0,
}


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RealRankerSmokeValidationError(f"{context} must be a positive integer")
    return value


def _materialized_request(
    value: Any,
    *,
    expected_kind: str,
    expected_held_out_fault_type: str | None,
    expected_evaluation_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RealRankerSmokeValidationError("authority request must be a mapping")
    request = deepcopy(dict(value))
    dataset = request.get("dataset")
    bundle = request.get("authority_score_bundle")
    budget = request.get("budget")
    if (
        request.get("schema_version")
        != "conservative-lofo-materialized-screen-request-v1"
        or request.get("dataset_id") != "rcabench"
        or request.get("kind") != expected_kind
        or request.get("held_out_fault_type") != expected_held_out_fault_type
        or request.get("seed") != 42
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
        or not isinstance(dataset, Mapping)
        or not isinstance(bundle, Mapping)
    ):
        raise RealRankerSmokeValidationError(
            f"{expected_kind} authority request drift"
        )
    fit_ids = tuple(str(value) for value in request.get("fit_case_ids", ()))
    selected_ids = tuple(str(value) for value in request.get("selected_case_ids", ()))
    evaluation_ids = tuple(
        str(value) for value in request.get("evaluation_case_ids", ())
    )
    if (
        len(fit_ids) != budget
        or fit_ids != selected_ids
        or len(evaluation_ids) != expected_evaluation_count
        or set(fit_ids) & set(evaluation_ids)
        or request.get("authority_score_bundle_sha256")
        != bundle.get("authority_score_bundle_sha256")
    ):
        raise RealRankerSmokeValidationError("authority membership or score binding drift")
    feature_names = tuple(str(value) for value in dataset.get("base_feature_names", ()))
    cases = dataset.get("cases")
    labels = dataset.get("labels_by_case")
    if (
        len(feature_names) != 122
        or not isinstance(cases, Mapping)
        or not isinstance(labels, Mapping)
        or not set(fit_ids + evaluation_ids) <= set(cases)
        or not set(fit_ids + evaluation_ids) <= set(labels)
    ):
        raise RealRankerSmokeValidationError("materialized dataset closure drift")
    for case_id in fit_ids + evaluation_ids:
        case = cases[case_id]
        candidates = tuple(str(value) for value in case.get("candidate_ids", ()))
        if (
            len(candidates) != 54
            or len(candidates) != len(set(candidates))
            or len(case.get("base_feature_rows", ())) != 54
            or len(case.get("observable_rows", ())) != 54
            or any(len(row) != 122 for row in case["base_feature_rows"])
        ):
            raise RealRankerSmokeValidationError(
                f"candidate-complete materialization drift for {case_id}"
            )
    return request


def _request(value: Any) -> dict[str, Any]:
    return _materialized_request(
        value,
        expected_kind="strict_lofo",
        expected_held_out_fault_type="NetworkDelay",
        expected_evaluation_count=21,
    )


def _candidate_rows(
    dataset: Mapping[str, Any], case_ids: Sequence[str]
) -> list[dict[str, Any]]:
    names = tuple(str(value) for value in dataset["base_feature_names"])
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = dataset["cases"][case_id]
        for candidate_id, values in zip(
            case["candidate_ids"], case["base_feature_rows"]
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": str(candidate_id),
                    **{name: float(value) for name, value in zip(names, values)},
                }
            )
    return rows


def _targets(
    dataset: Mapping[str, Any], case_ids: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    return {
        case_id: tuple(
            str(dataset["cases"][case_id]["candidate_ids"][index])
            for index in dataset["cases"][case_id]["positive_indices"]
        )
        for case_id in case_ids
    }


def _observable_by_candidate(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in case["observable_rows"]:
        row = dict(raw)
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in result:
            raise RealRankerSmokeValidationError("observable candidate ownership drift")
        result[candidate_id] = row
    if set(result) != set(str(value) for value in case["candidate_ids"]):
        raise RealRankerSmokeValidationError("observable candidate set drift")
    return result


def _values(row: Mapping[str, Any], fields: Sequence[str]) -> list[float]:
    values = [float(row[field]) for field in fields]
    if not all(math.isfinite(value) for value in values):
        raise RealRankerSmokeValidationError("observable state contains non-finite values")
    return values


def _mask_values(row: Mapping[str, Any]) -> dict[str, list[float]]:
    presence = {name: float(row[name]) for name in _PRESENCE_FIELDS}
    if any(value not in (0.0, 1.0) for value in presence.values()):
        raise RealRankerSmokeValidationError("observable modality mask drift")
    return {
        "mechanism": (
            [presence["has_metric_signal"]] * 4
            + [presence["has_log_signal"]] * 3
            + [presence["has_trace_signal"]] * 4
        ),
        "propagation": (
            [presence["has_topology_signal"]] * 3
            + [presence["has_time_signal"]] * 3
        ),
        "context": [presence["has_candidate_signal"]] * 4 + [1.0] * 6,
    }


def _state_values(row: Mapping[str, Any]) -> dict[str, list[float]]:
    return {
        "mechanism": _values(row, _MECHANISM_FIELDS),
        "propagation": _values(row, _PROPAGATION_FIELDS),
        "context": _values(row, _CONTEXT_FIELDS),
    }


def _vector_state(kind: str, values: Sequence[float], names: Sequence[str]) -> dict[str, Any]:
    identity = {
        "schema_version": f"service-continuous-materialized-{kind}-state-v1",
        "feature_names": [str(value) for value in names],
        "values": [float(value) for value in values],
    }
    return {**identity, "state_sha256": _semantic_hash(identity)}


def _compatibility(source: Mapping[str, Any], target: Mapping[str, Any]) -> float:
    source_state = _state_values(source)
    target_state = _state_values(target)
    differences = []
    for factor in ("mechanism", "propagation", "context"):
        left = np.asarray(source_state[factor], dtype=float)
        right = np.asarray(target_state[factor], dtype=float)
        scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
        differences.extend((np.abs(left - right) / scale).tolist())
    result = math.exp(-math.fsum(differences) / max(1, len(differences)))
    return min(1.0, max(0.0, result))


def _target_plan(
    source_row: Mapping[str, Any],
    observable_rows: Mapping[str, Mapping[str, Any]],
    source_targets: Sequence[str],
    *,
    max_targets: int,
    compatible: bool,
) -> list[dict[str, Any]]:
    candidates = [candidate for candidate in sorted(observable_rows) if candidate not in set(source_targets)]
    if len(candidates) < 2:
        raise RealRankerSmokeValidationError("service migration requires two non-root targets")
    scores = {candidate: _compatibility(source_row, observable_rows[candidate]) for candidate in candidates}
    ordered = sorted(candidates, key=lambda candidate: (scores[candidate], candidate))
    limit = min(_positive_int(max_targets, "max_targets"), len(ordered))
    if limit == 1:
        selected = [ordered[-1]]
    elif limit == len(ordered):
        selected = ordered
    else:
        selected = [ordered[0], ordered[-1]]
        for candidate in reversed(ordered[1:-1]):
            if len(selected) >= limit:
                break
            selected.append(candidate)
        selected = sorted(selected)
    raw_weights = (
        {candidate: max(scores[candidate], 1e-6) for candidate in selected}
        if compatible
        else {candidate: 1.0 for candidate in selected}
    )
    total = math.fsum(raw_weights.values())
    plan_identity = {
        "schema_version": "service-continuous-materialized-target-plan-v1",
        "target_policy": "compatible" if compatible else "arbitrary",
        "targets": selected,
        "compatibility": {candidate: scores[candidate] for candidate in selected},
        "weights": {candidate: raw_weights[candidate] / total for candidate in selected},
    }
    plan_hash = _semantic_hash(plan_identity)
    return [
        {
            "target_service_id": candidate,
            "target_weight": raw_weights[candidate] / total,
            "compatibility": scores[candidate],
            "target_weight_plan_sha256": plan_hash,
            "compatibility_sha256": _semantic_hash(
                {
                    "source_state": _semantic_hash(_state_values(source_row)),
                    "target_service_id": candidate,
                    "target_state": _semantic_hash(
                        _state_values(observable_rows[candidate])
                    ),
                    "score": scores[candidate],
                }
            ),
        }
        for candidate in selected
    ]


def _queried_label(
    dataset: Mapping[str, Any], case_id: str, targets: Sequence[str]
) -> dict[str, Any]:
    raw = dataset["labels_by_case"][case_id]
    root_cause = str(raw.get("root_cause", "")).strip()
    fault_type = str(raw.get("fault_type", "")).strip()
    if not root_cause or not fault_type or not targets:
        raise RealRankerSmokeValidationError("queried label materialization drift")
    return {
        "root_cause": root_cause,
        "fault_type": fault_type,
        "label_source": "queried_budget",
        "budget_cost": 1,
    }


def _explicit_rows(
    *,
    dataset: Mapping[str, Any],
    fit_ids: Sequence[str],
    source_ids: Sequence[str],
    targets_by_case: Mapping[str, Sequence[str]],
    max_targets: int,
) -> dict[str, list[dict[str, Any]]]:
    registry = build_explicit_arm_registry()
    arms = {"explicit_arbitrary": [], "explicit_compatible": []}
    for source_id in source_ids:
        case = dataset["cases"][source_id]
        observable = _observable_by_candidate(case)
        roots = tuple(targets_by_case[source_id])
        source_row = observable[roots[0]]
        states = _state_values(source_row)
        source_case = {
            "case_id": source_id,
            "queried_label": _queried_label(dataset, source_id, roots),
            "observation_case_ids": list(fit_ids),
            "label_case_ids": [source_id],
            "same_fault_type_partner_case_ids": [],
            "mechanism_state": _vector_state(
                "mechanism", states["mechanism"], _MECHANISM_FIELDS
            ),
            "propagation_state": _vector_state(
                "propagation", states["propagation"], _PROPAGATION_FIELDS
            ),
            "context_state": _vector_state(
                "context", states["context"], _CONTEXT_FIELDS
            ),
        }
        for arm_id in arms:
            compatible = arm_id.endswith("compatible") and not arm_id.endswith(
                "arbitrary"
            )
            plan = _target_plan(
                source_row,
                observable,
                roots,
                max_targets=max_targets,
                compatible=compatible,
            )
            for target in plan:
                target_row = observable[target["target_service_id"]]
                target_states = _state_values(target_row)
                bundle = generate_explicit_service_continuity(
                    source_case=source_case,
                    target_context={
                        **target,
                        "target_entity_level": "service",
                        "propagation_state": _vector_state(
                            "propagation",
                            target_states["propagation"],
                            _PROPAGATION_FIELDS,
                        ),
                        "context_state": _vector_state(
                            "context", target_states["context"], _CONTEXT_FIELDS
                        ),
                    },
                    arm_config=registry["arms"][arm_id],
                )
                arms[arm_id].extend(bundle["synthetic_rows"])
    return arms


def _normalization(
    dataset: Mapping[str, Any], fit_ids: Sequence[str], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    raw_states = {factor: [] for factor in ("mechanism", "propagation", "context")}
    raw_masks = {factor: [] for factor in raw_states}
    for case_id in fit_ids:
        for row in _observable_by_candidate(dataset["cases"][case_id]).values():
            values = _state_values(row)
            masks = _mask_values(row)
            for factor in raw_states:
                raw_states[factor].append(values[factor])
                raw_masks[factor].append(masks[factor])
    normalized: dict[str, torch.Tensor] = {}
    mask_tensors: dict[str, torch.Tensor] = {}
    identity: dict[str, Any] = {}
    for factor in raw_states:
        values = torch.tensor(raw_states[factor], dtype=torch.float32, device=device)
        masks = torch.tensor(raw_masks[factor], dtype=torch.float32, device=device)
        denominator = masks.sum(dim=0).clamp_min(1.0)
        mean = (values * masks).sum(dim=0) / denominator
        variance = (((values - mean) * masks) ** 2).sum(dim=0) / denominator
        scale = variance.sqrt().clamp_min(1e-6)
        normalized[factor] = ((values - mean) / scale) * masks
        mask_tensors[factor] = masks
        identity[factor] = {
            "mean": mean.detach().cpu().tolist(),
            "scale": scale.detach().cpu().tolist(),
            "observed_count": denominator.detach().cpu().tolist(),
        }
    return normalized, mask_tensors, {
        **identity,
        "normalization_sha256": _semantic_hash(identity),
    }


def _state_dict_hash(model: FactorizedConditionalVAE) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _train_cvae_and_generate(
    *,
    dataset: Mapping[str, Any],
    fit_ids: Sequence[str],
    source_ids: Sequence[str],
    targets_by_case: Mapping[str, Sequence[str]],
    max_targets: int,
    optimizer_steps: int,
    samples_per_target: int,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    steps = _positive_int(optimizer_steps, "cvae_optimizer_steps")
    samples = _positive_int(samples_per_target, "cvae_samples_per_target")
    normalized, masks, normalization = _normalization(dataset, fit_ids, device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = FactorizedConditionalVAE(
        mechanism_dim=normalized["mechanism"].shape[1],
        propagation_dim=normalized["propagation"].shape[1],
        context_dim=normalized["context"].shape[1],
        target_context_dim=normalized["context"].shape[1],
        hidden_width=_FROZEN_SMOKE_PROFILE["hidden_width"],
        mechanism_latent_width=_FROZEN_SMOKE_PROFILE["mechanism_latent_width"],
        propagation_latent_width=_FROZEN_SMOKE_PROFILE["propagation_latent_width"],
        context_latent_width=_FROZEN_SMOKE_PROFILE["context_latent_width"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_FROZEN_SMOKE_PROFILE["learning_rate"],
        weight_decay=_FROZEN_SMOKE_PROFILE["weight_decay"],
    )
    batch = {
        "mechanism": normalized["mechanism"],
        "mechanism_mask": masks["mechanism"],
        "propagation": normalized["propagation"],
        "propagation_mask": masks["propagation"],
        "context": normalized["context"],
        "context_mask": masks["context"],
        "target_context": normalized["context"],
        "target_context_mask": masks["context"],
    }
    weights = {
        "masked_reconstruction": _FROZEN_SMOKE_PROFILE[
            "masked_reconstruction_weight"
        ],
        "kl": _FROZEN_SMOKE_PROFILE["kl_weight"],
        "target_context": _FROZEN_SMOKE_PROFILE["target_context_weight"],
        "cycle_consistency": _FROZEN_SMOKE_PROFILE[
            "cycle_consistency_weight"
        ],
    }
    model.train()
    initial = compute_unlabeled_objectives(
        model=model, batch=batch, weights=weights, sample_seed=42
    )
    gradient_norms = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objectives = compute_unlabeled_objectives(
            model=model, batch=batch, weights=weights, sample_seed=42
        )
        objectives["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), _FROZEN_SMOKE_PROFILE["gradient_clip_norm"]
        )
        gradient_norms.append(float(norm.detach().cpu().item()))
        optimizer.step()
    model.eval()
    final = compute_unlabeled_objectives(
        model=model, batch=batch, weights=weights, sample_seed=42
    )
    checkpoint_sha256 = _state_dict_hash(model)

    # Fit rows are flattened in case/candidate order; retain a deterministic index.
    row_index = {}
    cursor = 0
    for case_id in fit_ids:
        for candidate in dataset["cases"][case_id]["candidate_ids"]:
            row_index[(case_id, str(candidate))] = cursor
            cursor += 1

    source_id = source_ids[0]
    source_root = targets_by_case[source_id][0]
    source_index = row_index[(source_id, source_root)]
    source_inputs = {
        factor: normalized[factor][source_index : source_index + 1]
        for factor in normalized
    }
    source_inputs.update(
        {
            f"{factor}_mask": masks[factor][source_index : source_index + 1]
            for factor in masks
        }
    )
    observable = _observable_by_candidate(dataset["cases"][source_id])
    source_observable = observable[source_root]
    plans = {
        "cvae_arbitrary": _target_plan(
            source_observable,
            observable,
            targets_by_case[source_id],
            max_targets=max_targets,
            compatible=False,
        ),
        "cvae_compatible": _target_plan(
            source_observable,
            observable,
            targets_by_case[source_id],
            max_targets=max_targets,
            compatible=True,
        ),
    }

    decoded_hashes = []
    decoded_contexts = []
    posterior_logvars = []
    with torch.no_grad():
        for target in plans["cvae_arbitrary"]:
            target_index = row_index[(source_id, target["target_service_id"])]
            output = model(
                **source_inputs,
                target_context=normalized["context"][target_index : target_index + 1],
                target_context_mask=masks["context"][target_index : target_index + 1],
                sample=False,
            )
            decoded = output["reconstruction"]
            decoded_hashes.append(
                _semantic_hash(
                    {factor: _tensor_hash(decoded[factor]) for factor in decoded}
                )
            )
            decoded_contexts.append(decoded["context"])
            if not posterior_logvars:
                posterior_logvars = [
                    float(value)
                    for factor in ("mechanism", "propagation", "context")
                    for value in output["posterior"][factor]["logvar"]
                    .detach()
                    .cpu()
                    .flatten()
                    .tolist()
                ]
    source_hash = _semantic_hash(
        {factor: _tensor_hash(source_inputs[factor]) for factor in normalized}
    )
    target_deltas = [
        float(torch.linalg.vector_norm(value - decoded_contexts[0]).cpu().item())
        for value in decoded_contexts[1:]
    ] or [0.0]
    activity = build_cvae_activity_audit(
        profile_id=_FROZEN_SMOKE_PROFILE["profile_id"],
        gradient_norms=gradient_norms,
        reconstruction_history=[
            float(initial["components"]["masked_reconstruction"].detach().cpu()),
            float(final["components"]["masked_reconstruction"].detach().cpu()),
        ],
        kl_history=[
            float(initial["components"]["kl"].detach().cpu()),
            float(final["components"]["kl"].detach().cpu()),
        ],
        posterior_logvars=posterior_logvars,
        source_state_sha256s=[source_hash] * len(decoded_hashes),
        generated_state_sha256s=decoded_hashes,
        target_context_deltas=target_deltas,
        mechanism_latent_distances=[0.0] * len(decoded_hashes),
        sampling_radius=_FROZEN_SMOKE_PROFILE["sampling_radius"],
    )
    if activity["status"] != "mechanistically_active":
        raise RealRankerSmokeValidationError(
            "bounded CVAE is inactive: " + ",".join(activity["rejection_reasons"])
        )
    profile = {
        "profile_id": _FROZEN_SMOKE_PROFILE["profile_id"],
        "sampling_radius": _FROZEN_SMOKE_PROFILE["sampling_radius"],
        "samples_per_target": samples,
    }
    profile["profile_sha256"] = _semantic_hash(profile)
    source_case = {
        "case_id": source_id,
        "queried_label": _queried_label(
            dataset, source_id, targets_by_case[source_id]
        ),
        "observation_case_ids": list(fit_ids),
        "label_case_ids": [source_id],
        **source_inputs,
    }
    arms = {"cvae_arbitrary": [], "cvae_compatible": []}
    for arm_index, arm_id in enumerate(arms):
        target_contexts = []
        for target in plans[arm_id]:
            index = row_index[(source_id, target["target_service_id"])]
            target_contexts.append(
                {
                    **target,
                    "target_context": normalized["context"][index : index + 1],
                    "target_context_mask": masks["context"][index : index + 1],
                }
            )
        bundle = generate_cvae_service_continuity(
            model=model,
            arm_id=arm_id,
            source_case=source_case,
            target_contexts=target_contexts,
            profile=profile,
            sample_seed=42 + arm_index * 100_003,
            checkpoint_sha256=checkpoint_sha256,
            activity_evidence=activity,
        )
        arms[arm_id] = bundle["synthetic_rows"]
    return arms, {
        **activity,
        "optimizer_step_count": steps,
        "checkpoint_sha256": checkpoint_sha256,
        "normalization_sha256": normalization["normalization_sha256"],
        "pretraining_case_count": len(fit_ids),
        "pretraining_candidate_row_count": cursor,
    }


def _build_neural_request(
    *,
    dataset: Mapping[str, Any],
    fit_ids: Sequence[str],
    source_ids: Sequence[str],
    targets_by_case: Mapping[str, Sequence[str]],
    held_out_fault_type: str | None,
    max_targets: int,
    optimizer_steps: int,
    samples_per_target: int,
    device: str,
) -> dict[str, Any]:
    records = []
    for case_id in fit_ids:
        for candidate_id, row in _observable_by_candidate(
            dataset["cases"][case_id]
        ).items():
            records.append(
                {
                    "case_id": case_id,
                    "candidate_id": candidate_id,
                    "state": _state_values(row),
                    "mask": _mask_values(row),
                }
            )
    source_id = str(source_ids[0])
    source_root = str(targets_by_case[source_id][0])
    observable = _observable_by_candidate(dataset["cases"][source_id])
    source_row = observable[source_root]
    plans = {
        "cvae_arbitrary": _target_plan(
            source_row,
            observable,
            targets_by_case[source_id],
            max_targets=max_targets,
            compatible=False,
        ),
        "cvae_compatible": _target_plan(
            source_row,
            observable,
            targets_by_case[source_id],
            max_targets=max_targets,
            compatible=True,
        ),
    }
    identity = {
        "schema_version": "service-continuous-neural-handshake-request-v1",
        "dataset_id": "rcabench",
        "held_out_fault_type": held_out_fault_type,
        "fit_case_ids": list(fit_ids),
        "held_out_case_ids": [],
        "pretraining_records": records,
        "source_case_id": source_id,
        "source_root_service_id": source_root,
        "queried_label": _queried_label(
            dataset, source_id, targets_by_case[source_id]
        ),
        "target_plans": plans,
        "profile": dict(_FROZEN_SMOKE_PROFILE),
        "optimizer_steps": _positive_int(
            optimizer_steps, "cvae_optimizer_steps"
        ),
        "samples_per_target": _positive_int(
            samples_per_target, "cvae_samples_per_target"
        ),
        "training_seed": 42,
        "device": str(device),
    }
    return {**identity, "request_sha256": _semantic_hash(identity)}


def _run_neural_handshake(
    neural_request: Mapping[str, Any],
    *,
    handshake_task: str = "task10.3",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    request_hash = str(neural_request["request_sha256"])
    root = Path(
        os.environ.get(
            "SERVICE_CONTINUOUS_HANDSHAKE_ROOT",
            "outputs/rcl_study/service_continuous_vae_rcl_augmentation/"
            f"smoke/{handshake_task}-neural-handshake",
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / f"input-{request_hash}.json"
    output_path = root / f"output-{request_hash}.json"
    serialized = (
        json.dumps(
            neural_request,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if input_path.exists() and input_path.read_text(encoding="utf-8") != serialized:
        raise RealRankerSmokeValidationError("neural handshake input hash collision")
    input_path.write_text(serialized, encoding="utf-8")
    executable = os.environ.get(
        "SERVICE_CONTINUOUS_TORCH_PYTHON",
        "${TORCH_PYTHON}",
    )
    command = [
        executable,
        "-m",
        "rcl_study.service_continuous_neural_runner",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    environment = dict(os.environ)
    workspace = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(workspace)
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RealRankerSmokeValidationError(
            "neural handshake failed: "
            + (completed.stderr or completed.stdout or "unknown error").strip()
        )
    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealRankerSmokeValidationError(
            "neural handshake output is absent or invalid"
        ) from exc
    request_schema = str(neural_request.get("schema_version", ""))
    expected_result_schema = {
        "service-continuous-neural-handshake-request-v1":
            "service-continuous-neural-handshake-result-v1",
        "service-continuous-neural-handshake-request-v2":
            "service-continuous-neural-handshake-result-v2",
    }.get(request_schema)
    expected_arms = (
        {"cvae_arbitrary", "cvae_compatible"}
        if request_schema == "service-continuous-neural-handshake-request-v1"
        else {
            "cvae_arbitrary",
            "cvae_compatible",
            "proxy_mode_cvae_compatible",
        }
    )
    if (
        expected_result_schema is None
        or
        not isinstance(result, Mapping)
        or result.get("schema_version")
        != expected_result_schema
        or result.get("request_sha256") != request_hash
        or result.get("status") != "completed"
        or set(result.get("arms", {})) != expected_arms
    ):
        raise RealRankerSmokeValidationError("neural handshake result closure drift")
    activity = dict(result["activity_audit"])
    if neural_request.get("checkpoint_output_path") is not None:
        artifact = result.get("checkpoint_artifact")
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("path")
            != str(Path(str(neural_request["checkpoint_output_path"])).resolve())
            or artifact.get("state_dict_sha256")
            != activity.get("checkpoint_sha256")
            or len(str(artifact.get("file_sha256", ""))) != 64
            or not Path(str(artifact.get("path", ""))).is_file()
        ):
            raise RealRankerSmokeValidationError(
                "neural checkpoint artifact closure drift"
            )
        activity["checkpoint_artifact"] = dict(artifact)
    return {
        arm_id: [dict(row) for row in rows]
        for arm_id, rows in result["arms"].items()
    }, activity


def _decoded_values(row: Mapping[str, Any], factor: str) -> list[float]:
    if row.get("arm_id", "").startswith("explicit_"):
        raw = row[f"{factor}_state"]["values"]
    else:
        raw = row[f"decoded_{factor}"]
    values = [float(value) for value in raw]
    if not values or not all(math.isfinite(value) for value in values):
        raise RealRankerSmokeValidationError("generated state is empty or non-finite")
    return values


def _adapt_candidate_rows(
    *,
    generated_row: Mapping[str, Any],
    dataset: Mapping[str, Any],
    feature_bounds: tuple[np.ndarray, np.ndarray],
    targets_by_case: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    source_id = str(generated_row["source_case_id"])
    target_id = str(generated_row["target_service_id"])
    case = dataset["cases"][source_id]
    candidates = [str(value) for value in case["candidate_ids"]]
    if target_id not in candidates:
        raise RealRankerSmokeValidationError("generated target is outside source candidates")
    source_root = str(targets_by_case[source_id][0])
    matrix = np.asarray(case["base_feature_rows"], dtype=float).copy()
    if matrix.shape != (54, 122) or not np.isfinite(matrix).all():
        raise RealRankerSmokeValidationError("authority base feature matrix drift")
    source_index = candidates.index(source_root)
    target_index = candidates.index(target_id)
    source_vector = matrix[source_index].copy()
    target_vector = matrix[target_index].copy()
    alpha = float(generated_row.get("interpolation_alpha", 1.0))
    if not 0.0 < alpha <= 1.0:
        raise RealRankerSmokeValidationError("generated interpolation alpha drift")
    state_values = [
        *_decoded_values(generated_row, "mechanism"),
        *_decoded_values(generated_row, "propagation"),
        *_decoded_values(generated_row, "context"),
    ]
    scale = 1.0 + 0.1 * math.tanh(math.fsum(state_values) / len(state_values))
    symptom = list(_SYMPTOM_INDICES)
    migrated = (1.0 - alpha) * target_vector[symptom] + alpha * source_vector[symptom]
    matrix[target_index, symptom] = migrated * scale
    matrix[source_index, symptom] = (
        (1.0 - alpha) * source_vector[symptom] + alpha * target_vector[symptom]
    )
    matrix[target_index, list(_IDENTITY_CONTEXT_INDICES)] = target_vector[
        list(_IDENTITY_CONTEXT_INDICES)
    ]
    lower, upper = feature_bounds
    matrix = np.minimum(upper, np.maximum(lower, matrix))
    if not np.isfinite(matrix).all():
        raise RealRankerSmokeValidationError("adapted candidate features are non-finite")
    feature_names = tuple(str(value) for value in dataset["base_feature_names"])
    candidate_rows = [
        {
            "candidate_id": candidate,
            **{
                feature: float(value)
                for feature, value in zip(feature_names, matrix[index].tolist())
            },
        }
        for index, candidate in enumerate(candidates)
    ]
    generator_hash = str(generated_row["synthetic_row_sha256"])
    adapter_identity = {
        "schema_version": "service-continuous-state-to-authority-feature-adapter-v1",
        "generator_row_sha256": generator_hash,
        "source_case_id": source_id,
        "source_root_service_id": source_root,
        "target_service_id": target_id,
        "candidate_ids": candidates,
        "feature_names": list(feature_names),
        "interpolation_alpha": alpha,
        "state_scale": scale,
        "candidate_rows_sha256": _semantic_hash(candidate_rows),
    }
    adapted_hash = _semantic_hash(adapter_identity)
    provisional = generated_row.get("provisional_training_weight")
    if provisional is None:
        provisional = generated_row.get("per_step_target_mass")
    return {
        **deepcopy(dict(generated_row)),
        "generator_row_sha256": generator_hash,
        "synthetic_row_sha256": adapted_hash,
        "provisional_training_weight": float(provisional),
        "adapter_sha256": adapted_hash,
        "candidate_rows": candidate_rows,
    }


def _feature_bounds(
    dataset: Mapping[str, Any], fit_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [
            row
            for case_id in fit_ids
            for row in dataset["cases"][case_id]["base_feature_rows"]
        ],
        dtype=float,
    )
    if matrix.shape != (len(fit_ids) * 54, 122) or not np.isfinite(matrix).all():
        raise RealRankerSmokeValidationError("fold-local feature bounds input drift")
    return matrix.min(axis=0), matrix.max(axis=0)


def _authority_rankings(bundle: Mapping[str, Any]) -> dict[str, list[str]]:
    partitions = bundle.get("partitions")
    if not isinstance(partitions, Mapping):
        raise RealRankerSmokeValidationError("authority score partitions are absent")
    rows = partitions.get("evaluation")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise RealRankerSmokeValidationError("authority evaluation partition drift")
    result = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        ranking = [str(value) for value in row.get("ranking", ())]
        if not case_id or len(ranking) != 54:
            raise RealRankerSmokeValidationError("authority ranking closure drift")
        result[case_id] = ranking
    return result


def _rankings_from_score(
    score_artifact: Mapping[str, Any], targets_by_case: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "candidate_count": len(ranking),
            "ranking_count": len(ranking),
            "ranking": list(ranking),
            "targets": list(targets_by_case[case_id]),
        }
        for case_id, ranking in score_artifact["rankings_by_case"].items()
    ]


def _run_bounded_materialized_ranker_smoke(
    *,
    prepared: Mapping[str, Any],
    evaluation_protocol: str,
    held_out_fault_type: str | None,
    ordinary: bool,
    evaluation_case_limit: int = 3,
    max_sources: int = 1,
    max_targets: int = 2,
    cvae_optimizer_steps: int = 8,
    cvae_samples_per_target: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    protocol = str(evaluation_protocol).strip()
    if protocol not in {
        "strict_lofo_bounded_smoke",
        "ordinary_query_only_bounded_smoke",
    }:
        raise RealRankerSmokeValidationError("bounded evaluation protocol drift")
    if ordinary != (protocol == "ordinary_query_only_bounded_smoke"):
        raise RealRankerSmokeValidationError("ordinary protocol flag drift")
    dataset = prepared["dataset"]
    fit_ids = tuple(str(value) for value in prepared["fit_case_ids"])
    query_budget = _positive_int(prepared.get("budget"), "query budget")
    full_evaluation_ids = tuple(
        str(value) for value in prepared["evaluation_case_ids"]
    )
    evaluation_limit = min(
        _positive_int(evaluation_case_limit, "evaluation_case_limit"),
        len(full_evaluation_ids),
    )
    evaluation_ids = full_evaluation_ids[:evaluation_limit]
    source_limit = min(_positive_int(max_sources, "max_sources"), len(fit_ids))
    source_ids = fit_ids[:source_limit]
    requested_device = str(device).strip()
    if not requested_device:
        raise RealRankerSmokeValidationError("device must be nonempty")

    real_rows = _candidate_rows(dataset, fit_ids)
    real_targets = _targets(dataset, fit_ids)
    evaluation_rows = _candidate_rows(dataset, evaluation_ids)
    evaluation_targets = _targets(dataset, evaluation_ids)
    expected_candidates = {
        case_id: tuple(str(value) for value in dataset["cases"][case_id]["candidate_ids"])
        for case_id in evaluation_ids
    }
    feature_names = tuple(str(value) for value in dataset["base_feature_names"])
    query_hash = str(prepared["query_plan_sha256"])
    membership_hash = _semantic_hash(list(evaluation_ids))
    bounds = _feature_bounds(dataset, fit_ids)

    explicit = _explicit_rows(
        dataset=dataset,
        fit_ids=fit_ids,
        source_ids=source_ids,
        targets_by_case=real_targets,
        max_targets=max_targets,
    )
    neural_request = _build_neural_request(
        dataset=dataset,
        fit_ids=fit_ids,
        source_ids=source_ids,
        targets_by_case=real_targets,
        held_out_fault_type=held_out_fault_type,
        max_targets=max_targets,
        optimizer_steps=cvae_optimizer_steps,
        samples_per_target=cvae_samples_per_target,
        device=requested_device,
    )
    cvae, cvae_activity = _run_neural_handshake(
        neural_request,
        handshake_task="task10.4" if ordinary else "task10.3",
    )
    generated_by_arm = {**explicit, **cvae}
    real_query_rows = [
        {
            "case_id": case_id,
            "query_budget_cost": 1,
            "training_weight": 1.0,
        }
        for case_id in fit_ids
    ]

    arms: dict[str, Any] = {}
    baseline_bridge = fit_service_continuity_pairwise_bridge(
        dataset_id="rcabench",
        arm_id="baseline",
        augmentation_enabled=False,
        real_training_rows=real_rows,
        real_targets_by_case=real_targets,
        synthetic_rows=[],
        feature_columns=feature_names,
        query_plan_sha256=query_hash,
        random_state=42,
        generation_disposition={
            "generated_count": 0,
            "dropped_count": 0,
            "invalid_count": 0,
        },
        query_budget=query_budget,
    )
    bridges = {"baseline": baseline_bridge}
    generation_activity = {
        "baseline": {
            "active": False,
            "synthetic_row_count": 0,
            "unique_generated_state_count": 0,
        }
    }
    for arm_id in _ARMS[1:]:
        adapted = [
            _adapt_candidate_rows(
                generated_row=row,
                dataset=dataset,
                feature_bounds=bounds,
                targets_by_case=real_targets,
            )
            for row in generated_by_arm[arm_id]
        ]
        training_ledger = build_synthetic_training_ledger(
            real_query_rows=real_query_rows,
            synthetic_rows=adapted,
            query_budget=query_budget,
        )
        admission = build_synthetic_admission_ledger(
            synthetic_rows=training_ledger["synthetic_rows"],
            evaluation_membership={
                "validation": [],
                "strict_lofo_test": [] if ordinary else list(evaluation_ids),
                "ordinary_test": list(evaluation_ids) if ordinary else [],
                "t1": list(evaluation_ids),
                "t2": [],
            },
            official_denominator_membership={
                "strict_lofo_test": [] if ordinary else list(evaluation_ids),
                "ordinary_test": list(evaluation_ids) if ordinary else [],
                "t1": list(evaluation_ids),
                "t2": [],
            },
        )
        synthetics = admission["train_only_rows"]
        bridges[arm_id] = fit_service_continuity_pairwise_bridge(
            dataset_id="rcabench",
            arm_id=arm_id,
            augmentation_enabled=True,
            real_training_rows=real_rows,
            real_targets_by_case=real_targets,
            synthetic_rows=synthetics,
            feature_columns=feature_names,
            query_plan_sha256=query_hash,
            random_state=42,
            generation_disposition={
                "generated_count": len(synthetics),
                "dropped_count": 0,
                "invalid_count": 0,
            },
            query_budget=query_budget,
        )
        state_hash_field = (
            "synthetic_row_sha256"
            if arm_id.startswith("explicit_")
            else "decoded_state_sha256"
        )
        generation_activity[arm_id] = {
            "active": True,
            "generator_kind": (
                "deterministic_service_continuity"
                if arm_id.startswith("explicit_")
                else "factorized_conditional_vae"
            ),
            "synthetic_row_count": len(synthetics),
            "unique_generated_state_count": len(
                {str(row.get(state_hash_field, row["synthetic_row_sha256"])) for row in synthetics}
            ),
            "synthetic_training_ledger_sha256": training_ledger["ledger_sha256"],
            "synthetic_admission_ledger_sha256": admission["ledger_sha256"],
        }

    for arm_id, fitted in bridges.items():
        score_artifact = score_base_score_bridge(
            fitted.bridge,
            evaluation_rows,
            evaluation_targets,
            protocol,
            expected_candidates_by_case=expected_candidates,
        )
        rankings = _rankings_from_score(score_artifact, evaluation_targets)
        if ordinary:
            count_evidence = build_ordinary_count_score(rankings)
            validate_ordinary_score(count_evidence)
        else:
            count_evidence = build_count_authoritative_score(
                str(held_out_fault_type), rankings
            )
            validate_type_score(count_evidence)
        candidate_complete = all(
            row["candidate_count"] == row["ranking_count"] == 54
            and set(row["ranking"]) == set(expected_candidates[row["case_id"]])
            for row in rankings
        )
        if not candidate_complete:
            raise RealRankerSmokeValidationError(
                f"{arm_id} produced candidate-incomplete rankings"
            )
        arms[arm_id] = {
            "query_plan_sha256": query_hash,
            "real_training_rows_sha256": fitted.training_audit[
                "real_training_rows_sha256"
            ],
            "evaluation_membership_sha256": membership_hash,
            "training_audit": fitted.training_audit,
            "generator_activity": generation_activity[arm_id],
            "candidate_complete_rankings": True,
            "count_reconstruction_valid": True,
            "per_case_rankings": rankings,
            "count_evidence": count_evidence,
            "score_artifact_sha256": score_artifact["score_artifact_sha256"],
        }

    authority_bundle = prepared["authority_score_bundle"]
    authority_rankings = _authority_rankings(authority_bundle)
    baseline_rankings = {
        row["case_id"]: row["ranking"]
        for row in arms["baseline"]["per_case_rankings"]
    }
    parity = {
        "model_sha256_equal": baseline_bridge.bridge.model_sha256
        == authority_bundle["model_sha256"],
        "feature_order_sha256_equal": baseline_bridge.bridge.feature_order_sha256
        == authority_bundle["feature_order_sha256"],
        "ranking_equal_for_scored_cases": all(
            baseline_rankings[case_id] == authority_rankings[case_id]
            for case_id in evaluation_ids
        ),
    }
    if not all(parity.values()):
        raise RealRankerSmokeValidationError("baseline authority parity failed")

    if ordinary:
        protocol_checks = {}
        baseline_score = arms["baseline"]["count_evidence"]
        for arm_id in _ARMS[1:]:
            gate = evaluate_ordinary_safety_gate(
                arm_id=arm_id,
                baseline_score=baseline_score,
                arm_score=arms[arm_id]["count_evidence"],
            )
            protocol_checks[arm_id] = {
                "bounded_protocol_only": True,
                "decision_authority": False,
                "count_schema_consumed": True,
                "gate_payload": gate,
            }
        identity = {
            "schema_version": (
                "service-continuous-bounded-ordinary-ranker-smoke-v1"
            ),
            "bounded_not_performance_result": True,
            "bounded_not_official_score": True,
            "official_ordinary_gate_decision_permitted": False,
            "dataset_id": "rcabench",
            "evaluation_protocol": protocol,
            "held_out_fault_type": None,
            "query_plan_sha256": query_hash,
            "authority_score_bundle_sha256": prepared[
                "authority_score_bundle_sha256"
            ],
            "real_query_budget_count": len(fit_ids),
            "synthetic_query_budget_count": 0,
            "fit_evaluation_overlap_count": len(
                set(fit_ids) & set(full_evaluation_ids)
            ),
            "base_feature_count": len(feature_names),
            "fit_case_ids": list(fit_ids),
            "source_case_ids": list(source_ids),
            "full_evaluation_case_count": len(full_evaluation_ids),
            "evaluation_case_ids": list(evaluation_ids),
            "evaluation_case_count": len(evaluation_ids),
            "candidate_count_by_case": {
                case_id: len(expected_candidates[case_id])
                for case_id in evaluation_ids
            },
            "baseline_authority_parity": parity,
            "cvae_activity_audit": cvae_activity,
            "ordinary_gate_protocol_checks": protocol_checks,
            "arms": arms,
        }
        return {**identity, "smoke_sha256": _semantic_hash(identity)}

    identity = {
        "schema_version": "service-continuous-real-ranker-smoke-v1",
        "bounded_not_performance_result": True,
        "dataset_id": "rcabench",
        "held_out_fault_type": "NetworkDelay",
        "query_plan_sha256": query_hash,
        "authority_score_bundle_sha256": prepared[
            "authority_score_bundle_sha256"
        ],
        "real_query_budget_count": len(fit_ids),
        "synthetic_query_budget_count": 0,
        "held_out_fitting_overlap_count": 0,
        "base_feature_count": len(feature_names),
        "fit_case_ids": list(fit_ids),
        "source_case_ids": list(source_ids),
        "evaluation_case_ids": list(evaluation_ids),
        "evaluation_case_count": len(evaluation_ids),
        "candidate_count_by_case": {
            case_id: len(expected_candidates[case_id]) for case_id in evaluation_ids
        },
        "baseline_authority_parity": parity,
        "cvae_activity_audit": cvae_activity,
        "arms": arms,
    }
    return {**identity, "smoke_sha256": _semantic_hash(identity)}


def run_bounded_real_ranker_smoke(
    *,
    request: Mapping[str, Any],
    evaluation_case_limit: int = 3,
    max_sources: int = 1,
    max_targets: int = 2,
    cvae_optimizer_steps: int = 8,
    cvae_samples_per_target: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a bounded real NetworkDelay fold through baseline plus four arms."""

    return _run_bounded_materialized_ranker_smoke(
        prepared=_request(request),
        evaluation_protocol="strict_lofo_bounded_smoke",
        held_out_fault_type="NetworkDelay",
        ordinary=False,
        evaluation_case_limit=evaluation_case_limit,
        max_sources=max_sources,
        max_targets=max_targets,
        cvae_optimizer_steps=cvae_optimizer_steps,
        cvae_samples_per_target=cvae_samples_per_target,
        device=device,
    )


__all__ = [
    "RealRankerSmokeValidationError",
    "run_bounded_real_ranker_smoke",
]
