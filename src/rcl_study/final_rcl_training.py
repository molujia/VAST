from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .service_continuous_synthetic_training import (
    SyntheticTrainingValidationError,
    build_synthetic_training_ledger,
)


class FinalRCLTrainingError(ValueError):
    """Raised when final-method supervision or synthetic training drifts."""


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _ids(value: Sequence[Any], context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalRCLTrainingError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if not result or "" in result or len(result) != len(set(result)):
        raise FinalRCLTrainingError(f"{context} must be unique and nonempty")
    return result


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise FinalRCLTrainingError(f"{context} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalRCLTrainingError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise FinalRCLTrainingError(f"{context} must be finite")
    return result


def build_final_synthetic_training_ledger(
    *,
    real_supervision_rows: Sequence[Mapping[str, Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
    training_mode: str,
    expected_supervised_case_ids: Sequence[Any],
    query_budget: int = 30,
) -> dict[str, Any]:
    """Normalize CVAE descendants under query-only or oracle-full ownership."""

    mode = str(training_mode)
    if mode not in {"query_only", "oracle_full"}:
        raise FinalRCLTrainingError("training_mode must be query_only or oracle_full")
    expected_ids = _ids(expected_supervised_case_ids, "supervised membership")
    real = [deepcopy(dict(row)) for row in real_supervision_rows]
    real_ids = [str(row.get("case_id", "")).strip() for row in real]
    if real_ids != expected_ids:
        raise FinalRCLTrainingError("real supervision membership or order drifted")
    if any(_finite(row.get("training_weight"), "real training weight") != 1.0 for row in real):
        raise FinalRCLTrainingError("every real supervised case must have weight 1.0")
    if isinstance(query_budget, bool) or not isinstance(query_budget, int) or query_budget <= 0:
        raise FinalRCLTrainingError("query_budget must be a positive integer")
    costs = [row.get("query_budget_cost") for row in real]
    if mode == "query_only":
        if len(real) != query_budget or any(cost != 1 for cost in costs):
            raise FinalRCLTrainingError(
                "query_only supervision must exactly consume the query budget"
            )
    elif any(cost != 0 for cost in costs):
        raise FinalRCLTrainingError("oracle_full supervision must have zero query cost")

    normalization_rows = [
        {"case_id": case_id, "query_budget_cost": 1, "training_weight": 1.0}
        for case_id in real_ids
    ]
    try:
        normalized = build_synthetic_training_ledger(
            real_query_rows=normalization_rows,
            synthetic_rows=synthetic_rows,
            query_budget=len(normalization_rows),
            source_total_mass_by_case={
                case_id: 1.0
                for case_id in real_ids
                if any(
                    str(row.get("source_case_id", "")) == case_id
                    for row in synthetic_rows
                )
            },
            max_source_mass=1.0,
        )
    except SyntheticTrainingValidationError as exc:
        raise FinalRCLTrainingError(f"invalid synthetic descendants: {exc}") from exc

    identity = {
        "schema_version": "final-rcl-synthetic-training-ledger-v1",
        "training_mode": mode,
        "supervised_case_ids": real_ids,
        "real_supervised_count": len(real_ids),
        "supervised_case_limit": len(real_ids),
        "query_budget": query_budget if mode == "query_only" else None,
        "real_query_budget_cost": sum(int(cost) for cost in costs),
        "real_case_training_weights": {case_id: 1.0 for case_id in real_ids},
        "synthetic_query_budget_cost": normalized["synthetic_query_budget_cost"],
        "synthetic_row_count": normalized["synthetic_row_count"],
        "synthetic_source_count": normalized["synthetic_source_count"],
        "synthetic_child_count_by_source": normalized[
            "synthetic_child_count_by_source"
        ],
        "synthetic_weight_sum_by_source": normalized[
            "synthetic_weight_sum_by_source"
        ],
        "synthetic_target_mass_by_source": normalized[
            "synthetic_target_mass_by_source"
        ],
        "source_weight_plan_sha256s": normalized["source_weight_plan_sha256s"],
        "weight_normalization_sha256": normalized[
            "weight_normalization_sha256"
        ],
        "synthetic_rows": normalized["synthetic_rows"],
        "normalization_ledger_sha256": normalized["ledger_sha256"],
    }
    return {**identity, "ledger_sha256": _semantic_hash(identity)}


def adapt_generated_candidate_case(
    *,
    generated_row: Mapping[str, Any],
    source_case: Mapping[str, Any],
    source_root_service_id: str,
    feature_names: Sequence[Any],
    feature_lower_bounds: Sequence[Any],
    feature_upper_bounds: Sequence[Any],
    symptom_feature_indices: Sequence[Any],
    identity_context_indices: Sequence[Any],
) -> dict[str, Any]:
    """Map one decoded service transfer into a candidate-complete ranker case."""

    row = deepcopy(dict(generated_row))
    source_id = str(row.get("source_case_id", "")).strip()
    target_id = str(row.get("target_service_id", "")).strip()
    source_root = str(source_root_service_id).strip()
    candidates = _ids(source_case.get("candidate_ids", ()), "candidate IDs")
    if not source_id or target_id not in candidates:
        raise FinalRCLTrainingError("generated target is outside the candidate set")
    if source_root not in candidates:
        raise FinalRCLTrainingError("source root is outside the candidate set")
    names = _ids(feature_names, "feature names")
    matrix = [list(values) for values in source_case.get("base_feature_rows", ())]
    if len(matrix) != len(candidates) or any(len(values) != len(names) for values in matrix):
        raise FinalRCLTrainingError("candidate feature matrix shape drifted")
    values = [[_finite(value, "candidate feature") for value in line] for line in matrix]
    lower = [_finite(value, "feature lower bound") for value in feature_lower_bounds]
    upper = [_finite(value, "feature upper bound") for value in feature_upper_bounds]
    if len(lower) != len(names) or len(upper) != len(names) or any(
        low > high for low, high in zip(lower, upper)
    ):
        raise FinalRCLTrainingError("feature bounds drifted")
    symptom = [int(index) for index in symptom_feature_indices]
    identity_context = [int(index) for index in identity_context_indices]
    if (
        not symptom
        or len(symptom) != len(set(symptom))
        or len(identity_context) != len(set(identity_context))
        or any(index < 0 or index >= len(names) for index in symptom + identity_context)
    ):
        raise FinalRCLTrainingError("feature index contract drifted")
    decoded = []
    for factor in ("mechanism", "propagation", "context"):
        factor_values = row.get(f"decoded_{factor}")
        if isinstance(factor_values, (str, bytes)) or not isinstance(
            factor_values, Sequence
        ):
            raise FinalRCLTrainingError("generated decoded state is incomplete")
        decoded.extend(_finite(value, "decoded state") for value in factor_values)
    if not decoded:
        raise FinalRCLTrainingError("generated decoded state is empty")
    alpha = _finite(row.get("interpolation_alpha", 1.0), "interpolation alpha")
    if not 0.0 < alpha <= 1.0:
        raise FinalRCLTrainingError("interpolation alpha must lie in (0,1]")
    scale = 1.0 + 0.1 * math.tanh(math.fsum(decoded) / len(decoded))
    source_index = candidates.index(source_root)
    target_index = candidates.index(target_id)
    original_source = list(values[source_index])
    original_target = list(values[target_index])
    for index in symptom:
        values[target_index][index] = (
            (1.0 - alpha) * original_target[index] + alpha * original_source[index]
        ) * scale
        values[source_index][index] = (
            (1.0 - alpha) * original_source[index] + alpha * original_target[index]
        )
    for index in identity_context:
        values[target_index][index] = original_target[index]
    values = [
        [min(high, max(low, item)) for item, low, high in zip(line, lower, upper)]
        for line in values
    ]
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            **{name: float(value) for name, value in zip(names, line)},
        }
        for candidate_id, line in zip(candidates, values)
    ]
    generator_hash = str(row.get("synthetic_row_sha256", ""))
    if len(generator_hash) != 64:
        raise FinalRCLTrainingError("generator row SHA-256 drifted")
    adapter_identity = {
        "schema_version": "final-rcl-state-to-candidate-feature-adapter-v1",
        "generator_row_sha256": generator_hash,
        "source_case_id": source_id,
        "source_root_service_id": source_root,
        "target_service_id": target_id,
        "candidate_ids": candidates,
        "feature_names": names,
        "symptom_feature_indices": symptom,
        "identity_context_indices": identity_context,
        "interpolation_alpha": alpha,
        "state_scale": scale,
        "candidate_rows_sha256": _semantic_hash(candidate_rows),
    }
    adapter_hash = _semantic_hash(adapter_identity)
    provisional = _finite(
        row.get("provisional_training_weight"), "provisional training weight"
    )
    if provisional <= 0.0:
        raise FinalRCLTrainingError("provisional training weight must be positive")
    return {
        **row,
        "generator_row_sha256": generator_hash,
        "synthetic_row_sha256": adapter_hash,
        "provisional_training_weight": provisional,
        "adapter_sha256": adapter_hash,
        "candidate_rows": candidate_rows,
    }


def build_final_neural_request(
    *,
    dataset_id: str,
    training_mode: str,
    fit_case_ids: Sequence[Any],
    supervised_case_ids: Sequence[Any],
    held_out_case_ids: Sequence[Any],
    expected_candidates_per_case: int,
    pretraining_records: Sequence[Mapping[str, Any]],
    source_cases: Sequence[Mapping[str, Any]],
    proxy_mode_partition: Mapping[str, Any],
    profile: Mapping[str, Any],
    optimizer_steps: int,
    samples_per_target: int,
    device: str,
    checkpoint_output_path: str | None = None,
) -> dict[str, Any]:
    """Build the final HDBSCAN-only neural request for query or oracle training."""

    from .final_rcl_hdbscan_proxy import validate_hdbscan_proxy_partition

    dataset = str(dataset_id)
    if dataset not in {"rcabench", "aiops2022_pre"}:
        raise FinalRCLTrainingError("unsupported final neural dataset")
    mode = str(training_mode)
    if mode not in {"query_only", "oracle_full"}:
        raise FinalRCLTrainingError("invalid final neural training mode")
    fit_ids = _ids(fit_case_ids, "fit case IDs")
    supervised_ids = _ids(supervised_case_ids, "supervised case IDs")
    held_out_ids = (
        []
        if len(held_out_case_ids) == 0
        else _ids(held_out_case_ids, "held-out case IDs")
    )
    if not set(supervised_ids) <= set(fit_ids) or set(fit_ids) & set(held_out_ids):
        raise FinalRCLTrainingError("final neural membership overlap or ownership drift")
    if mode == "oracle_full" and supervised_ids != fit_ids:
        raise FinalRCLTrainingError("oracle_full must supervise the complete fit pool")
    if isinstance(expected_candidates_per_case, bool) or int(expected_candidates_per_case) <= 0:
        raise FinalRCLTrainingError("expected candidate count must be positive")
    if isinstance(optimizer_steps, bool) or int(optimizer_steps) <= 0:
        raise FinalRCLTrainingError("optimizer_steps must be positive")
    if isinstance(samples_per_target, bool) or int(samples_per_target) <= 0:
        raise FinalRCLTrainingError("samples_per_target must be positive")
    records = [deepcopy(dict(row)) for row in pretraining_records]
    candidates_by_case = {case_id: [] for case_id in fit_ids}
    record_keys: set[tuple[str, str]] = set()
    for row in records:
        case_id = str(row.get("case_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        key = (case_id, candidate_id)
        if case_id not in candidates_by_case or not candidate_id or key in record_keys:
            raise FinalRCLTrainingError("pretraining candidate ownership drifted")
        record_keys.add(key)
        candidates_by_case[case_id].append(candidate_id)
    expected_count = int(expected_candidates_per_case)
    if any(len(values) != expected_count for values in candidates_by_case.values()):
        raise FinalRCLTrainingError("pretraining candidate coverage drifted")
    candidate_set = set(candidates_by_case[fit_ids[0]])
    if len(candidate_set) != expected_count or any(
        set(values) != candidate_set for values in candidates_by_case.values()
    ):
        raise FinalRCLTrainingError("dataset candidate set drifted across fit cases")

    sources = [deepcopy(dict(row)) for row in source_cases]
    if [str(row.get("source_case_id", "")) for row in sources] != supervised_ids:
        raise FinalRCLTrainingError("source cases must match supervised membership in order")
    for source in sources:
        label = dict(source.get("queried_label", {}))
        expected_label_source = "queried_budget" if mode == "query_only" else "oracle_full"
        expected_cost = 1 if mode == "query_only" else 0
        if (
            label.get("label_source") != expected_label_source
            or label.get("budget_cost") != expected_cost
        ):
            raise FinalRCLTrainingError("source label ownership drifted")
        plans = source.get("target_plans")
        if not isinstance(plans, Mapping) or set(plans) != {
            "proxy_mode_cvae_compatible"
        }:
            raise FinalRCLTrainingError("final neural request permits only proxy CVAE")
        targets = plans["proxy_mode_cvae_compatible"]
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise FinalRCLTrainingError("proxy CVAE target plan must be a sequence")
        target_ids = [
            str(dict(target).get("target_service_id", "")).strip()
            for target in targets
        ]
        if len(target_ids) != len(set(target_ids)) or set(target_ids) != candidate_set:
            raise FinalRCLTrainingError(
                "every dataset candidate requires one compatible-target decision"
            )
    partition = deepcopy(dict(proxy_mode_partition))
    try:
        validate_hdbscan_proxy_partition(
            partition,
            expected_dataset_id=dataset,
            expected_fit_case_ids=fit_ids,
            expected_queried_case_ids=supervised_ids,
            expected_held_out_case_ids=held_out_ids,
            expected_representation_matrix_sha256=str(
                partition.get("representation_matrix_sha256", "")
            ),
            expected_geometry_sha256=str(partition.get("geometry_sha256", "")),
            expected_active_partition_sha256=str(
                partition.get("active_partition_sha256", "")
            ),
        )
    except ValueError as exc:
        raise FinalRCLTrainingError(f"HDBSCAN proxy partition drift: {exc}") from exc
    if not isinstance(profile, Mapping) or profile.get("profile_id") != "balanced":
        raise FinalRCLTrainingError("final neural profile must be balanced")
    identity = {
        "schema_version": "service-continuous-neural-handshake-request-v3",
        "dataset_id": dataset,
        "training_mode": mode,
        "fit_case_ids": fit_ids,
        "supervised_case_ids": supervised_ids,
        "held_out_case_ids": held_out_ids,
        "expected_candidates_per_case": int(expected_candidates_per_case),
        "pretraining_records": records,
        "source_cases": sources,
        "proxy_mode_partition": partition,
        "profile": deepcopy(dict(profile)),
        "optimizer_steps": int(optimizer_steps),
        "samples_per_target": int(samples_per_target),
        "training_seed": 42,
        "device": str(device),
    }
    if checkpoint_output_path is not None:
        identity["checkpoint_output_path"] = str(checkpoint_output_path)
    return {**identity, "request_sha256": _semantic_hash(identity)}


__all__ = [
    "FinalRCLTrainingError",
    "adapt_generated_candidate_case",
    "build_final_neural_request",
    "build_final_synthetic_training_ledger",
]
