from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .conservative_lofo_base_bridge import (
    BaseScoreBridge,
    _backend_feature_columns,
    _candidate_ids,
    _model_payload,
    _normalize_rows,
    _normalize_targets,
    _rename_features_for_backend,
    fit_base_score_bridge,
)
from nexusrcl_rebuild.training.pairwise_backend import train_pairwise_linear_ranker


class ServiceContinuityPairwiseValidationError(ValueError):
    """Raised when an augmentation arm drifts from the authority ranker contract."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = {
    "explicit_arbitrary",
    "explicit_compatible",
    "cvae_arbitrary",
    "cvae_compatible",
    "proxy_mode_cvae_compatible",
    "repaired_proxy_base",
    "iv_selective",
    "plofo_selective",
    "residual_proxy_endpoint",
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


def _rows(value: Any, context: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ServiceContinuityPairwiseValidationError(
            f"{context} must be a sequence"
        )
    result = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ServiceContinuityPairwiseValidationError(
                f"{context} entries must be mappings"
            )
        result.append(deepcopy(dict(raw)))
    return result


def _generation_disposition(value: Any) -> dict[str, int]:
    expected = {"generated_count", "dropped_count", "invalid_count"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ServiceContinuityPairwiseValidationError(
            "generation disposition closure drift"
        )
    result = {}
    for field in sorted(expected):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ServiceContinuityPairwiseValidationError(
                "generation disposition counts must be nonnegative integers"
            )
        result[field] = raw
    return result


def _real_frame(
    rows: Sequence[Mapping[str, Any]],
    targets_by_case: Mapping[str, Sequence[Any]],
    feature_columns: Sequence[str],
    query_budget: int,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    frame = _normalize_rows(rows, feature_columns)
    targets = _normalize_targets(targets_by_case)
    candidates = _candidate_ids(frame)
    if len(candidates) != query_budget or set(candidates) != set(targets):
        raise ServiceContinuityPairwiseValidationError(
            "authority real-case count must equal query_budget"
        )
    labels = []
    for row in frame.itertuples(index=False):
        case_id = str(row.window_id)
        candidate_id = str(row.entity_id)
        if not set(targets[case_id]) <= set(candidates[case_id]):
            raise ServiceContinuityPairwiseValidationError(
                f"real target is absent from candidates for {case_id}"
            )
        labels.append(int(candidate_id in set(targets[case_id])))
    frame["label"] = labels
    frame["sample_weight"] = 1.0
    frame["pair_weight_mode"] = "legacy_query"
    frame["label_source"] = "queried_groundtruth"
    return frame, targets, candidates


@dataclass
class ServiceContinuityPairwiseBridge:
    bridge: BaseScoreBridge
    training_audit: dict[str, Any]


def fit_service_continuity_pairwise_bridge(
    *,
    dataset_id: str,
    arm_id: str,
    augmentation_enabled: bool,
    real_training_rows: Sequence[Mapping[str, Any]],
    real_targets_by_case: Mapping[str, Sequence[Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    query_plan_sha256: str,
    random_state: int,
    generation_disposition: Mapping[str, Any],
    query_budget: int = 30,
) -> ServiceContinuityPairwiseBridge:
    disposition = _generation_disposition(generation_disposition)
    synthetics = _rows(synthetic_rows, "synthetic_rows")
    indexed = ["generator_order_index" in row for row in synthetics]
    if any(indexed):
        if not all(indexed):
            raise ServiceContinuityPairwiseValidationError(
                "synthetic generator-order metadata is incomplete"
            )
        indices = [row["generator_order_index"] for row in synthetics]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in indices
            )
            or len(indices) != len(set(indices))
        ):
            raise ServiceContinuityPairwiseValidationError(
                "synthetic generator-order metadata is invalid"
            )
        synthetics.sort(
            key=lambda row: (
                int(row["generator_order_index"]),
                str(row.get("synthetic_row_sha256", "")),
            )
        )
    arm = str(arm_id).strip()
    if int(random_state) != 42:
        raise ServiceContinuityPairwiseValidationError(
            "pairwise authority random_state must be 42"
        )
    query_hash = str(query_plan_sha256).strip()
    if not _SHA256.fullmatch(query_hash):
        raise ServiceContinuityPairwiseValidationError(
            "query_plan_sha256 must be SHA-256"
        )
    real_rows_copy = _rows(real_training_rows, "real_training_rows")
    real_rows_sha256 = _semantic_hash(real_rows_copy)
    targets_copy = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in real_targets_by_case.items()
    }
    if (
        isinstance(query_budget, bool)
        or not isinstance(query_budget, int)
        or query_budget <= 0
        or len(targets_copy) != query_budget
    ):
        raise ServiceContinuityPairwiseValidationError(
            "real target count must equal positive query_budget"
        )

    if not augmentation_enabled:
        if arm != "baseline" or synthetics or any(disposition.values()):
            raise ServiceContinuityPairwiseValidationError(
                "disabled augmentation must be an empty baseline"
            )
        bridge = fit_base_score_bridge(
            dataset_id=dataset_id,
            training_rows=real_rows_copy,
            targets_by_case=targets_copy,
            feature_columns=feature_columns,
            query_plan_sha256=query_hash,
            random_state=42,
        )
        audit_identity = {
            "schema_version": "service-continuous-pairwise-training-audit-v1",
            "arm_id": "baseline",
            "augmentation_enabled": False,
            "backend_id": "pairwise_linear",
            "query_plan_sha256": query_hash,
            "random_state": 42,
            "real_case_count": len(targets_copy),
            "synthetic_case_count": 0,
            "effective_real_case_weight": float(len(targets_copy)),
            "effective_synthetic_case_weight": 0.0,
            "real_training_rows_sha256": real_rows_sha256,
            "synthetic_training_rows_sha256": _semantic_hash([]),
            "synthetic_child_count_by_source": {},
            "synthetic_target_mass_by_source": {},
            "generation_disposition": disposition,
        }
        return ServiceContinuityPairwiseBridge(
            bridge=bridge,
            training_audit={
                **audit_identity,
                "training_audit_sha256": _semantic_hash(audit_identity),
            },
        )

    if arm not in _ARMS:
        raise ServiceContinuityPairwiseValidationError("unknown augmentation arm")
    if not synthetics:
        raise ServiceContinuityPairwiseValidationError(
            "enabled augmentation arm is silently inactive"
        )
    if disposition["generated_count"] != len(synthetics):
        raise ServiceContinuityPairwiseValidationError(
            "generated-count disposition does not match admitted synthetics"
        )

    real_frame, real_targets, real_candidates = _real_frame(
        real_rows_copy, targets_copy, feature_columns, query_budget
    )
    synthetic_frames = []
    all_targets = dict(real_targets)
    all_candidates = dict(real_candidates)
    synthetic_hashes = set()
    child_counts: dict[str, int] = defaultdict(int)
    target_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    effective_synthetic_weight = 0.0
    for synthetic in synthetics:
        row_hash = str(synthetic.get("synthetic_row_sha256", "")).strip()
        source = str(synthetic.get("source_case_id", "")).strip()
        target = str(synthetic.get("target_service_id", "")).strip()
        weight_raw = synthetic.get("final_training_weight")
        candidate_rows = _rows(
            synthetic.get("candidate_rows"), f"synthetic {row_hash} candidate_rows"
        )
        if (
            not _SHA256.fullmatch(row_hash)
            or row_hash in synthetic_hashes
            or source not in real_candidates
            or not target
            or synthetic.get("query_budget_cost") != 0
            or isinstance(weight_raw, bool)
            or not isinstance(weight_raw, (int, float))
            or not math.isfinite(float(weight_raw))
            or float(weight_raw) < 0.0
        ):
            raise ServiceContinuityPairwiseValidationError(
                "synthetic case identity, source, budget, or weight drift"
            )
        synthetic_case_id = f"synthetic:{row_hash}"
        normalized_candidate_rows = [
            {
                **candidate,
                "case_id": synthetic_case_id,
            }
            for candidate in candidate_rows
        ]
        frame = _normalize_rows(normalized_candidate_rows, feature_columns)
        candidate_ids = _candidate_ids(frame)[synthetic_case_id]
        if set(candidate_ids) != set(real_candidates[source]) or target not in set(
            candidate_ids
        ):
            raise ServiceContinuityPairwiseValidationError(
                "synthetic case must preserve the source candidate-complete set"
            )
        frame["label"] = [
            int(str(candidate_id) == target) for candidate_id in frame["entity_id"]
        ]
        frame["sample_weight"] = float(weight_raw)
        frame["pair_weight_mode"] = "pseudo_case_normalized"
        frame["pseudo_case_weight"] = float(weight_raw)
        frame["label_source"] = "pseudo_service_continuity"
        synthetic_hashes.add(row_hash)
        child_counts[source] += 1
        target_mass[source][target] += float(weight_raw)
        effective_synthetic_weight += float(weight_raw)
        if float(weight_raw) > 0.0:
            synthetic_frames.append(frame)
            all_targets[synthetic_case_id] = (target,)
            all_candidates[synthetic_case_id] = candidate_ids

    combined = pd.concat([real_frame, *synthetic_frames], ignore_index=True)
    features = tuple(str(value) for value in feature_columns)
    backend_features = _backend_feature_columns(features)
    backend_frame = _rename_features_for_backend(combined, features, backend_features)
    ranker = train_pairwise_linear_ranker(
        training_frame=backend_frame,
        feature_columns=backend_features,
        random_state=42,
    )
    model_identity = _model_payload(ranker, features)
    model_identity.update(
        {
            "dataset_id": str(dataset_id),
            "arm_id": arm,
            "query_plan_sha256": query_hash,
            "targets_by_case": all_targets,
            "training_candidates_by_case": all_candidates,
            "random_state": 42,
            "backend_feature_columns": backend_features,
            "real_training_rows_sha256": real_rows_sha256,
            "synthetic_row_sha256s": sorted(synthetic_hashes),
        }
    )
    bridge = BaseScoreBridge(
        dataset_id=str(dataset_id),
        backend_id="pairwise_linear",
        ranker=ranker,
        feature_columns=features,
        backend_feature_columns=backend_features,
        targets_by_case=all_targets,
        training_candidates_by_case=all_candidates,
        query_plan_sha256=query_hash,
        random_state=42,
        model_sha256=_semantic_hash(model_identity),
        feature_order_sha256=_semantic_hash(features),
    )
    audit_identity = {
        "schema_version": "service-continuous-pairwise-training-audit-v1",
        "arm_id": arm,
        "augmentation_enabled": True,
        "backend_id": "pairwise_linear",
        "query_plan_sha256": query_hash,
        "random_state": 42,
        "real_case_count": len(real_targets),
        "synthetic_case_count": len(synthetics),
        "effective_real_case_weight": float(len(real_targets)),
        "effective_synthetic_case_weight": effective_synthetic_weight,
        "real_training_rows_sha256": real_rows_sha256,
        "synthetic_training_rows_sha256": _semantic_hash(synthetics),
        "synthetic_child_count_by_source": dict(sorted(child_counts.items())),
        "synthetic_target_mass_by_source": {
            source: dict(sorted(mass.items()))
            for source, mass in sorted(target_mass.items())
        },
        "generation_disposition": disposition,
        "backend_training_diagnostics": ranker.training_diagnostics,
    }
    return ServiceContinuityPairwiseBridge(
        bridge=bridge,
        training_audit={
            **audit_identity,
            "training_audit_sha256": _semantic_hash(audit_identity),
        },
    )


__all__ = [
    "ServiceContinuityPairwiseBridge",
    "ServiceContinuityPairwiseValidationError",
    "fit_service_continuity_pairwise_bridge",
]
