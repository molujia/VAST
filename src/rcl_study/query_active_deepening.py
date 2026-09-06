"""Contracts for the query-only active-learning deepening workflow.

The helpers in this module are intentionally lightweight. They validate the
experiment protocol, selector cleanliness, stage gates, and reporting metadata
before the heavier remote runners launch full experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


CHANGE_ID = "deepen-query-only-active-learning"
REMOTE_WORKSPACE = "${RCL_WORKSPACE}"
RUN_ID = "deepen-query-only-active-learning-20260820-01"
OUTPUT_ROOT = (
    REMOTE_WORKSPACE
    + "/outputs/rcl_study/deepen_query_only_active_learning/"
    + RUN_ID
)
ACTIVE_LEARNING_SEEDS = (42, 43, 44)
SPLIT_SEED = 42
CURRENT_METHOD_SELECTOR_ID = "sequential_proxy_mode_query"
CURRENT_METHOD_FAMILY_ID = "sequential_proxy_mode"
CURRENT_METHOD_REFERENCE_CONFIG = {
    "embedding_field": "embedding",
    "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
    "categorical_fields": ["cluster_id", "time_bucket"],
    "score_field": "inner_boundary_uncertainty",
    "proxy_mode_count": 14,
    "max_per_proxy_mode": 5,
    "seed_budget": 8,
    "metric_ad_required": False,
}
CURRENT_METHOD_BUDGETS = (30, 35, 40, 45, 50, 55, 60, 70, 80)
CURRENT_METHOD_PARAMETER_STABILITY_VALUES = {
    "proxy_mode_count": (12, 14, 16),
    "max_per_proxy_mode": (4, 5, 6),
    "seed_budget": (6, 8, 10),
    "score_field": ("inner_boundary_uncertainty", "baseline_score"),
}
CURRENT_METHOD_DATASETS = ("rcabench", "aiops2022_pre")
CURRENT_METHOD_STAGE_SEQUENCE = (
    "current_method_oracle",
    "current_method_label_free",
    "current_method_parameter_stability",
    "current_method_budget_check",
    "current_method_terminal_summary",
)
DIVERGENT_STRATEGY_FAMILIES = {
    "ranker_aware": {
        "stage": "divergent_ranker_aware",
        "selector_id": "ranker_aware_acquisition",
        "method_id": "ranker_aware_acquisition",
        "selector_config": {
            "score_field": "inner_boundary_uncertainty",
            "uncertainty_weight": 0.45,
            "baseline_weight": 0.20,
            "rank_prior_weight": 0.20,
            "diversity_weight": 0.15,
        },
    },
    "richer_metric_signature": {
        "stage": "divergent_richer_metric_signature",
        "selector_id": "richer_metric_signature_acquisition",
        "method_id": "richer_metric_signature_acquisition",
        "selector_config": {
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "metric_ad_required": False,
            "metric_ad_fields": [
                "metric_ad_anomaly_direction",
                "metric_ad_duration",
                "metric_ad_metric_family_count",
                "metric_ad_propagation_width",
                "metric_ad_sparsity",
                "metric_ad_robust_z_peak",
                "metric_ad_start_slope",
                "metric_ad_peak_lag",
                "metric_ad_recovery_slope",
            ],
        },
    },
    "budgeted_fault_mode_model": {
        "stage": "divergent_budgeted_fault_mode_model",
        "selector_id": "budgeted_fault_mode_modeling",
        "method_id": "budgeted_fault_mode_modeling",
        "selector_config": {
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "seed_budget": 8,
        },
    },
    "facility_location": {
        "stage": "divergent_facility_location",
        "selector_id": "facility_location_selection",
        "method_id": "facility_location_selection",
        "selector_config": {
            "score_field": "inner_boundary_uncertainty",
            "utility_weight": 0.15,
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score", "scheme1_rank"],
            "categorical_fields": ["cluster_id", "time_bucket"],
        },
    },
    "winner_like_distillation": {
        "stage": "divergent_winner_like_distillation",
        "selector_id": "winner_like_distillation",
        "method_id": "winner_like_distillation",
        "selector_config": {
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "winner_feature_weights": {
                "uncertainty": 0.30,
                "baseline": 0.20,
                "rank_prior": 0.20,
                "metric_ad_robust_z_peak": 0.15,
                "metric_ad_duration": 0.15,
            },
        },
    },
}
DIVERGENT_STAGE_SEQUENCE = tuple(
    str(spec["stage"]) for _, spec in sorted(DIVERGENT_STRATEGY_FAMILIES.items())
)
DIVERGENT_DEFAULT_BUDGETS = (30,)

HIT_METRICS = ("hit_at_1", "hit_at_3", "hit_at_5")
RANKING_METRICS = (*HIT_METRICS, "mrr")

SOTA_THRESHOLDS = {
    "rcabench": {
        "hit_at_1": 0.4175,
        "hit_at_3": 0.6351,
        "hit_at_5": 0.7368,
    },
    "aiops2022_pre": {
        "hit_at_1": 0.5210,
        "hit_at_3": 0.8270,
        "hit_at_5": 0.8960,
    },
}

RCABENCH_AUTHORITY_BASELINE = {
    "selector_id": "uncertainty_boundary",
    "T1": {
        "hit_at_1": 0.426230,
        "hit_at_3": 0.543326,
        "hit_at_5": 0.594848,
        "mrr": 0.515604,
    },
}
RCABENCH_STABLE_BUDGET30_REFERENCE = {
    "hit_at_1": 0.477752,
    "hit_at_3": 0.658080,
    "hit_at_5": 0.714286,
    "mrr": 0.589220,
}
RCABENCH_RANDOM_DIAGNOSTIC_TARGET = {
    "unit_id": "random.rcabench.round10.set0001",
    "T1": {
        "hit_at_1": 0.5971896955503513,
        "hit_at_3": 0.7540983606557377,
        "hit_at_5": 0.8009367681498829,
        "mrr": 0.693295655308535,
    },
}
RCABENCH_SERIOUS_DEGRADATION_MARGIN = 0.05
AIOPS2022_SERIOUS_DEGRADATION_MARGIN = 0.02
SERIOUS_DEGRADATION_MAX_RATE = 0.10

FORBIDDEN_UNQUERIED_LABEL_FIELDS = {
    "ground_truth",
    "groundtruth",
    "targets",
    "target_set",
    "root_cause",
    "rootcause",
    "fault_type",
    "true_fault_type",
}

_DATASET_REGISTRY = {
    "rcabench": {
        "canonical_dataset_id": "rcabench",
        "source_path": "${RCABENCH_ROOT}",
        "schema_adapter": "rcabench",
        "fault_type_scope": "oracle_only",
    },
    "aiops25": {
        "canonical_dataset_id": "aiops25",
        "source_path": "${AIOPS25_ROOT}",
        "schema_adapter": "aiops25",
        "fault_type_scope": "not_in_scope",
    },
    "aiops2022_pre": {
        "canonical_dataset_id": "aiops2022_pre",
        "source_path": "${AIOPS22_ROOT}",
        "schema_adapter": "aiops2022_pre",
        "fault_type_scope": "oracle_only",
    },
}

_LEGACY_ALIASES = {
    "hd4": "rcabench",
    "hd3": "aiops25",
}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_dataset_registry() -> dict[str, dict[str, Any]]:
    """Return canonical datasets supported by the deepening workflow."""

    return deepcopy(_DATASET_REGISTRY)


def resolve_canonical_dataset(value: str) -> dict[str, Any]:
    """Resolve a dataset ID, legacy alias, or registered source path."""

    text = str(value).strip()
    if not text:
        raise ValueError("dataset identifier must not be empty")
    canonical = _LEGACY_ALIASES.get(text.lower(), text)
    if canonical in _DATASET_REGISTRY:
        resolved = deepcopy(_DATASET_REGISTRY[canonical])
        if canonical != text:
            resolved["requested_alias"] = text
        return resolved
    for row in _DATASET_REGISTRY.values():
        if text == row.get("source_path"):
            return deepcopy(row)
    if text.lower().startswith("hd"):
        raise ValueError("ambiguous or unregistered HD-style dataset identifier: %s" % text)
    raise ValueError("unknown canonical dataset identifier or source path: %s" % text)


def _iter_forbidden_fields(value: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = "%s.%s" % (path, key_text) if path else key_text
            if key_text.lower() in FORBIDDEN_UNQUERIED_LABEL_FIELDS:
                findings.append(child_path)
            findings.extend(_iter_forbidden_fields(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            child_path = "%s[%d]" % (path, index) if path else "[%d]" % index
            findings.extend(_iter_forbidden_fields(child, child_path))
    return findings


def validate_clean_feature_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a selector feature bundle contains no free labels."""

    payload = deepcopy(dict(bundle))
    dataset_id = str(payload.get("canonical_dataset_id", "")).strip()
    if dataset_id:
        resolve_canonical_dataset(dataset_id)
    features = list(payload.get("features") or [])
    if not features:
        raise ValueError("clean feature bundle must contain features")
    findings = sorted(set(_iter_forbidden_fields(features, "features")))
    if findings:
        raise ValueError(
            "forbidden unqueried label field in clean feature bundle: %s" % findings
        )
    identity = {
        "canonical_dataset_id": dataset_id,
        "schema_version": str(payload.get("schema_version", "")),
        "feature_count": len(features),
        "features": features,
    }
    return {
        "valid": True,
        "canonical_dataset_id": dataset_id,
        "feature_count": len(features),
        "forbidden_findings": [],
        "feature_bundle_sha256": _semantic_sha256(identity),
    }


def _aiops2022_case_id(row: Mapping[str, Any]) -> str:
    timestamp = str(row.get("timestamp", "")).strip()
    level = str(row.get("level", "")).strip()
    cmdb_id = str(row.get("cmdb_id", "")).strip()
    failure_type = str(row.get("failure_type", "")).strip()
    if not timestamp or not level or not cmdb_id or not failure_type:
        raise ValueError("aiops2022_pre groundtruth row missing required fields")
    return "aiops2022_pre:%s:%s:%s:%s" % (timestamp, level, cmdb_id, failure_type)


def _utc_date_from_timestamp(value: Any) -> str:
    timestamp = int(str(value).strip())
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def fault_type_oracle_from_metadata_json(value: Any) -> str:
    """Extract a diagnostic-only fault mechanism label from window metadata."""

    if isinstance(value, Mapping):
        metadata = deepcopy(dict(value))
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            raise ValueError("fault_type oracle metadata is empty")
        metadata = json.loads(text)
        if not isinstance(metadata, Mapping):
            raise ValueError("fault_type oracle metadata must decode to an object")
        metadata = deepcopy(dict(metadata))
    case_metadata = metadata.get("case_metadata")
    if isinstance(case_metadata, Mapping):
        fault_type_code = case_metadata.get("fault_type_code")
        if fault_type_code not in (None, ""):
            return "fault_type_code:%s" % str(fault_type_code).strip()
    fault_type = str(metadata.get("fault_type", "")).strip()
    if fault_type:
        return fault_type
    for key in ("failure_type", "faultType", "fault_type_name"):
        text = str(metadata.get(key, "")).strip()
        if text:
            return text
    raise ValueError("fault_type oracle metadata does not contain a fault_type oracle")


def build_aiops2022_pre_inventory(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_path: str = "${AIOPS22_ROOT}",
    split_seed: int = 42,
    outer_test_ratio: float = 0.3,
) -> dict[str, Any]:
    """Build a stable AIOps22-pre active-learning case inventory from rows."""

    if not rows:
        raise ValueError("aiops2022_pre inventory requires groundtruth rows")
    if not (0.0 < float(outer_test_ratio) < 1.0):
        raise ValueError("outer_test_ratio must be between 0 and 1")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = _aiops2022_case_id(row)
        if case_id in seen:
            raise ValueError("aiops2022_pre inventory contains duplicate case IDs")
        seen.add(case_id)
        label_free = {
            "timestamp": int(str(row["timestamp"]).strip()),
            "date_utc": _utc_date_from_timestamp(row["timestamp"]),
            "level": str(row["level"]).strip(),
            "cmdb_id": str(row["cmdb_id"]).strip(),
        }
        cases.append(
            {
                "case_id": case_id,
                "native_case_id": case_id.removeprefix("aiops2022_pre:"),
                "canonical_dataset_id": "aiops2022_pre",
                "case_kind": "fault",
                "label_free_fields": label_free,
                "fault_type_oracle": str(row["failure_type"]).strip(),
                "fault_type_scope": "oracle_only",
            }
        )
    cases.sort(key=lambda row: str(row["case_id"]))
    shuffled_ids = [str(row["case_id"]) for row in cases]
    random.Random(int(split_seed)).shuffle(shuffled_ids)
    test_count = max(1, round(len(shuffled_ids) * float(outer_test_ratio)))
    if test_count >= len(shuffled_ids):
        test_count = len(shuffled_ids) - 1
    outer_test = sorted(shuffled_ids[:test_count])
    outer_train = sorted(shuffled_ids[test_count:])
    identity = {
        "canonical_dataset_id": "aiops2022_pre",
        "source_path": str(source_path),
        "split_seed": int(split_seed),
        "outer_test_ratio": float(outer_test_ratio),
        "outer_train_case_ids": outer_train,
        "outer_test_case_ids": outer_test,
    }
    return {
        "schema_version": "rcl-query-active-aiops2022-pre-inventory-v1",
        "canonical_dataset_id": "aiops2022_pre",
        "source_path": str(source_path),
        "schema_adapter": "aiops2022_pre",
        "fault_type_scope": "oracle_only",
        "split_seed": int(split_seed),
        "outer_test_ratio": float(outer_test_ratio),
        "cases": cases,
        "outer_train_case_ids": outer_train,
        "outer_test_case_ids": outer_test,
        "admission_counts": {
            "outer_train_fault_cases": len(outer_train),
            "outer_test_fault_cases": len(outer_test),
        },
        "split_identity_sha256": _semantic_sha256(identity),
    }


def build_label_free_feature_bundle(
    *,
    canonical_dataset_id: str,
    cases: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    adapter_name: str,
    feature_schema_version: str = "rcl-query-active-label-free-feature-bundle-v1",
) -> dict[str, Any]:
    """Normalize dataset-specific label-free feature rows into a common bundle."""

    dataset = resolve_canonical_dataset(canonical_dataset_id)["canonical_dataset_id"]
    case_lookup = {str(row["case_id"]): deepcopy(dict(row)) for row in cases}
    expected_fields = (
        "embedding",
        "cluster_id",
        "time_bucket",
        "inner_boundary_uncertainty",
        "baseline_score",
        "metric_ad_duration",
    )
    features: list[dict[str, Any]] = []
    for raw in feature_rows:
        row = deepcopy(dict(raw))
        case_id = str(row.pop("case_id", "")).strip()
        if case_id not in case_lookup:
            raise ValueError("feature row references unknown case_id: %s" % case_id)
        findings = sorted(set(_iter_forbidden_fields(row, "fields")))
        if findings:
            raise ValueError(
                "forbidden unqueried label field in clean feature bundle: %s" % findings
            )
        present = sorted(str(field) for field in row)
        missing = [field for field in expected_fields if field not in row]
        features.append(
            {
                "case_id": case_id,
                "canonical_dataset_id": dataset,
                "fields": row,
                "allowed_fields": present,
                "missing_feature_fields": missing,
                "adapter_provenance": {
                    "adapter_name": str(adapter_name),
                    "source_case_hash": _semantic_sha256(case_lookup[case_id]),
                },
            }
        )
    bundle = {
        "schema_version": "rcl-query-active-label-free-feature-bundle-v1",
        "canonical_dataset_id": dataset,
        "adapter_name": str(adapter_name),
        "adapter_hash": _semantic_sha256(
            {
                "adapter_name": str(adapter_name),
                "canonical_dataset_id": dataset,
                "feature_schema_version": feature_schema_version,
            }
        ),
        "feature_schema_version": feature_schema_version,
        "features": features,
    }
    validation = validate_clean_feature_bundle(bundle)
    bundle["feature_hash"] = validation["feature_bundle_sha256"]
    bundle["missing_feature_indicators"] = {
        row["case_id"]: list(row["missing_feature_fields"]) for row in features
    }
    return bundle


def _finite_float(value: Any, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % context) from exc
    if not math.isfinite(numeric):
        raise ValueError("%s must be finite" % context)
    return numeric


def _clean_unit_is_complete(unit: Mapping[str, Any]) -> bool:
    markers = dict(unit.get("completion_markers") or {})
    return (
        str(unit.get("status")) == "complete"
        and markers.get("COMPLETED.json") is True
        and markers.get("all.done") is True
    )


def _unit_score(unit: Mapping[str, Any]) -> tuple[float, float, float, float]:
    t1 = dict(dict(unit.get("metrics") or {}).get("T1") or {})
    return tuple(_finite_float(t1.get(metric), "T1.%s" % metric) for metric in RANKING_METRICS)  # type: ignore[return-value]


def build_clean_leaderboard(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate oracle diagnostics, clean candidates, and excluded units."""

    clean: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for unit in units:
        row = deepcopy(dict(unit))
        if bool(row.get("oracle_only")):
            oracle.append(row)
            continue
        if not _clean_unit_is_complete(row):
            excluded.append(row)
            continue
        _unit_score(row)
        clean.append(row)
    clean.sort(key=_unit_score, reverse=True)
    return {
        "schema_version": "rcl-query-active-deepening-clean-leaderboard-v1",
        "clean_candidates": clean,
        "oracle_diagnostics": oracle,
        "excluded_units": excluded,
    }


def validate_deepening_manifest_stage_order(
    manifest: Mapping[str, Any],
    *,
    completed_stage_markers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that divergent strategies do not launch too early."""

    stages = [str(row.get("stage_id", "")) for row in list(manifest.get("stages") or [])]
    if not stages or any(not stage for stage in stages):
        raise ValueError("deepening manifest stages must contain stage_id")
    divergent = [stage for stage in stages if stage.startswith("divergent_")]
    markers = dict(completed_stage_markers or {})
    if divergent and markers.get("current_method_terminal_summary") is not True:
        raise ValueError(
            "divergent strategies require current-method terminal summary evidence"
        )
    return {
        "valid": True,
        "stage_count": len(stages),
        "divergent_stage_count": len(divergent),
        "stages": stages,
    }


def _validate_metric_block(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    validated = {
        metric: _finite_float(metrics.get(metric), "%s.%s" % (prefix, metric))
        for metric in RANKING_METRICS
    }
    if "denominator" not in metrics:
        raise ValueError("%s.denominator is required" % prefix)
    denominator = int(metrics["denominator"])
    if denominator <= 0:
        raise ValueError("%s.denominator must be positive" % prefix)
    validated["denominator"] = denominator
    if "hit_counts" not in metrics:
        raise ValueError("%s.hit_counts is required" % prefix)
    hit_counts = deepcopy(dict(metrics["hit_counts"]))
    for metric in HIT_METRICS:
        if metric not in hit_counts:
            raise ValueError("%s.hit_counts missing %s" % (prefix, metric))
    validated["hit_counts"] = hit_counts
    if "mrr_numerator" in metrics:
        validated["mrr_numerator"] = _finite_float(
            metrics.get("mrr_numerator"),
            "%s.mrr_numerator" % prefix,
        )
    return validated


def _t1_t2_warnings(t1: Mapping[str, Any], t2: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for metric in RANKING_METRICS:
        t1_value = float(t1[metric])
        t2_value = float(t2[metric])
        diff = abs(t1_value - t2_value)
        if diff >= 0.05:
            warnings.append(
                {
                    "reason": "absolute_metric_gap_at_least_threshold",
                    "metric": metric,
                    "T1": t1_value,
                    "T2": t2_value,
                    "absolute_difference": diff,
                }
            )
    return warnings


def validate_scored_unit_contract(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate T1/T2 score payloads and budget-case exclusion."""

    payload = deepcopy(dict(unit))
    queried = {str(case_id) for case_id in payload.get("queried_case_ids") or []}
    partitions = dict(payload.get("partitions") or {})
    for partition_name in ("T1", "T2"):
        partition_ids = {str(case_id) for case_id in partitions.get(partition_name) or []}
        overlap = sorted(queried.intersection(partition_ids))
        if overlap:
            raise ValueError(
                "queried budget case appears in %s scoring partition: %s"
                % (partition_name, overlap)
            )
    metrics = dict(payload.get("metrics") or {})
    t1 = _validate_metric_block(dict(metrics.get("T1") or {}), "T1")
    t2 = _validate_metric_block(dict(metrics.get("T2") or {}), "T2")
    if not payload.get("per_case_ranking_evidence"):
        raise ValueError("scored unit requires per-case ranking evidence")
    if not payload.get("target_set_evidence"):
        raise ValueError("scored unit requires target-set evidence")
    return {
        "valid": True,
        "unit_id": str(payload.get("unit_id", "")),
        "metrics": {"T1": t1, "T2": t2},
        "t1_t2_warnings": _t1_t2_warnings(t1, t2),
    }


def _first_target_rank(
    ranking: Sequence[Any],
    targets: Sequence[Any],
) -> int:
    target_set = {str(target) for target in targets if str(target)}
    if not target_set:
        raise ValueError("targets must not be empty")
    for index, candidate in enumerate(ranking, start=1):
        if str(candidate) in target_set:
            return index
    return 0


def _ranking_lookup(
    ranking_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for row in ranking_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("ranking row missing case_id")
        if case_id in lookup:
            raise ValueError("duplicate ranking row for case_id: %s" % case_id)
        lookup[case_id] = row
    if not lookup:
        raise ValueError("ranking_rows must not be empty")
    return lookup


def _compute_ranking_metrics_for_cases(
    *,
    case_ids: Sequence[Any],
    ranking_rows: Sequence[Mapping[str, Any]],
    canonical_dataset_id: str | None = None,
) -> dict[str, Any]:
    ids = [str(case_id) for case_id in case_ids]
    if not ids:
        raise ValueError("case_ids must not be empty")
    lookup = _ranking_lookup(ranking_rows)
    hit_counts = {metric: 0 for metric in HIT_METRICS}
    reciprocal_ranks: list[float] = []
    per_case = []
    for case_id in ids:
        if case_id not in lookup:
            raise ValueError("missing ranking for %s" % case_id)
        row = lookup[case_id]
        ranking = row.get("ranking")
        targets = row.get("targets")
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            raise ValueError("ranking must be a sequence for %s" % case_id)
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise ValueError("targets must be a sequence for %s" % case_id)
        rank = _first_target_rank(ranking, targets)
        reciprocal_rank = 0.0 if rank == 0 else 1.0 / float(rank)
        reciprocal_ranks.append(reciprocal_rank)
        hit_flags = {
            "hit_at_1": bool(rank and rank <= 1),
            "hit_at_3": bool(rank and rank <= 3),
            "hit_at_5": bool(rank and rank <= 5),
        }
        for cutoff, metric in ((1, "hit_at_1"), (3, "hit_at_3"), (5, "hit_at_5")):
            if rank and rank <= cutoff:
                hit_counts[metric] += 1
        per_case_row = {
            "case_id": case_id,
            "first_target_rank": rank,
            "first_matching_rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "targets": [str(target) for target in targets],
            "ranking": [str(candidate) for candidate in ranking],
            **hit_flags,
        }
        if canonical_dataset_id is not None:
            dataset_id = str(canonical_dataset_id).strip()
            if not dataset_id:
                raise ValueError("canonical_dataset_id must not be empty")
            fault_type = str(row.get("fault_type", "")).strip()
            if not fault_type:
                raise ValueError(
                    "ranking row missing fault_type for unseen evidence: %s" % case_id
                )
            per_case_row["dataset_id"] = dataset_id
            per_case_row["fault_type"] = fault_type
        per_case.append(per_case_row)
    denominator = len(ids)
    mrr_numerator = sum(reciprocal_ranks)
    return {
        "denominator": denominator,
        "evaluation_fault_cases": denominator,
        "hit_at_1": hit_counts["hit_at_1"] / denominator,
        "hit_at_3": hit_counts["hit_at_3"] / denominator,
        "hit_at_5": hit_counts["hit_at_5"] / denominator,
        "mrr": mrr_numerator / denominator,
        "mrr_numerator": mrr_numerator,
        "hit_counts": hit_counts,
        "per_case": per_case,
        "case_ids": ids,
        "target_case_ids": ids,
    }


def score_t1_t2_generic(
    *,
    outer_train_case_ids: Sequence[Any],
    outer_test_case_ids: Sequence[Any],
    queried_case_ids: Sequence[Any],
    ranking_rows: Sequence[Mapping[str, Any]],
    extra_t2_case_ids: Sequence[Any] | None = None,
    canonical_dataset_id: str | None = None,
) -> dict[str, Any]:
    """Compute official T1 and diagnostic T2 metrics for any active-RCL dataset."""

    queried = {str(case_id) for case_id in queried_case_ids}
    t1 = [str(case_id) for case_id in outer_test_case_ids if str(case_id) not in queried]
    t2_tail = (
        [str(case_id) for case_id in extra_t2_case_ids]
        if extra_t2_case_ids is not None
        else [str(case_id) for case_id in outer_train_case_ids]
    )
    t2 = list(t1) + [case_id for case_id in t2_tail if case_id not in queried]
    for view_name, ids in (("T1", t1), ("T2", t2)):
        overlap = sorted(queried.intersection(ids))
        if overlap:
            raise ValueError(
                "queried budget case appears in %s scoring partition: %s"
                % (view_name, overlap)
            )
    if extra_t2_case_ids is not None:
        requested_overlap = sorted(queried.intersection(str(case_id) for case_id in extra_t2_case_ids))
        if requested_overlap:
            raise ValueError(
                "queried budget case requested for extra T2 partition: %s"
                % requested_overlap
            )
    t1_metrics = _compute_ranking_metrics_for_cases(
        case_ids=t1,
        ranking_rows=ranking_rows,
        canonical_dataset_id=canonical_dataset_id,
    )
    t2_metrics = _compute_ranking_metrics_for_cases(
        case_ids=t2,
        ranking_rows=ranking_rows,
        canonical_dataset_id=canonical_dataset_id,
    )
    return {
        "schema_version": "rcl-query-active-deepening-t1-t2-score-v1",
        "partitions": {"T1": t1, "T2": t2},
        "queried_case_ids": sorted(queried),
        "T1": t1_metrics,
        "T2": t2_metrics,
        "t1_t2_warnings": _t1_t2_warnings(t1_metrics, t2_metrics),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty values")
    return sum(values) / len(values)


def _dataset_serious_thresholds(dataset_id: str) -> dict[str, float]:
    if dataset_id == "rcabench":
        return {
            metric: RCABENCH_STABLE_BUDGET30_REFERENCE[metric]
            - RCABENCH_SERIOUS_DEGRADATION_MARGIN
            for metric in HIT_METRICS
        }
    if dataset_id == "aiops2022_pre":
        return {
            metric: SOTA_THRESHOLDS[dataset_id][metric]
            - AIOPS2022_SERIOUS_DEGRADATION_MARGIN
            for metric in HIT_METRICS
        }
    raise ValueError("unsupported acceptance dataset: %s" % dataset_id)


def evaluate_cross_dataset_acceptance(
    seed_results_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Evaluate mean SOTA and serious-degradation gates."""

    datasets: dict[str, Any] = {}
    final_ready = True
    for dataset_id, rows_raw in seed_results_by_dataset.items():
        rows = [deepcopy(dict(row)) for row in rows_raw]
        if not rows:
            raise ValueError("dataset %s has no seed results" % dataset_id)
        thresholds = SOTA_THRESHOLDS[str(dataset_id)]
        serious_thresholds = _dataset_serious_thresholds(str(dataset_id))
        means = {
            metric: _mean(
                [
                    _finite_float(dict(row.get("T1") or {}).get(metric), "%s.%s" % (dataset_id, metric))
                    for row in rows
                ]
            )
            for metric in HIT_METRICS
        }
        mean_sota_pass = all(means[metric] > thresholds[metric] for metric in HIT_METRICS)
        serious_seeds: list[dict[str, Any]] = []
        for row in rows:
            t1 = dict(row.get("T1") or {})
            degraded = [
                metric
                for metric in HIT_METRICS
                if _finite_float(t1.get(metric), "%s.seed.%s" % (dataset_id, metric))
                < serious_thresholds[metric]
            ]
            if degraded:
                serious_seeds.append(
                    {
                        "seed": row.get("seed"),
                        "degraded_metrics": degraded,
                    }
                )
        serious_rate = len(serious_seeds) / len(rows)
        requires_expanded = bool(serious_seeds)
        datasets[str(dataset_id)] = {
            "seed_count": len(rows),
            "mean_T1": means,
            "mean_sota_pass": mean_sota_pass,
            "serious_degradation_seed_count": len(serious_seeds),
            "serious_degradation_rate": serious_rate,
            "serious_degradation_seeds": serious_seeds,
            "requires_expanded_seed_check": requires_expanded,
            "serious_degradation_acceptable": (
                serious_rate <= SERIOUS_DEGRADATION_MAX_RATE
            ),
        }
        if not mean_sota_pass or requires_expanded:
            final_ready = False
    return {
        "schema_version": "rcl-query-active-deepening-acceptance-v1",
        "datasets": datasets,
        "final_acceptance_ready": final_ready,
    }


def build_detached_run_handoff(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize tmux handoff metadata for long runs."""

    required = (
        "tmux_session",
        "output_root",
        "progress_command",
        "log_command",
        "failed_command",
        "completed_json",
        "all_done",
        "wake_condition",
    )
    payload = deepcopy(dict(metadata))
    missing = [field for field in required if not str(payload.get(field, "")).strip()]
    if missing:
        raise ValueError("missing handoff field: %s" % missing[0])
    return {
        "schema_version": "rcl-query-active-deepening-handoff-v1",
        "valid": True,
        "tmux_session": str(payload["tmux_session"]),
        "output_root": str(payload["output_root"]),
        "commands": {
            "progress": str(payload["progress_command"]),
            "log": str(payload["log_command"]),
            "failed": str(payload["failed_command"]),
        },
        "completion_proofs": {
            "COMPLETED.json": str(payload["completed_json"]),
            "all.done": str(payload["all_done"]),
        },
        "wake_condition": str(payload["wake_condition"]),
    }


def reference_scorebook() -> dict[str, Any]:
    """Return frozen reference scores and SOTA thresholds for this change."""

    return {
        "schema_version": "rcl-query-active-deepening-reference-scorebook-v1",
        "rcabench": {
            "sota_thresholds": deepcopy(SOTA_THRESHOLDS["rcabench"]),
            "authority_baseline": deepcopy(RCABENCH_AUTHORITY_BASELINE),
            "stable_mechanism_budget30": {
                "selector_id": "sequential_proxy_mode_query",
                "T1": deepcopy(RCABENCH_STABLE_BUDGET30_REFERENCE),
            },
            "random_diagnostic_target": deepcopy(RCABENCH_RANDOM_DIAGNOSTIC_TARGET),
        },
        "aiops2022_pre": {
            "sota_thresholds": deepcopy(SOTA_THRESHOLDS["aiops2022_pre"]),
        },
    }


def evaluate_t1_sota_status(
    canonical_dataset_id: str,
    t1_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate strict T1 SOTA status for one dataset."""

    dataset = resolve_canonical_dataset(canonical_dataset_id)["canonical_dataset_id"]
    if dataset not in SOTA_THRESHOLDS:
        raise ValueError("dataset has no T1 SOTA thresholds: %s" % dataset)
    thresholds = SOTA_THRESHOLDS[dataset]
    metrics: dict[str, Any] = {}
    for metric in HIT_METRICS:
        actual = _finite_float(t1_metrics.get(metric), "T1.%s" % metric)
        threshold = float(thresholds[metric])
        metrics[metric] = {
            "actual": actual,
            "threshold": threshold,
            "strictly_exceeds": actual > threshold,
            "margin": actual - threshold,
        }
    return {
        "canonical_dataset_id": dataset,
        "passed": all(row["strictly_exceeds"] for row in metrics.values()),
        "metrics": metrics,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    import threading

    temporary = destination.with_name(
        destination.name + ".tmp-%d-%d" % (os.getpid(), threading.get_ident())
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(destination))


def ensure_formal_run_is_remote(
    *,
    os_name: str | None = None,
    workspace: str = REMOTE_WORKSPACE,
) -> dict[str, Any]:
    """Fail closed if a formal experiment is attempted outside the server workspace."""

    observed_os = os.name if os_name is None else str(os_name)
    observed_workspace = str(workspace)
    if observed_os == "nt" or observed_workspace != REMOTE_WORKSPACE:
        raise RuntimeError(
            "formal deepening experiments must run on the remote server under %s"
            % REMOTE_WORKSPACE
        )
    return {
        "valid": True,
        "workspace": observed_workspace,
        "os_name": observed_os,
    }


def build_deepening_stage_manifest(
    *,
    run_id: str = RUN_ID,
    output_root: str = OUTPUT_ROOT,
    include_divergent: bool = False,
) -> dict[str, Any]:
    """Build the stage-gated manifest skeleton for this change."""

    root = str(output_root).rstrip("/")
    if not root.startswith(REMOTE_WORKSPACE + "/outputs/rcl_study/"):
        raise ValueError("deepening output_root must be below the approved RCL output root")
    stages = [
        {"stage_id": "backup", "required_before": []},
        {"stage_id": "smoke_validation", "required_before": ["backup"]},
        {
            "stage_id": "current_method_oracle",
            "required_before": ["smoke_validation"],
        },
        {
            "stage_id": "current_method_label_free",
            "required_before": ["current_method_oracle"],
        },
        {
            "stage_id": "current_method_budget_and_stability",
            "required_before": ["current_method_label_free"],
        },
        {
            "stage_id": "current_method_terminal_summary",
            "required_before": ["current_method_budget_and_stability"],
        },
    ]
    if include_divergent:
        stages.extend(
            [
                {
                    "stage_id": "divergent_ranker_aware",
                    "required_before": ["current_method_terminal_summary"],
                },
                {
                    "stage_id": "divergent_richer_metric_signature",
                    "required_before": ["current_method_terminal_summary"],
                },
                {
                    "stage_id": "divergent_budgeted_fault_mode_model",
                    "required_before": ["current_method_terminal_summary"],
                },
                {
                    "stage_id": "divergent_facility_location",
                    "required_before": ["current_method_terminal_summary"],
                },
                {
                    "stage_id": "divergent_winner_like_distillation",
                    "required_before": ["current_method_terminal_summary"],
                },
            ]
        )
    stages.extend(
        [
            {"stage_id": "final_aggregation", "required_before": []},
            {"stage_id": "packaging", "required_before": ["final_aggregation"]},
        ]
    )
    manifest = {
        "schema_version": "rcl-query-active-deepening-stage-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": str(run_id),
        "output_root": root,
        "stages": stages,
        "completion_proofs": {
            "COMPLETED.json": root + "/COMPLETED.json",
            "all.done": root + "/all.done",
            ".failed": root + "/.failed",
        },
    }
    validate_deepening_manifest_stage_order(manifest)
    manifest["manifest_sha256"] = _semantic_sha256(manifest)
    return manifest


def _require_run_id(value: Any, context: str = "run_id") -> str:
    text = str(value).strip()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError("%s must be a path-safe identifier" % context)
    return text


def _require_deepening_output_root(
    value: Any,
    *,
    allow_local_output_root: bool = False,
) -> str:
    root = str(value).rstrip("/")
    if allow_local_output_root:
        if not root:
            raise ValueError("deepening output_root must not be empty")
        return root
    allowed_prefix = REMOTE_WORKSPACE + "/outputs/rcl_study/"
    if not root.startswith(allowed_prefix):
        raise ValueError("deepening output_root must be below %s" % allowed_prefix)
    return root


def _safe_token(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty token")
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text)


def _current_method_selector_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = deepcopy(CURRENT_METHOD_REFERENCE_CONFIG)
    config.update(deepcopy(dict(overrides or {})))
    return config


def _selector_config_sha256(selector_id: str, selector_config: Mapping[str, Any]) -> str:
    return _semantic_sha256(
        {
            "selector_id": str(selector_id),
            "selector_config": deepcopy(dict(selector_config)),
        }
    )


def _current_method_unit(
    *,
    base_root: str,
    run_id: str,
    stage: str,
    dataset: str,
    selector_id: str,
    selector_config: Mapping[str, Any],
    budget: int,
    seed: int,
    oracle_only: bool,
    strategy_family: str,
    method_id: str,
    parameter_name: str | None = None,
    parameter_value: Any = None,
) -> dict[str, Any]:
    dataset_id = resolve_canonical_dataset(dataset)["canonical_dataset_id"]
    selector_registry = list_deepening_selectors()
    if selector_id not in selector_registry:
        raise ValueError("unknown deepening selector: %s" % selector_id)
    selector_meta = selector_registry[selector_id]
    unit_token_parts = [method_id, dataset_id]
    if parameter_name is not None:
        unit_token_parts.extend([str(parameter_name), str(parameter_value)])
    unit_token_parts.extend(["budget%d" % int(budget), "seed%d" % int(seed)])
    unit_token = _safe_token(".".join(unit_token_parts))
    output_root = deepening_unit_output_root(
        base_root=base_root,
        stage=stage,
        strategy_family=strategy_family,
        dataset=dataset_id,
        budget=int(budget),
        seed=int(seed),
        run_id=unit_token,
    )
    unit = {
        "unit_id": "current.%s" % unit_token,
        "stage": str(stage),
        "canonical_dataset_id": dataset_id,
        "source_path": _DATASET_REGISTRY[dataset_id]["source_path"],
        "mode": "run_deepening_query_t1_t2_unit",
        "method_id": str(method_id),
        "selector_id": str(selector_id),
        "selector_family": str(strategy_family),
        "selector_config": deepcopy(dict(selector_config)),
        "selector_config_sha256": _selector_config_sha256(
            selector_id,
            selector_config,
        ),
        "allowed_input_contract": str(selector_meta["allowed_input_contract"]),
        "oracle_only": bool(oracle_only),
        "budget": int(budget),
        "budget_unit": "unique_outer_training_fault_cases",
        "active_learning_seed": int(seed),
        "seed": int(seed),
        "split_seed": SPLIT_SEED,
        "normal_policy": "fault_only",
        "evaluation_views": ["T1", "T2"],
        "output_root": output_root,
    }
    if parameter_name is not None:
        unit["parameter_stability"] = {
            "parameter_name": str(parameter_name),
            "parameter_value": deepcopy(parameter_value),
            "reference_config": deepcopy(CURRENT_METHOD_REFERENCE_CONFIG),
        }
    return unit


def _divergent_strategy_unit(
    *,
    base_root: str,
    run_id: str,
    stage: str,
    dataset: str,
    selector_id: str,
    selector_config: Mapping[str, Any],
    budget: int,
    seed: int,
    strategy_family: str,
    method_id: str,
) -> dict[str, Any]:
    dataset_id = resolve_canonical_dataset(dataset)["canonical_dataset_id"]
    selector_registry = list_deepening_selectors()
    if selector_id not in selector_registry:
        raise ValueError("unknown deepening selector: %s" % selector_id)
    selector_meta = selector_registry[selector_id]
    if bool(selector_meta.get("oracle_only")):
        raise ValueError("divergent strategy cannot use oracle-only selector")
    unit_token = _safe_token(
        ".".join(
            [
                method_id,
                dataset_id,
                "budget%d" % int(budget),
                "seed%d" % int(seed),
            ]
        )
    )
    output_root = deepening_unit_output_root(
        base_root=base_root,
        stage=stage,
        strategy_family=strategy_family,
        dataset=dataset_id,
        budget=int(budget),
        seed=int(seed),
        run_id=unit_token,
    )
    return {
        "unit_id": "divergent.%s" % unit_token,
        "stage": str(stage),
        "canonical_dataset_id": dataset_id,
        "source_path": _DATASET_REGISTRY[dataset_id]["source_path"],
        "mode": "run_deepening_query_t1_t2_unit",
        "method_id": str(method_id),
        "selector_id": str(selector_id),
        "selector_family": str(strategy_family),
        "selector_config": deepcopy(dict(selector_config)),
        "selector_config_sha256": _selector_config_sha256(
            selector_id,
            selector_config,
        ),
        "allowed_input_contract": str(selector_meta["allowed_input_contract"]),
        "oracle_only": False,
        "budget": int(budget),
        "budget_unit": "unique_outer_training_fault_cases",
        "active_learning_seed": int(seed),
        "seed": int(seed),
        "split_seed": SPLIT_SEED,
        "normal_policy": "fault_only",
        "evaluation_views": ["T1", "T2"],
        "output_root": output_root,
    }


def _coerce_unique_ints(values: Sequence[Any], context: str) -> list[int]:
    ints = [int(value) for value in values]
    if not ints or any(value <= 0 for value in ints):
        raise ValueError("%s must contain positive integers" % context)
    if len(ints) != len(set(ints)):
        raise ValueError("%s must not contain duplicates" % context)
    return ints


def build_current_method_formal_manifest(
    *,
    run_id: str,
    output_root: str,
    active_learning_seeds: Sequence[Any] = ACTIVE_LEARNING_SEEDS,
    budgets: Sequence[Any] = CURRENT_METHOD_BUDGETS,
    datasets: Sequence[str] = CURRENT_METHOD_DATASETS,
    selector_config: Mapping[str, Any] | None = None,
    parameter_stability_values: Mapping[str, Sequence[Any]]
    | None = CURRENT_METHOD_PARAMETER_STABILITY_VALUES,
    allow_local_output_root: bool = False,
) -> dict[str, Any]:
    """Build the ordered formal manifest for current-method deepening runs."""

    run_id_text = _require_run_id(run_id)
    root = _require_deepening_output_root(
        output_root,
        allow_local_output_root=allow_local_output_root,
    )
    seeds = _coerce_unique_ints(active_learning_seeds, "active_learning_seeds")
    budget_values = _coerce_unique_ints(budgets, "budgets")
    if 30 not in budget_values:
        raise ValueError("current method budget checks must include budget 30 reference")
    dataset_ids = [
        resolve_canonical_dataset(dataset)["canonical_dataset_id"] for dataset in datasets
    ]
    if sorted(dataset_ids) != sorted(CURRENT_METHOD_DATASETS):
        raise ValueError("current-method manifest must cover rcabench and aiops2022_pre")
    reference_config = _current_method_selector_config(selector_config)
    parameter_values = {
        str(name): tuple(values)
        for name, values in dict(parameter_stability_values or {}).items()
    }
    units: list[dict[str, Any]] = []
    for dataset in dataset_ids:
        for seed in seeds:
            units.append(
                _current_method_unit(
                    base_root=root,
                    run_id=run_id_text,
                    stage="current_method_oracle",
                    dataset=dataset,
                    selector_id="true_fault_type_oracle",
                    selector_config={"score_field": "inner_boundary_uncertainty"},
                    budget=30,
                    seed=seed,
                    oracle_only=True,
                    strategy_family="oracle_fault_type_coverage",
                    method_id="true_fault_type_oracle",
                )
            )
    for dataset in dataset_ids:
        for seed in seeds:
            units.append(
                _current_method_unit(
                    base_root=root,
                    run_id=run_id_text,
                    stage="current_method_label_free",
                    dataset=dataset,
                    selector_id=CURRENT_METHOD_SELECTOR_ID,
                    selector_config=reference_config,
                    budget=30,
                    seed=seed,
                    oracle_only=False,
                    strategy_family=CURRENT_METHOD_FAMILY_ID,
                    method_id="label_free_reference",
                )
            )
    for parameter_name in sorted(parameter_values):
        values = parameter_values[parameter_name]
        if not values:
            raise ValueError("parameter stability values must not be empty")
        for value in values:
            config = _current_method_selector_config({parameter_name: value})
            for dataset in dataset_ids:
                for seed in seeds:
                    units.append(
                        _current_method_unit(
                            base_root=root,
                            run_id=run_id_text,
                            stage="current_method_parameter_stability",
                            dataset=dataset,
                            selector_id=CURRENT_METHOD_SELECTOR_ID,
                            selector_config=config,
                            budget=30,
                            seed=seed,
                            oracle_only=False,
                            strategy_family=CURRENT_METHOD_FAMILY_ID,
                            method_id="parameter_stability",
                            parameter_name=parameter_name,
                            parameter_value=value,
                        )
                    )
    for budget in budget_values:
        for dataset in dataset_ids:
            for seed in seeds:
                units.append(
                    _current_method_unit(
                        base_root=root,
                        run_id=run_id_text,
                        stage="current_method_budget_check",
                        dataset=dataset,
                        selector_id=CURRENT_METHOD_SELECTOR_ID,
                        selector_config=reference_config,
                        budget=budget,
                        seed=seed,
                        oracle_only=False,
                        strategy_family=CURRENT_METHOD_FAMILY_ID,
                        method_id="budget_check",
                    )
                )
    manifest = {
        "schema_version": "rcl-query-active-deepening-current-method-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": run_id_text,
        "output_root": root,
        "stage_sequence": list(CURRENT_METHOD_STAGE_SEQUENCE),
        "target_datasets": {
            dataset: deepcopy(_DATASET_REGISTRY[dataset]) for dataset in sorted(dataset_ids)
        },
        "split_seed": SPLIT_SEED,
        "active_learning_seeds": seeds,
        "budget_grid": budget_values,
        "reference_budget": 30,
        "active_learning_method": {
            "selector_id": CURRENT_METHOD_SELECTOR_ID,
            "selector_family": CURRENT_METHOD_FAMILY_ID,
            "unified_logic_across_datasets": True,
            "reference_config": reference_config,
            "reference_config_sha256": _selector_config_sha256(
                CURRENT_METHOD_SELECTOR_ID,
                reference_config,
            ),
        },
        "parameter_stability_values": {
            name: list(values) for name, values in sorted(parameter_values.items())
        },
        "reference_scorebook": reference_scorebook(),
        "units": units,
        "execution": {
            "runner_entrypoint": "scripts/run_query_active_deepening_current.py",
            "unit_entrypoint": "scripts/run_query_active_deepening_unit.py",
            "launcher_entrypoint": "scripts/launch_query_active_deepening_current_tmux.sh",
            "status_entrypoint": "scripts/status_query_active_deepening.py",
            "conda_environment": "rcalab",
            "formal_execution_location": REMOTE_WORKSPACE,
            "gpu_policy": "auto_degrade_to_usable_gpus_up_to_24",
        },
        "completion_proofs": {
            "completed_json": root + "/COMPLETED.json",
            "all_done": root + "/all.done",
            ".failed": root + "/.failed",
            "terminal_summary": root + "/current_method_terminal_summary.json",
        },
    }
    manifest["manifest_sha256"] = _semantic_sha256(manifest)
    validate_current_method_formal_manifest(
        manifest,
        allow_local_output_root=allow_local_output_root,
    )
    return manifest


def validate_current_method_formal_manifest(
    manifest: Mapping[str, Any],
    *,
    allow_local_output_root: bool = False,
    allow_hash_drift: bool = False,
) -> dict[str, Any]:
    """Validate the current-method formal manifest before any long run starts."""

    payload = deepcopy(dict(manifest))
    if payload.get("schema_version") != (
        "rcl-query-active-deepening-current-method-manifest-v1"
    ):
        raise ValueError("unexpected current-method manifest schema")
    root = _require_deepening_output_root(
        payload.get("output_root"),
        allow_local_output_root=allow_local_output_root,
    )
    stages = [str(stage) for stage in payload.get("stage_sequence") or []]
    if stages != list(CURRENT_METHOD_STAGE_SEQUENCE):
        raise ValueError("current-method stage sequence drifted")
    units = [deepcopy(dict(unit)) for unit in payload.get("units") or []]
    if not units:
        raise ValueError("current-method manifest requires units")
    unit_ids = [str(unit.get("unit_id", "")) for unit in units]
    if any(not unit_id for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
        raise ValueError("current-method manifest unit IDs must be unique and non-empty")
    counts = {stage: 0 for stage in stages}
    parameter_names: set[str] = set()
    budgets_by_dataset: dict[str, set[int]] = {}
    divergent_units = []
    for unit in units:
        stage = str(unit.get("stage", "")).strip()
        if stage.startswith("divergent_"):
            divergent_units.append(str(unit.get("unit_id", "")))
            continue
        if stage not in counts:
            raise ValueError("unknown current-method stage: %s" % stage)
        counts[stage] += 1
        dataset = resolve_canonical_dataset(str(unit.get("canonical_dataset_id", "")))[
            "canonical_dataset_id"
        ]
        if dataset not in CURRENT_METHOD_DATASETS:
            raise ValueError("unsupported current-method dataset: %s" % dataset)
        budget = int(unit.get("budget", 0))
        if budget <= 0:
            raise ValueError("current-method unit budget must be positive")
        budgets_by_dataset.setdefault(dataset, set()).add(budget)
        if int(unit.get("split_seed", SPLIT_SEED)) != SPLIT_SEED:
            raise ValueError("current-method split_seed must remain 42")
        output_root = str(unit.get("output_root", ""))
        if not output_root.startswith(root + "/") and not output_root.startswith(root + "\\"):
            raise ValueError("current-method unit output_root escapes manifest root")
        selector_id = str(unit.get("selector_id", ""))
        oracle_only = bool(unit.get("oracle_only"))
        if stage == "current_method_oracle":
            if not oracle_only or selector_id != "true_fault_type_oracle":
                raise ValueError("oracle stage requires true_fault_type_oracle oracle_only")
            if str(unit.get("allowed_input_contract")) != "full_candidate_fault_type_oracle":
                raise ValueError("oracle stage input contract drifted")
        elif stage in {
            "current_method_label_free",
            "current_method_parameter_stability",
            "current_method_budget_check",
        }:
            if oracle_only:
                raise ValueError("oracle_only unit cannot appear in clean stage")
            if selector_id == "true_fault_type_oracle":
                raise ValueError("clean stage cannot use true fault-type oracle")
            if str(unit.get("allowed_input_contract")) != (
                "label_free_plus_budgeted_revealed_labels"
            ):
                raise ValueError("clean stage input contract drifted")
        if stage == "current_method_parameter_stability":
            info = dict(unit.get("parameter_stability") or {})
            name = str(info.get("parameter_name", "")).strip()
            if not name:
                raise ValueError("parameter stability unit missing parameter_name")
            parameter_names.add(name)
    if divergent_units:
        raise ValueError("current-method manifest must not contain divergent units")
    for stage in CURRENT_METHOD_STAGE_SEQUENCE[:-1]:
        if counts.get(stage, 0) <= 0:
            raise ValueError("current-method stage has no units: %s" % stage)
    for dataset in CURRENT_METHOD_DATASETS:
        if 30 not in budgets_by_dataset.get(dataset, set()):
            raise ValueError("budget check missing budget 30 for %s" % dataset)
    recorded_sha = str(payload.get("manifest_sha256", ""))
    if recorded_sha and not allow_hash_drift:
        unsigned = deepcopy(payload)
        unsigned.pop("manifest_sha256", None)
        if _semantic_sha256(unsigned) != recorded_sha:
            raise ValueError("current-method manifest hash drifted")
    return {
        "schema_version": "rcl-query-active-deepening-current-method-validation-v1",
        "valid": True,
        "unit_count": len(units),
        "unit_counts_by_stage": counts,
        "parameter_stability_names": sorted(parameter_names),
        "budgets_by_dataset": {
            dataset: sorted(values) for dataset, values in sorted(budgets_by_dataset.items())
        },
        "divergent_unit_count": 0,
    }


def _require_current_method_terminal_summary(path: Any) -> str:
    summary_path = Path(str(path))
    if not summary_path.is_file():
        raise ValueError("current-method terminal summary evidence is missing: %s" % path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("current-method terminal summary must be a JSON object")
    if int(payload.get("unit_count", 0)) <= 0:
        raise ValueError("current-method terminal summary has no completed units")
    return str(summary_path)


def _divergent_selector_config(
    family_id: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    family = deepcopy(dict(DIVERGENT_STRATEGY_FAMILIES[str(family_id)]))
    config = deepcopy(dict(family["selector_config"]))
    config.update(deepcopy(dict(overrides or {})))
    return config


def build_divergent_strategy_manifest(
    *,
    run_id: str,
    output_root: str,
    current_method_terminal_summary_path: str,
    active_learning_seeds: Sequence[Any] = ACTIVE_LEARNING_SEEDS,
    budgets: Sequence[Any] = DIVERGENT_DEFAULT_BUDGETS,
    datasets: Sequence[str] = CURRENT_METHOD_DATASETS,
    strategy_families: Sequence[str] | None = None,
    selector_config_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    allow_local_output_root: bool = False,
) -> dict[str, Any]:
    """Build the formal manifest for post-current divergent strategy screening."""

    run_id_text = _require_run_id(run_id)
    root = _require_deepening_output_root(
        output_root,
        allow_local_output_root=allow_local_output_root,
    )
    summary_path = _require_current_method_terminal_summary(
        current_method_terminal_summary_path
    )
    seeds = _coerce_unique_ints(active_learning_seeds, "active_learning_seeds")
    budget_values = _coerce_unique_ints(budgets, "budgets")
    dataset_ids = [
        resolve_canonical_dataset(dataset)["canonical_dataset_id"] for dataset in datasets
    ]
    if sorted(dataset_ids) != sorted(CURRENT_METHOD_DATASETS):
        raise ValueError("divergent manifest must cover rcabench and aiops2022_pre")
    requested_families = [
        str(family)
        for family in (
            strategy_families
            if strategy_families is not None
            else sorted(DIVERGENT_STRATEGY_FAMILIES)
        )
    ]
    if not requested_families:
        raise ValueError("divergent manifest requires at least one strategy family")
    overrides = {
        str(key): dict(value)
        for key, value in dict(selector_config_overrides or {}).items()
    }
    units: list[dict[str, Any]] = []
    for family_id in requested_families:
        if family_id not in DIVERGENT_STRATEGY_FAMILIES:
            raise ValueError("unknown divergent strategy family: %s" % family_id)
        family = DIVERGENT_STRATEGY_FAMILIES[family_id]
        config = _divergent_selector_config(family_id, overrides.get(family_id))
        for budget in budget_values:
            for dataset in dataset_ids:
                for seed in seeds:
                    units.append(
                        _divergent_strategy_unit(
                            base_root=root,
                            run_id=run_id_text,
                            stage=str(family["stage"]),
                            dataset=dataset,
                            selector_id=str(family["selector_id"]),
                            selector_config=config,
                            budget=budget,
                            seed=seed,
                            strategy_family=family_id,
                            method_id=str(family["method_id"]),
                        )
                    )
    manifest = {
        "schema_version": "rcl-query-active-deepening-divergent-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": run_id_text,
        "output_root": root,
        "stage_sequence": list(DIVERGENT_STAGE_SEQUENCE),
        "current_method_terminal_summary_path": summary_path,
        "target_datasets": {
            dataset: deepcopy(_DATASET_REGISTRY[dataset]) for dataset in sorted(dataset_ids)
        },
        "split_seed": SPLIT_SEED,
        "active_learning_seeds": seeds,
        "budget_grid": budget_values,
        "strategy_families": requested_families,
        "strategy_family_specs": deepcopy(
            {family: DIVERGENT_STRATEGY_FAMILIES[family] for family in requested_families}
        ),
        "reference_scorebook": reference_scorebook(),
        "units": units,
        "execution": {
            "runner_entrypoint": "scripts/run_query_active_deepening_divergent.py",
            "unit_entrypoint": "scripts/run_query_active_deepening_unit.py",
            "launcher_entrypoint": "scripts/launch_query_active_deepening_divergent_tmux.sh",
            "status_entrypoint": "scripts/status_query_active_deepening.py",
            "conda_environment": "rcalab",
            "formal_execution_location": REMOTE_WORKSPACE,
            "gpu_policy": "auto_degrade_to_usable_gpus_up_to_24",
        },
        "completion_proofs": {
            "completed_json": root + "/COMPLETED.json",
            "all_done": root + "/all.done",
            ".failed": root + "/.failed",
            "terminal_summary": root + "/divergent_strategy_terminal_summary.json",
        },
    }
    manifest["manifest_sha256"] = _semantic_sha256(manifest)
    validate_divergent_strategy_manifest(
        manifest,
        allow_local_output_root=allow_local_output_root,
    )
    return manifest


def validate_divergent_strategy_manifest(
    manifest: Mapping[str, Any],
    *,
    allow_local_output_root: bool = False,
    allow_hash_drift: bool = False,
) -> dict[str, Any]:
    """Validate divergent strategy manifests after current-method completion."""

    payload = deepcopy(dict(manifest))
    if payload.get("schema_version") != "rcl-query-active-deepening-divergent-manifest-v1":
        raise ValueError("unexpected divergent strategy manifest schema")
    root = _require_deepening_output_root(
        payload.get("output_root"),
        allow_local_output_root=allow_local_output_root,
    )
    _require_current_method_terminal_summary(
        payload.get("current_method_terminal_summary_path")
    )
    stages = [str(stage) for stage in payload.get("stage_sequence") or []]
    if stages != list(DIVERGENT_STAGE_SEQUENCE):
        raise ValueError("divergent stage sequence drifted")
    units = [deepcopy(dict(unit)) for unit in payload.get("units") or []]
    if not units:
        raise ValueError("divergent manifest requires units")
    unit_ids = [str(unit.get("unit_id", "")) for unit in units]
    if any(not unit_id for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
        raise ValueError("divergent manifest unit IDs must be unique and non-empty")
    counts = {stage: 0 for stage in stages}
    selector_registry = list_deepening_selectors()
    families_seen: set[str] = set()
    for unit in units:
        stage = str(unit.get("stage", "")).strip()
        if stage not in counts:
            raise ValueError("unknown divergent stage: %s" % stage)
        counts[stage] += 1
        dataset = resolve_canonical_dataset(str(unit.get("canonical_dataset_id", "")))[
            "canonical_dataset_id"
        ]
        if dataset not in CURRENT_METHOD_DATASETS:
            raise ValueError("unsupported divergent dataset: %s" % dataset)
        if int(unit.get("split_seed", SPLIT_SEED)) != SPLIT_SEED:
            raise ValueError("divergent split_seed must remain 42")
        if int(unit.get("budget", 0)) <= 0:
            raise ValueError("divergent unit budget must be positive")
        output_root = str(unit.get("output_root", ""))
        if not output_root.startswith(root + "/") and not output_root.startswith(root + "\\"):
            raise ValueError("divergent unit output_root escapes manifest root")
        selector_id = str(unit.get("selector_id", ""))
        if selector_id not in selector_registry:
            raise ValueError("unknown divergent selector: %s" % selector_id)
        meta = selector_registry[selector_id]
        if bool(unit.get("oracle_only")) or bool(meta.get("oracle_only")):
            raise ValueError("divergent unit must be publication-clean, not oracle-only")
        if str(unit.get("allowed_input_contract")) == "full_candidate_fault_type_oracle":
            raise ValueError("divergent unit cannot use full fault-type oracle input")
        families_seen.add(str(unit.get("selector_family", "")))
    missing = [stage for stage, count in counts.items() if count <= 0]
    if missing:
        raise ValueError("divergent manifest missing stages: %s" % missing)
    recorded_sha = str(payload.get("manifest_sha256", ""))
    if recorded_sha and not allow_hash_drift:
        unsigned = deepcopy(payload)
        unsigned.pop("manifest_sha256", None)
        if _semantic_sha256(unsigned) != recorded_sha:
            raise ValueError("divergent manifest hash drifted")
    return {
        "schema_version": "rcl-query-active-deepening-divergent-validation-v1",
        "valid": True,
        "unit_count": len(units),
        "unit_counts_by_stage": counts,
        "strategy_families": sorted(family for family in families_seen if family),
        "oracle_unit_count": 0,
    }


def format_deepening_progress_line(
    progress: Mapping[str, Any],
    *,
    width: int = 20,
) -> str:
    """Render a tqdm-style single-line progress snapshot for tmux logs."""

    completed = int(progress.get("completed_units", 0))
    total = int(progress.get("total_units", 0))
    fraction = 0.0 if total <= 0 else max(0.0, min(1.0, completed / float(total)))
    filled = int(round(fraction * int(width)))
    bar = "=" * filled + "." * (int(width) - filled)
    return (
        "[{bar}] {completed}/{total} {percent:.1f}% "
        "stage={stage} family={family} dataset={dataset} budget={budget} "
        "seed={seed} failures={failures}"
    ).format(
        bar=bar,
        completed=completed,
        total=total,
        percent=fraction * 100.0,
        stage=progress.get("stage", ""),
        family=progress.get("strategy_family", ""),
        dataset=progress.get("dataset", ""),
        budget=progress.get("budget", ""),
        seed=progress.get("seed", ""),
        failures=progress.get("failure_count", 0),
    )


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_completion_markers(
    run_root: Path,
    completion_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(dict(completion_payload))
    payload.setdefault("completed_at_utc", _utc_now())
    completed_path = run_root / "COMPLETED.json"
    all_done_path = run_root / "all.done"
    _write_json_atomic(completed_path, payload)
    completed_sha = _sha256_file(completed_path)
    all_done_path.write_text(completed_sha + "\n", encoding="utf-8")
    return {
        "completed_json": str(completed_path),
        "all_done": str(all_done_path),
        "completed_sha256": completed_sha,
    }


def _write_deepening_failure_marker(
    run_root: Path,
    exc: BaseException,
    *,
    completed_units: int,
    total_units: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": "rcl-query-active-deepening-current-method-failure-v1",
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "failed_at_utc": _utc_now(),
    }
    _write_json_atomic(run_root / "pipeline.failed.json", payload)
    (run_root / ".failed").write_text(
        "%s: %s\n" % (type(exc).__name__, exc),
        encoding="utf-8",
    )
    return payload


def _call_unit_executor(unit_executor: Any, unit: Mapping[str, Any], run_root: Path) -> Any:
    try:
        return unit_executor(unit, run_root)
    except TypeError as exc:
        try:
            return unit_executor(unit)
        except TypeError:
            raise exc


def _normalize_current_method_unit_result(
    unit: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(raw_result))
    if str(result.get("unit_id")) != str(unit.get("unit_id")):
        raise ValueError("current-method result unit_id mismatch")
    dataset = resolve_canonical_dataset(str(result.get("canonical_dataset_id")))[
        "canonical_dataset_id"
    ]
    if dataset != str(unit.get("canonical_dataset_id")):
        raise ValueError("current-method result dataset mismatch")
    if "metrics" not in result and "t1_metrics" in result and "t2_metrics" in result:
        result["metrics"] = {
            "T1": deepcopy(dict(result["t1_metrics"])),
            "T2": deepcopy(dict(result["t2_metrics"])),
        }
    result.setdefault("stage", str(unit.get("stage")))
    result.setdefault("selector_id", str(unit.get("selector_id")))
    result.setdefault("oracle_only", bool(unit.get("oracle_only")))
    result.setdefault("budget", int(unit.get("budget", 0)))
    result.setdefault("active_learning_seed", int(unit.get("active_learning_seed", 0)))
    result.setdefault("queried_case_ids", list(result.get("selected_case_ids") or []))
    validated_score = validate_scored_unit_contract(result)
    result["canonical_dataset_id"] = dataset
    result["metrics"] = validated_score["metrics"]
    result["t1_t2_warnings"] = validated_score["t1_t2_warnings"]
    result["t1_sota_status"] = evaluate_t1_sota_status(
        dataset,
        result["metrics"]["T1"],
    )
    result["status"] = "complete"
    result["completion_markers"] = {"COMPLETED.json": True, "all.done": True}
    result.setdefault("query_plan_sha256", "")
    result["schema_version"] = "rcl-query-active-deepening-current-method-unit-result-v1"
    return result


def _aggregate_current_method_results(
    unit_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str, bool], list[Mapping[str, Any]]] = {}
    for result in unit_results:
        key = (
            str(result.get("stage")),
            str(result.get("canonical_dataset_id")),
            str(result.get("method_id", result.get("selector_id", ""))),
            bool(result.get("oracle_only")),
        )
        buckets.setdefault(key, []).append(result)
    aggregates: dict[str, Any] = {}
    for (stage, dataset, method_id, oracle_only), rows in sorted(buckets.items()):
        aggregate_key = "%s.%s.%s" % (stage, dataset, method_id)
        view_summary = {}
        for view in ("T1", "T2"):
            view_summary[view] = {
                "seed_count": len(rows),
                "mean": {
                    metric: _mean(
                        [
                            _finite_float(
                                dict(dict(row["metrics"])[view]).get(metric),
                                "%s.%s.%s" % (aggregate_key, view, metric),
                            )
                            for row in rows
                        ]
                    )
                    for metric in RANKING_METRICS
                },
                "total_denominator": sum(
                    int(dict(dict(row["metrics"])[view])["denominator"])
                    for row in rows
                ),
            }
        aggregates[aggregate_key] = {
            "stage": stage,
            "canonical_dataset_id": dataset,
            "method_id": method_id,
            "oracle_only": oracle_only,
            "unit_ids": sorted(str(row["unit_id"]) for row in rows),
            "views": view_summary,
        }
    return aggregates


def execute_current_method_manifest(
    manifest: Mapping[str, Any],
    unit_executor: Any,
    *,
    allow_local_output_root: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Execute a current-method manifest and write durable terminal evidence."""

    validation = validate_current_method_formal_manifest(
        manifest,
        allow_local_output_root=allow_local_output_root,
        allow_hash_drift=allow_local_output_root,
    )
    run_root = Path(str(manifest["output_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    units = [deepcopy(dict(unit)) for unit in manifest.get("units") or []]
    unit_results: list[dict[str, Any]] = []
    try:
        _remove_if_present(run_root / "COMPLETED.json")
        _remove_if_present(run_root / "all.done")
        _remove_if_present(run_root / ".failed")
        initial_progress = build_deepening_progress_snapshot(
            stage="starting",
            strategy_family="current_method",
            dataset="",
            budget=None,
            seed=None,
            completed_units=0,
            total_units=len(units),
            failure_count=0,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        write_deepening_progress(run_root / "progress.json", initial_progress)
        print(format_deepening_progress_line(initial_progress), flush=True)
        def execute_one(index: int, unit: Mapping[str, Any]) -> dict[str, Any]:
            running_progress = build_deepening_progress_snapshot(
                stage=str(unit.get("stage")),
                strategy_family=str(unit.get("selector_family", unit.get("selector_id", ""))),
                dataset=str(unit.get("canonical_dataset_id", "")),
                budget=int(unit.get("budget", 0)),
                seed=int(unit.get("active_learning_seed", 0)),
                completed_units=index - 1,
                total_units=len(units),
                failure_count=0,
                recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
            )
            write_deepening_progress(run_root / "progress.json", running_progress)
            print(format_deepening_progress_line(running_progress), flush=True)
            raw = _call_unit_executor(unit_executor, unit, run_root)
            result = _normalize_current_method_unit_result(unit, raw)
            result_path = run_root / "unit_results" / _safe_token(unit["unit_id"]) / "result.json"
            _write_json_atomic(result_path, result)
            return result

        worker_count = max(1, int(max_workers))
        if worker_count == 1:
            for index, unit in enumerate(units, start=1):
                result = execute_one(index, unit)
                unit_results.append(result)
                completed_progress = build_deepening_progress_snapshot(
                    stage=str(unit.get("stage")),
                    strategy_family=str(unit.get("selector_family", unit.get("selector_id", ""))),
                    dataset=str(unit.get("canonical_dataset_id", "")),
                    budget=int(unit.get("budget", 0)),
                    seed=int(unit.get("active_learning_seed", 0)),
                    completed_units=index,
                    total_units=len(units),
                    failure_count=0,
                    recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
                )
                write_deepening_progress(run_root / "progress.json", completed_progress)
                print(format_deepening_progress_line(completed_progress), flush=True)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = {}
            with ThreadPoolExecutor(max_workers=min(worker_count, len(units))) as pool:
                for index, unit in enumerate(units, start=1):
                    futures[pool.submit(execute_one, index, unit)] = unit
                for future in as_completed(futures):
                    result = future.result()
                    unit_results.append(result)
                    completed_count = len(unit_results)
                    unit = futures[future]
                    completed_progress = build_deepening_progress_snapshot(
                        stage=str(unit.get("stage")),
                        strategy_family=str(
                            unit.get("selector_family", unit.get("selector_id", ""))
                        ),
                        dataset=str(unit.get("canonical_dataset_id", "")),
                        budget=int(unit.get("budget", 0)),
                        seed=int(unit.get("active_learning_seed", 0)),
                        completed_units=completed_count,
                        total_units=len(units),
                        failure_count=0,
                        recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
                    )
                    write_deepening_progress(run_root / "progress.json", completed_progress)
                    print(format_deepening_progress_line(completed_progress), flush=True)
        summary = {
            "schema_version": "rcl-query-active-deepening-current-method-terminal-summary-v1",
            "change_id": CHANGE_ID,
            "run_id": manifest.get("run_id"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "manifest_validation": validation,
            "unit_count": len(unit_results),
            "stage_sequence": list(manifest.get("stage_sequence") or []),
            "aggregates": _aggregate_current_method_results(unit_results),
            "clean_leaderboard": build_clean_leaderboard(unit_results),
            "t1_t2_warning_count": sum(
                len(result.get("t1_t2_warnings") or []) for result in unit_results
            ),
            "unit_results": {
                str(result["unit_id"]): result
                for result in sorted(unit_results, key=lambda row: str(row["unit_id"]))
            },
            "completed_at_utc": _utc_now(),
        }
        _write_json_atomic(run_root / "current_method_summary.json", summary)
        _write_json_atomic(run_root / "current_method_terminal_summary.json", summary)
        completion = {
            "schema_version": "rcl-query-active-deepening-current-method-completion-v1",
            "status": "complete",
            "manifest_sha256": manifest.get("manifest_sha256"),
            "terminal_summary_path": str(
                (run_root / "current_method_terminal_summary.json").resolve()
            ),
            "terminal_summary_sha256": _sha256_file(
                run_root / "current_method_terminal_summary.json"
            ),
            "unit_count": len(unit_results),
        }
        markers = _write_completion_markers(run_root, completion)
        final_progress = build_deepening_progress_snapshot(
            stage="complete",
            strategy_family="current_method",
            dataset="",
            budget=None,
            seed=None,
            completed_units=len(units),
            total_units=len(units),
            failure_count=0,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        write_deepening_progress(run_root / "progress.json", final_progress)
        print(format_deepening_progress_line(final_progress), flush=True)
        return {**completion, **markers}
    except Exception as exc:
        _remove_if_present(run_root / "COMPLETED.json")
        _remove_if_present(run_root / "all.done")
        _write_deepening_failure_marker(
            run_root,
            exc,
            completed_units=len(unit_results),
            total_units=len(units),
        )
        failed_progress = build_deepening_progress_snapshot(
            stage="failed",
            strategy_family="current_method",
            dataset="",
            budget=None,
            seed=None,
            completed_units=len(unit_results),
            total_units=len(units),
            failure_count=1,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        write_deepening_progress(run_root / "progress.json", failed_progress)
        print(format_deepening_progress_line(failed_progress), flush=True)
        raise


def execute_divergent_strategy_manifest(
    manifest: Mapping[str, Any],
    unit_executor: Any,
    *,
    allow_local_output_root: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Execute a divergent strategy manifest and write durable terminal evidence."""

    validation = validate_divergent_strategy_manifest(
        manifest,
        allow_local_output_root=allow_local_output_root,
        allow_hash_drift=allow_local_output_root,
    )
    run_root = Path(str(manifest["output_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    units = [deepcopy(dict(unit)) for unit in manifest.get("units") or []]
    unit_results: list[dict[str, Any]] = []
    try:
        _remove_if_present(run_root / "COMPLETED.json")
        _remove_if_present(run_root / "all.done")
        _remove_if_present(run_root / ".failed")
        initial_progress = build_deepening_progress_snapshot(
            stage="starting",
            strategy_family="divergent_strategy",
            dataset="",
            budget=None,
            seed=None,
            completed_units=0,
            total_units=len(units),
            failure_count=0,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        write_deepening_progress(run_root / "progress.json", initial_progress)
        print(format_deepening_progress_line(initial_progress), flush=True)

        def execute_one(index: int, unit: Mapping[str, Any]) -> dict[str, Any]:
            running_progress = build_deepening_progress_snapshot(
                stage=str(unit.get("stage")),
                strategy_family=str(unit.get("selector_family", unit.get("selector_id", ""))),
                dataset=str(unit.get("canonical_dataset_id", "")),
                budget=int(unit.get("budget", 0)),
                seed=int(unit.get("active_learning_seed", 0)),
                completed_units=index - 1,
                total_units=len(units),
                failure_count=0,
                recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
            )
            write_deepening_progress(run_root / "progress.json", running_progress)
            print(format_deepening_progress_line(running_progress), flush=True)
            raw = _call_unit_executor(unit_executor, unit, run_root)
            result = _normalize_current_method_unit_result(unit, raw)
            result["schema_version"] = (
                "rcl-query-active-deepening-divergent-strategy-unit-result-v1"
            )
            result_path = run_root / "unit_results" / _safe_token(unit["unit_id"]) / "result.json"
            _write_json_atomic(result_path, result)
            return result

        worker_count = max(1, int(max_workers))
        if worker_count == 1:
            for index, unit in enumerate(units, start=1):
                result = execute_one(index, unit)
                unit_results.append(result)
                completed_progress = build_deepening_progress_snapshot(
                    stage=str(unit.get("stage")),
                    strategy_family=str(unit.get("selector_family", unit.get("selector_id", ""))),
                    dataset=str(unit.get("canonical_dataset_id", "")),
                    budget=int(unit.get("budget", 0)),
                    seed=int(unit.get("active_learning_seed", 0)),
                    completed_units=index,
                    total_units=len(units),
                    failure_count=0,
                    recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
                )
                write_deepening_progress(run_root / "progress.json", completed_progress)
                print(format_deepening_progress_line(completed_progress), flush=True)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = {}
            with ThreadPoolExecutor(max_workers=min(worker_count, len(units))) as pool:
                for index, unit in enumerate(units, start=1):
                    futures[pool.submit(execute_one, index, unit)] = unit
                for future in as_completed(futures):
                    result = future.result()
                    unit_results.append(result)
                    completed_count = len(unit_results)
                    unit = futures[future]
                    completed_progress = build_deepening_progress_snapshot(
                        stage=str(unit.get("stage")),
                        strategy_family=str(
                            unit.get("selector_family", unit.get("selector_id", ""))
                        ),
                        dataset=str(unit.get("canonical_dataset_id", "")),
                        budget=int(unit.get("budget", 0)),
                        seed=int(unit.get("active_learning_seed", 0)),
                        completed_units=completed_count,
                        total_units=len(units),
                        failure_count=0,
                        recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
                    )
                    write_deepening_progress(run_root / "progress.json", completed_progress)
                    print(format_deepening_progress_line(completed_progress), flush=True)
        summary = {
            "schema_version": "rcl-query-active-deepening-divergent-strategy-terminal-summary-v1",
            "change_id": CHANGE_ID,
            "run_id": manifest.get("run_id"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "manifest_validation": validation,
            "current_method_terminal_summary_path": manifest.get(
                "current_method_terminal_summary_path"
            ),
            "unit_count": len(unit_results),
            "stage_sequence": list(manifest.get("stage_sequence") or []),
            "aggregates": _aggregate_current_method_results(unit_results),
            "clean_leaderboard": build_clean_leaderboard(unit_results),
            "t1_t2_warning_count": sum(
                len(result.get("t1_t2_warnings") or []) for result in unit_results
            ),
            "unit_results": {
                str(result["unit_id"]): result
                for result in sorted(unit_results, key=lambda row: str(row["unit_id"]))
            },
            "completed_at_utc": _utc_now(),
        }
        _write_json_atomic(run_root / "divergent_strategy_summary.json", summary)
        _write_json_atomic(run_root / "divergent_strategy_terminal_summary.json", summary)
        completion = {
            "schema_version": "rcl-query-active-deepening-divergent-strategy-completion-v1",
            "status": "complete",
            "manifest_sha256": manifest.get("manifest_sha256"),
            "terminal_summary_path": str(
                (run_root / "divergent_strategy_terminal_summary.json").resolve()
            ),
            "terminal_summary_sha256": _sha256_file(
                run_root / "divergent_strategy_terminal_summary.json"
            ),
            "unit_count": len(unit_results),
        }
        markers = _write_completion_markers(run_root, completion)
        final_progress = build_deepening_progress_snapshot(
            stage="complete",
            strategy_family="divergent_strategy",
            dataset="",
            budget=None,
            seed=None,
            completed_units=len(units),
            total_units=len(units),
            failure_count=0,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        write_deepening_progress(run_root / "progress.json", final_progress)
        print(format_deepening_progress_line(final_progress), flush=True)
        return {**completion, **markers}
    except Exception as exc:
        _remove_if_present(run_root / "COMPLETED.json")
        _remove_if_present(run_root / "all.done")
        _write_deepening_failure_marker(
            run_root,
            exc,
            completed_units=len(unit_results),
            total_units=len(units),
        )
        failed_progress = build_deepening_progress_snapshot(
            stage="failed",
            strategy_family="divergent_strategy",
            dataset="",
            budget=None,
            seed=None,
            completed_units=len(unit_results),
            total_units=len(units),
            failure_count=1,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        write_deepening_progress(run_root / "progress.json", failed_progress)
        print(format_deepening_progress_line(failed_progress), flush=True)
        raise


def _safe_path_component(value: Any, context: str) -> str:
    text = str(value).strip()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError("unsafe %s path component: %r" % (context, value))
    return text


def deepening_unit_output_root(
    *,
    base_root: str,
    stage: str,
    strategy_family: str,
    dataset: str,
    budget: int | None = None,
    seed: int | None = None,
    run_id: str,
) -> str:
    """Return an isolated output root for one unit."""

    parts = [
        str(base_root).rstrip("/"),
        _safe_path_component(stage, "stage"),
        _safe_path_component(strategy_family, "strategy_family"),
        _safe_path_component(dataset, "dataset"),
    ]
    if budget is not None:
        parts.append("budget%d" % int(budget))
    if seed is not None:
        parts.append("seed%d" % int(seed))
    parts.append(_safe_path_component(run_id, "run_id"))
    return "/".join(parts)


def build_deepening_progress_snapshot(
    *,
    stage: str,
    strategy_family: str,
    dataset: str,
    budget: int | None,
    seed: int | None,
    completed_units: int,
    total_units: int,
    failure_count: int,
    recent_log_path: str,
) -> dict[str, Any]:
    total = int(total_units)
    completed = int(completed_units)
    if total < 0 or completed < 0 or completed > total:
        raise ValueError("invalid progress completed/total counts")
    return {
        "schema_version": "rcl-query-active-deepening-progress-v1",
        "stage": str(stage),
        "strategy_family": str(strategy_family),
        "dataset": str(dataset),
        "budget": None if budget is None else int(budget),
        "seed": None if seed is None else int(seed),
        "completed_units": completed,
        "total_units": total,
        "failure_count": int(failure_count),
        "recent_log_path": str(recent_log_path),
        "updated_at_utc": _utc_now(),
    }


def write_deepening_progress(path: Path | str, progress: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(progress))
    if payload.get("schema_version") != "rcl-query-active-deepening-progress-v1":
        raise ValueError("unexpected deepening progress schema")
    _write_json_atomic(Path(path), payload)
    return payload


def build_deepening_status_snapshot(
    *,
    run_root: Path | str,
    tmux_session: str,
    tmux_alive: bool,
) -> dict[str, Any]:
    root = Path(run_root)
    progress_path = root / "progress.json"
    progress: dict[str, Any] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    failed_path = root / ".failed"
    completed_path = root / "COMPLETED.json"
    all_done_path = root / "all.done"
    return {
        "schema_version": "rcl-query-active-deepening-status-v1",
        "run_root": str(root),
        "tmux_session": str(tmux_session),
        "tmux_alive": bool(tmux_alive),
        "progress": progress,
        "completion": {
            "COMPLETED.json": str(completed_path),
            "COMPLETED_json_exists": completed_path.is_file(),
            "all.done": str(all_done_path),
            "all_done_exists": all_done_path.is_file(),
        },
        "failure": {
            ".failed": str(failed_path),
            "failed_exists": failed_path.is_file(),
        },
    }


def list_deepening_selectors() -> dict[str, dict[str, Any]]:
    """Return selector metadata with explicit clean/oracle boundaries."""

    return {
        "true_fault_type_oracle": {
            "selector_id": "true_fault_type_oracle",
            "family_id": "oracle_fault_type_coverage",
            "oracle_only": True,
            "allowed_input_contract": "full_candidate_fault_type_oracle",
        },
        "metric_signature_balanced_uncertainty": {
            "selector_id": "metric_signature_balanced_uncertainty",
            "family_id": "metric_signature_proxy_balance",
            "oracle_only": False,
            "allowed_input_contract": "label_free_only",
        },
        "sequential_proxy_mode_query": {
            "selector_id": "sequential_proxy_mode_query",
            "family_id": "sequential_proxy_mode",
            "oracle_only": False,
            "allowed_input_contract": "label_free_plus_budgeted_revealed_labels",
        },
        "ranker_aware_acquisition": {
            "selector_id": "ranker_aware_acquisition",
            "family_id": "ranker_aware",
            "oracle_only": False,
            "allowed_input_contract": "label_free_ranker_utility",
        },
        "richer_metric_signature_acquisition": {
            "selector_id": "richer_metric_signature_acquisition",
            "family_id": "richer_metric_signature",
            "oracle_only": False,
            "allowed_input_contract": "label_free_metric_ad_signature",
        },
        "budgeted_fault_mode_modeling": {
            "selector_id": "budgeted_fault_mode_modeling",
            "family_id": "budgeted_fault_mode_model",
            "oracle_only": False,
            "allowed_input_contract": "label_free_plus_budgeted_revealed_labels",
        },
        "facility_location_selection": {
            "selector_id": "facility_location_selection",
            "family_id": "facility_location",
            "oracle_only": False,
            "allowed_input_contract": "label_free_similarity",
        },
        "unknown_mode_signature_reserve": {
            "selector_id": "unknown_mode_signature_reserve",
            "family_id": "unknown_mode_signature_reserve",
            "oracle_only": False,
            "allowed_input_contract": "label_free_metric_signature_unknown_mode_reserve",
        },
        "winner_like_distillation": {
            "selector_id": "winner_like_distillation",
            "family_id": "winner_like_distillation",
            "oracle_only": False,
            "allowed_input_contract": "label_free_winner_profile",
        },
        "mpca_dual_factor_query": {
            "selector_id": "mpca_dual_factor_query",
            "family_id": "mpca_dual_factor",
            "oracle_only": False,
            "allowed_input_contract": "label_free_mechanism_propagation_proxy",
        },
    }


def _require_outer_train_fault_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [deepcopy(dict(row)) for row in candidates]
    if not rows:
        raise ValueError("selector candidates must not be empty")
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("selector candidate case_id must not be empty")
        if case_id in seen:
            raise ValueError("selector candidate case IDs must be unique")
        seen.add(case_id)
        if str(row.get("split")) != "outer_train" or str(row.get("case_kind")) != "fault":
            raise ValueError("selector candidates must be outer_train fault cases")
    return rows


def select_true_fault_type_oracle_cases(
    candidates: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    seed: int = 42,
    score_field: str = "inner_boundary_uncertainty",
) -> dict[str, Any]:
    """Diagnostic-only selector that balances true fault types."""

    rows = _require_outer_train_fault_candidates(candidates)
    if int(budget) <= 0 or int(budget) > len(rows):
        raise ValueError("budget must fit candidate count")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fault_type = str(row.get("fault_type_oracle", "")).strip()
        if not fault_type:
            raise ValueError("true fault-type oracle selector requires fault_type_oracle")
        grouped.setdefault(fault_type, []).append(row)
    selected: list[dict[str, Any]] = []
    for fault_type in sorted(grouped):
        group = sorted(
            grouped[fault_type],
            key=lambda row: (
                -_finite_float(row.get(score_field, 0.0), score_field),
                _semantic_sha256({"case_id": row["case_id"], "seed": int(seed)}),
            ),
        )
        if group and len(selected) < int(budget):
            selected.append(group[0])
    if len(selected) < int(budget):
        selected_ids = {str(row["case_id"]) for row in selected}
        leftovers = sorted(
            [row for row in rows if str(row["case_id"]) not in selected_ids],
            key=lambda row: (
                -_finite_float(row.get(score_field, 0.0), score_field),
                _semantic_sha256({"case_id": row["case_id"], "seed": int(seed)}),
            ),
        )
        selected.extend(leftovers[: int(budget) - len(selected)])
    coverage: dict[str, int] = {}
    for row in selected:
        fault_type = str(row["fault_type_oracle"])
        coverage[fault_type] = coverage.get(fault_type, 0) + 1
    return {
        "schema_version": "rcl-query-active-true-fault-type-oracle-selection-v1",
        "selector_id": "true_fault_type_oracle",
        "oracle_only": True,
        "budget": int(budget),
        "seed": int(seed),
        "selected_case_ids": [str(row["case_id"]) for row in selected],
        "selected_cases": selected,
        "diagnostics": {
            "fault_type_oracle_coverage": dict(sorted(coverage.items())),
        },
        "selector_result_sha256": _semantic_sha256(
            {
                "selector_id": "true_fault_type_oracle",
                "budget": int(budget),
                "seed": int(seed),
                "selected_case_ids": [str(row["case_id"]) for row in selected],
            }
        ),
    }


def validate_budgeted_label_access(
    *,
    requested_case_ids: Sequence[Any],
    queried_case_ids: Sequence[Any],
) -> dict[str, Any]:
    queried = {str(case_id) for case_id in queried_case_ids}
    requested = [str(case_id) for case_id in requested_case_ids]
    unbudgeted = sorted(case_id for case_id in requested if case_id not in queried)
    if unbudgeted:
        raise ValueError("unbudgeted label access attempted for cases: %s" % unbudgeted)
    return {
        "valid": True,
        "requested_case_ids": requested,
        "queried_case_ids": sorted(queried),
    }


def build_deepening_query_plan(
    *,
    canonical_dataset_id: str,
    selector_id: str,
    selector_config: Mapping[str, Any],
    selected_case_ids: Sequence[Any],
    authoritative_labels: Mapping[str, Sequence[Any]],
    active_learning_seed: int,
    split_seed: int,
    feature_hash: str,
    oracle_only: bool = False,
) -> dict[str, Any]:
    """Freeze a budgeted query plan with labels only for selected cases."""

    dataset = resolve_canonical_dataset(canonical_dataset_id)["canonical_dataset_id"]
    selected = [str(case_id) for case_id in selected_case_ids]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("query plan requires unique selected case IDs")
    validate_budgeted_label_access(
        requested_case_ids=list(authoritative_labels),
        queried_case_ids=selected,
    )
    annotations = []
    for case_id in selected:
        targets = [
            str(target).strip()
            for target in authoritative_labels.get(case_id, ())
            if str(target).strip()
        ]
        if not targets:
            raise ValueError("queried case has no authoritative label: %s" % case_id)
        annotations.append(
            {
                "case_id": case_id,
                "annotation_source": "simulated_manual_ground_truth",
                "targets": targets,
            }
        )
    identity = {
        "schema_version": "rcl-query-active-deepening-query-plan-v1",
        "canonical_dataset_id": dataset,
        "selector_id": str(selector_id),
        "selector_config": deepcopy(dict(selector_config)),
        "selected_case_ids": selected,
        "active_learning_seed": int(active_learning_seed),
        "split_seed": int(split_seed),
        "feature_hash": str(feature_hash),
        "oracle_only": bool(oracle_only),
    }
    return {
        **identity,
        "budget": len(selected),
        "annotations": annotations,
        "query_plan_sha256": _semantic_sha256(identity),
    }
