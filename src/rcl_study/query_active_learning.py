"""Protocol helpers for the budget-30 query-only active-learning change."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


CHANGE_ID = "optimize-query-only-active-learning"
RUN_ID = "query-only-active-20260817-01"
REMOTE_WORKSPACE = "${RCL_WORKSPACE}"
APPROVED_OUTPUT_ROOT = (
    REMOTE_WORKSPACE
    + "/outputs/rcl_study/query_active_learning/"
    + RUN_ID
)
TARGET_DATASETS = {
    "aiops2022_pre": "${AIOPS22_ROOT}",
    "rcabench": "${RCABENCH_ROOT}",
}
SOTA_THRESHOLDS = {
    "aiops2022_pre": {
        "hit_at_1": 0.5210,
        "hit_at_3": 0.8270,
        "hit_at_5": 0.8960,
    },
    "rcabench": {
        "hit_at_1": 0.4175,
        "hit_at_3": 0.6351,
        "hit_at_5": 0.7368,
    },
}
RCABENCH_EXISTING_ORACLE_RESULT = (
    REMOTE_WORKSPACE
    + "/outputs/rcl_study/clean_baseline/20260726_clean_baseline_04/"
    + "formal_units/semi.rcabench.oracle_full.seed_42/result.json"
)
RCABENCH_EXISTING_ORACLE_SHA256 = (
    "2faa28fc0d8b3207ce796f7640ce886a17b887c1e8dbba7b4d779ed27f515c18"
)
HIT_METRICS = ("hit_at_1", "hit_at_3", "hit_at_5")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_INVENTORY_COUNTS = (
    "outer_train_fault_cases",
    "outer_test_fault_cases",
    "inner_train_fault_cases",
    "inner_validation_fault_cases",
)
_FORBIDDEN_SELECTOR_KEY_TOKENS = (
    "outer_test_case_ids",
    "outer_test_label",
    "outer_test_labels",
    "final_hit",
    "hit_at_",
    "prediction",
    "predictions",
    "targets",
    "target_set",
    "root_cause",
    "rootcause",
    "fault_type",
    "true_fault_type",
    "groundtruth",
    "ground_truth",
    "oracle_label",
)
_FORBIDDEN_SELECTOR_VALUE_TOKENS = (
    "outer_test_case_ids",
    "labels_eval_only",
    "final_hit",
    "groundtruth",
    "ground_truth",
    "/final/",
    "\\final\\",
)
_FORMAL_TARGET_PAIR_TIERS = {
    "formal_target_pair",
    "formal_target_pair_oracle_gate",
}
_QUERY_SELECTOR_REGISTRY = {
    "scheme1_baseline": {
        "selector_id": "scheme1_baseline",
        "strategy_family": "matched_baseline",
        "description": (
            "Matched Scheme-1 query selector replay. Uses a pre-frozen "
            "scheme1_rank/rank field when available, otherwise deterministic "
            "baseline scores or case-id order without reading labels."
        ),
    },
    "diversity_maxmin": {
        "selector_id": "diversity_maxmin",
        "strategy_family": "diversity",
        "description": (
            "Deterministic k-center/max-min coreset selector on pre-annotation "
            "case embeddings."
        ),
    },
    "stratified_cluster_quota": {
        "selector_id": "stratified_cluster_quota",
        "strategy_family": "stratified",
        "description": (
            "Round-robin quota selector over available pre-annotation "
            "metadata such as cluster, service, or time buckets."
        ),
    },
    "uncertainty_boundary": {
        "selector_id": "uncertainty_boundary",
        "strategy_family": "uncertainty",
        "description": (
            "Boundary/uncertainty selector using inner-training or "
            "inner-validation evidence only."
        ),
    },
    "coverage_capped_uncertainty": {
        "selector_id": "coverage_capped_uncertainty",
        "strategy_family": "coverage_capped_uncertainty",
        "description": (
            "Uncertainty-first selector with pre-annotation coverage caps and "
            "embedding-distance tie-breaking to avoid narrow budget collapse."
        ),
    },
    "metric_signature_balanced_uncertainty": {
        "selector_id": "metric_signature_balanced_uncertainty",
        "strategy_family": "metric_signature_proxy_balance",
        "description": (
            "Label-clean metric-signature proxy-mode selector that balances "
            "fault-mode proxies before uncertainty/value tie-breaking."
        ),
    },
    "sequential_proxy_mode_query": {
        "selector_id": "sequential_proxy_mode_query",
        "strategy_family": "sequential_proxy_mode",
        "description": (
            "Budgeted multi-round proxy-mode selector with label-free seed "
            "selection and explicit accounting for revealed labels."
        ),
    },
    "two_stage_diverse_uncertainty": {
        "selector_id": "two_stage_diverse_uncertainty",
        "strategy_family": "two_stage",
        "description": (
            "Small diverse seed batch followed by inner-evidence "
            "uncertainty/value scoring."
        ),
    },
    "ranker_aware_acquisition": {
        "selector_id": "ranker_aware_acquisition",
        "strategy_family": "ranker_aware",
        "description": (
            "Label-clean acquisition that estimates candidate ranker impact "
            "from inner evidence, baseline rank/value, and diversity gain."
        ),
    },
    "richer_metric_signature_acquisition": {
        "selector_id": "richer_metric_signature_acquisition",
        "strategy_family": "richer_metric_signature",
        "description": (
            "Metric_AD-aware signature selector using richer anomaly direction, "
            "duration, propagation, sparsity, robust-z, and temporal features "
            "when present."
        ),
    },
    "budgeted_fault_mode_modeling": {
        "selector_id": "budgeted_fault_mode_modeling",
        "strategy_family": "budgeted_fault_mode_model",
        "description": (
            "Multi-round active selector that uses only already queried labels "
            "to train a coarse fault-mode surrogate before filling the budget."
        ),
    },
    "facility_location_selection": {
        "selector_id": "facility_location_selection",
        "strategy_family": "facility_location",
        "description": (
            "Submodular facility-location selector over label-free candidate "
            "similarities, with uncertainty and mode-coverage terms."
        ),
    },
    "winner_like_distillation": {
        "selector_id": "winner_like_distillation",
        "strategy_family": "winner_like_distillation",
        "description": (
            "Distills high-vs-low random diagnostic set preferences into a "
            "label-free feature-profile score without replaying lucky case IDs."
        ),
    },
    "unknown_mode_signature_reserve": {
        "selector_id": "unknown_mode_signature_reserve",
        "strategy_family": "unknown_mode_signature_reserve",
        "description": (
            "Label-clean richer-signature selector that reserves budget for "
            "metric regions far from already queried cases while greedily "
            "covering the whole candidate pool."
        ),
    },
    "mpca_dual_factor_query": {
        "selector_id": "mpca_dual_factor_query",
        "strategy_family": "mpca_dual_factor",
        "description": (
            "Mechanism-propagation compositional acquisition using only "
            "label-free mechanism proxies, propagation/context proxies, "
            "uncertainty, and joint novelty."
        ),
    },
}
_COVERAGE_DIAGNOSTIC_FIELDS = (
    "service",
    "service_name",
    "fault",
    "proxy_fault_mode",
    "time_bucket",
    "day",
    "cluster_id",
)
_FIXED_REPLAY_CONFIG_KEYS = (
    "selected_case_ids",
    "fixed_case_ids",
    "diagnostic_case_ids",
    "random_case_ids",
    "sample_set_hash",
)
_CASE_ID_SEMANTIC_CONFIG_TOKENS = (
    "parse_case_id",
    "parse_native_case_id",
    "case_id_regex",
    "native_case_id_regex",
    "derive_from_case_id",
    "derive_from_native_case_id",
    "id_fault",
    "id_service",
)
_METRIC_SIGNATURE_SCHEMA_VERSION = "rcl-query-active-metric-signature-v1"
_PROXY_ASSIGNMENT_SCHEMA_VERSION = "rcl-query-active-proxy-fault-mode-v1"


def _semantic_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    text = str(value)
    if not SHA256_RE.fullmatch(text):
        raise ValueError("%s must be a lowercase SHA-256" % context)
    return text


def _require_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("budget must be a positive integer")
    return value


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id", "")).strip()


def _prepare_selector_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selector_id: str,
    budget: int,
) -> list[Dict[str, Any]]:
    audit_selector_input_contract(
        {
            "selector_id": selector_id,
            "selection_partition": "outer_training_inner_validation_only",
            "candidates": list(candidates),
        }
    )
    rows = [deepcopy(dict(row)) for row in candidates]
    if len(rows) < budget:
        raise ValueError("selector requires at least 30 admitted candidate cases")
    seen: set[str] = set()
    for row in rows:
        case_id = _candidate_id(row)
        if not case_id:
            raise ValueError("candidate case_id must not be empty")
        if case_id in seen:
            raise ValueError("selector requires unique candidate case IDs")
        seen.add(case_id)
        if str(row.get("split", "")) != "outer_train" or str(
            row.get("case_kind", "")
        ) != "fault":
            raise ValueError("selector candidates must be outer_train fault cases")
    return rows


def _selector_config_payload(
    *,
    selector_id: str,
    budget: int,
    seed: int,
    selector_config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "selector_id": str(selector_id),
        "budget": int(budget),
        "seed": int(seed),
        "selector_config": deepcopy(dict(selector_config)),
    }


def _coverage_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    coverage: Dict[str, Dict[str, int]] = {}
    for field in _COVERAGE_DIAGNOSTIC_FIELDS:
        counts: Dict[str, int] = {}
        for row in rows:
            if field not in row:
                continue
            value = str(row.get(field)).strip()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        if counts:
            coverage[field] = dict(sorted(counts.items()))
    return coverage


def _stage_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        stage = str(row.get("selection_stage", "")).strip()
        if not stage:
            continue
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _query_rounds(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rounds: Dict[int, list[str]] = {}
    for row in rows:
        raw_round = row.get("query_round")
        if raw_round is None:
            continue
        round_index = int(raw_round)
        if round_index <= 0:
            raise ValueError("query_round must be positive")
        rounds.setdefault(round_index, []).append(_candidate_id(row))
    return [rounds[index] for index in sorted(rounds)]


def _proxy_mode_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    counts: Dict[str, int] = {}
    for row in rows:
        mode = str(row.get("proxy_fault_mode", "")).strip()
        if not mode:
            continue
        counts[mode] = counts.get(mode, 0) + 1
    if not counts:
        return None
    total = sum(counts.values())
    hhi = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
    return {
        "proxy_mode_count": len(counts),
        "proxy_mode_counts": dict(sorted(counts.items())),
        "top_proxy_mode_count": max(counts.values()),
        "proxy_mode_hhi": hhi,
    }


def _seed_tiebreak(selector_id: str, case_id: str, seed: int) -> str:
    return _semantic_sha256(
        {
            "selector_id": selector_id,
            "case_id": case_id,
            "seed": int(seed),
        }
    )


def _selector_result(
    *,
    selector_id: str,
    budget: int,
    seed: int,
    selector_config: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    config_payload = _selector_config_payload(
        selector_id=selector_id,
        budget=budget,
        seed=seed,
        selector_config=selector_config,
    )
    selected: list[Dict[str, Any]] = []
    extra_diagnostics: Dict[str, Any] = {}
    for raw_row in selected_cases:
        row = deepcopy(dict(raw_row))
        embedded = row.pop("_selector_diagnostics", None)
        if isinstance(embedded, Mapping):
            for key, value in embedded.items():
                extra_diagnostics.setdefault(str(key), deepcopy(value))
        selected.append(row)
    selected_ids = [_candidate_id(row) for row in selected]
    identity = {
        "schema_version": "rcl-query-active-selector-result-v1",
        "selector_id": selector_id,
        "budget": budget,
        "seed": int(seed),
        "selector_config": deepcopy(dict(selector_config)),
        "selector_config_sha256": _semantic_sha256(config_payload),
        "selected_case_ids": selected_ids,
        "diagnostics": {
            "coverage": _coverage_counts(selected),
            "selected_case_count": len(selected),
        },
    }
    identity["diagnostics"].update(extra_diagnostics)
    stages = _stage_counts(selected)
    if stages:
        identity["diagnostics"]["stage_counts"] = stages
    proxy_diagnostics = _proxy_mode_diagnostics(selected)
    if proxy_diagnostics:
        identity["diagnostics"]["proxy_modes"] = proxy_diagnostics
    rounds = _query_rounds(selected)
    if rounds:
        identity["diagnostics"]["query_rounds"] = rounds
        identity["diagnostics"]["query_round_counts"] = [
            len(round_ids) for round_ids in rounds
        ]
        first_stage = stages.get("proxy_diverse_seed", 0) if stages else 0
        identity["diagnostics"]["budgeted_revealed_label_count"] = first_stage
    return {
        **identity,
        "selected_cases": selected,
        "selector_result_sha256": _semantic_sha256(identity),
    }


def _numeric_rank(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        rank = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rank):
        return None
    return rank


def _scheme1_baseline_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    seed: int,
) -> list[Mapping[str, Any]]:
    rank_field = str(selector_config.get("rank_field", "scheme1_rank"))
    ranked = []
    fallback = []
    for row in candidates:
        rank = _numeric_rank(row.get(rank_field))
        if rank is not None and rank > 0:
            case_id = _candidate_id(row)
            ranked.append(
                (
                    rank,
                    _seed_tiebreak("scheme1_baseline", case_id, seed),
                    case_id,
                    row,
                )
            )
        else:
            fallback.append(row)
    if ranked:
        ordered = [row for _, _, _, row in sorted(ranked)]
        ordered.extend(
            sorted(
                fallback,
                key=lambda row: (
                    -float(row.get("baseline_score", 0.0) or 0.0),
                    _seed_tiebreak("scheme1_baseline", _candidate_id(row), seed),
                    _candidate_id(row),
                ),
            )
        )
        return ordered
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.get("baseline_score", 0.0) or 0.0),
            _seed_tiebreak("scheme1_baseline", _candidate_id(row), seed),
            _candidate_id(row),
        ),
    )


def _embedding_vector(row: Mapping[str, Any], field: str) -> tuple[float, ...]:
    raw = row.get(field)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("diversity_maxmin requires numeric case embeddings")
    vector = tuple(float(value) for value in raw)
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("diversity_maxmin requires finite numeric embeddings")
    return vector


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _config_key_findings(value: Any, tokens: Sequence[str], path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = "%s.%s" % (path, key_text) if path else key_text
            lowered = key_text.lower()
            if any(token in lowered for token in tokens):
                findings.append(child_path)
            findings.extend(_config_key_findings(child, tokens, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            child_path = "%s[%d]" % (path, index) if path else "[%d]" % index
            findings.extend(_config_key_findings(child, tokens, child_path))
    return findings


def _reject_case_id_semantic_config(selector_config: Mapping[str, Any]) -> None:
    findings = sorted(
        set(_config_key_findings(selector_config, _CASE_ID_SEMANTIC_CONFIG_TOKENS))
    )
    if findings:
        raise ValueError(
            "case-id semantic parsing is forbidden for stable selectors: %s"
            % findings
        )


def _stable_unit_interval(value: Any) -> float:
    digest = _semantic_sha256({"value": str(value)})
    return int(digest[:12], 16) / float(16**12 - 1)


def _config_sequence(
    selector_config: Mapping[str, Any],
    key: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    raw = selector_config.get(key, list(default))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("%s must be a sequence" % key)
    return tuple(str(value).strip() for value in raw if str(value).strip())


def build_metric_signatures(
    candidates: Sequence[Mapping[str, Any]],
    *,
    signature_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build label-clean metric signatures for active-query candidates."""

    config = deepcopy(dict(signature_config or {}))
    audit_selector_input_contract(
        {
            "selector_id": "metric_signature_builder",
            "selection_partition": "outer_training_inner_validation_only",
            "candidates": list(candidates),
            "signature_config": config,
        }
    )
    _reject_case_id_semantic_config(config)
    embedding_field = str(config.get("embedding_field", "embedding"))
    numeric_fields = _config_sequence(
        config,
        "numeric_fields",
        ("inner_boundary_uncertainty", "baseline_score", "scheme1_rank"),
    )
    categorical_fields = _config_sequence(
        config,
        "categorical_fields",
        ("cluster_id", "time_bucket"),
    )
    metric_ad_fields = _config_sequence(config, "metric_ad_fields", ())
    metric_ad_required = bool(config.get("metric_ad_required", False))
    used_fields: set[str] = set()
    rows: list[Dict[str, Any]] = []
    metric_ad_available = bool(metric_ad_fields) and all(
        all(field in row for field in metric_ad_fields) for row in candidates
    )
    for row in candidates:
        copied = deepcopy(dict(row))
        signature: list[float] = []
        source_fields: list[str] = []
        if embedding_field in row:
            vector = _embedding_vector(row, embedding_field)
            signature.extend(float(value) for value in vector)
            source_fields.extend("%s[%d]" % (embedding_field, i) for i in range(len(vector)))
            used_fields.add(embedding_field)
        for field in numeric_fields:
            score = _numeric_rank(row.get(field))
            if score is None:
                continue
            signature.append(score)
            source_fields.append(field)
            used_fields.add(field)
        for field in categorical_fields:
            if field not in row:
                continue
            signature.append(_stable_unit_interval(row.get(field)))
            source_fields.append(field)
            used_fields.add(field)
        if metric_ad_available:
            for field in metric_ad_fields:
                score = _numeric_rank(row.get(field))
                if score is None:
                    continue
                signature.append(score)
                source_fields.append(field)
                used_fields.add(field)
        if not signature:
            raise ValueError("metric signature requires at least one label-free field")
        if any(not math.isfinite(value) for value in signature):
            raise ValueError("metric signature values must be finite")
        copied["metric_signature"] = signature
        copied["metric_signature_fields"] = source_fields
        rows.append(copied)
    if metric_ad_available:
        metric_ad_status = {
            "status": "included",
            "fields": list(metric_ad_fields),
        }
    elif metric_ad_required:
        metric_ad_status = {
            "status": "fallback_metric_ad_unavailable",
            "fields": list(metric_ad_fields),
            "reason": "metric_AD event-summary fields were absent from candidates",
        }
    else:
        metric_ad_status = {"status": "fallback_not_required", "fields": []}
    identity = {
        "schema_version": _METRIC_SIGNATURE_SCHEMA_VERSION,
        "signature_config": config,
        "signature_config_sha256": _semantic_sha256(config),
        "allowed_input_fields": sorted(used_fields),
        "forbidden_fields_absent": True,
        "metric_ad_status": metric_ad_status,
        "candidate_signatures": rows,
    }
    identity["signature_bundle_sha256"] = _semantic_sha256(
        {
            "schema_version": identity["schema_version"],
            "signature_config_sha256": identity["signature_config_sha256"],
            "case_signatures": [
                {
                    "case_id": _candidate_id(row),
                    "metric_signature": row["metric_signature"],
                    "metric_signature_fields": row["metric_signature_fields"],
                }
                for row in rows
            ],
        }
    )
    return identity


def _normalized_vectors(rows: Sequence[Mapping[str, Any]]) -> Dict[str, tuple[float, ...]]:
    raw_vectors = {
        _candidate_id(row): tuple(float(value) for value in row["metric_signature"])
        for row in rows
    }
    dimensions = {len(vector) for vector in raw_vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("proxy-mode assignment requires equal signature dimensions")
    dimension = next(iter(dimensions))
    lows = [min(vector[index] for vector in raw_vectors.values()) for index in range(dimension)]
    highs = [max(vector[index] for vector in raw_vectors.values()) for index in range(dimension)]
    normalized: Dict[str, tuple[float, ...]] = {}
    for case_id, vector in raw_vectors.items():
        values = []
        for index, value in enumerate(vector):
            low = lows[index]
            high = highs[index]
            values.append(0.0 if high == low else (value - low) / (high - low))
        normalized[case_id] = tuple(values)
    return normalized


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("cannot average empty vectors")
    dimension = len(vectors[0])
    return tuple(
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(dimension)
    )


def assign_proxy_fault_modes(
    candidate_signatures: Sequence[Mapping[str, Any]],
    *,
    proxy_config: Mapping[str, Any] | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Assign deterministic label-free proxy fault modes from metric signatures."""

    config = deepcopy(dict(proxy_config or {}))
    audit_selector_input_contract(
        {
            "selector_id": "proxy_fault_mode_assignment",
            "selection_partition": "outer_training_inner_validation_only",
            "candidate_signatures": list(candidate_signatures),
            "proxy_config": config,
        }
    )
    _reject_case_id_semantic_config(config)
    rows = [deepcopy(dict(row)) for row in candidate_signatures]
    if not rows:
        raise ValueError("proxy-mode assignment requires candidates")
    requested = int(config.get("proxy_mode_count", min(14, len(rows))))
    if requested <= 0:
        raise ValueError("proxy_mode_count must be positive")
    k = min(requested, len(rows))
    algorithm = str(config.get("algorithm", "deterministic_kmeans"))
    if algorithm != "deterministic_kmeans":
        raise ValueError("unsupported proxy-mode algorithm: %s" % algorithm)
    vectors = _normalized_vectors(rows)
    case_ids = sorted(vectors)
    global_centroid = _mean_vector([vectors[case_id] for case_id in case_ids])
    first_id = min(
        case_ids,
        key=lambda case_id: (
            _squared_distance(vectors[case_id], global_centroid),
            _seed_tiebreak("proxy_fault_mode", case_id, seed),
            case_id,
        ),
    )
    center_ids = [first_id]
    while len(center_ids) < k:
        remaining = [case_id for case_id in case_ids if case_id not in center_ids]
        next_id = max(
            remaining,
            key=lambda case_id: (
                min(
                    _squared_distance(vectors[case_id], vectors[center_id])
                    for center_id in center_ids
                ),
                _seed_tiebreak("proxy_fault_mode", case_id, seed),
                case_id,
            ),
        )
        center_ids.append(next_id)
    centers = [vectors[case_id] for case_id in center_ids]
    iterations = int(config.get("iterations", 8))
    assignments: Dict[str, int] = {}
    for _ in range(max(1, iterations)):
        assignments = {
            case_id: min(
                range(len(centers)),
                key=lambda index: (
                    _squared_distance(vectors[case_id], centers[index]),
                    index,
                ),
            )
            for case_id in case_ids
        }
        new_centers = []
        for index, center in enumerate(centers):
            assigned_vectors = [
                vectors[case_id]
                for case_id in case_ids
                if assignments.get(case_id) == index
            ]
            new_centers.append(_mean_vector(assigned_vectors) if assigned_vectors else center)
        centers = new_centers
    by_id = {_candidate_id(row): row for row in rows}
    assigned_rows: list[Dict[str, Any]] = []
    for case_id in case_ids:
        mode_index = assignments[case_id]
        row = deepcopy(dict(by_id[case_id]))
        row["proxy_fault_mode"] = "proxy-%02d" % mode_index
        row["proxy_distance"] = math.sqrt(
            _squared_distance(vectors[case_id], centers[mode_index])
        )
        assigned_rows.append(row)
    proxy_diag = _proxy_mode_diagnostics(assigned_rows) or {
        "proxy_mode_count": 0,
        "proxy_mode_counts": {},
        "top_proxy_mode_count": 0,
        "proxy_mode_hhi": 0.0,
    }
    identity = {
        "schema_version": _PROXY_ASSIGNMENT_SCHEMA_VERSION,
        "proxy_config": config,
        "proxy_config_sha256": _semantic_sha256(config),
        "seed": int(seed),
        "algorithm": algorithm,
        "assigned_candidates": assigned_rows,
        "diagnostics": proxy_diag,
    }
    identity["proxy_assignment_sha256"] = _semantic_sha256(
        {
            "schema_version": identity["schema_version"],
            "proxy_config_sha256": identity["proxy_config_sha256"],
            "seed": int(seed),
            "assignments": [
                {
                    "case_id": _candidate_id(row),
                    "proxy_fault_mode": row["proxy_fault_mode"],
                }
                for row in assigned_rows
            ],
        }
    )
    return identity


def _diversity_maxmin_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    seed: int,
) -> list[Mapping[str, Any]]:
    embedding_field = str(selector_config.get("embedding_field", "embedding"))
    vectors = {
        _candidate_id(row): _embedding_vector(row, embedding_field)
        for row in candidates
    }
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("diversity_maxmin requires equal embedding dimensions")
    dim = next(iter(dimensions))
    centroid = tuple(
        sum(vector[index] for vector in vectors.values()) / len(vectors)
        for index in range(dim)
    )
    by_id = {_candidate_id(row): row for row in candidates}
    first_id = min(
        vectors,
        key=lambda case_id: (
            _squared_distance(vectors[case_id], centroid),
            _seed_tiebreak("diversity_maxmin", case_id, seed),
            case_id,
        ),
    )
    selected_ids = [first_id]
    remaining = set(vectors).difference(selected_ids)
    while remaining:
        next_id = max(
            remaining,
            key=lambda case_id: (
                min(
                    _squared_distance(vectors[case_id], vectors[selected_id])
                    for selected_id in selected_ids
                ),
                _seed_tiebreak("diversity_maxmin", case_id, seed),
                case_id,
            ),
        )
        selected_ids.append(next_id)
        remaining.remove(next_id)
    return [by_id[case_id] for case_id in selected_ids]


def _stratified_cluster_quota_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    seed: int,
) -> list[Mapping[str, Any]]:
    raw_fields = selector_config.get("stratify_fields", ["cluster_id"])
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise ValueError("stratified selector requires stratify_fields sequence")
    fields = tuple(str(field).strip() for field in raw_fields if str(field).strip())
    if not fields:
        raise ValueError("stratified selector requires at least one field")
    groups: Dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in candidates:
        key = tuple(str(row.get(field, "__missing__")).strip() for field in fields)
        groups.setdefault(key, []).append(row)
    for key in list(groups):
        groups[key] = sorted(
            groups[key],
            key=lambda row: (
                -float(row.get("baseline_score", 0.0) or 0.0),
                _seed_tiebreak(
                    "stratified_cluster_quota",
                    _candidate_id(row),
                    seed,
                ),
                _candidate_id(row),
            ),
        )
    ordered: list[Mapping[str, Any]] = []
    while groups:
        for key in sorted(list(groups)):
            bucket = groups.get(key)
            if not bucket:
                groups.pop(key, None)
                continue
            ordered.append(bucket.pop(0))
            if not bucket:
                groups.pop(key, None)
    return ordered


def _uncertainty_boundary_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    seed: int,
) -> list[Mapping[str, Any]]:
    field = str(selector_config.get("uncertainty_field", "inner_boundary_uncertainty"))
    scored = []
    for row in candidates:
        score = _numeric_rank(row.get(field))
        if score is None:
            raise ValueError("uncertainty selector requires finite inner evidence scores")
        case_id = _candidate_id(row)
        scored.append(
            (
                score,
                _seed_tiebreak("uncertainty_boundary", case_id, seed),
                case_id,
                row,
            )
        )
    return [
        row
        for _, _, _, row in sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    ]


def _reject_fixed_replay_config(selector_config: Mapping[str, Any]) -> None:
    bad_keys = []
    for key in selector_config:
        lowered = str(key).lower()
        if any(token in lowered for token in _FIXED_REPLAY_CONFIG_KEYS):
            bad_keys.append(str(key))
    if bad_keys:
        raise ValueError(
            "coverage_capped_uncertainty forbids fixed replay config keys: %s"
            % sorted(bad_keys)
        )


def _normalized_numeric_scores(
    candidates: Sequence[Mapping[str, Any]],
    field: str,
    *,
    selector_id: str,
) -> Dict[str, float]:
    raw_scores: Dict[str, float] = {}
    for row in candidates:
        score = _numeric_rank(row.get(field))
        if score is None:
            raise ValueError("%s requires finite %s scores" % (selector_id, field))
        raw_scores[_candidate_id(row)] = score
    low = min(raw_scores.values())
    high = max(raw_scores.values())
    if high == low:
        return {case_id: 0.0 for case_id in raw_scores}
    return {
        case_id: (score - low) / (high - low)
        for case_id, score in raw_scores.items()
    }


def _coverage_cap_fields(selector_config: Mapping[str, Any]) -> tuple[str, ...]:
    raw_fields = selector_config.get("cap_fields", ["cluster_id", "time_bucket"])
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise ValueError("coverage_capped_uncertainty requires cap_fields sequence")
    fields = tuple(str(field).strip() for field in raw_fields if str(field).strip())
    if not fields:
        raise ValueError("coverage_capped_uncertainty requires at least one cap field")
    return fields


def _coverage_caps(
    *,
    selector_config: Mapping[str, Any],
    fields: Sequence[str],
    budget: int,
) -> Dict[str, int]:
    explicit = dict(selector_config.get("max_per_field") or {})
    fractions = dict(selector_config.get("max_fraction_per_field") or {})
    caps: Dict[str, int] = {}
    for field in fields:
        if field in explicit:
            cap = int(explicit[field])
        else:
            default_fraction = 0.15 if field == "cluster_id" else 0.25
            fraction = float(fractions.get(field, default_fraction))
            if not math.isfinite(fraction) or fraction <= 0.0:
                raise ValueError("coverage cap fractions must be positive")
            cap = int(math.ceil(int(budget) * fraction))
        if cap <= 0:
            raise ValueError("coverage caps must be positive")
        caps[field] = cap
    return caps


def _coverage_value(row: Mapping[str, Any], field: str) -> str:
    value = str(row.get(field, "__missing__")).strip()
    return value or "__missing__"


def _coverage_caps_allow(
    row: Mapping[str, Any],
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    caps: Mapping[str, int],
) -> bool:
    for field, cap in caps.items():
        value = _coverage_value(row, field)
        count = sum(
            1
            for selected in selected_rows
            if _coverage_value(selected, field) == value
        )
        if count >= int(cap):
            return False
    return True


def _coverage_capped_uncertainty_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_fixed_replay_config(selector_config)
    score_field = str(selector_config.get("score_field", "inner_boundary_uncertainty"))
    embedding_field = str(selector_config.get("embedding_field", "embedding"))
    diversity_weight = float(selector_config.get("diversity_weight", 0.10))
    if not math.isfinite(diversity_weight) or diversity_weight < 0.0:
        raise ValueError("coverage_capped_uncertainty diversity_weight must be nonnegative")
    scores = _normalized_numeric_scores(
        candidates,
        score_field,
        selector_id="coverage_capped_uncertainty",
    )
    vectors = {
        _candidate_id(row): _embedding_vector(row, embedding_field)
        for row in candidates
    }
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("coverage_capped_uncertainty requires equal embedding dimensions")
    cap_fields = _coverage_cap_fields(selector_config)
    caps = _coverage_caps(
        selector_config=selector_config,
        fields=cap_fields,
        budget=budget,
    )
    by_id = {_candidate_id(row): row for row in candidates}
    remaining = set(by_id)
    selected_rows: list[Mapping[str, Any]] = []
    ordered: list[Mapping[str, Any]] = []
    while remaining:
        eligible = [
            by_id[case_id]
            for case_id in remaining
            if _coverage_caps_allow(by_id[case_id], selected_rows=selected_rows, caps=caps)
        ]
        stage = "coverage_capped_uncertainty"
        if not eligible:
            eligible = [by_id[case_id] for case_id in remaining]
            stage = "coverage_relaxed_fill"

        def priority(row: Mapping[str, Any]) -> tuple[float, str, str]:
            case_id = _candidate_id(row)
            if selected_rows:
                diversity = min(
                    math.sqrt(
                        _squared_distance(vectors[case_id], vectors[_candidate_id(selected)])
                    )
                    for selected in selected_rows
                )
                diversity_bonus = diversity / (1.0 + diversity)
            else:
                diversity_bonus = 0.0
            return (
                scores[case_id] + diversity_weight * diversity_bonus,
                _seed_tiebreak("coverage_capped_uncertainty", case_id, seed),
                case_id,
            )

        chosen = max(eligible, key=priority)
        chosen_id = _candidate_id(chosen)
        selected = {**deepcopy(dict(chosen)), "selection_stage": stage}
        ordered.append(selected)
        selected_rows.append(selected)
        remaining.remove(chosen_id)
    return ordered


def _reject_stable_selector_config(selector_config: Mapping[str, Any]) -> None:
    _reject_fixed_replay_config(selector_config)
    _reject_case_id_semantic_config(selector_config)


def _proxy_config_from_selector(selector_config: Mapping[str, Any]) -> Dict[str, Any]:
    config = deepcopy(dict(selector_config.get("proxy_config") or {}))
    for key in ("proxy_mode_count", "algorithm", "iterations"):
        if key in selector_config and key not in config:
            config[key] = selector_config[key]
    return config


def _signature_config_from_selector(selector_config: Mapping[str, Any]) -> Dict[str, Any]:
    config = deepcopy(dict(selector_config.get("signature_config") or {}))
    for key in (
        "embedding_field",
        "numeric_fields",
        "categorical_fields",
        "metric_ad_fields",
        "metric_ad_required",
    ):
        if key in selector_config and key not in config:
            config[key] = selector_config[key]
    return config


def _assign_selector_proxy_modes(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[list[Mapping[str, Any]], Dict[str, Any]]:
    signatures = build_metric_signatures(
        candidates,
        signature_config=_signature_config_from_selector(selector_config),
    )
    proxy = assign_proxy_fault_modes(
        signatures["candidate_signatures"],
        proxy_config=_proxy_config_from_selector(selector_config),
        seed=seed,
    )
    metadata = {
        "metric_signature": {
            "signature_config_sha256": signatures["signature_config_sha256"],
            "signature_bundle_sha256": signatures["signature_bundle_sha256"],
            "allowed_input_fields": list(signatures["allowed_input_fields"]),
            "metric_ad_status": deepcopy(dict(signatures["metric_ad_status"])),
        },
        "proxy_assignment": {
            "proxy_config_sha256": proxy["proxy_config_sha256"],
            "proxy_assignment_sha256": proxy["proxy_assignment_sha256"],
            "diagnostics": deepcopy(dict(proxy["diagnostics"])),
        },
    }
    return list(proxy["assigned_candidates"]), metadata


def _proxy_balanced_select(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
    selector_id: str,
    stage: str,
    initial_mode_counts: Mapping[str, int] | None = None,
    mode_priority_penalty: Mapping[str, int] | None = None,
) -> list[Mapping[str, Any]]:
    if len(candidates) < budget:
        raise ValueError("proxy-balanced selector requires enough candidates")
    score_field = str(selector_config.get("score_field", "inner_boundary_uncertainty"))
    scores = _normalized_numeric_scores(
        candidates,
        score_field,
        selector_id=selector_id,
    )
    proxy_modes = {
        str(row.get("proxy_fault_mode", "")).strip()
        for row in candidates
        if str(row.get("proxy_fault_mode", "")).strip()
    }
    if not proxy_modes:
        raise ValueError("proxy-balanced selector requires proxy_fault_mode")
    default_cap = max(1, int(math.ceil(int(budget) * 0.20)))
    max_per_proxy = int(selector_config.get("max_per_proxy_mode", default_cap))
    if max_per_proxy <= 0:
        raise ValueError("max_per_proxy_mode must be positive")
    cap_feasible = max_per_proxy * len(proxy_modes) >= budget
    by_id = {_candidate_id(row): row for row in candidates}
    remaining = set(by_id)
    selected: list[Mapping[str, Any]] = []
    mode_counts: Dict[str, int] = {
        str(key): int(value) for key, value in dict(initial_mode_counts or {}).items()
    }
    mode_penalty: Dict[str, int] = {
        str(key): int(value) for key, value in dict(mode_priority_penalty or {}).items()
    }
    seen_times: set[str] = set()
    seen_clusters: set[str] = set()
    while remaining and len(selected) < budget:
        eligible = []
        for case_id in remaining:
            row = by_id[case_id]
            mode = str(row.get("proxy_fault_mode", "")).strip()
            if cap_feasible and mode_counts.get(mode, 0) >= max_per_proxy:
                continue
            eligible.append(row)
        relaxed = False
        if not eligible:
            eligible = [by_id[case_id] for case_id in remaining]
            relaxed = True

        def priority(
            row: Mapping[str, Any],
        ) -> tuple[int, int, int, int, int, float, str, str]:
            case_id = _candidate_id(row)
            mode = str(row.get("proxy_fault_mode", "")).strip()
            time_bucket = str(row.get("time_bucket", "")).strip()
            cluster = str(row.get("cluster_id", "")).strip()
            return (
                1 if mode_counts.get(mode, 0) == 0 else 0,
                -mode_counts.get(mode, 0),
                -mode_penalty.get(mode, 0),
                1 if time_bucket and time_bucket not in seen_times else 0,
                1 if cluster and cluster not in seen_clusters else 0,
                scores[case_id],
                _seed_tiebreak(selector_id, case_id, seed),
                case_id,
            )

        chosen = max(eligible, key=priority)
        chosen_id = _candidate_id(chosen)
        chosen_mode = str(chosen.get("proxy_fault_mode", "")).strip()
        selected_row = {
            **deepcopy(dict(chosen)),
            "selection_stage": stage if not relaxed else "%s_relaxed_fill" % stage,
            "proxy_selection_score": scores[chosen_id],
        }
        selected.append(selected_row)
        remaining.remove(chosen_id)
        mode_counts[chosen_mode] = mode_counts.get(chosen_mode, 0) + 1
        if str(chosen.get("time_bucket", "")).strip():
            seen_times.add(str(chosen.get("time_bucket", "")).strip())
        if str(chosen.get("cluster_id", "")).strip():
            seen_clusters.add(str(chosen.get("cluster_id", "")).strip())
    return selected


def _metric_signature_balanced_uncertainty_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    assigned, metadata = _assign_selector_proxy_modes(candidates, selector_config, seed=seed)
    ordered = _proxy_balanced_select(
        assigned,
        selector_config,
        budget=budget,
        seed=seed,
        selector_id="metric_signature_balanced_uncertainty",
        stage="proxy_balanced",
    )
    for row in ordered:
        row["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        row["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
    return ordered


def _flatten_query_rounds(rounds: Any) -> set[str]:
    if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
        return set()
    flattened: set[str] = set()
    for round_ids in rounds:
        if not isinstance(round_ids, Sequence) or isinstance(round_ids, (str, bytes)):
            continue
        for case_id in round_ids:
            text = str(case_id).strip()
            if text:
                flattened.add(text)
    return flattened


def _reject_unbudgeted_sequential_feedback(selector_config: Mapping[str, Any]) -> None:
    feedback = selector_config.get("label_feedback")
    if feedback in (None, {}):
        return
    if not isinstance(feedback, Mapping):
        raise ValueError("label_feedback must be a mapping")
    prior = _flatten_query_rounds(selector_config.get("prior_query_rounds", ()))
    bad = sorted(str(case_id).strip() for case_id in feedback if str(case_id).strip() not in prior)
    if bad:
        raise ValueError("unbudgeted sequential label feedback: %s" % bad[:10])


def _mode_counts_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        mode = str(row.get("proxy_fault_mode", "")).strip()
        if mode:
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def _mode_label_feedback_penalty(
    assigned: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
) -> Dict[str, int]:
    feedback = selector_config.get("label_feedback") or {}
    if not isinstance(feedback, Mapping):
        return {}
    by_id = {_candidate_id(row): row for row in assigned}
    penalties: Dict[str, int] = {}
    for case_id, labels in feedback.items():
        row = by_id.get(str(case_id).strip())
        if row is None:
            continue
        mode = str(row.get("proxy_fault_mode", "")).strip()
        if not mode:
            continue
        if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
            label_count = len({str(label) for label in labels if str(label).strip()})
        else:
            label_count = 1
        penalties[mode] = penalties.get(mode, 0) + max(1, label_count)
    return penalties


def _sequential_proxy_mode_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    _reject_unbudgeted_sequential_feedback(selector_config)
    seed_budget = int(selector_config.get("seed_budget", min(8, max(1, budget // 3))))
    if seed_budget <= 0 or seed_budget >= budget:
        raise ValueError("sequential proxy selector requires 0 < seed_budget < budget")
    assigned, metadata = _assign_selector_proxy_modes(candidates, selector_config, seed=seed)
    seed_config = deepcopy(dict(selector_config))
    seed_config["max_per_proxy_mode"] = int(seed_config.get("seed_max_per_proxy_mode", 1))
    seed_rows = _proxy_balanced_select(
        assigned,
        seed_config,
        budget=seed_budget,
        seed=seed,
        selector_id="sequential_proxy_mode_query",
        stage="proxy_diverse_seed",
    )
    seed_ids = {_candidate_id(row) for row in seed_rows}
    remaining = [row for row in assigned if _candidate_id(row) not in seed_ids]
    fill_rows = _proxy_balanced_select(
        remaining,
        selector_config,
        budget=budget - seed_budget,
        seed=seed,
        selector_id="sequential_proxy_mode_query",
        stage="proxy_balanced_fill",
        initial_mode_counts=_mode_counts_for_rows(seed_rows),
        mode_priority_penalty=_mode_label_feedback_penalty(assigned, selector_config),
    )
    ordered: list[Mapping[str, Any]] = []
    for row in seed_rows:
        copied = deepcopy(dict(row))
        copied["query_round"] = 1
        copied["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        copied["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        ordered.append(copied)
    for row in fill_rows:
        copied = deepcopy(dict(row))
        copied["query_round"] = 2
        copied["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        copied["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        ordered.append(copied)
    return ordered


_RICH_METRIC_AD_FIELDS = (
    "metric_ad_anomaly_direction",
    "metric_ad_duration",
    "metric_ad_metric_family_count",
    "metric_ad_propagation_width",
    "metric_ad_sparsity",
    "metric_ad_robust_z_mean",
    "metric_ad_robust_z_peak",
    "metric_ad_start_slope",
    "metric_ad_peak_lag",
    "metric_ad_recovery_slope",
)


def _normalized_optional_scores(
    candidates: Sequence[Mapping[str, Any]],
    field: str,
    *,
    default: float = 0.0,
    invert: bool = False,
) -> Dict[str, float]:
    raw_scores: Dict[str, float] = {}
    for row in candidates:
        score = _numeric_rank(row.get(field))
        raw_scores[_candidate_id(row)] = float(default if score is None else score)
    low = min(raw_scores.values())
    high = max(raw_scores.values())
    if high == low:
        normalized = {case_id: 0.0 for case_id in raw_scores}
    else:
        normalized = {
            case_id: (score - low) / (high - low)
            for case_id, score in raw_scores.items()
        }
    if invert:
        normalized = {case_id: 1.0 - value for case_id, value in normalized.items()}
    return normalized


def _stable_unit_interval(value: Any) -> float:
    digest = _semantic_sha256(str(value))[:12]
    return int(digest, 16) / float(16**12 - 1)


def _mpca_vector(row: Mapping[str, Any], *, factor: str) -> tuple[float, ...]:
    embedding = _embedding_vector(row, "embedding")
    uncertainty = _numeric_rank(row.get("inner_boundary_uncertainty")) or 0.0
    baseline = _numeric_rank(row.get("baseline_score")) or 0.0
    if factor == "mechanism":
        metric_fields = (
            "metric_ad_anomaly_direction",
            "metric_ad_duration",
            "metric_ad_sparsity",
            "metric_ad_robust_z_peak",
            "metric_ad_start_slope",
            "metric_ad_peak_lag",
            "metric_ad_recovery_slope",
        )
        metric_values = tuple(
            _numeric_rank(row.get(field)) or 0.0 for field in metric_fields
        )
        return tuple(float(value) for value in embedding) + metric_values + (
            uncertainty,
            baseline,
        )
    if factor == "propagation":
        cluster = _stable_unit_interval(row.get("cluster_id", ""))
        time_bucket = _stable_unit_interval(row.get("time_bucket", ""))
        width = _numeric_rank(row.get("metric_ad_propagation_width")) or 0.0
        scheme_rank = _numeric_rank(row.get("scheme1_rank")) or 0.0
        if len(embedding) == 1:
            spread = 0.0
        else:
            spread = max(embedding) - min(embedding)
        return (
            cluster,
            time_bucket,
            width,
            scheme_rank,
            spread,
            sum(embedding) / len(embedding),
            uncertainty,
        )
    raise ValueError("unknown MPCA factor: %s" % factor)


def _min_distance_to_selected(
    case_id: str,
    selected_ids: Sequence[str],
    vectors: Mapping[str, Sequence[float]],
) -> float:
    if not selected_ids:
        centroid = tuple(
            sum(vector[index] for vector in vectors.values()) / len(vectors)
            for index in range(len(next(iter(vectors.values()))))
        )
        return math.sqrt(_squared_distance(vectors[case_id], centroid))
    return min(
        math.sqrt(_squared_distance(vectors[case_id], vectors[selected_id]))
        for selected_id in selected_ids
    )


def _normalize_mapping(values: Mapping[str, float]) -> Dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {case_id: 1.0 for case_id in values}
    return {
        case_id: (float(value) - low) / (high - low)
        for case_id, value in values.items()
    }


def _mpca_dual_factor_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    weights = {
        "mechanism_novelty": float(selector_config.get("mechanism_weight", 0.30)),
        "propagation_novelty": float(selector_config.get("propagation_weight", 0.30)),
        "uncertainty": float(selector_config.get("uncertainty_weight", 0.25)),
        "joint_novelty": float(selector_config.get("joint_weight", 0.15)),
    }
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("mpca_dual_factor_query weights must be finite and nonnegative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("mpca_dual_factor_query weights must have positive sum")
    uncertainty_field = str(
        selector_config.get("uncertainty_field", "inner_boundary_uncertainty")
    )
    uncertainty = _normalized_optional_scores(candidates, uncertainty_field)
    mechanism_vectors = {
        _candidate_id(row): _mpca_vector(row, factor="mechanism") for row in candidates
    }
    propagation_vectors = {
        _candidate_id(row): _mpca_vector(row, factor="propagation") for row in candidates
    }
    by_id = {_candidate_id(row): row for row in candidates}
    all_ids = [_candidate_id(row) for row in candidates]
    selected_ids: list[str] = []
    ordered: list[Mapping[str, Any]] = []
    for round_index in range(int(budget)):
        remaining = [case_id for case_id in all_ids if case_id not in selected_ids]
        mechanism_raw = {
            case_id: _min_distance_to_selected(case_id, selected_ids, mechanism_vectors)
            for case_id in remaining
        }
        propagation_raw = {
            case_id: _min_distance_to_selected(case_id, selected_ids, propagation_vectors)
            for case_id in remaining
        }
        mechanism_scores = _normalize_mapping(mechanism_raw)
        propagation_scores = _normalize_mapping(propagation_raw)
        scored = []
        for case_id in remaining:
            components = {
                "mechanism_novelty": mechanism_scores[case_id],
                "propagation_novelty": propagation_scores[case_id],
                "uncertainty": uncertainty.get(case_id, 0.0),
                "joint_novelty": min(
                    mechanism_scores[case_id],
                    propagation_scores[case_id],
                ),
            }
            acquisition = sum(
                components[name] * weights[name] for name in weights
            ) / weight_sum
            scored.append(
                (
                    -acquisition,
                    _seed_tiebreak("mpca_dual_factor_query", case_id, seed),
                    case_id,
                    components,
                    acquisition,
                )
            )
        scored.sort()
        _, _, chosen_id, components, acquisition = scored[0]
        selected_ids.append(chosen_id)
        row = deepcopy(dict(by_id[chosen_id]))
        row["selection_stage"] = "mpca_dual_factor_greedy"
        row["selection_round"] = round_index + 1
        row["mpca_component_scores"] = components
        row["mpca_acquisition_score"] = acquisition
        ordered.append(row)
    leftovers = [
        deepcopy(dict(by_id[case_id]))
        for case_id in all_ids
        if case_id not in selected_ids
    ]
    return ordered + leftovers


def _ranker_aware_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    score_field = str(selector_config.get("score_field", "inner_boundary_uncertainty"))
    embedding_field = str(selector_config.get("embedding_field", "embedding"))
    uncertainty = _normalized_numeric_scores(
        candidates,
        score_field,
        selector_id="ranker_aware_acquisition",
    )
    baseline = _normalized_optional_scores(candidates, "baseline_score")
    rank_prior = _normalized_optional_scores(candidates, "scheme1_rank", invert=True)
    vectors = {
        _candidate_id(row): _embedding_vector(row, embedding_field)
        for row in candidates
    }
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("ranker-aware acquisition requires equal embedding dimensions")
    weights = {
        "uncertainty": float(selector_config.get("uncertainty_weight", 0.45)),
        "baseline": float(selector_config.get("baseline_weight", 0.20)),
        "rank_prior": float(selector_config.get("rank_prior_weight", 0.20)),
        "diversity": float(selector_config.get("diversity_weight", 0.15)),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("ranker-aware acquisition weights must be nonnegative")
    by_id = {_candidate_id(row): row for row in candidates}
    remaining = set(by_id)
    selected_ids: list[str] = []
    ordered: list[Mapping[str, Any]] = []
    while remaining:
        def priority(case_id: str) -> tuple[float, str, str]:
            if selected_ids:
                nearest = min(
                    math.sqrt(_squared_distance(vectors[case_id], vectors[selected_id]))
                    for selected_id in selected_ids
                )
                diversity = nearest / (1.0 + nearest)
            else:
                diversity = 0.0
            utility = (
                weights["uncertainty"] * uncertainty[case_id]
                + weights["baseline"] * baseline[case_id]
                + weights["rank_prior"] * rank_prior[case_id]
                + weights["diversity"] * diversity
            )
            return (
                utility,
                _seed_tiebreak("ranker_aware_acquisition", case_id, seed),
                case_id,
            )

        chosen_id = max(remaining, key=priority)
        selected_ids.append(chosen_id)
        remaining.remove(chosen_id)
        row = deepcopy(dict(by_id[chosen_id]))
        row["selection_stage"] = "ranker_utility_greedy"
        row["ranker_utility_score"] = priority(chosen_id)[0]
        row["_selector_diagnostics"] = {
            "ranker_utility": {
                "label_clean": True,
                "utility_field": score_field,
                "surrogate": "weighted_inner_evidence_rank_prior_diversity",
                "utility_components": sorted(weights),
                "utility_evidence_count": min(len(selected_ids), int(budget)),
            }
        }
        ordered.append(row)
    return ordered


def _richer_metric_signature_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    config = deepcopy(dict(selector_config))
    config.setdefault(
        "metric_ad_fields",
        [field for field in _RICH_METRIC_AD_FIELDS if any(field in row for row in candidates)],
    )
    config.setdefault("numeric_fields", ["inner_boundary_uncertainty", "baseline_score", "scheme1_rank"])
    config.setdefault("categorical_fields", ["cluster_id", "time_bucket"])
    assigned, metadata = _assign_selector_proxy_modes(candidates, config, seed=seed)
    ordered = _proxy_balanced_select(
        assigned,
        config,
        budget=budget,
        seed=seed,
        selector_id="richer_metric_signature_acquisition",
        stage="rich_metric_proxy_balance",
    )
    diagnostic = {
        "metric_signature": {
            "label_clean": True,
            "metric_ad_status": deepcopy(metadata["metric_signature"]["metric_ad_status"]),
            "allowed_input_fields": list(metadata["metric_signature"]["allowed_input_fields"]),
            "signature_config_sha256": metadata["metric_signature"]["signature_config_sha256"],
        }
    }
    for row in ordered:
        row["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        row["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        row["_selector_diagnostics"] = diagnostic
    return ordered


def _feedback_mode_label(labels: Any) -> str:
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        values = sorted(str(label).strip() for label in labels if str(label).strip())
    else:
        values = [str(labels).strip()] if str(labels).strip() else []
    return "label-mode:" + _semantic_sha256(values)[:12] if values else "label-mode:missing"


def _nearest_mode_prediction(
    vector: Sequence[float],
    centroids: Mapping[str, Sequence[float]],
) -> tuple[str, float]:
    if not centroids:
        return "__proxy_fallback__", 1.0
    distances = sorted(
        (
            math.sqrt(_squared_distance(vector, centroid)),
            mode,
        )
        for mode, centroid in centroids.items()
    )
    nearest_distance, mode = distances[0]
    if len(distances) == 1:
        uncertainty = 1.0 / (1.0 + nearest_distance)
    else:
        margin = distances[1][0] - nearest_distance
        uncertainty = 1.0 / (1.0 + max(0.0, margin))
    return mode, uncertainty


def _budgeted_fault_mode_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    _reject_unbudgeted_sequential_feedback(selector_config)
    seed_budget = int(selector_config.get("seed_budget", min(8, max(1, budget // 3))))
    if seed_budget <= 0 or seed_budget >= budget:
        raise ValueError("budgeted fault-mode selector requires 0 < seed_budget < budget")
    assigned, metadata = _assign_selector_proxy_modes(candidates, selector_config, seed=seed)
    by_id = {_candidate_id(row): row for row in assigned}
    prior_ids = [
        case_id
        for case_id in _flatten_query_rounds(selector_config.get("prior_query_rounds", ()))
        if case_id in by_id
    ]
    seed_rows: list[Mapping[str, Any]]
    if prior_ids:
        seed_rows = [by_id[case_id] for case_id in prior_ids[:seed_budget]]
        if len(seed_rows) < seed_budget:
            remaining_seed = [
                row for row in assigned if _candidate_id(row) not in set(prior_ids)
            ]
            seed_rows.extend(
                _proxy_balanced_select(
                    remaining_seed,
                    {**deepcopy(dict(selector_config)), "max_per_proxy_mode": 1},
                    budget=seed_budget - len(seed_rows),
                    seed=seed,
                    selector_id="budgeted_fault_mode_modeling",
                    stage="fault_mode_seed",
                )
            )
    else:
        seed_rows = _proxy_balanced_select(
            assigned,
            {**deepcopy(dict(selector_config)), "max_per_proxy_mode": 1},
            budget=seed_budget,
            seed=seed,
            selector_id="budgeted_fault_mode_modeling",
            stage="fault_mode_seed",
        )
    seed_ids = {_candidate_id(row) for row in seed_rows}
    feedback = dict(selector_config.get("label_feedback") or {})
    signatures = {
        _candidate_id(row): tuple(float(value) for value in row["metric_signature"])
        for row in assigned
    }
    labeled_vectors: Dict[str, list[tuple[float, ...]]] = {}
    for case_id in seed_ids:
        if case_id in feedback:
            labeled_vectors.setdefault(_feedback_mode_label(feedback[case_id]), []).append(
                signatures[case_id]
            )
    centroids = {
        mode: _mean_vector(vectors)
        for mode, vectors in labeled_vectors.items()
        if vectors
    }
    remaining = [row for row in assigned if _candidate_id(row) not in seed_ids]
    score_field = str(selector_config.get("score_field", "inner_boundary_uncertainty"))
    utility = _normalized_numeric_scores(
        remaining,
        score_field,
        selector_id="budgeted_fault_mode_modeling",
    )
    predictions: Dict[str, tuple[str, float]] = {}
    for row in remaining:
        case_id = _candidate_id(row)
        if centroids:
            predictions[case_id] = _nearest_mode_prediction(signatures[case_id], centroids)
        else:
            mode = str(row.get("proxy_fault_mode", "__proxy_fallback__")).strip()
            uncertainty = float(row.get("proxy_distance", 0.0))
            predictions[case_id] = (mode or "__proxy_fallback__", uncertainty / (1.0 + uncertainty))
    by_remaining = {_candidate_id(row): row for row in remaining}
    mode_counts = _mode_counts_for_rows(seed_rows)
    selected_fill: list[Mapping[str, Any]] = []
    while by_remaining and len(selected_fill) < budget - seed_budget:
        def priority(case_id: str) -> tuple[int, int, float, float, str, str]:
            predicted_mode, classifier_uncertainty = predictions[case_id]
            return (
                1 if mode_counts.get(predicted_mode, 0) == 0 else 0,
                -mode_counts.get(predicted_mode, 0),
                classifier_uncertainty,
                utility[case_id],
                _seed_tiebreak("budgeted_fault_mode_modeling", case_id, seed),
                case_id,
            )

        chosen_id = max(by_remaining, key=priority)
        chosen = deepcopy(dict(by_remaining.pop(chosen_id)))
        predicted_mode, classifier_uncertainty = predictions[chosen_id]
        chosen["selection_stage"] = "fault_mode_model_fill"
        chosen["query_round"] = 2
        chosen["predicted_fault_mode"] = predicted_mode
        chosen["fault_mode_classifier_uncertainty"] = classifier_uncertainty
        mode_counts[predicted_mode] = mode_counts.get(predicted_mode, 0) + 1
        selected_fill.append(chosen)
    ordered: list[Mapping[str, Any]] = []
    diagnostic = {
        "fault_mode_model": {
            "label_clean": True,
            "feedback_case_count": len(feedback),
            "budgeted_seed_count": seed_budget,
            "predictor": "nearest_centroid_from_budgeted_label_feedback"
            if centroids
            else "proxy_mode_fallback_without_unqueried_labels",
            "predicted_mode_count": len({predictions[_candidate_id(row)][0] for row in remaining})
            if remaining
            else 0,
        }
    }
    for row in seed_rows:
        copied = deepcopy(dict(row))
        copied["selection_stage"] = "fault_mode_seed"
        copied["query_round"] = 1
        copied["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        copied["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        copied["_selector_diagnostics"] = diagnostic
        ordered.append(copied)
    for row in selected_fill:
        copied = deepcopy(dict(row))
        copied["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        copied["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        copied["_selector_diagnostics"] = diagnostic
        ordered.append(copied)
    leftovers = [row for row in assigned if _candidate_id(row) not in {_candidate_id(item) for item in ordered}]
    ordered.extend(leftovers)
    return ordered


def _facility_location_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    signatures = build_metric_signatures(
        candidates,
        signature_config=_signature_config_from_selector(selector_config),
    )
    rows = list(signatures["candidate_signatures"])
    vectors = _normalized_vectors(rows)
    score_field = str(selector_config.get("score_field", "inner_boundary_uncertainty"))
    utility = _normalized_numeric_scores(
        candidates,
        score_field,
        selector_id="facility_location_selection",
    )
    by_id = {_candidate_id(row): row for row in rows}
    remaining = set(by_id)
    selected_ids: list[str] = []
    best_similarity = {case_id: 0.0 for case_id in remaining}
    ordered: list[Mapping[str, Any]] = []
    utility_weight = float(selector_config.get("utility_weight", 0.15))
    if not math.isfinite(utility_weight) or utility_weight < 0.0:
        raise ValueError("facility-location utility_weight must be nonnegative")
    while remaining:
        def gain(case_id: str) -> tuple[float, str, str]:
            coverage_gain = 0.0
            for other_id in vectors:
                distance = math.sqrt(_squared_distance(vectors[case_id], vectors[other_id]))
                similarity = 1.0 / (1.0 + distance)
                coverage_gain += max(0.0, similarity - best_similarity[other_id])
            value = coverage_gain / max(1, len(vectors)) + utility_weight * utility[case_id]
            return (
                value,
                _seed_tiebreak("facility_location_selection", case_id, seed),
                case_id,
            )

        chosen_id = max(remaining, key=gain)
        selected_ids.append(chosen_id)
        remaining.remove(chosen_id)
        for other_id in vectors:
            distance = math.sqrt(_squared_distance(vectors[chosen_id], vectors[other_id]))
            best_similarity[other_id] = max(best_similarity[other_id], 1.0 / (1.0 + distance))
        row = deepcopy(dict(by_id[chosen_id]))
        row["selection_stage"] = "facility_location_greedy"
        row["facility_location_gain"] = gain(chosen_id)[0]
        row["_selector_diagnostics"] = {
            "facility_location": {
                "label_clean": True,
                "objective": "submodular_facility_location_plus_uncertainty",
                "candidate_count": len(candidates),
                "utility_field": score_field,
            }
        }
        ordered.append(row)
    return ordered


def _unknown_mode_signature_reserve_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    config = deepcopy(dict(selector_config))
    config.setdefault(
        "metric_ad_fields",
        [field for field in _RICH_METRIC_AD_FIELDS if any(field in row for row in candidates)],
    )
    config.setdefault("numeric_fields", ["inner_boundary_uncertainty", "baseline_score", "scheme1_rank"])
    config.setdefault("categorical_fields", ["cluster_id", "time_bucket"])
    assigned, metadata = _assign_selector_proxy_modes(candidates, config, seed=seed)
    vectors = _normalized_vectors(assigned)
    candidate_ids = [_candidate_id(row) for row in assigned]
    anchor_count = int(config.get("coverage_anchor_count", min(256, len(candidate_ids))))
    if anchor_count <= 0:
        raise ValueError("coverage_anchor_count must be positive")
    anchor_count = min(anchor_count, len(candidate_ids))
    anchor_ids = sorted(
        candidate_ids,
        key=lambda case_id: (
            _seed_tiebreak("unknown_mode_signature_reserve.coverage_anchor", case_id, seed),
            case_id,
        ),
    )[:anchor_count]
    by_id = {_candidate_id(row): row for row in assigned}
    utility = _normalized_numeric_scores(
        assigned,
        str(config.get("score_field", "inner_boundary_uncertainty")),
        selector_id="unknown_mode_signature_reserve",
    )
    reserve_budget = int(config.get("unknown_mode_reserve_budget", max(1, math.ceil(budget * 0.25))))
    reserve_budget = max(1, min(int(budget), reserve_budget))
    unknown_weight = float(config.get("unknown_mode_weight", 0.40))
    coverage_weight = float(config.get("coverage_weight", 0.35))
    utility_weight = float(config.get("utility_weight", 0.20))
    mode_novelty_weight = float(config.get("mode_novelty_weight", 0.05))
    weights = (unknown_weight, coverage_weight, utility_weight, mode_novelty_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("unknown-mode reserve weights must be nonnegative finite numbers")
    default_cap = max(1, int(math.ceil(int(budget) * 0.20)))
    max_per_proxy = int(config.get("max_per_proxy_mode", default_cap))
    if max_per_proxy <= 0:
        raise ValueError("max_per_proxy_mode must be positive")
    centroid = _mean_vector(list(vectors.values()))
    remaining = set(candidate_ids)
    selected_ids: list[str] = []
    ordered: list[Mapping[str, Any]] = []
    anchor_similarity = {
        case_id: tuple(
            1.0 / (1.0 + math.sqrt(_squared_distance(vectors[case_id], vectors[anchor_id])))
            for anchor_id in anchor_ids
        )
        for case_id in candidate_ids
    }
    best_similarity = [0.0 for _ in anchor_ids]
    nearest_distance = {
        case_id: math.sqrt(_squared_distance(vectors[case_id], centroid))
        for case_id in candidate_ids
    }
    mode_counts: Dict[str, int] = {}
    seen_times: set[str] = set()
    seen_clusters: set[str] = set()

    def distance_to_selected(case_id: str) -> float:
        return nearest_distance[case_id]

    def coverage_gain(case_id: str) -> float:
        similarities = anchor_similarity[case_id]
        return sum(
            max(0.0, similarity - best_similarity[index])
            for index, similarity in enumerate(similarities)
        ) / max(1, len(anchor_ids))

    while remaining:
        cap_feasible = max_per_proxy * len(
            {str(row.get("proxy_fault_mode", "")).strip() for row in assigned if str(row.get("proxy_fault_mode", "")).strip()}
        ) >= min(int(budget), len(candidate_ids))
        eligible = []
        for case_id in remaining:
            row = by_id[case_id]
            mode = str(row.get("proxy_fault_mode", "")).strip()
            if cap_feasible and mode_counts.get(mode, 0) >= max_per_proxy:
                continue
            eligible.append(case_id)
        relaxed = False
        if not eligible:
            eligible = list(remaining)
            relaxed = True

        def priority(case_id: str) -> tuple[float, int, int, int, str, str]:
            row = by_id[case_id]
            mode = str(row.get("proxy_fault_mode", "")).strip()
            time_bucket = str(row.get("time_bucket", "")).strip()
            cluster = str(row.get("cluster_id", "")).strip()
            distance = distance_to_selected(case_id)
            unknown_score = distance / (1.0 + distance)
            mode_novelty = 1.0 if mode_counts.get(mode, 0) == 0 else 0.0
            value = (
                unknown_weight * unknown_score
                + coverage_weight * coverage_gain(case_id)
                + utility_weight * utility[case_id]
                + mode_novelty_weight * mode_novelty
            )
            return (
                value,
                1 if time_bucket and time_bucket not in seen_times else 0,
                1 if cluster and cluster not in seen_clusters else 0,
                -mode_counts.get(mode, 0),
                _seed_tiebreak("unknown_mode_signature_reserve", case_id, seed),
                case_id,
            )

        chosen_id = max(eligible, key=priority)
        chosen_priority = priority(chosen_id)[0]
        chosen = deepcopy(dict(by_id[chosen_id]))
        chosen_distance = distance_to_selected(chosen_id)
        chosen_gain = coverage_gain(chosen_id)
        selected_ids.append(chosen_id)
        remaining.remove(chosen_id)
        for index, similarity in enumerate(anchor_similarity[chosen_id]):
            best_similarity[index] = max(best_similarity[index], similarity)
        for other_id in remaining:
            distance = math.sqrt(_squared_distance(vectors[other_id], vectors[chosen_id]))
            if distance < nearest_distance[other_id]:
                nearest_distance[other_id] = distance
        mode = str(chosen.get("proxy_fault_mode", "")).strip()
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        time_bucket = str(chosen.get("time_bucket", "")).strip()
        cluster = str(chosen.get("cluster_id", "")).strip()
        if time_bucket:
            seen_times.add(time_bucket)
        if cluster:
            seen_clusters.add(cluster)
        stage = (
            "unknown_mode_reserve"
            if len(selected_ids) <= reserve_budget
            else "signature_coverage_fill"
        )
        if relaxed:
            stage += "_relaxed"
        chosen["selection_stage"] = stage
        chosen["unknown_mode_distance"] = chosen_distance
        chosen["facility_location_gain"] = chosen_gain
        chosen["unknown_mode_score"] = chosen_priority
        chosen["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        chosen["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        chosen["_selector_diagnostics"] = {
            "unknown_mode_signature_reserve": {
                "label_clean": True,
                "objective": "metric_signature_unknown_mode_reserve_plus_facility_coverage",
                "unknown_mode_reserve_budget": reserve_budget,
                "candidate_count": len(candidate_ids),
                "coverage_anchor_count": len(anchor_ids),
                "selected_proxy_mode_count": len(mode_counts),
                "max_per_proxy_mode": max_per_proxy,
                "weights": {
                    "unknown_mode": unknown_weight,
                    "coverage": coverage_weight,
                    "utility": utility_weight,
                    "mode_novelty": mode_novelty_weight,
                },
                "metric_ad_status": deepcopy(metadata["metric_signature"]["metric_ad_status"]),
                "allowed_input_fields": list(metadata["metric_signature"]["allowed_input_fields"]),
                "signature_config_sha256": metadata["metric_signature"]["signature_config_sha256"],
                "signature_bundle_sha256": metadata["metric_signature"]["signature_bundle_sha256"],
                "proxy_assignment_sha256": metadata["proxy_assignment"]["proxy_assignment_sha256"],
            }
        }
        ordered.append(chosen)
    selected_budget_rows = ordered[: int(budget)]
    final_selected_modes = {
        str(row.get("proxy_fault_mode", "")).strip()
        for row in selected_budget_rows
        if str(row.get("proxy_fault_mode", "")).strip()
    }
    final_diagnostic = {
        "unknown_mode_signature_reserve": {
            "label_clean": True,
            "objective": "metric_signature_unknown_mode_reserve_plus_facility_coverage",
            "unknown_mode_reserve_budget": reserve_budget,
            "candidate_count": len(candidate_ids),
            "coverage_anchor_count": len(anchor_ids),
            "selected_proxy_mode_count": len(final_selected_modes),
            "max_per_proxy_mode": max_per_proxy,
            "weights": {
                "unknown_mode": unknown_weight,
                "coverage": coverage_weight,
                "utility": utility_weight,
                "mode_novelty": mode_novelty_weight,
            },
            "metric_ad_status": deepcopy(metadata["metric_signature"]["metric_ad_status"]),
            "allowed_input_fields": list(metadata["metric_signature"]["allowed_input_fields"]),
            "signature_config_sha256": metadata["metric_signature"]["signature_config_sha256"],
            "signature_bundle_sha256": metadata["metric_signature"]["signature_bundle_sha256"],
            "proxy_assignment_sha256": metadata["proxy_assignment"]["proxy_assignment_sha256"],
        }
    }
    for row in ordered:
        row["_selector_diagnostics"] = deepcopy(final_diagnostic)
    return ordered


def _winner_like_distillation_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    _reject_stable_selector_config(selector_config)
    config = deepcopy(dict(selector_config))
    config.setdefault(
        "metric_ad_fields",
        [field for field in _RICH_METRIC_AD_FIELDS if any(field in row for row in candidates)],
    )
    assigned, metadata = _assign_selector_proxy_modes(candidates, config, seed=seed)
    candidate_ids = [_candidate_id(row) for row in assigned]
    uncertainty = _normalized_optional_scores(assigned, "inner_boundary_uncertainty")
    baseline = _normalized_optional_scores(assigned, "baseline_score")
    rank_prior = _normalized_optional_scores(assigned, "scheme1_rank", invert=True)
    z_peak = _normalized_optional_scores(assigned, "metric_ad_robust_z_peak")
    duration = _normalized_optional_scores(assigned, "metric_ad_duration")
    weights = dict(
        selector_config.get("winner_feature_weights")
        or {
            "uncertainty": 0.30,
            "baseline": 0.20,
            "rank_prior": 0.20,
            "metric_ad_robust_z_peak": 0.15,
            "metric_ad_duration": 0.15,
        }
    )
    score_by_id = {
        case_id: float(weights.get("uncertainty", 0.0)) * uncertainty[case_id]
        + float(weights.get("baseline", 0.0)) * baseline[case_id]
        + float(weights.get("rank_prior", 0.0)) * rank_prior[case_id]
        + float(weights.get("metric_ad_robust_z_peak", 0.0)) * z_peak[case_id]
        + float(weights.get("metric_ad_duration", 0.0)) * duration[case_id]
        for case_id in candidate_ids
    }
    by_id = {_candidate_id(row): row for row in assigned}
    mode_counts: Dict[str, int] = {}
    default_cap = max(1, int(math.ceil(int(budget) * 0.20)))
    max_per_proxy = int(selector_config.get("max_per_proxy_mode", default_cap))
    remaining = set(candidate_ids)
    ordered: list[Mapping[str, Any]] = []
    while remaining:
        eligible = [
            case_id
            for case_id in remaining
            if mode_counts.get(str(by_id[case_id].get("proxy_fault_mode", "")), 0)
            < max_per_proxy
        ]
        if not eligible:
            eligible = list(remaining)
        chosen_id = max(
            eligible,
            key=lambda case_id: (
                score_by_id[case_id],
                _seed_tiebreak("winner_like_distillation", case_id, seed),
                case_id,
            ),
        )
        remaining.remove(chosen_id)
        mode = str(by_id[chosen_id].get("proxy_fault_mode", "")).strip()
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        row = deepcopy(dict(by_id[chosen_id]))
        row["selection_stage"] = "winner_like_feature_score"
        row["winner_like_score"] = score_by_id[chosen_id]
        row["metric_signature_config_sha256"] = metadata["metric_signature"][
            "signature_config_sha256"
        ]
        row["proxy_assignment_sha256"] = metadata["proxy_assignment"][
            "proxy_assignment_sha256"
        ]
        row["_selector_diagnostics"] = {
            "winner_like_distillation": {
                "label_clean": True,
                "training_set_source": "label_free_high_low_random_set_feature_profile",
                "case_id_semantics_used": False,
                "feature_fields": sorted(weights),
            }
        }
        ordered.append(row)
    return ordered


def _two_stage_diverse_uncertainty_order(
    candidates: Sequence[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    *,
    budget: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    default_seed_budget = max(1, min(5, budget // 6 or 1))
    seed_budget = int(selector_config.get("seed_budget", default_seed_budget))
    if seed_budget <= 0 or seed_budget >= budget:
        raise ValueError("two-stage selector requires 0 < seed_budget < budget")
    diverse_seed = _diversity_maxmin_order(candidates, selector_config, seed)[
        :seed_budget
    ]
    diverse_ids = {_candidate_id(row) for row in diverse_seed}
    remaining = [row for row in candidates if _candidate_id(row) not in diverse_ids]
    uncertainty_fill = _uncertainty_boundary_order(remaining, selector_config, seed)[
        : budget - seed_budget
    ]
    ordered = []
    for row in diverse_seed:
        ordered.append({**deepcopy(dict(row)), "selection_stage": "diverse_seed"})
    for row in uncertainty_fill:
        ordered.append({**deepcopy(dict(row)), "selection_stage": "uncertainty_fill"})
    return ordered


def list_query_selectors() -> Dict[str, Dict[str, Any]]:
    """Return registered query-only active selectors."""

    return deepcopy(_QUERY_SELECTOR_REGISTRY)


def select_query_cases(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selector_id: str,
    budget: int = 30,
    seed: int = 42,
    selector_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Select a deterministic budgeted query set without label leakage."""

    selector = str(selector_id)
    if selector not in _QUERY_SELECTOR_REGISTRY:
        raise ValueError("unknown query selector: %s" % selector)
    query_budget = _require_budget(budget)
    config = deepcopy(dict(selector_config or {}))
    rows = _prepare_selector_candidates(
        candidates,
        selector_id=selector,
        budget=query_budget,
    )
    if selector == "scheme1_baseline":
        ordered = _scheme1_baseline_order(rows, config, int(seed))
    elif selector == "diversity_maxmin":
        ordered = _diversity_maxmin_order(rows, config, int(seed))
    elif selector == "stratified_cluster_quota":
        ordered = _stratified_cluster_quota_order(rows, config, int(seed))
    elif selector == "uncertainty_boundary":
        ordered = _uncertainty_boundary_order(rows, config, int(seed))
    elif selector == "coverage_capped_uncertainty":
        ordered = _coverage_capped_uncertainty_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "metric_signature_balanced_uncertainty":
        ordered = _metric_signature_balanced_uncertainty_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "sequential_proxy_mode_query":
        ordered = _sequential_proxy_mode_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "ranker_aware_acquisition":
        ordered = _ranker_aware_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "richer_metric_signature_acquisition":
        ordered = _richer_metric_signature_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "budgeted_fault_mode_modeling":
        ordered = _budgeted_fault_mode_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "facility_location_selection":
        ordered = _facility_location_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "unknown_mode_signature_reserve":
        ordered = _unknown_mode_signature_reserve_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "winner_like_distillation":
        ordered = _winner_like_distillation_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "mpca_dual_factor_query":
        ordered = _mpca_dual_factor_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    elif selector == "two_stage_diverse_uncertainty":
        ordered = _two_stage_diverse_uncertainty_order(
            rows,
            config,
            budget=query_budget,
            seed=int(seed),
        )
    else:  # pragma: no cover - registry and dispatch stay in lockstep.
        raise ValueError("unimplemented query selector: %s" % selector)
    return _selector_result(
        selector_id=selector,
        budget=query_budget,
        seed=int(seed),
        selector_config=config,
        selected_cases=ordered[:query_budget],
    )


def validate_oracle_gate(
    oracle_metrics: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Mapping[str, float]] = SOTA_THRESHOLDS,
) -> Dict[str, Any]:
    """Return a strict SOTA-gate summary for matched oracle-full metrics.

    Missing datasets/metrics are schema failures and raise. Values equal to a
    threshold are valid observations but fail the gate, because the protocol
    requires strictly exceeding the user-declared SOTA tuple.
    """

    dataset_results: Dict[str, Dict[str, Dict[str, float | bool]]] = {}
    blocking_metrics = []
    for dataset, metric_thresholds in thresholds.items():
        dataset_metrics = oracle_metrics.get(dataset)
        if not isinstance(dataset_metrics, Mapping):
            raise ValueError("missing oracle metrics for dataset %s" % dataset)
        dataset_results[dataset] = {}
        for metric in HIT_METRICS:
            if metric not in dataset_metrics:
                raise ValueError(
                    "missing oracle metric %s.%s" % (dataset, metric)
                )
            actual = float(dataset_metrics[metric])
            threshold = float(metric_thresholds[metric])
            strictly_exceeds = actual > threshold
            dataset_results[dataset][metric] = {
                "actual": actual,
                "threshold": threshold,
                "strictly_exceeds": strictly_exceeds,
                "margin": actual - threshold,
            }
            if not strictly_exceeds:
                blocking_metrics.append(
                    {
                        "canonical_dataset_id": dataset,
                        "metric": metric,
                        "actual": actual,
                        "threshold": threshold,
                        "reason": "not_strictly_greater_than_sota",
                    }
                )
    return {
        "schema_version": "rcl-query-active-learning-oracle-gate-v1",
        "change_id": CHANGE_ID,
        "target_datasets": deepcopy(TARGET_DATASETS),
        "sota_thresholds": deepcopy(dict(thresholds)),
        "dataset_results": dataset_results,
        "blocking_metrics": blocking_metrics,
        "passed": not blocking_metrics,
    }


def target_pair_config() -> Dict[str, Any]:
    """Return the machine-readable static protocol config for this change."""

    return {
        "schema_version": "rcl-query-active-learning-config-v1",
        "change_id": CHANGE_ID,
        "run_id": RUN_ID,
        "remote_workspace": REMOTE_WORKSPACE,
        "approved_output_root": APPROVED_OUTPUT_ROOT,
        "target_datasets": deepcopy(TARGET_DATASETS),
        "sota_thresholds": deepcopy(SOTA_THRESHOLDS),
        "budget": {
            "case_count": 30,
            "case_kind": "fault",
            "split": "outer_train",
            "annotation_source": "simulated_manual_ground_truth",
        },
        "oracle_gate": {
            "comparison": "strictly_greater_than_sota",
            "metrics": list(HIT_METRICS),
        },
    }


def build_target_pair_inventory(
    dataset_summaries: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Build a hash-closed target-pair admission inventory.

    The inventory is allowed to be pre-gate/historical, but it must make that
    evidence tier explicit so the later formal oracle gate can replace or
    validate it without ambiguity.
    """

    missing = sorted(set(TARGET_DATASETS).difference(dataset_summaries))
    if missing:
        raise ValueError("missing target inventory for %s" % missing)
    datasets: Dict[str, Dict[str, Any]] = {}
    for dataset in sorted(TARGET_DATASETS):
        raw = dict(dataset_summaries[dataset])
        source_path = str(raw.get("source_path", "")).rstrip("/")
        if source_path != TARGET_DATASETS[dataset]:
            raise ValueError(
                "%s source path drifted: %r != %r"
                % (dataset, source_path, TARGET_DATASETS[dataset])
            )
        counts = dict(raw.get("admission_counts") or {})
        for field in _REQUIRED_INVENTORY_COUNTS:
            value = counts.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "%s.%s must be a positive integer" % (dataset, field)
                )
        if int(counts["outer_train_fault_cases"]) < 30:
            raise ValueError(
                "%s requires at least 30 outer-training fault cases"
                % dataset
            )
        split = deepcopy(dict(raw.get("split") or {}))
        if not split:
            raise ValueError("%s split identity fields must not be empty" % dataset)
        provenance = deepcopy(dict(raw.get("provenance") or {}))
        _require_sha256(
            provenance.get("source_evidence_sha256"),
            "%s.provenance.source_evidence_sha256" % dataset,
        )
        dataset_record = {
            "canonical_dataset_id": dataset,
            "source_path": source_path,
            "evidence_tier": str(raw.get("evidence_tier", "")),
            "admission_counts": counts,
            "split": split,
            "split_identity_sha256": _semantic_sha256(
                {
                    "canonical_dataset_id": dataset,
                    "source_path": source_path,
                    "admission_counts": counts,
                    "split": split,
                }
            ),
            "provenance": provenance,
        }
        if not dataset_record["evidence_tier"]:
            raise ValueError("%s evidence tier must not be empty" % dataset)
        datasets[dataset] = dataset_record
    payload = {
        "schema_version": "rcl-query-active-learning-inventory-v1",
        "change_id": CHANGE_ID,
        "canonical_dataset_ids": sorted(TARGET_DATASETS),
        "datasets": datasets,
        "formal_refresh_required": any(
            record["evidence_tier"] not in _FORMAL_TARGET_PAIR_TIERS
            for record in datasets.values()
        ),
    }
    payload["inventory_sha256"] = _semantic_sha256(payload)
    return payload


def _find_forbidden_selector_inputs(value: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = "%s.%s" % (path, key_text) if path else key_text
            lowered_key = key_text.lower()
            if any(token in lowered_key for token in _FORBIDDEN_SELECTOR_KEY_TOKENS):
                findings.append(child_path)
            findings.extend(_find_forbidden_selector_inputs(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_path = "%s[%d]" % (path, index) if path else "[%d]" % index
            findings.extend(_find_forbidden_selector_inputs(child, child_path))
    elif isinstance(value, str):
        lowered_value = value.lower()
        if any(token in lowered_value for token in _FORBIDDEN_SELECTOR_VALUE_TOKENS):
            findings.append(path or "<value>")
    return findings


def audit_selector_input_contract(
    selector_input_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed if selector inputs expose outer-test labels/predictions/scores."""

    payload = deepcopy(dict(selector_input_manifest))
    findings = sorted(set(_find_forbidden_selector_inputs(payload)))
    result = {
        "schema_version": "rcl-query-active-learning-selector-leakage-audit-v1",
        "change_id": CHANGE_ID,
        "selection_partition": payload.get("selection_partition"),
        "sealed_outer_test_not_read": not findings,
        "forbidden_findings": findings,
    }
    if findings:
        raise ValueError(
            "forbidden selector input fields detected: %s" % findings
        )
    return result


def build_oracle_gate_manifest(
    *,
    run_id: str,
    output_root: str,
    inventory_sha256: str,
) -> Dict[str, Any]:
    """Build the formal target-pair oracle gate manifest."""

    run_id_text = str(run_id).strip()
    if not run_id_text or "/" in run_id_text or "\\" in run_id_text:
        raise ValueError("run_id must be a path-safe identifier")
    root = str(output_root).rstrip("/")
    allowed_prefix = APPROVED_OUTPUT_ROOT + "/"
    if not root.startswith(allowed_prefix):
        raise ValueError(
            "oracle output_root must be below %s" % APPROVED_OUTPUT_ROOT
        )
    inventory_hash = _require_sha256(inventory_sha256, "inventory_sha256")
    units = [
        {
            "unit_id": "oracle.aiops2022_pre.seed42",
            "canonical_dataset_id": "aiops2022_pre",
            "mode": "run_aiops2022_pre_oracle_full",
            "environment": "rcalab",
            "source_path": TARGET_DATASETS["aiops2022_pre"],
            "output_root": root + "/units/oracle.aiops2022_pre.seed42",
            "budget": None,
            "seed": 42,
            "normal_policy": "fault_only",
            "command": [
                "python",
                "scripts/run_query_active_aiops2022_pre_oracle.py",
                "--output-root",
                root + "/units/oracle.aiops2022_pre.seed42",
            ],
        },
        {
            "unit_id": "oracle.rcabench.seed42",
            "canonical_dataset_id": "rcabench",
            "mode": "validate_existing_rcabench_oracle_full",
            "environment": "rcalab",
            "source_path": TARGET_DATASETS["rcabench"],
            "existing_result_path": RCABENCH_EXISTING_ORACLE_RESULT,
            "existing_result_sha256": RCABENCH_EXISTING_ORACLE_SHA256,
            "budget": None,
            "seed": 42,
            "normal_policy": "fault_only",
        },
    ]
    manifest: Dict[str, Any] = {
        "schema_version": "rcl-query-active-oracle-gate-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": run_id_text,
        "output_root": root,
        "target_datasets": deepcopy(TARGET_DATASETS),
        "sota_thresholds": deepcopy(SOTA_THRESHOLDS),
        "inventory_sha256": inventory_hash,
        "units": units,
        "execution": {
            "runner_entrypoint": "scripts/run_query_active_oracle_gate.py",
            "status_entrypoint": "scripts/status_query_active_oracle_gate.sh",
            "completion_policy": (
                "COMPLETED.json_and_all.done_after_gate_validation"
            ),
        },
        "completion_proofs": {
            "completed_json": root + "/COMPLETED.json",
            "all_done": root + "/all.done",
            "gate_summary": root + "/oracle_gate_summary.json",
        },
    }
    manifest["manifest_sha256"] = _semantic_sha256(manifest)
    return manifest


def _require_safe_run_id(run_id: Any, context: str = "run_id") -> str:
    run_id_text = str(run_id).strip()
    if not run_id_text or "/" in run_id_text or "\\" in run_id_text:
        raise ValueError("%s must be a path-safe identifier" % context)
    return run_id_text


def _selector_default_config(selector_id: str) -> Dict[str, Any]:
    if selector_id == "scheme1_baseline":
        return {"rank_field": "scheme1_rank"}
    if selector_id == "diversity_maxmin":
        return {"embedding_field": "embedding"}
    if selector_id == "stratified_cluster_quota":
        return {"stratify_fields": ["cluster_id", "service"]}
    if selector_id == "uncertainty_boundary":
        return {"uncertainty_field": "inner_boundary_uncertainty"}
    if selector_id == "coverage_capped_uncertainty":
        return {
            "score_field": "inner_boundary_uncertainty",
            "embedding_field": "embedding",
            "cap_fields": ["cluster_id", "time_bucket"],
            "max_fraction_per_field": {"cluster_id": 0.15, "time_bucket": 0.25},
            "diversity_weight": 0.10,
        }
    if selector_id == "metric_signature_balanced_uncertainty":
        return {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
        }
    if selector_id == "sequential_proxy_mode_query":
        return {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "seed_budget": 8,
        }
    if selector_id == "unknown_mode_signature_reserve":
        return {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score", "scheme1_rank"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "unknown_mode_reserve_budget": 8,
            "coverage_anchor_count": 256,
            "unknown_mode_weight": 0.40,
            "coverage_weight": 0.35,
            "utility_weight": 0.20,
            "mode_novelty_weight": 0.05,
            "metric_ad_required": False,
            "metric_ad_fields": list(_RICH_METRIC_AD_FIELDS),
        }
    if selector_id == "two_stage_diverse_uncertainty":
        return {
            "seed_budget": 5,
            "embedding_field": "embedding",
            "uncertainty_field": "inner_boundary_uncertainty",
        }
    if selector_id == "mpca_dual_factor_query":
        return {
            "mechanism_weight": 0.30,
            "propagation_weight": 0.30,
            "uncertainty_weight": 0.25,
            "joint_weight": 0.15,
            "uncertainty_field": "inner_boundary_uncertainty",
        }
    raise ValueError("unknown query selector: %s" % selector_id)


def build_active_query_screening_manifest(
    *,
    run_id: str,
    output_root: str,
    inventory_sha256: str,
    feature_bundle_sha256: str,
    selector_ids: Sequence[str],
    seeds: Sequence[int],
    normal_window_options: Sequence[bool],
) -> Dict[str, Any]:
    """Build a frozen query-only active-selection screening manifest."""

    run_id_text = _require_safe_run_id(run_id)
    root = str(output_root).rstrip("/")
    allowed_prefix = APPROVED_OUTPUT_ROOT + "/"
    if not root.startswith(allowed_prefix):
        raise ValueError(
            "screening output_root must be below %s" % APPROVED_OUTPUT_ROOT
        )
    inventory_hash = _require_sha256(inventory_sha256, "inventory_sha256")
    feature_hash = _require_sha256(feature_bundle_sha256, "feature_bundle_sha256")
    selectors = [str(selector_id) for selector_id in selector_ids]
    if not selectors:
        raise ValueError("screening manifest requires at least one selector")
    for selector_id in selectors:
        if selector_id not in _QUERY_SELECTOR_REGISTRY:
            raise ValueError("unknown query selector: %s" % selector_id)
    selector_registry_snapshot = {
        selector_id: deepcopy(_QUERY_SELECTOR_REGISTRY[selector_id])
        for selector_id in selectors
    }
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("screening manifest requires at least one seed")
    normal_values = [bool(value) for value in normal_window_options]
    if not normal_values:
        raise ValueError("screening manifest requires normal-window options")
    units = []
    for dataset in sorted(TARGET_DATASETS):
        for selector_id in selectors:
            selector_config = _selector_default_config(selector_id)
            selector_config_hash = _semantic_sha256(
                {
                    "selector_id": selector_id,
                    "selector_config": selector_config,
                }
            )
            for seed in seed_values:
                for use_normal in normal_values:
                    normal_policy_id = (
                        "with_normal_windows" if use_normal else "fault_only"
                    )
                    unit_id = (
                        "screen.%s.%s.seed%d.%s"
                        % (dataset, selector_id, seed, normal_policy_id)
                    )
                    units.append(
                        {
                            "unit_id": unit_id,
                            "canonical_dataset_id": dataset,
                            "source_path": TARGET_DATASETS[dataset],
                            "arm": "query_only",
                            "budget": 30,
                            "budget_split": "outer_train",
                            "budget_case_kind": "fault",
                            "selector_id": selector_id,
                            "selector_config": deepcopy(selector_config),
                            "selector_config_sha256": selector_config_hash,
                            "seed": seed,
                            "normal_policy": normal_policy_id,
                            "use_normal_windows_for_training": use_normal,
                            "evaluation_partition": "inner_validation",
                            "mode": "run_query_only_inner_validation_screen",
                            "output_root": root + "/units/" + unit_id,
                        }
                    )
    manifest: Dict[str, Any] = {
        "schema_version": "rcl-query-active-screening-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": run_id_text,
        "output_root": root,
        "target_datasets": deepcopy(TARGET_DATASETS),
        "budget": 30,
        "inventory_sha256": inventory_hash,
        "feature_bundle_sha256": feature_hash,
        "selector_registry_snapshot": selector_registry_snapshot,
        "objective": {
            "declared_before_results": True,
            "primary": "mean_inner_validation_hit_at_1",
            "tie_breakers": [
                "mean_inner_validation_hit_at_3",
                "mean_inner_validation_hit_at_5",
                "mean_oracle_retention_hit_at_1",
                "lower_selector_complexity",
                "selector_config_sha256",
            ],
        },
        "units": units,
        "execution": {
            "runner_entrypoint": "scripts/run_query_active_screening.py",
            "status_entrypoint": "scripts/status_query_active_screening.sh",
            "completion_policy": "COMPLETED.json_and_all.done_after_validation",
        },
        "completion_proofs": {
            "completed_json": root + "/COMPLETED.json",
            "all_done": root + "/all.done",
            "screening_summary": root + "/screening_summary.json",
        },
    }
    manifest["manifest_sha256"] = _semantic_sha256(manifest)
    return manifest


def build_gate_failed_report(
    gate_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the terminal report required when oracle-full misses SOTA."""

    summary = deepcopy(dict(gate_summary))
    blocking = list(summary.get("blocking_metrics") or [])
    if not blocking:
        raise ValueError("gate-failed report requires blocking metrics")
    report = {
        "schema_version": "rcl-query-active-oracle-gate-failed-v1",
        "change_id": CHANGE_ID,
        "status": "gate_failed",
        "do_not_launch_screening": True,
        "blocking_metrics": blocking,
        "gate_summary": summary,
    }
    report["report_sha256"] = _semantic_sha256(report)
    return report


def _targets_for_case(
    authoritative_labels: Mapping[str, Sequence[str]], case_id: str
) -> Tuple[str, ...]:
    raw_targets = authoritative_labels.get(case_id)
    if not isinstance(raw_targets, (list, tuple, set)):
        raise ValueError("missing authoritative labels for %s" % case_id)
    targets = tuple(
        sorted({str(target).strip() for target in raw_targets if str(target).strip()})
    )
    if not targets:
        raise ValueError("missing authoritative labels for %s" % case_id)
    return targets


def build_frozen_query_plan(
    *,
    canonical_dataset_id: str,
    selected_cases: Sequence[Mapping[str, Any]],
    authoritative_labels: Mapping[str, Sequence[str]],
    selector_id: str,
    selector_config: Mapping[str, Any],
    split_identity_sha256: str,
    feature_hashes: Mapping[str, str],
    case_inventory_sha256: str,
) -> Dict[str, Any]:
    """Freeze exactly 30 queried outer-training fault cases and annotations."""

    dataset = str(canonical_dataset_id)
    if dataset not in TARGET_DATASETS:
        raise ValueError("unsupported target dataset for query plan: %s" % dataset)
    if len(selected_cases) != 30:
        raise ValueError("frozen query plan must contain exactly 30 cases")
    selector_text = str(selector_id).strip()
    if not selector_text:
        raise ValueError("selector_id must not be empty")
    split_hash = _require_sha256(split_identity_sha256, "split_identity_sha256")
    inventory_hash = _require_sha256(case_inventory_sha256, "case_inventory_sha256")
    normalized_feature_hashes = {
        str(key): _require_sha256(value, "feature_hashes.%s" % key)
        for key, value in sorted(dict(feature_hashes).items())
    }
    annotations = []
    seen = set()
    for rank, raw in enumerate(selected_cases, start=1):
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("selected case ID must not be empty")
        if case_id in seen:
            raise ValueError("frozen query plan requires 30 unique case IDs")
        seen.add(case_id)
        if str(raw.get("split", "")) != "outer_train" or str(
            raw.get("case_kind", "")
        ) != "fault":
            raise ValueError("frozen query plan may contain only outer_train fault cases")
        targets = _targets_for_case(authoritative_labels, case_id)
        annotations.append(
            {
                "case_id": case_id,
                "native_case_id": str(raw.get("native_case_id", case_id)).strip()
                or case_id,
                "query_rank": rank,
                "split": "outer_train",
                "case_kind": "fault",
                "annotation_source": "simulated_manual_ground_truth",
                "targets": list(targets),
            }
        )
    if len(seen) != 30:
        raise ValueError("frozen query plan requires 30 unique case IDs")
    selector_config_payload = deepcopy(dict(selector_config))
    selector_config_sha256 = _semantic_sha256(selector_config_payload)
    plan_identity = {
        "schema_version": "rcl-query-active-frozen-query-plan-v1",
        "canonical_dataset_id": dataset,
        "budget": 30,
        "budget_unit": "unique_outer_training_fault_cases",
        "selector_id": selector_text,
        "selector_config": selector_config_payload,
        "selector_config_sha256": selector_config_sha256,
        "split_identity_sha256": split_hash,
        "feature_hashes": normalized_feature_hashes,
        "case_inventory_sha256": inventory_hash,
        "annotations": annotations,
    }
    return {**plan_identity, "query_plan_sha256": _semantic_sha256(plan_identity)}


class QueryAnnotationSimulator:
    """Reveal simulated manual labels only for cases frozen in a query plan."""

    def __init__(self, query_plan: Mapping[str, Any]):
        annotations = query_plan.get("annotations")
        if not isinstance(annotations, Sequence):
            raise ValueError("query plan annotations must be a sequence")
        self._labels = {
            str(row["case_id"]): tuple(str(target) for target in row["targets"])
            for row in annotations
        }

    def reveal(self, case_id: str) -> Tuple[str, ...]:
        key = str(case_id)
        if key not in self._labels:
            raise ValueError("unqueried case cannot reveal labels: %s" % key)
        return self._labels[key]


def validate_query_only_training_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    queried_case_ids: Iterable[str],
) -> Dict[str, Any]:
    """Ensure unqueried cases carry no authoritative/pseudo positive signal."""

    queried = {str(case_id) for case_id in queried_case_ids}
    bad_cases = []
    total = 0
    for row in rows:
        total += 1
        case_id = str(row.get("case_id", "")).strip()
        label_source = str(row.get("label_source", "")).lower()
        weight = float(row.get("positive_training_weight", 0.0) or 0.0)
        if case_id not in queried and (
            weight > 0.0
            or "ground_truth" in label_source
            or "manual" in label_source
            or "pseudo" in label_source
        ):
            bad_cases.append(case_id)
    if bad_cases:
        raise ValueError(
            "unqueried fault cases received positive supervision: %s"
            % sorted(set(bad_cases))[:10]
        )
    return {
        "schema_version": "rcl-query-only-training-frame-validation-v1",
        "valid": True,
        "row_count": total,
        "queried_case_count": len(queried),
    }


def normal_window_policy(use_normal_windows: bool) -> Dict[str, Any]:
    """Return the label-free normal-window ablation policy."""

    enabled = bool(use_normal_windows)
    return {
        "use_normal_windows_for_training": enabled,
        "normal_training_label": "__normal__" if enabled else None,
        "final_test_admission": "fault_only",
    }


__all__ = [
    "APPROVED_OUTPUT_ROOT",
    "CHANGE_ID",
    "HIT_METRICS",
    "REMOTE_WORKSPACE",
    "RUN_ID",
    "RCABENCH_EXISTING_ORACLE_RESULT",
    "RCABENCH_EXISTING_ORACLE_SHA256",
    "SOTA_THRESHOLDS",
    "TARGET_DATASETS",
    "QueryAnnotationSimulator",
    "assign_proxy_fault_modes",
    "audit_selector_input_contract",
    "build_gate_failed_report",
    "build_metric_signatures",
    "build_frozen_query_plan",
    "build_active_query_screening_manifest",
    "build_oracle_gate_manifest",
    "build_target_pair_inventory",
    "list_query_selectors",
    "normal_window_policy",
    "select_query_cases",
    "target_pair_config",
    "validate_query_only_training_rows",
    "validate_oracle_gate",
]
