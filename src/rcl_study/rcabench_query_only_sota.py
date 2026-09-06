"""RCABench-only query-only SOTA-search protocol helpers.

This module intentionally keeps the outer-test-guided search contract separate
from the earlier query-active final helpers, whose official protocol is
outer-test-only and budget-30 fixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


CHANGE_ID = "optimize-rcabench-query-only-sota"
STABLE_SELECTION_CHANGE_ID = "stabilize-rcabench-query-selection"
CANONICAL_DATASET_ID = "rcabench"
RCABENCH_SOURCE_PATH = "${RCABENCH_ROOT}"
REMOTE_WORKSPACE = "${RCL_WORKSPACE}"
SPLIT_SEED = 42
ACTIVE_LEARNING_SEEDS = (42, 43, 44)
BUDGET_GRID = (35, 40, 45, 50, 55, 60, 70, 80)
HIT_METRICS = ("hit_at_1", "hit_at_3", "hit_at_5")
RANDOM_EARLY_STOP_METRICS = (*HIT_METRICS, "mrr")
SOTA_THRESHOLDS = {
    "hit_at_1": 0.4175,
    "hit_at_3": 0.6351,
    "hit_at_5": 0.7368,
}
RCABENCH_BASELINE_MRR_SCORES = {
    "BARO": 0.43665025785279,
    "SimpleRCA": 0.468190342240976,
    "ART": 0.075906432748538,
    "MicroDig": 0.451090014064698,
    "MicroHECL": 0.349507735583685,
    "MicroRank": 0.0968471636193155,
    "MicroRCA": 0.45631739334271,
    "ShapleyIQ": 0.313748241912798,
    "Nezha": 0.455309423347398,
    "DiagFusion": 0.477894736842105,
    "Eadro": 0.183040935672515,
    "CausalRCA": 0.0443389592123769,
}
RCABENCH_BASELINE_MRR_SOTA_THRESHOLD = max(RCABENCH_BASELINE_MRR_SCORES.values())
RANDOM_EARLY_STOP_THRESHOLDS = {
    **SOTA_THRESHOLDS,
    "mrr": RCABENCH_BASELINE_MRR_SOTA_THRESHOLD,
}
AUTHORITY_BASELINE_METRICS = {
    "selector_id": "uncertainty_boundary",
    "source": "openspec_archive_authority_baseline",
    "T1": {
        "hit_at_1": 0.426230,
        "hit_at_3": 0.543326,
        "hit_at_5": 0.594848,
        "mrr": 0.515604,
    },
}
TARGET_RANDOM_CANDIDATE_METRICS = {
    "unit_id": "random.rcabench.round10.set0001",
    "source": "openspec_archive_random_diagnostic_winner",
    "diagnostic_only": True,
    "T1": {
        "hit_at_1": 0.5971896955503513,
        "hit_at_3": 0.7540983606557377,
        "hit_at_5": 0.8009367681498829,
        "mrr": 0.693295655308535,
    },
}
FAILED_TRANSFER_METRICS = {
    "selector_id": "coverage_capped_uncertainty",
    "source": "openspec_archive_failed_transfer_strategy",
    "T1": {
        "hit_at_1": 0.355972,
        "hit_at_3": 0.573770,
        "hit_at_5": 0.648712,
        "mrr": 0.497200,
    },
}
ARCHIVED_PROTOCOL_REFERENCES = {
    "archive_change_root": (
        "openspec/changes/archive/2026-08-19-optimize-rcabench-query-only-sota"
    ),
    "authority_baseline_summary": "baseline_replay_authority_summary.md",
    "random_diagnostic_report": "random_diagnostic_report_mrr_corrected.md",
    "final_status_report": "final_report_mrr_corrected.md",
}
STABLE_SELECTOR_FAMILIES = {
    "metric_signature_balanced_uncertainty": "metric_signature_proxy_balance",
    "sequential_proxy_mode_query": "sequential_proxy_mode",
}
STABLE_REPORT_TAGS = (
    "outer_test_guided_engineering_search",
    "label_clean_stable_selector",
)
STABLE_STRATEGY_SUITE_ID = "stable_query_selection_budget30_v1"
DEFAULT_STABLE_SELECTOR_CONFIGS = (
    {
        "config_id": "one_shot_proxy12_cap4",
        "selector_id": "metric_signature_balanced_uncertainty",
        "selector_config": {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 12,
            "max_per_proxy_mode": 4,
            "metric_ad_required": False,
        },
    },
    {
        "config_id": "one_shot_proxy14_cap5",
        "selector_id": "metric_signature_balanced_uncertainty",
        "selector_config": {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "metric_ad_required": False,
        },
    },
    {
        "config_id": "one_shot_proxy16_cap5",
        "selector_id": "metric_signature_balanced_uncertainty",
        "selector_config": {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 16,
            "max_per_proxy_mode": 5,
            "metric_ad_required": False,
        },
    },
    {
        "config_id": "sequential_proxy14_seed6_cap5",
        "selector_id": "sequential_proxy_mode_query",
        "selector_config": {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "seed_budget": 6,
            "metric_ad_required": False,
        },
    },
    {
        "config_id": "sequential_proxy14_seed8_cap5",
        "selector_id": "sequential_proxy_mode_query",
        "selector_config": {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "seed_budget": 8,
            "metric_ad_required": False,
        },
    },
    {
        "config_id": "sequential_proxy14_seed10_cap5",
        "selector_id": "sequential_proxy_mode_query",
        "selector_config": {
            "embedding_field": "embedding",
            "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
            "categorical_fields": ["cluster_id", "time_bucket"],
            "score_field": "inner_boundary_uncertainty",
            "proxy_mode_count": 14,
            "max_per_proxy_mode": 5,
            "seed_budget": 10,
            "metric_ad_required": False,
        },
    },
)
ALLOWED_PRE_ANNOTATION_FIELDS = (
    "case_id",
    "native_case_id",
    "embedding",
    "cluster_id",
    "time_bucket",
    "inner_boundary_uncertainty",
    "baseline_score",
    "scheme1_rank",
    "metric_anomaly_summary",
    "metric_ad_event_summary",
)
FORBIDDEN_UNQUERIED_LABEL_FIELDS = (
    "ground_truth",
    "targets",
    "target_set",
    "root_cause",
    "rootcause",
    "fault_type",
    "true_fault_type",
)
ACTIVE_LEARNING_SEED_ROLE = (
    "selection_randomness_query_order_tie_breaking_or_strategy_stochasticity"
)
SotaUnitExecutor = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("%s.tmp-%d" % (destination.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(destination))


def _require_rcabench(canonical_dataset_id: str) -> None:
    if str(canonical_dataset_id) != CANONICAL_DATASET_ID:
        raise ValueError("this change supports rcabench only")


def _unique_text_list(values: Sequence[Any], context: str) -> list[str]:
    rows = [str(value).strip() for value in values]
    if not rows or any(not row for row in rows):
        raise ValueError("%s must contain non-empty case IDs" % context)
    if len(rows) != len(set(rows)):
        raise ValueError("%s must contain unique case IDs" % context)
    return rows


def build_split_identity(
    *,
    canonical_dataset_id: str,
    source_path: str,
    outer_train_case_ids: Sequence[Any],
    outer_test_case_ids: Sequence[Any],
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Freeze the fixed RCABench split identity used by this change."""

    _require_rcabench(str(canonical_dataset_id))
    if int(split_seed) != SPLIT_SEED:
        raise ValueError("split_seed must remain fixed at 42")
    train_ids = _unique_text_list(outer_train_case_ids, "outer_train_case_ids")
    test_ids = _unique_text_list(outer_test_case_ids, "outer_test_case_ids")
    overlap = sorted(set(train_ids).intersection(test_ids))
    if overlap:
        raise ValueError("outer_train and outer_test overlap: %s" % overlap[:10])
    identity = {
        "schema_version": "rcabench-query-only-sota-split-identity-v1",
        "change_id": CHANGE_ID,
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "source_path": str(source_path),
        "split_seed": SPLIT_SEED,
        "active_learning_seed_role": ACTIVE_LEARNING_SEED_ROLE,
        "outer_train_case_ids": train_ids,
        "outer_test_case_ids": test_ids,
    }
    identity["split_identity_sha256"] = semantic_sha256(identity)
    return identity


def build_score_partitions(
    split_identity: Mapping[str, Any],
    *,
    queried_case_ids: Iterable[Any],
) -> dict[str, list[str]]:
    """Return official T1 and diagnostic T2 case IDs for one query plan."""

    _require_rcabench(str(split_identity.get("canonical_dataset_id")))
    queried = {str(case_id) for case_id in queried_case_ids}
    outer_test = [str(case_id) for case_id in split_identity["outer_test_case_ids"]]
    outer_train = [str(case_id) for case_id in split_identity["outer_train_case_ids"]]
    return {
        "T1": [case_id for case_id in outer_test if case_id not in queried],
        "T2": [case_id for case_id in outer_test if case_id not in queried]
        + [case_id for case_id in outer_train if case_id not in queried],
    }


def _first_target_rank(
    ranking: Sequence[Any],
    targets: Sequence[Any],
) -> int:
    target_set = {str(target) for target in targets}
    if not target_set:
        raise ValueError("targets must not be empty")
    for index, candidate in enumerate(ranking, start=1):
        if str(candidate) in target_set:
            return index
    return 0


def compute_ranking_metrics(
    *,
    case_ids: Sequence[Any],
    rankings_by_case: Mapping[str, Sequence[Any]],
    targets_by_case: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Compute Hit@1/3/5 and MRR for a fixed set of case IDs."""

    ids = [str(case_id) for case_id in case_ids]
    if not ids:
        raise ValueError("case_ids must not be empty")
    hit_counts = {metric: 0 for metric in HIT_METRICS}
    reciprocal_ranks: list[float] = []
    per_case = []
    for case_id in ids:
        if case_id not in rankings_by_case:
            raise ValueError("missing ranking for %s" % case_id)
        if case_id not in targets_by_case:
            raise ValueError("missing targets for %s" % case_id)
        ranking = [str(value) for value in rankings_by_case[case_id]]
        targets = [str(value) for value in targets_by_case[case_id]]
        rank = _first_target_rank(ranking, targets)
        reciprocal_ranks.append(0.0 if rank == 0 else 1.0 / float(rank))
        for k, metric in ((1, "hit_at_1"), (3, "hit_at_3"), (5, "hit_at_5")):
            if rank and rank <= k:
                hit_counts[metric] += 1
        per_case.append(
            {
                "case_id": case_id,
                "first_target_rank": rank,
                "targets": targets,
                "ranking": ranking,
            }
        )
    denominator = len(ids)
    return {
        "denominator": denominator,
        "evaluation_fault_cases": denominator,
        "hit_at_1": hit_counts["hit_at_1"] / denominator,
        "hit_at_3": hit_counts["hit_at_3"] / denominator,
        "hit_at_5": hit_counts["hit_at_5"] / denominator,
        "mrr": sum(reciprocal_ranks) / denominator,
        "hit_counts": hit_counts,
        "per_case": per_case,
    }


def evaluate_t1_sota_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] = SOTA_THRESHOLDS,
    required_metrics: Sequence[str] = HIT_METRICS,
) -> dict[str, Any]:
    metric_status = {}
    for metric in required_metrics:
        actual = float(metrics[metric])
        threshold = float(thresholds[metric])
        metric_status[metric] = {
            "actual": actual,
            "threshold": threshold,
            "strictly_exceeds": actual > threshold,
            "margin": actual - threshold,
        }
    return {
        "schema_version": "rcabench-query-only-sota-gate-v1",
        "passed": all(row["strictly_exceeds"] for row in metric_status.values()),
        "metrics": metric_status,
    }


def _validated_random_early_stop_thresholds(
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    rows = {str(metric): float(value) for metric, value in dict(thresholds).items()}
    missing = [metric for metric in RANDOM_EARLY_STOP_METRICS if metric not in rows]
    if missing:
        raise ValueError(
            "random early-stop thresholds must include %s"
            % ", ".join(sorted(missing))
        )
    return {metric: rows[metric] for metric in RANDOM_EARLY_STOP_METRICS}


def stability_warnings(
    *,
    t1_metrics: Mapping[str, Any],
    t2_metrics: Mapping[str, Any],
    diff_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    t1_gate = evaluate_t1_sota_gate(t1_metrics)
    t2_gate = evaluate_t1_sota_gate(t2_metrics)
    if bool(t1_gate["passed"]) != bool(t2_gate["passed"]):
        warnings.append(
            {
                "reason": "strict_sota_pass_status_disagreement",
                "t1_passed": bool(t1_gate["passed"]),
                "t2_passed": bool(t2_gate["passed"]),
            }
        )
    for metric in HIT_METRICS:
        t1_value = float(t1_metrics[metric])
        t2_value = float(t2_metrics[metric])
        absolute_difference = abs(t1_value - t2_value)
        if absolute_difference >= float(diff_threshold):
            warnings.append(
                {
                    "reason": "absolute_metric_gap_at_least_threshold",
                    "metric": metric,
                    "t1_value": t1_value,
                    "t2_value": t2_value,
                    "absolute_difference": absolute_difference,
                    "threshold": float(diff_threshold),
                }
            )
    return warnings


def _case_id_aliases(case_id: Any) -> list[str]:
    text = str(case_id).strip()
    aliases = [text]
    if "::" in text:
        aliases.append(text.split("::", 1)[1])
    elif text:
        aliases.append("%s::%s" % (CANONICAL_DATASET_ID, text))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _ranking_row_lookup(
    ranking_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for row in ranking_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("ranking row missing case_id")
        for alias in _case_id_aliases(case_id):
            existing = lookup.get(alias)
            if existing is not None and existing is not row:
                raise ValueError("ambiguous ranking rows for case alias %s" % alias)
            lookup[alias] = row
    if not lookup:
        raise ValueError("ranking_rows must not be empty")
    return lookup


def _lookup_ranking_row(
    lookup: Mapping[str, Mapping[str, Any]],
    case_id: str,
) -> Mapping[str, Any]:
    for alias in _case_id_aliases(case_id):
        row = lookup.get(alias)
        if row is not None:
            return row
    raise ValueError("missing ranking for %s" % case_id)


def score_t1_t2_from_rankings(
    *,
    split_identity: Mapping[str, Any],
    queried_case_ids: Iterable[Any],
    ranking_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute official T1 and diagnostic T2 metrics from per-case rankings."""

    _require_rcabench(str(split_identity.get("canonical_dataset_id")))
    queried = [str(case_id).strip() for case_id in queried_case_ids]
    partitions = build_score_partitions(split_identity, queried_case_ids=queried)
    lookup = _ranking_row_lookup(ranking_rows)
    view_metrics: dict[str, Any] = {}
    for view, case_ids in partitions.items():
        rankings_by_case = {}
        targets_by_case = {}
        for case_id in case_ids:
            row = _lookup_ranking_row(lookup, str(case_id))
            ranking = row.get("ranking")
            targets = row.get("targets")
            if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
                raise ValueError("ranking must be a sequence for %s" % case_id)
            if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
                raise ValueError("targets must be a sequence for %s" % case_id)
            rankings_by_case[str(case_id)] = [str(value) for value in ranking]
            targets_by_case[str(case_id)] = [str(value) for value in targets]
        metrics = compute_ranking_metrics(
            case_ids=case_ids,
            rankings_by_case=rankings_by_case,
            targets_by_case=targets_by_case,
        )
        metrics["target_case_ids"] = list(case_ids)
        view_metrics[view] = metrics
    t1_gate = evaluate_t1_sota_gate(view_metrics["T1"])
    warnings = stability_warnings(
        t1_metrics=view_metrics["T1"],
        t2_metrics=view_metrics["T2"],
    )
    return {
        "schema_version": "rcabench-query-only-sota-t1-t2-score-v1",
        "partitions": partitions,
        "queried_case_ids": queried,
        "T1": view_metrics["T1"],
        "T2": view_metrics["T2"],
        "t1_sota_gate": t1_gate,
        "t1_t2_warnings": warnings,
        "ranking_row_count": len(ranking_rows),
    }


def _targets_for_case(
    authoritative_labels: Mapping[str, Sequence[Any]],
    case_id: str,
) -> list[str]:
    raw = authoritative_labels.get(case_id)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("missing authoritative labels for %s" % case_id)
    targets = [str(target).strip() for target in raw if str(target).strip()]
    if not targets:
        raise ValueError("missing authoritative labels for %s" % case_id)
    return targets


def freeze_budget_query_plan(
    *,
    canonical_dataset_id: str,
    split_identity: Mapping[str, Any],
    selected_case_ids: Sequence[Any],
    authoritative_labels: Mapping[str, Sequence[Any]],
    selector_id: str,
    selector_config: Mapping[str, Any],
    active_learning_seed: int,
    query_rounds: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Freeze one one-shot or sequential budgeted query plan."""

    _require_rcabench(str(canonical_dataset_id))
    _require_rcabench(str(split_identity.get("canonical_dataset_id")))
    selected = _unique_text_list(selected_case_ids, "selected_case_ids")
    train_ids = {str(case_id) for case_id in split_identity["outer_train_case_ids"]}
    not_train = [case_id for case_id in selected if case_id not in train_ids]
    if not_train:
        raise ValueError("query plan may contain only outer_train cases")
    if query_rounds is None:
        normalized_rounds = [selected]
    else:
        normalized_rounds = [
            [str(case_id).strip() for case_id in round_ids]
            for round_ids in query_rounds
        ]
        flattened = [case_id for round_ids in normalized_rounds for case_id in round_ids]
        if flattened != selected:
            raise ValueError("query_rounds must flatten to selected_case_ids order")
    annotations = []
    for rank, case_id in enumerate(selected, start=1):
        annotations.append(
            {
                "case_id": case_id,
                "query_rank": rank,
                "split": "outer_train",
                "case_kind": "fault",
                "annotation_source": "simulated_manual_ground_truth",
                "targets": _targets_for_case(authoritative_labels, case_id),
            }
        )
    config = deepcopy(dict(selector_config))
    selector_config_sha256 = semantic_sha256(
        {
            "selector_id": str(selector_id),
            "selector_config": config,
        }
    )
    identity = {
        "schema_version": "rcabench-query-only-sota-query-plan-v1",
        "change_id": CHANGE_ID,
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "budget": len(selected),
        "budget_unit": "unique_outer_train_fault_cases",
        "active_learning_seed": int(active_learning_seed),
        "active_learning_seed_role": ACTIVE_LEARNING_SEED_ROLE,
        "selector_id": str(selector_id),
        "selector_config": config,
        "selector_config_sha256": selector_config_sha256,
        "split_identity_sha256": str(split_identity["split_identity_sha256"]),
        "selected_case_ids": selected,
        "query_rounds": normalized_rounds,
        "query_round_counts": [len(round_ids) for round_ids in normalized_rounds],
        "annotations": annotations,
    }
    identity["query_plan_sha256"] = semantic_sha256(identity)
    return identity


class BudgetAnnotationSimulator:
    """Reveal simulated manual labels only for cases in a frozen query plan."""

    def __init__(self, query_plan: Mapping[str, Any]):
        self._labels = {
            str(row["case_id"]): tuple(str(target) for target in row["targets"])
            for row in query_plan.get("annotations", [])
        }

    def reveal(self, case_id: str) -> tuple[str, ...]:
        key = str(case_id)
        if key not in self._labels:
            raise ValueError("unqueried case cannot reveal labels: %s" % key)
        return self._labels[key]


def validate_training_rows_no_unqueried_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    queried_case_ids: Iterable[Any],
) -> dict[str, Any]:
    queried = {str(case_id) for case_id in queried_case_ids}
    bad_cases = []
    total = 0
    for row in rows:
        total += 1
        case_id = str(row.get("case_id", "")).strip()
        source = str(row.get("label_source", "")).lower()
        weight = float(row.get("positive_training_weight", 0.0) or 0.0)
        if case_id not in queried and (
            weight > 0.0
            or "ground_truth" in source
            or "manual" in source
            or "pseudo" in source
        ):
            bad_cases.append(case_id)
    if bad_cases:
        raise ValueError(
            "unqueried cases received supervision: %s" % sorted(set(bad_cases))[:10]
        )
    return {
        "schema_version": "rcabench-query-only-sota-training-validation-v1",
        "valid": True,
        "row_count": total,
        "queried_case_count": len(queried),
    }


def _manifest_base(
    *,
    run_id: str,
    output_root: str,
    split_identity: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    _require_rcabench(str(split_identity.get("canonical_dataset_id")))
    return {
        "schema_version": "rcabench-query-only-sota-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": str(run_id),
        "phase": str(phase),
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "source_path": RCABENCH_SOURCE_PATH,
        "output_root": str(output_root).rstrip("/"),
        "split_seed": SPLIT_SEED,
        "active_learning_seed_role": ACTIVE_LEARNING_SEED_ROLE,
        "split_identity_sha256": str(split_identity["split_identity_sha256"]),
        "outer_train_case_ids": list(split_identity["outer_train_case_ids"]),
        "outer_test_case_ids": list(split_identity["outer_test_case_ids"]),
        "evaluation_views": ["T1", "T2"],
        "sota_thresholds": deepcopy(SOTA_THRESHOLDS),
        "execution_mode": "outer_test_guided_engineering_search",
    }


def _signature_config_from_selector_config(
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
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


def _proxy_config_from_selector_config(
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = deepcopy(dict(selector_config.get("proxy_config") or {}))
    for key in ("proxy_mode_count", "algorithm", "iterations"):
        if key in selector_config and key not in config:
            config[key] = selector_config[key]
    return config


def _stable_selector_metadata(
    *,
    selector_id: str,
    selector_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    family = STABLE_SELECTOR_FAMILIES.get(str(selector_id))
    if family is None:
        return None
    signature_config = _signature_config_from_selector_config(selector_config)
    proxy_config = _proxy_config_from_selector_config(selector_config)
    leakage_validation = {
        "schema_version": "rcabench-stable-query-selection-leakage-validation-v1",
        "allowed_pre_annotation_fields": list(ALLOWED_PRE_ANNOTATION_FIELDS),
        "forbidden_unqueried_fields": list(FORBIDDEN_UNQUERIED_LABEL_FIELDS),
        "case_id_semantics": "opaque_identifier_only",
        "ground_truth_access": "budgeted_annotations_only",
        "revealed_label_budget_policy": "all_reveals_count_against_budget",
        "fixed_replay_policy": "diagnostic_ids_forbidden_as_selector_input",
    }
    metadata = {
        "schema_version": "rcabench-stable-query-selection-metadata-v1",
        "stable_change_id": STABLE_SELECTION_CHANGE_ID,
        "archived_protocol_change_id": CHANGE_ID,
        "selector_id": str(selector_id),
        "selector_family": family,
        "signature_config": signature_config,
        "signature_config_sha256": semantic_sha256(signature_config),
        "proxy_config": proxy_config,
        "proxy_config_sha256": semantic_sha256(proxy_config),
        "leakage_validation": leakage_validation,
        "report_tags": list(STABLE_REPORT_TAGS),
        "archived_protocol_references": deepcopy(ARCHIVED_PROTOCOL_REFERENCES),
        "authority_baseline": deepcopy(AUTHORITY_BASELINE_METRICS),
        "target_random_candidate": deepcopy(TARGET_RANDOM_CANDIDATE_METRICS),
        "failed_transfer": deepcopy(FAILED_TRANSFER_METRICS),
    }
    metadata["stable_metadata_sha256"] = semantic_sha256(metadata)
    return metadata


def _stable_unit_metadata(
    stable_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if stable_metadata is None:
        return {}
    return {
        "stable_change_id": str(stable_metadata["stable_change_id"]),
        "archived_protocol_change_id": str(
            stable_metadata["archived_protocol_change_id"]
        ),
        "selector_family": str(stable_metadata["selector_family"]),
        "signature_config": deepcopy(dict(stable_metadata["signature_config"])),
        "signature_config_sha256": str(stable_metadata["signature_config_sha256"]),
        "proxy_config": deepcopy(dict(stable_metadata["proxy_config"])),
        "proxy_config_sha256": str(stable_metadata["proxy_config_sha256"]),
        "leakage_validation": deepcopy(dict(stable_metadata["leakage_validation"])),
        "report_tags": list(stable_metadata["report_tags"]),
        "target_random_candidate": deepcopy(
            dict(stable_metadata["target_random_candidate"])
        ),
        "authority_baseline": deepcopy(dict(stable_metadata["authority_baseline"])),
    }


def build_baseline_replay_manifest(
    *,
    run_id: str,
    output_root: str,
    split_identity: Mapping[str, Any],
) -> dict[str, Any]:
    base = _manifest_base(
        run_id=run_id,
        output_root=output_root,
        split_identity=split_identity,
        phase="baseline_replay",
    )
    selector_config: dict[str, Any] = {}
    selector_config_sha256 = semantic_sha256(
        {
            "selector_id": "uncertainty_boundary",
            "selector_config": selector_config,
        }
    )
    units = []
    for seed in ACTIVE_LEARNING_SEEDS:
        units.append(
            {
                "unit_id": "baseline.rcabench.uncertainty_boundary.seed%d" % seed,
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "method_id": "in_change_baseline_uncertainty_boundary",
                "budget": 30,
                "selector_id": "uncertainty_boundary",
                "selector_config": deepcopy(selector_config),
                "selector_config_sha256": selector_config_sha256,
                "active_learning_seed": seed,
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "normal_policy": "fault_only",
                "evaluation_views": ["T1", "T2"],
                "output_root": base["output_root"] + "/units/baseline.seed%d" % seed,
            }
        )
    manifest = {
        **base,
        "budget": 30,
        "selector_id": "uncertainty_boundary",
        "selector_config": selector_config,
        "selector_config_sha256": selector_config_sha256,
        "active_learning_seeds": list(ACTIVE_LEARNING_SEEDS),
        "normal_policy": "fault_only",
        "units": units,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    return manifest


def build_random_search_manifest(
    *,
    run_id: str,
    output_root: str,
    split_identity: Mapping[str, Any],
    round_size: int,
    sampling_seed: int,
    random_early_stop_thresholds: Mapping[str, float] = RANDOM_EARLY_STOP_THRESHOLDS,
) -> dict[str, Any]:
    if int(round_size) not in (10, 20, 40, 80, 160):
        raise ValueError("unsupported random-search round_size")
    random_thresholds = _validated_random_early_stop_thresholds(
        random_early_stop_thresholds
    )
    base = _manifest_base(
        run_id=run_id,
        output_root=output_root,
        split_identity=split_identity,
        phase="random_diagnostic_search",
    )
    train_ids = list(split_identity["outer_train_case_ids"])
    if len(train_ids) < 30:
        raise ValueError("random search requires at least 30 outer_train cases")
    rng = random.Random(int(sampling_seed))
    units = []
    for index in range(int(round_size)):
        selected = sorted(rng.sample(train_ids, 30))
        sample_identity = {
            "selected_case_ids": selected,
            "round_size": int(round_size),
            "sampling_seed": int(sampling_seed),
            "index": index,
        }
        sample_hash = semantic_sha256(sample_identity)
        units.append(
            {
                "unit_id": "random.rcabench.round%d.set%04d" % (round_size, index),
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "method_id": "random_budget30_diagnostic",
                "budget": 30,
                "selector_id": "fixed_random_budget_set",
                "selector_config": {
                    "sample_set_hash": sample_hash,
                    "sampling_seed": int(sampling_seed),
                    "random_set_index": index,
                },
                "selector_config_sha256": semantic_sha256(
                    {
                        "selector_id": "fixed_random_budget_set",
                        "selected_case_ids": selected,
                        "sample_set_hash": sample_hash,
                    }
                ),
                "selected_case_ids": selected,
                "sample_set_hash": sample_hash,
                "sampling_seed": int(sampling_seed),
                "random_set_index": index,
                "evaluation_views": ["T1", "T2"],
                "output_root": base["output_root"] + "/units/" + sample_hash[:12],
            }
        )
    manifest = {
        **base,
        "round_size": int(round_size),
        "sampling_seed": int(sampling_seed),
        "random_early_stop_thresholds": random_thresholds,
        "units": units,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    return manifest


def select_random_diagnostic_result(
    results: Sequence[Mapping[str, Any]],
    *,
    random_early_stop_thresholds: Mapping[str, float] = RANDOM_EARLY_STOP_THRESHOLDS,
    max_attempts_exhausted: bool = False,
) -> dict[str, Any]:
    random_thresholds = _validated_random_early_stop_thresholds(
        random_early_stop_thresholds
    )
    rows = [deepcopy(dict(result)) for result in results]
    if not rows:
        raise ValueError("random diagnostic results must not be empty")
    hit_only_winners: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for row in rows:
        metrics = dict(row.get("t1_metrics") or {})
        hit_gate = evaluate_t1_sota_gate(metrics)
        random_gate = evaluate_t1_sota_gate(
            metrics,
            thresholds=random_thresholds,
            required_metrics=RANDOM_EARLY_STOP_METRICS,
        )
        row["t1_sota_gate"] = hit_gate
        row["t1_sota_pass"] = bool(hit_gate["passed"])
        row["random_early_stop_gate"] = random_gate
        if bool(random_gate["passed"]):
            winners.append(row)
        elif bool(hit_gate["passed"]):
            hit_only_winners.append(row)
    if winners:
        selected = max(winners, key=_random_diagnostic_sort_key)
        return {
            "decision": "early_stop_strict_t1_hit_and_mrr_sota",
            "selected_result": selected,
            "rejected_hit_only_winners": sorted(
                hit_only_winners,
                key=lambda row: str(row.get("unit_id", "")),
            ),
        }

    selected = max(rows, key=_random_diagnostic_sort_key)
    if not bool(max_attempts_exhausted):
        return {
            "decision": "continue_staged_random_search",
            "best_so_far": selected,
            "rejected_hit_only_winners": sorted(
                hit_only_winners,
                key=lambda row: str(row.get("unit_id", "")),
            ),
        }
    return {
        "decision": "fallback_best_optimized_by_t1_mrr",
        "selected_result": selected,
        "rejected_hit_only_winners": sorted(
            hit_only_winners,
            key=lambda row: str(row.get("unit_id", "")),
        ),
    }


def _random_diagnostic_sort_key(
    row: Mapping[str, Any],
) -> tuple[float, float, float, float, str]:
    metrics = dict(row.get("t1_metrics") or {})
    return (
        float(metrics.get("mrr", 0.0)),
        float(metrics.get("hit_at_3", 0.0)) - SOTA_THRESHOLDS["hit_at_3"],
        float(metrics.get("hit_at_5", 0.0)) - SOTA_THRESHOLDS["hit_at_5"],
        float(metrics.get("hit_at_1", 0.0)) - SOTA_THRESHOLDS["hit_at_1"],
        str(row.get("sample_set_hash", "")),
    )


def build_strategy_validation_manifest(
    *,
    run_id: str,
    output_root: str,
    split_identity: Mapping[str, Any],
    selector_id: str,
    selector_config: Mapping[str, Any],
    budget: int = 30,
) -> dict[str, Any]:
    base = _manifest_base(
        run_id=run_id,
        output_root=output_root,
        split_identity=split_identity,
        phase="strategy_validation",
    )
    selector_hash = semantic_sha256(
        {"selector_id": str(selector_id), "selector_config": dict(selector_config)}
    )
    stable_metadata = _stable_selector_metadata(
        selector_id=str(selector_id),
        selector_config=selector_config,
    )
    stable_unit_fields = _stable_unit_metadata(stable_metadata)
    units = []
    for seed in ACTIVE_LEARNING_SEEDS:
        unit = {
            "unit_id": "strategy.rcabench.%s.seed%d.budget%d"
            % (selector_id, seed, int(budget)),
            "canonical_dataset_id": CANONICAL_DATASET_ID,
            "method_id": "strategy_validation_%s" % selector_id,
            "budget": int(budget),
            "selector_id": str(selector_id),
            "selector_config": deepcopy(dict(selector_config)),
            "selector_config_sha256": selector_hash,
            "active_learning_seed": seed,
            "seed": seed,
            "split_seed": SPLIT_SEED,
            "evaluation_views": ["T1", "T2"],
            "execution_mode": base["execution_mode"],
            "output_root": base["output_root"] + "/units/seed%d" % seed,
        }
        unit.update(deepcopy(stable_unit_fields))
        units.append(unit)
    manifest = {
        **base,
        "budget": int(budget),
        "selector_id": str(selector_id),
        "selector_config": deepcopy(dict(selector_config)),
        "selector_config_sha256": selector_hash,
        "active_learning_seeds": list(ACTIVE_LEARNING_SEEDS),
        "units": units,
    }
    if stable_metadata is not None:
        manifest["stable_selection_metadata"] = stable_metadata
        manifest["report_tags"] = list(STABLE_REPORT_TAGS)
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    return manifest


def _stable_selector_config_rows(
    selector_configs: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_configs = selector_configs or DEFAULT_STABLE_SELECTOR_CONFIGS
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_configs:
        config_id = str(raw.get("config_id", "")).strip()
        if not config_id:
            raise ValueError("stable selector config requires config_id")
        if config_id in seen_ids:
            raise ValueError("duplicate stable selector config_id: %s" % config_id)
        seen_ids.add(config_id)
        selector_id = str(raw.get("selector_id", "")).strip()
        selector_config = deepcopy(dict(raw.get("selector_config") or {}))
        stable_metadata = _stable_selector_metadata(
            selector_id=selector_id,
            selector_config=selector_config,
        )
        if stable_metadata is None:
            raise ValueError("stable selector suite rejects selector: %s" % selector_id)
        selector_config_sha256 = semantic_sha256(
            {"selector_id": selector_id, "selector_config": selector_config}
        )
        rows.append(
            {
                "config_id": config_id,
                "selector_id": selector_id,
                "selector_family": stable_metadata["selector_family"],
                "selector_config": selector_config,
                "selector_config_sha256": selector_config_sha256,
                "signature_config": deepcopy(dict(stable_metadata["signature_config"])),
                "signature_config_sha256": stable_metadata["signature_config_sha256"],
                "proxy_config": deepcopy(dict(stable_metadata["proxy_config"])),
                "proxy_config_sha256": stable_metadata["proxy_config_sha256"],
                "stable_selection_metadata": stable_metadata,
            }
        )
    if not rows:
        raise ValueError("stable selector suite requires at least one config")
    return rows


def _suite_metadata(config_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["selector_family"]) for row in config_rows})
    metadata = {
        "schema_version": "rcabench-stable-query-selection-suite-metadata-v1",
        "stable_change_id": STABLE_SELECTION_CHANGE_ID,
        "archived_protocol_change_id": CHANGE_ID,
        "strategy_suite_id": STABLE_STRATEGY_SUITE_ID,
        "selector_families": families,
        "report_tags": list(STABLE_REPORT_TAGS),
        "archived_protocol_references": deepcopy(ARCHIVED_PROTOCOL_REFERENCES),
        "authority_baseline": deepcopy(AUTHORITY_BASELINE_METRICS),
        "target_random_candidate": deepcopy(TARGET_RANDOM_CANDIDATE_METRICS),
        "failed_transfer": deepcopy(FAILED_TRANSFER_METRICS),
    }
    metadata["suite_metadata_sha256"] = semantic_sha256(metadata)
    return metadata


def build_stable_strategy_suite_manifest(
    *,
    run_id: str,
    output_root: str,
    split_identity: Mapping[str, Any],
    budget: int = 30,
    selector_configs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    base = _manifest_base(
        run_id=run_id,
        output_root=output_root,
        split_identity=split_identity,
        phase="stable_strategy_validation",
    )
    config_rows = _stable_selector_config_rows(selector_configs)
    suite_metadata = _suite_metadata(config_rows)
    units = []
    for config_row in config_rows:
        stable_unit_fields = _stable_unit_metadata(
            config_row["stable_selection_metadata"]
        )
        for seed in ACTIVE_LEARNING_SEEDS:
            unit = {
                "unit_id": "strategy.rcabench.%s.seed%d.budget%d"
                % (config_row["config_id"], seed, int(budget)),
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "method_id": "strategy_validation_%s" % config_row["config_id"],
                "stable_config_id": str(config_row["config_id"]),
                "budget": int(budget),
                "selector_id": str(config_row["selector_id"]),
                "selector_config": deepcopy(dict(config_row["selector_config"])),
                "selector_config_sha256": str(config_row["selector_config_sha256"]),
                "active_learning_seed": seed,
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "evaluation_views": ["T1", "T2"],
                "execution_mode": base["execution_mode"],
                "output_root": (
                    base["output_root"]
                    + "/units/%s.seed%d" % (config_row["config_id"], seed)
                ),
            }
            unit.update(deepcopy(stable_unit_fields))
            units.append(unit)
    frozen_configs = []
    for row in config_rows:
        frozen = {
            key: deepcopy(value)
            for key, value in dict(row).items()
            if key != "stable_selection_metadata"
        }
        frozen_configs.append(frozen)
    manifest = {
        **base,
        "phase": "stable_strategy_validation",
        "strategy_suite_id": STABLE_STRATEGY_SUITE_ID,
        "budget": int(budget),
        "selector_id": "stable_selector_suite",
        "selector_config": {},
        "selector_config_sha256": semantic_sha256(frozen_configs),
        "active_learning_seeds": list(ACTIVE_LEARNING_SEEDS),
        "stable_selector_configs": frozen_configs,
        "stable_selection_metadata": suite_metadata,
        "report_tags": list(STABLE_REPORT_TAGS),
        "units": units,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    return manifest


def build_budget_grid_manifest(
    *,
    run_id: str,
    output_root: str,
    split_identity: Mapping[str, Any],
    selector_id: str,
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
    base = _manifest_base(
        run_id=run_id,
        output_root=output_root,
        split_identity=split_identity,
        phase="budget_escalation",
    )
    selector_hash = semantic_sha256(
        {"selector_id": str(selector_id), "selector_config": dict(selector_config)}
    )
    stable_metadata = _stable_selector_metadata(
        selector_id=str(selector_id),
        selector_config=selector_config,
    )
    stable_unit_fields = _stable_unit_metadata(stable_metadata)
    units = []
    for budget in BUDGET_GRID:
        for seed in ACTIVE_LEARNING_SEEDS:
            unit = {
                "unit_id": "budget.rcabench.%s.budget%d.seed%d"
                % (selector_id, budget, seed),
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "method_id": "budget_escalation_%s" % selector_id,
                "budget": int(budget),
                "selector_id": str(selector_id),
                "selector_config": deepcopy(dict(selector_config)),
                "selector_config_sha256": selector_hash,
                "active_learning_seed": seed,
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "evaluation_views": ["T1", "T2"],
                "execution_mode": base["execution_mode"],
                "output_root": base["output_root"]
                + "/units/budget%d.seed%d" % (budget, seed),
            }
            unit.update(deepcopy(stable_unit_fields))
            units.append(unit)
    manifest = {
        **base,
        "budgets": list(BUDGET_GRID),
        "selector_id": str(selector_id),
        "selector_config": deepcopy(dict(selector_config)),
        "selector_config_sha256": selector_hash,
        "active_learning_seeds": list(ACTIVE_LEARNING_SEEDS),
        "units": units,
    }
    if stable_metadata is not None:
        manifest["stable_selection_metadata"] = stable_metadata
        manifest["report_tags"] = list(STABLE_REPORT_TAGS)
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    return manifest


def select_usable_gpus(
    gpu_rows: Sequence[Mapping[str, Any]],
    *,
    max_gpus: int = 24,
    max_memory_used_mib: int = 1024,
    max_utilization_pct: int = 10,
) -> dict[str, Any]:
    usable = []
    for row in gpu_rows:
        memory = int(row.get("memory_used_mib", 0))
        utilization = int(row.get("utilization_pct", 0))
        if memory <= max_memory_used_mib and utilization <= max_utilization_pct:
            usable.append(int(row["gpu_id"]))
    selected = usable[: int(max_gpus)]
    if not selected:
        raise ValueError("no usable GPU available for GPU-required full experiment")
    return {
        "declared_gpu_cap": int(max_gpus),
        "selected_gpu_count": len(selected),
        "selected_gpu_ids": selected,
        "scheduling_policy": "auto_degrade_to_usable_gpus",
    }


def build_progress_payload(
    *,
    phase: str,
    completed: int,
    total: int,
    current_round_size: int | None = None,
    current_budget: int | None = None,
    current_active_learning_seed: int | None = None,
    failures: int = 0,
    recent_log_path: str = "",
    recent_unit: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "rcabench-query-only-sota-progress-v1",
        "phase": str(phase),
        "completed": int(completed),
        "total": int(total),
        "current_round_size": current_round_size,
        "current_budget": current_budget,
        "current_active_learning_seed": current_active_learning_seed,
        "recent_unit": str(recent_unit),
        "failures": int(failures),
        "recent_log_path": str(recent_log_path),
        "updated_at_utc": _utc_now(),
    }


def format_tqdm_progress_line(progress: Mapping[str, Any], *, width: int = 20) -> str:
    completed = int(progress.get("completed", 0))
    total = int(progress.get("total", 0))
    fraction = 0.0 if total <= 0 else max(0.0, min(1.0, completed / float(total)))
    filled = int(round(fraction * int(width)))
    bar = "=" * filled + "." * (int(width) - filled)
    percent = fraction * 100.0
    return (
        "[{bar}] {completed}/{total} {percent:.1f}% phase={phase} "
        "round={round_size} budget={budget} seed={seed} unit={unit} failures={failures}"
    ).format(
        bar=bar,
        completed=completed,
        total=total,
        percent=percent,
        phase=progress.get("phase", ""),
        round_size=progress.get("current_round_size", ""),
        budget=progress.get("current_budget", ""),
        seed=progress.get("current_active_learning_seed", ""),
        unit=progress.get("recent_unit", ""),
        failures=progress.get("failures", 0),
    )


def write_completion_markers(
    run_root: Path | str,
    completion_payload: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(dict(completion_payload))
    payload.setdefault("completed_at_utc", _utc_now())
    completed_path = root / "COMPLETED.json"
    all_done_path = root / "all.done"
    _write_json_atomic(completed_path, payload)
    completed_sha = sha256_file(completed_path)
    all_done_path.write_text(completed_sha + "\n", encoding="utf-8")
    return {
        "completed_json": str(completed_path),
        "all_done": str(all_done_path),
        "completed_sha256": completed_sha,
    }


def write_failure_marker(
    run_root: Path | str,
    exc: BaseException,
) -> dict[str, Any]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rcabench-query-only-sota-failure-v1",
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "failed_at_utc": _utc_now(),
    }
    _write_json_atomic(root / "pipeline.failed.json", payload)
    (root / ".failed").write_text(
        "%s: %s\n" % (type(exc).__name__, exc),
        encoding="utf-8",
    )
    return payload


def build_status_snapshot(
    *,
    run_root: Path | str,
    tmux_session: str,
    tmux_running: bool,
    selected_gpu_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    progress_path = root / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "phase": "missing",
            "completed": 0,
            "total": 0,
            "recent_log_path": str(root / "logs" / "run.log"),
        }
    completed_path = root / "COMPLETED.json"
    all_done_path = root / "all.done"
    failure_json_path = root / "pipeline.failed.json"
    failed_marker_path = root / ".failed"
    if completed_path.is_file() and all_done_path.is_file():
        completion = "complete"
    elif failure_json_path.is_file() or failed_marker_path.is_file():
        completion = "failed"
    else:
        completion = "pending"
    return {
        "schema_version": "rcabench-query-only-sota-status-v1",
        "tmux_session": str(tmux_session),
        "tmux": "running" if tmux_running else "not-running",
        "selected_gpu_ids": [int(value) for value in (selected_gpu_ids or [])],
        "progress": progress,
        "completion": completion,
        "paths": {
            "run_root": str(root),
            "completed_json": str(completed_path),
            "all_done": str(all_done_path),
            "failure_json": str(failure_json_path),
            "failed_marker": str(failed_marker_path),
            "recent_log": str(progress.get("recent_log_path", root / "logs" / "run.log")),
        },
        "updated_at_utc": _utc_now(),
    }


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _safe_unit_path(unit_id: str) -> str:
    return "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in str(unit_id)
    )


def _write_progress(
    run_root: Path,
    progress: Mapping[str, Any],
) -> None:
    _write_json_atomic(run_root / "progress.json", progress)
    print(format_tqdm_progress_line(progress), flush=True)


def _unit_seed(result: Mapping[str, Any]) -> int:
    if "active_learning_seed" in result:
        return int(result["active_learning_seed"])
    return int(result.get("seed", 42))


def _copy_unit_metadata(
    result: dict[str, Any],
    unit: Mapping[str, Any],
) -> None:
    for key in (
        "stable_change_id",
        "archived_protocol_change_id",
        "selector_family",
        "signature_config",
        "signature_config_sha256",
        "proxy_config",
        "proxy_config_sha256",
        "leakage_validation",
        "report_tags",
        "target_random_candidate",
        "authority_baseline",
        "execution_mode",
    ):
        if key in unit and key not in result:
            result[key] = deepcopy(unit[key])


def _normalize_sota_unit_result(
    unit: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(raw_result))
    if str(result.get("unit_id")) != str(unit.get("unit_id")):
        raise ValueError("unit result ID mismatch")
    _require_rcabench(str(result.get("canonical_dataset_id")))
    for view_key in ("t1_metrics", "t2_metrics"):
        if not isinstance(result.get(view_key), Mapping):
            raise ValueError("unit result missing %s" % view_key)
        metrics = dict(result[view_key])
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr"):
            if metric not in metrics:
                raise ValueError("%s missing %s" % (view_key, metric))
            metrics[metric] = float(metrics[metric])
        denominator = int(
            metrics.get("denominator", metrics.get("evaluation_fault_cases", 0))
        )
        if denominator <= 0:
            raise ValueError("%s denominator must be positive" % view_key)
        metrics["denominator"] = denominator
        result[view_key] = metrics
    selected = [str(value) for value in result.get("selected_case_ids") or []]
    budget = int(unit.get("budget", result.get("budget", 0)))
    if len(selected) != budget or len(set(selected)) != budget:
        raise ValueError("unit result selected_case_ids must match budget")
    result["budget"] = budget
    result["selected_case_ids"] = selected
    result["active_learning_seed"] = _unit_seed(result)
    result.setdefault("selector_id", str(unit.get("selector_id", "")))
    result.setdefault(
        "selector_config_sha256",
        str(unit.get("selector_config_sha256", "")),
    )
    for identity_key in ("sample_set_hash", "sampling_seed", "random_set_index"):
        if identity_key in unit and result.get(identity_key) is None:
            result[identity_key] = unit[identity_key]
    _copy_unit_metadata(result, unit)
    result.setdefault("query_plan_sha256", "")
    result.setdefault("selector_diagnostics", {})
    result["t1_sota_gate"] = evaluate_t1_sota_gate(result["t1_metrics"])
    result["t1_t2_warnings"] = stability_warnings(
        t1_metrics=result["t1_metrics"],
        t2_metrics=result["t2_metrics"],
    )
    result["schema_version"] = "rcabench-query-only-sota-unit-result-v1"
    return result


def _aggregate_sota_results(
    unit_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not unit_results:
        raise ValueError("unit_results must not be empty")
    aggregates = {}
    for view_name, key in (("T1", "t1_metrics"), ("T2", "t2_metrics")):
        rows = [dict(result[key]) for result in unit_results]
        view = {
            "seed_count": len(rows),
            "total_denominator": sum(int(row["denominator"]) for row in rows),
        }
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr"):
            view["mean_%s" % metric] = sum(
                float(row[metric]) for row in rows
            ) / float(len(rows))
        aggregates[view_name] = view
    return aggregates


def _target_candidate_distance(
    aggregates: Mapping[str, Any],
    target_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    distances: dict[str, Any] = {
        "schema_version": "rcabench-target-random-candidate-distance-v1",
        "target_unit_id": str(target_candidate.get("unit_id", "")),
    }
    for view_name, aggregate in sorted(dict(aggregates).items()):
        target_metrics = target_candidate.get(view_name)
        if not isinstance(target_metrics, Mapping):
            continue
        view_distance = {}
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr"):
            mean_key = "mean_%s" % metric
            if mean_key not in aggregate or metric not in target_metrics:
                continue
            actual = float(aggregate[mean_key])
            target = float(target_metrics[metric])
            view_distance[metric] = {
                "actual": actual,
                "target": target,
                "gap": actual - target,
                "absolute_gap": abs(actual - target),
            }
        if view_distance:
            distances[view_name] = view_distance
    return distances


def _proxy_mode_coverage_summary(
    unit_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    by_unit = {}
    top_counts = []
    hhi_values = []
    for result in unit_results:
        diagnostics = result.get("selector_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        proxy = diagnostics.get("proxy_modes")
        if not isinstance(proxy, Mapping) or not proxy:
            continue
        unit_id = str(result["unit_id"])
        row = deepcopy(dict(proxy))
        by_unit[unit_id] = row
        if "top_proxy_mode_count" in row:
            top_counts.append(float(row["top_proxy_mode_count"]))
        if "proxy_mode_hhi" in row:
            hhi_values.append(float(row["proxy_mode_hhi"]))
    if not by_unit:
        return None
    summary = {
        "schema_version": "rcabench-stable-query-proxy-coverage-summary-v1",
        "seed_count": len(by_unit),
        "by_unit": by_unit,
    }
    if top_counts:
        summary["mean_top_proxy_mode_count"] = sum(top_counts) / float(len(top_counts))
    if hhi_values:
        summary["mean_proxy_mode_hhi"] = sum(hhi_values) / float(len(hhi_values))
    return summary


def execute_rcabench_query_only_sota(
    manifest: Mapping[str, Any],
    unit_executor: SotaUnitExecutor,
) -> dict[str, Any]:
    """Execute one RCABench query-only SOTA-search manifest."""

    if manifest.get("schema_version") != "rcabench-query-only-sota-manifest-v1":
        raise ValueError("unexpected rcabench query-only SOTA manifest schema")
    _require_rcabench(str(manifest.get("canonical_dataset_id")))
    phase = str(manifest.get("phase", ""))
    run_root = Path(str(manifest["output_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    units = list(manifest.get("units") or [])
    if not units:
        raise ValueError("manifest contains no units")
    unit_results: list[dict[str, Any]] = []
    try:
        _remove_if_present(run_root / "COMPLETED.json")
        _remove_if_present(run_root / "all.done")
        progress = build_progress_payload(
            phase=phase,
            completed=0,
            total=len(units),
            failures=0,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        _write_progress(run_root, progress)
        for index, unit in enumerate(units, start=1):
            unit_id = str(unit["unit_id"])
            progress = build_progress_payload(
                phase=phase,
                completed=index - 1,
                total=len(units),
                current_round_size=unit.get("round_size")
                or manifest.get("round_size"),
                current_budget=unit.get("budget"),
                current_active_learning_seed=unit.get("active_learning_seed"),
                failures=0,
                recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
                recent_unit=unit_id,
            )
            _write_progress(run_root, progress)
            raw_result = unit_executor(unit, run_root)
            result = _normalize_sota_unit_result(unit, raw_result)
            result_path = run_root / "unit_results" / _safe_unit_path(unit_id) / "result.json"
            _write_json_atomic(result_path, result)
            unit_results.append(result)
            progress = build_progress_payload(
                phase=phase,
                completed=index,
                total=len(units),
                current_round_size=unit.get("round_size")
                or manifest.get("round_size"),
                current_budget=unit.get("budget"),
                current_active_learning_seed=unit.get("active_learning_seed"),
                failures=0,
                recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
                recent_unit=unit_id,
            )
            _write_progress(run_root, progress)
        aggregates = _aggregate_sota_results(unit_results)
        strict_gate = evaluate_t1_sota_gate(
            {
                metric: aggregates["T1"]["mean_%s" % metric]
                for metric in HIT_METRICS
            }
        )
        mean_warnings = stability_warnings(
            t1_metrics={
                metric: aggregates["T1"]["mean_%s" % metric]
                for metric in HIT_METRICS
            },
            t2_metrics={
                metric: aggregates["T2"]["mean_%s" % metric]
                for metric in HIT_METRICS
            },
        )
        summary = {
            "schema_version": "rcabench-query-only-sota-summary-v1",
            "change_id": CHANGE_ID,
            "run_id": manifest.get("run_id"),
            "phase": phase,
            "execution_mode": manifest.get(
                "execution_mode",
                "outer_test_guided_engineering_search",
            ),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "unit_count": len(units),
            "split_seed": manifest.get("split_seed", SPLIT_SEED),
            "split_identity_sha256": manifest.get("split_identity_sha256"),
            "active_learning_seeds": [
                _unit_seed(result)
                for result in sorted(
                    unit_results,
                    key=lambda row: _unit_seed(row),
                )
            ],
            "aggregates": aggregates,
            "strict_t1_sota_gate": strict_gate,
            "t1_t2_warnings": mean_warnings,
            "unit_results": {
                str(result["unit_id"]): result
                for result in sorted(
                    unit_results,
                    key=lambda row: str(row["unit_id"]),
                )
            },
            "completed_at_utc": _utc_now(),
        }
        stable_metadata = manifest.get("stable_selection_metadata")
        if isinstance(stable_metadata, Mapping):
            target_candidate = dict(stable_metadata["target_random_candidate"])
            summary["stable_selection_metadata"] = deepcopy(dict(stable_metadata))
            summary["report_tags"] = list(stable_metadata.get("report_tags") or [])
            summary["archived_protocol_references"] = deepcopy(
                dict(stable_metadata["archived_protocol_references"])
            )
            summary["authority_baseline"] = deepcopy(
                dict(stable_metadata["authority_baseline"])
            )
            summary["target_random_candidate"] = deepcopy(target_candidate)
            summary["failed_transfer"] = deepcopy(dict(stable_metadata["failed_transfer"]))
            summary["distance_to_target_random_candidate"] = _target_candidate_distance(
                aggregates,
                target_candidate,
            )
        proxy_summary = _proxy_mode_coverage_summary(unit_results)
        if proxy_summary is not None:
            summary["proxy_mode_coverage_diagnostics"] = proxy_summary
        summary_name = (
            "baseline_replay_summary.json"
            if phase == "baseline_replay"
            else "%s_summary.json" % phase
        )
        _write_json_atomic(run_root / summary_name, summary)
        completion = {
            "schema_version": "rcabench-query-only-sota-completion-v1",
            "status": "complete",
            "phase": phase,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "summary_path": str((run_root / summary_name).resolve()),
            "summary_sha256": sha256_file(run_root / summary_name),
            "unit_count": len(units),
        }
        markers = write_completion_markers(run_root, completion)
        complete_progress = build_progress_payload(
            phase="complete",
            completed=len(units),
            total=len(units),
            failures=0,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
            recent_unit=str(units[-1]["unit_id"]),
        )
        _write_progress(run_root, complete_progress)
        return {**completion, **markers}
    except Exception as exc:
        _remove_if_present(run_root / "COMPLETED.json")
        _remove_if_present(run_root / "all.done")
        write_failure_marker(run_root, exc)
        failed_progress = build_progress_payload(
            phase="failed",
            completed=len(unit_results),
            total=len(units),
            failures=1,
            recent_log_path=str(run_root / "logs" / "tmux-runner.log"),
        )
        _write_progress(run_root, failed_progress)
        raise


__all__ = [
    "ACTIVE_LEARNING_SEEDS",
    "BUDGET_GRID",
    "CANONICAL_DATASET_ID",
    "CHANGE_ID",
    "HIT_METRICS",
    "RCABENCH_BASELINE_MRR_SCORES",
    "RCABENCH_BASELINE_MRR_SOTA_THRESHOLD",
    "RANDOM_EARLY_STOP_METRICS",
    "RANDOM_EARLY_STOP_THRESHOLDS",
    "RCABENCH_SOURCE_PATH",
    "REMOTE_WORKSPACE",
    "SPLIT_SEED",
    "SOTA_THRESHOLDS",
    "STABLE_SELECTION_CHANGE_ID",
    "STABLE_STRATEGY_SUITE_ID",
    "BudgetAnnotationSimulator",
    "build_baseline_replay_manifest",
    "build_budget_grid_manifest",
    "build_progress_payload",
    "format_tqdm_progress_line",
    "build_random_search_manifest",
    "build_score_partitions",
    "build_split_identity",
    "build_stable_strategy_suite_manifest",
    "build_status_snapshot",
    "build_strategy_validation_manifest",
    "compute_ranking_metrics",
    "execute_rcabench_query_only_sota",
    "evaluate_t1_sota_gate",
    "freeze_budget_query_plan",
    "select_random_diagnostic_result",
    "select_usable_gpus",
    "semantic_sha256",
    "sha256_file",
    "score_t1_t2_from_rankings",
    "stability_warnings",
    "validate_training_rows_no_unqueried_labels",
    "write_completion_markers",
    "write_failure_marker",
]
