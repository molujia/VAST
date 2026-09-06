"""Count-authoritative scoring, stability, and immutable trial lineage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Dict


SCORING_SCHEMA_VERSION = "ordinary-strategy-count-score-v1"
AGGREGATE_SCHEMA_VERSION = "ordinary-strategy-aggregate-v1"
TRIAL_LEDGER_SCHEMA_VERSION = "ordinary-strategy-trial-ledger-v1"
FINAL_CONFIG_SCHEMA_VERSION = "ordinary-strategy-final-config-v1"
METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")
HIT_METRICS = METRICS[:3]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("%s must be a canonical string" % field)
    return value


def _first_target_rank(ranking: Sequence[str], targets: set[str]) -> int:
    for index, candidate in enumerate(ranking, start=1):
        if candidate in targets:
            return index
    return 0


def score_ranking_rows(
    ranking_rows: Any,
    *,
    expected_case_ids: Sequence[Any],
) -> Dict[str, Any]:
    """Build Hit counts and the exact MRR numerator from per-case rankings."""

    if isinstance(ranking_rows, (str, bytes)) or not isinstance(ranking_rows, Sequence):
        raise ValueError("ranking rows must be an ordered sequence")
    expected = [str(value).strip() for value in expected_case_ids]
    if not expected or any(not value for value in expected) or len(expected) != len(set(expected)):
        raise ValueError("expected case IDs must be non-empty and unique")
    by_case: Dict[str, Mapping[str, Any]] = {}
    for source in ranking_rows:
        if not isinstance(source, Mapping):
            raise ValueError("ranking row must be a mapping")
        if set(source) != {"case_id", "targets", "ranking"}:
            raise ValueError("ranking row contains undeclared fields")
        case_id = _canonical_text(source["case_id"], "case_id")
        if case_id in by_case:
            raise ValueError("ranking case IDs must be unique")
        by_case[case_id] = source
    if set(by_case) != set(expected):
        raise ValueError("ranking case IDs must exactly match expected case IDs")

    hit_counts = {metric: 0 for metric in HIT_METRICS}
    mrr_numerator = 0.0
    per_case = []
    for case_id in expected:
        source = by_case[case_id]
        raw_targets = source["targets"]
        raw_ranking = source["ranking"]
        if (
            isinstance(raw_targets, (str, bytes))
            or not isinstance(raw_targets, Sequence)
            or isinstance(raw_ranking, (str, bytes))
            or not isinstance(raw_ranking, Sequence)
        ):
            raise ValueError("targets and ranking must be sequences")
        targets = [str(value).strip() for value in raw_targets]
        ranking = [str(value).strip() for value in raw_ranking]
        if not targets or any(not value for value in targets) or any(not value for value in ranking):
            raise ValueError("targets and ranking entries must be non-empty")
        rank = _first_target_rank(ranking, set(targets))
        reciprocal_rank = 0.0 if rank == 0 else 1.0 / float(rank)
        mrr_numerator += reciprocal_rank
        for limit, metric in ((1, "hit_at_1"), (3, "hit_at_3"), (5, "hit_at_5")):
            if rank and rank <= limit:
                hit_counts[metric] += 1
        per_case.append(
            {
                "case_id": case_id,
                "targets": targets,
                "ranking": ranking,
                "first_target_rank": rank,
                "reciprocal_rank": reciprocal_rank,
            }
        )
    denominator = len(expected)
    metrics = {
        metric: hit_counts[metric] / float(denominator) for metric in HIT_METRICS
    }
    metrics["mrr"] = mrr_numerator / float(denominator)
    result = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "denominator": denominator,
        "hit_counts": hit_counts,
        "mrr_numerator": mrr_numerator,
        "metrics": metrics,
        "case_ids": expected,
        "per_case": per_case,
    }
    result["score_sha256"] = _semantic_sha256(result)
    return result


def validate_count_authoritative_view(view: Any) -> Dict[str, Any]:
    if not isinstance(view, Mapping):
        raise ValueError("count-authoritative view must be a mapping")
    result = deepcopy(dict(view))
    denominator = result.get("denominator")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("count-authoritative denominator must be positive")
    hit_counts = result.get("hit_counts")
    metrics = result.get("metrics")
    if not isinstance(hit_counts, Mapping) or set(hit_counts) != set(HIT_METRICS):
        raise ValueError("count-authoritative hit counts are incomplete")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        raise ValueError("count-authoritative metrics are incomplete")
    reconstructed = {}
    for metric in HIT_METRICS:
        count = hit_counts[metric]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= denominator:
            raise ValueError("count-authoritative hit count is invalid")
        reconstructed[metric] = count / float(denominator)
    numerator = result.get("mrr_numerator")
    if isinstance(numerator, bool) or not isinstance(numerator, (int, float)) or not math.isfinite(float(numerator)):
        raise ValueError("count-authoritative MRR numerator is invalid")
    if not 0.0 <= float(numerator) <= float(denominator):
        raise ValueError("count-authoritative MRR numerator is out of range")
    reconstructed["mrr"] = float(numerator) / float(denominator)
    for metric, expected in reconstructed.items():
        actual = metrics[metric]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
            raise ValueError("reported score does not reconstruct from stored counts")
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reported score does not reconstruct from stored counts")
    declared_sha = result.pop("score_sha256", None)
    if declared_sha is not None and declared_sha != _semantic_sha256(result):
        raise ValueError("count-authoritative score SHA does not reconstruct")
    if declared_sha is not None:
        result["score_sha256"] = declared_sha
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _mean_view(rows: Sequence[Mapping[str, Any]], view_name: str) -> Dict[str, float]:
    values = [validate_count_authoritative_view(row["views"][view_name])["metrics"] for row in rows]
    return {metric: _mean([float(value[metric]) for value in values]) for metric in METRICS}


def _query_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    type_union = sorted(
        {
            str(value)
            for row in rows
            for value in row.get("queried_fault_types", ())
            if str(value)
        }
    )
    first_rounds: Dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for fault_type, round_index in dict(row.get("first_round_by_fault_type") or {}).items():
            first_rounds[str(fault_type)].append(int(round_index))
    query_sets = [set(str(value) for value in row.get("queried_case_ids", ())) for row in rows]
    overlaps = []
    for left in range(len(query_sets)):
        for right in range(left + 1, len(query_sets)):
            union = query_sets[left] | query_sets[right]
            overlaps.append(1.0 if not union else len(query_sets[left] & query_sets[right]) / float(len(union)))
    return {
        "queried_fault_type_union": type_union,
        "mean_first_round_by_fault_type": {
            key: _mean(value) for key, value in sorted(first_rounds.items())
        },
        "mean_proxy_type_agreement": _mean(
            [float(row.get("proxy_type_agreement", 0.0)) for row in rows]
        ),
        "mean_pairwise_query_jaccard": 1.0 if not overlaps else _mean(overlaps),
    }


def _group_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["mean_views"]["T1"]
    digest = _semantic_sha256(
        {"dataset_id": row["dataset_id"], "strategy_id": row["strategy_id"]}
    )
    return (
        -float(metrics["mrr"]),
        -float(metrics["hit_at_1"]),
        -float(metrics["hit_at_3"]),
        -float(metrics["hit_at_5"]),
        digest,
    )


def aggregate_strategy_results(
    result_rows: Any,
    *,
    authority_rows: Any,
) -> Dict[str, Any]:
    if isinstance(result_rows, (str, bytes)) or not isinstance(result_rows, Sequence) or not result_rows:
        raise ValueError("strategy results must be a non-empty sequence")
    if isinstance(authority_rows, (str, bytes)) or not isinstance(authority_rows, Sequence) or not authority_rows:
        raise ValueError("authority rows must be a non-empty sequence")
    authority_by_seed = {}
    for row in authority_rows:
        key = (str(row["dataset_id"]), int(row["active_learning_seed"]))
        if key in authority_by_seed:
            raise ValueError("authority dataset/seed rows must be unique")
        validate_count_authoritative_view(row["views"]["T1"])
        validate_count_authoritative_view(row["views"]["T2"])
        authority_by_seed[key] = row
    grouped: Dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in result_rows:
        key = (str(row["dataset_id"]), str(row["strategy_id"]), int(row["active_learning_seed"]))
        if key in seen:
            raise ValueError("strategy dataset/seed rows must be unique")
        seen.add(key)
        validate_count_authoritative_view(row["views"]["T1"])
        validate_count_authoritative_view(row["views"]["T2"])
        if (key[0], key[2]) not in authority_by_seed:
            raise ValueError("same-seed authority result is missing")
        grouped[(key[0], key[1])].append(row)

    groups = []
    for (dataset_id, strategy_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda value: int(value["active_learning_seed"]))
        authority = [authority_by_seed[(dataset_id, int(row["active_learning_seed"]))] for row in ordered]
        mean_views = {view: _mean_view(ordered, view) for view in ("T1", "T2")}
        authority_t1 = _mean_view(authority, "T1")
        per_seed = []
        serious = []
        for row, baseline in zip(ordered, authority):
            current = validate_count_authoritative_view(row["views"]["T1"])["metrics"]
            reference = validate_count_authoritative_view(baseline["views"]["T1"])["metrics"]
            hit_drops = [float(reference[metric]) - float(current[metric]) for metric in HIT_METRICS]
            mean_hit_drop = _mean(hit_drops)
            seed = int(row["active_learning_seed"])
            if mean_hit_drop > 0.10:
                serious.append(seed)
            per_seed.append(
                {
                    "active_learning_seed": seed,
                    "t1_mrr_delta": float(current["mrr"]) - float(reference["mrr"]),
                    "mean_hit_drop": mean_hit_drop,
                    "seriously_regressed": mean_hit_drop > 0.10,
                }
            )
        seed_count = len(ordered)
        prevalence = len(serious) / float(seed_count)
        mean_mrr_delta = _mean([row["t1_mrr_delta"] for row in per_seed])
        group = {
            "dataset_id": dataset_id,
            "strategy_id": strategy_id,
            "active_learning_seeds": [int(row["active_learning_seed"]) for row in ordered],
            "seed_count": seed_count,
            "mean_views": mean_views,
            "authority_mean_t1": authority_t1,
            "same_seed_authority_deltas": {
                "mean_mrr_delta": mean_mrr_delta,
                "per_seed": per_seed,
            },
            "strict_mean_t1_mrr_improvement": mean_views["T1"]["mrr"] > authority_t1["mrr"],
            "seriously_regressed_seeds": serious,
            "serious_regression_prevalence": prevalence,
            "minimum_expanded_seed_count": 10,
            "seed_expansion_required": bool(serious) and seed_count < 10,
            "stability_accepted": (not serious) or (seed_count >= 10 and prevalence <= 0.10),
            "query_diagnostics": _query_diagnostics(ordered),
        }
        group["aggregate_sha256"] = _semantic_sha256(group)
        groups.append(group)
    ranked = sorted(groups, key=_group_rank_key)
    result = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "groups": groups,
        "ranked_groups": ranked,
        "ranking_rule": "mean_T1_MRR_then_Hit1_Hit3_Hit5_then_semantic_hash",
        "outer_test_guided": True,
    }
    result["aggregate_sha256"] = _semantic_sha256(result)
    return result


def build_bounded_parameter_ranges() -> Dict[str, Any]:
    payload = {
        "kmeans_coverage": {"cluster_count": [8, 10, 12, 14, 16, 20]},
        "dbscan_coverage": {"eps": [0.25, 0.35, 0.5, 0.65, 0.8], "min_samples": [3, 4, 5, 6, 8]},
        "knn_fault_mode_coverage": {"neighbor_count": [3, 5, 7, 9, 11]},
        "mutual_knn_graph_coverage": {"neighbor_count": [3, 5, 7, 9, 11]},
        "falcon_hybrid": {"lc_fraction": [0.25, 0.4, 0.5, 0.6, 0.75], "time_chunk_count": [3, 4, 6, 8]},
        "hdbscan_coverage": {"min_cluster_size": [3, 5, 8, 12, 16], "min_samples": [2, 3, 5, 8]},
        "graph_facility_location": {"similarity_quantile": [0.25, 0.4, 0.5, 0.6, 0.75]},
        "fusion": {
            "fusion_id": [
                "masked_early",
                "coverage_normalized_late_affinity",
                "shared_state_alignment",
            ]
        },
    }
    return payload


_ATTEMPT_FIELDS = {
    "attempt_id",
    "parent_attempt_id",
    "dataset_id",
    "strategy_id",
    "active_learning_seed_scope",
    "config",
    "changed_parameters",
    "t1_feedback",
    "t2_feedback",
    "rationale",
    "outcome",
    "decision",
    "feedback_source",
}


def _validate_attempt(source: Any, *, previous: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != _ATTEMPT_FIELDS:
        raise ValueError("trial attempt fields do not match schema")
    attempt = deepcopy(dict(source))
    for field in ("attempt_id", "dataset_id", "strategy_id", "rationale", "outcome", "decision"):
        _canonical_text(attempt[field], field)
    if attempt["feedback_source"] != "outer_test_guided":
        raise ValueError("trial feedback must be labeled outer-test-guided")
    seeds = attempt["active_learning_seed_scope"]
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError("active-learning seed scope is invalid")
    seed_values = [int(value) for value in seeds]
    if len(seed_values) < 3 or len(seed_values) != len(set(seed_values)):
        raise ValueError("seed-specific final overrides are forbidden")
    attempt["active_learning_seed_scope"] = seed_values
    config = attempt["config"]
    if not isinstance(config, Mapping) or not config:
        raise ValueError("trial config must be a non-empty mapping")
    if any(isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)) for value in config.values()):
        raise ValueError("undeclared Cartesian sweeps are forbidden")
    changed = attempt["changed_parameters"]
    if isinstance(changed, (str, bytes)) or not isinstance(changed, Sequence) or len(changed) > 1:
        raise ValueError("trial attempts must be one-dimensional")
    changed_values = [str(value) for value in changed]
    if any(value not in config for value in changed_values):
        raise ValueError("changed parameter is absent from config")
    attempt["changed_parameters"] = changed_values
    if previous is None:
        if attempt["parent_attempt_id"] is not None:
            raise ValueError("first trial attempt cannot have a parent")
    else:
        if attempt["parent_attempt_id"] != previous["attempt_id"]:
            raise ValueError("trial lineage must extend the latest attempt")
        if attempt["dataset_id"] != previous["dataset_id"] or attempt["strategy_id"] != previous["strategy_id"]:
            raise ValueError("trial lineage cannot change dataset or strategy")
        prior_config = dict(previous["config"])
        delta = sorted(key for key in set(prior_config) | set(config) if prior_config.get(key) != config.get(key))
        if delta != changed_values:
            raise ValueError("declared one-dimensional delta does not match config")
    return attempt


def _ledger_sha(payload: Mapping[str, Any]) -> str:
    clean = deepcopy(dict(payload))
    clean.pop("ledger_sha256", None)
    return _semantic_sha256(clean)


def append_trial_attempt(ledger: Any, attempt: Any) -> Dict[str, Any]:
    if ledger is None:
        result = {
            "schema_version": TRIAL_LEDGER_SCHEMA_VERSION,
            "outer_test_guided": True,
            "untouched_test_claim": False,
            "parameter_ranges": build_bounded_parameter_ranges(),
            "attempts": [],
        }
    else:
        result = validate_trial_ledger(ledger)
    previous = result["attempts"][-1] if result["attempts"] else None
    entry = _validate_attempt(attempt, previous=previous)
    if any(existing["attempt_id"] == entry["attempt_id"] for existing in result["attempts"]):
        raise ValueError("trial attempt IDs must be immutable and unique")
    entry["previous_attempt_sha256"] = None if previous is None else previous["attempt_sha256"]
    entry["attempt_sha256"] = _semantic_sha256(entry)
    result["attempts"].append(entry)
    result["ledger_sha256"] = _ledger_sha(result)
    return result


def validate_trial_ledger(ledger: Any) -> Dict[str, Any]:
    if not isinstance(ledger, Mapping):
        raise ValueError("trial ledger must be a mapping")
    result = deepcopy(dict(ledger))
    if result.get("schema_version") != TRIAL_LEDGER_SCHEMA_VERSION:
        raise ValueError("trial ledger schema drifted")
    if result.get("outer_test_guided") is not True or result.get("untouched_test_claim") is not False:
        raise ValueError("trial ledger must deny untouched-test status")
    attempts = result.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("trial ledger attempts must be a list")
    rebuilt = []
    for index, source in enumerate(attempts):
        plain = deepcopy(dict(source))
        digest = plain.pop("attempt_sha256", None)
        previous_sha = plain.pop("previous_attempt_sha256", None)
        expected_previous = None if index == 0 else rebuilt[-1]["attempt_sha256"]
        if previous_sha != expected_previous:
            raise ValueError("trial attempt hash chain is invalid")
        validated = _validate_attempt(plain, previous=None if index == 0 else rebuilt[-1])
        validated["previous_attempt_sha256"] = expected_previous
        validated["attempt_sha256"] = _semantic_sha256(validated)
        if digest != validated["attempt_sha256"]:
            raise ValueError("trial attempt SHA is invalid")
        rebuilt.append(validated)
    result["attempts"] = rebuilt
    if result.get("ledger_sha256") != _ledger_sha(result):
        raise ValueError("trial ledger SHA is invalid")
    return result


def freeze_final_configuration(
    *,
    dataset_id: Any,
    strategy_id: Any,
    config: Any,
    active_learning_seeds: Any,
) -> Dict[str, Any]:
    dataset = _canonical_text(dataset_id, "dataset_id")
    strategy = _canonical_text(strategy_id, "strategy_id")
    if not isinstance(config, Mapping) or not config:
        raise ValueError("final config must be a non-empty mapping")
    if any(isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)) for value in config.values()):
        raise ValueError("final config cannot encode a sweep")
    if isinstance(active_learning_seeds, (str, bytes)) or not isinstance(active_learning_seeds, Sequence):
        raise ValueError("final seeds must be a sequence")
    seeds = [int(value) for value in active_learning_seeds]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("final config requires at least three unique seeds")
    payload = {
        "schema_version": FINAL_CONFIG_SCHEMA_VERSION,
        "dataset_id": dataset,
        "strategy_id": strategy,
        "active_learning_seeds": seeds,
        "config": deepcopy(dict(config)),
        "config_by_seed": {str(seed): deepcopy(dict(config)) for seed in seeds},
        "seed_specific_overrides": False,
        "outer_test_guided": True,
    }
    payload["freeze_sha256"] = _semantic_sha256(payload)
    return payload


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "FINAL_CONFIG_SCHEMA_VERSION",
    "SCORING_SCHEMA_VERSION",
    "TRIAL_LEDGER_SCHEMA_VERSION",
    "aggregate_strategy_results",
    "append_trial_attempt",
    "build_bounded_parameter_ranges",
    "freeze_final_configuration",
    "score_ranking_rows",
    "validate_count_authoritative_view",
    "validate_trial_ledger",
]
