from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class FinalRCLEvaluationError(ValueError):
    """Raised when the final matched evaluation contract is violated."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATASETS = ("rcabench", "aiops2022_pre")
_ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)
_QUERY_ARMS = frozenset(("hdbscan_query_cvae_oser", "hdbscan_query_pairwise"))
_COMPLETE_ARMS = frozenset(("hdbscan_query_cvae_oser", "oracle_full_cvae_oser"))
_METRIC_FIELDS = ("hit_at_1", "hit_at_3", "hit_at_5", "top135", "mrr")


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


def _sha256(value: Any, context: str) -> str:
    result = str(value).strip()
    if not _SHA256.fullmatch(result):
        raise FinalRCLEvaluationError(f"{context} must be SHA-256")
    return result


def _ids(value: Any, context: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalRCLEvaluationError(f"{context} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if (
        (not allow_empty and not result)
        or "" in result
        or len(result) != len(set(result))
    ):
        raise FinalRCLEvaluationError(f"{context} must be unique and nonempty")
    return result


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise FinalRCLEvaluationError(f"{context} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalRCLEvaluationError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise FinalRCLEvaluationError(f"{context} must be finite")
    return result


def _validate_hash_seal(value: Mapping[str, Any], field: str, context: str) -> None:
    identity = {key: item for key, item in value.items() if key != field}
    if value.get(field) != _semantic_hash(identity):
        raise FinalRCLEvaluationError(f"{context} hash drifted")


def build_fixed_split_adapter(
    *,
    expected_dataset_id: str,
    inventory_manifest: Mapping[str, Any],
    candidate_pool: Mapping[str, Any],
    query_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the frozen ordinary split to the immutable HDBSCAN authority."""

    dataset_id = str(expected_dataset_id)
    if dataset_id not in _DATASETS:
        raise FinalRCLEvaluationError("final split requires a declared dataset")
    inventory = deepcopy(dict(inventory_manifest))
    if (
        inventory.get("schema_version") != "ordinary-query-only-inventory-v1"
        or inventory.get("canonical_dataset_id") != dataset_id
        or inventory.get("split_seed") != 42
    ):
        raise FinalRCLEvaluationError("ordinary inventory identity drifted")
    records = inventory.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FinalRCLEvaluationError("ordinary inventory records must be a sequence")
    train: list[str] = []
    test: list[str] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise FinalRCLEvaluationError("ordinary inventory record must be a mapping")
        case_id = str(raw.get("case_id", "")).strip()
        split = str(raw.get("split", ""))
        if not case_id or case_id in seen:
            raise FinalRCLEvaluationError("ordinary inventory case IDs must be unique")
        seen.add(case_id)
        if split == "outer_train":
            train.append(case_id)
        elif split == "outer_test":
            test.append(case_id)
        else:
            raise FinalRCLEvaluationError("ordinary inventory split is invalid")
    train_ids = _ids(train, "outer-train case IDs")
    test_ids = _ids(test, "outer-test case IDs")
    overlap = set(train_ids).intersection(test_ids)
    if overlap:
        raise FinalRCLEvaluationError("outer-train/test overlap is forbidden")
    if "record_count" in inventory and inventory.get("record_count") != len(records):
        raise FinalRCLEvaluationError("ordinary inventory record count drifted")
    if "outer_train_count" in inventory and inventory.get("outer_train_count") != len(train_ids):
        raise FinalRCLEvaluationError("ordinary inventory train count drifted")
    if "outer_test_count" in inventory and inventory.get("outer_test_count") != len(test_ids):
        raise FinalRCLEvaluationError("ordinary inventory test count drifted")
    inventory_sha = _sha256(inventory.get("sha256"), "ordinary inventory identity")

    pool = deepcopy(dict(candidate_pool))
    if (
        pool.get("candidate_population")
        != "full_observable_outer_train_not_sampled"
        or pool.get("case_count") != len(train_ids)
        or ("dataset_id" in pool and pool.get("dataset_id") != dataset_id)
        or pool.get("ground_truth_included", False) is not False
    ):
        raise FinalRCLEvaluationError("HDBSCAN candidate-pool identity drifted")
    pool_ids = _ids(pool.get("case_ids"), "HDBSCAN candidate-pool case IDs")
    if set(pool_ids) != set(train_ids):
        raise FinalRCLEvaluationError(
            "HDBSCAN candidate pool must equal the complete outer-train membership"
        )

    plan = deepcopy(dict(query_plan))
    if (
        plan.get("dataset_id") != dataset_id
        or plan.get("clusterer_id") != "hdbscan"
        or plan.get("selector_id") != "center"
        or plan.get("active_learning_seed") != 42
        or plan.get("budget") != 30
    ):
        raise FinalRCLEvaluationError("final query authority must be HDBSCAN center seed 42")
    selected = _ids(plan.get("selected_case_ids"), "HDBSCAN selected case IDs")
    if len(selected) != 30 or not set(selected).issubset(train_ids):
        raise FinalRCLEvaluationError("HDBSCAN query budget or membership drifted")
    plan_sha = _sha256(plan.get("plan_sha256"), "HDBSCAN query plan")
    partition_sha = _sha256(plan.get("partition_sha256"), "HDBSCAN partition")
    representation_sha = (
        _sha256(plan.get("representation_matrix_sha256"), "HDBSCAN representation")
        if plan.get("representation_matrix_sha256") is not None
        else None
    )
    pool_sha = _semantic_hash(pool)
    identity = {
        "schema_version": "final-rcl-fixed-split-adapter-v1",
        "dataset_id": dataset_id,
        "split_seed": 42,
        "outer_train_case_ids": tuple(train_ids),
        "outer_test_case_ids": tuple(test_ids),
        "outer_train_case_count": len(train_ids),
        "outer_test_case_count": len(test_ids),
        "fit_test_overlap_count": 0,
        "inventory_sha256": inventory_sha,
        "candidate_pool_sha256": pool_sha,
        "hdbscan_candidate_order": pool_ids,
        "query_case_ids": selected,
        "query_plan_sha256": plan_sha,
        "hdbscan_partition_sha256": partition_sha,
        "hdbscan_representation_sha256": representation_sha,
    }
    return {**identity, "split_sha256": _semantic_hash(identity)}


def _validate_split_adapter(value: Mapping[str, Any]) -> dict[str, Any]:
    split = deepcopy(dict(value))
    _validate_hash_seal(split, "split_sha256", "fixed split adapter")
    train = _ids(split.get("outer_train_case_ids"), "outer-train case IDs")
    test = _ids(split.get("outer_test_case_ids"), "outer-test case IDs")
    query = _ids(split.get("query_case_ids"), "query case IDs")
    if (
        split.get("schema_version") != "final-rcl-fixed-split-adapter-v1"
        or split.get("dataset_id") not in _DATASETS
        or split.get("split_seed") != 42
        or split.get("outer_train_case_count") != len(train)
        or split.get("outer_test_case_count") != len(test)
        or split.get("fit_test_overlap_count") != 0
        or len(query) != 30
        or not set(query).issubset(train)
        or set(train).intersection(test)
    ):
        raise FinalRCLEvaluationError("fixed split adapter membership drifted")
    _sha256(split.get("query_plan_sha256"), "query plan")
    _sha256(split.get("hdbscan_partition_sha256"), "HDBSCAN partition")
    return split


def build_final_arm_contract(
    *, split_adapter: Mapping[str, Any], arm_id: str
) -> dict[str, Any]:
    split = _validate_split_adapter(split_adapter)
    arm = str(arm_id)
    if arm not in _ARMS:
        raise FinalRCLEvaluationError("unknown final evaluation arm")
    query_only = arm in _QUERY_ARMS
    complete = arm in _COMPLETE_ARMS
    real_training_ids = tuple(
        split["query_case_ids"] if query_only else split["outer_train_case_ids"]
    )
    supervision_sha = (
        split["query_plan_sha256"]
        if query_only
        else _semantic_hash(
            {
                "dataset_id": split["dataset_id"],
                "training_mode": "oracle_full",
                "real_training_case_ids": real_training_ids,
                "split_sha256": split["split_sha256"],
            }
        )
    )
    identity = {
        "schema_version": "final-rcl-arm-evaluation-contract-v1",
        "unit_id": f"{split['dataset_id']}--{arm}--seed42",
        "dataset_id": split["dataset_id"],
        "arm_id": arm,
        "seed": 42,
        "training_mode": "query_only" if query_only else "oracle_full",
        "augmentation": "proxy_mode_cvae_compatible" if complete else "none",
        "model": "oser_meta" if complete else "pairwise_linear",
        "outer_train_case_ids": tuple(split["outer_train_case_ids"]),
        "outer_test_case_ids": tuple(split["outer_test_case_ids"]),
        "real_training_case_ids": real_training_ids,
        "real_training_case_count": len(real_training_ids),
        "fit_test_overlap_count": len(
            set(real_training_ids).intersection(split["outer_test_case_ids"])
        ),
        "split_sha256": split["split_sha256"],
        "query_plan_sha256": split["query_plan_sha256"] if query_only else None,
        "supervision_authority_sha256": supervision_sha,
        "hdbscan_partition_sha256": split["hdbscan_partition_sha256"],
        "real_training_membership_sha256": _semantic_hash(real_training_ids),
    }
    if identity["fit_test_overlap_count"]:
        raise FinalRCLEvaluationError("arm fitting overlaps final test")
    return {**identity, "arm_contract_sha256": _semantic_hash(identity)}


def _validate_arm_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = deepcopy(dict(value))
    _validate_hash_seal(contract, "arm_contract_sha256", "arm contract")
    if (
        contract.get("schema_version") != "final-rcl-arm-evaluation-contract-v1"
        or contract.get("dataset_id") not in _DATASETS
        or contract.get("arm_id") not in _ARMS
        or contract.get("seed") != 42
        or contract.get("fit_test_overlap_count") != 0
    ):
        raise FinalRCLEvaluationError("arm contract identity drifted")
    train = _ids(contract.get("outer_train_case_ids"), "arm outer-train IDs")
    test = _ids(contract.get("outer_test_case_ids"), "arm outer-test IDs")
    fit = _ids(contract.get("real_training_case_ids"), "arm real-training IDs")
    if set(train).intersection(test) or not set(fit).issubset(train):
        raise FinalRCLEvaluationError("arm fit/test membership overlap or drift")
    if contract.get("real_training_case_count") != len(fit):
        raise FinalRCLEvaluationError("arm real-training count drifted")
    return contract


def validate_matched_arm_contracts(
    arm_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(arm_contracts, (str, bytes)) or len(arm_contracts) != 3:
        raise FinalRCLEvaluationError("matched comparison requires three arm contracts")
    contracts = tuple(_validate_arm_contract(value) for value in arm_contracts)
    if {value["arm_id"] for value in contracts} != set(_ARMS):
        raise FinalRCLEvaluationError("matched comparison arm coverage drifted")
    datasets = {value["dataset_id"] for value in contracts}
    splits = {value["split_sha256"] for value in contracts}
    train_sets = {tuple(value["outer_train_case_ids"]) for value in contracts}
    test_sets = {tuple(value["outer_test_case_ids"]) for value in contracts}
    if any(len(values) != 1 for values in (datasets, splits, train_sets, test_sets)):
        raise FinalRCLEvaluationError("matched arms do not share one fixed split")
    overlap_count = sum(
        len(set(value["real_training_case_ids"]).intersection(value["outer_test_case_ids"]))
        for value in contracts
    )
    if overlap_count:
        raise FinalRCLEvaluationError("matched arms contain fit/test overlap")
    return {
        "valid": True,
        "dataset_id": contracts[0]["dataset_id"],
        "arm_count": 3,
        "shared_outer_train_case_count": len(contracts[0]["outer_train_case_ids"]),
        "shared_outer_test_case_count": len(contracts[0]["outer_test_case_ids"]),
        "fit_test_overlap_count": overlap_count,
        "split_sha256": contracts[0]["split_sha256"],
    }


def build_final_ranking_artifact(
    *,
    arm_contract: Mapping[str, Any],
    score_artifact: Mapping[str, Any],
    expected_candidates_by_case: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Normalize base and OSER outputs through one candidate-complete path."""

    contract = _validate_arm_contract(arm_contract)
    score = deepcopy(dict(score_artifact))
    arm = contract["arm_id"]
    complete = arm in _COMPLETE_ARMS
    expected_schema = (
        "conservative-lofo-residual-output-v1"
        if complete
        else "conservative-lofo-base-score-artifact-v1"
    )
    if score.get("schema_version") != expected_schema:
        raise FinalRCLEvaluationError("arm score artifact belongs to the wrong scoring path")
    if score.get("artifact_role") != "final_test":
        raise FinalRCLEvaluationError("final score artifact role drifted")
    if str(score.get("query_plan_sha256", "")) != contract["supervision_authority_sha256"]:
        raise FinalRCLEvaluationError("score supervision authority drifted")
    if complete and score.get("arm") != "oser_meta":
        raise FinalRCLEvaluationError("complete arm must terminate in OSER-Meta")
    if not complete and score.get("backend_id") != "pairwise_linear":
        raise FinalRCLEvaluationError("control arm must terminate in pairwise-linear")
    candidate_map = {
        str(case_id): _ids(values, f"candidate IDs for {case_id}")
        for case_id, values in expected_candidates_by_case.items()
    }
    test_ids = tuple(contract["outer_test_case_ids"])
    if tuple(candidate_map) != test_ids:
        raise FinalRCLEvaluationError("final score case order or coverage drifted")
    recorded_candidates = {
        str(case_id): _ids(values, f"recorded candidates for {case_id}")
        for case_id, values in dict(score.get("candidate_ids_by_case", {})).items()
    }
    rankings = {
        str(case_id): _ids(values, f"ranking candidates for {case_id}")
        for case_id, values in dict(score.get("rankings_by_case", {})).items()
    }
    score_field = "final_scores_by_case" if complete else "scores_by_case"
    scores = {
        str(case_id): {
            str(candidate_id): _finite(raw, f"score for {case_id}/{candidate_id}")
            for candidate_id, raw in dict(values).items()
        }
        for case_id, values in dict(score.get(score_field, {})).items()
    }
    if not (
        tuple(recorded_candidates) == test_ids
        and tuple(rankings) == test_ids
        and tuple(scores) == test_ids
    ):
        raise FinalRCLEvaluationError("score artifact case coverage drifted")
    ranking_rows = []
    for case_id in test_ids:
        expected = candidate_map[case_id]
        if (
            set(recorded_candidates[case_id]) != set(expected)
            or set(rankings[case_id]) != set(expected)
            or set(scores[case_id]) != set(expected)
            or len(rankings[case_id]) != len(expected)
        ):
            raise FinalRCLEvaluationError(
                f"candidate-complete ranking drifted for {case_id}"
            )
        ranking_rows.append(
            {
                "case_id": case_id,
                "candidate_ids": expected,
                "ranking": rankings[case_id],
                "scores": {candidate: scores[case_id][candidate] for candidate in expected},
            }
        )
    source_hash_field = "artifact_sha256" if complete else "score_artifact_sha256"
    source_hash = _sha256(score.get(source_hash_field), "source score artifact")
    identity = {
        "schema_version": "final-rcl-candidate-complete-ranking-v1",
        "unit_id": contract["unit_id"],
        "dataset_id": contract["dataset_id"],
        "arm_id": arm,
        "seed": 42,
        "split_sha256": contract["split_sha256"],
        "supervision_authority_sha256": contract["supervision_authority_sha256"],
        "source_score_schema": expected_schema,
        "source_score_artifact_sha256": source_hash,
        "case_ids": test_ids,
        "case_count": len(test_ids),
        "ranking_rows": tuple(ranking_rows),
    }
    return {**identity, "ranking_sha256": _semantic_hash(identity)}


def _validate_ranking_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    ranking = deepcopy(dict(value))
    _validate_hash_seal(ranking, "ranking_sha256", "ranking artifact")
    case_ids = _ids(ranking.get("case_ids"), "ranking case IDs")
    rows = ranking.get("ranking_rows")
    if (
        ranking.get("schema_version") != "final-rcl-candidate-complete-ranking-v1"
        or isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or ranking.get("case_count") != len(case_ids)
        or len(rows) != len(case_ids)
    ):
        raise FinalRCLEvaluationError("ranking artifact identity drifted")
    if tuple(str(row.get("case_id", "")) for row in rows) != case_ids:
        raise FinalRCLEvaluationError("ranking row order drifted")
    return ranking


def aggregate_final_rank_metrics(
    *,
    ranking_artifact: Mapping[str, Any],
    targets_by_case: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Derive every reported score from case ranks and integer counts."""

    artifact = _validate_ranking_artifact(ranking_artifact)
    targets = {
        str(case_id): _ids(values, f"true roots for {case_id}")
        for case_id, values in targets_by_case.items()
    }
    case_ids = tuple(artifact["case_ids"])
    if tuple(targets) != case_ids:
        raise FinalRCLEvaluationError("true-root case order or coverage drifted")
    hit_counts = {1: 0, 3: 0, 5: 0}
    case_ranks: dict[str, int] = {}
    reciprocal_rank_sum = 0.0
    for row in artifact["ranking_rows"]:
        case_id = str(row["case_id"])
        candidates = _ids(row.get("candidate_ids"), f"candidate IDs for {case_id}")
        ranking = _ids(row.get("ranking"), f"ranking for {case_id}")
        if set(candidates) != set(ranking):
            raise FinalRCLEvaluationError(f"candidate ranking drifted for {case_id}")
        true_roots = set(targets[case_id])
        if not true_roots.intersection(candidates):
            raise FinalRCLEvaluationError(
                f"true root is absent from candidate set for {case_id}"
            )
        rank = min(index for index, candidate in enumerate(ranking, 1) if candidate in true_roots)
        case_ranks[case_id] = rank
        reciprocal_rank_sum += 1.0 / rank
        for cutoff in hit_counts:
            hit_counts[cutoff] += int(rank <= cutoff)
    denominator = len(case_ids)
    rates = {cutoff: hit_counts[cutoff] / denominator for cutoff in hit_counts}
    identity = {
        "schema_version": "final-rcl-count-derived-metrics-v1",
        "unit_id": artifact["unit_id"],
        "dataset_id": artifact["dataset_id"],
        "arm_id": artifact["arm_id"],
        "seed": 42,
        "ranking_sha256": artifact["ranking_sha256"],
        "denominator": denominator,
        "hit_at_1_count": hit_counts[1],
        "hit_at_3_count": hit_counts[3],
        "hit_at_5_count": hit_counts[5],
        "hit_at_1": rates[1],
        "hit_at_3": rates[3],
        "hit_at_5": rates[5],
        "top135": math.fsum((rates[1], rates[3], rates[5])) / 3.0,
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "mrr": reciprocal_rank_sum / denominator,
        "case_ranks": case_ranks,
    }
    return {**identity, "metrics_sha256": _semantic_hash(identity)}


def _dependency_hash(value: Any, context: str) -> str:
    if not isinstance(value, Mapping) or set(value) != {"sha256"}:
        raise FinalRCLEvaluationError(f"{context} dependency must contain only sha256")
    return _sha256(value.get("sha256"), context)


def build_final_unit_provenance(
    *,
    arm_contract: Mapping[str, Any],
    ranking_artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    code_config_closure: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _validate_arm_contract(arm_contract)
    ranking = _validate_ranking_artifact(ranking_artifact)
    measured = deepcopy(dict(metrics))
    _validate_hash_seal(measured, "metrics_sha256", "unit metrics")
    if (
        ranking.get("unit_id") != contract["unit_id"]
        or measured.get("unit_id") != contract["unit_id"]
        or measured.get("ranking_sha256") != ranking["ranking_sha256"]
    ):
        raise FinalRCLEvaluationError("unit ranking/metric ownership drifted")
    arm = contract["arm_id"]
    required = {"split", "hdbscan_partition"}
    if arm in _QUERY_ARMS:
        required.add("query_plan")
    if arm in _COMPLETE_ARMS:
        required.update(
            ("cvae_ledger", "cvae_checkpoint", "oser_checkpoint", "oser_activity")
        )
    deps = deepcopy(dict(dependencies))
    if set(deps) != required:
        raise FinalRCLEvaluationError("arm-specific provenance dependency closure drifted")
    normalized_dependencies = {
        name: {"sha256": _dependency_hash(deps[name], name)} for name in sorted(deps)
    }
    if normalized_dependencies["split"]["sha256"] != contract["split_sha256"]:
        raise FinalRCLEvaluationError("provenance split identity drifted")
    if (
        normalized_dependencies["hdbscan_partition"]["sha256"]
        != contract["hdbscan_partition_sha256"]
    ):
        raise FinalRCLEvaluationError("provenance HDBSCAN partition drifted")
    if arm in _QUERY_ARMS and (
        normalized_dependencies["query_plan"]["sha256"]
        != contract["query_plan_sha256"]
    ):
        raise FinalRCLEvaluationError("provenance query-plan identity drifted")
    closure = deepcopy(dict(code_config_closure))
    if set(closure) != {"code_sha256", "config_sha256", "input_sha256"}:
        raise FinalRCLEvaluationError("code/config/input closure fields drifted")
    closure = {name: _sha256(value, name) for name, value in sorted(closure.items())}
    identity = {
        "schema_version": "final-rcl-unit-provenance-v1",
        "unit_id": contract["unit_id"],
        "dataset_id": contract["dataset_id"],
        "arm_id": arm,
        "seed": 42,
        "training_mode": contract["training_mode"],
        "split_sha256": contract["split_sha256"],
        "real_training_case_ids": tuple(contract["real_training_case_ids"]),
        "real_training_case_count": contract["real_training_case_count"],
        "real_training_membership_sha256": contract["real_training_membership_sha256"],
        "fit_test_overlap_count": 0,
        "query_plan_sha256": contract["query_plan_sha256"],
        "hdbscan_partition_sha256": contract["hdbscan_partition_sha256"],
        "ranking_sha256": ranking["ranking_sha256"],
        "metrics_sha256": measured["metrics_sha256"],
        "dependencies": normalized_dependencies,
        "code_config_closure": closure,
    }
    return {**identity, "provenance_sha256": _semantic_hash(identity)}


def aggregate_final_matched_comparisons(
    unit_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(unit_records, (str, bytes)) or len(unit_records) != 6:
        raise FinalRCLEvaluationError("matched aggregate requires exactly six units")
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in unit_records:
        row = deepcopy(dict(raw))
        dataset_id = str(row.get("dataset_id", ""))
        arm_id = str(row.get("arm_id", ""))
        key = (dataset_id, arm_id)
        if (
            dataset_id not in _DATASETS
            or arm_id not in _ARMS
            or key in normalized
            or row.get("seed") != 42
        ):
            raise FinalRCLEvaluationError("six-unit registry coverage drifted")
        _sha256(row.get("split_sha256"), "unit split")
        _sha256(row.get("provenance_sha256"), "unit provenance")
        metrics = deepcopy(dict(row.get("metrics", {})))
        denominator = metrics.get("denominator")
        if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
            raise FinalRCLEvaluationError("unit metric denominator must be positive")
        projected = {name: _finite(metrics.get(name), name) for name in _METRIC_FIELDS}
        if any(not 0.0 <= value <= 1.0 for value in projected.values()):
            raise FinalRCLEvaluationError("unit metric rate is outside [0,1]")
        row["metrics"] = {"denominator": denominator, **projected}
        normalized[key] = row
    if set(normalized) != {(dataset, arm) for dataset in _DATASETS for arm in _ARMS}:
        raise FinalRCLEvaluationError("matched aggregate does not contain six declared units")

    datasets: dict[str, Any] = {}
    for dataset_id in _DATASETS:
        rows = {arm: normalized[(dataset_id, arm)] for arm in _ARMS}
        if len({row["split_sha256"] for row in rows.values()}) != 1:
            raise FinalRCLEvaluationError("matched unit split identity drifted")
        if len({row["metrics"]["denominator"] for row in rows.values()}) != 1:
            raise FinalRCLEvaluationError("matched unit denominator drifted")
        complete = rows["hdbscan_query_cvae_oser"]["metrics"]
        oracle = rows["oracle_full_cvae_oser"]["metrics"]
        control = rows["hdbscan_query_pairwise"]["metrics"]

        def delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
            return {name: float(left[name]) - float(right[name]) for name in _METRIC_FIELDS}

        datasets[dataset_id] = {
            "split_sha256": rows["hdbscan_query_cvae_oser"]["split_sha256"],
            "denominator": complete["denominator"],
            "arms": {arm: deepcopy(rows[arm]["metrics"]) for arm in _ARMS},
            "complete_minus_hdbscan_only": delta(complete, control),
            "query_only_minus_oracle_full": delta(complete, oracle),
        }
    identity = {
        "schema_version": "final-rcl-matched-comparison-v1",
        "seed": 42,
        "unit_count": 6,
        "datasets": datasets,
    }
    return {**identity, "aggregate_sha256": _semantic_hash(identity)}


__all__ = [
    "FinalRCLEvaluationError",
    "aggregate_final_matched_comparisons",
    "aggregate_final_rank_metrics",
    "build_final_arm_contract",
    "build_final_ranking_artifact",
    "build_final_unit_provenance",
    "build_fixed_split_adapter",
    "validate_matched_arm_contracts",
]
