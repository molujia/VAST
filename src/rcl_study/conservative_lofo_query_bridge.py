from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .ordinary_query_engine import (
    MappingAnnotationOracle,
    run_ordinary_query_engine,
    validate_ordinary_query_candidates,
    validate_query_event_log,
)


class QueryBridgeValidationError(ValueError):
    """Raised when a matched DBSCAN query plan drifts from its strict fold."""


_ARMS = ("baseline", "oser_meta", "mm_dro", "cope_gate")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _dataset_dbscan(authority_registry: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    try:
        record = dict(authority_registry["dbscan"][dataset_id])
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryBridgeValidationError(f"authority lacks DBSCAN config for {dataset_id}") from exc
    if record.get("strategy_id") != "dbscan_coverage":
        raise QueryBridgeValidationError("authority strategy is not dbscan_coverage")
    return deepcopy(dict(record.get("config", {})))


def _validated_rows_for_fold(
    fold: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    try:
        rows = validate_ordinary_query_candidates(candidate_rows)
    except ValueError as exc:
        raise QueryBridgeValidationError(f"candidate validation failed: {exc}") from exc
    ids = tuple(row["case_id"] for row in rows)
    expected = tuple(str(value) for value in fold.get("candidate_case_ids", ()))
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise QueryBridgeValidationError("candidate drift from strict-LOFO fold")
    held = set(str(value) for value in fold.get("test_only_case_ids", ()))
    unused = set(str(value) for value in fold.get("unused_outer_test_case_ids", ()))
    if set(ids) & held:
        raise QueryBridgeValidationError("candidate rows contain held-out cases")
    if set(ids) & unused:
        raise QueryBridgeValidationError("candidate rows contain non-held outer-test cases")
    return rows


def _validated_answers(
    fold: Mapping[str, Any],
    candidate_ids: Sequence[str],
    annotation_answers: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    answers = {str(case_id): dict(value) for case_id, value in annotation_answers.items()}
    if set(answers) != set(candidate_ids):
        raise QueryBridgeValidationError("annotation answer coverage differs from candidate ledger")
    membership = dict(fold.get("membership_by_case", {}))
    normalized: dict[str, dict[str, str]] = {}
    for case_id in candidate_ids:
        answer = answers[case_id]
        if set(answer) != {"root_cause", "fault_type"}:
            raise QueryBridgeValidationError(f"annotation answer is incomplete for {case_id}")
        row = {
            "root_cause": str(answer["root_cause"]).strip(),
            "fault_type": str(answer["fault_type"]).strip(),
        }
        if not all(row.values()):
            raise QueryBridgeValidationError(f"annotation answer is empty for {case_id}")
        if row["fault_type"] != str(membership.get(case_id, {}).get("fault_type", "")):
            raise QueryBridgeValidationError(f"annotation fault_type drift for {case_id}")
        normalized[case_id] = row
    return normalized


def build_matched_dbscan_query_plan(
    fold: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    annotation_answers: Mapping[str, Mapping[str, Any]],
    authority_registry: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id = str(fold.get("dataset_id", ""))
    if tuple(authority_registry.get("datasets", ())) not in (
        ("rcabench", "aiops2022_pre"),
        ["rcabench", "aiops2022_pre"],
    ):
        raise QueryBridgeValidationError("authority dataset scope drift")
    if int(authority_registry.get("budget", -1)) != 30:
        raise QueryBridgeValidationError("authority budget drift")
    if int(authority_registry.get("active_learning_seed", -1)) != 42:
        raise QueryBridgeValidationError("authority active_learning_seed drift")
    rows = _validated_rows_for_fold(fold, candidate_rows)
    candidate_ids = tuple(row["case_id"] for row in rows)
    answers = _validated_answers(fold, candidate_ids, annotation_answers)
    config = _dataset_dbscan(authority_registry, dataset_id)
    try:
        engine = run_ordinary_query_engine(
            candidate_rows,
            annotation_oracle=MappingAnnotationOracle(answers),
            strategy_id="dbscan_coverage",
            active_learning_seed=42,
            config=config,
        )
    except ValueError as exc:
        raise QueryBridgeValidationError(f"DBSCAN query engine failed: {exc}") from exc
    selected = tuple(str(value) for value in engine["selected_case_ids"])
    selected_answers = {case_id: answers[case_id] for case_id in selected}
    candidate_payload = [
        {
            "case_id": row["case_id"],
            "embedding": list(row["embedding"]),
            "timestamp": row["timestamp"],
            "incident_id": row["incident_id"],
            "uncertainty": row["uncertainty"],
        }
        for row in rows
    ]
    identity = {
        "schema_version": "conservative-lofo-matched-dbscan-query-plan-v1",
        "dataset_id": dataset_id,
        "held_out_fault_type": str(fold.get("held_out_fault_type", "")),
        "fold_sha256": str(fold.get("fold_sha256", "")),
        "strategy_id": "dbscan_coverage",
        "strategy_config": config,
        "active_learning_seed": 42,
        "budget": 30,
        "round_sizes": tuple(engine["round_sizes"]),
        "candidate_case_ids": candidate_ids,
        "candidate_rows_sha256": _semantic_hash(candidate_payload),
        "selected_case_ids": selected,
        "selected_case_ids_sha256": _semantic_hash(selected),
        "answers_by_selected_case": selected_answers,
        "selected_answers_sha256": _semantic_hash(selected_answers),
        "events": deepcopy(engine["events"]),
        "last_event_sha256": engine["events"][-1]["event_sha256"],
        "engine_query_plan_sha256": str(engine["query_plan_sha256"]),
    }
    plan = {**identity, "matched_query_plan_sha256": _semantic_hash(identity)}
    validate_matched_dbscan_query_plan(
        plan, fold, candidate_rows, annotation_answers, authority_registry
    )
    return plan


def validate_matched_dbscan_query_plan(
    plan: Mapping[str, Any],
    fold: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    annotation_answers: Mapping[str, Mapping[str, Any]],
    authority_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if str(plan.get("schema_version")) != "conservative-lofo-matched-dbscan-query-plan-v1":
        raise QueryBridgeValidationError("unexpected matched query-plan schema")
    dataset_id = str(fold.get("dataset_id", ""))
    if str(plan.get("dataset_id")) != dataset_id:
        raise QueryBridgeValidationError("query-plan dataset drift")
    if str(plan.get("held_out_fault_type")) != str(fold.get("held_out_fault_type")):
        raise QueryBridgeValidationError("query-plan target drift")
    if str(plan.get("fold_sha256")) != str(fold.get("fold_sha256")):
        raise QueryBridgeValidationError("query-plan fold hash drift")
    if plan.get("strategy_id") != "dbscan_coverage":
        raise QueryBridgeValidationError("query-plan strategy drift")
    if int(plan.get("budget", -1)) != 30:
        raise QueryBridgeValidationError("query-plan budget drift")
    if int(plan.get("active_learning_seed", -1)) != 42:
        raise QueryBridgeValidationError("query-plan seed drift")
    if dict(plan.get("strategy_config", {})) != _dataset_dbscan(authority_registry, dataset_id):
        raise QueryBridgeValidationError("query-plan DBSCAN config drift")

    rows = _validated_rows_for_fold(fold, candidate_rows)
    candidate_ids = tuple(row["case_id"] for row in rows)
    candidate_payload = [
        {
            "case_id": row["case_id"],
            "embedding": list(row["embedding"]),
            "timestamp": row["timestamp"],
            "incident_id": row["incident_id"],
            "uncertainty": row["uncertainty"],
        }
        for row in rows
    ]
    if tuple(plan.get("candidate_case_ids", ())) != candidate_ids:
        raise QueryBridgeValidationError("query-plan candidate order drift")
    if plan.get("candidate_rows_sha256") != _semantic_hash(candidate_payload):
        raise QueryBridgeValidationError("query-plan candidate content drift")
    answers = _validated_answers(fold, candidate_ids, annotation_answers)
    selected = tuple(str(value) for value in plan.get("selected_case_ids", ()))
    if len(selected) != len(set(selected)) or len(selected) != 30:
        raise QueryBridgeValidationError("query-plan selected budget is not exact")
    candidate_set = set(candidate_ids)
    if not set(selected) <= candidate_set:
        raise QueryBridgeValidationError("query-plan selected non-candidate case")
    if set(selected) & set(str(value) for value in fold.get("test_only_case_ids", ())):
        raise QueryBridgeValidationError("query-plan selected held-out case")
    if set(selected) & set(str(value) for value in fold.get("unused_outer_test_case_ids", ())):
        raise QueryBridgeValidationError("query-plan selected non-held outer-test case")
    if plan.get("selected_case_ids_sha256") != _semantic_hash(selected):
        raise QueryBridgeValidationError("query-plan selected-case hash drift")
    selected_answers = {case_id: answers[case_id] for case_id in selected}
    if plan.get("answers_by_selected_case") != selected_answers:
        raise QueryBridgeValidationError("query-plan selected answers drift")
    if plan.get("selected_answers_sha256") != _semantic_hash(selected_answers):
        raise QueryBridgeValidationError("query-plan selected-answer hash drift")
    try:
        event_audit = validate_query_event_log(plan.get("events", ()))
    except ValueError as exc:
        raise QueryBridgeValidationError(f"query-plan event history invalid: {exc}") from exc
    if tuple(event_audit["selected_case_ids"]) != selected or event_audit["annotation_cost"] != 30:
        raise QueryBridgeValidationError("query-plan event selection/cost drift")
    if str(plan.get("last_event_sha256")) != str(event_audit["last_event_sha256"]):
        raise QueryBridgeValidationError("query-plan event tail drift")
    identity = {key: deepcopy(value) for key, value in plan.items() if key != "matched_query_plan_sha256"}
    if plan.get("matched_query_plan_sha256") != _semantic_hash(identity):
        raise QueryBridgeValidationError("matched query-plan hash drift")
    return {
        "valid": True,
        "dataset_id": dataset_id,
        "selected_count": 30,
        "annotation_cost": 30,
        "event_count": event_audit["event_count"],
        "matched_query_plan_sha256": plan["matched_query_plan_sha256"],
    }


def bind_query_plan_to_arm(plan: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arm_id = str(arm)
    if arm_id not in _ARMS:
        raise QueryBridgeValidationError(f"unknown query-plan consumer arm: {arm_id}")
    query_hash = str(plan.get("matched_query_plan_sha256", ""))
    selected_hash = str(plan.get("selected_case_ids_sha256", ""))
    if len(query_hash) != 64 or len(selected_hash) != 64:
        raise QueryBridgeValidationError("query-plan binding hashes are invalid")
    return {
        "schema_version": "conservative-lofo-query-plan-binding-v1",
        "arm": arm_id,
        "matched_query_plan_sha256": query_hash,
        "selected_case_ids_sha256": selected_hash,
        "mutable_query_state_shared": False,
    }


__all__ = [
    "QueryBridgeValidationError",
    "bind_query_plan_to_arm",
    "build_matched_dbscan_query_plan",
    "validate_matched_dbscan_query_plan",
]
