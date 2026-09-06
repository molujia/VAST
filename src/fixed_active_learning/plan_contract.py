"""Stable query-plan hashing and validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


PLAN_BODY_FIELDS = (
    "dataset_id",
    "clusterer_id",
    "selector_id",
    "active_learning_seed",
    "budget",
    "representation_matrix_sha256",
    "geometry_sha256",
    "partition_sha256",
    "quota_sha256",
    "allocation_sha256",
    "queues_sha256",
    "case_sets_sha256",
    "input_sha256",
    "records",
)


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_plan_hash(plan: Mapping[str, Any]) -> str:
    missing = [field for field in PLAN_BODY_FIELDS if field not in plan]
    if missing:
        raise ValueError("query plan is missing fields: " + ", ".join(missing))
    return semantic_sha256({field: plan[field] for field in PLAN_BODY_FIELDS})


def validate_query_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping):
        raise ValueError("query plan must be an object")
    actual = semantic_plan_hash(plan)
    if actual != plan.get("plan_sha256"):
        raise ValueError("query-plan semantic hash mismatch")
    if plan.get("dataset_id") not in {"rcabench", "aiops2022_pre"}:
        raise ValueError("query plan dataset is unsupported")
    if plan.get("clusterer_id") != "hdbscan" or plan.get("selector_id") != "center":
        raise ValueError("query plan is not the fixed HDBSCAN center method")
    if plan.get("active_learning_seed") not in {41, 42, 43}:
        raise ValueError("query plan active-learning seed drifted")
    if plan.get("budget") != 30:
        raise ValueError("query plan budget drifted")
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("query plan must contain 30 records")
    case_ids = [record.get("case_id") for record in records if isinstance(record, Mapping)]
    if len(case_ids) != 30 or len(set(case_ids)) != 30 or any(not item for item in case_ids):
        raise ValueError("query plan selected cases must be 30 unique identifiers")
    if [record.get("query_index") for record in records] != list(range(30)):
        raise ValueError("query plan record order drifted")
    if plan.get("selected_case_ids") != case_ids:
        raise ValueError("query plan selected_case_ids drifted")
    expected_plan_id = (
        f"{plan['dataset_id']}.hdbscan.center.seed{plan['active_learning_seed']}."
        f"budget30.{actual[:12]}"
    )
    if plan.get("plan_id") != expected_plan_id:
        raise ValueError("query plan identifier drifted")

