"""Convert fixed active-learning output into a downstream LOFO input manifest."""

from __future__ import annotations

from typing import Any, Mapping

from .plan_contract import validate_query_plan


def export_lofo_manifest(
    plan: Mapping[str, Any], *, validate_frozen_budget: bool = True
) -> dict[str, Any]:
    if validate_frozen_budget:
        validate_query_plan(plan)
    records = plan.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("query plan records are required")
    selected = []
    seen: set[str] = set()
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("query plan record must be an object")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("selected case identifier is invalid")
        if case_id in seen:
            raise ValueError(f"duplicate selected case: {case_id}")
        seen.add(case_id)
        query_index = record.get("query_index")
        if query_index != expected_index:
            raise ValueError("selected case order is not contiguous")
        selected.append(
            {
                "case_id": case_id,
                "query_index": query_index,
                "cluster_label": record.get("raw_label"),
                "is_residual": bool(record.get("is_residual", False)),
            }
        )
    return {
        "schema_version": "rcl-lofo-input/v1",
        "lofo_boundary": "after_active_learning",
        "active_learning_frozen": True,
        "dataset_id": plan.get("dataset_id"),
        "active_learning_seed": plan.get("active_learning_seed"),
        "budget": plan.get("budget"),
        "selected_case_count": len(selected),
        "source_query_plan_sha256": plan.get("plan_sha256"),
        "selected_cases": selected,
    }

