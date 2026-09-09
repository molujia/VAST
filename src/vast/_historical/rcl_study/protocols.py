"""Leakage-safe protocol validators for the budgeted RCL study."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


CASE_FIELDS = (
    "case_id",
    "window_id",
    "split",
    "case_kind",
    "admitted",
    "targets",
)
QUERY_FIELDS = ("case_id", "annotation_source", "targets")
TRAINING_ROW_FIELDS = (
    "case_id",
    "window_id",
    "derived_row_id",
    "supervision_source",
)
NORMAL_POLICY_CASE_FIELDS = ("case_id", "split", "case_kind", "admitted")
NORMAL_POLICY_TRAINING_FIELDS = (
    "case_id",
    "case_kind",
    "training_label",
    "supervision_source",
    "annotation_cost",
)
EVALUATION_ROW_FIELDS = ("case_id", "split", "case_kind")


def _require_fields(payload: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError("%s missing fields: %s" % (context, ", ".join(missing)))


def _target_set(value: Any, context: str) -> set:
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("%s targets must be a sequence" % context)
    targets = {str(item) for item in value if str(item)}
    if not targets:
        raise ValueError("%s targets must not be empty" % context)
    return targets


def validate_case_budget(
    *,
    case_records: Sequence[Mapping[str, Any]],
    query_plan: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    expected_budget: int = 30,
) -> Dict[str, Any]:
    if isinstance(expected_budget, bool) or not isinstance(expected_budget, int):
        raise ValueError("expected_budget must be a positive integer")
    if expected_budget <= 0:
        raise ValueError("expected_budget must be a positive integer")

    cases_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(case_records):
        _require_fields(case, CASE_FIELDS, "case record %d" % index)
        case_id = str(case["case_id"])
        if not case_id or case_id in cases_by_id:
            raise ValueError("case records must have unique non-empty case_id values")
        if (
            bool(case["admitted"])
            and str(case["case_kind"]) == "fault"
            and str(case["window_id"]) != case_id
        ):
            raise ValueError("current admitted fault views must satisfy window_id == case_id")
        _target_set(case["targets"], "case record %d" % index)
        cases_by_id[case_id] = case

    query_ids = [str(query.get("case_id", "")) for query in query_plan]
    if len(query_ids) != expected_budget or len(set(query_ids)) != expected_budget:
        raise ValueError(
            "query plan must contain exactly %d unique fault case IDs" % expected_budget
        )

    annotated_target_count = 0
    for index, query in enumerate(query_plan):
        _require_fields(query, QUERY_FIELDS, "query record %d" % index)
        case_id = str(query["case_id"])
        case = cases_by_id.get(case_id)
        if (
            case is None
            or not bool(case["admitted"])
            or str(case["split"]) != "outer_train"
            or str(case["case_kind"]) != "fault"
        ):
            raise ValueError(
                "queried case %s must be an admitted outer_train fault case" % case_id
            )
        if str(query["annotation_source"]) != "simulated_manual_ground_truth":
            raise ValueError("queried case %s has an invalid annotation source" % case_id)
        authoritative_targets = _target_set(case["targets"], "case %s" % case_id)
        annotated_targets = _target_set(query["targets"], "query %s" % case_id)
        if annotated_targets != authoritative_targets:
            raise ValueError(
                "queried case %s annotation must return the complete target set" % case_id
            )
        annotated_target_count += len(annotated_targets)

    seen_derived_ids = set()
    training_case_ids = set()
    for index, row in enumerate(training_rows):
        _require_fields(row, TRAINING_ROW_FIELDS, "training row %d" % index)
        case_id = str(row["case_id"])
        if case_id not in set(query_ids):
            raise ValueError("training row uses an unqueried fault case %s" % case_id)
        if str(row["window_id"]) != case_id:
            raise ValueError("training row must preserve window_id == case_id")
        if str(row["supervision_source"]) != "queried_ground_truth":
            raise ValueError("training row has non-query supervision")
        derived_id = str(row["derived_row_id"])
        if not derived_id or derived_id in seen_derived_ids:
            raise ValueError("training derived_row_id values must be unique and non-empty")
        seen_derived_ids.add(derived_id)
        training_case_ids.add(case_id)
    if training_case_ids != set(query_ids):
        raise ValueError("every queried case must contribute at least one training row")

    return {
        "expected_budget": expected_budget,
        "annotated_fault_case_count": len(set(query_ids)),
        "annotated_target_count": annotated_target_count,
        "derived_training_row_count": len(training_rows),
        "queried_case_ids": query_ids,
        "passed": True,
    }


def validate_normal_policy(
    *,
    case_records: Sequence[Mapping[str, Any]],
    query_case_ids: Sequence[str],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    policy: str,
) -> Dict[str, Any]:
    policy_id = str(policy)
    if policy_id not in {"fault_only", "with_normal_class"}:
        raise ValueError("unsupported normal policy %r" % policy_id)
    query_ids = [str(case_id) for case_id in query_case_ids]
    if len(query_ids) != 30 or len(set(query_ids)) != 30:
        raise ValueError("normal policy requires the same 30 unique queried fault cases")

    cases_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(case_records):
        _require_fields(case, NORMAL_POLICY_CASE_FIELDS, "case record %d" % index)
        case_id = str(case["case_id"])
        if not case_id or case_id in cases_by_id:
            raise ValueError("normal-policy case records must have unique case IDs")
        cases_by_id[case_id] = case
    for case_id in query_ids:
        case = cases_by_id.get(case_id)
        if (
            case is None
            or not bool(case["admitted"])
            or str(case["split"]) != "outer_train"
            or str(case["case_kind"]) != "fault"
        ):
            raise ValueError("query IDs must identify admitted outer_train fault cases")

    eligible_normal_ids = {
        case_id
        for case_id, case in cases_by_id.items()
        if bool(case["admitted"])
        and str(case["split"]) == "outer_train"
        and str(case["case_kind"]) == "normal"
    }
    training_by_id: Dict[str, Mapping[str, Any]] = {}
    fault_training_ids = set()
    normal_training_ids = set()
    annotation_cost = 0
    for index, row in enumerate(training_rows):
        _require_fields(
            row, NORMAL_POLICY_TRAINING_FIELDS, "normal-policy training row %d" % index
        )
        case_id = str(row["case_id"])
        if case_id not in cases_by_id or case_id in training_by_id:
            raise ValueError("normal-policy training rows must reference unique known cases")
        training_by_id[case_id] = row
        kind = str(row["case_kind"])
        cost = row["annotation_cost"]
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
            raise ValueError("training annotation_cost must be a non-negative integer")
        annotation_cost += cost
        if kind == "fault":
            if (
                case_id not in set(query_ids)
                or str(row["supervision_source"]) != "queried_ground_truth"
                or cost != 1
            ):
                raise ValueError("fault training supervision must come from queried cases")
            fault_training_ids.add(case_id)
        elif kind == "normal":
            normal_training_ids.add(case_id)
        else:
            raise ValueError("training case_kind must be fault or normal")
    if fault_training_ids != set(query_ids):
        raise ValueError("training must retain exactly the same 30 queried fault cases")

    if policy_id == "fault_only":
        if normal_training_ids:
            raise ValueError("fault_only excludes normal training rows")
    else:
        if normal_training_ids != eligible_normal_ids:
            raise ValueError("with_normal_class must include all eligible normal windows")
        for case_id in normal_training_ids:
            row = training_by_id[case_id]
            if (
                str(row["training_label"]) != "__normal__"
                or str(row["supervision_source"]) != "normal_window"
                or row["annotation_cost"] != 0
            ):
                raise ValueError(
                    "normal windows must use one zero-cost normal class for training only"
                )
    if annotation_cost != 30:
        raise ValueError("normal windows must not change annotation cost from 30")

    expected_evaluation_ids = {
        case_id
        for case_id, case in cases_by_id.items()
        if bool(case["admitted"])
        and str(case["case_kind"]) == "fault"
        and str(case["split"]) in {"inner_validation", "outer_test"}
    }
    evaluation_ids = []
    for index, row in enumerate(evaluation_rows):
        _require_fields(row, EVALUATION_ROW_FIELDS, "evaluation row %d" % index)
        if str(row["case_kind"]) != "fault":
            raise ValueError("validation and test RCL denominators must be fault-only")
        if str(row["split"]) not in {"inner_validation", "outer_test"}:
            raise ValueError("evaluation rows must be inner_validation or outer_test")
        evaluation_ids.append(str(row["case_id"]))
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("evaluation rows must contain unique case IDs")
    if set(evaluation_ids) != expected_evaluation_ids:
        raise ValueError("evaluation rows do not match admitted fault-only cases")

    return {
        "policy": policy_id,
        "queried_fault_case_count": len(query_ids),
        "normal_training_case_count": len(normal_training_ids),
        "annotation_cost": annotation_cost,
        "evaluation_fault_case_count": len(evaluation_ids),
        "passed": True,
    }


__all__ = ["validate_case_budget", "validate_normal_policy"]
