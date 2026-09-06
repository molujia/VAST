from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


class ProtocolValidationError(ValueError):
    """Raised when a strict-LOFO membership or label-access contract drifts."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def serialize_membership(membership_by_case: Mapping[str, Mapping[str, Any]]) -> str:
    normalized = {
        str(case_id): dict(record)
        for case_id, record in sorted(membership_by_case.items(), key=lambda item: str(item[0]))
    }
    return _canonical_json(normalized)


def _inventory_parts(
    inventory: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    cases = {
        str(case_id): dict(record)
        for case_id, record in dict(inventory.get("cases_by_id", {})).items()
    }
    if not cases:
        raise ProtocolValidationError("inventory contains no cases_by_id")
    train = tuple(str(value) for value in inventory.get("train_case_ids", ()))
    test = tuple(str(value) for value in inventory.get("test_case_ids", ()))
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ProtocolValidationError("inventory split contains duplicate case IDs")
    overlap = set(train) & set(test)
    if overlap:
        raise ProtocolValidationError(f"inventory train/test overlap: {sorted(overlap)[:5]}")
    known = set(cases)
    assigned = set(train) | set(test)
    if assigned != known:
        missing = sorted(known - assigned)
        unknown = sorted(assigned - known)
        raise ProtocolValidationError(
            f"inventory split must cover cases exactly; missing={missing[:5]} unknown={unknown[:5]}"
        )
    for case_id, record in cases.items():
        fault_type = str(record.get("fault_type", "")).strip()
        if not fault_type:
            raise ProtocolValidationError(f"inventory case {case_id} missing fault_type")
        record["fault_type"] = fault_type
    return cases, train, test


def _inventory_identity(
    dataset_id: str,
    cases: Mapping[str, Mapping[str, Any]],
    train: Sequence[str],
    test: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset_id": str(dataset_id),
        "train_case_ids": sorted(str(value) for value in train),
        "test_case_ids": sorted(str(value) for value in test),
        "fault_type_by_case": {
            case_id: str(cases[case_id]["fault_type"]) for case_id in sorted(cases)
        },
    }


def build_initial_target_inventory(
    dataset_id: str,
    inventory: Mapping[str, Any],
    requested_targets: Sequence[Any],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    cases, train, test = _inventory_parts(inventory)
    available = {str(record["fault_type"]) for record in cases.values()}
    alias_map = {str(key): str(value) for key, value in dict(aliases or {}).items()}
    resolved: list[str] = []
    used_aliases: dict[str, str] = {}
    for raw in requested_targets:
        requested = str(raw)
        canonical = alias_map.get(requested, requested)
        if canonical not in available:
            raise ProtocolValidationError(
                f"unknown target fault type {requested!r} resolved as {canonical!r}"
            )
        if canonical in resolved:
            raise ProtocolValidationError(f"duplicate target after alias resolution: {canonical}")
        resolved.append(canonical)
        if requested != canonical:
            used_aliases[requested] = canonical
    support = Counter(str(record["fault_type"]) for record in cases.values())
    identity = {
        "schema_version": "conservative-lofo-initial-target-inventory-v1",
        "dataset_id": str(dataset_id),
        "inventory_sha256": _semantic_hash(
            _inventory_identity(dataset_id, cases, train, test)
        ),
        "target_fault_types": tuple(resolved),
        "resolved_aliases": used_aliases,
        "support_by_fault_type": {fault: support[fault] for fault in resolved},
        "case_count": len(cases),
    }
    return {**identity, "target_inventory_sha256": _semantic_hash(identity)}


def build_strict_lofo_fold(
    dataset_id: str,
    inventory: Mapping[str, Any],
    held_out_fault_type: str,
) -> dict[str, Any]:
    cases, train, test = _inventory_parts(inventory)
    held = str(held_out_fault_type).strip()
    if not held:
        raise ProtocolValidationError("held_out_fault_type must be nonempty")
    if held not in {str(record["fault_type"]) for record in cases.values()}:
        raise ProtocolValidationError(f"held_out_fault_type is absent from inventory: {held}")

    train_set = set(train)
    test_set = set(test)
    membership: dict[str, dict[str, Any]] = {}
    test_only: list[str] = []
    candidates: list[str] = []
    unused_outer_test: list[str] = []
    for case_id in sorted(cases):
        fault_type = str(cases[case_id]["fault_type"])
        original_split = "outer_train" if case_id in train_set else "outer_test"
        if fault_type == held:
            role = "held_out_test_only"
            test_only.append(case_id)
        elif case_id in train_set:
            role = "selector_visible_unlabeled_candidate"
            candidates.append(case_id)
        elif case_id in test_set:
            role = "unused_non_held_outer_test"
            unused_outer_test.append(case_id)
        else:  # pragma: no cover - guarded by _inventory_parts
            raise ProtocolValidationError(f"case lacks original split: {case_id}")
        membership[case_id] = {
            "case_id": case_id,
            "fault_type": fault_type,
            "original_split": original_split,
            "role": role,
        }

    membership_serialization = serialize_membership(membership)
    inventory_sha256 = _semantic_hash(_inventory_identity(dataset_id, cases, train, test))
    identity = {
        "schema_version": "conservative-lofo-fold-v1",
        "dataset_id": str(dataset_id),
        "held_out_fault_type": held,
        "inventory_sha256": inventory_sha256,
        "test_only_case_ids": tuple(test_only),
        "candidate_case_ids": tuple(candidates),
        "unused_outer_test_case_ids": tuple(unused_outer_test),
        "membership_by_case": membership,
        "membership_sha256": hashlib.sha256(
            membership_serialization.encode("utf-8")
        ).hexdigest(),
    }
    return {**identity, "fold_sha256": _semantic_hash(identity)}


def validate_strict_lofo_fold(
    fold: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    if str(fold.get("schema_version")) != "conservative-lofo-fold-v1":
        raise ProtocolValidationError("unexpected strict LOFO fold schema")
    dataset_id = str(fold.get("dataset_id", ""))
    held = str(fold.get("held_out_fault_type", ""))
    expected = build_strict_lofo_fold(dataset_id, inventory, held)
    for field in (
        "inventory_sha256",
        "test_only_case_ids",
        "candidate_case_ids",
        "unused_outer_test_case_ids",
        "membership_by_case",
        "membership_sha256",
        "fold_sha256",
    ):
        observed = fold.get(field)
        if observed != expected[field]:
            raise ProtocolValidationError(
                f"{field} mismatch for held-out type {held}: expected frozen membership"
            )
    role_sets = [
        set(str(value) for value in fold[field])
        for field in (
            "test_only_case_ids",
            "candidate_case_ids",
            "unused_outer_test_case_ids",
        )
    ]
    if role_sets[0] & role_sets[1] or role_sets[0] & role_sets[2] or role_sets[1] & role_sets[2]:
        raise ProtocolValidationError("strict LOFO fold roles overlap")
    cases = set(dict(inventory.get("cases_by_id", {})))
    if set.union(*role_sets) != cases:
        raise ProtocolValidationError("strict LOFO fold roles omit inventory cases")
    return {
        "valid": True,
        "dataset_id": dataset_id,
        "held_out_fault_type": held,
        "test_count": len(role_sets[0]),
        "candidate_count": len(role_sets[1]),
        "unused_count": len(role_sets[2]),
        "membership_sha256": fold["membership_sha256"],
    }


_ALLOWED_FIELDS = {
    "fold_membership_only": {"case_id", "fault_type", "original_split"},
    "unlabeled_observation_only": {"case_id", "observations", "topology", "original_split"},
    "queried_supervision": {
        "case_id",
        "observations",
        "topology",
        "root_cause",
        "fault_type",
    },
    "development_calibration_only": {
        "case_id",
        "observations",
        "topology",
        "root_cause",
        "fault_type",
    },
    "post_inference_scoring_only": {"case_id", "root_cause", "fault_type", "ranking"},
}


def build_label_access_ledger(
    fold: Mapping[str, Any],
    inventory: Mapping[str, Any],
    queried_case_ids: Sequence[Any],
) -> dict[str, Any]:
    validate_strict_lofo_fold(fold, inventory)
    queried = tuple(str(value) for value in queried_case_ids)
    if len(queried) != len(set(queried)):
        raise ProtocolValidationError("queried_case_ids contains duplicates")
    candidate_set = set(str(value) for value in fold["candidate_case_ids"])
    unknown = sorted(set(queried) - candidate_set)
    if unknown:
        raise ProtocolValidationError(f"queried cases are not fold candidates: {unknown[:5]}")

    events: list[dict[str, Any]] = []
    membership = dict(fold["membership_by_case"])
    for case_id in sorted(membership):
        events.append(
            {
                "case_id": case_id,
                "role": "fold_membership_only",
                "phase": "fold_construction",
                "fields": ("case_id", "fault_type", "original_split"),
                "label_cost": 0,
            }
        )
    queried_set = set(queried)
    for case_id in fold["candidate_case_ids"]:
        if case_id in queried_set:
            events.append(
                {
                    "case_id": case_id,
                    "role": "queried_supervision",
                    "phase": "post_query_training",
                    "fields": (
                        "case_id",
                        "observations",
                        "topology",
                        "root_cause",
                        "fault_type",
                    ),
                    "label_cost": 1,
                }
            )
        else:
            events.append(
                {
                    "case_id": case_id,
                    "role": "unlabeled_observation_only",
                    "phase": "selection_or_pretraining",
                    "fields": ("case_id", "observations", "topology", "original_split"),
                    "label_cost": 0,
                }
            )
    for case_id in fold["test_only_case_ids"]:
        events.append(
            {
                "case_id": case_id,
                "role": "post_inference_scoring_only",
                "phase": "after_frozen_inference",
                "fields": ("case_id", "root_cause", "fault_type", "ranking"),
                "label_cost": 0,
            }
        )
    identity = {
        "schema_version": "conservative-lofo-label-access-ledger-v1",
        "fold_sha256": fold["fold_sha256"],
        "queried_case_ids": queried,
        "events": events,
        "total_label_cost": len(queried),
    }
    return {**identity, "ledger_sha256": _semantic_hash(identity)}


def validate_label_access_ledger(
    ledger: Mapping[str, Any],
    fold: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    validate_strict_lofo_fold(fold, inventory)
    if str(ledger.get("schema_version")) != "conservative-lofo-label-access-ledger-v1":
        raise ProtocolValidationError("unexpected label access ledger schema")
    if ledger.get("fold_sha256") != fold.get("fold_sha256"):
        raise ProtocolValidationError("label access fold ownership mismatch")
    candidate_set = set(str(value) for value in fold["candidate_case_ids"])
    test_set = set(str(value) for value in fold["test_only_case_ids"])
    observed_cost = 0
    seen_scoring: set[str] = set()
    seen_supervision: set[str] = set()
    for raw in ledger.get("events", ()):
        event = dict(raw)
        case_id = str(event.get("case_id", ""))
        role = str(event.get("role", ""))
        fields = set(str(value) for value in event.get("fields", ()))
        allowed = _ALLOWED_FIELDS.get(role)
        if allowed is None or not fields <= allowed:
            raise ProtocolValidationError(
                f"label access violation for {case_id} role={role}: fields={sorted(fields)}"
            )
        if "fit_feedback" in fields:
            raise ProtocolValidationError(f"label access uses test feedback for {case_id}")
        if role in {"queried_supervision", "unlabeled_observation_only"} and case_id not in candidate_set:
            raise ProtocolValidationError(f"label access admits non-candidate {case_id} to fitting")
        if role == "queried_supervision":
            seen_supervision.add(case_id)
            if int(event.get("label_cost", -1)) != 1:
                raise ProtocolValidationError(f"label access cost mismatch for {case_id}")
            observed_cost += 1
        elif int(event.get("label_cost", 0)) != 0:
            raise ProtocolValidationError(f"label access non-query cost for {case_id}")
        if role == "post_inference_scoring_only":
            if case_id not in test_set or str(event.get("phase")) != "after_frozen_inference":
                raise ProtocolValidationError(f"label access scoring phase violation for {case_id}")
            seen_scoring.add(case_id)
    queried = tuple(str(value) for value in ledger.get("queried_case_ids", ()))
    if seen_supervision != set(queried):
        raise ProtocolValidationError("label access queried supervision coverage mismatch")
    if seen_scoring != test_set:
        raise ProtocolValidationError("label access held-out scoring coverage mismatch")
    if observed_cost != len(queried) or int(ledger.get("total_label_cost", -1)) != observed_cost:
        raise ProtocolValidationError("label access total cost mismatch")
    identity = {key: deepcopy(value) for key, value in ledger.items() if key != "ledger_sha256"}
    if ledger.get("ledger_sha256") != _semantic_hash(identity):
        raise ProtocolValidationError("label access ledger hash mismatch")
    return {"valid": True, "label_cost": observed_cost, "event_count": len(ledger["events"])}


def build_union_excluded_calibration_pool(
    dataset_id: str,
    inventory: Mapping[str, Any],
    excluded_fault_types: Sequence[Any],
    *,
    queried_case_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    cases, train, test = _inventory_parts(inventory)
    excluded = tuple(str(value) for value in excluded_fault_types)
    if not excluded or len(excluded) != len(set(excluded)):
        raise ProtocolValidationError("excluded_fault_types must be unique and nonempty")
    excluded_set = set(excluded)
    admitted = tuple(
        sorted(case_id for case_id in train if str(cases[case_id]["fault_type"]) not in excluded_set)
    )
    excluded_ids = tuple(
        sorted(case_id for case_id, row in cases.items() if str(row["fault_type"]) in excluded_set)
    )
    queried = tuple(str(value) for value in (queried_case_ids or ()))
    if len(queried) != len(set(queried)):
        raise ProtocolValidationError("queried development case IDs contain duplicates")
    if set(queried) - set(admitted):
        raise ProtocolValidationError("queried development cases must belong to admitted pool")
    counts = Counter(str(cases[case_id]["fault_type"]) for case_id in excluded_ids)
    events = [
        {
            "case_id": case_id,
            "role": "development_calibration_only",
            "fields": (
                "case_id",
                "observations",
                "topology",
                "root_cause",
                "fault_type",
            ),
            "label_cost": 1,
        }
        for case_id in queried
    ]
    identity = {
        "schema_version": "conservative-lofo-union-excluded-calibration-v1",
        "dataset_id": str(dataset_id),
        "inventory_sha256": _semantic_hash(
            _inventory_identity(dataset_id, cases, train, test)
        ),
        "excluded_fault_types": excluded,
        "excluded_case_ids": excluded_ids,
        "excluded_counts_by_fault_type": {fault: counts[fault] for fault in excluded},
        "admitted_candidate_case_ids": admitted,
        "unused_outer_test_case_ids": tuple(sorted(test)),
        "queried_development_case_ids": queried,
        "label_access_events": events,
        "development_label_cost": len(queried),
    }
    return {**identity, "calibration_pool_sha256": _semantic_hash(identity)}


def validate_union_excluded_calibration_pool(
    pool: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    formal_training_case_ids: Iterable[Any] = (),
) -> dict[str, Any]:
    if str(pool.get("schema_version")) != "conservative-lofo-union-excluded-calibration-v1":
        raise ProtocolValidationError("unexpected union-excluded calibration schema")
    queried = tuple(str(value) for value in pool.get("queried_development_case_ids", ()))
    if len(queried) != 30 or int(pool.get("development_label_cost", -1)) != 30:
        raise ProtocolValidationError("calibration must contain exactly 30 queried development cases")
    expected = build_union_excluded_calibration_pool(
        str(pool.get("dataset_id", "")),
        inventory,
        tuple(pool.get("excluded_fault_types", ())),
        queried_case_ids=queried,
    )
    for field in (
        "inventory_sha256",
        "excluded_case_ids",
        "excluded_counts_by_fault_type",
        "admitted_candidate_case_ids",
        "unused_outer_test_case_ids",
        "label_access_events",
        "development_label_cost",
        "calibration_pool_sha256",
    ):
        if pool.get(field) != expected[field]:
            if field == "admitted_candidate_case_ids":
                raise ProtocolValidationError("admitted pool contains an excluded fault type or drift")
            raise ProtocolValidationError(f"calibration {field} mismatch")
    formal = {str(value) for value in formal_training_case_ids}
    overlap = formal & set(queried)
    if overlap:
        raise ProtocolValidationError(
            f"calibration formal training overlap: {sorted(overlap)[:5]}"
        )
    for event in pool.get("label_access_events", ()):
        if str(event.get("role")) != "development_calibration_only":
            raise ProtocolValidationError("calibration label access role drift")
    return {
        "valid": True,
        "candidate_count": len(pool["admitted_candidate_case_ids"]),
        "excluded_count": len(pool["excluded_case_ids"]),
        "development_label_cost": 30,
    }


__all__ = [
    "ProtocolValidationError",
    "build_initial_target_inventory",
    "build_label_access_ledger",
    "build_strict_lofo_fold",
    "build_union_excluded_calibration_pool",
    "serialize_membership",
    "validate_label_access_ledger",
    "validate_strict_lofo_fold",
    "validate_union_excluded_calibration_pool",
]
