from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class SyntheticAdmissionValidationError(ValueError):
    """Raised when a synthetic row escapes its train-only partition."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVALUATION_PARTITIONS = {
    "validation",
    "strict_lofo_test",
    "ordinary_test",
    "t1",
    "t2",
}
_DENOMINATOR_PARTITIONS = {
    "strict_lofo_test",
    "ordinary_test",
    "t1",
    "t2",
}


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ids(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SyntheticAdmissionValidationError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise SyntheticAdmissionValidationError(
            f"{context} must contain unique nonempty IDs"
        )
    return result


def _membership(
    value: Any, expected: set[str], context: str
) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SyntheticAdmissionValidationError(f"{context} partition closure drift")
    return {
        partition: _ids(value[partition], f"{context}.{partition}")
        for partition in sorted(expected)
    }


def build_synthetic_admission_ledger(
    *,
    synthetic_rows: Sequence[Mapping[str, Any]],
    evaluation_membership: Mapping[str, Sequence[Any]],
    official_denominator_membership: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    if isinstance(synthetic_rows, (str, bytes)) or not isinstance(
        synthetic_rows, Sequence
    ):
        raise SyntheticAdmissionValidationError("synthetic_rows must be a sequence")
    admitted_rows = []
    admitted_ids = []
    seen: set[str] = set()
    for raw in synthetic_rows:
        if not isinstance(raw, Mapping):
            raise SyntheticAdmissionValidationError("synthetic row must be a mapping")
        row = deepcopy(dict(raw))
        row_id = str(row.get("synthetic_row_sha256", "")).strip()
        weight = row.get("final_training_weight")
        if (
            not _SHA256.fullmatch(row_id)
            or row_id in seen
            or row.get("query_budget_cost") != 0
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise SyntheticAdmissionValidationError(
                "synthetic admission identity, budget, or weight drift"
            )
        seen.add(row_id)
        admitted_ids.append(row_id)
        admitted_rows.append(row)

    evaluation = _membership(
        evaluation_membership, _EVALUATION_PARTITIONS, "evaluation membership"
    )
    denominators = _membership(
        official_denominator_membership,
        _DENOMINATOR_PARTITIONS,
        "official denominator",
    )
    synthetic_id_set = set(admitted_ids)
    evaluation_overlap = sorted(
        synthetic_id_set
        & {
            case_id
            for members in evaluation.values()
            for case_id in members
        }
    )
    if evaluation_overlap:
        raise SyntheticAdmissionValidationError(
            "synthetic rows are train-only and cannot enter evaluation"
        )
    denominator_overlap = sorted(
        synthetic_id_set
        & {
            case_id
            for members in denominators.values()
            for case_id in members
        }
    )
    if denominator_overlap:
        raise SyntheticAdmissionValidationError(
            "synthetic rows cannot enter an official denominator"
        )
    for partition in sorted(_DENOMINATOR_PARTITIONS):
        if denominators[partition] != evaluation[partition]:
            raise SyntheticAdmissionValidationError(
                f"official denominator membership drift for {partition}"
            )

    identity = {
        "schema_version": "service-continuous-synthetic-admission-ledger-v1",
        "admission_scope": "ranker_training_only",
        "admitted_synthetic_row_ids": admitted_ids,
        "admitted_synthetic_row_count": len(admitted_ids),
        "train_only_rows": admitted_rows,
        "excluded_partitions": sorted(_EVALUATION_PARTITIONS),
        "evaluation_membership": evaluation,
        "official_denominator_membership": denominators,
        "official_denominator_counts": {
            partition: len(members)
            for partition, members in sorted(denominators.items())
        },
        "synthetic_evaluation_overlap_count": 0,
        "synthetic_official_denominator_overlap_count": 0,
    }
    return {**identity, "ledger_sha256": _semantic_hash(identity)}


__all__ = [
    "SyntheticAdmissionValidationError",
    "build_synthetic_admission_ledger",
]
