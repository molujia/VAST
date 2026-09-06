from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class SyntheticTrainingValidationError(ValueError):
    """Raised when synthetic budget or per-source influence drifts."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rows(value: Any, context: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SyntheticTrainingValidationError(f"{context} must be a sequence")
    result = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise SyntheticTrainingValidationError(
                f"{context}[{index}] must be a mapping"
            )
        result.append(deepcopy(dict(raw)))
    return result


def _finite(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SyntheticTrainingValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise SyntheticTrainingValidationError(f"{context} must be finite")
    return result


def build_synthetic_training_ledger(
    *,
    real_query_rows: Sequence[Mapping[str, Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
    query_budget: int = 30,
    source_total_mass_by_case: Mapping[str, Any] | None = None,
    max_source_mass: float = 1.0,
) -> dict[str, Any]:
    real = _rows(real_query_rows, "real_query_rows")
    synthetic = _rows(synthetic_rows, "synthetic_rows")
    if (
        isinstance(query_budget, bool)
        or not isinstance(query_budget, int)
        or query_budget <= 0
    ):
        raise SyntheticTrainingValidationError(
            "query_budget must be a positive integer"
        )
    if len(real) != query_budget:
        raise SyntheticTrainingValidationError(
            "training real-query count must equal query_budget"
        )
    real_ids = []
    real_weights: dict[str, float] = {}
    for row in real:
        case_id = str(row.get("case_id", "")).strip()
        if (
            not case_id
            or case_id in real_weights
            or row.get("query_budget_cost") != 1
            or _finite(row.get("training_weight"), "real training weight") != 1.0
        ):
            raise SyntheticTrainingValidationError(
                "real queries require unique IDs, unit budget, and weight 1.0"
            )
        real_ids.append(case_id)
        real_weights[case_id] = 1.0

    maximum_mass = _finite(max_source_mass, "max_source_mass", positive=True)
    if source_total_mass_by_case is not None and not isinstance(
        source_total_mass_by_case, Mapping
    ):
        raise SyntheticTrainingValidationError(
            "source_total_mass_by_case must be a mapping"
        )

    groups: dict[str, list[int]] = defaultdict(list)
    synthetic_hashes: set[str] = set()
    for index, row in enumerate(synthetic):
        row_hash = str(row.get("synthetic_row_sha256", "")).strip()
        source = str(row.get("source_case_id", "")).strip()
        target = str(row.get("target_service_id", "")).strip()
        if row.get("query_budget_cost") != 0:
            raise SyntheticTrainingValidationError(
                "every synthetic row must have zero query budget cost"
            )
        if (
            not _SHA256.fullmatch(row_hash)
            or row_hash in synthetic_hashes
            or source not in real_weights
            or not target
        ):
            raise SyntheticTrainingValidationError(
                "synthetic row identity or source ownership drift"
            )
        _finite(
            row.get("provisional_training_weight"),
            "provisional synthetic training weight",
            positive=True,
        )
        synthetic_hashes.add(row_hash)
        groups[source].append(index)

    if source_total_mass_by_case is None:
        requested_source_mass = {source: 1.0 for source in groups}
    else:
        requested_source_mass = {}
        for raw_source, raw_mass in source_total_mass_by_case.items():
            source = str(raw_source).strip()
            mass = _finite(raw_mass, f"synthetic source mass for {source}")
            if (
                not source
                or source not in real_weights
                or mass < 0.0
                or mass > maximum_mass
            ):
                raise SyntheticTrainingValidationError(
                    "synthetic source mass must satisfy 0 <= W_i <= W_max/max_source_mass"
                )
            requested_source_mass[source] = mass
        if any(
            source not in groups and mass > 0.0
            for source, mass in requested_source_mass.items()
        ):
            raise SyntheticTrainingValidationError(
                "a positive synthetic source mass requires retained children"
            )

    normalized_rows = deepcopy(synthetic)
    per_source_sum: dict[str, float] = {}
    child_counts: dict[str, int] = {}
    per_source_plan_hashes: dict[str, str] = {}
    target_mass_by_source: dict[str, dict[str, float]] = {}
    for source in sorted(groups):
        indices = groups[source]
        source_mass = requested_source_mass.get(source, 0.0)
        raw_total = math.fsum(
            _finite(
                synthetic[index]["provisional_training_weight"],
                "provisional synthetic training weight",
                positive=True,
            )
            for index in indices
        )
        plan_identity = {
            "schema_version": "service-continuous-synthetic-source-weight-plan-v1",
            "source_case_id": source,
            "raw_weight_total": raw_total,
            "source_total_mass": source_mass,
            "max_source_mass": maximum_mass,
            "children": [
                {
                    "synthetic_row_sha256": synthetic[index][
                        "synthetic_row_sha256"
                    ],
                    "provisional_training_weight": float(
                        synthetic[index]["provisional_training_weight"]
                    ),
                }
                for index in indices
            ],
        }
        plan_hash = _semantic_hash(plan_identity)
        per_source_plan_hashes[source] = plan_hash
        target_mass: dict[str, float] = defaultdict(float)
        for index in indices:
            final_weight = source_mass * float(
                synthetic[index]["provisional_training_weight"]
            ) / raw_total
            normalized_rows[index]["final_training_weight"] = final_weight
            normalized_rows[index]["weight_normalization_sha256"] = plan_hash
            target_mass[synthetic[index]["target_service_id"]] += final_weight
        per_source_sum[source] = math.fsum(
            normalized_rows[index]["final_training_weight"] for index in indices
        )
        child_counts[source] = len(indices)
        target_mass_by_source[source] = dict(sorted(target_mass.items()))
        if not math.isclose(
            per_source_sum[source], source_mass, abs_tol=1e-8
        ):
            raise SyntheticTrainingValidationError(
                "per-source synthetic training mass must sum to W_i"
            )

    normalization_identity = {
        "schema_version": "service-continuous-synthetic-weight-normalization-v1",
        "real_query_case_ids": real_ids,
        "synthetic_row_sha256s": [
            row["synthetic_row_sha256"] for row in normalized_rows
        ],
        "source_weight_plan_sha256s": per_source_plan_hashes,
        "synthetic_weight_sum_by_source": per_source_sum,
        "source_total_mass_by_case": dict(sorted(requested_source_mass.items())),
        "max_source_mass": maximum_mass,
    }
    identity = {
        "schema_version": "service-continuous-synthetic-training-ledger-v1",
        "real_query_count": len(real),
        "real_query_budget_cost": sum(int(row["query_budget_cost"]) for row in real),
        "real_case_training_weights": dict(sorted(real_weights.items())),
        "synthetic_row_count": len(normalized_rows),
        "synthetic_query_budget_cost": sum(
            int(row["query_budget_cost"]) for row in normalized_rows
        ),
        "synthetic_source_count": len(groups),
        "synthetic_child_count_by_source": child_counts,
        "synthetic_weight_sum_by_source": per_source_sum,
        "source_total_mass_by_case": dict(sorted(requested_source_mass.items())),
        "max_source_mass": maximum_mass,
        "synthetic_target_mass_by_source": target_mass_by_source,
        "source_weight_plan_sha256s": per_source_plan_hashes,
        "synthetic_rows": normalized_rows,
        "weight_normalization_sha256": _semantic_hash(normalization_identity),
    }
    return {**identity, "ledger_sha256": _semantic_hash(identity)}


__all__ = ["SyntheticTrainingValidationError", "build_synthetic_training_ledger"]
