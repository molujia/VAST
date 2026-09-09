"""Result records with evidence tiers and auditable native denominators."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .datasets import resolve_dataset


RESULT_SCHEMA_VERSION = "rcl-result-record-v1"
EVIDENCE_TIERS = {"historical_reference", "clean_baseline", "optimized"}
ACCEPTANCE_STATUSES = {"not_applicable", "pending", "accepted", "rejected"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RESULT_FIELDS = (
    "schema_version",
    "evidence_tier",
    "method_id",
    "arm_or_candidate_id",
    "canonical_dataset_id",
    "legacy_alias",
    "budget",
    "selector_seeds",
    "downstream_seed",
    "case_manifest_sha256",
    "event_manifest_sha256",
    "config_sha256",
    "code_revision",
    "environment_manifest_sha256",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "scored_cases",
    "total_cases",
    "candidate_count",
    "validity_status",
    "acceptance_status",
    "runtime_seconds",
    "denominator",
    "per_case_rankings",
)
DENOMINATOR_FIELDS = (
    "view_id",
    "fault_only",
    "total_cases",
    "admitted_cases",
    "scored_cases",
    "scored_case_ids",
    "exclusion_counts",
    "target_recall",
)
RANKING_FIELDS = (
    "case_id",
    "case_kind",
    "scored",
    "status",
    "ranked_candidates",
    "targets",
)


def _require_fields(payload: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError("%s missing required fields: %s" % (context, ", ".join(missing)))


def _sha256(value: Any, context: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError("%s must be a lowercase SHA-256 digest" % context)
    return text


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % context)
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError("%s must be finite" % context)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be finite" % context)
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % context)
    return number


def _rate(value: Any, context: str) -> float:
    number = _finite(value, context)
    if not 0.0 <= number <= 1.0:
        raise ValueError("%s must be within [0, 1]" % context)
    return number


def recompute_hit_at_k(
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical_dataset_id: str,
    candidate_count: int,
) -> Dict[str, Any]:
    seen_cases: Set[str] = set()
    scored_case_ids: List[str] = []
    hits = {1: 0, 3: 0, 5: 0}
    for index, row in enumerate(rows):
        _require_fields(row, RANKING_FIELDS, "ranking row %d" % index)
        case_id = str(row["case_id"])
        if not case_id.startswith(canonical_dataset_id + "::"):
            raise ValueError("ranking row %d case_id is not canonical" % index)
        if case_id in seen_cases:
            raise ValueError("duplicate per-case ranking for %s" % case_id)
        seen_cases.add(case_id)
        if not isinstance(row["scored"], bool):
            raise ValueError("ranking row %d scored must be boolean" % index)
        if not row["scored"]:
            continue
        if str(row["case_kind"]) != "fault":
            raise ValueError("RCL denominator must be fault-only")
        ranked = [str(item) for item in row["ranked_candidates"]]
        targets = {str(item) for item in row["targets"]}
        if not ranked or len(ranked) > candidate_count or len(ranked) != len(set(ranked)):
            raise ValueError("ranking row %d has an invalid candidate ranking" % index)
        if not targets:
            raise ValueError("ranking row %d has no authoritative target" % index)
        scored_case_ids.append(case_id)
        for k in hits:
            hits[k] += int(bool(targets.intersection(ranked[:k])))
    scored_cases = len(scored_case_ids)
    if not scored_cases:
        raise ValueError("per-case rankings contain no scored fault cases")
    return {
        "hit_at_1": hits[1] / scored_cases,
        "hit_at_3": hits[3] / scored_cases,
        "hit_at_5": hits[5] / scored_cases,
        "scored_cases": scored_cases,
        "scored_case_ids": scored_case_ids,
    }


def _validate_denominator(
    denominator: Mapping[str, Any],
    *,
    total_cases: int,
    scored_cases: int,
) -> Dict[str, Any]:
    _require_fields(denominator, DENOMINATOR_FIELDS, "denominator")
    if not str(denominator["view_id"]).strip():
        raise ValueError("denominator view_id must be non-empty")
    if denominator["fault_only"] is not True:
        raise ValueError("RCL denominator must be fault-only")
    denominator_total = _nonnegative_int(denominator["total_cases"], "denominator total_cases")
    admitted = _nonnegative_int(denominator["admitted_cases"], "denominator admitted_cases")
    denominator_scored = _nonnegative_int(
        denominator["scored_cases"], "denominator scored_cases"
    )
    if denominator_total != total_cases or denominator_scored != scored_cases:
        raise ValueError("result and denominator counts do not match")
    if not denominator_total >= admitted >= denominator_scored:
        raise ValueError("denominator counts must satisfy total >= admitted >= scored")
    scored_ids = [str(item) for item in denominator["scored_case_ids"]]
    if len(scored_ids) != len(set(scored_ids)) or len(scored_ids) != denominator_scored:
        raise ValueError("denominator scored_case_ids do not match scored_cases")
    exclusions = denominator["exclusion_counts"]
    if not isinstance(exclusions, Mapping):
        raise ValueError("denominator exclusion_counts must be an object")
    excluded_count = sum(
        _nonnegative_int(value, "denominator exclusion count %s" % key)
        for key, value in exclusions.items()
    )
    if excluded_count != denominator_total - admitted:
        raise ValueError("denominator exclusion counts do not explain non-admitted cases")
    _rate(denominator["target_recall"], "denominator target_recall")
    return dict(denominator)


def _validate_historical_source(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("historical_source must be an object")
    _require_fields(payload, ("path", "sha256", "protocol_difference"), "historical_source")
    if not str(payload["path"]).strip() or not str(payload["protocol_difference"]).strip():
        raise ValueError("historical_source path and protocol_difference must be non-empty")
    _sha256(payload["sha256"], "historical_source sha256")
    return dict(payload)


def validate_result_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    _require_fields(record, RESULT_FIELDS, "result")
    if str(record["schema_version"]) != RESULT_SCHEMA_VERSION:
        raise ValueError("result schema_version must be %s" % RESULT_SCHEMA_VERSION)
    tier = str(record["evidence_tier"])
    if tier not in EVIDENCE_TIERS:
        raise ValueError("unsupported evidence_tier %r" % tier)
    resolution = resolve_dataset(str(record["canonical_dataset_id"]))
    if resolution.alias_used:
        raise ValueError("result canonical_dataset_id must not use a legacy alias")
    dataset_id = resolution.canonical_id
    if str(record["legacy_alias"]) != resolution.legacy_alias:
        raise ValueError("result legacy_alias mismatch")
    for field in (
        "case_manifest_sha256",
        "event_manifest_sha256",
        "config_sha256",
        "environment_manifest_sha256",
    ):
        _sha256(record[field], "result %s" % field)
    if not str(record["method_id"]).strip() or not str(record["arm_or_candidate_id"]).strip():
        raise ValueError("result method_id and arm_or_candidate_id must be non-empty")
    if not str(record["code_revision"]).strip():
        raise ValueError("result code_revision must be non-empty")
    budget = record["budget"]
    if budget is not None:
        _nonnegative_int(budget, "result budget")
    selector_seeds = record["selector_seeds"]
    if not isinstance(selector_seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in selector_seeds
    ):
        raise ValueError("result selector_seeds must be a list of integers")
    if len(selector_seeds) != len(set(selector_seeds)):
        raise ValueError("result selector_seeds must be unique")
    if isinstance(record["downstream_seed"], bool) or not isinstance(
        record["downstream_seed"], int
    ):
        raise ValueError("result downstream_seed must be an integer")

    hit_values = {
        "hit_at_1": _rate(record["hit_at_1"], "result hit_at_1"),
        "hit_at_3": _rate(record["hit_at_3"], "result hit_at_3"),
        "hit_at_5": _rate(record["hit_at_5"], "result hit_at_5"),
    }
    scored_cases = _nonnegative_int(record["scored_cases"], "result scored_cases")
    total_cases = _nonnegative_int(record["total_cases"], "result total_cases")
    candidate_count = _nonnegative_int(record["candidate_count"], "result candidate_count")
    if not candidate_count or scored_cases > total_cases:
        raise ValueError("result candidate/scored/total counts are invalid")
    runtime = _finite(record["runtime_seconds"], "result runtime_seconds")
    if runtime < 0:
        raise ValueError("result runtime_seconds must be non-negative")
    denominator = record["denominator"]
    if not isinstance(denominator, Mapping):
        raise ValueError("result denominator must be an object")
    validated_denominator = _validate_denominator(
        denominator, total_cases=total_cases, scored_cases=scored_cases
    )
    acceptance = str(record["acceptance_status"])
    if acceptance not in ACCEPTANCE_STATUSES:
        raise ValueError("unsupported acceptance_status %r" % acceptance)

    output = dict(record)
    output["denominator"] = validated_denominator
    if tier == "historical_reference":
        _validate_historical_source(record.get("historical_source"))
        if str(record["validity_status"]) != "reference_only":
            raise ValueError("historical reference validity_status must be reference_only")
        if acceptance == "accepted":
            raise ValueError("historical reference cannot be accepted")
        if record["per_case_rankings"] not in (None, []):
            raise ValueError("historical reference per_case_rankings must be absent")
        output["recomputed_metrics"] = None
        return output

    rankings = record["per_case_rankings"]
    if not isinstance(rankings, list) or not rankings:
        raise ValueError("clean and optimized results require per_case_rankings")
    recomputed = recompute_hit_at_k(
        rankings,
        canonical_dataset_id=dataset_id,
        candidate_count=candidate_count,
    )
    if recomputed["scored_cases"] != scored_cases:
        raise ValueError("scored_cases does not match per-case rankings")
    if set(recomputed["scored_case_ids"]) != set(
        str(item) for item in denominator["scored_case_ids"]
    ):
        raise ValueError("denominator IDs do not match per-case rankings")
    for field, expected in hit_values.items():
        if not math.isclose(recomputed[field], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("%s does not match per-case rankings" % field)
    output["recomputed_metrics"] = {
        "hit_at_1": recomputed["hit_at_1"],
        "hit_at_3": recomputed["hit_at_3"],
        "hit_at_5": recomputed["hit_at_5"],
        "scored_cases": recomputed["scored_cases"],
    }
    return output


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "recompute_hit_at_k",
    "validate_result_record",
]
