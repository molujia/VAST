from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ResidualValidationError(ValueError):
    """Raised when a conservative residual violates safety or ownership."""


_ARMS = ("oser_meta", "mm_dro", "cope_gate")
_EVIDENCE_STATUSES = ("supported", "missing", "nonfinite", "weak")


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


def _finite(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResidualValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ResidualValidationError(f"{field_name} must be finite")
    return result


def _normalize_string_tuple(values: Sequence[Any], field_name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or "" in result or len(result) != len(set(result)):
        raise ResidualValidationError(f"{field_name} must be unique and nonempty")
    return result


def _validate_base_artifact(base: Mapping[str, Any]) -> dict[str, Any]:
    if base.get("backend_id") != "pairwise_linear":
        raise ResidualValidationError("base artifact backend is not pairwise_linear")
    query_hash = str(base.get("query_plan_sha256", ""))
    model_hash = str(base.get("model_sha256", ""))
    score_hash = str(base.get("score_artifact_sha256", ""))
    if any(len(value) != 64 for value in (query_hash, model_hash, score_hash)):
        raise ResidualValidationError("base artifact ownership hashes are invalid")
    candidates = {
        str(case_id): _normalize_string_tuple(values, f"candidates[{case_id}]")
        for case_id, values in dict(base.get("candidate_ids_by_case", {})).items()
    }
    scores = {
        str(case_id): {str(candidate): _finite(value, "base score") for candidate, value in values.items()}
        for case_id, values in dict(base.get("scores_by_case", {})).items()
    }
    rankings = {
        str(case_id): [str(value) for value in values]
        for case_id, values in dict(base.get("rankings_by_case", {})).items()
    }
    targets = {
        str(case_id): _normalize_string_tuple(values, f"targets[{case_id}]")
        for case_id, values in dict(base.get("targets_by_case", {})).items()
    }
    if not candidates or not (
        set(candidates) == set(scores) == set(rankings) == set(targets)
    ):
        raise ResidualValidationError("base artifact case coverage drift")
    for case_id, expected_order in candidates.items():
        expected = set(expected_order)
        if set(scores[case_id]) != expected or set(rankings[case_id]) != expected:
            raise ResidualValidationError(f"base candidate coverage drift for {case_id}")
        if len(rankings[case_id]) != len(expected):
            raise ResidualValidationError(f"base ranking duplicates candidates for {case_id}")
        if not set(targets[case_id]) <= expected:
            raise ResidualValidationError(f"base valid positive missing for {case_id}")
    return {
        "query_plan_sha256": query_hash,
        "model_sha256": model_hash,
        "score_artifact_sha256": score_hash,
        "candidates": candidates,
        "scores": scores,
        "rankings": rankings,
        "targets": targets,
    }


def _validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    arm = str(profile.get("arm", ""))
    if arm not in _ARMS:
        raise ResidualValidationError(f"unknown residual arm: {arm}")
    profile_id = str(profile.get("profile_id", "")).strip()
    if not profile_id:
        raise ResidualValidationError("residual profile_id must be nonempty")
    cap = _finite(profile.get("residual_cap"), "residual_cap")
    threshold = _finite(profile.get("gate_threshold"), "gate_threshold")
    if not 0.0 < cap <= 1.0:
        raise ResidualValidationError("residual_cap must lie in (0,1]")
    if not 0.0 <= threshold <= 1.0:
        raise ResidualValidationError("gate_threshold must lie in [0,1]")
    return {
        "arm": arm,
        "profile_id": profile_id,
        "residual_cap": cap,
        "gate_threshold": threshold,
    }


def _validate_checkpoint(
    checkpoint_ref: Mapping[str, Any], arm: str, query_plan_sha256: str
) -> dict[str, str]:
    owner = str(checkpoint_ref.get("owner_arm", ""))
    checkpoint_sha = str(checkpoint_ref.get("checkpoint_sha256", ""))
    query_hash = str(checkpoint_ref.get("query_plan_sha256", ""))
    if owner != arm or query_hash != query_plan_sha256 or len(checkpoint_sha) != 64:
        raise ResidualValidationError("checkpoint ownership does not match arm/query plan")
    return {
        "owner_arm": owner,
        "checkpoint_sha256": checkpoint_sha,
        "query_plan_sha256": query_hash,
    }


def _evidence_disposition(
    status: str,
    evidence_value: Any,
    gate: float,
    threshold: float,
) -> tuple[float, str, float | None, bool]:
    if status not in _EVIDENCE_STATUSES:
        raise ResidualValidationError(f"unknown evidence status: {status}")
    evidence_nonfinite = False
    normalized_evidence: float | None
    if evidence_value is None:
        normalized_evidence = None
    else:
        try:
            normalized_evidence = float(evidence_value)
        except (TypeError, ValueError):
            normalized_evidence = None
        if normalized_evidence is not None and not math.isfinite(normalized_evidence):
            evidence_nonfinite = True
            normalized_evidence = None
    if status == "missing":
        return 0.0, "missing_evidence", normalized_evidence, evidence_nonfinite
    if status == "nonfinite" or evidence_nonfinite:
        return 0.0, "nonfinite_evidence", normalized_evidence, True
    if status == "weak":
        return 0.0, "weak_evidence", normalized_evidence, evidence_nonfinite
    if normalized_evidence is None:
        return 0.0, "missing_evidence", None, evidence_nonfinite
    if gate == 0.0:
        return 0.0, "gate_zero", normalized_evidence, evidence_nonfinite
    if gate < threshold:
        return 0.0, "gate_below_threshold", normalized_evidence, evidence_nonfinite
    return gate, "active", normalized_evidence, evidence_nonfinite


@dataclass(frozen=True)
class ConservativeResidualOutput:
    artifact: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.artifact))


def apply_conservative_residual(
    base_artifact: Mapping[str, Any],
    residual_rows: Sequence[Mapping[str, Any]],
    selected_profile: Mapping[str, Any],
    checkpoint_ref: Mapping[str, Any],
    artifact_role: str,
) -> ConservativeResidualOutput:
    base = _validate_base_artifact(base_artifact)
    profile = _validate_profile(selected_profile)
    checkpoint = _validate_checkpoint(
        checkpoint_ref, profile["arm"], base["query_plan_sha256"]
    )
    expected_keys = {
        (case_id, candidate_id)
        for case_id, values in base["candidates"].items()
        for candidate_id in values
    }
    normalized_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in residual_rows:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        key = (case_id, candidate_id)
        if key in normalized_rows:
            raise ResidualValidationError(f"duplicate residual candidate row: {key}")
        if str(row.get("arm", "")) != profile["arm"]:
            raise ResidualValidationError("residual row arm ownership mismatch")
        residual = _finite(row.get("residual"), "residual")
        gate = _finite(row.get("gate"), "gate")
        if abs(residual) > profile["residual_cap"] + 1e-12:
            raise ResidualValidationError("residual cap violation")
        if not 0.0 <= gate <= 1.0:
            raise ResidualValidationError("gate must lie in [0,1]")
        effective_gate, reason, evidence, evidence_nonfinite = _evidence_disposition(
            str(row.get("evidence_status", "")),
            row.get("evidence_value"),
            gate,
            profile["gate_threshold"],
        )
        normalized_rows[key] = {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "raw_residual": residual,
            "declared_gate": gate,
            "effective_gate": effective_gate,
            "no_op_reason": reason,
            "evidence_status": str(row.get("evidence_status", "")),
            "evidence_value": evidence,
            "evidence_was_nonfinite": evidence_nonfinite,
        }
    if set(normalized_rows) != expected_keys:
        missing = len(expected_keys - set(normalized_rows))
        extra = len(set(normalized_rows) - expected_keys)
        raise ResidualValidationError(
            f"residual candidate coverage drift: missing={missing}, extra={extra}"
        )

    final_scores: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}
    audit_rows = []
    no_op_case_count = 0
    active_candidate_count = 0
    for case_id, candidate_order in base["candidates"].items():
        case_active = False
        case_scores: dict[str, float] = {}
        for candidate_id in candidate_order:
            row = normalized_rows[(case_id, candidate_id)]
            base_score = base["scores"][case_id][candidate_id]
            if row["effective_gate"] == 0.0:
                final_score = base_score
            else:
                case_active = True
                active_candidate_count += 1
                clipped = max(
                    -profile["residual_cap"],
                    min(profile["residual_cap"], row["raw_residual"]),
                )
                final_score = base_score + row["effective_gate"] * clipped
            if not math.isfinite(final_score):
                raise ResidualValidationError("final residual score is non-finite")
            case_scores[candidate_id] = float(final_score)
            audit_rows.append(
                {
                    **row,
                    "base_score": base_score,
                    "final_score": float(final_score),
                }
            )
        final_scores[case_id] = case_scores
        if not case_active:
            no_op_case_count += 1
            rankings[case_id] = list(base["rankings"][case_id])
        else:
            rankings[case_id] = sorted(
                candidate_order,
                key=lambda candidate_id: (
                    -case_scores[candidate_id],
                    -base["scores"][case_id][candidate_id],
                    candidate_id,
                ),
            )
    identity = {
        "schema_version": "conservative-lofo-residual-output-v1",
        "artifact_role": str(artifact_role),
        "arm": profile["arm"],
        "profile_id": profile["profile_id"],
        "residual_cap": profile["residual_cap"],
        "gate_threshold": profile["gate_threshold"],
        "checkpoint_owner": checkpoint["owner_arm"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "query_plan_sha256": base["query_plan_sha256"],
        "base_model_sha256": base["model_sha256"],
        "base_score_artifact_sha256": base["score_artifact_sha256"],
        "candidate_ids_by_case": base["candidates"],
        "targets_by_case": base["targets"],
        "base_scores_by_case": base["scores"],
        "final_scores_by_case": final_scores,
        "rankings_by_case": rankings,
        "audit_rows": audit_rows,
        "no_op_case_count": no_op_case_count,
        "active_candidate_count": active_candidate_count,
    }
    artifact = {**identity, "artifact_sha256": _semantic_hash(identity)}
    output = ConservativeResidualOutput(artifact)
    validate_conservative_residual_output(
        output,
        base_artifact,
        expected_arm=profile["arm"],
        checkpoint_ref=checkpoint_ref,
    )
    return output


def validate_conservative_residual_output(
    output: ConservativeResidualOutput,
    base_artifact: Mapping[str, Any],
    expected_arm: str,
    checkpoint_ref: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = output.to_dict()
    base = _validate_base_artifact(base_artifact)
    arm = str(expected_arm)
    checkpoint = _validate_checkpoint(checkpoint_ref, arm, base["query_plan_sha256"])
    if artifact.get("schema_version") != "conservative-lofo-residual-output-v1":
        raise ResidualValidationError("unexpected residual output schema")
    if artifact.get("arm") != arm or artifact.get("checkpoint_owner") != arm:
        raise ResidualValidationError("residual output arm/checkpoint ownership mismatch")
    if artifact.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
        raise ResidualValidationError("residual checkpoint ownership mismatch")
    if artifact.get("query_plan_sha256") != base["query_plan_sha256"]:
        raise ResidualValidationError("residual query-plan ownership mismatch")
    if artifact.get("base_model_sha256") != base["model_sha256"] or artifact.get(
        "base_score_artifact_sha256"
    ) != base["score_artifact_sha256"]:
        raise ResidualValidationError("residual base-score ownership mismatch")
    candidates = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(artifact.get("candidate_ids_by_case", {})).items()
    }
    targets = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(artifact.get("targets_by_case", {})).items()
    }
    if candidates != base["candidates"] or targets != base["targets"]:
        raise ResidualValidationError("residual candidate/target coverage drift")
    final_scores = dict(artifact.get("final_scores_by_case", {}))
    rankings = dict(artifact.get("rankings_by_case", {}))
    if set(final_scores) != set(candidates) or set(rankings) != set(candidates):
        raise ResidualValidationError("residual output case coverage drift")
    audit_by_key = {}
    for row in artifact.get("audit_rows", ()):
        key = (str(row.get("case_id", "")), str(row.get("candidate_id", "")))
        if key in audit_by_key:
            raise ResidualValidationError("residual audit duplicates candidates")
        audit_by_key[key] = row
    expected_keys = {
        (case_id, candidate_id)
        for case_id, values in candidates.items()
        for candidate_id in values
    }
    if set(audit_by_key) != expected_keys:
        raise ResidualValidationError("residual audit candidate coverage drift")
    cap = _finite(artifact.get("residual_cap"), "residual_cap")
    threshold = _finite(artifact.get("gate_threshold"), "gate_threshold")
    for case_id, candidate_ids in candidates.items():
        if set(final_scores[case_id]) != set(candidate_ids) or set(rankings[case_id]) != set(
            candidate_ids
        ):
            raise ResidualValidationError(f"residual candidates incomplete for {case_id}")
        case_active = False
        for candidate_id in candidate_ids:
            row = audit_by_key[(case_id, candidate_id)]
            residual = _finite(row.get("raw_residual"), "raw_residual")
            declared_gate = _finite(row.get("declared_gate"), "declared_gate")
            effective_gate = _finite(row.get("effective_gate"), "effective_gate")
            if abs(residual) > cap + 1e-12 or not 0.0 <= declared_gate <= 1.0 or not 0.0 <= effective_gate <= 1.0:
                raise ResidualValidationError("residual audit numeric bounds drift")
            base_score = base["scores"][case_id][candidate_id]
            final_score = _finite(final_scores[case_id][candidate_id], "final_score")
            if effective_gate == 0.0:
                if final_score != base_score:
                    raise ResidualValidationError("inactive residual changed a score")
            else:
                case_active = True
                expected = base_score + effective_gate * max(-cap, min(cap, residual))
                if abs(final_score - expected) > 1e-12:
                    raise ResidualValidationError("active residual formula drift")
        if not case_active:
            if list(rankings[case_id]) != list(base["rankings"][case_id]):
                raise ResidualValidationError("inactive residual changed a ranking")
        else:
            expected_ranking = sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    -float(final_scores[case_id][candidate_id]),
                    -base["scores"][case_id][candidate_id],
                    candidate_id,
                ),
            )
            if list(rankings[case_id]) != expected_ranking:
                raise ResidualValidationError("active residual ranking drift")
    if not 0.0 <= threshold <= 1.0:
        raise ResidualValidationError("residual gate threshold drift")
    identity = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _semantic_hash(identity):
        raise ResidualValidationError("residual artifact hash drift")
    return {
        "valid": True,
        "arm": arm,
        "case_count": len(candidates),
        "candidate_count": len(expected_keys),
        "artifact_sha256": artifact["artifact_sha256"],
    }


__all__ = [
    "ConservativeResidualOutput",
    "ResidualValidationError",
    "apply_conservative_residual",
    "validate_conservative_residual_output",
]
