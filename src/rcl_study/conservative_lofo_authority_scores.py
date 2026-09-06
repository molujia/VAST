from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence


class AuthorityScoreValidationError(ValueError):
    """Raised when an immutable authority score bundle is incomplete or drifts."""


BACKEND_ID = "pairwise_linear"
TRAINER_ID = "nexusrcl_pairwise_classifier_ranker"
ROLES = ("support", "evaluation")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _valid_hash(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _ordered_rows(
    artifact: Mapping[str, Any],
    *,
    role: str,
    dataset_id: str,
    case_ids: Sequence[Any],
    query_plan_sha256: str,
    feature_order_sha256: str,
    model_sha256: str,
) -> tuple[dict[str, Any], ...]:
    source = deepcopy(dict(artifact))
    if source.get("schema_version") != "conservative-lofo-base-score-artifact-v1":
        raise AuthorityScoreValidationError("unexpected base score artifact schema")
    if source.get("artifact_role") != role:
        raise AuthorityScoreValidationError(f"{role} artifact role drift")
    if source.get("dataset_id") != dataset_id:
        raise AuthorityScoreValidationError(f"{role} dataset drift")
    if source.get("backend_id") != BACKEND_ID:
        raise AuthorityScoreValidationError(f"{role} backend drift")
    for field, expected in (
        ("query_plan_sha256", query_plan_sha256),
        ("feature_order_sha256", feature_order_sha256),
        ("model_sha256", model_sha256),
    ):
        if source.get(field) != expected:
            raise AuthorityScoreValidationError(f"{role} {field} drift")
    if not _valid_hash(source.get("score_artifact_sha256")):
        raise AuthorityScoreValidationError(f"{role} score artifact hash is invalid")
    expected_cases = tuple(str(value) for value in case_ids)
    if not expected_cases or len(expected_cases) != len(set(expected_cases)):
        raise AuthorityScoreValidationError(f"{role} case order is invalid")
    candidates = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(source.get("candidate_ids_by_case", {})).items()
    }
    scores = {
        str(case_id): {str(candidate): float(value) for candidate, value in dict(values).items()}
        for case_id, values in dict(source.get("scores_by_case", {})).items()
    }
    targets = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(source.get("targets_by_case", {})).items()
    }
    rankings = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(source.get("rankings_by_case", {})).items()
    }
    if (
        tuple(candidates) != expected_cases
        or tuple(scores) != expected_cases
        or tuple(targets) != expected_cases
        or tuple(rankings) != expected_cases
    ):
        raise AuthorityScoreValidationError(f"{role} case order or coverage drift")
    rows = []
    for case_id in expected_cases:
        candidate_ids = candidates[case_id]
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise AuthorityScoreValidationError(f"{role} candidate order is invalid for {case_id}")
        if set(scores[case_id]) != set(candidate_ids) or set(rankings[case_id]) != set(candidate_ids):
            raise AuthorityScoreValidationError(f"{role} candidate coverage drift for {case_id}")
        if not targets[case_id] or not set(targets[case_id]) <= set(candidate_ids):
            raise AuthorityScoreValidationError(f"{role} target coverage drift for {case_id}")
        ordered_scores = tuple(scores[case_id][candidate] for candidate in candidate_ids)
        if not all(math.isfinite(value) for value in ordered_scores):
            raise AuthorityScoreValidationError(f"{role} scores must be finite for {case_id}")
        expected_ranking = tuple(
            candidate_ids[index]
            for index in sorted(
                range(len(candidate_ids)),
                key=lambda index: (-ordered_scores[index], candidate_ids[index]),
            )
        )
        if rankings[case_id] != expected_ranking:
            raise AuthorityScoreValidationError(f"{role} ranking drift for {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "candidate_ids": candidate_ids,
                "targets": targets[case_id],
                "scores": ordered_scores,
                "ranking": expected_ranking,
            }
        )
    return tuple(rows)


def build_authority_score_bundle(
    *,
    dataset_id: str,
    fold_id: str,
    fold_kind: str,
    held_out_fault_type: str | None,
    query_plan_sha256: str,
    feature_order_sha256: str,
    model_sha256: str,
    support_case_ids: Sequence[Any],
    evaluation_case_ids: Sequence[Any],
    support_artifact: Mapping[str, Any],
    evaluation_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = str(dataset_id).strip()
    fold = str(fold_id).strip()
    kind = str(fold_kind).strip()
    if not dataset or not fold or kind not in {"ordinary", "strict_lofo", "pseudo_lofo"}:
        raise AuthorityScoreValidationError("authority bundle ownership is invalid")
    for field, value in (
        ("query plan", query_plan_sha256),
        ("feature order", feature_order_sha256),
        ("model", model_sha256),
    ):
        if not _valid_hash(value):
            raise AuthorityScoreValidationError(f"authority {field} hash is invalid")
    support_ids = tuple(str(value) for value in support_case_ids)
    evaluation_ids = tuple(str(value) for value in evaluation_case_ids)
    if set(support_ids) & set(evaluation_ids):
        raise AuthorityScoreValidationError("authority support/evaluation overlap")
    support_rows = _ordered_rows(
        support_artifact,
        role="support",
        dataset_id=dataset,
        case_ids=support_ids,
        query_plan_sha256=query_plan_sha256,
        feature_order_sha256=feature_order_sha256,
        model_sha256=model_sha256,
    )
    evaluation_rows = _ordered_rows(
        evaluation_artifact,
        role="evaluation",
        dataset_id=dataset,
        case_ids=evaluation_ids,
        query_plan_sha256=query_plan_sha256,
        feature_order_sha256=feature_order_sha256,
        model_sha256=model_sha256,
    )
    identity = {
        "schema_version": "conservative-lofo-authority-score-bundle-v1",
        "dataset_id": dataset,
        "fold_id": fold,
        "fold_kind": kind,
        "held_out_fault_type": None if held_out_fault_type in (None, "") else str(held_out_fault_type),
        "authority_backend_id": BACKEND_ID,
        "authority_trainer_id": TRAINER_ID,
        "query_plan_sha256": str(query_plan_sha256),
        "feature_order_sha256": str(feature_order_sha256),
        "model_sha256": str(model_sha256),
        "score_artifact_sha256": {
            "support": str(support_artifact["score_artifact_sha256"]),
            "evaluation": str(evaluation_artifact["score_artifact_sha256"]),
        },
        "support_case_ids": support_ids,
        "evaluation_case_ids": evaluation_ids,
        "partitions": {"support": support_rows, "evaluation": evaluation_rows},
    }
    bundle = {**identity, "authority_score_bundle_sha256": _hash(identity)}
    validate_authority_score_bundle(bundle)
    return bundle


def validate_authority_score_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(bundle))
    if source.get("schema_version") != "conservative-lofo-authority-score-bundle-v1":
        raise AuthorityScoreValidationError("unexpected authority score bundle schema")
    if source.get("authority_backend_id") != BACKEND_ID:
        raise AuthorityScoreValidationError("authority backend drift")
    if source.get("authority_trainer_id") != TRAINER_ID:
        raise AuthorityScoreValidationError("authority trainer drift")
    for field in (
        "query_plan_sha256",
        "feature_order_sha256",
        "model_sha256",
        "authority_score_bundle_sha256",
    ):
        if not _valid_hash(source.get(field)):
            raise AuthorityScoreValidationError(f"authority {field} is invalid")
    artifact_hashes = dict(source.get("score_artifact_sha256", {}))
    if set(artifact_hashes) != set(ROLES) or not all(_valid_hash(value) for value in artifact_hashes.values()):
        raise AuthorityScoreValidationError("authority score artifact hashes are invalid")
    support_ids = tuple(str(value) for value in source.get("support_case_ids", ()))
    evaluation_ids = tuple(str(value) for value in source.get("evaluation_case_ids", ()))
    if (
        not support_ids
        or not evaluation_ids
        or len(support_ids) != len(set(support_ids))
        or len(evaluation_ids) != len(set(evaluation_ids))
        or set(support_ids) & set(evaluation_ids)
    ):
        raise AuthorityScoreValidationError("authority case ownership is invalid")
    partitions = dict(source.get("partitions", {}))
    if set(partitions) != set(ROLES):
        raise AuthorityScoreValidationError("authority partitions are incomplete")
    for role, case_ids in (("support", support_ids), ("evaluation", evaluation_ids)):
        rows = tuple(dict(value) for value in partitions[role])
        if tuple(str(row.get("case_id", "")) for row in rows) != case_ids:
            raise AuthorityScoreValidationError(f"authority {role} case order drift")
        for row in rows:
            candidates = tuple(str(value) for value in row.get("candidate_ids", ()))
            targets = tuple(str(value) for value in row.get("targets", ()))
            scores = tuple(float(value) for value in row.get("scores", ()))
            ranking = tuple(str(value) for value in row.get("ranking", ()))
            if (
                not candidates
                or len(candidates) != len(set(candidates))
                or len(scores) != len(candidates)
                or set(ranking) != set(candidates)
                or len(ranking) != len(candidates)
            ):
                raise AuthorityScoreValidationError(f"authority {role} candidate coverage drift")
            if not targets or not set(targets) <= set(candidates):
                raise AuthorityScoreValidationError(f"authority {role} target coverage drift")
            if not all(math.isfinite(value) for value in scores):
                raise AuthorityScoreValidationError(f"authority {role} scores must be finite")
            expected_ranking = tuple(
                candidates[index]
                for index in sorted(
                    range(len(candidates)),
                    key=lambda index: (-scores[index], candidates[index]),
                )
            )
            if ranking != expected_ranking:
                raise AuthorityScoreValidationError(f"authority {role} ranking drift")
    identity = {key: value for key, value in source.items() if key != "authority_score_bundle_sha256"}
    if source.get("authority_score_bundle_sha256") != _hash(identity):
        raise AuthorityScoreValidationError("authority score bundle hash drift")
    return {
        "valid": True,
        "dataset_id": str(source["dataset_id"]),
        "fold_id": str(source["fold_id"]),
        "support_case_count": len(support_ids),
        "evaluation_case_count": len(evaluation_ids),
        "model_sha256": str(source["model_sha256"]),
        "authority_score_bundle_sha256": str(source["authority_score_bundle_sha256"]),
    }


def extract_authority_scores(
    bundle: Mapping[str, Any],
    *,
    role: str,
    expected_case_ids: Sequence[Any],
    expected_candidates_by_case: Mapping[str, Sequence[Any]],
) -> dict[str, list[float]]:
    validate_authority_score_bundle(bundle)
    role_id = str(role)
    if role_id not in ROLES:
        raise AuthorityScoreValidationError("unknown authority score role")
    case_field = "support_case_ids" if role_id == "support" else "evaluation_case_ids"
    expected_cases = tuple(str(value) for value in expected_case_ids)
    if tuple(str(value) for value in bundle[case_field]) != expected_cases:
        raise AuthorityScoreValidationError(f"authority {role_id} expected case order drift")
    expected_candidates = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in expected_candidates_by_case.items()
    }
    if tuple(expected_candidates) != expected_cases:
        raise AuthorityScoreValidationError(f"authority {role_id} candidate case order drift")
    result: dict[str, list[float]] = {}
    for row in bundle["partitions"][role_id]:
        case_id = str(row["case_id"])
        if tuple(str(value) for value in row["candidate_ids"]) != expected_candidates[case_id]:
            raise AuthorityScoreValidationError(f"authority {role_id} candidate order drift for {case_id}")
        result[case_id] = [float(value) for value in row["scores"]]
    return result


__all__ = [
    "AuthorityScoreValidationError",
    "BACKEND_ID",
    "TRAINER_ID",
    "build_authority_score_bundle",
    "extract_authority_scores",
    "validate_authority_score_bundle",
]
