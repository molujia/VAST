from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class ScreenScoringValidationError(ValueError):
    """Raised when service-continuity scores cannot be rebuilt from ranks."""


METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")
FROZEN_RCABENCH_SUPPORT = {
    "NetworkDelay": 21,
    "HTTPResponsePatchBody": 4,
    "HTTPResponseDelay": 89,
    "HTTPRequestAbort": 60,
    "PodKill": 10,
    "JVMMemoryStress": 171,
    "NetworkBandwidth": 42,
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


def _metrics(ranks: Sequence[int]) -> dict[str, float]:
    if not ranks:
        raise ScreenScoringValidationError("cannot score empty ranks")
    count = len(ranks)
    return {
        "hit_at_1": sum(rank <= 1 for rank in ranks) / count,
        "hit_at_3": sum(rank <= 3 for rank in ranks) / count,
        "hit_at_5": sum(rank <= 5 for rank in ranks) / count,
        "mrr": math.fsum(1.0 / rank for rank in ranks) / count,
    }


def _score_rows(case_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    if isinstance(case_rows, (str, bytes)) or not isinstance(case_rows, Sequence) or not case_rows:
        raise ScreenScoringValidationError("case rows must be a nonempty sequence")
    seen = set()
    per_case = []
    ranks = []
    for raw in case_rows:
        if not isinstance(raw, Mapping):
            raise ScreenScoringValidationError("case row must be a mapping")
        case_id = str(raw.get("case_id", "")).strip()
        ranking = tuple(str(value).strip() for value in raw.get("ranking", ()))
        targets = tuple(str(value).strip() for value in raw.get("targets", ()))
        if (
            not case_id
            or case_id in seen
            or not ranking
            or any(not value for value in ranking)
            or len(ranking) != len(set(ranking))
            or not targets
            or any(not value for value in targets)
            or len(targets) != len(set(targets))
            or not set(targets) <= set(ranking)
        ):
            raise ScreenScoringValidationError(
                "case, ranking, or target membership drift"
            )
        seen.add(case_id)
        positive_ranks = tuple(sorted(ranking.index(target) + 1 for target in targets))
        rank = positive_ranks[0]
        ranks.append(rank)
        per_case.append(
            {
                "case_id": case_id,
                "candidate_count": len(ranking),
                "ranking": ranking,
                "targets": targets,
                "positive_ranks": positive_ranks,
                "best_positive_rank": rank,
                "hit_at_1": int(rank <= 1),
                "hit_at_3": int(rank <= 3),
                "hit_at_5": int(rank <= 5),
                "reciprocal_rank": 1.0 / rank,
            }
        )
    return per_case, ranks


def _count_identity(schema_version: str, case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_case, ranks = _score_rows(case_rows)
    return {
        "schema_version": schema_version,
        "case_count": len(ranks),
        "hit_at_1_count": sum(rank <= 1 for rank in ranks),
        "hit_at_3_count": sum(rank <= 3 for rank in ranks),
        "hit_at_5_count": sum(rank <= 5 for rank in ranks),
        "reciprocal_rank_sum": math.fsum(1.0 / rank for rank in ranks),
        "metrics": _metrics(ranks),
        "per_case": per_case,
    }


def build_count_authoritative_score(
    fault_type: str, case_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fault = str(fault_type).strip()
    if not fault:
        raise ScreenScoringValidationError("fault_type must be nonempty")
    identity = {
        **_count_identity("service-continuous-type-score-v1", case_rows),
        "fault_type": fault,
    }
    return {**identity, "type_score_sha256": _semantic_hash(identity)}


def build_ordinary_count_score(
    case_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity = _count_identity("service-continuous-ordinary-score-v1", case_rows)
    return {**identity, "ordinary_score_sha256": _semantic_hash(identity)}


def _validate_count_score(
    score: Mapping[str, Any], *, schema: str, hash_field: str
) -> dict[str, Any]:
    if not isinstance(score, Mapping) or score.get("schema_version") != schema:
        raise ScreenScoringValidationError("score schema drift")
    value = deepcopy(dict(score))
    supplied = str(value.pop(hash_field, ""))
    if supplied != _semantic_hash(value):
        raise ScreenScoringValidationError("score content hash drift")
    count = value.get("case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ScreenScoringValidationError("score denominator drift")
    per_case = value.get("per_case")
    if not isinstance(per_case, Sequence) or len(per_case) != count:
        raise ScreenScoringValidationError("score per-case denominator drift")
    reconstructed = {
        "hit_at_1": int(value["hit_at_1_count"]) / count,
        "hit_at_3": int(value["hit_at_3_count"]) / count,
        "hit_at_5": int(value["hit_at_5_count"]) / count,
        "mrr": float(value["reciprocal_rank_sum"]) / count,
    }
    if any(
        not math.isclose(float(value["metrics"][metric]), expected, abs_tol=1e-12)
        for metric, expected in reconstructed.items()
    ):
        raise ScreenScoringValidationError("score is not count-authoritative")
    return {**value, hash_field: supplied}


def validate_type_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_count_score(
        score,
        schema="service-continuous-type-score-v1",
        hash_field="type_score_sha256",
    )


def validate_ordinary_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_count_score(
        score,
        schema="service-continuous-ordinary-score-v1",
        hash_field="ordinary_score_sha256",
    )


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_seven_type_lofo_score(
    type_scores: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 1000
) -> dict[str, Any]:
    if isinstance(type_scores, (str, bytes)) or not isinstance(type_scores, Sequence):
        raise ScreenScoringValidationError("type_scores must be a sequence")
    validated = [validate_type_score(score) for score in type_scores]
    by_type = {str(score.get("fault_type", "")): score for score in validated}
    if set(by_type) != set(FROZEN_RCABENCH_SUPPORT) or len(by_type) != 7:
        raise ScreenScoringValidationError("frozen seven-type coverage drift")
    ordered = [by_type[fault_type] for fault_type in FROZEN_RCABENCH_SUPPORT]
    support = {fault_type: int(by_type[fault_type]["case_count"]) for fault_type in by_type}
    if support != FROZEN_RCABENCH_SUPPORT:
        raise ScreenScoringValidationError("frozen seven-type denominator drift")
    primary = {
        metric: math.fsum(float(score["metrics"][metric]) for score in ordered) / 7.0
        for metric in METRICS
    }
    pooled_count = sum(score["case_count"] for score in ordered)
    if pooled_count != 397:
        raise ScreenScoringValidationError("pooled RCABench denominator must be 397")
    diagnostic = {
        "hit_at_1": sum(score["hit_at_1_count"] for score in ordered) / pooled_count,
        "hit_at_3": sum(score["hit_at_3_count"] for score in ordered) / pooled_count,
        "hit_at_5": sum(score["hit_at_5_count"] for score in ordered) / pooled_count,
        "mrr": math.fsum(score["reciprocal_rank_sum"] for score in ordered)
        / pooled_count,
    }
    samples = int(bootstrap_samples)
    if samples <= 0:
        raise ScreenScoringValidationError("bootstrap_samples must be positive")
    generator = random.Random(42)
    bootstrapped = {metric: [] for metric in METRICS}
    ranks_by_type = {
        score["fault_type"]: [case["best_positive_rank"] for case in score["per_case"]]
        for score in ordered
    }
    for _ in range(samples):
        replicate_type_metrics = []
        for fault_type in FROZEN_RCABENCH_SUPPORT:
            ranks = ranks_by_type[fault_type]
            sampled = [ranks[generator.randrange(len(ranks))] for _ in ranks]
            replicate_type_metrics.append(_metrics(sampled))
        for metric in METRICS:
            bootstrapped[metric].append(
                math.fsum(value[metric] for value in replicate_type_metrics) / 7.0
            )
    identity = {
        "schema_version": "service-continuous-seven-type-lofo-score-v1",
        "dataset_id": "rcabench",
        "type_scores": ordered,
        "support_by_fault_type": dict(FROZEN_RCABENCH_SUPPORT),
        "primary_type_macro": primary,
        "diagnostic_pooled": diagnostic,
        "pooled_case_count": pooled_count,
        "bootstrap": {
            "samples": samples,
            "seed": 42,
            "intervals": {
                metric: {
                    "lower_2p5": _quantile(bootstrapped[metric], 0.025),
                    "upper_97p5": _quantile(bootstrapped[metric], 0.975),
                }
                for metric in METRICS
            },
        },
    }
    return {**identity, "dataset_score_sha256": _semantic_hash(identity)}


def validate_seven_type_lofo_score(score: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(score, Mapping) or score.get("schema_version") != (
        "service-continuous-seven-type-lofo-score-v1"
    ):
        raise ScreenScoringValidationError("LOFO score schema drift")
    value = deepcopy(dict(score))
    supplied = str(value.pop("dataset_score_sha256", ""))
    if supplied != _semantic_hash(value):
        raise ScreenScoringValidationError("LOFO score hash drift")
    if value.get("support_by_fault_type") != FROZEN_RCABENCH_SUPPORT or value.get(
        "pooled_case_count"
    ) != 397:
        raise ScreenScoringValidationError("LOFO frozen denominator drift")
    for type_score in value.get("type_scores", ()):
        validate_type_score(type_score)
    return {**value, "dataset_score_sha256": supplied}


__all__ = [
    "FROZEN_RCABENCH_SUPPORT",
    "METRICS",
    "ScreenScoringValidationError",
    "build_count_authoritative_score",
    "build_ordinary_count_score",
    "build_seven_type_lofo_score",
    "validate_ordinary_score",
    "validate_seven_type_lofo_score",
    "validate_type_score",
]
