from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .service_continuous_screen_scoring import (
    METRICS,
    ScreenScoringValidationError,
    validate_ordinary_score,
    validate_seven_type_lofo_score,
)


class ScreenGateValidationError(ValueError):
    """Raised when LOFO and ordinary evidence are mixed or staged incorrectly."""


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _arm(value: Any) -> str:
    arm = str(value).strip()
    if not arm or arm == "baseline":
        raise ScreenGateValidationError("gate requires a non-baseline arm ID")
    return arm


def _case_ids(score: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        type_score["fault_type"]: tuple(
            case["case_id"] for case in type_score["per_case"]
        )
        for type_score in score["type_scores"]
    }


def evaluate_strict_lofo_gate(
    *,
    arm_id: str,
    baseline_score: Mapping[str, Any],
    arm_score: Mapping[str, Any],
) -> dict[str, Any]:
    arm = _arm(arm_id)
    try:
        baseline = validate_seven_type_lofo_score(baseline_score)
        enhanced = validate_seven_type_lofo_score(arm_score)
    except ScreenScoringValidationError as exc:
        raise ScreenGateValidationError(
            f"LOFO score schema or count evidence is invalid: {exc}"
        ) from exc
    if _case_ids(baseline) != _case_ids(enhanced):
        raise ScreenGateValidationError("matched LOFO case membership drift")
    deltas = {
        metric: float(enhanced["primary_type_macro"][metric])
        - float(baseline["primary_type_macro"][metric])
        for metric in METRICS
    }
    improvement = deltas["mrr"] > 0.0
    identity = {
        "schema_version": "service-continuous-strict-lofo-gate-v1",
        "arm_id": arm,
        "comparison_protocol": "matched_strict_lofo_arm_minus_baseline",
        "baseline_score_sha256": baseline["dataset_score_sha256"],
        "arm_score_sha256": enhanced["dataset_score_sha256"],
        "macro_metric_deltas": deltas,
        "macro_mrr_delta": deltas["mrr"],
        "strict_improvement": improvement,
        "status": (
            "eligible_for_ordinary_safety_screen"
            if improvement
            else "rejected_non_improving_lofo"
        ),
    }
    return {**identity, "gate_sha256": _semantic_hash(identity)}


def evaluate_ordinary_safety_gate(
    *,
    arm_id: str,
    baseline_score: Mapping[str, Any],
    arm_score: Mapping[str, Any],
) -> dict[str, Any]:
    arm = _arm(arm_id)
    try:
        baseline = validate_ordinary_score(baseline_score)
        enhanced = validate_ordinary_score(arm_score)
    except ScreenScoringValidationError as exc:
        raise ScreenGateValidationError(
            f"ordinary score schema or count evidence is invalid: {exc}"
        ) from exc
    baseline_ids = tuple(case["case_id"] for case in baseline["per_case"])
    arm_ids = tuple(case["case_id"] for case in enhanced["per_case"])
    if baseline_ids != arm_ids:
        raise ScreenGateValidationError("matched ordinary case membership drift")
    arm_minus_baseline = {
        metric: float(enhanced["metrics"][metric])
        - float(baseline["metrics"][metric])
        for metric in METRICS
    }
    declines = {
        metric: float(baseline["metrics"][metric])
        - float(enhanced["metrics"][metric])
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5")
    }
    mean_decline = math.fsum(declines.values()) / 3.0
    safe = mean_decline <= 0.05 + 1e-12
    identity = {
        "schema_version": "service-continuous-ordinary-safety-gate-v1",
        "arm_id": arm,
        "comparison_protocol": "matched_ordinary_arm_minus_baseline",
        "baseline_score_sha256": baseline["ordinary_score_sha256"],
        "arm_score_sha256": enhanced["ordinary_score_sha256"],
        "arm_minus_baseline_deltas": arm_minus_baseline,
        "baseline_minus_arm_hit_declines": declines,
        "mean_hit_decline": mean_decline,
        "threshold": 0.05,
        "equality_accepted": True,
        "ordinary_safe": safe,
        "status": "ordinary_safe" if safe else "ordinary_unsafe",
    }
    return {**identity, "gate_sha256": _semantic_hash(identity)}


def _validated_gate(
    value: Any, schema: str, context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise ScreenGateValidationError(f"{context} schema drift")
    gate = deepcopy(dict(value))
    supplied = str(gate.pop("gate_sha256", ""))
    if supplied != _semantic_hash(gate):
        raise ScreenGateValidationError(f"{context} hash drift")
    return {**gate, "gate_sha256": supplied}


def validate_staged_decision(
    *,
    requested_stage: str,
    lofo_gate: Mapping[str, Any],
    ordinary_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage = str(requested_stage).strip()
    lofo = _validated_gate(
        lofo_gate, "service-continuous-strict-lofo-gate-v1", "LOFO gate"
    )
    if not lofo.get("strict_improvement"):
        raise ScreenGateValidationError(
            "ordinary launch is forbidden for a non-improving LOFO arm"
        )
    if stage == "ordinary":
        identity = {
            "schema_version": "service-continuous-staged-decision-v1",
            "requested_stage": stage,
            "arm_id": lofo["arm_id"],
            "lofo_gate_sha256": lofo["gate_sha256"],
            "ordinary_gate_sha256": None,
            "authorized": True,
        }
    elif stage == "later":
        ordinary = _validated_gate(
            ordinary_gate,
            "service-continuous-ordinary-safety-gate-v1",
            "ordinary gate",
        )
        if ordinary["arm_id"] != lofo["arm_id"] or not ordinary.get(
            "ordinary_safe"
        ):
            raise ScreenGateValidationError(
                "later stages require the same arm to pass ordinary safety"
            )
        identity = {
            "schema_version": "service-continuous-staged-decision-v1",
            "requested_stage": stage,
            "arm_id": lofo["arm_id"],
            "lofo_gate_sha256": lofo["gate_sha256"],
            "ordinary_gate_sha256": ordinary["gate_sha256"],
            "authorized": True,
        }
    else:
        raise ScreenGateValidationError("requested stage must be ordinary or later")
    return {**identity, "decision_sha256": _semantic_hash(identity)}


__all__ = [
    "ScreenGateValidationError",
    "evaluate_ordinary_safety_gate",
    "evaluate_strict_lofo_gate",
    "validate_staged_decision",
]
