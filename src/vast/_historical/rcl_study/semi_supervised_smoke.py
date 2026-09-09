"""Leakage-safe helpers for bounded semi-supervised smoke experiments."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from nexusrcl_rebuild.pseudo_labeling.contracts import (
    EvidenceRecord,
    FrozenPseudoLabelPrediction,
    SanitizedFeatureView,
    StrategyResult,
)


RAW_POOL_SCHEMA_VERSION = "rcl-raw-pseudo-pool-v1"


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_ids(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _ordered_unique_ids(values: Sequence[str]) -> Tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        item = str(value)
        if not item:
            continue
        if item in seen:
            raise ValueError("queried window IDs must be unique")
        result.append(item)
        seen.add(item)
    return tuple(result)


def _finite_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _entity_score(entity: Any) -> float:
    features = getattr(entity, "features", {})
    if not isinstance(features, Mapping):
        return 0.0
    values = [_finite_score(value) for value in features.values()]
    return max(values, default=0.0)


def _prediction_for_view(view: SanitizedFeatureView) -> Dict[str, Any]:
    ranked = sorted(
        (
            (str(entity.entity_id), _entity_score(entity))
            for entity in view.entities
            if str(entity.entity_id)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        raise ValueError(
            "fault view has no candidate entities: %s" % view.window_id
        )
    candidate_ranking = [entity_id for entity_id, _ in ranked]
    top_score = max(0.0, ranked[0][1])
    score_mass = sum(max(0.0, score) for _, score in ranked)
    confidence = top_score / score_mass if score_mass > 0.0 else 0.0
    return {
        "window_id": str(view.window_id),
        "predicted_ids": [ranked[0][0]],
        "confidence": float(confidence),
        "candidate_ranking": candidate_ranking,
    }


def build_raw_pseudo_pool(
    views: Sequence[SanitizedFeatureView],
    *,
    queried_window_ids: Sequence[str],
) -> Dict[str, Any]:
    """Generate deterministic top-one predictions from label-free telemetry."""

    queried = _ordered_unique_ids(queried_window_ids)
    queried_set = set(queried)
    datasets = {str(view.dataset) for view in views}
    if len(datasets) != 1:
        raise ValueError("raw pseudo-pool views must belong to one dataset")
    eligible = []
    seen = set()
    for view in views:
        window_id = str(view.window_id)
        if window_id in seen:
            raise ValueError("duplicate feature view: %s" % window_id)
        seen.add(window_id)
        if window_id in queried_set:
            continue
        if str(view.window_kind) != "fault":
            continue
        eligible.append(_prediction_for_view(view))
    payload = {
        "schema_version": RAW_POOL_SCHEMA_VERSION,
        "dataset": next(iter(datasets)),
        "queried_window_ids": list(queried),
        "predictions": sorted(
            eligible,
            key=lambda prediction: prediction["window_id"],
        ),
    }
    return freeze_raw_pseudo_pool(payload)


def freeze_raw_pseudo_pool(raw_pool: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonicalize a raw pool and bind its immutable semantic hash."""

    if str(raw_pool.get("schema_version", "")) != RAW_POOL_SCHEMA_VERSION:
        raise ValueError("unsupported raw pseudo-pool schema version")
    dataset = str(raw_pool.get("dataset", ""))
    if not dataset:
        raise ValueError("raw pseudo-pool dataset must not be empty")
    queried = _ordered_unique_ids(raw_pool.get("queried_window_ids", ()))
    queried_set = set(queried)
    predictions = []
    seen = set()
    for raw_prediction in raw_pool.get("predictions", ()):
        window_id = str(raw_prediction.get("window_id", ""))
        if not window_id:
            raise ValueError("raw pseudo prediction window_id must not be empty")
        if window_id in queried_set:
            raise ValueError("raw pseudo pool contains a queried window")
        if window_id in seen:
            raise ValueError("raw pseudo pool contains duplicate predictions")
        seen.add(window_id)
        predicted_ids = _stable_ids(raw_prediction.get("predicted_ids", ()))
        if not predicted_ids:
            raise ValueError("raw pseudo prediction must contain predicted_ids")
        confidence = _finite_score(raw_prediction.get("confidence", 0.0))
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("raw pseudo confidence must be within [0, 1]")
        ranking = _stable_ranking(
            raw_prediction.get("candidate_ranking", ()),
            predicted_ids,
        )
        predictions.append(
            {
                "window_id": window_id,
                "predicted_ids": list(predicted_ids),
                "confidence": confidence,
                "candidate_ranking": list(ranking),
            }
        )
    frozen = {
        "schema_version": RAW_POOL_SCHEMA_VERSION,
        "dataset": dataset,
        "queried_window_ids": list(queried),
        "predictions": sorted(
            predictions,
            key=lambda prediction: prediction["window_id"],
        ),
    }
    frozen["pool_sha256"] = _semantic_sha256(frozen)
    return frozen


def _stable_ranking(
    values: Sequence[str],
    predicted_ids: Sequence[str],
) -> Tuple[str, ...]:
    ranking = []
    seen = set()
    for value in values:
        entity_id = str(value)
        if entity_id and entity_id not in seen:
            ranking.append(entity_id)
            seen.add(entity_id)
    for entity_id in predicted_ids:
        if entity_id not in seen:
            ranking.append(entity_id)
            seen.add(entity_id)
    return tuple(ranking)


def _validated_raw_pool(raw_pool: Mapping[str, Any]) -> Dict[str, Any]:
    frozen = freeze_raw_pseudo_pool(raw_pool)
    supplied_hash = str(raw_pool.get("pool_sha256", ""))
    if supplied_hash and supplied_hash != frozen["pool_sha256"]:
        raise ValueError("raw pseudo-pool hash mismatch")
    return frozen


def select_raw_pseudo_pool(
    raw_pool: Mapping[str, Any],
    *,
    authoritative_labels: Mapping[str, Sequence[str]],
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    """Select a confidence threshold using inner-validation labels only."""

    frozen = _validated_raw_pool(raw_pool)
    threshold_values = sorted(
        {_finite_score(value) for value in thresholds}
    )
    if not threshold_values:
        raise ValueError("at least one selector threshold is required")
    if any(value < 0.0 or value > 1.0 for value in threshold_values):
        raise ValueError("selector thresholds must be within [0, 1]")
    validation_ids = set(str(value) for value in authoritative_labels)
    candidates = []
    for threshold in threshold_values:
        selected_validation = [
            prediction
            for prediction in frozen["predictions"]
            if prediction["window_id"] in validation_ids
            and float(prediction["confidence"]) >= threshold
        ]
        if not selected_validation:
            continue
        hits = 0
        for prediction in selected_validation:
            expected = {
                str(value)
                for value in authoritative_labels[prediction["window_id"]]
            }
            ranking = prediction["candidate_ranking"]
            hits += int(bool(ranking) and ranking[0] in expected)
        hit_at_1 = float(hits) / float(len(selected_validation))
        candidates.append(
            {
                "threshold": threshold,
                "hit_at_1": hit_at_1,
                "selected_validation_count": len(selected_validation),
            }
        )
    if not candidates:
        raise ValueError("no threshold selects an inner-validation prediction")
    winner = max(
        candidates,
        key=lambda row: (
            row["hit_at_1"],
            row["selected_validation_count"],
            row["threshold"],
        ),
    )
    selected_ids = sorted(
        prediction["window_id"]
        for prediction in frozen["predictions"]
        if float(prediction["confidence"]) >= winner["threshold"]
    )
    return {
        "schema_version": "rcl-pseudo-selector-v1",
        "raw_pool_sha256": frozen["pool_sha256"],
        "selection_partition": "inner_validation",
        "selected_threshold": winner["threshold"],
        "selected_window_ids": selected_ids,
        "threshold_scoreboard": candidates,
    }


def strategy_result_from_raw_pool(
    raw_pool: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    mode: str,
) -> StrategyResult:
    """Project one immutable raw pool into all- or selected-pseudo supervision."""

    frozen = _validated_raw_pool(raw_pool)
    if mode not in {"all_pseudo", "selected_pseudo"}:
        raise ValueError("unsupported raw-pool adapter mode: %s" % mode)
    if str(selector.get("raw_pool_sha256", "")) != frozen["pool_sha256"]:
        raise ValueError("selector does not reference the supplied raw pool")
    selected_ids = {
        str(value) for value in selector.get("selected_window_ids", ())
    }
    strategy_id = "raw_pool_%s" % mode
    predictions = []
    for row in frozen["predictions"]:
        emitted = mode == "all_pseudo" or row["window_id"] in selected_ids
        evidence = EvidenceRecord(
            source="label_free_raw_pool",
            supported_ids=tuple(row["predicted_ids"]),
            score=float(row["confidence"]),
            details={"candidate_ranking": row["candidate_ranking"]},
        )
        predictions.append(
            FrozenPseudoLabelPrediction(
                window_id=row["window_id"],
                strategy_id=strategy_id,
                predicted_ids=(
                    tuple(row["predicted_ids"]) if emitted else ()
                ),
                confidence=float(row["confidence"]) if emitted else 0.0,
                evidence=(evidence,),
                abstention_reason=None if emitted else "selector_rejected",
            )
        )
    config_payload = {
        "mode": mode,
        "selector_raw_pool_sha256": selector.get("raw_pool_sha256"),
        "selected_threshold": selector.get("selected_threshold"),
        "selected_window_ids": sorted(selected_ids),
    }
    return StrategyResult(
        dataset=frozen["dataset"],
        strategy_id=strategy_id,
        predictions=tuple(predictions),
        queried_window_ids=tuple(frozen["queried_window_ids"]),
        input_hash=frozen["pool_sha256"],
        config_hash=_semantic_sha256(config_payload),
    )


def choose_normal_policy(
    inner_validation_metrics: Mapping[str, Mapping[str, Any]],
) -> str:
    """Choose normal supervision only on a strict inner-validation Hit@1 gain."""

    def hit_at_1(policy: str) -> float:
        metrics = inner_validation_metrics.get(policy)
        if not isinstance(metrics, Mapping):
            raise ValueError("missing normal-policy metrics: %s" % policy)
        raw_value = metrics.get("A@1", metrics.get("hit_at_1"))
        value = _finite_score(raw_value)
        if raw_value is None or not math.isfinite(value):
            raise ValueError("normal-policy Hit@1 must be finite")
        return value

    return (
        "with_normal_class"
        if hit_at_1("with_normal_class") > hit_at_1("fault_only")
        else "fault_only"
    )


__all__ = [
    "RAW_POOL_SCHEMA_VERSION",
    "build_raw_pseudo_pool",
    "choose_normal_policy",
    "freeze_raw_pseudo_pool",
    "select_raw_pseudo_pool",
    "strategy_result_from_raw_pool",
]
