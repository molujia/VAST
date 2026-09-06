from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class BaseBridgeValidationError(ValueError):
    """Raised when pairwise-linear authority scores cannot be frozen safely."""


def _ensure_backend_importable() -> None:
    source_root = Path(__file__).resolve().parents[1] / "half_supervise" / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


_ensure_backend_importable()
from nexusrcl_rebuild.training.pairwise_backend import (  # noqa: E402
    PairwiseClassifierRanker,
    train_pairwise_linear_ranker,
)


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


def _normalize_targets(
    targets_by_case: Mapping[str, Sequence[Any]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for case_id, raw_targets in targets_by_case.items():
        targets = tuple(str(value).strip() for value in raw_targets if str(value).strip())
        if not targets or len(targets) != len(set(targets)):
            raise BaseBridgeValidationError(f"invalid positive targets for {case_id}")
        normalized[str(case_id)] = targets
    if not normalized:
        raise BaseBridgeValidationError("targets_by_case must be nonempty")
    return normalized


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]
) -> pd.DataFrame:
    features = tuple(str(value) for value in feature_columns)
    if not features or len(features) != len(set(features)):
        raise BaseBridgeValidationError("feature_columns must be unique and nonempty")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not case_id or not candidate_id:
            raise BaseBridgeValidationError("base bridge row lacks case_id or candidate_id")
        key = (case_id, candidate_id)
        if key in seen:
            raise BaseBridgeValidationError(f"duplicate candidate row: {key}")
        seen.add(key)
        record = {"window_id": case_id, "entity_id": candidate_id}
        for feature in features:
            try:
                value = float(row[feature])
            except (KeyError, TypeError, ValueError) as exc:
                raise BaseBridgeValidationError(
                    f"invalid feature {feature} for {case_id}/{candidate_id}"
                ) from exc
            if not math.isfinite(value):
                raise BaseBridgeValidationError(
                    f"non-finite feature {feature} for {case_id}/{candidate_id}"
                )
            record[feature] = value
        records.append(record)
    if not records:
        raise BaseBridgeValidationError("base bridge rows are empty")
    return pd.DataFrame.from_records(records)


def _candidate_ids(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    return {
        str(case_id): tuple(str(value) for value in group["entity_id"].tolist())
        for case_id, group in frame.groupby("window_id", sort=False)
    }


def _backend_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"authority_feature_{index:04d}" for index, _ in enumerate(feature_columns))


def _rename_features_for_backend(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    backend_feature_columns: Sequence[str],
) -> pd.DataFrame:
    return frame.rename(
        columns=dict(zip(tuple(feature_columns), tuple(backend_feature_columns)))
    )


def _model_payload(ranker: PairwiseClassifierRanker, feature_columns: Sequence[str]) -> dict[str, Any]:
    classifier = ranker.classifier
    payload: dict[str, Any] = {
        "backend_id": "pairwise_linear",
        "feature_columns": list(feature_columns),
        "classifier_type": type(classifier).__name__,
    }
    if hasattr(classifier, "named_steps"):
        scaler = classifier.named_steps["scaler"]
        logreg = classifier.named_steps["logreg"]
        payload.update(
            {
                "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
                "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
                "classes": np.asarray(logreg.classes_, dtype=int).tolist(),
                "coef": np.asarray(logreg.coef_, dtype=float).tolist(),
                "intercept": np.asarray(logreg.intercept_, dtype=float).tolist(),
                "n_iter": np.asarray(logreg.n_iter_, dtype=int).tolist(),
            }
        )
    else:
        payload.update(
            {
                "probability": float(getattr(classifier, "probability", 0.5)),
                "classes": np.asarray(getattr(classifier, "classes_", [0, 1]), dtype=int).tolist(),
            }
        )
    return payload


@dataclass
class BaseScoreBridge:
    dataset_id: str
    backend_id: str
    ranker: PairwiseClassifierRanker
    feature_columns: tuple[str, ...]
    backend_feature_columns: tuple[str, ...]
    targets_by_case: dict[str, tuple[str, ...]]
    training_candidates_by_case: dict[str, tuple[str, ...]]
    query_plan_sha256: str
    random_state: int
    model_sha256: str
    feature_order_sha256: str

    def to_feature_frame(
        self, rows: Sequence[Mapping[str, Any]], *, include_labels: bool = False
    ) -> pd.DataFrame:
        frame = _normalize_rows(rows, self.feature_columns)
        frame = _rename_features_for_backend(
            frame, self.feature_columns, self.backend_feature_columns
        )
        if include_labels:
            labels = []
            for row in frame.itertuples(index=False):
                targets = set(self.targets_by_case.get(str(row.window_id), ()))
                labels.append(int(str(row.entity_id) in targets))
            frame["label"] = labels
        return frame


def fit_base_score_bridge(
    dataset_id: str,
    training_rows: Sequence[Mapping[str, Any]],
    targets_by_case: Mapping[str, Sequence[Any]],
    feature_columns: Sequence[str],
    query_plan_sha256: str,
    random_state: int = 42,
) -> BaseScoreBridge:
    query_hash = str(query_plan_sha256)
    if len(query_hash) != 64:
        raise BaseBridgeValidationError("query_plan_sha256 must contain 64 hex characters")
    if int(random_state) != 42:
        raise BaseBridgeValidationError("base bridge random_state must be 42")
    features = tuple(str(value) for value in feature_columns)
    frame = _normalize_rows(training_rows, features)
    backend_features = _backend_feature_columns(features)
    targets = _normalize_targets(targets_by_case)
    candidates = _candidate_ids(frame)
    if set(candidates) != set(targets):
        raise BaseBridgeValidationError("training target/case coverage mismatch")
    labels: list[int] = []
    for row in frame.itertuples(index=False):
        case_id = str(row.window_id)
        candidate_id = str(row.entity_id)
        labels.append(int(candidate_id in set(targets[case_id])))
    frame["label"] = labels
    frame["sample_weight"] = 1.0
    frame["pair_weight_mode"] = "legacy_query"
    frame["label_source"] = "queried_groundtruth"
    for case_id, group in frame.groupby("window_id", sort=False):
        positives = set(targets[str(case_id)])
        actual = set(str(value) for value in group["entity_id"])
        if not positives <= actual:
            raise BaseBridgeValidationError(f"valid positive missing from candidates for {case_id}")
        if not positives or positives == actual:
            raise BaseBridgeValidationError(f"pairwise case lacks positive/negative contrast: {case_id}")
    backend_frame = _rename_features_for_backend(frame, features, backend_features)
    ranker = train_pairwise_linear_ranker(
        training_frame=backend_frame,
        feature_columns=backend_features,
        random_state=42,
    )
    model_payload = _model_payload(ranker, features)
    model_payload.update(
        {
            "dataset_id": str(dataset_id),
            "query_plan_sha256": query_hash,
            "targets_by_case": targets,
            "training_candidates_by_case": candidates,
            "random_state": 42,
            "backend_feature_columns": backend_features,
        }
    )
    return BaseScoreBridge(
        dataset_id=str(dataset_id),
        backend_id="pairwise_linear",
        ranker=ranker,
        feature_columns=features,
        backend_feature_columns=backend_features,
        targets_by_case=targets,
        training_candidates_by_case=candidates,
        query_plan_sha256=query_hash,
        random_state=42,
        model_sha256=_semantic_hash(model_payload),
        feature_order_sha256=_semantic_hash(features),
    )


def score_base_score_bridge(
    bridge: BaseScoreBridge,
    rows: Sequence[Mapping[str, Any]],
    targets_by_case: Mapping[str, Sequence[Any]],
    artifact_role: str,
    *,
    expected_candidates_by_case: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    if bridge.backend_id != "pairwise_linear" or bridge.random_state != 42:
        raise BaseBridgeValidationError("base bridge authority identity drift")
    frame = bridge.to_feature_frame(rows, include_labels=False)
    observed_candidates = _candidate_ids(frame)
    expected = (
        {
            str(case_id): tuple(str(value) for value in values)
            for case_id, values in expected_candidates_by_case.items()
        }
        if expected_candidates_by_case is not None
        else observed_candidates
    )
    if set(observed_candidates) != set(expected):
        raise BaseBridgeValidationError("base score case coverage mismatch")
    for case_id in expected:
        if set(observed_candidates[case_id]) != set(expected[case_id]):
            raise BaseBridgeValidationError(f"candidate coverage mismatch for {case_id}")
    targets = _normalize_targets(targets_by_case)
    if set(targets) != set(expected):
        raise BaseBridgeValidationError("base score target/case coverage mismatch")
    for case_id, positives in targets.items():
        if not set(positives) <= set(expected[case_id]):
            raise BaseBridgeValidationError(f"valid positive missing from candidates for {case_id}")
    scored = bridge.ranker.score_frame(frame)
    if not np.isfinite(scored[["score", "raw_score"]].to_numpy(dtype=float)).all():
        raise BaseBridgeValidationError("base scores contain non-finite values")
    scores_by_case: dict[str, dict[str, float]] = {}
    rankings_by_case: dict[str, list[str]] = {}
    for case_id, group in scored.groupby("window_id", sort=False):
        case_key = str(case_id)
        scores_by_case[case_key] = {
            str(row.entity_id): float(row.score) for row in group.itertuples(index=False)
        }
        ordered = group.sort_values(
            by=["score", "raw_score", "entity_id"],
            ascending=[False, False, True],
        )
        rankings_by_case[case_key] = ordered["entity_id"].astype(str).tolist()
    identity = {
        "schema_version": "conservative-lofo-base-score-artifact-v1",
        "dataset_id": bridge.dataset_id,
        "artifact_role": str(artifact_role),
        "backend_id": bridge.backend_id,
        "query_plan_sha256": bridge.query_plan_sha256,
        "model_sha256": bridge.model_sha256,
        "feature_order_sha256": bridge.feature_order_sha256,
        "feature_columns": bridge.feature_columns,
        "candidate_ids_by_case": expected,
        "targets_by_case": targets,
        "scores_by_case": scores_by_case,
        "rankings_by_case": rankings_by_case,
    }
    artifact = {**identity, "score_artifact_sha256": _semantic_hash(identity)}
    validate_base_score_artifact(artifact, bridge)
    return artifact


def validate_base_score_artifact(
    artifact: Mapping[str, Any], bridge: BaseScoreBridge
) -> dict[str, Any]:
    if str(artifact.get("schema_version")) != "conservative-lofo-base-score-artifact-v1":
        raise BaseBridgeValidationError("unexpected base score artifact schema")
    if artifact.get("backend_id") != "pairwise_linear":
        raise BaseBridgeValidationError("base score backend drift")
    if artifact.get("query_plan_sha256") != bridge.query_plan_sha256:
        raise BaseBridgeValidationError("base score query-plan ownership mismatch")
    if artifact.get("model_sha256") != bridge.model_sha256:
        raise BaseBridgeValidationError("base score model ownership mismatch")
    candidates = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(artifact.get("candidate_ids_by_case", {})).items()
    }
    scores = dict(artifact.get("scores_by_case", {}))
    rankings = dict(artifact.get("rankings_by_case", {}))
    targets = {
        str(case_id): tuple(str(value) for value in values)
        for case_id, values in dict(artifact.get("targets_by_case", {})).items()
    }
    if not candidates or set(candidates) != set(scores) or set(candidates) != set(rankings):
        raise BaseBridgeValidationError("base score candidate/case coverage drift")
    for case_id in candidates:
        expected = set(candidates[case_id])
        if set(scores[case_id]) != expected or set(rankings[case_id]) != expected:
            raise BaseBridgeValidationError(f"base score candidates are incomplete for {case_id}")
        if len(rankings[case_id]) != len(set(rankings[case_id])):
            raise BaseBridgeValidationError(f"base score ranking duplicates candidates for {case_id}")
        if not set(targets.get(case_id, ())) <= expected:
            raise BaseBridgeValidationError(f"base score valid positive missing for {case_id}")
        if not all(math.isfinite(float(value)) for value in scores[case_id].values()):
            raise BaseBridgeValidationError(f"base score non-finite value for {case_id}")
    identity = {
        key: value for key, value in artifact.items() if key != "score_artifact_sha256"
    }
    if artifact.get("score_artifact_sha256") != _semantic_hash(identity):
        raise BaseBridgeValidationError("base score artifact hash drift")
    return {
        "valid": True,
        "case_count": len(candidates),
        "model_sha256": bridge.model_sha256,
        "score_artifact_sha256": artifact["score_artifact_sha256"],
    }


__all__ = [
    "BaseBridgeValidationError",
    "BaseScoreBridge",
    "fit_base_score_bridge",
    "score_base_score_bridge",
    "validate_base_score_artifact",
]
