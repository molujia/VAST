"""Immutable contracts shared by pseudo-label generators and evaluators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _stable_ids(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def _freeze_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if isinstance(value, Mapping):
        source = value
    else:
        source = dict(value)
    frozen = _freeze_value(source)
    if not isinstance(frozen, Mapping):
        raise TypeError("expected a mapping")
    return frozen


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


@dataclass(frozen=True)
class SanitizedEntityFeatureView:
    """One candidate entity with telemetry-only features."""

    entity_id: str
    entity_type: str
    entity_index: Optional[int]
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        object.__setattr__(self, "features", _freeze_mapping(self.features))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "entity_index": self.entity_index,
            "features": _plain_value(self.features),
        }


@dataclass(frozen=True)
class SanitizedFeatureView:
    """Label-free generator input for one normal or fault window."""

    dataset: str
    window_id: str
    source_id: str
    window_kind: str
    day: str
    start_ts: float
    end_ts: float
    cluster_id: Optional[int]
    is_normal_cluster: bool
    window_features: Mapping[str, float] = field(default_factory=dict)
    entities: Tuple[SanitizedEntityFeatureView, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset or not self.window_id:
            raise ValueError("dataset and window_id must not be empty")
        object.__setattr__(self, "window_features", _freeze_mapping(self.window_features))
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_id": self.window_id,
            "source_id": self.source_id,
            "window_kind": self.window_kind,
            "day": self.day,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "cluster_id": self.cluster_id,
            "is_normal_cluster": self.is_normal_cluster,
            "window_features": _plain_value(self.window_features),
            "entities": [entity.to_dict() for entity in self.entities],
            "metadata": _plain_value(self.metadata),
        }


@dataclass(frozen=True)
class AnnotationQuery:
    dataset: str
    window_id: str
    query_rank: int
    role: str

    def __post_init__(self) -> None:
        if not self.dataset or not self.window_id or not self.role:
            raise ValueError("annotation query fields must not be empty")
        if self.query_rank < 0:
            raise ValueError("query_rank must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_id": self.window_id,
            "query_rank": self.query_rank,
            "role": self.role,
        }


@dataclass(frozen=True)
class AnnotationAnswer:
    query: AnnotationQuery
    authoritative_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative_ids", _stable_ids(self.authoritative_ids))
        if not self.authoritative_ids:
            raise ValueError("an authorized fault query must return at least one entity")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "authoritative_ids": list(self.authoritative_ids),
        }


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    supported_ids: Tuple[str, ...] = ()
    score: Optional[float] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("evidence source must not be empty")
        object.__setattr__(self, "supported_ids", _stable_ids(self.supported_ids))
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("evidence score must be finite")
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "supported_ids": list(self.supported_ids),
            "score": self.score,
            "details": _plain_value(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            source=str(payload["source"]),
            supported_ids=tuple(str(item) for item in payload.get("supported_ids", ())),
            score=(None if payload.get("score") is None else float(payload["score"])),
            details=payload.get("details", {}),
        )


@dataclass(frozen=True)
class FrozenPseudoLabelPrediction:
    """An emitted pseudo-label or an explicit abstention."""

    window_id: str
    strategy_id: str
    predicted_ids: Tuple[str, ...]
    confidence: float
    evidence: Tuple[EvidenceRecord, ...] = ()
    abstention_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.window_id or not self.strategy_id:
            raise ValueError("prediction window_id and strategy_id must not be empty")
        object.__setattr__(self, "predicted_ids", _stable_ids(self.predicted_ids))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("prediction confidence must be within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if self.predicted_ids and self.abstention_reason:
            raise ValueError("an emitted prediction cannot have an abstention reason")
        if not self.predicted_ids and not self.abstention_reason:
            raise ValueError("an empty prediction must record an abstention reason")

    @property
    def emitted(self) -> bool:
        return bool(self.predicted_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "strategy_id": self.strategy_id,
            "predicted_ids": list(self.predicted_ids),
            "confidence": self.confidence,
            "emitted": self.emitted,
            "evidence": [record.to_dict() for record in self.evidence],
            "abstention_reason": self.abstention_reason,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FrozenPseudoLabelPrediction":
        return cls(
            window_id=str(payload["window_id"]),
            strategy_id=str(payload["strategy_id"]),
            predicted_ids=tuple(str(item) for item in payload.get("predicted_ids", ())),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=tuple(
                EvidenceRecord.from_dict(item)
                for item in payload.get("evidence", ())
            ),
            abstention_reason=(
                None
                if payload.get("abstention_reason") is None
                else str(payload["abstention_reason"])
            ),
        )


@dataclass(frozen=True)
class StrategyResult:
    """Frozen output from one strategy before any quality audit."""

    dataset: str
    strategy_id: str
    predictions: Tuple[FrozenPseudoLabelPrediction, ...]
    queried_window_ids: Tuple[str, ...]
    input_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        if not self.dataset or not self.strategy_id:
            raise ValueError("strategy result identity must not be empty")
        if not self.input_hash or not self.config_hash:
            raise ValueError("strategy result hashes must not be empty")
        predictions = tuple(self.predictions)
        window_ids = [prediction.window_id for prediction in predictions]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("strategy result contains duplicate window predictions")
        if any(prediction.strategy_id != self.strategy_id for prediction in predictions):
            raise ValueError("prediction strategy_id does not match its result")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "queried_window_ids", tuple(self.queried_window_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "strategy_id": self.strategy_id,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "queried_window_ids": list(self.queried_window_ids),
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyResult":
        return cls(
            dataset=str(payload["dataset"]),
            strategy_id=str(payload["strategy_id"]),
            predictions=tuple(
                FrozenPseudoLabelPrediction.from_dict(item)
                for item in payload.get("predictions", ())
            ),
            queried_window_ids=tuple(
                str(item) for item in payload.get("queried_window_ids", ())
            ),
            input_hash=str(payload["input_hash"]),
            config_hash=str(payload["config_hash"]),
        )


@dataclass(frozen=True)
class PseudoAuditSummary:
    eligible_count: int
    emitted_count: int
    exact_match_count: int
    any_hit_count: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "eligible_count": self.eligible_count,
            "emitted_count": self.emitted_count,
            "exact_match_count": self.exact_match_count,
            "any_hit_count": self.any_hit_count,
        }
