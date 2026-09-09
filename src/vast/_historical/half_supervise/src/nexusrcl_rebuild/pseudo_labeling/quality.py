"""Deterministic quality metrics for frozen pseudo-label artifacts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import FrozenPseudoLabelPrediction, StrategyResult


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f_score(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


@dataclass(frozen=True)
class PseudoQualitySummary:
    eligible_count: int
    emitted_count: int
    abstained_count: int
    exact_match_count: int
    any_hit_count: int
    true_positive_label_count: int
    predicted_label_count: int
    authoritative_label_count: int
    exact_accuracy: float
    any_hit_accuracy: float
    micro_label_precision: float
    micro_label_recall: float
    micro_label_f1: float
    coverage: float
    abstention_rate: float
    exact_selective_recall: float
    any_selective_recall: float
    selective_exact_f0_5: float
    selective_any_f0_5: float
    eligible_for_selection: bool

    def to_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class PseudoQualityReport:
    dataset: str
    strategy_id: str
    input_hash: str
    config_hash: str
    summary: PseudoQualitySummary
    per_entity: Mapping[str, PseudoQualitySummary]
    per_entity_type: Mapping[str, PseudoQualitySummary]
    per_cluster: Mapping[str, PseudoQualitySummary]
    by_label_cardinality: Mapping[str, PseudoQualitySummary]

    def __post_init__(self) -> None:
        for attribute in (
            "per_entity",
            "per_entity_type",
            "per_cluster",
            "by_label_cardinality",
        ):
            value = getattr(self, attribute)
            object.__setattr__(
                self,
                attribute,
                MappingProxyType(
                    {
                        str(key): item
                        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                    }
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "strategy_id": self.strategy_id,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "summary": self.summary.to_dict(),
            "per_entity": {
                key: value.to_dict() for key, value in self.per_entity.items()
            },
            "per_entity_type": {
                key: value.to_dict() for key, value in self.per_entity_type.items()
            },
            "per_cluster": {
                key: value.to_dict() for key, value in self.per_cluster.items()
            },
            "by_label_cardinality": {
                key: value.to_dict() for key, value in self.by_label_cardinality.items()
            },
        }


def _summarize(
    window_ids: Sequence[str],
    prediction_by_window: Mapping[str, FrozenPseudoLabelPrediction],
    authoritative_labels: Mapping[str, Tuple[str, ...]],
) -> PseudoQualitySummary:
    eligible_count = len(window_ids)
    emitted_count = 0
    exact_match_count = 0
    any_hit_count = 0
    true_positive_label_count = 0
    predicted_label_count = 0
    authoritative_label_count = 0
    for window_id in window_ids:
        prediction = prediction_by_window[window_id]
        predicted = set(prediction.predicted_ids)
        expected = set(authoritative_labels[window_id])
        if prediction.emitted:
            emitted_count += 1
        exact_match_count += int(bool(predicted) and predicted == expected)
        any_hit_count += int(bool(predicted.intersection(expected)))
        true_positive_label_count += len(predicted.intersection(expected))
        predicted_label_count += len(predicted)
        authoritative_label_count += len(expected)

    abstained_count = eligible_count - emitted_count
    exact_accuracy = _ratio(exact_match_count, emitted_count)
    any_hit_accuracy = _ratio(any_hit_count, emitted_count)
    micro_precision = _ratio(true_positive_label_count, predicted_label_count)
    micro_recall = _ratio(true_positive_label_count, authoritative_label_count)
    micro_f1 = _f_score(micro_precision, micro_recall, beta=1.0)
    coverage = _ratio(emitted_count, eligible_count)
    exact_selective_recall = _ratio(exact_match_count, eligible_count)
    any_selective_recall = _ratio(any_hit_count, eligible_count)
    return PseudoQualitySummary(
        eligible_count=eligible_count,
        emitted_count=emitted_count,
        abstained_count=abstained_count,
        exact_match_count=exact_match_count,
        any_hit_count=any_hit_count,
        true_positive_label_count=true_positive_label_count,
        predicted_label_count=predicted_label_count,
        authoritative_label_count=authoritative_label_count,
        exact_accuracy=exact_accuracy,
        any_hit_accuracy=any_hit_accuracy,
        micro_label_precision=micro_precision,
        micro_label_recall=micro_recall,
        micro_label_f1=micro_f1,
        coverage=coverage,
        abstention_rate=(1.0 - coverage if eligible_count else 0.0),
        exact_selective_recall=exact_selective_recall,
        any_selective_recall=any_selective_recall,
        selective_exact_f0_5=_f_score(
            exact_accuracy,
            exact_selective_recall,
            beta=0.5,
        ),
        selective_any_f0_5=_f_score(
            any_hit_accuracy,
            any_selective_recall,
            beta=0.5,
        ),
        eligible_for_selection=bool(eligible_count and emitted_count),
    )


def _entity_type(entity_id: str, entity_types: Mapping[str, str]) -> str:
    explicit = entity_types.get(entity_id)
    if explicit:
        return str(explicit)
    return entity_id.split(":", 1)[0] if ":" in entity_id else "unknown"


def audit_pseudo_label_quality(
    result: StrategyResult,
    authoritative_labels: Mapping[str, Sequence[str]],
    eligible_window_ids: Sequence[str],
    entity_types: Optional[Mapping[str, str]] = None,
    cluster_ids: Optional[Mapping[str, Optional[int]]] = None,
) -> PseudoQualityReport:
    """Audit a complete frozen strategy result after generation has finished."""

    eligible_ids = tuple(str(item) for item in eligible_window_ids)
    if len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("eligible_window_ids contains duplicates")
    queried_overlap = set(eligible_ids).intersection(result.queried_window_ids)
    if queried_overlap:
        raise ValueError("queried windows cannot enter a pseudo-label quality audit")
    prediction_by_window = {
        prediction.window_id: prediction for prediction in result.predictions
    }
    missing_predictions = sorted(set(eligible_ids).difference(prediction_by_window))
    if missing_predictions:
        raise ValueError(
            "eligible windows missing prediction or abstention records: %s"
            % missing_predictions
        )
    extra_predictions = sorted(set(prediction_by_window).difference(eligible_ids))
    if extra_predictions:
        raise ValueError("strategy result contains ineligible predictions: %s" % extra_predictions)

    normalized_truth = {}
    for window_id in eligible_ids:
        if window_id not in authoritative_labels:
            raise KeyError("eligible window missing authoritative labels: %s" % window_id)
        labels = tuple(
            sorted({str(item) for item in authoritative_labels[window_id] if str(item)})
        )
        if not labels:
            raise ValueError("eligible fault window has an empty authoritative label set")
        normalized_truth[window_id] = labels

    entity_type_map = dict(entity_types or {})
    windows_by_entity = {}
    windows_by_type = {}
    for window_id in eligible_ids:
        for entity_id in normalized_truth[window_id]:
            windows_by_entity.setdefault(entity_id, []).append(window_id)
            entity_type = _entity_type(entity_id, entity_type_map)
            windows_by_type.setdefault(entity_type, []).append(window_id)
    windows_by_type = {
        key: tuple(dict.fromkeys(value)) for key, value in windows_by_type.items()
    }

    windows_by_cluster = {}
    if cluster_ids is not None:
        missing_clusters = sorted(set(eligible_ids).difference(cluster_ids))
        if missing_clusters:
            raise ValueError("eligible windows missing cluster IDs: %s" % missing_clusters)
        for window_id in eligible_ids:
            cluster_value = cluster_ids[window_id]
            cluster_key = "none" if cluster_value is None else str(int(cluster_value))
            windows_by_cluster.setdefault(cluster_key, []).append(window_id)

    cardinality_groups = {
        "single_positive": tuple(
            window_id
            for window_id in eligible_ids
            if len(normalized_truth[window_id]) == 1
        ),
        "multi_positive": tuple(
            window_id
            for window_id in eligible_ids
            if len(normalized_truth[window_id]) > 1
        ),
    }
    return PseudoQualityReport(
        dataset=result.dataset,
        strategy_id=result.strategy_id,
        input_hash=result.input_hash,
        config_hash=result.config_hash,
        summary=_summarize(eligible_ids, prediction_by_window, normalized_truth),
        per_entity={
            entity_id: _summarize(window_ids, prediction_by_window, normalized_truth)
            for entity_id, window_ids in windows_by_entity.items()
        },
        per_entity_type={
            entity_type: _summarize(window_ids, prediction_by_window, normalized_truth)
            for entity_type, window_ids in windows_by_type.items()
        },
        per_cluster={
            cluster_id: _summarize(window_ids, prediction_by_window, normalized_truth)
            for cluster_id, window_ids in windows_by_cluster.items()
        },
        by_label_cardinality={
            group_name: _summarize(window_ids, prediction_by_window, normalized_truth)
            for group_name, window_ids in cardinality_groups.items()
        },
    )
