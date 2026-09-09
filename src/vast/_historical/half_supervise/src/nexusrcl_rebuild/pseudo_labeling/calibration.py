"""Cross-fitted confidence calibration without hidden-label access."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


def _stable_ids(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if value is not None and str(value)}))


@dataclass(frozen=True)
class ScoredLabelPrediction:
    raw_score: float
    predicted_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        score = float(self.raw_score)
        if not math.isfinite(score):
            raise ValueError("raw calibration score must be finite")
        object.__setattr__(self, "raw_score", score)
        object.__setattr__(self, "predicted_ids", _stable_ids(self.predicted_ids))


@dataclass(frozen=True)
class CalibrationObservation:
    window_id: str
    raw_score: float
    exact_correct: bool
    any_hit_correct: bool
    fit_window_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("calibration observation window_id must not be empty")
        score = float(self.raw_score)
        if not math.isfinite(score):
            raise ValueError("raw calibration score must be finite")
        fit_window_ids = _stable_ids(self.fit_window_ids)
        if self.window_id in fit_window_ids:
            raise ValueError("calibration observation cannot train on its own window")
        if bool(self.exact_correct) and not bool(self.any_hit_correct):
            raise ValueError("an exact match must also be an any-positive hit")
        object.__setattr__(self, "raw_score", score)
        object.__setattr__(self, "exact_correct", bool(self.exact_correct))
        object.__setattr__(self, "any_hit_correct", bool(self.any_hit_correct))
        object.__setattr__(self, "fit_window_ids", fit_window_ids)


@dataclass(frozen=True)
class CalibratedConfidence:
    exact: float
    any_hit: float

    def to_dict(self) -> Dict[str, float]:
        return {"exact": self.exact, "any_hit": self.any_hit}


@dataclass(frozen=True)
class ReliabilityBin:
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    absolute_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "empirical_accuracy": self.empirical_accuracy,
            "absolute_gap": self.absolute_gap,
        }


@dataclass(frozen=True)
class CalibrationTargetReport:
    sample_count: int
    expected_calibration_error: float
    brier_score: float
    bins: Tuple[ReliabilityBin, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "bins": [item.to_dict() for item in self.bins],
        }


@dataclass(frozen=True)
class DualCalibrationReport:
    exact: CalibrationTargetReport
    any_hit: CalibrationTargetReport

    def to_dict(self) -> Dict[str, Any]:
        return {"exact": self.exact.to_dict(), "any_hit": self.any_hit.to_dict()}


@dataclass(frozen=True)
class _MonotonicCurve:
    scores: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    minimum_probability: float
    maximum_probability: float

    def predict(self, raw_score: float) -> float:
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("raw calibration score must be finite")
        if len(self.scores) == 1 or score <= self.scores[0]:
            probability = self.probabilities[0]
        elif score >= self.scores[-1]:
            probability = self.probabilities[-1]
        else:
            right = bisect.bisect_right(self.scores, score)
            left = right - 1
            span = self.scores[right] - self.scores[left]
            fraction = 0.0 if span == 0.0 else (score - self.scores[left]) / span
            probability = (
                self.probabilities[left]
                + fraction * (self.probabilities[right] - self.probabilities[left])
            )
        return min(self.maximum_probability, max(self.minimum_probability, probability))


def _fit_monotonic_curve(
    scores: Sequence[float],
    outcomes: Sequence[bool],
    smoothing: float,
    minimum_probability: float,
    maximum_probability: float,
) -> _MonotonicCurve:
    grouped = {}
    for score, outcome in zip(scores, outcomes):
        grouped.setdefault(float(score), []).append(bool(outcome))
    blocks = []
    for score in sorted(grouped):
        values = grouped[score]
        count = float(len(values))
        probability = (float(sum(values)) + smoothing) / (count + 2.0 * smoothing)
        blocks.append(
            {
                "score_weighted_sum": score * count,
                "probability_weighted_sum": probability * count,
                "weight": count,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_probability = left["probability_weighted_sum"] / left["weight"]
            right_probability = right["probability_weighted_sum"] / right["weight"]
            if left_probability <= right_probability:
                break
            blocks[-2:] = [
                {
                    "score_weighted_sum": (
                        left["score_weighted_sum"] + right["score_weighted_sum"]
                    ),
                    "probability_weighted_sum": (
                        left["probability_weighted_sum"]
                        + right["probability_weighted_sum"]
                    ),
                    "weight": left["weight"] + right["weight"],
                }
            ]
    return _MonotonicCurve(
        scores=tuple(block["score_weighted_sum"] / block["weight"] for block in blocks),
        probabilities=tuple(
            block["probability_weighted_sum"] / block["weight"] for block in blocks
        ),
        minimum_probability=minimum_probability,
        maximum_probability=maximum_probability,
    )


@dataclass(frozen=True)
class DualOutcomeCalibrator:
    exact_curve: _MonotonicCurve
    any_hit_curve: _MonotonicCurve

    @classmethod
    def fit(
        cls,
        observations: Sequence[CalibrationObservation],
        smoothing: float = 1.0,
        minimum_probability: float = 0.01,
        maximum_probability: float = 0.99,
    ) -> "DualOutcomeCalibrator":
        rows = tuple(observations)
        if not rows:
            raise ValueError("at least one calibration observation is required")
        if smoothing <= 0.0:
            raise ValueError("calibration smoothing must be positive")
        if not (
            0.0 <= minimum_probability < maximum_probability < 1.0
        ):
            raise ValueError("calibration probability bounds must satisfy 0 <= min < max < 1")
        scores = tuple(row.raw_score for row in rows)
        return cls(
            exact_curve=_fit_monotonic_curve(
                scores=scores,
                outcomes=tuple(row.exact_correct for row in rows),
                smoothing=float(smoothing),
                minimum_probability=float(minimum_probability),
                maximum_probability=float(maximum_probability),
            ),
            any_hit_curve=_fit_monotonic_curve(
                scores=scores,
                outcomes=tuple(row.any_hit_correct for row in rows),
                smoothing=float(smoothing),
                minimum_probability=float(minimum_probability),
                maximum_probability=float(maximum_probability),
            ),
        )

    def predict(self, raw_score: float) -> CalibratedConfidence:
        exact = self.exact_curve.predict(raw_score)
        any_hit = self.any_hit_curve.predict(raw_score)
        return CalibratedConfidence(exact=min(exact, any_hit), any_hit=any_hit)


def build_leave_one_out_observations(
    queried_labels: Mapping[str, Sequence[str]],
    predictor: Callable[[str, Mapping[str, Tuple[str, ...]]], ScoredLabelPrediction],
) -> Tuple[CalibrationObservation, ...]:
    """Score each queried window while withholding its own annotation."""

    labels = {
        str(window_id): _stable_ids(window_labels)
        for window_id, window_labels in queried_labels.items()
    }
    if len(labels) < 2:
        raise ValueError("leave-one-out calibration requires at least two queried windows")
    if any(not values for values in labels.values()):
        raise ValueError("queried calibration labels must not be empty")
    observations = []
    for window_id in sorted(labels):
        visible_labels = MappingProxyType(
            {
                other_id: labels[other_id]
                for other_id in sorted(labels)
                if other_id != window_id
            }
        )
        prediction = predictor(window_id, visible_labels)
        if not isinstance(prediction, ScoredLabelPrediction):
            raise TypeError("leave-one-out predictor must return ScoredLabelPrediction")
        predicted = set(prediction.predicted_ids)
        expected = set(labels[window_id])
        observations.append(
            CalibrationObservation(
                window_id=window_id,
                raw_score=prediction.raw_score,
                exact_correct=bool(predicted) and predicted == expected,
                any_hit_correct=bool(predicted.intersection(expected)),
                fit_window_ids=tuple(visible_labels),
            )
        )
    return tuple(observations)


def _target_report(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    bin_count: int,
) -> CalibrationTargetReport:
    bins_confidence = [[] for _ in range(bin_count)]
    bins_outcome = [[] for _ in range(bin_count)]
    for confidence, outcome in zip(confidences, outcomes):
        bin_index = min(int(float(confidence) * bin_count), bin_count - 1)
        bins_confidence[bin_index].append(float(confidence))
        bins_outcome[bin_index].append(bool(outcome))
    reliability_bins = []
    expected_calibration_error = 0.0
    sample_count = len(confidences)
    for bin_index in range(bin_count):
        values = bins_confidence[bin_index]
        outcomes_in_bin = bins_outcome[bin_index]
        count = len(values)
        if count:
            mean_confidence = sum(values) / float(count)
            empirical_accuracy = sum(outcomes_in_bin) / float(count)
            absolute_gap = abs(mean_confidence - empirical_accuracy)
            expected_calibration_error += (count / float(sample_count)) * absolute_gap
        else:
            mean_confidence = 0.0
            empirical_accuracy = 0.0
            absolute_gap = 0.0
        reliability_bins.append(
            ReliabilityBin(
                lower_bound=bin_index / float(bin_count),
                upper_bound=(bin_index + 1) / float(bin_count),
                count=count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=absolute_gap,
            )
        )
    brier_score = sum(
        (float(confidence) - float(bool(outcome))) ** 2
        for confidence, outcome in zip(confidences, outcomes)
    ) / float(sample_count)
    return CalibrationTargetReport(
        sample_count=sample_count,
        expected_calibration_error=expected_calibration_error,
        brier_score=brier_score,
        bins=tuple(reliability_bins),
    )


def evaluate_dual_calibration(
    observations: Sequence[CalibrationObservation],
    exact_confidences: Sequence[float],
    any_hit_confidences: Sequence[float],
    bin_count: int = 10,
) -> DualCalibrationReport:
    rows = tuple(observations)
    exact_values = tuple(float(value) for value in exact_confidences)
    any_values = tuple(float(value) for value in any_hit_confidences)
    if not rows:
        raise ValueError("at least one calibration observation is required")
    if len(rows) != len(exact_values) or len(rows) != len(any_values):
        raise ValueError("observations and confidence arrays must have the same length")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    for confidence in exact_values + any_values:
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be within [0, 1]")
    return DualCalibrationReport(
        exact=_target_report(
            exact_values,
            tuple(row.exact_correct for row in rows),
            bin_count,
        ),
        any_hit=_target_report(
            any_values,
            tuple(row.any_hit_correct for row in rows),
            bin_count,
        ),
    )
