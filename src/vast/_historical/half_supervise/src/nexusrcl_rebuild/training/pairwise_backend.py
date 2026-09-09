"""Pairwise ranking backends for within-window RCA ordering."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANOMALY_HINT_COLUMNS = [
    "log_error_count",
    "log_error_ratio",
    "metric_event_score_sum",
    "metric_event_score_max",
    "metric_event_active_kpi_count",
    "metric_abs_z_max",
    "metric_anomalous_kpi_count",
    "trace_error_count",
    "trace_error_ratio",
    "trace_client_error_ratio",
    "trace_server_error_count",
    "trace_server_error_ratio",
    "trace_client_latency_abs_z_max",
    "trace_server_latency_abs_z_mean",
    "trace_server_latency_abs_z_max",
    "trace_client_anomalous_operation_count",
    "trace_server_anomalous_operation_count",
    "topology_change_count",
]

ANOMALY_HINT_PREFIXES = (
    "metric_kpi_peak_",
    "metric_kpi_hit_",
    "trace_operation_z_",
    "trace_peer_share_",
)


def _select_anomaly_columns(columns: Sequence[str]) -> Sequence[str]:
    return [
        column
        for column in columns
        if column in ANOMALY_HINT_COLUMNS
        or any(str(column).startswith(prefix) for prefix in ANOMALY_HINT_PREFIXES)
    ]


class ConstantPairwiseClassifier:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(probability)
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full((features.shape[0], 1), self.probability, dtype=float)
        negative = np.full((features.shape[0], 1), 1.0 - self.probability, dtype=float)
        return np.concatenate([negative, positive], axis=1)


@dataclass
class PairwiseClassifierRanker:
    classifier: object
    feature_columns: Sequence[str]

    @property
    def training_diagnostics(self) -> Dict[str, Any]:
        return dict(getattr(self.classifier, "training_diagnostics", {}))

    def score_frame(self, entity_features: pd.DataFrame) -> pd.DataFrame:
        rows = entity_features.copy()
        if rows.empty:
            rows["raw_score"] = pd.Series(dtype=float)
            rows["score"] = pd.Series(dtype=float)
            return rows

        classes = getattr(self.classifier, "classes_", np.array([0, 1], dtype=int))
        if len(classes) == 1:
            positive_score = 1.0 if int(classes[0]) == 1 else 0.0
            rows["raw_score"] = positive_score
            rows["score"] = positive_score
            return rows

        positive_index = int(np.where(classes == 1)[0][0])
        outputs = []
        for _window_id, group in rows.groupby("window_id", sort=False):
            feature_matrix = group[list(self.feature_columns)].fillna(0.0).to_numpy(dtype=float)
            if feature_matrix.shape[0] <= 1:
                window_scores = np.ones(feature_matrix.shape[0], dtype=float)
            else:
                candidate_count = feature_matrix.shape[0]
                left_indices = np.repeat(np.arange(candidate_count), candidate_count)
                right_indices = np.tile(np.arange(candidate_count), candidate_count)
                pair_mask = left_indices != right_indices
                left_indices = left_indices[pair_mask]
                right_indices = right_indices[pair_mask]
                comparisons = feature_matrix[left_indices] - feature_matrix[right_indices]
                probabilities = self.classifier.predict_proba(comparisons)[:, positive_index]
                score_sums = np.zeros(candidate_count, dtype=float)
                np.add.at(score_sums, left_indices, probabilities)
                window_scores = score_sums / float(candidate_count - 1)
            scored_group = group.copy()
            scored_group["raw_score"] = window_scores
            scored_group["score"] = window_scores
            outputs.append(scored_group)
        return pd.concat(outputs, axis=0, ignore_index=True)


@dataclass(frozen=True)
class PairwiseTrainingArrays:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    diagnostics: Dict[str, Any]


def _build_pairwise_training_arrays(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    negative_top_k: int = 0,
    with_diagnostics: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    PairwiseTrainingArrays,
]:
    pair_features = []
    pair_labels = []
    pair_weights = []
    queried_pair_count = 0
    pseudo_pair_count = 0
    queried_pair_weight_sum = 0.0
    pseudo_pair_weight_sum = 0.0
    pseudo_case_count = 0
    eligible_pseudo_case_count = 0
    ineligible_reason_counts: Dict[str, int] = {}
    pseudo_initial_gradient = np.zeros(len(feature_columns), dtype=float)

    for _window_id, group in training_frame.groupby("window_id", sort=False):
        if "label_source" in group.columns:
            source_values = {
                str(value)
                for value in group["label_source"].dropna().tolist()
            }
        else:
            source_values = set()
        is_pseudo_case = any(
            value.startswith("pseudo_") for value in source_values
        )
        if is_pseudo_case:
            pseudo_case_count += 1
            if "pseudo_eligibility_reason" in group.columns:
                reasons = {
                    str(value)
                    for value in group["pseudo_eligibility_reason"].dropna().tolist()
                    if str(value)
                }
                for reason in reasons:
                    ineligible_reason_counts[reason] = (
                        ineligible_reason_counts.get(reason, 0) + 1
                    )
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        if positives.empty or negatives.empty:
            continue
        supervised = group[group["label"].isin([0, 1])]
        if "pair_weight_mode" in supervised.columns:
            pair_weight_modes = {
                str(value)
                for value in supervised["pair_weight_mode"].dropna().tolist()
            }
        else:
            pair_weight_modes = {"legacy_query"}
        if len(pair_weight_modes) != 1:
            raise ValueError("pairwise case has mixed pair-weight modes")
        pair_weight_mode = next(iter(pair_weight_modes))
        if pair_weight_mode not in {
            "legacy_query",
            "pseudo_case_normalized",
        }:
            raise ValueError("pairwise case has unsupported pair-weight mode")
        if int(negative_top_k) > 0 and len(negatives) > int(negative_top_k):
            anomaly_columns = _select_anomaly_columns(negatives.columns.tolist())
            negatives = negatives.copy()
            if anomaly_columns:
                negatives["_pairwise_anomaly_strength"] = (
                    negatives[anomaly_columns].fillna(0.0).abs().sum(axis=1)
                )
            else:
                negatives["_pairwise_anomaly_strength"] = 0.0
            negatives = negatives.sort_values(
                by=["_pairwise_anomaly_strength", "entity_id"],
                ascending=[False, True],
            ).head(int(negative_top_k))
        pseudo_directional_weight = None
        if pair_weight_mode == "pseudo_case_normalized":
            if "pseudo_case_weight" not in supervised.columns:
                raise ValueError("pseudo pairwise case weight is missing")
            raw_case_weights = [
                float(value)
                for value in supervised["pseudo_case_weight"].tolist()
            ]
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in raw_case_weights
            ):
                raise ValueError(
                    "pseudo pairwise case weight must be finite positive"
                )
            case_weights = set(raw_case_weights)
            if len(case_weights) != 1:
                raise ValueError("pseudo pairwise case weight must be constant")
            pseudo_case_weight = next(iter(case_weights))
            eligible_pseudo_case_count += 1
            pseudo_directional_weight = pseudo_case_weight / (
                2.0 * float(len(positives)) * float(len(negatives))
            )
        negative_matrix = negatives[list(feature_columns)].fillna(0.0).to_numpy(dtype=float)
        negative_weights = negatives["sample_weight"].astype(float).to_numpy()
        for positive_row in positives.itertuples(index=False):
            positive_vector = np.asarray(
                [getattr(positive_row, column) for column in feature_columns],
                dtype=float,
            )
            positive_weight = float(getattr(positive_row, "sample_weight"))
            diffs = positive_vector[None, :] - negative_matrix
            if pseudo_directional_weight is not None:
                weights = np.full(
                    len(negatives),
                    pseudo_directional_weight,
                    dtype=float,
                )
            else:
                weights = positive_weight + negative_weights
            for diff_vector, weight in zip(diffs, weights):
                pair_features.append(diff_vector)
                pair_labels.append(1)
                pair_weights.append(weight)
                pair_features.append(-diff_vector)
                pair_labels.append(0)
                pair_weights.append(weight)
                if pseudo_directional_weight is not None:
                    pseudo_pair_count += 2
                    pseudo_pair_weight_sum += 2.0 * float(weight)
                    pseudo_initial_gradient -= float(weight) * diff_vector
                else:
                    queried_pair_count += 2
                    queried_pair_weight_sum += 2.0 * float(weight)

    if not pair_features:
        features = np.empty((0, len(feature_columns)), dtype=float)
        labels = np.empty((0,), dtype=int)
        weights = np.empty((0,), dtype=float)
    else:
        features = np.asarray(pair_features, dtype=float)
        labels = np.asarray(pair_labels, dtype=int)
        weights = np.asarray(pair_weights, dtype=float)
    pseudo_initial_gradient_l2 = float(
        np.linalg.norm(pseudo_initial_gradient)
    )
    diagnostics = {
        "schema_version": "pairwise-training-arrays-v2",
        "pair_count": int(labels.size),
        "queried_pair_count": queried_pair_count,
        "pseudo_pair_count": pseudo_pair_count,
        "queried_pair_weight_sum": queried_pair_weight_sum,
        "pseudo_pair_weight_sum": pseudo_pair_weight_sum,
        "pseudo_case_count": pseudo_case_count,
        "eligible_pseudo_case_count": eligible_pseudo_case_count,
        "ineligible_pseudo_case_count": (
            pseudo_case_count - eligible_pseudo_case_count
        ),
        "ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "pseudo_initial_gradient_l2": pseudo_initial_gradient_l2,
        "pseudo_nonzero_gradient_epoch_count": int(
            pseudo_initial_gradient_l2 > 0.0
        ),
    }
    arrays = PairwiseTrainingArrays(
        features=features,
        labels=labels,
        weights=weights,
        diagnostics=diagnostics,
    )
    if with_diagnostics:
        return arrays
    return arrays.features, arrays.labels, arrays.weights


def train_pairwise_tree_ranker(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int = 42,
    negative_top_k: int = 0,
) -> PairwiseClassifierRanker:
    arrays = _build_pairwise_training_arrays(
        training_frame=training_frame,
        feature_columns=feature_columns,
        negative_top_k=negative_top_k,
        with_diagnostics=True,
    )
    pair_features = arrays.features
    pair_labels = arrays.labels
    pair_weights = arrays.weights
    if pair_features.size == 0:
        classifier = ConstantPairwiseClassifier(probability=0.5)
        classifier.training_diagnostics = dict(arrays.diagnostics)
        return PairwiseClassifierRanker(classifier=classifier, feature_columns=list(feature_columns))

    classifier = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(pair_features, pair_labels, sample_weight=pair_weights)
    classifier.training_diagnostics = dict(arrays.diagnostics)
    return PairwiseClassifierRanker(classifier=classifier, feature_columns=list(feature_columns))


def train_pairwise_linear_ranker(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int = 42,
    negative_top_k: int = 0,
) -> PairwiseClassifierRanker:
    arrays = _build_pairwise_training_arrays(
        training_frame=training_frame,
        feature_columns=feature_columns,
        negative_top_k=negative_top_k,
        with_diagnostics=True,
    )
    pair_features = arrays.features
    pair_labels = arrays.labels
    pair_weights = arrays.weights
    if pair_features.size == 0:
        classifier = ConstantPairwiseClassifier(probability=0.5)
        classifier.training_diagnostics = dict(arrays.diagnostics)
        return PairwiseClassifierRanker(classifier=classifier, feature_columns=list(feature_columns))

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    classifier.fit(pair_features, pair_labels, logreg__sample_weight=pair_weights)
    classifier.training_diagnostics = dict(arrays.diagnostics)
    return PairwiseClassifierRanker(classifier=classifier, feature_columns=list(feature_columns))
