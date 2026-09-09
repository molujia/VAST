"""Cautious training adaptation for frozen pseudo-label predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from nexusrcl_rebuild.pseudo_labeling.contracts import StrategyResult
from nexusrcl_rebuild.pseudo_labeling.sanitizer import is_label_bearing_field


@dataclass(frozen=True)
class PseudoTrainingAdapterConfig:
    pseudo_positive_weight: float = 0.5
    pseudo_negative_weight: float = 0.2
    consensus_bottom_k: int = 2
    minimum_negative_agreement: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.pseudo_positive_weight <= 1.0:
            raise ValueError("pseudo_positive_weight must be within [0, 1]")
        if not 0.0 <= self.pseudo_negative_weight <= 1.0:
            raise ValueError("pseudo_negative_weight must be within [0, 1]")
        if self.consensus_bottom_k < 0:
            raise ValueError("consensus_bottom_k must be non-negative")
        if self.minimum_negative_agreement <= 0:
            raise ValueError("minimum_negative_agreement must be positive")


def _stable_ids(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if value is not None and str(value)}))


def _normalize_rankings(
    candidate_rankings: Optional[Mapping[str, Mapping[str, Sequence[str]]]],
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    return {
        str(window_id): {
            str(source): tuple(str(entity_id) for entity_id in ranking)
            for source, ranking in sorted(source_rankings.items())
        }
        for window_id, source_rankings in sorted((candidate_rankings or {}).items())
    }


def _consensus_negatives(
    candidate_ids: Set[str],
    positive_ids: Set[str],
    source_rankings: Mapping[str, Sequence[str]],
    config: PseudoTrainingAdapterConfig,
) -> Set[str]:
    if config.consensus_bottom_k <= 0 or config.pseudo_negative_weight <= 0.0:
        return set()
    votes: Counter = Counter()
    for source, ranking in sorted(source_rankings.items()):
        ordered = tuple(str(entity_id) for entity_id in ranking)
        if len(ordered) != len(set(ordered)):
            raise ValueError("candidate ranking %s contains duplicate entity IDs" % source)
        unknown = set(ordered).difference(candidate_ids)
        if unknown:
            raise ValueError(
                "candidate ranking %s contains unknown entities: %s"
                % (source, sorted(unknown))
            )
        eligible = [entity_id for entity_id in ordered if entity_id not in positive_ids]
        for entity_id in eligible[-config.consensus_bottom_k :]:
            votes[entity_id] += 1
    return {
        entity_id
        for entity_id, count in votes.items()
        if count >= config.minimum_negative_agreement
    }


def adapt_pseudo_training_rows(
    entity_features: pd.DataFrame,
    queried_labels: Mapping[str, Sequence[str]],
    strategy_result: StrategyResult,
    candidate_rankings: Optional[
        Mapping[str, Mapping[str, Sequence[str]]]
    ] = None,
    config: Optional[PseudoTrainingAdapterConfig] = None,
) -> pd.DataFrame:
    resolved_config = config or PseudoTrainingAdapterConfig()
    required_columns = {"window_id", "window_kind", "entity_id", "entity_type"}
    missing = sorted(required_columns.difference(entity_features.columns))
    if missing:
        raise ValueError("entity_features missing required columns: %s" % missing)
    rows = entity_features.drop(
        columns=[
            column
            for column in entity_features.columns
            if is_label_bearing_field(column)
        ],
        errors="ignore",
    ).copy()
    if "dataset" in rows.columns:
        datasets = set(rows["dataset"].astype(str).unique().tolist())
        if datasets and datasets != {strategy_result.dataset}:
            raise ValueError("entity features and strategy result use different datasets")
    query_ids = tuple(str(item) for item in strategy_result.queried_window_ids)
    normalized_queries = {
        str(window_id): _stable_ids(labels)
        for window_id, labels in queried_labels.items()
    }
    if set(normalized_queries) != set(query_ids):
        raise ValueError("queried labels must match the frozen strategy queried IDs")
    if any(not labels for labels in normalized_queries.values()):
        raise ValueError("queried fault labels must not be empty")
    prediction_by_window = {
        prediction.window_id: prediction for prediction in strategy_result.predictions
    }
    if set(prediction_by_window).intersection(query_ids):
        raise ValueError("frozen pseudo results must not predict queried windows")
    available_window_ids = set(rows["window_id"].astype(str).tolist())
    unknown_predictions = set(prediction_by_window).difference(available_window_ids)
    if unknown_predictions:
        raise ValueError(
            "frozen pseudo predictions reference unknown windows: %s"
            % sorted(unknown_predictions)
        )
    normalized_rankings = _normalize_rankings(candidate_rankings)

    candidate_ids_by_window = {
        str(window_id): set(group["entity_id"].astype(str).tolist())
        for window_id, group in rows.groupby("window_id", sort=False)
    }
    negative_ids_by_window = {}
    for window_id, prediction in prediction_by_window.items():
        if not prediction.emitted:
            continue
        candidate_ids = candidate_ids_by_window[window_id]
        positive_ids = set(prediction.predicted_ids)
        unknown_positives = positive_ids.difference(candidate_ids)
        if unknown_positives:
            raise ValueError(
                "frozen pseudo label references unknown candidate entities: %s"
                % sorted(unknown_positives)
            )
        negative_ids_by_window[window_id] = _consensus_negatives(
            candidate_ids=candidate_ids,
            positive_ids=positive_ids,
            source_rankings=normalized_rankings.get(window_id, {}),
            config=resolved_config,
        )

    labels = []
    weights = []
    sources = []
    for row in rows.itertuples(index=False):
        window_id = str(row.window_id)
        entity_id = str(row.entity_id)
        if str(row.window_kind) == "normal":
            labels.append(0)
            weights.append(1.0)
            sources.append("normal_window")
            continue
        if window_id in normalized_queries:
            candidate_ids = candidate_ids_by_window[window_id]
            missing_query_ids = set(normalized_queries[window_id]).difference(candidate_ids)
            if missing_query_ids:
                raise ValueError(
                    "queried label references unknown candidate entities: %s"
                    % sorted(missing_query_ids)
                )
            labels.append(1 if entity_id in normalized_queries[window_id] else 0)
            weights.append(1.0)
            sources.append("queried")
            continue
        prediction = prediction_by_window.get(window_id)
        if prediction is None:
            labels.append(-1)
            weights.append(0.0)
            sources.append("unlabeled_no_prediction")
            continue
        if not prediction.emitted:
            labels.append(-1)
            weights.append(0.0)
            sources.append("pseudo_abstention")
            continue
        if entity_id in prediction.predicted_ids:
            labels.append(1)
            weights.append(
                prediction.confidence * resolved_config.pseudo_positive_weight
            )
            sources.append("pseudo_positive")
        elif entity_id in negative_ids_by_window.get(window_id, set()):
            labels.append(0)
            weights.append(
                prediction.confidence * resolved_config.pseudo_negative_weight
            )
            sources.append("pseudo_consensus_negative")
        else:
            labels.append(-1)
            weights.append(0.0)
            sources.append("unlabeled_ambiguous")
    rows["label"] = labels
    rows["sample_weight"] = weights
    rows["label_source"] = sources
    return rows


def _matched_arm_manifest(matched_arm: Any) -> Dict[str, Any]:
    fields = (
        "arm",
        "canonical_dataset_id",
        "raw_pool_sha256",
        "raw_pool_path",
        "query_plan_sha256",
        "query_case_ids",
        "label_mode",
        "class_conditional_coverage",
        "pseudo_rows",
        "rejections",
        "pseudo_count",
        "pseudo_weight_policy",
        "negative_strategy",
        "pseudo_loss_ratio",
        "teacher_bottom_fraction",
        "minimum_rank_margin",
    )
    defaults = {
        "negative_strategy": "all_non_positive",
        "pseudo_loss_ratio": 0.25,
        "teacher_bottom_fraction": 0.25,
        "minimum_rank_margin": 0.50,
    }
    payload = {
        field: (
            matched_arm.get(field, defaults.get(field))
            if isinstance(matched_arm, Mapping)
            else getattr(matched_arm, field, defaults.get(field))
        )
        for field in fields
    }
    payload["query_case_ids"] = [
        str(value) for value in payload["query_case_ids"]
    ]
    payload["pseudo_rows"] = [
        dict(value) for value in payload["pseudo_rows"]
    ]
    payload["rejections"] = [
        dict(value) for value in payload["rejections"]
    ]
    return payload


def _complete_candidate_ranking(
    raw: Any,
    *,
    candidate_ids: Set[str],
    role: str,
) -> Tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("%s must be a candidate ranking" % role)
    ranking = tuple(str(value) for value in raw)
    if (
        len(ranking) != len(set(ranking))
        or set(ranking) != candidate_ids
    ):
        raise ValueError("%s must cover every candidate exactly once" % role)
    return ranking


def _matched_pseudo_candidate_partition(
    *,
    candidate_ids: Set[str],
    positive_ids: Set[str],
    pseudo: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Tuple[Set[str], Set[str], Set[str], str]:
    if not positive_ids:
        raise ValueError("matched pseudo positive set must not be empty")
    if not positive_ids.issubset(candidate_ids):
        raise ValueError("matched pseudo positives are outside the candidate set")
    strategy = str(
        pseudo.get("negative_strategy", manifest["negative_strategy"])
    )
    if strategy == "all_non_positive":
        negatives = set(candidate_ids).difference(positive_ids)
    elif strategy == "teacher_bottom_k":
        fraction = float(
            pseudo.get(
                "teacher_bottom_fraction",
                manifest["teacher_bottom_fraction"],
            )
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("teacher bottom fraction must be within (0, 1]")
        raw_rankings = pseudo.get("teacher_candidate_rankings", ())
        if not isinstance(raw_rankings, (list, tuple)) or len(raw_rankings) != 2:
            raise ValueError("teacher_bottom_k requires exactly two teacher rankings")
        rankings = tuple(
            _complete_candidate_ranking(
                raw,
                candidate_ids=candidate_ids,
                role="teacher ranking %d" % index,
            )
            for index, raw in enumerate(raw_rankings)
        )
        bottom_count = max(1, int(math.ceil(len(candidate_ids) * fraction)))
        negatives = set(rankings[0][-bottom_count:]).intersection(
            rankings[1][-bottom_count:]
        )
        negatives.difference_update(positive_ids)
    elif strategy == "margin_abstain":
        margin = float(
            pseudo.get("minimum_rank_margin", manifest["minimum_rank_margin"])
        )
        if not 0.0 <= margin <= 1.0:
            raise ValueError("minimum rank margin must be within [0, 1]")
        ranking = _complete_candidate_ranking(
            pseudo.get("candidate_ranking", ()),
            candidate_ids=candidate_ids,
            role="mean teacher ranking",
        )
        denominator = float(max(len(ranking) - 1, 1))
        rank_scores = {
            candidate_id: 1.0 - float(index) / denominator
            for index, candidate_id in enumerate(ranking)
        }
        worst_positive_score = min(
            rank_scores[candidate_id] for candidate_id in positive_ids
        )
        negatives = {
            candidate_id
            for candidate_id in candidate_ids.difference(positive_ids)
            if worst_positive_score - rank_scores[candidate_id]
            >= margin - 1e-12
        }
    else:
        raise ValueError("unsupported matched pseudo negative strategy")
    ambiguous = set(candidate_ids).difference(positive_ids).difference(negatives)
    if (
        positive_ids.intersection(negatives)
        or positive_ids.intersection(ambiguous)
        or negatives.intersection(ambiguous)
        or positive_ids | negatives | ambiguous != candidate_ids
    ):
        raise ValueError("matched pseudo P/N/U partition is invalid")
    reason = "" if negatives else "no_reliable_negative_candidates"
    return set(positive_ids), set(negatives), ambiguous, reason


def adapt_matched_pseudo_training_rows(
    entity_features: pd.DataFrame,
    queried_labels: Mapping[str, Sequence[str]],
    matched_arm: Any,
) -> pd.DataFrame:
    manifest = _matched_arm_manifest(matched_arm)
    rows = entity_features.drop(
        columns=[
            column
            for column in entity_features.columns
            if is_label_bearing_field(column)
        ],
        errors="ignore",
    ).copy()
    required = {"window_id", "window_kind", "entity_id", "entity_type"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(
            "entity_features missing required columns: %s" % missing
        )
    query_ids = tuple(str(value) for value in manifest["query_case_ids"])
    normalized_queries = {
        str(case_id): _stable_ids(targets)
        for case_id, targets in queried_labels.items()
    }
    if set(normalized_queries) != set(query_ids):
        raise ValueError(
            "queried labels must match matched-arm query case IDs"
        )
    pseudo_by_case = {
        str(row["case_id"]): dict(row)
        for row in manifest["pseudo_rows"]
    }
    if set(pseudo_by_case) & set(query_ids):
        raise ValueError("matched pseudo arm predicts a queried case")
    available = set(rows["window_id"].astype(str))
    unknown = sorted(set(pseudo_by_case).difference(available))
    if unknown:
        raise ValueError(
            "matched pseudo arm references unknown cases: %s" % unknown
        )
    candidates_by_case = {
        str(case_id): set(group["entity_id"].astype(str))
        for case_id, group in rows.groupby("window_id", sort=False)
    }
    for case_id, targets in normalized_queries.items():
        missing_targets = set(targets).difference(
            candidates_by_case.get(case_id, set())
        )
        if missing_targets:
            raise ValueError(
                "queried target set contains unknown candidates: %s"
                % sorted(missing_targets)
            )
    for case_id, pseudo in pseudo_by_case.items():
        targets = _stable_ids(pseudo.get("pseudo_target_set", ()))
        if not targets:
            raise ValueError("matched pseudo target set must not be empty")
        missing_targets = set(targets).difference(
            candidates_by_case.get(case_id, set())
        )
        if missing_targets:
            raise ValueError(
                "pseudo target set contains unknown candidates: %s"
                % sorted(missing_targets)
            )

    partitions_by_case = {}
    ineligible_by_case = {}
    for case_id, pseudo in pseudo_by_case.items():
        positive_ids = set(
            _stable_ids(pseudo.get("pseudo_target_set", ()))
        )
        partition = _matched_pseudo_candidate_partition(
            candidate_ids=candidates_by_case[case_id],
            positive_ids=positive_ids,
            pseudo=pseudo,
            manifest=manifest,
        )
        partitions_by_case[case_id] = partition[:3]
        if partition[3]:
            ineligible_by_case[case_id] = partition[3]

    labels = []
    weights = []
    sources = []
    confidences = []
    target_sets = []
    pseudo_partitions = []
    negative_strategies = []
    pseudo_case_weights = []
    pair_weight_modes = []
    pseudo_eligibility_reasons = []
    for row in rows.itertuples(index=False):
        case_id = str(row.window_id)
        entity_id = str(row.entity_id)
        if str(row.window_kind) == "normal":
            labels.append(-1)
            weights.append(0.0)
            sources.append("normal_window_excluded")
            confidences.append(0.0)
            target_sets.append(())
            pseudo_partitions.append("not_pseudo")
            negative_strategies.append("")
            pseudo_case_weights.append(0.0)
            pair_weight_modes.append("excluded")
            pseudo_eligibility_reasons.append("")
            continue
        if case_id in normalized_queries:
            targets = normalized_queries[case_id]
            labels.append(int(entity_id in targets))
            weights.append(1.0)
            sources.append("queried")
            confidences.append(0.0)
            target_sets.append(targets)
            pseudo_partitions.append("not_pseudo")
            negative_strategies.append("")
            pseudo_case_weights.append(0.0)
            pair_weight_modes.append("legacy_query")
            pseudo_eligibility_reasons.append("")
            continue
        pseudo = pseudo_by_case.get(case_id)
        if pseudo is None:
            labels.append(-1)
            weights.append(0.0)
            sources.append("unlabeled_no_matched_pseudo")
            confidences.append(0.0)
            target_sets.append(())
            pseudo_partitions.append("not_pseudo")
            negative_strategies.append("")
            pseudo_case_weights.append(0.0)
            pair_weight_modes.append("excluded")
            pseudo_eligibility_reasons.append("")
            continue
        targets = _stable_ids(pseudo.get("pseudo_target_set", ()))
        confidence = float(pseudo.get("confidence", 0.0))
        if not 0.0 < confidence <= 1.0:
            raise ValueError("matched pseudo confidence must be within (0, 1]")
        label_mode = str(pseudo.get("label_mode", manifest["label_mode"]))
        if label_mode not in {"hard_top1", "partial_top2"}:
            raise ValueError("unsupported matched pseudo label mode")
        strategy = str(
            pseudo.get("negative_strategy", manifest["negative_strategy"])
        )
        loss_ratio = float(
            pseudo.get("pseudo_loss_ratio", manifest["pseudo_loss_ratio"])
        )
        if loss_ratio not in {0.1, 0.25, 0.5}:
            raise ValueError("matched pseudo loss ratio is outside the frozen grid")
        case_weight = loss_ratio * confidence
        positive_ids, negative_ids, ambiguous_ids = partitions_by_case[case_id]
        ineligible_reason = ineligible_by_case.get(case_id, "")
        if ineligible_reason:
            label = -1
            sample_weight = 0.0
            partition_name = "ineligible"
            pair_mode = "excluded"
        elif entity_id in positive_ids:
            label = 1
            sample_weight = case_weight
            partition_name = "positive"
            pair_mode = "pseudo_case_normalized"
        elif entity_id in negative_ids:
            label = 0
            sample_weight = case_weight
            partition_name = "reliable_negative"
            pair_mode = "pseudo_case_normalized"
        elif entity_id in ambiguous_ids:
            label = -1
            sample_weight = 0.0
            partition_name = "abstained"
            pair_mode = "excluded"
        else:
            raise ValueError("matched pseudo partition omitted a candidate")
        labels.append(label)
        weights.append(sample_weight)
        sources.append("pseudo_%s" % label_mode)
        confidences.append(confidence)
        target_sets.append(targets)
        pseudo_partitions.append(partition_name)
        negative_strategies.append(strategy)
        pseudo_case_weights.append(case_weight)
        pair_weight_modes.append(pair_mode)
        pseudo_eligibility_reasons.append(ineligible_reason)
    rows["label"] = labels
    rows["sample_weight"] = weights
    rows["label_source"] = sources
    rows["pseudo_case_confidence"] = confidences
    rows["target_set"] = target_sets
    rows["pseudo_partition"] = pseudo_partitions
    rows["pseudo_negative_strategy"] = negative_strategies
    rows["pseudo_case_weight"] = pseudo_case_weights
    rows["pair_weight_mode"] = pair_weight_modes
    rows["pseudo_eligibility_reason"] = pseudo_eligibility_reasons
    rows.attrs["matched_pseudo_arm"] = manifest
    rows.attrs["matched_pseudo_diagnostics"] = {
        "pseudo_case_count": len(pseudo_by_case),
        "eligible_pseudo_case_count": len(pseudo_by_case) - len(ineligible_by_case),
        "ineligible_pseudo_case_count": len(ineligible_by_case),
        "ineligible_reasons": dict(sorted(ineligible_by_case.items())),
    }
    return rows


def attach_matched_pseudo_arm(query_plan: Any, matched_arm: Any) -> Any:
    manifest = _matched_arm_manifest(matched_arm)
    query_ids = tuple(str(value) for value in query_plan.queried_window_ids)
    arm_query_ids = tuple(
        str(value) for value in manifest["query_case_ids"]
    )
    if (
        len(query_ids) != len(set(query_ids))
        or len(arm_query_ids) != len(set(arm_query_ids))
    ):
        raise ValueError(
            "query plan and matched pseudo arm require unique queried IDs"
        )
    if set(query_ids) != set(arm_query_ids):
        raise ValueError(
            "query plan and matched pseudo arm use different queried IDs"
        )
    pseudo_labels = {
        str(row["case_id"]): list(row["pseudo_target_set"])
        for row in manifest["pseudo_rows"]
    }
    pseudo_confidence = {
        str(row["case_id"]): float(row["confidence"])
        for row in manifest["pseudo_rows"]
    }
    metadata = dict(query_plan.metadata)
    manifest["training_dataset"] = str(query_plan.dataset)
    metadata["matched_pseudo_arm"] = manifest
    return replace(
        query_plan,
        pseudo_labels=pseudo_labels,
        pseudo_confidence=pseudo_confidence,
        metadata=metadata,
    )


def attach_frozen_pseudo_result(
    query_plan: Any,
    strategy_result: StrategyResult,
    candidate_rankings: Optional[
        Mapping[str, Mapping[str, Sequence[str]]]
    ] = None,
) -> Any:
    if str(query_plan.dataset) != strategy_result.dataset:
        raise ValueError("query plan and frozen pseudo result use different datasets")
    if tuple(str(item) for item in query_plan.queried_window_ids) != tuple(
        strategy_result.queried_window_ids
    ):
        raise ValueError("query plan and frozen pseudo result use different queried IDs")
    emitted = {
        prediction.window_id: list(prediction.predicted_ids)
        for prediction in strategy_result.predictions
        if prediction.emitted
    }
    confidences = {
        prediction.window_id: float(prediction.confidence)
        for prediction in strategy_result.predictions
        if prediction.emitted
    }
    normalized_rankings = _normalize_rankings(candidate_rankings)
    manifest = strategy_result.to_dict()
    manifest.update(
        {
            "artifact_version": "frozen-pseudo-supervision-v1",
            "candidate_rankings": {
                window_id: {
                    source: list(ranking)
                    for source, ranking in sorted(source_rankings.items())
                }
                for window_id, source_rankings in sorted(normalized_rankings.items())
            },
        }
    )
    metadata = dict(query_plan.metadata)
    metadata["frozen_pseudo_supervision"] = manifest
    updates = {
        "pseudo_labels": emitted,
        "pseudo_confidence": confidences,
        "metadata": metadata,
    }
    if hasattr(query_plan, "frozen_pseudo_result"):
        updates["frozen_pseudo_result"] = strategy_result
    if hasattr(query_plan, "pseudo_candidate_rankings"):
        updates["pseudo_candidate_rankings"] = normalized_rankings
    return replace(query_plan, **updates)


__all__ = [
    "PseudoTrainingAdapterConfig",
    "adapt_matched_pseudo_training_rows",
    "adapt_pseudo_training_rows",
    "attach_frozen_pseudo_result",
    "attach_matched_pseudo_arm",
]
