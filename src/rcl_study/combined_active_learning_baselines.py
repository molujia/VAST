"""Independent label-free acquisition baselines for combined study 2.0."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any

import numpy as np

from rcl_study.combined_active_learning_schemas import (
    ACTIVE_LEARNING_SEEDS,
    ANNOTATION_BUDGET,
    DatasetId,
    NO_REPRESENTATION_DEPENDENCY_SHA256,
    SelectorPlan,
)


NO_CLUSTERING_DEPENDENCY_SHA256 = hashlib.sha256(
    b"rcl-active-learning-combined-2.0:no-clustering-dependency"
).hexdigest()
GLOBAL_RANDOM_QUOTA_SHA256 = hashlib.sha256(
    b"rcl-active-learning-combined-2.0:uniform-full-pool-without-replacement"
).hexdigest()
INDEPENDENT_ROUND_SIZES = (8,) + (2,) * 11
KNN_FAULT_MODE_NEIGHBOR_COUNT = 5
FALCON_TIME_CHUNK_COUNT = 4
FALCON_EXPLORATION_FRACTION = 0.5
FALCON_MINIMUM_EXPLORATION_PER_ROUND = 1


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if (
        len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class IndependentQueryRecord:
    query_index: int
    case_id: str
    selector_score: float
    score_definition: str
    score_components: tuple[tuple[str, Any], ...]
    reason: str
    active_learning_seed: int
    input_sha256: str

    def __post_init__(self) -> None:
        if self.query_index < 0:
            raise ValueError("independent query index must be non-negative")
        if not self.case_id:
            raise ValueError("independent query case ID must be non-empty")
        if not math.isfinite(self.selector_score) or self.selector_score < 0.0:
            raise ValueError("independent selector score must be finite and non-negative")
        if not self.score_definition or not self.reason:
            raise ValueError("independent selector provenance must be non-empty")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("active-learning seed is outside the frozen set")
        _require_sha256(self.input_sha256, "independent query input SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "case_id": self.case_id,
            "selector_score": self.selector_score,
            "score_definition": self.score_definition,
            "score_components": dict(self.score_components),
            "reason": self.reason,
            "active_learning_seed": self.active_learning_seed,
            "input_sha256": self.input_sha256,
        }


@dataclass(frozen=True)
class IndependentQueryPlan:
    dataset_id: str
    method_id: str
    selector_id: str
    active_learning_seed: int
    budget: int
    eligible_case_count: int
    candidate_pool_sha256: str
    representation_dependency_sha256: str
    clustering_dependency_sha256: str
    quota_sha256: str
    input_sha256: str
    records: tuple[IndependentQueryRecord, ...]

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if not self.method_id or not self.selector_id:
            raise ValueError("independent method and selector IDs must be non-empty")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("active-learning seed is outside the frozen set")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError(f"independent query budget must be {ANNOTATION_BUDGET}")
        if self.eligible_case_count < self.budget:
            raise ValueError("eligible case count must cover the annotation budget")
        if len(self.records) != self.budget:
            raise ValueError("independent query plan must fill the annotation budget")
        if tuple(record.query_index for record in self.records) != tuple(
            range(self.budget)
        ):
            raise ValueError("independent query indices must be contiguous")
        if len(set(self.selected_case_ids)) != self.budget:
            raise ValueError("independent selected case IDs must be unique")
        if any(
            record.active_learning_seed != self.active_learning_seed
            or record.input_sha256 != self.input_sha256
            for record in self.records
        ):
            raise ValueError("independent query record scope drifted")
        if self.method_id == "global_random":
            if (
                self.representation_dependency_sha256
                != NO_REPRESENTATION_DEPENDENCY_SHA256
            ):
                raise ValueError("global random gained a representation dependency")
        elif self.method_id in {
            "knn_fault_mode",
            "falcon_hybrid",
            "facility_location",
        }:
            if (
                self.representation_dependency_sha256
                == NO_REPRESENTATION_DEPENDENCY_SHA256
            ):
                raise ValueError("independent acquisition requires its label-free geometry")
        else:
            raise ValueError("unsupported independent acquisition method")
        if self.clustering_dependency_sha256 != NO_CLUSTERING_DEPENDENCY_SHA256:
            raise ValueError("independent query plan gained a clustering dependency")
        for value, name in (
            (self.candidate_pool_sha256, "candidate pool SHA-256"),
            (self.representation_dependency_sha256, "representation dependency SHA-256"),
            (self.clustering_dependency_sha256, "clustering dependency SHA-256"),
            (self.quota_sha256, "independent quota SHA-256"),
            (self.input_sha256, "independent input SHA-256"),
        ):
            _require_sha256(value, name)

    @property
    def selected_case_ids(self) -> tuple[str, ...]:
        return tuple(record.case_id for record in self.records)

    def _plan_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "active_learning_seed": self.active_learning_seed,
            "budget": self.budget,
            "eligible_case_count": self.eligible_case_count,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "representation_dependency_sha256": (
                self.representation_dependency_sha256
            ),
            "clustering_dependency_sha256": self.clustering_dependency_sha256,
            "quota_sha256": self.quota_sha256,
            "input_sha256": self.input_sha256,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def plan_sha256(self) -> str:
        return _semantic_sha256(self._plan_dict())

    @property
    def plan_id(self) -> str:
        return (
            f"{self.dataset_id}.{self.method_id}.{self.selector_id}."
            f"seed{self.active_learning_seed}.budget{self.budget}."
            f"{self.plan_sha256[:12]}"
        )

    def to_selector_plan(self) -> SelectorPlan:
        geometry_sha256 = (
            self.clustering_dependency_sha256
            if self.method_id == "global_random"
            else self.representation_dependency_sha256
        )
        return SelectorPlan(
            plan_id=self.plan_id,
            dataset_id=DatasetId(self.dataset_id),
            method_id=self.method_id,
            selector_id=self.selector_id,
            active_learning_seed=self.active_learning_seed,
            budget=self.budget,
            selected_case_ids=self.selected_case_ids,
            geometry_sha256=geometry_sha256,
            quota_sha256=self.quota_sha256,
            plan_sha256=self.plan_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            **self._plan_dict(),
            "selected_case_ids": list(self.selected_case_ids),
            "plan_sha256": self.plan_sha256,
        }


def build_global_random_query_plan(
    *,
    dataset_id: str,
    eligible_case_ids: tuple[str, ...],
    active_learning_seed: int,
    budget: int = ANNOTATION_BUDGET,
) -> IndependentQueryPlan:
    """Uniformly sample the complete eligible pool without replacement."""

    DatasetId(dataset_id)
    if active_learning_seed not in ACTIVE_LEARNING_SEEDS:
        raise ValueError("active-learning seed is outside the frozen set")
    if isinstance(budget, bool) or budget != ANNOTATION_BUDGET:
        raise ValueError(f"global-random budget must be {ANNOTATION_BUDGET}")
    population = tuple(eligible_case_ids)
    if any(not isinstance(case_id, str) or not case_id for case_id in population):
        raise ValueError("eligible case IDs must be non-empty strings")
    if len(population) < budget:
        raise ValueError("global random requires at least 30 eligible cases")
    if len(set(population)) != len(population):
        raise ValueError("global-random eligible case IDs must be unique")

    canonical_population = tuple(sorted(population))
    candidate_pool_sha256 = _semantic_sha256(list(canonical_population))
    input_sha256 = _semantic_sha256(
        {
            "role": "global-random-full-pool-input",
            "dataset_id": dataset_id,
            "active_learning_seed": active_learning_seed,
            "budget": budget,
            "candidate_pool_sha256": candidate_pool_sha256,
            "representation_dependency_sha256": (
                NO_REPRESENTATION_DEPENDENCY_SHA256
            ),
            "clustering_dependency_sha256": NO_CLUSTERING_DEPENDENCY_SHA256,
            "quota_sha256": GLOBAL_RANDOM_QUOTA_SHA256,
        }
    )
    selected = random.Random(active_learning_seed).sample(
        canonical_population,
        k=budget,
    )
    records = tuple(
        IndependentQueryRecord(
            query_index=query_index,
            case_id=case_id,
            selector_score=float(query_index + 1),
            score_definition="uniform_without_replacement_draw_rank",
            score_components=(
                ("draw_rank", float(query_index + 1)),
                ("eligible_case_count", float(len(canonical_population))),
            ),
            reason="uniform_full_pool_without_replacement",
            active_learning_seed=active_learning_seed,
            input_sha256=input_sha256,
        )
        for query_index, case_id in enumerate(selected)
    )
    return IndependentQueryPlan(
        dataset_id=dataset_id,
        method_id="global_random",
        selector_id="global_random",
        active_learning_seed=active_learning_seed,
        budget=budget,
        eligible_case_count=len(canonical_population),
        candidate_pool_sha256=candidate_pool_sha256,
        representation_dependency_sha256=NO_REPRESENTATION_DEPENDENCY_SHA256,
        clustering_dependency_sha256=NO_CLUSTERING_DEPENDENCY_SHA256,
        quota_sha256=GLOBAL_RANDOM_QUOTA_SHA256,
        input_sha256=input_sha256,
        records=records,
    )


def _seeded_tie(active_learning_seed: int, case_id: str, namespace: str) -> str:
    return hashlib.sha256(
        f"{namespace}:{active_learning_seed}:{case_id}".encode("utf-8")
    ).hexdigest()


def _validated_matrix_inputs(
    *,
    dataset_id: str,
    eligible_case_ids: tuple[str, ...],
    representation_matrix: np.ndarray,
    active_learning_seed: int,
) -> tuple[tuple[str, ...], np.ndarray, str, str]:
    DatasetId(dataset_id)
    if active_learning_seed not in ACTIVE_LEARNING_SEEDS:
        raise ValueError("active-learning seed is outside the frozen set")
    case_ids = tuple(eligible_case_ids)
    if len(case_ids) < ANNOTATION_BUDGET:
        raise ValueError("independent acquisition requires at least 30 eligible cases")
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("eligible case IDs must be non-empty strings")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("independent eligible case IDs must be unique")
    matrix = np.asarray(representation_matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(case_ids)
        or matrix.shape[1] < 1
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("representation matrix must be finite and case-aligned")
    order = np.argsort(np.asarray(case_ids, dtype=str), kind="stable")
    canonical_ids = tuple(case_ids[int(index)] for index in order)
    canonical_matrix = np.ascontiguousarray(matrix[order], dtype=np.float64)
    candidate_pool_sha256 = _semantic_sha256(list(canonical_ids))
    representation_sha256 = _semantic_sha256(
        {
            "case_ids": list(canonical_ids),
            "shape": list(canonical_matrix.shape),
            "values": canonical_matrix.tolist(),
        }
    )
    return (
        canonical_ids,
        canonical_matrix,
        candidate_pool_sha256,
        representation_sha256,
    )


def _farthest_first_batch(
    *,
    case_ids: tuple[str, ...],
    matrix: np.ndarray,
    available_indices: list[int],
    selected_indices: list[int],
    batch_size: int,
    active_learning_seed: int,
) -> tuple[list[int], dict[int, float]]:
    chosen: list[int] = []
    scores: dict[int, float] = {}
    reference = list(selected_indices)
    remaining = list(available_indices)
    while remaining and len(chosen) < batch_size:
        if reference:
            distances = np.linalg.norm(
                matrix[np.asarray(remaining)][:, np.newaxis, :]
                - matrix[np.asarray(reference)][np.newaxis, :, :],
                axis=2,
            ).min(axis=1)
        else:
            centroid = matrix[np.asarray(remaining)].mean(axis=0)
            distances = np.linalg.norm(
                matrix[np.asarray(remaining)] - centroid,
                axis=1,
            )
        best_position = max(
            range(len(remaining)),
            key=lambda position: (
                float(distances[position]),
                _seeded_tie(
                    active_learning_seed,
                    case_ids[remaining[position]],
                    "kff",
                ),
            ),
        )
        index = remaining.pop(best_position)
        chosen.append(index)
        scores[index] = float(distances[best_position])
        reference.append(index)
    return chosen, scores


def _knn_mode_predictions(
    *,
    matrix: np.ndarray,
    remaining_indices: list[int],
    selected_indices: list[int],
    revealed_fault_types: Mapping[str, str],
    case_ids: tuple[str, ...],
) -> dict[int, tuple[str, float, dict[str, float]]]:
    selected_labels = [revealed_fault_types[case_ids[index]] for index in selected_indices]
    predictions = {}
    for index in remaining_indices:
        distances = np.linalg.norm(matrix[np.asarray(selected_indices)] - matrix[index], axis=1)
        neighbor_order = np.argsort(distances, kind="stable")[:
            min(KNN_FAULT_MODE_NEIGHBOR_COUNT, len(selected_indices))
        ]
        weights: defaultdict[str, float] = defaultdict(float)
        for position in neighbor_order:
            distance = max(float(distances[int(position)]), 1e-9)
            weights[selected_labels[int(position)]] += 1.0 / distance
        total = sum(weights.values())
        normalized = {
            label: float(weight / total) for label, weight in weights.items()
        }
        predicted = max(
            normalized,
            key=lambda label: (normalized[label], label),
        )
        predictions[index] = (
            predicted,
            normalized[predicted],
            dict(sorted(normalized.items())),
        )
    return predictions


def _round_robin_mode_indices(
    *,
    queues: Mapping[str, list[int]],
    mode_order: list[str],
    batch_size: int,
) -> list[int]:
    mutable = {mode: list(queues[mode]) for mode in mode_order}
    chosen: list[int] = []
    while len(chosen) < batch_size:
        progressed = False
        for mode in mode_order:
            if mutable[mode] and len(chosen) < batch_size:
                chosen.append(mutable[mode].pop(0))
                progressed = True
        if not progressed:
            break
    return chosen


def build_knn_fault_mode_query_plan(
    *,
    dataset_id: str,
    eligible_case_ids: tuple[str, ...],
    representation_matrix: np.ndarray,
    active_learning_seed: int,
    fault_type_oracle: Any,
    neighbor_count: int = KNN_FAULT_MODE_NEIGHBOR_COUNT,
    representation_dependency_sha256: str | None = None,
) -> IndependentQueryPlan:
    """Port frozen kNN fault-mode acquisition with commit-then-reveal labels."""

    if isinstance(neighbor_count, bool) or neighbor_count != 5:
        raise ValueError("kNN fault-mode neighbor count is frozen at 5")
    if not hasattr(fault_type_oracle, "reveal_fault_types"):
        raise ValueError("fault-type oracle must expose reveal_fault_types")
    oracle_provenance_sha256 = getattr(
        fault_type_oracle,
        "provenance_sha256",
        "",
    )
    _require_sha256(oracle_provenance_sha256, "fault-type oracle provenance SHA-256")
    (
        case_ids,
        matrix,
        candidate_pool_sha256,
        matrix_value_sha256,
    ) = _validated_matrix_inputs(
        dataset_id=dataset_id,
        eligible_case_ids=eligible_case_ids,
        representation_matrix=representation_matrix,
        active_learning_seed=active_learning_seed,
    )
    formal_representation_sha256 = (
        matrix_value_sha256
        if representation_dependency_sha256 is None
        else representation_dependency_sha256
    )
    _require_sha256(
        formal_representation_sha256,
        "formal representation dependency SHA-256",
    )
    quota_sha256 = _semantic_sha256(
        {
            "method_id": "knn_fault_mode",
            "round_sizes": list(INDEPENDENT_ROUND_SIZES),
            "neighbor_count": KNN_FAULT_MODE_NEIGHBOR_COUNT,
        }
    )
    input_sha256 = _semantic_sha256(
        {
            "role": "knn-fault-mode-input",
            "dataset_id": dataset_id,
            "active_learning_seed": active_learning_seed,
            "candidate_pool_sha256": candidate_pool_sha256,
            "representation_sha256": matrix_value_sha256,
            "formal_representation_dependency_sha256": (
                formal_representation_sha256
            ),
            "oracle_provenance_sha256": oracle_provenance_sha256,
            "quota_sha256": quota_sha256,
            "clustering_dependency_sha256": NO_CLUSTERING_DEPENDENCY_SHA256,
        }
    )

    index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
    selected_indices: list[int] = []
    revealed_fault_types: dict[str, str] = {}
    records: list[IndependentQueryRecord] = []
    for round_index, batch_size in enumerate(INDEPENDENT_ROUND_SIZES):
        selected_set = set(selected_indices)
        remaining = [
            index for index in range(len(case_ids)) if index not in selected_set
        ]
        if not selected_indices:
            chosen, farthest_scores = _farthest_first_batch(
                case_ids=case_ids,
                matrix=matrix,
                available_indices=remaining,
                selected_indices=[],
                batch_size=batch_size,
                active_learning_seed=active_learning_seed,
            )
            score_details = {
                index: (
                    farthest_scores[index],
                    "no_revealed_fault_type_k_center",
                    (
                        ("selection_stage", "label_free_k_center"),
                        ("k_center_distance", farthest_scores[index]),
                        ("revealed_label_count_before_selection", 0.0),
                        ("round_index", float(round_index)),
                        ("neighbor_count", float(KNN_FAULT_MODE_NEIGHBOR_COUNT)),
                    ),
                )
                for index in chosen
            }
        else:
            predictions = _knn_mode_predictions(
                matrix=matrix,
                remaining_indices=remaining,
                selected_indices=selected_indices,
                revealed_fault_types=revealed_fault_types,
                case_ids=case_ids,
            )
            queues: defaultdict[str, list[int]] = defaultdict(list)
            for index in remaining:
                queues[predictions[index][0]].append(index)
            for mode, indices in queues.items():
                queues[mode] = sorted(
                    indices,
                    key=lambda index: (
                        predictions[index][1],
                        _seeded_tie(
                            active_learning_seed,
                            case_ids[index],
                            "knn-mode",
                        ),
                    ),
                )
            queried_counts = Counter(revealed_fault_types.values())
            mode_order = sorted(
                queues,
                key=lambda mode: (
                    queried_counts[mode],
                    _seeded_tie(active_learning_seed, mode, "knn-types"),
                ),
            )
            chosen = _round_robin_mode_indices(
                queues=queues,
                mode_order=mode_order,
                batch_size=batch_size,
            )
            score_details = {}
            for index in chosen:
                predicted, confidence, weights = predictions[index]
                score_details[index] = (
                    1.0 - confidence,
                    "predicted_mode_coverage_then_within_mode_uncertainty",
                    (
                        (
                            "selection_stage",
                            "revealed_label_knn_fault_mode_coverage",
                        ),
                        ("predicted_fault_type", predicted),
                        ("prediction_confidence", confidence),
                        ("neighbor_label_weights", weights),
                        (
                            "revealed_label_count_before_selection",
                            float(len(revealed_fault_types)),
                        ),
                        ("round_index", float(round_index)),
                        ("neighbor_count", float(KNN_FAULT_MODE_NEIGHBOR_COUNT)),
                    ),
                )
        if len(chosen) != batch_size:
            raise ValueError("kNN fault-mode could not fill a frozen round")

        chosen_case_ids = tuple(case_ids[index] for index in chosen)
        for round_position, index in enumerate(chosen):
            selector_score, reason, components = score_details[index]
            records.append(
                IndependentQueryRecord(
                    query_index=len(records),
                    case_id=case_ids[index],
                    selector_score=float(selector_score),
                    score_definition=(
                        "k_center_distance"
                        if not selected_indices
                        else "one_minus_knn_prediction_confidence"
                    ),
                    score_components=components
                    + (("round_position", float(round_position)),),
                    reason=reason,
                    active_learning_seed=active_learning_seed,
                    input_sha256=input_sha256,
                )
            )
        selected_indices.extend(chosen)

        revealed = fault_type_oracle.reveal_fault_types(chosen_case_ids)
        if not isinstance(revealed, Mapping) or set(revealed) != set(chosen_case_ids):
            raise ValueError("fault-type oracle reveal does not match committed cases")
        for case_id in chosen_case_ids:
            fault_type = revealed[case_id]
            if not isinstance(fault_type, str) or not fault_type.strip():
                raise ValueError("revealed fault type must be a non-empty string")
            revealed_fault_types[case_id] = fault_type.strip()
        if any(case_id not in revealed_fault_types for case_id in chosen_case_ids):
            raise ValueError("committed case lacks revealed fault type")
        if any(case_id not in index_by_case for case_id in revealed_fault_types):
            raise ValueError("fault-type oracle revealed an ineligible case")

    return IndependentQueryPlan(
        dataset_id=dataset_id,
        method_id="knn_fault_mode",
        selector_id="knn_fault_mode",
        active_learning_seed=active_learning_seed,
        budget=ANNOTATION_BUDGET,
        eligible_case_count=len(case_ids),
        candidate_pool_sha256=candidate_pool_sha256,
        representation_dependency_sha256=formal_representation_sha256,
        clustering_dependency_sha256=NO_CLUSTERING_DEPENDENCY_SHA256,
        quota_sha256=quota_sha256,
        input_sha256=input_sha256,
        records=tuple(records),
    )


def build_falcon_hybrid_query_plan(
    *,
    dataset_id: str,
    eligible_case_ids: tuple[str, ...],
    representation_matrix: np.ndarray,
    relative_time_order: tuple[float, ...],
    uncertainty_scores: tuple[float, ...],
    active_learning_seed: int,
    time_chunk_count: int = FALCON_TIME_CHUNK_COUNT,
    exploration_fraction: float = FALCON_EXPLORATION_FRACTION,
    minimum_exploration_per_round: int = FALCON_MINIMUM_EXPLORATION_PER_ROUND,
    representation_dependency_sha256: str | None = None,
) -> IndependentQueryPlan:
    """Port frozen Falcon LC/KFF hybrid over label-free 2.0 inputs."""

    if isinstance(time_chunk_count, bool) or time_chunk_count != 4:
        raise ValueError("Falcon time chunk count is frozen at 4")
    if exploration_fraction != 0.5:
        raise ValueError("Falcon exploration fraction is frozen at 0.5")
    if (
        isinstance(minimum_exploration_per_round, bool)
        or minimum_exploration_per_round != 1
    ):
        raise ValueError("Falcon minimum exploration per round is frozen at 1")
    original_case_ids = tuple(eligible_case_ids)
    if len(relative_time_order) != len(original_case_ids):
        raise ValueError("Falcon relative-time order must be case-aligned")
    if len(uncertainty_scores) != len(original_case_ids):
        raise ValueError("Falcon uncertainty scores must be case-aligned")
    try:
        time_by_case = {
            case_id: float(value)
            for case_id, value in zip(original_case_ids, relative_time_order)
        }
        uncertainty_by_case = {
            case_id: float(value)
            for case_id, value in zip(original_case_ids, uncertainty_scores)
        }
    except (TypeError, ValueError) as error:
        raise ValueError("Falcon temporal and uncertainty values must be finite") from error
    if not all(math.isfinite(value) for value in time_by_case.values()):
        raise ValueError("Falcon relative-time order must be finite")
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in uncertainty_by_case.values()
    ):
        raise ValueError("Falcon uncertainty scores must be finite in [0, 1]")

    (
        case_ids,
        matrix,
        candidate_pool_sha256,
        matrix_value_sha256,
    ) = _validated_matrix_inputs(
        dataset_id=dataset_id,
        eligible_case_ids=original_case_ids,
        representation_matrix=representation_matrix,
        active_learning_seed=active_learning_seed,
    )
    formal_representation_sha256 = (
        matrix_value_sha256
        if representation_dependency_sha256 is None
        else representation_dependency_sha256
    )
    _require_sha256(
        formal_representation_sha256,
        "formal representation dependency SHA-256",
    )
    times = np.asarray([time_by_case[case_id] for case_id in case_ids], dtype=float)
    uncertainties = np.asarray(
        [uncertainty_by_case[case_id] for case_id in case_ids],
        dtype=float,
    )
    temporal_order = np.argsort(times, kind="stable")
    chunks = np.zeros(len(case_ids), dtype=int)
    for rank, index in enumerate(temporal_order):
        chunks[int(index)] = min(
            FALCON_TIME_CHUNK_COUNT - 1,
            int(rank * FALCON_TIME_CHUNK_COUNT / len(case_ids)),
        )

    quota_sha256 = _semantic_sha256(
        {
            "method_id": "falcon_hybrid",
            "round_sizes": list(INDEPENDENT_ROUND_SIZES),
            "time_chunk_count": FALCON_TIME_CHUNK_COUNT,
            "exploration_fraction": FALCON_EXPLORATION_FRACTION,
            "minimum_exploration_per_round": (
                FALCON_MINIMUM_EXPLORATION_PER_ROUND
            ),
        }
    )
    input_sha256 = _semantic_sha256(
        {
            "role": "falcon-hybrid-input",
            "dataset_id": dataset_id,
            "active_learning_seed": active_learning_seed,
            "candidate_pool_sha256": candidate_pool_sha256,
            "representation_sha256": matrix_value_sha256,
            "formal_representation_dependency_sha256": (
                formal_representation_sha256
            ),
            "relative_time_order": times.tolist(),
            "uncertainty_scores": uncertainties.tolist(),
            "quota_sha256": quota_sha256,
            "clustering_dependency_sha256": NO_CLUSTERING_DEPENDENCY_SHA256,
        }
    )

    selected_indices: list[int] = []
    records: list[IndependentQueryRecord] = []
    for round_index, batch_size in enumerate(INDEPENDENT_ROUND_SIZES):
        selected_set = set(selected_indices)
        available = [
            index for index in range(len(case_ids)) if index not in selected_set
        ]
        explore_count = max(
            FALCON_MINIMUM_EXPLORATION_PER_ROUND,
            int(round(batch_size * FALCON_EXPLORATION_FRACTION)),
        )
        explore_count = min(
            batch_size - 1 if batch_size > 1 else 1,
            explore_count,
        )
        exploit_count = batch_size - explore_count
        lc_order = sorted(
            available,
            key=lambda index: (
                -float(uncertainties[index]),
                int(chunks[index]),
                _seeded_tie(active_learning_seed, case_ids[index], "falcon-lc"),
            ),
        )
        chosen_lc = lc_order[:exploit_count]
        remaining = [index for index in available if index not in set(chosen_lc)]
        chosen_kff, _ = _farthest_first_batch(
            case_ids=case_ids,
            matrix=matrix,
            available_indices=remaining,
            selected_indices=selected_indices + chosen_lc,
            batch_size=explore_count,
            active_learning_seed=active_learning_seed,
        )
        chosen: list[int] = []
        components: dict[int, str] = {}
        for offset in range(max(len(chosen_lc), len(chosen_kff))):
            if offset < len(chosen_lc):
                index = chosen_lc[offset]
                chosen.append(index)
                components[index] = "least_confidence"
            if offset < len(chosen_kff):
                index = chosen_kff[offset]
                chosen.append(index)
                components[index] = "kernel_furthest_first"
        if len(chosen) != batch_size:
            raise ValueError("Falcon hybrid could not fill a frozen round")

        for round_position, index in enumerate(chosen):
            if selected_indices:
                kff_distance = float(
                    np.linalg.norm(
                        matrix[np.asarray(selected_indices)] - matrix[index],
                        axis=1,
                    ).min()
                )
            else:
                kff_distance = float(
                    np.linalg.norm(matrix[index] - matrix[np.asarray(available)].mean(axis=0))
                )
            component = components[index]
            uncertainty = float(uncertainties[index])
            selector_score = (
                uncertainty if component == "least_confidence" else kff_distance
            )
            records.append(
                IndependentQueryRecord(
                    query_index=len(records),
                    case_id=case_ids[index],
                    selector_score=selector_score,
                    score_definition=(
                        "least_confidence_score"
                        if component == "least_confidence"
                        else "kff_distance"
                    ),
                    score_components=(
                        ("timestamp_chunk", float(chunks[index])),
                        ("least_confidence_score", uncertainty),
                        ("kff_distance", kff_distance),
                        ("selection_component", component),
                        ("round_index", float(round_index)),
                        ("round_position", float(round_position)),
                    ),
                    reason="falcon_real_time_lc_kff_hybrid",
                    active_learning_seed=active_learning_seed,
                    input_sha256=input_sha256,
                )
            )
        selected_indices.extend(chosen)

    return IndependentQueryPlan(
        dataset_id=dataset_id,
        method_id="falcon_hybrid",
        selector_id="falcon_hybrid",
        active_learning_seed=active_learning_seed,
        budget=ANNOTATION_BUDGET,
        eligible_case_count=len(case_ids),
        candidate_pool_sha256=candidate_pool_sha256,
        representation_dependency_sha256=formal_representation_sha256,
        clustering_dependency_sha256=NO_CLUSTERING_DEPENDENCY_SHA256,
        quota_sha256=quota_sha256,
        input_sha256=input_sha256,
        records=tuple(records),
    )


def build_facility_location_query_plan(
    *,
    dataset_id: str,
    eligible_case_ids: tuple[str, ...],
    case_affinity: np.ndarray,
    active_learning_seed: int,
    budget: int = ANNOTATION_BUDGET,
    representation_dependency_sha256: str | None = None,
) -> IndependentQueryPlan:
    """Greedily maximize coverage on an approved label-free case affinity."""

    DatasetId(dataset_id)
    if active_learning_seed not in ACTIVE_LEARNING_SEEDS:
        raise ValueError("active-learning seed is outside the frozen set")
    if isinstance(budget, bool) or budget != ANNOTATION_BUDGET:
        raise ValueError(f"Facility-Location budget must be {ANNOTATION_BUDGET}")
    original_case_ids = tuple(eligible_case_ids)
    if len(original_case_ids) < budget:
        raise ValueError("Facility-Location requires at least 30 eligible cases")
    if any(
        not isinstance(case_id, str) or not case_id
        for case_id in original_case_ids
    ):
        raise ValueError("eligible case IDs must be non-empty strings")
    if len(set(original_case_ids)) != len(original_case_ids):
        raise ValueError("Facility-Location eligible case IDs must be unique")
    affinity = np.asarray(case_affinity, dtype=float)
    expected_shape = (len(original_case_ids), len(original_case_ids))
    if affinity.shape != expected_shape or not np.all(np.isfinite(affinity)):
        raise ValueError("case affinity must be a finite square case-aligned matrix")
    if not np.allclose(affinity, affinity.T, rtol=0.0, atol=1e-12):
        raise ValueError("case affinity must be symmetric")
    if np.any(affinity < -1e-12) or np.any(affinity > 1.0 + 1e-12):
        raise ValueError("case affinity must be bounded in [0, 1]")
    if not np.allclose(np.diag(affinity), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("case affinity diagonal must equal one")

    order = np.argsort(np.asarray(original_case_ids, dtype=str), kind="stable")
    case_ids = tuple(original_case_ids[int(index)] for index in order)
    affinity = np.clip(
        affinity[np.ix_(order, order)],
        0.0,
        1.0,
    )
    candidate_pool_sha256 = _semantic_sha256(list(case_ids))
    affinity_sha256 = _semantic_sha256(
        {
            "case_ids": list(case_ids),
            "shape": list(affinity.shape),
            "values": affinity.tolist(),
        }
    )
    formal_representation_sha256 = (
        affinity_sha256
        if representation_dependency_sha256 is None
        else representation_dependency_sha256
    )
    _require_sha256(
        formal_representation_sha256,
        "formal representation dependency SHA-256",
    )
    quota_sha256 = _semantic_sha256(
        {
            "method_id": "facility_location",
            "objective": "sum_max_label_free_case_affinity",
            "budget": budget,
            "weighting": "uniform_cases",
        }
    )
    input_sha256 = _semantic_sha256(
        {
            "role": "facility-location-input",
            "dataset_id": dataset_id,
            "active_learning_seed": active_learning_seed,
            "candidate_pool_sha256": candidate_pool_sha256,
            "case_affinity_sha256": affinity_sha256,
            "formal_representation_dependency_sha256": (
                formal_representation_sha256
            ),
            "quota_sha256": quota_sha256,
            "clustering_dependency_sha256": NO_CLUSTERING_DEPENDENCY_SHA256,
        }
    )

    coverage = np.zeros(len(case_ids), dtype=float)
    remaining = list(range(len(case_ids)))
    records: list[IndependentQueryRecord] = []
    while remaining and len(records) < budget:
        gains = {
            index: float(
                np.sum(np.maximum(coverage, affinity[:, index]) - coverage)
            )
            for index in remaining
        }
        best = max(
            remaining,
            key=lambda index: (
                gains[index],
                _seeded_tie(active_learning_seed, case_ids[index], "facility"),
            ),
        )
        coverage_sum_before = float(np.sum(coverage))
        coverage = np.maximum(coverage, affinity[:, best])
        coverage_sum_after = float(np.sum(coverage))
        gain = gains[best]
        records.append(
            IndependentQueryRecord(
                query_index=len(records),
                case_id=case_ids[best],
                selector_score=gain,
                score_definition="facility_location_marginal_coverage_gain",
                score_components=(
                    ("marginal_gain", gain),
                    ("coverage_sum_before", coverage_sum_before),
                    ("coverage_sum_after", coverage_sum_after),
                    (
                        "mean_coverage_after",
                        coverage_sum_after / float(len(case_ids)),
                    ),
                ),
                reason="greedy_label_free_facility_location_gain",
                active_learning_seed=active_learning_seed,
                input_sha256=input_sha256,
            )
        )
        remaining.remove(best)
    if len(records) != budget:
        raise ValueError("Facility-Location could not fill the annotation budget")

    return IndependentQueryPlan(
        dataset_id=dataset_id,
        method_id="facility_location",
        selector_id="facility_location",
        active_learning_seed=active_learning_seed,
        budget=budget,
        eligible_case_count=len(case_ids),
        candidate_pool_sha256=candidate_pool_sha256,
        representation_dependency_sha256=formal_representation_sha256,
        clustering_dependency_sha256=NO_CLUSTERING_DEPENDENCY_SHA256,
        quota_sha256=quota_sha256,
        input_sha256=input_sha256,
        records=tuple(records),
    )


__all__ = [
    "FALCON_EXPLORATION_FRACTION",
    "FALCON_MINIMUM_EXPLORATION_PER_ROUND",
    "FALCON_TIME_CHUNK_COUNT",
    "GLOBAL_RANDOM_QUOTA_SHA256",
    "IndependentQueryPlan",
    "IndependentQueryRecord",
    "INDEPENDENT_ROUND_SIZES",
    "KNN_FAULT_MODE_NEIGHBOR_COUNT",
    "NO_CLUSTERING_DEPENDENCY_SHA256",
    "NO_REPRESENTATION_DEPENDENCY_SHA256",
    "build_facility_location_query_plan",
    "build_global_random_query_plan",
    "build_falcon_hybrid_query_plan",
    "build_knn_fault_mode_query_plan",
]
