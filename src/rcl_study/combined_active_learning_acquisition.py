"""Matched cluster acquisition primitives for combined active learning 2.0."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from rcl_study.combined_active_learning_clustering import (
    DBSCANClusteringResult,
    EffectiveClusterPartition,
    HDBSCANClusteringResult,
    KMeansClusteringResult,
    MutualKNNClusteringResult,
    evaluate_structure_gate,
)
from rcl_study.combined_active_learning_representation import (
    RepresentationMatrixCandidate,
)
from rcl_study.combined_active_learning_schemas import (
    ACTIVE_LEARNING_SEEDS,
    ANNOTATION_BUDGET,
    DatasetId,
    SelectorPlan,
)
from sklearn.metrics import pairwise_distances


QUOTA_POLICY = "equal_weight_round_robin"
MAX_RESIDUAL_QUERIES = 2


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ClusterAllocationStep:
    query_index: int
    allocation_kind: str
    raw_label: int | None
    effective_round_index: int | None

    def __post_init__(self) -> None:
        if self.query_index < 0:
            raise ValueError("allocation query index must be non-negative")
        if self.allocation_kind == "effective_cluster":
            if self.raw_label is None or self.raw_label < 0:
                raise ValueError("effective allocation requires a cluster label")
            if self.effective_round_index is None or self.effective_round_index < 0:
                raise ValueError("effective allocation requires a round index")
        elif self.allocation_kind == "residual":
            if self.raw_label is not None or self.effective_round_index is not None:
                raise ValueError("residual allocation cannot carry cluster coordinates")
        else:
            raise ValueError("unsupported cluster allocation kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "allocation_kind": self.allocation_kind,
            "raw_label": self.raw_label,
            "effective_round_index": self.effective_round_index,
        }


@dataclass(frozen=True)
class MatchedClusterAllocation:
    dataset_id: str
    clusterer_id: str
    active_learning_seed: int
    partition_sha256: str
    budget: int
    quota_policy: str
    cluster_visit_order: tuple[int, ...]
    effective_cluster_quotas: tuple[tuple[int, int], ...]
    residual_quota: int
    steps: tuple[ClusterAllocationStep, ...]

    def __post_init__(self) -> None:
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError(f"matched allocation budget must be {ANNOTATION_BUDGET}")
        if self.quota_policy != QUOTA_POLICY:
            raise ValueError("matched allocation must use equal-weight round-robin")
        if len(self.steps) != self.budget:
            raise ValueError("matched allocation must fill the annotation budget")
        if tuple(step.query_index for step in self.steps) != tuple(
            range(self.budget)
        ):
            raise ValueError("matched allocation query indices must be contiguous")
        if not 0 <= self.residual_quota <= MAX_RESIDUAL_QUERIES:
            raise ValueError("matched allocation residual quota exceeds two")

    def _quota_dict(self) -> dict[str, Any]:
        return {
            "quota_policy": self.quota_policy,
            "budget": self.budget,
            "cluster_visit_order": list(self.cluster_visit_order),
            "effective_cluster_quotas": {
                str(label): quota for label, quota in self.effective_cluster_quotas
            },
            "residual_quota": self.residual_quota,
        }

    @property
    def quota_sha256(self) -> str:
        return _semantic_sha256(self._quota_dict())

    @property
    def allocation_sha256(self) -> str:
        return _semantic_sha256(
            {
                "dataset_id": self.dataset_id,
                "clusterer_id": self.clusterer_id,
                "active_learning_seed": self.active_learning_seed,
                "partition_sha256": self.partition_sha256,
                **self._quota_dict(),
                "steps": [step.to_dict() for step in self.steps],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "clusterer_id": self.clusterer_id,
            "active_learning_seed": self.active_learning_seed,
            "partition_sha256": self.partition_sha256,
            **self._quota_dict(),
            "quota_sha256": self.quota_sha256,
            "steps": [step.to_dict() for step in self.steps],
            "allocation_sha256": self.allocation_sha256,
        }


@dataclass(frozen=True)
class RankedClusterCase:
    case_id: str
    raw_label: int
    selector_id: str
    rank_within_cluster: int
    selector_score: float
    score_definition: str
    score_components: tuple[tuple[str, float], ...]
    reason: str
    is_core: bool | None = None
    is_border: bool | None = None
    is_noise: bool = False
    is_residual: bool = False

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("ranked cluster case ID must be non-empty")
        if self.raw_label < 0:
            raise ValueError("ranked cluster case requires an effective label")
        if self.selector_id not in {"center", "boundary", "within_cluster_random"}:
            raise ValueError("ranked cluster selector is unsupported")
        if self.rank_within_cluster < 1:
            raise ValueError("rank within cluster must be one-based")
        if not math.isfinite(self.selector_score) or self.selector_score < 0.0:
            raise ValueError("selector score must be finite and non-negative")
        if not self.score_definition or not self.reason:
            raise ValueError("selector score provenance must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "raw_label": self.raw_label,
            "selector_id": self.selector_id,
            "rank_within_cluster": self.rank_within_cluster,
            "selector_score": self.selector_score,
            "score_definition": self.score_definition,
            "score_components": dict(self.score_components),
            "reason": self.reason,
            "is_core": self.is_core,
            "is_border": self.is_border,
            "is_noise": self.is_noise,
            "is_residual": self.is_residual,
        }


@dataclass(frozen=True)
class ClusterSelectorQueues:
    clusterer_id: str
    selector_id: str
    active_learning_seed: int
    geometry_sha256: str
    representation_matrix_sha256: str
    cluster_queues: tuple[tuple[int, tuple[RankedClusterCase, ...]], ...]

    def queue_for(self, raw_label: int) -> tuple[RankedClusterCase, ...]:
        for label, queue in self.cluster_queues:
            if label == raw_label:
                return queue
        raise KeyError(raw_label)

    @property
    def queues_sha256(self) -> str:
        return _semantic_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "clusterer_id": self.clusterer_id,
            "selector_id": self.selector_id,
            "active_learning_seed": self.active_learning_seed,
            "geometry_sha256": self.geometry_sha256,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "cluster_queues": {
                str(label): [item.to_dict() for item in queue]
                for label, queue in self.cluster_queues
            },
        }
        if include_hash:
            payload["queues_sha256"] = self.queues_sha256
        return payload


@dataclass(frozen=True)
class MatchedSelectorContract:
    selector_id: str
    dataset_id: str
    clusterer_id: str
    active_learning_seed: int
    geometry_sha256: str
    partition_sha256: str
    representation_matrix_sha256: str
    quota_sha256: str
    allocation_sha256: str
    queues_sha256: str
    case_sets_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_id": self.selector_id,
            "dataset_id": self.dataset_id,
            "clusterer_id": self.clusterer_id,
            "active_learning_seed": self.active_learning_seed,
            "geometry_sha256": self.geometry_sha256,
            "partition_sha256": self.partition_sha256,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "quota_sha256": self.quota_sha256,
            "allocation_sha256": self.allocation_sha256,
            "queues_sha256": self.queues_sha256,
            "case_sets_sha256": self.case_sets_sha256,
        }


@dataclass(frozen=True)
class QueryCaseStatus:
    is_core: bool | None
    is_border: bool | None
    is_noise: bool

    def __post_init__(self) -> None:
        if not isinstance(self.is_noise, bool):
            raise ValueError("query case noise status must be boolean")
        for value, name in (
            (self.is_core, "core"),
            (self.is_border, "border"),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"query case {name} status must be boolean or null")
        if self.is_core is True and self.is_border is True:
            raise ValueError("query case cannot be both core and border")
        if self.is_noise and (self.is_core is True or self.is_border is True):
            raise ValueError("query case noise status conflicts with core/border")

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "is_core": self.is_core,
            "is_border": self.is_border,
            "is_noise": self.is_noise,
        }


@dataclass(frozen=True)
class ClusterQueryRecord:
    query_index: int
    case_id: str
    raw_label: int
    selector_id: str
    rank_within_cluster: int
    selector_score: float
    score_definition: str
    score_components: tuple[tuple[str, float], ...]
    reason: str
    is_core: bool | None
    is_border: bool | None
    is_noise: bool
    is_residual: bool
    active_learning_seed: int
    input_sha256: str

    def __post_init__(self) -> None:
        if self.query_index < 0:
            raise ValueError("query index must be non-negative")
        if not self.case_id:
            raise ValueError("query case ID must be non-empty")
        if isinstance(self.raw_label, bool) or not isinstance(self.raw_label, int):
            raise ValueError("query raw label must be an integer")
        if self.selector_id not in {"center", "boundary", "within_cluster_random"}:
            raise ValueError("query selector is unsupported")
        if self.rank_within_cluster < 1:
            raise ValueError("query rank within cluster must be one-based")
        if not math.isfinite(self.selector_score) or self.selector_score < 0.0:
            raise ValueError("query selector score must be finite and non-negative")
        if not self.score_definition or not self.reason:
            raise ValueError("query selector provenance must be non-empty")
        QueryCaseStatus(
            is_core=self.is_core,
            is_border=self.is_border,
            is_noise=self.is_noise,
        )
        if not isinstance(self.is_residual, bool):
            raise ValueError("query residual status must be boolean")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("query active-learning seed is outside the frozen set")
        _require_sha256(self.input_sha256, "query input SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "case_id": self.case_id,
            "raw_label": self.raw_label,
            "selector_id": self.selector_id,
            "rank_within_cluster": self.rank_within_cluster,
            "selector_score": self.selector_score,
            "score_definition": self.score_definition,
            "score_components": dict(self.score_components),
            "reason": self.reason,
            "is_core": self.is_core,
            "is_border": self.is_border,
            "is_noise": self.is_noise,
            "is_residual": self.is_residual,
            "active_learning_seed": self.active_learning_seed,
            "input_sha256": self.input_sha256,
        }


@dataclass(frozen=True)
class ClusterQueryPlan:
    dataset_id: str
    clusterer_id: str
    selector_id: str
    active_learning_seed: int
    budget: int
    representation_matrix_sha256: str
    geometry_sha256: str
    partition_sha256: str
    quota_sha256: str
    allocation_sha256: str
    queues_sha256: str
    case_sets_sha256: str
    input_sha256: str
    records: tuple[ClusterQueryRecord, ...]

    def __post_init__(self) -> None:
        if self.selector_id not in {"center", "boundary", "within_cluster_random"}:
            raise ValueError("cluster query-plan selector is unsupported")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("query-plan active-learning seed is outside the frozen set")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError(f"cluster query-plan budget must be {ANNOTATION_BUDGET}")
        if len(self.records) != self.budget:
            raise ValueError("cluster query plan must fill the annotation budget")
        if tuple(record.query_index for record in self.records) != tuple(
            range(self.budget)
        ):
            raise ValueError("cluster query indices must be contiguous")
        selected_case_ids = self.selected_case_ids
        if len(set(selected_case_ids)) != len(selected_case_ids):
            raise ValueError("cluster query-plan case IDs must be unique")
        if any(
            record.selector_id != self.selector_id
            or record.active_learning_seed != self.active_learning_seed
            or record.input_sha256 != self.input_sha256
            for record in self.records
        ):
            raise ValueError("cluster query record scope drifted")
        for value, name in (
            (self.representation_matrix_sha256, "representation matrix SHA-256"),
            (self.geometry_sha256, "geometry SHA-256"),
            (self.partition_sha256, "partition SHA-256"),
            (self.quota_sha256, "quota SHA-256"),
            (self.allocation_sha256, "allocation SHA-256"),
            (self.queues_sha256, "queues SHA-256"),
            (self.case_sets_sha256, "case sets SHA-256"),
            (self.input_sha256, "query input SHA-256"),
        ):
            _require_sha256(value, name)

    @property
    def selected_case_ids(self) -> tuple[str, ...]:
        return tuple(record.case_id for record in self.records)

    def _plan_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "clusterer_id": self.clusterer_id,
            "selector_id": self.selector_id,
            "active_learning_seed": self.active_learning_seed,
            "budget": self.budget,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "geometry_sha256": self.geometry_sha256,
            "partition_sha256": self.partition_sha256,
            "quota_sha256": self.quota_sha256,
            "allocation_sha256": self.allocation_sha256,
            "queues_sha256": self.queues_sha256,
            "case_sets_sha256": self.case_sets_sha256,
            "input_sha256": self.input_sha256,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def plan_sha256(self) -> str:
        return _semantic_sha256(self._plan_dict())

    @property
    def plan_id(self) -> str:
        return (
            f"{self.dataset_id}.{self.clusterer_id}.{self.selector_id}."
            f"seed{self.active_learning_seed}.budget{self.budget}."
            f"{self.plan_sha256[:12]}"
        )

    def to_selector_plan(self) -> SelectorPlan:
        return SelectorPlan(
            plan_id=self.plan_id,
            dataset_id=DatasetId(self.dataset_id),
            method_id=self.clusterer_id,
            selector_id=self.selector_id,
            active_learning_seed=self.active_learning_seed,
            budget=self.budget,
            selected_case_ids=self.selected_case_ids,
            geometry_sha256=self.geometry_sha256,
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


def _cluster_visit_order(partition: EffectiveClusterPartition) -> tuple[int, ...]:
    def key(raw_label: int) -> str:
        return _semantic_sha256(
            {
                "role": "matched-cluster-visit-order",
                "partition_sha256": partition.partition_sha256,
                "active_learning_seed": partition.active_learning_seed,
                "raw_label": raw_label,
            }
        )

    return tuple(
        sorted(
            (cluster.raw_label for cluster in partition.effective_clusters),
            key=key,
        )
    )


def build_equal_weight_round_robin_allocation(
    partition: EffectiveClusterPartition,
    *,
    budget: int = ANNOTATION_BUDGET,
) -> MatchedClusterAllocation:
    """Allocate matched effective-cluster and residual slots at budget 30."""

    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("matched allocation requires an effective-cluster partition")
    if isinstance(budget, bool) or budget != ANNOTATION_BUDGET:
        raise ValueError(f"matched allocation budget must be {ANNOTATION_BUDGET}")
    if len(partition.effective_clusters) < 2:
        raise ValueError("matched allocation requires at least two effective clusters")
    if len(partition.case_ids) < budget:
        raise ValueError("candidate population cannot fill the annotation budget")

    cluster_order = _cluster_visit_order(partition)
    if len(cluster_order) > budget:
        raise ValueError("budget cannot complete the first effective-cluster round")
    residual_quota = min(
        MAX_RESIDUAL_QUERIES,
        len(partition.residual_case_ids),
        budget - len(cluster_order),
    )
    effective_slots = budget - residual_quota
    remaining = {
        cluster.raw_label: cluster.support
        for cluster in partition.effective_clusters
    }
    effective_steps: list[tuple[int, int]] = []
    round_index = 0
    while len(effective_steps) < effective_slots:
        progressed = False
        for raw_label in cluster_order:
            if remaining[raw_label] <= 0:
                continue
            effective_steps.append((raw_label, round_index))
            remaining[raw_label] -= 1
            progressed = True
            if len(effective_steps) == effective_slots:
                break
        if not progressed:
            raise ValueError("effective clusters cannot fill budget under residual cap")
        round_index += 1

    first_round_count = len(cluster_order)
    scheduled: list[tuple[str, int | None, int | None]] = [
        ("effective_cluster", raw_label, effective_round)
        for raw_label, effective_round in effective_steps[:first_round_count]
    ]
    scheduled.extend(("residual", None, None) for _ in range(residual_quota))
    scheduled.extend(
        ("effective_cluster", raw_label, effective_round)
        for raw_label, effective_round in effective_steps[first_round_count:]
    )
    steps = tuple(
        ClusterAllocationStep(
            query_index=query_index,
            allocation_kind=kind,
            raw_label=raw_label,
            effective_round_index=effective_round,
        )
        for query_index, (kind, raw_label, effective_round) in enumerate(scheduled)
    )
    quotas = Counter(
        step.raw_label
        for step in steps
        if step.allocation_kind == "effective_cluster"
    )
    return MatchedClusterAllocation(
        dataset_id=partition.dataset_id,
        clusterer_id=partition.clusterer_id,
        active_learning_seed=partition.active_learning_seed,
        partition_sha256=partition.partition_sha256,
        budget=budget,
        quota_policy=QUOTA_POLICY,
        cluster_visit_order=cluster_order,
        effective_cluster_quotas=tuple(
            (raw_label, int(quotas[raw_label]))
            for raw_label in sorted(cluster_order)
        ),
        residual_quota=residual_quota,
        steps=steps,
    )


def _validate_kmeans_inputs(
    *,
    partition: EffectiveClusterPartition,
    result: KMeansClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> None:
    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("K-means queues require an effective-cluster partition")
    if type(result) is not KMeansClusteringResult:
        raise ValueError("K-means queues require native K-means geometry")
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("K-means queues require a formal representation matrix")
    if len(partition.effective_clusters) < 2:
        raise ValueError("K-means queues require at least two effective clusters")
    if (
        partition.clusterer_id != "kmeans"
        or result.clusterer_id != "kmeans"
        or partition.dataset_id != result.dataset_id
        or partition.active_learning_seed != result.active_learning_seed
    ):
        raise ValueError("K-means queue geometry scope drifted")
    if (
        candidate.matrix_sha256 != result.representation_matrix_sha256
        or candidate.representation_id != result.representation_id
    ):
        raise ValueError("K-means representation matrix does not match geometry")
    if (
        partition.source_geometry_sha256 != result.geometry_sha256
        or partition.case_ids != result.case_ids
        or candidate.case_ids != result.case_ids
        or partition.raw_labels != result.labels
    ):
        raise ValueError("K-means partition does not match native geometry")
    if (
        candidate.matrix.ndim != 2
        or result.cluster_centers.ndim != 2
        or candidate.matrix.shape[0] != len(result.case_ids)
        or candidate.matrix.shape[1] != result.cluster_centers.shape[1]
        or result.cluster_centers.shape[0] != result.cluster_count
    ):
        raise ValueError("K-means centroid geometry has incompatible dimensions")


def _ranked_kmeans_queue(
    *,
    selector_id: str,
    raw_label: int,
    indices: tuple[int, ...],
    result: KMeansClusteringResult,
    distances: np.ndarray,
) -> tuple[RankedClusterCase, ...]:
    rows: list[tuple[float, str, tuple[tuple[str, float], ...]]] = []
    for index in indices:
        all_distances = distances[index]
        nearest = np.sort(all_distances)[:2]
        assigned_distance = float(all_distances[raw_label])
        if selector_id == "center":
            score = assigned_distance
            components = (("assigned_centroid_distance", assigned_distance),)
        else:
            nearest_distance = float(nearest[0])
            second_distance = float(nearest[1])
            denominator = max(second_distance, np.finfo(float).eps)
            score = (second_distance - nearest_distance) / denominator
            components = (
                ("nearest_centroid_distance", nearest_distance),
                ("second_nearest_centroid_distance", second_distance),
                ("normalized_gap", score),
            )
        case_id = result.case_ids[index]
        tie = _semantic_sha256(
            {
                "role": "kmeans-selector-tie",
                "selector_id": selector_id,
                "active_learning_seed": result.active_learning_seed,
                "geometry_sha256": result.geometry_sha256,
                "case_id": case_id,
            }
        )
        rows.append((score, tie, components))
    order = sorted(range(len(indices)), key=lambda offset: (rows[offset][0], rows[offset][1]))
    definition = (
        "assigned_centroid_distance"
        if selector_id == "center"
        else (
            "(second_nearest_centroid_distance-nearest_centroid_distance)"
            "/second_nearest_centroid_distance"
        )
    )
    reason = (
        "kmeans_nearest_to_centroid"
        if selector_id == "center"
        else "kmeans_smallest_normalized_centroid_gap"
    )
    return tuple(
        RankedClusterCase(
            case_id=result.case_ids[indices[offset]],
            raw_label=raw_label,
            selector_id=selector_id,
            rank_within_cluster=rank,
            selector_score=float(rows[offset][0]),
            score_definition=definition,
            score_components=rows[offset][2],
            reason=reason,
        )
        for rank, offset in enumerate(order, start=1)
    )


def build_kmeans_selector_queues(
    *,
    partition: EffectiveClusterPartition,
    result: KMeansClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> dict[str, ClusterSelectorQueues]:
    """Build center and boundary queues from native K-means centroids."""

    _validate_kmeans_inputs(
        partition=partition,
        result=result,
        candidate=candidate,
    )
    distances = np.linalg.norm(
        candidate.matrix[:, np.newaxis, :] - result.cluster_centers[np.newaxis, :, :],
        axis=2,
    )
    result_by_selector: dict[str, ClusterSelectorQueues] = {}
    for selector_id in ("center", "boundary"):
        queues = []
        for cluster in partition.effective_clusters:
            indices = tuple(
                index
                for index, label in enumerate(result.labels)
                if label == cluster.raw_label
            )
            queues.append(
                (
                    cluster.raw_label,
                    _ranked_kmeans_queue(
                        selector_id=selector_id,
                        raw_label=cluster.raw_label,
                        indices=indices,
                        result=result,
                        distances=distances,
                    ),
                )
            )
        result_by_selector[selector_id] = ClusterSelectorQueues(
            clusterer_id="kmeans",
            selector_id=selector_id,
            active_learning_seed=result.active_learning_seed,
            geometry_sha256=result.geometry_sha256,
            representation_matrix_sha256=result.representation_matrix_sha256,
            cluster_queues=tuple(queues),
        )
    return result_by_selector


def _validate_dbscan_inputs(
    *,
    partition: EffectiveClusterPartition,
    result: DBSCANClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> None:
    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("DBSCAN queues require an effective-cluster partition")
    if type(result) is not DBSCANClusteringResult:
        raise ValueError("DBSCAN queues require native DBSCAN geometry")
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("DBSCAN queues require a formal representation matrix")
    core = set(result.core_case_ids)
    border = set(result.border_case_ids)
    noise = set(result.noise_case_ids)
    if (
        core.intersection(border)
        or core.intersection(noise)
        or border.intersection(noise)
        or core.union(border, noise) != set(result.case_ids)
    ):
        raise ValueError("DBSCAN core/border/noise metadata is inconsistent")
    if any(
        (case_id in noise) != (label < 0)
        for case_id, label in zip(result.case_ids, result.labels)
    ):
        raise ValueError("DBSCAN core/border/noise labels are inconsistent")
    if len(partition.effective_clusters) < 2:
        raise ValueError("DBSCAN queues require at least two effective clusters")
    if (
        partition.clusterer_id != "dbscan"
        or result.clusterer_id != "dbscan"
        or partition.dataset_id != result.dataset_id
        or partition.active_learning_seed != result.active_learning_seed
    ):
        raise ValueError("DBSCAN queue geometry scope drifted")
    if (
        candidate.matrix_sha256 != result.representation_matrix_sha256
        or candidate.representation_id != result.representation_id
    ):
        raise ValueError("DBSCAN representation matrix does not match geometry")
    if (
        partition.source_geometry_sha256 != result.geometry_sha256
        or partition.case_ids != result.case_ids
        or candidate.case_ids != result.case_ids
        or partition.raw_labels != result.labels
    ):
        raise ValueError("DBSCAN partition does not match native geometry")
    if candidate.matrix.ndim != 2 or candidate.matrix.shape[0] != len(
        result.case_ids
    ):
        raise ValueError("DBSCAN representation matrix dimensions are invalid")


def build_dbscan_selector_queues(
    *,
    partition: EffectiveClusterPartition,
    result: DBSCANClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> dict[str, ClusterSelectorQueues]:
    """Build arithmetic-center and native density-boundary DBSCAN queues."""

    _validate_dbscan_inputs(
        partition=partition,
        result=result,
        candidate=candidate,
    )
    distances = pairwise_distances(candidate.matrix, metric=result.metric)
    eps_tolerance = max(np.finfo(float).eps, abs(result.eps) * 1e-12)
    neighbor_counts = np.sum(distances <= result.eps + eps_tolerance, axis=1)
    index_by_case = {
        case_id: index for index, case_id in enumerate(result.case_ids)
    }
    core = set(result.core_case_ids)
    border = set(result.border_case_ids)
    selector_queues: dict[str, list[tuple[int, tuple[RankedClusterCase, ...]]]] = {
        "center": [],
        "boundary": [],
    }
    for cluster in partition.effective_clusters:
        indices = tuple(index_by_case[case_id] for case_id in cluster.case_ids)
        center = np.mean(candidate.matrix[np.asarray(indices)], axis=0, keepdims=True)
        center_distances = pairwise_distances(
            candidate.matrix[np.asarray(indices)],
            center,
            metric=result.metric,
        ).reshape(-1)
        rows = []
        for local_index, global_index in enumerate(indices):
            case_id = result.case_ids[global_index]
            is_border = case_id in border
            is_core = case_id in core
            density = int(neighbor_counts[global_index])
            center_distance = float(center_distances[local_index])
            tie = _semantic_sha256(
                {
                    "role": "dbscan-selector-tie",
                    "active_learning_seed": result.active_learning_seed,
                    "geometry_sha256": result.geometry_sha256,
                    "case_id": case_id,
                }
            )
            rows.append(
                {
                    "case_id": case_id,
                    "is_border": is_border,
                    "is_core": is_core,
                    "density": density,
                    "center_distance": center_distance,
                    "tie": tie,
                }
            )
        center_rows = sorted(
            rows,
            key=lambda row: (row["center_distance"], row["tie"]),
        )
        center_queue = tuple(
            RankedClusterCase(
                case_id=str(row["case_id"]),
                raw_label=cluster.raw_label,
                selector_id="center",
                rank_within_cluster=rank,
                selector_score=float(row["center_distance"]),
                score_definition="distance_to_arithmetic_cluster_center",
                score_components=(
                    ("arithmetic_center_distance", float(row["center_distance"])),
                ),
                reason="dbscan_nearest_to_arithmetic_center",
                is_core=bool(row["is_core"]),
                is_border=bool(row["is_border"]),
            )
            for rank, row in enumerate(center_rows, start=1)
        )
        boundary_rows = sorted(
            rows,
            key=lambda row: (
                0 if row["is_border"] else 1,
                row["density"],
                -float(row["center_distance"]),
                row["tie"],
            ),
        )
        boundary_queue = tuple(
            RankedClusterCase(
                case_id=str(row["case_id"]),
                raw_label=cluster.raw_label,
                selector_id="boundary",
                rank_within_cluster=rank,
                selector_score=(
                    (0.0 if row["is_border"] else 1.0)
                    + float(row["density"]) / (len(result.case_ids) + 1.0)
                ),
                score_definition="native_border_priority_then_eps_neighbor_count",
                score_components=(
                    ("native_border_priority", 0.0 if row["is_border"] else 1.0),
                    ("eps_neighbor_count", float(row["density"])),
                    ("arithmetic_center_distance", float(row["center_distance"])),
                ),
                reason=(
                    "dbscan_native_border"
                    if row["is_border"]
                    else "dbscan_low_density_core_supplement"
                ),
                is_core=bool(row["is_core"]),
                is_border=bool(row["is_border"]),
            )
            for rank, row in enumerate(boundary_rows, start=1)
        )
        selector_queues["center"].append((cluster.raw_label, center_queue))
        selector_queues["boundary"].append((cluster.raw_label, boundary_queue))
    return {
        selector_id: ClusterSelectorQueues(
            clusterer_id="dbscan",
            selector_id=selector_id,
            active_learning_seed=result.active_learning_seed,
            geometry_sha256=result.geometry_sha256,
            representation_matrix_sha256=result.representation_matrix_sha256,
            cluster_queues=tuple(queues),
        )
        for selector_id, queues in selector_queues.items()
    }


def _validate_hdbscan_inputs(
    *,
    partition: EffectiveClusterPartition,
    result: HDBSCANClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> None:
    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("HDBSCAN queues require an effective-cluster partition")
    if type(result) is not HDBSCANClusteringResult:
        raise ValueError("HDBSCAN queues require native HDBSCAN geometry")
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("HDBSCAN queues require a formal representation matrix")
    memberships = np.asarray(result.membership_strengths, dtype=float)
    if (
        memberships.shape != (len(result.case_ids),)
        or not np.isfinite(memberships).all()
        or np.any(memberships < 0.0)
        or np.any(memberships > 1.0)
    ):
        raise ValueError("HDBSCAN membership evidence is invalid")
    if len(partition.effective_clusters) < 2:
        raise ValueError("HDBSCAN queues require at least two effective clusters")
    if (
        partition.clusterer_id != "hdbscan"
        or result.clusterer_id != "hdbscan"
        or partition.dataset_id != result.dataset_id
        or partition.active_learning_seed != result.active_learning_seed
    ):
        raise ValueError("HDBSCAN queue geometry scope drifted")
    if (
        candidate.matrix_sha256 != result.representation_matrix_sha256
        or candidate.representation_id != result.representation_id
    ):
        raise ValueError("HDBSCAN representation matrix does not match geometry")
    if (
        partition.source_geometry_sha256 != result.geometry_sha256
        or partition.case_ids != result.case_ids
        or candidate.case_ids != result.case_ids
        or partition.raw_labels != result.labels
    ):
        raise ValueError("HDBSCAN partition does not match native geometry")
    cluster_labels = tuple(sorted({label for label in result.labels if label >= 0}))
    if (
        tuple(result.medoid_cluster_labels) != cluster_labels
        or result.medoids.shape != (len(cluster_labels), candidate.matrix.shape[1])
        or not np.isfinite(result.medoids).all()
    ):
        raise ValueError("HDBSCAN medoid evidence is invalid")
    noise = set(result.noise_case_ids)
    if any(
        (case_id in noise) != (label < 0)
        for case_id, label in zip(result.case_ids, result.labels)
    ):
        raise ValueError("HDBSCAN noise evidence is inconsistent")


def build_hdbscan_selector_queues(
    *,
    partition: EffectiveClusterPartition,
    result: HDBSCANClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> dict[str, ClusterSelectorQueues]:
    """Build medoid/membership center and low-membership boundary queues."""

    _validate_hdbscan_inputs(
        partition=partition,
        result=result,
        candidate=candidate,
    )
    index_by_case = {
        case_id: index for index, case_id in enumerate(result.case_ids)
    }
    medoid_by_label = {
        label: result.medoids[index]
        for index, label in enumerate(result.medoid_cluster_labels)
    }
    selector_queues: dict[str, list[tuple[int, tuple[RankedClusterCase, ...]]]] = {
        "center": [],
        "boundary": [],
    }
    for cluster in partition.effective_clusters:
        rows = []
        medoid = medoid_by_label[cluster.raw_label].reshape(1, -1)
        indices = tuple(index_by_case[case_id] for case_id in cluster.case_ids)
        medoid_distances = pairwise_distances(
            candidate.matrix[np.asarray(indices)],
            medoid,
            metric="euclidean",
        ).reshape(-1)
        for local_index, global_index in enumerate(indices):
            membership = float(result.membership_strengths[global_index])
            outlier_score = 1.0 - membership
            case_id = result.case_ids[global_index]
            rows.append(
                {
                    "case_id": case_id,
                    "membership": membership,
                    "outlier_score": outlier_score,
                    "medoid_distance": float(medoid_distances[local_index]),
                    "tie": _semantic_sha256(
                        {
                            "role": "hdbscan-selector-tie",
                            "active_learning_seed": result.active_learning_seed,
                            "geometry_sha256": result.geometry_sha256,
                            "case_id": case_id,
                        }
                    ),
                }
            )
        center_rows = sorted(
            rows,
            key=lambda row: (
                row["medoid_distance"],
                -float(row["membership"]),
                row["tie"],
            ),
        )
        center_queue = tuple(
            RankedClusterCase(
                case_id=str(row["case_id"]),
                raw_label=cluster.raw_label,
                selector_id="center",
                rank_within_cluster=rank,
                selector_score=float(row["medoid_distance"]),
                score_definition=(
                    "medoid_distance_then_descending_membership_strength"
                ),
                score_components=(
                    ("medoid_distance", float(row["medoid_distance"])),
                    ("membership_strength", float(row["membership"])),
                ),
                reason="hdbscan_medoid_high_membership",
            )
            for rank, row in enumerate(center_rows, start=1)
        )
        boundary_rows = sorted(
            rows,
            key=lambda row: (
                row["membership"],
                -float(row["outlier_score"]),
                -float(row["medoid_distance"]),
                row["tie"],
            ),
        )
        boundary_queue = tuple(
            RankedClusterCase(
                case_id=str(row["case_id"]),
                raw_label=cluster.raw_label,
                selector_id="boundary",
                rank_within_cluster=rank,
                selector_score=float(row["membership"]),
                score_definition=(
                    "ascending_membership_strength_then_descending_outlier_score"
                ),
                score_components=(
                    ("membership_strength", float(row["membership"])),
                    ("outlier_score", float(row["outlier_score"])),
                    ("medoid_distance", float(row["medoid_distance"])),
                ),
                reason="hdbscan_low_membership_high_outlier",
            )
            for rank, row in enumerate(boundary_rows, start=1)
        )
        selector_queues["center"].append((cluster.raw_label, center_queue))
        selector_queues["boundary"].append((cluster.raw_label, boundary_queue))
    return {
        selector_id: ClusterSelectorQueues(
            clusterer_id="hdbscan",
            selector_id=selector_id,
            active_learning_seed=result.active_learning_seed,
            geometry_sha256=result.geometry_sha256,
            representation_matrix_sha256=result.representation_matrix_sha256,
            cluster_queues=tuple(queues),
        )
        for selector_id, queues in selector_queues.items()
    }


def _validate_mutual_knn_inputs(
    *,
    partition: EffectiveClusterPartition,
    result: MutualKNNClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> dict[str, set[str]]:
    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("mutual-kNN queues require an effective-cluster partition")
    if type(result) is not MutualKNNClusteringResult:
        raise ValueError("mutual-kNN queues require native graph geometry")
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("mutual-kNN queues require a formal representation matrix")
    if len(partition.effective_clusters) < 2:
        raise ValueError("mutual-kNN queues require at least two effective clusters")
    if (
        partition.clusterer_id != "mutual_knn"
        or result.clusterer_id != "mutual_knn"
        or partition.dataset_id != result.dataset_id
        or partition.active_learning_seed != result.active_learning_seed
    ):
        raise ValueError("mutual-kNN queue geometry scope drifted")
    if (
        candidate.matrix_sha256 != result.representation_matrix_sha256
        or candidate.representation_id != result.representation_id
    ):
        raise ValueError("mutual-kNN representation matrix does not match geometry")
    if (
        partition.case_ids != result.case_ids
        or candidate.case_ids != result.case_ids
        or partition.raw_labels != result.labels
    ):
        raise ValueError("mutual-kNN partition does not match native geometry")
    if (
        len(result.neighbor_case_ids) != len(result.case_ids)
        or len(result.within_component_degrees) != len(result.case_ids)
    ):
        raise ValueError("mutual-kNN neighbor or degree evidence length drifted")
    label_by_case = dict(zip(result.case_ids, result.labels))
    adjacency = {case_id: set() for case_id in result.case_ids}
    for left, right in result.mutual_edges:
        if (
            left not in adjacency
            or right not in adjacency
            or left == right
            or label_by_case[left] != label_by_case[right]
        ):
            raise ValueError("mutual-kNN edge evidence is inconsistent")
        adjacency[left].add(right)
        adjacency[right].add(left)
    observed_degrees = tuple(len(adjacency[case_id]) for case_id in result.case_ids)
    if observed_degrees != result.within_component_degrees:
        raise ValueError("mutual-kNN within-component degree evidence drifted")
    if partition.source_geometry_sha256 != result.geometry_sha256:
        raise ValueError("mutual-kNN partition geometry hash drifted")
    return adjacency


def _shortest_path_lengths(
    source: str,
    *,
    members: frozenset[str],
    adjacency: dict[str, set[str]],
) -> dict[str, int]:
    distances = {source: 0}
    pending = deque((source,))
    while pending:
        current = pending.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor in members and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                pending.append(neighbor)
    if set(distances) != set(members):
        raise ValueError("mutual-kNN effective component is disconnected")
    return distances


def build_mutual_knn_selector_queues(
    *,
    partition: EffectiveClusterPartition,
    result: MutualKNNClusteringResult,
    candidate: RepresentationMatrixCandidate,
) -> dict[str, ClusterSelectorQueues]:
    """Build graph-medoid center and graph-periphery boundary queues."""

    adjacency = _validate_mutual_knn_inputs(
        partition=partition,
        result=result,
        candidate=candidate,
    )
    selector_queues: dict[str, list[tuple[int, tuple[RankedClusterCase, ...]]]] = {
        "center": [],
        "boundary": [],
    }
    for cluster in partition.effective_clusters:
        members = frozenset(cluster.case_ids)
        rows = []
        for case_id in cluster.case_ids:
            distances = _shortest_path_lengths(
                case_id,
                members=members,
                adjacency=adjacency,
            )
            distance_sum = sum(distances.values())
            mean_distance = distance_sum / (len(members) - 1)
            closeness = (len(members) - 1) / distance_sum
            eccentricity = max(distances.values())
            degree = len(adjacency[case_id])
            rows.append(
                {
                    "case_id": case_id,
                    "mean_distance": float(mean_distance),
                    "closeness": float(closeness),
                    "eccentricity": int(eccentricity),
                    "degree": int(degree),
                    "tie": _semantic_sha256(
                        {
                            "role": "mutual-knn-selector-tie",
                            "active_learning_seed": result.active_learning_seed,
                            "geometry_sha256": result.geometry_sha256,
                            "case_id": case_id,
                        }
                    ),
                }
            )
        center_rows = sorted(
            rows,
            key=lambda row: (
                row["mean_distance"],
                -float(row["closeness"]),
                -int(row["degree"]),
                row["tie"],
            ),
        )
        center_queue = tuple(
            RankedClusterCase(
                case_id=str(row["case_id"]),
                raw_label=cluster.raw_label,
                selector_id="center",
                rank_within_cluster=rank,
                selector_score=float(row["mean_distance"]),
                score_definition=(
                    "ascending_mean_shortest_path_distance_graph_medoid"
                ),
                score_components=(
                    ("mean_shortest_path_distance", float(row["mean_distance"])),
                    ("closeness", float(row["closeness"])),
                    ("eccentricity", float(row["eccentricity"])),
                    ("within_cluster_degree", float(row["degree"])),
                ),
                reason="mutual_knn_graph_medoid_high_closeness",
            )
            for rank, row in enumerate(center_rows, start=1)
        )
        max_eccentricity = max(int(row["eccentricity"]) for row in rows)
        max_degree = max(int(row["degree"]) for row in rows)
        boundary_rows = sorted(
            rows,
            key=lambda row: (
                -int(row["eccentricity"]),
                int(row["degree"]),
                float(row["closeness"]),
                row["tie"],
            ),
        )
        boundary_queue = tuple(
            RankedClusterCase(
                case_id=str(row["case_id"]),
                raw_label=cluster.raw_label,
                selector_id="boundary",
                rank_within_cluster=rank,
                selector_score=float(
                    (max_eccentricity - int(row["eccentricity"]))
                    * (max_degree + 1)
                    + int(row["degree"])
                ),
                score_definition=(
                    "descending_eccentricity_then_ascending_within_cluster_degree"
                ),
                score_components=(
                    ("eccentricity", float(row["eccentricity"])),
                    ("within_cluster_degree", float(row["degree"])),
                    ("closeness", float(row["closeness"])),
                ),
                reason="mutual_knn_high_eccentricity_low_degree",
            )
            for rank, row in enumerate(boundary_rows, start=1)
        )
        selector_queues["center"].append((cluster.raw_label, center_queue))
        selector_queues["boundary"].append((cluster.raw_label, boundary_queue))
    return {
        selector_id: ClusterSelectorQueues(
            clusterer_id="mutual_knn",
            selector_id=selector_id,
            active_learning_seed=result.active_learning_seed,
            geometry_sha256=result.geometry_sha256,
            representation_matrix_sha256=result.representation_matrix_sha256,
            cluster_queues=tuple(queues),
        )
        for selector_id, queues in selector_queues.items()
    }


def _require_sha256(value: str, name: str) -> None:
    if (
        len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def build_within_cluster_random_queues(
    *,
    partition: EffectiveClusterPartition,
    representation_matrix_sha256: str,
) -> ClusterSelectorQueues:
    """Build a seeded permutation inside each effective cluster only."""

    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("within-cluster random requires an effective partition")
    _require_sha256(
        representation_matrix_sha256,
        "representation matrix SHA-256",
    )
    if len(partition.effective_clusters) < 2:
        raise ValueError(
            "within-cluster random requires at least two effective clusters"
        )
    queues = []
    for cluster in partition.effective_clusters:
        keyed = []
        for case_id in cluster.case_ids:
            key = _semantic_sha256(
                {
                    "role": "within-cluster-seeded-random",
                    "active_learning_seed": partition.active_learning_seed,
                    "geometry_sha256": partition.source_geometry_sha256,
                    "raw_label": cluster.raw_label,
                    "case_id": case_id,
                }
            )
            keyed.append((key, case_id))
        keyed.sort()
        queue = tuple(
            RankedClusterCase(
                case_id=case_id,
                raw_label=cluster.raw_label,
                selector_id="within_cluster_random",
                rank_within_cluster=rank,
                selector_score=int(key[:16], 16) / float(0xFFFFFFFFFFFFFFFF),
                score_definition="seeded_sha256_permutation_key",
                score_components=(
                    (
                        "seeded_key_prefix_u64",
                        float(int(key[:16], 16)),
                    ),
                ),
                reason="within_cluster_seeded_random",
            )
            for rank, (key, case_id) in enumerate(keyed, start=1)
        )
        queues.append((cluster.raw_label, queue))
    return ClusterSelectorQueues(
        clusterer_id=partition.clusterer_id,
        selector_id="within_cluster_random",
        active_learning_seed=partition.active_learning_seed,
        geometry_sha256=partition.source_geometry_sha256,
        representation_matrix_sha256=representation_matrix_sha256,
        cluster_queues=tuple(queues),
    )


def assert_matched_selector_contracts(
    *,
    partition: EffectiveClusterPartition,
    allocation: MatchedClusterAllocation,
    selector_queues: Mapping[str, ClusterSelectorQueues],
) -> dict[str, MatchedSelectorContract]:
    """Prove selectors share geometry, population, visit order, and quotas."""

    expected_selectors = {"center", "boundary", "within_cluster_random"}
    if set(selector_queues) != expected_selectors:
        raise ValueError("matched selector bundle must contain exactly three selectors")
    if (
        allocation.dataset_id != partition.dataset_id
        or allocation.clusterer_id != partition.clusterer_id
        or allocation.active_learning_seed != partition.active_learning_seed
        or allocation.partition_sha256 != partition.partition_sha256
    ):
        raise ValueError("matched selector allocation does not match partition")
    expected_labels = tuple(
        cluster.raw_label for cluster in partition.effective_clusters
    )
    expected_case_sets = {
        cluster.raw_label: frozenset(cluster.case_ids)
        for cluster in partition.effective_clusters
    }
    representation_hashes = {
        queues.representation_matrix_sha256
        for queues in selector_queues.values()
    }
    if len(representation_hashes) != 1:
        raise ValueError("matched selectors use different representation matrices")
    case_sets_sha256 = _semantic_sha256(
        {
            str(label): sorted(case_ids)
            for label, case_ids in expected_case_sets.items()
        }
    )
    contracts: dict[str, MatchedSelectorContract] = {}
    for selector_id in ("center", "boundary", "within_cluster_random"):
        queues = selector_queues[selector_id]
        if queues.selector_id != selector_id:
            raise ValueError("matched selector queue identity drifted")
        if (
            queues.clusterer_id != partition.clusterer_id
            or queues.active_learning_seed != partition.active_learning_seed
            or queues.geometry_sha256 != partition.source_geometry_sha256
        ):
            raise ValueError("matched selector geometry drifted")
        labels = tuple(label for label, _ in queues.cluster_queues)
        if labels != expected_labels:
            raise ValueError("matched selector cluster labels drifted")
        for label, queue in queues.cluster_queues:
            if (
                frozenset(item.case_id for item in queue) != expected_case_sets[label]
                or len(queue) != len(expected_case_sets[label])
                or tuple(item.rank_within_cluster for item in queue)
                != tuple(range(1, len(queue) + 1))
                or any(
                    item.raw_label != label or item.selector_id != selector_id
                    for item in queue
                )
            ):
                raise ValueError("matched selector case set or rank drifted")
        contracts[selector_id] = MatchedSelectorContract(
            selector_id=selector_id,
            dataset_id=partition.dataset_id,
            clusterer_id=partition.clusterer_id,
            active_learning_seed=partition.active_learning_seed,
            geometry_sha256=partition.source_geometry_sha256,
            partition_sha256=partition.partition_sha256,
            representation_matrix_sha256=queues.representation_matrix_sha256,
            quota_sha256=allocation.quota_sha256,
            allocation_sha256=allocation.allocation_sha256,
            queues_sha256=queues.queues_sha256,
            case_sets_sha256=case_sets_sha256,
        )
    return contracts


def _shared_residual_rows(
    *,
    partition: EffectiveClusterPartition,
    residual_statuses: Mapping[str, QueryCaseStatus] | None,
) -> tuple[tuple[str, int, int, float, QueryCaseStatus], ...]:
    residual_ids = set(partition.residual_case_ids)
    if residual_statuses is not None and set(residual_statuses) != residual_ids:
        raise ValueError("residual status case set does not match partition")
    if partition.clusterer_id == "dbscan" and residual_ids and residual_statuses is None:
        raise ValueError("DBSCAN residual cases require native core/border/noise status")

    raw_label_by_case = dict(zip(partition.case_ids, partition.raw_labels))
    noise_ids = set(partition.noise_case_ids)
    keyed = []
    for case_id in partition.residual_case_ids:
        digest = _semantic_sha256(
            {
                "role": "matched-shared-residual-order",
                "partition_sha256": partition.partition_sha256,
                "active_learning_seed": partition.active_learning_seed,
                "case_id": case_id,
            }
        )
        score = int(digest[:13], 16) / float((16**13) - 1)
        if residual_statuses is None:
            status = QueryCaseStatus(
                is_core=None,
                is_border=None,
                is_noise=case_id in noise_ids,
            )
        else:
            status = residual_statuses[case_id]
            if type(status) is not QueryCaseStatus:
                raise ValueError("residual status values must be QueryCaseStatus")
        if status.is_noise != (case_id in noise_ids):
            raise ValueError("residual noise status does not match partition")
        keyed.append(
            (
                digest,
                case_id,
                int(raw_label_by_case[case_id]),
                score,
                status,
            )
        )
    keyed.sort(key=lambda row: row[0])
    return tuple(
        (case_id, raw_label, rank, score, status)
        for rank, (_, case_id, raw_label, score, status) in enumerate(
            keyed,
            start=1,
        )
    )


def build_matched_cluster_query_plans(
    *,
    partition: EffectiveClusterPartition,
    allocation: MatchedClusterAllocation,
    selector_queues: Mapping[str, ClusterSelectorQueues],
    residual_statuses: Mapping[str, QueryCaseStatus] | None = None,
) -> dict[str, ClusterQueryPlan]:
    """Materialize matched, immutable budget-30 cluster query sequences."""

    contracts = assert_matched_selector_contracts(
        partition=partition,
        allocation=allocation,
        selector_queues=selector_queues,
    )
    residual_rows = _shared_residual_rows(
        partition=partition,
        residual_statuses=residual_statuses,
    )
    if allocation.residual_quota > len(residual_rows):
        raise ValueError("residual queue cannot fill the matched allocation")

    plans: dict[str, ClusterQueryPlan] = {}
    for selector_id in ("center", "boundary", "within_cluster_random"):
        queues = selector_queues[selector_id]
        contract = contracts[selector_id]
        input_sha256 = _semantic_sha256(
            {
                "role": "matched-cluster-query-plan-input",
                "contract": contract.to_dict(),
                "allocation_sha256": allocation.allocation_sha256,
                "residual_rows": [
                    {
                        "case_id": case_id,
                        "raw_label": raw_label,
                        "rank_within_cluster": rank,
                        "selector_score": score,
                        **status.to_dict(),
                    }
                    for case_id, raw_label, rank, score, status in residual_rows
                ],
            }
        )
        cluster_offsets = {
            raw_label: 0 for raw_label in allocation.cluster_visit_order
        }
        residual_offset = 0
        records = []
        for step in allocation.steps:
            if step.allocation_kind == "effective_cluster":
                assert step.raw_label is not None
                offset = cluster_offsets[step.raw_label]
                queue = queues.queue_for(step.raw_label)
                if offset >= len(queue):
                    raise ValueError(
                        "selector cluster queue exhausted before matched allocation"
                    )
                source = queue[offset]
                cluster_offsets[step.raw_label] = offset + 1
                records.append(
                    ClusterQueryRecord(
                        query_index=step.query_index,
                        case_id=source.case_id,
                        raw_label=source.raw_label,
                        selector_id=selector_id,
                        rank_within_cluster=source.rank_within_cluster,
                        selector_score=source.selector_score,
                        score_definition=source.score_definition,
                        score_components=source.score_components,
                        reason=source.reason,
                        is_core=source.is_core,
                        is_border=source.is_border,
                        is_noise=source.is_noise,
                        is_residual=False,
                        active_learning_seed=partition.active_learning_seed,
                        input_sha256=input_sha256,
                    )
                )
                continue

            case_id, raw_label, rank, score, status = residual_rows[
                residual_offset
            ]
            residual_offset += 1
            records.append(
                ClusterQueryRecord(
                    query_index=step.query_index,
                    case_id=case_id,
                    raw_label=raw_label,
                    selector_id=selector_id,
                    rank_within_cluster=rank,
                    selector_score=score,
                    score_definition="shared_residual_sha256_order_score",
                    score_components=(("sha256_order_score", score),),
                    reason="matched_selector_independent_residual_order",
                    is_core=status.is_core,
                    is_border=status.is_border,
                    is_noise=status.is_noise,
                    is_residual=True,
                    active_learning_seed=partition.active_learning_seed,
                    input_sha256=input_sha256,
                )
            )

        plans[selector_id] = ClusterQueryPlan(
            dataset_id=partition.dataset_id,
            clusterer_id=partition.clusterer_id,
            selector_id=selector_id,
            active_learning_seed=partition.active_learning_seed,
            budget=allocation.budget,
            representation_matrix_sha256=(
                contract.representation_matrix_sha256
            ),
            geometry_sha256=contract.geometry_sha256,
            partition_sha256=contract.partition_sha256,
            quota_sha256=contract.quota_sha256,
            allocation_sha256=contract.allocation_sha256,
            queues_sha256=contract.queues_sha256,
            case_sets_sha256=contract.case_sets_sha256,
            input_sha256=input_sha256,
            records=tuple(records),
        )
    return plans


__all__ = [
    "ClusterQueryPlan",
    "ClusterQueryRecord",
    "ClusterSelectorQueues",
    "ClusterAllocationStep",
    "MAX_RESIDUAL_QUERIES",
    "MatchedClusterAllocation",
    "MatchedSelectorContract",
    "QUOTA_POLICY",
    "QueryCaseStatus",
    "RankedClusterCase",
    "build_dbscan_selector_queues",
    "build_equal_weight_round_robin_allocation",
    "build_hdbscan_selector_queues",
    "build_kmeans_selector_queues",
    "build_matched_cluster_query_plans",
    "build_mutual_knn_selector_queues",
    "build_within_cluster_random_queues",
    "assert_matched_selector_contracts",
]
