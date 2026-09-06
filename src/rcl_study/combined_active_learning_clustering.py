"""Native label-free clustering for combined active learning 2.0.

Every clusterer in this module consumes a frozen representation matrix
directly.  It does not replay or import historical authority query plans.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from threadpoolctl import threadpool_limits

from rcl_study.combined_active_learning_representation import (
    FORMAL_REPRESENTATION_IDS,
    RepresentationMatrixCandidate,
)
from rcl_study.combined_active_learning_schemas import (
    ACTIVE_LEARNING_SEEDS,
    DatasetId,
    GroundTruthClusterCap,
    SeedStructureObservation,
    StructureTripletAssessment,
    StructureGateResult,
    amended_structure_policy,
    assess_structure_triplet,
    default_protocol_manifest,
)


KMEANS_N_INIT = 20
KMEANS_ALGORITHM = "lloyd"
KMEANS_CLUSTER_COUNTS = {
    "rcabench": (13, 14, 15, 16, 17),
    "aiops2022_pre": (8, 9, 10, 11, 12),
}
DBSCAN_MIN_SAMPLES = (3, 4, 5, 6, 8, 10, 12, 16)
DBSCAN_EPS_QUANTILES = (0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05)
DBSCAN_METRICS = ("euclidean", "cosine")
HDBSCAN_MIN_CLUSTER_SIZES = (5, 8, 10, 12, 16, 20, 24)
HDBSCAN_MIN_SAMPLES = (3, 5, 8, 10, 12)
HDBSCAN_CLUSTER_SELECTION_METHODS = ("eom", "leaf")
HDBSCAN_METRIC = "euclidean"
HDBSCAN_ALLOW_SINGLE_CLUSTER = False
HDBSCAN_STORE_CENTERS = "medoid"
MUTUAL_KNN_NEIGHBOR_COUNTS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
MUTUAL_KNN_METRICS = ("euclidean", "cosine")
EFFECTIVE_CLUSTER_MIN_SUPPORT = 5
FORMAL_CLUSTERER_IDS = frozenset(("kmeans", "dbscan", "hdbscan", "mutual_knn"))
FORMAL_CLUSTERER_ORDER = ("kmeans", "dbscan", "hdbscan", "mutual_knn")


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class KMeansConfiguration:
    dataset_id: str
    cluster_count: int
    n_init: int = KMEANS_N_INIT
    algorithm: str = KMEANS_ALGORITHM

    def __post_init__(self) -> None:
        if self.dataset_id not in KMEANS_CLUSTER_COUNTS:
            raise ValueError("K-means dataset is outside the formal protocol")
        if self.cluster_count not in KMEANS_CLUSTER_COUNTS[self.dataset_id]:
            raise ValueError("K-means cluster count is outside the dataset band")
        if self.n_init != KMEANS_N_INIT:
            raise ValueError("combined 2.0 K-means requires n_init=20")
        if self.algorithm != KMEANS_ALGORITHM:
            raise ValueError("combined 2.0 K-means requires the lloyd algorithm")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "cluster_count": self.cluster_count,
            "n_init": self.n_init,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True)
class DBSCANConfiguration:
    min_samples: int
    eps_quantile: float
    metric: str

    def __post_init__(self) -> None:
        if self.min_samples not in DBSCAN_MIN_SAMPLES:
            raise ValueError("DBSCAN min_samples is outside the frozen grid")
        if self.eps_quantile not in DBSCAN_EPS_QUANTILES:
            raise ValueError("DBSCAN eps quantile is outside the frozen grid")
        if self.metric not in DBSCAN_METRICS:
            raise ValueError("DBSCAN metric is outside the frozen grid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "eps_quantile": self.eps_quantile,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class DBSCANEpsDerivation:
    min_samples: int
    eps_quantile: float
    metric: str
    eps: float
    neighbor_rank: int
    includes_self: bool
    k_distances: tuple[float, ...]
    representation_id: str
    representation_matrix_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "candidate_pool_k_distance_quantile",
            "min_samples": self.min_samples,
            "eps_quantile": self.eps_quantile,
            "metric": self.metric,
            "eps": self.eps,
            "neighbor_rank": self.neighbor_rank,
            "includes_self": self.includes_self,
            "k_distances": list(self.k_distances),
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
        }

    @property
    def derivation_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class DBSCANClusteringResult:
    dataset_id: str
    active_learning_seed: int
    min_samples: int
    eps_quantile: float
    metric: str
    eps: float
    representation_id: str
    representation_matrix_sha256: str
    eps_derivation_sha256: str
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    core_case_ids: tuple[str, ...]
    border_case_ids: tuple[str, ...]
    noise_case_ids: tuple[str, ...]
    non_noise_cluster_count: int
    non_noise_case_count: int
    noise_count: int
    non_noise_coverage: float
    largest_non_noise_cluster_share: float
    density_phase: str
    clusterer_id: str = "dbscan"
    backend: str = "sklearn.cluster.DBSCAN"
    input_contract: str = "shared_final_representation_pairwise_distance"
    seed_influences_geometry: bool = False

    def _geometry_dict(self) -> dict[str, Any]:
        return {
            "clusterer_id": self.clusterer_id,
            "backend": self.backend,
            "input_contract": self.input_contract,
            "seed_influences_geometry": self.seed_influences_geometry,
            "dataset_id": self.dataset_id,
            "parameters": {
                "min_samples": self.min_samples,
                "eps_quantile": self.eps_quantile,
                "metric": self.metric,
                "eps": self.eps,
            },
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "eps_derivation_sha256": self.eps_derivation_sha256,
            "case_ids": list(self.case_ids),
            "labels": list(self.labels),
            "core_case_ids": list(self.core_case_ids),
            "border_case_ids": list(self.border_case_ids),
            "noise_case_ids": list(self.noise_case_ids),
            "non_noise_cluster_count": self.non_noise_cluster_count,
            "non_noise_case_count": self.non_noise_case_count,
            "noise_count": self.noise_count,
            "non_noise_coverage": self.non_noise_coverage,
            "largest_non_noise_cluster_share": (
                self.largest_non_noise_cluster_share
            ),
            "density_phase": self.density_phase,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._geometry_dict()
        payload["active_learning_seed"] = self.active_learning_seed
        payload["geometry_sha256"] = self.geometry_sha256
        return payload

    @property
    def geometry_sha256(self) -> str:
        return _semantic_sha256(self._geometry_dict())


@dataclass(frozen=True)
class DBSCANSearchAttempt:
    configuration: DBSCANConfiguration
    trajectory_position: int
    status: str
    eps_derivation: DBSCANEpsDerivation | None
    result: DBSCANClusteringResult | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.trajectory_position not in range(1, len(DBSCAN_EPS_QUANTILES) + 1):
            raise ValueError("DBSCAN trajectory position is invalid")
        if self.status == "fitted":
            if (
                self.eps_derivation is None
                or self.result is None
                or self.rejection_reason is not None
            ):
                raise ValueError("fitted DBSCAN attempt has inconsistent evidence")
        elif self.status == "rejected":
            if self.result is not None or not self.rejection_reason:
                raise ValueError("rejected DBSCAN attempt has inconsistent evidence")
        else:
            raise ValueError("DBSCAN attempt status is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "trajectory_position": self.trajectory_position,
            "status": self.status,
            "eps_derivation": (
                self.eps_derivation.to_dict()
                if self.eps_derivation is not None
                else None
            ),
            "result": self.result.to_dict() if self.result is not None else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class DBSCANGridSearchResult:
    dataset_id: str
    active_learning_seed: int
    representation_id: str
    representation_matrix_sha256: str
    attempts: tuple[DBSCANSearchAttempt, ...]
    seed_influences_geometry: bool = False

    @property
    def attempted_configuration_count(self) -> int:
        return len(self.attempts)

    @property
    def fitted_configuration_count(self) -> int:
        return sum(attempt.status == "fitted" for attempt in self.attempts)

    @property
    def rejected_configuration_count(self) -> int:
        return sum(attempt.status == "rejected" for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "active_learning_seed": self.active_learning_seed,
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "seed_influences_geometry": self.seed_influences_geometry,
            "attempted_configuration_count": self.attempted_configuration_count,
            "fitted_configuration_count": self.fitted_configuration_count,
            "rejected_configuration_count": self.rejected_configuration_count,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @property
    def search_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class HDBSCANConfiguration:
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str
    allow_single_cluster: bool = HDBSCAN_ALLOW_SINGLE_CLUSTER
    metric: str = HDBSCAN_METRIC

    def __post_init__(self) -> None:
        if self.min_cluster_size not in HDBSCAN_MIN_CLUSTER_SIZES:
            raise ValueError("HDBSCAN min_cluster_size is outside the frozen grid")
        if self.min_samples not in HDBSCAN_MIN_SAMPLES:
            raise ValueError("HDBSCAN min_samples is outside the frozen grid")
        if self.cluster_selection_method not in HDBSCAN_CLUSTER_SELECTION_METHODS:
            raise ValueError("HDBSCAN selection method is outside the frozen grid")
        if self.allow_single_cluster is not HDBSCAN_ALLOW_SINGLE_CLUSTER:
            raise ValueError("combined 2.0 HDBSCAN forbids single-cluster output")
        if self.metric != HDBSCAN_METRIC:
            raise ValueError("combined 2.0 HDBSCAN requires Euclidean distance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class HDBSCANClusteringResult:
    dataset_id: str
    active_learning_seed: int
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str
    representation_id: str
    representation_matrix_sha256: str
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    membership_strengths: tuple[float, ...]
    medoid_cluster_labels: tuple[int, ...]
    medoids: np.ndarray
    noise_case_ids: tuple[str, ...]
    non_noise_cluster_count: int
    non_noise_case_count: int
    noise_count: int
    non_noise_coverage: float
    largest_non_noise_cluster_share: float
    clusterer_id: str = "hdbscan"
    backend: str = "sklearn.cluster.HDBSCAN"
    input_contract: str = "shared_final_representation_matrix"
    metric: str = HDBSCAN_METRIC
    allow_single_cluster: bool = HDBSCAN_ALLOW_SINGLE_CLUSTER
    store_centers: str = HDBSCAN_STORE_CENTERS
    seed_influences_geometry: bool = False

    def _geometry_dict(self) -> dict[str, Any]:
        return {
            "clusterer_id": self.clusterer_id,
            "backend": self.backend,
            "input_contract": self.input_contract,
            "metric": self.metric,
            "allow_single_cluster": self.allow_single_cluster,
            "store_centers": self.store_centers,
            "seed_influences_geometry": self.seed_influences_geometry,
            "dataset_id": self.dataset_id,
            "parameters": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "cluster_selection_method": self.cluster_selection_method,
            },
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "case_ids": list(self.case_ids),
            "labels": list(self.labels),
            "membership_strengths": list(self.membership_strengths),
            "medoid_cluster_labels": list(self.medoid_cluster_labels),
            "medoids": self.medoids.tolist(),
            "noise_case_ids": list(self.noise_case_ids),
            "non_noise_cluster_count": self.non_noise_cluster_count,
            "non_noise_case_count": self.non_noise_case_count,
            "noise_count": self.noise_count,
            "non_noise_coverage": self.non_noise_coverage,
            "largest_non_noise_cluster_share": (
                self.largest_non_noise_cluster_share
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._geometry_dict()
        payload["active_learning_seed"] = self.active_learning_seed
        payload["geometry_sha256"] = self.geometry_sha256
        return payload

    @property
    def geometry_sha256(self) -> str:
        return _semantic_sha256(self._geometry_dict())


@dataclass(frozen=True)
class HDBSCANSearchAttempt:
    configuration: HDBSCANConfiguration
    status: str
    result: HDBSCANClusteringResult | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.status == "fitted":
            if self.result is None or self.rejection_reason is not None:
                raise ValueError("fitted HDBSCAN attempt has inconsistent evidence")
        elif self.status == "rejected":
            if self.result is not None or not self.rejection_reason:
                raise ValueError("rejected HDBSCAN attempt has inconsistent evidence")
        else:
            raise ValueError("HDBSCAN attempt status is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "status": self.status,
            "result": self.result.to_dict() if self.result is not None else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class HDBSCANGridSearchResult:
    dataset_id: str
    active_learning_seed: int
    representation_id: str
    representation_matrix_sha256: str
    attempts: tuple[HDBSCANSearchAttempt, ...]
    seed_influences_geometry: bool = False

    @property
    def attempted_configuration_count(self) -> int:
        return len(self.attempts)

    @property
    def fitted_configuration_count(self) -> int:
        return sum(attempt.status == "fitted" for attempt in self.attempts)

    @property
    def rejected_configuration_count(self) -> int:
        return sum(attempt.status == "rejected" for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "active_learning_seed": self.active_learning_seed,
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "seed_influences_geometry": self.seed_influences_geometry,
            "attempted_configuration_count": self.attempted_configuration_count,
            "fitted_configuration_count": self.fitted_configuration_count,
            "rejected_configuration_count": self.rejected_configuration_count,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @property
    def search_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class MutualKNNConfiguration:
    neighbor_count: int
    metric: str

    def __post_init__(self) -> None:
        if self.neighbor_count not in MUTUAL_KNN_NEIGHBOR_COUNTS:
            raise ValueError("mutual-kNN neighbor count is outside the frozen grid")
        if self.metric not in MUTUAL_KNN_METRICS:
            raise ValueError("mutual-kNN metric is outside the frozen grid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "neighbor_count": self.neighbor_count,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class MutualKNNClusteringResult:
    dataset_id: str
    active_learning_seed: int
    neighbor_count: int
    metric: str
    representation_id: str
    representation_matrix_sha256: str
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    neighbor_case_ids: tuple[tuple[str, ...], ...]
    mutual_edges: tuple[tuple[str, str], ...]
    within_component_degrees: tuple[int, ...]
    component_count: int
    largest_component_share: float
    clusterer_id: str = "mutual_knn"
    backend: str = "mutual_knn_connected_components"
    input_contract: str = "shared_final_representation_pairwise_distance"
    seed_influences_geometry: bool = False

    def _geometry_dict(self) -> dict[str, Any]:
        return {
            "clusterer_id": self.clusterer_id,
            "backend": self.backend,
            "input_contract": self.input_contract,
            "seed_influences_geometry": self.seed_influences_geometry,
            "dataset_id": self.dataset_id,
            "parameters": {
                "neighbor_count": self.neighbor_count,
                "metric": self.metric,
            },
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "case_ids": list(self.case_ids),
            "labels": list(self.labels),
            "neighbor_case_ids": [list(items) for items in self.neighbor_case_ids],
            "mutual_edges": [list(edge) for edge in self.mutual_edges],
            "within_component_degrees": list(self.within_component_degrees),
            "component_count": self.component_count,
            "largest_component_share": self.largest_component_share,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._geometry_dict()
        payload["active_learning_seed"] = self.active_learning_seed
        payload["geometry_sha256"] = self.geometry_sha256
        return payload

    @property
    def geometry_sha256(self) -> str:
        return _semantic_sha256(self._geometry_dict())


@dataclass(frozen=True)
class MutualKNNSearchAttempt:
    configuration: MutualKNNConfiguration
    status: str
    result: MutualKNNClusteringResult | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.status == "fitted":
            if self.result is None or self.rejection_reason is not None:
                raise ValueError("fitted mutual-kNN attempt has inconsistent evidence")
        elif self.status == "rejected":
            if self.result is not None or not self.rejection_reason:
                raise ValueError("rejected mutual-kNN attempt has inconsistent evidence")
        else:
            raise ValueError("mutual-kNN attempt status is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "status": self.status,
            "result": self.result.to_dict() if self.result is not None else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class MutualKNNGridSearchResult:
    dataset_id: str
    active_learning_seed: int
    representation_id: str
    representation_matrix_sha256: str
    attempts: tuple[MutualKNNSearchAttempt, ...]
    seed_influences_geometry: bool = False

    @property
    def attempted_configuration_count(self) -> int:
        return len(self.attempts)

    @property
    def fitted_configuration_count(self) -> int:
        return sum(attempt.status == "fitted" for attempt in self.attempts)

    @property
    def rejected_configuration_count(self) -> int:
        return sum(attempt.status == "rejected" for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "active_learning_seed": self.active_learning_seed,
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "seed_influences_geometry": self.seed_influences_geometry,
            "attempted_configuration_count": self.attempted_configuration_count,
            "fitted_configuration_count": self.fitted_configuration_count,
            "rejected_configuration_count": self.rejected_configuration_count,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @property
    def search_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class EffectiveClusterRecord:
    raw_label: int
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.raw_label, bool) or self.raw_label < 0:
            raise ValueError("effective cluster raw label must be non-negative")
        if len(self.case_ids) < EFFECTIVE_CLUSTER_MIN_SUPPORT:
            raise ValueError("effective cluster support must be at least five")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("effective cluster case IDs must be unique")

    @property
    def support(self) -> int:
        return len(self.case_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_label": self.raw_label,
            "support": self.support,
            "case_ids": list(self.case_ids),
        }


@dataclass(frozen=True)
class EffectiveClusterPartition:
    dataset_id: str
    clusterer_id: str
    active_learning_seed: int
    min_effective_support: int
    source_geometry_sha256: str
    case_ids: tuple[str, ...]
    raw_labels: tuple[int, ...]
    effective_labels: tuple[int, ...]
    effective_clusters: tuple[EffectiveClusterRecord, ...]
    residual_case_ids: tuple[str, ...]
    noise_case_ids: tuple[str, ...]
    undersized_component_case_ids: tuple[str, ...]

    @property
    def effective_cluster_count(self) -> int:
        return len(self.effective_clusters)

    @property
    def effective_case_count(self) -> int:
        return sum(cluster.support for cluster in self.effective_clusters)

    @property
    def effective_coverage(self) -> float:
        if not self.case_ids:
            return 0.0
        return self.effective_case_count / len(self.case_ids)

    @property
    def largest_effective_cluster_size(self) -> int:
        if not self.effective_clusters:
            return 0
        return max(cluster.support for cluster in self.effective_clusters)

    @property
    def largest_effective_cluster_share_of_population(self) -> float:
        if not self.case_ids:
            return 0.0
        return self.largest_effective_cluster_size / len(self.case_ids)

    @property
    def largest_effective_cluster_share(self) -> float:
        if not self.effective_clusters:
            return 0.0
        return max(
            cluster.support for cluster in self.effective_clusters
        ) / self.effective_case_count

    def _partition_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "clusterer_id": self.clusterer_id,
            "active_learning_seed": self.active_learning_seed,
            "min_effective_support": self.min_effective_support,
            "source_geometry_sha256": self.source_geometry_sha256,
            "case_ids": list(self.case_ids),
            "raw_labels": list(self.raw_labels),
            "effective_labels": list(self.effective_labels),
            "effective_clusters": [
                cluster.to_dict() for cluster in self.effective_clusters
            ],
            "residual_case_ids": list(self.residual_case_ids),
            "noise_case_ids": list(self.noise_case_ids),
            "undersized_component_case_ids": list(
                self.undersized_component_case_ids
            ),
            "effective_cluster_count": self.effective_cluster_count,
            "effective_case_count": self.effective_case_count,
            "effective_coverage": self.effective_coverage,
            "largest_effective_cluster_share": (
                self.largest_effective_cluster_share
            ),
            "largest_effective_cluster_size": self.largest_effective_cluster_size,
            "largest_effective_cluster_share_of_population": (
                self.largest_effective_cluster_share_of_population
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._partition_dict()
        payload["partition_sha256"] = self.partition_sha256
        return payload

    @property
    def partition_sha256(self) -> str:
        return _semantic_sha256(self._partition_dict())


@dataclass(frozen=True)
class StructuralConfigurationScore:
    configuration_id: str
    dataset_id: str
    clusterer_id: str
    active_learning_seeds: tuple[int, ...]
    eligible: bool
    rank: int | None
    worst_target_cluster_count_deviation: int
    worst_largest_effective_cluster_share: float
    worst_effective_coverage: float
    cross_seed_stability: float
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "dataset_id": self.dataset_id,
            "clusterer_id": self.clusterer_id,
            "active_learning_seeds": list(self.active_learning_seeds),
            "eligible": self.eligible,
            "rank": self.rank,
            "worst_target_cluster_count_deviation": (
                self.worst_target_cluster_count_deviation
            ),
            "worst_largest_effective_cluster_share": (
                self.worst_largest_effective_cluster_share
            ),
            "worst_effective_coverage": self.worst_effective_coverage,
            "cross_seed_stability": self.cross_seed_stability,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @property
    def score_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class AmendedStructuralConfigurationScore:
    """Three-seed structure assessment for one approved-grid configuration."""

    configuration_id: str
    dataset_id: str
    clusterer_id: str
    assessment: StructureTripletAssessment
    eligible: bool
    rank: int | None
    worst_target_cluster_count_deviation: int
    worst_largest_effective_cluster_share_of_population: float
    mean_effective_coverage: float
    cross_seed_stability: float
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "dataset_id": self.dataset_id,
            "clusterer_id": self.clusterer_id,
            "assessment": self.assessment.to_dict(),
            "eligible": self.eligible,
            "rank": self.rank,
            "worst_target_cluster_count_deviation": (
                self.worst_target_cluster_count_deviation
            ),
            "worst_largest_effective_cluster_share_of_population": (
                self.worst_largest_effective_cluster_share_of_population
            ),
            "mean_effective_coverage": self.mean_effective_coverage,
            "cross_seed_stability": self.cross_seed_stability,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class AmendedStructureCellDecision:
    """Independent decision for one representation-by-clusterer cell."""

    dataset_id: str
    representation_id: str
    clusterer_id: str
    status: str
    structural_cohort: str
    selected_configuration_id: str | None
    nonconformance_reasons: tuple[str, ...]
    closest_configuration_ids: tuple[str, ...]
    configuration_scores: tuple[AmendedStructuralConfigurationScore, ...]
    extension_jobs: tuple[str, ...] = ()
    shared_representation_required: bool = False
    global_hard_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "representation_id": self.representation_id,
            "clusterer_id": self.clusterer_id,
            "status": self.status,
            "structural_cohort": self.structural_cohort,
            "selected_configuration_id": self.selected_configuration_id,
            "nonconformance_reasons": list(self.nonconformance_reasons),
            "closest_configuration_ids": list(self.closest_configuration_ids),
            "configuration_scores": [
                score.to_dict() for score in self.configuration_scores
            ],
            "extension_jobs": list(self.extension_jobs),
            "shared_representation_required": self.shared_representation_required,
            "global_hard_stop": self.global_hard_stop,
        }


@dataclass(frozen=True)
class SharedRepresentationCandidateScore:
    representation_id: str
    dataset_id: str
    feasible: bool
    rank: int | None
    selected_configuration_ids: tuple[tuple[str, str], ...]
    worst_family_target_cluster_count_deviation: int | None
    worst_family_largest_effective_cluster_share: float | None
    worst_family_effective_coverage: float | None
    worst_family_cross_seed_stability: float | None
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "dataset_id": self.dataset_id,
            "feasible": self.feasible,
            "rank": self.rank,
            "selected_configuration_ids": {
                clusterer_id: configuration_id
                for clusterer_id, configuration_id in self.selected_configuration_ids
            },
            "worst_family_target_cluster_count_deviation": (
                self.worst_family_target_cluster_count_deviation
            ),
            "worst_family_largest_effective_cluster_share": (
                self.worst_family_largest_effective_cluster_share
            ),
            "worst_family_effective_coverage": (
                self.worst_family_effective_coverage
            ),
            "worst_family_cross_seed_stability": (
                self.worst_family_cross_seed_stability
            ),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class SharedRepresentationSelectionResult:
    dataset_id: str
    status: str
    hard_stop: bool
    selected_representation_id: str | None
    required_clusterers: tuple[str, ...]
    candidates: tuple[SharedRepresentationCandidateScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "hard_stop": self.hard_stop,
            "selected_representation_id": self.selected_representation_id,
            "required_clusterers": list(self.required_clusterers),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_sha256": self.selection_sha256,
        }

    @property
    def selection_sha256(self) -> str:
        return _semantic_sha256(
            {
                "dataset_id": self.dataset_id,
                "status": self.status,
                "hard_stop": self.hard_stop,
                "selected_representation_id": self.selected_representation_id,
                "required_clusterers": list(self.required_clusterers),
                "candidates": [candidate.to_dict() for candidate in self.candidates],
            }
        )


@dataclass(frozen=True)
class KMeansClusteringResult:
    dataset_id: str
    active_learning_seed: int
    cluster_count: int
    n_init: int
    algorithm: str
    representation_id: str
    representation_matrix_sha256: str
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    cluster_centers: np.ndarray
    inertia: float
    n_iter: int
    clusterer_id: str = "kmeans"
    backend: str = "sklearn.cluster.KMeans"
    input_contract: str = "shared_final_representation_matrix"
    authority_plan_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusterer_id": self.clusterer_id,
            "backend": self.backend,
            "input_contract": self.input_contract,
            "authority_plan_used": self.authority_plan_used,
            "dataset_id": self.dataset_id,
            "active_learning_seed": self.active_learning_seed,
            "parameters": {
                "cluster_count": self.cluster_count,
                "n_init": self.n_init,
                "algorithm": self.algorithm,
            },
            "representation_id": self.representation_id,
            "representation_matrix_sha256": self.representation_matrix_sha256,
            "case_ids": list(self.case_ids),
            "labels": list(self.labels),
            "cluster_centers": self.cluster_centers.tolist(),
            "inertia": self.inertia,
            "n_iter": self.n_iter,
        }

    @property
    def geometry_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


def kmeans_parameter_grid(dataset_id: str) -> tuple[KMeansConfiguration, ...]:
    """Return the exact frozen K-means grid for one dataset."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("K-means dataset is outside the formal protocol")
    return tuple(
        KMeansConfiguration(dataset_id=dataset_id, cluster_count=cluster_count)
        for cluster_count in KMEANS_CLUSTER_COUNTS[dataset_id]
    )


def dbscan_parameter_grid() -> tuple[DBSCANConfiguration, ...]:
    """Return the exact frozen DBSCAN Cartesian grid in trajectory order."""

    return tuple(
        DBSCANConfiguration(
            min_samples=min_samples,
            eps_quantile=eps_quantile,
            metric=metric,
        )
        for metric in DBSCAN_METRICS
        for min_samples in DBSCAN_MIN_SAMPLES
        for eps_quantile in DBSCAN_EPS_QUANTILES
    )


def hdbscan_parameter_grid() -> tuple[HDBSCANConfiguration, ...]:
    """Return the exact frozen HDBSCAN Cartesian grid."""

    return tuple(
        HDBSCANConfiguration(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
        for cluster_selection_method in HDBSCAN_CLUSTER_SELECTION_METHODS
        for min_cluster_size in HDBSCAN_MIN_CLUSTER_SIZES
        for min_samples in HDBSCAN_MIN_SAMPLES
    )


def mutual_knn_parameter_grid() -> tuple[MutualKNNConfiguration, ...]:
    """Return the exact frozen mutual-kNN Cartesian grid."""

    return tuple(
        MutualKNNConfiguration(neighbor_count=neighbor_count, metric=metric)
        for metric in MUTUAL_KNN_METRICS
        for neighbor_count in MUTUAL_KNN_NEIGHBOR_COUNTS
    )


def build_effective_cluster_partition(
    *,
    dataset_id: str,
    clusterer_id: str,
    active_learning_seed: int,
    case_ids: tuple[str, ...],
    raw_labels: tuple[int, ...],
    source_geometry_sha256: str,
    min_effective_support: int = EFFECTIVE_CLUSTER_MIN_SUPPORT,
) -> EffectiveClusterPartition:
    """Separate support-at-least-five clusters from the residual pool."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("effective-cluster dataset is outside the formal protocol")
    if clusterer_id not in FORMAL_CLUSTERER_IDS:
        raise ValueError("effective-cluster source is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    if (
        isinstance(min_effective_support, bool)
        or min_effective_support != EFFECTIVE_CLUSTER_MIN_SUPPORT
    ):
        raise ValueError("combined 2.0 effective-cluster support must equal five")
    if (
        len(source_geometry_sha256) != 64
        or source_geometry_sha256.lower() != source_geometry_sha256
        or any(character not in "0123456789abcdef" for character in source_geometry_sha256)
    ):
        raise ValueError("source geometry must use a lowercase SHA-256")
    normalized_case_ids = tuple(str(case_id) for case_id in case_ids)
    if (
        not normalized_case_ids
        or any(not case_id for case_id in normalized_case_ids)
        or len(normalized_case_ids) != len(set(normalized_case_ids))
    ):
        raise ValueError("effective-cluster case IDs must be non-empty and unique")
    if len(raw_labels) != len(normalized_case_ids) or any(
        isinstance(label, bool) or not isinstance(label, (int, np.integer))
        for label in raw_labels
    ):
        raise ValueError("raw cluster labels must match the case population")
    normalized_labels = tuple(int(label) for label in raw_labels)
    members_by_label: dict[int, list[str]] = {}
    for case_id, label in zip(normalized_case_ids, normalized_labels):
        if label >= 0:
            members_by_label.setdefault(label, []).append(case_id)
    effective_raw_labels = frozenset(
        label
        for label, members in members_by_label.items()
        if len(members) >= min_effective_support
    )
    effective_clusters = tuple(
        EffectiveClusterRecord(
            raw_label=label,
            case_ids=tuple(members_by_label[label]),
        )
        for label in sorted(effective_raw_labels)
    )
    noise_case_ids = tuple(
        case_id
        for case_id, label in zip(normalized_case_ids, normalized_labels)
        if label < 0
    )
    undersized_case_ids = tuple(
        case_id
        for case_id, label in zip(normalized_case_ids, normalized_labels)
        if label >= 0 and label not in effective_raw_labels
    )
    residual_case_ids = tuple(
        case_id
        for case_id, label in zip(normalized_case_ids, normalized_labels)
        if label < 0 or label not in effective_raw_labels
    )
    effective_labels = tuple(
        label if label in effective_raw_labels else -1 for label in normalized_labels
    )
    return EffectiveClusterPartition(
        dataset_id=dataset_id,
        clusterer_id=clusterer_id,
        active_learning_seed=active_learning_seed,
        min_effective_support=min_effective_support,
        source_geometry_sha256=source_geometry_sha256,
        case_ids=normalized_case_ids,
        raw_labels=normalized_labels,
        effective_labels=effective_labels,
        effective_clusters=effective_clusters,
        residual_case_ids=residual_case_ids,
        noise_case_ids=noise_case_ids,
        undersized_component_case_ids=undersized_case_ids,
    )


def evaluate_structure_gate(
    partition: EffectiveClusterPartition,
) -> StructureGateResult:
    """Apply the frozen dataset-specific structure gates to one partition."""

    if type(partition) is not EffectiveClusterPartition:
        raise ValueError("structure gating requires an effective-cluster partition")
    protocol = default_protocol_manifest()
    gate = protocol.structure_gates[partition.dataset_id]
    rejection_reasons: list[str] = []
    if partition.effective_cluster_count == 1:
        rejection_reasons.append("single_effective_cluster_prohibited")
    if partition.effective_cluster_count < gate.min_cluster_count:
        rejection_reasons.append(
            "effective_cluster_count_below_minimum:"
            f"observed={partition.effective_cluster_count},"
            f"minimum={gate.min_cluster_count}"
        )
    if partition.effective_cluster_count > gate.max_cluster_count:
        rejection_reasons.append(
            "effective_cluster_count_above_maximum:"
            f"observed={partition.effective_cluster_count},"
            f"maximum={gate.max_cluster_count}"
        )
    if partition.effective_coverage < gate.min_effective_coverage:
        rejection_reasons.append(
            "effective_coverage_below_minimum:"
            f"observed={partition.effective_coverage:.12g},"
            f"minimum={gate.min_effective_coverage:.12g}"
        )
    if (
        partition.largest_effective_cluster_share
        > gate.max_largest_effective_cluster_share
    ):
        rejection_reasons.append(
            "largest_effective_cluster_share_above_maximum:"
            f"observed={partition.largest_effective_cluster_share:.12g},"
            f"maximum={gate.max_largest_effective_cluster_share:.12g}"
        )
    return StructureGateResult(
        dataset_id=DatasetId(partition.dataset_id),
        clusterer_id=partition.clusterer_id,
        active_learning_seed=partition.active_learning_seed,
        effective_cluster_count=partition.effective_cluster_count,
        effective_coverage=partition.effective_coverage,
        largest_effective_cluster_share=(
            partition.largest_effective_cluster_share
        ),
        passed=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
    )


def _validated_cross_seed_partitions(
    partitions: tuple[EffectiveClusterPartition, ...],
) -> tuple[EffectiveClusterPartition, ...]:
    if len(partitions) != len(ACTIVE_LEARNING_SEEDS) or any(
        type(partition) is not EffectiveClusterPartition for partition in partitions
    ):
        raise ValueError("cross-seed evidence requires partitions for seeds 41/42/43")
    by_seed = {partition.active_learning_seed: partition for partition in partitions}
    if tuple(sorted(by_seed)) != ACTIVE_LEARNING_SEEDS or len(by_seed) != len(partitions):
        raise ValueError("cross-seed evidence requires partitions for seeds 41/42/43")
    ordered = tuple(by_seed[seed] for seed in ACTIVE_LEARNING_SEEDS)
    reference = ordered[0]
    for partition in ordered[1:]:
        if (
            partition.dataset_id != reference.dataset_id
            or partition.clusterer_id != reference.clusterer_id
        ):
            raise ValueError("cross-seed partitions must share dataset and clusterer")
        if partition.case_ids != reference.case_ids:
            raise ValueError("cross-seed partitions must share the exact case population")
    return ordered


def compute_cross_seed_stability(
    partitions: tuple[EffectiveClusterPartition, ...],
) -> float:
    """Mean pairwise ARI over the frozen active-learning seed axis."""

    ordered = _validated_cross_seed_partitions(partitions)
    pairwise_scores = tuple(
        float(adjusted_rand_score(left.effective_labels, right.effective_labels))
        for left, right in combinations(ordered, 2)
    )
    stability = float(np.mean(pairwise_scores))
    if not np.isfinite(stability) or not -1.0 <= stability <= 1.0:
        raise ValueError("cross-seed adjusted Rand stability is invalid")
    return stability


def _structural_score_without_rank(
    configuration_id: str,
    partitions: tuple[EffectiveClusterPartition, ...],
) -> StructuralConfigurationScore:
    if not str(configuration_id).strip():
        raise ValueError("structural configuration ID must be non-empty")
    ordered = _validated_cross_seed_partitions(partitions)
    stability = compute_cross_seed_stability(ordered)
    gates = tuple(evaluate_structure_gate(partition) for partition in ordered)
    gate_config = default_protocol_manifest().structure_gates[ordered[0].dataset_id]
    rejection_reasons = tuple(
        f"seed{gate.active_learning_seed}:{reason}"
        for gate in gates
        for reason in gate.rejection_reasons
    )
    return StructuralConfigurationScore(
        configuration_id=configuration_id,
        dataset_id=ordered[0].dataset_id,
        clusterer_id=ordered[0].clusterer_id,
        active_learning_seeds=ACTIVE_LEARNING_SEEDS,
        eligible=not rejection_reasons,
        rank=None,
        worst_target_cluster_count_deviation=max(
            abs(gate.effective_cluster_count - gate_config.target_cluster_count)
            for gate in gates
        ),
        worst_largest_effective_cluster_share=max(
            gate.largest_effective_cluster_share for gate in gates
        ),
        worst_effective_coverage=min(gate.effective_coverage for gate in gates),
        cross_seed_stability=stability,
        rejection_reasons=rejection_reasons,
    )


def _ranked_structural_score(
    score: StructuralConfigurationScore,
    rank: int,
) -> StructuralConfigurationScore:
    return StructuralConfigurationScore(
        configuration_id=score.configuration_id,
        dataset_id=score.dataset_id,
        clusterer_id=score.clusterer_id,
        active_learning_seeds=score.active_learning_seeds,
        eligible=score.eligible,
        rank=rank,
        worst_target_cluster_count_deviation=(
            score.worst_target_cluster_count_deviation
        ),
        worst_largest_effective_cluster_share=(
            score.worst_largest_effective_cluster_share
        ),
        worst_effective_coverage=score.worst_effective_coverage,
        cross_seed_stability=score.cross_seed_stability,
        rejection_reasons=score.rejection_reasons,
    )


def rank_structural_configurations(
    configurations: Mapping[str, tuple[EffectiveClusterPartition, ...]],
) -> tuple[StructuralConfigurationScore, ...]:
    """Rank valid configurations by the frozen four-level structural key."""

    if not isinstance(configurations, Mapping) or not configurations:
        raise ValueError("structural ranking requires at least one configuration")
    scores = tuple(
        _structural_score_without_rank(configuration_id, tuple(partitions))
        for configuration_id, partitions in configurations.items()
    )
    reference = scores[0]
    reference_population = tuple(configurations[reference.configuration_id])[0].case_ids
    for score in scores[1:]:
        population = tuple(configurations[score.configuration_id])[0].case_ids
        if (
            score.dataset_id != reference.dataset_id
            or score.clusterer_id != reference.clusterer_id
            or population != reference_population
        ):
            raise ValueError(
                "ranked structural configurations must share dataset, clusterer, and population"
            )
    eligible = sorted(
        (score for score in scores if score.eligible),
        key=lambda score: (
            score.worst_target_cluster_count_deviation,
            score.worst_largest_effective_cluster_share,
            -score.worst_effective_coverage,
            -score.cross_seed_stability,
            score.configuration_id,
        ),
    )
    ranked = tuple(
        _ranked_structural_score(score, rank)
        for rank, score in enumerate(eligible, start=1)
    )
    rejected = tuple(
        sorted(
            (score for score in scores if not score.eligible),
            key=lambda score: score.configuration_id,
        )
    )
    return (*ranked, *rejected)


def _amended_structure_observation(
    partition: EffectiveClusterPartition,
) -> SeedStructureObservation:
    return SeedStructureObservation(
        active_learning_seed=partition.active_learning_seed,
        effective_cluster_count=partition.effective_cluster_count,
        effective_coverage=partition.effective_coverage,
        largest_effective_cluster_size=partition.largest_effective_cluster_size,
        candidate_count=len(partition.case_ids),
        residual_or_noise_count=len(partition.residual_case_ids),
    )


def _amended_rejection_reasons(
    assessment: StructureTripletAssessment,
) -> tuple[str, ...]:
    reasons = [
        f"seed{seed}:effective_cluster_count={count}:outside_legal_band"
        for seed, count in sorted(assessment.illegal_count_by_seed.items())
    ]
    if assessment.counts_legal and not assessment.mean_coverage_passed:
        reasons.append(
            "three_seed_mean_effective_coverage_not_strictly_above_half:"
            f"observed={assessment.mean_effective_coverage:.12g}"
        )
    if assessment.counts_legal and not assessment.largest_cluster_passed:
        reasons.append(
            "largest_effective_cluster_full_pool_cap_failed:"
            f"passing_seeds={assessment.largest_cluster_passing_seed_count}"
        )
    return tuple(reasons)


def _amended_score_without_rank(
    *,
    configuration_id: str,
    partitions: tuple[EffectiveClusterPartition, ...],
    ground_truth_cap: GroundTruthClusterCap,
) -> AmendedStructuralConfigurationScore:
    if not str(configuration_id).strip():
        raise ValueError("amended structural configuration ID must be non-empty")
    ordered = _validated_cross_seed_partitions(partitions)
    if len(ordered[0].case_ids) != ground_truth_cap.candidate_count:
        raise ValueError("ground-truth cap candidate count drifted")
    assessment = assess_structure_triplet(
        dataset_id=ordered[0].dataset_id,
        observations=tuple(_amended_structure_observation(item) for item in ordered),
        ground_truth_cap=ground_truth_cap,
    )
    policy = amended_structure_policy(ordered[0].dataset_id)
    return AmendedStructuralConfigurationScore(
        configuration_id=configuration_id,
        dataset_id=ordered[0].dataset_id,
        clusterer_id=ordered[0].clusterer_id,
        assessment=assessment,
        eligible=assessment.status == "usable",
        rank=None,
        worst_target_cluster_count_deviation=max(
            abs(item.effective_cluster_count - policy.target_cluster_count)
            for item in ordered
        ),
        worst_largest_effective_cluster_share_of_population=max(
            item.largest_effective_cluster_share_of_population for item in ordered
        ),
        mean_effective_coverage=assessment.mean_effective_coverage,
        cross_seed_stability=compute_cross_seed_stability(ordered),
        rejection_reasons=_amended_rejection_reasons(assessment),
    )


def _ranked_amended_score(
    score: AmendedStructuralConfigurationScore,
    rank: int,
) -> AmendedStructuralConfigurationScore:
    return AmendedStructuralConfigurationScore(
        configuration_id=score.configuration_id,
        dataset_id=score.dataset_id,
        clusterer_id=score.clusterer_id,
        assessment=score.assessment,
        eligible=score.eligible,
        rank=rank,
        worst_target_cluster_count_deviation=(
            score.worst_target_cluster_count_deviation
        ),
        worst_largest_effective_cluster_share_of_population=(
            score.worst_largest_effective_cluster_share_of_population
        ),
        mean_effective_coverage=score.mean_effective_coverage,
        cross_seed_stability=score.cross_seed_stability,
        rejection_reasons=score.rejection_reasons,
    )


def _count_band_distance(score: AmendedStructuralConfigurationScore) -> int:
    policy = score.assessment.policy
    return sum(
        max(
            policy.min_cluster_count - item.effective_cluster_count,
            item.effective_cluster_count - policy.max_cluster_count,
            0,
        )
        for item in score.assessment.observations
    )


def _amended_label_free_order(
    score: AmendedStructuralConfigurationScore,
) -> tuple[Any, ...]:
    return (
        score.worst_target_cluster_count_deviation,
        score.worst_largest_effective_cluster_share_of_population,
        -score.mean_effective_coverage,
        -score.cross_seed_stability,
        score.configuration_id,
    )


def rank_amended_structural_configurations(
    configurations: Mapping[str, tuple[EffectiveClusterPartition, ...]],
    *,
    ground_truth_cap: GroundTruthClusterCap,
) -> tuple[AmendedStructuralConfigurationScore, ...]:
    """Assess one cell without shared-representation or per-seed hard stops."""

    if not isinstance(configurations, Mapping) or not configurations:
        raise ValueError("amended structural ranking requires configurations")
    if type(ground_truth_cap) is not GroundTruthClusterCap:
        raise ValueError("amended structural ranking requires a frozen cap")
    scores = tuple(
        _amended_score_without_rank(
            configuration_id=configuration_id,
            partitions=tuple(partitions),
            ground_truth_cap=ground_truth_cap,
        )
        for configuration_id, partitions in configurations.items()
    )
    reference = scores[0]
    if any(
        score.dataset_id != reference.dataset_id
        or score.clusterer_id != reference.clusterer_id
        for score in scores[1:]
    ):
        raise ValueError("amended ranking cannot mix datasets or clustering families")
    usable = sorted((score for score in scores if score.eligible), key=_amended_label_free_order)
    ranked = tuple(
        _ranked_amended_score(score, rank)
        for rank, score in enumerate(usable, start=1)
    )
    legal_rejected = sorted(
        (
            score
            for score in scores
            if not score.eligible and score.assessment.counts_legal
        ),
        key=_amended_label_free_order,
    )
    illegal = sorted(
        (score for score in scores if not score.assessment.counts_legal),
        key=lambda score: (
            _count_band_distance(score),
            *_amended_label_free_order(score),
        ),
    )
    return (*ranked, *legal_rejected, *illegal)


def select_amended_structure_cell(
    *,
    dataset_id: str,
    representation_id: str,
    clusterer_id: str,
    ranking: tuple[AmendedStructuralConfigurationScore, ...],
) -> AmendedStructureCellDecision:
    """Freeze one approved-grid multi-cluster configuration for downstream RCL."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("amended cell dataset is outside the formal protocol")
    if representation_id not in FORMAL_REPRESENTATION_IDS:
        raise ValueError("amended cell representation is outside the formal protocol")
    if clusterer_id not in FORMAL_CLUSTERER_IDS:
        raise ValueError("amended cell clusterer is outside the formal protocol")
    if not ranking or any(
        type(score) is not AmendedStructuralConfigurationScore
        or score.dataset_id != dataset_id
        or score.clusterer_id != clusterer_id
        for score in ranking
    ):
        raise ValueError("amended cell ranking identity drifted")
    selected = next((score for score in ranking if score.eligible), None)
    if selected is None:
        selected = next(
            (
                score
                for score in ranking
                if all(
                    observation.effective_cluster_count >= 2
                    for observation in score.assessment.observations
                )
            ),
            None,
        )
    if selected is None:
        raise ValueError(
            "approved grid has no complete multi-cluster configuration for RCL"
        )
    structural_cohort = (
        "structurally_admissible"
        if selected.eligible
        else "structurally_nonconforming"
    )
    status = "usable" if selected.eligible else "usable_structurally_nonconforming"
    return AmendedStructureCellDecision(
        dataset_id=dataset_id,
        representation_id=representation_id,
        clusterer_id=clusterer_id,
        status=status,
        structural_cohort=structural_cohort,
        selected_configuration_id=selected.configuration_id,
        nonconformance_reasons=selected.rejection_reasons,
        closest_configuration_ids=tuple(
            score.configuration_id for score in ranking[:5]
        ),
        configuration_scores=ranking,
        extension_jobs=(),
        shared_representation_required=False,
        global_hard_stop=False,
    )


def _ranked_shared_representation_candidate(
    score: SharedRepresentationCandidateScore,
    rank: int,
) -> SharedRepresentationCandidateScore:
    return SharedRepresentationCandidateScore(
        representation_id=score.representation_id,
        dataset_id=score.dataset_id,
        feasible=score.feasible,
        rank=rank,
        selected_configuration_ids=score.selected_configuration_ids,
        worst_family_target_cluster_count_deviation=(
            score.worst_family_target_cluster_count_deviation
        ),
        worst_family_largest_effective_cluster_share=(
            score.worst_family_largest_effective_cluster_share
        ),
        worst_family_effective_coverage=score.worst_family_effective_coverage,
        worst_family_cross_seed_stability=(
            score.worst_family_cross_seed_stability
        ),
        rejection_reasons=score.rejection_reasons,
    )


def select_shared_representation(
    rankings_by_representation: Mapping[
        str,
        Mapping[str, tuple[StructuralConfigurationScore, ...]],
    ],
) -> SharedRepresentationSelectionResult:
    """Select one ten-candidate representation by four-family minimax."""

    if set(rankings_by_representation) != set(FORMAL_REPRESENTATION_IDS) or len(
        rankings_by_representation
    ) != len(FORMAL_REPRESENTATION_IDS):
        raise ValueError("shared selection requires exactly the ten formal representations")
    candidate_scores: list[SharedRepresentationCandidateScore] = []
    observed_dataset_id: str | None = None
    for representation_id in FORMAL_REPRESENTATION_IDS:
        family_rankings = rankings_by_representation[representation_id]
        if set(family_rankings) != FORMAL_CLUSTERER_IDS or len(
            family_rankings
        ) != len(FORMAL_CLUSTERER_ORDER):
            raise ValueError("shared selection requires all four clustering families")
        selected: list[tuple[str, StructuralConfigurationScore]] = []
        rejection_reasons: list[str] = []
        for clusterer_id in FORMAL_CLUSTERER_ORDER:
            ranking = tuple(family_rankings[clusterer_id])
            if not ranking or any(
                type(score) is not StructuralConfigurationScore for score in ranking
            ):
                raise ValueError("each clustering family requires a typed structural ranking")
            for score in ranking:
                if score.clusterer_id != clusterer_id:
                    raise ValueError("structural ranking clusterer does not match its family")
                if observed_dataset_id is None:
                    observed_dataset_id = score.dataset_id
                elif score.dataset_id != observed_dataset_id:
                    raise ValueError("shared representation selection cannot mix datasets")
            eligible = tuple(
                score
                for score in ranking
                if score.eligible and score.rank is not None
            )
            if not eligible:
                rejection_reasons.append(
                    f"{clusterer_id}:no_structurally_valid_configuration"
                )
            else:
                selected.append(
                    (
                        clusterer_id,
                        min(
                            eligible,
                            key=lambda score: (score.rank, score.configuration_id),
                        ),
                    )
                )
        feasible = not rejection_reasons
        chosen_scores = tuple(score for _, score in selected)
        candidate_scores.append(
            SharedRepresentationCandidateScore(
                representation_id=representation_id,
                dataset_id=observed_dataset_id or "",
                feasible=feasible,
                rank=None,
                selected_configuration_ids=tuple(
                    (clusterer_id, score.configuration_id)
                    for clusterer_id, score in selected
                ),
                worst_family_target_cluster_count_deviation=(
                    max(
                        score.worst_target_cluster_count_deviation
                        for score in chosen_scores
                    )
                    if feasible
                    else None
                ),
                worst_family_largest_effective_cluster_share=(
                    max(
                        score.worst_largest_effective_cluster_share
                        for score in chosen_scores
                    )
                    if feasible
                    else None
                ),
                worst_family_effective_coverage=(
                    min(score.worst_effective_coverage for score in chosen_scores)
                    if feasible
                    else None
                ),
                worst_family_cross_seed_stability=(
                    min(score.cross_seed_stability for score in chosen_scores)
                    if feasible
                    else None
                ),
                rejection_reasons=tuple(rejection_reasons),
            )
        )
    formal_order = {
        representation_id: index
        for index, representation_id in enumerate(FORMAL_REPRESENTATION_IDS)
    }
    feasible_scores = sorted(
        (score for score in candidate_scores if score.feasible),
        key=lambda score: (
            score.worst_family_target_cluster_count_deviation,
            score.worst_family_largest_effective_cluster_share,
            -score.worst_family_effective_coverage,
            -score.worst_family_cross_seed_stability,
            formal_order[score.representation_id],
        ),
    )
    ranked = tuple(
        _ranked_shared_representation_candidate(score, rank)
        for rank, score in enumerate(feasible_scores, start=1)
    )
    infeasible = tuple(
        score for score in candidate_scores if not score.feasible
    )
    ordered_candidates = (*ranked, *infeasible)
    if ranked:
        return SharedRepresentationSelectionResult(
            dataset_id=observed_dataset_id or "",
            status="selected",
            hard_stop=False,
            selected_representation_id=ranked[0].representation_id,
            required_clusterers=FORMAL_CLUSTERER_ORDER,
            candidates=ordered_candidates,
        )
    return SharedRepresentationSelectionResult(
        dataset_id=observed_dataset_id or "",
        status="no_feasible_shared_representation",
        hard_stop=True,
        selected_representation_id=None,
        required_clusterers=FORMAL_CLUSTERER_ORDER,
        candidates=ordered_candidates,
    )


def _validated_dbscan_distance_matrix(
    candidate: RepresentationMatrixCandidate,
    metric: str,
) -> np.ndarray:
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("DBSCAN requires a typed representation candidate")
    if (
        candidate.representation_id not in FORMAL_REPRESENTATION_IDS
        or not candidate.formal_eligible
    ):
        raise ValueError("DBSCAN requires a formal representation candidate")
    if metric not in DBSCAN_METRICS:
        raise ValueError("DBSCAN metric is outside the frozen grid")
    matrix = np.asarray(candidate.matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(candidate.case_ids)
        or matrix.shape[1] != len(candidate.feature_names)
        or matrix.shape[0] < 1
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("shared representation matrix is invalid for DBSCAN")
    distances = pairwise_distances(matrix, metric=metric, n_jobs=1)
    np.fill_diagonal(distances, 0.0)
    if metric == "cosine":
        # Cosine distance between identical nonzero vectors is mathematically
        # zero, but BLAS rounding can leave a one-ulp positive residue.
        distances[np.isclose(distances, 0.0, rtol=0.0, atol=1e-12)] = 0.0
    if not np.isfinite(distances).all() or np.any(distances < -1e-12):
        raise ValueError("DBSCAN pairwise distance matrix is invalid")
    distances = np.maximum(distances, 0.0)
    distances.setflags(write=False)
    return distances


def _derive_dbscan_eps_from_distances(
    *,
    candidate: RepresentationMatrixCandidate,
    configuration: DBSCANConfiguration,
    distance_matrix: np.ndarray,
) -> DBSCANEpsDerivation:
    if distance_matrix.shape != (len(candidate.case_ids), len(candidate.case_ids)):
        raise ValueError("DBSCAN distance matrix population is invalid")
    if len(candidate.case_ids) < configuration.min_samples:
        raise ValueError("DBSCAN min_samples exceeds the candidate population")
    k_distances_array = np.sort(distance_matrix, axis=1)[
        :, configuration.min_samples - 1
    ]
    eps = float(
        np.quantile(
            k_distances_array,
            configuration.eps_quantile,
            method="linear",
        )
    )
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("DBSCAN k-distance quantile produced a non-positive eps")
    return DBSCANEpsDerivation(
        min_samples=configuration.min_samples,
        eps_quantile=configuration.eps_quantile,
        metric=configuration.metric,
        eps=eps,
        neighbor_rank=configuration.min_samples,
        includes_self=True,
        k_distances=tuple(float(value) for value in k_distances_array),
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
    )


def derive_dbscan_eps(
    *,
    candidate: RepresentationMatrixCandidate,
    min_samples: int,
    eps_quantile: float,
    metric: str,
) -> DBSCANEpsDerivation:
    """Derive eps from the candidate-pool including-self k-distance curve."""

    configuration = DBSCANConfiguration(
        min_samples=min_samples,
        eps_quantile=eps_quantile,
        metric=metric,
    )
    distances = _validated_dbscan_distance_matrix(candidate, metric)
    return _derive_dbscan_eps_from_distances(
        candidate=candidate,
        configuration=configuration,
        distance_matrix=distances,
    )


def _fit_dbscan_from_distances(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    derivation: DBSCANEpsDerivation,
    distance_matrix: np.ndarray,
) -> DBSCANClusteringResult:
    estimator = DBSCAN(
        eps=derivation.eps,
        min_samples=derivation.min_samples,
        metric="precomputed",
        n_jobs=1,
    )
    labels_array = estimator.fit_predict(distance_matrix)
    if labels_array.shape != (len(candidate.case_ids),):
        raise ValueError("native DBSCAN returned an invalid label population")
    core_indices = frozenset(int(index) for index in estimator.core_sample_indices_)
    assigned_indices = tuple(
        index for index, label in enumerate(labels_array) if int(label) != -1
    )
    noise_indices = tuple(
        index for index, label in enumerate(labels_array) if int(label) == -1
    )
    border_indices = tuple(
        index for index in assigned_indices if index not in core_indices
    )
    cluster_labels = tuple(sorted({int(label) for label in labels_array if label != -1}))
    cluster_sizes = tuple(
        int(np.sum(labels_array == label)) for label in cluster_labels
    )
    non_noise_count = len(assigned_indices)
    noise_count = len(noise_indices)
    coverage = non_noise_count / len(candidate.case_ids)
    largest_share = (
        max(cluster_sizes) / non_noise_count if non_noise_count else 0.0
    )
    if not cluster_labels:
        density_phase = "all_noise"
    elif len(cluster_labels) == 1:
        density_phase = "single_density_component"
    elif noise_count / len(candidate.case_ids) > 0.70:
        density_phase = "excessive_noise"
    else:
        density_phase = "multiple_density_components"
    return DBSCANClusteringResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        min_samples=derivation.min_samples,
        eps_quantile=derivation.eps_quantile,
        metric=derivation.metric,
        eps=derivation.eps,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        eps_derivation_sha256=derivation.derivation_sha256,
        case_ids=candidate.case_ids,
        labels=tuple(int(label) for label in labels_array),
        core_case_ids=tuple(candidate.case_ids[index] for index in sorted(core_indices)),
        border_case_ids=tuple(candidate.case_ids[index] for index in border_indices),
        noise_case_ids=tuple(candidate.case_ids[index] for index in noise_indices),
        non_noise_cluster_count=len(cluster_labels),
        non_noise_case_count=non_noise_count,
        noise_count=noise_count,
        non_noise_coverage=float(coverage),
        largest_non_noise_cluster_share=float(largest_share),
        density_phase=density_phase,
    )


def fit_native_dbscan(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    min_samples: int,
    eps_quantile: float,
    metric: str,
) -> DBSCANClusteringResult:
    """Fit deterministic DBSCAN from one frozen k-distance configuration."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("DBSCAN dataset is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    configuration = DBSCANConfiguration(
        min_samples=min_samples,
        eps_quantile=eps_quantile,
        metric=metric,
    )
    distances = _validated_dbscan_distance_matrix(candidate, metric)
    derivation = _derive_dbscan_eps_from_distances(
        candidate=candidate,
        configuration=configuration,
        distance_matrix=distances,
    )
    return _fit_dbscan_from_distances(
        dataset_id=dataset_id,
        candidate=candidate,
        active_learning_seed=active_learning_seed,
        derivation=derivation,
        distance_matrix=distances,
    )


def run_dbscan_grid(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
) -> DBSCANGridSearchResult:
    """Fit all 144 DBSCAN settings and preserve each ordered trajectory."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("DBSCAN dataset is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    attempts: list[DBSCANSearchAttempt] = []
    for metric in DBSCAN_METRICS:
        distances = _validated_dbscan_distance_matrix(candidate, metric)
        for min_samples in DBSCAN_MIN_SAMPLES:
            for position, eps_quantile in enumerate(
                DBSCAN_EPS_QUANTILES, start=1
            ):
                configuration = DBSCANConfiguration(
                    min_samples=min_samples,
                    eps_quantile=eps_quantile,
                    metric=metric,
                )
                derivation: DBSCANEpsDerivation | None = None
                try:
                    derivation = _derive_dbscan_eps_from_distances(
                        candidate=candidate,
                        configuration=configuration,
                        distance_matrix=distances,
                    )
                    result = _fit_dbscan_from_distances(
                        dataset_id=dataset_id,
                        candidate=candidate,
                        active_learning_seed=active_learning_seed,
                        derivation=derivation,
                        distance_matrix=distances,
                    )
                except ValueError as exc:
                    attempts.append(
                        DBSCANSearchAttempt(
                            configuration=configuration,
                            trajectory_position=position,
                            status="rejected",
                            eps_derivation=derivation,
                            result=None,
                            rejection_reason=str(exc),
                        )
                    )
                else:
                    attempts.append(
                        DBSCANSearchAttempt(
                            configuration=configuration,
                            trajectory_position=position,
                            status="fitted",
                            eps_derivation=derivation,
                            result=result,
                            rejection_reason=None,
                        )
                    )
    return DBSCANGridSearchResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        attempts=tuple(attempts),
    )


def _validated_hdbscan_matrix(
    candidate: RepresentationMatrixCandidate,
) -> np.ndarray:
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("HDBSCAN requires a typed representation candidate")
    if (
        candidate.representation_id not in FORMAL_REPRESENTATION_IDS
        or not candidate.formal_eligible
    ):
        raise ValueError("HDBSCAN requires a formal representation candidate")
    matrix = np.asarray(candidate.matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(candidate.case_ids)
        or matrix.shape[1] != len(candidate.feature_names)
        or matrix.shape[0] < 1
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("shared representation matrix is invalid for HDBSCAN")
    return matrix


def _fit_hdbscan_from_matrix(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    configuration: HDBSCANConfiguration,
    matrix: np.ndarray,
) -> HDBSCANClusteringResult:
    if matrix.shape[0] < configuration.min_cluster_size:
        raise ValueError("HDBSCAN min_cluster_size exceeds the candidate population")
    estimator = HDBSCAN(
        min_cluster_size=configuration.min_cluster_size,
        min_samples=configuration.min_samples,
        metric=configuration.metric,
        n_jobs=1,
        cluster_selection_method=configuration.cluster_selection_method,
        allow_single_cluster=configuration.allow_single_cluster,
        store_centers=HDBSCAN_STORE_CENTERS,
        copy=True,
    )
    with threadpool_limits(limits=1):
        labels_array = estimator.fit_predict(matrix)
    memberships = np.asarray(estimator.probabilities_, dtype=float)
    if (
        labels_array.shape != (matrix.shape[0],)
        or memberships.shape != (matrix.shape[0],)
        or not np.isfinite(memberships).all()
        or np.any(memberships < 0.0)
        or np.any(memberships > 1.0)
    ):
        raise ValueError("native HDBSCAN returned invalid membership evidence")
    cluster_labels = tuple(sorted({int(label) for label in labels_array if label >= 0}))
    if len(cluster_labels) == 1:
        raise ValueError("combined 2.0 HDBSCAN single-cluster output is prohibited")
    if cluster_labels:
        medoids = np.asarray(estimator.medoids_, dtype=float).copy()
    else:
        medoids = np.empty((0, matrix.shape[1]), dtype=float)
    if medoids.shape != (len(cluster_labels), matrix.shape[1]) or not np.isfinite(
        medoids
    ).all():
        raise ValueError("native HDBSCAN returned invalid medoid evidence")
    noise_indices = tuple(
        index for index, label in enumerate(labels_array) if int(label) < 0
    )
    cluster_sizes = tuple(
        int(np.sum(labels_array == label)) for label in cluster_labels
    )
    non_noise_count = matrix.shape[0] - len(noise_indices)
    largest_share = (
        max(cluster_sizes) / non_noise_count if non_noise_count else 0.0
    )
    medoids.setflags(write=False)
    return HDBSCANClusteringResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        min_cluster_size=configuration.min_cluster_size,
        min_samples=configuration.min_samples,
        cluster_selection_method=configuration.cluster_selection_method,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        case_ids=candidate.case_ids,
        labels=tuple(int(label) for label in labels_array),
        membership_strengths=tuple(float(value) for value in memberships),
        medoid_cluster_labels=cluster_labels,
        medoids=medoids,
        noise_case_ids=tuple(candidate.case_ids[index] for index in noise_indices),
        non_noise_cluster_count=len(cluster_labels),
        non_noise_case_count=non_noise_count,
        noise_count=len(noise_indices),
        non_noise_coverage=float(non_noise_count / matrix.shape[0]),
        largest_non_noise_cluster_share=float(largest_share),
    )


def fit_native_hdbscan(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    allow_single_cluster: bool = HDBSCAN_ALLOW_SINGLE_CLUSTER,
) -> HDBSCANClusteringResult:
    """Fit deterministic sklearn HDBSCAN on the shared representation."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("HDBSCAN dataset is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    configuration = HDBSCANConfiguration(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    matrix = _validated_hdbscan_matrix(candidate)
    return _fit_hdbscan_from_matrix(
        dataset_id=dataset_id,
        candidate=candidate,
        active_learning_seed=active_learning_seed,
        configuration=configuration,
        matrix=matrix,
    )


def run_hdbscan_grid(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
) -> HDBSCANGridSearchResult:
    """Fit and retain every setting in the frozen 70-item HDBSCAN grid."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("HDBSCAN dataset is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    matrix = _validated_hdbscan_matrix(candidate)
    attempts: list[HDBSCANSearchAttempt] = []
    for configuration in hdbscan_parameter_grid():
        try:
            result = _fit_hdbscan_from_matrix(
                dataset_id=dataset_id,
                candidate=candidate,
                active_learning_seed=active_learning_seed,
                configuration=configuration,
                matrix=matrix,
            )
        except ValueError as exc:
            attempts.append(
                HDBSCANSearchAttempt(
                    configuration=configuration,
                    status="rejected",
                    result=None,
                    rejection_reason=str(exc),
                )
            )
        else:
            attempts.append(
                HDBSCANSearchAttempt(
                    configuration=configuration,
                    status="fitted",
                    result=result,
                    rejection_reason=None,
                )
            )
    return HDBSCANGridSearchResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        attempts=tuple(attempts),
    )


def _validated_mutual_knn_distance_matrix(
    candidate: RepresentationMatrixCandidate,
    metric: str,
) -> np.ndarray:
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("mutual-kNN requires a typed representation candidate")
    if (
        candidate.representation_id not in FORMAL_REPRESENTATION_IDS
        or not candidate.formal_eligible
    ):
        raise ValueError("mutual-kNN requires a formal representation candidate")
    if metric not in MUTUAL_KNN_METRICS:
        raise ValueError("mutual-kNN metric is outside the frozen grid")
    matrix = np.asarray(candidate.matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(candidate.case_ids)
        or matrix.shape[1] != len(candidate.feature_names)
        or matrix.shape[0] < 1
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("shared representation matrix is invalid for mutual-kNN")
    distances = pairwise_distances(matrix, metric=metric, n_jobs=1)
    np.fill_diagonal(distances, 0.0)
    if metric == "cosine":
        distances[np.isclose(distances, 0.0, rtol=0.0, atol=1e-12)] = 0.0
    if not np.isfinite(distances).all() or np.any(distances < -1e-12):
        raise ValueError("mutual-kNN pairwise distance matrix is invalid")
    distances = np.maximum(distances, 0.0)
    distances.setflags(write=False)
    return distances


def _fit_mutual_knn_from_distances(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    configuration: MutualKNNConfiguration,
    distance_matrix: np.ndarray,
) -> MutualKNNClusteringResult:
    population = len(candidate.case_ids)
    if configuration.neighbor_count >= population:
        raise ValueError("mutual-kNN neighbor count must be below the population")
    neighbor_indices: list[tuple[int, ...]] = []
    for index in range(population):
        ordered = np.argsort(distance_matrix[index], kind="mergesort")
        neighbors = tuple(
            int(neighbor)
            for neighbor in ordered
            if int(neighbor) != index
        )[: configuration.neighbor_count]
        neighbor_indices.append(neighbors)
    neighbor_sets = tuple(frozenset(items) for items in neighbor_indices)
    mutual_index_edges = tuple(
        (left, right)
        for left in range(population)
        for right in neighbor_indices[left]
        if left < right and left in neighbor_sets[right]
    )

    parents = list(range(population))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in mutual_index_edges:
        union(left, right)
    members_by_root: dict[int, list[int]] = {}
    for index in range(population):
        members_by_root.setdefault(find(index), []).append(index)
    ordered_components = tuple(
        sorted(members) for members in sorted(members_by_root.values(), key=min)
    )
    labels_array = np.empty(population, dtype=int)
    for label, members in enumerate(ordered_components):
        labels_array[members] = label
    degrees = [0] * population
    for left, right in mutual_index_edges:
        degrees[left] += 1
        degrees[right] += 1
    mutual_edges = tuple(
        (candidate.case_ids[left], candidate.case_ids[right])
        for left, right in mutual_index_edges
    )
    return MutualKNNClusteringResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        neighbor_count=configuration.neighbor_count,
        metric=configuration.metric,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        case_ids=candidate.case_ids,
        labels=tuple(int(label) for label in labels_array),
        neighbor_case_ids=tuple(
            tuple(candidate.case_ids[index] for index in items)
            for items in neighbor_indices
        ),
        mutual_edges=mutual_edges,
        within_component_degrees=tuple(degrees),
        component_count=len(ordered_components),
        largest_component_share=float(
            max(len(members) for members in ordered_components) / population
        ),
    )


def fit_native_mutual_knn(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    neighbor_count: int,
    metric: str,
) -> MutualKNNClusteringResult:
    """Build a deterministic mutual-kNN graph and its connected components."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("mutual-kNN dataset is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    configuration = MutualKNNConfiguration(
        neighbor_count=neighbor_count,
        metric=metric,
    )
    distances = _validated_mutual_knn_distance_matrix(candidate, metric)
    return _fit_mutual_knn_from_distances(
        dataset_id=dataset_id,
        candidate=candidate,
        active_learning_seed=active_learning_seed,
        configuration=configuration,
        distance_matrix=distances,
    )


def run_mutual_knn_grid(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
) -> MutualKNNGridSearchResult:
    """Fit and retain every setting in the frozen 20-item mutual-kNN grid."""

    if dataset_id not in KMEANS_CLUSTER_COUNTS:
        raise ValueError("mutual-kNN dataset is outside the formal protocol")
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    attempts: list[MutualKNNSearchAttempt] = []
    for metric in MUTUAL_KNN_METRICS:
        distances = _validated_mutual_knn_distance_matrix(candidate, metric)
        for neighbor_count in MUTUAL_KNN_NEIGHBOR_COUNTS:
            configuration = MutualKNNConfiguration(
                neighbor_count=neighbor_count,
                metric=metric,
            )
            try:
                result = _fit_mutual_knn_from_distances(
                    dataset_id=dataset_id,
                    candidate=candidate,
                    active_learning_seed=active_learning_seed,
                    configuration=configuration,
                    distance_matrix=distances,
                )
            except ValueError as exc:
                attempts.append(
                    MutualKNNSearchAttempt(
                        configuration=configuration,
                        status="rejected",
                        result=None,
                        rejection_reason=str(exc),
                    )
                )
            else:
                attempts.append(
                    MutualKNNSearchAttempt(
                        configuration=configuration,
                        status="fitted",
                        result=result,
                        rejection_reason=None,
                    )
                )
    return MutualKNNGridSearchResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        attempts=tuple(attempts),
    )


def fit_native_kmeans(
    *,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    cluster_count: int,
    n_init: int = KMEANS_N_INIT,
) -> KMeansClusteringResult:
    """Fit sklearn K-means directly on the shared frozen representation."""

    configuration = KMeansConfiguration(
        dataset_id=dataset_id,
        cluster_count=cluster_count,
        n_init=n_init,
    )
    if isinstance(active_learning_seed, bool) or (
        active_learning_seed not in ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("active-learning seed must be one of 41, 42, and 43")
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("K-means requires a typed representation candidate")
    if (
        candidate.representation_id not in FORMAL_REPRESENTATION_IDS
        or not candidate.formal_eligible
    ):
        raise ValueError("K-means requires a formal representation candidate")
    matrix = np.asarray(candidate.matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(candidate.case_ids)
        or matrix.shape[1] != len(candidate.feature_names)
        or matrix.shape[0] < cluster_count
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("shared representation matrix is invalid for K-means")

    estimator = KMeans(
        n_clusters=configuration.cluster_count,
        n_init=configuration.n_init,
        random_state=active_learning_seed,
        algorithm=configuration.algorithm,
    )
    # Symmetric candidate pools can contain exactly tied local optima.  The
    # frozen single-thread numerical boundary prevents OpenMP reduction order
    # from choosing different tied optima for an identical acquisition seed.
    with threadpool_limits(limits=1):
        labels_array = estimator.fit_predict(matrix)
    centers = np.asarray(estimator.cluster_centers_, dtype=float).copy()
    if (
        labels_array.shape != (matrix.shape[0],)
        or centers.shape != (cluster_count, matrix.shape[1])
        or not np.isfinite(centers).all()
        or not np.isfinite(estimator.inertia_)
        or int(estimator.n_iter_) < 1
    ):
        raise ValueError("native K-means returned invalid geometry")
    centers.setflags(write=False)
    return KMeansClusteringResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        cluster_count=configuration.cluster_count,
        n_init=configuration.n_init,
        algorithm=configuration.algorithm,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        case_ids=candidate.case_ids,
        labels=tuple(int(label) for label in labels_array),
        cluster_centers=centers,
        inertia=float(estimator.inertia_),
        n_iter=int(estimator.n_iter_),
    )


__all__ = [
    "AmendedStructuralConfigurationScore",
    "AmendedStructureCellDecision",
    "DBSCAN_EPS_QUANTILES",
    "DBSCAN_METRICS",
    "DBSCAN_MIN_SAMPLES",
    "DBSCANClusteringResult",
    "DBSCANConfiguration",
    "DBSCANEpsDerivation",
    "DBSCANGridSearchResult",
    "DBSCANSearchAttempt",
    "EFFECTIVE_CLUSTER_MIN_SUPPORT",
    "EffectiveClusterPartition",
    "EffectiveClusterRecord",
    "FORMAL_CLUSTERER_IDS",
    "FORMAL_CLUSTERER_ORDER",
    "SharedRepresentationCandidateScore",
    "SharedRepresentationSelectionResult",
    "StructuralConfigurationScore",
    "HDBSCAN_ALLOW_SINGLE_CLUSTER",
    "HDBSCAN_CLUSTER_SELECTION_METHODS",
    "HDBSCAN_METRIC",
    "HDBSCAN_MIN_CLUSTER_SIZES",
    "HDBSCAN_MIN_SAMPLES",
    "HDBSCAN_STORE_CENTERS",
    "HDBSCANClusteringResult",
    "HDBSCANConfiguration",
    "HDBSCANGridSearchResult",
    "HDBSCANSearchAttempt",
    "KMEANS_ALGORITHM",
    "KMEANS_CLUSTER_COUNTS",
    "KMEANS_N_INIT",
    "KMeansClusteringResult",
    "KMeansConfiguration",
    "MUTUAL_KNN_METRICS",
    "MUTUAL_KNN_NEIGHBOR_COUNTS",
    "MutualKNNClusteringResult",
    "MutualKNNConfiguration",
    "MutualKNNGridSearchResult",
    "MutualKNNSearchAttempt",
    "dbscan_parameter_grid",
    "build_effective_cluster_partition",
    "compute_cross_seed_stability",
    "derive_dbscan_eps",
    "evaluate_structure_gate",
    "fit_native_dbscan",
    "fit_native_hdbscan",
    "fit_native_kmeans",
    "fit_native_mutual_knn",
    "hdbscan_parameter_grid",
    "kmeans_parameter_grid",
    "mutual_knn_parameter_grid",
    "rank_structural_configurations",
    "rank_amended_structural_configurations",
    "run_dbscan_grid",
    "run_hdbscan_grid",
    "run_mutual_knn_grid",
    "select_shared_representation",
    "select_amended_structure_cell",
]
