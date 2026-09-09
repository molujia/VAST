"""Semi-supervised window clustering, pseudo labeling, and unified ranking."""

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from nexusrcl_rebuild.features.entities import load_entity_index
from nexusrcl_rebuild.features.topology import load_topology_bundle
from nexusrcl_rebuild.pseudo_labeling.contracts import StrategyResult
from nexusrcl_rebuild.pseudo_labeling.supervision import validate_supervision_request
from nexusrcl_rebuild.training.pseudo_adapter import (
    PseudoTrainingAdapterConfig,
    adapt_matched_pseudo_training_rows,
    adapt_pseudo_training_rows,
    attach_frozen_pseudo_result,
)


WINDOW_META_COLUMNS = [
    "dataset",
    "window_id",
    "source_id",
    "window_kind",
    "day",
    "start_ts",
    "end_ts",
]

ENTITY_META_COLUMNS = WINDOW_META_COLUMNS + [
    "entity_id",
    "entity_index",
    "entity_type",
    "entity_name",
    "is_positive",
]

ANOMALY_FEATURE_HINTS = [
    "log_error_count",
    "log_error_ratio",
    "metric_event_score_sum",
    "metric_event_score_max",
    "metric_event_active_kpi_count",
    "metric_abs_z_max",
    "metric_anomalous_kpi_count",
    "trace_error_count",
    "trace_error_ratio",
    "trace_latency_abs_z_max",
    "trace_anomalous_operation_count",
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

ANOMALY_SIGNATURE_WEIGHTS = {
    "log_error_count": 0.75,
    "log_error_ratio": 0.50,
    "metric_event_score_sum": 1.50,
    "metric_event_score_max": 1.00,
    "metric_event_active_kpi_count": 0.75,
    "metric_abs_z_max": 1.00,
    "metric_anomalous_kpi_count": 0.75,
    "trace_error_count": 0.75,
    "trace_error_ratio": 0.75,
    "trace_latency_abs_z_max": 1.00,
    "trace_anomalous_operation_count": 1.00,
    "trace_client_error_ratio": 0.50,
    "trace_server_error_count": 0.75,
    "trace_server_error_ratio": 0.75,
    "trace_client_latency_abs_z_max": 0.75,
    "trace_server_latency_abs_z_mean": 0.75,
    "trace_server_latency_abs_z_max": 1.00,
    "trace_client_anomalous_operation_count": 0.50,
    "trace_server_anomalous_operation_count": 1.00,
    "topology_change_count": 0.50,
}

ANOMALY_PREFIX_WEIGHTS = {
    "metric_kpi_peak_": 1.25,
    "metric_kpi_hit_": 0.50,
    "trace_operation_z_": 1.00,
    "trace_peer_share_": 0.35,
}


@dataclass(frozen=True)
class SemiSupervisedConfig:
    """Experiment-facing knobs for clustering, propagation, and fusion."""

    model_backend: str = "pairwise_linear"
    cluster_mode: str = "dbscan"
    fixed_cluster_count: int = 3
    max_clusters: int = 8
    pca_components: int = 16
    dbscan_min_samples: int = 4
    dbscan_eps_quantiles: Sequence[float] = (0.6, 0.7, 0.8, 0.9)
    cluster_representation: str = "feature_aggregate"
    supervision_mode: str = "auto"
    normal_training_policy: str = "with_normal_class"
    query_strategy: str = "sequential_medoid_noise_boundary"
    propagate_mode: str = "cluster_medoid"
    restrict_neighbors_to_cluster: bool = True
    neighbor_count: int = 5
    multi_positive_margin: float = 0.85
    pseudo_min_confidence: float = 0.0
    pseudo_positive_weight: float = 0.5
    pseudo_negative_weight: float = 0.2
    auto_normal_weight: float = 0.35
    use_topology_features: bool = True
    use_log_features: bool = True
    use_metric_features: bool = True
    use_trace_features: bool = True
    use_summary_features: bool = True
    use_graph_diffusion: bool = False
    use_graph_context_features: bool = False
    gated_onehop_relations: Sequence[str] = ()
    use_edge_refinement: bool = True
    use_window_relative_features: bool = True
    use_entity_baseline_features: bool = False
    entity_baseline_feature_mode: str = "zscore_center"
    entity_baseline_min_std: float = 0.05
    diffusion_alpha: float = 0.75
    diffusion_steps: int = 1
    label_faults_in_normal_cluster: bool = False
    query_fault_only: bool = True
    query_include_normal_cluster_faults: bool = True
    pseudo_fault_only: bool = True
    pseudo_include_normal_cluster_faults: bool = True
    pseudo_cluster_consensus_min: float = 0.0
    pseudo_cluster_core_fraction: float = 1.0
    pseudo_cluster_min_core: int = 1
    pseudo_cluster_max_core: int = 0
    pseudo_windows_per_queried: int = 0
    pseudo_entity_query_multiplier: int = 0
    pseudo_entity_min_cap: int = 1
    pseudo_negative_mode: str = "all"
    pseudo_negative_exclusion_top_k: int = 0
    pseudo_consensus_negative_bottom_k: int = 2
    pseudo_consensus_negative_min_agreement: int = 2
    pseudo_selftrain_min_score: float = 0.70
    pseudo_selftrain_min_margin: float = 0.08
    pseudo_selftrain_global_cap: int = 0
    pairwise_negative_top_k: int = 0
    entity_score_calibration_mode: str = "none"
    entity_score_calibration_min_std: float = 0.05
    entity_score_calibration_center_alpha: float = 1.0
    entity_score_calibration_host_positive_max_ratio: float = 0.05
    prototype_rerank_mode: str = "none"
    prototype_rerank_alpha: float = 0.0
    prototype_rerank_same_entity_weight: float = 0.7
    prototype_rerank_same_type_weight: float = 0.3
    prototype_rerank_host_positive_max_ratio: float = 0.05
    edge_refinement_scale: float = 1.0
    edge_refinement_corr_threshold: float = 0.10
    edge_refinement_min_support: int = 5
    cluster_pretrain_epochs: int = 80
    cluster_pretrain_patience: int = 10
    cluster_pretrain_lr: float = 1e-3
    cluster_pretrain_weight_decay: float = 1e-4
    hgcn_hidden_dim: int = 64
    hgcn_layers: int = 2
    hgcn_dropout: float = 0.10
    hgcn_epochs: int = 80
    hgcn_patience: int = 10
    hgcn_lr: float = 1e-3
    hgcn_weight_decay: float = 1e-4
    hgcn_batch_size: int = 32
    hgcn_device: str = "auto"
    supervised_pairwise_loss_weight: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "SemiSupervisedConfig":
        if not payload:
            return cls()
        values = dict(payload)
        return cls(
            model_backend=str(values.get("model_backend", cls.model_backend)),
            cluster_mode=str(values.get("cluster_mode", cls.cluster_mode)),
            fixed_cluster_count=int(values.get("fixed_cluster_count", cls.fixed_cluster_count)),
            max_clusters=int(values.get("max_clusters", cls.max_clusters)),
            pca_components=int(values.get("pca_components", cls.pca_components)),
            dbscan_min_samples=int(values.get("dbscan_min_samples", cls.dbscan_min_samples)),
            dbscan_eps_quantiles=tuple(
                float(item)
                for item in values.get("dbscan_eps_quantiles", cls.dbscan_eps_quantiles)
            ),
            cluster_representation=str(
                values.get("cluster_representation", cls.cluster_representation)
            ),
            supervision_mode=str(values.get("supervision_mode", cls.supervision_mode)),
            normal_training_policy=str(
                values.get(
                    "normal_training_policy",
                    cls.normal_training_policy,
                )
            ),
            query_strategy=str(values.get("query_strategy", cls.query_strategy)),
            propagate_mode=str(values.get("propagate_mode", cls.propagate_mode)),
            restrict_neighbors_to_cluster=bool(
                values.get("restrict_neighbors_to_cluster", cls.restrict_neighbors_to_cluster)
            ),
            neighbor_count=int(values.get("neighbor_count", cls.neighbor_count)),
            multi_positive_margin=float(
                values.get("multi_positive_margin", cls.multi_positive_margin)
            ),
            pseudo_min_confidence=float(
                values.get("pseudo_min_confidence", cls.pseudo_min_confidence)
            ),
            pseudo_positive_weight=float(
                values.get("pseudo_positive_weight", cls.pseudo_positive_weight)
            ),
            pseudo_negative_weight=float(
                values.get("pseudo_negative_weight", cls.pseudo_negative_weight)
            ),
            auto_normal_weight=float(
                values.get("auto_normal_weight", cls.auto_normal_weight)
            ),
            use_topology_features=bool(
                values.get("use_topology_features", cls.use_topology_features)
            ),
            use_log_features=bool(values.get("use_log_features", cls.use_log_features)),
            use_metric_features=bool(values.get("use_metric_features", cls.use_metric_features)),
            use_trace_features=bool(values.get("use_trace_features", cls.use_trace_features)),
            use_summary_features=bool(
                values.get("use_summary_features", cls.use_summary_features)
            ),
            use_graph_diffusion=bool(
                values.get("use_graph_diffusion", cls.use_graph_diffusion)
            ),
            use_graph_context_features=bool(
                values.get(
                    "use_graph_context_features",
                    cls.use_graph_context_features,
                )
            ),
            gated_onehop_relations=tuple(
                str(item)
                for item in values.get(
                    "gated_onehop_relations",
                    cls.gated_onehop_relations,
                )
            ),
            use_edge_refinement=bool(
                values.get("use_edge_refinement", cls.use_edge_refinement)
            ),
            use_window_relative_features=bool(
                values.get("use_window_relative_features", cls.use_window_relative_features)
            ),
            use_entity_baseline_features=bool(
                values.get(
                    "use_entity_baseline_features",
                    cls.use_entity_baseline_features,
                )
            ),
            entity_baseline_feature_mode=str(
                values.get(
                    "entity_baseline_feature_mode",
                    cls.entity_baseline_feature_mode,
                )
            ),
            entity_baseline_min_std=float(
                values.get(
                    "entity_baseline_min_std",
                    cls.entity_baseline_min_std,
                )
            ),
            diffusion_alpha=float(values.get("diffusion_alpha", cls.diffusion_alpha)),
            diffusion_steps=int(values.get("diffusion_steps", cls.diffusion_steps)),
            label_faults_in_normal_cluster=bool(
                values.get("label_faults_in_normal_cluster", cls.label_faults_in_normal_cluster)
            ),
            query_fault_only=bool(
                values.get("query_fault_only", cls.query_fault_only)
            ),
            query_include_normal_cluster_faults=bool(
                values.get(
                    "query_include_normal_cluster_faults",
                    cls.query_include_normal_cluster_faults,
                )
            ),
            pseudo_fault_only=bool(
                values.get("pseudo_fault_only", cls.pseudo_fault_only)
            ),
            pseudo_include_normal_cluster_faults=bool(
                values.get(
                    "pseudo_include_normal_cluster_faults",
                    cls.pseudo_include_normal_cluster_faults,
                )
            ),
            pseudo_cluster_consensus_min=float(
                values.get(
                    "pseudo_cluster_consensus_min",
                    cls.pseudo_cluster_consensus_min,
                )
            ),
            pseudo_cluster_core_fraction=float(
                values.get(
                    "pseudo_cluster_core_fraction",
                    cls.pseudo_cluster_core_fraction,
                )
            ),
            pseudo_cluster_min_core=int(
                values.get(
                    "pseudo_cluster_min_core",
                    cls.pseudo_cluster_min_core,
                )
            ),
            pseudo_cluster_max_core=int(
                values.get(
                    "pseudo_cluster_max_core",
                    cls.pseudo_cluster_max_core,
                )
            ),
            pseudo_windows_per_queried=int(
                values.get(
                    "pseudo_windows_per_queried",
                    cls.pseudo_windows_per_queried,
                )
            ),
            pseudo_entity_query_multiplier=int(
                values.get(
                    "pseudo_entity_query_multiplier",
                    cls.pseudo_entity_query_multiplier,
                )
            ),
            pseudo_entity_min_cap=int(
                values.get(
                    "pseudo_entity_min_cap",
                    cls.pseudo_entity_min_cap,
                )
            ),
            pseudo_negative_mode=str(
                values.get(
                    "pseudo_negative_mode",
                    cls.pseudo_negative_mode,
                )
            ),
            pseudo_negative_exclusion_top_k=int(
                values.get(
                    "pseudo_negative_exclusion_top_k",
                    cls.pseudo_negative_exclusion_top_k,
                )
            ),
            pseudo_consensus_negative_bottom_k=int(
                values.get(
                    "pseudo_consensus_negative_bottom_k",
                    cls.pseudo_consensus_negative_bottom_k,
                )
            ),
            pseudo_consensus_negative_min_agreement=int(
                values.get(
                    "pseudo_consensus_negative_min_agreement",
                    cls.pseudo_consensus_negative_min_agreement,
                )
            ),
            pseudo_selftrain_min_score=float(
                values.get(
                    "pseudo_selftrain_min_score",
                    cls.pseudo_selftrain_min_score,
                )
            ),
            pseudo_selftrain_min_margin=float(
                values.get(
                    "pseudo_selftrain_min_margin",
                    cls.pseudo_selftrain_min_margin,
                )
            ),
            pseudo_selftrain_global_cap=int(
                values.get(
                    "pseudo_selftrain_global_cap",
                    cls.pseudo_selftrain_global_cap,
                )
            ),
            pairwise_negative_top_k=int(
                values.get(
                    "pairwise_negative_top_k",
                    cls.pairwise_negative_top_k,
                )
            ),
            entity_score_calibration_mode=str(
                values.get(
                    "entity_score_calibration_mode",
                    cls.entity_score_calibration_mode,
                )
            ),
            entity_score_calibration_min_std=float(
                values.get(
                    "entity_score_calibration_min_std",
                    cls.entity_score_calibration_min_std,
                )
            ),
            entity_score_calibration_center_alpha=float(
                values.get(
                    "entity_score_calibration_center_alpha",
                    cls.entity_score_calibration_center_alpha,
                )
            ),
            entity_score_calibration_host_positive_max_ratio=float(
                values.get(
                    "entity_score_calibration_host_positive_max_ratio",
                    cls.entity_score_calibration_host_positive_max_ratio,
                )
            ),
            prototype_rerank_mode=str(
                values.get(
                    "prototype_rerank_mode",
                    cls.prototype_rerank_mode,
                )
            ),
            prototype_rerank_alpha=float(
                values.get(
                    "prototype_rerank_alpha",
                    cls.prototype_rerank_alpha,
                )
            ),
            prototype_rerank_same_entity_weight=float(
                values.get(
                    "prototype_rerank_same_entity_weight",
                    cls.prototype_rerank_same_entity_weight,
                )
            ),
            prototype_rerank_same_type_weight=float(
                values.get(
                    "prototype_rerank_same_type_weight",
                    cls.prototype_rerank_same_type_weight,
                )
            ),
            prototype_rerank_host_positive_max_ratio=float(
                values.get(
                    "prototype_rerank_host_positive_max_ratio",
                    cls.prototype_rerank_host_positive_max_ratio,
                )
            ),
            edge_refinement_scale=float(
                values.get("edge_refinement_scale", cls.edge_refinement_scale)
            ),
            edge_refinement_corr_threshold=float(
                values.get(
                    "edge_refinement_corr_threshold",
                    cls.edge_refinement_corr_threshold,
                )
            ),
            edge_refinement_min_support=int(
                values.get(
                    "edge_refinement_min_support",
                    cls.edge_refinement_min_support,
                )
            ),
            cluster_pretrain_epochs=int(
                values.get("cluster_pretrain_epochs", cls.cluster_pretrain_epochs)
            ),
            cluster_pretrain_patience=int(
                values.get("cluster_pretrain_patience", cls.cluster_pretrain_patience)
            ),
            cluster_pretrain_lr=float(
                values.get("cluster_pretrain_lr", cls.cluster_pretrain_lr)
            ),
            cluster_pretrain_weight_decay=float(
                values.get(
                    "cluster_pretrain_weight_decay",
                    cls.cluster_pretrain_weight_decay,
                )
            ),
            hgcn_hidden_dim=int(values.get("hgcn_hidden_dim", cls.hgcn_hidden_dim)),
            hgcn_layers=int(values.get("hgcn_layers", cls.hgcn_layers)),
            hgcn_dropout=float(values.get("hgcn_dropout", cls.hgcn_dropout)),
            hgcn_epochs=int(values.get("hgcn_epochs", cls.hgcn_epochs)),
            hgcn_patience=int(values.get("hgcn_patience", cls.hgcn_patience)),
            hgcn_lr=float(values.get("hgcn_lr", cls.hgcn_lr)),
            hgcn_weight_decay=float(values.get("hgcn_weight_decay", cls.hgcn_weight_decay)),
            hgcn_batch_size=int(values.get("hgcn_batch_size", cls.hgcn_batch_size)),
            hgcn_device=str(values.get("hgcn_device", cls.hgcn_device)),
            supervised_pairwise_loss_weight=float(
                values.get(
                    "supervised_pairwise_loss_weight",
                    cls.supervised_pairwise_loss_weight,
                )
            ),
        )


@dataclass(frozen=True)
class FeatureBundleTables:
    dataset: str
    windows: pd.DataFrame
    entity_features: pd.DataFrame
    metadata: Mapping[str, Any]
    feature_columns: Sequence[str]


@dataclass(frozen=True)
class QueryPlan:
    dataset: str
    normal_cluster_id: int
    window_clusters: Mapping[str, int]
    queried_window_ids: Sequence[str]
    queried_roles: Mapping[str, str]
    queried_labels: Mapping[str, Sequence[str]]
    pseudo_labels: Mapping[str, Sequence[str]]
    pseudo_confidence: Mapping[str, float]
    metadata: Mapping[str, Any]
    frozen_pseudo_result: Optional[StrategyResult] = None
    pseudo_candidate_rankings: Mapping[
        str, Mapping[str, Sequence[str]]
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class PrototypeRerankState:
    feature_columns: Sequence[str]
    center: np.ndarray
    scale: np.ndarray
    entity_positive: Mapping[str, np.ndarray]
    entity_negative: Mapping[str, np.ndarray]
    type_positive: Mapping[str, np.ndarray]
    type_negative: Mapping[str, np.ndarray]


@dataclass
class SemiSupervisedModel:
    dataset: str
    feature_columns: Sequence[str]
    base_feature_columns: Sequence[str]
    classifier: Any
    adjacency: Optional[np.ndarray]
    diffusion_alpha: float
    diffusion_steps: int
    query_plan: QueryPlan
    config: SemiSupervisedConfig
    entity_feature_baselines: Optional[pd.DataFrame] = None
    entity_score_calibration: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    score_calibration_mode: str = "none"
    prototype_rerank_state: Optional[PrototypeRerankState] = None
    prototype_rerank_mode: str = "none"
    graph_context_bundle: Optional[Any] = None
    gated_onehop_topology: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    training_frame: Optional[pd.DataFrame] = None

    def _score_prepared_entity_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        missing_features = sorted(set(self.feature_columns).difference(rows.columns))
        if missing_features:
            raise ValueError(
                "prepared entity features are missing model columns: %s"
                % missing_features[:10]
            )
        if rows.empty:
            rows["raw_score"] = pd.Series(dtype=float)
            rows["score"] = pd.Series(dtype=float)
            return rows
        score_frame = getattr(self.classifier, "score_frame", None)
        if callable(score_frame):
            rows = score_frame(rows)
            rows = _apply_entity_score_calibration(
                rows=rows,
                calibration=self.entity_score_calibration,
                mode=self.score_calibration_mode,
                center_alpha=float(self.config.entity_score_calibration_center_alpha),
            )
            return _apply_prototype_rerank(
                rows=rows,
                state=self.prototype_rerank_state,
                mode=self.prototype_rerank_mode,
                alpha=float(self.config.prototype_rerank_alpha),
                same_entity_weight=float(self.config.prototype_rerank_same_entity_weight),
                same_type_weight=float(self.config.prototype_rerank_same_type_weight),
            )
        probabilities = self.classifier.predict_proba(rows[list(self.feature_columns)].values)
        if probabilities.shape[1] == 1:
            class_value = int(self.classifier.classes_[0])
            raw_scores = probabilities[:, 0] if class_value == 1 else np.zeros(len(rows), dtype=float)
        else:
            positive_index = int(np.where(self.classifier.classes_ == 1)[0][0])
            raw_scores = probabilities[:, positive_index]
        rows["raw_score"] = raw_scores
        rows["score"] = raw_scores
        if self.adjacency is not None and self.diffusion_steps > 0:
            for window_id, group in rows.groupby("window_id", sort=False):
                indices = group["entity_index"].astype(int).to_numpy()
                scores = group["raw_score"].to_numpy(dtype=float)
                dense = np.zeros(self.adjacency.shape[0], dtype=float)
                dense[indices] = scores
                propagated = dense.copy()
                for _ in range(self.diffusion_steps):
                    propagated = (
                        self.diffusion_alpha * dense
                        + (1.0 - self.diffusion_alpha) * self.adjacency.dot(propagated)
                    )
                rows.loc[group.index, "score"] = propagated[indices]
        rows = _apply_entity_score_calibration(
            rows=rows,
            calibration=self.entity_score_calibration,
            mode=self.score_calibration_mode,
            center_alpha=float(self.config.entity_score_calibration_center_alpha),
        )
        return _apply_prototype_rerank(
            rows=rows,
            state=self.prototype_rerank_state,
            mode=self.prototype_rerank_mode,
            alpha=float(self.config.prototype_rerank_alpha),
            same_entity_weight=float(self.config.prototype_rerank_same_entity_weight),
            same_type_weight=float(self.config.prototype_rerank_same_type_weight),
        )

    def score_prepared_entity_features(self, entity_features: pd.DataFrame) -> pd.DataFrame:
        """Score rows that already contain the model's prepared feature columns."""
        return self._score_prepared_entity_features(entity_features)

    def score_entity_features(self, entity_features: pd.DataFrame) -> pd.DataFrame:
        rows = prepare_entity_features_for_model(
            entity_features=entity_features.copy(),
            base_feature_columns=self.base_feature_columns,
            config=self.config,
            entity_feature_baselines=self.entity_feature_baselines,
        )
        gated_relations = tuple(
            str(value)
            for value in getattr(
                self.config,
                "gated_onehop_relations",
                (),
            )
        )
        if gated_relations:
            if self.gated_onehop_topology is None:
                raise ValueError(
                    "gated one-hop scoring requires the frozen topology"
                )
            pre_gated_columns = [
                column
                for column in self.feature_columns
                if "gated_onehop_" not in str(column)
                and "graph_ctx_" not in str(column)
            ]
            gated_tables = augment_prepared_feature_bundle_with_gated_onehop(
                tables=FeatureBundleTables(
                    dataset=self.dataset,
                    windows=pd.DataFrame(),
                    entity_features=rows,
                    metadata={},
                    feature_columns=pre_gated_columns,
                ),
                topology=self.gated_onehop_topology,
                source_feature_columns=self.base_feature_columns,
                config=self.config,
            )
            rows = gated_tables.entity_features
        if (
            self.graph_context_bundle is not None
            and bool(getattr(self.config, "use_graph_context_features", False))
        ):
            rows, _feature_columns = _augment_prepared_entity_features_with_graph_context(
                entity_features=rows,
                feature_columns=[
                    column
                    for column in self.feature_columns
                    if "graph_ctx_" not in str(column)
                ],
                graph_context_bundle=self.graph_context_bundle,
                config=self.config,
            )
        return self._score_prepared_entity_features(rows)


class ConstantProbabilityClassifier:
    """Predict a constant probability when the training labels collapse to one class."""

    def __init__(self, probability: float):
        self.probability = float(probability)
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = np.clip(self.probability, 0.0, 1.0)
        negative = np.full((features.shape[0], 1), 1.0 - probability, dtype=float)
        positive = np.full((features.shape[0], 1), probability, dtype=float)
        return np.concatenate([negative, positive], axis=1)


def load_feature_bundle_tables(feature_root: Path, dataset: str) -> FeatureBundleTables:
    dataset_root = feature_root / dataset
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(dataset_root / "windows.csv")
    entity_features = pd.read_csv(dataset_root / "entity_features.csv")
    windows["positive_ids_list"] = windows["positive_ids"].fillna("").map(
        lambda value: [item for item in str(value).split(";") if item]
    )
    feature_columns = list(metadata.get("all_feature_columns") or [])
    if not feature_columns:
        feature_columns = [
            column
            for column in entity_features.columns
            if column not in ENTITY_META_COLUMNS
        ]
    return FeatureBundleTables(
        dataset=dataset,
        windows=windows,
        entity_features=entity_features,
        metadata=metadata,
        feature_columns=feature_columns,
    )


def subset_feature_bundle_tables(
    tables: FeatureBundleTables,
    window_ids: Sequence[str],
) -> FeatureBundleTables:
    selected = set(str(item) for item in window_ids)
    windows = tables.windows[tables.windows["window_id"].astype(str).isin(selected)].copy()
    entity_features = tables.entity_features[
        tables.entity_features["window_id"].astype(str).isin(selected)
    ].copy()
    return FeatureBundleTables(
        dataset=tables.dataset,
        windows=windows,
        entity_features=entity_features,
        metadata=dict(tables.metadata),
        feature_columns=list(tables.feature_columns),
    )


def _feature_group_for_column(column: str) -> str:
    if column.startswith("entity_is_") or column.startswith("topo_"):
        return "topology"
    if column.startswith("log_"):
        return "log"
    if column.startswith("metric_"):
        return "metric"
    if column.startswith("trace_"):
        return "trace"
    if column.startswith("has_") or column == "modalities_present_count":
        return "summary"
    return "other"


def resolve_feature_columns(
    feature_columns: Sequence[str],
    config: Optional[SemiSupervisedConfig] = None,
) -> List[str]:
    config = config or SemiSupervisedConfig()
    selected = []
    for column in feature_columns:
        group = _feature_group_for_column(column)
        if group == "topology" and not config.use_topology_features:
            continue
        if group == "log" and not config.use_log_features:
            continue
        if group == "metric" and not config.use_metric_features:
            continue
        if group == "trace" and not config.use_trace_features:
            continue
        if group == "summary" and not config.use_summary_features:
            continue
        selected.append(column)
    if not selected:
        raise ValueError("No feature columns remain after applying the semi-supervised config.")
    return selected


def _baseline_feature_columns(
    feature_columns: Sequence[str],
) -> List[str]:
    return [
        str(column)
        for column in feature_columns
        if not str(column).startswith(("entity_is_", "topo_"))
    ]


def build_entity_feature_baselines(
    entity_features: pd.DataFrame,
    base_feature_columns: Sequence[str],
    config: Optional[SemiSupervisedConfig] = None,
) -> Optional[pd.DataFrame]:
    config = config or SemiSupervisedConfig()
    if not bool(config.use_entity_baseline_features):
        return None
    baseline_columns = _baseline_feature_columns(base_feature_columns)
    if entity_features.empty or not baseline_columns:
        return None
    normal_rows = entity_features[entity_features["window_kind"] == "normal"].copy()
    if normal_rows.empty:
        return None

    grouped = (
        normal_rows.groupby("entity_id", sort=False)[baseline_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    flattened = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            base_name, statistic = column
            if not statistic:
                flattened.append(str(base_name))
            else:
                flattened.append("entity_baseline_%s_%s" % (statistic, base_name))
        else:
            flattened.append(str(column))
    grouped.columns = flattened
    return grouped


def prepare_entity_features_for_model(
    entity_features: pd.DataFrame,
    base_feature_columns: Sequence[str],
    config: Optional[SemiSupervisedConfig] = None,
    entity_feature_baselines: Optional[pd.DataFrame] = None,
    return_feature_columns: bool = False,
) -> pd.DataFrame:
    config = config or SemiSupervisedConfig()
    rows = entity_features.copy()
    feature_columns = list(base_feature_columns)
    if rows.empty or not feature_columns:
        if return_feature_columns:
            return rows, feature_columns
        return rows

    baseline_feature_columns: List[str] = []
    baseline_mode = str(getattr(config, "entity_baseline_feature_mode", "none")).lower()
    if (
        bool(getattr(config, "use_entity_baseline_features", False))
        and entity_feature_baselines is not None
        and not entity_feature_baselines.empty
        and baseline_mode != "none"
    ):
        rows = rows.merge(entity_feature_baselines, on="entity_id", how="left")
        baseline_stat_columns: List[str] = []
        baseline_extra_columns: Dict[str, pd.Series] = {}
        min_std = float(getattr(config, "entity_baseline_min_std", 0.05))
        for column in _baseline_feature_columns(feature_columns):
            mean_column = "entity_baseline_mean_%s" % column
            std_column = "entity_baseline_std_%s" % column
            if mean_column not in rows.columns or std_column not in rows.columns:
                continue
            baseline_stat_columns.extend([mean_column, std_column])
            mean_values = rows[mean_column].fillna(0.0).astype(float)
            std_values = rows[std_column].fillna(0.0).astype(float).clip(lower=min_std)
            if baseline_mode in ("center", "zscore_center", "center_zscore"):
                center_column = "entity_base_center_%s" % column
                baseline_extra_columns[center_column] = rows[column].astype(float) - mean_values
                baseline_feature_columns.append(center_column)
            if baseline_mode in ("zscore", "zscore_center", "center_zscore"):
                zscore_column = "entity_base_z_%s" % column
                baseline_extra_columns[zscore_column] = (
                    rows[column].astype(float) - mean_values
                ) / std_values
                baseline_feature_columns.append(zscore_column)
        if baseline_extra_columns:
            rows = pd.concat(
                [rows, pd.DataFrame(baseline_extra_columns, index=rows.index)],
                axis=1,
            )
        if baseline_stat_columns:
            rows = rows.drop(columns=sorted(set(baseline_stat_columns)))
        feature_columns.extend(baseline_feature_columns)
    if not bool(config.use_window_relative_features):
        if return_feature_columns:
            return rows, feature_columns
        return rows

    window_groups = rows.groupby("window_id", sort=False)
    type_groups = rows.groupby(["window_id", "entity_type"], sort=False)
    extra_columns = {}
    for column in feature_columns:
        mean_values = window_groups[column].transform("mean")
        std_values = window_groups[column].transform("std").fillna(0.0)
        std_values = std_values.replace(0.0, 1.0)
        extra_columns["rel_win_z_%s" % column] = (rows[column] - mean_values) / std_values
        extra_columns["rel_win_rank_%s" % column] = window_groups[column].rank(
            method="average",
            pct=True,
            ascending=False,
        )
        extra_columns["rel_type_rank_%s" % column] = type_groups[column].rank(
            method="average",
            pct=True,
            ascending=False,
        )
    if not extra_columns:
        if return_feature_columns:
            return rows, feature_columns
        return rows
    output = pd.concat([rows, pd.DataFrame(extra_columns, index=rows.index)], axis=1)
    final_feature_columns = list(feature_columns)
    for column in feature_columns:
        final_feature_columns.extend(
            [
                "rel_win_z_%s" % column,
                "rel_win_rank_%s" % column,
                "rel_type_rank_%s" % column,
            ]
        )
    if return_feature_columns:
        return output, final_feature_columns
    return output


def prepare_feature_bundle_tables_for_model(
    tables: FeatureBundleTables,
    base_feature_columns: Sequence[str],
    config: Optional[SemiSupervisedConfig] = None,
    entity_feature_baselines: Optional[pd.DataFrame] = None,
) -> FeatureBundleTables:
    entity_features, feature_columns = prepare_entity_features_for_model(
        entity_features=tables.entity_features,
        base_feature_columns=base_feature_columns,
        config=config,
        entity_feature_baselines=entity_feature_baselines,
        return_feature_columns=True,
    )
    return FeatureBundleTables(
        dataset=tables.dataset,
        windows=tables.windows.copy(),
        entity_features=entity_features,
        metadata=dict(tables.metadata),
        feature_columns=feature_columns,
    )


def _merge_graph_context_columns(
    entity_features: pd.DataFrame,
    graph_context: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    if entity_features.empty or graph_context.empty:
        return entity_features.copy(), []
    merge_keys = ["window_id", "entity_id"]
    graph_columns = [
        str(column)
        for column in graph_context.columns
        if str(column) not in merge_keys
    ]
    if not graph_columns:
        return entity_features.copy(), []
    merged = entity_features.merge(
        graph_context[merge_keys + graph_columns],
        on=merge_keys,
        how="left",
    )
    for column in graph_columns:
        merged[column] = merged[column].fillna(0.0).astype(float)
    return merged, graph_columns


def _append_window_relative_columns(
    entity_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    rows = entity_features.copy()
    extra_columns = {}
    final_feature_columns = list(feature_columns)
    if rows.empty or not feature_columns:
        return rows, final_feature_columns
    window_groups = rows.groupby("window_id", sort=False)
    type_groups = rows.groupby(["window_id", "entity_type"], sort=False)
    for column in feature_columns:
        rel_z_column = "rel_win_z_%s" % column
        rel_rank_column = "rel_win_rank_%s" % column
        rel_type_rank_column = "rel_type_rank_%s" % column
        if (
            rel_z_column in rows.columns
            and rel_rank_column in rows.columns
            and rel_type_rank_column in rows.columns
        ):
            for derived_column in (
                rel_z_column,
                rel_rank_column,
                rel_type_rank_column,
            ):
                if derived_column not in final_feature_columns:
                    final_feature_columns.append(derived_column)
            continue
        mean_values = window_groups[column].transform("mean")
        std_values = window_groups[column].transform("std").fillna(0.0)
        std_values = std_values.replace(0.0, 1.0)
        extra_columns[rel_z_column] = (rows[column] - mean_values) / std_values
        extra_columns[rel_rank_column] = window_groups[column].rank(
            method="average",
            pct=True,
            ascending=False,
        )
        extra_columns[rel_type_rank_column] = type_groups[column].rank(
            method="average",
            pct=True,
            ascending=False,
        )
        for derived_column in (
            rel_z_column,
            rel_rank_column,
            rel_type_rank_column,
        ):
            if derived_column not in final_feature_columns:
                final_feature_columns.append(derived_column)
    if extra_columns:
        rows = pd.concat([rows, pd.DataFrame(extra_columns, index=rows.index)], axis=1)
    return rows, final_feature_columns


def augment_prepared_feature_bundle_with_gated_onehop(
    tables: FeatureBundleTables,
    topology: Any,
    source_feature_columns: Sequence[str],
    config: Optional[SemiSupervisedConfig] = None,
) -> FeatureBundleTables:
    config = config or SemiSupervisedConfig()
    from .gated_onehop import augment_entity_features_with_gated_onehop

    configured_relations = tuple(
        str(value)
        for value in getattr(
            config,
            "gated_onehop_relations",
            (),
        )
    )
    service_service_edges = topology.service_service_edges
    topology_projection = None
    if configured_relations:
        service_rows = tables.entity_features[
            tables.entity_features["entity_type"].astype(str) == "service"
        ]
        known_services = set(
            service_rows["entity_name"].astype(str)
        )
        projected_edges = {
            (str(source), str(target)): weight
            for (source, target), weight in service_service_edges.items()
            if str(source) in known_services
            and str(target) in known_services
        }
        unknown_services = sorted(
            {
                str(service)
                for edge in service_service_edges
                for service in edge
                if str(service) not in known_services
            }
        )
        topology_projection = {
            "mode": "known_feature_service_induced_subgraph",
            "source_edge_count": len(service_service_edges),
            "retained_edge_count": len(projected_edges),
            "dropped_edge_count": (
                len(service_service_edges) - len(projected_edges)
            ),
            "unknown_service_count": len(unknown_services),
            "unknown_services": unknown_services,
        }
        service_service_edges = projected_edges
    result = augment_entity_features_with_gated_onehop(
        tables.entity_features,
        feature_columns=source_feature_columns,
        service_service_edges=service_service_edges,
        relations=configured_relations,
    )
    entity_features = result.frame
    feature_columns = list(tables.feature_columns)
    for column in result.feature_columns:
        if column not in feature_columns:
            feature_columns.append(column)
    if (
        result.feature_columns
        and bool(config.use_window_relative_features)
    ):
        entity_features, derived_columns = _append_window_relative_columns(
            entity_features=entity_features,
            feature_columns=result.feature_columns,
        )
        for column in derived_columns:
            if column not in feature_columns:
                feature_columns.append(column)
    metadata = dict(tables.metadata)
    gated_onehop_metadata = dict(result.diagnostics)
    if topology_projection is not None:
        gated_onehop_metadata["topology_projection"] = topology_projection
    metadata["gated_onehop"] = gated_onehop_metadata
    return FeatureBundleTables(
        dataset=tables.dataset,
        windows=tables.windows.copy(),
        entity_features=entity_features,
        metadata=metadata,
        feature_columns=feature_columns,
    )


def augment_prepared_feature_bundle_with_graph_context(
    tables: FeatureBundleTables,
    graph_context: pd.DataFrame,
    config: Optional[SemiSupervisedConfig] = None,
) -> FeatureBundleTables:
    config = config or SemiSupervisedConfig()
    entity_features, graph_columns = _merge_graph_context_columns(
        entity_features=tables.entity_features,
        graph_context=graph_context,
    )
    feature_columns = list(tables.feature_columns)
    if graph_columns:
        for graph_column in graph_columns:
            if graph_column not in feature_columns:
                feature_columns.append(graph_column)
        if bool(config.use_window_relative_features):
            entity_features, feature_columns = _append_window_relative_columns(
                entity_features=entity_features,
                feature_columns=graph_columns,
            )
    return FeatureBundleTables(
        dataset=tables.dataset,
        windows=tables.windows.copy(),
        entity_features=entity_features,
        metadata=dict(tables.metadata),
        feature_columns=feature_columns,
    )


def _augment_prepared_entity_features_with_graph_context(
    entity_features: pd.DataFrame,
    feature_columns: Sequence[str],
    graph_context_bundle: Any,
    config: Optional[SemiSupervisedConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    config = config or SemiSupervisedConfig()
    if entity_features.empty:
        return entity_features.copy(), list(feature_columns)
    from .hgcn_backend import build_entity_graph_context_features

    graph_context = build_entity_graph_context_features(
        entity_features=entity_features,
        bundle=graph_context_bundle,
        config=config,
    )
    augmented = augment_prepared_feature_bundle_with_graph_context(
        tables=FeatureBundleTables(
            dataset=str(entity_features["dataset"].iloc[0]) if "dataset" in entity_features.columns and not entity_features.empty else "",
            windows=pd.DataFrame(),
            entity_features=entity_features,
            metadata={},
            feature_columns=list(feature_columns),
        ),
        graph_context=graph_context,
        config=config,
    )
    return augmented.entity_features, list(augmented.feature_columns)


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return numeric


def _sanitize_embedding_column_fragment(value: str) -> str:
    fragments = []
    for character in str(value):
        if character.isalnum():
            fragments.append(character.lower())
        else:
            fragments.append("_")
    sanitized = "".join(fragments)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def _entity_signature_column(prefix: str, entity_id: str) -> str:
    return "%s_%s" % (prefix, _sanitize_embedding_column_fragment(entity_id))


def _entity_feature_matrix_column(entity_id: str, feature_column: str) -> str:
    return "entityfeat_%s__%s" % (
        _sanitize_embedding_column_fragment(entity_id),
        _sanitize_embedding_column_fragment(feature_column),
    )


def _anomaly_feature_weight(column: str) -> Optional[float]:
    if column in ANOMALY_SIGNATURE_WEIGHTS:
        return float(ANOMALY_SIGNATURE_WEIGHTS[column])
    for prefix, weight in ANOMALY_PREFIX_WEIGHTS.items():
        if str(column).startswith(prefix):
            return float(weight)
    return None


def _select_anomaly_feature_columns(feature_columns: Sequence[str]) -> List[str]:
    return [
        column
        for column in feature_columns
        if _anomaly_feature_weight(column) is not None
    ]


def _row_anomaly_signature_scalar(
    row: Any,
    anomaly_columns: Sequence[str],
) -> float:
    score = 0.0
    for column in anomaly_columns:
        value = max(0.0, _safe_float(getattr(row, column, 0.0)))
        if value <= 0.0:
            continue
        weight = _anomaly_feature_weight(column)
        score += float(weight if weight is not None else 1.0) * value
    return float(score)


def _window_type_rank_scores(
    values: Mapping[str, float],
) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(
        values.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    denominator = max(1, len(ordered) - 1)
    rank_scores = {}
    for index, (entity_id, value) in enumerate(ordered):
        if float(value) <= 0.0:
            rank_scores[str(entity_id)] = 0.0
            continue
        rank_scores[str(entity_id)] = 1.0 - (float(index) / float(denominator))
    return rank_scores


def _window_positive_count(group: pd.DataFrame) -> int:
    # Semi-supervised feature views intentionally remove this audit-only target.
    if "is_positive" not in group.columns:
        return 0
    return int(group["is_positive"].sum())


def build_window_anomaly_signature_table(
    tables: FeatureBundleTables,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    entity_features = tables.entity_features
    selected_feature_columns = list(feature_columns or tables.feature_columns)
    anomaly_columns = _select_anomaly_feature_columns(selected_feature_columns)
    entity_index_rows = (
        entity_features[
            ["entity_id", "entity_type", "entity_name", "entity_index"]
        ]
        .drop_duplicates(subset=["entity_id"])
        .sort_values(by=["entity_index", "entity_id"])
        .reset_index(drop=True)
    )
    entity_records = [
        {
            "entity_id": str(row.entity_id),
            "entity_type": str(row.entity_type),
        }
        for row in entity_index_rows.itertuples(index=False)
    ]
    records = []

    for window_id, group in entity_features.groupby("window_id", sort=False):
        record = group.iloc[0][WINDOW_META_COLUMNS].to_dict()
        scores_by_entity = {}
        scores_by_type = {"service": {}, "host": {}}
        for row in group.itertuples(index=False):
            entity_id = str(row.entity_id)
            entity_type = str(row.entity_type)
            anomaly_score = _row_anomaly_signature_scalar(row, anomaly_columns)
            scores_by_entity[entity_id] = anomaly_score
            if entity_type in scores_by_type:
                scores_by_type[entity_type][entity_id] = anomaly_score

        rank_by_type = {
            entity_type: _window_type_rank_scores(values)
            for entity_type, values in scores_by_type.items()
        }
        totals_by_type = {
            entity_type: float(sum(values.values()))
            for entity_type, values in scores_by_type.items()
        }
        active_counts = {
            entity_type: int(sum(1 for value in values.values() if float(value) > 0.0))
            for entity_type, values in scores_by_type.items()
        }
        peak_scores = {
            entity_type: (
                max(values.values()) if values else 0.0
            )
            for entity_type, values in scores_by_type.items()
        }

        for entity_record in entity_records:
            entity_id = entity_record["entity_id"]
            entity_type = entity_record["entity_type"]
            score = float(scores_by_entity.get(entity_id, 0.0))
            total = float(totals_by_type.get(entity_type, 0.0))
            record[_entity_signature_column("signature_raw", entity_id)] = score
            record[_entity_signature_column("signature_share", entity_id)] = (
                score / total if total > 0.0 else 0.0
            )
            record[_entity_signature_column("signature_rank", entity_id)] = float(
                rank_by_type.get(entity_type, {}).get(entity_id, 0.0)
            )

        record["signature_total_service"] = float(totals_by_type.get("service", 0.0))
        record["signature_total_host"] = float(totals_by_type.get("host", 0.0))
        record["signature_peak_service"] = float(peak_scores.get("service", 0.0))
        record["signature_peak_host"] = float(peak_scores.get("host", 0.0))
        record["signature_active_service_count"] = float(active_counts.get("service", 0))
        record["signature_active_host_count"] = float(active_counts.get("host", 0))
        record["window_positive_count"] = _window_positive_count(group)
        if scores_by_entity:
            anomaly_values = list(scores_by_entity.values())
            record["window_anomaly_score_max"] = float(max(anomaly_values))
            record["window_anomaly_score_mean"] = float(np.mean(anomaly_values))
        else:
            record["window_anomaly_score_max"] = 0.0
            record["window_anomaly_score_mean"] = 0.0
        records.append(record)

    return pd.DataFrame.from_records(records)


def build_window_embedding_table(
    tables: FeatureBundleTables,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    entity_features = tables.entity_features
    selected_feature_columns = list(feature_columns or tables.feature_columns)
    records = []

    anomaly_columns = _select_anomaly_feature_columns(selected_feature_columns)
    for window_id, group in entity_features.groupby("window_id", sort=False):
        record = group.iloc[0][WINDOW_META_COLUMNS].to_dict()
        for entity_type in ("service", "host"):
            subset = group[group["entity_type"] == entity_type]
            for column in selected_feature_columns:
                prefix = "%s_%s" % (entity_type, column)
                if subset.empty:
                    record[prefix + "_mean"] = 0.0
                    record[prefix + "_max"] = 0.0
                else:
                    record[prefix + "_mean"] = float(subset[column].mean())
                    record[prefix + "_max"] = float(subset[column].max())
        if anomaly_columns:
            anomaly_values = np.asarray(
                [
                    _row_anomaly_signature_scalar(row, anomaly_columns)
                    for row in group.itertuples(index=False)
                ],
                dtype=float,
            )
            record["window_anomaly_score_max"] = float(anomaly_values.max())
            record["window_anomaly_score_mean"] = float(anomaly_values.mean())
        else:
            record["window_anomaly_score_max"] = 0.0
            record["window_anomaly_score_mean"] = 0.0
        record["window_positive_count"] = _window_positive_count(group)
        records.append(record)

    return pd.DataFrame.from_records(records)


def build_window_entity_feature_matrix_table(
    tables: FeatureBundleTables,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    entity_features = tables.entity_features
    selected_feature_columns = [
        column
        for column in list(feature_columns or tables.feature_columns)
        if not str(column).startswith("rel_")
    ]
    anomaly_columns = _select_anomaly_feature_columns(selected_feature_columns)
    records = []

    for _window_id, group in entity_features.groupby("window_id", sort=False):
        record = group.iloc[0][WINDOW_META_COLUMNS].to_dict()
        for row in group.itertuples(index=False):
            entity_id = str(row.entity_id)
            for column in selected_feature_columns:
                record[_entity_feature_matrix_column(entity_id, column)] = _safe_float(
                    getattr(row, column, 0.0)
                )
        if anomaly_columns:
            anomaly_values = np.asarray(
                [
                    _row_anomaly_signature_scalar(row, anomaly_columns)
                    for row in group.itertuples(index=False)
                ],
                dtype=float,
            )
            record["window_anomaly_score_max"] = float(anomaly_values.max())
            record["window_anomaly_score_mean"] = float(anomaly_values.mean())
        else:
            record["window_anomaly_score_max"] = 0.0
            record["window_anomaly_score_mean"] = 0.0
        record["window_positive_count"] = _window_positive_count(group)
        records.append(record)

    return pd.DataFrame.from_records(records)


def merge_window_embedding_tables(
    base_frame: pd.DataFrame,
    extra_frame: pd.DataFrame,
) -> pd.DataFrame:
    if base_frame.empty:
        return extra_frame.copy()
    if extra_frame.empty:
        return base_frame.copy()

    merge_columns = [
        column
        for column in extra_frame.columns
        if (
            column not in WINDOW_META_COLUMNS + ["window_positive_count"]
            and column not in base_frame.columns
        )
    ]
    if not merge_columns:
        return base_frame.copy()
    merged = base_frame.merge(
        extra_frame[["window_id"] + merge_columns],
        on="window_id",
        how="left",
    )
    for column in merge_columns:
        merged[column] = merged[column].fillna(0.0)
    return merged


def resolve_cluster_embedding_table(
    tables: FeatureBundleTables,
    feature_columns: Sequence[str],
    cluster_representation: str,
    graph_embedding_frame: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    representation = str(cluster_representation or "feature_aggregate").lower()
    if representation == "entity_anomaly_signature":
        return build_window_anomaly_signature_table(
            tables=tables,
            feature_columns=feature_columns,
        )
    if representation == "hybrid_feature_anomaly_signature":
        return merge_window_embedding_tables(
            build_window_embedding_table(tables, feature_columns),
            build_window_anomaly_signature_table(tables, feature_columns),
        )
    if representation == "hybrid_hgcn_anomaly_signature":
        signature_frame = build_window_anomaly_signature_table(
            tables=tables,
            feature_columns=feature_columns,
        )
        if graph_embedding_frame is None:
            return signature_frame
        return merge_window_embedding_tables(
            graph_embedding_frame,
            signature_frame,
        )
    if representation == "entity_feature_matrix":
        return build_window_entity_feature_matrix_table(
            tables=tables,
            feature_columns=feature_columns,
        )
    if representation == "hybrid_hgcn_entity_feature_matrix":
        matrix_frame = build_window_entity_feature_matrix_table(
            tables=tables,
            feature_columns=feature_columns,
        )
        if graph_embedding_frame is None:
            return matrix_frame
        return merge_window_embedding_tables(
            graph_embedding_frame,
            matrix_frame,
        )
    if representation == "hgcn_autoencoder" and graph_embedding_frame is not None:
        return graph_embedding_frame.copy()
    return build_window_embedding_table(
        tables=tables,
        feature_columns=feature_columns,
    )


def cluster_windows(
    embedding_frame: pd.DataFrame,
    config: Optional[SemiSupervisedConfig] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    config = config or SemiSupervisedConfig()
    if embedding_frame.empty:
        clustered = embedding_frame.copy()
        clustered.attrs["normal_cluster_id"] = 0
        clustered.attrs["cluster_sizes"] = {}
        clustered.attrs["silhouette_score"] = -1.0
        return clustered

    feature_columns = [
        column
        for column in embedding_frame.columns
        if column not in WINDOW_META_COLUMNS + ["window_positive_count"]
    ]
    features = embedding_frame[feature_columns].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    reduced = scaled
    if scaled.shape[0] > 2 and scaled.shape[1] > 2:
        component_count = min(
            max(2, int(config.pca_components)),
            scaled.shape[1],
            scaled.shape[0] - 1,
        )
        if component_count >= 2:
            reduced = PCA(n_components=component_count, random_state=random_state).fit_transform(scaled)

    cluster_mode = str(config.cluster_mode).lower()
    cluster_labels = None
    centroids = None
    best_score = -1.0

    if cluster_mode == "dbscan":
        cluster_labels, best_score = _select_dbscan_labels(reduced, config)

    if cluster_labels is None:
        cluster_labels, centroids, best_score = _select_kmeans_like_labels(
            reduced,
            config,
            random_state=random_state,
        )
    else:
        centroids = _cluster_centroids(reduced, cluster_labels)

    clustered = embedding_frame.copy().reset_index(drop=True)
    clustered["cluster_id"] = cluster_labels
    cluster_sizes = clustered["cluster_id"].value_counts().to_dict()
    non_noise_sizes = {
        int(cluster_id): int(size)
        for cluster_id, size in cluster_sizes.items()
        if int(cluster_id) != -1
    }
    if non_noise_sizes:
        normal_cluster_id = int(max(non_noise_sizes.items(), key=lambda item: item[1])[0])
    else:
        normal_cluster_id = int(max(cluster_sizes.items(), key=lambda item: item[1])[0])
    clustered["is_normal_cluster"] = (clustered["cluster_id"] == normal_cluster_id).astype(int)

    normal_centroid = centroids.get(normal_cluster_id, reduced.mean(axis=0))
    clustered["distance_to_normal_centroid"] = np.linalg.norm(reduced - normal_centroid, axis=1)
    embed_frame = pd.DataFrame(
        reduced,
        columns=["embed_%02d" % dim_index for dim_index in range(reduced.shape[1])],
    )
    clustered = pd.concat([clustered, embed_frame], axis=1)
    clustered["cluster_size"] = clustered["cluster_id"].map(cluster_sizes).astype(int)
    clustered["distance_to_cluster_medoid"] = 0.0
    clustered["cluster_medoid_window_id"] = ""
    clustered["is_cluster_medoid"] = 0
    clustered = _annotate_cluster_geometry(clustered)
    clustered.attrs["normal_cluster_id"] = normal_cluster_id
    clustered.attrs["cluster_sizes"] = cluster_sizes
    clustered.attrs["silhouette_score"] = best_score
    clustered.attrs["cluster_mode"] = config.cluster_mode
    clustered.attrs["noise_count"] = int((clustered["cluster_id"] == -1).sum())
    return clustered


def _window_label_map(windows: pd.DataFrame) -> Dict[str, List[str]]:
    mapping = {}
    for row in windows.itertuples(index=False):
        mapping[str(row.window_id)] = list(getattr(row, "positive_ids_list", []))
    return mapping


def _resolve_supervision_mode(config: SemiSupervisedConfig) -> str:
    return validate_supervision_request(
        supervision_mode=str(getattr(config, "supervision_mode", "auto") or "auto"),
        query_strategy=str(getattr(config, "query_strategy", "") or ""),
    )


def _resolve_normal_training_policy(config: SemiSupervisedConfig) -> str:
    policy = str(
        getattr(config, "normal_training_policy", "with_normal_class")
        or "with_normal_class"
    ).strip().lower()
    if policy not in {"fault_only", "with_normal_class"}:
        raise ValueError(
            "unsupported normal_training_policy: %s"
            % getattr(config, "normal_training_policy", policy)
        )
    return policy


def _ordered_fault_window_ids(
    windows: pd.DataFrame,
    true_labels: Mapping[str, Sequence[str]],
) -> List[str]:
    fault_windows = windows[windows["window_kind"].astype(str) == "fault"].copy()
    if fault_windows.empty:
        return []
    ordered = fault_windows.sort_values(by=["start_ts", "window_id"]).reset_index(drop=True)
    return [
        str(window_id)
        for window_id in ordered["window_id"].astype(str).tolist()
        if true_labels.get(str(window_id))
    ]


def _cluster_centroids(reduced: np.ndarray, cluster_labels: np.ndarray) -> Dict[int, np.ndarray]:
    centroids = {}
    for cluster_id in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster_id
        centroids[int(cluster_id)] = reduced[mask].mean(axis=0)
    return centroids


def _select_dbscan_labels(
    reduced: np.ndarray,
    config: SemiSupervisedConfig,
) -> Tuple[Optional[np.ndarray], float]:
    if reduced.shape[0] < max(3, int(config.dbscan_min_samples)):
        return None, -1.0
    neighbor_count = min(max(2, int(config.dbscan_min_samples)), reduced.shape[0])
    neighbors = NearestNeighbors(n_neighbors=neighbor_count)
    neighbors.fit(reduced)
    distances, _ = neighbors.kneighbors(reduced)
    kth_distances = distances[:, -1]
    candidates = sorted(
        {
            float(np.quantile(kth_distances, quantile))
            for quantile in config.dbscan_eps_quantiles
            if 0.0 < float(quantile) < 1.0
        }
    )
    best_labels = None
    best_objective = -1e9
    best_silhouette = -1.0
    for eps in candidates:
        if eps <= 0.0:
            continue
        labels = DBSCAN(
            eps=float(eps),
            min_samples=int(config.dbscan_min_samples),
        ).fit_predict(reduced)
        non_noise_mask = labels != -1
        cluster_count = len(set(labels[non_noise_mask].tolist())) if non_noise_mask.any() else 0
        if cluster_count == 0:
            continue
        noise_ratio = float((labels == -1).sum()) / float(len(labels))
        if cluster_count >= 2 and int(non_noise_mask.sum()) > cluster_count:
            silhouette = float(silhouette_score(reduced[non_noise_mask], labels[non_noise_mask]))
        else:
            silhouette = -0.25
        objective = silhouette + (0.05 * cluster_count) - (0.10 * noise_ratio)
        if objective > best_objective:
            best_objective = objective
            best_labels = labels
            best_silhouette = silhouette
    return best_labels, best_silhouette


def _select_kmeans_like_labels(
    reduced: np.ndarray,
    config: SemiSupervisedConfig,
    random_state: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], float]:
    cluster_mode = str(config.cluster_mode).lower()
    best_k = 1
    best_score = -1.0
    if cluster_mode == "fixed_k":
        best_k = min(max(1, int(config.fixed_cluster_count)), reduced.shape[0])
    elif cluster_mode == "single_cluster":
        best_k = 1
    else:
        upper_k = min(max(2, int(config.max_clusters)), max(2, reduced.shape[0] - 1))
        if reduced.shape[0] >= 3:
            for cluster_count in range(2, upper_k + 1):
                candidate = KMeans(
                    n_clusters=cluster_count,
                    n_init=20,
                    random_state=random_state,
                ).fit_predict(reduced)
                if len(set(candidate)) < 2:
                    continue
                score = silhouette_score(reduced, candidate)
                if score > best_score:
                    best_score = score
                    best_k = cluster_count

    if best_k <= 1:
        cluster_labels = np.zeros(reduced.shape[0], dtype=int)
    else:
        cluster_labels = KMeans(
            n_clusters=best_k,
            n_init=20,
            random_state=random_state,
        ).fit_predict(reduced)
    return cluster_labels, _cluster_centroids(reduced, cluster_labels), best_score


def _annotate_cluster_geometry(clustered: pd.DataFrame) -> pd.DataFrame:
    embed_columns = [column for column in clustered.columns if column.startswith("embed_")]
    if not embed_columns:
        return clustered
    for cluster_id in sorted(set(clustered["cluster_id"].astype(int).tolist())):
        if int(cluster_id) == -1:
            continue
        cluster_rows = clustered[clustered["cluster_id"] == cluster_id].copy()
        if cluster_rows.empty:
            continue
        vectors = cluster_rows[embed_columns].to_numpy(dtype=float)
        distance_matrix = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
        medoid_index = int(np.argmin(distance_matrix.sum(axis=1)))
        medoid_row = cluster_rows.iloc[medoid_index]
        medoid_window_id = str(medoid_row["window_id"])
        medoid_vector = medoid_row[embed_columns].to_numpy(dtype=float)
        distances = np.linalg.norm(vectors - medoid_vector, axis=1)
        clustered.loc[cluster_rows.index, "distance_to_cluster_medoid"] = distances
        clustered.loc[cluster_rows.index, "cluster_medoid_window_id"] = medoid_window_id
        clustered.loc[cluster_rows.index, "is_cluster_medoid"] = 0
        clustered.loc[cluster_rows.index[medoid_index], "is_cluster_medoid"] = 1
    return clustered


def _fallback_query_windows(clustered_windows: pd.DataFrame, budget: int) -> List[Tuple[str, str]]:
    fallback = clustered_windows[clustered_windows["window_kind"] == "fault"].copy()
    if fallback.empty:
        return []
    fallback = fallback.sort_values(
        by=["window_anomaly_score_max", "window_anomaly_score_mean"],
        ascending=[False, False],
    )
    return [
        (str(window_id), "fallback")
        for window_id in fallback["window_id"].head(budget).astype(str).tolist()
    ]


def _select_fault_cluster_representatives(
    candidate_windows: pd.DataFrame,
) -> List[Tuple[str, str]]:
    if candidate_windows.empty:
        return []

    representatives: List[Tuple[str, str, int, float, float]] = []
    abnormal = candidate_windows[candidate_windows["cluster_id"] != -1].copy()
    if abnormal.empty:
        return []

    for cluster_id, cluster_rows in abnormal.groupby("cluster_id", sort=False):
        ranked = cluster_rows.sort_values(
            by=[
                "distance_to_cluster_medoid",
                "distance_to_normal_centroid",
                "window_anomaly_score_max",
            ],
            ascending=[True, False, False],
        )
        row = ranked.iloc[0]
        representatives.append(
            (
                str(row["window_id"]),
                "cluster_medoid",
                int(cluster_id),
                float(row["cluster_size"]),
                float(row["distance_to_normal_centroid"]),
            )
        )

    representatives.sort(
        key=lambda item: (-item[3], -item[4], item[2], item[0]),
    )
    return [(window_id, role) for window_id, role, *_ in representatives]


def _fill_diverse_query_windows(
    candidate_windows: pd.DataFrame,
    selected: Sequence[Tuple[str, str]],
    budget: int,
) -> List[Tuple[str, str]]:
    if len(selected) >= budget or candidate_windows.empty:
        return list(selected)[:budget]

    chosen = list(selected)
    selected_ids = {window_id for window_id, _ in chosen}
    selected_clusters = {
        int(row.cluster_id)
        for row in candidate_windows[
            candidate_windows["window_id"].astype(str).isin(selected_ids)
        ].itertuples(index=False)
    }
    remaining = candidate_windows[
        ~candidate_windows["window_id"].astype(str).isin(selected_ids)
    ].copy()
    embed_columns = [column for column in candidate_windows.columns if column.startswith("embed_")]

    while len(chosen) < budget and not remaining.empty:
        if not embed_columns or not selected_ids:
            ranked = remaining.sort_values(
                by=["distance_to_normal_centroid", "window_anomaly_score_max"],
                ascending=[False, False],
            )
            row = ranked.iloc[0]
        else:
            selected_points = candidate_windows[
                candidate_windows["window_id"].astype(str).isin(selected_ids)
            ][embed_columns].to_numpy(dtype=float)
            candidate_points = remaining[embed_columns].to_numpy(dtype=float)
            min_distances = []
            for point in candidate_points:
                distances = np.linalg.norm(selected_points - point, axis=1)
                min_distances.append(float(distances.min()))
            cluster_bonus = (
                ~remaining["cluster_id"].astype(int).isin(selected_clusters)
            ).astype(float).to_numpy(dtype=float)
            candidate_scores = (
                np.asarray(min_distances, dtype=float)
                + (0.10 * cluster_bonus)
                + (0.01 * remaining["distance_to_normal_centroid"].astype(float).to_numpy())
            )
            row = remaining.iloc[int(np.argmax(candidate_scores))]

        window_id = str(row["window_id"])
        role = "noise" if int(row["cluster_id"]) == -1 else "boundary"
        chosen.append((window_id, role))
        selected_ids.add(window_id)
        selected_clusters.add(int(row["cluster_id"]))
        remaining = remaining[remaining["window_id"].astype(str) != window_id].copy()

    return chosen[:budget]


def _select_oracle_label_coverage_windows(
    candidate_windows: pd.DataFrame,
    budget: int,
    true_labels: Mapping[str, Sequence[str]],
) -> List[Tuple[str, str]]:
    if budget <= 0 or candidate_windows.empty:
        return []

    label_map: Dict[str, Tuple[str, ...]] = {}
    label_frequency: Counter = Counter()
    for row in candidate_windows.itertuples(index=False):
        window_id = str(row.window_id)
        labels = tuple(sorted(set(str(label) for label in true_labels.get(window_id, []))))
        if not labels:
            continue
        label_map[window_id] = labels
        label_frequency.update(labels)
    if not label_map:
        return []

    representative_ids = {
        window_id
        for window_id, _role in _select_fault_cluster_representatives(candidate_windows)
    }
    selected: List[Tuple[str, str]] = []
    selected_ids: Set[str] = set()
    selected_clusters: Set[int] = set()
    uncovered_labels = set(label_frequency.keys())

    while len(selected) < budget:
        best_window_id: Optional[str] = None
        best_role = "oracle_coverage"
        best_key: Optional[Tuple[float, float, float, float, str]] = None
        for row in candidate_windows.itertuples(index=False):
            window_id = str(row.window_id)
            if window_id in selected_ids or window_id not in label_map:
                continue
            labels = set(label_map[window_id])
            new_labels = [label for label in labels if label in uncovered_labels]
            coverage_score = float(
                sum(1.0 / max(1, int(label_frequency[label])) for label in new_labels)
            )
            if coverage_score <= 0.0:
                continue
            cluster_bonus = 1.0 if int(row.cluster_id) not in selected_clusters else 0.0
            representative_bonus = 1.0 if window_id in representative_ids else 0.0
            score_key = (
                coverage_score,
                cluster_bonus,
                representative_bonus,
                float(getattr(row, "distance_to_normal_centroid", 0.0)),
                window_id,
            )
            if best_key is None or score_key > best_key:
                best_window_id = window_id
                best_role = "cluster_medoid" if window_id in representative_ids else "oracle_coverage"
                best_key = score_key

        if best_window_id is None:
            break
        selected.append((best_window_id, best_role))
        selected_ids.add(best_window_id)
        selected_row = candidate_windows[
            candidate_windows["window_id"].astype(str) == best_window_id
        ].iloc[0]
        selected_clusters.add(int(selected_row["cluster_id"]))
        uncovered_labels.difference_update(label_map.get(best_window_id, ()))

    return _fill_diverse_query_windows(candidate_windows, selected, budget)


def _filter_cluster_candidates(
    clustered_windows: pd.DataFrame,
    normal_cluster_id: int,
    *,
    fault_only: bool,
    include_normal_cluster_faults: bool,
) -> pd.DataFrame:
    candidates = clustered_windows.copy()
    if fault_only:
        candidates = candidates[candidates["window_kind"] == "fault"].copy()
    if not include_normal_cluster_faults:
        candidates = candidates[candidates["cluster_id"] != normal_cluster_id].copy()
    return candidates.reset_index(drop=True)


def _cluster_label_consensus(
    cluster_rows: pd.DataFrame,
    true_labels: Mapping[str, Sequence[str]],
) -> float:
    if cluster_rows.empty:
        return 0.0
    signatures = []
    for window_id in cluster_rows["window_id"].astype(str).tolist():
        labels = tuple(sorted(set(true_labels.get(window_id, ()))))
        if labels:
            signatures.append(labels)
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    signature_consensus = float(max(counts.values())) / float(len(signatures))
    entity_counts: Counter = Counter()
    for labels in signatures:
        for entity_id in labels:
            entity_counts[str(entity_id)] += 1
    entity_consensus = (
        float(max(entity_counts.values())) / float(len(signatures))
        if entity_counts
        else 0.0
    )
    return max(signature_consensus, entity_consensus)


def _cluster_support_factor(query_count: int) -> float:
    if query_count <= 0:
        return 0.0
    return float(1.0 - (0.35 ** int(query_count)))


def _resolve_pseudo_cluster_limit(
    unlabeled_count: int,
    query_count: int,
    config: SemiSupervisedConfig,
) -> int:
    if unlabeled_count <= 0:
        return 0
    fraction = max(0.0, float(config.pseudo_cluster_core_fraction))
    limit = unlabeled_count
    if fraction > 0.0:
        limit = max(1, int(math.ceil(unlabeled_count * fraction)))
    limit = max(int(config.pseudo_cluster_min_core), limit)
    max_core = int(config.pseudo_cluster_max_core)
    if max_core > 0:
        limit = min(limit, max_core)
    per_query = int(config.pseudo_windows_per_queried)
    if per_query > 0:
        limit = min(limit, max(1, int(query_count) * per_query))
    return max(0, min(unlabeled_count, limit))


def _collect_positive_label_counts(true_labels: Mapping[str, Sequence[str]]) -> Counter:
    counts: Counter = Counter()
    for labels in true_labels.values():
        for entity_id in set(labels):
            counts[str(entity_id)] += 1
    return counts


def _resolve_pseudo_entity_limit(
    label_signature: Sequence[str],
    queried_positive_counts: Mapping[str, int],
    config: SemiSupervisedConfig,
) -> Optional[int]:
    multiplier = int(config.pseudo_entity_query_multiplier)
    if multiplier <= 0:
        return None
    min_cap = max(1, int(config.pseudo_entity_min_cap))
    caps = []
    for entity_id in sorted(set(str(item) for item in label_signature)):
        caps.append(max(min_cap, int(queried_positive_counts.get(entity_id, 0)) * multiplier))
    if not caps:
        return None
    return max(1, min(caps))


def _queried_positive_type_ratios(
    queried_labels: Mapping[str, Sequence[str]],
) -> Mapping[str, float]:
    counts: Counter = Counter()
    total = 0
    for labels in queried_labels.values():
        for entity_id in labels:
            entity_type = str(entity_id).split(":", 1)[0]
            counts[entity_type] += 1
            total += 1
    if total <= 0:
        return {}
    return {
        str(entity_type): float(count) / float(total)
        for entity_type, count in counts.items()
    }


def _resolve_entity_score_calibration_mode(
    config: SemiSupervisedConfig,
    query_plan: QueryPlan,
) -> str:
    mode = str(config.entity_score_calibration_mode).lower()
    if mode != "auto_service_zscore":
        return mode
    type_ratios = _queried_positive_type_ratios(query_plan.queried_labels)
    host_ratio = float(type_ratios.get("host", 0.0))
    if host_ratio <= float(config.entity_score_calibration_host_positive_max_ratio):
        return "service_zscore"
    return "none"


def _resolve_prototype_rerank_mode(
    config: SemiSupervisedConfig,
    query_plan: QueryPlan,
) -> str:
    mode = str(config.prototype_rerank_mode).lower()
    if mode != "auto_service_margin":
        return mode
    type_ratios = _queried_positive_type_ratios(query_plan.queried_labels)
    host_ratio = float(type_ratios.get("host", 0.0))
    if host_ratio <= float(config.prototype_rerank_host_positive_max_ratio):
        return "service_margin"
    return "none"


def _build_entity_score_calibration(
    scored_rows: pd.DataFrame,
    min_std: float,
) -> Mapping[str, Mapping[str, float]]:
    normal_rows = scored_rows[scored_rows["window_kind"] == "normal"].copy()
    if normal_rows.empty:
        return {}
    grouped = (
        normal_rows.groupby("entity_id", sort=False)["score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    calibration = {}
    for row in grouped.itertuples(index=False):
        std = float(getattr(row, "std"))
        if not np.isfinite(std):
            std = 0.0
        calibration[str(row.entity_id)] = {
            "mean": float(getattr(row, "mean")),
            "std": max(float(min_std), std),
        }
    return calibration


def _apply_entity_score_calibration(
    rows: pd.DataFrame,
    calibration: Mapping[str, Mapping[str, float]],
    mode: str,
    center_alpha: float,
) -> pd.DataFrame:
    resolved_mode = str(mode or "none").lower()
    if rows.empty or resolved_mode == "none" or not calibration:
        return rows

    calibrated = rows.copy()
    mean_series = calibrated["entity_id"].astype(str).map(
        lambda entity_id: float(calibration.get(entity_id, {}).get("mean", 0.0))
    )
    std_series = calibrated["entity_id"].astype(str).map(
        lambda entity_id: float(calibration.get(entity_id, {}).get("std", 1.0))
    )
    std_series = std_series.replace(0.0, 1.0)
    score_values = calibrated["score"].astype(float).copy()

    if resolved_mode.startswith("service_"):
        target_mask = calibrated["entity_type"].astype(str) == "service"
        base_mode = resolved_mode[len("service_") :]
    else:
        target_mask = pd.Series(True, index=calibrated.index)
        base_mode = resolved_mode

    if base_mode == "center":
        score_values.loc[target_mask] = (
            score_values.loc[target_mask]
            - (float(center_alpha) * mean_series.loc[target_mask])
        )
    elif base_mode == "zscore":
        score_values.loc[target_mask] = (
            (score_values.loc[target_mask] - mean_series.loc[target_mask])
            / std_series.loc[target_mask]
        )
    else:
        return rows

    calibrated["score"] = score_values
    return calibrated


def _normalized_matrix(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    if values.size == 0:
        return values
    normalized = (values.astype(float) - center[None, :]) / scale[None, :]
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return normalized / norms


def _build_prototype_rerank_state(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Optional[PrototypeRerankState]:
    if training_frame.empty or not feature_columns:
        return None
    queried_positive = training_frame[
        (training_frame["label"] == 1)
        & (training_frame["label_source"] == "queried")
    ].copy()
    normal_rows = training_frame[training_frame["label_source"] == "normal_window"].copy()
    if queried_positive.empty or normal_rows.empty:
        return None

    feature_matrix = training_frame[list(feature_columns)].fillna(0.0).to_numpy(dtype=float)
    center = feature_matrix.mean(axis=0)
    scale = feature_matrix.std(axis=0)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale < 1e-6] = 1.0

    queried_vectors = _normalized_matrix(
        queried_positive[list(feature_columns)].fillna(0.0).to_numpy(dtype=float),
        center=center,
        scale=scale,
    )
    normal_vectors = _normalized_matrix(
        normal_rows[list(feature_columns)].fillna(0.0).to_numpy(dtype=float),
        center=center,
        scale=scale,
    )

    def _group_means(frame: pd.DataFrame, matrix: np.ndarray, key: str) -> Dict[str, np.ndarray]:
        grouped = {}
        for value, indices in frame.groupby(key, sort=False).groups.items():
            grouped[str(value)] = matrix[list(indices)].mean(axis=0)
        return grouped

    return PrototypeRerankState(
        feature_columns=list(feature_columns),
        center=center,
        scale=scale,
        entity_positive=_group_means(queried_positive.reset_index(drop=True), queried_vectors, "entity_id"),
        entity_negative=_group_means(normal_rows.reset_index(drop=True), normal_vectors, "entity_id"),
        type_positive=_group_means(queried_positive.reset_index(drop=True), queried_vectors, "entity_type"),
        type_negative=_group_means(normal_rows.reset_index(drop=True), normal_vectors, "entity_type"),
    )


def _apply_prototype_rerank(
    rows: pd.DataFrame,
    state: Optional[PrototypeRerankState],
    mode: str,
    alpha: float,
    same_entity_weight: float,
    same_type_weight: float,
) -> pd.DataFrame:
    resolved_mode = str(mode or "none").lower()
    if (
        rows.empty
        or state is None
        or not state.feature_columns
        or resolved_mode == "none"
        or float(alpha) == 0.0
    ):
        return rows

    reranked = rows.copy()
    values = reranked[list(state.feature_columns)].fillna(0.0).to_numpy(dtype=float)
    normalized = _normalized_matrix(values, center=state.center, scale=state.scale)
    support = np.zeros(len(reranked), dtype=float)
    same_entity_weight = float(same_entity_weight)
    same_type_weight = float(same_type_weight)

    for index, row in enumerate(reranked.itertuples(index=False)):
        vector = normalized[index]
        entity_id = str(row.entity_id)
        entity_type = str(row.entity_type)

        if entity_id in state.entity_positive:
            support[index] += same_entity_weight * float(
                np.dot(vector, state.entity_positive[entity_id])
            )
        if entity_id in state.entity_negative:
            support[index] -= same_entity_weight * float(
                np.dot(vector, state.entity_negative[entity_id])
            )
        if entity_type in state.type_positive:
            support[index] += same_type_weight * float(
                np.dot(vector, state.type_positive[entity_type])
            )
        if entity_type in state.type_negative:
            support[index] -= same_type_weight * float(
                np.dot(vector, state.type_negative[entity_type])
            )

    target_mask = reranked["window_kind"].astype(str) == "fault"
    if resolved_mode.startswith("service_"):
        target_mask = target_mask & (reranked["entity_type"].astype(str) == "service")
    reranked["prototype_support"] = support
    reranked.loc[target_mask, "score"] = (
        reranked.loc[target_mask, "score"].astype(float)
        + (float(alpha) * reranked.loc[target_mask, "prototype_support"].astype(float))
    )
    return reranked


def select_query_windows_with_roles(
    clustered_windows: pd.DataFrame,
    budget: int,
    random_state: int = 42,
    query_strategy: str = "sequential_medoid_noise_boundary",
    true_labels: Optional[Mapping[str, Sequence[str]]] = None,
    config: Optional[SemiSupervisedConfig] = None,
) -> List[Tuple[str, str]]:
    config = config or SemiSupervisedConfig()
    if budget <= 0 or clustered_windows.empty:
        return []

    normal_cluster_id = int(clustered_windows.attrs["normal_cluster_id"])
    candidate_windows = _filter_cluster_candidates(
        clustered_windows,
        normal_cluster_id,
        fault_only=bool(config.query_fault_only),
        include_normal_cluster_faults=bool(config.query_include_normal_cluster_faults),
    )
    if candidate_windows.empty:
        return [] if bool(config.query_fault_only) else _fallback_query_windows(clustered_windows, budget)

    query_strategy = str(query_strategy).lower()
    if query_strategy == "distance_only":
        candidate_windows = candidate_windows.sort_values(
            by=["distance_to_normal_centroid", "window_anomaly_score_max", "window_anomaly_score_mean"],
            ascending=[False, False, False],
        )
        return [
            (str(window_id), "boundary")
            for window_id in candidate_windows["window_id"].head(budget).astype(str).tolist()
        ]
    if query_strategy == "random_abnormal":
        sampled = candidate_windows["window_id"].astype(str).tolist()
        rng = np.random.RandomState(random_state)
        rng.shuffle(sampled)
        return [(window_id, "random") for window_id in sampled[:budget]]
    if query_strategy == "oracle_label_coverage":
        oracle_selected = _select_oracle_label_coverage_windows(
            candidate_windows=candidate_windows,
            budget=budget,
            true_labels=(true_labels or {}),
        )
        if oracle_selected:
            return oracle_selected

    if query_strategy == "sequential_medoid_noise_boundary":
        selected = _select_fault_cluster_representatives(candidate_windows)[:budget]

        if len(selected) < budget:
            noise_rows = candidate_windows[candidate_windows["cluster_id"] == -1].copy()
            noise_rows = noise_rows.sort_values(
                by=["distance_to_normal_centroid", "window_anomaly_score_max"],
                ascending=[False, False],
            )
            for row in noise_rows.itertuples(index=False):
                if len(selected) >= budget:
                    break
                selected.append((str(row.window_id), "noise"))

        if len(selected) < budget:
            boundary_rows = candidate_windows[
                (candidate_windows["cluster_id"] != -1) & (candidate_windows["is_cluster_medoid"] == 0)
            ].copy()
            boundary_rows = boundary_rows.sort_values(
                by=["distance_to_cluster_medoid", "distance_to_normal_centroid"],
                ascending=[False, False],
            )
            selected_ids = {window_id for window_id, _ in selected}
            for row in boundary_rows.itertuples(index=False):
                if len(selected) >= budget:
                    break
                window_id = str(row.window_id)
                if window_id in selected_ids:
                    continue
                selected.append((window_id, "boundary"))
                selected_ids.add(window_id)

        return selected[:budget]

    embed_columns = [column for column in candidate_windows.columns if column.startswith("embed_")]
    if not embed_columns:
        return [
            (str(window_id), "boundary")
            for window_id in candidate_windows["window_id"].head(budget).astype(str).tolist()
        ]

    rng = np.random.RandomState(random_state)
    selected = []
    cluster_order = (
        candidate_windows["cluster_id"]
        .value_counts()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for cluster_id in cluster_order:
        if len(selected) >= budget:
            break
        cluster_rows = candidate_windows[candidate_windows["cluster_id"] == cluster_id].copy()
        center = cluster_rows[embed_columns].mean(axis=0).to_numpy(dtype=float)
        distances = np.linalg.norm(cluster_rows[embed_columns].to_numpy(dtype=float) - center, axis=1)
        medoid = cluster_rows.iloc[int(np.argmin(distances))]
        selected.append((str(medoid["window_id"]), "cluster_medoid"))

    if query_strategy == "cluster_medoids_only":
        return selected[:budget]

    selected_ids = {window_id for window_id, _ in selected}
    remaining = candidate_windows[~candidate_windows["window_id"].isin(selected_ids)].copy()
    if remaining.empty or len(selected) >= budget:
        return selected[:budget]

    remaining = remaining.sort_values(
        by=["distance_to_normal_centroid", "window_anomaly_score_max"],
        ascending=[False, False],
    ).reset_index(drop=True)

    while len(selected) < budget and not remaining.empty:
        if not selected:
            candidate_index = 0
        else:
            chosen_ids = {window_id for window_id, _ in selected}
            selected_points = candidate_windows[
                candidate_windows["window_id"].isin(chosen_ids)
            ][embed_columns].to_numpy(dtype=float)
            candidate_points = remaining[embed_columns].to_numpy(dtype=float)
            min_distances = []
            for point in candidate_points:
                distances = np.linalg.norm(selected_points - point, axis=1)
                min_distances.append(float(distances.min()))
            candidate_index = int(np.argmax(min_distances))
        selected.append((str(remaining.iloc[candidate_index]["window_id"]), "boundary"))
        remaining = remaining.drop(index=candidate_index).reset_index(drop=True)

    if len(selected) > budget:
        rng.shuffle(selected)
    return selected[:budget]


def select_query_windows(
    clustered_windows: pd.DataFrame,
    budget: int,
    random_state: int = 42,
    query_strategy: str = "cluster_medoid_diversity",
) -> List[str]:
    return [
        window_id
        for window_id, _ in select_query_windows_with_roles(
            clustered_windows=clustered_windows,
            budget=budget,
            random_state=random_state,
            query_strategy=query_strategy,
            true_labels=None,
            config=None,
        )
    ]


def _row_embedding_vector(row: Any, embed_columns: Sequence[str]) -> np.ndarray:
    return np.asarray([getattr(row, column) for column in embed_columns], dtype=float)


def propagate_pseudo_labels(
    clustered_windows: pd.DataFrame,
    queried_window_ids: Sequence[str],
    queried_roles: Optional[Mapping[str, str]],
    true_labels: Mapping[str, Sequence[str]],
    config: Optional[SemiSupervisedConfig] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    config = config or SemiSupervisedConfig()
    if str(config.propagate_mode).lower() == "disabled":
        return {}, {}
    if clustered_windows.empty or not queried_window_ids:
        return {}, {}

    queried_set = set(str(item) for item in queried_window_ids)
    normal_cluster_id = int(clustered_windows.attrs["normal_cluster_id"])
    embed_columns = [column for column in clustered_windows.columns if column.startswith("embed_")]
    if not embed_columns:
        return {}, {}

    candidate_windows = _filter_cluster_candidates(
        clustered_windows,
        normal_cluster_id,
        fault_only=bool(config.pseudo_fault_only),
        include_normal_cluster_faults=bool(config.pseudo_include_normal_cluster_faults),
    )
    queried = candidate_windows[candidate_windows["window_id"].isin(queried_set)].copy()
    unlabeled = candidate_windows[~candidate_windows["window_id"].isin(queried_set)].copy()
    if queried.empty or unlabeled.empty:
        return {}, {}

    pseudo_labels = {}
    pseudo_confidence = {}
    propagate_mode = str(config.propagate_mode).lower()
    queried_roles = dict(queried_roles or {})
    queried_positive_counts = _collect_positive_label_counts(true_labels)
    pseudo_positive_counts: Counter = Counter()

    if propagate_mode == "cluster_medoid":
        medoid_window_ids = {
            str(window_id)
            for window_id in (
                str(row.window_id)
                for row in queried.itertuples(index=False)
            )
            if queried_roles.get(window_id) == "cluster_medoid"
        }
        cluster_candidates: Dict[int, List[Tuple[str, List[str], float]]] = defaultdict(list)
        for row in unlabeled.itertuples(index=False):
            row_cluster = int(row.cluster_id)
            if row_cluster == -1:
                continue
            cluster_rows = candidate_windows[candidate_windows["cluster_id"] == row_cluster]
            cluster_queried = queried[queried["cluster_id"] == row_cluster]
            if cluster_queried.empty:
                continue
            source_pool = cluster_queried[
                cluster_queried["window_id"].astype(str).isin(medoid_window_ids)
            ].copy()
            if source_pool.empty:
                source_pool = cluster_queried.copy()
            if source_pool.empty:
                continue

            pool_vectors = source_pool[embed_columns].to_numpy(dtype=float)
            row_vector = _row_embedding_vector(row, embed_columns)
            distances = np.linalg.norm(pool_vectors - row_vector, axis=1)
            vote_scores = {}
            total_vote = 0.0
            for source_index, source_row in enumerate(source_pool.itertuples(index=False)):
                source_window_id = str(source_row.window_id)
                labels = list(true_labels.get(source_window_id, []))
                if not labels:
                    continue
                similarity = 1.0 / (float(distances[source_index]) + 1e-6)
                total_vote += similarity
                for entity_id in labels:
                    vote_scores[entity_id] = vote_scores.get(entity_id, 0.0) + similarity
            if not vote_scores or total_vote <= 0.0:
                continue

            max_vote = max(vote_scores.values())
            labels = [
                entity_id
                for entity_id, score in vote_scores.items()
                if score >= (float(config.multi_positive_margin) * max_vote)
            ]
            labels = sorted(labels) or [max(vote_scores.items(), key=lambda item: item[1])[0]]
            max_distance = (
                float(cluster_rows["distance_to_cluster_medoid"].max())
                if not cluster_rows.empty
                else 0.0
            )
            row_distance = float(getattr(row, "distance_to_cluster_medoid"))
            if max_distance > 0.0:
                distance_confidence = max(0.0, 1.0 - (row_distance / max_distance))
            else:
                distance_confidence = 1.0
            consensus = _cluster_label_consensus(source_pool, true_labels)
            if consensus <= 0.0:
                continue
            if consensus < float(config.pseudo_cluster_consensus_min):
                continue
            query_count = len(source_pool)
            support_factor = _cluster_support_factor(query_count)
            purity = float(max_vote / total_vote)
            confidence = float(distance_confidence * consensus * support_factor * purity)
            if confidence < float(config.pseudo_min_confidence):
                continue
            cluster_candidates[row_cluster].append((str(row.window_id), labels, confidence))

        for cluster_id, rows in cluster_candidates.items():
            cluster_queried = queried[queried["cluster_id"] == cluster_id]
            cluster_limit = _resolve_pseudo_cluster_limit(
                unlabeled_count=len(rows),
                query_count=len(cluster_queried),
                config=config,
            )
            if cluster_limit <= 0:
                continue
            assigned = 0
            for window_id, labels, confidence in sorted(
                rows,
                key=lambda item: (-float(item[2]), item[0]),
            ):
                label_limit = _resolve_pseudo_entity_limit(
                    label_signature=labels,
                    queried_positive_counts=queried_positive_counts,
                    config=config,
                )
                if label_limit is not None:
                    if any(
                        int(pseudo_positive_counts.get(entity_id, 0)) >= label_limit
                        for entity_id in labels
                    ):
                        continue
                pseudo_labels[window_id] = labels
                pseudo_confidence[window_id] = confidence
                for entity_id in labels:
                    pseudo_positive_counts[entity_id] += 1
                assigned += 1
                if assigned >= cluster_limit:
                    break
        return pseudo_labels, pseudo_confidence

    for row in unlabeled.itertuples(index=False):
        row_cluster = int(row.cluster_id)
        same_cluster = queried[queried["cluster_id"] == row_cluster]
        if config.restrict_neighbors_to_cluster:
            pool = same_cluster
        else:
            pool = same_cluster if not same_cluster.empty else queried
        if pool.empty:
            continue

        consensus = _cluster_label_consensus(pool, true_labels)
        if consensus <= 0.0:
            continue
        if consensus < float(config.pseudo_cluster_consensus_min):
            continue
        vote_scores = {}
        total_vote = 0.0
        if propagate_mode == "cluster_majority":
            for pool_row in pool.itertuples(index=False):
                labels = list(true_labels.get(str(pool_row.window_id), []))
                if not labels:
                    continue
                total_vote += 1.0
                for entity_id in labels:
                    vote_scores[entity_id] = vote_scores.get(entity_id, 0.0) + 1.0
        else:
            pool_vectors = pool[embed_columns].to_numpy(dtype=float)
            row_vector = _row_embedding_vector(row, embed_columns)
            distances = np.linalg.norm(pool_vectors - row_vector, axis=1)
            order = np.argsort(distances)[: max(1, min(int(config.neighbor_count), len(pool)))]
            for neighbor_index in order:
                neighbor_window_id = str(pool.iloc[int(neighbor_index)]["window_id"])
                labels = list(true_labels.get(neighbor_window_id, []))
                similarity = 1.0 / (float(distances[int(neighbor_index)]) + 1e-6)
                total_vote += similarity
                for entity_id in labels:
                    vote_scores[entity_id] = vote_scores.get(entity_id, 0.0) + similarity

        if not vote_scores:
            continue
        max_vote = max(vote_scores.values())
        support_factor = _cluster_support_factor(len(pool))
        confidence = float(max_vote / max(total_vote, 1e-6)) * max(consensus, 1e-6) * support_factor
        if confidence < float(config.pseudo_min_confidence):
            continue
        labels = [
            entity_id
            for entity_id, score in vote_scores.items()
            if score >= (float(config.multi_positive_margin) * max_vote)
        ]
        labels = sorted(labels) or [max(vote_scores.items(), key=lambda item: item[1])[0]]
        label_limit = _resolve_pseudo_entity_limit(
            label_signature=labels,
            queried_positive_counts=queried_positive_counts,
            config=config,
        )
        if label_limit is not None:
            if any(
                int(pseudo_positive_counts.get(entity_id, 0)) >= label_limit
                for entity_id in labels
            ):
                continue
        window_id = str(row.window_id)
        pseudo_labels[window_id] = labels
        pseudo_confidence[window_id] = confidence
        for entity_id in labels:
            pseudo_positive_counts[entity_id] += 1

    return pseudo_labels, pseudo_confidence


def _clone_query_plan_with_pseudo_labels(
    query_plan: QueryPlan,
    pseudo_labels: Mapping[str, Sequence[str]],
    pseudo_confidence: Mapping[str, float],
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> QueryPlan:
    metadata = dict(query_plan.metadata)
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return replace(
        query_plan,
        pseudo_labels={
            str(key): list(value)
            for key, value in pseudo_labels.items()
        },
        pseudo_confidence={
            str(key): float(value)
            for key, value in pseudo_confidence.items()
        },
        metadata=metadata,
    )


def _query_plan_cluster_consensus(
    query_plan: QueryPlan,
) -> Mapping[int, float]:
    by_cluster: DefaultDict[int, List[str]] = defaultdict(list)
    for window_id in query_plan.queried_window_ids:
        cluster_id = int(query_plan.window_clusters.get(str(window_id), query_plan.normal_cluster_id))
        by_cluster[cluster_id].append(str(window_id))
    consensus = {}
    for cluster_id, window_ids in by_cluster.items():
        cluster_rows = pd.DataFrame({"window_id": window_ids})
        consensus[int(cluster_id)] = _cluster_label_consensus(
            cluster_rows=cluster_rows,
            true_labels=query_plan.queried_labels,
        )
    return consensus


def propagate_model_pseudo_labels(
    scored_rows: pd.DataFrame,
    query_plan: QueryPlan,
    config: Optional[SemiSupervisedConfig] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    config = config or SemiSupervisedConfig()
    propagate_mode = str(config.propagate_mode).lower()
    if propagate_mode not in ("self_train_top1", "self_train_cluster_top1"):
        return {}, {}
    if scored_rows.empty or not query_plan.queried_window_ids:
        return {}, {}

    rows = scored_rows.copy()
    rows["cluster_id"] = rows["window_id"].astype(str).map(
        lambda value: int(query_plan.window_clusters.get(str(value), query_plan.normal_cluster_id))
    )
    if bool(config.pseudo_fault_only):
        rows = rows[rows["window_kind"].astype(str) == "fault"].copy()
    if not bool(config.pseudo_include_normal_cluster_faults):
        rows = rows[rows["cluster_id"].astype(int) != int(query_plan.normal_cluster_id)].copy()
    if rows.empty:
        return {}, {}

    queried_set = {str(item) for item in query_plan.queried_window_ids}
    unlabeled = rows[~rows["window_id"].astype(str).isin(queried_set)].copy()
    if unlabeled.empty:
        return {}, {}

    queried_positive_counts = _collect_positive_label_counts(query_plan.queried_labels)
    pseudo_positive_counts: Counter = Counter()
    cluster_consensus = _query_plan_cluster_consensus(query_plan)
    cluster_label_support: DefaultDict[int, set] = defaultdict(set)
    cluster_query_counts: Counter = Counter()
    for window_id, labels in query_plan.queried_labels.items():
        cluster_id = int(query_plan.window_clusters.get(str(window_id), query_plan.normal_cluster_id))
        cluster_query_counts[cluster_id] += 1
        for entity_id in labels:
            cluster_label_support[cluster_id].add(str(entity_id))

    candidates: List[Tuple[str, List[str], float]] = []
    for window_id, group in unlabeled.groupby("window_id", sort=False):
        ordered = group.sort_values(by=["score", "raw_score"], ascending=False).reset_index(drop=True)
        if ordered.empty:
            continue
        cluster_id = int(ordered.iloc[0]["cluster_id"])
        top1 = ordered.iloc[0]
        top1_entity_id = str(top1["entity_id"])
        top1_score = float(top1.get("score", top1.get("raw_score", 0.0)))
        second_score = (
            float(ordered.iloc[1].get("score", ordered.iloc[1].get("raw_score", 0.0)))
            if len(ordered) > 1
            else 0.0
        )
        margin = top1_score - second_score
        if top1_score < float(config.pseudo_selftrain_min_score):
            continue
        if margin < float(config.pseudo_selftrain_min_margin):
            continue
        if propagate_mode == "self_train_cluster_top1":
            if top1_entity_id not in cluster_label_support.get(cluster_id, set()):
                continue

        support_factor = _cluster_support_factor(int(cluster_query_counts.get(cluster_id, 0)))
        consensus = float(cluster_consensus.get(cluster_id, 1.0))
        margin_scale = max(float(config.pseudo_selftrain_min_margin), 1e-6)
        margin_confidence = min(1.0, margin / margin_scale)
        confidence = float(top1_score * (0.5 + (0.5 * margin_confidence)) * max(consensus, 1e-6))
        if support_factor > 0.0:
            confidence *= support_factor
        if confidence < float(config.pseudo_min_confidence):
            continue

        label_limit = _resolve_pseudo_entity_limit(
            label_signature=[top1_entity_id],
            queried_positive_counts=queried_positive_counts,
            config=config,
        )
        if label_limit is not None and int(pseudo_positive_counts.get(top1_entity_id, 0)) >= label_limit:
            continue
        candidates.append((str(window_id), [top1_entity_id], confidence))

    if not candidates:
        return {}, {}

    global_cap = int(config.pseudo_selftrain_global_cap)
    if global_cap <= 0:
        global_cap = len(candidates)
        if int(config.pseudo_windows_per_queried) > 0:
            global_cap = min(global_cap, len(queried_set) * int(config.pseudo_windows_per_queried))
    global_cap = max(0, min(global_cap, len(candidates)))

    pseudo_labels = {}
    pseudo_confidence = {}
    for window_id, labels, confidence in sorted(
        candidates,
        key=lambda item: (-float(item[2]), item[0]),
    )[:global_cap]:
        pseudo_labels[window_id] = list(labels)
        pseudo_confidence[window_id] = float(confidence)
        for entity_id in labels:
            pseudo_positive_counts[entity_id] += 1
    return pseudo_labels, pseudo_confidence


def build_query_plan(
    tables: FeatureBundleTables,
    budget: int,
    config: Optional[SemiSupervisedConfig] = None,
    random_state: int = 42,
    embedding_frame_override: Optional[pd.DataFrame] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[QueryPlan, pd.DataFrame]:
    config = config or SemiSupervisedConfig()
    supervision_mode = _resolve_supervision_mode(config)
    has_relative_columns = any(
        str(column).startswith(("rel_win_z_", "rel_win_rank_", "rel_type_rank_"))
        for column in tables.entity_features.columns
    )
    if has_relative_columns:
        prepared_tables = tables
        selected_feature_columns = list(prepared_tables.feature_columns)
    else:
        base_feature_columns = resolve_feature_columns(tables.feature_columns, config)
        entity_feature_baselines = build_entity_feature_baselines(
            entity_features=tables.entity_features,
            base_feature_columns=base_feature_columns,
            config=config,
        )
        prepared_tables = prepare_feature_bundle_tables_for_model(
            tables=tables,
            base_feature_columns=base_feature_columns,
            config=config,
            entity_feature_baselines=entity_feature_baselines,
        )
        selected_feature_columns = list(prepared_tables.feature_columns)
    embedding_frame = (
        embedding_frame_override.copy()
        if embedding_frame_override is not None
        else build_window_embedding_table(prepared_tables, selected_feature_columns)
    )
    clustered = cluster_windows(embedding_frame, config=config, random_state=random_state)
    true_labels = _window_label_map(prepared_tables.windows)
    if supervision_mode == "oracle_full":
        queried_window_ids = _ordered_fault_window_ids(prepared_tables.windows, true_labels)
        queried_roles = {window_id: "oracle_full" for window_id in queried_window_ids}
        queried_labels = {
            window_id: list(true_labels.get(window_id, []))
            for window_id in queried_window_ids
        }
        pseudo_labels, pseudo_confidence = {}, {}
    else:
        query_selection = select_query_windows_with_roles(
            clustered_windows=clustered,
            budget=budget,
            random_state=random_state,
            query_strategy=config.query_strategy,
            true_labels=true_labels,
            config=config,
        )
        queried_window_ids = [window_id for window_id, _ in query_selection]
        queried_roles = {window_id: role for window_id, role in query_selection}
        queried_labels = {
            window_id: list(true_labels.get(window_id, []))
            for window_id in queried_window_ids
        }
        propagate_mode = str(config.propagate_mode).lower()
        if (
            supervision_mode == "oracle_budget"
            or propagate_mode in ("self_train_top1", "self_train_cluster_top1")
        ):
            pseudo_labels, pseudo_confidence = {}, {}
        else:
            pseudo_labels, pseudo_confidence = propagate_pseudo_labels(
                clustered_windows=clustered,
                queried_window_ids=queried_window_ids,
                queried_roles=queried_roles,
                true_labels=queried_labels,
                config=config,
            )
    plan = QueryPlan(
        dataset=tables.dataset,
        normal_cluster_id=int(clustered.attrs.get("normal_cluster_id", 0)),
        window_clusters={
            str(row.window_id): int(row.cluster_id)
            for row in clustered.itertuples(index=False)
        },
        queried_window_ids=list(queried_window_ids),
        queried_roles=queried_roles,
        queried_labels=queried_labels,
        pseudo_labels=pseudo_labels,
        pseudo_confidence=pseudo_confidence,
        metadata={
            "cluster_sizes": dict(clustered.attrs.get("cluster_sizes", {})),
            "silhouette_score": float(clustered.attrs.get("silhouette_score", -1.0)),
            "noise_count": int(clustered.attrs.get("noise_count", 0)),
            "supervision_mode": supervision_mode,
            "budget": budget,
            "fault_window_count": len(_ordered_fault_window_ids(prepared_tables.windows, true_labels)),
            "effective_queried_window_count": len(queried_window_ids),
            "embedding_count": len(clustered),
            "embedding_source": (
                "override" if embedding_frame_override is not None else "feature_aggregate"
            ),
            "config": config.to_dict(),
            "selected_feature_columns": list(selected_feature_columns),
            "queried_role_counts": {
                role: sum(1 for value in queried_roles.values() if value == role)
                for role in sorted(set(queried_roles.values()))
            },
            **dict(extra_metadata or {}),
            "clustering_random_state": int(random_state),
        },
    )
    return plan, clustered


def build_training_frame(
    tables: FeatureBundleTables,
    query_plan: QueryPlan,
    config: Optional[SemiSupervisedConfig] = None,
    frozen_pseudo_result: Optional[StrategyResult] = None,
    pseudo_candidate_rankings: Optional[
        Mapping[str, Mapping[str, Sequence[str]]]
    ] = None,
    pseudo_adapter_config: Optional[PseudoTrainingAdapterConfig] = None,
) -> pd.DataFrame:
    config = config or SemiSupervisedConfig()
    normal_training_policy = _resolve_normal_training_policy(config)
    matched_manifest = query_plan.metadata.get("matched_pseudo_arm")
    if matched_manifest is not None:
        if not isinstance(matched_manifest, Mapping):
            raise ValueError("matched_pseudo_arm metadata must be a mapping")
        adapted = adapt_matched_pseudo_training_rows(
            entity_features=tables.entity_features,
            queried_labels=query_plan.queried_labels,
            matched_arm=matched_manifest,
        )
        adapted["cluster_id"] = adapted["window_id"].astype(str).map(
            lambda window_id: int(
                query_plan.window_clusters.get(
                    window_id,
                    query_plan.normal_cluster_id,
                )
            )
        )
        if normal_training_policy == "fault_only":
            adapted = adapted[
                adapted["window_kind"].astype(str) != "normal"
            ].copy()
        return adapted[adapted["label"] >= 0].reset_index(drop=True)
    serialized_manifest = query_plan.metadata.get("frozen_pseudo_supervision")
    restored_frozen_result = None
    restored_rankings = None
    if serialized_manifest is not None:
        if not isinstance(serialized_manifest, Mapping):
            raise ValueError("frozen_pseudo_supervision metadata must be a mapping")
        restored_frozen_result = StrategyResult.from_dict(serialized_manifest)
        restored_rankings = serialized_manifest.get("candidate_rankings", {})
        if (
            query_plan.frozen_pseudo_result is not None
            and query_plan.frozen_pseudo_result.to_dict()
            != restored_frozen_result.to_dict()
        ):
            raise ValueError("runtime frozen pseudo result does not match serialized manifest")
    resolved_frozen_result = (
        frozen_pseudo_result
        if frozen_pseudo_result is not None
        else (
            query_plan.frozen_pseudo_result
            if query_plan.frozen_pseudo_result is not None
            else restored_frozen_result
        )
    )
    if resolved_frozen_result is not None:
        resolved_rankings = (
            pseudo_candidate_rankings
            if pseudo_candidate_rankings is not None
            else (
                query_plan.pseudo_candidate_rankings
                if query_plan.pseudo_candidate_rankings
                else (restored_rankings or {})
            )
        )
        resolved_adapter_config = pseudo_adapter_config or PseudoTrainingAdapterConfig(
            pseudo_positive_weight=float(config.pseudo_positive_weight),
            pseudo_negative_weight=float(config.pseudo_negative_weight),
            consensus_bottom_k=int(config.pseudo_consensus_negative_bottom_k),
            minimum_negative_agreement=int(
                config.pseudo_consensus_negative_min_agreement
            ),
        )
        adapted = adapt_pseudo_training_rows(
            entity_features=tables.entity_features,
            queried_labels=query_plan.queried_labels,
            strategy_result=resolved_frozen_result,
            candidate_rankings=resolved_rankings,
            config=resolved_adapter_config,
        )
        adapted["cluster_id"] = adapted["window_id"].astype(str).map(
            lambda window_id: int(
                query_plan.window_clusters.get(
                    window_id,
                    query_plan.normal_cluster_id,
                )
            )
        )
        if normal_training_policy == "fault_only":
            adapted = adapted[
                adapted["window_kind"].astype(str) != "normal"
            ].copy()
        return adapted[adapted["label"] >= 0].reset_index(drop=True)
    rows = tables.entity_features.copy()
    queried_set = set(query_plan.queried_window_ids)
    pseudo_labels = {key: set(value) for key, value in query_plan.pseudo_labels.items()}
    queried_labels = {key: set(value) for key, value in query_plan.queried_labels.items()}
    pseudo_negative_mode = str(config.pseudo_negative_mode).lower()
    pseudo_negative_exclusions: Dict[str, set] = {}

    if (
        pseudo_labels
        and pseudo_negative_mode == "exclude_top_anomaly"
        and int(config.pseudo_negative_exclusion_top_k) > 0
    ):
        anomaly_columns = _select_anomaly_feature_columns(list(rows.columns))
        if anomaly_columns:
            anomaly_strength = rows[anomaly_columns].fillna(0.0).abs().sum(axis=1)
        else:
            anomaly_strength = pd.Series(np.zeros(len(rows), dtype=float), index=rows.index)
        rows["_pseudo_anomaly_strength"] = anomaly_strength
        for window_id, group in rows.groupby("window_id", sort=False):
            window_key = str(window_id)
            positives = pseudo_labels.get(window_key, set())
            if not positives:
                continue
            candidates = group[
                ~group["entity_id"].astype(str).isin(positives)
            ].copy()
            if candidates.empty:
                continue
            excluded_ids = (
                candidates.sort_values(
                    by=["_pseudo_anomaly_strength", "entity_id"],
                    ascending=[False, True],
                )
                .head(int(config.pseudo_negative_exclusion_top_k))["entity_id"]
                .astype(str)
                .tolist()
            )
            pseudo_negative_exclusions[window_key] = set(excluded_ids)

    labels = []
    weights = []
    sources = []
    cluster_ids = []

    for row in rows.itertuples(index=False):
        window_id = str(row.window_id)
        entity_id = str(row.entity_id)
        cluster_id = int(query_plan.window_clusters.get(window_id, query_plan.normal_cluster_id))
        cluster_ids.append(cluster_id)

        if row.window_kind == "normal":
            if normal_training_policy == "fault_only":
                labels.append(-1)
                weights.append(0.0)
                sources.append("normal_window_excluded")
            else:
                labels.append(0)
                weights.append(1.0)
                sources.append("normal_window")
            continue

        if window_id in queried_set:
            positives = queried_labels.get(window_id, set())
            labels.append(1 if entity_id in positives else 0)
            weights.append(1.0)
            sources.append("queried")
            continue

        if cluster_id == query_plan.normal_cluster_id:
            if bool(config.label_faults_in_normal_cluster):
                labels.append(0)
                weights.append(float(config.auto_normal_weight))
                sources.append("auto_normal_cluster")
            else:
                labels.append(-1)
                weights.append(0.0)
                sources.append("normal_cluster_skipped")
            continue

        if window_id in pseudo_labels:
            positives = pseudo_labels.get(window_id, set())
            confidence = float(query_plan.pseudo_confidence.get(window_id, 0.5))
            if entity_id in positives:
                labels.append(1)
                weights.append(confidence * float(config.pseudo_positive_weight))
                sources.append("pseudo")
                continue
            if pseudo_negative_mode == "none":
                labels.append(-1)
                weights.append(0.0)
                sources.append("pseudo_negative_skipped")
                continue
            if entity_id in pseudo_negative_exclusions.get(window_id, set()):
                labels.append(-1)
                weights.append(0.0)
                sources.append("pseudo_negative_skipped")
                continue
            labels.append(0)
            if float(config.pseudo_negative_weight) > 0.0:
                weights.append(confidence * float(config.pseudo_negative_weight))
            else:
                weights.append(0.0)
            sources.append("pseudo")
            continue

        labels.append(-1)
        weights.append(0.0)
        sources.append("unlabeled")

    rows["label"] = labels
    rows["sample_weight"] = weights
    rows["label_source"] = sources
    rows["cluster_id"] = cluster_ids
    output = rows[rows["label"] >= 0].reset_index(drop=True)
    if "_pseudo_anomaly_strength" in output.columns:
        output = output.drop(columns=["_pseudo_anomaly_strength"])
    return output


def build_unified_adjacency(feature_root: Path, dataset: str) -> np.ndarray:
    graph_root = feature_root / dataset / "graph"
    entity_index = load_entity_index(graph_root / "entity_index.json")
    topology = load_topology_bundle(graph_root / "topology.json")
    size = len(entity_index.entity_to_index)
    matrix = np.zeros((size, size), dtype=float)

    def _add_edge(entity_type_a: str, name_a: str, entity_type_b: str, name_b: str, weight: float) -> None:
        source_id = "%s:%s" % (entity_type_a, name_a)
        target_id = "%s:%s" % (entity_type_b, name_b)
        if source_id not in entity_index.entity_to_index or target_id not in entity_index.entity_to_index:
            return
        source_index = entity_index.entity_to_index[source_id]
        target_index = entity_index.entity_to_index[target_id]
        matrix[source_index, target_index] += weight
        matrix[target_index, source_index] += weight

    for (source, target), weight in topology.service_service_edges.items():
        _add_edge("service", source, "service", target, float(weight))
    for (source, target), weight in topology.service_host_edges.items():
        _add_edge("service", source, "host", target, float(weight))
    for (source, target), weight in topology.host_host_edges.items():
        _add_edge("host", source, "host", target, float(weight))

    matrix += np.eye(size, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums


def train_ranker(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int = 42,
) -> Any:
    if training_frame.empty:
        return ConstantProbabilityClassifier(probability=0.0)
    unique_labels = sorted(training_frame["label"].astype(int).unique().tolist())
    if len(unique_labels) < 2:
        probability = 1.0 if unique_labels and unique_labels[0] == 1 else 0.0
        return ConstantProbabilityClassifier(probability=probability)

    classifier = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(
        training_frame[list(feature_columns)].fillna(0.0).values,
        training_frame["label"].astype(int).values,
        sample_weight=training_frame["sample_weight"].astype(float).values,
    )
    return classifier


def _train_backend_classifier(
    backend: str,
    tables: FeatureBundleTables,
    training_frame: pd.DataFrame,
    feature_root: Path,
    selected_feature_columns: Sequence[str],
    config: SemiSupervisedConfig,
    random_state: int,
    graph_bundle: Optional[Any] = None,
) -> Any:
    backend = str(backend).lower()
    if backend == "hgcn":
        from .hgcn_backend import train_hgcn_classifier

        return train_hgcn_classifier(
            tables=tables,
            training_frame=training_frame,
            feature_root=feature_root,
            selected_feature_columns=selected_feature_columns,
            config=config,
            random_state=random_state,
            pretrained_bundle=graph_bundle,
        )
    if backend == "pairwise_tree":
        from .pairwise_backend import train_pairwise_tree_ranker

        return train_pairwise_tree_ranker(
            training_frame=training_frame,
            feature_columns=selected_feature_columns,
            random_state=random_state,
            negative_top_k=int(config.pairwise_negative_top_k),
        )
    if backend == "pairwise_linear":
        from .pairwise_backend import train_pairwise_linear_ranker

        return train_pairwise_linear_ranker(
            training_frame=training_frame,
            feature_columns=selected_feature_columns,
            random_state=random_state,
            negative_top_k=int(config.pairwise_negative_top_k),
        )
    if backend == "set_mass_linear_v1":
        from .set_mass_backend import fit_set_mass_linear

        return fit_set_mass_linear(
            training_frame,
            feature_columns=selected_feature_columns,
            epochs=200,
            learning_rate=0.01,
            weight_decay=0.0001,
            seed=42,
            pseudo_to_query_loss_ratio=0.25,
        )
    return train_ranker(training_frame, selected_feature_columns, random_state=random_state)


def fit_semisupervised_ranker(
    feature_root: Path,
    dataset: str,
    budget: int,
    random_state: int = 42,
    diffusion_alpha: float = 0.75,
    diffusion_steps: int = 1,
    model_config: Optional[SemiSupervisedConfig] = None,
    query_plan_override: Optional[QueryPlan] = None,
    clustered_windows_override: Optional[pd.DataFrame] = None,
) -> Tuple[SemiSupervisedModel, pd.DataFrame, pd.DataFrame]:
    config = model_config or SemiSupervisedConfig(
        diffusion_alpha=diffusion_alpha,
        diffusion_steps=diffusion_steps,
    )
    tables = load_feature_bundle_tables(feature_root, dataset)
    return fit_semisupervised_ranker_on_tables(
        tables=tables,
        feature_root=feature_root,
        budget=budget,
        random_state=random_state,
        diffusion_alpha=config.diffusion_alpha,
        diffusion_steps=config.diffusion_steps,
        model_config=config,
        query_plan_override=query_plan_override,
        clustered_windows_override=clustered_windows_override,
    )


def _validate_query_plan_override(
    query_plan: QueryPlan,
    tables: FeatureBundleTables,
    clustered_windows: pd.DataFrame,
    config: SemiSupervisedConfig,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> QueryPlan:
    if not isinstance(query_plan, QueryPlan):
        raise TypeError("query_plan_override must be a QueryPlan")
    if str(query_plan.dataset) != str(tables.dataset):
        raise ValueError("persisted query plan and feature tables use different datasets")
    supervision_mode = _resolve_supervision_mode(config)
    if supervision_mode == "oracle_full":
        raise ValueError("oracle-full training cannot use a budgeted query plan override")
    if str(config.propagate_mode).lower() in ("self_train_top1", "self_train_cluster_top1"):
        raise ValueError("persisted query plan cannot be combined with runtime self-training")

    windows = tables.windows.copy()
    known_window_ids = set(windows["window_id"].astype(str))
    queried_window_ids = tuple(str(item) for item in query_plan.queried_window_ids)
    if len(queried_window_ids) != len(set(queried_window_ids)):
        raise ValueError("persisted query plan contains duplicate queried window IDs")
    unknown_queries = sorted(set(queried_window_ids).difference(known_window_ids))
    if unknown_queries:
        raise ValueError("persisted query plan contains unknown queried windows: %s" % unknown_queries)
    window_kind_by_id = {
        str(row.window_id): str(row.window_kind)
        for row in windows.itertuples(index=False)
    }
    non_fault_queries = sorted(
        window_id
        for window_id in queried_window_ids
        if window_kind_by_id.get(window_id) != "fault"
    )
    if non_fault_queries:
        raise ValueError("persisted query plan may query fault windows only: %s" % non_fault_queries)
    if set(str(item) for item in query_plan.queried_labels) != set(queried_window_ids):
        raise ValueError("persisted queried labels must match queried window IDs exactly")
    if any(not tuple(values) for values in query_plan.queried_labels.values()):
        raise ValueError("persisted queried fault labels must not be empty")

    expected_clusters = {
        str(row.window_id): int(row.cluster_id)
        for row in clustered_windows.itertuples(index=False)
    }
    persisted_clusters = {
        str(window_id): int(cluster_id)
        for window_id, cluster_id in query_plan.window_clusters.items()
    }
    if set(persisted_clusters) != known_window_ids:
        missing = sorted(known_window_ids.difference(persisted_clusters))
        extra = sorted(set(persisted_clusters).difference(known_window_ids))
        raise ValueError(
            "persisted query plan cluster coverage mismatch: missing=%s extra=%s"
            % (missing, extra)
        )
    if persisted_clusters != expected_clusters:
        raise ValueError("persisted query plan clusters do not match frozen preprocessing")
    normal_cluster_id = int(clustered_windows.attrs.get("normal_cluster_id", 0))
    if int(query_plan.normal_cluster_id) != normal_cluster_id:
        raise ValueError("persisted normal cluster does not match frozen preprocessing")

    frozen_result = query_plan.frozen_pseudo_result
    if frozen_result is not None:
        if frozen_result.dataset != tables.dataset:
            raise ValueError("frozen pseudo result and feature tables use different datasets")
        if tuple(frozen_result.queried_window_ids) != queried_window_ids:
            raise ValueError("frozen pseudo result changed the persisted queried IDs")
    metadata = dict(query_plan.metadata)
    metadata.update(
        {
            "query_plan_source": "persisted_override",
            "supervision_mode": supervision_mode,
            **dict(extra_metadata or {}),
        }
    )
    return replace(query_plan, metadata=metadata)


def _resolve_query_plan_clustering_random_state(
    query_plan: QueryPlan,
    fallback: int,
) -> int:
    if not isinstance(query_plan, QueryPlan):
        raise TypeError("query_plan_override must be a QueryPlan")
    metadata = query_plan.metadata
    if "clustering_random_state" not in metadata:
        return int(fallback)
    value = metadata["clustering_random_state"]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            "persisted query plan clustering_random_state must be an integer"
        )
    return int(value)


def fit_semisupervised_ranker_on_tables(
    tables: FeatureBundleTables,
    feature_root: Path,
    budget: int,
    random_state: int = 42,
    diffusion_alpha: float = 0.75,
    diffusion_steps: int = 1,
    model_config: Optional[SemiSupervisedConfig] = None,
    query_plan_override: Optional[QueryPlan] = None,
    clustered_windows_override: Optional[pd.DataFrame] = None,
) -> Tuple[SemiSupervisedModel, pd.DataFrame, pd.DataFrame]:
    if (
        clustered_windows_override is not None
        and query_plan_override is None
    ):
        raise ValueError(
            "clustered_windows_override requires query_plan_override"
        )
    config = model_config or SemiSupervisedConfig(
        diffusion_alpha=diffusion_alpha,
        diffusion_steps=diffusion_steps,
    )
    _resolve_supervision_mode(config)
    _resolve_normal_training_policy(config)
    base_feature_columns = resolve_feature_columns(tables.feature_columns, config)
    entity_feature_baselines = build_entity_feature_baselines(
        entity_features=tables.entity_features,
        base_feature_columns=base_feature_columns,
        config=config,
    )
    prepared_tables = prepare_feature_bundle_tables_for_model(
        tables=tables,
        base_feature_columns=base_feature_columns,
        config=config,
        entity_feature_baselines=entity_feature_baselines,
    )
    selected_feature_columns = list(prepared_tables.feature_columns)
    gated_onehop_topology = None
    gated_onehop_relations = tuple(
        str(value)
        for value in getattr(config, "gated_onehop_relations", ())
    )
    if gated_onehop_relations:
        gated_onehop_topology = load_topology_bundle(
            feature_root / tables.dataset / "graph" / "topology.json"
        )
    graph_bundle = None
    cluster_representation = str(getattr(config, "cluster_representation", "feature_aggregate")).lower()
    if (
        cluster_representation in ("hgcn_autoencoder", "hybrid_hgcn_anomaly_signature")
        or bool(getattr(config, "use_graph_context_features", False))
        or str(config.model_backend).lower() == "hgcn"
    ):
        from .hgcn_backend import build_window_hgcn_embeddings

        graph_bundle = build_window_hgcn_embeddings(
            tables=prepared_tables,
            feature_root=feature_root,
            selected_feature_columns=selected_feature_columns,
            config=config,
            random_state=random_state,
        )
    embedding_frame = resolve_cluster_embedding_table(
        tables=prepared_tables,
        feature_columns=selected_feature_columns,
        cluster_representation=cluster_representation,
        graph_embedding_frame=(graph_bundle.embedding_frame if graph_bundle is not None else None),
    )
    graph_metadata = (
        {"graph_pretraining": dict(graph_bundle.metadata)}
        if graph_bundle is not None
        else None
    )
    if query_plan_override is None:
        query_plan, clustered_windows = build_query_plan(
            prepared_tables,
            budget,
            config=config,
            random_state=random_state,
            embedding_frame_override=embedding_frame,
            extra_metadata=graph_metadata,
        )
    else:
        if clustered_windows_override is None:
            clustering_random_state = (
                _resolve_query_plan_clustering_random_state(
                    query_plan=query_plan_override,
                    fallback=random_state,
                )
            )
            clustered_windows = cluster_windows(
                embedding_frame,
                config=config,
                random_state=clustering_random_state,
            )
        else:
            if not isinstance(clustered_windows_override, pd.DataFrame):
                raise TypeError(
                    "clustered_windows_override must be a pandas DataFrame"
                )
            clustered_windows = clustered_windows_override.copy()
        query_plan = _validate_query_plan_override(
            query_plan=query_plan_override,
            tables=prepared_tables,
            clustered_windows=clustered_windows,
            config=config,
            extra_metadata=graph_metadata,
        )
    model_tables = prepared_tables
    if gated_onehop_topology is not None:
        model_tables = augment_prepared_feature_bundle_with_gated_onehop(
            tables=model_tables,
            topology=gated_onehop_topology,
            source_feature_columns=base_feature_columns,
            config=config,
        )
    if graph_bundle is not None and bool(getattr(config, "use_graph_context_features", False)):
        from .hgcn_backend import build_entity_graph_context_features

        graph_context = build_entity_graph_context_features(
            entity_features=prepared_tables.entity_features,
            bundle=graph_bundle,
            config=config,
        )
        model_tables = augment_prepared_feature_bundle_with_graph_context(
            tables=model_tables,
            graph_context=graph_context,
            config=config,
        )
    selected_feature_columns = list(model_tables.feature_columns)
    backend = str(config.model_backend).lower()
    if str(config.propagate_mode).lower() in ("self_train_top1", "self_train_cluster_top1"):
        seed_training_frame = build_training_frame(model_tables, query_plan, config=config)
        seed_classifier = _train_backend_classifier(
            backend=backend,
            tables=tables,
            training_frame=seed_training_frame,
            feature_root=feature_root,
            selected_feature_columns=selected_feature_columns,
            config=config,
            random_state=random_state,
            graph_bundle=graph_bundle,
        )
        seed_model = SemiSupervisedModel(
            dataset=tables.dataset,
            feature_columns=selected_feature_columns,
            base_feature_columns=base_feature_columns,
            classifier=seed_classifier,
            adjacency=None,
            diffusion_alpha=float(config.diffusion_alpha),
            diffusion_steps=int(config.diffusion_steps),
            query_plan=query_plan,
            config=config,
            entity_feature_baselines=entity_feature_baselines,
            entity_score_calibration={},
            score_calibration_mode="none",
            prototype_rerank_state=None,
            prototype_rerank_mode="none",
            graph_context_bundle=(
                graph_bundle if bool(getattr(config, "use_graph_context_features", False)) else None
            ),
            gated_onehop_topology=gated_onehop_topology,
            metadata={
                "stage": "seed_self_train",
                "training_row_count": len(seed_training_frame),
                "gated_onehop": dict(
                    model_tables.metadata.get("gated_onehop", {})
                ),
            },
        )
        seed_calibration_mode = _resolve_entity_score_calibration_mode(config, query_plan)
        if seed_calibration_mode != "none":
            seed_train_scored = seed_model.score_entity_features(tables.entity_features)
            seed_model.entity_score_calibration = _build_entity_score_calibration(
                scored_rows=seed_train_scored,
                min_std=float(config.entity_score_calibration_min_std),
            )
            seed_model.score_calibration_mode = seed_calibration_mode
        seed_scored_rows = seed_model.score_entity_features(tables.entity_features)
        model_pseudo_labels, model_pseudo_confidence = propagate_model_pseudo_labels(
            scored_rows=seed_scored_rows,
            query_plan=query_plan,
            config=config,
        )
        if model_pseudo_labels:
            query_plan = _clone_query_plan_with_pseudo_labels(
                query_plan,
                pseudo_labels=model_pseudo_labels,
                pseudo_confidence=model_pseudo_confidence,
                extra_metadata={
                    "pseudo_generation_mode": str(config.propagate_mode),
                    "seed_training_row_count": len(seed_training_frame),
                    "seed_pseudo_window_count": len(model_pseudo_labels),
                },
            )

    training_frame = build_training_frame(model_tables, query_plan, config=config)
    classifier = _train_backend_classifier(
        backend=backend,
        tables=tables,
        training_frame=training_frame,
        feature_root=feature_root,
        selected_feature_columns=selected_feature_columns,
        config=config,
        random_state=random_state,
        graph_bundle=graph_bundle,
    )
    adjacency = None
    if backend == "tree" and config.use_graph_diffusion and int(config.diffusion_steps) > 0:
        adjacency = build_unified_adjacency(feature_root, tables.dataset)
    calibration_mode = _resolve_entity_score_calibration_mode(config, query_plan)
    prototype_rerank_mode = _resolve_prototype_rerank_mode(config, query_plan)
    prototype_rerank_state = None
    if prototype_rerank_mode != "none" and float(config.prototype_rerank_alpha) != 0.0:
        prototype_rerank_state = _build_prototype_rerank_state(
            training_frame=training_frame,
            feature_columns=selected_feature_columns,
        )
        if prototype_rerank_state is None:
            prototype_rerank_mode = "none"
    model = SemiSupervisedModel(
        dataset=tables.dataset,
        feature_columns=selected_feature_columns,
        base_feature_columns=base_feature_columns,
        classifier=classifier,
        adjacency=adjacency,
        diffusion_alpha=float(config.diffusion_alpha),
        diffusion_steps=int(config.diffusion_steps),
        query_plan=query_plan,
        config=config,
        entity_feature_baselines=entity_feature_baselines,
        entity_score_calibration={},
        score_calibration_mode="none",
        prototype_rerank_state=prototype_rerank_state,
        prototype_rerank_mode=prototype_rerank_mode,
        graph_context_bundle=(
            graph_bundle if bool(getattr(config, "use_graph_context_features", False)) else None
        ),
        gated_onehop_topology=gated_onehop_topology,
        metadata={
            "budget": budget,
            "model_backend": str(config.model_backend),
            "feature_column_count": len(selected_feature_columns),
            "base_feature_column_count": len(base_feature_columns),
            "selected_feature_columns": list(selected_feature_columns),
            "selected_base_feature_columns": list(base_feature_columns),
            "training_row_count": len(training_frame),
            "queried_window_count": len(query_plan.queried_window_ids),
            "pseudo_window_count": len(query_plan.pseudo_labels),
            "label_source_counts": training_frame["label_source"].value_counts().to_dict(),
            "score_calibration_mode": calibration_mode,
            "prototype_rerank_mode": prototype_rerank_mode,
            "gated_onehop": dict(
                model_tables.metadata.get("gated_onehop", {})
            ),
            "config": config.to_dict(),
        },
        training_frame=training_frame.copy(),
    )
    if calibration_mode != "none":
        train_scored_for_calibration = model.score_entity_features(tables.entity_features)
        model.entity_score_calibration = _build_entity_score_calibration(
            scored_rows=train_scored_for_calibration,
            min_std=float(config.entity_score_calibration_min_std),
        )
        model.score_calibration_mode = calibration_mode
        model.metadata = {
            **dict(model.metadata),
            "score_calibration_entity_count": len(model.entity_score_calibration),
        }
    scored_rows = model.score_entity_features(tables.entity_features)
    return model, clustered_windows, scored_rows


def rank_scored_rows(scored_rows: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    rankings = {}
    for window_id, group in scored_rows.groupby("window_id", sort=False):
        ordered = group.sort_values(by=["score", "raw_score"], ascending=False)
        rankings[str(window_id)] = ordered[
            ["entity_id", "entity_name", "entity_type", "score", "raw_score"]
        ].to_dict("records")
    return rankings
