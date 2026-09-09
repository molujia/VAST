"""Heterogeneous graph backends for semi-supervised NexusRCL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv

from nexusrcl_rebuild.features.entities import load_entity_index
from nexusrcl_rebuild.features.topology import load_topology_bundle


WINDOW_META_COLUMNS = [
    "dataset",
    "window_id",
    "source_id",
    "window_kind",
    "day",
    "start_ts",
    "end_ts",
]

ANOMALY_SCORE_WEIGHTS = {
    "log_error_count": 0.75,
    "log_error_ratio": 0.50,
    "metric_abs_z_max": 1.00,
    "metric_event_score_sum": 1.50,
    "metric_event_score_max": 1.00,
    "metric_event_active_kpi_count": 0.75,
    "trace_error_count": 0.75,
    "trace_error_ratio": 0.75,
    "trace_latency_abs_z_max": 1.00,
    "trace_anomalous_operation_count": 1.00,
    "topology_change_count": 0.50,
}

ANOMALY_SCORE_PREFIX_WEIGHTS = {
    "metric_kpi_peak_": 1.25,
    "metric_kpi_hit_": 0.50,
    "trace_operation_z_": 1.00,
    "trace_peer_share_": 0.35,
}

DUMMY_HOST_ENTITY_ID = "host:__dummy__"


@dataclass(frozen=True)
class HeteroGraphSchema:
    dataset: str
    edge_dict: Mapping[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]
    num_nodes_dict: Mapping[str, int]
    service_ids: Sequence[str]
    host_ids: Sequence[str]
    service_index: Mapping[str, int]
    host_index: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowGraphSample:
    graph: dgl.DGLHeteroGraph
    window_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PretrainedGraphBundle:
    schema: HeteroGraphSchema
    service_feature_columns: Sequence[str]
    host_feature_columns: Sequence[str]
    service_feature_mean: Sequence[float]
    service_feature_std: Sequence[float]
    host_feature_mean: Sequence[float]
    host_feature_std: Sequence[float]
    embedding_frame: pd.DataFrame
    encoder_state_dict: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _hetero_relation_modules(
    canonical_etypes: Sequence[Tuple[str, str, str]],
    hidden_dim: int,
) -> Mapping[Tuple[str, str, str], GraphConv]:
    return {
        canonical_etype: GraphConv(
            hidden_dim,
            hidden_dim,
            allow_zero_in_degree=True,
        )
        for canonical_etype in canonical_etypes
    }


class HGCNEncoder(nn.Module):
    def __init__(
        self,
        service_input_dim: int,
        host_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        canonical_etypes: Sequence[Tuple[str, str, str]],
    ) -> None:
        super().__init__()
        self.service_input = nn.Linear(service_input_dim, hidden_dim)
        self.host_input = nn.Linear(host_input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                HeteroGraphConv(
                    _hetero_relation_modules(canonical_etypes, hidden_dim),
                    aggregate="sum",
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.service_skip = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.layers))]
        )
        self.host_skip = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.layers))]
        )
        self.dropout = float(dropout)

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        service_features: torch.Tensor,
        host_features: torch.Tensor,
        edge_weight_dict: Optional[Mapping[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = {
            "service": F.relu(self.service_input(service_features)),
            "host": F.relu(self.host_input(host_features)),
        }
        for layer_index, layer in enumerate(self.layers):
            mod_kwargs = None
            if edge_weight_dict:
                mod_kwargs = {
                    canonical_etype: {"edge_weight": edge_weight}
                    for canonical_etype, edge_weight in edge_weight_dict.items()
                }
            conv_hidden = layer(graph, hidden, mod_kwargs=mod_kwargs)
            service_conv = conv_hidden.get("service")
            if service_conv is None:
                service_conv = torch.zeros_like(hidden["service"])
            host_conv = conv_hidden.get("host")
            if host_conv is None:
                host_conv = torch.zeros_like(hidden["host"])
            next_hidden = {
                "service": F.relu(
                    service_conv + self.service_skip[layer_index](hidden["service"])
                ),
                "host": F.relu(
                    host_conv + self.host_skip[layer_index](hidden["host"])
                ),
            }
            if self.dropout > 0.0:
                next_hidden = {
                    key: F.dropout(value, p=self.dropout, training=self.training)
                    for key, value in next_hidden.items()
                }
            hidden = next_hidden
        return hidden


class HGCNNodeScorer(nn.Module):
    def __init__(
        self,
        service_input_dim: int,
        host_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        canonical_etypes: Sequence[Tuple[str, str, str]],
    ) -> None:
        super().__init__()
        self.encoder = HGCNEncoder(
            service_input_dim=service_input_dim,
            host_input_dim=host_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            canonical_etypes=canonical_etypes,
        )
        self.service_head = nn.Linear(hidden_dim, 1)
        self.host_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        service_features: torch.Tensor,
        host_features: torch.Tensor,
        edge_weight_dict: Optional[Mapping[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(
            graph,
            service_features,
            host_features,
            edge_weight_dict=edge_weight_dict,
        )
        return {
            "service": self.service_head(hidden["service"]).squeeze(-1),
            "host": self.host_head(hidden["host"]).squeeze(-1),
        }


class HGCNReconstructionModel(nn.Module):
    def __init__(
        self,
        service_input_dim: int,
        host_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        canonical_etypes: Sequence[Tuple[str, str, str]],
    ) -> None:
        super().__init__()
        self.encoder = HGCNEncoder(
            service_input_dim=service_input_dim,
            host_input_dim=host_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            canonical_etypes=canonical_etypes,
        )
        self.service_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, service_input_dim),
        )
        self.host_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, host_input_dim),
        )

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        service_features: torch.Tensor,
        host_features: torch.Tensor,
        edge_weight_dict: Optional[Mapping[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        hidden = self.encoder(
            graph,
            service_features,
            host_features,
            edge_weight_dict=edge_weight_dict,
        )
        reconstruction = {
            "service": self.service_decoder(hidden["service"]),
            "host": self.host_decoder(hidden["host"]),
        }
        return hidden, reconstruction


class HGCNClassifierWrapper:
    """Adapter exposing a score_frame method for SemiSupervisedModel."""

    def __init__(
        self,
        model: HGCNNodeScorer,
        schema: HeteroGraphSchema,
        service_feature_columns: Sequence[str],
        host_feature_columns: Sequence[str],
        service_feature_mean: Sequence[float],
        service_feature_std: Sequence[float],
        host_feature_mean: Sequence[float],
        host_feature_std: Sequence[float],
        device: torch.device,
        use_edge_refinement: bool,
        edge_refinement_scale: float,
    ) -> None:
        self.model = model
        self.schema = schema
        self.service_feature_columns = list(service_feature_columns)
        self.host_feature_columns = list(host_feature_columns)
        self.service_feature_mean = np.asarray(service_feature_mean, dtype=np.float32)
        self.service_feature_std = np.asarray(service_feature_std, dtype=np.float32)
        self.host_feature_mean = np.asarray(host_feature_mean, dtype=np.float32)
        self.host_feature_std = np.asarray(host_feature_std, dtype=np.float32)
        self.device = device
        self.use_edge_refinement = bool(use_edge_refinement)
        self.edge_refinement_scale = float(edge_refinement_scale)

    def score_frame(self, entity_features: pd.DataFrame) -> pd.DataFrame:
        rows = entity_features.copy()
        if rows.empty:
            rows["raw_score"] = pd.Series(dtype=float)
            rows["score"] = pd.Series(dtype=float)
            return rows

        self.model.eval()
        outputs = []
        with torch.no_grad():
            for window_id, group in rows.groupby("window_id", sort=False):
                service_features, host_features = _build_window_feature_tensors(
                    group,
                    self.schema,
                    self.service_feature_columns,
                    self.host_feature_columns,
                    service_feature_mean=self.service_feature_mean,
                    service_feature_std=self.service_feature_std,
                    host_feature_mean=self.host_feature_mean,
                    host_feature_std=self.host_feature_std,
                )
                graph = _instantiate_graph(self.schema)
                if self.use_edge_refinement:
                    _attach_edge_weights(
                        graph,
                        _build_window_edge_weights(
                            group,
                            self.schema,
                            scale=self.edge_refinement_scale,
                        ),
                    )
                graph = graph.to(self.device)
                edge_weight_dict = (
                    _graph_edge_weight_dict(graph) if self.use_edge_refinement else None
                )
                logits = self.model(
                    graph,
                    service_features.to(self.device),
                    host_features.to(self.device),
                    edge_weight_dict=edge_weight_dict,
                )
                service_scores = torch.sigmoid(logits["service"]).cpu().numpy()
                host_scores = torch.sigmoid(logits["host"]).cpu().numpy()

                service_map = {
                    entity_id: float(service_scores[index])
                    for entity_id, index in self.schema.service_index.items()
                }
                host_map = {
                    entity_id: float(host_scores[index])
                    for entity_id, index in self.schema.host_index.items()
                }

                scored_group = group.copy()
                raw_scores = []
                for row in scored_group.itertuples(index=False):
                    entity_id = str(row.entity_id)
                    if str(row.entity_type) == "service":
                        raw_scores.append(service_map.get(entity_id, 0.0))
                    else:
                        raw_scores.append(host_map.get(entity_id, 0.0))
                scored_group["raw_score"] = raw_scores
                scored_group["score"] = raw_scores
                outputs.append(scored_group)
        return pd.concat(outputs, axis=0, ignore_index=True)


def _edge_tensor_pair(
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(list(source_indices), dtype=torch.int64),
        torch.tensor(list(target_indices), dtype=torch.int64),
    )


def build_hetero_graph_schema(feature_root: Path, dataset: str) -> HeteroGraphSchema:
    graph_root = feature_root / dataset / "graph"
    entity_index = load_entity_index(graph_root / "entity_index.json")
    topology = load_topology_bundle(graph_root / "topology.json")

    service_ids = ["service:%s" % name for name in entity_index.services]
    real_host_ids = ["host:%s" % name for name in entity_index.hosts]
    uses_dummy_host = not real_host_ids
    host_ids = list(real_host_ids) or [DUMMY_HOST_ENTITY_ID]
    service_index = {entity_id: index for index, entity_id in enumerate(service_ids)}
    host_index = {entity_id: index for index, entity_id in enumerate(host_ids)}

    edge_dict: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    service_service_src = []
    service_service_dst = []
    for (source, target), _weight in topology.service_service_edges.items():
        source_id = "service:%s" % source
        target_id = "service:%s" % target
        if source_id in service_index and target_id in service_index:
            service_service_src.append(service_index[source_id])
            service_service_dst.append(service_index[target_id])
    edge_dict[("service", "service_service", "service")] = _edge_tensor_pair(
        service_service_src,
        service_service_dst,
    )

    service_host_src = []
    service_host_dst = []
    for (source, target), _weight in topology.service_host_edges.items():
        source_id = "service:%s" % source
        target_id = "host:%s" % target
        if source_id in service_index and target_id in host_index:
            service_host_src.append(service_index[source_id])
            service_host_dst.append(host_index[target_id])
    edge_dict[("service", "service_host", "host")] = _edge_tensor_pair(
        service_host_src,
        service_host_dst,
    )

    host_host_src = []
    host_host_dst = []
    for (source, target), _weight in topology.host_host_edges.items():
        source_id = "host:%s" % source
        target_id = "host:%s" % target
        if source_id in host_index and target_id in host_index:
            host_host_src.append(host_index[source_id])
            host_host_dst.append(host_index[target_id])
    edge_dict[("host", "host_host", "host")] = _edge_tensor_pair(
        host_host_src,
        host_host_dst,
    )

    return HeteroGraphSchema(
        dataset=dataset,
        edge_dict=edge_dict,
        num_nodes_dict={"service": len(service_ids), "host": len(host_ids)},
        service_ids=service_ids,
        host_ids=host_ids,
        service_index=service_index,
        host_index=host_index,
        metadata={
            "uses_dummy_host": uses_dummy_host,
            "real_host_count": len(real_host_ids),
            "relation_edge_counts": {
                canonical_etype[1]: int(len(src))
                for canonical_etype, (src, _dst) in edge_dict.items()
            }
        },
    )


def _instantiate_graph(schema: HeteroGraphSchema) -> dgl.DGLHeteroGraph:
    return dgl.heterograph(
        data_dict={
            relation: (src.clone(), dst.clone())
            for relation, (src, dst) in schema.edge_dict.items()
        },
        num_nodes_dict=dict(schema.num_nodes_dict),
    )


def _nonzero_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> List[str]:
    if frame.empty:
        return list(feature_columns)
    selected = []
    for column in feature_columns:
        series = frame[column]
        if float(series.abs().sum()) > 0.0:
            selected.append(column)
    return selected or list(feature_columns)


def _compute_feature_stats(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        mean = np.zeros(len(feature_columns), dtype=np.float32)
        std = np.ones(len(feature_columns), dtype=np.float32)
        return mean, std
    values = frame[list(feature_columns)].fillna(0.0).to_numpy(dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _build_window_feature_tensors(
    frame: pd.DataFrame,
    schema: HeteroGraphSchema,
    service_feature_columns: Sequence[str],
    host_feature_columns: Sequence[str],
    service_feature_mean: Optional[np.ndarray] = None,
    service_feature_std: Optional[np.ndarray] = None,
    host_feature_mean: Optional[np.ndarray] = None,
    host_feature_std: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    service_features = np.zeros(
        (len(schema.service_ids), len(service_feature_columns)),
        dtype=np.float32,
    )
    host_features = np.zeros(
        (len(schema.host_ids), len(host_feature_columns)),
        dtype=np.float32,
    )

    for row in frame.itertuples(index=False):
        entity_id = str(row.entity_id)
        if str(row.entity_type) == "service":
            index = schema.service_index.get(entity_id)
            if index is None:
                continue
            service_features[index, :] = np.asarray(
                [getattr(row, column) for column in service_feature_columns],
                dtype=np.float32,
            )
        else:
            index = schema.host_index.get(entity_id)
            if index is None:
                continue
            host_features[index, :] = np.asarray(
                [getattr(row, column) for column in host_feature_columns],
                dtype=np.float32,
            )

    if service_feature_mean is not None and service_feature_std is not None and service_features.size:
        service_features = (service_features - service_feature_mean) / service_feature_std
    if host_feature_mean is not None and host_feature_std is not None and host_features.size:
        host_features = (host_features - host_feature_mean) / host_feature_std
    service_features = np.clip(service_features, -10.0, 10.0)
    host_features = np.clip(host_features, -10.0, 10.0)
    return torch.from_numpy(service_features), torch.from_numpy(host_features)


def _row_value(payload: Any, column: str) -> float:
    if isinstance(payload, Mapping):
        value = payload.get(column)
    else:
        value = getattr(payload, column, None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _anomaly_score_weight(column: str) -> Optional[float]:
    if column in ANOMALY_SCORE_WEIGHTS:
        return float(ANOMALY_SCORE_WEIGHTS[column])
    for prefix, weight in ANOMALY_SCORE_PREFIX_WEIGHTS.items():
        if str(column).startswith(prefix):
            return float(weight)
    return None


def _payload_columns(payload: Any) -> Sequence[str]:
    if hasattr(payload, "_fields"):
        return list(getattr(payload, "_fields"))
    if isinstance(payload, Mapping):
        return [str(key) for key in payload.keys()]
    index = getattr(payload, "index", None)
    if index is not None:
        return [str(key) for key in index]
    return []


def _entity_anomaly_scalar(payload: Any) -> float:
    score = 0.0
    consumed = set()
    for column, weight in ANOMALY_SCORE_WEIGHTS.items():
        score += weight * max(0.0, _row_value(payload, column))
        consumed.add(column)
    for column in _payload_columns(payload):
        if column in consumed:
            continue
        weight = _anomaly_score_weight(column)
        if weight is None:
            continue
        score += weight * max(0.0, _row_value(payload, column))
    return float(score)


def _normalize_score_map(score_map: Mapping[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return {key: 0.0 for key in score_map}
    scale = maximum - minimum
    return {
        key: float(value - minimum) / float(scale)
        for key, value in score_map.items()
    }


def _entity_anomaly_score_maps(frame: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    service_scores = {}
    host_scores = {}
    for row in frame.to_dict("records"):
        entity_id = str(row.get("entity_id"))
        entity_type = str(row.get("entity_type"))
        score = _entity_anomaly_scalar(row)
        if entity_type == "service":
            service_scores[entity_id] = score
        elif entity_type == "host":
            host_scores[entity_id] = score
    return _normalize_score_map(service_scores), _normalize_score_map(host_scores)


def _build_window_edge_weights(
    frame: pd.DataFrame,
    schema: HeteroGraphSchema,
    scale: float,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    service_scores, host_scores = _entity_anomaly_score_maps(frame)
    edge_weights: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for canonical_etype, (source_indices, target_indices) in schema.edge_dict.items():
        weights = []
        for source_index, target_index in zip(source_indices.tolist(), target_indices.tolist()):
            if canonical_etype[0] == "service":
                source_id = schema.service_ids[int(source_index)]
                source_score = service_scores.get(source_id, 0.0)
            else:
                source_id = schema.host_ids[int(source_index)]
                source_score = host_scores.get(source_id, 0.0)
            if canonical_etype[2] == "service":
                target_id = schema.service_ids[int(target_index)]
                target_score = service_scores.get(target_id, 0.0)
            else:
                target_id = schema.host_ids[int(target_index)]
                target_score = host_scores.get(target_id, 0.0)
            refined_weight = 1.0 + float(scale) * ((source_score + target_score) / 2.0)
            weights.append(refined_weight)
        edge_weights[canonical_etype] = torch.tensor(weights, dtype=torch.float32)
    return edge_weights


def _attach_edge_weights(
    graph: dgl.DGLHeteroGraph,
    edge_weights: Mapping[Tuple[str, str, str], torch.Tensor],
) -> None:
    for canonical_etype, weights in edge_weights.items():
        graph.edges[canonical_etype].data["weight"] = weights.clone()


def _graph_edge_weight_dict(
    graph: dgl.DGLHeteroGraph,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    edge_weight_dict = {}
    for canonical_etype in graph.canonical_etypes:
        edge_data = graph.edges[canonical_etype].data
        if "weight" not in edge_data:
            continue
        edge_weight_dict[canonical_etype] = edge_data["weight"]
    return edge_weight_dict


def _compute_entity_anomaly_series(
    tables,
    schema: HeteroGraphSchema,
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    ordered_windows = (
        tables.windows.sort_values(by=["start_ts", "window_id"]).reset_index(drop=True)
    )
    window_ids = ordered_windows["window_id"].astype(str).tolist()
    window_index = {window_id: index for index, window_id in enumerate(window_ids)}
    service_series = {
        entity_id: np.zeros(len(window_ids), dtype=float)
        for entity_id in schema.service_ids
    }
    host_series = {
        entity_id: np.zeros(len(window_ids), dtype=float)
        for entity_id in schema.host_ids
    }

    for row in tables.entity_features.itertuples(index=False):
        index = window_index.get(str(row.window_id))
        if index is None:
            continue
        entity_id = str(row.entity_id)
        score = _entity_anomaly_scalar(row)
        if str(row.entity_type) == "service" and entity_id in service_series:
            service_series[entity_id][index] = score
        elif str(row.entity_type) == "host" and entity_id in host_series:
            host_series[entity_id][index] = score
    return service_series, host_series


def _series_correlation(left: np.ndarray, right: np.ndarray) -> Tuple[float, int, bool]:
    mask = np.logical_or(np.abs(left) > 1e-9, np.abs(right) > 1e-9)
    support = int(mask.sum())
    if support < 2:
        return 0.0, support, False
    left_values = left[mask]
    right_values = right[mask]
    if float(np.std(left_values)) <= 1e-8 or float(np.std(right_values)) <= 1e-8:
        return 0.0, support, False
    correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    return correlation, support, True


def _restore_best_edges_if_needed(
    records: Sequence[Dict[str, Any]],
    source_key: str,
    selected_flags: List[bool],
) -> List[bool]:
    if not records:
        return selected_flags
    grouped = {}
    for index, record in enumerate(records):
        grouped.setdefault(record[source_key], []).append((index, record))
    selected = list(selected_flags)
    for _source, members in grouped.items():
        if any(selected[index] for index, _record in members):
            continue
        best_index = max(
            members,
            key=lambda item: (item[1]["correlation"], item[1]["support"]),
        )[0]
        selected[best_index] = True
    return selected


def _refine_graph_schema_by_correlation(
    schema: HeteroGraphSchema,
    tables,
    config,
) -> HeteroGraphSchema:
    if not bool(getattr(config, "use_edge_refinement", True)):
        return schema

    threshold = float(getattr(config, "edge_refinement_corr_threshold", 0.0))
    min_support = max(1, int(getattr(config, "edge_refinement_min_support", 1)))
    service_series, host_series = _compute_entity_anomaly_series(tables, schema)
    refined_edges: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    refinement_metadata = {}

    for canonical_etype, (source_indices, target_indices) in schema.edge_dict.items():
        if len(source_indices) == 0:
            refined_edges[canonical_etype] = _edge_tensor_pair([], [])
            refinement_metadata[canonical_etype[1]] = {
                "before": 0,
                "after": 0,
                "threshold": threshold,
                "min_support": min_support,
            }
            continue
        records = []
        for source_index, target_index in zip(source_indices.tolist(), target_indices.tolist()):
            if canonical_etype[0] == "service":
                source_id = schema.service_ids[int(source_index)]
                source_values = service_series.get(source_id)
            else:
                source_id = schema.host_ids[int(source_index)]
                source_values = host_series.get(source_id)
            if canonical_etype[2] == "service":
                target_id = schema.service_ids[int(target_index)]
                target_values = service_series.get(target_id)
            else:
                target_id = schema.host_ids[int(target_index)]
                target_values = host_series.get(target_id)
            correlation, support, valid = _series_correlation(
                np.asarray(source_values, dtype=float),
                np.asarray(target_values, dtype=float),
            )
            keep = (not valid) or support < min_support or correlation >= threshold
            records.append(
                {
                    "source_index": int(source_index),
                    "target_index": int(target_index),
                    "source_id": source_id,
                    "target_id": target_id,
                    "correlation": float(correlation),
                    "support": int(support),
                    "keep": bool(keep),
                }
            )

        selected_flags = [record["keep"] for record in records]
        if canonical_etype == ("service", "service_host", "host"):
            selected_flags = _restore_best_edges_if_needed(
                records=records,
                source_key="source_id",
                selected_flags=selected_flags,
            )
        elif canonical_etype == ("service", "service_service", "service"):
            selected_flags = _restore_best_edges_if_needed(
                records=records,
                source_key="source_id",
                selected_flags=selected_flags,
            )

        kept_sources = [
            record["source_index"]
            for record, keep in zip(records, selected_flags)
            if keep
        ]
        kept_targets = [
            record["target_index"]
            for record, keep in zip(records, selected_flags)
            if keep
        ]
        if not kept_sources:
            fallback = max(
                records,
                key=lambda record: (record["correlation"], record["support"]),
            )
            kept_sources = [fallback["source_index"]]
            kept_targets = [fallback["target_index"]]
        refined_edges[canonical_etype] = _edge_tensor_pair(
            kept_sources,
            kept_targets,
        )
        refinement_metadata[canonical_etype[1]] = {
            "before": int(len(records)),
            "after": int(len(kept_sources)),
            "threshold": threshold,
            "min_support": min_support,
        }

    return HeteroGraphSchema(
        dataset=schema.dataset,
        edge_dict=refined_edges,
        num_nodes_dict=dict(schema.num_nodes_dict),
        service_ids=list(schema.service_ids),
        host_ids=list(schema.host_ids),
        service_index=dict(schema.service_index),
        host_index=dict(schema.host_index),
        metadata={
            **dict(schema.metadata),
            "edge_refinement": refinement_metadata,
        },
    )


def _window_anomaly_summary(frame: pd.DataFrame) -> Mapping[str, float]:
    anomaly_values = frame.apply(_entity_anomaly_scalar, axis=1).to_numpy(dtype=float)
    if anomaly_values.size == 0:
        return {
            "window_anomaly_score_mean": 0.0,
            "window_anomaly_score_max": 0.0,
        }
    return {
        "window_anomaly_score_mean": float(anomaly_values.mean()),
        "window_anomaly_score_max": float(anomaly_values.max()),
    }


def _build_window_graph_samples(
    frame: pd.DataFrame,
    schema: HeteroGraphSchema,
    service_feature_columns: Sequence[str],
    host_feature_columns: Sequence[str],
    service_feature_mean: np.ndarray,
    service_feature_std: np.ndarray,
    host_feature_mean: np.ndarray,
    host_feature_std: np.ndarray,
    use_edge_refinement: bool,
    edge_refinement_scale: float,
) -> List[WindowGraphSample]:
    samples = []
    for window_id, group in frame.groupby("window_id", sort=False):
        service_features, host_features = _build_window_feature_tensors(
            group,
            schema,
            service_feature_columns,
            host_feature_columns,
            service_feature_mean=service_feature_mean,
            service_feature_std=service_feature_std,
            host_feature_mean=host_feature_mean,
            host_feature_std=host_feature_std,
        )
        graph = _instantiate_graph(schema)
        if use_edge_refinement:
            _attach_edge_weights(
                graph,
                _build_window_edge_weights(
                    group,
                    schema,
                    scale=edge_refinement_scale,
                ),
            )
        graph.nodes["service"].data["x"] = service_features
        graph.nodes["host"].data["x"] = host_features

        if "label" in group.columns and "sample_weight" in group.columns:
            service_labels = torch.zeros(len(schema.service_ids), dtype=torch.float32)
            host_labels = torch.zeros(len(schema.host_ids), dtype=torch.float32)
            service_weights = torch.zeros(len(schema.service_ids), dtype=torch.float32)
            host_weights = torch.zeros(len(schema.host_ids), dtype=torch.float32)
            for row in group.itertuples(index=False):
                entity_id = str(row.entity_id)
                label = float(row.label)
                weight = float(row.sample_weight)
                if str(row.entity_type) == "service":
                    index = schema.service_index.get(entity_id)
                    if index is None:
                        continue
                    service_labels[index] = label
                    service_weights[index] = weight
                else:
                    index = schema.host_index.get(entity_id)
                    if index is None:
                        continue
                    host_labels[index] = label
                    host_weights[index] = weight
            graph.nodes["service"].data["y"] = service_labels
            graph.nodes["host"].data["y"] = host_labels
            graph.nodes["service"].data["w"] = service_weights
            graph.nodes["host"].data["w"] = host_weights

        metadata = group.iloc[0][WINDOW_META_COLUMNS].to_dict()
        metadata.update(_window_anomaly_summary(group))
        samples.append(
            WindowGraphSample(
                graph=graph,
                window_id=str(window_id),
                metadata=metadata,
            )
        )
    return samples


def _iter_batches(items: Sequence[WindowGraphSample], batch_size: int) -> Iterable[Sequence[WindowGraphSample]]:
    for start in range(0, len(items), max(1, int(batch_size))):
        yield items[start : start + max(1, int(batch_size))]


def _resolve_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_name))


def _pooled_window_embedding(hidden: Mapping[str, torch.Tensor]) -> np.ndarray:
    pooled_parts = []
    for node_type in ("service", "host"):
        node_hidden = hidden[node_type]
        pooled_parts.append(node_hidden.mean(dim=0))
        pooled_parts.append(node_hidden.max(dim=0).values)
    return torch.cat(pooled_parts, dim=0).detach().cpu().numpy()


def build_window_hgcn_embeddings(
    tables,
    feature_root: Path,
    selected_feature_columns: Sequence[str],
    config,
    random_state: int = 42,
) -> PretrainedGraphBundle:
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    base_schema = build_hetero_graph_schema(feature_root, tables.dataset)
    service_rows = tables.entity_features[tables.entity_features["entity_type"] == "service"]
    host_rows = tables.entity_features[tables.entity_features["entity_type"] == "host"]
    service_feature_columns = _nonzero_feature_columns(service_rows, selected_feature_columns)
    host_feature_columns = _nonzero_feature_columns(host_rows, selected_feature_columns)
    service_feature_mean, service_feature_std = _compute_feature_stats(
        service_rows,
        service_feature_columns,
    )
    host_feature_mean, host_feature_std = _compute_feature_stats(
        host_rows,
        host_feature_columns,
    )
    schema = _refine_graph_schema_by_correlation(base_schema, tables, config)

    samples = _build_window_graph_samples(
        frame=tables.entity_features,
        schema=schema,
        service_feature_columns=service_feature_columns,
        host_feature_columns=host_feature_columns,
        service_feature_mean=service_feature_mean,
        service_feature_std=service_feature_std,
        host_feature_mean=host_feature_mean,
        host_feature_std=host_feature_std,
        use_edge_refinement=bool(getattr(config, "use_edge_refinement", True)),
        edge_refinement_scale=float(getattr(config, "edge_refinement_scale", 1.0)),
    )
    if not samples:
        raise ValueError("No window graphs are available for HGCN pretraining.")

    device = _resolve_device(str(getattr(config, "hgcn_device", "auto")))
    model = HGCNReconstructionModel(
        service_input_dim=len(service_feature_columns),
        host_input_dim=len(host_feature_columns),
        hidden_dim=int(getattr(config, "hgcn_hidden_dim", 64)),
        num_layers=int(getattr(config, "hgcn_layers", 2)),
        dropout=float(getattr(config, "hgcn_dropout", 0.10)),
        canonical_etypes=list(schema.edge_dict.keys()),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(getattr(config, "cluster_pretrain_lr", 1e-3)),
        weight_decay=float(getattr(config, "cluster_pretrain_weight_decay", 1e-4)),
    )

    best_state = None
    best_loss = None
    stale_epochs = 0
    patience = max(5, int(getattr(config, "cluster_pretrain_patience", 10)))

    for _epoch in range(max(1, int(getattr(config, "cluster_pretrain_epochs", 80)))):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in _iter_batches(samples, int(getattr(config, "hgcn_batch_size", 16))):
            batched_graph = dgl.batch([sample.graph for sample in batch]).to(device)
            service_features = batched_graph.nodes["service"].data["x"].to(device)
            host_features = batched_graph.nodes["host"].data["x"].to(device)
            edge_weight_dict = (
                _graph_edge_weight_dict(batched_graph)
                if bool(getattr(config, "use_edge_refinement", True))
                else None
            )
            _hidden, reconstruction = model(
                batched_graph,
                service_features,
                host_features,
                edge_weight_dict=edge_weight_dict,
            )
            loss = F.mse_loss(reconstruction["service"], service_features) + F.mse_loss(
                reconstruction["host"],
                host_features,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1

        mean_loss = epoch_loss / float(max(batch_count, 1))
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    records = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            graph = sample.graph.to(device)
            service_features = graph.nodes["service"].data["x"].to(device)
            host_features = graph.nodes["host"].data["x"].to(device)
            edge_weight_dict = (
                _graph_edge_weight_dict(graph)
                if bool(getattr(config, "use_edge_refinement", True))
                else None
            )
            hidden = model.encoder(
                graph,
                service_features,
                host_features,
                edge_weight_dict=edge_weight_dict,
            )
            pooled = _pooled_window_embedding(hidden)
            record = dict(sample.metadata)
            for index, value in enumerate(pooled):
                record["embed_%03d" % index] = float(value)
            records.append(record)

    embedding_frame = pd.DataFrame.from_records(records)
    return PretrainedGraphBundle(
        schema=schema,
        service_feature_columns=list(service_feature_columns),
        host_feature_columns=list(host_feature_columns),
        service_feature_mean=service_feature_mean.tolist(),
        service_feature_std=service_feature_std.tolist(),
        host_feature_mean=host_feature_mean.tolist(),
        host_feature_std=host_feature_std.tolist(),
        embedding_frame=embedding_frame,
        encoder_state_dict={
            key: value.detach().cpu().clone()
            for key, value in model.encoder.state_dict().items()
        },
        metadata={
            "embedding_source": "hgcn_autoencoder",
            "reconstruction_loss": float(best_loss or 0.0),
            "embedding_dim": int(
                4 * int(getattr(config, "hgcn_hidden_dim", 64))
            ),
            "relation_edge_counts": {
                canonical_etype[1]: int(len(source_indices))
                for canonical_etype, (source_indices, _target_indices) in schema.edge_dict.items()
            },
            "schema_metadata": dict(schema.metadata),
        },
    )


def _restore_pretrained_encoder(
    bundle: PretrainedGraphBundle,
    config,
    device: torch.device,
) -> HGCNEncoder:
    encoder = HGCNEncoder(
        service_input_dim=len(bundle.service_feature_columns),
        host_input_dim=len(bundle.host_feature_columns),
        hidden_dim=int(getattr(config, "hgcn_hidden_dim", 64)),
        num_layers=int(getattr(config, "hgcn_layers", 2)),
        dropout=float(getattr(config, "hgcn_dropout", 0.10)),
        canonical_etypes=list(bundle.schema.edge_dict.keys()),
    ).to(device)
    encoder.load_state_dict(bundle.encoder_state_dict, strict=False)
    encoder.eval()
    return encoder


def build_entity_graph_context_features(
    entity_features: pd.DataFrame,
    bundle: PretrainedGraphBundle,
    config,
) -> pd.DataFrame:
    if entity_features.empty:
        return pd.DataFrame(columns=["window_id", "entity_id"])

    device = _resolve_device(str(getattr(config, "hgcn_device", "auto")))
    encoder = _restore_pretrained_encoder(bundle=bundle, config=config, device=device)
    service_feature_mean = np.asarray(bundle.service_feature_mean, dtype=np.float32)
    service_feature_std = np.asarray(bundle.service_feature_std, dtype=np.float32)
    host_feature_mean = np.asarray(bundle.host_feature_mean, dtype=np.float32)
    host_feature_std = np.asarray(bundle.host_feature_std, dtype=np.float32)

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for window_id, group in entity_features.groupby("window_id", sort=False):
            service_features, host_features = _build_window_feature_tensors(
                group,
                bundle.schema,
                bundle.service_feature_columns,
                bundle.host_feature_columns,
                service_feature_mean=service_feature_mean,
                service_feature_std=service_feature_std,
                host_feature_mean=host_feature_mean,
                host_feature_std=host_feature_std,
            )
            graph = _instantiate_graph(bundle.schema)
            if bool(getattr(config, "use_edge_refinement", True)):
                _attach_edge_weights(
                    graph,
                    _build_window_edge_weights(
                        group,
                        bundle.schema,
                        scale=float(getattr(config, "edge_refinement_scale", 1.0)),
                    ),
                )
            graph = graph.to(device)
            edge_weight_dict = (
                _graph_edge_weight_dict(graph)
                if bool(getattr(config, "use_edge_refinement", True))
                else None
            )
            hidden = encoder(
                graph,
                service_features.to(device),
                host_features.to(device),
                edge_weight_dict=edge_weight_dict,
            )
            service_hidden = hidden["service"].detach().cpu().numpy()
            host_hidden = hidden["host"].detach().cpu().numpy()
            hidden_dim = int(service_hidden.shape[1] if service_hidden.size else host_hidden.shape[1])
            for row in group.itertuples(index=False):
                entity_id = str(row.entity_id)
                if str(row.entity_type) == "service":
                    index = bundle.schema.service_index.get(entity_id)
                    vector = (
                        service_hidden[index]
                        if index is not None and index < len(service_hidden)
                        else np.zeros(hidden_dim, dtype=np.float32)
                    )
                else:
                    index = bundle.schema.host_index.get(entity_id)
                    vector = (
                        host_hidden[index]
                        if index is not None and index < len(host_hidden)
                        else np.zeros(hidden_dim, dtype=np.float32)
                    )
                record = {
                    "window_id": str(window_id),
                    "entity_id": entity_id,
                }
                for feature_index, value in enumerate(vector):
                    record["graph_ctx_%03d" % feature_index] = float(value)
                records.append(record)
    return pd.DataFrame.from_records(records)


def _batched_pairwise_margin_loss(
    service_logits: torch.Tensor,
    host_logits: torch.Tensor,
    batched_graph: dgl.DGLHeteroGraph,
) -> torch.Tensor:
    service_labels = batched_graph.nodes["service"].data["y"]
    host_labels = batched_graph.nodes["host"].data["y"]
    service_weights = batched_graph.nodes["service"].data["w"]
    host_weights = batched_graph.nodes["host"].data["w"]
    service_counts = batched_graph.batch_num_nodes("service").tolist()
    host_counts = batched_graph.batch_num_nodes("host").tolist()

    service_offset = 0
    host_offset = 0
    losses = []
    for service_count, host_count in zip(service_counts, host_counts):
        service_slice = slice(service_offset, service_offset + int(service_count))
        host_slice = slice(host_offset, host_offset + int(host_count))
        combined_logits = torch.cat(
            [service_logits[service_slice], host_logits[host_slice]],
            dim=0,
        )
        combined_labels = torch.cat(
            [service_labels[service_slice], host_labels[host_slice]],
            dim=0,
        )
        combined_weights = torch.cat(
            [service_weights[service_slice], host_weights[host_slice]],
            dim=0,
        )
        positive_mask = (combined_labels > 0.5) & (combined_weights > 0.0)
        negative_mask = (combined_labels < 0.5) & (combined_weights > 0.0)
        if positive_mask.any() and negative_mask.any():
            positive_logits = combined_logits[positive_mask]
            negative_logits = combined_logits[negative_mask]
            positive_weights = combined_weights[positive_mask]
            negative_weights = combined_weights[negative_mask]
            margins = 1.0 - (positive_logits[:, None] - negative_logits[None, :])
            pair_weights = (positive_weights[:, None] + negative_weights[None, :]) / 2.0
            pair_loss = F.relu(margins)
            losses.append(
                (pair_loss * pair_weights).sum()
                / torch.clamp(pair_weights.sum(), min=1.0)
            )
        service_offset += int(service_count)
        host_offset += int(host_count)
    if not losses:
        return service_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_hgcn_classifier(
    tables,
    training_frame: pd.DataFrame,
    feature_root: Path,
    selected_feature_columns: Sequence[str],
    config,
    random_state: int = 42,
    pretrained_bundle: Optional[PretrainedGraphBundle] = None,
):
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    bundle = pretrained_bundle or build_window_hgcn_embeddings(
        tables=tables,
        feature_root=feature_root,
        selected_feature_columns=selected_feature_columns,
        config=config,
        random_state=random_state,
    )
    schema = bundle.schema
    samples = _build_window_graph_samples(
        frame=training_frame,
        schema=schema,
        service_feature_columns=bundle.service_feature_columns,
        host_feature_columns=bundle.host_feature_columns,
        service_feature_mean=np.asarray(bundle.service_feature_mean, dtype=np.float32),
        service_feature_std=np.asarray(bundle.service_feature_std, dtype=np.float32),
        host_feature_mean=np.asarray(bundle.host_feature_mean, dtype=np.float32),
        host_feature_std=np.asarray(bundle.host_feature_std, dtype=np.float32),
        use_edge_refinement=bool(getattr(config, "use_edge_refinement", True)),
        edge_refinement_scale=float(getattr(config, "edge_refinement_scale", 1.0)),
    )
    if not samples:
        raise ValueError("No labeled samples are available for HGCN training.")

    device = _resolve_device(str(getattr(config, "hgcn_device", "auto")))
    model = HGCNNodeScorer(
        service_input_dim=len(bundle.service_feature_columns),
        host_input_dim=len(bundle.host_feature_columns),
        hidden_dim=int(getattr(config, "hgcn_hidden_dim", 64)),
        num_layers=int(getattr(config, "hgcn_layers", 2)),
        dropout=float(getattr(config, "hgcn_dropout", 0.10)),
        canonical_etypes=list(schema.edge_dict.keys()),
    ).to(device)
    if bundle.encoder_state_dict:
        model.encoder.load_state_dict(bundle.encoder_state_dict, strict=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(getattr(config, "hgcn_lr", 1e-3)),
        weight_decay=float(getattr(config, "hgcn_weight_decay", 1e-4)),
    )

    best_state = None
    best_loss = None
    stale_epochs = 0
    patience = max(5, int(getattr(config, "hgcn_patience", 10)))
    pairwise_weight = float(getattr(config, "supervised_pairwise_loss_weight", 0.0))

    for _epoch in range(max(1, int(getattr(config, "hgcn_epochs", 80)))):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in _iter_batches(samples, int(getattr(config, "hgcn_batch_size", 16))):
            batched_graph = dgl.batch([sample.graph for sample in batch]).to(device)
            service_features = batched_graph.nodes["service"].data["x"].to(device)
            host_features = batched_graph.nodes["host"].data["x"].to(device)
            service_labels = batched_graph.nodes["service"].data["y"].to(device)
            host_labels = batched_graph.nodes["host"].data["y"].to(device)
            service_weights = batched_graph.nodes["service"].data["w"].to(device)
            host_weights = batched_graph.nodes["host"].data["w"].to(device)
            edge_weight_dict = (
                _graph_edge_weight_dict(batched_graph)
                if bool(getattr(config, "use_edge_refinement", True))
                else None
            )

            logits = model(
                batched_graph,
                service_features,
                host_features,
                edge_weight_dict=edge_weight_dict,
            )
            service_loss = F.binary_cross_entropy_with_logits(
                logits["service"],
                service_labels,
                reduction="none",
            )
            host_loss = F.binary_cross_entropy_with_logits(
                logits["host"],
                host_labels,
                reduction="none",
            )
            weighted_service = (service_loss * service_weights).sum()
            weighted_host = (host_loss * host_weights).sum()
            normalizer = torch.clamp(service_weights.sum() + host_weights.sum(), min=1.0)
            loss = (weighted_service + weighted_host) / normalizer
            if pairwise_weight > 0.0:
                loss = loss + pairwise_weight * _batched_pairwise_margin_loss(
                    logits["service"],
                    logits["host"],
                    batched_graph,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1

        mean_loss = epoch_loss / float(max(batch_count, 1))
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return HGCNClassifierWrapper(
        model=model,
        schema=schema,
        service_feature_columns=bundle.service_feature_columns,
        host_feature_columns=bundle.host_feature_columns,
        service_feature_mean=bundle.service_feature_mean,
        service_feature_std=bundle.service_feature_std,
        host_feature_mean=bundle.host_feature_mean,
        host_feature_std=bundle.host_feature_std,
        device=device,
        use_edge_refinement=bool(getattr(config, "use_edge_refinement", True)),
        edge_refinement_scale=float(getattr(config, "edge_refinement_scale", 1.0)),
    )
