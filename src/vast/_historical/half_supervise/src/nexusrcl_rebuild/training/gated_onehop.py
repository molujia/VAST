"""Relation-normalized one-hop residual features for semi-supervised RCL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


ALLOWED_RELATIONS = ("CALLS", "CALLED_BY")


@dataclass(frozen=True)
class GatedOnehopOutput:
    relation_contexts: Mapping[str, np.ndarray]
    relation_residuals: Mapping[str, np.ndarray]
    gate_by_relation: Mapping[str, np.ndarray]
    active_by_relation: Mapping[str, np.ndarray]
    derived_features: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class GatedOnehopFrame:
    frame: pd.DataFrame
    feature_columns: Tuple[str, ...]
    diagnostics: Mapping[str, Any]


def _relations(values: Sequence[str]) -> Tuple[str, ...]:
    relations = tuple(str(value) for value in values)
    if len(relations) != len(set(relations)):
        raise ValueError("gated one-hop relations must be unique")
    unknown = sorted(set(relations) - set(ALLOWED_RELATIONS))
    if unknown:
        raise ValueError(
            "unsupported gated one-hop relation: %s" % unknown
        )
    return relations


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("feature frame contains a non-finite value")
        return float(value)
    if value is None:
        return None
    return value


def feature_frame_sha256(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> str:
    columns = [
        column
        for column in (
            "window_id",
            "entity_id",
            "entity_type",
            "entity_name",
        )
        if column in frame.columns
    ]
    for column in feature_columns:
        text = str(column)
        if text not in frame.columns:
            raise ValueError("feature frame is missing column %s" % text)
        if text not in columns:
            columns.append(text)
    records = [
        {
            column: _canonical_scalar(value)
            for column, value in zip(columns, row)
        }
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_edges(
    edges: Sequence[Sequence[Any]],
    *,
    node_count: int,
) -> Tuple[Tuple[int, int, float], ...]:
    output = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError(
                "gated one-hop edges must be source/target/weight triples"
            )
        source, target, weight = edge
        if (
            isinstance(source, bool)
            or isinstance(target, bool)
            or not isinstance(source, (int, np.integer))
            or not isinstance(target, (int, np.integer))
        ):
            raise ValueError("gated one-hop edge endpoints must be integers")
        source = int(source)
        target = int(target)
        weight = float(weight)
        if (
            source < 0
            or target < 0
            or source >= node_count
            or target >= node_count
        ):
            raise ValueError("gated one-hop edge endpoint is out of range")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "gated one-hop edge weight must be finite and positive"
            )
        output.append((source, target, weight))
    return tuple(output)


def gated_onehop_features(
    features: np.ndarray,
    *,
    edges: Sequence[Sequence[Any]],
    relations: Sequence[str],
) -> GatedOnehopOutput:
    raw = np.asarray(features, dtype=float)
    if raw.ndim != 2:
        raise ValueError("gated one-hop features must be a matrix")
    if not np.isfinite(raw).all():
        raise ValueError("gated one-hop features must be finite")
    configured = _relations(relations)
    validated_edges = _validated_edges(edges, node_count=raw.shape[0])
    contexts: Dict[str, np.ndarray] = {}
    residuals: Dict[str, np.ndarray] = {}
    gates: Dict[str, np.ndarray] = {}
    active_masks: Dict[str, np.ndarray] = {}
    active_counts: Dict[str, int] = {}
    gate_diagnostics: Dict[str, Dict[str, float]] = {}
    for relation in configured:
        context = np.zeros_like(raw)
        degree = np.zeros(raw.shape[0], dtype=float)
        for source, target, weight in validated_edges:
            if relation == "CALLS":
                sender, receiver = source, target
            else:
                sender, receiver = target, source
            context[receiver] += weight * raw[sender]
            degree[receiver] += weight
        active = degree > 0.0
        context[active] = context[active] / degree[active, None]
        gate = np.zeros(raw.shape[0], dtype=float)
        if np.any(active):
            raw_norm = np.linalg.norm(raw[active], axis=1)
            context_norm = np.linalg.norm(context[active], axis=1)
            denominator = raw_norm * context_norm
            similarity = np.zeros(int(np.sum(active)), dtype=float)
            nonzero = denominator > 0.0
            if np.any(nonzero):
                similarity[nonzero] = (
                    np.sum(raw[active][nonzero] * context[active][nonzero], axis=1)
                    / denominator[nonzero]
                )
            similarity = np.clip(similarity, -1.0, 1.0)
            gate[active] = 1.0 / (1.0 + np.exp(-similarity))
        residual = raw + gate[:, None] * context
        if not np.isfinite(residual).all():
            raise FloatingPointError(
                "gated one-hop residual contains a non-finite value"
            )
        contexts[relation] = context
        residuals[relation] = residual
        gates[relation] = gate
        active_masks[relation] = active
        active_count = int(np.sum(active))
        active_counts[relation] = active_count
        active_gates = gate[active]
        gate_diagnostics[relation] = {
            "mean": (
                float(np.mean(active_gates)) if active_gates.size else 0.0
            ),
            "std": (
                float(np.std(active_gates)) if active_gates.size else 0.0
            ),
            "finite_count": active_count,
        }
    derived = (
        np.concatenate([residuals[relation] for relation in configured], axis=1)
        if configured
        else np.empty((raw.shape[0], 0), dtype=float)
    )
    return GatedOnehopOutput(
        relation_contexts=contexts,
        relation_residuals=residuals,
        gate_by_relation=gates,
        active_by_relation=active_masks,
        derived_features=derived,
        diagnostics={
            "mode": (
                "gated_onehop_rcl_v1" if configured else "G0_raw_control"
            ),
            "configured_relations": list(configured),
            "relation_edge_count": {
                relation: len(validated_edges) for relation in configured
            },
            "active_node_count_by_relation": active_counts,
            "gate_by_relation": gate_diagnostics,
            "hop_count": 1,
            "relation_normalization": "weighted_receiver_mean",
            "uniform_score_diffusion": False,
            "raw_columns_overwritten": False,
        },
    )


def _derived_column(relation: str, feature_column: str) -> str:
    return "gated_onehop_%s_%s" % (
        relation.lower(),
        str(feature_column),
    )


def augment_entity_features_with_gated_onehop(
    entity_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    service_service_edges: Mapping[Tuple[str, str], Any],
    relations: Sequence[str],
) -> GatedOnehopFrame:
    configured = _relations(relations)
    frame = entity_features.copy()
    source_columns = tuple(str(column) for column in feature_columns)
    for column in source_columns:
        if column not in frame.columns:
            raise ValueError(
                "gated one-hop source feature is missing: %s" % column
            )
    input_hash = feature_frame_sha256(frame, source_columns)
    if not configured:
        return GatedOnehopFrame(
            frame=frame,
            feature_columns=(),
            diagnostics={
                "mode": "G0_raw_control",
                "configured_relations": [],
                "input_feature_sha256": input_hash,
                "output_feature_sha256": input_hash,
                "hop_count": 1,
                "uniform_score_diffusion": False,
                "raw_columns_overwritten": False,
            },
        )
    required = {"window_id", "entity_type", "entity_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "gated one-hop entity frame is missing: %s" % missing
        )
    service_rows = frame[frame["entity_type"].astype(str) == "service"]
    known_services = set(service_rows["entity_name"].astype(str))
    unknown_services = sorted(
        {
            str(service)
            for edge in service_service_edges
            for service in edge
            if str(service) not in known_services
        }
    )
    if unknown_services:
        raise ValueError(
            "gated one-hop topology contains unknown service: %s"
            % unknown_services
        )
    derived_columns = tuple(
        _derived_column(relation, column)
        for relation in configured
        for column in source_columns
    )
    collisions = sorted(set(derived_columns) & set(frame.columns))
    if collisions:
        raise ValueError(
            "gated one-hop would overwrite derived columns: %s" % collisions
        )
    for column in derived_columns:
        frame[column] = 0.0

    gate_values: Dict[str, list] = {
        relation: [] for relation in configured
    }
    active_counts = {relation: 0 for relation in configured}
    applied_edge_counts = {relation: 0 for relation in configured}
    for _, group in frame.groupby("window_id", sort=False):
        services = group[group["entity_type"].astype(str) == "service"]
        if services.empty:
            continue
        names = services["entity_name"].astype(str).tolist()
        if len(names) != len(set(names)):
            raise ValueError(
                "gated one-hop window contains duplicate service rows"
            )
        name_to_index = {
            name: index for index, name in enumerate(names)
        }
        local_edges = []
        for (source, target), weight in service_service_edges.items():
            source = str(source)
            target = str(target)
            if source not in name_to_index or target not in name_to_index:
                continue
            local_edges.append(
                (
                    name_to_index[source],
                    name_to_index[target],
                    float(weight),
                )
            )
        matrix = (
            services[list(source_columns)]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        output = gated_onehop_features(
            matrix,
            edges=local_edges,
            relations=configured,
        )
        for relation in configured:
            columns = [
                _derived_column(relation, column)
                for column in source_columns
            ]
            frame.loc[
                services.index,
                columns,
            ] = output.relation_residuals[relation]
            active = output.active_by_relation[relation]
            gate_values[relation].extend(
                output.gate_by_relation[relation][active].tolist()
            )
            active_counts[relation] += int(np.sum(active))
            applied_edge_counts[relation] += len(local_edges)
    gate_diagnostics = {}
    for relation in configured:
        values = np.asarray(gate_values[relation], dtype=float)
        gate_diagnostics[relation] = {
            "mean": float(np.mean(values)) if values.size else 0.0,
            "std": float(np.std(values)) if values.size else 0.0,
            "finite_count": int(values.size),
        }
    return GatedOnehopFrame(
        frame=frame,
        feature_columns=derived_columns,
        diagnostics={
            "mode": "gated_onehop_rcl_v1",
            "configured_relations": list(configured),
            "input_feature_sha256": input_hash,
            "output_feature_sha256": feature_frame_sha256(
                frame,
                tuple(source_columns) + derived_columns,
            ),
            "relation_edge_count": applied_edge_counts,
            "active_node_count_by_relation": active_counts,
            "gate_by_relation": gate_diagnostics,
            "hop_count": 1,
            "relation_normalization": "weighted_receiver_mean",
            "uniform_score_diffusion": False,
            "raw_columns_overwritten": False,
        },
    )


__all__ = [
    "ALLOWED_RELATIONS",
    "GatedOnehopFrame",
    "GatedOnehopOutput",
    "augment_entity_features_with_gated_onehop",
    "feature_frame_sha256",
    "gated_onehop_features",
]
