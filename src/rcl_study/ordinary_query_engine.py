"""Event-sourced ordinary budget-30 multimodal active-learning strategies."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


QUERY_ENGINE_SCHEMA_VERSION = "ordinary-query-engine-v1"
EVENT_SCHEMA_VERSION = "ordinary-query-event-v1"
ROUND_SIZES = (8,) + (2,) * 11
BUDGET = 30
STRATEGY_IDS = (
    "kmeans_coverage",
    "dbscan_coverage",
    "knn_fault_mode_coverage",
    "mutual_knn_graph_coverage",
    "falcon_hybrid",
    "hdbscan_coverage",
    "graph_facility_location",
)
_CANDIDATE_FIELDS = frozenset(
    ("case_id", "embedding", "timestamp", "incident_id", "uncertainty")
)
_FORBIDDEN_FIELDS = frozenset(
    (
        "fault_type",
        "root_cause",
        "root_causes",
        "positive_ids",
        "positive_ids_list",
        "positive_names",
        "positive_types",
        "targets",
        "time_bucket",
    )
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _tie(seed: int, case_id: str, namespace: str = "tie") -> str:
    return hashlib.sha256(
        ("%s:%d:%s" % (namespace, int(seed), case_id)).encode("utf-8")
    ).hexdigest()


def _unit_tie(seed: int, case_id: str, namespace: str = "tie") -> float:
    return int(_tie(seed, case_id, namespace)[:12], 16) / float(16**12 - 1)


def _strategy_defaults() -> Dict[str, Dict[str, Any]]:
    return {
        "kmeans_coverage": {
            "cluster_count": 14,
            "max_per_proxy_mode": 5,
            "seed_budget": 8,
            "time_chunk_count": 4,
            "representative_pool_size": 3,
        },
        "dbscan_coverage": {
            "min_samples": 5,
            "eps_quantile": 0.80,
            "noise_budget_cap": 2,
            "boundary_fraction": 0.20,
        },
        "knn_fault_mode_coverage": {"neighbor_count": 5},
        "mutual_knn_graph_coverage": {
            "neighbor_count": 10,
            "tiny_component_size": 2,
            "tiny_component_budget_cap": 2,
        },
        "falcon_hybrid": {
            "time_chunk_count": 4,
            "exploration_fraction": 0.50,
            "minimum_exploration_per_round": 1,
        },
        "hdbscan_coverage": {
            "min_cluster_size": 8,
            "min_samples": 5,
            "noise_budget_cap": 2,
            "boundary_fraction": 0.20,
        },
        "graph_facility_location": {
            "kernel_temperature": 1.0,
            "incident_weighting": "inverse_incident_window_count",
        },
    }


def build_strategy_registry() -> Dict[str, Any]:
    payload = {
        "schema_version": QUERY_ENGINE_SCHEMA_VERSION,
        "protocol": "ordinary",
        "study_mode": "query_only",
        "budget": BUDGET,
        "round_sizes": list(ROUND_SIZES),
        "strategy_ids": list(STRATEGY_IDS),
        "strategy_configs": _strategy_defaults(),
        "annotation_contract": "one_case_reveals_root_cause_plus_fault_type_at_cost_one",
    }
    payload["registry_sha256"] = _semantic_sha256(payload)
    return payload


def _validated_config(strategy_id: Any, config: Any) -> Dict[str, Any]:
    if not isinstance(strategy_id, str) or strategy_id not in STRATEGY_IDS:
        raise ValueError("unknown ordinary strategy_id")
    defaults = deepcopy(_strategy_defaults()[strategy_id])
    if config is None:
        return defaults
    if not isinstance(config, Mapping):
        raise ValueError("strategy config must be a mapping")
    unknown = sorted(set(config).difference(defaults))
    if unknown:
        raise ValueError("strategy config contains undeclared field: %s" % unknown[0])
    defaults.update(deepcopy(dict(config)))
    return defaults


def validate_ordinary_query_candidates(candidates: Any) -> list[Dict[str, Any]]:
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(
        candidates, Sequence
    ):
        raise ValueError("ordinary candidates must be an ordered sequence")
    if len(candidates) < BUDGET:
        raise ValueError("ordinary candidates cannot fill budget 30")
    rows = []
    seen = set()
    dimensions = set()
    for index, source in enumerate(candidates):
        if not isinstance(source, Mapping):
            raise ValueError("ordinary candidate must be a mapping")
        forbidden = sorted(set(source).intersection(_FORBIDDEN_FIELDS))
        if forbidden:
            raise ValueError("forbidden candidate field: %s" % forbidden[0])
        unknown = sorted(set(source).difference(_CANDIDATE_FIELDS))
        if unknown:
            raise ValueError("ordinary candidate contains undeclared field: %s" % unknown[0])
        case_id = source.get("case_id")
        incident_id = source.get("incident_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id.strip() != case_id
            or not isinstance(incident_id, str)
            or not incident_id
            or incident_id.strip() != incident_id
        ):
            raise ValueError("candidate identities must be canonical strings")
        if case_id in seen:
            raise ValueError("candidate case IDs must be unique")
        seen.add(case_id)
        embedding = source.get("embedding")
        if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
            raise ValueError("candidate embedding must be a sequence")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as error:
            raise ValueError("candidate embedding must be finite") from error
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("candidate embedding must be finite")
        dimensions.add(len(vector))
        timestamp = source.get("timestamp")
        uncertainty = source.get("uncertainty", 0.0)
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or isinstance(uncertainty, bool)
            or not isinstance(uncertainty, (int, float))
            or not math.isfinite(float(uncertainty))
        ):
            raise ValueError("candidate timestamp and uncertainty must be finite")
        rows.append(
            {
                "case_id": case_id,
                "embedding": vector,
                "timestamp": float(timestamp),
                "incident_id": incident_id,
                "uncertainty": float(uncertainty),
                "_input_index": index,
            }
        )
    if len(dimensions) != 1:
        raise ValueError("candidate embeddings require one finite dimension")
    return rows


class MappingAnnotationOracle:
    """Reveal joint annotations only for a previously committed case list."""

    def __init__(self, annotations: Any) -> None:
        if not isinstance(annotations, Mapping):
            raise ValueError("annotations must be a mapping")
        self._annotations: Dict[str, Dict[str, str]] = {}
        for case_id, value in annotations.items():
            if not isinstance(case_id, str) or not isinstance(value, Mapping):
                raise ValueError("annotation record is malformed")
            if set(value) != {"root_cause", "fault_type"}:
                raise ValueError("annotation must contain root_cause plus fault_type")
            annotation = {
                "root_cause": str(value["root_cause"]).strip(),
                "fault_type": str(value["fault_type"]).strip(),
            }
            if not all(annotation.values()):
                raise ValueError("joint annotation values must be non-empty")
            self._annotations[case_id] = annotation
        self.reveal_calls: list[list[str]] = []

    def reveal(self, case_ids: Sequence[str]) -> list[Dict[str, Any]]:
        ordered = list(case_ids)
        if not ordered or len(ordered) != len(set(ordered)):
            raise ValueError("reveal requires unique committed case IDs")
        if any(case_id not in self._annotations for case_id in ordered):
            raise ValueError("committed case lacks an annotation")
        self.reveal_calls.append(list(ordered))
        return [
            {
                "case_id": case_id,
                "annotation": deepcopy(self._annotations[case_id]),
                "annotation_cost": 1,
            }
            for case_id in ordered
        ]


def _normalized_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.asarray([row["embedding"] for row in rows], dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (matrix - mean) / scale


def _row_maps(rows: Sequence[Mapping[str, Any]]):
    ids = [row["case_id"] for row in rows]
    return ids, {case_id: index for index, case_id in enumerate(ids)}


def _round_robin_queues(
    queues: Mapping[str, Sequence[str]],
    mode_order: Sequence[str],
    batch_size: int,
) -> list[str]:
    mutable = {mode: list(queues[mode]) for mode in mode_order}
    chosen = []
    while len(chosen) < batch_size:
        progressed = False
        for mode in mode_order:
            if mutable[mode] and len(chosen) < batch_size:
                chosen.append(mutable[mode].pop(0))
                progressed = True
        if not progressed:
            break
    return chosen


def _score_rows_base(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        row["case_id"]: {
            "case_id": row["case_id"],
            "primary_score": 0.0,
            "reason_code": "candidate_scored",
        }
        for row in rows
    }


def _authority_kmeans_candidate_rows(rows, *, time_chunk_count):
    ordered = sorted(
        rows,
        key=lambda row: (float(row["timestamp"]), str(row["case_id"])),
    )
    chunks = max(1, int(time_chunk_count))
    bucket_by_case = {
        str(row["case_id"]): "time-%d"
        % min(chunks - 1, index * chunks // len(ordered))
        for index, row in enumerate(ordered)
    }
    return [
        {
            "case_id": str(row["case_id"]),
            "split": "outer_train",
            "case_kind": "fault",
            "embedding": list(row["embedding"]),
            "inner_boundary_uncertainty": float(row["uncertainty"]),
            "baseline_score": 0.0,
            "cluster_id": "__unavailable__",
            "time_bucket": bucket_by_case[str(row["case_id"])],
        }
        for row in rows
    ]


def _authority_kmeans_selector_config(config):
    return {
        "embedding_field": "embedding",
        "numeric_fields": [
            "inner_boundary_uncertainty",
            "baseline_score",
        ],
        "categorical_fields": ["cluster_id", "time_bucket"],
        "score_field": "inner_boundary_uncertainty",
        "proxy_mode_count": int(config["cluster_count"]),
        "max_per_proxy_mode": int(config["max_per_proxy_mode"]),
        "seed_budget": int(config["seed_budget"]),
        "metric_ad_required": False,
    }


def _authority_kmeans_plan(rows, *, seed, config):
    from .query_active_learning import (
        assign_proxy_fault_modes,
        build_metric_signatures,
        select_query_cases,
    )

    candidates = _authority_kmeans_candidate_rows(
        rows,
        time_chunk_count=int(config["time_chunk_count"]),
    )
    selector_config = _authority_kmeans_selector_config(config)
    signatures = build_metric_signatures(
        candidates,
        signature_config={
            key: deepcopy(selector_config[key])
            for key in (
                "embedding_field",
                "numeric_fields",
                "categorical_fields",
                "metric_ad_required",
            )
        },
    )
    proxy = assign_proxy_fault_modes(
        signatures["candidate_signatures"],
        proxy_config={"proxy_mode_count": int(config["cluster_count"])},
        seed=int(seed),
    )
    selection = select_query_cases(
        candidates,
        selector_id="sequential_proxy_mode_query",
        budget=BUDGET,
        seed=int(seed),
        selector_config=selector_config,
    )
    return {
        "selection": selection,
        "assigned_by_case": {
            str(row["case_id"]): row
            for row in proxy["assigned_candidates"]
        },
        "selector_config": selector_config,
        "proxy_assignment_sha256": proxy["proxy_assignment_sha256"],
    }


def _kmeans_snapshot(rows, matrix, selected, batch_size, seed, config):
    del matrix
    authority = _authority_kmeans_plan(rows, seed=seed, config=config)
    selection = authority["selection"]
    full_order = [str(value) for value in selection["selected_case_ids"]]
    if list(selected) != full_order[: len(selected)]:
        raise ValueError("authority K-means selected prefix drifted")
    chosen = full_order[len(selected) : len(selected) + batch_size]
    assigned = authority["assigned_by_case"]
    selected_set = set(selected)
    covered = {
        str(assigned[case_id]["proxy_fault_mode"])
        for case_id in selected_set
    }
    ranks_by_mode = defaultdict(dict)
    for mode in sorted(
        {str(row["proxy_fault_mode"]) for row in assigned.values()}
    ):
        members = sorted(
            (
                row
                for row in assigned.values()
                if str(row["proxy_fault_mode"]) == mode
            ),
            key=lambda row: (
                float(row["proxy_distance"]),
                str(row["case_id"]),
            ),
        )
        ranks_by_mode[mode] = {
            str(row["case_id"]): rank
            for rank, row in enumerate(members)
        }
    authority_rank = {
        case_id: rank
        for rank, case_id in enumerate(full_order)
    }
    scores = _score_rows_base(rows)
    for row in rows:
        case_id = str(row["case_id"])
        assigned_row = assigned[case_id]
        mode = str(assigned_row["proxy_fault_mode"])
        rank = authority_rank.get(case_id)
        scores[case_id].update(
            {
                "proxy_mode_id": mode,
                "representative_distance": float(
                    assigned_row["proxy_distance"]
                ),
                "mode_covered_before_round": mode in covered,
                "within_mode_rank": int(ranks_by_mode[mode][case_id]),
                "authority_plan_rank": rank,
                "primary_score": float(row["uncertainty"]),
                "reason_code": (
                    "authority_proxy_diverse_seed"
                    if rank is not None and rank < int(config["seed_budget"])
                    else "authority_proxy_balanced_fill"
                ),
            }
        )
    nonempty = len(
        {str(row["proxy_fault_mode"]) for row in assigned.values()}
    )
    return scores, chosen, {
        "strategy_id": "kmeans_coverage",
        "authority_selector_id": "sequential_proxy_mode_query",
        "authority_selector_result_sha256": selection[
            "selector_result_sha256"
        ],
        "proxy_assignment_sha256": authority[
            "proxy_assignment_sha256"
        ],
        "nonempty_mode_count": nonempty,
        "empty_mode_count": max(
            0,
            int(config["cluster_count"]) - nonempty,
        ),
    }


def _dbscan_snapshot(rows, matrix, selected, batch_size, seed, config):
    minimum = max(2, int(config["min_samples"]))
    neighbors = NearestNeighbors(n_neighbors=min(minimum, len(rows))).fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)
    kth = distances[:, -1]
    eps = float(np.quantile(kth, float(config["eps_quantile"])))
    if eps <= 0.0:
        positive = kth[kth > 0.0]
        eps = float(positive.min()) if len(positive) else 1e-6
    model = DBSCAN(eps=eps, min_samples=minimum).fit(matrix)
    labels = model.labels_.astype(int)
    core = set(int(index) for index in model.core_sample_indices_)
    ids, by_id = _row_maps(rows)
    selected_set = set(selected)
    selected_noise = sum(int(labels[by_id[case_id]]) == -1 for case_id in selected_set)
    noise_cap = int(config["noise_budget_cap"])
    scores = _score_rows_base(rows)
    queues: Dict[str, list[str]] = {}
    cluster_labels = sorted(set(int(value) for value in labels if int(value) >= 0))
    for label in cluster_labels:
        indices = [index for index, value in enumerate(labels) if int(value) == label]
        centroid = matrix[indices].mean(axis=0)
        distances_to_center = {index: float(np.linalg.norm(matrix[index] - centroid)) for index in indices}
        ordered = sorted(
            [index for index in indices if ids[index] not in selected_set],
            key=lambda index: (distances_to_center[index], _tie(seed, ids[index], "dbscan")),
        )
        queues[str(label)] = [ids[index] for index in ordered]
        boundary_cut = max(1, int(math.ceil(len(indices) * float(config["boundary_fraction"]))))
        boundary = set(sorted(indices, key=lambda index: distances_to_center[index], reverse=True)[:boundary_cut])
        for index in indices:
            role = "core" if index in core else ("boundary" if index in boundary else "member")
            scores[ids[index]].update(
                {
                    "proxy_mode_id": str(label),
                    "density_role": role,
                    "core_distance": float(kth[index]),
                    "representative_distance": distances_to_center[index],
                    "primary_score": -distances_to_center[index],
                    "reason_code": "density_cluster_coverage",
                }
            )
    noise_indices = [index for index, value in enumerate(labels) if int(value) == -1]
    noise_queue = sorted(
        [ids[index] for index in noise_indices if ids[index] not in selected_set],
        key=lambda case_id: _tie(seed, case_id, "dbscan-noise"),
    )
    allowed_noise = max(0, noise_cap - selected_noise)
    queues["noise"] = noise_queue[:allowed_noise]
    for index in noise_indices:
        scores[ids[index]].update(
            {
                "proxy_mode_id": "noise",
                "density_role": "noise",
                "core_distance": float(kth[index]),
                "representative_distance": float(kth[index]),
                "primary_score": float(kth[index]),
                "reason_code": "bounded_density_noise_audit",
            }
        )
    mode_order = sorted(
        [mode for mode in queues if mode != "noise"],
        key=lambda mode: _tie(seed, mode, "dbscan-mode"),
    )
    if queues["noise"]:
        mode_order.append("noise")
    chosen = _round_robin_queues(queues, mode_order, batch_size)
    if len(chosen) < batch_size:
        fallback = [
            case_id
            for case_id in ids
            if case_id not in selected_set and case_id not in chosen and int(labels[by_id[case_id]]) >= 0
        ]
        chosen.extend(sorted(fallback, key=lambda case_id: _tie(seed, case_id, "dbscan-fill"))[: batch_size - len(chosen)])
    if len(chosen) != batch_size:
        raise ValueError("DBSCAN non-noise candidates cannot fill the declared batch under its noise cap")
    final_noise = selected_noise + sum(int(labels[by_id[case_id]]) == -1 for case_id in chosen)
    return scores, chosen, {
        "strategy_id": "dbscan_coverage",
        "cluster_count": len(cluster_labels),
        "noise_case_count": len(noise_indices),
        "selected_noise_count": final_noise,
        "noise_budget_cap": noise_cap,
        "eps": eps,
    }


def _farthest_first(ids, matrix, available_indices, selected_indices, batch_size, seed):
    chosen = []
    reference = list(selected_indices)
    remaining = list(available_indices)
    while remaining and len(chosen) < batch_size:
        if reference:
            distances = pairwise_distances(matrix[remaining], matrix[reference]).min(axis=1)
        else:
            centroid = matrix[remaining].mean(axis=0)
            distances = np.linalg.norm(matrix[remaining] - centroid, axis=1)
        best = max(
            range(len(remaining)),
            key=lambda position: (
                float(distances[position]),
                _tie(seed, ids[remaining[position]], "kff"),
            ),
        )
        index = remaining.pop(best)
        chosen.append(index)
        reference.append(index)
    return chosen


def _knn_predictions(rows, matrix, remaining_indices, selected_indices, annotations, neighbor_count):
    labels = [annotations[rows[index]["case_id"]]["fault_type"] for index in selected_indices]
    result = {}
    for index in remaining_indices:
        distance = np.linalg.norm(matrix[selected_indices] - matrix[index], axis=1)
        order = np.argsort(distance)[: min(neighbor_count, len(selected_indices))]
        weights = defaultdict(float)
        for position in order:
            weights[labels[int(position)]] += 1.0 / max(float(distance[int(position)]), 1e-9)
        total = sum(weights.values())
        normalized = {label: weight / total for label, weight in weights.items()}
        predicted = max(normalized, key=lambda label: (normalized[label], label))
        result[index] = (predicted, normalized[predicted], dict(sorted(normalized.items())))
    return result


def _knn_snapshot(rows, matrix, selected, annotations, batch_size, seed, config):
    ids, by_id = _row_maps(rows)
    selected_indices = [by_id[case_id] for case_id in selected]
    remaining_indices = [index for index, case_id in enumerate(ids) if case_id not in set(selected)]
    scores = _score_rows_base(rows)
    if not selected_indices:
        chosen_indices = _farthest_first(ids, matrix, remaining_indices, [], batch_size, seed)
        for index in remaining_indices:
            scores[ids[index]].update(
                {
                    "selection_stage": "label_free_k_center",
                    "primary_score": 0.0,
                    "reason_code": "no_revealed_fault_type_k_center",
                }
            )
    else:
        predictions = _knn_predictions(
            rows,
            matrix,
            remaining_indices,
            selected_indices,
            annotations,
            max(1, int(config["neighbor_count"])),
        )
        queues = defaultdict(list)
        for index in remaining_indices:
            predicted, confidence, weights = predictions[index]
            queues[predicted].append(index)
            scores[ids[index]].update(
                {
                    "selection_stage": "revealed_label_knn_fault_mode_coverage",
                    "predicted_fault_type": predicted,
                    "prediction_confidence": float(confidence),
                    "neighbor_label_weights": weights,
                    "primary_score": 1.0 - float(confidence),
                    "reason_code": "predicted_mode_coverage_then_within_mode_uncertainty",
                }
            )
        ordered_queues = {}
        for mode, indices in queues.items():
            ordered_queues[mode] = [
                ids[index]
                for index in sorted(
                    indices,
                    key=lambda index: (
                        predictions[index][1],
                        _tie(seed, ids[index], "knn-mode"),
                    ),
                )
            ]
        queried_counts = Counter(annotation["fault_type"] for annotation in annotations.values())
        mode_order = sorted(
            ordered_queues,
            key=lambda mode: (queried_counts[mode], _tie(seed, mode, "knn-types")),
        )
        chosen_ids = _round_robin_queues(ordered_queues, mode_order, batch_size)
        chosen_indices = [by_id[case_id] for case_id in chosen_ids]
    return scores, [ids[index] for index in chosen_indices], {
        "strategy_id": "knn_fault_mode_coverage",
        "revealed_label_count": len(selected_indices),
        "neighbor_count": int(config["neighbor_count"]),
    }


def _mutual_knn_snapshot(rows, matrix, selected, batch_size, seed, config):
    ids, by_id = _row_maps(rows)
    neighbor_count = min(max(1, int(config["neighbor_count"])), len(rows) - 1)
    distances, neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(matrix).kneighbors(matrix)
    directed = [set(int(value) for value in row[1:]) for row in neighbors]
    adjacency = np.zeros((len(rows), len(rows)), dtype=float)
    for left in range(len(rows)):
        for right_position, right in enumerate(neighbors[left][1:], start=1):
            right = int(right)
            if left in directed[right]:
                weight = max(float(distances[left][right_position]), 1e-12)
                adjacency[left, right] = weight
                adjacency[right, left] = weight
    graph = csr_matrix(adjacency)
    component_count, component_labels = connected_components(graph, directed=False)
    geodesic = shortest_path(graph, directed=False, unweighted=False)
    selected_set = set(selected)
    scores = _score_rows_base(rows)
    queues = {}
    tiny_size = int(config["tiny_component_size"])
    tiny_cap = int(config["tiny_component_budget_cap"])
    selected_tiny = 0
    for component in range(component_count):
        members = [index for index, label in enumerate(component_labels) if int(label) == component]
        selected_members = [by_id[case_id] for case_id in selected if int(component_labels[by_id[case_id]]) == component]
        if len(members) <= tiny_size:
            selected_tiny += len(selected_members)
        candidates = [index for index in members if ids[index] not in selected_set]
        if selected_members:
            coverage = np.min(geodesic[np.ix_(candidates, selected_members)], axis=1) if candidates else np.asarray([])
        else:
            sub = geodesic[np.ix_(members, members)]
            finite = np.where(np.isfinite(sub), sub, 0.0)
            central = finite.sum(axis=1)
            representative = members[int(np.argmin(central))]
            coverage = np.asarray(
                [float(geodesic[index, representative]) if np.isfinite(geodesic[index, representative]) else 0.0 for index in candidates]
            )
        order = sorted(
            range(len(candidates)),
            key=lambda position: (
                -float(coverage[position]),
                _tie(seed, ids[candidates[position]], "mutual-knn"),
            ),
        )
        queue = [ids[candidates[position]] for position in order]
        if len(members) <= tiny_size:
            queue = queue[: max(0, tiny_cap - selected_tiny)]
        queues[str(component)] = queue
        coverage_by_index = {candidates[position]: float(coverage[position]) for position in range(len(candidates))}
        for index in members:
            scores[ids[index]].update(
                {
                    "component_id": str(component),
                    "component_size": len(members),
                    "geodesic_coverage_distance": coverage_by_index.get(index, 0.0),
                    "primary_score": coverage_by_index.get(index, 0.0),
                    "reason_code": "uncovered_component_or_conditional_graph_k_center",
                }
            )
    mode_order = sorted(
        queues,
        key=lambda component: (
            any(int(component_labels[by_id[case_id]]) == int(component) for case_id in selected),
            -sum(int(value) == int(component) for value in component_labels),
            _tie(seed, component, "mutual-component"),
        ),
    )
    chosen = _round_robin_queues(queues, mode_order, batch_size)
    if len(chosen) < batch_size:
        fallback = [case_id for case_id in ids if case_id not in selected_set and case_id not in chosen]
        chosen.extend(sorted(fallback, key=lambda case_id: _tie(seed, case_id, "mutual-fill"))[: batch_size - len(chosen)])
    return scores, chosen, {
        "strategy_id": "mutual_knn_graph_coverage",
        "component_count": int(component_count),
        "neighbor_count": neighbor_count,
    }


def _falcon_snapshot(rows, matrix, selected, batch_size, seed, config):
    ids, by_id = _row_maps(rows)
    selected_set = set(selected)
    available = [index for index, case_id in enumerate(ids) if case_id not in selected_set]
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=float)
    chunk_count = max(1, int(config["time_chunk_count"]))
    order = np.argsort(timestamps, kind="stable")
    chunks = np.zeros(len(rows), dtype=int)
    for rank, index in enumerate(order):
        chunks[index] = min(chunk_count - 1, int(rank * chunk_count / len(rows)))
    explore_count = max(
        int(config["minimum_exploration_per_round"]),
        int(round(batch_size * float(config["exploration_fraction"]))),
    )
    explore_count = min(batch_size - 1 if batch_size > 1 else 1, explore_count)
    exploit_count = batch_size - explore_count
    scores = _score_rows_base(rows)
    uncertainty = {index: float(rows[index].get("uncertainty", 0.0)) for index in available}
    lc_order = sorted(
        available,
        key=lambda index: (-uncertainty[index], int(chunks[index]), _tie(seed, ids[index], "falcon-lc")),
    )
    chosen_lc = lc_order[:exploit_count]
    remaining = [index for index in available if index not in chosen_lc]
    selected_indices = [by_id[case_id] for case_id in selected] + chosen_lc
    chosen_kff = _farthest_first(ids, matrix, remaining, selected_indices, explore_count, seed)
    chosen_indices = []
    for offset in range(max(len(chosen_lc), len(chosen_kff))):
        if offset < len(chosen_lc):
            chosen_indices.append(chosen_lc[offset])
        if offset < len(chosen_kff):
            chosen_indices.append(chosen_kff[offset])
    kff_reference = [by_id[case_id] for case_id in selected]
    for index in available:
        if kff_reference:
            kff_distance = float(np.min(np.linalg.norm(matrix[kff_reference] - matrix[index], axis=1)))
        else:
            kff_distance = float(np.linalg.norm(matrix[index] - matrix[available].mean(axis=0)))
        component = "least_confidence" if index in chosen_lc else (
            "kernel_furthest_first" if index in chosen_kff else (
                "least_confidence" if uncertainty[index] >= 0.5 else "kernel_furthest_first"
            )
        )
        scores[ids[index]].update(
            {
                "timestamp_chunk": int(chunks[index]),
                "least_confidence_score": uncertainty[index],
                "kff_distance": kff_distance,
                "selection_component": component,
                "primary_score": uncertainty[index] if component == "least_confidence" else kff_distance,
                "reason_code": "falcon_real_time_lc_kff_hybrid",
            }
        )
    return scores, [ids[index] for index in chosen_indices[:batch_size]], {
        "strategy_id": "falcon_hybrid",
        "time_chunk_count": chunk_count,
        "exploration_fraction": float(config["exploration_fraction"]),
        "annotation_cost_per_case": 1,
    }


def _hdbscan_snapshot(rows, matrix, selected, batch_size, seed, config):
    model = HDBSCAN(
        min_cluster_size=max(2, int(config["min_cluster_size"])),
        min_samples=max(1, int(config["min_samples"])),
        store_centers="medoid",
        allow_single_cluster=True,
    ).fit(matrix)
    labels = model.labels_.astype(int)
    probabilities = np.asarray(model.probabilities_, dtype=float)
    ids, by_id = _row_maps(rows)
    selected_set = set(selected)
    selected_noise = sum(int(labels[by_id[case_id]]) == -1 for case_id in selected)
    noise_cap = int(config["noise_budget_cap"])
    scores = _score_rows_base(rows)
    queues = {}
    cluster_labels = sorted(set(int(value) for value in labels if int(value) >= 0))
    for label in cluster_labels:
        indices = [index for index, value in enumerate(labels) if int(value) == label]
        stability = float(len(indices) / len(rows) * probabilities[indices].mean())
        ordered = sorted(
            [index for index in indices if ids[index] not in selected_set],
            key=lambda index: (-probabilities[index], _tie(seed, ids[index], "hdbscan")),
        )
        queues[str(label)] = [ids[index] for index in ordered]
        for index in indices:
            scores[ids[index]].update(
                {
                    "proxy_mode_id": str(label),
                    "membership_probability": float(probabilities[index]),
                    "glosh_outlier_score": float(1.0 - probabilities[index]),
                    "branch_stability": stability,
                    "primary_score": float(probabilities[index]),
                    "reason_code": "stable_branch_exemplar_or_bounded_boundary",
                }
            )
    noise_indices = [index for index, value in enumerate(labels) if int(value) == -1]
    allowed_noise = max(0, noise_cap - selected_noise)
    queues["noise"] = [
        ids[index]
        for index in sorted(
            [index for index in noise_indices if ids[index] not in selected_set],
            key=lambda index: (-float(1.0 - probabilities[index]), _tie(seed, ids[index], "hdbscan-noise")),
        )[:allowed_noise]
    ]
    for index in noise_indices:
        scores[ids[index]].update(
            {
                "proxy_mode_id": "noise",
                "membership_probability": float(probabilities[index]),
                "glosh_outlier_score": float(1.0 - probabilities[index]),
                "branch_stability": 0.0,
                "primary_score": float(1.0 - probabilities[index]),
                "reason_code": "bounded_hdbscan_outlier_audit",
            }
        )
    mode_order = sorted(
        [mode for mode in queues if mode != "noise"],
        key=lambda mode: _tie(seed, mode, "hdbscan-mode"),
    )
    if queues["noise"]:
        mode_order.append("noise")
    chosen = _round_robin_queues(queues, mode_order, batch_size)
    if len(chosen) < batch_size:
        fallback = [
            case_id for case_id in ids
            if case_id not in selected_set and case_id not in chosen and int(labels[by_id[case_id]]) >= 0
        ]
        chosen.extend(sorted(fallback, key=lambda case_id: _tie(seed, case_id, "hdbscan-fill"))[: batch_size - len(chosen)])
    if len(chosen) != batch_size:
        raise ValueError("HDBSCAN stable branches cannot fill batch under noise cap")
    final_noise = selected_noise + sum(int(labels[by_id[case_id]]) == -1 for case_id in chosen)
    return scores, chosen, {
        "strategy_id": "hdbscan_coverage",
        "cluster_count": len(cluster_labels),
        "noise_case_count": len(noise_indices),
        "selected_noise_count": final_noise,
        "glosh_definition": "one_minus_sklearn_hdbscan_membership_probability",
    }


def _facility_snapshot(rows, matrix, selected, batch_size, seed, config):
    ids, by_id = _row_maps(rows)
    distance = pairwise_distances(matrix)
    positive = distance[distance > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    temperature = float(config["kernel_temperature"])
    if temperature <= 0.0:
        raise ValueError("facility kernel_temperature must be positive")
    similarity = np.exp(-np.square(distance) / (2.0 * (scale * temperature) ** 2))
    incident_counts = Counter(row["incident_id"] for row in rows)
    weights = np.asarray([1.0 / incident_counts[row["incident_id"]] for row in rows], dtype=float)
    selected_indices = [by_id[case_id] for case_id in selected]
    coverage = np.max(similarity[:, selected_indices], axis=1) if selected_indices else np.zeros(len(rows))
    remaining = [index for index, case_id in enumerate(ids) if case_id not in set(selected)]
    scores = _score_rows_base(rows)
    chosen = []
    chosen_gains = {}
    while remaining and len(chosen) < batch_size:
        gains = {
            index: float(np.sum(weights * np.maximum(coverage, similarity[:, index]) - weights * coverage))
            for index in remaining
        }
        best = max(
            remaining,
            key=lambda index: (gains[index], _tie(seed, ids[index], "facility")),
        )
        chosen.append(best)
        chosen_gains[best] = gains[best]
        coverage = np.maximum(coverage, similarity[:, best])
        remaining.remove(best)
    baseline_coverage = np.max(similarity[:, selected_indices], axis=1) if selected_indices else np.zeros(len(rows))
    for index, case_id in enumerate(ids):
        gain = float(np.sum(weights * np.maximum(baseline_coverage, similarity[:, index]) - weights * baseline_coverage))
        if index in chosen_gains:
            gain = chosen_gains[index]
        scores[case_id].update(
            {
                "marginal_gain": gain,
                "incident_weight": float(weights[index]),
                "primary_score": gain,
                "reason_code": "conditional_incident_weighted_facility_gain",
            }
        )
    return scores, [ids[index] for index in chosen], {
        "strategy_id": "graph_facility_location",
        "incident_weighting": "inverse_incident_window_count",
        "kernel_scale": scale,
    }


def _strategy_snapshot(rows, matrix, selected, annotations, strategy_id, batch_size, seed, config):
    if strategy_id == "kmeans_coverage":
        return _kmeans_snapshot(rows, matrix, selected, batch_size, seed, config)
    if strategy_id == "dbscan_coverage":
        return _dbscan_snapshot(rows, matrix, selected, batch_size, seed, config)
    if strategy_id == "knn_fault_mode_coverage":
        return _knn_snapshot(rows, matrix, selected, annotations, batch_size, seed, config)
    if strategy_id == "mutual_knn_graph_coverage":
        return _mutual_knn_snapshot(rows, matrix, selected, batch_size, seed, config)
    if strategy_id == "falcon_hybrid":
        return _falcon_snapshot(rows, matrix, selected, batch_size, seed, config)
    if strategy_id == "hdbscan_coverage":
        return _hdbscan_snapshot(rows, matrix, selected, batch_size, seed, config)
    if strategy_id == "graph_facility_location":
        return _facility_snapshot(rows, matrix, selected, batch_size, seed, config)
    raise AssertionError("unreachable strategy")


def _append_event(events: list[Dict[str, Any]], payload: Mapping[str, Any]) -> Dict[str, Any]:
    event = deepcopy(dict(payload))
    event["schema_version"] = EVENT_SCHEMA_VERSION
    event["sequence"] = len(events)
    event["previous_event_sha256"] = events[-1]["event_sha256"] if events else "0" * 64
    event["event_sha256"] = _semantic_sha256(event)
    events.append(event)
    return event


def validate_query_event_log(events: Any) -> Dict[str, Any]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("query event log must be a sequence")
    previous = "0" * 64
    pending = None
    selected = []
    annotation_cost = 0
    for sequence, source in enumerate(events):
        if not isinstance(source, Mapping):
            raise ValueError("query event is malformed")
        event = dict(source)
        digest = event.pop("event_sha256", None)
        if not isinstance(digest, str) or _semantic_sha256(event) != digest:
            raise ValueError("query event hash drifted")
        if event.get("sequence") != sequence or event.get("previous_event_sha256") != previous:
            raise ValueError("query event hash chain drifted")
        previous = digest
        event_type = event.get("event_type")
        if event_type == "selection_commit":
            if pending is not None:
                raise ValueError("selection commit occurred before pending reveal")
            case_ids = list(event.get("case_ids") or ())
            if not case_ids or len(case_ids) != len(set(case_ids)) or set(case_ids).intersection(selected):
                raise ValueError("selection commit contains duplicate cost")
            selected.extend(case_ids)
            pending = case_ids
        elif event_type == "annotation_reveal":
            if pending is None or list(event.get("case_ids") or ()) != pending:
                raise ValueError("annotation reveal lacks matching commit")
            reveals = event.get("reveals")
            if not isinstance(reveals, list) or len(reveals) != len(pending):
                raise ValueError("annotation reveal records are incomplete")
            annotation_cost += sum(int(row.get("annotation_cost", 0)) for row in reveals)
            pending = None
        elif event_type == "selector_update":
            if pending is not None:
                raise ValueError("selector updated before annotation reveal")
        else:
            raise ValueError("unknown query event type")
    return {
        "selected_case_ids": selected,
        "annotation_cost": annotation_cost,
        "pending_case_ids": pending,
        "event_count": len(events),
        "last_event_sha256": previous,
    }


def _annotations_from_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    result = {}
    for event in events:
        if event.get("event_type") == "annotation_reveal":
            for row in event["reveals"]:
                result[str(row["case_id"])] = deepcopy(dict(row["annotation"]))
    return result


def _append_reveal_and_update(events, oracle, committed_event, annotations):
    reveals = oracle.reveal(committed_event["case_ids"])
    if [row["case_id"] for row in reveals] != list(committed_event["case_ids"]):
        raise ValueError("annotation oracle changed committed order")
    for row in reveals:
        if set(row) != {"case_id", "annotation", "annotation_cost"} or row["annotation_cost"] != 1:
            raise ValueError("joint reveal must cost exactly one per case")
        if set(row["annotation"]) != {"root_cause", "fault_type"}:
            raise ValueError("joint reveal is incomplete")
    _append_event(
        events,
        {
            "event_type": "annotation_reveal",
            "round_index": committed_event["round_index"],
            "case_ids": list(committed_event["case_ids"]),
            "reveals": reveals,
        },
    )
    for row in reveals:
        annotations[row["case_id"]] = deepcopy(dict(row["annotation"]))
    state = {
        "selected_case_ids": [
            case_id
            for event in events
            if event.get("event_type") == "selection_commit"
            for case_id in event["case_ids"]
        ],
        "revealed_annotations": annotations,
    }
    _append_event(
        events,
        {
            "event_type": "selector_update",
            "round_index": committed_event["round_index"],
            "covered_fault_types": sorted(
                {annotation["fault_type"] for annotation in annotations.values()}
            ),
            "cumulative_annotation_cost": len(annotations),
            "selector_state_sha256": _semantic_sha256(state),
        },
    )


def _run_engine(
    candidates,
    *,
    annotation_oracle,
    strategy_id,
    active_learning_seed,
    config,
    prior_events,
):
    rows = validate_ordinary_query_candidates(candidates)
    if isinstance(active_learning_seed, bool) or not isinstance(active_learning_seed, int) or active_learning_seed <= 0:
        raise ValueError("active_learning_seed must be a positive integer")
    if not hasattr(annotation_oracle, "reveal"):
        raise ValueError("annotation_oracle must expose reveal")
    strategy_config = _validated_config(strategy_id, config)
    events = deepcopy(list(prior_events or ()))
    state = validate_query_event_log(events)
    selected = list(state["selected_case_ids"])
    annotations = _annotations_from_events(events)
    matrix = _normalized_matrix(rows)
    commit_events = [event for event in events if event["event_type"] == "selection_commit"]
    if state["pending_case_ids"] is not None:
        pending_commit = commit_events[-1]
        _append_reveal_and_update(events, annotation_oracle, pending_commit, annotations)
    start_round = len(commit_events)
    strategy_summary = {"strategy_id": strategy_id}
    for round_index in range(start_round, len(ROUND_SIZES)):
        batch_size = ROUND_SIZES[round_index]
        scores, chosen, strategy_summary = _strategy_snapshot(
            rows,
            matrix,
            selected,
            annotations,
            strategy_id,
            batch_size,
            active_learning_seed,
            strategy_config,
        )
        if len(chosen) != batch_size or len(chosen) != len(set(chosen)) or set(chosen).intersection(selected):
            raise ValueError("strategy failed exact unique round budget")
        score_rows = [scores[row["case_id"]] for row in rows if row["case_id"] not in set(selected)]
        commit = _append_event(
            events,
            {
                "event_type": "selection_commit",
                "round_index": round_index,
                "round_size": batch_size,
                "strategy_id": strategy_id,
                "active_learning_seed": active_learning_seed,
                "case_ids": list(chosen),
                "candidate_scores": score_rows,
                "strategy_config_sha256": _semantic_sha256(strategy_config),
                "precommit_state_sha256": _semantic_sha256(
                    {"selected_case_ids": selected, "annotations": annotations}
                ),
                "rng_state_sha256": _semantic_sha256(
                    {"seed": active_learning_seed, "round_index": round_index}
                ),
            },
        )
        selected.extend(chosen)
        _append_reveal_and_update(events, annotation_oracle, commit, annotations)
    validated = validate_query_event_log(events)
    if len(selected) != BUDGET or validated["annotation_cost"] != BUDGET:
        raise ValueError("ordinary query engine did not close exact budget")
    query_plan_payload = {
        "strategy_id": strategy_id,
        "active_learning_seed": active_learning_seed,
        "strategy_config": strategy_config,
        "selected_case_ids": selected,
        "round_sizes": list(ROUND_SIZES),
    }
    final_state = {
        "selected_case_ids": selected,
        "annotations": annotations,
        "last_event_sha256": validated["last_event_sha256"],
    }
    if strategy_id in ("dbscan_coverage", "hdbscan_coverage"):
        # The final round summary already contains the cumulative noise count.
        pass
    return {
        "schema_version": QUERY_ENGINE_SCHEMA_VERSION,
        "protocol": "ordinary",
        "study_mode": "query_only",
        "strategy_id": strategy_id,
        "active_learning_seed": active_learning_seed,
        "strategy_config": strategy_config,
        "budget": BUDGET,
        "round_sizes": list(ROUND_SIZES),
        "selected_case_ids": selected,
        "annotation_cost": validated["annotation_cost"],
        "events": events,
        "strategy_summary": strategy_summary,
        "query_plan_sha256": _semantic_sha256(query_plan_payload),
        "final_state_sha256": _semantic_sha256(final_state),
    }


def run_ordinary_query_engine(
    candidates: Any,
    *,
    annotation_oracle: Any,
    strategy_id: Any,
    active_learning_seed: Any,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return _run_engine(
        candidates,
        annotation_oracle=annotation_oracle,
        strategy_id=strategy_id,
        active_learning_seed=active_learning_seed,
        config=config,
        prior_events=(),
    )


def resume_ordinary_query_engine(
    candidates: Any,
    *,
    annotation_oracle: Any,
    strategy_id: Any,
    active_learning_seed: Any,
    prior_events: Any,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return _run_engine(
        candidates,
        annotation_oracle=annotation_oracle,
        strategy_id=strategy_id,
        active_learning_seed=active_learning_seed,
        config=config,
        prior_events=prior_events,
    )


__all__ = [
    "BUDGET",
    "EVENT_SCHEMA_VERSION",
    "MappingAnnotationOracle",
    "QUERY_ENGINE_SCHEMA_VERSION",
    "ROUND_SIZES",
    "STRATEGY_IDS",
    "build_strategy_registry",
    "resume_ordinary_query_engine",
    "run_ordinary_query_engine",
    "validate_ordinary_query_candidates",
    "validate_query_event_log",
]
