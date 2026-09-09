"""Deterministic, label-free shared query planning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import AnnotationQuery, SanitizedFeatureView


_PLANNER_VERSION = "shared-query-v1"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _seed_tie(seed: int, window_id: str) -> str:
    return hashlib.sha256(("%d:%s" % (seed, window_id)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SharedQueryPlan:
    dataset: str
    budget: int
    seed: int
    normal_cluster_id: Optional[int]
    queries: Tuple[AnnotationQuery, ...]
    input_hash: str
    planner_version: str = _PLANNER_VERSION

    def __post_init__(self) -> None:
        queries = tuple(self.queries)
        if not self.dataset or not self.input_hash or not self.planner_version:
            raise ValueError("shared query plan identity fields must not be empty")
        if self.budget < 0:
            raise ValueError("query budget must be non-negative")
        if len(queries) > self.budget:
            raise ValueError("query count exceeds the requested budget")
        window_ids = [query.window_id for query in queries]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("shared query plan contains duplicate window IDs")
        if any(query.dataset != self.dataset for query in queries):
            raise ValueError("query dataset does not match its shared plan")
        if tuple(query.query_rank for query in queries) != tuple(range(len(queries))):
            raise ValueError("query ranks must be contiguous and zero-based")
        object.__setattr__(self, "queries", queries)

    @property
    def queried_window_ids(self) -> Tuple[str, ...]:
        return tuple(query.window_id for query in self.queries)

    def _identity_payload(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "budget": self.budget,
            "seed": self.seed,
            "normal_cluster_id": self.normal_cluster_id,
            "queries": [query.to_dict() for query in self.queries],
            "input_hash": self.input_hash,
            "planner_version": self.planner_version,
        }

    @property
    def plan_id(self) -> str:
        return _sha256(self._identity_payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._identity_payload()
        payload["plan_id"] = self.plan_id
        payload["effective_query_count"] = len(self.queries)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SharedQueryPlan":
        plan = cls(
            dataset=str(payload["dataset"]),
            budget=int(payload["budget"]),
            seed=int(payload["seed"]),
            normal_cluster_id=(
                None
                if payload.get("normal_cluster_id") is None
                else int(payload["normal_cluster_id"])
            ),
            queries=tuple(
                AnnotationQuery(
                    dataset=str(item["dataset"]),
                    window_id=str(item["window_id"]),
                    query_rank=int(item["query_rank"]),
                    role=str(item["role"]),
                )
                for item in payload.get("queries", [])
            ),
            input_hash=str(payload["input_hash"]),
            planner_version=str(payload.get("planner_version", _PLANNER_VERSION)),
        )
        expected_plan_id = payload.get("plan_id")
        if expected_plan_id is not None and str(expected_plan_id) != plan.plan_id:
            raise ValueError("serialized shared query plan hash does not match its contents")
        return plan

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: Path) -> "SharedQueryPlan":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _window_feature(view: SanitizedFeatureView, name: str, default: float = 0.0) -> float:
    value = view.window_features.get(name, default)
    return float(value)


def _anomaly_score(view: SanitizedFeatureView) -> float:
    for column in (
        "window_anomaly_score",
        "window_anomaly_score_max",
        "distance_to_normal_centroid",
    ):
        if column in view.window_features:
            return abs(float(view.window_features[column]))
    values = [abs(float(value)) for value in view.window_features.values()]
    return max(values) if values else 0.0


def _entity_signature(view: SanitizedFeatureView) -> Dict[str, float]:
    signature = {}
    for entity in view.entities:
        values = [abs(float(value)) for value in entity.features.values()]
        signature[entity.entity_id] = max(values) if values else 0.0
    return signature


def _signature_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    entity_ids = set(left).union(right)
    return math.sqrt(
        sum(
            (float(left.get(entity_id, 0.0)) - float(right.get(entity_id, 0.0))) ** 2
            for entity_id in entity_ids
        )
    )


def _medoid_key(view: SanitizedFeatureView, seed: int) -> Tuple[Any, ...]:
    explicit_medoid = _window_feature(view, "is_cluster_medoid", 0.0) >= 0.5
    medoid_distance = _window_feature(view, "distance_to_cluster_medoid", float("inf"))
    return (
        0 if explicit_medoid else 1,
        medoid_distance,
        -_anomaly_score(view),
        _seed_tie(seed, view.window_id),
        view.window_id,
    )


def _normal_cluster_id(views: Sequence[SanitizedFeatureView]) -> Optional[int]:
    counts = Counter(
        int(view.cluster_id)
        for view in views
        if view.cluster_id is not None and int(view.cluster_id) != -1
    )
    if not counts:
        return None
    return min(counts, key=lambda cluster_id: (-counts[cluster_id], cluster_id))


def _boundary_order(
    candidates: Sequence[SanitizedFeatureView],
    selected: Sequence[SanitizedFeatureView],
    seed: int,
) -> List[SanitizedFeatureView]:
    remaining = {view.window_id: view for view in candidates}
    ordered = []
    selected_signatures = [_entity_signature(view) for view in selected]
    while remaining:
        candidate_rows = []
        for view in remaining.values():
            signature = _entity_signature(view)
            if selected_signatures:
                diversity = min(
                    _signature_distance(signature, selected_signature)
                    for selected_signature in selected_signatures
                )
            else:
                diversity = math.sqrt(sum(value * value for value in signature.values()))
            candidate_rows.append(
                (
                    -diversity,
                    -_anomaly_score(view),
                    _seed_tie(seed, view.window_id),
                    view.window_id,
                    view,
                    signature,
                )
            )
        candidate_rows.sort(key=lambda item: item[:-2])
        chosen = candidate_rows[0]
        view = chosen[-2]
        signature = chosen[-1]
        ordered.append(view)
        selected_signatures.append(signature)
        del remaining[view.window_id]
    return ordered


def build_shared_query_plan(
    views: Sequence[SanitizedFeatureView],
    budget: int,
    seed: int = 42,
) -> SharedQueryPlan:
    """Allocate a shared query budget without consulting authoritative labels."""

    sanitized_views = tuple(views)
    if not sanitized_views:
        raise ValueError("cannot build a shared query plan without feature views")
    if budget < 0:
        raise ValueError("query budget must be non-negative")
    datasets = {view.dataset for view in sanitized_views}
    if len(datasets) != 1:
        raise ValueError("a shared query plan must contain exactly one dataset")
    window_ids = [view.window_id for view in sanitized_views]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("sanitized feature views contain duplicate window IDs")

    dataset = next(iter(datasets))
    normal_cluster_id = _normal_cluster_id(sanitized_views)
    eligible = [
        view
        for view in sanitized_views
        if view.window_kind == "fault"
        and (normal_cluster_id is None or view.cluster_id != normal_cluster_id)
    ]
    abnormal_clusters = defaultdict(list)
    noise = []
    boundary_candidates = []
    for view in eligible:
        if view.cluster_id == -1:
            noise.append(view)
        elif view.cluster_id is not None:
            abnormal_clusters[int(view.cluster_id)].append(view)
        else:
            boundary_candidates.append(view)

    medoids = []
    for cluster_id in sorted(abnormal_clusters):
        medoid = min(abnormal_clusters[cluster_id], key=lambda view: _medoid_key(view, seed))
        medoids.append(medoid)
        boundary_candidates.extend(
            view
            for view in abnormal_clusters[cluster_id]
            if view.window_id != medoid.window_id
        )
    noise.sort(
        key=lambda view: (
            -_anomaly_score(view),
            _seed_tie(seed, view.window_id),
            view.window_id,
        )
    )

    selected_views = []
    selected_roles = []

    def add_stage(stage_views: Sequence[SanitizedFeatureView], role: str) -> None:
        for view in stage_views:
            if len(selected_views) >= budget:
                return
            selected_views.append(view)
            selected_roles.append(role)

    add_stage(medoids, "abnormal_medoid")
    add_stage(noise, "noise")
    if len(selected_views) < budget:
        add_stage(
            _boundary_order(boundary_candidates, selected_views, seed),
            "boundary_diverse",
        )

    queries = tuple(
        AnnotationQuery(
            dataset=dataset,
            window_id=view.window_id,
            query_rank=query_rank,
            role=selected_roles[query_rank],
        )
        for query_rank, view in enumerate(selected_views)
    )
    input_hash = _sha256(
        [
            view.to_dict()
            for view in sorted(sanitized_views, key=lambda item: item.window_id)
        ]
    )
    return SharedQueryPlan(
        dataset=dataset,
        budget=int(budget),
        seed=int(seed),
        normal_cluster_id=normal_cluster_id,
        queries=queries,
        input_hash=input_hash,
    )
