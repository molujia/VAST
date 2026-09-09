"""Feature builders for logs, metrics, traces, and topology."""

from .artifacts import WindowFeatureBundle, WindowRecord
from .entities import EntityIndex, build_entity_index
from .topology import TopologyArtifacts, build_topology_artifacts
from .window_features import build_window_feature_bundle


def build_entity_indices_and_topologies(*args, **kwargs):
    from .build_topology import build_entity_indices_and_topologies as _build_entity_indices_and_topologies

    return _build_entity_indices_and_topologies(*args, **kwargs)


def build_window_feature_artifacts(*args, **kwargs):
    from .build_window_features import build_window_feature_artifacts as _build_window_feature_artifacts

    return _build_window_feature_artifacts(*args, **kwargs)


__all__ = [
    "EntityIndex",
    "TopologyArtifacts",
    "WindowFeatureBundle",
    "WindowRecord",
    "build_entity_index",
    "build_entity_indices_and_topologies",
    "build_window_feature_artifacts",
    "build_window_feature_bundle",
    "build_topology_artifacts",
]
