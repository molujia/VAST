"""Build fail-closed, label-free feature views for pseudo-label strategies."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .contracts import SanitizedEntityFeatureView, SanitizedFeatureView


_WINDOW_IDENTITY_COLUMNS = (
    "dataset",
    "window_id",
    "source_id",
    "window_kind",
    "day",
    "start_ts",
    "end_ts",
)
_FORBIDDEN_TOKENS = {
    "authoritative",
    "culprit",
    "culprits",
    "groundtruth",
    "label",
    "labels",
    "positive",
    "positives",
    "target",
    "targets",
}
_FORBIDDEN_FIELDS = {
    "metadata_json",
}


def _normalized_key(value: object) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def is_label_bearing_field(name: object) -> bool:
    normalized = _normalized_key(name)
    if normalized in _FORBIDDEN_FIELDS:
        return True
    tokens = set(normalized.split("_"))
    if tokens.intersection(_FORBIDDEN_TOKENS):
        return True
    if "ground" in tokens and "truth" in tokens:
        return True
    return normalized.startswith("root_cause") or normalized.startswith("rootcause")


def _validate_requested_features(columns: Sequence[str], frame_name: str) -> Tuple[str, ...]:
    result = tuple(str(column) for column in columns)
    forbidden = [column for column in result if is_label_bearing_field(column)]
    if forbidden:
        raise ValueError(
            "label-bearing feature requested from %s: %s"
            % (frame_name, ", ".join(sorted(forbidden)))
        )
    return result


def _numeric_value(value: Any, column: str) -> float:
    if pd.isna(value):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature %s must be numeric" % column) from exc
    if not math.isfinite(numeric):
        raise ValueError("feature %s must be finite" % column)
    return numeric


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_metadata_value(item)
            for key, item in value.items()
            if not is_label_bearing_field(key)
        }
    if isinstance(value, (list, tuple)):
        return tuple(_sanitize_metadata_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_sanitize_metadata_value(item) for item in value))
    return value


def _safe_metadata(
    metadata: Optional[Mapping[str, Any]],
    safe_metadata_keys: Sequence[str],
) -> Dict[str, Any]:
    if not metadata:
        return {}
    requested = tuple(str(key) for key in safe_metadata_keys)
    forbidden = [key for key in requested if is_label_bearing_field(key)]
    if forbidden:
        raise ValueError(
            "label-bearing metadata key requested: %s" % ", ".join(sorted(forbidden))
        )
    return {
        key: _sanitize_metadata_value(metadata[key])
        for key in requested
        if key in metadata
    }


def sanitize_feature_views(
    windows: pd.DataFrame,
    entity_features: pd.DataFrame,
    entity_feature_columns: Sequence[str],
    window_feature_columns: Sequence[str],
    clustered_windows: Optional[pd.DataFrame] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    safe_metadata_keys: Sequence[str] = (),
) -> Tuple[SanitizedFeatureView, ...]:
    """Return deterministic feature views with no authoritative target fields."""

    entity_columns = _validate_requested_features(entity_feature_columns, "entity_features")
    window_columns = _validate_requested_features(window_feature_columns, "windows")
    required_window_columns = set(_WINDOW_IDENTITY_COLUMNS)
    missing_window_columns = sorted(required_window_columns.difference(windows.columns))
    if missing_window_columns:
        raise ValueError("windows missing required columns: %s" % missing_window_columns)
    required_entity_columns = {"window_id", "entity_id", "entity_type"}
    missing_entity_columns = sorted(required_entity_columns.difference(entity_features.columns))
    if missing_entity_columns:
        raise ValueError("entity_features missing required columns: %s" % missing_entity_columns)
    missing_entity_features = sorted(set(entity_columns).difference(entity_features.columns))
    if missing_entity_features:
        raise ValueError("entity feature columns missing: %s" % missing_entity_features)

    cluster_lookup: Dict[str, Mapping[str, Any]] = {}
    if clustered_windows is not None and not clustered_windows.empty:
        if "window_id" not in clustered_windows.columns:
            raise ValueError("clustered_windows must contain window_id")
        if clustered_windows["window_id"].astype(str).duplicated().any():
            raise ValueError("clustered_windows contains duplicate window_id values")
        cluster_lookup = {
            str(row["window_id"]): row
            for row in clustered_windows.to_dict(orient="records")
        }

    for column in window_columns:
        if column not in windows.columns and not any(
            column in row for row in cluster_lookup.values()
        ):
            raise ValueError("window feature column missing: %s" % column)

    shared_metadata = _safe_metadata(metadata, safe_metadata_keys)
    entity_groups = {
        str(window_id): group.copy()
        for window_id, group in entity_features.groupby("window_id", sort=False)
    }
    views = []
    for window_row in windows.to_dict(orient="records"):
        window_id = str(window_row["window_id"])
        cluster_row = cluster_lookup.get(window_id, {})
        window_feature_values = {}
        for column in window_columns:
            if column in cluster_row:
                value = cluster_row[column]
            else:
                value = window_row[column]
            window_feature_values[column] = _numeric_value(value, column)

        entity_views = []
        entity_group = entity_groups.get(window_id)
        if entity_group is not None:
            sort_columns = ["entity_id"]
            if "entity_index" in entity_group.columns:
                sort_columns = ["entity_index", "entity_id"]
            for entity_row in entity_group.sort_values(sort_columns).to_dict(orient="records"):
                entity_index = entity_row.get("entity_index")
                if pd.isna(entity_index):
                    entity_index = None
                elif entity_index is not None:
                    entity_index = int(entity_index)
                entity_views.append(
                    SanitizedEntityFeatureView(
                        entity_id=str(entity_row["entity_id"]),
                        entity_type=str(entity_row["entity_type"]),
                        entity_index=entity_index,
                        features={
                            column: _numeric_value(entity_row[column], column)
                            for column in entity_columns
                        },
                    )
                )

        cluster_id_value = cluster_row.get("cluster_id", window_row.get("cluster_id"))
        if cluster_id_value is None or pd.isna(cluster_id_value):
            cluster_id = None
        else:
            cluster_id = int(cluster_id_value)
        normal_value = cluster_row.get(
            "is_normal_cluster",
            window_row.get("is_normal_cluster", False),
        )
        views.append(
            SanitizedFeatureView(
                dataset=str(window_row["dataset"]),
                window_id=window_id,
                source_id=str(window_row["source_id"]),
                window_kind=str(window_row["window_kind"]),
                day=str(window_row["day"]),
                start_ts=float(window_row["start_ts"]),
                end_ts=float(window_row["end_ts"]),
                cluster_id=cluster_id,
                is_normal_cluster=bool(normal_value),
                window_features=window_feature_values,
                entities=tuple(entity_views),
                metadata=shared_metadata,
            )
        )
    return tuple(views)
