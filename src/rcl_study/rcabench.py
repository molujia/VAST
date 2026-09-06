"""Canonical metric preprocessing for one rcabench case.

The adapter follows ``HD3_PARQUET_PREPROCESSING.md``.  It deliberately stops
at canonical series construction; anomaly detection is a separate artifact
stage.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .artifacts import SERIES_SCHEMA_VERSION


CANONICAL_DATASET_ID = "rcabench"

_FILE_SPECS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("normal", "metrics", ("value",)),
    ("abnormal", "metrics", ("value",)),
    ("normal", "metrics_sum", ("value",)),
    ("abnormal", "metrics_sum", ("value",)),
    ("normal", "metrics_histogram", ("count", "sum", "min", "max")),
    ("abnormal", "metrics_histogram", ("count", "sum", "min", "max")),
)
_TIME_COLUMNS = frozenset(("time", "timestamp"))
_VALUE_COLUMNS = frozenset(("value", "count", "sum", "min", "max"))
_METRIC_NAME_COLUMNS = ("metric_name", "metric")
_SOURCE_COLUMNS = ("source_workload", "attr.source_workload", "attr.source")
_DESTINATION_COLUMNS = (
    "destination_workload",
    "attr.destination_workload",
    "attr.destination",
)
_ENDPOINT_COLUMNS = frozenset(_SOURCE_COLUMNS + _DESTINATION_COLUMNS)
_REPLICASET_SUFFIX = re.compile(r"-[0-9a-fA-F]{8,10}$")
_POD_SUFFIX = re.compile(r"-[0-9a-fA-F]{8,10}-[a-z0-9]{5}$")


def normalize_timestamps(series: pd.Series) -> Tuple[List[float], str]:
    """Convert one Parquet time column to UTC epoch seconds.

    Numeric units are inferred from the median absolute finite magnitude,
    exactly as specified in the preprocessing guide.
    """

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        converted = pd.to_datetime(series, utc=True, errors="coerce")
        return [
            float(value.timestamp()) if not pd.isna(value) else float("nan")
            for value in converted
        ], "datetime"

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, copy=False))]
    if finite.empty:
        magnitude = 0.0
    else:
        magnitude = float(np.median(np.abs(finite.to_numpy(dtype=float))))

    if magnitude > 1e17:
        divisor, unit = 1e9, "nanoseconds"
    elif magnitude > 1e14:
        divisor, unit = 1e6, "microseconds"
    elif magnitude > 1e11:
        divisor, unit = 1e3, "milliseconds"
    else:
        divisor, unit = 1.0, "seconds"
    return (numeric / divisor).tolist(), unit


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    if result.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return result


def _first_text(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        value = _text(row.get(column))
        if value:
            return value
    return ""


def _without_namespace(value: str) -> str:
    return value.rsplit("/", 1)[-1]


def _service_from_replicaset(value: str) -> str:
    return _REPLICASET_SUFFIX.sub("", _without_namespace(value))


def _service_from_pod(value: str) -> str:
    without_namespace = _without_namespace(value)
    stripped = _POD_SUFFIX.sub("", without_namespace)
    if stripped == without_namespace:
        stripped = _REPLICASET_SUFFIX.sub("", without_namespace)
    return stripped


def _endpoint_service(value: str) -> str:
    return _service_from_pod(value)


def _pod(row: Mapping[str, Any]) -> str:
    return _first_text(row, ("attr.k8s.pod.name", "pod", "pod_name"))


def _ordinary_service(row: Mapping[str, Any]) -> str:
    direct = _first_text(row, ("service",))
    if direct:
        return direct
    named = _first_text(row, ("service_name",))
    if named:
        return named
    replicaset = _first_text(
        row, ("attr.k8s.replicaset.name", "replicaset_name")
    )
    if replicaset:
        return _service_from_replicaset(replicaset)
    deployment = _first_text(
        row, ("attr.k8s.deployment.name", "deployment_name")
    )
    if deployment:
        return deployment
    statefulset = _first_text(
        row, ("attr.k8s.statefulset.name", "statefulset_name")
    )
    if statefulset:
        return statefulset
    pod = _pod(row)
    if pod:
        return _service_from_pod(pod)
    return ""


def _metric_name(row: Mapping[str, Any], filename: str) -> str:
    name = _first_text(row, _METRIC_NAME_COLUMNS)
    if not name:
        raise ValueError(
            "%s contains a row without metric_name or metric" % filename
        )
    return name


def _identity_labels(row: Mapping[str, Any]) -> Dict[str, str]:
    excluded = (
        _TIME_COLUMNS
        | _VALUE_COLUMNS
        | frozenset(_METRIC_NAME_COLUMNS)
        | _ENDPOINT_COLUMNS
    )
    labels: Dict[str, str] = {}
    for column in sorted(row):
        if column in excluded:
            continue
        value = _text(row.get(column))
        if value:
            labels[str(column)] = value
    return labels


def _metric_id(
    *,
    modality: str,
    value_column: str,
    metric_name: str,
    service: str,
    pod: str,
    source: str,
    destination: str,
    labels: Mapping[str, str],
) -> str:
    # Keep the documented identity-field order in the encoded string.  A JSON
    # object with sort_keys=True would sort by field name, so arbitrary label
    # keys could dominate the stable series ordering.
    fields = (
        ("modality", modality),
        ("value_column", value_column),
        ("metric_name", metric_name),
        ("service", service),
        ("pod", pod),
        ("source", source),
        ("destination", destination),
        (
            "labels",
            json.dumps(
                dict(labels),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )
    return "|".join(
        "%s=%s"
        % (
            name,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        )
        for name, value in fields
    )


def _canonical_observation(
    *,
    case_id: str,
    window: str,
    modality: str,
    value_column: str,
    timestamp: float,
    value: float,
    raw: Mapping[str, Any],
    filename: str,
) -> Dict[str, Any]:
    base_name = _metric_name(raw, filename)
    source = ""
    destination = ""
    service = ""
    expanded_name = base_name

    if "hubble" in base_name.lower():
        source_candidate = _first_text(raw, _SOURCE_COLUMNS)
        destination_candidate = _first_text(raw, _DESTINATION_COLUMNS)
        resolved_source = _endpoint_service(source_candidate)
        resolved_destination = _endpoint_service(destination_candidate)
        if resolved_source and resolved_destination:
            source = resolved_source
            destination = resolved_destination
            service = source
            expanded_name = "%s_%s_%s" % (
                base_name,
                source,
                destination,
            )

    if not service:
        service = _ordinary_service(raw)

    metric_semantic = base_name
    if modality == "metrics_histogram":
        expanded_name = "%s_%s" % (expanded_name, value_column)
        metric_semantic = "%s_%s" % (metric_semantic, value_column)

    pod = _pod(raw)
    labels = _identity_labels(raw)
    return {
        "schema_version": SERIES_SCHEMA_VERSION,
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "case_id": str(case_id),
        "metric_id": _metric_id(
            modality=modality,
            value_column=value_column,
            metric_name=expanded_name,
            service=service,
            pod=pod,
            source=source,
            destination=destination,
            labels=labels,
        ),
        "service": service,
        "pod": pod,
        "source": source,
        "destination": destination,
        "metric_name": expanded_name,
        "metric_semantic": metric_semantic,
        "modality": modality,
        "value_column": value_column,
        "window": window,
        "timestamp": float(timestamp),
        "value": float(value),
    }


def _remove_duplicates_and_conflicts(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    exact: Dict[Tuple[str, str, float, float], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["metric_id"]),
            float(row["timestamp"]),
            float(row["value"]),
        )
        exact.setdefault(key, row)
    deduplicated = list(exact.values())
    exact_duplicates_removed = len(rows) - len(deduplicated)

    values_by_observation: Dict[Tuple[str, str, float], set] = {}
    for row in deduplicated:
        key = (
            str(row["case_id"]),
            str(row["metric_id"]),
            float(row["timestamp"]),
        )
        values_by_observation.setdefault(key, set()).add(float(row["value"]))

    conflicting_observations = {
        key: values
        for key, values in values_by_observation.items()
        if len(values) > 1
    }
    conflicting_series = {
        (case_id, metric_id)
        for case_id, metric_id, _timestamp in conflicting_observations
    }
    retained = [
        row
        for row in deduplicated
        if (str(row["case_id"]), str(row["metric_id"])) not in conflicting_series
    ]
    records = [
        {
            "case_id": case_id,
            "metric_id": metric_id,
            "timestamp": timestamp,
            "values": sorted(values),
        }
        for (case_id, metric_id, timestamp), values in sorted(
            conflicting_observations.items()
        )
    ]
    return retained, {
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_series_count": len(conflicting_series),
        "conflicting_observation_count": len(conflicting_observations),
        "conflict_records": records,
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    observation_keys = [
        (str(row["case_id"]), str(row["metric_id"]), float(row["timestamp"]))
        for row in rows
    ]
    unique = len(observation_keys) == len(set(observation_keys))
    services: Dict[Tuple[str, str], set] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["metric_id"]))
        services.setdefault(key, set()).add(str(row["service"]))
    one_service = all(len(values) == 1 for values in services.values())
    stable_order = observation_keys == sorted(observation_keys)
    invariants = {
        "unique_case_metric_timestamp": unique,
        "one_service_per_series": one_service,
        "stable_order": stable_order,
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise ValueError("canonical rcabench invariants failed: %s" % ", ".join(failed))
    return invariants


def canonicalize_case(case_id: str, case_dir: Path) -> Dict[str, Any]:
    """Canonicalize the six documented metric inputs for one case directory."""

    case_dir = Path(case_dir)
    rows: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {
        "existing_file_count": 0,
        "input_file_count": 0,
        "empty_file_count": 0,
        "missing_file_count": 0,
        "non_finite_row_count": 0,
        "unresolved_service_row_count": 0,
        "timestamp_units": {},
    }

    for window, modality, value_columns in _FILE_SPECS:
        filename = "%s_%s.parquet" % (window, modality)
        path = case_dir / filename
        if not path.is_file():
            audit["missing_file_count"] += 1
            continue
        audit["existing_file_count"] += 1
        frame = pd.read_parquet(path)
        if frame.empty:
            audit["empty_file_count"] += 1
            continue
        audit["input_file_count"] += 1

        if "time" in frame.columns:
            time_column = "time"
        elif "timestamp" in frame.columns:
            time_column = "timestamp"
        else:
            raise ValueError(
                "%s must contain an explicit time or timestamp column" % filename
            )
        missing_value_columns = [
            column for column in value_columns if column not in frame.columns
        ]
        if missing_value_columns:
            raise ValueError(
                "%s is missing value columns: %s"
                % (filename, ", ".join(missing_value_columns))
            )

        timestamps, timestamp_unit = normalize_timestamps(frame[time_column])
        audit["timestamp_units"][filename] = timestamp_unit
        for row_position, (_index, raw_series) in enumerate(frame.iterrows()):
            raw = raw_series.to_dict()
            timestamp = float(timestamps[row_position])
            for value_column in value_columns:
                value = pd.to_numeric(
                    pd.Series([raw.get(value_column)]), errors="coerce"
                ).iloc[0]
                try:
                    finite = math.isfinite(timestamp) and math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    audit["non_finite_row_count"] += 1
                    continue
                observation = _canonical_observation(
                    case_id=str(case_id),
                    window=window,
                    modality=modality,
                    value_column=value_column,
                    timestamp=timestamp,
                    value=float(value),
                    raw=raw,
                    filename=filename,
                )
                if not observation["service"]:
                    audit["unresolved_service_row_count"] += 1
                    continue
                rows.append(observation)

    rows, duplicate_audit = _remove_duplicates_and_conflicts(rows)
    rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["metric_id"]),
            float(row["timestamp"]),
        )
    )
    audit.update(duplicate_audit)
    audit["output_row_count"] = len(rows)
    audit["output_series_count"] = len(
        {(str(row["case_id"]), str(row["metric_id"])) for row in rows}
    )
    audit["invariants"] = _validate_rows(rows)
    return {"rows": rows, "audit": audit}


__all__ = ["CANONICAL_DATASET_ID", "canonicalize_case", "normalize_timestamps"]
