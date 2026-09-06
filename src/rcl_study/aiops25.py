"""Native aiops25 case manifest and canonical metric adapter."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd

from .artifacts import (
    CASE_SCHEMA_VERSION,
    SERIES_SCHEMA_VERSION,
    validate_canonical_series_rows,
)


CANONICAL_DATASET_ID = "aiops25"
LEGACY_ALIAS = "hd3"
_LOCAL_TIMEZONE = timezone(timedelta(hours=8))
_ISO_UTC = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"
)
_POD_SUFFIX = re.compile(r"-\d+$")
_DATE_SUFFIX = re.compile(r"_\d{4}-\d{2}-\d{2}$")
_PARTITIONS = (
    "apm/service",
    "apm/pod",
    "infra/infra_node",
    "infra/infra_pod",
    "infra/infra_tidb",
    "other",
)
_METADATA_COLUMNS = frozenset(
    (
        "time",
        "instance",
        "cf",
        "device",
        "kpi_key",
        "kpi_name",
        "kubernetes_node",
        "mountpoint",
        "namespace",
        "object_id",
        "object_type",
        "pod",
        "service",
        "sql_type",
        "type",
    )
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "<na>", "unknown"}:
        return ""
    return text


def normalize_service_name(value: Any) -> str:
    text = _text(value).lower().replace("_", "-")
    text = re.sub(r"\s*\(deleted\)\s*$", "", text)
    text = _POD_SUFFIX.sub("", text)
    return text.strip("-")


def _epoch_seconds(value: str) -> float:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return float(parsed.timestamp())


def _parse_input_window(description: Any) -> Tuple[float, float]:
    timestamps = _ISO_UTC.findall(str(description or ""))
    if len(timestamps) < 2:
        raise ValueError("input window does not contain two UTC timestamps")
    start = _epoch_seconds(timestamps[0])
    end = _epoch_seconds(timestamps[1])
    if end <= start:
        raise ValueError("input window end must be after start")
    return start, end


def _values(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        output: List[str] = []
        for item in value:
            output.extend(_values(item))
        return output
    text = _text(value)
    return [text] if text else []


def _target_services(label: Mapping[str, Any]) -> List[str]:
    candidates: List[str] = []
    candidates.extend(_values(label.get("service")))
    if not candidates:
        candidates.extend(_values(label.get("source")))
        candidates.extend(_values(label.get("destination")))
    if not candidates and str(label.get("instance_type", "")).lower() in {
        "service",
        "pod",
        "container",
        "instance",
    }:
        candidates.extend(_values(label.get("instance")))
    output: List[str] = []
    for candidate in candidates:
        normalized = normalize_service_name(candidate)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _method_admission(
    *,
    admitted: bool,
    exclusion_reason: str,
    label: Mapping[str, Any],
    target_services: Sequence[str],
) -> Dict[str, Any]:
    if not admitted:
        return {
            "half_supervise": {
                "admitted": False,
                "reason": exclusion_reason,
            },
            "self_supervise": {
                "full_inference_admitted": False,
                "primary_rcl_evaluable": False,
                "reason": exclusion_reason,
            },
        }
    fault_type = str(label.get("fault_type", "")).strip().lower()
    half_admitted = "pod kill" not in fault_type
    instance_type = str(label.get("instance_type", "")).strip().lower()
    primary = instance_type != "node" and bool(target_services)
    if instance_type == "node":
        self_reason = "node_target_not_service_evaluable"
    elif not target_services:
        self_reason = "no_service_level_target"
    else:
        self_reason = "service_evaluable"
    return {
        "half_supervise": {
            "admitted": half_admitted,
            "reason": "admitted" if half_admitted else "pod_kill",
        },
        "self_supervise": {
            "full_inference_admitted": True,
            "primary_rcl_evaluable": primary,
            "reason": self_reason,
        },
    }


def _read_groundtruth(path: Path) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                uuid = _text(payload.get("uuid"))
                if uuid:
                    labels[uuid] = dict(payload)
    return labels


def build_case_manifest(dataset_root: Path) -> List[Dict[str, Any]]:
    """Build all input cases while retaining both methods' native admission."""

    root = Path(dataset_root)
    input_payload = json.loads((root / "input.json").read_text(encoding="utf-8"))
    input_rows = (
        input_payload
        if isinstance(input_payload, list)
        else input_payload.get("cases", [])
    )
    labels = _read_groundtruth(root / "groundtruth.jsonl")
    rows: List[Dict[str, Any]] = []
    for item in input_rows:
        native_case_id = _text(item.get("uuid"))
        if not native_case_id:
            continue
        label = labels.get(native_case_id)
        exclusion_reason = ""
        try:
            window_start, window_end = _parse_input_window(
                item.get("Anomaly Description")
            )
        except ValueError:
            window_start, window_end = 0.0, 0.0
            exclusion_reason = "input_window_unparseable"
        if label is None and not exclusion_reason:
            exclusion_reason = "missing_groundtruth"
        admitted = not bool(exclusion_reason)
        target_services = _target_services(label or {})
        width = window_end - window_start if admitted else 0.0
        normal_intervals = (
            [
                [window_start - width, window_start],
                [window_end, window_end + width],
            ]
            if admitted
            else []
        )
        rows.append(
            {
                "schema_version": CASE_SCHEMA_VERSION,
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "legacy_alias": LEGACY_ALIAS,
                "case_id": "%s::%s"
                % (CANONICAL_DATASET_ID, native_case_id),
                "native_case_id": native_case_id,
                "admitted": admitted,
                "exclusion_reason": exclusion_reason,
                "method_native_admission": _method_admission(
                    admitted=admitted,
                    exclusion_reason=exclusion_reason,
                    label=label or {},
                    target_services=target_services,
                ),
                "window_start": window_start,
                "window_end": window_end,
                "normal_intervals": normal_intervals,
                "instance_type": _text((label or {}).get("instance_type")),
                "fault_type": _text((label or {}).get("fault_type")),
                "target_services": target_services,
                "target_nodes": (
                    _values((label or {}).get("instance"))
                    if str((label or {}).get("instance_type", "")).lower()
                    == "node"
                    else []
                ),
                "groundtruth_present": label is not None,
                "source_input_path": str(root / "input.json"),
                "source_groundtruth_path": str(root / "groundtruth.jsonl"),
            }
        )
    return rows


def _partition_dates(case: Mapping[str, Any]) -> List[str]:
    intervals = list(case["normal_intervals"]) + [
        [float(case["window_start"]), float(case["window_end"])]
    ]
    earliest = min(float(interval[0]) for interval in intervals)
    latest = max(float(interval[1]) for interval in intervals)
    cursor = datetime.fromtimestamp(earliest, tz=_LOCAL_TIMEZONE).date()
    final = datetime.fromtimestamp(
        max(earliest, latest - 1e-6), tz=_LOCAL_TIMEZONE
    ).date()
    dates: List[str] = []
    while cursor <= final:
        dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return dates


def _window_name(timestamp: float, case: Mapping[str, Any]) -> str:
    start = float(case["window_start"])
    end = float(case["window_end"])
    if start <= timestamp < end:
        return "abnormal"
    for normal_start, normal_end in case["normal_intervals"]:
        if float(normal_start) <= timestamp < float(normal_end):
            return "normal"
    return ""


def _path_identity(path: Path, partition: str) -> str:
    stem = _DATE_SUFFIX.sub("", path.stem)
    if partition == "apm/service":
        return stem[len("service_") :] if stem.startswith("service_") else ""
    if partition == "apm/pod":
        return stem[len("pod_") :] if stem.startswith("pod_") else ""
    if partition == "infra/infra_tidb":
        return "tidb-tidb"
    if partition == "other":
        return "tidb-pd" if "_pd_" in ("_%s_" % stem) else "tidb-tikv"
    return ""


def _ownership(
    row: Mapping[str, Any],
    path: Path,
    partition: str,
) -> Tuple[str, str, str]:
    path_identity = _path_identity(path, partition)
    instance = _text(row.get("instance"))
    if partition == "apm/service":
        service = normalize_service_name(
            _text(row.get("object_id")) or path_identity
        )
        return service, "", instance
    if partition == "apm/pod":
        pod = (
            _text(row.get("object_id"))
            or _text(row.get("pod"))
            or path_identity
        )
        cleaned_pod = re.sub(r"\s*\(deleted\)\s*$", "", pod).strip()
        return normalize_service_name(cleaned_pod), cleaned_pod, instance
    if partition == "infra/infra_pod":
        pod = _text(row.get("pod"))
        return normalize_service_name(pod), pod, instance
    if partition == "infra/infra_node":
        host = _text(row.get("kubernetes_node")) or instance
        return ("host::%s" % host if host else ""), "", host
    return normalize_service_name(path_identity), "", instance


def _labels(
    row: Mapping[str, Any],
    value_columns: Set[str],
    partition: str,
) -> Dict[str, str]:
    output = {"source_partition": partition}
    for column in sorted(row):
        if column == "time" or column in value_columns:
            continue
        value = _text(row.get(column))
        if value:
            output[column] = value
    return output


def _metric_id(
    *,
    metric_name: str,
    service: str,
    pod: str,
    instance: str,
    partition: str,
    labels: Mapping[str, str],
) -> str:
    fields = (
        ("modality", "metrics"),
        ("source_partition", partition),
        ("value_column", metric_name),
        ("metric_name", metric_name),
        ("service", service),
        ("pod", pod),
        ("instance", instance),
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


def _deduplicate(
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
    values_by_observation: Dict[Tuple[str, str, float], Set[float]] = {}
    for row in deduplicated:
        key = (
            str(row["case_id"]),
            str(row["metric_id"]),
            float(row["timestamp"]),
        )
        values_by_observation.setdefault(key, set()).add(float(row["value"]))
    conflicts = {
        key: values
        for key, values in values_by_observation.items()
        if len(values) > 1
    }
    conflicting_series = {
        (case_id, metric_id)
        for case_id, metric_id, _timestamp in conflicts
    }
    retained = [
        row
        for row in deduplicated
        if (str(row["case_id"]), str(row["metric_id"]))
        not in conflicting_series
    ]
    return retained, {
        "exact_duplicates_removed": len(rows) - len(deduplicated),
        "conflicting_series_count": len(conflicting_series),
        "conflicting_observation_count": len(conflicts),
        "conflict_records": [
            {
                "case_id": key[0],
                "metric_id": key[1],
                "timestamp": key[2],
                "values": sorted(values),
            }
            for key, values in sorted(conflicts.items())
        ],
    }


def canonicalize_case(
    dataset_root: Path,
    case: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map native date-partitioned metrics for one admitted case."""

    if str(case.get("canonical_dataset_id")) != CANONICAL_DATASET_ID:
        raise ValueError("case canonical_dataset_id must be aiops25")
    if not bool(case.get("admitted")):
        raise ValueError(
            "case %s is not admitted: %s"
            % (case.get("case_id"), case.get("exclusion_reason"))
        )
    root = Path(dataset_root)
    dates = _partition_dates(case)
    output: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {
        "partition_dates": dates,
        "input_file_count": 0,
        "input_row_count": 0,
        "in_window_source_row_count": 0,
        "non_finite_row_count": 0,
        "unresolved_service_row_count": 0,
        "source_files": [],
    }
    for date in dates:
        metric_root = root / date / "metric-parquet"
        for partition in _PARTITIONS:
            for path in sorted((metric_root / partition).glob("*.parquet")):
                frame = pd.read_parquet(path)
                audit["input_file_count"] += 1
                audit["source_files"].append(str(path))
                audit["input_row_count"] += len(frame)
                if frame.empty or "time" not in frame.columns:
                    continue
                value_columns = {
                    column
                    for column in frame.columns
                    if column not in _METADATA_COLUMNS
                    and pd.api.types.is_numeric_dtype(frame[column].dtype)
                }
                if not value_columns:
                    continue
                timestamps = pd.to_datetime(
                    frame["time"], utc=True, errors="coerce"
                )
                for position, (_index, raw_series) in enumerate(
                    frame.iterrows()
                ):
                    raw = raw_series.to_dict()
                    parsed_time = timestamps.iloc[position]
                    timestamp = (
                        float(parsed_time.timestamp())
                        if not pd.isna(parsed_time)
                        else float("nan")
                    )
                    window = (
                        _window_name(timestamp, case)
                        if math.isfinite(timestamp)
                        else ""
                    )
                    if not window:
                        continue
                    audit["in_window_source_row_count"] += 1
                    service, pod, instance = _ownership(raw, path, partition)
                    entity_type = (
                        "host" if service.startswith("host::") else "service"
                    )
                    labels = _labels(raw, value_columns, partition)
                    for value_column in sorted(value_columns):
                        value = pd.to_numeric(
                            pd.Series([raw.get(value_column)]), errors="coerce"
                        ).iloc[0]
                        try:
                            finite = math.isfinite(timestamp) and math.isfinite(
                                float(value)
                            )
                        except (TypeError, ValueError):
                            finite = False
                        if not finite:
                            audit["non_finite_row_count"] += 1
                            continue
                        if not service:
                            audit["unresolved_service_row_count"] += 1
                            continue
                        output.append(
                            {
                                "schema_version": SERIES_SCHEMA_VERSION,
                                "canonical_dataset_id": CANONICAL_DATASET_ID,
                                "case_id": str(case["case_id"]),
                                "metric_id": _metric_id(
                                    metric_name=value_column,
                                    service=service,
                                    pod=pod,
                                    instance=instance,
                                    partition=partition,
                                    labels=labels,
                                ),
                                "service": service,
                                "entity_type": entity_type,
                                "pod": pod,
                                "instance": instance,
                                "metric_name": value_column,
                                "metric_semantic": value_column,
                                "modality": "metrics",
                                "source_partition": partition,
                                "window": window,
                                "timestamp": timestamp,
                                "value": float(value),
                            }
                        )

    output, duplicate_audit = _deduplicate(output)
    output.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["metric_id"]),
            float(row["timestamp"]),
        )
    )
    if output:
        summary = validate_canonical_series_rows(
            output, CANONICAL_DATASET_ID
        )
        empty_metric_reason = ""
    else:
        summary = {
            "row_count": 0,
            "series_count": 0,
            "case_count": 0,
        }
        if not audit["input_file_count"]:
            empty_metric_reason = "no_metric_files_for_declared_partitions"
        elif not audit["in_window_source_row_count"]:
            empty_metric_reason = (
                "no_metric_samples_in_declared_windows"
            )
        else:
            empty_metric_reason = "no_valid_canonical_metric_rows"
    audit.update(duplicate_audit)
    audit["output_row_count"] = summary["row_count"]
    audit["output_series_count"] = summary["series_count"]
    audit["empty_metric_case"] = not bool(output)
    audit["empty_metric_reason"] = empty_metric_reason
    audit["rows_by_window"] = {
        "normal": sum(row["window"] == "normal" for row in output),
        "abnormal": sum(row["window"] == "abnormal" for row in output),
    }
    audit["invariants"] = {
        "unique_case_metric_timestamp": True,
        "one_service_per_series": True,
        "stable_order": True,
    }
    return {"rows": output, "audit": audit}


__all__ = [
    "CANONICAL_DATASET_ID",
    "LEGACY_ALIAS",
    "build_case_manifest",
    "canonicalize_case",
    "normalize_service_name",
]
