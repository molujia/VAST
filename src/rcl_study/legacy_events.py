"""Method-native metric event generation over canonical series."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd

from .artifacts import (
    AUDIT_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION,
    COMPLETION_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    SERIES_SCHEMA_VERSION,
    sha256_file,
    validate_audit,
    validate_anomaly_event_file,
    validate_anomaly_event_rows,
    validate_canonical_series_rows,
    validate_case_manifest_rows,
    validate_completed_bundle,
)
from .datasets import resolve_dataset


HALF_SUPERVISE_METHOD_ID = "half_supervise"
SELF_SUPERVISE_METHOD_ID = "self_supervise"
LEGACY_GENERATOR_ID = "legacy_canonical"
BUCKET_SECONDS = 60


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    scale = maximum - minimum
    return [float(value - minimum) / float(scale) for value in values]


def _erode_binary_sequence(values: Sequence[int], max_gap: int) -> List[int]:
    result = list(values)
    ones = [index for index, value in enumerate(result) if value == 1]
    for index in range(len(ones) - 1):
        start = ones[index]
        end = ones[index + 1]
        if end - start - 1 <= max_gap:
            for gap_index in range(start + 1, end):
                result[gap_index] = 1
    return result


def _blur_binary_sequence(values: Sequence[int], max_gap: int) -> List[int]:
    result = list(values)
    zeros = [index for index, value in enumerate(result) if value == 0]
    for index in range(len(zeros) - 1):
        start = zeros[index]
        end = zeros[index + 1]
        if end - start - 1 <= max_gap:
            for gap_index in range(start + 1, end):
                result[gap_index] = 0
    return result


def _periodic_pattern_signature(
    values: Sequence[int], threshold: int = 6
) -> Tuple[bool, int, int, int, int]:
    if not values:
        return False, 0, -1, -1, -1
    count_zeros = 0
    while count_zeros < len(values) and values[count_zeros] == 0:
        count_zeros += 1
    if count_zeros >= len(values):
        return False, count_zeros, -1, -1, -1

    periods_of_ones: List[int] = []
    periods_of_zeros: List[int] = []
    current = values[count_zeros]
    count = 1
    for index in range(count_zeros + 1, len(values)):
        if values[index] == current:
            count += 1
        else:
            if current == 1:
                periods_of_ones.append(count)
            else:
                periods_of_zeros.append(count)
            current = values[index]
            count = 1
    if current == 1:
        periods_of_ones.append(count)
    else:
        periods_of_zeros.append(count)
    if len(periods_of_ones) < threshold or len(periods_of_zeros) < threshold:
        return False, count_zeros, -1, -1, -1

    one_counts = Counter(periods_of_ones)
    zero_counts = Counter(periods_of_zeros)
    one_period_length, one_period_times = one_counts.most_common(1)[0]
    zero_period_length, zero_period_times = zero_counts.most_common(1)[0]
    if (
        one_period_times <= threshold
        or zero_period_times <= threshold
        or abs(one_period_times - zero_period_times) > 2
    ):
        return False, count_zeros, -1, -1, -1

    offset = count_zeros
    pair_count = min(len(periods_of_ones), len(periods_of_zeros))
    for index in range(pair_count):
        if (
            periods_of_ones[index] == one_period_length
            and periods_of_zeros[index] == zero_period_length
        ):
            break
        offset += periods_of_ones[index]
        offset += periods_of_zeros[index]
    return (
        True,
        offset,
        one_period_length,
        zero_period_length,
        max(one_period_times, zero_period_times),
    )


def _synthetic_periodic_mask(
    leading_zeros: int,
    one_period_length: int,
    zero_period_length: int,
    period_times: int,
    full_length: int,
) -> List[int]:
    if full_length <= 0:
        return []
    active_length = int(one_period_length) + 2
    inactive_length = max(0, int(zero_period_length) - 2)
    result = [0] * max(0, int(leading_zeros))
    period_width = max(1, active_length + inactive_length)
    fitted_period_times = max(1, int(full_length / period_width))
    if abs(int(period_times) - fitted_period_times) <= 1:
        period_times = fitted_period_times
    for _ in range(max(0, int(period_times))):
        result.extend([1] * active_length)
        result.extend([0] * inactive_length)
    if len(result) < full_length:
        result.extend([0] * (full_length - len(result)))
    return result[:full_length]


def _sliding_window_sigma_scores(
    values: Sequence[float],
    window_size: int,
    sigma: float = 3.0,
) -> List[float]:
    sample_count = len(values)
    if sample_count == 0:
        return []
    if sample_count <= max(2, window_size + 1):
        return [0.0] * sample_count
    labels = [0.0] * sample_count
    for start_index in range(0, sample_count - window_size - 1):
        window = values[start_index : start_index + window_size]
        local_mean = _mean(window)
        local_std = float(np.std(window))
        if local_std <= 1e-12:
            continue
        lower_limit = local_mean - sigma * local_std
        upper_limit = local_mean + sigma * local_std
        for offset, value in enumerate(window):
            if value < lower_limit or value > upper_limit:
                labels[start_index + offset] += local_std
    return labels


def half_supervise_event_scores(
    values: Sequence[float],
    window_size: int = 120,
    sigma: float = 3.0,
) -> List[float]:
    """Exact shared copy of the original semi-supervised event score core."""

    if not values:
        return []
    raw_scores = _sliding_window_sigma_scores(
        values, window_size=window_size, sigma=sigma
    )
    binary_scores = [1 if score > 0.0 else 0 for score in raw_scores]
    if sum(binary_scores) == 0:
        return [0.0] * len(values)
    eroded = _erode_binary_sequence(binary_scores, 5)
    (
        has_period,
        leading_zeros,
        one_length,
        zero_length,
        period_times,
    ) = _periodic_pattern_signature(eroded)
    if has_period:
        periodic_mask = _synthetic_periodic_mask(
            leading_zeros=leading_zeros,
            one_period_length=one_length,
            zero_period_length=zero_length,
            period_times=period_times,
            full_length=len(values),
        )
        eroded = [
            max(0, eroded[index] - periodic_mask[index])
            for index in range(len(eroded))
        ]
    filtered = _blur_binary_sequence(eroded, 1)
    kept_scores = [
        raw_scores[index] if filtered[index] == 1 else 0.0
        for index in range(len(raw_scores))
    ]
    return _minmax_normalize(kept_scores)


def _finite(values: Iterable[Any]) -> np.ndarray:
    numeric = pd.to_numeric(
        pd.Series(list(values), dtype="object"), errors="coerce"
    )
    data = numeric.to_numpy(dtype=float, na_value=np.nan)
    return data[np.isfinite(data)]


def self_supervise_window_effect_signature(
    normal_values: Iterable[Any],
    abnormal_values: Iterable[Any],
) -> Dict[str, float]:
    """Exact shared copy of the original self-supervised effect signature."""

    normal = _finite(normal_values)
    abnormal = _finite(abnormal_values)
    normal_mean = float(np.mean(normal)) if normal.size else 0.0
    abnormal_mean = float(np.mean(abnormal)) if abnormal.size else 0.0
    normal_std = float(np.std(normal)) if normal.size else 0.0
    abnormal_std = float(np.std(abnormal)) if abnormal.size else 0.0
    normal_median = float(np.median(normal)) if normal.size else 0.0
    abnormal_median = float(np.median(abnormal)) if abnormal.size else 0.0
    mad = (
        float(np.median(np.abs(normal - normal_median)))
        if normal.size
        else 0.0
    )
    baseline_floor = max(abs(normal_median) * 0.01, 1e-6)
    scale = max(1.4826 * mad, normal_std, baseline_floor)
    location_effect = abs(abnormal_median - normal_median) / scale
    mean_effect = abs(abnormal_mean - normal_mean) / scale
    peak_effect = (
        float(np.max(np.abs(abnormal - normal_median)) / scale)
        if abnormal.size
        else 0.0
    )
    effect = max(location_effect, mean_effect, peak_effect)
    return {
        "normal_count": float(normal.size),
        "abnormal_count": float(abnormal.size),
        "normal_mean": normal_mean,
        "abnormal_mean": abnormal_mean,
        "normal_std": normal_std,
        "abnormal_std": abnormal_std,
        "normal_median": normal_median,
        "abnormal_median": abnormal_median,
        "scale": scale,
        "location_effect": float(location_effect),
        "mean_effect": float(mean_effect),
        "peak_effect": float(peak_effect),
        "effect": float(effect),
    }


def _event_id(*parts: Any) -> str:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "legacy-%s" % hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_metric_id(
    rows: Sequence[Mapping[str, Any]], service: str, metric_name: str
) -> str:
    metric_ids = sorted({str(row["metric_id"]) for row in rows})
    if len(metric_ids) == 1:
        return metric_ids[0]
    return "legacy-group|service=%s|metric_name=%s|sources=%s" % (
        json.dumps(service, ensure_ascii=False),
        json.dumps(metric_name, ensure_ascii=False),
        hashlib.sha256(
            json.dumps(metric_ids, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    )


def _half_supervise_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_series_sha256: str,
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["service"]),
            str(row["metric_name"]),
        )
        groups.setdefault(key, []).append(row)
    events: List[Dict[str, Any]] = []
    for (case_id, service, metric_name), group_rows in sorted(groups.items()):
        bucket_values: Dict[int, float] = {}
        windows_by_bucket: Dict[int, Set[str]] = {}
        for row in group_rows:
            bucket = (
                int(float(row["timestamp"]) // BUCKET_SECONDS) * BUCKET_SECONDS
            )
            bucket_values[bucket] = bucket_values.get(bucket, 0.0) + float(
                row["value"]
            )
            windows_by_bucket.setdefault(bucket, set()).add(str(row["window"]))
        first_bucket = min(bucket_values)
        last_bucket = max(bucket_values)
        bucket_count = int((last_bucket - first_bucket) / BUCKET_SECONDS) + 1
        dense_values = [0.0] * bucket_count
        for timestamp, value in bucket_values.items():
            dense_values[
                int((timestamp - first_bucket) / BUCKET_SECONDS)
            ] = float(value)
        normalized_values = _minmax_normalize(dense_values)
        window_size = min(120, max(5, len(normalized_values) - 2))
        anomaly_scores = half_supervise_event_scores(
            normalized_values,
            window_size=window_size,
            sigma=3.0,
        )
        grouped_metric_id = _group_metric_id(
            group_rows, service, metric_name
        )
        for bucket_index, score in enumerate(anomaly_scores):
            if score <= 0.0:
                continue
            timestamp = first_bucket + bucket_index * BUCKET_SECONDS
            windows = windows_by_bucket.get(timestamp, set())
            if not windows:
                continue
            window = "abnormal" if "abnormal" in windows else sorted(windows)[0]
            events.append(
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "canonical_dataset_id": str(
                        group_rows[0]["canonical_dataset_id"]
                    ),
                    "case_id": case_id,
                    "event_id": _event_id(
                        HALF_SUPERVISE_METHOD_ID,
                        case_id,
                        service,
                        metric_name,
                        timestamp,
                    ),
                    "metric_id": grouped_metric_id,
                    "metric_name": metric_name,
                    "service": service,
                    "window": window,
                    "start_ts": float(timestamp),
                    "end_ts": float(timestamp),
                    "peak_ts": float(timestamp),
                    "score": float(score),
                    "detector_id": "%s:%s"
                    % (LEGACY_GENERATOR_ID, HALF_SUPERVISE_METHOD_ID),
                    "source_series_sha256": source_series_sha256,
                    "legacy_payload": {
                        "bucket_seconds": BUCKET_SECONDS,
                        "sigma_window_size": window_size,
                        "sigma": 3.0,
                    },
                }
            )
    events.sort(
        key=lambda event: (
            event["case_id"],
            event["service"],
            event["start_ts"],
            event["metric_id"],
        )
    )
    return {"events": events, "service_scores": {}}


def _self_supervise_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_series_sha256: str,
) -> Dict[str, Any]:
    services: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        services.setdefault(
            (str(row["case_id"]), str(row["service"])), []
        ).append(row)
    events: List[Dict[str, Any]] = []
    service_scores: Dict[str, Dict[str, float]] = {}
    for (case_id, service), service_rows in sorted(services.items()):
        metric_groups: Dict[str, List[Mapping[str, Any]]] = {}
        for row in service_rows:
            metric_groups.setdefault(str(row["metric_id"]), []).append(row)
        signatures: List[Tuple[str, Dict[str, float]]] = []
        for metric_id, metric_rows in sorted(metric_groups.items()):
            normal_values = [
                float(row["value"])
                for row in metric_rows
                if str(row["window"]) == "normal"
            ]
            abnormal_values = [
                float(row["value"])
                for row in metric_rows
                if str(row["window"]) == "abnormal"
            ]
            signature = self_supervise_window_effect_signature(
                normal_values, abnormal_values
            )
            if signature["normal_count"] and signature["abnormal_count"]:
                signatures.append((metric_id, signature))
        if not signatures:
            continue
        signatures.sort(key=lambda item: (-item[1]["effect"], item[0]))
        top = signatures[: min(5, len(signatures))]
        effects = [item[1]["effect"] for item in top]
        score = float(
            math.log1p(
                max(
                    0.0,
                    0.6 * max(effects) + 0.4 * float(np.mean(effects)),
                )
            )
        )
        service_scores.setdefault(case_id, {})[service] = score
        top_metric_id, top_signature = top[0]
        top_abnormal_rows = [
            row
            for row in metric_groups[top_metric_id]
            if str(row["window"]) == "abnormal"
        ]
        abnormal_service_rows = [
            row for row in service_rows if str(row["window"]) == "abnormal"
        ]
        if not top_abnormal_rows or not abnormal_service_rows:
            continue
        normal_median = float(top_signature["normal_median"])
        peak_row = max(
            top_abnormal_rows,
            key=lambda row: (
                abs(float(row["value"]) - normal_median),
                -float(row["timestamp"]),
            ),
        )
        start_ts = min(float(row["timestamp"]) for row in abnormal_service_rows)
        end_ts = max(float(row["timestamp"]) for row in abnormal_service_rows)
        metric_names = {
            metric_id: str(metric_groups[metric_id][0]["metric_name"])
            for metric_id, _signature in top
        }
        events.append(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "canonical_dataset_id": str(
                    service_rows[0]["canonical_dataset_id"]
                ),
                "case_id": case_id,
                "event_id": _event_id(
                    SELF_SUPERVISE_METHOD_ID,
                    case_id,
                    service,
                    top_metric_id,
                ),
                "metric_id": top_metric_id,
                "metric_name": metric_names[top_metric_id],
                "service": service,
                "window": "abnormal",
                "start_ts": start_ts,
                "end_ts": end_ts,
                "peak_ts": float(peak_row["timestamp"]),
                "score": score,
                "detector_id": "%s:%s"
                % (LEGACY_GENERATOR_ID, SELF_SUPERVISE_METHOD_ID),
                "source_series_sha256": source_series_sha256,
                "legacy_payload": {
                    "aggregation_method": (
                        "normal_abnormal_robust_effect_top5"
                    ),
                    "top_metric_ids": [metric_id for metric_id, _ in top],
                    "top_metric_names": metric_names,
                    "top_metric_signatures": {
                        metric_id: signature for metric_id, signature in top
                    },
                    "service_anomaly_score": score,
                },
            }
        )
    events.sort(
        key=lambda event: (
            event["case_id"],
            event["service"],
            event["start_ts"],
            event["metric_id"],
        )
    )
    return {"events": events, "service_scores": service_scores}


def build_legacy_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    source_series_sha256: str,
) -> Dict[str, Any]:
    """Generate one method's unchanged legacy signal from canonical rows."""

    if method_id not in {
        HALF_SUPERVISE_METHOD_ID,
        SELF_SUPERVISE_METHOD_ID,
    }:
        raise ValueError(
            "method_id must be half_supervise or self_supervise"
        )
    if not rows:
        raise ValueError("canonical series rows must not be empty")
    dataset_id = str(rows[0].get("canonical_dataset_id", ""))
    series_summary = validate_canonical_series_rows(rows, dataset_id)
    if method_id == HALF_SUPERVISE_METHOD_ID:
        result = _half_supervise_events(
            rows, source_series_sha256=source_series_sha256
        )
        algorithm_id = "half_supervise:eventized_sliding_sigma_v1"
    else:
        result = _self_supervise_events(
            rows, source_series_sha256=source_series_sha256
        )
        algorithm_id = (
            "self_supervise:normal_abnormal_robust_effect_top5_v1"
        )
    event_summary = validate_anomaly_event_rows(
        result["events"], dataset_id, source_series_sha256
    )
    result["audit"] = {
        "event_generator_id": LEGACY_GENERATOR_ID,
        "method_id": method_id,
        "algorithm_id": algorithm_id,
        "canonical_dataset_id": dataset_id,
        "source_series_sha256": source_series_sha256,
        "input_row_count": series_summary["row_count"],
        "input_series_count": series_summary["series_count"],
        "input_case_count": series_summary["case_count"],
        "event_count": event_summary["event_count"],
        "event_case_count": event_summary["case_count"],
    }
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(destination)


def _artifact_entry(
    root: Path,
    path: Path,
    *,
    schema_version: str,
    row_count: int,
    sha256: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": str(sha256 or sha256_file(path)),
        "schema_version": schema_version,
        "row_count": int(row_count),
    }


def write_legacy_event_artifact(
    rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    output_root: Path,
    artifact_id: str,
    code_revision: str,
    code_sha256: str,
    environment_manifest_path: str,
    environment_manifest_sha256: str,
) -> Dict[str, Any]:
    """Write one validated ``legacy_canonical`` event artifact bundle."""

    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output_root must be new or empty: %s" % root)
    if not str(artifact_id).strip():
        raise ValueError("artifact_id must be non-empty")
    if not str(code_revision).strip():
        raise ValueError("code_revision must be non-empty")
    for value, context in (
        (code_sha256, "code_sha256"),
        (environment_manifest_sha256, "environment_manifest_sha256"),
    ):
        if (
            len(str(value)) != 64
            or any(character not in "0123456789abcdef" for character in str(value))
        ):
            raise ValueError("%s must be a lowercase SHA-256" % context)
    if not rows:
        raise ValueError("canonical series rows must not be empty")
    dataset_id = str(rows[0].get("canonical_dataset_id", ""))
    resolution = resolve_dataset(dataset_id)
    validate_canonical_series_rows(rows, dataset_id)
    case_summary = validate_case_manifest_rows(case_rows, dataset_id)
    root.mkdir(parents=True, exist_ok=True)
    try:
        ordered_rows = sorted(
            [dict(row) for row in rows],
            key=lambda row: (
                str(row["case_id"]),
                str(row["metric_id"]),
                float(row["timestamp"]),
            ),
        )
        ordered_cases = sorted(
            [dict(row) for row in case_rows],
            key=lambda row: str(row["case_id"]),
        )
        cases_path = root / "cases.jsonl"
        series_path = root / "series.jsonl"
        _write_jsonl(cases_path, ordered_cases)
        _write_jsonl(series_path, ordered_rows)
        source_series_sha256 = sha256_file(series_path)
        result = build_legacy_events(
            ordered_rows,
            method_id=method_id,
            source_series_sha256=source_series_sha256,
        )
        events_path = (
            root / "events" / LEGACY_GENERATOR_ID / "events.jsonl"
        )
        _write_jsonl(events_path, result["events"])
        config_identity = {
            "schema_version": "legacy-canonical-config-v1",
            "event_generator_id": LEGACY_GENERATOR_ID,
            "method_id": method_id,
            "algorithm_id": result["audit"]["algorithm_id"],
        }
        config_sha256 = hashlib.sha256(
            json.dumps(
                config_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        admitted = sum(bool(row["admitted"]) for row in ordered_cases)
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "canonical_dataset_id": dataset_id,
            "input_file_count": 2,
            "input_case_count": len(ordered_cases),
            "admitted_case_count": admitted,
            "excluded_case_count": len(ordered_cases) - admitted,
            "exact_duplicates_removed": 0,
            "conflicting_series_count": 0,
            "non_finite_row_count": 0,
            "unresolved_service_row_count": 0,
            "output_series_count": len(
                {
                    (str(row["case_id"]), str(row["metric_id"]))
                    for row in ordered_rows
                }
            ),
            "output_row_count": len(ordered_rows),
            "source_checksums": {
                "canonical_cases": sha256_file(cases_path),
                "canonical_series": source_series_sha256,
            },
            "code_revision": str(code_revision),
            "code_sha256": str(code_sha256),
            "config_sha256": config_sha256,
            "environment_manifest_path": str(environment_manifest_path),
            "environment_manifest_sha256": str(
                environment_manifest_sha256
            ),
            "event_generator_id": LEGACY_GENERATOR_ID,
            "method_id": method_id,
            "algorithm_id": result["audit"]["algorithm_id"],
            "event_count": len(result["events"]),
            "root_cause_label_access": "forbidden",
            "passed": True,
        }
        audit_path = root / "audit.json"
        _write_json(audit_path, audit)
        validate_audit(audit, dataset_id)
        artifacts = {
            "cases": _artifact_entry(
                root,
                cases_path,
                schema_version=CASE_SCHEMA_VERSION,
                row_count=len(ordered_cases),
            ),
            "series": _artifact_entry(
                root,
                series_path,
                schema_version=SERIES_SCHEMA_VERSION,
                row_count=len(ordered_rows),
            ),
            "events": _artifact_entry(
                root,
                events_path,
                schema_version=EVENT_SCHEMA_VERSION,
                row_count=len(result["events"]),
            ),
            "audit": _artifact_entry(
                root,
                audit_path,
                schema_version=AUDIT_SCHEMA_VERSION,
                row_count=1,
            ),
        }
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "artifact_id": str(artifact_id),
            "canonical_dataset_id": dataset_id,
            "legacy_alias": resolution.legacy_alias,
            "producer_environment": "rcalab",
            "case_ids": sorted(case_summary["case_ids"]),
            "artifacts": artifacts,
            "event_generator": {
                "id": LEGACY_GENERATOR_ID,
                "method_id": method_id,
                "algorithm_id": result["audit"]["algorithm_id"],
                "config_sha256": config_sha256,
                "source_series_sha256": source_series_sha256,
                "code_revision": str(code_revision),
                "code_sha256": str(code_sha256),
                "root_cause_label_access": "forbidden",
            },
            "provenance": {
                "case_manifest_sha256": artifacts["cases"]["sha256"],
                "environment_manifest_path": str(environment_manifest_path),
                "environment_manifest_sha256": str(
                    environment_manifest_sha256
                ),
            },
        }
        manifest_path = root / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha256 = sha256_file(manifest_path)
        _write_json(
            root / "COMPLETED.json",
            {
                "schema_version": COMPLETION_SCHEMA_VERSION,
                "artifact_id": str(artifact_id),
                "status": "complete",
                "artifact_manifest_sha256": manifest_sha256,
                "event_count": len(result["events"]),
            },
        )
        (root / "all.done").write_text(
            manifest_sha256 + "\n", encoding="utf-8"
        )
        validated = validate_completed_bundle(
            root,
            expected_dataset_id=dataset_id,
            expected_event_generator_id=LEGACY_GENERATOR_ID,
            expected_config_sha256=config_sha256,
            expected_case_ids=set(case_summary["case_ids"]),
        )
        return {
            "manifest": validated,
            "config_sha256": config_sha256,
            "source_series_sha256": source_series_sha256,
            "event_count": len(result["events"]),
        }
    except Exception as exc:
        for marker_name in ("COMPLETED.json", "all.done"):
            marker = root / marker_name
            if marker.exists():
                marker.unlink()
        _write_json(
            root / "run.failed",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def write_legacy_event_artifact_from_file(
    series_source_path: Path,
    case_rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    output_root: Path,
    artifact_id: str,
    code_revision: str,
    code_sha256: str,
    environment_manifest_path: str,
    environment_manifest_sha256: str,
    expected_source_series_sha256: str,
    progress_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    """Build events from one case at a time and hard-link canonical rows."""

    if method_id not in {
        HALF_SUPERVISE_METHOD_ID,
        SELF_SUPERVISE_METHOD_ID,
    }:
        raise ValueError(
            "method_id must be half_supervise or self_supervise"
        )
    source = Path(series_source_path)
    if not source.is_file():
        raise ValueError("canonical series file is missing: %s" % source)
    source_sha256 = str(expected_source_series_sha256)
    if (
        len(source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_sha256
        )
    ):
        raise ValueError(
            "expected_source_series_sha256 must be a lowercase SHA-256"
        )
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output_root must be new or empty: %s" % root)
    if not str(artifact_id).strip():
        raise ValueError("artifact_id must be non-empty")
    if not str(code_revision).strip():
        raise ValueError("code_revision must be non-empty")
    for value, context in (
        (code_sha256, "code_sha256"),
        (
            environment_manifest_sha256,
            "environment_manifest_sha256",
        ),
    ):
        if (
            len(str(value)) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(value)
            )
        ):
            raise ValueError(
                "%s must be a lowercase SHA-256" % context
            )

    first_row: Optional[Dict[str, Any]] = None
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "canonical series line is not an object"
                    )
                first_row = dict(payload)
                break
    if first_row is None:
        raise ValueError("canonical series rows must not be empty")
    dataset_id = str(
        first_row.get("canonical_dataset_id", "")
    )
    resolution = resolve_dataset(dataset_id)
    case_summary = validate_case_manifest_rows(
        case_rows, dataset_id
    )
    ordered_cases = sorted(
        [dict(row) for row in case_rows],
        key=lambda row: str(row["case_id"]),
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        cases_path = root / "cases.jsonl"
        series_path = root / "series.jsonl"
        _write_jsonl(cases_path, ordered_cases)
        try:
            os.link(str(source.resolve()), str(series_path))
        except OSError:
            shutil.copyfile(str(source), str(series_path))

        events_path = (
            root / "events" / LEGACY_GENERATOR_ID / "events.jsonl"
        )
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_temporary = events_path.with_name(
            events_path.name + ".tmp"
        )
        source_digest = hashlib.sha256()
        input_row_count = 0
        input_series_count = 0
        observed_cases: Set[str] = set()
        event_count = 0
        previous_case_id = ""
        algorithm_id = (
            "half_supervise:eventized_sliding_sigma_v1"
            if method_id == HALF_SUPERVISE_METHOD_ID
            else (
                "self_supervise:"
                "normal_abnormal_robust_effect_top5_v1"
            )
        )

        def iter_rows():
            with series_path.open("rb") as handle:
                for line_number, raw_line in enumerate(
                    handle, start=1
                ):
                    source_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line.decode("utf-8"))
                    if not isinstance(payload, Mapping):
                        raise ValueError(
                            "canonical series line %d is not an object"
                            % line_number
                        )
                    yield dict(payload)

        with events_temporary.open("w", encoding="utf-8") as event_file:
            for completed, (case_id, grouped) in enumerate(
                groupby(
                    iter_rows(),
                    key=lambda row: str(row["case_id"]),
                ),
                start=1,
            ):
                if previous_case_id and case_id <= previous_case_id:
                    raise ValueError(
                        "canonical series case blocks are not sorted"
                    )
                previous_case_id = case_id
                rows = list(grouped)
                result = build_legacy_events(
                    rows,
                    method_id=method_id,
                    source_series_sha256=source_sha256,
                )
                input_row_count += int(
                    result["audit"]["input_row_count"]
                )
                input_series_count += int(
                    result["audit"]["input_series_count"]
                )
                observed_cases.add(case_id)
                for event in result["events"]:
                    event_file.write(
                        json.dumps(
                            event,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    event_count += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "generate_events",
                            "completed": completed,
                            "total": len(ordered_cases),
                            "current_item": case_id,
                        }
                    )
        events_temporary.replace(events_path)
        if source_digest.hexdigest() != source_sha256:
            raise ValueError("canonical series source hash mismatch")

        event_summary = validate_anomaly_event_file(
            events_path,
            dataset_id,
            source_sha256,
        )
        if int(event_summary["event_count"]) != event_count:
            raise ValueError("streamed event count mismatch")
        config_identity = {
            "schema_version": "legacy-canonical-config-v1",
            "event_generator_id": LEGACY_GENERATOR_ID,
            "method_id": method_id,
            "algorithm_id": algorithm_id,
        }
        config_sha256 = hashlib.sha256(
            json.dumps(
                config_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        admitted = sum(
            bool(row["admitted"]) for row in ordered_cases
        )
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "canonical_dataset_id": dataset_id,
            "input_file_count": 2,
            "input_case_count": len(ordered_cases),
            "admitted_case_count": admitted,
            "excluded_case_count": len(ordered_cases) - admitted,
            "exact_duplicates_removed": 0,
            "conflicting_series_count": 0,
            "non_finite_row_count": 0,
            "unresolved_service_row_count": 0,
            "output_series_count": input_series_count,
            "output_row_count": input_row_count,
            "source_checksums": {
                "canonical_cases": sha256_file(cases_path),
                "canonical_series": source_sha256,
            },
            "code_revision": str(code_revision),
            "code_sha256": str(code_sha256),
            "config_sha256": config_sha256,
            "environment_manifest_path": str(
                environment_manifest_path
            ),
            "environment_manifest_sha256": str(
                environment_manifest_sha256
            ),
            "event_generator_id": LEGACY_GENERATOR_ID,
            "method_id": method_id,
            "algorithm_id": algorithm_id,
            "event_count": event_count,
            "event_case_count": int(
                event_summary["case_count"]
            ),
            "source_case_count": len(observed_cases),
            "root_cause_label_access": "forbidden",
            "passed": True,
        }
        audit_path = root / "audit.json"
        _write_json(audit_path, audit)
        validate_audit(audit, dataset_id)
        artifacts = {
            "cases": _artifact_entry(
                root,
                cases_path,
                schema_version=CASE_SCHEMA_VERSION,
                row_count=len(ordered_cases),
            ),
            "series": _artifact_entry(
                root,
                series_path,
                schema_version=SERIES_SCHEMA_VERSION,
                row_count=input_row_count,
                sha256=source_sha256,
            ),
            "events": _artifact_entry(
                root,
                events_path,
                schema_version=EVENT_SCHEMA_VERSION,
                row_count=event_count,
                sha256=str(event_summary["sha256"]),
            ),
            "audit": _artifact_entry(
                root,
                audit_path,
                schema_version=AUDIT_SCHEMA_VERSION,
                row_count=1,
            ),
        }
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "artifact_id": str(artifact_id),
            "canonical_dataset_id": dataset_id,
            "legacy_alias": resolution.legacy_alias,
            "producer_environment": "rcalab",
            "case_ids": sorted(case_summary["case_ids"]),
            "artifacts": artifacts,
            "event_generator": {
                "id": LEGACY_GENERATOR_ID,
                "method_id": method_id,
                "algorithm_id": algorithm_id,
                "config_sha256": config_sha256,
                "source_series_sha256": source_sha256,
                "code_revision": str(code_revision),
                "code_sha256": str(code_sha256),
                "root_cause_label_access": "forbidden",
            },
            "provenance": {
                "case_manifest_sha256": artifacts["cases"][
                    "sha256"
                ],
                "environment_manifest_path": str(
                    environment_manifest_path
                ),
                "environment_manifest_sha256": str(
                    environment_manifest_sha256
                ),
            },
        }
        manifest_path = root / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha256 = sha256_file(manifest_path)
        _write_json(
            root / "COMPLETED.json",
            {
                "schema_version": COMPLETION_SCHEMA_VERSION,
                "artifact_id": str(artifact_id),
                "status": "complete",
                "artifact_manifest_sha256": manifest_sha256,
                "event_count": event_count,
            },
        )
        (root / "all.done").write_text(
            manifest_sha256 + "\n", encoding="utf-8"
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "complete",
                    "completed": len(ordered_cases),
                    "total": len(ordered_cases),
                    "current_item": str(root),
                }
            )
        validated = validate_completed_bundle(
            root,
            expected_dataset_id=dataset_id,
            expected_event_generator_id=LEGACY_GENERATOR_ID,
            expected_config_sha256=config_sha256,
            expected_case_ids=set(case_summary["case_ids"]),
        )
        return {
            "manifest": validated,
            "config_sha256": config_sha256,
            "source_series_sha256": source_sha256,
            "event_count": event_count,
        }
    except Exception as exc:
        for marker_name in ("COMPLETED.json", "all.done"):
            marker = root / marker_name
            if marker.exists():
                marker.unlink()
        _write_json(
            root / "run.failed",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


__all__ = [
    "BUCKET_SECONDS",
    "HALF_SUPERVISE_METHOD_ID",
    "LEGACY_GENERATOR_ID",
    "SELF_SUPERVISE_METHOD_ID",
    "build_legacy_events",
    "half_supervise_event_scores",
    "self_supervise_window_effect_signature",
    "write_legacy_event_artifact",
    "write_legacy_event_artifact_from_file",
]
