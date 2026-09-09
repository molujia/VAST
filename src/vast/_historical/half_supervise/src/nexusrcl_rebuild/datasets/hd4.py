"""HD4 authoritative case reconstruction from RCAbench case directories."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import (
    DatasetManifest,
    DaySpan,
    FaultCase,
    NormalWindow,
    dedupe_targets,
    make_service_target,
    median_int,
)


def _parse_epoch_seconds(value: Any) -> int:
    return int(str(value).strip())


def _parse_iso_timestamp(value: Any) -> int:
    return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_case_dirs(dataset_root: Path) -> Sequence[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "injection.json").exists() and (path / "env.json").exists()
    )


def _is_pod_kill_case(case_dir: Path, injection: Mapping[str, Any]) -> bool:
    haystacks = [
        case_dir.name,
        str(injection.get("injection_name") or ""),
        str(injection.get("description") or ""),
        str(injection.get("display_config") or ""),
    ]
    return any("pod-kill" in value.lower() or "pod kill" in value.lower() for value in haystacks)


def _service_targets(ground_truth: Mapping[str, Any]) -> List[Any]:
    services = ground_truth.get("service") or []
    if isinstance(services, str):
        services = [services]
    return dedupe_targets(make_service_target(str(service_name)) for service_name in services if service_name)


def build_hd4_manifest(
    dataset_root: Path,
    window_size_seconds: int = 300,
    guard_band_seconds: int = 20,
) -> DatasetManifest:
    source_rows = 0
    skipped = Counter()
    pod_kill_durations = Counter()
    pending_rows: List[Dict[str, Any]] = []

    for case_dir in _iter_case_dirs(dataset_root):
        source_rows += 1
        injection = _load_json(case_dir / "injection.json")
        env = _load_json(case_dir / "env.json")
        ground_truth = dict(injection.get("ground_truth") or {})
        positives = _service_targets(ground_truth)
        if not positives:
            skipped["missing_service_ground_truth"] += 1
            continue

        normal_start_ts = _parse_epoch_seconds(env["NORMAL_START"])
        normal_end_ts = _parse_epoch_seconds(env["NORMAL_END"])
        abnormal_start_ts = _parse_epoch_seconds(env["ABNORMAL_START"])
        abnormal_end_ts = _parse_epoch_seconds(env["ABNORMAL_END"])
        injection_start_ts = _parse_iso_timestamp(injection["start_time"])
        injection_end_ts = _parse_iso_timestamp(injection["end_time"])

        duration_seconds = abnormal_end_ts - abnormal_start_ts
        if duration_seconds <= 0:
            skipped["non_positive_fault_duration"] += 1
            continue

        start_drift = abnormal_start_ts - injection_start_ts
        end_drift = abnormal_end_ts - injection_end_ts
        row = {
            "case_dir": case_dir,
            "injection": injection,
            "env": env,
            "positives": positives,
            "normal_start_ts": normal_start_ts,
            "normal_end_ts": normal_end_ts,
            "abnormal_start_ts": abnormal_start_ts,
            "abnormal_end_ts": abnormal_end_ts,
            "duration_seconds": duration_seconds,
            "start_drift_seconds": start_drift,
            "end_drift_seconds": end_drift,
            "is_pod_kill": _is_pod_kill_case(case_dir, injection),
        }
        if row["is_pod_kill"]:
            pod_kill_durations[duration_seconds] += 1
        pending_rows.append(row)

    fixed_pod_kill_duration: Optional[int] = None
    if pod_kill_durations:
        if len(pod_kill_durations) == 1:
            fixed_pod_kill_duration = next(iter(pod_kill_durations))
        else:
            skipped["pod_kill_inconsistent_duration"] += sum(pod_kill_durations.values())

    admitted_cases: List[FaultCase] = []
    normal_windows: List[NormalWindow] = []
    day_spans: List[DaySpan] = []
    start_drift_counter = Counter()
    end_drift_counter = Counter()

    for row in pending_rows:
        case_dir = row["case_dir"]
        injection = row["injection"]
        env = row["env"]
        start_drift_counter[row["start_drift_seconds"]] += 1
        end_drift_counter[row["end_drift_seconds"]] += 1

        if row["is_pod_kill"] and fixed_pod_kill_duration is None:
            skipped["pod_kill_skipped_no_fixed_duration"] += 1
            continue
        if row["is_pod_kill"] and row["duration_seconds"] != fixed_pod_kill_duration:
            skipped["pod_kill_skipped_wrong_duration"] += 1
            continue

        case_id = case_dir.name
        case_day = case_dir.name
        abnormal_start_ts = int(row["abnormal_start_ts"])
        abnormal_end_ts = int(row["abnormal_end_ts"])
        day_spans.append(
            DaySpan(
                day=case_day,
                start_ts=int(row["normal_start_ts"]),
                end_ts=abnormal_end_ts,
                coverage_source="case_dir_env",
                modality_counts={
                    "normal_logs": int((case_dir / "normal_logs.parquet").exists()),
                    "normal_metrics": int((case_dir / "normal_metrics.parquet").exists()),
                    "normal_metrics_sum": int((case_dir / "normal_metrics_sum.parquet").exists()),
                    "normal_traces": int((case_dir / "normal_traces.parquet").exists()),
                    "abnormal_logs": int((case_dir / "abnormal_logs.parquet").exists()),
                    "abnormal_metrics": int((case_dir / "abnormal_metrics.parquet").exists()),
                    "abnormal_metrics_sum": int((case_dir / "abnormal_metrics_sum.parquet").exists()),
                    "abnormal_traces": int((case_dir / "abnormal_traces.parquet").exists()),
                },
            )
        )
        admitted_cases.append(
            FaultCase(
                dataset="hd4",
                case_id=case_id,
                day=case_day,
                start_ts=abnormal_start_ts,
                end_ts=abnormal_end_ts,
                interval_source="env_abnormal_span",
                label_granularity="service",
                fault_type=str(injection.get("injection_name") or case_dir.name),
                raw_target="|".join(target.name for target in row["positives"]),
                positives=list(row["positives"]),
                metadata={
                    "case_dir": case_dir.name,
                    "benchmark": injection.get("benchmark"),
                    "description": injection.get("description"),
                    "display_config": injection.get("display_config"),
                    "engine_config": injection.get("engine_config"),
                    "fault_type_code": injection.get("fault_type"),
                    "ground_truth": dict(injection.get("ground_truth") or {}),
                    "task_id": injection.get("task_id"),
                    "namespace": env.get("NAMESPACE"),
                    "timezone": env.get("TIMEZONE"),
                    "normal_start_ts": int(row["normal_start_ts"]),
                    "normal_end_ts": int(row["normal_end_ts"]),
                    "injection_start_drift_seconds": int(row["start_drift_seconds"]),
                    "injection_end_drift_seconds": int(row["end_drift_seconds"]),
                    "pod_kill_duration_seconds": fixed_pod_kill_duration if row["is_pod_kill"] else None,
                },
            )
        )

        guarded_normal_end = min(int(row["normal_end_ts"]), abnormal_start_ts - int(guard_band_seconds))
        if int(row["normal_start_ts"]) < guarded_normal_end:
            normal_windows.append(
                NormalWindow(
                    dataset="hd4",
                    window_id="hd4:normal:%s" % case_dir.name,
                    day=case_day,
                    start_ts=int(row["normal_start_ts"]),
                    end_ts=guarded_normal_end,
                    source="env_normal_span",
                    metadata={
                        "case_dir": case_dir.name,
                        "guard_band_seconds": int(guard_band_seconds),
                        "original_normal_end_ts": int(row["normal_end_ts"]),
                    },
                )
            )
        else:
            skipped["normal_window_too_short_after_guard"] += 1

    admitted_cases.sort(key=lambda item: (item.start_ts, item.case_id))
    normal_windows.sort(key=lambda item: (item.start_ts, item.window_id))
    duration_values = [case.end_ts - case.start_ts for case in admitted_cases]

    validation = {
        "dataset": "hd4",
        "source_rows": source_rows,
        "admitted_case_count": len(admitted_cases),
        "normal_window_count": len(normal_windows),
        "case_ids": [case.case_id for case in admitted_cases],
        "cases_by_namespace": dict(
            Counter(str(case.metadata.get("namespace") or "") for case in admitted_cases)
        ),
        "cases_by_service": dict(
            Counter(case.positives[0].name for case in admitted_cases if case.positives)
        ),
        "pod_kill_duration_counts": dict(sorted(pod_kill_durations.items())),
        "fixed_pod_kill_duration_seconds": fixed_pod_kill_duration,
        "interval_sources": dict(Counter(case.interval_source for case in admitted_cases)),
        "start_drift_seconds": dict(sorted(start_drift_counter.items())),
        "end_drift_seconds": dict(sorted(end_drift_counter.items())),
        "median_fault_duration_seconds": median_int(duration_values, window_size_seconds),
        "skipped_counts": dict(skipped),
    }

    return DatasetManifest(
        dataset="hd4",
        case_window_seconds=median_int(duration_values, window_size_seconds),
        guard_band_seconds=guard_band_seconds,
        cases=admitted_cases,
        normal_windows=normal_windows,
        day_spans=day_spans,
        validation=validation,
    )
