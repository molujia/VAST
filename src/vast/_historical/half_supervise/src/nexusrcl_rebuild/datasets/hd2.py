"""HD2 authoritative case reconstruction."""

import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .common import (
    DatasetManifest,
    DaySpan,
    FaultCase,
    count_files_by_name,
    local_day_bounds,
    make_case_id,
    parse_local_datetime,
    to_positive_targets,
)
from .normal_windows import sample_normal_windows


FORBIDDEN_HD2_DAYS = {"2025-05-10", "2025-05-11"}
REQUIRED_MODALITIES = ("log", "metric", "trace")


def _load_day_spans(dataset_root: Path) -> Dict[str, DaySpan]:
    day_spans = {}
    for day_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        day = day_dir.name
        modality_counts = count_files_by_name(day_dir, REQUIRED_MODALITIES)
        if any(modality_counts[name] == 0 for name in REQUIRED_MODALITIES):
            continue
        start_ts, end_ts = local_day_bounds(day)
        day_spans[day] = DaySpan(
            day=day,
            start_ts=start_ts,
            end_ts=end_ts,
            coverage_source="directory_modalities",
            modality_counts=modality_counts,
        )
    return day_spans


def _parse_timeout_seconds(matchers: str) -> Optional[int]:
    parsed = ast.literal_eval(matchers)
    for item in parsed:
        if item.get("name") != "timeout":
            continue
        values = item.get("value") or []
        if not values:
            return None
        return int(values[0])
    return None


def build_hd2_manifest(
    dataset_root: Path,
    window_size_seconds: int = 300,
    guard_band_seconds: int = 20,
) -> DatasetManifest:
    day_spans_by_day = _load_day_spans(dataset_root)
    day_spans = [day_spans_by_day[day] for day in sorted(day_spans_by_day)]

    source_rows = 0
    admitted_cases: List[FaultCase] = []
    skipped = Counter()

    inject_log_path = dataset_root / "inject_log.csv"
    with inject_log_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_rows += 1
            if row.get("Result") != "Success":
                skipped["non_success"] += 1
                continue

            day = row["Timestamp"].split()[0]
            if day in FORBIDDEN_HD2_DAYS:
                skipped["forbidden_day"] += 1
                continue
            if day not in day_spans_by_day:
                skipped["incomplete_telemetry_day"] += 1
                continue
            if row["Scope"] == "pod" and (
                row["Experiment Name"] == "pod-fail" or row["Action"] == "fail"
            ):
                skipped["pod_fail"] += 1
                continue

            duration_seconds = _parse_timeout_seconds(row["Matchers"])
            if duration_seconds is None:
                skipped["missing_timeout"] += 1
                continue

            start_ts = parse_local_datetime(row["Timestamp"], "%Y-%m-%d %H:%M:%S")
            end_ts = start_ts + duration_seconds
            positives = to_positive_targets(row["Scope"], row["Location"])
            case_id = make_case_id(
                "hd2",
                day,
                row["Timestamp"],
                row["Experiment Name"],
                row["Location"],
            )
            admitted_cases.append(
                FaultCase(
                    dataset="hd2",
                    case_id=case_id,
                    day=day,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    interval_source="inject_log_timeout",
                    label_granularity=row["Scope"],
                    fault_type=row["Experiment Name"],
                    raw_target=row["Location"],
                    positives=positives,
                    metadata={
                        "target": row["Target"],
                        "action": row["Action"],
                        "matchers": row["Matchers"],
                        "result": row["Result"],
                        "duration_seconds": duration_seconds,
                    },
                )
            )

    admitted_cases.sort(key=lambda item: (item.start_ts, item.case_id))
    target_windows_by_day = Counter(case.day for case in admitted_cases)
    normal_windows, normal_diagnostics = sample_normal_windows(
        dataset="hd2",
        day_spans=day_spans,
        fault_cases=admitted_cases,
        window_size_seconds=window_size_seconds,
        guard_band_seconds=guard_band_seconds,
        target_windows_by_day=target_windows_by_day,
    )

    validation = {
        "dataset": "hd2",
        "source_rows": source_rows,
        "admitted_case_count": len(admitted_cases),
        "admitted_days": sorted(target_windows_by_day),
        "telemetry_days": sorted(day_spans_by_day),
        "skipped_counts": dict(skipped),
        "cases_by_day": dict(target_windows_by_day),
        "cases_by_scope": dict(Counter(case.label_granularity for case in admitted_cases)),
        "interval_sources": dict(Counter(case.interval_source for case in admitted_cases)),
        "normal_window_diagnostics": normal_diagnostics,
    }

    return DatasetManifest(
        dataset="hd2",
        case_window_seconds=window_size_seconds,
        guard_band_seconds=guard_band_seconds,
        cases=admitted_cases,
        normal_windows=normal_windows,
        day_spans=day_spans,
        validation=validation,
    )
