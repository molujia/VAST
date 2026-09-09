"""HD1 authoritative case reconstruction with conservative interval recovery."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from .common import (
    DatasetManifest,
    DaySpan,
    FaultCase,
    count_files_by_name,
    local_day_bounds,
    local_day_from_timestamp,
    make_case_id,
    median_int,
    normalize_text,
    parse_local_datetime,
    to_positive_targets,
)
from .normal_windows import sample_normal_windows


DEFAULT_HD1_DURATION_SECONDS = 300


def _load_day_spans(dataset_root: Path) -> List[DaySpan]:
    day_spans = []
    for day_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        day = day_dir.name
        if not day.startswith("2022-"):
            continue
        cloudbed_dir = day_dir / "cloudbed"
        modality_counts = {
            "log": count_files_by_name(cloudbed_dir, ("log",)).get("log", 0),
            "metric": count_files_by_name(cloudbed_dir, ("metric",)).get("metric", 0),
            "trace": count_files_by_name(cloudbed_dir, ("trace",)).get("trace", 0),
        }
        start_ts, end_ts = local_day_bounds(day)
        day_spans.append(
            DaySpan(
                day=day,
                start_ts=start_ts,
                end_ts=end_ts,
                coverage_source="cloudbed_directory",
                modality_counts=modality_counts,
            )
        )
    return day_spans


def _normalized_run_key(row: Mapping[str, str]) -> Tuple[str, str, str, str]:
    return (
        row["timestamp"],
        normalize_text(row["level"]),
        normalize_text(row["cmdb_id"]),
        normalize_text(row["failure_type"]),
    )


def _load_run_table(dataset_root: Path):
    run_rows_by_key = {}
    durations_by_type = defaultdict(list)
    durations_by_level = defaultdict(list)
    offsets_by_type = defaultdict(list)
    offsets_by_level = defaultdict(list)
    split_by_key = {}

    with (dataset_root / "22AIOps_run_table.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _normalized_run_key(row)
            start_ts = parse_local_datetime(row["start"], "%Y/%m/%d %H:%M")
            end_ts = parse_local_datetime(row["end"], "%Y/%m/%d %H:%M")
            duration_seconds = end_ts - start_ts
            offset_seconds = start_ts - int(row["timestamp"])

            payload = dict(row)
            payload["start_ts"] = start_ts
            payload["end_ts"] = end_ts
            payload["duration_seconds"] = duration_seconds
            payload["offset_seconds"] = offset_seconds

            run_rows_by_key[key] = payload
            durations_by_type[(key[1], key[3])].append(duration_seconds)
            durations_by_level[key[1]].append(duration_seconds)
            offsets_by_type[(key[1], key[3])].append(offset_seconds)
            offsets_by_level[key[1]].append(offset_seconds)
            split_by_key[key] = row.get("type") or "unknown"

    return {
        "rows_by_key": run_rows_by_key,
        "durations_by_type": durations_by_type,
        "durations_by_level": durations_by_level,
        "offsets_by_type": offsets_by_type,
        "offsets_by_level": offsets_by_level,
        "split_by_key": split_by_key,
    }


def _infer_interval(
    timestamp: int,
    level: str,
    failure_type: str,
    stats: Mapping[str, Mapping[Tuple[str, str], List[int]]],
) -> Tuple[int, int, str]:
    level_key = normalize_text(level)
    type_key = normalize_text(failure_type)
    pair_key = (level_key, type_key)

    pair_durations = stats["durations_by_type"].get(pair_key, [])
    level_durations = stats["durations_by_level"].get(level_key, [])
    pair_offsets = stats["offsets_by_type"].get(pair_key, [])
    level_offsets = stats["offsets_by_level"].get(level_key, [])

    if pair_durations:
        duration_seconds = median_int(pair_durations, DEFAULT_HD1_DURATION_SECONDS)
        offset_seconds = median_int(pair_offsets, 0)
        source = "run_table_type_median"
    elif level_durations:
        duration_seconds = median_int(level_durations, DEFAULT_HD1_DURATION_SECONDS)
        offset_seconds = median_int(level_offsets, 0)
        source = "run_table_level_median"
    else:
        duration_seconds = DEFAULT_HD1_DURATION_SECONDS
        offset_seconds = 0
        source = "global_default"

    start_ts = timestamp + offset_seconds
    end_ts = start_ts + duration_seconds
    return start_ts, end_ts, source


def build_hd1_manifest(
    dataset_root: Path,
    window_size_seconds: int = 300,
    guard_band_seconds: int = 20,
) -> DatasetManifest:
    day_spans = _load_day_spans(dataset_root)
    run_stats = _load_run_table(dataset_root)

    admitted_cases: List[FaultCase] = []
    source_rows = 0
    local_day_mismatch_count = 0
    interval_source_counter = Counter()
    split_counter = Counter()

    checked_dir = dataset_root / "checked_groundtruth_new"
    for checked_path in sorted(checked_dir.glob("groundtruth-2022-*.csv")):
        day = checked_path.stem.replace("groundtruth-", "")
        with checked_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_rows += 1
                timestamp = int(row["timestamp"])
                if local_day_from_timestamp(timestamp) != day:
                    local_day_mismatch_count += 1

                key = _normalized_run_key(row)
                matched_run = run_stats["rows_by_key"].get(key)
                if matched_run is not None:
                    start_ts = matched_run["start_ts"]
                    end_ts = matched_run["end_ts"]
                    interval_source = "run_table_exact"
                    split_counter[run_stats["split_by_key"].get(key, "unknown")] += 1
                else:
                    start_ts, end_ts, interval_source = _infer_interval(
                        timestamp=timestamp,
                        level=row["level"],
                        failure_type=row["failure_type"],
                        stats=run_stats,
                    )
                    split_counter["unknown"] += 1

                interval_source_counter[interval_source] += 1
                positives = to_positive_targets(row["level"], row["cmdb_id"])
                admitted_cases.append(
                    FaultCase(
                        dataset="hd1",
                        case_id=make_case_id(
                            "hd1",
                            day,
                            row["timestamp"],
                            row["level"],
                            row["cmdb_id"],
                            row["failure_type"],
                        ),
                        day=day,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        interval_source=interval_source,
                        label_granularity=row["level"],
                        fault_type=row["failure_type"],
                        raw_target=row["cmdb_id"],
                        positives=positives,
                        metadata={
                            "original_timestamp": timestamp,
                            "matched_run_table": matched_run is not None,
                            "train_test_split": run_stats["split_by_key"].get(key, "unknown"),
                        },
                    )
                )

    admitted_cases.sort(key=lambda item: (item.start_ts, item.case_id))
    target_windows_by_day = Counter(case.day for case in admitted_cases)
    normal_windows, normal_diagnostics = sample_normal_windows(
        dataset="hd1",
        day_spans=day_spans,
        fault_cases=admitted_cases,
        window_size_seconds=window_size_seconds,
        guard_band_seconds=guard_band_seconds,
        target_windows_by_day=target_windows_by_day,
    )

    validation = {
        "dataset": "hd1",
        "source_rows": source_rows,
        "admitted_case_count": len(admitted_cases),
        "day_case_counts": dict(target_windows_by_day),
        "cases_by_scope": dict(Counter(case.label_granularity for case in admitted_cases)),
        "interval_sources": dict(interval_source_counter),
        "split_counts": dict(split_counter),
        "local_day_mismatch_count": local_day_mismatch_count,
        "matched_run_table_count": interval_source_counter.get("run_table_exact", 0),
        "inferred_interval_count": len(admitted_cases) - interval_source_counter.get("run_table_exact", 0),
        "normal_window_diagnostics": normal_diagnostics,
    }

    return DatasetManifest(
        dataset="hd1",
        case_window_seconds=window_size_seconds,
        guard_band_seconds=guard_band_seconds,
        cases=admitted_cases,
        normal_windows=normal_windows,
        day_spans=day_spans,
        validation=validation,
    )
