"""HD3 authoritative case reconstruction."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .common import (
    DatasetManifest,
    DaySpan,
    FaultCase,
    dedupe_targets,
    local_day_bounds,
    local_day_from_timestamp,
    make_case_id,
    make_host_target,
    make_service_target,
    parse_utc_iso8601,
)
from .normal_windows import sample_normal_windows


POD_KILL_KEYWORD = "pod kill"


def _load_pod_mapping(path: Path) -> Dict[str, List[Tuple[int, int, Dict[str, str]]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping_by_day = {}
    for day, interval_map in raw.items():
        entries = []
        _, default_day_end = local_day_bounds(day)
        for interval_key, pod_map in interval_map.items():
            start_raw, end_raw = interval_key.split("&", 1)
            start_ts = parse_utc_iso8601(start_raw)
            end_ts = parse_utc_iso8601(end_raw) if end_raw != "None" else default_day_end
            entries.append((start_ts, end_ts, pod_map))
        mapping_by_day[day] = entries
    return mapping_by_day


def _load_day_spans(
    dataset_root: Path,
    pod_mapping: Dict[str, List[Tuple[int, int, Dict[str, str]]]],
) -> List[DaySpan]:
    day_spans = []
    archive_days = {path.name.replace(".tar.gz", "") for path in dataset_root.glob("*.tar.gz")}

    for day in sorted(archive_days):
        if day not in pod_mapping:
            continue
        if pod_mapping[day]:
            start_ts = min(item[0] for item in pod_mapping[day])
            end_candidates = [item[1] for item in pod_mapping[day] if item[1] >= item[0]]
            end_ts = max(end_candidates) if end_candidates else local_day_bounds(day)[1]
        else:
            start_ts, end_ts = local_day_bounds(day)
        day_spans.append(
            DaySpan(
                day=day,
                start_ts=start_ts,
                end_ts=end_ts,
                coverage_source="archive_plus_pod_mapping",
                modality_counts={"archive": 1, "pod_mapping_intervals": len(pod_mapping[day])},
            )
        )
    return day_spans


def _to_positive_targets(row: Dict[str, object]):
    instance_type = row["instance_type"]
    if instance_type == "node":
        return [make_host_target(str(row["instance"]))]

    if row.get("source") and row.get("destination"):
        return dedupe_targets(
            [
                make_service_target(str(row["source"])),
                make_service_target(str(row["destination"])),
            ]
        )

    service_name = row.get("service")
    if service_name:
        return [make_service_target(str(service_name))]

    instance = row.get("instance")
    if isinstance(instance, list) and instance:
        return [make_service_target(str(instance[0]))]
    return [make_service_target(str(instance))]


def build_hd3_manifest(
    dataset_root: Path,
    window_size_seconds: int = 300,
    guard_band_seconds: int = 20,
) -> DatasetManifest:
    pod_mapping = _load_pod_mapping(dataset_root / "pod_instance_mapping.json")
    day_spans = _load_day_spans(dataset_root, pod_mapping)
    day_spans_by_day = {item.day: item for item in day_spans}

    source_rows = 0
    admitted_cases: List[FaultCase] = []
    skipped = Counter()

    with (dataset_root / "groundtruth.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            source_rows += 1
            row = json.loads(line)
            if POD_KILL_KEYWORD in row["fault_type"].lower():
                skipped["pod_kill"] += 1
                continue

            start_ts = parse_utc_iso8601(row["start_time"])
            end_ts = parse_utc_iso8601(row["end_time"])
            day = local_day_from_timestamp(start_ts)
            if day not in day_spans_by_day:
                skipped["missing_archive_day"] += 1
                continue

            positives = _to_positive_targets(row)
            raw_target = row["instance"] if not isinstance(row["instance"], list) else "|".join(row["instance"])
            admitted_cases.append(
                FaultCase(
                    dataset="hd3",
                    case_id=make_case_id("hd3", row["uuid"]),
                    day=day,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    interval_source="groundtruth_jsonl",
                    label_granularity=row["instance_type"],
                    fault_type=row["fault_type"],
                    raw_target=str(raw_target),
                    positives=positives,
                    metadata={
                        "uuid": row["uuid"],
                        "fault_category": row["fault_category"],
                        "service": row["service"],
                        "instance": row["instance"],
                        "source": row["source"],
                        "destination": row["destination"],
                        "key_metrics": row.get("key_metrics", []),
                    },
                )
            )

    admitted_cases.sort(key=lambda item: (item.start_ts, item.case_id))
    target_windows_by_day = Counter(case.day for case in admitted_cases)
    normal_windows, normal_diagnostics = sample_normal_windows(
        dataset="hd3",
        day_spans=day_spans,
        fault_cases=admitted_cases,
        window_size_seconds=window_size_seconds,
        guard_band_seconds=guard_band_seconds,
        target_windows_by_day=target_windows_by_day,
    )

    validation = {
        "dataset": "hd3",
        "source_rows": source_rows,
        "admitted_case_count": len(admitted_cases),
        "archive_days": [item.day for item in day_spans],
        "cases_by_day": dict(target_windows_by_day),
        "cases_by_instance_type": dict(Counter(case.label_granularity for case in admitted_cases)),
        "interval_sources": dict(Counter(case.interval_source for case in admitted_cases)),
        "skipped_counts": dict(skipped),
        "multi_positive_case_count": sum(1 for case in admitted_cases if len(case.positives) > 1),
        "normal_window_diagnostics": normal_diagnostics,
    }

    return DatasetManifest(
        dataset="hd3",
        case_window_seconds=window_size_seconds,
        guard_band_seconds=guard_band_seconds,
        cases=admitted_cases,
        normal_windows=normal_windows,
        day_spans=day_spans,
        validation=validation,
    )
