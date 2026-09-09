"""Shared manifest types and label normalization helpers."""

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


LOCAL_TZ = timezone(timedelta(hours=8))
UTC = timezone.utc


@dataclass(frozen=True)
class EntityTarget:
    """A ranked RCA target in the unified service-plus-host space."""

    entity_id: str
    entity_type: str
    name: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
        }


@dataclass(frozen=True)
class DaySpan:
    """A day-level telemetry coverage interval."""

    day: str
    start_ts: int
    end_ts: int
    coverage_source: str
    modality_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "start_time": isoformat_local(self.start_ts),
            "end_time": isoformat_local(self.end_ts),
            "coverage_source": self.coverage_source,
            "modality_counts": dict(self.modality_counts),
        }


@dataclass(frozen=True)
class FaultCase:
    """A validated fault case used for downstream RCA experiments."""

    dataset: str
    case_id: str
    day: str
    start_ts: int
    end_ts: int
    interval_source: str
    label_granularity: str
    fault_type: str
    raw_target: str
    positives: Sequence[EntityTarget]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "day": self.day,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "start_time": isoformat_local(self.start_ts),
            "end_time": isoformat_local(self.end_ts),
            "duration_seconds": self.end_ts - self.start_ts,
            "interval_source": self.interval_source,
            "label_granularity": self.label_granularity,
            "fault_type": self.fault_type,
            "raw_target": self.raw_target,
            "positives": [item.to_dict() for item in self.positives],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NormalWindow:
    """A safe normal window sampled outside all admitted fault intervals."""

    dataset: str
    window_id: str
    day: str
    start_ts: int
    end_ts: int
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_id": self.window_id,
            "day": self.day,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "start_time": isoformat_local(self.start_ts),
            "end_time": isoformat_local(self.end_ts),
            "duration_seconds": self.end_ts - self.start_ts,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatasetManifest:
    """Serializable dataset rebuild artifact."""

    dataset: str
    case_window_seconds: int
    guard_band_seconds: int
    cases: Sequence[FaultCase]
    normal_windows: Sequence[NormalWindow]
    day_spans: Sequence[DaySpan]
    validation: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "case_window_seconds": self.case_window_seconds,
            "guard_band_seconds": self.guard_band_seconds,
            "case_count": len(self.cases),
            "normal_window_count": len(self.normal_windows),
            "day_span_count": len(self.day_spans),
            "cases": [item.to_dict() for item in self.cases],
            "normal_windows": [item.to_dict() for item in self.normal_windows],
            "day_spans": [item.to_dict() for item in self.day_spans],
            "validation": dict(self.validation),
        }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def isoformat_local(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, LOCAL_TZ).isoformat()


def local_day_bounds(day: str) -> Tuple[int, int]:
    start_dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=LOCAL_TZ)
    end_dt = start_dt + timedelta(days=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def parse_local_datetime(value: str, fmt: str) -> int:
    dt = datetime.strptime(value, fmt).replace(tzinfo=LOCAL_TZ)
    return int(dt.timestamp())


def parse_utc_iso8601(value: str) -> int:
    dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return int(dt.timestamp())


def local_day_from_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, LOCAL_TZ).strftime("%Y-%m-%d")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def natural_sort_key(value: Any) -> Tuple[Any, ...]:
    text = str(value)
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def make_entity_id(entity_type: str, name: str) -> str:
    return "%s:%s" % (entity_type, name)


@lru_cache(maxsize=4096)
def normalize_service_name(value: str) -> str:
    if not value:
        return value

    match = re.match(r"^(.*?service)(?:\d+)?-\d+$", value)
    if match:
        return match.group(1)

    match = re.match(r"^(.*?)(?:\d+)-\d+$", value)
    if match:
        return match.group(1)

    match = re.match(r"^(.*?)-\d+$", value)
    if match:
        return match.group(1)

    match = re.match(r"^(.*?service)\d+$", value)
    if match:
        return match.group(1)

    return value


def make_service_target(name: str) -> EntityTarget:
    canonical = normalize_service_name(name)
    return EntityTarget(
        entity_id=make_entity_id("service", canonical),
        entity_type="service",
        name=canonical,
    )


def make_host_target(name: str) -> EntityTarget:
    return EntityTarget(
        entity_id=make_entity_id("host", name),
        entity_type="host",
        name=name,
    )


def dedupe_targets(targets: Iterable[EntityTarget]) -> List[EntityTarget]:
    seen = set()
    result = []
    for target in targets:
        if target.entity_id in seen:
            continue
        seen.add(target.entity_id)
        result.append(target)
    return result


def to_positive_targets(label_granularity: str, raw_target: str) -> List[EntityTarget]:
    if label_granularity == "node":
        return [make_host_target(raw_target)]
    if label_granularity in ("pod", "service"):
        return [make_service_target(raw_target)]
    raise ValueError("Unsupported label granularity: %s" % label_granularity)


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for file_path in path.rglob("*") if file_path.is_file())


def count_files_by_name(day_root: Path, names: Sequence[str]) -> Dict[str, int]:
    return {name: count_files(day_root / name) for name in names}


def median_int(values: Sequence[int], fallback: int) -> int:
    if not values:
        return fallback
    return int(round(median(list(values))))


def make_case_id(dataset: str, *parts: Any) -> str:
    slug_parts = []
    for part in parts:
        text = str(part)
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
        slug_parts.append(text or "na")
    return "%s:%s" % (dataset, ":".join(slug_parts))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def flatten_case_row(case: FaultCase) -> Dict[str, Any]:
    case_dict = case.to_dict()
    return {
        "dataset": case_dict["dataset"],
        "case_id": case_dict["case_id"],
        "day": case_dict["day"],
        "start_ts": case_dict["start_ts"],
        "end_ts": case_dict["end_ts"],
        "start_time": case_dict["start_time"],
        "end_time": case_dict["end_time"],
        "duration_seconds": case_dict["duration_seconds"],
        "interval_source": case_dict["interval_source"],
        "label_granularity": case_dict["label_granularity"],
        "fault_type": case_dict["fault_type"],
        "raw_target": case_dict["raw_target"],
        "positive_ids": ";".join(item["entity_id"] for item in case_dict["positives"]),
        "positive_types": ";".join(item["entity_type"] for item in case_dict["positives"]),
        "positive_names": ";".join(item["name"] for item in case_dict["positives"]),
        "metadata_json": json.dumps(case_dict["metadata"], ensure_ascii=False, sort_keys=True),
    }


def flatten_normal_window_row(window: NormalWindow) -> Dict[str, Any]:
    window_dict = window.to_dict()
    return {
        "dataset": window_dict["dataset"],
        "window_id": window_dict["window_id"],
        "day": window_dict["day"],
        "start_ts": window_dict["start_ts"],
        "end_ts": window_dict["end_ts"],
        "start_time": window_dict["start_time"],
        "end_time": window_dict["end_time"],
        "duration_seconds": window_dict["duration_seconds"],
        "source": window_dict["source"],
        "metadata_json": json.dumps(window_dict["metadata"], ensure_ascii=False, sort_keys=True),
    }


def write_manifest_bundle(output_dir: Path, manifest: DatasetManifest) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "manifest.json", manifest.to_dict())
    write_json(output_dir / "validation.json", dict(manifest.validation))
    write_jsonl(output_dir / "cases.jsonl", [item.to_dict() for item in manifest.cases])
    write_jsonl(
        output_dir / "normal_windows.jsonl",
        [item.to_dict() for item in manifest.normal_windows],
    )
    write_csv(
        output_dir / "cases.csv",
        [flatten_case_row(item) for item in manifest.cases],
        [
            "dataset",
            "case_id",
            "day",
            "start_ts",
            "end_ts",
            "start_time",
            "end_time",
            "duration_seconds",
            "interval_source",
            "label_granularity",
            "fault_type",
            "raw_target",
            "positive_ids",
            "positive_types",
            "positive_names",
            "metadata_json",
        ],
    )
    write_csv(
        output_dir / "normal_windows.csv",
        [flatten_normal_window_row(item) for item in manifest.normal_windows],
        [
            "dataset",
            "window_id",
            "day",
            "start_ts",
            "end_ts",
            "start_time",
            "end_time",
            "duration_seconds",
            "source",
            "metadata_json",
        ],
    )


def entity_target_from_dict(payload: Mapping[str, Any]) -> EntityTarget:
    return EntityTarget(
        entity_id=str(payload["entity_id"]),
        entity_type=str(payload["entity_type"]),
        name=str(payload["name"]),
    )


def day_span_from_dict(payload: Mapping[str, Any]) -> DaySpan:
    return DaySpan(
        day=str(payload["day"]),
        start_ts=int(payload["start_ts"]),
        end_ts=int(payload["end_ts"]),
        coverage_source=str(payload["coverage_source"]),
        modality_counts=dict(payload.get("modality_counts", {})),
    )


def fault_case_from_dict(payload: Mapping[str, Any]) -> FaultCase:
    return FaultCase(
        dataset=str(payload["dataset"]),
        case_id=str(payload["case_id"]),
        day=str(payload["day"]),
        start_ts=int(payload["start_ts"]),
        end_ts=int(payload["end_ts"]),
        interval_source=str(payload["interval_source"]),
        label_granularity=str(payload["label_granularity"]),
        fault_type=str(payload["fault_type"]),
        raw_target=str(payload["raw_target"]),
        positives=[entity_target_from_dict(item) for item in payload.get("positives", [])],
        metadata=dict(payload.get("metadata", {})),
    )


def normal_window_from_dict(payload: Mapping[str, Any]) -> NormalWindow:
    return NormalWindow(
        dataset=str(payload["dataset"]),
        window_id=str(payload["window_id"]),
        day=str(payload["day"]),
        start_ts=int(payload["start_ts"]),
        end_ts=int(payload["end_ts"]),
        source=str(payload["source"]),
        metadata=dict(payload.get("metadata", {})),
    )


def load_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetManifest(
        dataset=str(payload["dataset"]),
        case_window_seconds=int(payload["case_window_seconds"]),
        guard_band_seconds=int(payload["guard_band_seconds"]),
        cases=[fault_case_from_dict(item) for item in payload.get("cases", [])],
        normal_windows=[normal_window_from_dict(item) for item in payload.get("normal_windows", [])],
        day_spans=[day_span_from_dict(item) for item in payload.get("day_spans", [])],
        validation=dict(payload.get("validation", {})),
    )
