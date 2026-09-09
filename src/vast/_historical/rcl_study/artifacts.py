"""Versioned disk contracts shared by rcalab producers and dgl consumers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .datasets import resolve_dataset


CASE_SCHEMA_VERSION = "canonical-case-manifest-v1"
SERIES_SCHEMA_VERSION = "canonical-series-v1"
EVENT_SCHEMA_VERSION = "anomaly-event-v1"
AUDIT_SCHEMA_VERSION = "canonical-artifact-audit-v1"
BUNDLE_SCHEMA_VERSION = "rcl-disk-artifact-manifest-v1"
COMPLETION_SCHEMA_VERSION = "rcl-artifact-completion-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
JSON_NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)

CASE_FIELDS = (
    "schema_version",
    "canonical_dataset_id",
    "legacy_alias",
    "case_id",
    "native_case_id",
    "admitted",
    "exclusion_reason",
    "method_native_admission",
)
SERIES_FIELDS = (
    "schema_version",
    "canonical_dataset_id",
    "case_id",
    "metric_id",
    "service",
    "pod",
    "metric_name",
    "metric_semantic",
    "modality",
    "window",
    "timestamp",
    "value",
)
EVENT_FIELDS = (
    "schema_version",
    "canonical_dataset_id",
    "case_id",
    "event_id",
    "metric_id",
    "service",
    "window",
    "start_ts",
    "end_ts",
    "peak_ts",
    "score",
    "detector_id",
    "source_series_sha256",
)
AUDIT_FIELDS = (
    "schema_version",
    "canonical_dataset_id",
    "input_file_count",
    "input_case_count",
    "admitted_case_count",
    "excluded_case_count",
    "exact_duplicates_removed",
    "conflicting_series_count",
    "non_finite_row_count",
    "unresolved_service_row_count",
    "output_series_count",
    "output_row_count",
    "source_checksums",
    "code_revision",
    "config_sha256",
    "environment_manifest_path",
    "environment_manifest_sha256",
    "passed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(payload: Mapping[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError("%s missing required fields: %s" % (context, ", ".join(missing)))


def _require_sha256(value: Any, context: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError("%s must be a lowercase SHA-256 digest" % context)
    return text


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number" % context)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number" % context)
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % context)
    return number


def _canonical_dataset(identifier: str) -> str:
    resolution = resolve_dataset(identifier)
    if resolution.alias_used:
        raise ValueError("canonical_dataset_id must not use legacy alias %r" % identifier)
    return resolution.canonical_id


def _validate_dataset_row(
    row: Mapping[str, Any],
    expected_dataset_id: str,
    schema_version: str,
    index: int,
) -> None:
    if str(row.get("schema_version", "")) != schema_version:
        raise ValueError("row %d schema_version must be %s" % (index, schema_version))
    if str(row.get("canonical_dataset_id", "")) != expected_dataset_id:
        raise ValueError(
            "row %d canonical_dataset_id %r != %r"
            % (index, row.get("canonical_dataset_id"), expected_dataset_id)
        )
    case_id = str(row.get("case_id", ""))
    if not case_id.startswith(expected_dataset_id + "::"):
        raise ValueError("row %d case_id is not canonical: %r" % (index, case_id))


def validate_case_manifest_rows(
    rows: Sequence[Mapping[str, Any]], expected_dataset_id: str
) -> Dict[str, Any]:
    dataset_id = _canonical_dataset(expected_dataset_id)
    if not rows:
        raise ValueError("case manifest must not be empty")
    case_ids: List[str] = []
    expected_alias = resolve_dataset(dataset_id).legacy_alias
    for index, row in enumerate(rows):
        _require_fields(row, CASE_FIELDS, "case row %d" % index)
        _validate_dataset_row(row, dataset_id, CASE_SCHEMA_VERSION, index)
        if str(row["legacy_alias"]) != expected_alias:
            raise ValueError("case row %d legacy_alias mismatch" % index)
        if not isinstance(row["admitted"], bool):
            raise ValueError("case row %d admitted must be boolean" % index)
        if not row["admitted"] and not str(row["exclusion_reason"]).strip():
            raise ValueError("case row %d excluded without exclusion_reason" % index)
        if not isinstance(row["method_native_admission"], Mapping):
            raise ValueError("case row %d method_native_admission must be an object" % index)
        case_ids.append(str(row["case_id"]))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate canonical case_id in case manifest")
    return {"case_count": len(case_ids), "case_ids": case_ids}


def validate_canonical_series_rows(
    rows: Sequence[Mapping[str, Any]], expected_dataset_id: str
) -> Dict[str, int]:
    dataset_id = _canonical_dataset(expected_dataset_id)
    if not rows:
        raise ValueError("canonical series rows must not be empty")
    observation_keys: Set[Tuple[str, str, float]] = set()
    services_by_series: Dict[Tuple[str, str], Set[str]] = {}
    ordering: List[Tuple[str, str, float]] = []
    cases: Set[str] = set()
    for index, row in enumerate(rows):
        _require_fields(row, SERIES_FIELDS, "series row %d" % index)
        _validate_dataset_row(row, dataset_id, SERIES_SCHEMA_VERSION, index)
        case_id = str(row["case_id"])
        metric_id = str(row["metric_id"]).strip()
        service = str(row["service"]).strip()
        if not metric_id or not service:
            raise ValueError("series row %d metric_id and service must be non-empty" % index)
        timestamp = _finite_number(row["timestamp"], "series row %d timestamp" % index)
        _finite_number(row["value"], "series row %d value" % index)
        observation_key = (case_id, metric_id, timestamp)
        if observation_key in observation_keys:
            raise ValueError("duplicate canonical series observation: %r" % (observation_key,))
        observation_keys.add(observation_key)
        series_key = (case_id, metric_id)
        services_by_series.setdefault(series_key, set()).add(service)
        ordering.append(observation_key)
        cases.add(case_id)
    ambiguous = [key for key, services in services_by_series.items() if len(services) != 1]
    if ambiguous:
        raise ValueError(
            "canonical metric series maps to multiple services: %r" % (ambiguous[0],)
        )
    if ordering != sorted(ordering):
        raise ValueError("canonical series rows are not stably sorted")
    return {
        "row_count": len(rows),
        "series_count": len(services_by_series),
        "case_count": len(cases),
    }


def validate_canonical_series_file(
    path: Path, expected_dataset_id: str
) -> Dict[str, Any]:
    """Validate a sorted canonical JSONL file without retaining all rows."""

    dataset_id = _canonical_dataset(expected_dataset_id)
    source = Path(path)
    digest = hashlib.sha256()
    row_count = 0
    series_count = 0
    cases: Set[str] = set()
    previous_observation: Optional[Tuple[str, str, float]] = None
    current_series: Optional[Tuple[str, str]] = None
    current_service = ""
    try:
        with source.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "series line %d is not an object" % line_number
                    )
                row = dict(payload)
                _require_fields(
                    row, SERIES_FIELDS, "series row %d" % line_number
                )
                _validate_dataset_row(
                    row,
                    dataset_id,
                    SERIES_SCHEMA_VERSION,
                    line_number,
                )
                case_id = str(row["case_id"])
                metric_id = str(row["metric_id"]).strip()
                service = str(row["service"]).strip()
                if not metric_id or not service:
                    raise ValueError(
                        "series row %d metric_id and service must be non-empty"
                        % line_number
                    )
                timestamp = _finite_number(
                    row["timestamp"],
                    "series row %d timestamp" % line_number,
                )
                _finite_number(
                    row["value"], "series row %d value" % line_number
                )
                observation = (case_id, metric_id, timestamp)
                if (
                    previous_observation is not None
                    and observation <= previous_observation
                ):
                    if observation == previous_observation:
                        raise ValueError(
                            "duplicate canonical series observation: %r"
                            % (observation,)
                        )
                    raise ValueError(
                        "canonical series rows are not stably sorted"
                    )
                series_key = (case_id, metric_id)
                if series_key != current_series:
                    current_series = series_key
                    current_service = service
                    series_count += 1
                elif service != current_service:
                    raise ValueError(
                        "canonical metric series maps to multiple services: %r"
                        % (series_key,)
                    )
                previous_observation = observation
                cases.add(case_id)
                row_count += 1
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            "cannot validate canonical series %s: %s" % (source, exc)
        )
    if row_count == 0:
        raise ValueError("canonical series rows must not be empty")
    return {
        "row_count": row_count,
        "series_count": series_count,
        "case_count": len(cases),
        "case_ids": sorted(cases),
        "sha256": digest.hexdigest(),
    }


def validate_anomaly_event_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_dataset_id: str,
    expected_source_series_sha256: str,
) -> Dict[str, int]:
    dataset_id = _canonical_dataset(expected_dataset_id)
    source_sha256 = _require_sha256(
        expected_source_series_sha256, "expected_source_series_sha256"
    )
    event_ids: Set[str] = set()
    cases: Set[str] = set()
    for index, row in enumerate(rows):
        _require_fields(row, EVENT_FIELDS, "event row %d" % index)
        _validate_dataset_row(row, dataset_id, EVENT_SCHEMA_VERSION, index)
        event_id = str(row["event_id"]).strip()
        if not event_id:
            raise ValueError("event row %d event_id must be non-empty" % index)
        if event_id in event_ids:
            raise ValueError("duplicate anomaly event_id %r" % event_id)
        event_ids.add(event_id)
        start = _finite_number(row["start_ts"], "event row %d start_ts" % index)
        peak = _finite_number(row["peak_ts"], "event row %d peak_ts" % index)
        end = _finite_number(row["end_ts"], "event row %d end_ts" % index)
        _finite_number(row["score"], "event row %d score" % index)
        if not start <= peak <= end:
            raise ValueError("event time order must satisfy start_ts <= peak_ts <= end_ts")
        if str(row["source_series_sha256"]) != source_sha256:
            raise ValueError("event row %d source series hash mismatch" % index)
        cases.add(str(row["case_id"]))
    return {"event_count": len(rows), "case_count": len(cases)}


def validate_anomaly_event_file(
    path: Path,
    expected_dataset_id: str,
    expected_source_series_sha256: str,
) -> Dict[str, Any]:
    """Validate an event JSONL file with bounded row memory."""

    dataset_id = _canonical_dataset(expected_dataset_id)
    source_sha256 = _require_sha256(
        expected_source_series_sha256,
        "expected_source_series_sha256",
    )
    source = Path(path)
    digest = hashlib.sha256()
    event_ids: Set[str] = set()
    cases: Set[str] = set()
    row_count = 0
    try:
        with source.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "event line %d is not an object" % line_number
                    )
                row = dict(payload)
                _require_fields(
                    row, EVENT_FIELDS, "event row %d" % line_number
                )
                _validate_dataset_row(
                    row,
                    dataset_id,
                    EVENT_SCHEMA_VERSION,
                    line_number,
                )
                event_id = str(row["event_id"]).strip()
                if not event_id:
                    raise ValueError(
                        "event row %d event_id must be non-empty"
                        % line_number
                    )
                if event_id in event_ids:
                    raise ValueError(
                        "duplicate anomaly event_id %r" % event_id
                    )
                event_ids.add(event_id)
                start = _finite_number(
                    row["start_ts"],
                    "event row %d start_ts" % line_number,
                )
                peak = _finite_number(
                    row["peak_ts"],
                    "event row %d peak_ts" % line_number,
                )
                end = _finite_number(
                    row["end_ts"],
                    "event row %d end_ts" % line_number,
                )
                _finite_number(
                    row["score"], "event row %d score" % line_number
                )
                if not start <= peak <= end:
                    raise ValueError(
                        "event time order must satisfy "
                        "start_ts <= peak_ts <= end_ts"
                    )
                if str(row["source_series_sha256"]) != source_sha256:
                    raise ValueError(
                        "event row %d source series hash mismatch"
                        % line_number
                    )
                cases.add(str(row["case_id"]))
                row_count += 1
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            "cannot validate anomaly events %s: %s" % (source, exc)
        )
    return {
        "event_count": row_count,
        "case_count": len(cases),
        "case_ids": sorted(cases),
        "sha256": digest.hexdigest(),
    }


def validate_audit(
    payload: Mapping[str, Any], expected_dataset_id: str
) -> Dict[str, Any]:
    dataset_id = _canonical_dataset(expected_dataset_id)
    _require_fields(payload, AUDIT_FIELDS, "audit")
    if str(payload["schema_version"]) != AUDIT_SCHEMA_VERSION:
        raise ValueError("audit schema_version must be %s" % AUDIT_SCHEMA_VERSION)
    if str(payload["canonical_dataset_id"]) != dataset_id:
        raise ValueError("audit canonical_dataset_id mismatch")
    count_fields = AUDIT_FIELDS[2:12]
    for field in count_fields:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("audit %s must be a non-negative integer" % field)
    checksums = payload["source_checksums"]
    if not isinstance(checksums, Mapping) or not checksums:
        raise ValueError("audit source_checksums must be a non-empty object")
    for name, digest in checksums.items():
        _require_sha256(digest, "audit source_checksums[%s]" % name)
    _require_sha256(payload["config_sha256"], "audit config_sha256")
    _require_sha256(
        payload["environment_manifest_sha256"], "audit environment_manifest_sha256"
    )
    if not str(payload["code_revision"]).strip():
        raise ValueError("audit code_revision must be non-empty")
    if not str(payload["environment_manifest_path"]).strip():
        raise ValueError("audit environment_manifest_path must be non-empty")
    if not isinstance(payload["passed"], bool):
        raise ValueError("audit passed must be boolean")
    return dict(payload)


def _read_json(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a JSON object" % context)
    return dict(payload)


def _read_jsonl(path: Path, context: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("line %d is not an object" % line_number)
                rows.append(dict(payload))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    return rows


def _skip_json_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _scan_json_string_end(text: str, index: int) -> int:
    if index >= len(text) or text[index] != '"':
        raise ValueError("expected JSON string")
    index += 1
    while index < len(text):
        character = text[index]
        if character == '"':
            return index + 1
        if character == "\\":
            if index + 1 >= len(text):
                raise ValueError("unterminated JSON escape")
            escape = text[index + 1]
            if escape == "u":
                digits = text[index + 2 : index + 6]
                if (
                    len(digits) != 4
                    or any(
                        value not in "0123456789abcdefABCDEF"
                        for value in digits
                    )
                ):
                    raise ValueError("invalid JSON unicode escape")
                index += 6
                continue
            if escape not in '"\\/bfnrt':
                raise ValueError("invalid JSON escape")
            index += 2
            continue
        if ord(character) < 0x20:
            raise ValueError("unescaped control character in JSON string")
        index += 1
    raise ValueError("unterminated JSON string")


def _skip_json_value(text: str, index: int, depth: int = 0) -> int:
    if depth > 128:
        raise ValueError("JSON nesting exceeds case scanner limit")
    index = _skip_json_whitespace(text, index)
    if index >= len(text):
        raise ValueError("missing JSON value")
    character = text[index]
    if character == '"':
        return _scan_json_string_end(text, index)
    if character == "{":
        index = _skip_json_whitespace(text, index + 1)
        if index < len(text) and text[index] == "}":
            return index + 1
        while True:
            key_end = _scan_json_string_end(text, index)
            index = _skip_json_whitespace(text, key_end)
            if index >= len(text) or text[index] != ":":
                raise ValueError("JSON object is missing a colon")
            index = _skip_json_value(text, index + 1, depth + 1)
            index = _skip_json_whitespace(text, index)
            if index >= len(text):
                raise ValueError("unterminated JSON object")
            if text[index] == "}":
                return index + 1
            if text[index] != ",":
                raise ValueError("JSON object is missing a comma")
            index = _skip_json_whitespace(text, index + 1)
    if character == "[":
        index = _skip_json_whitespace(text, index + 1)
        if index < len(text) and text[index] == "]":
            return index + 1
        while True:
            index = _skip_json_value(text, index, depth + 1)
            index = _skip_json_whitespace(text, index)
            if index >= len(text):
                raise ValueError("unterminated JSON array")
            if text[index] == "]":
                return index + 1
            if text[index] != ",":
                raise ValueError("JSON array is missing a comma")
            index = _skip_json_whitespace(text, index + 1)
    for literal in ("true", "false", "null"):
        if text.startswith(literal, index):
            return index + len(literal)
    match = JSON_NUMBER_PATTERN.match(text, index)
    if match is not None:
        return match.end()
    raise ValueError("invalid JSON value")


def _selected_json_object_fields(
    text: str, selected_fields: Sequence[str]
) -> Dict[str, Any]:
    selected = {str(value) for value in selected_fields}
    result: Dict[str, Any] = {}
    index = _skip_json_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        raise ValueError("case manifest row is not an object")
    index = _skip_json_whitespace(text, index + 1)
    if index < len(text) and text[index] == "}":
        index += 1
    else:
        while True:
            key_start = index
            key_end = _scan_json_string_end(text, key_start)
            key = json.loads(text[key_start:key_end])
            if not isinstance(key, str):
                raise ValueError("JSON object key is not text")
            index = _skip_json_whitespace(text, key_end)
            if index >= len(text) or text[index] != ":":
                raise ValueError("JSON object is missing a colon")
            value_start = _skip_json_whitespace(text, index + 1)
            value_end = _skip_json_value(text, value_start, 1)
            if key in selected:
                if key in result:
                    raise ValueError(
                        "case manifest row contains duplicate field %s" % key
                    )
                result[key] = json.loads(text[value_start:value_end])
            index = _skip_json_whitespace(text, value_end)
            if index >= len(text):
                raise ValueError("unterminated JSON object")
            if text[index] == "}":
                index += 1
                break
            if text[index] != ",":
                raise ValueError("JSON object is missing a comma")
            index = _skip_json_whitespace(text, index + 1)
    if _skip_json_whitespace(text, index) != len(text):
        raise ValueError("case manifest row has trailing data")
    return result


def _read_case_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                content = line.rstrip("\r\n")
                if not content.strip():
                    continue
                try:
                    rows.append(
                        _selected_json_object_fields(content, CASE_FIELDS)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "line %d is invalid: %s" % (line_number, exc)
                    )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "cannot read case manifest %s: %s" % (path, exc)
        )
    return rows


def _artifact_path(root: Path, relative_path: Any, context: str) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("%s path is unsafe: %r" % (context, str(relative)))
    path = root / relative
    if not path.is_file():
        raise ValueError("%s file is missing: %s" % (context, path))
    return path


def validate_completed_bundle(
    root: Path,
    *,
    expected_dataset_id: str,
    expected_event_generator_id: str,
    expected_config_sha256: str,
    expected_case_ids: Set[str],
    verify_source_series_content: bool = True,
) -> Dict[str, Any]:
    root = Path(root)
    dataset_id = _canonical_dataset(expected_dataset_id)
    config_sha256 = _require_sha256(expected_config_sha256, "expected_config_sha256")
    all_done_path = root / "all.done"
    if not all_done_path.is_file():
        raise ValueError("all.done success marker is missing")
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise ValueError("COMPLETED.json success record is missing")
    if any(root.rglob("*.failed")):
        raise ValueError("bundle contains a .failed marker")

    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "artifact manifest")
    completion = _read_json(completion_path, "completion record")
    if str(manifest.get("schema_version", "")) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("artifact manifest schema_version mismatch")
    if str(completion.get("schema_version", "")) != COMPLETION_SCHEMA_VERSION:
        raise ValueError("completion schema_version mismatch")
    if str(completion.get("status", "")) != "complete":
        raise ValueError("completion status is not complete")
    if str(manifest.get("canonical_dataset_id", "")) != dataset_id:
        raise ValueError("artifact manifest canonical_dataset_id mismatch")
    expected_alias = resolve_dataset(dataset_id).legacy_alias
    if str(manifest.get("legacy_alias", "")) != expected_alias:
        raise ValueError("artifact manifest legacy_alias mismatch")
    if str(manifest.get("producer_environment", "")) not in {"rcalab", "dgl"}:
        raise ValueError("artifact manifest producer_environment is invalid")

    manifest_sha256 = sha256_file(manifest_path)
    if str(completion.get("artifact_manifest_sha256", "")) != manifest_sha256:
        raise ValueError("completion artifact manifest hash mismatch")
    if all_done_path.read_text(encoding="utf-8").strip() != manifest_sha256:
        raise ValueError("all.done artifact manifest hash mismatch")
    if str(completion.get("artifact_id", "")) != str(manifest.get("artifact_id", "")):
        raise ValueError("completion artifact_id mismatch")

    case_ids = {str(case_id) for case_id in manifest.get("case_ids", [])}
    if case_ids != set(expected_case_ids):
        raise ValueError("artifact manifest case coverage mismatch")
    event_generator = manifest.get("event_generator", {})
    if not isinstance(event_generator, Mapping):
        raise ValueError("event_generator must be an object")
    if str(event_generator.get("id", "")) != str(expected_event_generator_id):
        raise ValueError("event generator id mismatch")
    if str(event_generator.get("config_sha256", "")) != config_sha256:
        raise ValueError("event generator config hash mismatch")

    artifact_entries = manifest.get("artifacts", {})
    if not isinstance(artifact_entries, Mapping):
        raise ValueError("artifacts must be an object")
    required = {
        "cases": CASE_SCHEMA_VERSION,
        "series": SERIES_SCHEMA_VERSION,
        "events": EVENT_SCHEMA_VERSION,
        "audit": AUDIT_SCHEMA_VERSION,
    }
    resolved: Dict[str, Path] = {}
    for name, schema_version in required.items():
        entry = artifact_entries.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError("artifact entry %s is missing" % name)
        _require_fields(
            entry, ("path", "sha256", "schema_version", "row_count"), "artifact %s" % name
        )
        if str(entry["schema_version"]) != schema_version:
            raise ValueError("artifact %s schema_version mismatch" % name)
        _require_sha256(
            entry["sha256"], "artifact %s sha256" % name
        )
        row_count = entry["row_count"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
        ):
            raise ValueError(
                "artifact %s row_count must be a non-negative integer"
                % name
            )
        path = _artifact_path(root, entry["path"], "artifact %s" % name)
        if (
            name != "series" or verify_source_series_content
        ) and sha256_file(path) != str(entry["sha256"]):
            raise ValueError("artifact %s hash mismatch" % name)
        resolved[name] = path

    cases = _read_case_manifest_rows(resolved["cases"])
    audit = _read_json(resolved["audit"], "artifact audit")
    case_summary = validate_case_manifest_rows(cases, dataset_id)
    series_sha256 = str(artifact_entries["series"]["sha256"])
    if verify_source_series_content:
        series_summary = validate_canonical_series_file(
            resolved["series"], dataset_id
        )
    else:
        series_summary = {
            "row_count": int(
                artifact_entries["series"]["row_count"]
            ),
            "sha256": series_sha256,
        }
    event_summary = validate_anomaly_event_file(
        resolved["events"], dataset_id, series_sha256
    )
    if series_summary["sha256"] != series_sha256:
        raise ValueError("artifact series hash mismatch")
    if (
        event_summary["sha256"]
        != str(artifact_entries["events"]["sha256"])
    ):
        raise ValueError("artifact events hash mismatch")
    validated_audit = validate_audit(audit, dataset_id)

    summaries = {
        "cases": case_summary["case_count"],
        "series": series_summary["row_count"],
        "events": event_summary["event_count"],
        "audit": 1,
    }
    for name, row_count in summaries.items():
        if int(artifact_entries[name]["row_count"]) != row_count:
            raise ValueError("artifact %s row_count mismatch" % name)
    if set(case_summary["case_ids"]) != set(expected_case_ids):
        raise ValueError("case manifest coverage mismatch")
    if str(event_generator.get("source_series_sha256", "")) != series_sha256:
        raise ValueError("event generator source series hash mismatch")
    if not bool(validated_audit["passed"]):
        raise ValueError("artifact audit did not pass")
    return manifest


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "BUNDLE_SCHEMA_VERSION",
    "CASE_SCHEMA_VERSION",
    "COMPLETION_SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "SERIES_SCHEMA_VERSION",
    "sha256_file",
    "validate_anomaly_event_rows",
    "validate_anomaly_event_file",
    "validate_audit",
    "validate_canonical_series_file",
    "validate_canonical_series_rows",
    "validate_case_manifest_rows",
    "validate_completed_bundle",
]
