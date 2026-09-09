"""Python 3.8-compatible validated event selection for DGL consumers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .artifacts import sha256_file, validate_completed_bundle
from .datasets import resolve_dataset


ARTIFACT_MODE = "artifact"
LEGACY_FALLBACK_MODE = "legacy_fallback"
SELECTION_AUDIT_SCHEMA_VERSION = "rcl-event-selection-audit-v1"


@dataclass
class EventArtifactSelection:
    mode: str
    canonical_dataset_id: str
    generator_id: str
    events: List[Dict[str, Any]]
    case_id_to_native_id: Dict[str, str]
    manifest: Dict[str, Any]
    audit: Dict[str, Any]


def _read_json(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a JSON object" % context)
    return dict(payload)


def _read_jsonl(path: Path, context: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "%s line %d is not an object"
                        % (context, line_number)
                    )
                rows.append(dict(payload))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    return rows


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    artifact_name: str,
) -> Path:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifact manifest artifacts must be an object")
    entry = artifacts.get(artifact_name)
    if not isinstance(entry, Mapping):
        raise ValueError("artifact %s is missing" % artifact_name)
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("artifact %s path is unsafe" % artifact_name)
    path = root / relative
    if not path.is_file():
        raise ValueError("artifact %s file is missing: %s" % (artifact_name, path))
    return path


def _canonical_dataset(identifier: str) -> str:
    resolution = resolve_dataset(identifier)
    if resolution.alias_used:
        raise ValueError(
            "expected_dataset_id must be canonical, got legacy alias %r"
            % identifier
        )
    return resolution.canonical_id


def _normalized_native_case_id(
    dataset_id: str, native_case_id: str
) -> str:
    text = str(native_case_id).strip()
    resolution = resolve_dataset(dataset_id)
    for prefix in (
        resolution.canonical_id + "::",
        resolution.legacy_alias + "::",
        resolution.legacy_alias + ":",
    ):
        if text.startswith(prefix):
            suffix = text[len(prefix) :].strip()
            if not suffix:
                raise ValueError(
                    "native case ID has an empty dataset-prefixed suffix: %r"
                    % text
                )
            return suffix
    return text


def _fallback_selection(
    *,
    dataset_id: str,
    expected_native_case_ids: Set[str],
    fallback_reason: str,
) -> EventArtifactSelection:
    reason = str(fallback_reason).strip()
    if not reason:
        raise ValueError("legacy_fallback requires a non-empty fallback_reason")
    audit = {
        "schema_version": SELECTION_AUDIT_SCHEMA_VERSION,
        "mode": LEGACY_FALLBACK_MODE,
        "canonical_dataset_id": dataset_id,
        "event_generator_id": "legacy_canonical",
        "bundle_validated": False,
        "used_legacy_fallback": True,
        "fallback_reason": reason,
        "expected_native_case_count": len(expected_native_case_ids),
        "event_count": 0,
        "root_cause_label_access": "forbidden",
    }
    return EventArtifactSelection(
        mode=LEGACY_FALLBACK_MODE,
        canonical_dataset_id=dataset_id,
        generator_id="legacy_canonical",
        events=[],
        case_id_to_native_id={},
        manifest={},
        audit=audit,
    )


def load_event_selection(
    *,
    mode: str,
    expected_dataset_id: str,
    expected_native_case_ids: Sequence[str],
    artifact_root: Optional[Path] = None,
    expected_event_generator_id: Optional[str] = None,
    expected_config_sha256: Optional[str] = None,
    fallback_reason: str = "",
) -> EventArtifactSelection:
    """Resolve either a validated artifact or an explicit legacy fallback."""

    dataset_id = _canonical_dataset(expected_dataset_id)
    expected_native = {
        str(case_id).strip()
        for case_id in expected_native_case_ids
        if str(case_id).strip()
    }
    selected_mode = str(mode).strip()
    if selected_mode == LEGACY_FALLBACK_MODE:
        if artifact_root is not None:
            raise ValueError(
                "legacy_fallback must not provide an artifact_root"
            )
        return _fallback_selection(
            dataset_id=dataset_id,
            expected_native_case_ids=expected_native,
            fallback_reason=fallback_reason,
        )
    if selected_mode != ARTIFACT_MODE:
        raise ValueError("event selection mode must be artifact or legacy_fallback")
    if artifact_root is None:
        raise ValueError("artifact mode requires artifact_root")
    if not str(expected_event_generator_id or "").strip():
        raise ValueError(
            "artifact mode requires expected_event_generator_id"
        )
    if not str(expected_config_sha256 or "").strip():
        raise ValueError("artifact mode requires expected_config_sha256")

    root = Path(artifact_root)
    manifest_path = root / "manifest.json"
    untrusted_manifest = _read_json(manifest_path, "artifact manifest")
    manifest_case_ids = {
        str(case_id) for case_id in untrusted_manifest.get("case_ids", [])
    }
    if not manifest_case_ids:
        raise ValueError("artifact manifest case_ids must not be empty")
    manifest = validate_completed_bundle(
        root,
        expected_dataset_id=dataset_id,
        expected_event_generator_id=str(expected_event_generator_id),
        expected_config_sha256=str(expected_config_sha256),
        expected_case_ids=manifest_case_ids,
        verify_source_series_content=False,
    )
    cases_path = _artifact_path(root, manifest, "cases")
    events_path = _artifact_path(root, manifest, "events")
    case_rows = _read_jsonl(cases_path, "canonical case manifest")
    event_rows = _read_jsonl(events_path, "selected anomaly events")

    expected_by_normalized = {}
    for expected_native_id in sorted(expected_native):
        normalized = _normalized_native_case_id(
            dataset_id, expected_native_id
        )
        previous = expected_by_normalized.get(normalized)
        if previous is not None and previous != expected_native_id:
            raise ValueError(
                "expected native case IDs collide after dataset-prefix "
                "normalization: %s and %s"
                % (previous, expected_native_id)
            )
        expected_by_normalized[normalized] = expected_native_id

    case_id_to_native_id = {}
    native_to_case_id = {}
    artifact_native_by_normalized = {}
    for row in case_rows:
        case_id = str(row.get("case_id", "")).strip()
        native_case_id = str(row.get("native_case_id", "")).strip()
        if not case_id or not native_case_id:
            raise ValueError(
                "canonical case rows require case_id and native_case_id"
            )
        if native_case_id in native_to_case_id:
            raise ValueError(
                "duplicate native_case_id in event artifact: %s"
                % native_case_id
            )
        native_to_case_id[native_case_id] = case_id
        normalized = _normalized_native_case_id(
            dataset_id, native_case_id
        )
        previous_native = artifact_native_by_normalized.get(normalized)
        if (
            previous_native is not None
            and previous_native != native_case_id
        ):
            raise ValueError(
                "artifact native case IDs collide after dataset-prefix "
                "normalization: %s and %s"
                % (previous_native, native_case_id)
            )
        artifact_native_by_normalized[normalized] = native_case_id
        case_id_to_native_id[case_id] = expected_by_normalized.get(
            normalized, native_case_id
        )

    available_native = set(native_to_case_id)
    available_normalized = set(artifact_native_by_normalized)
    missing_normalized = sorted(
        set(expected_by_normalized) - available_normalized
    )
    missing_native = [
        expected_by_normalized[value]
        for value in missing_normalized
    ]
    if missing_native:
        raise ValueError(
            "event artifact native case coverage missing: %s"
            % ", ".join(missing_native[:10])
        )
    alias_resolution_count = sum(
        int(
            expected_by_normalized[normalized]
            != artifact_native_by_normalized[normalized]
        )
        for normalized in expected_by_normalized
    )
    events = []
    for row in event_rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in case_id_to_native_id:
            raise ValueError(
                "event references unknown canonical case_id %r" % case_id
            )
        event = dict(row)
        event["native_case_id"] = case_id_to_native_id[case_id]
        events.append(event)
    events.sort(
        key=lambda row: (
            str(row["native_case_id"]),
            str(row.get("service", "")),
            float(row.get("start_ts", 0.0)),
            str(row.get("metric_id", "")),
            str(row.get("event_id", "")),
        )
    )
    event_generator = dict(manifest.get("event_generator", {}))
    audit = {
        "schema_version": SELECTION_AUDIT_SCHEMA_VERSION,
        "mode": ARTIFACT_MODE,
        "canonical_dataset_id": dataset_id,
        "event_generator_id": str(event_generator.get("id", "")),
        "event_generator_config_sha256": str(
            event_generator.get("config_sha256", "")
        ),
        "event_generator_source_series_sha256": str(
            event_generator.get("source_series_sha256", "")
        ),
        "artifact_root": str(root),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "bundle_validated": True,
        "source_series_content_validated": False,
        "used_legacy_fallback": False,
        "fallback_reason": "",
        "event_count": len(events),
        "native_case_alias_resolution_count": (
            alias_resolution_count
        ),
        "native_case_coverage": {
            "expected": len(expected_native),
            "available": len(available_native),
            "missing": missing_native,
        },
        "root_cause_label_access": "forbidden",
    }
    return EventArtifactSelection(
        mode=ARTIFACT_MODE,
        canonical_dataset_id=dataset_id,
        generator_id=str(event_generator.get("id", "")),
        events=events,
        case_id_to_native_id=case_id_to_native_id,
        manifest=manifest,
        audit=audit,
    )


def write_event_selection_audit(
    path: Path,
    selection: EventArtifactSelection,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = dict(selection.audit)
    if extra:
        payload["consumer"] = dict(extra)
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


__all__ = [
    "ARTIFACT_MODE",
    "EventArtifactSelection",
    "LEGACY_FALLBACK_MODE",
    "SELECTION_AUDIT_SCHEMA_VERSION",
    "load_event_selection",
    "write_event_selection_audit",
]
