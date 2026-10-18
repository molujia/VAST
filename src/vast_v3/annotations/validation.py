"""Structural and evidence consistency checks, not scientific adjudication.

No model predictions are accepted or consulted. A valid document does not prove
causality, genuine expert review, or the correctness of an evidence interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    is_draft: bool
    artifact_files_verified: bool


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(files("vast_v3.annotations").joinpath("schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def derive_ordered_paths(fragment: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Return every maximal nonempty resolved path, retaining shared vertices.

    Unknown-order edges remain in the fragment but do not enter ordered paths.
    Parallel relations with the same event sequence share one vertex path; their
    distinct relation/evidence records are retained in ``edges``. Isolated events
    have no ordered edge path. Cycles and excessive path counts fail explicitly.
    """
    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for edge in fragment["edges"]:
        if edge["temporal_order"] != "before":
            continue
        source, target = edge["source_event"], edge["target_event"]
        outgoing.setdefault(source, set()).add(target)
        outgoing.setdefault(target, set())
        incoming.setdefault(target, set()).add(source)
        incoming.setdefault(source, set())
    # Kahn's algorithm verifies all components, including a disconnected cycle.
    degrees = {node: len(parents) for node, parents in incoming.items()}
    ready = [node for node in outgoing if degrees[node] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in outgoing[node]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if visited != len(outgoing):
        raise ValueError("resolved temporal order contains a cycle")
    paths = []
    stack = [(node,) for node in sorted(outgoing) if not incoming[node]]
    while stack:
        path = stack.pop()
        children = outgoing[path[-1]]
        if not children:
            paths.append(path)
            if len(paths) > 100_000:
                raise ValueError("maximal path count exceeds validation safety limit; no truncation applied")
        else:
            stack.extend(path + (child,) for child in sorted(children))
    return tuple(sorted(paths))


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))


def _safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    posix, windows = PurePosixPath(normalized), PureWindowsPath(value)
    return (not posix.is_absolute() and not windows.is_absolute()
            and not windows.drive and not windows.root
            and ".." not in posix.parts and ":" not in normalized
            and "\x00" not in normalized and normalized not in ("", "."))


def validate_annotation(
    document: Mapping[str, Any], *, for_submission: bool = False,
    evidence_root: str | Path | None = None,
) -> ValidationReport:
    """Validate a draft or completed annotation and optionally hash raw evidence.

    ``submitted`` always invokes submission checks, even without the keyword.
    Empty/unknown drafts are legal. File verification is opt-in and never reported
    for an empty evidence set. SHA256 checks authenticate bytes, not interpretations.
    """
    warnings = ["Structural validation does not verify scientific truth, causal influence, or genuine expert confirmation."]
    errors = [f"{'/'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
              for error in _validator().iter_errors(document)]
    is_draft = isinstance(document, Mapping) and document.get("status") == "draft"
    if errors:
        return ValidationReport(False, tuple(errors), tuple(warnings), is_draft, False)
    submitted = for_submission or document["status"] == "submitted"

    def error(path: str, message: str) -> None:
        errors.append(f"{path}: {message}")

    def finite_numbers(value: Any, path: str = "$") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            error(path, "non-finite numbers are not JSON annotation values")
        elif isinstance(value, dict):
            for key, item in value.items():
                finite_numbers(item, f"{path}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                finite_numbers(item, f"{path}/{index}")

    finite_numbers(document)
    case = document["case"]
    if submitted:
        if document["status"] != "submitted":
            error("status", "a draft cannot be submitted without explicit completion")
        if document["purpose"] is None:
            error("purpose", "a completed annotation needs its intended use")
        for key, value in case.items():
            if value is None:
                error(f"case/{key}", "required for submission")
        if document["provenance"]["preparation_basis"] != "observation_based":
            error("provenance/preparation_basis", "template-only material is not a completed annotation")
        if not document["entities"]:
            error("entities", "a completed annotation needs the observed entity catalog")
        if document["coverage"]["overall"] != "sufficient" and not document["coverage"]["limitations"]:
            error("coverage/limitations", "explain partial, insufficient, or unknown coverage")

    window_start = _timestamp(case["window_start"]) if case["window_start"] else None
    window_end = _timestamp(case["window_end"]) if case["window_end"] else None
    if window_start and window_end and window_end < window_start:
        error("case", "window_end precedes window_start")
    origin = _timestamp(case["time_origin"]) if case["time_origin"] else None
    if origin and window_start and window_end and not window_start <= origin <= window_end:
        error("case/time_origin", "must lie within the declared case window")

    def interval(value: dict, path: str) -> tuple[datetime, datetime]:
        start, end = _timestamp(value["start"]), _timestamp(value["end"])
        if end < start:
            error(path, "interval end precedes start")
        if window_start and start < window_start or window_end and end > window_end:
            error(path, "interval lies outside the declared case window")
        return start, end

    indexes: dict[str, dict[str, dict]] = {}
    for collection in ("entities", "evidence", "events", "symptoms", "relations", "fragments"):
        indexes[collection] = {}
        for item in document[collection]:
            if item["id"] in indexes[collection]:
                error(collection, f"duplicate ID {item['id']!r}")
            indexes[collection][item["id"]] = item

    def refs(values: list[str], collection: str, path: str) -> None:
        for value in values:
            if value not in indexes[collection]:
                error(path, f"unknown {collection} reference {value!r}")

    def evidence_refs(item: dict, path: str, *, required: bool = False) -> None:
        refs(item["evidence_refs"], "evidence", path + "/evidence_refs")
        if required and not item["evidence_refs"]:
            error(path + "/evidence_refs", "observed assertions require evidence references")

    def observation_evidence(item: dict, path: str) -> None:
        """Require at least one entity-and-time matching observation, not mere IDs."""
        if not item["evidence_refs"]:
            return  # Draft emptiness/submission requirements are handled separately.
        start, end = _timestamp(item["interval"]["start"]), _timestamp(item["interval"]["end"])
        supporting = []
        for evidence_id in item["evidence_refs"]:
            source = indexes["evidence"].get(evidence_id)
            if source is None:
                continue
            source_start = _timestamp(source["interval"]["start"])
            source_end = _timestamp(source["interval"]["end"])
            if item["entity_id"] in source["entity_ids"] and start <= source_end and source_start <= end:
                supporting.append(source)
        if not supporting:
            error(path + "/evidence_refs", "no referenced evidence covers this entity and overlaps its event interval")
        if "modality" in item and supporting and not any(source["modality"] == item["modality"] for source in supporting):
            error(path + "/evidence_refs", "symptom modality does not match any supporting evidence")

    confirmation = document["provenance"]["expert_confirmation"]
    confirmation_fields = ("confirmed_by", "confirmed_at", "confirmation_ref")
    if confirmation["status"] == "confirmed":
        for key in confirmation_fields:
            if confirmation[key] is None:
                error("provenance/expert_confirmation", f"confirmed status requires actual {key}")
        warnings.append("Expert confirmation metadata is present; its authenticity requires an independent human check.")
    elif any(confirmation[key] is not None for key in confirmation_fields):
        error("provenance/expert_confirmation", "non-confirmed status must not carry a confirmation signature")

    supervision = document["root_supervision"]
    if supervision["availability"] == "complete_set" and not supervision["source_ref"]:
        error("root_supervision/source_ref", "complete root facts need an authorized label-sidecar reference")
    for entity in document["entities"]:
        path = f"entities/{entity['id']}"
        evidence_refs(entity, path, required=entity["observation_state"] != "unknown")
        if entity["evidence_refs"] and not any(
            entity["id"] in indexes["evidence"][reference]["entity_ids"]
            for reference in entity["evidence_refs"] if reference in indexes["evidence"]
        ):
            error(path + "/evidence_refs", "no referenced evidence concerns this entity's observation state")
        if supervision["availability"] == "unavailable" and entity["root_role"] != "unknown":
            error(path + "/root_role", "root/nonroot facts require complete root supervision, not observed symptoms")
        if not entity["candidate"] and entity["root_role"] == "root":
            error(path + "/root_role", "a root must belong to the candidate catalog")

    file_errors = 0
    evidence_base = Path(evidence_root).resolve() if evidence_root is not None else None
    for item in document["evidence"]:
        path = f"evidence/{item['id']}"
        refs(item["entity_ids"], "entities", path + "/entity_ids")
        interval(item["interval"], path + "/interval")
        if not _safe_relative_path(item["path"]):
            error(path + "/path", "must be a package-relative path without traversal or drive names")
            file_errors += 1
            continue
        if evidence_base is not None:
            try:
                evidence_file = (evidence_base / item["path"].replace("\\", "/")).resolve()
                if not evidence_file.is_relative_to(evidence_base):
                    raise ValueError("resolved evidence path escapes the package")
                if not evidence_file.is_file():
                    raise ValueError("evidence file is missing or is not a regular file")
                with evidence_file.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest.lower() != item["sha256"].lower():
                    raise ValueError("evidence SHA256 does not match")
            except (OSError, ValueError) as issue:
                error(path, str(issue))
                file_errors += 1
    if evidence_base is None:
        warnings.append("Evidence file existence and SHA256 were not checked; supply evidence_root to check package bytes.")
    elif not document["evidence"]:
        warnings.append("No evidence files were supplied; file verification is not claimed.")

    event_times = {}
    for event in document["events"]:
        path = f"events/{event['id']}"
        refs([event["entity_id"]], "entities", path + "/entity_id")
        refs(event["symptom_ids"], "symptoms", path + "/symptom_ids")
        evidence_refs(event, path, required=submitted)
        event_times[event["id"]] = interval(event["interval"], path + "/interval")
        observation_evidence(event, path)
        for symptom_id in event["symptom_ids"]:
            symptom = indexes["symptoms"].get(symptom_id)
            if symptom and symptom["entity_id"] != event["entity_id"]:
                error(path, "symptom belongs to a different entity")
            if symptom:
                event_start, event_end = event_times[event["id"]]
                symptom_start = _timestamp(symptom["interval"]["start"])
                symptom_end = _timestamp(symptom["interval"]["end"])
                if event_end < symptom_start or symptom_end < event_start:
                    error(path + "/symptom_ids", "referenced symptom does not overlap this event's time interval")
    for symptom in document["symptoms"]:
        path = f"symptoms/{symptom['id']}"
        refs([symptom["entity_id"]], "entities", path + "/entity_id")
        interval(symptom["interval"], path + "/interval")
        evidence_refs(symptom, path, required=submitted)
        observation_evidence(symptom, path)
        if (symptom["category"] == "unknown" or symptom["direction"] == "unknown"
                or symptom["magnitude"] is None) and not symptom["unknown_reason"]:
            error(path + "/unknown_reason", "explain unknown category, direction, or unquantified magnitude")

    def event_context(event_ids: list[str]) -> tuple[set[str], datetime | None, datetime | None]:
        known = [indexes["events"][event_id] for event_id in event_ids if event_id in indexes["events"]]
        if not known:
            return set(), window_start, window_end
        return ({event["entity_id"] for event in known},
                min(event_times[event["id"]][0] for event in known),
                max(event_times[event["id"]][1] for event in known))

    def graph_evidence(item: dict, path: str, contexts: list[tuple[set[str], datetime | None, datetime | None]]) -> None:
        """Check citation relevance, not causal support or the expert interpretation.

        A context spans the linked events: relation evidence between endpoint
        observations is allowed. At least one citation must match an entity and
        this same temporal context; unrelated extra background is not adjudicated.
        """
        if not item["evidence_refs"]:
            return  # Submission requirements are handled by evidence_refs.
        for reference in item["evidence_refs"]:
            source = indexes["evidence"].get(reference)
            if source is None:
                continue
            source_start = _timestamp(source["interval"]["start"])
            source_end = _timestamp(source["interval"]["end"])
            for entity_ids, start, end in contexts:
                if (entity_ids.intersection(source["entity_ids"])
                        and (start is None or start <= source_end)
                        and (end is None or source_start <= end)):
                    return
        error(path + "/evidence_refs", "no referenced evidence matches a linked entity and its graph-claim time context")

    relation_contexts: dict[str, list[tuple[set[str], datetime | None, datetime | None]]] = {}
    for fragment in document["fragments"]:
        for edge in fragment["edges"]:
            relation_contexts.setdefault(edge["relation_id"], []).append(
                event_context([edge["source_event"], edge["target_event"]]))

    for relation in document["relations"]:
        path = f"relations/{relation['id']}"
        endpoints = {relation["source_entity"], relation["target_entity"]}
        refs(list(endpoints), "entities", path)
        evidence_refs(relation, path, required=submitted)
        contexts = relation_contexts.get(relation["id"])
        if not contexts:
            related_events = [event["id"] for event in document["events"] if event["entity_id"] in endpoints]
            _, start, end = event_context(related_events)
            contexts = [(endpoints, start, end)]
        graph_evidence(relation, path, contexts)
        for field in ("call_direction", "impact_direction"):
            direction = relation[field]
            if direction is not None and {direction["source"], direction["target"]} != endpoints:
                error(path + "/" + field, "direction endpoints must match the relation entities")
        if relation["type"] == "call" and relation["call_direction"] is None:
            error(path + "/call_direction", "call relations need an observed call direction")
        if relation["type"] != "call" and relation["call_direction"] is not None:
            error(path + "/call_direction", "non-call relations must not imply a call")
        if relation["type"] == "same_entity" and len(endpoints) != 1:
            error(path, "same_entity relation must retain one entity identity")
        if relation["impact_status"] == "unknown" and relation["impact_direction"] is not None:
            error(path + "/impact_direction", "unknown impact must not invent a direction")
        if relation["impact_status"] != "unknown" and relation["impact_direction"] is None:
            error(path + "/impact_direction", "a supported or hypothesized impact needs an explicit direction")

    for fragment in document["fragments"]:
        path = f"fragments/{fragment['id']}"
        refs(fragment["event_ids"], "events", path + "/event_ids")
        evidence_refs(fragment, path, required=submitted)
        graph_evidence(fragment, path, [event_context(fragment["event_ids"])])
        if not fragment["event_ids"]:
            error(path, "a fragment must contain at least one real event")
        edge_ids = set()
        for edge in fragment["edges"]:
            edge_path = path + f"/edges/{edge['id']}"
            if edge["id"] in edge_ids:
                error(edge_path, "duplicate fragment edge ID")
            edge_ids.add(edge["id"])
            refs([edge["relation_id"]], "relations", edge_path + "/relation_id")
            evidence_refs(edge, edge_path, required=submitted)
            source, target = edge["source_event"], edge["target_event"]
            graph_evidence(edge, edge_path, [event_context([source, target])])
            for endpoint in (source, target):
                if endpoint not in fragment["event_ids"]:
                    error(edge_path, f"event {endpoint!r} is not a member of this fragment")
            if source == target:
                error(edge_path, "an edge must join distinct event vertices")
            relation = indexes["relations"].get(edge["relation_id"])
            if relation and source in indexes["events"] and target in indexes["events"]:
                actual = {indexes["events"][source]["entity_id"], indexes["events"][target]["entity_id"]}
                if actual != {relation["source_entity"], relation["target_entity"]}:
                    error(edge_path, "event entities do not match the referenced relation")
            lower, upper = edge["delay_seconds"]["lower"], edge["delay_seconds"]["upper"]
            if edge["temporal_order"] == "unknown":
                if lower is not None or upper is not None:
                    error(edge_path + "/delay_seconds", "unknown order requires null delay bounds, not invented zero delay")
                if not edge["unknown_reason"]:
                    error(edge_path + "/unknown_reason", "unknown order requires an explanation")
            else:
                if lower is None or upper is None:
                    error(edge_path + "/delay_seconds", "resolved order requires delay bounds in physical seconds")
                elif lower > upper:
                    error(edge_path + "/delay_seconds", "lower delay exceeds upper delay")
                if source in event_times and target in event_times:
                    start_s, end_s = event_times[source]
                    start_t, end_t = event_times[target]
                    if end_s >= start_t:
                        error(edge_path + "/temporal_order", "before order requires strictly disjoint source/target uncertainty intervals")
                    expected = ((start_t - end_s).total_seconds(), (end_t - start_s).total_seconds())
                    if lower is not None and upper is not None and any(
                        not math.isclose(value, bound, rel_tol=1e-9, abs_tol=1e-9)
                        for value, bound in zip((lower, upper), expected)
                    ):
                        error(edge_path + "/delay_seconds", "delay bounds must equal interval-derived physical-time bounds")
        try:
            expected_paths = derive_ordered_paths(fragment)
            if set(map(tuple, fragment["ordered_paths"])) != set(expected_paths):
                error(path + "/ordered_paths", "must list all complete maximal ordered paths, preserving shared nodes")
        except ValueError as issue:
            error(path + "/ordered_paths", str(issue))

    verified = evidence_base is not None and bool(document["evidence"]) and file_errors == 0
    return ValidationReport(not errors, tuple(errors), tuple(warnings), is_draft, verified)
