"""Synthetic schema fixtures only: no real incident or expert annotation data."""

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def api():
    try:
        return importlib.import_module("vast_v3.annotations.validation")
    except ModuleNotFoundError as error:
        pytest.fail(f"annotation validator is missing: {error}")


def blank():
    path = ROOT / "annotations" / "blank-template.json"
    assert path.is_file(), "blank annotation template is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def interval(second, end=None):
    return {"start": f"2026-01-01T00:00:{second:02d}+00:00",
            "end": f"2026-01-01T00:00:{second if end is None else end:02d}+00:00",
            "resolution_seconds": 0.1}


def synthetic_document():
    doc = blank()
    doc["status"] = "submitted"
    doc["purpose"] = "semantic_evaluation"
    doc["case"] = {"dataset": "D1", "case_id": "synthetic-unit-fixture-only",
                   "window_start": "2026-01-01T00:00:00+00:00",
                   "window_end": "2026-01-01T00:01:00+00:00",
                   "time_origin": "2026-01-01T00:00:00+00:00",
                   "time_unit": "seconds", "source_manifest_sha256": "1" * 64}
    doc["provenance"]["preparation_basis"] = "observation_based"
    doc["coverage"] = {"overall": "partial", "missing_modalities": ["log"],
                       "limitations": ["Synthetic test fixture; not research evidence."]}
    doc["entities"] = [{"id": name, "category": "service", "candidate": True,
                        "root_role": "unknown", "observation_state": "unknown",
                        "evidence_refs": []} for name in ("a", "b", "c", "d")]
    doc["evidence"] = [{"id": "raw", "modality": "trace", "path": "raw.csv",
                        "sha256": hashlib.sha256(b"synthetic-test-only\n").hexdigest(),
                        "locator": "synthetic fixture row 1", "interval": interval(0, 10),
                        "entity_ids": ["a", "b", "c", "d"],
                        "note": "Unit-test data, not actual observations."}]
    doc["events"] = [{"id": name, "entity_id": name, "interval": interval(time),
                      "symptom_ids": [], "evidence_refs": ["raw"]}
                     for name, time in (("a", 1), ("b", 1), ("c", 3), ("d", 5))]
    doc["relations"] = [{"id": name, "type": "call", "source_entity": source,
                          "target_entity": target,
                          "call_direction": {"source": source, "target": target},
                          "impact_direction": None, "impact_status": "unknown",
                          "evidence_refs": ["raw"], "explanation": "Observed call; impact unresolved."}
                         for name, source, target in (("ac", "a", "c"), ("bc", "b", "c"), ("cd", "c", "d"))]
    doc["fragments"] = [{"id": "convergence", "event_ids": ["a", "b", "c", "d"],
                          "edges": [{"id": name, "source_event": source, "target_event": target,
                                     "relation_id": name, "temporal_order": "before",
                                     "delay_seconds": {"lower": 2., "upper": 2.},
                                     "evidence_refs": ["raw"], "unknown_reason": None}
                                    for name, source, target in (("ac", "a", "c"), ("bc", "b", "c"), ("cd", "c", "d"))],
                          "family": "convergence", "ordered_paths": [["a", "c", "d"], ["b", "c", "d"]],
                          "interpretation": "Synthetic shared-node test only.",
                          "certainty": "supported_interpretation", "evidence_refs": ["raw"],
                          "unknowns": ["Calls alone do not establish causal influence."]}]
    return doc


def test_blank_draft_validates_without_fake_case_times_or_expert_values():
    doc = blank()
    report = api().validate_annotation(doc)
    assert report.valid, report.errors
    assert report.is_draft and not report.artifact_files_verified
    assert doc["case"]["case_id"] is None
    assert doc["case"]["window_start"] is None
    assert doc["case"]["source_manifest_sha256"] is None
    assert doc["entities"] == doc["events"] == doc["evidence"] == []
    assert doc["provenance"]["expert_confirmation"]["status"] == "not_reviewed"


def test_blank_cannot_be_submitted_as_a_completed_annotation():
    report = api().validate_annotation(blank(), for_submission=True)
    assert not report.valid
    assert any("case" in error or "status" in error for error in report.errors)


def test_complete_maximal_paths_keep_shared_event_vertices_once():
    doc = synthetic_document()
    report = api().validate_annotation(doc, for_submission=True)
    assert report.valid, report.errors
    assert set(api().derive_ordered_paths(doc["fragments"][0])) == {("a", "c", "d"), ("b", "c", "d")}
    assert len(doc["events"]) == 4
    assert report.warnings and not report.artifact_files_verified


def test_abbreviated_paths_are_not_substituted_for_maximal_paths():
    doc = synthetic_document()
    doc["fragments"][0]["ordered_paths"] = [["a", "c"], ["b", "c"], ["c", "d"]]
    report = api().validate_annotation(doc, for_submission=True)
    assert not report.valid and any("maximal" in error for error in report.errors)


def test_unknown_order_preserves_edge_without_inventing_a_delay():
    doc = synthetic_document()
    doc["events"][0]["interval"] = interval(1, 4)
    edge = doc["fragments"][0]["edges"][0]
    edge.update(temporal_order="unknown", delay_seconds={"lower": None, "upper": None},
                unknown_reason="Intervals overlap at available resolution.")
    doc["fragments"][0]["ordered_paths"] = [["b", "c", "d"]]
    report = api().validate_annotation(doc, for_submission=True)
    assert report.valid, report.errors
    assert len(doc["fragments"][0]["edges"]) == 3
    edge["delay_seconds"] = {"lower": 0., "upper": 0.}
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_before_relation_requires_disjoint_evidence_intervals():
    doc = synthetic_document()
    doc["events"][0]["interval"] = interval(1, 4)
    report = api().validate_annotation(doc, for_submission=True)
    assert not report.valid and any("order" in error for error in report.errors)


@pytest.mark.parametrize("change", ["unknown_entity", "unknown_evidence", "duplicate_event", "prediction"])
def test_broken_references_duplicates_and_predictions_fail(change):
    doc = synthetic_document()
    if change == "unknown_entity":
        doc["events"][0]["entity_id"] = "absent"
    elif change == "unknown_evidence":
        doc["events"][0]["evidence_refs"] = ["absent"]
    elif change == "duplicate_event":
        doc["events"].append(deepcopy(doc["events"][0]))
    else:
        doc["predicted_top1"] = "a"
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_root_role_and_observation_state_are_independent():
    doc = synthetic_document()
    doc["root_supervision"] = {"availability": "complete_set", "source_ref": "authorized-ground-truth-sidecar"}
    for entity in doc["entities"]:
        entity.update(root_role="root" if entity["id"] == "a" else "nonroot",
                      observation_state="not_detected" if entity["id"] == "a" else "detected",
                      evidence_refs=["raw"])
    report = api().validate_annotation(doc, for_submission=True)
    assert report.valid, report.errors
    doc["root_supervision"]["availability"] = "unavailable"
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_unreviewed_assistant_annotation_cannot_claim_expert_confirmation():
    doc = synthetic_document()
    doc["provenance"]["expert_confirmation"]["status"] = "confirmed"
    report = api().validate_annotation(doc, for_submission=True)
    assert not report.valid and any("confirmation" in error for error in report.errors)


def test_evidence_file_hashes_are_optional_but_never_claimed_without_checks(tmp_path):
    doc = synthetic_document()
    (tmp_path / "raw.csv").write_bytes(b"synthetic-test-only\n")
    verified = api().validate_annotation(doc, for_submission=True, evidence_root=tmp_path)
    assert verified.valid and verified.artifact_files_verified
    (tmp_path / "raw.csv").write_bytes(b"changed synthetic fixture\n")
    failed = api().validate_annotation(doc, for_submission=True, evidence_root=tmp_path)
    assert not failed.valid and not failed.artifact_files_verified


def test_evidence_paths_cannot_escape_the_case_package(tmp_path):
    doc = synthetic_document()
    doc["evidence"][0]["path"] = "../outside.csv"
    report = api().validate_annotation(doc, for_submission=True, evidence_root=tmp_path)
    assert not report.valid


def test_insufficient_observation_can_be_honestly_submitted_without_fragments():
    doc = synthetic_document()
    doc["fragments"] = []
    doc["coverage"]["overall"] = "insufficient"
    assert api().validate_annotation(doc, for_submission=True).valid


def test_resolved_order_cycle_is_rejected_not_truncated():
    fragment = {"edges": [{"source_event": "a", "target_event": "b", "temporal_order": "before"},
                          {"source_event": "b", "target_event": "a", "temporal_order": "before"}]}
    with pytest.raises(ValueError, match="cycle"):
        api().derive_ordered_paths(fragment)


def test_packaged_schema_is_identical_to_expert_facing_schema():
    from importlib.resources import files

    packaged = files("vast_v3.annotations").joinpath("schema.json")
    assert json.loads(packaged.read_text(encoding="utf-8")) == json.loads(
        (ROOT / "annotations" / "schema.json").read_text(encoding="utf-8"))


def test_event_evidence_must_cover_its_entity_and_overlap_its_time():
    doc = synthetic_document()
    doc["evidence"][0]["interval"] = interval(10, 20)
    assert not api().validate_annotation(doc, for_submission=True).valid
    doc = synthetic_document()
    doc["evidence"][0]["entity_ids"] = ["b", "c", "d"]
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_unknown_impact_is_not_filled_from_call_direction():
    doc = synthetic_document()
    doc["relations"][0]["impact_direction"] = {"source": "a", "target": "c"}
    assert not api().validate_annotation(doc, for_submission=True).valid
    doc["relations"][0].update(impact_direction={"source": "c", "target": "a"},
                               impact_status="hypothesized")
    # A reverse impact hypothesis is legal and never rewrites the observed call.
    assert api().validate_annotation(doc, for_submission=True).valid


@pytest.mark.parametrize("path", ["../raw.csv", "C:/raw.csv", r"C:raw.csv", r"\\server\raw.csv", "raw.csv:stream"])
def test_unsafe_evidence_paths_fail_even_without_file_access(path):
    doc = synthetic_document()
    doc["evidence"][0]["path"] = path
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_delay_bounds_cannot_be_arbitrarily_narrowed():
    doc = synthetic_document()
    doc["events"][0]["interval"] = interval(1, 2)
    doc["fragments"][0]["edges"][0]["delay_seconds"] = {"lower": 1., "upper": 2.}
    assert api().validate_annotation(doc, for_submission=True).valid
    doc["fragments"][0]["edges"][0]["delay_seconds"] = {"lower": 1.5, "upper": 1.5}
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_symptoms_keep_unquantified_magnitude_unknown():
    doc = synthetic_document()
    doc["symptoms"] = [{"id": "s", "entity_id": "a", "category": "latency", "modality": "trace",
                        "direction": "increase", "magnitude": None, "interval": interval(1),
                        "evidence_refs": ["raw"], "unknown_reason": "Baseline not available."}]
    doc["events"][0]["symptom_ids"] = ["s"]
    assert api().validate_annotation(doc, for_submission=True).valid
    doc["symptoms"][0]["unknown_reason"] = None
    assert not api().validate_annotation(doc, for_submission=True).valid


def test_lowercase_utc_timestamp_is_valid_rfc3339():
    doc = synthetic_document()
    doc["case"]["window_start"] = "2026-01-01t00:00:00z"
    assert api().validate_annotation(doc, for_submission=True).valid


@pytest.mark.parametrize("record", ["relation", "edge", "fragment"])
@pytest.mark.parametrize("mismatch", ["entity", "time"])
def test_graph_evidence_checks_entity_and_time_separately(record, mismatch):
    doc = synthetic_document()
    extra = deepcopy(doc["evidence"][0])
    extra["id"] = "mismatched"
    if mismatch == "entity":
        doc["entities"].append({"id": "unrelated", "category": "service", "candidate": True,
                                "root_role": "unknown", "observation_state": "unknown", "evidence_refs": []})
        extra["entity_ids"] = ["unrelated"]
    else:
        extra["interval"] = interval(40, 50)
    doc["evidence"].append(extra)
    target = {"relation": doc["relations"][0],
              "edge": doc["fragments"][0]["edges"][0],
              "fragment": doc["fragments"][0]}[record]
    target["evidence_refs"] = ["mismatched"]
    report = api().validate_annotation(doc, for_submission=True)
    assert not report.valid
    assert any("evidence" in error for error in report.errors)


@pytest.mark.parametrize("record", ["relation", "edge", "fragment"])
def test_graph_context_can_use_evidence_between_endpoint_events(record):
    doc = synthetic_document()
    context = deepcopy(doc["evidence"][0])
    context.update(id="context", entity_ids=["a", "c"], interval=interval(2))
    doc["evidence"].append(context)
    target = {"relation": doc["relations"][0],
              "edge": doc["fragments"][0]["edges"][0],
              "fragment": doc["fragments"][0]}[record]
    target["evidence_refs"] = ["context"]
    report = api().validate_annotation(doc, for_submission=True)
    assert report.valid, report.errors


def test_event_and_symptom_can_share_a_boundary_timestamp():
    doc = synthetic_document()
    doc["symptoms"] = [{"id": "s", "entity_id": "a", "category": "latency", "modality": "trace",
                        "direction": "increase", "magnitude": None, "interval": interval(1, 2),
                        "evidence_refs": ["raw"], "unknown_reason": "Baseline not available."}]
    doc["events"][0]["symptom_ids"] = ["s"]
    report = api().validate_annotation(doc, for_submission=True)
    assert report.valid, report.errors
