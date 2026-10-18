"""Independent annotation-review probes; no real cases or implementation edits.

Assertions describe the required structural consistency, not causal truth.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vast_v3.annotations import validate_annotation


ROOT = Path(__file__).resolve().parents[2]


def interval(start, end=None):
    return {
        "start": f"2026-01-01T00:00:{start:02d}+00:00",
        "end": f"2026-01-01T00:00:{start if end is None else end:02d}+00:00",
        "resolution_seconds": 0.1,
    }


def document():
    doc = json.loads((ROOT / "annotations/blank-template.json").read_text(encoding="utf-8"))
    doc.update(status="submitted", purpose="semantic_evaluation")
    doc["case"].update(
        dataset="D1", case_id="synthetic-annotation-review-only",
        window_start="2026-01-01T00:00:00+00:00",
        window_end="2026-01-01T00:01:00+00:00",
        time_origin="2026-01-01T00:00:00+00:00",
        source_manifest_sha256="1" * 64,
    )
    doc["provenance"]["preparation_basis"] = "observation_based"
    doc["coverage"].update(overall="partial", limitations=["Synthetic fixture only."])
    doc["entities"] = [
        {"id": name, "category": "service", "candidate": True,
         "root_role": "unknown", "observation_state": "unknown", "evidence_refs": []}
        for name in ("a", "b", "unrelated")
    ]
    doc["evidence"] = [
        {"id": "raw", "modality": "trace", "path": "synthetic.txt",
         "sha256": hashlib.sha256(b"synthetic-review-only\n").hexdigest(),
         "locator": "synthetic fixture", "interval": interval(0, 10),
         "entity_ids": ["a", "b"], "note": "Not research evidence."}
    ]
    doc["events"] = [
        {"id": name, "entity_id": name, "interval": interval(time),
         "symptom_ids": [], "evidence_refs": ["raw"]}
        for name, time in (("a", 1), ("b", 3))
    ]
    doc["relations"] = [
        {"id": "ab", "type": "call", "source_entity": "a", "target_entity": "b",
         "call_direction": {"source": "a", "target": "b"},
         "impact_direction": None, "impact_status": "unknown",
         "evidence_refs": ["raw"], "explanation": "Synthetic observed call only."}
    ]
    doc["fragments"] = [
        {"id": "f", "event_ids": ["a", "b"],
         "edges": [{"id": "ab", "source_event": "a", "target_event": "b",
                    "relation_id": "ab", "temporal_order": "before",
                    "delay_seconds": {"lower": 2., "upper": 2.},
                    "evidence_refs": ["raw"], "unknown_reason": None}],
         "family": "chain", "ordered_paths": [["a", "b"]],
         "interpretation": "Synthetic test only.", "certainty": "hypothesis",
         "evidence_refs": ["raw"], "unknowns": ["No causal claim."]}
    ]
    return doc


def add_unrelated_evidence(doc):
    unrelated = deepcopy(doc["evidence"][0])
    unrelated.update(id="unrelated", entity_ids=["unrelated"], interval=interval(40, 50))
    doc["evidence"].append(unrelated)


def test_review_baseline_has_real_synthetic_file_and_hash(tmp_path):
    doc = document()
    (tmp_path / "synthetic.txt").write_bytes(b"synthetic-review-only\n")
    report = validate_annotation(doc, evidence_root=tmp_path)
    assert report.valid, report.errors
    assert report.artifact_files_verified


@pytest.mark.parametrize("state", ["detected", "not_detected"])
def test_entity_observation_cannot_be_supported_only_by_another_entity(state):
    doc = document()
    add_unrelated_evidence(doc)
    doc["entities"][0].update(observation_state=state, evidence_refs=["unrelated"])
    report = validate_annotation(doc)
    assert not report.valid, "An observation about a accepted evidence solely about unrelated."


def test_event_cannot_link_a_symptom_in_a_disjoint_time_interval():
    doc = document()
    doc["symptoms"] = [
        {"id": "late", "entity_id": "a", "category": "latency", "modality": "trace",
         "direction": "increase", "magnitude": None, "interval": interval(8, 9),
         "evidence_refs": ["raw"], "unknown_reason": "No quantitative baseline."}
    ]
    doc["events"][0]["symptom_ids"] = ["late"]
    report = validate_annotation(doc)
    assert not report.valid, "Event at second 1 accepted a symptom occurring at seconds 8-9."


@pytest.mark.parametrize("record", ["relation", "edge", "fragment"])
def test_graph_claim_cannot_reference_only_unrelated_entity_and_time(record):
    doc = document()
    add_unrelated_evidence(doc)
    target = {"relation": doc["relations"][0],
              "edge": doc["fragments"][0]["edges"][0],
              "fragment": doc["fragments"][0]}[record]
    target["evidence_refs"] = ["unrelated"]
    report = validate_annotation(doc)
    assert not report.valid, f"{record} accepted evidence for unrelated entity at seconds 40-50."
