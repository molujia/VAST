"""Independent adapter review probes; no benchmark telemetry or implementation edits."""

from dataclasses import asdict

import pytest

from vast_v3.data.adapters import aiops22_rows_to_case, rcabench_rows_to_case
from vast_v3.data.manifests import CandidateCatalog, WindowSpec, curate_frozen_windows
from vast_v3.data.schema import Entity


def catalog():
    return CandidateCatalog(
        tuple(Entity("service:" + name, "service") for name in ("a", "b", "c", "absent")),
        "review-catalog", "a" * 64,
    )


def window(dataset="rcabench"):
    return WindowSpec("review", dataset, "opaque-source", "2022-05-03", 0., 3., 4., 8.,
                      "explicit review reference", "b" * 64)


def span(time, span_id, parent, owner, *, trace="t", duration=1.):
    return {"time": time, "trace_id": trace, "span_id": span_id,
            "parent_span_id": parent, "service_name": owner, "duration": duration}


def test_in_window_child_retains_known_owner_when_parent_starts_in_window_gap():
    loaded = rcabench_rows_to_case(window(), catalog(), timestamp_unit="s", trace_rows=[
        span(3.9, "p", "", "a"),
        span(4.1, "c", "p", "b", duration=7.),
    ])
    assert len(loaded.case.numeric) == 1, "A known out-of-window parent must not erase its in-window child measurement"
    edge, = loaded.case.relations
    series, = loaded.case.numeric
    assert (edge.source, edge.target) == ("service:a", "service:b")
    assert series.owner == edge.id
    assert series.times == (4.1,)
    assert series.values == (7.,)
    assert loaded.audit["trace_unresolved_parent_records"] == 0


@pytest.mark.parametrize("source", ["rcabench", "aiops"])
def test_missing_parent_column_is_structural_parse_failure(source):
    row = span(4.1, "c", "", "b")
    del row["parent_span_id"]
    if source == "rcabench":
        with pytest.raises(ValueError):
            rcabench_rows_to_case(window(), catalog(), trace_rows=[row], timestamp_unit="s")
    else:
        row = {"timestamp": 4100, "cmdb_id": "b", "span_id": "c", "trace_id": "t", "duration": 1}
        with pytest.raises(ValueError):
            aiops22_rows_to_case(window("aiops2022_pre"), catalog(), trace_rows=[row])


def test_identical_span_ids_across_traces_keep_distinct_directed_owner_values():
    loaded = rcabench_rows_to_case(window(), catalog(), timestamp_unit="s", trace_rows=[
        span(4., "p", "", "a", trace="first"),
        span(4.2, "c", "p", "b", trace="first", duration=11.),
        span(4., "p", "", "c", trace="second"),
        span(4.2, "c", "p", "b", trace="second", duration=29.),
    ])
    owners = {edge.id: (edge.source, edge.target) for edge in loaded.case.relations}
    assert {owners[s.owner]: s.values for s in loaded.case.numeric} == {
        ("service:a", "service:b"): (11.,),
        ("service:c", "service:b"): (29.,),
    }
    assert loaded.case.candidate_ids == catalog().candidate_ids
    assert set(asdict(loaded.case)) == {
        "case_id", "entities", "reference_start", "incident_start", "incident_end", "numeric", "logs", "relations"
    }


def test_malformed_out_of_window_observation_is_not_silently_filtered():
    row = {"time": 100., "service_name": "a", "metric": "cpu", "value": "broken"}
    with pytest.raises(ValueError):
        rcabench_rows_to_case(window(), catalog(), metric_rows=[row], timestamp_unit="s")


def test_all_earlier_overlapping_faults_bound_reference_not_only_previous_start():
    rows = [
        {"window_id": name, "source_id": name, "window_kind": "fault", "day": "day",
         "start_ts": start, "end_ts": end, "metadata_json": {"interval_source": provenance}}
        for name, start, end, provenance in [
            ("long", 400, 1200, "run_table_exact"),
            ("short", 500, 600, "run_table_level_median"),
            ("inside", 1000, 1100, "run_table_type_median"),
            ("after", 1250, 1400, "run_table_exact"),
        ]
    ]
    windows, audit = curate_frozen_windows(rows, dataset="aiops2022_pre", source_sha256="c" * 64)
    assert [w.case_id for w in windows] == ["long", "short", "inside", "after"]
    assert [(w.reference_start, w.reference_end) for w in windows] == [
        (100., 400.), (500., 500.), (1000., 1000.), (1200., 1250.)
    ]
    assert audit["reference_counts"] == {"full_300": 1, "zero": 2, "shortened": 1}
