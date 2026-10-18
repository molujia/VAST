"""Search-order fixtures explicitly contain no measured benchmark cases."""

import csv
import importlib
import json
from dataclasses import replace

import pytest


def api():
    try:
        return importlib.import_module("vast_v3.evaluation.search_ledger")
    except ModuleNotFoundError:
        pytest.fail("The append-only search ledger is not implemented yet")


def manifest():
    return api().SearchManifest(
        dataset="synthetic-fixture", search_run_id="unit-test-only",
        eligible_case_ids=("one", "two", "three", "four", "five"),
        excluded_semantic_ids=("reserved",), candidate_ids=("a", "b", "c"),
        case_manifest_hash="1"*64, code_revision="test-code-revision",
        config_hash="2"*64, preprocessing_hash="3"*64,
        checkpoint_hash="4"*64, catalog_hash="5"*64, data_hash="6"*64,
    )


def append(ledger, case, root="c", **kwargs):
    fields = dict(case_id=case, stage="search", status="scored",
                  started_at="2026-10-15T00:00:00+00:00",
                  finished_at="2026-10-15T00:00:02+00:00",
                  scores={"a": 3, "b": 2, "c": 1}, roots=[root],
                  evidence_paths=["synthetic-fixture/evidence.json"])
    fields.update(kwargs)
    return ledger.append_attempt(**fields)


def test_first_success_at_fourth_case_preserves_all_attempts_and_report(tmp_path):
    ledger = api().SearchLedger(tmp_path / "attempts.jsonl", manifest())
    for case in ("one", "two", "three"):
        append(ledger, case)
    success = append(ledger, "four", root="a")
    assert success["attempt_index"] == 4
    assert success["cumulative_hits"] == [1, 1, 4, 4, 4]
    ledger.select_case("four", reason="Synthetic observable evidence is sufficient",
                       evidence_paths=["fixture/selection.json"])
    summary = ledger.summary()
    assert summary["total_attempts"] == 4
    assert summary["attempted_unique_cases"] == 4
    assert summary["scored_unique_cases"] == 4
    assert summary["first_top1_index"] == 4
    assert summary["first_top1_case_id"] == "four"
    assert summary["selected_case_id"] == "four"
    assert summary["benchmark_estimate"] is False
    csv_path, markdown_path = ledger.export_reports(tmp_path / "reports")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_id"] for row in rows] == ["one", "two", "three", "four"]
    assert rows[-1]["selection_status"] == "selected"
    assert rows[-1]["checkpoint_hash"] == "4"*64
    assert rows[-1]["root_ranks"] == '{"a":1}'
    assert "not an unbiased benchmark" in markdown_path.read_text(encoding="utf-8")


def test_batch_case_after_first_success_is_still_recorded(tmp_path):
    ledger = api().SearchLedger(tmp_path / "attempts.jsonl", manifest())
    append(ledger, "one", root="a")
    append(ledger, "two")
    assert ledger.summary()["total_attempts"] == 2
    assert ledger.summary()["first_top1_index"] == 1


def test_error_retry_and_resume_keep_error_unscored(tmp_path):
    path = tmp_path / "attempts.jsonl"
    ledger = api().SearchLedger(path, manifest())
    first = append(ledger, "one", status="error", scores=None, roots=None,
                   error_reason="Synthetic parser failure")
    assert first["first_root_rank"] is None
    assert first["hit_at_1"] is None
    old_bytes = path.read_bytes()
    resumed = api().SearchLedger(path, manifest(), resume=True,
                                 expected_head_hash=ledger.head_hash)
    retry = append(resumed, "one", root="a", stage="raw_retry", repeat_of=1)
    assert path.read_bytes().startswith(old_bytes)
    assert retry["repeat_of"] == 1
    summary = resumed.summary()
    assert summary["total_attempts"] == 2
    assert summary["attempted_unique_cases"] == 1
    assert summary["scored_unique_cases"] == 1
    assert summary["errors"] == 1
    assert summary["retries"] == 1
    assert summary["first_top1_index"] == 2


@pytest.mark.parametrize("case,kwargs", [
    ("two", {}), ("reserved", {}), ("not-eligible", {}),
    ("one", {"repeat_of": 99}),
    ("one", {"finished_at": "2026-10-14T00:00:00+00:00"}),
    ("one", {"started_at": "2026-10-15T00:00:00"}),
    ("one", {"status": "error", "scores": None, "roots": None}),
    ("one", {"stage": "export_replay"}),
])
def test_invalid_attempt_is_rejected_without_writing(tmp_path, case, kwargs):
    path = tmp_path / "attempts.jsonl"
    ledger = api().SearchLedger(path, manifest())
    before = path.read_bytes()
    with pytest.raises(ValueError):
        append(ledger, case, **kwargs)
    assert path.read_bytes() == before


def test_manifest_excludes_semantic_evaluation_and_duplicate_candidates():
    frozen = manifest()
    with pytest.raises(ValueError):
        replace(frozen, excluded_semantic_ids=("one",))
    with pytest.raises(ValueError):
        replace(frozen, candidate_ids=("a", "a"))


def test_resume_rejects_changed_identity_old_record_or_truncation(tmp_path):
    path = tmp_path / "attempts.jsonl"
    ledger = api().SearchLedger(path, manifest())
    append(ledger, "one")
    original = path.read_bytes()
    head = ledger.head_hash
    with pytest.raises(ValueError):
        api().SearchLedger(path, replace(manifest(), checkpoint_hash="f"*64), resume=True)
    changed = original.replace(b'"first_root_rank":3', b'"first_root_rank":1')
    assert changed != original
    path.write_bytes(changed)
    with pytest.raises(ValueError):
        api().SearchLedger(path, manifest(), resume=True)
    path.write_bytes(original.splitlines(keepends=True)[0])
    with pytest.raises(ValueError):
        api().SearchLedger(path, manifest(), resume=True, expected_head_hash=head)


def test_stale_writer_and_existing_nonresume_are_refused(tmp_path):
    path = tmp_path / "attempts.jsonl"
    writer = api().SearchLedger(path, manifest())
    stale = api().SearchLedger(path, manifest(), resume=True)
    append(writer, "one")
    with pytest.raises(ValueError):
        append(stale, "one")
    with pytest.raises(FileExistsError):
        api().SearchLedger(path, manifest())


def test_exhausted_search_records_unmet_requirement(tmp_path):
    ledger = api().SearchLedger(tmp_path / "attempts.jsonl", manifest())
    for case in manifest().eligible_case_ids:
        append(ledger, case)
    assert ledger.summary()["eligible_pool_exhausted"] is True
    assert ledger.summary()["demo_requirement_met"] is False
    assert ledger.summary()["first_top1_index"] is None
    with pytest.raises(ValueError):
        ledger.select_case("one", reason="not a winner", evidence_paths=["fixture"])


def test_rejected_top1_and_export_replay_attempts_remain_visible(tmp_path):
    ledger = api().SearchLedger(tmp_path / "attempts.jsonl", manifest())
    append(ledger, "one", root="a")
    ledger.reject_case("one", reason="No observable propagation evidence",
                       evidence_paths=["fixture/absence.json"])
    append(ledger, "two", root="a")
    ledger.select_case("two", reason="Observable synthetic evidence",
                       evidence_paths=["fixture/evidence.json"])
    append(ledger, "two", root="c", stage="export_replay", repeat_of=2)
    summary = ledger.summary()
    assert summary["first_top1_case_id"] == "one"
    assert summary["selected_case_id"] == "two"
    assert summary["export_replay_attempts"] == 1
    assert summary["demo_requirement_met"] is False
    assert summary["transfer_authorized"] is False
    records = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert any(row["payload"].get("event") == "rejected" for row in records)


def test_later_top1_cannot_bypass_unreviewed_first_top1(tmp_path):
    ledger = api().SearchLedger(tmp_path / "attempts.jsonl", manifest())
    append(ledger, "one", root="a")
    append(ledger, "two", root="a")
    with pytest.raises(ValueError, match="earlier"):
        ledger.select_case("two", reason="Unsupported cherry-picking", evidence_paths=["fixture"])


def test_resume_keeps_selection_rejection_and_replay_events(tmp_path):
    path = tmp_path / "attempts.jsonl"
    ledger = api().SearchLedger(path, manifest())
    append(ledger, "one", root="a")
    ledger.reject_case("one", reason="Synthetic evidence missing", evidence_paths=["fixture"])
    append(ledger, "two", root="a")
    ledger.select_case("two", reason="Synthetic evidence available", evidence_paths=["fixture"])
    append(ledger, "two", root="a", stage="export_replay", repeat_of=2)
    resumed = api().SearchLedger(path, manifest(), resume=True, expected_head_hash=ledger.head_hash)
    assert resumed.summary() == ledger.summary()
    before = path.read_bytes()
    csv_path, markdown = resumed.export_reports(tmp_path / "reports")
    assert path.read_bytes() == before
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["selection_status"] == "rejected"
    assert rows[0]["rejection_reason"] == "Synthetic evidence missing"
    assert '"a":1' in markdown.read_text(encoding="utf-8")


def test_partial_line_and_invalid_json_do_not_resume(tmp_path):
    path = tmp_path / "attempts.jsonl"
    ledger = api().SearchLedger(path, manifest())
    before = path.read_bytes()
    for tail in (b'{"unfinished":', b'not-json\n'):
        path.write_bytes(before + tail)
        with pytest.raises(ValueError):
            api().SearchLedger(path, manifest(), resume=True)
