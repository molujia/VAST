import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from rcl_study.final_rcl_contract import build_formal_registry, load_final_method_config
from rcl_study.final_rcl_execution import (
    FINAL_STAGE_KINDS,
    FinalRCLExecutionError,
    build_final_execution_manifest,
    build_stage_spec,
    finalize_final_run,
    read_final_run_status,
    run_final_execution,
    run_or_reuse_stage,
    validate_completed_stage,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_rcl/hdbscan_proxy_cvae_oser_seed42.json"


@pytest.mark.parametrize("stage_kind", FINAL_STAGE_KINDS)
def test_completed_stage_manifest_is_content_addressed_and_validated(
    tmp_path: Path, stage_kind: str
) -> None:
    spec = build_stage_spec(
        stage_kind=stage_kind,
        owner_id="rcabench--hdbscan_query_cvae_oser--seed42",
        input_identity={"dataset_id": "rcabench", "value": 1},
        dependency_manifest_sha256s=("a" * 64,),
        code_sha256="b" * 64,
        config_sha256="c" * 64,
        resource_class="gpu" if stage_kind in {"cvae_neural_pool", "oser_checkpoint"} else "cpu",
    )
    stage_dir = tmp_path / stage_kind

    def producer(work_dir: Path):
        output = work_dir / "payload.json"
        output.write_text(json.dumps({"stage": stage_kind}), encoding="utf-8")
        return {"payload": output}

    result = run_or_reuse_stage(stage_dir=stage_dir, expected_spec=spec, producer=producer)
    audit = validate_completed_stage(stage_dir=stage_dir, expected_spec=spec)

    assert result["decision"] == "computed"
    assert audit["valid"] is True
    assert audit["stage_kind"] == stage_kind
    assert Path(audit["success_marker_path"]).is_file()


def test_restart_reuses_valid_stage_and_recomputes_only_corrupt_output(tmp_path: Path) -> None:
    spec = build_stage_spec(
        stage_kind="cvae_neural_pool",
        owner_id="unit-a",
        input_identity={"profile": "balanced"},
        dependency_manifest_sha256s=(),
        code_sha256="b" * 64,
        config_sha256="c" * 64,
        resource_class="gpu",
    )
    stage_dir = tmp_path / "stage"
    calls = Counter()

    def producer(work_dir: Path):
        calls["count"] += 1
        output = work_dir / "pool.json"
        output.write_text(json.dumps({"attempt": calls["count"]}), encoding="utf-8")
        return {"pool": output}

    first = run_or_reuse_stage(stage_dir=stage_dir, expected_spec=spec, producer=producer)
    second = run_or_reuse_stage(stage_dir=stage_dir, expected_spec=spec, producer=producer)
    (stage_dir / "pool.json").write_text("corrupt", encoding="utf-8")
    third = run_or_reuse_stage(stage_dir=stage_dir, expected_spec=spec, producer=producer)

    assert first["decision"] == "computed"
    assert second["decision"] == "reused"
    assert third["decision"] == "recomputed"
    assert calls["count"] == 2
    assert list((stage_dir / "reuse_rejections").glob("*.json"))


def _execution_manifest(tmp_path: Path):
    config = load_final_method_config(CONFIG)
    registry = build_formal_registry(config)
    return build_final_execution_manifest(
        registry=registry,
        run_root=tmp_path / "formal-run",
        tmux_session="final-rcl-seed42-test",
        code_sha256="d" * 64,
        config_sha256="e" * 64,
        input_sha256="f" * 64,
        max_cpu_workers=2,
        max_gpu_workers=1,
    )


def test_six_unit_orchestrator_uses_unique_ownership_and_reuses_all_aggregates(
    tmp_path: Path,
) -> None:
    manifest = _execution_manifest(tmp_path)
    calls = Counter()

    def worker(unit, unit_dir: Path, log_path: Path):
        calls[unit["unit_id"]] += 1
        log_path.write_text("worker completed\n", encoding="utf-8")
        spec = build_stage_spec(
            stage_kind="unit_aggregate",
            owner_id=unit["unit_id"],
            input_identity={"unit": unit["unit_id"]},
            dependency_manifest_sha256s=(),
            code_sha256="d" * 64,
            config_sha256="e" * 64,
            resource_class=unit["resource_class"],
        )

        def produce(stage_dir: Path):
            output = stage_dir / "aggregate.json"
            output.write_text(json.dumps({"unit_id": unit["unit_id"]}), encoding="utf-8")
            return {"aggregate": output}

        return run_or_reuse_stage(
            stage_dir=unit_dir / "unit_aggregate",
            expected_spec=spec,
            producer=produce,
        )

    first = run_final_execution(execution_manifest=manifest, unit_worker=worker)
    assert first["completed"] == 6
    assert first["failed"] == 0
    assert Path(first["all_done_path"]).is_file()
    assert len({unit["unit_output_root"] for unit in manifest["units"]}) == 6
    assert {unit["resource_class"] for unit in manifest["units"]} == {"cpu", "gpu"}

    second = run_final_execution(execution_manifest=manifest, unit_worker=worker)
    assert second["completed"] == 6
    assert sum(calls.values()) == 6
    assert second["reused_unit_count"] == 6


def test_execution_manifest_reopens_the_same_json_identity(tmp_path: Path) -> None:
    first = _execution_manifest(tmp_path)
    config = load_final_method_config(CONFIG)
    registry = build_formal_registry(config)

    second = build_final_execution_manifest(
        registry=registry,
        run_root=tmp_path / "formal-run",
        tmux_session="final-rcl-seed42-test",
        code_sha256="d" * 64,
        config_sha256="e" * 64,
        input_sha256="f" * 64,
        max_cpu_workers=2,
        max_gpu_workers=1,
    )

    assert second == first


def test_failure_is_atomic_and_all_done_cannot_precede_six_valid_aggregates(
    tmp_path: Path,
) -> None:
    manifest = _execution_manifest(tmp_path)

    def worker(unit, unit_dir: Path, log_path: Path):
        if unit["unit_id"].startswith("aiops2022_pre--oracle"):
            raise RuntimeError("intentional failure")
        spec = build_stage_spec(
            stage_kind="unit_aggregate",
            owner_id=unit["unit_id"],
            input_identity={"unit": unit["unit_id"]},
            dependency_manifest_sha256s=(),
            code_sha256="d" * 64,
            config_sha256="e" * 64,
            resource_class=unit["resource_class"],
        )

        def produce(stage_dir: Path):
            output = stage_dir / "aggregate.json"
            output.write_text("{}", encoding="utf-8")
            return {"aggregate": output}

        return run_or_reuse_stage(
            stage_dir=unit_dir / "unit_aggregate",
            expected_spec=spec,
            producer=produce,
        )

    status = run_final_execution(execution_manifest=manifest, unit_worker=worker)

    assert status["completed"] == 5
    assert status["failed"] == 1
    assert not Path(status["all_done_path"]).exists()
    assert status["recent_failures"][0]["error_type"] == "RuntimeError"
    with pytest.raises(FinalRCLExecutionError, match="six validated"):
        finalize_final_run(Path(manifest["run_root"]))


def test_status_reports_durable_counts_eta_tmux_and_expected_marker(tmp_path: Path) -> None:
    manifest = _execution_manifest(tmp_path)
    status = read_final_run_status(
        run_root=Path(manifest["run_root"]),
        tmux_alive=True,
    )

    assert status["expected"] == 6
    assert status["completed"] == 0
    assert status["running"] == 0
    assert status["failed"] == 0
    assert status["pending"] == 6
    assert status["tmux_session"] == "final-rcl-seed42-test"
    assert status["tmux_alive"] is True
    assert status["estimated_remaining_seconds"] is None
    assert status["all_done_path"].endswith("all.done")


def test_status_command_emits_machine_readable_progress(tmp_path: Path) -> None:
    manifest = _execution_manifest(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/status_final_rcl.py"),
            "--run-root",
            manifest["run_root"],
            "--tmux-alive",
            "true",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["expected"] == 6
    assert status["pending"] == 6
    assert status["tmux_alive"] is True


def test_required_aggregate_report_gates_all_done(tmp_path: Path) -> None:
    config = load_final_method_config(CONFIG)
    registry = build_formal_registry(config)
    manifest = build_final_execution_manifest(
        registry=registry,
        run_root=tmp_path / "formal-with-aggregate",
        tmux_session="final-rcl-seed42-aggregate-test",
        code_sha256="d" * 64,
        config_sha256="e" * 64,
        input_sha256="f" * 64,
        max_cpu_workers=2,
        max_gpu_workers=1,
        require_aggregate_report=True,
    )

    def worker(unit, unit_dir: Path, log_path: Path):
        spec = build_stage_spec(
            stage_kind="unit_aggregate",
            owner_id=unit["unit_id"],
            input_identity={"unit": unit["unit_id"]},
            dependency_manifest_sha256s=(),
            code_sha256="d" * 64,
            config_sha256="e" * 64,
            resource_class=unit["resource_class"],
        )

        def produce(stage_dir: Path):
            output = stage_dir / "aggregate.json"
            output.write_text("{}", encoding="utf-8")
            return {"aggregate": output}

        return run_or_reuse_stage(
            stage_dir=unit_dir / "unit_aggregate",
            expected_spec=spec,
            producer=produce,
        )

    status = run_final_execution(
        execution_manifest=manifest,
        unit_worker=worker,
        auto_finalize=False,
    )

    assert status["completed"] == 6
    assert not Path(status["all_done_path"]).exists()
    with pytest.raises(FinalRCLExecutionError, match="aggregate report"):
        finalize_final_run(Path(manifest["run_root"]))

    aggregate_identity = {
        "schema_version": "final-rcl-matched-comparison-v1",
        "seed": 42,
        "unit_count": 6,
        "datasets": {"rcabench": {}, "aiops2022_pre": {}},
    }
    import hashlib

    aggregate_sha = hashlib.sha256(
        json.dumps(
            aggregate_identity,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    aggregate_path = Path(manifest["aggregate_report_path"])
    aggregate_path.write_text(
        json.dumps({**aggregate_identity, "aggregate_sha256": aggregate_sha}),
        encoding="utf-8",
    )

    marker = finalize_final_run(Path(manifest["run_root"]))

    assert marker["aggregate_report_sha256"] == aggregate_sha
    assert Path(status["all_done_path"]).is_file()
