from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.run_final_rcl_formal as formal_runner
from scripts.run_final_rcl_formal import build_formal_runtime_units


ROOT = Path(__file__).resolve().parents[1]


def test_formal_cvae_stage_binds_the_proxy_partition_contract_hash() -> None:
    assert formal_runner._proxy_partition_stage_sha256(
        {
            "proxy_partition_sha256": "a" * 64,
            "partition_sha256": "b" * 64,
        }
    ) == "a" * 64


def test_formal_runtime_units_use_full_outer_split_and_frozen_query_budget() -> None:
    registry = {
        "units": (
            {"unit_id": "d--hdbscan_query_cvae_oser--seed42", "dataset_id": "d", "arm_id": "hdbscan_query_cvae_oser"},
            {"unit_id": "d--oracle_full_cvae_oser--seed42", "dataset_id": "d", "arm_id": "oracle_full_cvae_oser"},
            {"unit_id": "d--hdbscan_query_pairwise--seed42", "dataset_id": "d", "arm_id": "hdbscan_query_pairwise"},
        )
    }
    prepared = {
        "d": {
            "split": {
                "outer_train_case_ids": ("a", "b", "c", "d"),
                "outer_test_case_ids": ("e", "f"),
                "query_case_ids": ("b", "d"),
            }
        }
    }

    units = build_formal_runtime_units(
        registry=registry,
        prepared_by_dataset=prepared,
        cvae_optimizer_steps=150,
    )

    assert [unit["fit_case_ids"] for unit in units] == [("a", "b", "c", "d")] * 3
    assert units[0]["supervised_case_ids"] == ("b", "d")
    assert units[1]["supervised_case_ids"] == ("a", "b", "c", "d")
    assert units[2]["supervised_case_ids"] == ("b", "d")
    assert [unit["evaluation_case_ids"] for unit in units] == [("e", "f")] * 3
    assert all(unit["cvae_optimizer_steps"] == 150 for unit in units)


def test_formal_cli_exposes_preflight_and_resumable_execution_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_final_rcl_formal.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--run-root",
        "--tested-closure",
        "--rcabench-feature-dir",
        "--aiops22-feature-dir",
        "--rcabench-inventory",
        "--aiops22-inventory",
        "--torch-python",
        "--tmux-session",
        "--preflight-only",
    ):
        assert flag in completed.stdout
