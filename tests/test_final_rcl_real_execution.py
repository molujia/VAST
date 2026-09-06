from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from rcl_study.final_rcl_real_execution import (
    FinalRCLRealExecutionError,
    build_all_candidate_compatible_target_plan,
    build_final_oser_handshake_request,
    build_final_smoke_manifest,
    validate_final_smoke_report,
)


ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)


def _registry() -> dict[str, object]:
    units = []
    for dataset_id in ("rcabench", "aiops2022_pre"):
        for arm_id in ARMS:
            units.append(
                {
                    "unit_id": f"{dataset_id}--{arm_id}--seed42",
                    "dataset_id": dataset_id,
                    "arm_id": arm_id,
                    "seed": 42,
                }
            )
    return {
        "schema_version": "final-rcl-formal-registry-v1",
        "seed": 42,
        "clusterer_id": "hdbscan",
        "outer_router_enabled": False,
        "unit_count": 6,
        "units": units,
        "registry_sha256": "a" * 64,
    }


def _split(dataset_id: str) -> dict[str, object]:
    train = tuple(f"{dataset_id}-train-{index:03d}" for index in range(40))
    test = tuple(f"{dataset_id}-test-{index:03d}" for index in range(5))
    query = train[5:35]
    return {
        "schema_version": "final-rcl-fixed-split-adapter-v1",
        "dataset_id": dataset_id,
        "split_seed": 42,
        "outer_train_case_ids": train,
        "outer_test_case_ids": test,
        "outer_train_case_count": len(train),
        "outer_test_case_count": len(test),
        "fit_test_overlap_count": 0,
        "inventory_sha256": "b" * 64,
        "candidate_pool_sha256": "c" * 64,
        "hdbscan_candidate_order": train,
        "query_case_ids": query,
        "query_plan_sha256": "d" * 64,
        "hdbscan_partition_sha256": "e" * 64,
        "hdbscan_representation_sha256": "f" * 64,
        "split_sha256": ("1" if dataset_id == "rcabench" else "2") * 64,
    }


def test_smoke_manifest_has_six_bounded_real_branches_and_is_not_promotable(
    tmp_path: Path,
) -> None:
    manifest = build_final_smoke_manifest(
        registry=_registry(),
        split_adapters={dataset: _split(dataset) for dataset in ("rcabench", "aiops2022_pre")},
        run_root=tmp_path / "smoke",
        oracle_train_limit=30,
        test_case_limit=2,
        cvae_optimizer_steps=2,
    )

    assert manifest["schema_version"] == "final-rcl-six-branch-smoke-v1"
    assert manifest["evidence_role"] == "smoke_only"
    assert manifest["promotable_to_formal"] is False
    assert manifest["unit_count"] == 6
    assert {(row["dataset_id"], row["arm_id"]) for row in manifest["units"]} == {
        (dataset, arm)
        for dataset in ("rcabench", "aiops2022_pre")
        for arm in ARMS
    }
    for unit in manifest["units"]:
        split = _split(unit["dataset_id"])
        assert unit["evaluation_case_ids"] == split["outer_test_case_ids"][:2]
        assert set(unit["fit_case_ids"]).isdisjoint(unit["evaluation_case_ids"])
        assert unit["cvae_optimizer_steps"] == 2
        if unit["arm_id"] == "oracle_full_cvae_oser":
            assert unit["training_mode"] == "oracle_full"
            assert len(unit["fit_case_ids"]) == 30
            assert unit["supervised_case_ids"] == unit["fit_case_ids"]
        else:
            assert unit["training_mode"] == "query_only"
            assert unit["supervised_case_ids"] == split["query_case_ids"]


def _state_record(candidate_id: str, offset: float, *, propagation: bool = True):
    return {
        "case_id": "case-a",
        "candidate_id": candidate_id,
        "state": {
            "mechanism": [1.0 + offset, 0.2],
            "propagation": [0.3 + offset, 0.4],
            "context": [0.5, 0.6 + offset],
        },
        "mask": {
            "mechanism": [1.0, 1.0],
            "propagation": [1.0 if propagation else 0.0, 1.0 if propagation else 0.0],
            "context": [1.0, 1.0],
        },
    }


def test_target_plan_records_a_hard_compatibility_decision_for_every_candidate() -> None:
    plan = build_all_candidate_compatible_target_plan(
        source_case_id="case-a",
        source_root_service_id="svc-a",
        candidate_records=(
            _state_record("svc-a", 0.0),
            _state_record("svc-b", 0.1),
            _state_record("svc-c", 0.2, propagation=False),
        ),
    )

    assert [row["target_service_id"] for row in plan] == ["svc-a", "svc-b", "svc-c"]
    assert [row["hard_eligible"] for row in plan] == [True, True, False]
    assert plan[2]["hard_rejection_reasons"] == ["missing_target_propagation"]
    assert sum(row["target_weight"] for row in plan if row["hard_eligible"]) == pytest.approx(1.0)


def _result(unit: dict[str, object]) -> dict[str, object]:
    complete = unit["arm_id"] != "hdbscan_query_pairwise"
    return {
        "unit_id": unit["unit_id"],
        "dataset_id": unit["dataset_id"],
        "arm_id": unit["arm_id"],
        "evidence_role": "smoke_only",
        "status": "completed",
        "fit_test_overlap_count": 0,
        "candidate_complete_rankings": True,
        "finite_rankings": True,
        "training_mode": unit["training_mode"],
        "real_training_case_count": len(unit["supervised_case_ids"]),
        "evaluation_case_count": len(unit["evaluation_case_ids"]),
        "oser_status": "active" if complete else "not_applicable",
    }


def test_smoke_report_requires_all_branches_and_rejects_formal_promotion(
    tmp_path: Path,
) -> None:
    manifest = build_final_smoke_manifest(
        registry=_registry(),
        split_adapters={dataset: _split(dataset) for dataset in ("rcabench", "aiops2022_pre")},
        run_root=tmp_path / "smoke",
        oracle_train_limit=30,
        test_case_limit=1,
        cvae_optimizer_steps=2,
    )
    report = validate_final_smoke_report(
        smoke_manifest=manifest,
        unit_results=[_result(unit) for unit in manifest["units"]],
    )

    assert report["status"] == "passed"
    assert report["passed_unit_count"] == 6
    assert report["evidence_role"] == "smoke_only"
    assert report["promotable_to_formal"] is False

    attacked = [_result(unit) for unit in manifest["units"]]
    attacked[0] = deepcopy(attacked[0])
    attacked[0]["evidence_role"] = "formal"
    with pytest.raises(FinalRCLRealExecutionError, match="smoke"):
        validate_final_smoke_report(smoke_manifest=manifest, unit_results=attacked)


def test_oser_handshake_is_hash_sealed_and_keeps_real_outer_queries() -> None:
    request = build_final_oser_handshake_request(
        dataset_id="rcabench",
        training_mode="query_only",
        supervised_case_ids=("case-a", "case-b"),
        real_label_records=(
            {"case_id": "case-a", "fault_type": "type-a"},
            {"case_id": "case-b", "fault_type": "type-b"},
        ),
        synthetic_family_rows=(
            {
                "synthetic_row_sha256": "a" * 64,
                "source_case_id": "case-a",
                "target_service_id": "svc-b",
                "query_budget_cost": 0,
                "final_training_weight": 1.0,
            },
        ),
        training_cases={
            "case-a": {"states": [[1.0]], "base_scores": [0.5], "positive_indices": [0]},
            "case-b": {"states": [[0.5]], "base_scores": [0.4], "positive_indices": [0]},
            "synthetic:" + "a" * 64: {
                "states": [[0.8]],
                "base_scores": [0.6],
                "positive_indices": [0],
            },
        },
        inference_cases={
            "test-a": {"states": [[0.2]], "base_scores": [0.1], "positive_indices": [0]}
        },
        base_score_artifact={"schema_version": "conservative-lofo-base-score-artifact-v1"},
        profile={"profile_id": "oser-p02"},
        state_transform_sha256="b" * 64,
        artifact_role="bounded_real_smoke",
    )

    assert request["schema_version"] == "final-rcl-oser-handshake-request-v1"
    assert request["supervised_case_limit"] == 2
    assert request["outer_query_population"] == "real_cases_only"
    assert request["request_sha256"]
