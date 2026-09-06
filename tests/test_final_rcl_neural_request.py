from __future__ import annotations

from copy import deepcopy

import pytest

from rcl_study.final_rcl_hdbscan_proxy import build_hdbscan_proxy_partition
from rcl_study.final_rcl_training import FinalRCLTrainingError, build_final_neural_request


def _partition(supervised: list[str]) -> dict[str, object]:
    return build_hdbscan_proxy_partition(
        dataset_id="rcabench",
        fit_case_ids=["case-a", "case-b"],
        queried_case_ids=supervised,
        held_out_case_ids=["test-a"],
        raw_labels=[0, 0],
        membership_strengths=[1.0, 0.8],
        representation_matrix_sha256="1" * 64,
        geometry_sha256="2" * 64,
        active_partition_sha256="3" * 64,
        min_cluster_size=5,
        min_samples=5,
        cluster_selection_method="leaf",
    )


def _records() -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "state": {
                "mechanism": [float(case_index + candidate_index), 0.2],
                "propagation": [0.1, 0.3],
                "context": [0.4, float(candidate_index)],
            },
            "mask": {
                "mechanism": [1.0, 1.0],
                "propagation": [1.0, 1.0],
                "context": [1.0, 1.0],
            },
        }
        for case_index, case_id in enumerate(("case-a", "case-b"))
        for candidate_index, candidate_id in enumerate(("svc-a", "svc-b"))
    ]


def _profile() -> dict[str, object]:
    return {
        "profile_id": "balanced",
        "hidden_width": 128,
        "mechanism_latent_width": 12,
        "propagation_latent_width": 12,
        "context_latent_width": 12,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "masked_reconstruction_weight": 1.0,
        "kl_weight": 0.0005,
        "target_context_weight": 0.75,
        "cycle_consistency_weight": 0.75,
        "sampling_radius": 0.5,
    }


def _targets() -> list[dict[str, object]]:
    return [
        {
            "target_service_id": "svc-a",
            "target_weight": 0.5,
            "compatibility": 1.0,
            "hard_eligible": True,
            "hard_rejection_reasons": [],
            "target_weight_plan_sha256": "4" * 64,
            "compatibility_sha256": "5" * 64,
        },
        {
            "target_service_id": "svc-b",
            "target_weight": 0.5,
            "compatibility": 0.8,
            "hard_eligible": True,
            "hard_rejection_reasons": [],
            "target_weight_plan_sha256": "4" * 64,
            "compatibility_sha256": "6" * 64,
        },
    ]


def _sources(case_ids: list[str], mode: str) -> list[dict[str, object]]:
    return [
        {
            "source_case_id": case_id,
            "source_root_service_id": "svc-a",
            "queried_label": {
                "root_cause": "svc-a",
                "fault_type": f"type-{index}",
                "label_source": "queried_budget" if mode == "query_only" else "oracle_full",
                "budget_cost": 1 if mode == "query_only" else 0,
            },
            "target_plans": {"proxy_mode_cvae_compatible": _targets()},
        }
        for index, case_id in enumerate(case_ids)
    ]


@pytest.mark.parametrize(
    ("mode", "supervised"),
    [("query_only", ["case-a"]), ("oracle_full", ["case-a", "case-b"])],
)
def test_final_v3_neural_request_accepts_both_supervision_modes(
    mode: str, supervised: list[str]
) -> None:
    request = build_final_neural_request(
        dataset_id="rcabench",
        training_mode=mode,
        fit_case_ids=["case-a", "case-b"],
        supervised_case_ids=supervised,
        held_out_case_ids=["test-a"],
        expected_candidates_per_case=2,
        pretraining_records=_records(),
        source_cases=_sources(supervised, mode),
        proxy_mode_partition=_partition(supervised),
        profile=_profile(),
        optimizer_steps=2,
        samples_per_target=1,
        device="cpu",
    )

    assert request["schema_version"] == "service-continuous-neural-handshake-request-v3"
    assert request["dataset_id"] == "rcabench"
    assert request["training_mode"] == mode
    assert request["supervised_case_ids"] == supervised


@pytest.mark.parametrize(
    ("mode", "supervised"),
    [("query_only", ["case-a"]), ("oracle_full", ["case-a", "case-b"])],
)
def test_torch_neural_runner_accepts_final_v3_request(
    mode: str, supervised: list[str]
) -> None:
    pytest.importorskip("torch")
    from rcl_study.service_continuous_neural_runner import (
        execute_neural_handshake,
    )

    request = build_final_neural_request(
        dataset_id="rcabench",
        training_mode=mode,
        fit_case_ids=["case-a", "case-b"],
        supervised_case_ids=supervised,
        held_out_case_ids=["test-a"],
        expected_candidates_per_case=2,
        pretraining_records=_records(),
        source_cases=_sources(supervised, mode),
        proxy_mode_partition=_partition(supervised),
        profile=_profile(),
        optimizer_steps=16,
        samples_per_target=1,
        device="cpu",
    )
    result = execute_neural_handshake(request)

    assert result["schema_version"] == "service-continuous-neural-handshake-result-v3"
    assert result["request_sha256"] == request["request_sha256"]
    assert set(result["arms"]) == {"proxy_mode_cvae_compatible"}
    rows = result["arms"]["proxy_mode_cvae_compatible"]
    assert len(rows) == 2 * len(supervised)
    expected_transfer_source = (
        "queried_budget_transfer"
        if mode == "query_only"
        else "oracle_full_transfer"
    )
    assert {row["synthetic_label"]["label_source"] for row in rows} == {
        expected_transfer_source
    }
    assert {row["query_budget_cost"] for row in rows} == {0}
    assert {row["proxy_mode_partition_sha256"] for row in rows} == {
        request["proxy_mode_partition"]["proxy_partition_sha256"]
    }
    assert result["activity_audit"]["source_case_count"] == len(supervised)
    assert result["activity_audit"]["proxy_mode_activity"][
        "active_partition_sha256"
    ] == "3" * 64


def test_final_v3_neural_request_rejects_dbscan_partition() -> None:
    partition = _partition(["case-a"])
    partition["algorithm"] = "DBSCAN"
    with pytest.raises(ValueError, match="HDBSCAN"):
        build_final_neural_request(
            dataset_id="rcabench",
            training_mode="query_only",
            fit_case_ids=["case-a", "case-b"],
            supervised_case_ids=["case-a"],
            held_out_case_ids=["test-a"],
            expected_candidates_per_case=2,
            pretraining_records=_records(),
            source_cases=_sources(["case-a"], "query_only"),
            proxy_mode_partition=partition,
            profile=_profile(),
            optimizer_steps=2,
            samples_per_target=1,
            device="cpu",
        )


def test_final_v3_neural_request_requires_a_decision_for_every_candidate_target() -> None:
    sources = _sources(["case-a"], "query_only")
    sources[0]["target_plans"]["proxy_mode_cvae_compatible"] = _targets()[:1]

    with pytest.raises(FinalRCLTrainingError, match="candidate.*target|target.*candidate"):
        build_final_neural_request(
            dataset_id="rcabench",
            training_mode="query_only",
            fit_case_ids=["case-a", "case-b"],
            supervised_case_ids=["case-a"],
            held_out_case_ids=["test-a"],
            expected_candidates_per_case=2,
            pretraining_records=_records(),
            source_cases=sources,
            proxy_mode_partition=_partition(["case-a"]),
            profile=_profile(),
            optimizer_steps=2,
            samples_per_target=1,
            device="cpu",
        )
