from __future__ import annotations

from copy import deepcopy

import pytest

from rcl_study.final_rcl_training import (
    FinalRCLTrainingError,
    adapt_generated_candidate_case,
    build_final_synthetic_training_ledger,
)


def _real(case_ids: list[str], *, cost: int) -> list[dict[str, object]]:
    return [
        {"case_id": case_id, "query_budget_cost": cost, "training_weight": 1.0}
        for case_id in case_ids
    ]


def _synthetics(case_ids: list[str]) -> list[dict[str, object]]:
    rows = []
    for source_index, source in enumerate(case_ids):
        for child_index, provisional in enumerate((1.0, 3.0)):
            rows.append(
                {
                    "source_case_id": source,
                    "target_service_id": f"service-{child_index}",
                    "query_budget_cost": 0,
                    "provisional_training_weight": provisional,
                    "synthetic_row_sha256": format(
                        source_index * 2 + child_index + 1, "064x"
                    ),
                    "candidate_rows": [
                        {"candidate_id": "service-0", "f0": 1.0},
                        {"candidate_id": "service-1", "f0": 0.0},
                    ],
                }
            )
    return rows


def test_query_only_ledger_preserves_budget_and_normalizes_each_parent() -> None:
    case_ids = [f"q{index:02d}" for index in range(30)]
    ledger = build_final_synthetic_training_ledger(
        real_supervision_rows=_real(case_ids, cost=1),
        synthetic_rows=_synthetics(case_ids),
        training_mode="query_only",
        expected_supervised_case_ids=case_ids,
        query_budget=30,
    )

    assert ledger["training_mode"] == "query_only"
    assert ledger["real_supervised_count"] == 30
    assert ledger["real_query_budget_cost"] == 30
    assert ledger["synthetic_query_budget_cost"] == 0
    assert set(ledger["synthetic_weight_sum_by_source"].values()) == {1.0}
    assert [row["final_training_weight"] for row in ledger["synthetic_rows"][:2]] == [
        0.25,
        0.75,
    ]


def test_oracle_full_ledger_accepts_all_training_cases_without_query_cost() -> None:
    case_ids = ["train-a", "train-b", "train-c"]
    ledger = build_final_synthetic_training_ledger(
        real_supervision_rows=_real(case_ids, cost=0),
        synthetic_rows=_synthetics(case_ids),
        training_mode="oracle_full",
        expected_supervised_case_ids=case_ids,
        query_budget=30,
    )

    assert ledger["training_mode"] == "oracle_full"
    assert ledger["real_supervised_count"] == 3
    assert ledger["supervised_case_limit"] == 3
    assert ledger["real_query_budget_cost"] == 0
    assert set(ledger["synthetic_weight_sum_by_source"].values()) == {1.0}


def test_supervision_ledger_rejects_mode_cost_and_membership_drift() -> None:
    case_ids = ["a", "b"]
    with pytest.raises(FinalRCLTrainingError, match="oracle_full.*query cost"):
        build_final_synthetic_training_ledger(
            real_supervision_rows=_real(case_ids, cost=1),
            synthetic_rows=_synthetics(case_ids),
            training_mode="oracle_full",
            expected_supervised_case_ids=case_ids,
        )
    with pytest.raises(FinalRCLTrainingError, match="membership"):
        build_final_synthetic_training_ledger(
            real_supervision_rows=_real(case_ids, cost=0),
            synthetic_rows=_synthetics(case_ids),
            training_mode="oracle_full",
            expected_supervised_case_ids=["a", "missing"],
        )


def test_generated_state_adapter_is_candidate_complete_and_dimension_agnostic() -> None:
    generated = {
        "source_case_id": "case-a",
        "target_service_id": "svc-c",
        "synthetic_row_sha256": "a" * 64,
        "provisional_training_weight": 0.4,
        "decoded_mechanism": [0.2, -0.1],
        "decoded_propagation": [0.3],
        "decoded_context": [0.4],
    }
    source_case = {
        "candidate_ids": ["svc-a", "svc-b", "svc-c"],
        "base_feature_rows": [
            [0.9, 0.8, 0.7, 0.6],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ],
    }
    adapted = adapt_generated_candidate_case(
        generated_row=generated,
        source_case=source_case,
        source_root_service_id="svc-a",
        feature_names=["f0", "f1", "identity", "context"],
        feature_lower_bounds=[0.0, 0.0, 0.0, 0.0],
        feature_upper_bounds=[1.0, 1.0, 1.0, 1.0],
        symptom_feature_indices=[0, 1],
        identity_context_indices=[2, 3],
    )

    assert [row["candidate_id"] for row in adapted["candidate_rows"]] == [
        "svc-a",
        "svc-b",
        "svc-c",
    ]
    assert all(set(row) == {"candidate_id", "f0", "f1", "identity", "context"} for row in adapted["candidate_rows"])
    target = adapted["candidate_rows"][2]
    assert target["identity"] == 0.4
    assert target["context"] == 0.5
    assert adapted["generator_row_sha256"] == "a" * 64
    assert adapted["synthetic_row_sha256"] != "a" * 64


def test_generated_state_adapter_rejects_target_outside_candidate_set() -> None:
    generated = {
        "source_case_id": "case-a",
        "target_service_id": "missing",
        "synthetic_row_sha256": "a" * 64,
        "provisional_training_weight": 1.0,
        "decoded_mechanism": [0.2],
        "decoded_propagation": [0.3],
        "decoded_context": [0.4],
    }
    with pytest.raises(FinalRCLTrainingError, match="target"):
        adapt_generated_candidate_case(
            generated_row=generated,
            source_case={
                "candidate_ids": ["svc-a", "svc-b"],
                "base_feature_rows": [[0.0, 0.0], [1.0, 1.0]],
            },
            source_root_service_id="svc-a",
            feature_names=["f0", "f1"],
            feature_lower_bounds=[0.0, 0.0],
            feature_upper_bounds=[1.0, 1.0],
            symptom_feature_indices=[0],
            identity_context_indices=[1],
        )
