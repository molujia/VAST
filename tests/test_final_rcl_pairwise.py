from __future__ import annotations

from rcl_study.final_rcl_pairwise import fit_final_augmented_pairwise_base


FEATURES = ("metric_signal", "propagation_signal", "context_signal")


def _real_rows() -> tuple[list[dict[str, object]], dict[str, tuple[str, ...]]]:
    rows: list[dict[str, object]] = []
    targets: dict[str, tuple[str, ...]] = {}
    for case_index, case_id in enumerate(("real-a", "real-b", "real-c")):
        positive = f"svc-{case_index}"
        targets[case_id] = (positive,)
        for candidate_index in range(3):
            candidate = f"svc-{candidate_index}"
            signal = float(candidate == positive)
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": candidate,
                    "metric_signal": signal + case_index * 0.01,
                    "propagation_signal": signal * 0.8,
                    "context_signal": candidate_index * 0.02,
                }
            )
    return rows, targets


def _synthetic() -> dict[str, object]:
    return {
        "synthetic_row_sha256": "a" * 64,
        "source_case_id": "real-a",
        "target_service_id": "svc-1",
        "query_budget_cost": 0,
        "final_training_weight": 1.0,
        "candidate_rows": [
            {
                "candidate_id": f"svc-{candidate_index}",
                "metric_signal": float(candidate_index == 1) + 0.1,
                "propagation_signal": float(candidate_index == 1) * 0.8,
                "context_signal": candidate_index * 0.05,
            }
            for candidate_index in range(3)
        ],
    }


def test_oracle_full_augmented_pairwise_accepts_non_budget_cardinality_and_weights() -> None:
    rows, targets = _real_rows()
    fitted = fit_final_augmented_pairwise_base(
        dataset_id="rcabench",
        training_mode="oracle_full",
        supervised_case_limit=3,
        real_training_rows=rows,
        real_targets_by_case=targets,
        synthetic_rows=[_synthetic()],
        feature_columns=FEATURES,
        supervision_authority_sha256="d" * 64,
        generation_disposition={
            "generated_count": 1,
            "dropped_count": 0,
            "invalid_count": 0,
        },
    )

    assert fitted.training_audit["training_mode"] == "oracle_full"
    assert fitted.training_audit["supervised_case_limit"] == 3
    assert fitted.training_audit["effective_real_case_weight"] == 3.0
    assert fitted.training_audit["effective_synthetic_case_weight"] == 1.0
