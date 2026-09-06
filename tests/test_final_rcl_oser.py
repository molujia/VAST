from __future__ import annotations

import hashlib
import inspect
import json

import pytest

torch = pytest.importorskip("torch")

import rcl_study.final_rcl_oser as final_oser
from rcl_study.final_rcl_oser import (
    FinalOSERValidationError,
    build_final_oser_family_episodes,
    fit_final_oser,
    score_final_oser,
)


FEATURES = ("metric_signal", "propagation_signal", "context_signal")
OSER_SHA = "ef4f2c0229e8549cc2b85d33bf6478d10870606929cf156a4e99a60d70eb581a"


def _oser_profile() -> dict[str, object]:
    return {
        "profile_id": "oser-p02",
        "profile_sha256": OSER_SHA,
        "state_width": 32,
        "residual_cap": 0.05,
        "gate_threshold": 0.5,
        "lambda_meta": 0.5,
        "inner_updates": 1,
        "inner_learning_rate": 0.05,
        "training_steps": 30,
        "base_training_steps": 50,
    }


def _real_rows(case_ids: list[str]) -> tuple[list[dict[str, object]], dict[str, tuple[str, ...]]]:
    rows: list[dict[str, object]] = []
    targets: dict[str, tuple[str, ...]] = {}
    for case_index, case_id in enumerate(case_ids):
        positive = f"svc-{case_index % 3}"
        targets[case_id] = (positive,)
        for candidate_index in range(3):
            candidate = f"svc-{candidate_index}"
            signal = float(candidate == positive)
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": candidate,
                    "metric_signal": signal + case_index * 0.01,
                    "propagation_signal": signal * 0.8 - candidate_index * 0.01,
                    "context_signal": signal * 0.6 + candidate_index * 0.02,
                }
            )
    return rows, targets


def _synthetic(source: str, target: str, character: str, weight: float) -> dict[str, object]:
    return {
        "synthetic_row_sha256": character * 64,
        "source_case_id": source,
        "target_service_id": target,
        "query_budget_cost": 0,
        "final_training_weight": weight,
        "candidate_rows": [
            {
                "candidate_id": f"svc-{candidate_index}",
                "metric_signal": float(f"svc-{candidate_index}" == target) + 0.1,
                "propagation_signal": float(f"svc-{candidate_index}" == target) * 0.8,
                "context_signal": candidate_index * 0.05,
            }
            for candidate_index in range(3)
        ],
    }


def _labels() -> list[dict[str, str]]:
    return [
        {"case_id": "real-a", "fault_type": "type-a"},
        {"case_id": "real-b", "fault_type": "type-b"},
        {"case_id": "real-c", "fault_type": "type-c"},
    ]


def _synthetics() -> list[dict[str, object]]:
    return [
        _synthetic("real-a", "svc-1", "a", 0.25),
        _synthetic("real-a", "svc-2", "b", 0.75),
        _synthetic("real-b", "svc-0", "c", 1.0),
    ]


def test_family_episodes_keep_outer_real_only_and_parent_families_isolated() -> None:
    artifact = build_final_oser_family_episodes(
        real_label_records=_labels(),
        supervised_case_ids=["real-a", "real-b", "real-c"],
        synthetic_rows=_synthetics(),
        training_mode="oracle_full",
        supervised_case_limit=3,
    )

    hidden_a = next(row for row in artifact["episodes"] if row["query_fault_type"] == "type-a")
    assert hidden_a["outer_case_keys"] == ("real-a",)
    assert set(hidden_a["support_case_keys"]) == {
        "real-b",
        "real-c",
        "synthetic:" + "c" * 64,
    }
    assert set(hidden_a["quarantined_hidden_descendant_keys"]) == {
        "synthetic:" + "a" * 64,
        "synthetic:" + "b" * 64,
    }
    assert artifact["case_weights"]["real-a"] == 1.0
    assert artifact["case_weights"]["synthetic:" + "a" * 64] == 0.25
    assert artifact["case_weights"]["synthetic:" + "b" * 64] == 0.75
    assert artifact["parent_family_isolation_audit"]["split_family_count"] == 0


def test_supervision_cardinality_is_exact_for_budget_and_oracle_modes() -> None:
    query_ids = [f"query-{index:02d}" for index in range(30)]
    query = build_final_oser_family_episodes(
        real_label_records=[
            {"case_id": case_id, "fault_type": f"type-{index % 3}"}
            for index, case_id in enumerate(query_ids)
        ],
        supervised_case_ids=query_ids,
        synthetic_rows=[],
        training_mode="query_only",
        supervised_case_limit=30,
    )
    assert query["supervised_real_case_count"] == 30
    assert query["supervised_case_limit"] == 30

    with pytest.raises(FinalOSERValidationError, match="cardinality"):
        build_final_oser_family_episodes(
            real_label_records=_labels(),
            supervised_case_ids=["real-a", "real-b", "real-c"],
            synthetic_rows=[],
            training_mode="oracle_full",
            supervised_case_limit=4,
        )


def _state_case(case_index: int, *, supported: bool = True) -> dict[str, object]:
    return {
        "states": [
            [float(case_index), float(candidate), 0.2, 0.4]
            for candidate in range(3)
        ],
        "base_scores": [0.8, 0.4, 0.1],
        "evidence_supported": [supported, supported, supported],
        "positive_indices": (case_index % 3,),
        "candidate_ids": ("svc-0", "svc-1", "svc-2"),
    }


def _base_artifact() -> dict[str, object]:
    identity = {
        "schema_version": "conservative-lofo-base-score-artifact-v1",
        "dataset_id": "rcabench",
        "artifact_role": "final_test",
        "backend_id": "pairwise_linear",
        "query_plan_sha256": "d" * 64,
        "model_sha256": "f" * 64,
        "feature_order_sha256": "e" * 64,
        "feature_columns": FEATURES,
        "candidate_ids_by_case": {"test": ("svc-0", "svc-1", "svc-2")},
        "targets_by_case": {"test": ("svc-0",)},
        "scores_by_case": {"test": {"svc-0": 0.8, "svc-1": 0.4, "svc-2": 0.1}},
        "rankings_by_case": {"test": ["svc-0", "svc-1", "svc-2"]},
    }
    payload = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        **identity,
        "score_artifact_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def test_oser_training_uses_family_support_and_missing_evidence_is_exact_noop() -> None:
    episodes = build_final_oser_family_episodes(
        real_label_records=_labels(),
        supervised_case_ids=["real-a", "real-b", "real-c"],
        synthetic_rows=_synthetics(),
        training_mode="oracle_full",
        supervised_case_limit=3,
    )
    training_cases = {
        case_key: _state_case(index)
        for index, case_key in enumerate(episodes["training_case_keys"])
    }
    fitted = fit_final_oser(
        training_cases=training_cases,
        episode_artifact=episodes,
        profile=_oser_profile(),
        state_transform_sha256="e" * 64,
        seed=42,
    )
    assert fitted.audit["objective_step_count"] == 30
    assert fitted.audit["gradient_l1"] > 0.0
    assert fitted.audit["parent_family_isolation"]["split_family_count"] == 0
    assert fitted.audit["outer_query_synthetic_count"] == 0

    base = _base_artifact()
    no_evidence = {"test": _state_case(0, supported=False)}
    scored = score_final_oser(
        fitted=fitted,
        inference_cases=no_evidence,
        base_score_artifact=base,
        artifact_role="final_test",
    ).to_dict()
    assert scored["final_scores_by_case"] == scored["base_scores_by_case"]
    assert scored["no_op_case_count"] == 1


def test_final_oser_implementation_does_not_import_or_invoke_legacy_outer_router() -> None:
    source = inspect.getsource(final_oser)
    forbidden = "residual" + "_only"
    assert forbidden not in source
