from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rcl_study.final_rcl_oser_runner import (
    FinalRCLOSERHandshakeError,
    validate_final_oser_handshake_request,
)
from rcl_study.final_rcl_real_execution import build_final_oser_handshake_request


def _request() -> dict[str, object]:
    return build_final_oser_handshake_request(
        dataset_id="rcabench",
        training_mode="query_only",
        supervised_case_ids=("case-a", "case-b"),
        real_label_records=(
            {"case_id": "case-a", "fault_type": "type-a"},
            {"case_id": "case-b", "fault_type": "type-b"},
        ),
        synthetic_family_rows=(),
        training_cases={
            "case-a": {"states": [[1.0]], "base_scores": [0.5], "positive_indices": [0]},
            "case-b": {"states": [[0.5]], "base_scores": [0.4], "positive_indices": [0]},
        },
        inference_cases={
            "test-a": {"states": [[0.2]], "base_scores": [0.1], "positive_indices": [0]}
        },
        base_score_artifact={"schema_version": "conservative-lofo-base-score-artifact-v1"},
        profile={"profile_id": "oser-p02"},
        state_transform_sha256="b" * 64,
        artifact_role="bounded_real_smoke",
    )


def test_oser_runner_rejects_any_mutation_after_hash_sealing() -> None:
    request = _request()
    assert validate_final_oser_handshake_request(request)["seed"] == 42

    attacked = deepcopy(request)
    attacked["outer_query_population"] = "synthetic_allowed"
    with pytest.raises(FinalRCLOSERHandshakeError, match="identity|real"):
        validate_final_oser_handshake_request(attacked)


def test_oser_runner_restores_declared_family_order_after_sorted_json_wire() -> None:
    synthetic_hash = "1" * 64
    request = build_final_oser_handshake_request(
        dataset_id="rcabench",
        training_mode="query_only",
        supervised_case_ids=("case-z", "case-a"),
        real_label_records=(
            {"case_id": "case-z", "fault_type": "type-z"},
            {"case_id": "case-a", "fault_type": "type-a"},
        ),
        synthetic_family_rows=(
            {
                "synthetic_row_sha256": synthetic_hash,
                "source_case_id": "case-z",
                "target_service_id": "svc-a",
                "query_budget_cost": 0,
                "final_training_weight": 1.0,
            },
        ),
        training_cases={
            "case-z": {"states": [[1.0]], "base_scores": [0.5], "positive_indices": [0]},
            "case-a": {"states": [[0.5]], "base_scores": [0.4], "positive_indices": [0]},
            "synthetic:" + synthetic_hash: {
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
    wire_value = json.loads(json.dumps(request, sort_keys=True))

    validated = validate_final_oser_handshake_request(wire_value)

    assert tuple(validated["training_cases"]) == (
        "case-z",
        "case-a",
        "synthetic:" + synthetic_hash,
    )
