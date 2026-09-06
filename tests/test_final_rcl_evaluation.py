from copy import deepcopy

import pytest

from rcl_study.final_rcl_evaluation import (
    FinalRCLEvaluationError,
    aggregate_final_matched_comparisons,
    aggregate_final_rank_metrics,
    build_final_arm_contract,
    build_final_ranking_artifact,
    build_final_unit_provenance,
    build_fixed_split_adapter,
    validate_matched_arm_contracts,
)


ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)


def _split_inputs(dataset_id: str = "rcabench"):
    train = [f"{dataset_id}-train-{index:02d}" for index in range(32)]
    test = [f"{dataset_id}-test-{index:02d}" for index in range(2)]
    inventory = {
        "schema_version": "ordinary-query-only-inventory-v1",
        "canonical_dataset_id": dataset_id,
        "split_seed": 42,
        "records": [
            {
                "case_id": case_id,
                "split": "outer_train",
                "fault_type": f"type-{index % 3}",
                "incident_id": f"incident-{index}",
            }
            for index, case_id in enumerate(train)
        ]
        + [
            {
                "case_id": case_id,
                "split": "outer_test",
                "fault_type": f"test-type-{index}",
                "incident_id": f"test-incident-{index}",
            }
            for index, case_id in enumerate(test)
        ],
        "sha256": "a" * 64,
    }
    pool = {
        "candidate_population": "full_observable_outer_train_not_sampled",
        "case_count": len(train),
        "case_ids": train,
        "sha256": "b" * 64,
    }
    plan = {
        "dataset_id": dataset_id,
        "active_learning_seed": 42,
        "budget": 30,
        "clusterer_id": "hdbscan",
        "selector_id": "center",
        "selected_case_ids": train[:30],
        "plan_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
    }
    return train, test, inventory, pool, plan


def _split(dataset_id: str = "rcabench"):
    _train, _test, inventory, pool, plan = _split_inputs(dataset_id)
    return build_fixed_split_adapter(
        expected_dataset_id=dataset_id,
        inventory_manifest=inventory,
        candidate_pool=pool,
        query_plan=plan,
    )


def _arm_contract(arm_id: str = "hdbscan_query_cvae_oser", dataset_id: str = "rcabench"):
    return build_final_arm_contract(split_adapter=_split(dataset_id), arm_id=arm_id)


def _score_artifact(arm_id: str):
    supervision_sha256 = _arm_contract(arm_id)["supervision_authority_sha256"]
    candidates = {
        "rcabench-test-00": ("svc-a", "svc-b", "svc-c"),
        "rcabench-test-01": ("svc-a", "svc-b", "svc-c"),
    }
    rankings = {
        "rcabench-test-00": ["svc-a", "svc-b", "svc-c"],
        "rcabench-test-01": ["svc-c", "svc-a", "svc-b"],
    }
    scores = {
        case_id: {candidate: float(3 - index) for index, candidate in enumerate(order)}
        for case_id, order in rankings.items()
    }
    if arm_id == "hdbscan_query_pairwise":
        return {
            "schema_version": "conservative-lofo-base-score-artifact-v1",
            "dataset_id": "rcabench",
            "artifact_role": "final_test",
            "backend_id": "pairwise_linear",
            "query_plan_sha256": supervision_sha256,
            "model_sha256": "e" * 64,
            "score_artifact_sha256": "f" * 64,
            "candidate_ids_by_case": candidates,
            "scores_by_case": scores,
            "rankings_by_case": rankings,
        }
    return {
        "schema_version": "conservative-lofo-residual-output-v1",
        "artifact_role": "final_test",
        "arm": "oser_meta",
        "query_plan_sha256": supervision_sha256,
        "checkpoint_sha256": "1" * 64,
        "artifact_sha256": "2" * 64,
        "candidate_ids_by_case": candidates,
        "final_scores_by_case": scores,
        "rankings_by_case": rankings,
    }


def _ranking(arm_id: str = "hdbscan_query_cvae_oser"):
    expected = {
        "rcabench-test-00": ("svc-a", "svc-b", "svc-c"),
        "rcabench-test-01": ("svc-a", "svc-b", "svc-c"),
    }
    return build_final_ranking_artifact(
        arm_contract=_arm_contract(arm_id),
        score_artifact=_score_artifact(arm_id),
        expected_candidates_by_case=expected,
    )


def _metrics(arm_id: str = "hdbscan_query_cvae_oser"):
    return aggregate_final_rank_metrics(
        ranking_artifact=_ranking(arm_id),
        targets_by_case={
            "rcabench-test-00": ("svc-a",),
            "rcabench-test-01": ("svc-b",),
        },
    )


def test_fixed_split_adapter_and_all_arms_share_exact_memberships() -> None:
    train, test, _inventory, _pool, _plan = _split_inputs()
    split = _split()
    contracts = [build_final_arm_contract(split_adapter=split, arm_id=arm) for arm in ARMS]
    audit = validate_matched_arm_contracts(contracts)

    assert split["outer_train_case_ids"] == tuple(train)
    assert split["outer_test_case_ids"] == tuple(test)
    assert contracts[0]["real_training_case_ids"] == tuple(train[:30])
    assert contracts[1]["real_training_case_ids"] == tuple(train)
    assert contracts[2]["real_training_case_ids"] == tuple(train[:30])
    assert audit["valid"] is True
    assert audit["fit_test_overlap_count"] == 0
    assert audit["shared_outer_test_case_count"] == 2


def test_split_adapter_rejects_fit_test_overlap_and_non_hdbscan_authority() -> None:
    _train, _test, inventory, pool, plan = _split_inputs()
    attacked = deepcopy(inventory)
    attacked["records"].append(
        {
            "case_id": attacked["records"][-1]["case_id"],
            "split": "outer_train",
            "fault_type": "bad",
            "incident_id": "bad",
        }
    )
    with pytest.raises(FinalRCLEvaluationError, match="unique|overlap"):
        build_fixed_split_adapter(
            expected_dataset_id="rcabench",
            inventory_manifest=attacked,
            candidate_pool=pool,
            query_plan=plan,
        )

    plan["clusterer_id"] = "dbscan"
    with pytest.raises(FinalRCLEvaluationError, match="HDBSCAN"):
        build_fixed_split_adapter(
            expected_dataset_id="rcabench",
            inventory_manifest=inventory,
            candidate_pool=pool,
            query_plan=plan,
        )


@pytest.mark.parametrize("arm_id", ARMS)
def test_all_three_arms_use_one_candidate_complete_ranking_path(arm_id: str) -> None:
    artifact = _ranking(arm_id)

    assert artifact["arm_id"] == arm_id
    assert artifact["case_ids"] == ("rcabench-test-00", "rcabench-test-01")
    assert len(artifact["ranking_rows"]) == 2
    assert artifact["ranking_rows"][0]["ranking"] == (
        "svc-a",
        "svc-b",
        "svc-c",
    )


def test_candidate_complete_ranking_rejects_missing_or_duplicate_candidates() -> None:
    score = _score_artifact("hdbscan_query_pairwise")
    score["rankings_by_case"]["rcabench-test-00"] = ["svc-a", "svc-a", "svc-c"]
    with pytest.raises(FinalRCLEvaluationError, match="candidate"):
        build_final_ranking_artifact(
            arm_contract=_arm_contract("hdbscan_query_pairwise"),
            score_artifact=score,
            expected_candidates_by_case={
                "rcabench-test-00": ("svc-a", "svc-b", "svc-c"),
                "rcabench-test-01": ("svc-a", "svc-b", "svc-c"),
            },
        )


def test_metrics_are_derived_from_integer_counts_and_first_valid_ranks() -> None:
    metrics = _metrics()

    assert metrics["denominator"] == 2
    assert metrics["hit_at_1_count"] == 1
    assert metrics["hit_at_3_count"] == 2
    assert metrics["hit_at_5_count"] == 2
    assert metrics["hit_at_1"] == 0.5
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["top135"] == pytest.approx(5.0 / 6.0)
    assert metrics["reciprocal_rank_sum"] == pytest.approx(4.0 / 3.0)
    assert metrics["mrr"] == pytest.approx(2.0 / 3.0)
    assert metrics["case_ranks"] == {
        "rcabench-test-00": 1,
        "rcabench-test-01": 3,
    }


def test_metrics_reject_a_target_absent_from_candidate_set() -> None:
    with pytest.raises(FinalRCLEvaluationError, match="true root"):
        aggregate_final_rank_metrics(
            ranking_artifact=_ranking(),
            targets_by_case={
                "rcabench-test-00": ("svc-a",),
                "rcabench-test-01": ("svc-missing",),
            },
        )


def _dependencies(arm_id: str):
    contract = _arm_contract(arm_id)
    common = {
        "split": {"sha256": contract["split_sha256"]},
        "hdbscan_partition": {"sha256": "d" * 64},
    }
    if arm_id != "oracle_full_cvae_oser":
        common["query_plan"] = {"sha256": "c" * 64}
    if arm_id != "hdbscan_query_pairwise":
        common.update(
            {
                "cvae_ledger": {"sha256": "4" * 64},
                "cvae_checkpoint": {"sha256": "5" * 64},
                "oser_checkpoint": {"sha256": "6" * 64},
                "oser_activity": {"sha256": "7" * 64},
            }
        )
    return common


@pytest.mark.parametrize("arm_id", ARMS)
def test_unit_provenance_requires_exact_arm_specific_dependencies(arm_id: str) -> None:
    provenance = build_final_unit_provenance(
        arm_contract=_arm_contract(arm_id),
        ranking_artifact=_ranking(arm_id),
        metrics=_metrics(arm_id),
        dependencies=_dependencies(arm_id),
        code_config_closure={
            "code_sha256": "8" * 64,
            "config_sha256": "9" * 64,
            "input_sha256": "0" * 64,
        },
    )

    assert provenance["arm_id"] == arm_id
    assert provenance["ranking_sha256"] == _ranking(arm_id)["ranking_sha256"]
    assert provenance["fit_test_overlap_count"] == 0
    assert ("cvae_checkpoint" in provenance["dependencies"]) == (
        arm_id != "hdbscan_query_pairwise"
    )


def test_matched_comparisons_require_six_units_and_compute_declared_deltas() -> None:
    units = []
    values = {
        "hdbscan_query_cvae_oser": (0.7, 0.8),
        "oracle_full_cvae_oser": (0.8, 0.9),
        "hdbscan_query_pairwise": (0.6, 0.7),
    }
    for dataset_id in ("rcabench", "aiops2022_pre"):
        for arm_id in ARMS:
            top135, mrr = values[arm_id]
            units.append(
                {
                    "dataset_id": dataset_id,
                    "arm_id": arm_id,
                    "seed": 42,
                    "split_sha256": ("a" if dataset_id == "rcabench" else "b") * 64,
                    "metrics": {
                        "denominator": 10,
                        "hit_at_1": top135 - 0.1,
                        "hit_at_3": top135,
                        "hit_at_5": top135 + 0.1,
                        "top135": top135,
                        "mrr": mrr,
                    },
                    "provenance_sha256": "f" * 64,
                }
            )

    aggregate = aggregate_final_matched_comparisons(units)

    assert aggregate["unit_count"] == 6
    rcabench = aggregate["datasets"]["rcabench"]
    assert rcabench["complete_minus_hdbscan_only"]["top135"] == pytest.approx(0.1)
    assert rcabench["complete_minus_hdbscan_only"]["mrr"] == pytest.approx(0.1)
    assert rcabench["query_only_minus_oracle_full"]["top135"] == pytest.approx(-0.1)
    assert rcabench["query_only_minus_oracle_full"]["mrr"] == pytest.approx(-0.1)

    with pytest.raises(FinalRCLEvaluationError, match="six"):
        aggregate_final_matched_comparisons(units[:-1])
