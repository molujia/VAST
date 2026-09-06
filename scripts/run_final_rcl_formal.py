#!/usr/bin/env python3
"""Run the resumable six-unit formal HDBSCAN + CVAE + OSER study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rcl_study.final_rcl_contract import (  # noqa: E402
    build_formal_registry,
    load_final_method_config,
    validate_formal_registry,
)
from rcl_study.final_rcl_evaluation import (  # noqa: E402
    aggregate_final_matched_comparisons,
    aggregate_final_rank_metrics,
    build_final_arm_contract,
    build_final_ranking_artifact,
    build_final_unit_provenance,
)
from rcl_study.final_rcl_execution import (  # noqa: E402
    build_final_execution_manifest,
    build_stage_spec,
    finalize_final_run,
    read_final_run_status,
    run_final_execution,
    run_or_reuse_stage,
    validate_completed_stage,
)
from scripts.run_final_rcl_real_smoke import (  # noqa: E402
    CLOSURE_CODE_RELATIVE_PATHS,
    _attach_base_scores,
    _factorized_records,
    _feature_alias_root,
    _file_hash,
    _prepare_dataset,
    _proxy_partition,
    _read,
    _rows_for_cases,
    _run_torch_worker,
    _semantic_hash,
    _state_materialization,
    _synthetic_rows_for_scoring,
    _synthetic_state_cases,
    _targets,
    _transfer_feature_indices,
    _write,
)


FORMAL_CODE_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (*CLOSURE_CODE_RELATIVE_PATHS, "rcl_study/final_rcl_execution.py", "rcl_study/final_rcl_evaluation.py", "scripts/run_final_rcl_formal.py", "scripts/status_final_rcl.py")
    )
)


def build_formal_runtime_units(
    *,
    registry: Mapping[str, Any],
    prepared_by_dataset: Mapping[str, Mapping[str, Any]],
    cvae_optimizer_steps: int,
) -> tuple[dict[str, Any], ...]:
    """Bind each frozen arm to the complete outer split used by formal scoring."""

    steps = int(cvae_optimizer_steps)
    if isinstance(cvae_optimizer_steps, bool) or steps <= 0:
        raise ValueError("formal CVAE optimizer steps must be positive")
    result = []
    for raw in registry.get("units", ()):
        unit = dict(raw)
        dataset_id = str(unit["dataset_id"])
        split = dict(prepared_by_dataset[dataset_id]["split"])
        train = tuple(str(value) for value in split["outer_train_case_ids"])
        test = tuple(str(value) for value in split["outer_test_case_ids"])
        query = tuple(str(value) for value in split["query_case_ids"])
        oracle = str(unit["arm_id"]) == "oracle_full_cvae_oser"
        result.append(
            {
                **unit,
                "fit_case_ids": train,
                "supervised_case_ids": train if oracle else query,
                "evaluation_case_ids": test,
                "training_mode": "oracle_full" if oracle else "query_only",
                "cvae_optimizer_steps": steps,
            }
        )
    return tuple(result)


def _formal_code_closure() -> tuple[dict[str, str], str]:
    hashes = {
        relative: _file_hash(REPO_ROOT / relative)
        for relative in FORMAL_CODE_RELATIVE_PATHS
    }
    return hashes, _semantic_hash(hashes)


def _proxy_partition_stage_sha256(partition: Mapping[str, Any]) -> str:
    value = str(partition.get("proxy_partition_sha256", "")).strip()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("formal CVAE requires the HDBSCAN proxy partition SHA-256")
    return value


def _validate_tested_closure(path: Path, config_path: Path) -> dict[str, Any]:
    closure = _read(path)
    supplied = str(closure.get("closure_sha256", ""))
    identity = {key: value for key, value in closure.items() if key != "closure_sha256"}
    if (
        closure.get("schema_version") != "final-rcl-tested-closure-v1"
        or supplied != _semantic_hash(identity)
        or closure.get("config_sha256") != _file_hash(config_path)
    ):
        raise ValueError("tested smoke closure identity drifted")
    for relative, expected in dict(closure.get("code_file_sha256s", {})).items():
        if _file_hash(REPO_ROOT / relative) != expected:
            raise ValueError(f"tested smoke code drifted: {relative}")
    smoke_report = _read(path.parent / "smoke-report.json")
    report_sha = str(smoke_report.get("smoke_report_sha256", ""))
    report_identity = {
        key: value for key, value in smoke_report.items() if key != "smoke_report_sha256"
    }
    if (
        smoke_report.get("status") != "passed"
        or smoke_report.get("passed_unit_count") != 6
        or smoke_report.get("evidence_role") != "smoke_only"
        or smoke_report.get("promotable_to_formal") is not False
        or report_sha != _semantic_hash(report_identity)
        or closure.get("smoke_report_sha256") != report_sha
    ):
        raise ValueError("six-branch smoke report is not a valid formal gate")
    return closure


def _output_path(stage_audit: Mapping[str, Any], name: str) -> Path:
    outputs = [dict(value) for value in stage_audit["outputs"]]
    matches = [value for value in outputs if value["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"stage output is absent or duplicate: {name}")
    return Path(stage_audit["manifest_path"]).parent / matches[0]["relative_path"]


def _append_log(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, **fields},
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _runtime_identity(
    unit: Mapping[str, Any], prepared: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "unit_id": unit["unit_id"],
        "dataset_id": unit["dataset_id"],
        "arm_id": unit["arm_id"],
        "training_mode": unit["training_mode"],
        "fit_case_ids": tuple(unit["fit_case_ids"]),
        "supervised_case_ids": tuple(unit["supervised_case_ids"]),
        "evaluation_case_ids": tuple(unit["evaluation_case_ids"]),
        "split_sha256": prepared["split"]["split_sha256"],
        "query_plan_sha256": prepared["query_plan"]["plan_sha256"],
        "candidate_count": int(prepared["candidate_count"]),
        "cvae_optimizer_steps": int(unit["cvae_optimizer_steps"]),
    }


def _execute_runtime_unit(
    *,
    unit: Mapping[str, Any],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    torch_python: Path,
    unit_root: Path,
    log_path: Path,
    code_sha256: str,
    config_sha256: str,
    input_sha256: str,
) -> dict[str, Any]:
    import numpy as np

    from rcl_study.conservative_lofo_base_bridge import score_base_score_bridge
    from rcl_study.final_rcl_pairwise import fit_final_augmented_pairwise_base
    from rcl_study.final_rcl_real_execution import (
        build_all_candidate_compatible_target_plan,
        build_final_oser_handshake_request,
    )
    from rcl_study.final_rcl_training import (
        adapt_generated_candidate_case,
        build_final_neural_request,
        build_final_synthetic_training_ledger,
    )
    from rcl_study.service_continuous_pairwise_bridge import (
        fit_service_continuity_pairwise_bridge,
    )

    dataset_id = str(unit["dataset_id"])
    arm_id = str(unit["arm_id"])
    complete = arm_id != "hdbscan_query_pairwise"
    dataset = prepared["dataset"]
    fit_ids = tuple(str(value) for value in unit["fit_case_ids"])
    supervised_ids = tuple(str(value) for value in unit["supervised_case_ids"])
    evaluation_ids = tuple(str(value) for value in unit["evaluation_case_ids"])
    feature_names = tuple(str(value) for value in dataset["base_feature_names"])
    symptom_indices, identity_indices = _transfer_feature_indices(feature_names)
    real_rows = _rows_for_cases(dataset, supervised_ids)
    real_targets = _targets(dataset, supervised_ids)
    evaluation_rows = _rows_for_cases(dataset, evaluation_ids)
    evaluation_targets = _targets(dataset, evaluation_ids)
    arm_contract = build_final_arm_contract(
        split_adapter=prepared["split"], arm_id=arm_id
    )
    if tuple(arm_contract["real_training_case_ids"]) != supervised_ids:
        raise ValueError("formal runtime supervision drifted from arm contract")
    runtime_identity = _runtime_identity(unit, prepared)
    supervision_sha = str(arm_contract["supervision_authority_sha256"])

    representation_spec = build_stage_spec(
        stage_kind="representation_hdbscan",
        owner_id=str(unit["unit_id"]),
        input_identity={
            **runtime_identity,
            "representation_matrix_sha256": config["active_learning"]["datasets"][dataset_id]["representation_matrix_sha256"],
            "geometry_sha256": config["active_learning"]["datasets"][dataset_id]["geometry_sha256"],
            "partition_sha256": config["active_learning"]["datasets"][dataset_id]["partition_sha256"],
        },
        dependency_manifest_sha256s=(),
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        resource_class="cpu",
    )

    def produce_partition(stage_dir: Path) -> Mapping[str, Path]:
        partition_path = stage_dir / "proxy-partition.json"
        _write(
            partition_path,
            _proxy_partition(
                dataset_id=dataset_id,
                prepared=prepared,
                fit_ids=fit_ids,
                supervised_ids=supervised_ids,
            ),
        )
        return {"proxy_partition": partition_path}

    representation = run_or_reuse_stage(
        stage_dir=unit_root / "representation_hdbscan",
        expected_spec=representation_spec,
        producer=produce_partition,
    )
    _append_log(log_path, "stage_complete", stage="representation_hdbscan", decision=representation["decision"])
    partition = _read(_output_path(representation, "proxy_partition"))

    neural: dict[str, Any] | None = None
    cvae_stage: dict[str, Any] | None = None
    if complete:
        cvae_spec = build_stage_spec(
            stage_kind="cvae_neural_pool",
            owner_id=str(unit["unit_id"]),
            input_identity={
                **runtime_identity,
                "proxy_partition_sha256": _proxy_partition_stage_sha256(partition),
                "profile_sha256": config["cvae"]["generation_profile_sha256"],
                "samples_per_target": 1,
            },
            dependency_manifest_sha256s=(representation["manifest_sha256"],),
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            resource_class="gpu",
        )

        def produce_cvae(stage_dir: Path) -> Mapping[str, Path]:
            factorized = _factorized_records(dataset, fit_ids)
            by_case: dict[str, list[dict[str, Any]]] = {
                case_id: [] for case_id in fit_ids
            }
            for row in factorized:
                by_case[str(row["case_id"])].append(row)
            source_targets = _targets(dataset, supervised_ids)
            source_cases = []
            for case_id in supervised_ids:
                root = source_targets[case_id][0]
                label = dataset["labels_by_case"][case_id]
                source_cases.append(
                    {
                        "source_case_id": case_id,
                        "source_root_service_id": root,
                        "queried_label": {
                            "root_cause": str(label["root_cause"]),
                            "fault_type": str(label["fault_type"]),
                            "label_source": "queried_budget" if unit["training_mode"] == "query_only" else "oracle_full",
                            "budget_cost": 1 if unit["training_mode"] == "query_only" else 0,
                        },
                        "target_plans": {
                            "proxy_mode_cvae_compatible": build_all_candidate_compatible_target_plan(
                                source_case_id=case_id,
                                source_root_service_id=root,
                                candidate_records=by_case[case_id],
                            )
                        },
                    }
                )
            checkpoint = stage_dir / "cvae-checkpoint.pt"
            request = build_final_neural_request(
                dataset_id=dataset_id,
                training_mode=str(unit["training_mode"]),
                fit_case_ids=fit_ids,
                supervised_case_ids=supervised_ids,
                held_out_case_ids=evaluation_ids,
                expected_candidates_per_case=int(prepared["candidate_count"]),
                pretraining_records=factorized,
                source_cases=source_cases,
                proxy_mode_partition=partition,
                profile=dict(config["cvae"]),
                optimizer_steps=int(unit["cvae_optimizer_steps"]),
                samples_per_target=1,
                device="cuda",
                checkpoint_output_path=str(checkpoint.resolve()),
            )
            request_path = stage_dir / "cvae-request.json"
            result_path = stage_dir / "cvae-result.json"
            _write(request_path, request)
            _run_torch_worker(
                torch_python=torch_python,
                module="rcl_study.service_continuous_neural_runner",
                request=request_path,
                output=result_path,
            )
            return {
                "cvae_request": request_path,
                "cvae_result": result_path,
                "cvae_checkpoint": checkpoint,
            }

        cvae_stage = run_or_reuse_stage(
            stage_dir=unit_root / "cvae_neural_pool",
            expected_spec=cvae_spec,
            producer=produce_cvae,
        )
        _append_log(log_path, "stage_complete", stage="cvae_neural_pool", decision=cvae_stage["decision"])
        neural = _read(_output_path(cvae_stage, "cvae_result"))

    base_dependencies = (
        (cvae_stage["manifest_sha256"],) if cvae_stage is not None else (representation["manifest_sha256"],)
    )
    base_spec = build_stage_spec(
        stage_kind="augmented_base_ranker",
        owner_id=str(unit["unit_id"]),
        input_identity={
            **runtime_identity,
            "augmentation": "proxy_mode_cvae_compatible" if complete else "none",
            "supervision_authority_sha256": supervision_sha,
        },
        dependency_manifest_sha256s=base_dependencies,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        resource_class="cpu",
    )

    def produce_base(stage_dir: Path) -> Mapping[str, Path]:
        if not complete:
            fitted = fit_service_continuity_pairwise_bridge(
                dataset_id=dataset_id,
                arm_id="baseline",
                augmentation_enabled=False,
                real_training_rows=real_rows,
                real_targets_by_case=real_targets,
                synthetic_rows=(),
                feature_columns=feature_names,
                query_plan_sha256=supervision_sha,
                random_state=42,
                generation_disposition={"generated_count": 0, "dropped_count": 0, "invalid_count": 0},
                query_budget=len(supervised_ids),
            )
            score = score_base_score_bridge(
                fitted.bridge, evaluation_rows, evaluation_targets, "final_test"
            )
            result_path = stage_dir / "base-result.json"
            _write(result_path, {"score_artifact": score, "training_audit": fitted.training_audit})
            return {"base_result": result_path}

        assert neural is not None
        generated = [
            dict(row) for row in neural["arms"]["proxy_mode_cvae_compatible"]
        ]
        source_targets = _targets(dataset, supervised_ids)
        bounds = np.asarray(
            [
                row
                for case_id in fit_ids
                for row in dataset["cases"][case_id]["base_feature_rows"]
            ],
            dtype=float,
        )
        lower = bounds.min(axis=0)
        upper = bounds.max(axis=0)
        adapted = [
            adapt_generated_candidate_case(
                generated_row=row,
                source_case=dataset["cases"][str(row["source_case_id"])],
                source_root_service_id=source_targets[str(row["source_case_id"])][0],
                feature_names=feature_names,
                feature_lower_bounds=lower,
                feature_upper_bounds=upper,
                symptom_feature_indices=symptom_indices,
                identity_context_indices=identity_indices,
            )
            for row in generated
        ]
        ledger = build_final_synthetic_training_ledger(
            real_supervision_rows=[
                {
                    "case_id": case_id,
                    "training_weight": 1.0,
                    "query_budget_cost": 1 if unit["training_mode"] == "query_only" else 0,
                }
                for case_id in supervised_ids
            ],
            synthetic_rows=adapted,
            training_mode=str(unit["training_mode"]),
            expected_supervised_case_ids=supervised_ids,
            query_budget=30,
        )
        synthetics = [dict(row) for row in ledger["synthetic_rows"]]
        fitted = fit_final_augmented_pairwise_base(
            dataset_id=dataset_id,
            training_mode=str(unit["training_mode"]),
            supervised_case_limit=len(supervised_ids),
            real_training_rows=real_rows,
            real_targets_by_case=real_targets,
            synthetic_rows=synthetics,
            feature_columns=feature_names,
            supervision_authority_sha256=supervision_sha,
            generation_disposition={"generated_count": len(synthetics), "dropped_count": 0, "invalid_count": 0},
        )
        base_evaluation = score_base_score_bridge(
            fitted.bridge, evaluation_rows, evaluation_targets, "final_test"
        )
        state_cases, state_transform_sha = _state_materialization(
            dataset, fit_ids, evaluation_ids
        )
        real_state = {case_id: state_cases[case_id] for case_id in supervised_ids}
        real_training_score = score_base_score_bridge(
            fitted.bridge, real_rows, real_targets, "oser_training_real"
        )
        real_state = _attach_base_scores(real_state, real_training_score)
        synthetic_rows_flat, synthetic_targets, synthetic_candidates = _synthetic_rows_for_scoring(synthetics)
        synthetic_score = score_base_score_bridge(
            fitted.bridge,
            synthetic_rows_flat,
            synthetic_targets,
            "oser_training_synthetic",
            expected_candidates_by_case=synthetic_candidates,
        )
        synthetic_state = _synthetic_state_cases(
            synthetics, real_state, synthetic_score
        )
        oser_request = build_final_oser_handshake_request(
            dataset_id=dataset_id,
            training_mode=str(unit["training_mode"]),
            supervised_case_ids=supervised_ids,
            real_label_records=[
                {"case_id": case_id, "fault_type": str(dataset["labels_by_case"][case_id]["fault_type"])}
                for case_id in supervised_ids
            ],
            synthetic_family_rows=[
                {
                    "synthetic_row_sha256": row["synthetic_row_sha256"],
                    "source_case_id": row["source_case_id"],
                    "target_service_id": row["target_service_id"],
                    "query_budget_cost": 0,
                    "final_training_weight": row["final_training_weight"],
                }
                for row in synthetics
            ],
            training_cases={**real_state, **synthetic_state},
            inference_cases=_attach_base_scores(
                {case_id: state_cases[case_id] for case_id in evaluation_ids},
                base_evaluation,
            ),
            base_score_artifact=base_evaluation,
            profile=dict(config["oser"]),
            state_transform_sha256=state_transform_sha,
            artifact_role="final_test",
        )
        request_path = stage_dir / "oser-request.json"
        audit_path = stage_dir / "base-audit.json"
        _write(request_path, oser_request)
        _write(
            audit_path,
            {
                "pairwise_training_audit": fitted.training_audit,
                "cvae_activity_audit": neural["activity_audit"],
                "synthetic_ledger_sha256": ledger["ledger_sha256"],
                "synthetic_row_count": len(synthetics),
            },
        )
        return {"oser_request": request_path, "base_audit": audit_path}

    base_stage = run_or_reuse_stage(
        stage_dir=unit_root / "augmented_base_ranker",
        expected_spec=base_spec,
        producer=produce_base,
    )
    _append_log(log_path, "stage_complete", stage="augmented_base_ranker", decision=base_stage["decision"])

    oser_stage: dict[str, Any] | None = None
    oser_result: dict[str, Any] | None = None
    if complete:
        oser_spec = build_stage_spec(
            stage_kind="oser_checkpoint",
            owner_id=str(unit["unit_id"]),
            input_identity={
                **runtime_identity,
                "profile_sha256": config["oser"]["profile_sha256"],
            },
            dependency_manifest_sha256s=(base_stage["manifest_sha256"],),
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            resource_class="gpu",
        )

        def produce_oser(stage_dir: Path) -> Mapping[str, Path]:
            result_path = stage_dir / "oser-result.json"
            _run_torch_worker(
                torch_python=torch_python,
                module="rcl_study.final_rcl_oser_runner",
                request=_output_path(base_stage, "oser_request"),
                output=result_path,
            )
            return {"oser_result": result_path}

        oser_stage = run_or_reuse_stage(
            stage_dir=unit_root / "oser_checkpoint",
            expected_spec=oser_spec,
            producer=produce_oser,
        )
        _append_log(log_path, "stage_complete", stage="oser_checkpoint", decision=oser_stage["decision"])
        oser_result = _read(_output_path(oser_stage, "oser_result"))
        score_artifact = dict(oser_result["score_artifact"])
        ranking_dependency = oser_stage
    else:
        base_result = _read(_output_path(base_stage, "base_result"))
        score_artifact = dict(base_result["score_artifact"])
        ranking_dependency = base_stage

    expected_candidates = {
        case_id: tuple(str(value) for value in dataset["cases"][case_id]["candidate_ids"])
        for case_id in evaluation_ids
    }
    ranking_spec = build_stage_spec(
        stage_kind="ranking",
        owner_id=str(unit["unit_id"]),
        input_identity={
            **runtime_identity,
            "score_artifact_sha256": score_artifact[
                "artifact_sha256" if complete else "score_artifact_sha256"
            ],
        },
        dependency_manifest_sha256s=(ranking_dependency["manifest_sha256"],),
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        resource_class="cpu",
    )

    def produce_ranking(stage_dir: Path) -> Mapping[str, Path]:
        ranking_path = stage_dir / "ranking.json"
        _write(
            ranking_path,
            build_final_ranking_artifact(
                arm_contract=arm_contract,
                score_artifact=score_artifact,
                expected_candidates_by_case=expected_candidates,
            ),
        )
        return {"ranking": ranking_path}

    ranking_stage = run_or_reuse_stage(
        stage_dir=unit_root / "ranking",
        expected_spec=ranking_spec,
        producer=produce_ranking,
    )
    _append_log(log_path, "stage_complete", stage="ranking", decision=ranking_stage["decision"])
    ranking = _read(_output_path(ranking_stage, "ranking"))
    metrics = aggregate_final_rank_metrics(
        ranking_artifact=ranking,
        targets_by_case=evaluation_targets,
    )
    dependencies = {
        "split": {"sha256": arm_contract["split_sha256"]},
        "hdbscan_partition": {"sha256": arm_contract["hdbscan_partition_sha256"]},
    }
    if arm_contract["query_plan_sha256"] is not None:
        dependencies["query_plan"] = {"sha256": arm_contract["query_plan_sha256"]}
    if complete:
        assert neural is not None and oser_result is not None
        base_audit = _read(_output_path(base_stage, "base_audit"))
        dependencies.update(
            {
                "cvae_ledger": {"sha256": base_audit["synthetic_ledger_sha256"]},
                "cvae_checkpoint": {"sha256": neural["activity_audit"]["checkpoint_sha256"]},
                "oser_checkpoint": {"sha256": oser_result["training_audit"]["checkpoint_sha256"]},
                "oser_activity": {"sha256": oser_result["result_sha256"]},
            }
        )
    provenance = build_final_unit_provenance(
        arm_contract=arm_contract,
        ranking_artifact=ranking,
        metrics=metrics,
        dependencies=dependencies,
        code_config_closure={
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
        },
    )
    aggregate_spec = build_stage_spec(
        stage_kind="unit_aggregate",
        owner_id=str(unit["unit_id"]),
        input_identity={
            **runtime_identity,
            "ranking_sha256": ranking["ranking_sha256"],
            "metrics_sha256": metrics["metrics_sha256"],
            "provenance_sha256": provenance["provenance_sha256"],
        },
        dependency_manifest_sha256s=(ranking_stage["manifest_sha256"],),
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        resource_class=str(unit["resource_class"]),
    )

    def produce_aggregate(stage_dir: Path) -> Mapping[str, Path]:
        output = stage_dir / "aggregate.json"
        payload = {
            "schema_version": "final-rcl-unit-aggregate-v1",
            "unit_id": unit["unit_id"],
            "dataset_id": dataset_id,
            "arm_id": arm_id,
            "seed": 42,
            "split_sha256": arm_contract["split_sha256"],
            "metrics": metrics,
            "provenance": provenance,
            "provenance_sha256": provenance["provenance_sha256"],
            "ranking_sha256": ranking["ranking_sha256"],
            "oser_status": (
                str(oser_result["oser_status"]) if oser_result is not None else "not_applicable"
            ),
        }
        _write(output, payload)
        return {"aggregate": output}

    aggregate_stage = run_or_reuse_stage(
        stage_dir=unit_root / "unit_aggregate",
        expected_spec=aggregate_spec,
        producer=produce_aggregate,
    )
    _append_log(log_path, "stage_complete", stage="unit_aggregate", decision=aggregate_stage["decision"])
    return aggregate_stage


def _aggregate_six_units(
    execution_manifest: Mapping[str, Any], run_root: Path
) -> dict[str, Any]:
    records = []
    for unit in execution_manifest["units"]:
        audit = validate_completed_stage(
            stage_dir=Path(unit["unit_output_root"]) / "unit_aggregate"
        )
        records.append(_read(_output_path(audit, "aggregate")))
    report = aggregate_final_matched_comparisons(records)
    _write(run_root / "aggregate-report.json", report)
    return report


def _gpu_inventory() -> tuple[str, ...]:
    completed = subprocess.run(
        ("nvidia-smi", "-L"), check=False, capture_output=True, text=True
    )
    values = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if completed.returncode != 0 or not values:
        raise ValueError("formal execution requires an available NVIDIA GPU")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tested-closure", required=True, type=Path)
    parser.add_argument("--rcabench-feature-dir", required=True, type=Path)
    parser.add_argument("--aiops22-feature-dir", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--rcabench-inventory", required=True, type=Path)
    parser.add_argument("--aiops22-inventory", required=True, type=Path)
    parser.add_argument("--torch-python", required=True, type=Path)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--cvae-optimizer-steps", type=int)
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--max-gpu-workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_root = args.run_root.resolve()
    config_path = REPO_ROOT / "configs/final_rcl/hdbscan_proxy_cvae_oser_seed42.json"
    config = load_final_method_config(config_path)
    registry = build_formal_registry(config)
    validate_formal_registry(registry, config)
    tested_closure = _validate_tested_closure(args.tested_closure.resolve(), config_path)
    gpu_inventory = _gpu_inventory()
    alias_root = _feature_alias_root(
        rcabench_feature_dir=args.rcabench_feature_dir.resolve(),
        aiops22_feature_dir=args.aiops22_feature_dir.resolve(),
        run_root=run_root,
    )
    inventories = {
        "rcabench": args.rcabench_inventory.resolve(),
        "aiops2022_pre": args.aiops22_inventory.resolve(),
    }
    prepared = {
        dataset_id: _prepare_dataset(
            dataset_id=dataset_id,
            package_root=args.package_root.resolve(),
            feature_root=alias_root,
            alias_root=alias_root,
            inventory_path=inventories[dataset_id],
            config=config,
        )
        for dataset_id in ("rcabench", "aiops2022_pre")
    }
    optimizer_steps = int(
        args.cvae_optimizer_steps
        if args.cvae_optimizer_steps is not None
        else config["cvae"]["max_epochs"]
    )
    runtime_units = build_formal_runtime_units(
        registry=registry,
        prepared_by_dataset=prepared,
        cvae_optimizer_steps=optimizer_steps,
    )
    runtime_by_id = {str(unit["unit_id"]): unit for unit in runtime_units}
    code_files, code_sha = _formal_code_closure()
    config_sha = _file_hash(config_path)
    input_identity = {
        "tested_closure_sha256": tested_closure["closure_sha256"],
        "inventory_file_sha256s": {
            dataset: _file_hash(path) for dataset, path in inventories.items()
        },
        "split_sha256s": {
            dataset: value["split"]["split_sha256"] for dataset, value in prepared.items()
        },
        "query_plan_sha256s": {
            dataset: value["query_plan"]["plan_sha256"] for dataset, value in prepared.items()
        },
        "representation_matrix_sha256s": {
            dataset: value["candidate"].matrix_sha256 for dataset, value in prepared.items()
        },
        "geometry_sha256s": {
            dataset: value["geometry"].geometry_sha256 for dataset, value in prepared.items()
        },
        "candidate_counts": {
            dataset: int(value["candidate_count"]) for dataset, value in prepared.items()
        },
        "train_counts": {
            dataset: len(value["split"]["outer_train_case_ids"]) for dataset, value in prepared.items()
        },
        "test_counts": {
            dataset: len(value["split"]["outer_test_case_ids"]) for dataset, value in prepared.items()
        },
        "cvae_optimizer_steps": optimizer_steps,
    }
    input_sha = _semantic_hash(input_identity)
    execution_manifest = build_final_execution_manifest(
        registry=registry,
        run_root=run_root,
        tmux_session=str(args.tmux_session),
        code_sha256=code_sha,
        config_sha256=config_sha,
        input_sha256=input_sha,
        max_cpu_workers=int(args.max_cpu_workers),
        max_gpu_workers=int(args.max_gpu_workers),
        require_aggregate_report=True,
    )
    preflight = {
        "schema_version": "final-rcl-formal-preflight-v1",
        "status": "passed",
        "unit_count": 6,
        "seed": 42,
        "run_root": str(run_root),
        "tmux_session": str(args.tmux_session),
        "execution_manifest_sha256": execution_manifest["manifest_sha256"],
        "tested_closure_sha256": tested_closure["closure_sha256"],
        "code_file_sha256s": code_files,
        "code_sha256": code_sha,
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "input_sha256": input_sha,
        "gpu_inventory": gpu_inventory,
        "historical_output_overwrite": False,
    }
    preflight = {**preflight, "preflight_sha256": _semantic_hash(preflight)}
    _write(run_root / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    def worker(raw_unit: Mapping[str, Any], unit_root: Path, log_path: Path) -> Mapping[str, Any]:
        runtime = runtime_by_id[str(raw_unit["unit_id"])]
        _append_log(log_path, "unit_started", unit_id=runtime["unit_id"])
        try:
            return _execute_runtime_unit(
                unit={**runtime, "resource_class": raw_unit["resource_class"]},
                prepared=prepared[str(runtime["dataset_id"])],
                config=config,
                torch_python=args.torch_python.resolve(),
                unit_root=unit_root,
                log_path=log_path,
                code_sha256=code_sha,
                config_sha256=config_sha,
                input_sha256=input_sha,
            )
        except Exception as exc:
            _append_log(log_path, "unit_failed", error_type=type(exc).__name__, message=str(exc))
            raise

    status = run_final_execution(
        execution_manifest=execution_manifest,
        unit_worker=worker,
        auto_finalize=False,
    )
    if status["failed"]:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    if status["completed"] != 6:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    report = _aggregate_six_units(execution_manifest, run_root)
    if report.get("unit_count") != 6 or any(
        not math.isfinite(float(value))
        for dataset in report["datasets"].values()
        for arm in dataset["arms"].values()
        for key, value in arm.items()
        if key != "denominator"
    ):
        raise ValueError("formal aggregate is incomplete or nonfinite")
    finalize_final_run(run_root)
    final_status = read_final_run_status(run_root=run_root)
    print(json.dumps(final_status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_formal_runtime_units", "main"]
