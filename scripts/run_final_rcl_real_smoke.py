#!/usr/bin/env python3
"""Run all six bounded real-data branches for the final RCL method."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DATASETS = ("rcabench", "aiops2022_pre")
ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)
CLOSURE_CODE_RELATIVE_PATHS = (
    "rcl_study/conservative_lofo_base_bridge.py",
    "rcl_study/conservative_lofo_oser.py",
    "rcl_study/conservative_lofo_residual.py",
    "rcl_study/conservative_lofo_state.py",
    "rcl_study/final_rcl_hdbscan_proxy.py",
    "rcl_study/final_rcl_oser.py",
    "rcl_study/final_rcl_oser_runner.py",
    "rcl_study/final_rcl_pairwise.py",
    "rcl_study/final_rcl_real_execution.py",
    "rcl_study/final_rcl_training.py",
    "rcl_study/service_continuous_neural_runner.py",
    "rcl_study/service_continuous_pairwise_bridge.py",
    "rcl_study/service_continuous_real_ranker_smoke.py",
    "rcl_study/service_continuous_unlabeled_pool.py",
    "scripts/run_final_rcl_real_smoke.py",
)


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _extend_package_path(package: Any, extension: Path) -> None:
    resolved = str(Path(extension).resolve())
    existing = [str(value) for value in package.__path__ if str(value) != resolved]
    package.__path__[:] = [resolved, *existing]


def _feature_alias_sources(
    *, rcabench_feature_dir: Path, aiops22_feature_dir: Path
) -> dict[str, Path]:
    """Bind public and legacy aliases to the two frozen authority bundles."""

    rcabench = Path(rcabench_feature_dir).resolve()
    aiops22 = Path(aiops22_feature_dir).resolve()
    return {"rcabench": rcabench, "hd4": rcabench, "hd1": aiops22}


def _transfer_feature_indices(
    feature_names: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split movable symptoms from candidate-owned context by feature semantics."""

    names = tuple(str(value) for value in feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("authority feature names must be nonempty and unique")

    def is_identity_context(name: str) -> bool:
        return (
            name.startswith("entity_is_")
            or name.startswith("topo_")
            or name == "topology_change_count"
            or (name.startswith("has_") and name.endswith("_signal"))
            or name == "modalities_present_count"
        )

    identity_context = tuple(
        index for index, name in enumerate(names) if is_identity_context(name)
    )
    symptom = tuple(
        index for index, name in enumerate(names) if not is_identity_context(name)
    )
    if not identity_context or not symptom:
        raise ValueError("authority features lack symptom or identity/context support")
    return symptom, identity_context


def _feature_alias_root(
    *, rcabench_feature_dir: Path, aiops22_feature_dir: Path, run_root: Path
) -> Path:
    """Expose the fixed package's aliases without mutating source bundles."""

    alias_root = run_root / "input-aliases"
    alias_root.mkdir(parents=True, exist_ok=True)
    sources = _feature_alias_sources(
        rcabench_feature_dir=rcabench_feature_dir,
        aiops22_feature_dir=aiops22_feature_dir,
    )
    for alias, source in sources.items():
        if not source.is_dir():
            raise ValueError(f"feature source is absent: {source}")
        target = alias_root / alias
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"feature alias already points elsewhere: {target}")
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)
    return alias_root


def _fixed_geometry(dataset_id: str, candidate: Any, config: Any) -> Any:
    if dataset_id == "rcabench":
        from rcl_study.combined_active_learning_clustering import fit_native_hdbscan

        return fit_native_hdbscan(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=42,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_method=config.cluster_selection_method,
        )
    from rcl_study.combined_active_learning_tuning import generate_round_a_grid
    from rcl_study.combined_active_learning_tuning_structure import (
        fit_configuration_geometry,
    )
    from rcl_study.combined_active_learning_winner_analysis import (
        to_native_hdbscan_result,
    )

    matches = [
        item
        for item in generate_round_a_grid()
        if item.min_cluster_size == config.min_cluster_size
        and item.min_samples == config.min_samples
        and item.cluster_selection_method == config.cluster_selection_method
        and item.metric == "euclidean"
        and item.max_cluster_size is None
        and item.allow_single_cluster is False
    ]
    if len(matches) != 1:
        raise ValueError("frozen AIOps22 HDBSCAN configuration is not unique")
    fitted = fit_configuration_geometry(matches[0], candidate=candidate)
    return to_native_hdbscan_result(
        fitted_geometry=fitted,
        candidate=candidate,
        dataset_id=dataset_id,
        active_learning_seed=42,
    )


def _rows_for_cases(
    dataset: Mapping[str, Any], case_ids: Sequence[str]
) -> list[dict[str, Any]]:
    names = tuple(str(value) for value in dataset["base_feature_names"])
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = dataset["cases"][case_id]
        for candidate_id, values in zip(
            case["candidate_ids"], case["base_feature_rows"]
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": str(candidate_id),
                    **{
                        name: float(value)
                        for name, value in zip(names, values)
                    },
                }
            )
    return rows


def _targets(
    dataset: Mapping[str, Any], case_ids: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    return {
        case_id: tuple(
            str(dataset["cases"][case_id]["candidate_ids"][index])
            for index in dataset["cases"][case_id]["positive_indices"]
        )
        for case_id in case_ids
    }


def _state_schema_from_payload(value: Mapping[str, Any]) -> Any:
    """Rehydrate the canonical ``ObservableStateSchema.to_dict`` payload."""

    from rcl_study.conservative_lofo_state import ObservableStateSchema

    payload = dict(value)
    fields = payload.get("fields_by_type")
    if not isinstance(fields, Mapping):
        from scripts.prepare_conservative_lofo_formal_calibration import (
            _schema_from_payload,
        )

        return _schema_from_payload(payload)
    expected = ("metric", "log", "trace", "topology", "time", "candidate")
    if set(fields) != set(expected):
        raise ValueError("serialized observable schema field types drifted")
    normalized = {name: tuple(str(item) for item in fields[name]) for name in expected}
    feature_order = tuple(item for name in expected for item in normalized[name])
    if tuple(payload.get("feature_order", ())) != feature_order:
        raise ValueError("serialized observable schema feature order drifted")
    return ObservableStateSchema(
        metric_fields=normalized["metric"],
        log_fields=normalized["log"],
        trace_fields=normalized["trace"],
        topology_fields=normalized["topology"],
        time_fields=normalized["time"],
        candidate_fields=normalized["candidate"],
        modality_presence_fields=dict(payload.get("modality_presence_fields", {})),
        clip_value=float(payload.get("clip_value", 20.0)),
    )


def _factorized_records(
    dataset: Mapping[str, Any], case_ids: Sequence[str]
) -> list[dict[str, Any]]:
    from rcl_study.service_continuous_unlabeled_pool import (
        OBSERVABLE_STATE_FIELDS,
        build_unlabeled_state_record,
    )

    records = []
    for case_id in case_ids:
        for raw in dataset["cases"][case_id]["observable_rows"]:
            row = dict(raw)
            records.append(
                build_unlabeled_state_record(
                    {
                        "case_id": case_id,
                        "candidate_id": str(row["candidate_id"]),
                        **{field: row[field] for field in OBSERVABLE_STATE_FIELDS},
                    }
                )
            )
    return records


def _state_materialization(
    dataset: Mapping[str, Any], fit_ids: Sequence[str], evaluation_ids: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], str]:
    from rcl_study.conservative_lofo_state import (
        build_candidate_state_rows,
        fit_fold_state_transform,
    )
    admitted = tuple(fit_ids) + tuple(evaluation_ids)
    schema = _state_schema_from_payload(dict(dataset["observable_schema"]))
    rows = [
        dict(row)
        for case_id in admitted
        for row in dataset["cases"][case_id]["observable_rows"]
    ]
    transform = fit_fold_state_transform(rows, schema, tuple(fit_ids), tuple(evaluation_ids))
    artifact = build_candidate_state_rows(
        rows,
        schema,
        transform,
        admitted,
        "final_rcl_bounded_real_smoke_state",
    )
    lookup = {
        (str(member["case_id"]), str(member["candidate_id"])): dict(state)
        for member, state in zip(artifact["membership"], artifact["state_rows"])
    }
    cases: dict[str, dict[str, Any]] = {}
    for case_id in admitted:
        raw = dataset["cases"][case_id]
        state_rows = [lookup[(case_id, str(candidate))] for candidate in raw["candidate_ids"]]
        cases[case_id] = {
            "candidate_ids": tuple(str(value) for value in raw["candidate_ids"]),
            "positive_indices": tuple(int(value) for value in raw["positive_indices"]),
            "states": [list(row["state_vector"]) for row in state_rows],
            "evidence_supported": [
                any(int(value) for value in row["state_mask"]) for row in state_rows
            ],
        }
    return cases, str(transform["transform_sha256"])


def _attach_base_scores(
    cases: Mapping[str, Mapping[str, Any]], score: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case_id, raw in cases.items():
        case = deepcopy(dict(raw))
        candidates = tuple(str(value) for value in case["candidate_ids"])
        scores = dict(score["scores_by_case"][case_id])
        if set(scores) != set(candidates):
            raise ValueError(f"base score candidate coverage drifted for {case_id}")
        case["base_scores"] = [float(scores[candidate]) for candidate in candidates]
        result[case_id] = case
    return result


def _synthetic_rows_for_scoring(
    synthetic_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    rows: list[dict[str, Any]] = []
    targets: dict[str, tuple[str, ...]] = {}
    candidates: dict[str, tuple[str, ...]] = {}
    for raw in synthetic_rows:
        child = dict(raw)
        case_id = "synthetic:" + str(child["synthetic_row_sha256"])
        target = str(child["target_service_id"])
        candidate_rows = [dict(row) for row in child["candidate_rows"]]
        candidate_ids = tuple(str(row["candidate_id"]) for row in candidate_rows)
        rows.extend({**row, "case_id": case_id} for row in candidate_rows)
        targets[case_id] = (target,)
        candidates[case_id] = candidate_ids
    return rows, targets, candidates


def _synthetic_state_cases(
    synthetic_rows: Sequence[Mapping[str, Any]],
    real_state_cases: Mapping[str, Mapping[str, Any]],
    synthetic_score: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in synthetic_rows:
        child = dict(raw)
        key = "synthetic:" + str(child["synthetic_row_sha256"])
        parent = deepcopy(dict(real_state_cases[str(child["source_case_id"])]))
        candidates = list(parent["candidate_ids"])
        target_index = candidates.index(str(child["target_service_id"]))
        root_index = int(parent["positive_indices"][0])
        states = [list(row) for row in parent["states"]]
        states[target_index], states[root_index] = states[root_index], states[target_index]
        evidence = list(parent["evidence_supported"])
        evidence[target_index], evidence[root_index] = evidence[root_index], evidence[target_index]
        scores = dict(synthetic_score["scores_by_case"][key])
        result[key] = {
            "candidate_ids": tuple(candidates),
            "positive_indices": (target_index,),
            "states": states,
            "evidence_supported": evidence,
            "base_scores": [float(scores[candidate]) for candidate in candidates],
        }
    return result


def _run_torch_worker(
    *, torch_python: Path, module: str, request: Path, output: Path
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(torch_python),
            "-m",
            module,
            "--input",
            str(request),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{module} failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
        )
    return _read(output)


def _prepare_dataset(
    *,
    dataset_id: str,
    package_root: Path,
    feature_root: Path,
    alias_root: Path,
    inventory_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    package_src = package_root / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    import rcl_study

    _extend_package_path(rcl_study, package_src / "rcl_study")
    from fixed_active_learning.config import load_fixed_config
    from fixed_active_learning.pipeline import build_global_pca_dim32
    from fixed_active_learning.plan_contract import validate_query_plan
    from rcl_study.final_rcl_evaluation import build_fixed_split_adapter
    from scripts.prepare_conservative_lofo_formal_calibration import _inventory
    from scripts.prepare_conservative_lofo_formal_screen import (
        _load_feature_tables,
        _materialized_dataset,
    )

    fixed_config = load_fixed_config(
        package_root / "configs" / f"{dataset_id}.json",
        data_root_override=alias_root,
    )
    candidate = build_global_pca_dim32(fixed_config)
    geometry = _fixed_geometry(dataset_id, candidate, fixed_config)
    expected = dict(config["active_learning"]["datasets"][dataset_id])
    if (
        candidate.matrix_sha256 != expected["representation_matrix_sha256"]
        or geometry.geometry_sha256 != expected["geometry_sha256"]
    ):
        raise ValueError(f"{dataset_id} fixed HDBSCAN geometry drifted")
    query_plan_path = (
        package_root
        / "reference"
        / "query_plans"
        / dataset_id
        / "center-seed42.json"
    )
    query_plan = _read(query_plan_path)
    validate_query_plan(query_plan)
    if query_plan["plan_sha256"] != expected["query_plan_sha256"]:
        raise ValueError(f"{dataset_id} fixed query plan drifted")
    inventory_raw = _read(inventory_path)
    candidate_pool = _read(package_root / "reference" / "candidate_pools" / f"{dataset_id}.json")
    split = build_fixed_split_adapter(
        expected_dataset_id=dataset_id,
        inventory_manifest=inventory_raw,
        candidate_pool=candidate_pool,
        query_plan=query_plan,
    )
    inventory, inventory_rows = _inventory(inventory_path, dataset_id)
    tables, fault, entity, features = _load_feature_tables(feature_root, dataset_id)
    del tables
    all_ids = tuple(split["outer_train_case_ids"]) + tuple(split["outer_test_case_ids"])
    materialized = _materialized_dataset(
        dataset_id=dataset_id,
        case_ids=all_ids,
        inventory_rows=inventory_rows,
        fault_windows=fault,
        entity_features=entity,
        feature_names=features,
    )
    candidate_counts = {len(case["candidate_ids"]) for case in materialized["cases"].values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"{dataset_id} candidate cardinality is not constant")
    return {
        "split": split,
        "dataset": materialized,
        "candidate": candidate,
        "geometry": geometry,
        "query_plan": query_plan,
        "query_plan_path": query_plan_path,
        "inventory_path": inventory_path,
        "candidate_count": next(iter(candidate_counts)),
    }


def _proxy_partition(
    *, dataset_id: str, prepared: Mapping[str, Any], fit_ids: Sequence[str], supervised_ids: Sequence[str]
) -> dict[str, Any]:
    from rcl_study.final_rcl_hdbscan_proxy import build_hdbscan_proxy_partition

    geometry = prepared["geometry"]
    full_ids = tuple(str(value) for value in geometry.case_ids)
    positions = {case_id: index for index, case_id in enumerate(full_ids)}
    indices = [positions[str(case_id)] for case_id in fit_ids]
    config_path = REPO_ROOT / "configs" / "final_rcl" / "hdbscan_proxy_cvae_oser_seed42.json"
    config = _read(config_path)
    parameters = config["active_learning"]["datasets"][dataset_id]
    is_full = tuple(fit_ids) == full_ids
    return build_hdbscan_proxy_partition(
        dataset_id=dataset_id,
        fit_case_ids=tuple(fit_ids),
        queried_case_ids=tuple(supervised_ids),
        held_out_case_ids=tuple(prepared["split"]["outer_test_case_ids"]),
        raw_labels=[int(geometry.labels[index]) for index in indices],
        membership_strengths=[float(geometry.membership_strengths[index]) for index in indices],
        representation_matrix_sha256=(
            parameters["representation_matrix_sha256"]
            if is_full
            else _semantic_hash({"full": parameters["representation_matrix_sha256"], "fit": tuple(fit_ids)})
        ),
        geometry_sha256=(
            parameters["geometry_sha256"]
            if is_full
            else _semantic_hash({"full": parameters["geometry_sha256"], "fit": tuple(fit_ids)})
        ),
        active_partition_sha256=(
            parameters["partition_sha256"]
            if is_full
            else _semantic_hash({"full": parameters["partition_sha256"], "fit": tuple(fit_ids)})
        ),
        min_cluster_size=int(parameters["min_cluster_size"]),
        min_samples=int(parameters["min_samples"]),
        cluster_selection_method="leaf",
    )


def _run_unit(
    *,
    unit: Mapping[str, Any],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    torch_python: Path,
    unit_root: Path,
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
    dataset = prepared["dataset"]
    fit_ids = tuple(str(value) for value in unit["fit_case_ids"])
    supervised_ids = tuple(str(value) for value in unit["supervised_case_ids"])
    evaluation_ids = tuple(str(value) for value in unit["evaluation_case_ids"])
    feature_names = tuple(str(value) for value in dataset["base_feature_names"])
    real_rows = _rows_for_cases(dataset, supervised_ids)
    real_targets = _targets(dataset, supervised_ids)
    evaluation_rows = _rows_for_cases(dataset, evaluation_ids)
    evaluation_targets = _targets(dataset, evaluation_ids)
    symptom_feature_indices, identity_context_indices = _transfer_feature_indices(
        feature_names
    )
    supervision_sha = (
        prepared["query_plan"]["plan_sha256"]
        if unit["training_mode"] == "query_only"
        else _semantic_hash(
            {"dataset_id": dataset_id, "training_mode": "oracle_full", "cases": supervised_ids}
        )
    )
    unit_root.mkdir(parents=True, exist_ok=True)
    if arm_id == "hdbscan_query_pairwise":
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
            fitted.bridge,
            evaluation_rows,
            evaluation_targets,
            "bounded_real_smoke",
        )
        oser_status = "not_applicable"
        training_audit = fitted.training_audit
    else:
        factorized = _factorized_records(dataset, fit_ids)
        by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in fit_ids}
        for row in factorized:
            by_case[str(row["case_id"])].append(row)
        source_cases = []
        source_targets = _targets(dataset, supervised_ids)
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
        partition = _proxy_partition(
            dataset_id=dataset_id,
            prepared=prepared,
            fit_ids=fit_ids,
            supervised_ids=supervised_ids,
        )
        neural_request = build_final_neural_request(
            dataset_id=dataset_id,
            training_mode=str(unit["training_mode"]),
            fit_case_ids=fit_ids,
            supervised_case_ids=supervised_ids,
            held_out_case_ids=tuple(prepared["split"]["outer_test_case_ids"]),
            expected_candidates_per_case=int(prepared["candidate_count"]),
            pretraining_records=factorized,
            source_cases=source_cases,
            proxy_mode_partition=partition,
            profile=dict(config["cvae"]),
            optimizer_steps=int(unit["cvae_optimizer_steps"]),
            samples_per_target=1,
            device="cuda",
            checkpoint_output_path=str((unit_root / "cvae-checkpoint.pt").resolve()),
        )
        neural_input = unit_root / "cvae-request.json"
        neural_output = unit_root / "cvae-result.json"
        _write(neural_input, neural_request)
        neural_result = _run_torch_worker(
            torch_python=torch_python,
            module="rcl_study.service_continuous_neural_runner",
            request=neural_input,
            output=neural_output,
        )
        generated = [
            dict(row)
            for row in neural_result["arms"]["proxy_mode_cvae_compatible"]
        ]
        bounds_matrix = np.asarray(
            [
                row
                for case_id in fit_ids
                for row in dataset["cases"][case_id]["base_feature_rows"]
            ],
            dtype=float,
        )
        lower = bounds_matrix.min(axis=0)
        upper = bounds_matrix.max(axis=0)
        adapted = [
            adapt_generated_candidate_case(
                generated_row=row,
                source_case=dataset["cases"][str(row["source_case_id"])],
                source_root_service_id=source_targets[str(row["source_case_id"])][0],
                feature_names=feature_names,
                feature_lower_bounds=lower,
                feature_upper_bounds=upper,
                symptom_feature_indices=symptom_feature_indices,
                identity_context_indices=identity_context_indices,
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
            fitted.bridge,
            evaluation_rows,
            evaluation_targets,
            "bounded_real_smoke",
        )
        state_cases, state_transform_sha = _state_materialization(
            dataset, fit_ids, evaluation_ids
        )
        real_state = {case_id: state_cases[case_id] for case_id in supervised_ids}
        real_training_score = score_base_score_bridge(
            fitted.bridge,
            real_rows,
            real_targets,
            "oser_training_real",
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
        training_cases = {**real_state, **synthetic_state}
        inference_cases = _attach_base_scores(
            {case_id: state_cases[case_id] for case_id in evaluation_ids},
            base_evaluation,
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
            training_cases=training_cases,
            inference_cases=inference_cases,
            base_score_artifact=base_evaluation,
            profile=dict(config["oser"]),
            state_transform_sha256=state_transform_sha,
            artifact_role="bounded_real_smoke",
        )
        oser_input = unit_root / "oser-request.json"
        oser_output = unit_root / "oser-result.json"
        _write(oser_input, oser_request)
        oser_result = _run_torch_worker(
            torch_python=torch_python,
            module="rcl_study.final_rcl_oser_runner",
            request=oser_input,
            output=oser_output,
        )
        score = oser_result["score_artifact"]
        oser_status = str(oser_result["oser_status"])
        training_audit = {
            "pairwise": fitted.training_audit,
            "cvae": neural_result["activity_audit"],
            "oser": oser_result["training_audit"],
            "synthetic_ledger_sha256": ledger["ledger_sha256"],
        }
    score_field = "final_scores_by_case" if arm_id != "hdbscan_query_pairwise" else "scores_by_case"
    scores = dict(score[score_field])
    rankings = dict(score["rankings_by_case"])
    candidate_complete = all(
        set(rankings[case_id]) == set(dataset["cases"][case_id]["candidate_ids"])
        and set(scores[case_id]) == set(dataset["cases"][case_id]["candidate_ids"])
        for case_id in evaluation_ids
    )
    finite = all(
        math.isfinite(float(value))
        for case_id in evaluation_ids
        for value in scores[case_id].values()
    )
    result = {
        "schema_version": "final-rcl-real-smoke-unit-result-v1",
        "unit_id": unit["unit_id"],
        "dataset_id": dataset_id,
        "arm_id": arm_id,
        "evidence_role": "smoke_only",
        "status": "completed",
        "fit_test_overlap_count": len(set(fit_ids) & set(evaluation_ids)),
        "candidate_complete_rankings": candidate_complete,
        "finite_rankings": finite,
        "training_mode": unit["training_mode"],
        "real_training_case_count": len(supervised_ids),
        "evaluation_case_count": len(evaluation_ids),
        "oser_status": oser_status,
        "training_audit": training_audit,
        "rankings_by_case": rankings,
        "scores_by_case": scores,
    }
    result = {**result, "unit_result_sha256": _semantic_hash(result)}
    _write(unit_root / "unit-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--rcabench-feature-dir", type=Path, required=True)
    parser.add_argument("--aiops22-feature-dir", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--rcabench-inventory", type=Path, required=True)
    parser.add_argument("--aiops22-inventory", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--oracle-train-limit", type=int, default=30)
    parser.add_argument("--test-case-limit", type=int, default=2)
    parser.add_argument("--cvae-optimizer-steps", type=int, default=2)
    args = parser.parse_args()

    from rcl_study.final_rcl_contract import build_formal_registry, load_final_method_config
    from rcl_study.final_rcl_real_execution import (
        build_final_smoke_manifest,
        validate_final_smoke_report,
    )

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = REPO_ROOT / "configs" / "final_rcl" / "hdbscan_proxy_cvae_oser_seed42.json"
    config = load_final_method_config(config_path)
    registry = build_formal_registry(config)
    alias_root = _feature_alias_root(
        rcabench_feature_dir=args.rcabench_feature_dir,
        aiops22_feature_dir=args.aiops22_feature_dir,
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
        for dataset_id in DATASETS
    }
    smoke = build_final_smoke_manifest(
        registry=registry,
        split_adapters={dataset_id: value["split"] for dataset_id, value in prepared.items()},
        run_root=run_root,
        oracle_train_limit=int(args.oracle_train_limit),
        test_case_limit=int(args.test_case_limit),
        cvae_optimizer_steps=int(args.cvae_optimizer_steps),
    )
    _write(run_root / "smoke-manifest.json", smoke)
    results = []
    for index, unit in enumerate(smoke["units"], 1):
        progress = {
            "schema_version": "final-rcl-smoke-progress-v1",
            "expected": 6,
            "completed": len(results),
            "running_unit_id": unit["unit_id"],
            "next_unit_index": index,
        }
        _write(run_root / "progress.json", progress)
        results.append(
            _run_unit(
                unit=unit,
                prepared=prepared[str(unit["dataset_id"])],
                config=config,
                torch_python=args.torch_python.resolve(),
                unit_root=Path(str(unit["run_directory"])),
            )
        )
    report = validate_final_smoke_report(smoke_manifest=smoke, unit_results=results)
    closure_identity = {
        "schema_version": "final-rcl-tested-closure-v1",
        "smoke_manifest_sha256": smoke["smoke_manifest_sha256"],
        "smoke_report_sha256": report["smoke_report_sha256"],
        "config_sha256": _file_hash(config_path),
        "code_file_sha256s": {
            str(path.relative_to(REPO_ROOT)).replace(os.sep, "/"): _file_hash(path)
            for path in (
                REPO_ROOT / relative for relative in CLOSURE_CODE_RELATIVE_PATHS
            )
        },
        "input_file_sha256s": {
            "rcabench_inventory": _file_hash(inventories["rcabench"]),
            "aiops2022_pre_inventory": _file_hash(inventories["aiops2022_pre"]),
            "rcabench_query_plan": _file_hash(prepared["rcabench"]["query_plan_path"]),
            "aiops2022_pre_query_plan": _file_hash(prepared["aiops2022_pre"]["query_plan_path"]),
        },
    }
    closure = {**closure_identity, "closure_sha256": _semantic_hash(closure_identity)}
    _write(run_root / "smoke-report.json", report)
    _write(run_root / "tested-closure.json", closure)
    _write(
        run_root / "progress.json",
        {
            "schema_version": "final-rcl-smoke-progress-v1",
            "expected": 6,
            "completed": 6,
            "running_unit_id": None,
            "status": "passed",
        },
    )
    print(json.dumps({"report": report, "closure": closure}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
