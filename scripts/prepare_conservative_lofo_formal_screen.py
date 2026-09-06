from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NEXUS_SOURCE_ROOT = Path(
    "${NEXUSRCL_REBUILD_ROOT}/src"
)
if str(NEXUS_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEXUS_SOURCE_ROOT))

from rcl_study.conservative_lofo_authority import load_authority_registry
from rcl_study.conservative_lofo_authority_scores import build_authority_score_bundle
from rcl_study.conservative_lofo_base_bridge import (
    fit_base_score_bridge,
    score_base_score_bridge,
)
from rcl_study.conservative_lofo_orchestration import DATASETS, EXPECTED_TARGETS
from rcl_study.conservative_lofo_protocol import (
    build_label_access_ledger,
    build_strict_lofo_fold,
    validate_label_access_ledger,
    validate_strict_lofo_fold,
)
from rcl_study.conservative_lofo_query_bridge import build_matched_dbscan_query_plan
from rcl_study.conservative_lofo_screen_preparation import prepare_materialized_screen_matrix
from rcl_study.datasets import resolve_dataset
from rcl_study.ordinary_query_engine import MappingAnnotationOracle, run_ordinary_query_engine
from rcl_study.ordinary_strategy_pipeline import merge_fusion_and_candidate_context
from scripts.prepare_conservative_lofo_formal_calibration import (
    _actual_exclusions,
    _authority_rows_and_targets,
    _default_schema,
    _inventory,
    _ordinary_unit,
    _rows_from_training_frame,
)


DEFAULT_FEATURE_ROOT = Path(
    "${NEXUSRCL_REBUILD_ROOT}/"
    "artifacts/window_feature_artifacts_hd134_stage1"
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _targets_from_window(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item for item in value.split(";") if item]
        raw = parsed if isinstance(parsed, (tuple, list, set)) else (parsed,)
    elif isinstance(value, (tuple, list, set)):
        raw = value
    else:
        raw = ()
    result = tuple(str(item).strip() for item in raw if str(item).strip())
    if not result:
        raise ValueError("fault case has no root-cause target")
    return result


def _load_feature_tables(feature_root: Path, dataset_id: str) -> Any:
    from nexusrcl_rebuild.training.semisupervised import load_feature_bundle_tables

    dataset = resolve_dataset(dataset_id)
    tables = load_feature_bundle_tables(feature_root, dataset.legacy_alias)
    windows = tables.windows.copy()
    windows["window_id"] = windows["window_id"].astype(str)
    fault = windows[windows["window_kind"].astype(str) == "fault"]
    if fault["window_id"].duplicated().any():
        raise ValueError(f"feature bundle contains duplicate fault IDs for {dataset_id}")
    entity = tables.entity_features.copy()
    entity["window_id"] = entity["window_id"].astype(str)
    entity["entity_id"] = entity["entity_id"].astype(str)
    features = tuple(
        str(name)
        for name in tables.feature_columns
        if name in entity.columns and pd.api.types.is_numeric_dtype(entity[name])
    )
    if not features:
        raise ValueError(f"feature bundle lacks numeric authority features for {dataset_id}")
    return tables, fault, entity, features


def _materialized_dataset(
    *,
    dataset_id: str,
    case_ids: Sequence[str],
    inventory_rows: Mapping[str, Mapping[str, Any]],
    fault_windows: pd.DataFrame,
    entity_features: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    ordered = tuple(str(item) for item in case_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("materialized dataset case IDs contain duplicates")
    by_window = fault_windows.set_index("window_id", drop=False)
    missing_windows = sorted(set(ordered) - set(by_window.index))
    if missing_windows:
        raise ValueError(f"feature windows miss formal cases: {missing_windows[:5]}")
    frame = entity_features[entity_features["window_id"].isin(ordered)].copy()
    if "is_positive" not in frame.columns:
        raise ValueError("feature rows lack is_positive scoring evidence")
    frame["label"] = frame["is_positive"].astype(int)
    raw_rows = _rows_from_training_frame(frame, feature_names)
    grouped: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in ordered}
    for row in raw_rows:
        grouped[str(row["case_id"])].append(dict(row))
    cases: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, str]] = {}
    for case_id in ordered:
        rows = grouped[case_id]
        if not rows:
            raise ValueError(f"candidate feature coverage is empty for {case_id}")
        candidates = tuple(str(row["candidate_id"]) for row in rows)
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"candidate feature rows duplicate an entity for {case_id}")
        positives = tuple(index for index, row in enumerate(rows) if int(row.get("label", 0)) == 1)
        window_targets = _targets_from_window(by_window.loc[case_id]["positive_ids_list"])
        target_indices = tuple(index for index, candidate in enumerate(candidates) if candidate in set(window_targets))
        if positives != target_indices or not positives:
            raise ValueError(f"positive candidate evidence drift for {case_id}")
        base_rows = [list(map(float, row["base_features"])) for row in rows]
        if any(
            len(values) != len(feature_names)
            or not all(math.isfinite(value) for value in values)
            for values in base_rows
        ):
            raise ValueError(f"base feature row drift for {case_id}")
        cases[case_id] = {
            "candidate_ids": candidates,
            "positive_indices": positives,
            "base_feature_rows": base_rows,
            "observable_rows": rows,
        }
        labels[case_id] = {
            "root_cause": ";".join(window_targets),
            "fault_type": str(inventory_rows[case_id]["fault_type"]),
        }
    return {
        "dataset_id": dataset_id,
        "base_feature_names": feature_names,
        "observable_schema": _default_schema().to_dict(),
        "cases": cases,
        "labels_by_case": labels,
    }


def _ordinary_protocol(
    *,
    dataset_id: str,
    inventory: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    strategy_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    engine = run_ordinary_query_engine(
        candidate_rows,
        annotation_oracle=MappingAnnotationOracle(answers),
        strategy_id="dbscan_coverage",
        active_learning_seed=42,
        config=dict(strategy_config),
    )
    selected = tuple(str(item) for item in engine["selected_case_ids"])
    membership_identity = {
        "schema_version": "conservative-lofo-ordinary-membership-v1",
        "dataset_id": dataset_id,
        "candidate_case_ids": tuple(str(item) for item in inventory["train_case_ids"]),
        "evaluation_case_ids": tuple(str(item) for item in inventory["test_case_ids"]),
    }
    membership = {**membership_identity, "membership_sha256": _hash(membership_identity)}
    plan_identity = {
        "schema_version": "conservative-lofo-ordinary-dbscan-query-plan-v1",
        "dataset_id": dataset_id,
        "strategy_id": "dbscan_coverage",
        "strategy_config": dict(strategy_config),
        "active_learning_seed": 42,
        "budget": 30,
        "candidate_case_ids": tuple(str(row["case_id"]) for row in candidate_rows),
        "selected_case_ids": selected,
        "events": deepcopy(engine["events"]),
        "engine_query_plan_sha256": str(engine["query_plan_sha256"]),
    }
    plan = {**plan_identity, "query_plan_sha256": _hash(plan_identity)}
    ledger_identity = {
        "schema_version": "conservative-lofo-ordinary-label-access-ledger-v1",
        "dataset_id": dataset_id,
        "queried_case_ids": selected,
        "evaluation_case_ids": membership["evaluation_case_ids"],
        "total_label_cost": 30,
        "queried_role": "queried_supervision",
        "evaluation_role": "post_inference_scoring_only",
    }
    ledger = {**ledger_identity, "ledger_sha256": _hash(ledger_identity)}
    return membership, plan, ledger


def _request(
    *,
    dataset_id: str,
    kind: str,
    target_display: str | None,
    target_internal: str | None,
    selected: Sequence[str],
    evaluation: Sequence[str],
    unused: Sequence[str],
    membership_sha256: str,
    query_plan_sha256: str,
    events: Sequence[Mapping[str, Any]],
    label_ledger_sha256: str,
    code_sha256: str,
    materialized_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = tuple(str(item) for item in selected)
    evaluation_ids = tuple(str(item) for item in evaluation)
    cases = {
        str(key): dict(value)
        for key, value in dict(materialized_dataset["cases"]).items()
    }
    feature_names = tuple(str(value) for value in materialized_dataset["base_feature_names"])
    support_rows, support_targets, support_candidates = _authority_rows_and_targets(
        cases, selected_ids, feature_names
    )
    evaluation_rows, evaluation_targets, evaluation_candidates = _authority_rows_and_targets(
        cases, evaluation_ids, feature_names
    )
    bridge = fit_base_score_bridge(
        dataset_id=dataset_id,
        training_rows=support_rows,
        targets_by_case=support_targets,
        feature_columns=feature_names,
        query_plan_sha256=query_plan_sha256,
        random_state=42,
    )
    support_artifact = score_base_score_bridge(
        bridge=bridge,
        rows=support_rows,
        targets_by_case=support_targets,
        artifact_role="support",
        expected_candidates_by_case=support_candidates,
    )
    evaluation_artifact = score_base_score_bridge(
        bridge=bridge,
        rows=evaluation_rows,
        targets_by_case=evaluation_targets,
        artifact_role="evaluation",
        expected_candidates_by_case=evaluation_candidates,
    )
    target_token = "ordinary" if target_display is None else _hash(target_display)[:12]
    fold_id = f"{kind}__{dataset_id}__{target_token}"
    authority_bundle = build_authority_score_bundle(
        dataset_id=dataset_id,
        fold_id=fold_id,
        fold_kind=kind,
        held_out_fault_type=target_display,
        query_plan_sha256=query_plan_sha256,
        feature_order_sha256=bridge.feature_order_sha256,
        model_sha256=bridge.model_sha256,
        support_case_ids=selected_ids,
        evaluation_case_ids=evaluation_ids,
        support_artifact=support_artifact,
        evaluation_artifact=evaluation_artifact,
    )
    authority_path = f"authority_scores/{dataset_id}/{target_token}.json"
    base_contract_identity = {
        "schema_version": "conservative-lofo-base-score-contract-v1",
        "dataset_id": dataset_id,
        "kind": kind,
        "held_out_fault_type": target_display,
        "membership_sha256": membership_sha256,
        "query_plan_sha256": query_plan_sha256,
        "feature_names": feature_names,
        "backend_id": "pairwise_linear",
        "authority_score_bundle_sha256": authority_bundle[
            "authority_score_bundle_sha256"
        ],
        "authority_model_sha256": authority_bundle["model_sha256"],
        "authority_feature_order_sha256": authority_bundle["feature_order_sha256"],
        "authority_score_artifact_sha256": dict(
            authority_bundle["score_artifact_sha256"]
        ),
        "code_sha256": code_sha256,
    }
    identity = {
        "schema_version": "conservative-lofo-materialized-screen-request-v1",
        "dataset_id": dataset_id,
        "kind": kind,
        "held_out_fault_type": target_display,
        "held_out_fault_type_internal": target_internal,
        "seed": 42,
        "budget": 30,
        "selected_case_ids": selected_ids,
        "fit_case_ids": selected_ids,
        "evaluation_case_ids": evaluation_ids,
        "unused_nonheld_outer_test_case_ids": tuple(str(item) for item in unused),
        "query_plan_sha256": query_plan_sha256,
        "query_event_history_sha256": _hash(tuple(events)),
        "label_ledger_sha256": label_ledger_sha256,
        "membership_sha256": membership_sha256,
        "base_score_contract_sha256": _hash(base_contract_identity),
        "authority_score_bundle_path": authority_path,
        "authority_score_bundle_sha256": authority_bundle[
            "authority_score_bundle_sha256"
        ],
        "authority_model_sha256": authority_bundle["model_sha256"],
        "authority_feature_order_sha256": authority_bundle["feature_order_sha256"],
        "authority_score_artifact_sha256": dict(
            authority_bundle["score_artifact_sha256"]
        ),
        "authority_score_bundle": authority_bundle,
        "dataset": deepcopy(dict(materialized_dataset)),
    }
    return {**identity, "request_sha256": _hash(identity)}


def _prepare_requests(
    *,
    ordinary_run_root: Path,
    feature_root: Path,
    code_sha256: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    ordinary_manifest = _read(ordinary_run_root / "ordinary_strategy_manifest.json")
    authority = load_authority_registry(REPO_ROOT)
    requests: dict[str, dict[str, dict[str, Any]]] = {}
    audits = []
    for dataset_id in DATASETS:
        unit = _ordinary_unit(ordinary_manifest, dataset_id)
        inventory, inventory_rows = _inventory(Path(unit["inventory_path"]), dataset_id)
        fusion = _read(Path(unit["fusion_artifact_path"]))
        context = _read(Path(unit["candidate_context_path"]))
        merged = merge_fusion_and_candidate_context(fusion, context)
        candidates_by_id = {str(row["case_id"]): dict(row) for row in merged["candidates"]}
        answers = {str(key): dict(value) for key, value in merged["private_annotations"].items()}
        train_ids = tuple(str(item) for item in inventory["train_case_ids"])
        if set(candidates_by_id) != set(train_ids) or set(answers) != set(train_ids):
            raise ValueError(f"ordinary candidate/context coverage drift for {dataset_id}")
        _tables, fault_windows, entity_features, feature_names = _load_feature_tables(feature_root, dataset_id)
        if set(inventory_rows) != set(fault_windows["window_id"].astype(str)):
            raise ValueError(f"inventory/feature fault-case coverage drift for {dataset_id}")
        actual_targets = _actual_exclusions(dataset_id, inventory)
        display_to_internal = dict(zip(EXPECTED_TARGETS[dataset_id], actual_targets))
        requests[dataset_id] = {}
        for display_target in EXPECTED_TARGETS[dataset_id]:
            internal_target = display_to_internal[display_target]
            fold = build_strict_lofo_fold(dataset_id, inventory, internal_target)
            validate_strict_lofo_fold(fold, inventory)
            fold_candidates = [candidates_by_id[case_id] for case_id in fold["candidate_case_ids"]]
            fold_answers = {case_id: answers[case_id] for case_id in fold["candidate_case_ids"]}
            plan = build_matched_dbscan_query_plan(fold, fold_candidates, fold_answers, authority)
            selected = tuple(str(item) for item in plan["selected_case_ids"])
            ledger = build_label_access_ledger(fold, inventory, selected)
            validate_label_access_ledger(ledger, fold, inventory)
            evaluation = tuple(str(item) for item in fold["test_only_case_ids"])
            materialized = _materialized_dataset(
                dataset_id=dataset_id,
                case_ids=selected + evaluation,
                inventory_rows=inventory_rows,
                fault_windows=fault_windows,
                entity_features=entity_features,
                feature_names=feature_names,
            )
            requests[dataset_id][display_target] = _request(
                dataset_id=dataset_id,
                kind="strict_lofo",
                target_display=display_target,
                target_internal=internal_target,
                selected=selected,
                evaluation=evaluation,
                unused=fold["unused_outer_test_case_ids"],
                membership_sha256=str(fold["membership_sha256"]),
                query_plan_sha256=str(plan["matched_query_plan_sha256"]),
                events=plan["events"],
                label_ledger_sha256=str(ledger["ledger_sha256"]),
                code_sha256=code_sha256,
                materialized_dataset=materialized,
            )
            audits.append(
                {
                    "dataset_id": dataset_id,
                    "kind": "strict_lofo",
                    "held_out_fault_type": display_target,
                    "held_out_fault_type_internal": internal_target,
                    "candidate_case_count": len(fold["candidate_case_ids"]),
                    "selected_case_count": len(selected),
                    "evaluation_case_count": len(evaluation),
                    "unused_nonheld_outer_test_case_count": len(fold["unused_outer_test_case_ids"]),
                    "held_out_fitting_overlap_count": len(set(selected) & set(evaluation)),
                    "membership_sha256": fold["membership_sha256"],
                    "query_plan_sha256": plan["matched_query_plan_sha256"],
                    "label_ledger_sha256": ledger["ledger_sha256"],
                    "request_sha256": requests[dataset_id][display_target]["request_sha256"],
                }
            )
        ordinary_candidates = [candidates_by_id[case_id] for case_id in train_ids]
        ordinary_answers = {case_id: answers[case_id] for case_id in train_ids}
        membership, plan, ledger = _ordinary_protocol(
            dataset_id=dataset_id,
            inventory=inventory,
            candidate_rows=ordinary_candidates,
            answers=ordinary_answers,
            strategy_config=unit["strategy_config"],
        )
        selected = tuple(str(item) for item in plan["selected_case_ids"])
        evaluation = tuple(str(item) for item in inventory["test_case_ids"])
        materialized = _materialized_dataset(
            dataset_id=dataset_id,
            case_ids=selected + evaluation,
            inventory_rows=inventory_rows,
            fault_windows=fault_windows,
            entity_features=entity_features,
            feature_names=feature_names,
        )
        requests[dataset_id]["ordinary"] = _request(
            dataset_id=dataset_id,
            kind="ordinary",
            target_display=None,
            target_internal=None,
            selected=selected,
            evaluation=evaluation,
            unused=(),
            membership_sha256=membership["membership_sha256"],
            query_plan_sha256=plan["query_plan_sha256"],
            events=plan["events"],
            label_ledger_sha256=ledger["ledger_sha256"],
            code_sha256=code_sha256,
            materialized_dataset=materialized,
        )
        audits.append(
            {
                "dataset_id": dataset_id,
                "kind": "ordinary",
                "held_out_fault_type": None,
                "candidate_case_count": len(train_ids),
                "selected_case_count": len(selected),
                "evaluation_case_count": len(evaluation),
                "held_out_fitting_overlap_count": len(set(selected) & set(evaluation)),
                "membership_sha256": membership["membership_sha256"],
                "query_plan_sha256": plan["query_plan_sha256"],
                "label_ledger_sha256": ledger["ledger_sha256"],
                "request_sha256": requests[dataset_id]["ordinary"]["request_sha256"],
            }
        )
    audit_identity = {
        "schema_version": "conservative-lofo-formal-screen-materialization-audit-v1",
        "status": "passed",
        "dataset_ids": DATASETS,
        "strict_lofo_fold_count": 10,
        "ordinary_fold_count": 2,
        "folds": audits,
    }
    return requests, {**audit_identity, "audit_sha256": _hash(audit_identity)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-run-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--selected-profile-registry", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--run-id", default="20260901-01")
    parser.add_argument("--code-sha256", required=True)
    args = parser.parse_args()
    if len(args.code_sha256) != 64 or any(character not in "0123456789abcdef" for character in args.code_sha256.lower()):
        raise ValueError("code-sha256 must contain 64 lowercase hex characters")
    requests, materialization_audit = _prepare_requests(
        ordinary_run_root=args.ordinary_run_root,
        feature_root=args.feature_root,
        code_sha256=args.code_sha256,
    )
    registry = _read(args.selected_profile_registry)
    content_identity = {
        "schema_version": "conservative-lofo-formal-screen-content-v1",
        "code_sha256": args.code_sha256,
        "selected_profile_registry_sha256": registry["selected_profile_registry_sha256"],
        "request_sha256_by_dataset": {
            dataset: {key: value["request_sha256"] for key, value in requests[dataset].items()}
            for dataset in DATASETS
        },
    }
    content_sha = _hash(content_identity)
    output_root = args.output_parent.resolve() / f"{content_sha[:12]}-{args.run_id}"
    result = prepare_materialized_screen_matrix(
        requests_by_dataset=requests,
        selected_profile_registry=registry,
        code_sha256=args.code_sha256,
        output_root=output_root,
    )
    _write(output_root / "materialization-audit.json", materialization_audit)
    execution = {
        "schema_version": "conservative-lofo-formal-screen-execution-v1",
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "execution_content_sha256": content_sha,
        "code_sha256": args.code_sha256,
        "selected_profile_registry_sha256": registry["selected_profile_registry_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "strict_lofo_fold_count": 10,
        "ordinary_fold_count": 2,
        "lofo_unit_count": 40,
        "ordinary_unit_count": 8,
        "unit_count": 48,
        "full_lofo_authorized": False,
        "automatic_follow_on": False,
    }
    execution["execution_sha256"] = _hash(execution)
    _write(output_root / "execution.json", execution)
    print(json.dumps(execution, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
