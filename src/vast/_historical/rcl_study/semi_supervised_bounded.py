"""Bounded, auditable semi-supervised smoke experiments for both datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from nexusrcl_rebuild.pseudo_labeling.sanitizer import (
    is_label_bearing_field,
    sanitize_feature_views,
)
from nexusrcl_rebuild.training.pseudo_adapter import attach_frozen_pseudo_result
from nexusrcl_rebuild.training.semisupervised import (
    FeatureBundleTables,
    QueryPlan,
    SemiSupervisedConfig,
    build_query_plan,
    fit_semisupervised_ranker_on_tables,
    load_feature_bundle_tables,
    subset_feature_bundle_tables,
)

from .datasets import resolve_dataset
from .orchestration import ProgressTracker
from .pseudo_artifacts import (
    MatchedPseudoArms,
    derive_matched_pseudo_arms,
    load_raw_pseudo_pool,
)
from .semi_supervised_smoke import (
    build_raw_pseudo_pool,
    choose_normal_policy,
    select_raw_pseudo_pool,
    strategy_result_from_raw_pool,
)


SMOKE_SCHEMA_VERSION = "rcl-semi-supervised-bounded-smoke-v1"
ARM_IDS = ("query_only", "all_pseudo", "selected_pseudo", "oracle_full")
NORMAL_POLICIES = ("fault_only", "with_normal_class")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError("value is not JSON serializable: %r" % (value,))


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
    os.replace(str(temporary), str(path))


def _prepare_output_root(path: Path) -> Path:
    root = Path(path)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return root
    allowed = {"logs", "tmux.json"}
    unexpected = sorted(
        item.name for item in root.iterdir() if item.name not in allowed
    )
    if unexpected:
        raise ValueError(
            "smoke output root is not a fresh tmux launch root: %s"
            % unexpected
        )
    return root


def select_bounded_window_ids(
    windows: pd.DataFrame,
    *,
    fault_limit: int,
    normal_limit: int,
) -> Tuple[str, ...]:
    """Select deterministic chronological fault and normal smoke windows."""

    required = {"window_id", "window_kind", "start_ts"}
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError("windows missing bounded-selection fields: %s" % missing)
    if int(fault_limit) <= 0 or int(normal_limit) < 0:
        raise ValueError("bounded window limits are invalid")
    if windows["window_id"].astype(str).duplicated().any():
        raise ValueError("windows contain duplicate window IDs")

    def select(window_kind: str, limit: int) -> List[str]:
        frame = windows[
            windows["window_kind"].astype(str) == window_kind
        ].copy()
        if len(frame) < int(limit):
            raise ValueError(
                "not enough %s windows for bounded smoke: %d < %d"
                % (window_kind, len(frame), int(limit))
            )
        return (
            frame.sort_values(["start_ts", "window_id"])
            .head(int(limit))["window_id"]
            .astype(str)
            .tolist()
        )

    return tuple(select("fault", fault_limit) + select("normal", normal_limit))


def freeze_outer_query_plan(
    inner_query_plan: QueryPlan,
    outer_cluster_plan: QueryPlan,
    *,
    budget: int,
) -> QueryPlan:
    """Reuse inner-training queries while adopting outer-training clusters."""

    if str(inner_query_plan.dataset) != str(outer_cluster_plan.dataset):
        raise ValueError("inner and outer query plans use different datasets")
    queried_ids = tuple(str(value) for value in inner_query_plan.queried_window_ids)
    if len(queried_ids) != int(budget) or len(set(queried_ids)) != int(budget):
        raise ValueError("frozen query plan must contain exactly budget unique cases")
    outer_ids = set(str(value) for value in outer_cluster_plan.window_clusters)
    unknown = sorted(set(queried_ids).difference(outer_ids))
    if unknown:
        raise ValueError("inner queries are absent from outer training: %s" % unknown)
    labels = {
        window_id: tuple(
            sorted(
                {
                    str(value)
                    for value in inner_query_plan.queried_labels.get(
                        window_id, ()
                    )
                    if str(value)
                }
            )
        )
        for window_id in queried_ids
    }
    if any(not values for values in labels.values()):
        raise ValueError("every frozen query must have a complete annotation answer")
    roles = {
        window_id: str(
            inner_query_plan.queried_roles.get(window_id, "inner_train_query")
        )
        for window_id in queried_ids
    }
    metadata = dict(outer_cluster_plan.metadata)
    metadata.update(
        {
            "budget": int(budget),
            "effective_queried_window_count": len(queried_ids),
            "query_plan_source": "inner_train_frozen",
            "inner_query_plan_sha256": _semantic_sha256(
                _query_plan_payload(inner_query_plan)
            ),
        }
    )
    return replace(
        outer_cluster_plan,
        queried_window_ids=queried_ids,
        queried_roles=roles,
        queried_labels=labels,
        pseudo_labels={},
        pseudo_confidence={},
        metadata=metadata,
        frozen_pseudo_result=None,
        pseudo_candidate_rankings={},
    )


def apply_frozen_query_annotations(
    cluster_plan: QueryPlan,
    annotations: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> QueryPlan:
    """Replace feature-dependent queries with the manifest-frozen cases."""

    if len(annotations) != int(budget):
        raise ValueError(
            "formal query annotations must contain exactly budget cases"
        )
    queried_ids = tuple(
        str(row.get("native_case_id", "")).strip()
        for row in annotations
    )
    if (
        any(not case_id for case_id in queried_ids)
        or len(set(queried_ids)) != int(budget)
    ):
        raise ValueError(
            "formal query annotations require unique native case IDs"
        )
    unknown = sorted(
        set(queried_ids).difference(
            str(value) for value in cluster_plan.window_clusters
        )
    )
    if unknown:
        raise ValueError(
            "formal queried cases are absent from inner training: %s"
            % unknown[:10]
        )
    labels = {}
    for row, case_id in zip(annotations, queried_ids):
        raw_targets = row.get("targets")
        if not isinstance(raw_targets, (list, tuple, set)):
            raise ValueError("formal query targets must be a sequence")
        targets = tuple(
            sorted(
                {
                    str(target).strip()
                    for target in raw_targets
                    if str(target).strip()
                }
            )
        )
        if not targets:
            raise ValueError(
                "formal queried case %s has no targets" % case_id
            )
        labels[case_id] = targets
    metadata = dict(cluster_plan.metadata)
    metadata.update(
        {
            "budget": int(budget),
            "effective_queried_window_count": len(queried_ids),
            "query_plan_source": "formal_baseline_manifest",
            "manifest_query_annotation_sha256": _semantic_sha256(
                [
                    {
                        "native_case_id": case_id,
                        "targets": list(labels[case_id]),
                    }
                    for case_id in queried_ids
                ]
            ),
        }
    )
    return replace(
        cluster_plan,
        queried_window_ids=queried_ids,
        queried_roles={
            case_id: "formal_manifest_query" for case_id in queried_ids
        },
        queried_labels=labels,
        pseudo_labels={},
        pseudo_confidence={},
        metadata=metadata,
        frozen_pseudo_result=None,
        pseudo_candidate_rankings={},
    )


def build_per_case_rankings(
    scored_rows: pd.DataFrame,
    windows: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Build fault-only rankings with complete target sets and score evidence."""

    required_scores = {"window_id", "entity_id", "score", "raw_score"}
    missing_scores = sorted(required_scores.difference(scored_rows.columns))
    if missing_scores:
        raise ValueError("scored rows missing ranking fields: %s" % missing_scores)
    required_windows = {"window_id", "window_kind", "positive_ids_list"}
    missing_windows = sorted(required_windows.difference(windows.columns))
    if missing_windows:
        raise ValueError("windows missing ranking fields: %s" % missing_windows)
    groups = {
        str(window_id): group.copy()
        for window_id, group in scored_rows.groupby("window_id", sort=False)
    }
    records = []
    fault_windows = windows[
        windows["window_kind"].astype(str) == "fault"
    ].copy()
    for window in fault_windows.sort_values(["window_id"]).itertuples(index=False):
        case_id = str(window.window_id)
        targets = sorted(
            {
                str(value)
                for value in getattr(window, "positive_ids_list", ())
                if str(value)
            }
        )
        group = groups.get(case_id)
        if not targets or group is None or group.empty:
            continue
        ordered = group.sort_values(
            ["score", "raw_score", "entity_id"],
            ascending=[False, False, True],
        )
        ranked_entities = [
            {
                "entity_id": str(row.entity_id),
                "score": float(row.score),
                "raw_score": float(row.raw_score),
            }
            for row in ordered.itertuples(index=False)
        ]
        records.append(
            {
                "case_id": case_id,
                "targets": targets,
                "ranking": [
                    row["entity_id"] for row in ranked_entities
                ],
                "ranked_entities": ranked_entities,
            }
        )
    return records


def recompute_hit_at_k(
    rankings: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    """Recompute Hit@k directly from persisted per-case ranking records."""

    normalized_ks = tuple(sorted({int(value) for value in ks}))
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("ranking cutoffs must be positive")
    hits = {value: 0 for value in normalized_ks}
    for record in rankings:
        targets = {str(value) for value in record.get("targets", ())}
        ranking = [str(value) for value in record.get("ranking", ())]
        for cutoff in normalized_ks:
            hits[cutoff] += int(bool(targets.intersection(ranking[:cutoff])))
    count = len(rankings)
    result = {
        "hit_at_%d" % cutoff: (
            float(hits[cutoff]) / float(count) if count else 0.0
        )
        for cutoff in normalized_ks
    }
    result["scored_cases"] = count
    return result


def _query_plan_payload(query_plan: QueryPlan) -> Dict[str, Any]:
    return {
        "dataset": str(query_plan.dataset),
        "normal_cluster_id": int(query_plan.normal_cluster_id),
        "window_clusters": {
            str(key): int(value)
            for key, value in sorted(query_plan.window_clusters.items())
        },
        "queried_window_ids": [
            str(value) for value in query_plan.queried_window_ids
        ],
        "queried_roles": {
            str(key): str(value)
            for key, value in sorted(query_plan.queried_roles.items())
        },
        "queried_labels": {
            str(key): list(value)
            for key, value in sorted(query_plan.queried_labels.items())
        },
        "pseudo_labels": {
            str(key): list(value)
            for key, value in sorted(query_plan.pseudo_labels.items())
        },
        "pseudo_confidence": {
            str(key): float(value)
            for key, value in sorted(query_plan.pseudo_confidence.items())
        },
        "metadata": dict(query_plan.metadata),
    }


def query_plan_sha256(query_plan: QueryPlan) -> str:
    return _semantic_sha256(_query_plan_payload(query_plan))


def prepare_optimization_matched_arms(
    raw_pool_manifest_path: Path,
    *,
    query_plan: QueryPlan,
    label_mode: str,
    class_conditional_coverage: float,
) -> MatchedPseudoArms:
    """Bind optimization arms to one frozen budget-30 query plan and pool."""

    if query_plan.pseudo_labels or query_plan.pseudo_confidence:
        raise ValueError(
            "frozen optimization query plan must not contain pseudo labels"
        )
    queried = tuple(str(value) for value in query_plan.queried_window_ids)
    if len(queried) != 30 or len(set(queried)) != 30:
        raise ValueError(
            "optimization matched arms require exactly 30 queried cases"
        )
    pool = load_raw_pseudo_pool(raw_pool_manifest_path)
    resolution = resolve_dataset(query_plan.dataset)
    if resolution.canonical_id != pool.canonical_dataset_id:
        raise ValueError("query plan and raw pseudo pool dataset mismatch")
    expected_hash = query_plan_sha256(query_plan)
    if pool.query_plan_sha256 != expected_hash:
        raise ValueError("query plan hash does not match raw pseudo pool")
    if pool.query_case_ids != tuple(sorted(queried)):
        raise ValueError("query case IDs do not match raw pseudo pool")
    return derive_matched_pseudo_arms(
        pool,
        label_mode=label_mode,
        class_conditional_coverage=class_conditional_coverage,
    )


def _strip_label_bearing_tables(
    tables: FeatureBundleTables,
) -> FeatureBundleTables:
    window_columns = [
        column
        for column in tables.windows.columns
        if not is_label_bearing_field(column)
    ]
    entity_columns = [
        column
        for column in tables.entity_features.columns
        if not is_label_bearing_field(column)
    ]
    feature_columns = [
        column
        for column in tables.feature_columns
        if column in entity_columns and not is_label_bearing_field(column)
    ]
    return FeatureBundleTables(
        dataset=tables.dataset,
        windows=tables.windows[window_columns].copy(),
        entity_features=tables.entity_features[entity_columns].copy(),
        metadata={},
        feature_columns=feature_columns,
    )


def _model_config(normal_policy: str, *, oracle_full: bool = False) -> SemiSupervisedConfig:
    return SemiSupervisedConfig.from_dict(
        {
            "supervision_mode": (
                "oracle_full" if oracle_full else "semi_supervised"
            ),
            "normal_training_policy": normal_policy,
            "query_strategy": "sequential_medoid_noise_boundary",
            "propagate_mode": "disabled",
            "model_backend": "pairwise_linear",
        }
    )


def _feature_views(
    tables: FeatureBundleTables,
    clustered_windows: pd.DataFrame,
) -> Tuple[Any, ...]:
    safe_tables = _strip_label_bearing_tables(tables)
    numeric_columns = [
        column
        for column in safe_tables.feature_columns
        if column in safe_tables.entity_features.columns
        and pd.api.types.is_numeric_dtype(
            safe_tables.entity_features[column]
        )
    ]
    return sanitize_feature_views(
        windows=safe_tables.windows,
        entity_features=safe_tables.entity_features,
        entity_feature_columns=numeric_columns,
        window_feature_columns=(),
        clustered_windows=clustered_windows,
    )


def _label_map(windows: pd.DataFrame) -> Dict[str, Tuple[str, ...]]:
    return {
        str(row.window_id): tuple(
            sorted(
                {
                    str(value)
                    for value in getattr(row, "positive_ids_list", ())
                    if str(value)
                }
            )
        )
        for row in windows[
            windows["window_kind"].astype(str) == "fault"
        ].itertuples(index=False)
    }


def _candidate_rankings(
    raw_pool: Mapping[str, Any],
    emitted_window_ids: Sequence[str],
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    emitted = {str(value) for value in emitted_window_ids}
    return {
        str(row["window_id"]): {
            "raw_pool_telemetry": tuple(row["candidate_ranking"])
        }
        for row in raw_pool["predictions"]
        if str(row["window_id"]) in emitted
    }


def _save_evaluation(
    root: Path,
    *,
    model: Any,
    scored_rows: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
    config: SemiSupervisedConfig,
    extra: Mapping[str, Any],
) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if model.training_frame is None:
        raise RuntimeError("semi-supervised model did not expose its training frame")
    training_frame = model.training_frame.copy()
    rankings = build_per_case_rankings(scored_rows, evaluation_windows)
    metrics = recompute_hit_at_k(rankings)
    fault_count = int(
        (
            evaluation_windows["window_kind"].astype(str) == "fault"
        ).sum()
    )
    normal_count = int(
        (
            evaluation_windows["window_kind"].astype(str) == "normal"
        ).sum()
    )
    if metrics["scored_cases"] != fault_count:
        raise ValueError(
            "fault-only evaluation coverage mismatch: %d != %d"
            % (metrics["scored_cases"], fault_count)
        )
    training_frame.to_csv(root / "training_frame.csv", index=False)
    scored_rows.to_csv(root / "predictions.csv", index=False)
    _write_jsonl(root / "per_case_rankings.jsonl", rankings)
    _write_json(root / "query_plan.json", _query_plan_payload(model.query_plan))
    _write_json(root / "model_config.json", config.to_dict())
    result = {
        **metrics,
        "evaluation_fault_cases": fault_count,
        "evaluation_normal_windows_excluded": normal_count,
        "queried_case_count": len(model.query_plan.queried_window_ids),
        "queried_case_ids": list(model.query_plan.queried_window_ids),
        "training_fault_windows": int(
            training_frame.loc[
                training_frame["window_kind"].astype(str) == "fault",
                "window_id",
            ].nunique()
        ),
        "training_normal_windows": int(
            training_frame.loc[
                training_frame["window_kind"].astype(str) == "normal",
                "window_id",
            ].nunique()
        ),
        "training_rows": len(training_frame),
        "label_source_counts": {
            str(key): int(value)
            for key, value in training_frame["label_source"]
            .value_counts()
            .to_dict()
            .items()
        },
        "per_case_rankings_sha256": hashlib.sha256(
            (root / "per_case_rankings.jsonl").read_bytes()
        ).hexdigest(),
        **dict(extra),
    }
    for name in ("hit_at_1", "hit_at_3", "hit_at_5"):
        if not math.isfinite(float(result[name])):
            raise ValueError("non-finite smoke metric: %s" % name)
    _write_json(root / "metrics.json", result)
    return result


def _run_budgeted_arm(
    *,
    arm: str,
    outer_train_tables: FeatureBundleTables,
    safe_outer_train_tables: FeatureBundleTables,
    safe_test_tables: FeatureBundleTables,
    test_windows: pd.DataFrame,
    feature_root: Path,
    normal_policy: str,
    query_plan: QueryPlan,
    raw_pool: Mapping[str, Any],
    selector: Mapping[str, Any],
    output_root: Path,
    seed: int,
    budget: int,
) -> Dict[str, Any]:
    config = _model_config(normal_policy, oracle_full=(arm == "oracle_full"))
    training_tables = safe_outer_train_tables
    plan_override = query_plan
    pseudo_result = None
    if arm == "oracle_full":
        training_tables = outer_train_tables
        plan_override = None
    elif arm in {"all_pseudo", "selected_pseudo"}:
        pseudo_result = strategy_result_from_raw_pool(
            raw_pool,
            selector=selector,
            mode=arm,
        )
        emitted_ids = [
            prediction.window_id
            for prediction in pseudo_result.predictions
            if prediction.emitted
        ]
        plan_override = attach_frozen_pseudo_result(
            query_plan,
            pseudo_result,
            candidate_rankings=_candidate_rankings(
                raw_pool,
                emitted_ids,
            ),
        )
    fault_count = int(
        (
            outer_train_tables.windows["window_kind"].astype(str)
            == "fault"
        ).sum()
    )
    model, clustered, _ = fit_semisupervised_ranker_on_tables(
        tables=training_tables,
        feature_root=feature_root,
        budget=(fault_count if arm == "oracle_full" else int(budget)),
        random_state=int(seed),
        model_config=config,
        query_plan_override=plan_override,
    )
    scored = model.score_entity_features(safe_test_tables.entity_features)
    output_root.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(output_root / "clustered_windows.csv", index=False)
    result = _save_evaluation(
        output_root,
        model=model,
        scored_rows=scored,
        evaluation_windows=test_windows,
        config=config,
        extra={
            "arm": arm,
            "normal_training_policy": normal_policy,
            "budget": None if arm == "oracle_full" else int(budget),
            "upper_bound_reference": arm == "oracle_full",
            "raw_pool_sha256": (
                None if pseudo_result is None else pseudo_result.input_hash
            ),
            "pseudo_emitted_count": (
                0
                if pseudo_result is None
                else sum(
                    prediction.emitted
                    for prediction in pseudo_result.predictions
                )
            ),
        },
    )
    if arm != "oracle_full":
        if tuple(model.query_plan.queried_window_ids) != tuple(
            query_plan.queried_window_ids
        ):
            raise ValueError("budgeted arm changed the frozen query plan")
        if result["queried_case_count"] != int(budget):
            raise ValueError("budgeted arm did not preserve budget 30")
    return result


def run_bounded_semi_supervised_smoke(
    *,
    feature_root: Path,
    output_root: Path,
    canonical_dataset_id: str,
    budget: int = 30,
    fault_limit: int = 60,
    normal_limit: int = 20,
    outer_test_ratio: float = 0.30,
    inner_val_ratio: float = 0.20,
    seed: int = 42,
    selector_seed: Optional[int] = None,
    downstream_seed: Optional[int] = None,
    frozen_query_annotations: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Run one bounded dataset smoke with a frozen normal policy and four arms."""

    from nexusrcl_rebuild.evaluation.splits import (
        build_nested_chronological_splits,
    )

    started = time.time()
    selector_seed_value = int(
        seed if selector_seed is None else selector_seed
    )
    downstream_seed_value = int(
        seed if downstream_seed is None else downstream_seed
    )
    resolution = resolve_dataset(canonical_dataset_id)
    if resolution.requested_id != resolution.canonical_id:
        raise ValueError("new smoke runs require canonical dataset identifiers")
    output_root = _prepare_output_root(Path(output_root))
    tracker = ProgressTracker(output_root, total_units=7)
    tracker.initialize("loading_features")
    try:
        full_tables = load_feature_bundle_tables(
            Path(feature_root),
            resolution.legacy_alias,
        )
        bounded_ids = select_bounded_window_ids(
            full_tables.windows,
            fault_limit=int(fault_limit),
            normal_limit=int(normal_limit),
        )
        tables = subset_feature_bundle_tables(full_tables, bounded_ids)
        split = build_nested_chronological_splits(
            tables.windows,
            outer_test_ratio=float(outer_test_ratio),
            inner_val_ratio=float(inner_val_ratio),
        )
        outer_train_tables = subset_feature_bundle_tables(
            tables, split.outer_train_window_ids
        )
        test_tables = subset_feature_bundle_tables(
            tables, split.outer_test_window_ids
        )
        inner_train_tables = subset_feature_bundle_tables(
            outer_train_tables, split.inner_train_window_ids
        )
        val_tables = subset_feature_bundle_tables(
            outer_train_tables, split.inner_val_window_ids
        )
        safe_outer_train = _strip_label_bearing_tables(outer_train_tables)
        safe_test = _strip_label_bearing_tables(test_tables)
        safe_inner_train = _strip_label_bearing_tables(inner_train_tables)
        safe_val = _strip_label_bearing_tables(val_tables)

        fault_counts = {
            "inner_train": int(
                (
                    inner_train_tables.windows["window_kind"].astype(str)
                    == "fault"
                ).sum()
            ),
            "inner_val": int(
                (
                    val_tables.windows["window_kind"].astype(str) == "fault"
                ).sum()
            ),
            "outer_train": int(
                (
                    outer_train_tables.windows["window_kind"].astype(str)
                    == "fault"
                ).sum()
            ),
            "outer_test": int(
                (
                    test_tables.windows["window_kind"].astype(str) == "fault"
                ).sum()
            ),
        }
        if fault_counts["inner_train"] < int(budget):
            raise ValueError("bounded inner training has fewer faults than budget")

        base_config = _model_config("fault_only")
        inner_query_plan, inner_clustered = build_query_plan(
            inner_train_tables,
            budget=int(budget),
            config=base_config,
            random_state=selector_seed_value,
        )
        if frozen_query_annotations:
            inner_query_plan = apply_frozen_query_annotations(
                inner_query_plan,
                frozen_query_annotations,
                budget=int(budget),
            )
        if len(inner_query_plan.queried_window_ids) != int(budget):
            raise ValueError("inner query plan did not contain exactly budget cases")
        query_plan_hash = _semantic_sha256(
            _query_plan_payload(inner_query_plan)
        )
        _write_json(
            output_root / "split.json",
            {
                "canonical_dataset_id": resolution.canonical_id,
                "legacy_alias": resolution.legacy_alias,
                "bounded_window_ids": list(bounded_ids),
                "outer_train_window_ids": list(split.outer_train_window_ids),
                "outer_test_window_ids": list(split.outer_test_window_ids),
                "inner_train_window_ids": list(split.inner_train_window_ids),
                "inner_val_window_ids": list(split.inner_val_window_ids),
                "fault_counts": fault_counts,
                "outer_test_normal_windows": int(
                    (
                        test_tables.windows["window_kind"].astype(str)
                        == "normal"
                    ).sum()
                ),
            },
        )
        _write_json(
            output_root / "inner_query_plan.json",
            _query_plan_payload(inner_query_plan),
        )
        inner_clustered.to_csv(
            output_root / "inner_clustered_windows.csv",
            index=False,
        )

        normal_ablation = {}
        ablation_query_ids = {}
        for policy in NORMAL_POLICIES:
            tracker.transition(
                "normal_policy_ablation",
                current_unit=policy,
            )
            config = _model_config(policy)
            model, clustered, _ = fit_semisupervised_ranker_on_tables(
                tables=safe_inner_train,
                feature_root=Path(feature_root),
                budget=int(budget),
                random_state=downstream_seed_value,
                model_config=config,
                query_plan_override=inner_query_plan,
            )
            scored = model.score_entity_features(safe_val.entity_features)
            policy_root = output_root / "normal_policy" / policy
            policy_root.mkdir(parents=True, exist_ok=True)
            clustered.to_csv(
                policy_root / "clustered_windows.csv",
                index=False,
            )
            normal_ablation[policy] = _save_evaluation(
                policy_root,
                model=model,
                scored_rows=scored,
                evaluation_windows=val_tables.windows,
                config=config,
                extra={
                    "partition": "inner_validation",
                    "normal_training_policy": policy,
                    "query_plan_sha256": query_plan_hash,
                },
            )
            ablation_query_ids[policy] = tuple(
                model.query_plan.queried_window_ids
            )
            tracker.advance(policy)
        if len(set(ablation_query_ids.values())) != 1:
            raise ValueError("normal-policy ablation changed queried case IDs")
        chosen_policy = choose_normal_policy(
            {
                policy: {"A@1": normal_ablation[policy]["hit_at_1"]}
                for policy in NORMAL_POLICIES
            }
        )
        policy_artifact = {
            "schema_version": "rcl-normal-policy-selection-v1",
            "canonical_dataset_id": resolution.canonical_id,
            "selection_partition": "inner_validation_fault_only",
            "query_plan_sha256": query_plan_hash,
            "queried_case_ids": list(inner_query_plan.queried_window_ids),
            "metrics": {
                policy: {
                    name: normal_ablation[policy][name]
                    for name in (
                        "hit_at_1",
                        "hit_at_3",
                        "hit_at_5",
                        "scored_cases",
                        "training_fault_windows",
                        "training_normal_windows",
                    )
                }
                for policy in NORMAL_POLICIES
            },
            "selected_policy": chosen_policy,
            "selection_rule": (
                "with_normal_class only on strict inner-validation Hit@1 gain; "
                "otherwise fault_only"
            ),
        }
        policy_artifact["artifact_sha256"] = _semantic_sha256(policy_artifact)
        _write_json(
            output_root / "frozen_normal_policy.json",
            policy_artifact,
        )

        outer_cluster_plan, outer_clustered = build_query_plan(
            outer_train_tables,
            budget=int(budget),
            config=_model_config(chosen_policy),
            random_state=selector_seed_value,
        )
        outer_query_plan = freeze_outer_query_plan(
            inner_query_plan,
            outer_cluster_plan,
            budget=int(budget),
        )
        _write_json(
            output_root / "outer_query_plan.json",
            _query_plan_payload(outer_query_plan),
        )
        outer_clustered.to_csv(
            output_root / "outer_clustered_windows.csv",
            index=False,
        )

        inner_val_views = _feature_views(val_tables, outer_clustered[
            outer_clustered["window_id"].astype(str).isin(
                set(val_tables.windows["window_id"].astype(str))
            )
        ].copy())
        validation_pool = build_raw_pseudo_pool(
            inner_val_views,
            queried_window_ids=inner_query_plan.queried_window_ids,
        )
        threshold_grid = sorted(
            {
                float(row["confidence"])
                for row in validation_pool["predictions"]
            }
        )
        validation_selector = select_raw_pseudo_pool(
            validation_pool,
            authoritative_labels=_label_map(val_tables.windows),
            thresholds=threshold_grid,
        )
        _write_json(
            output_root / "inner_validation_raw_pool.json",
            validation_pool,
        )
        _write_json(
            output_root / "inner_validation_selector.json",
            validation_selector,
        )

        outer_views = _feature_views(outer_train_tables, outer_clustered)
        raw_pool = build_raw_pseudo_pool(
            outer_views,
            queried_window_ids=outer_query_plan.queried_window_ids,
        )
        selected_threshold = float(
            validation_selector["selected_threshold"]
        )
        selected_ids = sorted(
            row["window_id"]
            for row in raw_pool["predictions"]
            if float(row["confidence"]) >= selected_threshold
        )
        selector = {
            "schema_version": "rcl-pseudo-selector-v1",
            "raw_pool_sha256": raw_pool["pool_sha256"],
            "inner_validation_raw_pool_sha256": validation_pool[
                "pool_sha256"
            ],
            "selection_partition": "inner_validation",
            "selected_threshold": selected_threshold,
            "selected_window_ids": selected_ids,
            "threshold_scoreboard": validation_selector[
                "threshold_scoreboard"
            ],
        }
        selector["selector_sha256"] = _semantic_sha256(selector)
        _write_json(output_root / "raw_pseudo_pool.json", raw_pool)
        _write_json(output_root / "frozen_pseudo_selector.json", selector)

        tracker.transition(
            "selector_qualification",
            current_unit="selector_seed_%d" % selector_seed_value,
        )
        inner_views = _feature_views(inner_train_tables, inner_clustered)
        inner_raw_pool = build_raw_pseudo_pool(
            inner_views,
            queried_window_ids=inner_query_plan.queried_window_ids,
        )
        inner_selected_ids = sorted(
            row["window_id"]
            for row in inner_raw_pool["predictions"]
            if float(row["confidence"]) >= selected_threshold
        )
        inner_selector = {
            "schema_version": "rcl-pseudo-selector-v1",
            "raw_pool_sha256": inner_raw_pool["pool_sha256"],
            "selection_partition": "inner_validation",
            "selected_threshold": selected_threshold,
            "selected_window_ids": inner_selected_ids,
            "source_selector_sha256": selector["selector_sha256"],
        }
        inner_selector["selector_sha256"] = _semantic_sha256(
            inner_selector
        )
        qualification_selected = _run_budgeted_arm(
            arm="selected_pseudo",
            outer_train_tables=inner_train_tables,
            safe_outer_train_tables=safe_inner_train,
            safe_test_tables=safe_val,
            test_windows=val_tables.windows,
            feature_root=Path(feature_root),
            normal_policy=chosen_policy,
            query_plan=inner_query_plan,
            raw_pool=inner_raw_pool,
            selector=inner_selector,
            output_root=(
                output_root
                / "selector_qualification"
                / "selected_pseudo"
            ),
            seed=downstream_seed_value,
            budget=int(budget),
        )
        query_inner_hit1 = float(
            normal_ablation[chosen_policy]["hit_at_1"]
        )
        selected_inner_hit1 = float(
            qualification_selected["hit_at_1"]
        )
        selected_pseudo_eligibility = {
            "eligible": bool(inner_selected_ids)
            and selected_inner_hit1 > query_inner_hit1,
            "partition": "inner_validation_fault_only",
            "selector_seed": selector_seed_value,
            "downstream_seed": downstream_seed_value,
            "query_only_hit_at_1": query_inner_hit1,
            "selected_pseudo_hit_at_1": selected_inner_hit1,
            "selected_pseudo_count": len(inner_selected_ids),
            "rule": (
                "nonempty_selected_pseudo_and_strict_hit_at_1_"
                "improvement_over_matched_query_only"
            ),
            "outer_test_access": "forbidden",
        }
        _write_json(
            output_root / "inner_training_raw_pool.json",
            inner_raw_pool,
        )
        _write_json(
            output_root / "inner_training_selector.json",
            inner_selector,
        )
        _write_json(
            output_root / "selected_pseudo_eligibility.json",
            selected_pseudo_eligibility,
        )
        tracker.advance("selector_seed_%d" % selector_seed_value)

        arms = {}
        for arm in ARM_IDS:
            tracker.transition("arms", current_unit=arm)
            arms[arm] = _run_budgeted_arm(
                arm=arm,
                outer_train_tables=outer_train_tables,
                safe_outer_train_tables=safe_outer_train,
                safe_test_tables=safe_test,
                test_windows=test_tables.windows,
                feature_root=Path(feature_root),
                normal_policy=chosen_policy,
                query_plan=outer_query_plan,
                raw_pool=raw_pool,
                selector=selector,
                output_root=output_root / "arms" / arm,
                seed=downstream_seed_value,
                budget=int(budget),
            )
            tracker.advance(arm)
        budgeted_query_plans = {
            tuple(arms[arm]["queried_case_ids"])
            for arm in ("query_only", "all_pseudo", "selected_pseudo")
        }
        if len(budgeted_query_plans) != 1:
            raise ValueError("budgeted arms changed queried case IDs")
        all_result = strategy_result_from_raw_pool(
            raw_pool, selector=selector, mode="all_pseudo"
        )
        selected_result = strategy_result_from_raw_pool(
            raw_pool, selector=selector, mode="selected_pseudo"
        )
        if all_result.input_hash != selected_result.input_hash:
            raise ValueError("all and selected pseudo arms changed raw pool")

        oracle_hit1 = float(arms["oracle_full"]["hit_at_1"])
        for arm in ("query_only", "all_pseudo", "selected_pseudo"):
            hit1 = float(arms[arm]["hit_at_1"])
            arms[arm]["oracle_hit_at_1_gap"] = oracle_hit1 - hit1
            arms[arm]["oracle_hit_at_1_retention"] = (
                hit1 / oracle_hit1 if oracle_hit1 > 0.0 else None
            )
            _write_json(
                output_root / "arms" / arm / "metrics.json",
                arms[arm],
            )

        summary = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "canonical_dataset_id": resolution.canonical_id,
            "legacy_alias": resolution.legacy_alias,
            "evidence_tier": "smoke",
            "budget": int(budget),
            "seed": downstream_seed_value,
            "selector_seed": selector_seed_value,
            "downstream_seed": downstream_seed_value,
            "fault_limit": int(fault_limit),
            "normal_limit": int(normal_limit),
            "outer_test_ratio": float(outer_test_ratio),
            "inner_val_ratio": float(inner_val_ratio),
            "fault_counts": fault_counts,
            "frozen_normal_policy": chosen_policy,
            "normal_policy_artifact_sha256": policy_artifact[
                "artifact_sha256"
            ],
            "raw_pool_sha256": raw_pool["pool_sha256"],
            "selector_sha256": selector["selector_sha256"],
            "raw_pseudo_count": len(raw_pool["predictions"]),
            "selected_pseudo_count": len(selected_ids),
            "selected_pseudo_eligibility": selected_pseudo_eligibility,
            "arms": arms,
            "started_at_utc": datetime.fromtimestamp(
                started, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.time() - started,
        }
        _write_json(output_root / "summary.json", summary)
        completion = {
            "schema_version": "rcl-smoke-completion-v1",
            "status": "complete",
            "canonical_dataset_id": resolution.canonical_id,
            "summary_sha256": hashlib.sha256(
                (output_root / "summary.json").read_bytes()
            ).hexdigest(),
            "validated_units": list(NORMAL_POLICIES) + list(ARM_IDS),
            "completed_at_utc": _utc_now(),
        }
        _write_json(output_root / "COMPLETED.json", completion)
        (output_root / "all.done").write_text(
            hashlib.sha256(
                (output_root / "COMPLETED.json").read_bytes()
            ).hexdigest()
            + "\n",
            encoding="utf-8",
        )
        tracker.transition("complete")
        return summary
    except Exception as exc:
        _write_json(
            output_root / "error.json",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failed_at_utc": _utc_now(),
            },
        )
        (output_root / ".failed").write_text(
            "%s: %s\n" % (type(exc).__name__, exc),
            encoding="utf-8",
        )
        if (output_root / "all.done").exists():
            (output_root / "all.done").unlink()
        if (output_root / "COMPLETED.json").exists():
            (output_root / "COMPLETED.json").unlink()
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded semi-supervised RCL smoke experiment."
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=("rcabench", "aiops25"),
        required=True,
    )
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--fault-limit", type=int, default=60)
    parser.add_argument("--normal-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-seed", type=int, default=None)
    parser.add_argument("--downstream-seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    if os.name == "nt":
        raise SystemExit(
            "Semi-supervised dataset/model execution is server-only; "
            "run this command on wangrunzhou@10.10.1.226."
        )
    args = _parse_args()
    summary = run_bounded_semi_supervised_smoke(
        feature_root=args.feature_root,
        output_root=args.output_root,
        canonical_dataset_id=args.dataset,
        budget=args.budget,
        fault_limit=args.fault_limit,
        normal_limit=args.normal_limit,
        seed=args.seed,
        selector_seed=args.selector_seed,
        downstream_seed=args.downstream_seed,
    )
    print(
        "%s complete: %s"
        % (summary["canonical_dataset_id"], args.output_root)
    )
    return 0


__all__ = [
    "ARM_IDS",
    "NORMAL_POLICIES",
    "SMOKE_SCHEMA_VERSION",
    "apply_frozen_query_annotations",
    "build_per_case_rankings",
    "freeze_outer_query_plan",
    "prepare_optimization_matched_arms",
    "query_plan_sha256",
    "recompute_hit_at_k",
    "run_bounded_semi_supervised_smoke",
    "select_bounded_window_ids",
]


if __name__ == "__main__":
    raise SystemExit(main())
