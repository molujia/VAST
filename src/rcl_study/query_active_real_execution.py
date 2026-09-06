"""Real inner-validation execution for query-only active-learning screening."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from .datasets import resolve_dataset
from .query_active_learning import select_query_cases
from .query_active_deepening import (
    fault_type_oracle_from_metadata_json,
    score_t1_t2_generic,
    select_true_fault_type_oracle_cases,
)
from .unseen_fault_type_generalization import (
    build_fault_type_inventory,
    score_fault_type_slices,
)
from .rcabench_query_only_sota import (
    RCABENCH_SOURCE_PATH,
    SPLIT_SEED,
    build_split_identity,
    score_t1_t2_from_rankings,
    semantic_sha256,
)


DEFAULT_SEMI_FEATURE_ROOT = Path(
    "${NEXUSRCL_REBUILD_ROOT}/"
    "artifacts/window_feature_artifacts_hd134_stage1"
)

DEEPENING_EXECUTION_DATASETS = frozenset(
    {"rcabench", "aiops2022_pre", "aiops25"}
)


def validate_deepening_execution_dataset(identifier: str):
    """Resolve a dataset admitted by the truthful current-method runner."""

    resolution = resolve_dataset(identifier)
    if resolution.canonical_id not in DEEPENING_EXECUTION_DATASETS:
        raise ValueError(
            "unsupported deepening execution dataset: %s"
            % resolution.canonical_id
        )
    return resolution


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError("value is not JSON serializable: %r" % (value,))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-%d" % os.getpid())
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
    os.replace(str(temporary), str(destination))


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _clone_feature_bundle_tables(
    tables: Any,
    *,
    windows: pd.DataFrame,
    entity_features: pd.DataFrame,
    metadata: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> Any:
    """Clone the half-supervise FeatureBundleTables dataclass or a test double."""

    payload = {
        "dataset": getattr(tables, "dataset"),
        "windows": windows,
        "entity_features": entity_features,
        "metadata": (
            dict(getattr(tables, "metadata", {}) or {})
            if metadata is None
            else dict(metadata)
        ),
        "feature_columns": (
            list(getattr(tables, "feature_columns", ()) or ())
            if feature_columns is None
            else list(feature_columns)
        ),
    }
    try:
        return replace(tables, **payload)
    except TypeError:
        return SimpleNamespace(**payload)


def _clone_query_plan(query_plan: Any, **updates: Any) -> Any:
    """Clone the half-supervise QueryPlan dataclass or a test double."""

    try:
        return replace(query_plan, **updates)
    except TypeError:
        payload = {
            "dataset": getattr(query_plan, "dataset"),
            "normal_cluster_id": getattr(query_plan, "normal_cluster_id"),
            "window_clusters": getattr(query_plan, "window_clusters"),
            "queried_window_ids": getattr(query_plan, "queried_window_ids"),
            "queried_roles": getattr(query_plan, "queried_roles"),
            "queried_labels": getattr(query_plan, "queried_labels"),
            "pseudo_labels": getattr(query_plan, "pseudo_labels"),
            "pseudo_confidence": getattr(query_plan, "pseudo_confidence"),
            "metadata": getattr(query_plan, "metadata"),
        }
        payload.update(updates)
        return SimpleNamespace(**payload)


def _target_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = value.split(";")
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = ()
    targets = sorted({str(item).strip() for item in raw if str(item).strip()})
    if not targets:
        raise ValueError("selected query case has no annotation targets")
    return targets


def bounded_normal_policy(normal_policy: str) -> str:
    """Map query-active manifest policy IDs to the bounded runner policy IDs."""

    value = str(normal_policy)
    if value == "fault_only":
        return "fault_only"
    if value == "with_normal_windows":
        return "with_normal_class"
    raise ValueError("unsupported normal policy: %s" % normal_policy)


def _training_seed(unit: Mapping[str, Any]) -> int:
    """Resolve an explicit training seed while preserving historical default 42."""

    return int(unit.get("training_seed", 42))


def _requested_evaluation_views(unit: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve requested views while preserving historical T1/T2 behavior."""

    scope = str(unit.get("evaluation_scope", "T1_T2")).strip()
    if scope == "T1_only":
        return ("T1",)
    if scope == "T1_T2":
        return ("T1", "T2")
    raise ValueError("unsupported evaluation scope: %s" % scope)


def _score_t1_only_generic(
    *,
    outer_test_case_ids: Sequence[Any],
    queried_case_ids: Sequence[Any],
    ranking_rows: Sequence[Mapping[str, Any]],
    canonical_dataset_id: str | None = None,
) -> dict[str, Any]:
    """Compute formal outer-test metrics without requesting diagnostic T2."""

    score = score_t1_t2_generic(
        outer_train_case_ids=(),
        outer_test_case_ids=outer_test_case_ids,
        queried_case_ids=queried_case_ids,
        ranking_rows=ranking_rows,
        extra_t2_case_ids=(),
        canonical_dataset_id=canonical_dataset_id,
    )
    return {
        "schema_version": "rcl-query-active-deepening-t1-only-score-v1",
        "evaluation_scope": "T1_only",
        "partitions": {"T1": list(score["partitions"]["T1"])},
        "queried_case_ids": list(score["queried_case_ids"]),
        "T1": dict(score["T1"]),
        "T2": {"status": "not_requested"},
        "t1_t2_warnings": [],
        "per_case_ranking_evidence": {
            "T1": list(score["T1"]["per_case"]),
        },
    }


def _per_case_evidence_by_view(score_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only requested per-case views without touching absent metrics."""

    declared = score_payload.get("per_case_ranking_evidence")
    if declared is not None:
        return dict(declared)
    return {
        "T1": list(dict(score_payload["T1"])["per_case"]),
        "T2": list(dict(score_payload["T2"])["per_case"]),
    }


def _fit_with_compatible_cluster_override(
    fit_function: Any,
    *,
    clustered_windows_override: Any,
    require_clustered_override: bool,
    **fit_kwargs: Any,
) -> Any:
    """Use the optional cluster override only when the installed API supports it."""

    parameters = inspect.signature(fit_function).parameters
    supports_override = "clustered_windows_override" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if require_clustered_override and (
        clustered_windows_override is None or not supports_override
    ):
        raise TypeError(
            "MPCA augmentation requires clustered_windows_override support"
        )
    if supports_override and clustered_windows_override is not None:
        fit_kwargs["clustered_windows_override"] = clustered_windows_override
    return fit_function(**fit_kwargs)


def build_active_query_candidate_rows(
    *,
    canonical_dataset_id: str,
    windows: pd.DataFrame,
    entity_features: pd.DataFrame,
    clustered_windows: pd.DataFrame,
    feature_columns: Sequence[str],
    baseline_query_ids: Sequence[str],
) -> list[Dict[str, Any]]:
    """Build label-free selector candidates from inner-training feature tables."""

    dataset = resolve_dataset(canonical_dataset_id).canonical_id
    if "window_id" not in windows or "window_kind" not in windows:
        raise ValueError("windows missing active-query candidate fields")
    if "window_id" not in entity_features:
        raise ValueError("entity features missing window_id")
    if not {"window_id", "cluster_id"}.issubset(clustered_windows.columns):
        raise ValueError("clustered windows missing cluster_id")
    numeric_columns = [
        str(column)
        for column in feature_columns
        if column in entity_features.columns
        and pd.api.types.is_numeric_dtype(entity_features[column])
    ]
    if not numeric_columns:
        raise ValueError("active-query candidates require numeric features")
    fault_windows = windows[
        windows["window_kind"].astype(str) == "fault"
    ].copy()
    if fault_windows["window_id"].astype(str).duplicated().any():
        raise ValueError("active-query windows contain duplicate IDs")
    feature_frame = entity_features.copy()
    feature_frame["window_id"] = feature_frame["window_id"].astype(str)
    grouped_features = (
        feature_frame.groupby("window_id", sort=True)[numeric_columns]
        .mean()
        .astype(float)
    )
    cluster_frame = clustered_windows.copy()
    cluster_frame["window_id"] = cluster_frame["window_id"].astype(str)
    if cluster_frame["window_id"].duplicated().any():
        raise ValueError("active-query clustered windows contain duplicate IDs")
    cluster_by_window = {
        str(row.window_id): str(row.cluster_id)
        for row in cluster_frame[["window_id", "cluster_id"]].itertuples(
            index=False
        )
    }
    baseline_ranks = {
        str(case_id): rank
        for rank, case_id in enumerate(tuple(baseline_query_ids), start=1)
    }
    raw_embeddings: Dict[str, tuple[float, ...]] = {}
    for window_id, row in grouped_features.iterrows():
        raw_embeddings[str(window_id)] = tuple(float(row[column]) for column in numeric_columns)
    clusters: Dict[str, list[tuple[float, ...]]] = {}
    for window_id, vector in raw_embeddings.items():
        clusters.setdefault(cluster_by_window.get(window_id, "__missing__"), []).append(
            vector
        )
    centroids = {
        cluster: tuple(
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        )
        for cluster, vectors in clusters.items()
        if vectors
    }
    rows: list[Dict[str, Any]] = []
    for window in fault_windows.sort_values(["window_id"]).itertuples(index=False):
        native_id = str(window.window_id)
        if native_id not in raw_embeddings:
            raise ValueError("active-query feature coverage missing %s" % native_id)
        vector = raw_embeddings[native_id]
        cluster = cluster_by_window.get(native_id, "__missing__")
        centroid = centroids[cluster]
        distance = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(vector, centroid))
        )
        rows.append(
            {
                "case_id": "%s::%s" % (dataset, native_id),
                "native_case_id": native_id,
                "split": "outer_train",
                "case_kind": "fault",
                "embedding": [float(value) for value in vector],
                "cluster_id": cluster,
                "time_bucket": "bucket-%d" % (len(rows) % 4),
                "scheme1_rank": baseline_ranks.get(native_id, 10**9),
                "baseline_score": float(-distance),
                "inner_boundary_uncertainty": float(distance),
            }
        )
    return rows


def annotations_from_selected_cases(
    selected_cases: Sequence[Mapping[str, Any]],
    *,
    windows: pd.DataFrame,
    budget: int = 30,
) -> list[Dict[str, Any]]:
    """Reveal simulated manual labels after the query plan is frozen."""

    if len(selected_cases) != int(budget):
        raise ValueError("active annotations require exactly budget cases")
    window_frame = windows.copy()
    window_frame["window_id"] = window_frame["window_id"].astype(str)
    if window_frame["window_id"].duplicated().any():
        raise ValueError("annotation windows contain duplicate IDs")
    by_window = window_frame.set_index("window_id", drop=False)
    annotations = []
    seen: set[str] = set()
    for row in selected_cases:
        native_id = str(row.get("native_case_id", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        if not native_id or native_id in seen or native_id not in by_window.index:
            raise ValueError("selected query annotation coverage drifted")
        seen.add(native_id)
        source = by_window.loc[native_id]
        targets = (
            _target_list(source["positive_ids_list"])
            if "positive_ids_list" in by_window.columns
            else _target_list(source.get("positive_ids"))
        )
        annotations.append(
            {
                "case_id": case_id,
                "native_case_id": native_id,
                "annotation_source": "simulated_manual_ground_truth",
                "targets": targets,
                "split": "outer_train",
                "case_kind": "fault",
            }
        )
    if len(seen) != int(budget):
        raise ValueError("active annotations require unique selected cases")
    return annotations


def oracle_full_annotations_from_outer_train(
    windows: pd.DataFrame,
    *,
    outer_train_window_ids: Sequence[Any],
) -> list[Dict[str, Any]]:
    """Open all outer-training fault labels for an explicit oracle_full arm."""

    frame = windows.copy()
    frame["window_id"] = frame["window_id"].astype(str)
    if frame["window_id"].duplicated().any():
        raise ValueError("oracle_full windows contain duplicate IDs")
    by_window = frame.set_index("window_id", drop=False)
    annotations: list[Dict[str, Any]] = []
    for raw_id in outer_train_window_ids:
        native_id = str(raw_id).strip()
        if native_id not in by_window.index:
            raise ValueError("oracle_full outer_train ID missing from windows: %s" % native_id)
        source = by_window.loc[native_id]
        if str(source.get("window_kind", "")) != "fault":
            continue
        targets = (
            _target_list(source["positive_ids_list"])
            if "positive_ids_list" in by_window.columns
            else _target_list(source.get("positive_ids"))
        )
        if not targets:
            raise ValueError("oracle_full fault case has no targets: %s" % native_id)
        annotations.append(
            {
                "case_id": native_id,
                "native_case_id": native_id,
                "annotation_source": "oracle_full_outer_train_ground_truth",
                "targets": targets,
                "split": "outer_train",
                "case_kind": "fault",
            }
        )
    if not annotations:
        raise ValueError("oracle_full annotations must not be empty")
    return annotations


def _ranker_fit_inputs_for_training_mode(
    *,
    is_oracle_full: bool,
    outer_train_tables: Any,
    safe_outer_train_tables: Any,
    query_plan: Any,
) -> tuple[Any, Any]:
    """Choose ranker-fit tables and query-plan override for the training mode."""

    if is_oracle_full:
        return outer_train_tables, None
    return safe_outer_train_tables, query_plan


def _mpca_enabled_arm_config(unit_or_config: Mapping[str, Any]) -> dict[str, Any]:
    if "mpca_arm_config" in unit_or_config:
        return dict(unit_or_config.get("mpca_arm_config") or {})
    return dict(unit_or_config or {})


def _mpca_required_bridge_effects(arm_config: Mapping[str, Any]) -> list[str]:
    config = dict(arm_config or {})
    effects: list[str] = []
    if str(config.get("augmentation", "none")) != "none":
        effects.append("synthetic_training_supervision")
    ranker_features = str(config.get("ranker_features", "current_reference"))
    if ranker_features not in {"", "current_reference"}:
        if str(config.get("score_overlay_policy", "blend")) == "residual_gate":
            effects.append("score_overlay_gate")
        else:
            effects.append("score_overlay")
    if bool(config.get("ood_fallback", False)):
        effects.append("ood_score_mixing")
    return effects


def _mpca_augmentation_modes(augmentation: str) -> list[str]:
    value = str(augmentation or "none")
    if value == "none":
        return []
    if value == "metric_only":
        return ["metric_family_replacement"]
    if value == "propagation_only":
        return ["propagation_motif_replacement"]
    if value == "joint_mechanism_propagation":
        return ["metric_family_replacement", "propagation_motif_replacement"]
    raise ValueError("unsupported MPCA augmentation mode: %s" % augmentation)


def _safe_synthetic_id(*parts: Any) -> str:
    text = "::".join(str(part) for part in parts if str(part))
    return "mpca_synth_%s" % semantic_sha256({"id": text})[:24]


def _window_ids_by_kind(windows: pd.DataFrame, kind: str) -> list[str]:
    frame = windows.copy()
    frame["window_id"] = frame["window_id"].astype(str)
    return [
        str(row.window_id)
        for row in frame[frame["window_kind"].astype(str) == str(kind)].itertuples(
            index=False
        )
    ]


def _annotation_targets_by_native_id(
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    for row in annotations:
        native_id = str(row.get("native_case_id", row.get("case_id", ""))).strip()
        if not native_id:
            continue
        values = [str(value) for value in row.get("targets") or [] if str(value)]
        if values:
            targets[native_id] = sorted(dict.fromkeys(values))
    return targets


def _entity_rows_for_window(entity_features: pd.DataFrame, window_id: str) -> pd.DataFrame:
    frame = entity_features.copy()
    frame["window_id"] = frame["window_id"].astype(str)
    return frame[frame["window_id"] == str(window_id)].copy()


def _has_all_targets(entity_rows: pd.DataFrame, targets: Sequence[str]) -> bool:
    if entity_rows.empty or "entity_id" not in entity_rows:
        return False
    entity_ids = set(entity_rows["entity_id"].astype(str))
    return set(str(target) for target in targets).issubset(entity_ids)


def _transform_synthetic_entity_rows(
    rows: pd.DataFrame,
    *,
    mode: str,
    feature_columns: Sequence[str],
    targets: Sequence[str],
) -> pd.DataFrame:
    output = rows.copy()
    numeric_columns = [
        column
        for column in feature_columns
        if column in output.columns and pd.api.types.is_numeric_dtype(output[column])
    ]
    if not numeric_columns:
        return output
    if mode == "metric_family_replacement":
        output[numeric_columns] = output[numeric_columns].astype(float)
        return output
    if mode == "propagation_motif_replacement":
        target_set = set(str(target) for target in targets)
        is_target = output["entity_id"].astype(str).isin(target_set)
        output.loc[is_target, numeric_columns] = (
            output.loc[is_target, numeric_columns].astype(float) * 0.95
        )
        output.loc[~is_target, numeric_columns] = (
            output.loc[~is_target, numeric_columns].astype(float) * 1.05
        )
        return output
    return output


def _append_cluster_rows(
    clustered_windows: pd.DataFrame | None,
    synthetic_clusters: Mapping[str, int],
) -> pd.DataFrame | None:
    if clustered_windows is None:
        return None
    clustered = clustered_windows.copy()
    existing = set(clustered["window_id"].astype(str))
    additions = [
        {"window_id": window_id, "cluster_id": int(cluster_id)}
        for window_id, cluster_id in synthetic_clusters.items()
        if window_id not in existing
    ]
    if additions:
        clustered = pd.concat([clustered, pd.DataFrame(additions)], ignore_index=True)
    for key, value in getattr(clustered_windows, "attrs", {}).items():
        clustered.attrs[key] = value
    return clustered


def _apply_mpca_synthetic_training_bridge(
    *,
    tables: Any,
    query_plan: Any | None,
    clustered_windows: pd.DataFrame | None,
    annotations: Sequence[Mapping[str, Any]],
    arm_config: Mapping[str, Any],
    seed: int,
) -> tuple[Any, Any | None, pd.DataFrame | None, dict[str, Any]]:
    """Inject MPCA synthetic supervision through the existing weighted trainer.

    The bridge uses only already queried/oracle-opened labels as synthetic targets.
    Unqueried cases may contribute observable feature rows as donor mechanisms or
    propagation contexts, but their labels/fault_type values are never read here.
    """

    config = dict(arm_config or {})
    augmentation = str(config.get("augmentation", "none"))
    modes = _mpca_augmentation_modes(augmentation)
    audit: dict[str, Any] = {
        "schema_version": "rcl-mpca-real-training-bridge-audit-v1",
        "augmentation": augmentation,
        "requested_modes": list(modes),
        "synthetic_window_count": 0,
        "synthetic_entity_row_count": 0,
        "synthetic_window_ids": [],
        "synthetic_confidence": float(config.get("synthetic_confidence", 0.5)),
        "training_effective": False,
        "label_access": "queried_or_oracle_opened_targets_only",
    }
    if not modes:
        return tables, query_plan, clustered_windows, audit

    windows = tables.windows.copy()
    entity_features = tables.entity_features.copy()
    feature_columns = list(getattr(tables, "feature_columns", ()) or [])
    windows["window_id"] = windows["window_id"].astype(str)
    entity_features["window_id"] = entity_features["window_id"].astype(str)
    by_window = windows.set_index("window_id", drop=False)
    targets_by_id = _annotation_targets_by_native_id(annotations)
    if not targets_by_id:
        return tables, query_plan, clustered_windows, audit

    query_ids = [case_id for case_id in targets_by_id if case_id in set(windows["window_id"])]
    if not query_ids:
        return tables, query_plan, clustered_windows, audit
    donor_pool = [
        case_id
        for case_id in _window_ids_by_kind(windows, "fault")
        if case_id not in set(query_ids)
    ]
    synthetic_windows: list[dict[str, Any]] = []
    synthetic_entity_frames: list[pd.DataFrame] = []
    synthetic_clusters: dict[str, int] = {}
    window_clusters = (
        dict(getattr(query_plan, "window_clusters", {}) or {})
        if query_plan is not None
        else {}
    )
    synthetic_confidence = float(config.get("synthetic_confidence", 0.5))
    synthetic_confidence = max(0.0, min(1.0, synthetic_confidence))

    for query_index, query_id in enumerate(query_ids):
        targets = targets_by_id[query_id]
        query_rows = _entity_rows_for_window(entity_features, query_id)
        if not _has_all_targets(query_rows, targets):
            continue
        for mode_index, mode in enumerate(modes):
            donor_id = query_id
            if mode == "metric_family_replacement" and donor_pool:
                donor_id = donor_pool[(query_index + mode_index + int(seed)) % len(donor_pool)]
            donor_rows = _entity_rows_for_window(entity_features, donor_id)
            source_rows = donor_rows if _has_all_targets(donor_rows, targets) else query_rows
            if source_rows.empty:
                continue
            synthetic_id = _safe_synthetic_id(augmentation, mode, query_id, donor_id, query_index)
            window_row = dict(by_window.loc[query_id])
            window_row["window_id"] = synthetic_id
            window_row["window_kind"] = "fault"
            if "positive_ids" in windows.columns:
                window_row["positive_ids"] = ";".join(targets)
            if "positive_ids_list" in windows.columns:
                window_row["positive_ids_list"] = list(targets)
            window_row["mpca_synthetic"] = True
            window_row["mpca_source_query_case_id"] = query_id
            window_row["mpca_source_observable_case_id"] = donor_id
            window_row["mpca_mockup_mode"] = mode
            synthetic_windows.append(window_row)

            synthetic_rows = source_rows.copy()
            synthetic_rows["window_id"] = synthetic_id
            if "window_kind" in synthetic_rows.columns:
                synthetic_rows["window_kind"] = "fault"
            synthetic_rows["mpca_synthetic"] = True
            synthetic_rows["mpca_source_query_case_id"] = query_id
            synthetic_rows["mpca_source_observable_case_id"] = donor_id
            synthetic_rows["mpca_mockup_mode"] = mode
            synthetic_rows = _transform_synthetic_entity_rows(
                synthetic_rows,
                mode=mode,
                feature_columns=feature_columns,
                targets=targets,
            )
            synthetic_entity_frames.append(synthetic_rows)
            synthetic_clusters[synthetic_id] = int(
                window_clusters.get(
                    query_id,
                    1 if int(getattr(query_plan, "normal_cluster_id", 0) or 0) == 0 else 0,
                )
            )

    if not synthetic_windows or not synthetic_entity_frames:
        return tables, query_plan, clustered_windows, audit

    augmented_windows = pd.concat(
        [windows, pd.DataFrame(synthetic_windows)],
        ignore_index=True,
        sort=False,
    )
    augmented_entity_features = pd.concat(
        [entity_features, *synthetic_entity_frames],
        ignore_index=True,
        sort=False,
    )
    augmented_metadata = {
        **dict(getattr(tables, "metadata", {}) or {}),
        "mpca_synthetic_training_bridge": {
            "augmentation": augmentation,
            "synthetic_window_count": len(synthetic_windows),
            "synthetic_entity_row_count": int(
                sum(len(frame) for frame in synthetic_entity_frames)
            ),
        },
    }
    augmented_tables = _clone_feature_bundle_tables(
        tables,
        windows=augmented_windows,
        entity_features=augmented_entity_features,
        metadata=augmented_metadata,
        feature_columns=feature_columns,
    )
    augmented_plan = query_plan
    augmented_clustered = clustered_windows
    if query_plan is not None:
        pseudo_labels = {
            str(key): list(value)
            for key, value in dict(getattr(query_plan, "pseudo_labels", {}) or {}).items()
        }
        pseudo_confidence = {
            str(key): float(value)
            for key, value in dict(getattr(query_plan, "pseudo_confidence", {}) or {}).items()
        }
        for window_row in synthetic_windows:
            synthetic_id = str(window_row["window_id"])
            source_query_id = str(window_row["mpca_source_query_case_id"])
            pseudo_labels[synthetic_id] = list(targets_by_id[source_query_id])
            pseudo_confidence[synthetic_id] = synthetic_confidence
        updated_clusters = {
            str(key): int(value)
            for key, value in dict(getattr(query_plan, "window_clusters", {}) or {}).items()
        }
        updated_clusters.update(synthetic_clusters)
        metadata = {
            **dict(getattr(query_plan, "metadata", {}) or {}),
            "mpca_synthetic_training_bridge": {
                "augmentation": augmentation,
                "synthetic_window_count": len(synthetic_windows),
                "synthetic_confidence": synthetic_confidence,
            },
        }
        augmented_plan = _clone_query_plan(
            query_plan,
            window_clusters=updated_clusters,
            pseudo_labels=pseudo_labels,
            pseudo_confidence=pseudo_confidence,
            metadata=metadata,
        )
        augmented_clustered = _append_cluster_rows(clustered_windows, synthetic_clusters)

    audit.update(
        {
            "synthetic_window_count": len(synthetic_windows),
            "synthetic_entity_row_count": int(
                sum(len(frame) for frame in synthetic_entity_frames)
            ),
            "synthetic_window_ids": [str(row["window_id"]) for row in synthetic_windows],
            "training_effective": True,
        }
    )
    audit["training_bridge_sha256"] = semantic_sha256(audit)
    return augmented_tables, augmented_plan, augmented_clustered, audit


def _unit_interval_from_text(value: Any) -> float:
    digest = semantic_sha256({"value": str(value)})
    return int(digest[:12], 16) / float(16**12 - 1)


def _normalize_series_per_window(rows: pd.DataFrame, values: pd.Series) -> pd.Series:
    output = pd.Series(0.5, index=rows.index, dtype=float)
    for _window_id, indices in rows.groupby("window_id", sort=False).groups.items():
        group_values = values.loc[list(indices)].astype(float)
        low = float(group_values.min())
        high = float(group_values.max())
        if math.isclose(low, high):
            output.loc[list(indices)] = 0.5
        else:
            output.loc[list(indices)] = (group_values - low) / (high - low)
    return output


def _apply_mpca_score_bridge(
    *,
    scored_rows: pd.DataFrame,
    arm_config: Mapping[str, Any],
    view_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply propagation/OOD score mixing before official ranking export."""

    config = dict(arm_config or {})
    ranker_features = str(config.get("ranker_features", "current_reference"))
    ood_enabled = bool(config.get("ood_fallback", False))
    score_overlay_policy = str(config.get("score_overlay_policy", "blend"))
    requested = ranker_features not in {"", "current_reference"} or ood_enabled
    rows = scored_rows.copy()
    audit: dict[str, Any] = {
        "schema_version": "rcl-mpca-score-bridge-audit-v1",
        "view": str(view_name),
        "ranker_features": ranker_features,
        "ood_fallback": ood_enabled,
        "requested": requested,
        "score_overlay_policy": score_overlay_policy,
        "score_gate_audited": False,
        "gated_window_count": 0,
        "rejected_window_count": 0,
        "score_overlay_alpha": 0.0,
        "fallback_weight": 0.0,
        "changed_row_count": 0,
        "score_bridge_effective": False,
    }
    if not requested or rows.empty:
        return rows, audit

    numeric_columns = [
        column
        for column in rows.columns
        if column not in {
            "score",
            "raw_score",
            "rank",
            "label",
            "window_id",
            "entity_id",
            "entity_name",
            "entity_type",
        }
        and pd.api.types.is_numeric_dtype(rows[column])
    ]
    if numeric_columns:
        strength = rows[numeric_columns].fillna(0.0).astype(float).abs().sum(axis=1)
    else:
        strength = rows["entity_id"].astype(str).map(_unit_interval_from_text).astype(float)
    propagation_score = _normalize_series_per_window(rows, strength)
    base_alpha = float(
        config.get(
            "score_overlay_alpha",
            0.03 if ranker_features in {"propagation", "joint", "mechanism"} else 0.0,
        )
    )
    base_alpha = max(0.0, min(1.0, base_alpha))
    fallback_weight = base_alpha
    if ood_enabled:
        fallback_weight = max(
            fallback_weight,
            max(0.0, min(1.0, float(config.get("ood_max_fallback_weight", 0.08)))),
        )
    learned = rows["score"].astype(float)
    per_row_weight = pd.Series(fallback_weight, index=rows.index, dtype=float)
    if score_overlay_policy == "residual_gate":
        min_margin = max(
            0.0, min(1.0, float(config.get("score_overlay_min_margin", 0.2)))
        )
        max_learned_gap = max(
            0.0, float(config.get("score_overlay_max_learned_gap", 0.05))
        )
        mixed = learned.copy()
        per_row_weight = pd.Series(0.0, index=rows.index, dtype=float)
        gated_window_count = 0
        rejected_window_count = 0
        for _window_id, indices in rows.groupby("window_id", sort=False).groups.items():
            index_list = list(indices)
            window_prop = propagation_score.loc[index_list].astype(float)
            window_learned = learned.loc[index_list].astype(float)
            if window_prop.empty:
                rejected_window_count += 1
                continue
            ordered_prop = window_prop.sort_values(ascending=False)
            top_idx = ordered_prop.index[0]
            top_prop = float(ordered_prop.iloc[0])
            second_prop = float(ordered_prop.iloc[1]) if len(ordered_prop) > 1 else 0.0
            propagation_margin = top_prop - second_prop
            learned_gap = float(window_learned.max()) - float(window_learned.loc[top_idx])
            if propagation_margin >= min_margin and learned_gap <= max_learned_gap:
                centered_residual = window_prop - 0.5
                mixed.loc[index_list] = window_learned + (base_alpha * centered_residual)
                per_row_weight.loc[index_list] = base_alpha
                gated_window_count += 1
            else:
                rejected_window_count += 1
        fallback_weight = base_alpha
        audit.update(
            {
                "score_gate_audited": True,
                "score_overlay_min_margin": min_margin,
                "score_overlay_max_learned_gap": max_learned_gap,
                "gated_window_count": gated_window_count,
                "rejected_window_count": rejected_window_count,
            }
        )
    else:
        mixed = ((1.0 - fallback_weight) * learned) + (
            fallback_weight * propagation_score
        )
    changed = (mixed - learned).abs() > 1e-12
    rows["mpca_learned_score"] = learned
    rows["mpca_propagation_score"] = propagation_score
    rows["mpca_score_bridge_weight"] = per_row_weight
    rows["score"] = mixed
    audit.update(
        {
            "score_overlay_alpha": base_alpha,
            "fallback_weight": fallback_weight,
            "changed_row_count": int(changed.sum()),
            "score_bridge_effective": bool(changed.any()),
            "scored_row_count": int(len(rows)),
            "numeric_feature_count": len(numeric_columns),
        }
    )
    audit["score_bridge_sha256"] = semantic_sha256(audit)
    return rows, audit


def _validate_mpca_bridge_effects(
    *,
    unit: Mapping[str, Any],
    bridge_audit: Mapping[str, Any],
) -> None:
    arm_config = _mpca_enabled_arm_config(unit)
    required = _mpca_required_bridge_effects(arm_config)
    if not required:
        return
    training = dict(bridge_audit.get("training") or {})
    views = dict(bridge_audit.get("views") or {})
    effects = {
        "synthetic_training_supervision": bool(training.get("training_effective")),
        "score_overlay": any(
            bool(dict(view).get("score_bridge_effective")) for view in views.values()
        ),
        "score_overlay_gate": any(
            bool(dict(view).get("requested"))
            and str(dict(view).get("score_overlay_policy", "")) == "residual_gate"
            and bool(dict(view).get("score_gate_audited"))
            for view in views.values()
        ),
        "ood_score_mixing": any(
            float(dict(view).get("fallback_weight", 0.0)) > float(
                dict(view).get("score_overlay_alpha", 0.0)
            )
            and bool(dict(view).get("score_bridge_effective"))
            for view in views.values()
        ),
    }
    missing = [effect for effect in required if not effects.get(effect, False)]
    if missing:
        raise RuntimeError(
            "MPCA real bridge declared effects were absent for %s: %s"
            % (str(unit.get("unit_id", "")), ", ".join(missing))
        )


def _case_aliases(case_id: Any) -> tuple[str, ...]:
    text = str(case_id).strip()
    aliases = [text]
    if "::" in text:
        aliases.append(text.split("::", 1)[1])
    elif text:
        aliases.append("rcabench::%s" % text)
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _select_fixed_cases(
    candidates: Sequence[Mapping[str, Any]],
    selected_case_ids: Sequence[Any],
    *,
    budget: int,
) -> Dict[str, Any]:
    selected_ids = [str(case_id).strip() for case_id in selected_case_ids]
    if len(selected_ids) != int(budget) or len(set(selected_ids)) != int(budget):
        raise ValueError("fixed selected_case_ids must match the declared budget")
    lookup: Dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        for field in ("case_id", "native_case_id"):
            for alias in _case_aliases(candidate.get(field, "")):
                lookup[alias] = candidate
    selected_cases = []
    for case_id in selected_ids:
        match = None
        for alias in _case_aliases(case_id):
            match = lookup.get(alias)
            if match is not None:
                break
        if match is None:
            raise ValueError("fixed selected case is not an outer-train candidate: %s" % case_id)
        selected_cases.append(dict(match))
    return {
        "schema_version": "rcl-query-active-fixed-selection-v1",
        "selector_id": "fixed_budget_set",
        "selected_case_ids": [str(row["case_id"]) for row in selected_cases],
        "selected_cases": selected_cases,
        "diagnostics": {
            "selected_case_count": len(selected_cases),
            "source": "manifest_selected_case_ids",
            "selection_sha256": semantic_sha256(selected_ids),
        },
    }


def _fault_ids_in_order(tables: Any, ordered_ids: Sequence[Any]) -> list[str]:
    windows = tables.windows.copy()
    windows["window_id"] = windows["window_id"].astype(str)
    fault_ids = set(
        windows.loc[
            windows["window_kind"].astype(str) == "fault",
            "window_id",
        ].astype(str)
    )
    return [str(window_id) for window_id in ordered_ids if str(window_id) in fault_ids]


def build_unseen_fault_type_inventory_from_feature_bundle(
    *,
    canonical_dataset_id: str,
    budget: int,
    feature_root: Path = DEFAULT_SEMI_FEATURE_ROOT,
    outer_test_ratio: float = 0.30,
    inner_val_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Build an unseen-category inventory from the same feature bundle split."""

    if os.name == "nt":
        raise RuntimeError("real unseen fault-type inventory is server-only")
    from nexusrcl_rebuild.evaluation.splits import build_nested_chronological_splits
    from nexusrcl_rebuild.training.semisupervised import (
        load_feature_bundle_tables,
        subset_feature_bundle_tables,
    )

    from .semi_supervised_bounded import select_bounded_window_ids

    dataset = resolve_dataset(str(canonical_dataset_id))
    if dataset.canonical_id not in {"rcabench", "aiops2022_pre"}:
        raise ValueError("unseen inventory supports rcabench and aiops2022_pre only")
    full_tables = load_feature_bundle_tables(Path(feature_root), dataset.legacy_alias)
    kinds = full_tables.windows["window_kind"].astype(str)
    bounded_ids = select_bounded_window_ids(
        full_tables.windows,
        fault_limit=int((kinds == "fault").sum()),
        normal_limit=int((kinds == "normal").sum()),
    )
    tables = subset_feature_bundle_tables(full_tables, bounded_ids)
    split = build_nested_chronological_splits(
        tables.windows,
        outer_test_ratio=float(outer_test_ratio),
        inner_val_ratio=float(inner_val_ratio),
    )
    outer_train_fault_ids = _fault_ids_in_order(
        tables,
        split.outer_train_window_ids,
    )
    outer_test_fault_ids = _fault_ids_in_order(
        tables,
        split.outer_test_window_ids,
    )
    by_window = _fault_type_oracle_by_window_id(tables.windows)
    all_fault_ids = sorted(set(outer_train_fault_ids).union(outer_test_fault_ids))
    cases = []
    for case_id in all_fault_ids:
        cases.append(
            {
                "case_id": case_id,
                "native_case_id": case_id,
                "canonical_dataset_id": dataset.canonical_id,
                "split": (
                    "outer_train"
                    if case_id in set(outer_train_fault_ids)
                    else "outer_test"
                ),
                "case_kind": "fault",
                "fault_type": by_window[case_id],
                "groundtruth_provenance": "feature_bundle.windows.metadata_json",
            }
        )
    return build_fault_type_inventory(
        canonical_dataset_id=dataset.canonical_id,
        cases=cases,
        outer_train_case_ids=outer_train_fault_ids,
        outer_test_case_ids=outer_test_fault_ids,
        t2_case_ids=outer_test_fault_ids + outer_train_fault_ids,
        budget=int(budget),
    )


def _fault_type_oracle_by_window_id(windows: pd.DataFrame) -> Dict[str, str]:
    frame = windows.copy()
    if "window_id" not in frame or "metadata_json" not in frame:
        raise ValueError("windows missing fault_type oracle metadata fields")
    frame["window_id"] = frame["window_id"].astype(str)
    if frame["window_id"].duplicated().any():
        raise ValueError("windows contain duplicate IDs for fault_type oracle")
    result: Dict[str, str] = {}
    fault_rows = frame[frame["window_kind"].astype(str) == "fault"]
    for row in fault_rows[["window_id", "metadata_json"]].itertuples(index=False):
        result[str(row.window_id)] = fault_type_oracle_from_metadata_json(
            row.metadata_json
        )
    if not result:
        raise ValueError("fault_type oracle requires fault windows")
    return result


def _add_fault_type_oracle_to_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    windows: pd.DataFrame,
) -> list[Dict[str, Any]]:
    by_window = _fault_type_oracle_by_window_id(windows)
    enriched: list[Dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        native_id = str(row.get("native_case_id", "")).strip()
        if native_id not in by_window:
            raise ValueError("fault_type oracle missing for candidate %s" % native_id)
        row["fault_type_oracle"] = by_window[native_id]
        enriched.append(row)
    return enriched


def _enrich_ranking_rows_with_fault_type(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    windows: pd.DataFrame,
) -> list[Dict[str, Any]]:
    by_window = _fault_type_oracle_by_window_id(windows)
    enriched: list[Dict[str, Any]] = []
    for raw in ranking_rows:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        native_id = case_id.split("::", 1)[1] if "::" in case_id else case_id
        if native_id not in by_window:
            raise ValueError("fault_type oracle missing for ranking case %s" % case_id)
        row["fault_type"] = by_window[native_id]
        enriched.append(row)
    return enriched


def _deepening_selector_result(
    unit: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    windows: pd.DataFrame,
    budget: int,
    seed: int,
) -> Dict[str, Any]:
    selector_id = str(unit["selector_id"])
    selector_config = dict(unit.get("selector_config") or {})
    if selector_id == "fixed_budget_set":
        if unit.get("selected_case_ids") is None:
            raise ValueError("fixed_budget_set requires manifest selected_case_ids")
        fixed = _select_fixed_cases(
            candidates,
            unit.get("selected_case_ids") or (),
            budget=budget,
        )
        fixed["diagnostics"].update(
            {
                "arm_id": str(unit.get("method_id", "")),
                "frozen_query_plan_sha256": str(
                    unit.get("query_plan_sha256", "")
                ),
            }
        )
        return fixed
    held_out_fault_type = str(unit.get("held_out_fault_type", "")).strip()
    if held_out_fault_type:
        enriched = _add_fault_type_oracle_to_candidates(
            candidates,
            windows=windows,
        )
        nonheldout = [
            row
            for row in enriched
            if str(row.get("fault_type_oracle", "")).strip() != held_out_fault_type
        ]
        if len(nonheldout) < int(budget):
            raise ValueError(
                "cannot fill budget after excluding held-out fault type %s: %d < %d"
                % (held_out_fault_type, len(nonheldout), int(budget))
            )
        clean_candidates = []
        for row in nonheldout:
            clean_row = dict(row)
            clean_row.pop("fault_type_oracle", None)
            clean_row.pop("fault_type", None)
            clean_row.pop("true_fault_type", None)
            clean_candidates.append(clean_row)
        selector = select_query_cases(
            clean_candidates,
            selector_id=selector_id,
            budget=budget,
            seed=seed,
            selector_config=selector_config,
        )
        selector["oracle_only"] = True
        selector.setdefault("diagnostics", {})
        selector["diagnostics"].update(
            {
                "held_out_fault_type": held_out_fault_type,
                "excluded_held_out_candidate_count": len(enriched)
                - len(nonheldout),
                "available_nonheldout_candidate_count": len(nonheldout),
                "ordered_candidate_count": len(enriched),
                "oracle_use": "diagnostic_forced_exclusion_only",
            }
        )
        return selector
    if selector_id == "true_fault_type_oracle":
        enriched = _add_fault_type_oracle_to_candidates(
            candidates,
            windows=windows,
        )
        return select_true_fault_type_oracle_cases(
            enriched,
            budget=budget,
            seed=seed,
            score_field=str(
                selector_config.get("score_field", "inner_boundary_uncertainty")
            ),
        )
    return select_query_cases(
        candidates,
        selector_id=selector_id,
        budget=budget,
        seed=seed,
        selector_config=selector_config,
    )


def run_active_query_inner_validation_unit(
    unit: Mapping[str, Any],
    run_root: Path,
    *,
    feature_root: Path = DEFAULT_SEMI_FEATURE_ROOT,
) -> Dict[str, Any]:
    """Run one real query-only selector arm on the inner-validation split."""

    if os.name == "nt":
        raise RuntimeError("real active-query screening execution is server-only")
    from nexusrcl_rebuild.evaluation.splits import build_nested_chronological_splits
    from nexusrcl_rebuild.training.semisupervised import (
        build_query_plan,
        fit_semisupervised_ranker_on_tables,
        load_feature_bundle_tables,
        subset_feature_bundle_tables,
    )

    from .semi_supervised_bounded import (
        _model_config,
        _save_evaluation,
        _strip_label_bearing_tables,
        apply_frozen_query_annotations,
        query_plan_sha256,
        select_bounded_window_ids,
    )

    dataset = resolve_dataset(str(unit["canonical_dataset_id"]))
    output_root = Path(str(unit["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_tables = load_feature_bundle_tables(Path(feature_root), dataset.legacy_alias)
    kinds = full_tables.windows["window_kind"].astype(str)
    fault_limit = int(unit.get("fault_limit") or int((kinds == "fault").sum()))
    normal_limit = int(unit.get("normal_limit") or int((kinds == "normal").sum()))
    bounded_ids = select_bounded_window_ids(
        full_tables.windows,
        fault_limit=fault_limit,
        normal_limit=normal_limit,
    )
    tables = subset_feature_bundle_tables(full_tables, bounded_ids)
    split = build_nested_chronological_splits(
        tables.windows,
        outer_test_ratio=float(unit.get("outer_test_ratio", 0.30)),
        inner_val_ratio=float(unit.get("inner_val_ratio", 0.20)),
    )
    outer_train_tables = subset_feature_bundle_tables(
        tables,
        split.outer_train_window_ids,
    )
    inner_train_tables = subset_feature_bundle_tables(
        outer_train_tables,
        split.inner_train_window_ids,
    )
    val_tables = subset_feature_bundle_tables(
        outer_train_tables,
        split.inner_val_window_ids,
    )
    selector_seed = int(unit.get("seed", 42))
    base_plan, inner_clustered = build_query_plan(
        inner_train_tables,
        budget=int(unit.get("budget", 30)),
        config=_model_config("fault_only"),
        random_state=selector_seed,
    )
    candidates = build_active_query_candidate_rows(
        canonical_dataset_id=dataset.canonical_id,
        windows=inner_train_tables.windows,
        entity_features=inner_train_tables.entity_features,
        clustered_windows=inner_clustered,
        feature_columns=inner_train_tables.feature_columns,
        baseline_query_ids=base_plan.queried_window_ids,
    )
    selector = select_query_cases(
        candidates,
        selector_id=str(unit["selector_id"]),
        budget=int(unit.get("budget", 30)),
        seed=selector_seed,
        selector_config=dict(unit.get("selector_config") or {}),
    )
    annotations = annotations_from_selected_cases(
        selector["selected_cases"],
        windows=inner_train_tables.windows,
        budget=int(unit.get("budget", 30)),
    )
    query_plan = apply_frozen_query_annotations(
        base_plan,
        annotations,
        budget=int(unit.get("budget", 30)),
    )
    safe_inner_train = _strip_label_bearing_tables(inner_train_tables)
    safe_val = _strip_label_bearing_tables(val_tables)
    normal_policy = bounded_normal_policy(str(unit.get("normal_policy", "fault_only")))
    config = _model_config(normal_policy)
    model, _clustered, _training_scores = fit_semisupervised_ranker_on_tables(
        tables=safe_inner_train,
        feature_root=Path(feature_root),
        budget=int(unit.get("budget", 30)),
        random_state=42,
        model_config=config,
        query_plan_override=query_plan,
    )
    scored = model.score_entity_features(safe_val.entity_features)
    query_hash = query_plan_sha256(query_plan)
    evaluation = _save_evaluation(
        output_root,
        model=model,
        scored_rows=scored,
        evaluation_windows=val_tables.windows,
        config=config,
        extra={
            "partition": "inner_validation",
            "normal_training_policy": normal_policy,
            "query_plan_sha256": query_hash,
            "selector_id": str(unit["selector_id"]),
            "selector_config_sha256": str(unit["selector_config_sha256"]),
        },
    )
    active_plan = {
        "schema_version": "rcl-query-active-real-selection-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "selector_result": selector,
        "annotations": annotations,
        "bounded_window_counts": {
            "fault_limit": fault_limit,
            "normal_limit": normal_limit,
            "inner_train_fault_cases": int(
                (
                    inner_train_tables.windows["window_kind"].astype(str) == "fault"
                ).sum()
            ),
            "inner_validation_fault_cases": int(
                (val_tables.windows["window_kind"].astype(str) == "fault").sum()
            ),
        },
    }
    _write_json(output_root / "active_query_selection.json", active_plan)
    return {
        "schema_version": "rcl-query-active-screening-unit-result-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "arm": "query_only",
        "selector_id": str(unit["selector_id"]),
        "selector_config_sha256": str(unit["selector_config_sha256"]),
        "seed": selector_seed,
        "normal_policy": str(unit.get("normal_policy", "fault_only")),
        "evaluation_partition": "inner_validation",
        "query_plan_sha256": query_hash,
        "selected_case_ids": list(selector["selected_case_ids"]),
        "selector_diagnostics": dict(selector["diagnostics"]),
        "metrics": {
            "hit_at_1": float(evaluation["hit_at_1"]),
            "hit_at_3": float(evaluation["hit_at_3"]),
            "hit_at_5": float(evaluation["hit_at_5"]),
            "evaluation_fault_cases": int(evaluation["scored_cases"]),
        },
        "result_path": str((output_root / "metrics.json").resolve()),
        "run_root": str(Path(run_root).resolve()),
    }


def run_active_query_outer_test_unit(
    unit: Mapping[str, Any],
    run_root: Path,
    *,
    feature_root: Path = DEFAULT_SEMI_FEATURE_ROOT,
) -> Dict[str, Any]:
    """Run one real query-only final arm on the sealed outer-test split."""

    if os.name == "nt":
        raise RuntimeError("real active-query final execution is server-only")
    from nexusrcl_rebuild.evaluation.splits import build_nested_chronological_splits
    from nexusrcl_rebuild.training.semisupervised import (
        build_query_plan,
        fit_semisupervised_ranker_on_tables,
        load_feature_bundle_tables,
        subset_feature_bundle_tables,
    )

    from .semi_supervised_bounded import (
        _model_config,
        _save_evaluation,
        _strip_label_bearing_tables,
        apply_frozen_query_annotations,
        query_plan_sha256,
        select_bounded_window_ids,
    )

    dataset = resolve_dataset(str(unit["canonical_dataset_id"]))
    output_root = Path(str(unit["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_tables = load_feature_bundle_tables(Path(feature_root), dataset.legacy_alias)
    kinds = full_tables.windows["window_kind"].astype(str)
    fault_limit = int(unit.get("fault_limit") or int((kinds == "fault").sum()))
    normal_limit = int(unit.get("normal_limit") or int((kinds == "normal").sum()))
    bounded_ids = select_bounded_window_ids(
        full_tables.windows,
        fault_limit=fault_limit,
        normal_limit=normal_limit,
    )
    tables = subset_feature_bundle_tables(full_tables, bounded_ids)
    split = build_nested_chronological_splits(
        tables.windows,
        outer_test_ratio=float(unit.get("outer_test_ratio", 0.30)),
        inner_val_ratio=float(unit.get("inner_val_ratio", 0.20)),
    )
    outer_train_tables = subset_feature_bundle_tables(
        tables,
        split.outer_train_window_ids,
    )
    test_tables = subset_feature_bundle_tables(
        tables,
        split.outer_test_window_ids,
    )
    selector_seed = int(unit.get("seed", 42))
    base_plan, outer_clustered = build_query_plan(
        outer_train_tables,
        budget=int(unit.get("budget", 30)),
        config=_model_config("fault_only"),
        random_state=selector_seed,
    )
    candidates = build_active_query_candidate_rows(
        canonical_dataset_id=dataset.canonical_id,
        windows=outer_train_tables.windows,
        entity_features=outer_train_tables.entity_features,
        clustered_windows=outer_clustered,
        feature_columns=outer_train_tables.feature_columns,
        baseline_query_ids=base_plan.queried_window_ids,
    )
    selector = select_query_cases(
        candidates,
        selector_id=str(unit["selector_id"]),
        budget=int(unit.get("budget", 30)),
        seed=selector_seed,
        selector_config=dict(unit.get("selector_config") or {}),
    )
    annotations = annotations_from_selected_cases(
        selector["selected_cases"],
        windows=outer_train_tables.windows,
        budget=int(unit.get("budget", 30)),
    )
    query_plan = apply_frozen_query_annotations(
        base_plan,
        annotations,
        budget=int(unit.get("budget", 30)),
    )
    safe_outer_train = _strip_label_bearing_tables(outer_train_tables)
    safe_test = _strip_label_bearing_tables(test_tables)
    normal_policy = bounded_normal_policy(str(unit.get("normal_policy", "fault_only")))
    config = _model_config(normal_policy)
    model, _clustered, _training_scores = fit_semisupervised_ranker_on_tables(
        tables=safe_outer_train,
        feature_root=Path(feature_root),
        budget=int(unit.get("budget", 30)),
        random_state=42,
        model_config=config,
        query_plan_override=query_plan,
    )
    scored = model.score_entity_features(safe_test.entity_features)
    query_hash = query_plan_sha256(query_plan)
    evaluation = _save_evaluation(
        output_root,
        model=model,
        scored_rows=scored,
        evaluation_windows=test_tables.windows,
        config=config,
        extra={
            "partition": "outer_test",
            "normal_training_policy": normal_policy,
            "query_plan_sha256": query_hash,
            "selector_id": str(unit["selector_id"]),
            "selector_config_sha256": str(unit["selector_config_sha256"]),
            "method_id": str(unit["method_id"]),
        },
    )
    active_plan = {
        "schema_version": "rcl-query-active-final-selection-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "method_id": str(unit["method_id"]),
        "selector_result": selector,
        "annotations": annotations,
        "bounded_window_counts": {
            "fault_limit": fault_limit,
            "normal_limit": normal_limit,
            "outer_train_fault_cases": int(
                (
                    outer_train_tables.windows["window_kind"].astype(str) == "fault"
                ).sum()
            ),
            "outer_test_fault_cases": int(
                (test_tables.windows["window_kind"].astype(str) == "fault").sum()
            ),
        },
    }
    _write_json(output_root / "active_query_selection.json", active_plan)
    return {
        "schema_version": "rcl-query-active-final-unit-result-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "method_id": str(unit["method_id"]),
        "arm": "query_only",
        "selector_id": str(unit["selector_id"]),
        "selector_config_sha256": str(unit["selector_config_sha256"]),
        "seed": selector_seed,
        "normal_policy": str(unit.get("normal_policy", "fault_only")),
        "evaluation_partition": "outer_test",
        "query_plan_sha256": query_hash,
        "selected_case_ids": list(selector["selected_case_ids"]),
        "selector_diagnostics": dict(selector["diagnostics"]),
        "metrics": {
            "hit_at_1": float(evaluation["hit_at_1"]),
            "hit_at_3": float(evaluation["hit_at_3"]),
            "hit_at_5": float(evaluation["hit_at_5"]),
            "evaluation_fault_cases": int(evaluation["scored_cases"]),
        },
        "result_path": str((output_root / "metrics.json").resolve()),
        "run_root": str(Path(run_root).resolve()),
    }


def run_rcabench_query_only_sota_unit(
    unit: Mapping[str, Any],
    run_root: Path,
    *,
    feature_root: Path = DEFAULT_SEMI_FEATURE_ROOT,
) -> Dict[str, Any]:
    """Run one RCABench query-only SOTA-search unit with T1/T2 scoring."""

    if os.name == "nt":
        raise RuntimeError("real RCABench SOTA-search execution is server-only")
    from nexusrcl_rebuild.evaluation.splits import build_nested_chronological_splits
    from nexusrcl_rebuild.training.semisupervised import (
        build_query_plan,
        fit_semisupervised_ranker_on_tables,
        load_feature_bundle_tables,
        subset_feature_bundle_tables,
    )

    from .semi_supervised_bounded import (
        _model_config,
        _save_evaluation,
        _strip_label_bearing_tables,
        apply_frozen_query_annotations,
        query_plan_sha256,
        select_bounded_window_ids,
    )

    dataset = resolve_dataset(str(unit["canonical_dataset_id"]))
    if dataset.canonical_id != "rcabench":
        raise ValueError("SOTA-search unit supports rcabench only")
    output_root = Path(str(unit["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_tables = load_feature_bundle_tables(Path(feature_root), dataset.legacy_alias)
    kinds = full_tables.windows["window_kind"].astype(str)
    fault_limit = int(unit.get("fault_limit") or int((kinds == "fault").sum()))
    normal_limit = int(unit.get("normal_limit") or int((kinds == "normal").sum()))
    bounded_ids = select_bounded_window_ids(
        full_tables.windows,
        fault_limit=fault_limit,
        normal_limit=normal_limit,
    )
    tables = subset_feature_bundle_tables(full_tables, bounded_ids)
    split = build_nested_chronological_splits(
        tables.windows,
        outer_test_ratio=float(unit.get("outer_test_ratio", 0.30)),
        inner_val_ratio=float(unit.get("inner_val_ratio", 0.20)),
    )
    outer_train_tables = subset_feature_bundle_tables(
        tables,
        split.outer_train_window_ids,
    )
    test_tables = subset_feature_bundle_tables(
        tables,
        split.outer_test_window_ids,
    )
    selector_seed = int(unit.get("active_learning_seed", unit.get("seed", 42)))
    budget = int(unit.get("budget", 30))
    base_plan, outer_clustered = build_query_plan(
        outer_train_tables,
        budget=budget,
        config=_model_config("fault_only"),
        random_state=selector_seed,
    )
    candidates = build_active_query_candidate_rows(
        canonical_dataset_id=dataset.canonical_id,
        windows=outer_train_tables.windows,
        entity_features=outer_train_tables.entity_features,
        clustered_windows=outer_clustered,
        feature_columns=outer_train_tables.feature_columns,
        baseline_query_ids=base_plan.queried_window_ids,
    )
    if unit.get("selected_case_ids"):
        selector = _select_fixed_cases(
            candidates,
            unit.get("selected_case_ids") or (),
            budget=budget,
        )
    else:
        selector = select_query_cases(
            candidates,
            selector_id=str(unit["selector_id"]),
            budget=budget,
            seed=selector_seed,
            selector_config=dict(unit.get("selector_config") or {}),
        )
    annotations = annotations_from_selected_cases(
        selector["selected_cases"],
        windows=outer_train_tables.windows,
        budget=budget,
    )
    query_plan = apply_frozen_query_annotations(
        base_plan,
        annotations,
        budget=budget,
    )
    queried_native_ids = [str(value) for value in query_plan.queried_window_ids]
    safe_outer_train = _strip_label_bearing_tables(outer_train_tables)
    safe_test = _strip_label_bearing_tables(test_tables)
    normal_policy = bounded_normal_policy(str(unit.get("normal_policy", "fault_only")))
    config = _model_config(normal_policy)
    model, _clustered, _training_scores = fit_semisupervised_ranker_on_tables(
        tables=safe_outer_train,
        feature_root=Path(feature_root),
        budget=budget,
        random_state=42,
        model_config=config,
        query_plan_override=query_plan,
    )
    query_hash = query_plan_sha256(query_plan)
    t1_scored = model.score_entity_features(safe_test.entity_features)
    t1_eval = _save_evaluation(
        output_root / "views" / "T1",
        model=model,
        scored_rows=t1_scored,
        evaluation_windows=test_tables.windows,
        config=config,
        extra={
            "partition": "T1_outer_test",
            "normal_training_policy": normal_policy,
            "query_plan_sha256": query_hash,
            "selector_id": str(unit["selector_id"]),
            "selector_config_sha256": str(unit["selector_config_sha256"]),
            "method_id": str(unit.get("method_id", "rcabench_query_only_sota")),
        },
    )
    outer_train_fault_ids = _fault_ids_in_order(
        outer_train_tables,
        split.outer_train_window_ids,
    )
    outer_test_fault_ids = _fault_ids_in_order(
        test_tables,
        split.outer_test_window_ids,
    )
    t2_ids = outer_test_fault_ids + [
        case_id for case_id in outer_train_fault_ids if case_id not in set(queried_native_ids)
    ]
    t2_tables = subset_feature_bundle_tables(tables, t2_ids)
    safe_t2 = _strip_label_bearing_tables(t2_tables)
    t2_scored = model.score_entity_features(safe_t2.entity_features)
    t2_eval = _save_evaluation(
        output_root / "views" / "T2",
        model=model,
        scored_rows=t2_scored,
        evaluation_windows=t2_tables.windows,
        config=config,
        extra={
            "partition": "T2_outer_test_plus_unqueried_outer_train",
            "normal_training_policy": normal_policy,
            "query_plan_sha256": query_hash,
            "selector_id": str(unit["selector_id"]),
            "selector_config_sha256": str(unit["selector_config_sha256"]),
            "method_id": str(unit.get("method_id", "rcabench_query_only_sota")),
        },
    )
    split_identity = build_split_identity(
        canonical_dataset_id=dataset.canonical_id,
        source_path=str(unit.get("source_path", RCABENCH_SOURCE_PATH)),
        outer_train_case_ids=outer_train_fault_ids,
        outer_test_case_ids=outer_test_fault_ids,
        split_seed=SPLIT_SEED,
    )
    score_payload = score_t1_t2_from_rankings(
        split_identity=split_identity,
        queried_case_ids=queried_native_ids,
        ranking_rows=_read_jsonl(output_root / "views" / "T2" / "per_case_rankings.jsonl"),
    )
    _write_json(output_root / "split_identity.json", split_identity)
    _write_json(output_root / "t1_t2_metrics.json", score_payload)
    active_plan = {
        "schema_version": "rcabench-query-only-sota-selection-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "method_id": str(unit.get("method_id", "rcabench_query_only_sota")),
        "selector_result": selector,
        "annotations": annotations,
        "queried_native_case_ids": queried_native_ids,
        "split_identity_sha256": split_identity["split_identity_sha256"],
        "bounded_window_counts": {
            "fault_limit": fault_limit,
            "normal_limit": normal_limit,
            "outer_train_fault_cases": len(outer_train_fault_ids),
            "outer_test_fault_cases": len(outer_test_fault_ids),
            "t1_fault_cases": int(t1_eval["evaluation_fault_cases"]),
            "t2_fault_cases": int(t2_eval["evaluation_fault_cases"]),
        },
    }
    _write_json(output_root / "active_query_selection.json", active_plan)
    result = {
        "schema_version": "rcabench-query-only-sota-unit-result-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "method_id": str(unit.get("method_id", "rcabench_query_only_sota")),
        "arm": "query_only",
        "selector_id": str(unit["selector_id"]),
        "selector_config_sha256": str(unit["selector_config_sha256"]),
        "active_learning_seed": selector_seed,
        "split_seed": SPLIT_SEED,
        "normal_policy": str(unit.get("normal_policy", "fault_only")),
        "budget": budget,
        "query_plan_sha256": query_hash,
        "selected_case_ids": queried_native_ids,
        "selected_case_ids_from_selector": list(selector["selected_case_ids"]),
        "selector_diagnostics": dict(selector["diagnostics"]),
        "split_identity_sha256": split_identity["split_identity_sha256"],
        "t1_metrics": score_payload["T1"],
        "t2_metrics": score_payload["T2"],
        "t1_sota_gate": score_payload["t1_sota_gate"],
        "t1_t2_warnings": score_payload["t1_t2_warnings"],
        "view_result_paths": {
            "T1": str((output_root / "views" / "T1" / "metrics.json").resolve()),
            "T2": str((output_root / "views" / "T2" / "metrics.json").resolve()),
        },
        "result_path": str((output_root / "t1_t2_metrics.json").resolve()),
        "run_root": str(Path(run_root).resolve()),
    }
    _write_json(output_root / "metrics.json", result)
    return result


def run_deepening_query_only_t1_t2_unit(
    unit: Mapping[str, Any],
    run_root: Path,
    *,
    feature_root: Path = DEFAULT_SEMI_FEATURE_ROOT,
) -> Dict[str, Any]:
    """Run one cross-dataset current-method deepening unit with T1/T2 metrics."""

    if os.name == "nt":
        raise RuntimeError("real query-active deepening execution is server-only")
    from nexusrcl_rebuild.evaluation.splits import build_nested_chronological_splits
    from nexusrcl_rebuild.training.semisupervised import (
        build_query_plan,
        fit_semisupervised_ranker_on_tables,
        load_feature_bundle_tables,
        subset_feature_bundle_tables,
    )

    from .semi_supervised_bounded import (
        _model_config,
        _save_evaluation,
        _strip_label_bearing_tables,
        apply_frozen_query_annotations,
        query_plan_sha256,
        select_bounded_window_ids,
    )

    dataset = validate_deepening_execution_dataset(
        str(unit["canonical_dataset_id"])
    )
    requested_evaluation_views = _requested_evaluation_views(unit)
    evaluation_scope = (
        "T1_only" if requested_evaluation_views == ("T1",) else "T1_T2"
    )
    output_root = Path(str(unit["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_tables = load_feature_bundle_tables(Path(feature_root), dataset.legacy_alias)
    kinds = full_tables.windows["window_kind"].astype(str)
    fault_limit = int(unit.get("fault_limit") or int((kinds == "fault").sum()))
    normal_limit = int(unit.get("normal_limit") or int((kinds == "normal").sum()))
    bounded_ids = select_bounded_window_ids(
        full_tables.windows,
        fault_limit=fault_limit,
        normal_limit=normal_limit,
    )
    tables = subset_feature_bundle_tables(full_tables, bounded_ids)
    split = build_nested_chronological_splits(
        tables.windows,
        outer_test_ratio=float(unit.get("outer_test_ratio", 0.30)),
        inner_val_ratio=float(unit.get("inner_val_ratio", 0.20)),
    )
    outer_train_tables = subset_feature_bundle_tables(
        tables,
        split.outer_train_window_ids,
    )
    test_tables = subset_feature_bundle_tables(
        tables,
        split.outer_test_window_ids,
    )
    outer_train_fault_ids = _fault_ids_in_order(
        outer_train_tables,
        split.outer_train_window_ids,
    )
    outer_test_fault_ids = _fault_ids_in_order(
        test_tables,
        split.outer_test_window_ids,
    )
    seed = int(unit.get("active_learning_seed", unit.get("seed", 42)))
    declared_budget = int(unit.get("budget", 30))
    is_oracle_full = bool(unit.get("oracle_full")) or str(unit.get("training_mode", "")) == "oracle_full"
    budget = len(outer_train_fault_ids) if is_oracle_full else declared_budget
    base_plan, outer_clustered = build_query_plan(
        outer_train_tables,
        budget=budget,
        config=_model_config("fault_only", oracle_full=is_oracle_full),
        random_state=seed,
    )
    if is_oracle_full:
        selector = {
            "schema_version": "rcl-query-active-oracle-full-selection-v1",
            "selector_id": "oracle_full_all_outer_train_fault_labels",
            "oracle_only": True,
            "budget": budget,
            "seed": seed,
            "selected_case_ids": list(outer_train_fault_ids),
            "selected_cases": [
                {
                    "case_id": case_id,
                    "native_case_id": case_id,
                    "split": "outer_train",
                    "case_kind": "fault",
                    "selection_stage": "oracle_full_all_labels",
                }
                for case_id in outer_train_fault_ids
            ],
            "diagnostics": {
                "oracle_full": True,
                "oracle_training_case_count": len(outer_train_fault_ids),
                "declared_budget": declared_budget,
            },
        }
        annotations = oracle_full_annotations_from_outer_train(
            outer_train_tables.windows,
            outer_train_window_ids=outer_train_fault_ids,
        )
    else:
        candidates = build_active_query_candidate_rows(
            canonical_dataset_id=dataset.canonical_id,
            windows=outer_train_tables.windows,
            entity_features=outer_train_tables.entity_features,
            clustered_windows=outer_clustered,
            feature_columns=outer_train_tables.feature_columns,
            baseline_query_ids=base_plan.queried_window_ids,
        )
        selector = _deepening_selector_result(
            unit,
            candidates,
            windows=outer_train_tables.windows,
            budget=budget,
            seed=seed,
        )
        annotations = annotations_from_selected_cases(
            selector["selected_cases"],
            windows=outer_train_tables.windows,
            budget=budget,
        )
    query_plan = apply_frozen_query_annotations(
        base_plan,
        annotations,
        budget=budget,
    )
    queried_native_ids = [str(value) for value in query_plan.queried_window_ids]
    safe_outer_train = _strip_label_bearing_tables(outer_train_tables)
    safe_test = _strip_label_bearing_tables(test_tables)
    normal_policy = bounded_normal_policy(str(unit.get("normal_policy", "fault_only")))
    config = _model_config(normal_policy, oracle_full=is_oracle_full)
    fit_tables, query_plan_override = _ranker_fit_inputs_for_training_mode(
        is_oracle_full=is_oracle_full,
        outer_train_tables=outer_train_tables,
        safe_outer_train_tables=safe_outer_train,
        query_plan=query_plan,
    )
    arm_config = _mpca_enabled_arm_config(unit)
    fit_tables, query_plan_override, clustered_override, training_bridge_audit = (
        _apply_mpca_synthetic_training_bridge(
            tables=fit_tables,
            query_plan=query_plan_override,
            clustered_windows=(outer_clustered if query_plan_override is not None else None),
            annotations=annotations,
            arm_config=arm_config,
            seed=seed,
        )
    )
    training_seed = _training_seed(unit)
    model, _clustered, _training_scores = _fit_with_compatible_cluster_override(
        fit_semisupervised_ranker_on_tables,
        tables=fit_tables,
        feature_root=Path(feature_root),
        budget=budget,
        random_state=training_seed,
        model_config=config,
        query_plan_override=query_plan_override,
        clustered_windows_override=clustered_override,
        require_clustered_override=bool(
            _mpca_augmentation_modes(str(arm_config.get("augmentation", "none")))
        ),
    )
    query_plan = model.query_plan
    query_hash = query_plan_sha256(query_plan)
    t1_scored = model.score_entity_features(safe_test.entity_features)
    t1_scored, t1_bridge_audit = _apply_mpca_score_bridge(
        scored_rows=t1_scored,
        arm_config=arm_config,
        view_name="T1",
    )
    _save_evaluation(
        output_root / "views" / "T1",
        model=model,
        scored_rows=t1_scored,
        evaluation_windows=test_tables.windows,
        config=config,
        extra={
            "partition": "T1_outer_test",
            "normal_training_policy": normal_policy,
            "query_plan_sha256": query_hash,
            "selector_id": str(unit["selector_id"]),
            "selector_config_sha256": str(unit["selector_config_sha256"]),
            "method_id": str(unit.get("method_id", "current_method_deepening")),
        },
    )
    held_out_fault_type = str(unit.get("held_out_fault_type", "")).strip()
    if "T2" in requested_evaluation_views:
        t2_ids = outer_test_fault_ids + [
            case_id
            for case_id in outer_train_fault_ids
            if case_id not in set(queried_native_ids)
        ]
        t2_tables = subset_feature_bundle_tables(tables, t2_ids)
        safe_t2 = _strip_label_bearing_tables(t2_tables)
        t2_scored = model.score_entity_features(safe_t2.entity_features)
        t2_scored, t2_bridge_audit = _apply_mpca_score_bridge(
            scored_rows=t2_scored,
            arm_config=arm_config,
            view_name="T2",
        )
        _save_evaluation(
            output_root / "views" / "T2",
            model=model,
            scored_rows=t2_scored,
            evaluation_windows=t2_tables.windows,
            config=config,
            extra={
                "partition": "T2_outer_test_plus_unqueried_outer_train",
                "normal_training_policy": normal_policy,
                "query_plan_sha256": query_hash,
                "selector_id": str(unit["selector_id"]),
                "selector_config_sha256": str(unit["selector_config_sha256"]),
                "method_id": str(unit.get("method_id", "current_method_deepening")),
            },
        )
        ranking_rows = _enrich_ranking_rows_with_fault_type(
            _read_jsonl(output_root / "views" / "T2" / "per_case_rankings.jsonl"),
            windows=tables.windows,
        )
        if held_out_fault_type:
            score_payload = score_fault_type_slices(
                canonical_dataset_id=dataset.canonical_id,
                outer_train_case_ids=outer_train_fault_ids,
                outer_test_case_ids=outer_test_fault_ids,
                queried_case_ids=queried_native_ids,
                ranking_rows=ranking_rows,
                held_out_fault_type=held_out_fault_type,
            )
        else:
            score_payload = score_t1_t2_generic(
                outer_train_case_ids=outer_train_fault_ids,
                outer_test_case_ids=outer_test_fault_ids,
                queried_case_ids=queried_native_ids,
                ranking_rows=ranking_rows,
                canonical_dataset_id=dataset.canonical_id,
            )
    else:
        if held_out_fault_type:
            raise ValueError(
                "T1_only evaluation does not support held-out fault-type slicing"
            )
        t2_bridge_audit = {"status": "not_requested"}
        ranking_rows = _enrich_ranking_rows_with_fault_type(
            _read_jsonl(output_root / "views" / "T1" / "per_case_rankings.jsonl"),
            windows=tables.windows,
        )
        score_payload = _score_t1_only_generic(
            outer_test_case_ids=outer_test_fault_ids,
            queried_case_ids=queried_native_ids,
            ranking_rows=ranking_rows,
            canonical_dataset_id=dataset.canonical_id,
        )
    mpca_bridge_audit = {
        "schema_version": "rcl-mpca-real-bridge-audit-v1",
        "unit_id": str(unit["unit_id"]),
        "mpca_arm_id": str(unit.get("mpca_arm_id", "")),
        "arm_config": arm_config,
        "required_effects": _mpca_required_bridge_effects(arm_config),
        "training": training_bridge_audit,
        "views": {
            "T1": t1_bridge_audit,
            "T2": t2_bridge_audit,
        },
        "model_metadata": {
            "training_row_count": int(dict(model.metadata).get("training_row_count", 0)),
            "pseudo_window_count": int(dict(model.metadata).get("pseudo_window_count", 0)),
            "feature_column_count": int(dict(model.metadata).get("feature_column_count", 0)),
            "label_source_counts": dict(dict(model.metadata).get("label_source_counts", {})),
        },
    }
    mpca_bridge_audit["bridge_audit_sha256"] = semantic_sha256(mpca_bridge_audit)
    _validate_mpca_bridge_effects(unit=unit, bridge_audit=mpca_bridge_audit)
    _write_json(output_root / "mpca_bridge_audit.json", mpca_bridge_audit)
    active_plan = {
        "schema_version": "rcl-query-active-deepening-current-method-selection-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "stage": str(unit.get("stage", "")),
        "method_id": str(unit.get("method_id", "current_method_deepening")),
        "oracle_only": bool(unit.get("oracle_only", False)),
        "held_out_fault_type": held_out_fault_type,
        "selector_result": selector,
        "annotations": annotations,
        "queried_native_case_ids": queried_native_ids,
        "mpca_bridge_audit": mpca_bridge_audit,
        "bounded_window_counts": {
            "fault_limit": fault_limit,
            "normal_limit": normal_limit,
            "outer_train_fault_cases": len(outer_train_fault_ids),
            "outer_test_fault_cases": len(outer_test_fault_ids),
            "t1_fault_cases": int(score_payload["T1"]["denominator"]),
            "t2_fault_cases": (
                int(score_payload["T2"]["denominator"])
                if "denominator" in score_payload["T2"]
                else None
            ),
        },
    }
    _write_json(output_root / "active_query_selection.json", active_plan)
    result = {
        "schema_version": "rcl-query-active-deepening-current-method-unit-result-v1",
        "unit_id": str(unit["unit_id"]),
        "canonical_dataset_id": dataset.canonical_id,
        "stage": str(unit.get("stage", "")),
        "method_id": str(unit.get("method_id", "current_method_deepening")),
        "arm": "query_only",
        "selector_id": str(unit["selector_id"]),
        "selector_config_sha256": str(unit["selector_config_sha256"]),
        "oracle_only": bool(unit.get("oracle_only", False)),
        "active_learning_seed": seed,
        "training_seed": training_seed,
        "split_seed": SPLIT_SEED,
        "evaluation_scope": evaluation_scope,
        "normal_policy": str(unit.get("normal_policy", "fault_only")),
        "budget": budget,
        "query_plan_sha256": query_hash,
        "queried_case_ids": queried_native_ids,
        "selected_case_ids": queried_native_ids,
        "selected_case_ids_from_selector": list(selector["selected_case_ids"]),
        "selector_diagnostics": dict(selector["diagnostics"]),
        "mpca_bridge_audit": mpca_bridge_audit,
        "mpca_bridge_audit_path": str((output_root / "mpca_bridge_audit.json").resolve()),
        "partitions": score_payload["partitions"],
        "metrics": {
            "T1": score_payload["T1"],
            "T2": score_payload["T2"],
        },
        "t1_t2_warnings": score_payload["t1_t2_warnings"],
        "per_case_ranking_evidence": score_payload["T1"]["per_case"],
        "per_case_ranking_evidence_by_view": _per_case_evidence_by_view(
            score_payload
        ),
        "target_set_evidence": [
            {
                "case_id": row["case_id"],
                "targets": row["targets"],
            }
            for row in score_payload["T1"]["per_case"]
        ],
        "view_result_paths": {
            "T1": str((output_root / "views" / "T1" / "metrics.json").resolve()),
        },
        "evaluation_status_by_view": {
            "T1": "completed",
            "T2": (
                "completed" if "T2" in requested_evaluation_views else "not_requested"
            ),
        },
        "result_path": str((output_root / "metrics.json").resolve()),
        "run_root": str(Path(run_root).resolve()),
    }
    if "T2" in requested_evaluation_views:
        result["view_result_paths"]["T2"] = str(
            (output_root / "views" / "T2" / "metrics.json").resolve()
        )
    if held_out_fault_type:
        result["held_out_fault_type"] = held_out_fault_type
        result["fault_type_slices"] = score_payload["slices"]
    _write_json(output_root / "t1_t2_metrics.json", score_payload)
    _write_json(output_root / "metrics.json", result)
    return result


__all__ = [
    "DEEPENING_EXECUTION_DATASETS",
    "DEFAULT_SEMI_FEATURE_ROOT",
    "annotations_from_selected_cases",
    "bounded_normal_policy",
    "build_active_query_candidate_rows",
    "build_unseen_fault_type_inventory_from_feature_bundle",
    "oracle_full_annotations_from_outer_train",
    "run_active_query_inner_validation_unit",
    "run_active_query_outer_test_unit",
    "run_deepening_query_only_t1_t2_unit",
    "validate_deepening_execution_dataset",
    "run_rcabench_query_only_sota_unit",
]
