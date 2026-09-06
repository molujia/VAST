"""Leakage-free multimodal acquisition representations for ordinary query-only RCL."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


FUSION_SCHEMA_VERSION = "ordinary-multimodal-fusion-v1"
GEOMETRY_SCHEMA_VERSION = "ordinary-proxy-geometry-v1"
FUSION_FREEZE_SCHEMA_VERSION = "ordinary-dataset-fusion-freeze-v1"
FORMAL_DATASETS = frozenset(("rcabench", "aiops2022_pre"))
MODALITIES = ("metric", "log", "trace", "topology", "time")
FUSION_IDS = (
    "masked_early",
    "coverage_normalized_late_affinity",
    "shared_state_alignment",
)
_ENTRY_FIELDS = frozenset(("feature_names", "values", "mask", "coverage"))
_SIGNAL_COLUMNS = {
    "metric": "has_metric_signal",
    "log": "has_log_signal",
    "trace": "has_trace_signal",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LABEL_FIELDS = frozenset(
    (
        "fault_type",
        "root_cause",
        "root_causes",
        "positive_ids",
        "positive_ids_list",
        "positive_names",
        "positive_types",
        "targets",
        "is_positive",
    )
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _dataset_id(value: Any) -> str:
    if not isinstance(value, str) or value not in FORMAL_DATASETS:
        raise ValueError("formal multimodal fusion dataset is out of scope")
    return value


def _case_ids(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("case IDs must be an ordered sequence")
    result = tuple(value)
    if (
        not result
        or any(
            not isinstance(case_id, str)
            or not case_id
            or case_id.strip() != case_id
            for case_id in result
        )
        or len(result) != len(set(result))
    ):
        raise ValueError("case IDs must be unique canonical strings")
    return result


def _finite_matrix(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> np.ndarray:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("%s must be a complete finite numeric block" % context)
    return matrix


def _feature_groups(feature_columns: Sequence[Any]) -> Dict[str, Tuple[str, ...]]:
    groups: Dict[str, list[str]] = {modality: [] for modality in MODALITIES[:-1]}
    seen = set()
    for raw in feature_columns:
        column = str(raw)
        if not column or column in seen:
            raise ValueError("feature columns must be unique non-empty names")
        seen.add(column)
        if column in _SIGNAL_COLUMNS.values() or column == "modalities_present_count":
            continue
        if column.startswith("metric_"):
            groups["metric"].append(column)
        elif column.startswith("log_"):
            groups["log"].append(column)
        elif column.startswith("trace_"):
            groups["trace"].append(column)
        elif (
            column in ("entity_is_service", "entity_is_host")
            or column.startswith("topo_")
            or column.startswith("topology_")
        ):
            groups["topology"].append(column)
    if any(not groups[modality] for modality in MODALITIES[:-1]):
        raise ValueError("every observable non-time modality requires feature columns")
    return {modality: tuple(columns) for modality, columns in groups.items()}


def _aggregate_block(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    matrix = _finite_matrix(frame, columns, context="case modality features")
    names = []
    values = []
    for index, column in enumerate(columns):
        vector = matrix[:, index]
        for statistic, value in (
            ("mean", float(vector.mean())),
            ("max", float(vector.max())),
            ("std", float(vector.std(ddof=0))),
        ):
            names.append("%s.%s" % (column, statistic))
            values.append(0.0 if value == 0.0 else value)
    return tuple(names), tuple(values)


def _time_block(row: Any) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    start = float(row.start_ts)
    end = float(row.end_ts)
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("time modality requires a finite strict start/end interval")
    duration = float(getattr(row, "duration_seconds", end - start))
    if not math.isfinite(duration) or duration <= 0.0:
        duration = end - start
    day_phase = (start % 86400.0) / 86400.0
    week_phase = (start % (7.0 * 86400.0)) / (7.0 * 86400.0)
    names = (
        "log_duration_seconds",
        "time_of_day_sin",
        "time_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    )
    values = (
        math.log1p(duration),
        math.sin(2.0 * math.pi * day_phase),
        math.cos(2.0 * math.pi * day_phase),
        math.sin(2.0 * math.pi * week_phase),
        math.cos(2.0 * math.pi * week_phase),
    )
    return names, tuple(float(value) for value in values)


def build_case_modality_records(
    *,
    windows: pd.DataFrame,
    entity_features: pd.DataFrame,
    feature_columns: Sequence[Any],
    case_ids: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate observable entity features into five label-free case blocks."""

    ordered_ids = _case_ids(case_ids)
    expected = set(ordered_ids)
    if not isinstance(windows, pd.DataFrame) or "window_id" not in windows:
        raise ValueError("window table is missing window_id")
    if not isinstance(entity_features, pd.DataFrame) or "window_id" not in entity_features:
        raise ValueError("entity feature table is missing window_id")
    window_frame = windows.copy()
    window_frame["window_id"] = window_frame["window_id"].astype(str)
    window_frame = window_frame[window_frame["window_id"].isin(expected)].copy()
    if window_frame["window_id"].duplicated().any() or set(window_frame["window_id"]) != expected:
        raise ValueError("window membership must exactly cover case IDs once")
    if not {"start_ts", "end_ts"}.issubset(window_frame.columns):
        raise ValueError("window time fields are missing")
    feature_frame = entity_features.copy()
    feature_frame["window_id"] = feature_frame["window_id"].astype(str)
    feature_frame = feature_frame[feature_frame["window_id"].isin(expected)].copy()
    if set(feature_frame["window_id"]) != expected:
        raise ValueError("entity feature membership must exactly cover case IDs")
    groups = _feature_groups(feature_columns)
    required_columns = set(column for columns in groups.values() for column in columns)
    required_columns.update(_SIGNAL_COLUMNS.values())
    missing = sorted(required_columns.difference(feature_frame.columns))
    if missing:
        raise ValueError("observable feature columns are missing: %s" % missing[0])
    windows_by_id = window_frame.set_index("window_id", drop=False)
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for case_id in sorted(ordered_ids):
        case_frame = feature_frame[feature_frame["window_id"] == case_id]
        modalities: Dict[str, Dict[str, Any]] = {}
        for modality in MODALITIES[:-1]:
            names, values = _aggregate_block(case_frame, groups[modality])
            if modality in _SIGNAL_COLUMNS:
                signal = pd.to_numeric(
                    case_frame[_SIGNAL_COLUMNS[modality]], errors="coerce"
                ).to_numpy(dtype=float)
                if not np.isfinite(signal).all() or not np.isin(signal, (0.0, 1.0)).all():
                    raise ValueError("modality signal mask must be complete binary data")
                mask = int(bool(np.max(signal)))
                coverage = float(np.mean(signal))
            else:
                mask = 1
                coverage = 1.0
            if mask == 0:
                values = tuple(0.0 for _ in values)
            modalities[modality] = {
                "feature_names": list(names),
                "values": list(values),
                "mask": mask,
                "coverage": coverage,
            }
        time_names, time_values = _time_block(windows_by_id.loc[case_id])
        modalities["time"] = {
            "feature_names": list(time_names),
            "values": list(time_values),
            "mask": 1,
            "coverage": 1.0,
        }
        result[case_id] = modalities
    return result


def _validated_case_modalities(
    value: Any,
) -> Tuple[
    Tuple[str, ...],
    Dict[str, Dict[str, Tuple[float, ...]]],
    Dict[str, Dict[str, float]],
    Dict[str, Tuple[str, ...]],
]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("case modalities must be a non-empty mapping")
    case_ids = tuple(sorted(_case_ids(tuple(value))))
    names_by_modality: Dict[str, Tuple[str, ...]] = {}
    vectors: Dict[str, Dict[str, Tuple[float, ...]]] = {}
    observations: Dict[str, Dict[str, float]] = {}
    for case_id in case_ids:
        modalities = value[case_id]
        if not isinstance(modalities, Mapping) or frozenset(modalities) != frozenset(MODALITIES):
            unknown = set(modalities) if isinstance(modalities, Mapping) else set()
            forbidden = sorted(unknown.intersection(_FORBIDDEN_LABEL_FIELDS))
            if forbidden:
                raise ValueError("forbidden label field in case modalities: %s" % forbidden[0])
            raise ValueError("case modalities must contain exactly five modalities")
        vectors[case_id] = {}
        observations[case_id] = {}
        for modality in MODALITIES:
            entry = modalities[modality]
            if not isinstance(entry, Mapping) or frozenset(entry) != _ENTRY_FIELDS:
                raise ValueError("modality entry fields are invalid")
            raw_names = entry["feature_names"]
            raw_values = entry["values"]
            if (
                isinstance(raw_names, (str, bytes))
                or not isinstance(raw_names, Sequence)
                or isinstance(raw_values, (str, bytes))
                or not isinstance(raw_values, Sequence)
            ):
                raise ValueError("modality names and values must be sequences")
            names = tuple(raw_names)
            if (
                not names
                or len(names) != len(set(names))
                or any(not isinstance(name, str) or not name for name in names)
            ):
                raise ValueError("modality feature names must be unique strings")
            try:
                values = tuple(float(item) for item in raw_values)
            except (TypeError, ValueError) as error:
                raise ValueError("modality values must be finite numbers") from error
            if len(values) != len(names) or not all(math.isfinite(item) for item in values):
                raise ValueError("modality values must match names and be finite")
            mask = entry["mask"]
            coverage = entry["coverage"]
            if isinstance(mask, bool) or not isinstance(mask, int) or mask not in (0, 1):
                raise ValueError("modality mask must be integer zero or one")
            if (
                isinstance(coverage, bool)
                or not isinstance(coverage, (int, float))
                or not math.isfinite(float(coverage))
                or not 0.0 <= float(coverage) <= 1.0
            ):
                raise ValueError("modality coverage must be finite within [0, 1]")
            coverage = float(coverage)
            if mask == 0 and (coverage != 0.0 or any(item != 0.0 for item in values)):
                raise ValueError("missing modality requires zero values and coverage")
            if mask == 1 and coverage <= 0.0:
                raise ValueError("observed modality requires positive coverage")
            if modality in names_by_modality and names_by_modality[modality] != names:
                raise ValueError("modality feature layout must be identical across cases")
            names_by_modality.setdefault(modality, names)
            vectors[case_id][modality] = values
            observations[case_id][modality + ".mask"] = float(mask)
            observations[case_id][modality + ".coverage"] = coverage
    return case_ids, vectors, observations, names_by_modality


def build_fusion_candidate_registry(dataset_id: Any) -> Dict[str, Any]:
    dataset = _dataset_id(dataset_id)
    configs = {
        "masked_early": {
            "normalization": "fold_local_observed_zscore",
            "modality_balance": "coverage_normalized_dimension_balance",
            "append_mask_and_coverage": True,
        },
        "coverage_normalized_late_affinity": {
            "normalization": "fold_local_observed_zscore",
            "distance": "jointly_observed_coverage_normalized_euclidean",
            "mds_dimension": 16,
        },
        "shared_state_alignment": {
            "normalization": "fold_local_observed_zscore",
            "shared_dimension": 4,
            "residual_weight": 0.25,
            "alignment": "paired_orthogonal_procrustes",
        },
    }
    payload = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "canonical_dataset_id": dataset,
        "scope": "one_candidate_registry_per_dataset_all_strategies_and_seeds",
        "candidate_configs": configs,
    }
    payload["registry_sha256"] = _semantic_sha256(payload)
    return payload


def _normalized_blocks(
    case_ids: Sequence[str],
    vectors: Mapping[str, Mapping[str, Sequence[float]]],
    observations: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    normalized: Dict[str, Dict[str, np.ndarray]] = {
        case_id: {} for case_id in case_ids
    }
    summary = {}
    for modality in MODALITIES:
        matrix = np.asarray([vectors[case_id][modality] for case_id in case_ids], dtype=float)
        mask = np.asarray(
            [observations[case_id][modality + ".mask"] for case_id in case_ids],
            dtype=bool,
        )
        if mask.any():
            mean = matrix[mask].mean(axis=0)
            scale = matrix[mask].std(axis=0)
            scale = np.where(scale == 0.0, 1.0, scale)
        else:
            mean = np.zeros(matrix.shape[1], dtype=float)
            scale = np.ones(matrix.shape[1], dtype=float)
        transformed = (matrix - mean) / scale
        transformed[~mask] = 0.0
        for index, case_id in enumerate(case_ids):
            normalized[case_id][modality] = transformed[index]
        summary[modality] = {
            "dimension": int(matrix.shape[1]),
            "observed_case_count": int(mask.sum()),
            "mean_sha256": _semantic_sha256([float(value) for value in mean]),
            "scale_sha256": _semantic_sha256([float(value) for value in scale]),
        }
    return normalized, summary


def _contributions(
    case_id: str,
    observations: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    weights = {
        modality: float(observations[case_id][modality + ".coverage"])
        for modality in MODALITIES
    }
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("case must contain at least one observed modality")
    return {modality: weights[modality] / total for modality in MODALITIES}


def _metadata_suffix(
    case_id: str,
    observations: Mapping[str, Mapping[str, float]],
) -> list[float]:
    return [
        float(observations[case_id][modality + ".mask"])
        for modality in MODALITIES
    ] + [
        float(observations[case_id][modality + ".coverage"])
        for modality in MODALITIES
    ]


def _masked_early_embeddings(
    case_ids: Sequence[str],
    normalized: Mapping[str, Mapping[str, np.ndarray]],
    observations: Mapping[str, Mapping[str, float]],
) -> Dict[str, list[float]]:
    result = {}
    for case_id in case_ids:
        contribution = _contributions(case_id, observations)
        vector = []
        for modality in MODALITIES:
            block = normalized[case_id][modality]
            scale = math.sqrt(contribution[modality] / max(1, len(block)))
            vector.extend(float(value * scale) for value in block)
        vector.extend(_metadata_suffix(case_id, observations))
        result[case_id] = vector
    return result


def _late_distance(
    case_ids: Sequence[str],
    normalized: Mapping[str, Mapping[str, np.ndarray]],
    observations: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    count = len(case_ids)
    distance = np.zeros((count, count), dtype=float)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left_id = case_ids[left_index]
            right_id = case_ids[right_index]
            weighted = 0.0
            total_weight = 0.0
            for modality in MODALITIES:
                left_coverage = observations[left_id][modality + ".coverage"]
                right_coverage = observations[right_id][modality + ".coverage"]
                weight = math.sqrt(left_coverage * right_coverage)
                if weight == 0.0:
                    continue
                left = normalized[left_id][modality]
                right = normalized[right_id][modality]
                block_distance = float(np.linalg.norm(left - right)) / math.sqrt(
                    max(1, len(left))
                )
                weighted += weight * block_distance
                total_weight += weight
            value = 1.0 if total_weight == 0.0 else weighted / total_weight
            distance[left_index, right_index] = value
            distance[right_index, left_index] = value
    return distance


def _classical_mds(distance: np.ndarray, dimension: int) -> np.ndarray:
    count = distance.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / float(count)
    gram = -0.5 * centering @ np.square(distance) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = [index for index in order if eigenvalues[index] > 1e-12]
    retained = positive[: max(1, min(int(dimension), count - 1))]
    if not retained:
        return np.zeros((count, 1), dtype=float)
    vectors = eigenvectors[:, retained].copy()
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    return vectors * np.sqrt(eigenvalues[retained])


def _pca_scores(matrix: np.ndarray, mask: np.ndarray, dimension: int) -> np.ndarray:
    scores = np.zeros((matrix.shape[0], dimension), dtype=float)
    if int(mask.sum()) < 2:
        return scores
    _, singular_values, right = np.linalg.svd(matrix[mask], full_matrices=False)
    rank = min(dimension, right.shape[0], int((singular_values > 1e-12).sum()))
    if rank:
        basis = right[:rank].copy()
        for index in range(rank):
            pivot = int(np.argmax(np.abs(basis[index])))
            if basis[index, pivot] < 0.0:
                basis[index] *= -1.0
        scores[:, :rank] = matrix @ basis.T
        scores[~mask] = 0.0
    return scores


def _shared_state_embeddings(
    case_ids: Sequence[str],
    normalized: Mapping[str, Mapping[str, np.ndarray]],
    observations: Mapping[str, Mapping[str, float]],
    *,
    shared_dimension: int,
    residual_weight: float,
) -> Dict[str, list[float]]:
    matrices = {
        modality: np.asarray(
            [normalized[case_id][modality] for case_id in case_ids], dtype=float
        )
        for modality in MODALITIES
    }
    masks = {
        modality: np.asarray(
            [
                observations[case_id][modality + ".mask"] > 0.0
                for case_id in case_ids
            ],
            dtype=bool,
        )
        for modality in MODALITIES
    }
    scores = {
        modality: _pca_scores(matrices[modality], masks[modality], shared_dimension)
        for modality in MODALITIES
    }
    anchor = max(MODALITIES, key=lambda modality: (int(masks[modality].sum()), -MODALITIES.index(modality)))
    aligned = {anchor: scores[anchor]}
    for modality in MODALITIES:
        if modality == anchor:
            continue
        overlap = masks[anchor] & masks[modality]
        rotation = np.eye(shared_dimension)
        if int(overlap.sum()) >= 2:
            cross = scores[modality][overlap].T @ scores[anchor][overlap]
            left, _, right = np.linalg.svd(cross, full_matrices=False)
            rotation = left @ right
        aligned[modality] = scores[modality] @ rotation
    result = {}
    for index, case_id in enumerate(case_ids):
        weights = np.asarray(
            [observations[case_id][modality + ".coverage"] for modality in MODALITIES],
            dtype=float,
        )
        stacked = np.stack([aligned[modality][index] for modality in MODALITIES])
        shared = np.average(stacked, axis=0, weights=weights)
        vector = [float(value) for value in shared]
        for modality_index, modality in enumerate(MODALITIES):
            residual = (stacked[modality_index] - shared) * float(residual_weight)
            if weights[modality_index] == 0.0:
                residual[:] = 0.0
            vector.extend(float(value) for value in residual)
        vector.extend(_metadata_suffix(case_id, observations))
        result[case_id] = vector
    return result


def fit_fusion_artifact(
    dataset_id: Any,
    *,
    case_modalities: Any,
    fusion_id: Any,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit one fold-local acquisition fusion on the complete outer-train pool."""

    dataset = _dataset_id(dataset_id)
    if not isinstance(fusion_id, str) or fusion_id not in FUSION_IDS:
        raise ValueError("unknown multimodal fusion_id")
    registry_config = deepcopy(
        build_fusion_candidate_registry(dataset)["candidate_configs"][fusion_id]
    )
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("fusion config must be a mapping")
        unknown = set(config).difference(registry_config)
        if unknown:
            raise ValueError("fusion config has undeclared fields")
        registry_config.update(deepcopy(dict(config)))
    case_ids, vectors, observations, names = _validated_case_modalities(case_modalities)
    normalized, normalization = _normalized_blocks(case_ids, vectors, observations)
    pairwise_distance = None
    transform_summary: Dict[str, Any] = {
        "normalization": "fold_local_observed_zscore",
        "normalization_by_modality": normalization,
    }
    if fusion_id == "masked_early":
        embeddings = _masked_early_embeddings(case_ids, normalized, observations)
    elif fusion_id == "coverage_normalized_late_affinity":
        dimension = int(registry_config["mds_dimension"])
        if dimension <= 0:
            raise ValueError("mds_dimension must be positive")
        pairwise_distance = _late_distance(case_ids, normalized, observations)
        mds = _classical_mds(pairwise_distance, dimension)
        embeddings = {
            case_id: [float(value) for value in mds[index]]
            + _metadata_suffix(case_id, observations)
            for index, case_id in enumerate(case_ids)
        }
        transform_summary.update(
            {
                "distance": "jointly_observed_coverage_normalized_euclidean",
                "mds_requested_dimension": dimension,
                "mds_retained_dimension": int(mds.shape[1]),
            }
        )
    else:
        shared_dimension = int(registry_config["shared_dimension"])
        residual_weight = float(registry_config["residual_weight"])
        if shared_dimension <= 0 or not 0.0 <= residual_weight <= 1.0:
            raise ValueError("shared-state fusion parameters are out of range")
        embeddings = _shared_state_embeddings(
            case_ids,
            normalized,
            observations,
            shared_dimension=shared_dimension,
            residual_weight=residual_weight,
        )
        transform_summary.update(
            {
                "shared_dimension": shared_dimension,
                "residual_weight": residual_weight,
                "alignment": "paired_orthogonal_procrustes",
                "alignment_scope": "fold_local",
            }
        )
    case_rows = []
    for case_id in case_ids:
        case_rows.append(
            {
                "case_id": case_id,
                "embedding": embeddings[case_id],
                "modality_mask": {
                    modality: int(observations[case_id][modality + ".mask"])
                    for modality in MODALITIES
                },
                "modality_coverage": {
                    modality: float(observations[case_id][modality + ".coverage"])
                    for modality in MODALITIES
                },
                "modality_contributions": _contributions(case_id, observations),
            }
        )
    observable_payload = {
        "case_ids": list(case_ids),
        "feature_names_by_modality": {
            modality: list(names[modality]) for modality in MODALITIES
        },
        "case_modalities": case_modalities,
    }
    artifact: Dict[str, Any] = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "canonical_dataset_id": dataset,
        "fusion_id": fusion_id,
        "fusion_config": registry_config,
        "fit_scope": "outer_train_candidate_pool_only",
        "case_count": len(case_ids),
        "case_ids_sha256": _semantic_sha256(list(case_ids)),
        "observable_input_sha256": _semantic_sha256(observable_payload),
        "feature_names_by_modality": observable_payload["feature_names_by_modality"],
        "transform_summary": transform_summary,
        "case_embeddings": case_rows,
    }
    if pairwise_distance is not None:
        artifact["pairwise_distance_matrix"] = [
            [float(value) for value in row] for row in pairwise_distance
        ]
    artifact["artifact_sha256"] = _semantic_sha256(artifact)
    return artifact


def _validate_frozen_artifact(artifact: Any) -> Tuple[list[dict[str, Any]], str]:
    if not isinstance(artifact, Mapping):
        raise ValueError("fusion artifact must be a mapping")
    digest = artifact.get("artifact_sha256")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError("fusion artifact SHA-256 is missing")
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    if _semantic_sha256(payload) != digest:
        raise ValueError("fusion artifact digest drifted")
    rows = artifact.get("case_embeddings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("fusion artifact has no case embeddings")
    return rows, digest


def evaluate_frozen_proxy_geometry(
    fusion_artifact: Any,
    *,
    evaluation_fault_type_by_case: Any,
    cluster_count: int,
) -> Dict[str, Any]:
    """Evaluate type agreement only after the label-free fusion is hash-frozen."""

    rows, artifact_sha256 = _validate_frozen_artifact(fusion_artifact)
    if not isinstance(evaluation_fault_type_by_case, Mapping):
        raise ValueError("evaluation fault-type map must be a mapping")
    case_ids = [str(row["case_id"]) for row in rows]
    if set(evaluation_fault_type_by_case) != set(case_ids):
        raise ValueError("evaluation fault-type membership must match fusion cases")
    labels = [str(evaluation_fault_type_by_case[case_id]) for case_id in case_ids]
    if any(not label for label in labels):
        raise ValueError("evaluation fault types must be non-empty")
    if isinstance(cluster_count, bool) or not isinstance(cluster_count, int) or not 2 <= cluster_count <= len(case_ids):
        raise ValueError("cluster_count must fit frozen cases")
    matrix = np.asarray([row["embedding"] for row in rows], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("frozen fusion embeddings must be finite")
    distinct_count = int(np.unique(matrix, axis=0).shape[0])
    effective_cluster_count = min(cluster_count, distinct_count)
    if effective_cluster_count == 1:
        proxy = np.zeros(len(case_ids), dtype=int)
        geometry_status = "degenerate_single_distinct_embedding"
    else:
        proxy = KMeans(
            n_clusters=effective_cluster_count,
            random_state=42,
            n_init=20,
        ).fit_predict(matrix)
        geometry_status = (
            "complete"
            if effective_cluster_count == cluster_count
            else "reduced_to_distinct_embedding_count"
        )
    purity_hits = 0
    mode_reports = {}
    missingness_dominated = 0
    for mode in sorted(set(int(value) for value in proxy)):
        indices = [index for index, value in enumerate(proxy) if int(value) == mode]
        type_counts = Counter(labels[index] for index in indices)
        purity_hits += max(type_counts.values())
        signatures = Counter(
            tuple(
                int(rows[index]["modality_mask"][modality])
                for modality in MODALITIES
            )
            for index in indices
        )
        dominance = max(signatures.values()) / float(len(indices))
        dominated = dominance >= 0.8 and len(signatures) > 1
        missingness_dominated += int(dominated)
        mode_reports[str(mode)] = {
            "case_count": len(indices),
            "fault_type_counts": dict(sorted(type_counts.items())),
            "dominant_missingness_fraction": dominance,
            "missingness_dominated": dominated,
        }
    proxy_by_case = {
        case_id: int(proxy[index]) for index, case_id in enumerate(case_ids)
    }
    report = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "canonical_dataset_id": fusion_artifact["canonical_dataset_id"],
        "fusion_id": fusion_artifact["fusion_id"],
        "fusion_artifact_sha256": artifact_sha256,
        "fault_type_role": "evaluation_only_after_fusion_freeze",
        "case_count": len(case_ids),
        "cluster_count": cluster_count,
        "requested_cluster_count": cluster_count,
        "effective_cluster_count": effective_cluster_count,
        "distinct_embedding_count": distinct_count,
        "geometry_status": geometry_status,
        "purity": purity_hits / float(len(case_ids)),
        "adjusted_rand_index": float(adjusted_rand_score(labels, proxy)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(labels, proxy)
        ),
        "proxy_modes_by_case": proxy_by_case,
        "proxy_mode_reports": mode_reports,
        "missingness_dominated_proxy_mode_count": missingness_dominated,
    }
    report["report_sha256"] = _semantic_sha256(report)
    return report


def freeze_dataset_fusion(
    dataset_id: Any,
    *,
    fusion_id: Any,
    fusion_artifact_sha256: Any,
) -> Dict[str, Any]:
    dataset = _dataset_id(dataset_id)
    if not isinstance(fusion_id, str) or fusion_id not in FUSION_IDS:
        raise ValueError("unknown multimodal fusion_id")
    if (
        not isinstance(fusion_artifact_sha256, str)
        or not _SHA256_PATTERN.fullmatch(fusion_artifact_sha256)
    ):
        raise ValueError("fusion artifact SHA-256 must be canonical")
    payload = {
        "schema_version": FUSION_FREEZE_SCHEMA_VERSION,
        "canonical_dataset_id": dataset,
        "fusion_id": fusion_id,
        "fusion_artifact_sha256": fusion_artifact_sha256,
        "scope": "one_fusion_per_dataset_all_strategies_and_seeds",
    }
    payload["freeze_sha256"] = _semantic_sha256(payload)
    return payload


__all__ = [
    "FORMAL_DATASETS",
    "FUSION_IDS",
    "FUSION_SCHEMA_VERSION",
    "MODALITIES",
    "build_case_modality_records",
    "build_fusion_candidate_registry",
    "evaluate_frozen_proxy_geometry",
    "fit_fusion_artifact",
    "freeze_dataset_fusion",
]
