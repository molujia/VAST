"""Reproducible, preparation-only AIOps22 HDBSCAN structure-grid builder.

This module reconstructs the frozen ``global_pca_dim32`` representation,
fits each approved HDBSCAN configuration exactly once, and freezes label-free
center query plans.  It never launches RCL and never reads private labels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from rcl_study.combined_active_learning_tuning import (
    ACQUISITION_SEEDS,
    BUDGET,
    CANDIDATE_POOL_SIZE,
    EFFECTIVE_SUPPORT,
    CandidateStructureEvaluation,
    CanonicalJsonPublication,
    TuningCandidate,
    TuningConfiguration,
    assess_candidate_structure,
    build_default_contract,
    build_tuning_candidate,
    build_tuning_manifest,
    deduplicate_query_plan_triples,
    preflight_canonical_json_idempotent,
    validate_frozen_tuning_followup_root,
    write_canonical_json_idempotent,
)


DATASET_ID = "aiops2022_pre"
REPRESENTATION_ID = "global_pca_dim32"
EXPECTED_DESCRIPTOR_INPUT_SHA256 = (
    "e78a35565c1c87bb211857ae5a964ce5b57b6866740a1d00c2548b8adf1d6e33"
)
EXPECTED_REPRESENTATION_MATRIX_SHA256 = (
    "c9ed2855078e8ddc6c9d75821d2c92302371b9d6e26956f607a354b8a0c069ff"
)
EXPECTED_ORDERED_POOL_SHA256 = (
    "078036d5808d21c6f4292ef54ee30abe47bf838753351af8e881471ea8400037"
)
REPRESENTATION_SCOPED_CODE_PATHS = (
    "rcl_study/combined_active_learning_representation.py",
    "rcl_study/combined_active_learning_geometry_smoke.py",
    "rcl_study/combined_active_learning_schemas.py",
    "rcl_study/ordinary_multimodal_fusion.py",
)
DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "candidate_context_path",
        "candidate_context_sha256",
        "feature_inventory_path",
        "feature_inventory_sha256",
        "feature_root",
        "feature_source_manifest_path",
        "feature_source_artifact_sha256",
        "candidate_population",
        "active_learning_seeds",
        "required_core_modalities",
        "trace_repair_required",
        "approved_initial_grid_only",
        "code_files_sha256",
        "code_bundle_sha256",
        "input_sha256",
    }
)
COMPATIBILITY_SCHEMA_VERSION = (
    "combined-active-learning-tuning-representation-compatibility-v1"
)
STRUCTURE_SUMMARY_SCHEMA_VERSION = (
    "combined-active-learning-hdbscan-structure-grid-summary-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_pool_sha256(case_ids: Sequence[str]) -> str:
    encoded = json.dumps(
        list(case_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _workspace_path(value: Any, *, workspace: Path, field_name: str) -> Path:
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{field_name} must be an absolute normalized path")
    resolved = raw.resolve()
    if not _within(resolved, workspace):
        raise ValueError(f"{field_name} escaped the isolated workspace")
    return resolved


def _read_json_object(path: Path, field_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{field_name} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must contain an object")
    return value


def validate_sealed_representation_input(
    *,
    workspace_root: str | Path,
    descriptor_path: str | Path,
    trusted_input_sha256: str = EXPECTED_DESCRIPTOR_INPUT_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a v2 descriptor while tolerating unrelated code-bundle drift."""

    workspace = Path(workspace_root).resolve()
    if not workspace.is_dir():
        raise ValueError("isolated workspace root must be an existing directory")
    descriptor_file = _workspace_path(
        descriptor_path, workspace=workspace, field_name="sealed descriptor path"
    )
    descriptor = _read_json_object(descriptor_file, "sealed descriptor")
    if frozenset(descriptor) != DESCRIPTOR_FIELDS:
        raise ValueError("sealed descriptor field allowlist drifted")

    trusted = _require_sha256(trusted_input_sha256, "trusted input SHA-256")
    body = dict(descriptor)
    declared_input_sha256 = body.pop("input_sha256", None)
    if (
        declared_input_sha256 != semantic_sha256(body)
        or declared_input_sha256 != trusted
    ):
        raise ValueError("sealed descriptor semantic seal or trust anchor drifted")
    if (
        descriptor.get("schema_version")
        != "combined-active-learning-formal-structure-input-v2"
        or descriptor.get("dataset_id") != DATASET_ID
        or descriptor.get("candidate_population")
        != "full_observable_outer_train_not_sampled"
        or descriptor.get("active_learning_seeds") != list(ACQUISITION_SEEDS)
        or descriptor.get("required_core_modalities")
        != ["metric", "log", "trace"]
        or descriptor.get("trace_repair_required") is not True
        or descriptor.get("approved_initial_grid_only") is not True
    ):
        raise ValueError("sealed descriptor protocol controls drifted")

    declared_code_files = descriptor.get("code_files_sha256")
    if not isinstance(declared_code_files, dict) or not set(
        REPRESENTATION_SCOPED_CODE_PATHS
    ).issubset(declared_code_files):
        raise ValueError("sealed descriptor lacks the representation-scoped sources")
    if any(
        not isinstance(relative, str)
        or not relative
        or not _SHA256.fullmatch(str(digest))
        for relative, digest in declared_code_files.items()
    ):
        raise ValueError("sealed code-file hash map is invalid")
    if descriptor.get("code_bundle_sha256") != semantic_sha256(
        declared_code_files
    ):
        raise ValueError("sealed whole-code-bundle hash is internally invalid")

    current_code_files: dict[str, str | None] = {}
    for relative in declared_code_files:
        source = _workspace_path(
            workspace / relative,
            workspace=workspace,
            field_name="declared code source",
        )
        current_code_files[relative] = _file_sha256(source) if source.is_file() else None
    representation_drift = sorted(
        relative
        for relative in REPRESENTATION_SCOPED_CODE_PATHS
        if current_code_files.get(relative) != declared_code_files.get(relative)
    )
    if representation_drift:
        raise ValueError(
            "representation-scoped source drift: " + ", ".join(representation_drift)
        )
    whole_bundle_drift = sorted(
        relative
        for relative, declared in declared_code_files.items()
        if current_code_files.get(relative) != declared
    )
    current_bundle_sha256 = semantic_sha256(current_code_files)

    context_path = _workspace_path(
        descriptor["candidate_context_path"],
        workspace=workspace,
        field_name="candidate context path",
    )
    expected_context = (
        workspace
        / "inputs"
        / "authority_freeze"
        / DATASET_ID
        / "candidate-context.json"
    ).resolve()
    if (
        context_path != expected_context
        or not context_path.is_file()
        or _file_sha256(context_path) != descriptor["candidate_context_sha256"]
    ):
        raise ValueError("candidate context hash or canonical path drifted")

    inventory_path = _workspace_path(
        descriptor["feature_inventory_path"],
        workspace=workspace,
        field_name="feature inventory path",
    )
    if (
        not inventory_path.is_file()
        or _file_sha256(inventory_path) != descriptor["feature_inventory_sha256"]
    ):
        raise ValueError("feature inventory hash drifted")
    source_path = _workspace_path(
        descriptor["feature_source_manifest_path"],
        workspace=workspace,
        field_name="feature source path",
    )
    feature_root = _workspace_path(
        descriptor["feature_root"],
        workspace=workspace,
        field_name="feature root",
    )
    if (
        source_path != (feature_root / "hd1" / "window_feature_manifest.json").resolve()
        or not source_path.is_file()
        or _file_sha256(source_path)
        != descriptor["feature_source_artifact_sha256"]
    ):
        raise ValueError("feature source hash or canonical path drifted")

    context = _read_json_object(context_path, "candidate context")
    inventory = _read_json_object(inventory_path, "feature inventory")
    _read_json_object(source_path, "feature source manifest")
    if context.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise ValueError("candidate context dataset drifted")
    inventory_body = dict(inventory)
    inventory_seal = inventory_body.pop("inventory_sha256", None)
    if inventory_seal is not None and inventory_seal != semantic_sha256(inventory_body):
        raise ValueError("feature inventory semantic seal drifted")
    if (
        inventory.get("dataset_id") != DATASET_ID
        or inventory.get("source_artifact_sha256")
        != descriptor["feature_source_artifact_sha256"]
    ):
        raise ValueError("feature inventory source identity drifted")

    audit = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "descriptor_path": descriptor_file.as_posix(),
        "descriptor_file_sha256": _file_sha256(descriptor_file),
        "descriptor_input_sha256": declared_input_sha256,
        "representation_scoped_paths": list(REPRESENTATION_SCOPED_CODE_PATHS),
        "representation_scoped_source_sha256": {
            relative: current_code_files[relative]
            for relative in REPRESENTATION_SCOPED_CODE_PATHS
        },
        "representation_scope_exact_match": True,
        "sealed_whole_bundle_sha256": descriptor["code_bundle_sha256"],
        "current_whole_bundle_sha256": current_bundle_sha256,
        "whole_bundle_exact_match": not whole_bundle_drift,
        "whole_bundle_drifted_paths": whole_bundle_drift,
        "whole_bundle_drift_tolerated": bool(whole_bundle_drift),
        "candidate_context_sha256": descriptor["candidate_context_sha256"],
        "candidate_context_exact_match": True,
        "feature_inventory_sha256": descriptor["feature_inventory_sha256"],
        "feature_inventory_exact_match": True,
        "feature_source_artifact_sha256": descriptor[
            "feature_source_artifact_sha256"
        ],
        "feature_source_exact_match": True,
        "descriptor_rewritten_or_resealed": False,
        "ground_truth_read": False,
    }
    audit["compatibility_sha256"] = semantic_sha256(audit)
    return descriptor, audit


def rebuild_global_pca_dim32(
    *, workspace_root: str | Path, descriptor: Mapping[str, Any]
) -> Any:
    """Rebuild the selected representation from observable candidates only."""

    if not isinstance(descriptor, Mapping) or descriptor.get("dataset_id") != DATASET_ID:
        raise ValueError("representation rebuild requires the validated AIOps22 descriptor")
    root = Path(workspace_root).resolve()
    context_path = Path(descriptor["candidate_context_path"]).resolve()
    inventory_path = Path(descriptor["feature_inventory_path"]).resolve()
    context = _read_json_object(context_path, "candidate context")
    inventory = _read_json_object(inventory_path, "feature inventory")

    from rcl_study.combined_active_learning_geometry_smoke import (
        DATASET_CONFIG,
        _external_feature_loaders,
        filter_case_modalities_to_inventory,
        observable_candidate_case_ids,
    )
    from rcl_study.combined_active_learning_representation import (
        balance_modality_blocks,
        build_global_representation_candidates,
        fit_modality_standardizer,
        transform_modality_blocks,
    )

    case_ids = tuple(
        sorted(
            observable_candidate_case_ids(
                context,
                expected_dataset_id=DATASET_ID,
            )
        )
    )
    if len(case_ids) != CANDIDATE_POOL_SIZE:
        raise ValueError("observable candidate pool count drifted from 169")
    eligible = inventory.get("eligible_features_by_modality")
    if not isinstance(eligible, Mapping):
        raise ValueError("feature inventory has no eligible modality mapping")
    feature_root = Path(descriptor["feature_root"]).resolve()
    load_tables, build_records = _external_feature_loaders(root)
    tables = load_tables(feature_root, DATASET_CONFIG[DATASET_ID]["alias"])
    raw = build_records(
        windows=tables.windows,
        entity_features=tables.entity_features,
        feature_columns=tables.feature_columns,
        case_ids=case_ids,
    )
    filtered = filter_case_modalities_to_inventory(
        raw,
        eligible_features_by_modality=eligible,
    )
    standardizer = fit_modality_standardizer(
        fit_case_ids=case_ids,
        case_modalities=filtered,
        feature_names_by_modality={
            modality: tuple(names) for modality, names in eligible.items()
        },
    )
    standardized = transform_modality_blocks(
        artifact=standardizer,
        case_modalities=filtered,
        case_ids=case_ids,
    )
    candidates = build_global_representation_candidates(
        balance_modality_blocks(standardized)
    )
    if REPRESENTATION_ID not in candidates:
        raise ValueError("global_pca_dim32 was not rebuilt")
    selected = candidates[REPRESENTATION_ID]
    if tuple(selected.case_ids) != case_ids:
        raise ValueError("rebuilt representation changed the observable pool order")
    return selected


def validate_rebuilt_representation(
    candidate: Any,
    *,
    expected_matrix_sha256: str = EXPECTED_REPRESENTATION_MATRIX_SHA256,
    expected_ordered_pool_sha256: str = EXPECTED_ORDERED_POOL_SHA256,
) -> str:
    expected_matrix = _require_sha256(
        expected_matrix_sha256, "expected representation matrix SHA-256"
    )
    expected_pool = _require_sha256(
        expected_ordered_pool_sha256, "expected ordered pool SHA-256"
    )
    case_ids = tuple(getattr(candidate, "case_ids", ()))
    matrix = np.asarray(getattr(candidate, "matrix", ()), dtype=float)
    feature_names = tuple(getattr(candidate, "feature_names", ()))
    if (
        getattr(candidate, "representation_id", None) != REPRESENTATION_ID
        or getattr(candidate, "formal_eligible", None) is not True
        or len(case_ids) != CANDIDATE_POOL_SIZE
        or len(set(case_ids)) != CANDIDATE_POOL_SIZE
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or matrix.ndim != 2
        or matrix.shape[0] != CANDIDATE_POOL_SIZE
        or matrix.shape[1] != len(feature_names)
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("rebuilt global_pca_dim32 representation is invalid")
    if getattr(candidate, "matrix_sha256", None) != expected_matrix:
        raise ValueError("rebuilt representation matrix SHA-256 drifted")
    observed_pool = ordered_pool_sha256(case_ids)
    if observed_pool != expected_pool:
        raise ValueError("rebuilt ordered pool SHA-256 drifted")
    return observed_pool


def backend_parameters(configuration: TuningConfiguration) -> dict[str, Any]:
    if not isinstance(configuration, TuningConfiguration):
        raise ValueError("HDBSCAN fitting requires a frozen tuning configuration")
    return {
        "min_cluster_size": configuration.min_cluster_size,
        "min_samples": configuration.min_samples,
        "cluster_selection_method": configuration.cluster_selection_method,
        "max_cluster_size": configuration.max_cluster_size,
        "metric": configuration.metric,
        "alpha": configuration.alpha,
        "cluster_selection_epsilon": configuration.cluster_selection_epsilon,
        "allow_single_cluster": configuration.allow_single_cluster,
        "store_centers": configuration.store_centers,
        "copy": True,
        "n_jobs": 1,
    }


@dataclass(frozen=True, eq=False)
class FittedHDBSCANGeometry:
    configuration: TuningConfiguration
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    membership_strengths: tuple[float, ...]
    medoid_cluster_labels: tuple[int, ...]
    medoids: np.ndarray
    matrix: np.ndarray
    representation_matrix_sha256: str
    backend: str
    backend_parameters: Mapping[str, Any]
    seed_influences_geometry: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, TuningConfiguration):
            raise ValueError("geometry configuration is invalid")
        if (
            len(self.case_ids) != CANDIDATE_POOL_SIZE
            or len(set(self.case_ids)) != CANDIDATE_POOL_SIZE
            or len(self.labels) != CANDIDATE_POOL_SIZE
            or len(self.membership_strengths) != CANDIDATE_POOL_SIZE
            or any(isinstance(label, bool) or not isinstance(label, (int, np.integer)) for label in self.labels)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in self.membership_strengths
            )
        ):
            raise ValueError("HDBSCAN geometry population evidence is invalid")
        labels = tuple(sorted({int(label) for label in self.labels if int(label) >= 0}))
        medoids = np.asarray(self.medoids, dtype=float)
        matrix = np.asarray(self.matrix, dtype=float)
        if (
            self.medoid_cluster_labels != labels
            or medoids.ndim != 2
            or medoids.shape[0] != len(labels)
            or not np.isfinite(medoids).all()
            or matrix.ndim != 2
            or matrix.shape[0] != CANDIDATE_POOL_SIZE
            or matrix.shape[1] != medoids.shape[1]
            or not np.isfinite(matrix).all()
            or dict(self.backend_parameters) != backend_parameters(self.configuration)
            or self.seed_influences_geometry is not False
            or not self.backend
        ):
            raise ValueError("HDBSCAN geometry controls or medoid evidence are invalid")
        _require_sha256(
            self.representation_matrix_sha256,
            "geometry representation matrix SHA-256",
        )

    @property
    def geometry_sha256(self) -> str:
        return semantic_sha256(
            {
                "configuration": self.configuration.to_dict(),
                "case_ids": list(self.case_ids),
                "labels": [int(value) for value in self.labels],
                "membership_strengths": [
                    float(value) for value in self.membership_strengths
                ],
                "medoid_cluster_labels": list(self.medoid_cluster_labels),
                "medoids": np.asarray(self.medoids).tolist(),
                "representation_matrix_sha256": self.representation_matrix_sha256,
                "backend": self.backend,
                "backend_parameters": dict(self.backend_parameters),
                "seed_influences_geometry": False,
            }
        )


def _default_hdbscan_factory(**kwargs: Any) -> Any:
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(**kwargs)


def fit_configuration_geometry(
    configuration: TuningConfiguration,
    *,
    candidate: Any,
    backend_factory: Callable[..., Any] | None = None,
    thread_limit_context_factory: Callable[[int], Any] | None = None,
) -> FittedHDBSCANGeometry:
    """Fit one deterministic sklearn-compatible HDBSCAN geometry."""

    case_ids = tuple(getattr(candidate, "case_ids", ()))
    matrix = np.asarray(getattr(candidate, "matrix", ()), dtype=float)
    if (
        len(case_ids) != CANDIDATE_POOL_SIZE
        or matrix.ndim != 2
        or matrix.shape[0] != CANDIDATE_POOL_SIZE
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("HDBSCAN candidate matrix is invalid")
    parameters = backend_parameters(configuration)
    factory = backend_factory or _default_hdbscan_factory
    estimator = factory(**parameters)
    if thread_limit_context_factory is None:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    else:
        limiter = thread_limit_context_factory(1)
    with limiter:
        fitted = estimator.fit(np.array(matrix, dtype=float, copy=True))
    labels = tuple(int(value) for value in np.asarray(fitted.labels_).reshape(-1))
    memberships = tuple(
        float(value) for value in np.asarray(fitted.probabilities_).reshape(-1)
    )
    cluster_labels = tuple(sorted({label for label in labels if label >= 0}))
    medoids = (
        np.asarray(fitted.medoids_, dtype=float)
        if cluster_labels
        else np.empty((0, matrix.shape[1]), dtype=float)
    )
    return FittedHDBSCANGeometry(
        configuration=configuration,
        case_ids=case_ids,
        labels=labels,
        membership_strengths=memberships,
        medoid_cluster_labels=cluster_labels,
        medoids=medoids,
        matrix=np.array(matrix, dtype=float, copy=True),
        representation_matrix_sha256=str(candidate.matrix_sha256),
        backend=(
            "sklearn.cluster.HDBSCAN"
            if backend_factory is None
            else f"{type(fitted).__module__}.{type(fitted).__qualname__}"
        ),
        backend_parameters=parameters,
    )


def _effective_members(
    geometry: FittedHDBSCANGeometry,
) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...]]:
    members: dict[int, list[int]] = {}
    for index, label in enumerate(geometry.labels):
        if label >= 0:
            members.setdefault(int(label), []).append(index)
    effective = {
        label: tuple(indices)
        for label, indices in members.items()
        if len(indices) >= EFFECTIVE_SUPPORT
    }
    residual = tuple(
        index
        for index, label in enumerate(geometry.labels)
        if int(label) not in effective
    )
    return dict(sorted(effective.items())), residual


def _structure_evaluation(
    geometry: FittedHDBSCANGeometry,
) -> tuple[CandidateStructureEvaluation, int, int]:
    effective, _residual = _effective_members(geometry)
    support = sum(len(indices) for indices in effective.values())
    largest = max((len(indices) for indices in effective.values()), default=0)
    summaries = tuple(
        tuning_summary(
            seed=seed,
            effective_cluster_count=len(effective),
            effective_coverage=support / CANDIDATE_POOL_SIZE,
            largest_effective_cluster_share=largest / CANDIDATE_POOL_SIZE,
        )
        for seed in ACQUISITION_SEEDS
    )
    return assess_candidate_structure(geometry.configuration, summaries), support, largest


def tuning_summary(
    *,
    seed: int,
    effective_cluster_count: int,
    effective_coverage: float,
    largest_effective_cluster_share: float,
):
    from rcl_study.combined_active_learning_tuning import StructureSeedSummary

    return StructureSeedSummary(
        seed=seed,
        effective_cluster_count=effective_cluster_count,
        effective_coverage=float(effective_coverage),
        largest_effective_cluster_share=float(largest_effective_cluster_share),
    )


def _seeded_digest(
    *,
    configuration_id: str,
    geometry_sha256: str,
    seed: int,
    role: str,
    value: Any,
) -> str:
    return semantic_sha256(
        {
            "role": role,
            "configuration_id": configuration_id,
            "geometry_sha256": geometry_sha256,
            "seed": seed,
            "value": value,
        }
    )


def build_seeded_center_plan(
    geometry: FittedHDBSCANGeometry,
    *,
    seed: int,
    pool_case_ids: Sequence[str],
    budget: int = BUDGET,
    _geometry_sha256: str | None = None,
) -> tuple[str, ...]:
    if seed not in ACQUISITION_SEEDS or budget != BUDGET:
        raise ValueError("center plan seed or budget drifted")
    pool = tuple(pool_case_ids)
    if pool != geometry.case_ids:
        raise ValueError("center planning requires the exact ordered full pool")
    evaluation, _support, _largest = _structure_evaluation(geometry)
    if not evaluation.eligible:
        raise ValueError("center plans require an eligible structure")
    effective, residual = _effective_members(geometry)
    medoid_by_label = {
        label: np.asarray(geometry.medoids[index], dtype=float)
        for index, label in enumerate(geometry.medoid_cluster_labels)
    }
    matrix_by_case_index = np.asarray(geometry.matrix, dtype=float)
    geometry_hash = _geometry_sha256 or geometry.geometry_sha256
    _require_sha256(geometry_hash, "center-plan geometry SHA-256")
    configuration_id = geometry.configuration.configuration_id

    cluster_order = tuple(
        sorted(
            effective,
            key=lambda label: _seeded_digest(
                configuration_id=configuration_id,
                geometry_sha256=geometry_hash,
                seed=seed,
                role="hdbscan-cluster-traversal",
                value=label,
            ),
        )
    )
    queues: dict[int, tuple[int, ...]] = {}
    for label, indices in effective.items():
        medoid = medoid_by_label[label]
        queues[label] = tuple(
            sorted(
                indices,
                key=lambda index: (
                    float(np.linalg.norm(matrix_by_case_index[index] - medoid)),
                    -float(geometry.membership_strengths[index]),
                    _seeded_digest(
                        configuration_id=configuration_id,
                        geometry_sha256=geometry_hash,
                        seed=seed,
                        role="hdbscan-center-case-tie",
                        value=geometry.case_ids[index],
                    ),
                ),
            )
        )
    residual_order = tuple(
        sorted(
            residual,
            key=lambda index: _seeded_digest(
                configuration_id=configuration_id,
                geometry_sha256=geometry_hash,
                seed=seed,
                role="hdbscan-residual-order",
                value=geometry.case_ids[index],
            ),
        )
    )
    residual_quota = min(2, len(residual_order), budget - len(cluster_order))
    selected_indices: list[int] = []
    offsets = {label: 0 for label in cluster_order}
    for label in cluster_order:
        selected_indices.append(queues[label][0])
        offsets[label] = 1
    selected_indices.extend(residual_order[:residual_quota])
    while len(selected_indices) < budget:
        progressed = False
        for label in cluster_order:
            offset = offsets[label]
            if offset >= len(queues[label]):
                continue
            selected_indices.append(queues[label][offset])
            offsets[label] += 1
            progressed = True
            if len(selected_indices) == budget:
                break
        if not progressed:
            raise ValueError("effective clusters cannot fill the budget under residual cap")
    plan = tuple(geometry.case_ids[index] for index in selected_indices)
    if len(plan) != len(set(plan)) or len(plan) != budget:
        raise ValueError("center plan is not a unique budget-30 plan")
    return plan


@dataclass(frozen=True)
class PreparedConfigurationResult:
    evaluation: CandidateStructureEvaluation
    candidate: TuningCandidate | None
    case_free_summary: dict[str, Any]


def prepare_configuration_result(
    geometry: FittedHDBSCANGeometry,
) -> PreparedConfigurationResult:
    if not isinstance(geometry, FittedHDBSCANGeometry):
        raise ValueError("configuration preparation requires fitted geometry")
    evaluation, support, largest = _structure_evaluation(geometry)
    geometry_hash = geometry.geometry_sha256
    candidate = None
    if evaluation.eligible:
        candidate = build_tuning_candidate(
            evaluation,
            pool_case_ids=geometry.case_ids,
            ordered_pool_sha256=ordered_pool_sha256(geometry.case_ids),
            center_plan_fn=lambda *, configuration, seed, pool_case_ids, budget: (
                build_seeded_center_plan(
                    geometry,
                    seed=seed,
                    pool_case_ids=pool_case_ids,
                    budget=budget,
                    _geometry_sha256=geometry_hash,
                )
            ),
        )
    summary = {
        "configuration_id": geometry.configuration.configuration_id,
        "round_id": geometry.configuration.round_id,
        "configuration": geometry.configuration.to_dict(),
        "geometry_sha256": geometry_hash,
        "backend": geometry.backend,
        "backend_parameters": dict(geometry.backend_parameters),
        "seed_influences_geometry": False,
        "geometry_reused_for_acquisition_seeds": list(ACQUISITION_SEEDS),
        "effective_support": EFFECTIVE_SUPPORT,
        "effective_cluster_count": evaluation.seed_summaries[0].effective_cluster_count,
        "effective_coverage_numerator": support,
        "effective_coverage_denominator": CANDIDATE_POOL_SIZE,
        "largest_effective_cluster_numerator": largest,
        "largest_effective_cluster_denominator": CANDIDATE_POOL_SIZE,
        "seed_summaries": [item.to_dict() for item in evaluation.seed_summaries],
        "eligible": evaluation.eligible,
        "rejection_reasons": list(evaluation.rejection_reasons),
        "query_plan_status": "frozen" if candidate is not None else "not_created",
        "query_plan_triple_sha256": (
            candidate.query_plan_triple_hash if candidate is not None else None
        ),
    }
    return PreparedConfigurationResult(evaluation, candidate, summary)


@dataclass(frozen=True)
class StructureGridBuild:
    manifest: dict[str, Any]
    structure_summary: dict[str, Any]


def build_structure_grid(
    candidate: Any,
    *,
    expected_matrix_sha256: str = EXPECTED_REPRESENTATION_MATRIX_SHA256,
    expected_ordered_pool_sha256: str = EXPECTED_ORDERED_POOL_SHA256,
    fit_geometry_fn: Callable[..., FittedHDBSCANGeometry] | None = None,
) -> StructureGridBuild:
    pool_hash = validate_rebuilt_representation(
        candidate,
        expected_matrix_sha256=expected_matrix_sha256,
        expected_ordered_pool_sha256=expected_ordered_pool_sha256,
    )
    contract = build_default_contract()
    configurations = contract.round_a_configurations + contract.round_b_configurations
    fitter = fit_geometry_fn or fit_configuration_geometry
    prepared: list[PreparedConfigurationResult] = []
    for configuration in configurations:
        geometry = fitter(configuration, candidate=candidate)
        if geometry.configuration != configuration:
            raise ValueError("structure fitter returned the wrong configuration")
        if geometry.representation_matrix_sha256 != candidate.matrix_sha256:
            raise ValueError("structure fitter returned the wrong representation hash")
        prepared.append(prepare_configuration_result(geometry))
    evaluations = tuple(item.evaluation for item in prepared)
    legal_candidates = tuple(
        item.candidate for item in prepared if item.candidate is not None
    )
    deduplication = deduplicate_query_plan_triples(legal_candidates)
    manifest = build_tuning_manifest(
        contract,
        evaluations,
        deduplication,
    ).to_dict()
    summary = {
        "schema_version": STRUCTURE_SUMMARY_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "representation_id": REPRESENTATION_ID,
        "representation_matrix_sha256": candidate.matrix_sha256,
        "ordered_pool_sha256": pool_hash,
        "candidate_pool_count": CANDIDATE_POOL_SIZE,
        "candidate_population": "full_observable_outer_train_not_sampled",
        "attempted_configuration_count": len(configurations),
        "geometry_fit_count": len(prepared),
        "seed_influences_geometry": False,
        "acquisition_seeds": list(ACQUISITION_SEEDS),
        "legal_configuration_count": len(legal_candidates),
        "eligible_configuration_count": len(legal_candidates),
        "excluded_configuration_count": len(configurations) - len(legal_candidates),
        "canonical_candidate_count": len(deduplication.canonical_candidates),
        "duplicate_configuration_count": len(deduplication.duplicate_mappings),
        "batch_count": len(manifest["batches"]),
        "ground_truth_used_for_ranking": False,
        "rcl_launched": False,
        "configurations": [item.case_free_summary for item in prepared],
    }
    summary["structure_summary_sha256"] = semantic_sha256(summary)
    return StructureGridBuild(manifest=manifest, structure_summary=summary)


def _finalize_compatibility(
    audit: Mapping[str, Any], *, candidate: Any, ordered_pool_hash: str
) -> dict[str, Any]:
    value = dict(audit)
    value.pop("compatibility_sha256", None)
    value.update(
        {
            "representation_id": REPRESENTATION_ID,
            "rebuilt_representation_matrix_sha256": candidate.matrix_sha256,
            "expected_representation_matrix_sha256": (
                EXPECTED_REPRESENTATION_MATRIX_SHA256
            ),
            "representation_matrix_exact_match": True,
            "rebuilt_ordered_pool_sha256": ordered_pool_hash,
            "expected_ordered_pool_sha256": EXPECTED_ORDERED_POOL_SHA256,
            "ordered_pool_exact_match": True,
            "candidate_pool_count": CANDIDATE_POOL_SIZE,
        }
    )
    value["compatibility_sha256"] = semantic_sha256(value)
    return value


@dataclass(frozen=True)
class StructureGridArtifacts:
    representation_compatibility: Mapping[str, Any]
    structure_summary: Mapping[str, Any]
    manifest: Mapping[str, Any]


def prepare_structure_grid_artifacts(
    *,
    workspace_root: str | Path,
    descriptor_path: str | Path,
) -> tuple[StructureGridArtifacts, StructureGridBuild]:
    descriptor, initial_audit = validate_sealed_representation_input(
        workspace_root=workspace_root,
        descriptor_path=descriptor_path,
    )
    candidate = rebuild_global_pca_dim32(
        workspace_root=workspace_root,
        descriptor=descriptor,
    )
    pool_hash = validate_rebuilt_representation(candidate)
    grid = build_structure_grid(candidate)
    compatibility = _finalize_compatibility(
        initial_audit,
        candidate=candidate,
        ordered_pool_hash=pool_hash,
    )
    return (
        StructureGridArtifacts(
            representation_compatibility=compatibility,
            structure_summary=grid.structure_summary,
            manifest=grid.manifest,
        ),
        grid,
    )


def publish_structure_grid_artifacts(
    artifacts: StructureGridArtifacts,
    *,
    isolated_workspace_root: str | Path,
    first_stage_root: str | Path,
    output_root: str | Path,
    followup_root: str | Path,
) -> dict[str, CanonicalJsonPublication]:
    if not isinstance(artifacts, StructureGridArtifacts):
        raise ValueError("structure-grid publication requires typed artifacts")
    followup, workspace, outputs, first_stage = _validate_publication_roots(
        isolated_workspace_root=isolated_workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
        followup_root=followup_root,
    )

    targets = {
        "representation_compatibility": followup / "representation-compatibility.json",
        "structure_summary": followup / "structure-grid-summary.json",
        "manifest": followup / "tuning-manifest.json",
    }
    payloads = {
        "representation_compatibility": artifacts.representation_compatibility,
        "structure_summary": artifacts.structure_summary,
        "manifest": artifacts.manifest,
    }
    roots = {
        "isolated_workspace_root": workspace,
        "first_stage_root": first_stage,
        "output_root": outputs,
        "allowed_followup_root": followup,
    }
    for name, target in targets.items():
        preflight_canonical_json_idempotent(target, payloads[name], **roots)
    return {
        name: write_canonical_json_idempotent(target, payloads[name], **roots)
        for name, target in targets.items()
    }


def _validate_publication_roots(
    *,
    isolated_workspace_root: str | Path,
    first_stage_root: str | Path,
    output_root: str | Path,
    followup_root: str | Path,
) -> tuple[Path, Path, Path, Path]:
    followup = validate_frozen_tuning_followup_root(
        followup_root,
        isolated_workspace_root=isolated_workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
    )
    workspace = Path(isolated_workspace_root).resolve(strict=True)
    outputs = Path(output_root).resolve(strict=True)
    first_stage = Path(first_stage_root).resolve(strict=True)
    return followup, workspace, outputs, first_stage


def prepare_and_publish_structure_grid(
    *,
    descriptor_path: str | Path,
    workspace_root: str | Path,
    output_root: str | Path,
    first_stage_root: str | Path,
    followup_root: str | Path,
) -> dict[str, Any]:
    _validate_publication_roots(
        isolated_workspace_root=workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
        followup_root=followup_root,
    )
    artifacts, grid = prepare_structure_grid_artifacts(
        workspace_root=workspace_root,
        descriptor_path=descriptor_path,
    )
    publications = publish_structure_grid_artifacts(
        artifacts,
        isolated_workspace_root=workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
        followup_root=followup_root,
    )
    summary = grid.structure_summary
    manifest = grid.manifest
    return {
        "attempted_configuration_count": summary["attempted_configuration_count"],
        "eligible_configuration_count": summary["eligible_configuration_count"],
        "excluded_configuration_count": summary["excluded_configuration_count"],
        "duplicate_configuration_count": summary["duplicate_configuration_count"],
        "canonical_candidate_count": summary["canonical_candidate_count"],
        "batch_count": summary["batch_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": publications["manifest"].path.as_posix(),
        "representation_compatibility_sha256": artifacts.representation_compatibility[
            "compatibility_sha256"
        ],
        "representation_compatibility_path": publications[
            "representation_compatibility"
        ].path.as_posix(),
        "structure_summary_sha256": summary["structure_summary_sha256"],
        "structure_summary_path": publications["structure_summary"].path.as_posix(),
        "rcl_launched": False,
        "ground_truth_used_for_ranking": False,
    }


__all__ = [
    "EXPECTED_DESCRIPTOR_INPUT_SHA256",
    "EXPECTED_ORDERED_POOL_SHA256",
    "EXPECTED_REPRESENTATION_MATRIX_SHA256",
    "FittedHDBSCANGeometry",
    "PreparedConfigurationResult",
    "REPRESENTATION_SCOPED_CODE_PATHS",
    "StructureGridArtifacts",
    "StructureGridBuild",
    "backend_parameters",
    "build_seeded_center_plan",
    "build_structure_grid",
    "fit_configuration_geometry",
    "ordered_pool_sha256",
    "prepare_and_publish_structure_grid",
    "prepare_configuration_result",
    "prepare_structure_grid_artifacts",
    "publish_structure_grid_artifacts",
    "rebuild_global_pca_dim32",
    "semantic_sha256",
    "validate_rebuilt_representation",
    "validate_sealed_representation_input",
]
