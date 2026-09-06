"""Label-free feature boundary for combined active learning 2.0.

This module owns the last schema check before observable case features may
enter a clustering matrix.  Case IDs remain row keys outside the matrix;
identity, label, absolute/calendar time, and split-membership fields are
inventoried as exclusions and can never cross this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances


FEATURE_INVENTORY_SCHEMA_VERSION = (
    "rcl-active-learning-combined-2.0-feature-inventory-v1"
)
MODALITY_STANDARDIZER_SCHEMA_VERSION = (
    "rcl-active-learning-combined-2.0-modality-standardizer-v1"
)
MODALITY_BALANCE_SCHEMA_VERSION = (
    "rcl-active-learning-combined-2.0-modality-balance-v1"
)
DIRECT_FUSION_SCHEMA_VERSION = (
    "rcl-active-learning-combined-2.0-direct-fusion-v1"
)
REPRESENTATION_MATRIX_SCHEMA_VERSION = (
    "rcl-active-learning-combined-2.0-representation-matrix-v1"
)
REPRESENTATION_MANIFEST_SCHEMA_VERSION = (
    "rcl-active-learning-combined-2.0-representation-manifest-v1"
)
GLOBAL_REPRESENTATION_IDS = (
    "balanced_none",
    "global_pca_var90",
    "global_pca_var95",
    "global_pca_dim8",
    "global_pca_dim16",
    "global_pca_dim32",
)
PER_MODALITY_REPRESENTATION_IDS = (
    "per_modality_pca_var90",
    "per_modality_pca_var95",
    "per_modality_pca_dim8",
)
LATE_AFFINITY_REPRESENTATION_ID = "late_affinity_mds16"
FORMAL_REPRESENTATION_IDS = (
    *GLOBAL_REPRESENTATION_IDS,
    *PER_MODALITY_REPRESENTATION_IDS,
    LATE_AFFINITY_REPRESENTATION_ID,
)
DIAGNOSTIC_REPRESENTATION_IDS = (
    "diagnostic_umap",
    "diagnostic_legacy_rcl_guided",
)
FEATURE_POLICY_VERSION = "identity-and-time-exclusion-v1"
FORMAL_DATASETS = frozenset(("rcabench", "aiops2022_pre"))
MODALITIES = ("metric", "log", "trace", "topology", "time")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEPARATOR = re.compile(r"[^a-z0-9]+")
_AGGREGATE_SUFFIXES = frozenset(
    ("mean", "max", "min", "std", "median", "sum", "p95", "p99")
)
_ABSOLUTE_TIME_VALUE_SUFFIX = (
    r"(?:$|_(?:value|utc|gmt|epoch|seconds|milliseconds|millis|ms|"
    r"nanoseconds|ns|of_peak|at_peak)(?:_|$))"
)


_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "case_identity",
        re.compile(r"(?:^|_)case_(?:id|uuid|name|key|index)(?:_|$)"),
    ),
    (
        "service_identity",
        re.compile(r"(?:^|_)(?:service|svc)_(?:id|uuid|name|key)(?:_|$)"),
    ),
    (
        "fault_identity",
        re.compile(r"(?:^|_)fault_(?:type|id|name|class|label)(?:_|$)"),
    ),
    (
        "injection_identity",
        re.compile(r"(?:^|_)injection_(?:type|id|uuid|name|class|label)(?:_|$)"),
    ),
    (
        "absolute_time",
        re.compile(
            r"(?:^|_)(?:"
            r"(?:absolute_)?timestamp"
            + _ABSOLUTE_TIME_VALUE_SUFFIX
            + r"|epoch"
            + _ABSOLUTE_TIME_VALUE_SUFFIX
            + r"|unix_time"
            + _ABSOLUTE_TIME_VALUE_SUFFIX
            + r"|wall_clock"
            + _ABSOLUTE_TIME_VALUE_SUFFIX
            + r"|"
            r"(?:start|end|event|injection|incident|case|window)_"
            r"(?:ts|timestamp|time|datetime)"
            + _ABSOLUTE_TIME_VALUE_SUFFIX
            + r")"
        ),
    ),
    (
        "calendar_time",
        re.compile(
            r"(?:^|_)(?:time_of_day|hour_of_day|day_of_week|weekday|"
            r"week_of_year|month_of_year|calendar_time)(?:_|$)"
        ),
    ),
    (
        "split_membership",
        re.compile(
            r"(?:^|_)(?:split|split_id|fold|fold_id|partition|outer_train|"
            r"outer_test|held_out|train_indicator|test_indicator)(?:_|$)"
        ),
    ),
)


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_feature_name(value: str) -> str:
    parts = value.casefold().split(".")
    while len(parts) > 1 and parts[-1] in _AGGREGATE_SUFFIXES:
        parts.pop()
    return _SEPARATOR.sub("_", ".".join(parts)).strip("_")


def forbidden_feature_reason(feature_name: str) -> str | None:
    """Return the frozen exclusion class for one matrix feature name."""

    if not isinstance(feature_name, str) or not feature_name.strip():
        raise ValueError("feature names must be non-empty strings")
    if feature_name != feature_name.strip():
        raise ValueError("feature names must not contain surrounding whitespace")
    normalized = _normalized_feature_name(feature_name)
    for reason, pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(normalized):
            return reason
    return None


def _validate_feature_layout(
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(feature_names_by_modality, Mapping):
        raise ValueError("feature inventory must be a modality mapping")
    if set(feature_names_by_modality) != set(MODALITIES):
        raise ValueError("feature inventory must contain exactly the canonical modalities")
    canonical: dict[str, tuple[str, ...]] = {}
    for modality in MODALITIES:
        raw_names = feature_names_by_modality[modality]
        if isinstance(raw_names, (str, bytes, bytearray)) or not isinstance(
            raw_names, Sequence
        ):
            raise ValueError(f"{modality} feature names must be an ordered sequence")
        names = tuple(raw_names)
        if not names:
            raise ValueError(f"{modality} feature names must not be empty")
        if any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            for name in names
        ):
            raise ValueError(f"{modality} feature names must be canonical strings")
        if len(names) != len(set(names)):
            raise ValueError(f"{modality} feature names must be unique")
        canonical[modality] = names
    return canonical


@dataclass(frozen=True)
class FeatureExclusion:
    modality: str
    feature_name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "modality": self.modality,
            "feature_name": self.feature_name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FeatureInventory:
    dataset_id: str
    modalities: tuple[str, ...]
    source_artifact_sha256: str
    raw_features_by_modality: Mapping[str, tuple[str, ...]]
    eligible_features_by_modality: Mapping[str, tuple[str, ...]]
    exclusions: tuple[FeatureExclusion, ...]
    raw_feature_count: int
    eligible_feature_count: int
    excluded_feature_count: int
    schema_version: str = FEATURE_INVENTORY_SCHEMA_VERSION
    policy_version: str = FEATURE_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "dataset_id": self.dataset_id,
            "modalities": list(self.modalities),
            "source_artifact_sha256": self.source_artifact_sha256,
            "raw_features_by_modality": {
                modality: list(self.raw_features_by_modality[modality])
                for modality in MODALITIES
            },
            "eligible_features_by_modality": {
                modality: list(self.eligible_features_by_modality[modality])
                for modality in MODALITIES
            },
            "exclusions": [item.to_dict() for item in self.exclusions],
            "raw_feature_count": self.raw_feature_count,
            "eligible_feature_count": self.eligible_feature_count,
            "excluded_feature_count": self.excluded_feature_count,
        }

    @property
    def inventory_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class ModalityStandardizer:
    fit_case_ids: tuple[str, ...]
    fit_population_sha256: str
    feature_names_by_modality: Mapping[str, tuple[str, ...]]
    retained_feature_names_by_modality: Mapping[str, tuple[str, ...]]
    dropped_zero_variance_by_modality: Mapping[str, tuple[str, ...]]
    retained_indices_by_modality: Mapping[str, tuple[int, ...]]
    means_by_modality: Mapping[str, tuple[float, ...]]
    scales_by_modality: Mapping[str, tuple[float, ...]]
    observed_case_count_by_modality: Mapping[str, int]
    schema_version: str = MODALITY_STANDARDIZER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fit_case_ids": list(self.fit_case_ids),
            "fit_population_sha256": self.fit_population_sha256,
            "feature_names_by_modality": {
                modality: list(self.feature_names_by_modality[modality])
                for modality in MODALITIES
            },
            "retained_feature_names_by_modality": {
                modality: list(self.retained_feature_names_by_modality[modality])
                for modality in MODALITIES
            },
            "dropped_zero_variance_by_modality": {
                modality: list(self.dropped_zero_variance_by_modality[modality])
                for modality in MODALITIES
            },
            "retained_indices_by_modality": {
                modality: list(self.retained_indices_by_modality[modality])
                for modality in MODALITIES
            },
            "means_by_modality": {
                modality: list(self.means_by_modality[modality])
                for modality in MODALITIES
            },
            "scales_by_modality": {
                modality: list(self.scales_by_modality[modality])
                for modality in MODALITIES
            },
            "observed_case_count_by_modality": dict(
                self.observed_case_count_by_modality
            ),
        }

    @property
    def artifact_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class StandardizedModalityBlocks:
    case_ids: tuple[str, ...]
    blocks_by_modality: Mapping[str, np.ndarray]
    feature_names_by_modality: Mapping[str, tuple[str, ...]]
    masks_by_modality: Mapping[str, np.ndarray]
    coverages_by_modality: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ModalityContributionRecord:
    modality: str
    dimension: int
    observed_case_count: int
    pair_count: int
    positive_pair_count: int
    zero_distance_pair_fraction: float
    raw_median_pairwise_distance: float
    block_multiplier: float
    scaled_median_pairwise_distance: float
    active: bool
    inactive_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "dimension": self.dimension,
            "observed_case_count": self.observed_case_count,
            "pair_count": self.pair_count,
            "positive_pair_count": self.positive_pair_count,
            "zero_distance_pair_fraction": self.zero_distance_pair_fraction,
            "raw_median_pairwise_distance": self.raw_median_pairwise_distance,
            "block_multiplier": self.block_multiplier,
            "scaled_median_pairwise_distance": (
                self.scaled_median_pairwise_distance
            ),
            "active": self.active,
            "inactive_reason": self.inactive_reason,
        }


@dataclass(frozen=True)
class ModalityContributionAudit:
    records_by_modality: Mapping[str, ModalityContributionRecord]
    active_modalities: tuple[str, ...]
    scaled_median_ratio: float
    passed: bool
    tolerance: float = 1e-10
    schema_version: str = MODALITY_BALANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": "median_positive_observed_pairwise_distance",
            "records_by_modality": {
                modality: self.records_by_modality[modality].to_dict()
                for modality in MODALITIES
            },
            "active_modalities": list(self.active_modalities),
            "scaled_median_ratio": self.scaled_median_ratio,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BalancedModalityBlocks:
    case_ids: tuple[str, ...]
    blocks_by_modality: Mapping[str, np.ndarray]
    feature_names_by_modality: Mapping[str, tuple[str, ...]]
    masks_by_modality: Mapping[str, np.ndarray]
    coverages_by_modality: Mapping[str, np.ndarray]
    contribution_audit: ModalityContributionAudit


@dataclass(frozen=True)
class FusedRepresentationMatrix:
    case_ids: tuple[str, ...]
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    modality_slices: Mapping[str, tuple[int, int]]
    modality_weights: Mapping[str, float]
    effective_distance_multiplier_by_modality: Mapping[str, float]
    transform: str = "direct_concatenation_no_feature_rescaling"
    schema_version: str = DIRECT_FUSION_SCHEMA_VERSION

    @property
    def matrix_sha256(self) -> str:
        return _semantic_sha256(
            {
                "case_ids": list(self.case_ids),
                "feature_names": list(self.feature_names),
                "matrix": self.matrix.tolist(),
                "modality_slices": {
                    modality: list(self.modality_slices[modality])
                    for modality in MODALITIES
                },
                "modality_weights": dict(self.modality_weights),
                "transform": self.transform,
            }
        )


@dataclass(frozen=True)
class RepresentationMatrixCandidate:
    representation_id: str
    family: str
    case_ids: tuple[str, ...]
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    formal_eligible: bool
    source_matrix_sha256: str
    parameters: Mapping[str, Any]
    schema_version: str = REPRESENTATION_MATRIX_SCHEMA_VERSION

    @property
    def fit_population_sha256(self) -> str:
        return _semantic_sha256(list(self.case_ids))

    @property
    def matrix_sha256(self) -> str:
        return _semantic_sha256(
            {
                "schema_version": self.schema_version,
                "representation_id": self.representation_id,
                "family": self.family,
                "case_ids": list(self.case_ids),
                "matrix": self.matrix.tolist(),
                "feature_names": list(self.feature_names),
                "formal_eligible": self.formal_eligible,
                "source_matrix_sha256": self.source_matrix_sha256,
                "parameters": dict(self.parameters),
            }
        )


@dataclass(frozen=True)
class LateAffinityGeometry:
    case_ids: tuple[str, ...]
    distance_matrix: np.ndarray
    jointly_observed_weight_matrix: np.ndarray
    jointly_observed_modality_count: np.ndarray
    pair_count: int
    no_overlap_pair_count: int
    no_overlap_distance: float

    @property
    def geometry_sha256(self) -> str:
        return _semantic_sha256(
            {
                "case_ids": list(self.case_ids),
                "distance_matrix": self.distance_matrix.tolist(),
                "jointly_observed_weight_matrix": (
                    self.jointly_observed_weight_matrix.tolist()
                ),
                "jointly_observed_modality_count": (
                    self.jointly_observed_modality_count.tolist()
                ),
                "pair_count": self.pair_count,
                "no_overlap_pair_count": self.no_overlap_pair_count,
                "no_overlap_distance": self.no_overlap_distance,
                "distance_definition": (
                    "coverage_geometric_mean_weighted_observed_modality_mean"
                ),
            }
        )


@dataclass(frozen=True)
class DiagnosticRepresentationAdapter:
    adapter_id: str
    family: str
    backend: str
    diagnostic_only: bool = True
    formal_eligible: bool = False
    selection_eligible: bool = False

    def __post_init__(self) -> None:
        if self.adapter_id not in DIAGNOSTIC_REPRESENTATION_IDS:
            raise ValueError("unknown diagnostic representation adapter")
        if (
            not self.diagnostic_only
            or self.formal_eligible
            or self.selection_eligible
        ):
            raise ValueError("diagnostic adapters can never be selection eligible")


@dataclass(frozen=True)
class RepresentationFitScope:
    dataset_id: str
    stage: str
    held_out_fault_type: str | None
    fit_case_ids: tuple[str, ...]
    original_outer_train_case_ids: tuple[str, ...]
    original_outer_test_case_ids: tuple[str, ...]
    test_only_case_ids: tuple[str, ...]
    unused_outer_test_case_ids: tuple[str, ...]
    split_manifest_sha256: str
    label_access: str

    @property
    def fit_population_sha256(self) -> str:
        return _semantic_sha256(list(self.fit_case_ids))

    @property
    def scope_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "stage": self.stage,
            "held_out_fault_type": self.held_out_fault_type,
            "fit_case_ids": list(self.fit_case_ids),
            "original_outer_train_case_ids": list(
                self.original_outer_train_case_ids
            ),
            "original_outer_test_case_ids": list(self.original_outer_test_case_ids),
            "test_only_case_ids": list(self.test_only_case_ids),
            "unused_outer_test_case_ids": list(self.unused_outer_test_case_ids),
            "split_manifest_sha256": self.split_manifest_sha256,
            "label_access": self.label_access,
        }


@dataclass(frozen=True)
class FittedStandardizedRepresentation:
    scope: RepresentationFitScope
    standardizer: ModalityStandardizer
    standardized: StandardizedModalityBlocks


@dataclass(frozen=True)
class RepresentationModalityManifest:
    modality: str
    input_feature_names: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    dropped_zero_variance_features: tuple[str, ...]
    input_dimension: int
    retained_dimension: int
    observed_case_count: int
    standardization_means: tuple[float, ...]
    standardization_scales: tuple[float, ...]
    block_multiplier: float
    raw_median_pairwise_distance: float
    scaled_median_pairwise_distance: float
    active: bool
    inactive_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "input_feature_names": list(self.input_feature_names),
            "retained_feature_names": list(self.retained_feature_names),
            "dropped_zero_variance_features": list(
                self.dropped_zero_variance_features
            ),
            "input_dimension": self.input_dimension,
            "retained_dimension": self.retained_dimension,
            "observed_case_count": self.observed_case_count,
            "standardization_means": list(self.standardization_means),
            "standardization_scales": list(self.standardization_scales),
            "block_multiplier": self.block_multiplier,
            "raw_median_pairwise_distance": self.raw_median_pairwise_distance,
            "scaled_median_pairwise_distance": (
                self.scaled_median_pairwise_distance
            ),
            "active": self.active,
            "inactive_reason": self.inactive_reason,
        }


@dataclass(frozen=True)
class RepresentationNumericalHealth:
    row_count: int
    output_dimension: int
    feature_name_count: int
    shape_consistent: bool
    all_finite: bool
    non_finite_value_count: int
    zero_variance_dimension_count: int
    numerical_rank: int
    minimum_value: float | None
    maximum_value: float | None
    maximum_absolute_value: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "output_dimension": self.output_dimension,
            "feature_name_count": self.feature_name_count,
            "shape_consistent": self.shape_consistent,
            "all_finite": self.all_finite,
            "non_finite_value_count": self.non_finite_value_count,
            "zero_variance_dimension_count": self.zero_variance_dimension_count,
            "numerical_rank": self.numerical_rank,
            "minimum_value": self.minimum_value,
            "maximum_value": self.maximum_value,
            "maximum_absolute_value": self.maximum_absolute_value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RepresentationManifest:
    dataset_id: str
    stage: str
    held_out_fault_type: str | None
    representation_id: str
    family: str
    formal_eligible: bool
    fit_case_ids: tuple[str, ...]
    fit_population_sha256: str
    fit_scope_sha256: str
    split_manifest_sha256: str
    feature_inventory_sha256: str
    feature_source_artifact_sha256: str
    feature_policy_version: str
    feature_exclusions: tuple[FeatureExclusion, ...]
    modalities: Mapping[str, RepresentationModalityManifest]
    reduction_parameters: Mapping[str, Any]
    row_count: int
    output_dimension: int
    source_matrix_sha256: str
    matrix_sha256: str
    numerical_health: RepresentationNumericalHealth
    schema_version: str = REPRESENTATION_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "stage": self.stage,
            "held_out_fault_type": self.held_out_fault_type,
            "representation_id": self.representation_id,
            "family": self.family,
            "formal_eligible": self.formal_eligible,
            "fit_case_ids": list(self.fit_case_ids),
            "fit_population_sha256": self.fit_population_sha256,
            "fit_scope_sha256": self.fit_scope_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "feature_inventory_sha256": self.feature_inventory_sha256,
            "feature_source_artifact_sha256": (
                self.feature_source_artifact_sha256
            ),
            "feature_policy_version": self.feature_policy_version,
            "feature_exclusions": [
                item.to_dict() for item in self.feature_exclusions
            ],
            "modalities": {
                modality: self.modalities[modality].to_dict()
                for modality in MODALITIES
            },
            "reduction_parameters": dict(self.reduction_parameters),
            "row_count": self.row_count,
            "output_dimension": self.output_dimension,
            "source_matrix_sha256": self.source_matrix_sha256,
            "matrix_sha256": self.matrix_sha256,
            "numerical_health": self.numerical_health.to_dict(),
        }

    @property
    def manifest_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


def _canonical_case_ids(case_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(case_ids, (str, bytes, bytearray)) or not isinstance(
        case_ids, Sequence
    ):
        raise ValueError("case IDs must be an ordered sequence")
    ids = tuple(case_ids)
    if (
        not ids
        or any(
            not isinstance(case_id, str)
            or not case_id
            or case_id != case_id.strip()
            for case_id in ids
        )
        or len(ids) != len(set(ids))
    ):
        raise ValueError("case IDs must be unique canonical strings")
    return tuple(sorted(ids))


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be a finite real number")
    return 0.0 if result == 0.0 else result


def _extract_modality_arrays(
    *,
    case_ids: tuple[str, ...],
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    feature_names_by_modality: Mapping[str, tuple[str, ...]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    if not isinstance(case_modalities, Mapping):
        raise ValueError("case modalities must be a mapping")
    values: dict[str, list[tuple[float, ...]]] = {
        modality: [] for modality in MODALITIES
    }
    masks: dict[str, list[bool]] = {modality: [] for modality in MODALITIES}
    coverages: dict[str, list[float]] = {
        modality: [] for modality in MODALITIES
    }
    required_entry_fields = {"feature_names", "values", "mask", "coverage"}
    for case_id in case_ids:
        if case_id not in case_modalities:
            raise ValueError(f"case modalities are missing candidate: {case_id}")
        record = case_modalities[case_id]
        if not isinstance(record, Mapping) or set(record) != set(MODALITIES):
            raise ValueError("each case must contain exactly the canonical modalities")
        for modality in MODALITIES:
            entry = record[modality]
            if not isinstance(entry, Mapping) or set(entry) != required_entry_fields:
                raise ValueError(f"{modality} entry fields are invalid")
            entry_names = entry["feature_names"]
            if isinstance(entry_names, (str, bytes, bytearray)) or not isinstance(
                entry_names, Sequence
            ):
                raise ValueError(f"{modality} feature layout is invalid")
            if tuple(entry_names) != feature_names_by_modality[modality]:
                raise ValueError(f"{modality} feature layout drifted")
            raw_values = entry["values"]
            if isinstance(raw_values, (str, bytes, bytearray)) or not isinstance(
                raw_values, Sequence
            ):
                raise ValueError(f"{modality} values must be an ordered sequence")
            if len(raw_values) != len(feature_names_by_modality[modality]):
                raise ValueError(f"{modality} values do not match feature layout")
            row = tuple(
                _finite_float(value, f"{case_id}.{modality}.values[{index}]")
                for index, value in enumerate(raw_values)
            )
            mask = entry["mask"]
            if isinstance(mask, bool) or not isinstance(mask, int) or mask not in (0, 1):
                raise ValueError(f"{modality} mask must be integer zero or one")
            coverage = _finite_float(entry["coverage"], f"{modality} coverage")
            if not 0.0 <= coverage <= 1.0:
                raise ValueError(f"{modality} coverage must be within [0, 1]")
            if mask == 0 and (coverage != 0.0 or any(row)):
                raise ValueError("missing modality requires zero values and coverage")
            if mask == 1 and coverage <= 0.0:
                raise ValueError("observed modality requires positive coverage")
            values[modality].append(row)
            masks[modality].append(bool(mask))
            coverages[modality].append(coverage)
    return (
        {
            modality: np.asarray(values[modality], dtype=float)
            for modality in MODALITIES
        },
        {
            modality: np.asarray(masks[modality], dtype=bool)
            for modality in MODALITIES
        },
        {
            modality: np.asarray(coverages[modality], dtype=float)
            for modality in MODALITIES
        },
    )


def fit_modality_standardizer(
    *,
    fit_case_ids: Sequence[str],
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> ModalityStandardizer:
    """Fit five independent observed-only z-score transforms."""

    ids = _canonical_case_ids(fit_case_ids)
    names = validate_clustering_matrix_features(feature_names_by_modality)
    values, masks, _coverages = _extract_modality_arrays(
        case_ids=ids,
        case_modalities=case_modalities,
        feature_names_by_modality=names,
    )
    retained_names: dict[str, tuple[str, ...]] = {}
    dropped_names: dict[str, tuple[str, ...]] = {}
    retained_indices: dict[str, tuple[int, ...]] = {}
    means: dict[str, tuple[float, ...]] = {}
    scales: dict[str, tuple[float, ...]] = {}
    observed_counts: dict[str, int] = {}
    for modality in MODALITIES:
        observed = masks[modality]
        observed_count = int(observed.sum())
        if observed_count:
            block_mean = values[modality][observed].mean(axis=0)
            block_scale = values[modality][observed].std(axis=0, ddof=0)
        else:
            dimension = values[modality].shape[1]
            block_mean = np.zeros(dimension, dtype=float)
            block_scale = np.zeros(dimension, dtype=float)
        kept = tuple(int(index) for index in np.flatnonzero(block_scale > 0.0))
        dropped = tuple(
            index for index in range(len(names[modality])) if index not in kept
        )
        retained_indices[modality] = kept
        retained_names[modality] = tuple(names[modality][index] for index in kept)
        dropped_names[modality] = tuple(names[modality][index] for index in dropped)
        means[modality] = tuple(float(value) for value in block_mean)
        scales[modality] = tuple(float(value) for value in block_scale)
        observed_counts[modality] = observed_count
    return ModalityStandardizer(
        fit_case_ids=ids,
        fit_population_sha256=_semantic_sha256(list(ids)),
        feature_names_by_modality=MappingProxyType(names),
        retained_feature_names_by_modality=MappingProxyType(retained_names),
        dropped_zero_variance_by_modality=MappingProxyType(dropped_names),
        retained_indices_by_modality=MappingProxyType(retained_indices),
        means_by_modality=MappingProxyType(means),
        scales_by_modality=MappingProxyType(scales),
        observed_case_count_by_modality=MappingProxyType(observed_counts),
    )


def transform_modality_blocks(
    *,
    artifact: ModalityStandardizer,
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    case_ids: Sequence[str],
) -> StandardizedModalityBlocks:
    """Apply one frozen standardizer without reading or changing its fit set."""

    if type(artifact) is not ModalityStandardizer:
        raise ValueError("artifact must be a ModalityStandardizer")
    ids = _canonical_case_ids(case_ids)
    names = {
        modality: artifact.feature_names_by_modality[modality]
        for modality in MODALITIES
    }
    values, masks, coverages = _extract_modality_arrays(
        case_ids=ids,
        case_modalities=case_modalities,
        feature_names_by_modality=names,
    )
    blocks: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        indices = np.asarray(
            artifact.retained_indices_by_modality[modality], dtype=int
        )
        if indices.size:
            means = np.asarray(artifact.means_by_modality[modality], dtype=float)[
                indices
            ]
            scales = np.asarray(artifact.scales_by_modality[modality], dtype=float)[
                indices
            ]
            transformed = (values[modality][:, indices] - means) / scales
            transformed[~masks[modality]] = 0.0
        else:
            transformed = np.zeros((len(ids), 0), dtype=float)
        transformed.setflags(write=False)
        masks[modality].setflags(write=False)
        coverages[modality].setflags(write=False)
        blocks[modality] = transformed
    return StandardizedModalityBlocks(
        case_ids=ids,
        blocks_by_modality=MappingProxyType(blocks),
        feature_names_by_modality=artifact.retained_feature_names_by_modality,
        masks_by_modality=MappingProxyType(masks),
        coverages_by_modality=MappingProxyType(coverages),
    )


def _validated_standardized_blocks(
    value: StandardizedModalityBlocks,
) -> tuple[
    tuple[str, ...],
    dict[str, np.ndarray],
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    if type(value) is not StandardizedModalityBlocks:
        raise ValueError("standardized blocks use an invalid schema")
    ids = _canonical_case_ids(value.case_ids)
    if ids != value.case_ids:
        raise ValueError("standardized block case order must be canonical")
    mappings = (
        value.blocks_by_modality,
        value.feature_names_by_modality,
        value.masks_by_modality,
        value.coverages_by_modality,
    )
    if any(not isinstance(item, Mapping) or set(item) != set(MODALITIES) for item in mappings):
        raise ValueError("standardized blocks must contain every canonical modality")
    blocks: dict[str, np.ndarray] = {}
    names: dict[str, tuple[str, ...]] = {}
    masks: dict[str, np.ndarray] = {}
    coverages: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        block = np.asarray(value.blocks_by_modality[modality], dtype=float)
        modality_names = tuple(value.feature_names_by_modality[modality])
        mask = np.asarray(value.masks_by_modality[modality], dtype=bool)
        coverage = np.asarray(value.coverages_by_modality[modality], dtype=float)
        if block.ndim != 2 or block.shape != (len(ids), len(modality_names)):
            raise ValueError(f"{modality} standardized block shape is invalid")
        if len(modality_names) != len(set(modality_names)) or any(
            not isinstance(name, str) or not name for name in modality_names
        ):
            raise ValueError(f"{modality} standardized feature names are invalid")
        if mask.shape != (len(ids),) or coverage.shape != (len(ids),):
            raise ValueError(f"{modality} observation vectors have invalid shape")
        if not np.isfinite(block).all() or not np.isfinite(coverage).all():
            raise ValueError(f"{modality} standardized block must be finite")
        if np.any((coverage < 0.0) | (coverage > 1.0)):
            raise ValueError(f"{modality} coverage must be within [0, 1]")
        if np.any(coverage[~mask] != 0.0) or np.any(coverage[mask] <= 0.0):
            raise ValueError(f"{modality} mask and coverage disagree")
        if block.shape[1] and np.any(block[~mask] != 0.0):
            raise ValueError(f"{modality} missing rows must be zero")
        blocks[modality] = block
        names[modality] = modality_names
        masks[modality] = mask
        coverages[modality] = coverage
    return ids, blocks, names, masks, coverages


def _block_scale_record(
    modality: str,
    block: np.ndarray,
    mask: np.ndarray,
) -> ModalityContributionRecord:
    dimension = int(block.shape[1])
    observed = block[mask]
    observed_count = int(observed.shape[0])
    pair_count = observed_count * (observed_count - 1) // 2
    positive = np.asarray((), dtype=float)
    zero_count = 0
    if dimension and pair_count:
        left, right = np.triu_indices(observed_count, k=1)
        distances = np.linalg.norm(observed[left] - observed[right], axis=1)
        positive = distances[distances > 0.0]
        zero_count = int(pair_count - positive.size)
    if dimension == 0:
        inactive_reason = "zero_dimension"
    elif observed_count < 2:
        inactive_reason = "fewer_than_two_observed_cases"
    elif not positive.size:
        inactive_reason = "no_positive_pairwise_distance"
    else:
        inactive_reason = None
    active = inactive_reason is None
    raw_median = float(np.median(positive)) if active else 0.0
    multiplier = 1.0 / raw_median if active else 1.0
    scaled_median = raw_median * multiplier if active else 0.0
    return ModalityContributionRecord(
        modality=modality,
        dimension=dimension,
        observed_case_count=observed_count,
        pair_count=pair_count,
        positive_pair_count=int(positive.size),
        zero_distance_pair_fraction=(
            0.0 if pair_count == 0 else zero_count / float(pair_count)
        ),
        raw_median_pairwise_distance=raw_median,
        block_multiplier=multiplier,
        scaled_median_pairwise_distance=scaled_median,
        active=active,
        inactive_reason=inactive_reason,
    )


def balance_modality_blocks(
    standardized: StandardizedModalityBlocks,
) -> BalancedModalityBlocks:
    """Equalize active modality geometry by robust observed-pair scale."""

    ids, blocks, names, masks, coverages = _validated_standardized_blocks(
        standardized
    )
    records = {
        modality: _block_scale_record(modality, blocks[modality], masks[modality])
        for modality in MODALITIES
    }
    balanced_blocks: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        balanced = blocks[modality] * records[modality].block_multiplier
        if balanced.shape[1]:
            balanced[~masks[modality]] = 0.0
        balanced.setflags(write=False)
        masks[modality].setflags(write=False)
        coverages[modality].setflags(write=False)
        balanced_blocks[modality] = balanced
    active = tuple(modality for modality in MODALITIES if records[modality].active)
    scaled_medians = tuple(
        records[modality].scaled_median_pairwise_distance for modality in active
    )
    ratio = (
        max(scaled_medians) / min(scaled_medians) if scaled_medians else float("inf")
    )
    tolerance = 1e-10
    audit = ModalityContributionAudit(
        records_by_modality=MappingProxyType(records),
        active_modalities=active,
        scaled_median_ratio=ratio,
        passed=bool(active) and ratio <= 1.0 + tolerance,
        tolerance=tolerance,
    )
    return BalancedModalityBlocks(
        case_ids=ids,
        blocks_by_modality=MappingProxyType(balanced_blocks),
        feature_names_by_modality=MappingProxyType(names),
        masks_by_modality=MappingProxyType(masks),
        coverages_by_modality=MappingProxyType(coverages),
        contribution_audit=audit,
    )


def fuse_balanced_blocks(
    balanced: BalancedModalityBlocks,
    *,
    modality_weights: Mapping[str, float] | None = None,
) -> FusedRepresentationMatrix:
    """Concatenate balanced blocks without fitting a fusion-time transform."""

    if type(balanced) is not BalancedModalityBlocks:
        raise ValueError("balanced blocks use an invalid schema")
    ids, blocks, names, _masks, _coverages = _validated_standardized_blocks(
        StandardizedModalityBlocks(
            case_ids=balanced.case_ids,
            blocks_by_modality=balanced.blocks_by_modality,
            feature_names_by_modality=balanced.feature_names_by_modality,
            masks_by_modality=balanced.masks_by_modality,
            coverages_by_modality=balanced.coverages_by_modality,
        )
    )
    if not balanced.contribution_audit.passed:
        raise ValueError("balanced blocks failed their contribution audit")
    raw_weights: Mapping[str, float]
    if modality_weights is None:
        raw_weights = {modality: 1.0 for modality in MODALITIES}
    else:
        raw_weights = modality_weights
    if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(MODALITIES):
        raise ValueError("modality weights must name every canonical modality")
    weights: dict[str, float] = {}
    multipliers: dict[str, float] = {}
    for modality in MODALITIES:
        raw = raw_weights[modality]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError("modality weights must be finite positive numbers")
        weight = float(raw)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("modality weights must be finite positive numbers")
        weights[modality] = weight
        multipliers[modality] = float(np.sqrt(weight))
    parts: list[np.ndarray] = []
    feature_names: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for modality in MODALITIES:
        part = blocks[modality] * multipliers[modality]
        parts.append(part)
        stop = offset + part.shape[1]
        slices[modality] = (offset, stop)
        feature_names.extend(f"{modality}.{name}" for name in names[modality])
        offset = stop
    matrix = (
        np.concatenate(parts, axis=1)
        if parts
        else np.zeros((len(ids), 0), dtype=float)
    )
    matrix.setflags(write=False)
    return FusedRepresentationMatrix(
        case_ids=ids,
        matrix=matrix,
        feature_names=tuple(feature_names),
        modality_slices=MappingProxyType(slices),
        modality_weights=MappingProxyType(weights),
        effective_distance_multiplier_by_modality=MappingProxyType(multipliers),
    )


def _pca_candidate(
    source: FusedRepresentationMatrix,
    *,
    representation_id: str,
    requested_dimension: int | None = None,
    requested_variance: float | None = None,
) -> RepresentationMatrixCandidate:
    if (requested_dimension is None) == (requested_variance is None):
        raise ValueError("PCA requires exactly one dimension selection policy")
    matrix = np.asarray(source.matrix, dtype=float)
    center = matrix.mean(axis=0)
    centered = matrix - center
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size:
        tolerance = (
            max(centered.shape)
            * np.finfo(float).eps
            * float(singular_values[0])
        )
    else:
        tolerance = 0.0
    admissible_rank = int(np.sum(singular_values > tolerance))
    if admissible_rank < 1:
        raise ValueError("global PCA source has no admissible numerical rank")
    variance = np.square(singular_values[:admissible_rank])
    ratios = variance / variance.sum()
    if requested_dimension is not None:
        if isinstance(requested_dimension, bool) or requested_dimension < 1:
            raise ValueError("requested PCA dimension must be positive")
        retained = min(int(requested_dimension), admissible_rank)
        dimension_capped = retained != requested_dimension
    else:
        if (
            requested_variance is None
            or not np.isfinite(requested_variance)
            or not 0.0 < requested_variance <= 1.0
        ):
            raise ValueError("requested PCA variance must be within (0, 1]")
        retained = int(
            np.searchsorted(np.cumsum(ratios), requested_variance, side="left")
            + 1
        )
        retained = min(retained, admissible_rank)
        dimension_capped = False
    components = right[:retained].copy()
    for index in range(retained):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    scores = centered @ components.T
    scores.setflags(write=False)
    parameters: dict[str, Any] = {
        "reduction": "global_pca",
        "centering": "candidate_pool_column_mean",
        "feature_rescaling_after_fusion": False,
        "admissible_rank": admissible_rank,
        "numerical_rank_tolerance": float(tolerance),
        "retained_dimension": retained,
        "dimension_capped": dimension_capped,
        "explained_variance_ratio_by_component": tuple(
            float(value) for value in ratios
        ),
        "retained_explained_variance": float(ratios[:retained].sum()),
        "component_sign_rule": "largest_absolute_loading_positive",
    }
    if requested_dimension is not None:
        parameters["requested_dimension"] = int(requested_dimension)
    else:
        parameters["requested_variance"] = float(requested_variance)
    return RepresentationMatrixCandidate(
        representation_id=representation_id,
        family="global_pca",
        case_ids=source.case_ids,
        matrix=scores,
        feature_names=tuple(f"pc{index + 1:03d}" for index in range(retained)),
        formal_eligible=True,
        source_matrix_sha256=source.matrix_sha256,
        parameters=MappingProxyType(parameters),
    )


def build_global_representation_candidates(
    balanced: BalancedModalityBlocks,
    *,
    modality_weights: Mapping[str, float] | None = None,
) -> dict[str, RepresentationMatrixCandidate]:
    """Build balanced no-reduction and five frozen global-PCA candidates."""

    fused = fuse_balanced_blocks(balanced, modality_weights=modality_weights)
    direct = RepresentationMatrixCandidate(
        representation_id="balanced_none",
        family="balanced_fusion",
        case_ids=fused.case_ids,
        matrix=fused.matrix,
        feature_names=fused.feature_names,
        formal_eligible=True,
        source_matrix_sha256=fused.matrix_sha256,
        parameters=MappingProxyType(
            {
                "reduction": "none",
                "fusion_transform": fused.transform,
                "modality_weights": dict(fused.modality_weights),
                "retained_dimension": int(fused.matrix.shape[1]),
            }
        ),
    )
    candidates = {direct.representation_id: direct}
    for representation_id, target in (
        ("global_pca_var90", 0.90),
        ("global_pca_var95", 0.95),
    ):
        candidates[representation_id] = _pca_candidate(
            fused,
            representation_id=representation_id,
            requested_variance=target,
        )
    for requested in (8, 16, 32):
        representation_id = f"global_pca_dim{requested}"
        candidates[representation_id] = _pca_candidate(
            fused,
            representation_id=representation_id,
            requested_dimension=requested,
        )
    if tuple(candidates) != GLOBAL_REPRESENTATION_IDS:
        raise RuntimeError("global representation registry drifted")
    return candidates


def _standardized_blocks_sha256(
    ids: tuple[str, ...],
    blocks: Mapping[str, np.ndarray],
    names: Mapping[str, tuple[str, ...]],
    masks: Mapping[str, np.ndarray],
    coverages: Mapping[str, np.ndarray],
) -> str:
    return _semantic_sha256(
        {
            "case_ids": list(ids),
            "blocks_by_modality": {
                modality: blocks[modality].tolist() for modality in MODALITIES
            },
            "feature_names_by_modality": {
                modality: list(names[modality]) for modality in MODALITIES
            },
            "masks_by_modality": {
                modality: masks[modality].astype(int).tolist()
                for modality in MODALITIES
            },
            "coverages_by_modality": {
                modality: coverages[modality].tolist()
                for modality in MODALITIES
            },
        }
    )


def _project_observed_pca(
    block: np.ndarray,
    mask: np.ndarray,
    *,
    requested_dimension: int | None = None,
    requested_variance: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (requested_dimension is None) == (requested_variance is None):
        raise ValueError("per-modality PCA requires one dimension selection policy")
    observed = block[mask]
    center = (
        observed.mean(axis=0)
        if observed.shape[0]
        else np.zeros(block.shape[1], dtype=float)
    )
    centered_observed = observed - center
    if centered_observed.size:
        _left, singular_values, right = np.linalg.svd(
            centered_observed, full_matrices=False
        )
    else:
        singular_values = np.asarray((), dtype=float)
        right = np.zeros((0, block.shape[1]), dtype=float)
    tolerance = (
        max(centered_observed.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    admissible_rank = int(np.sum(singular_values > tolerance))
    if admissible_rank:
        variance = np.square(singular_values[:admissible_rank])
        ratios = variance / variance.sum()
    else:
        ratios = np.asarray((), dtype=float)
    if requested_dimension is not None:
        if isinstance(requested_dimension, bool) or requested_dimension < 1:
            raise ValueError("requested PCA dimension must be positive")
        retained = min(int(requested_dimension), admissible_rank)
        dimension_capped = retained != requested_dimension
    else:
        if (
            requested_variance is None
            or not np.isfinite(requested_variance)
            or not 0.0 < requested_variance <= 1.0
        ):
            raise ValueError("requested PCA variance must be within (0, 1]")
        retained = (
            int(
                np.searchsorted(
                    np.cumsum(ratios), requested_variance, side="left"
                )
                + 1
            )
            if admissible_rank
            else 0
        )
        retained = min(retained, admissible_rank)
        dimension_capped = False
    components = right[:retained].copy()
    for index in range(retained):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    scores = np.zeros((block.shape[0], retained), dtype=float)
    if retained:
        scores[mask] = centered_observed @ components.T
    parameters: dict[str, Any] = {
        "centering": "observed_candidate_rows_only",
        "observed_case_count": int(mask.sum()),
        "input_dimension": int(block.shape[1]),
        "admissible_rank": admissible_rank,
        "numerical_rank_tolerance": float(tolerance),
        "retained_dimension": retained,
        "dimension_capped": dimension_capped,
        "explained_variance_ratio_by_component": tuple(
            float(value) for value in ratios
        ),
        "retained_explained_variance": (
            float(ratios[:retained].sum()) if retained else 0.0
        ),
        "component_sign_rule": "largest_absolute_loading_positive",
    }
    if requested_dimension is not None:
        parameters["requested_dimension"] = int(requested_dimension)
    else:
        parameters["requested_variance"] = float(requested_variance)
    return scores, parameters


def _per_modality_candidate(
    standardized: StandardizedModalityBlocks,
    *,
    representation_id: str,
    requested_dimension: int | None = None,
    requested_variance: float | None = None,
) -> RepresentationMatrixCandidate:
    ids, blocks, names, masks, coverages = _validated_standardized_blocks(
        standardized
    )
    reduced_blocks: dict[str, np.ndarray] = {}
    reduced_names: dict[str, tuple[str, ...]] = {}
    modality_parameters: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        scores, parameters = _project_observed_pca(
            blocks[modality],
            masks[modality],
            requested_dimension=requested_dimension,
            requested_variance=requested_variance,
        )
        reduced_blocks[modality] = scores
        reduced_names[modality] = tuple(
            f"pc{index + 1:03d}" for index in range(scores.shape[1])
        )
        modality_parameters[modality] = parameters
    reduced = StandardizedModalityBlocks(
        case_ids=ids,
        blocks_by_modality=MappingProxyType(reduced_blocks),
        feature_names_by_modality=MappingProxyType(reduced_names),
        masks_by_modality=MappingProxyType(masks),
        coverages_by_modality=MappingProxyType(coverages),
    )
    balanced = balance_modality_blocks(reduced)
    fused = fuse_balanced_blocks(balanced)
    source_sha256 = _standardized_blocks_sha256(
        ids, blocks, names, masks, coverages
    )
    return RepresentationMatrixCandidate(
        representation_id=representation_id,
        family="per_modality_pca",
        case_ids=ids,
        matrix=fused.matrix,
        feature_names=fused.feature_names,
        formal_eligible=True,
        source_matrix_sha256=source_sha256,
        parameters=MappingProxyType(
            {
                "reduction": "per_modality_pca_then_balanced_fusion",
                "feature_rescaling_after_fusion": False,
                "modality_parameters": modality_parameters,
                "post_reduction_balance": balanced.contribution_audit.to_dict(),
                "modality_slices": dict(fused.modality_slices),
                "retained_dimension": int(fused.matrix.shape[1]),
            }
        ),
    )


def build_per_modality_representation_candidates(
    standardized: StandardizedModalityBlocks,
) -> dict[str, RepresentationMatrixCandidate]:
    """Build the three frozen per-modality PCA then balanced-fusion candidates."""

    candidates: dict[str, RepresentationMatrixCandidate] = {}
    for representation_id, target in (
        ("per_modality_pca_var90", 0.90),
        ("per_modality_pca_var95", 0.95),
    ):
        candidates[representation_id] = _per_modality_candidate(
            standardized,
            representation_id=representation_id,
            requested_variance=target,
        )
    candidates["per_modality_pca_dim8"] = _per_modality_candidate(
        standardized,
        representation_id="per_modality_pca_dim8",
        requested_dimension=8,
    )
    if tuple(candidates) != PER_MODALITY_REPRESENTATION_IDS:
        raise RuntimeError("per-modality representation registry drifted")
    return candidates


def coverage_normalized_late_distance(
    balanced: BalancedModalityBlocks,
    *,
    no_overlap_distance: float = 1.0,
) -> LateAffinityGeometry:
    """Fuse modality distances only where each case pair shares observations."""

    if type(balanced) is not BalancedModalityBlocks:
        raise ValueError("balanced blocks use an invalid schema")
    ids, blocks, _names, masks, coverages = _validated_standardized_blocks(
        StandardizedModalityBlocks(
            case_ids=balanced.case_ids,
            blocks_by_modality=balanced.blocks_by_modality,
            feature_names_by_modality=balanced.feature_names_by_modality,
            masks_by_modality=balanced.masks_by_modality,
            coverages_by_modality=balanced.coverages_by_modality,
        )
    )
    if not balanced.contribution_audit.passed:
        raise ValueError("balanced blocks failed their contribution audit")
    if (
        isinstance(no_overlap_distance, bool)
        or not isinstance(no_overlap_distance, Real)
        or not np.isfinite(no_overlap_distance)
        or no_overlap_distance <= 0.0
    ):
        raise ValueError("no-overlap distance must be a finite positive number")
    count = len(ids)
    weighted_distance = np.zeros((count, count), dtype=float)
    total_weight = np.zeros((count, count), dtype=float)
    shared_count = np.zeros((count, count), dtype=int)
    for modality in MODALITIES:
        record = balanced.contribution_audit.records_by_modality[modality]
        if not record.active:
            continue
        pair_distance = pairwise_distances(blocks[modality], metric="euclidean")
        observed_coverage = np.where(
            masks[modality], coverages[modality], 0.0
        )
        weight = np.sqrt(np.outer(observed_coverage, observed_coverage))
        jointly_observed = np.outer(masks[modality], masks[modality])
        weight = np.where(jointly_observed, weight, 0.0)
        weighted_distance += weight * pair_distance
        total_weight += weight
        shared_count += jointly_observed.astype(int)
    distance = np.full((count, count), float(no_overlap_distance), dtype=float)
    available = total_weight > 0.0
    distance[available] = weighted_distance[available] / total_weight[available]
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(total_weight, 0.0)
    np.fill_diagonal(shared_count, 0)
    left, right = np.triu_indices(count, k=1)
    no_overlap_count = int(np.sum(total_weight[left, right] == 0.0))
    for matrix in (distance, total_weight, shared_count):
        matrix.setflags(write=False)
    return LateAffinityGeometry(
        case_ids=ids,
        distance_matrix=distance,
        jointly_observed_weight_matrix=total_weight,
        jointly_observed_modality_count=shared_count,
        pair_count=count * (count - 1) // 2,
        no_overlap_pair_count=no_overlap_count,
        no_overlap_distance=float(no_overlap_distance),
    )


def build_late_affinity_candidate(
    balanced: BalancedModalityBlocks,
    *,
    requested_dimension: int = 16,
) -> RepresentationMatrixCandidate:
    """Embed the frozen coverage-normalized late distance with classical MDS."""

    if isinstance(requested_dimension, bool) or requested_dimension != 16:
        raise ValueError("formal late-affinity MDS dimension must be 16")
    geometry = coverage_normalized_late_distance(balanced)
    count = len(geometry.case_ids)
    centering = np.eye(count) - np.ones((count, count)) / float(count)
    gram = (
        -0.5
        * centering
        @ np.square(geometry.distance_matrix)
        @ centering
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    maximum = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = max(1, count) * np.finfo(float).eps * maximum
    positive = tuple(int(index) for index in order if eigenvalues[index] > tolerance)
    retained_indices = positive[: min(requested_dimension, max(0, count - 1))]
    if not retained_indices:
        raise ValueError("late-affinity MDS has no positive admissible dimension")
    vectors = eigenvectors[:, retained_indices].copy()
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    retained_values = eigenvalues[list(retained_indices)]
    coordinates = vectors * np.sqrt(retained_values)
    coordinates.setflags(write=False)
    negative_mass = float(
        np.abs(eigenvalues[eigenvalues < -tolerance]).sum()
    )
    positive_mass = float(eigenvalues[eigenvalues > tolerance].sum())
    return RepresentationMatrixCandidate(
        representation_id=LATE_AFFINITY_REPRESENTATION_ID,
        family="coverage_normalized_late_affinity",
        case_ids=geometry.case_ids,
        matrix=coordinates,
        feature_names=tuple(
            f"mds{index + 1:03d}" for index in range(coordinates.shape[1])
        ),
        formal_eligible=True,
        source_matrix_sha256=geometry.geometry_sha256,
        parameters=MappingProxyType(
            {
                "distance": (
                    "coverage_geometric_mean_weighted_observed_modality_mean"
                ),
                "no_overlap_policy": "unit_distance",
                "no_overlap_distance": geometry.no_overlap_distance,
                "pair_count": geometry.pair_count,
                "no_overlap_pair_count": geometry.no_overlap_pair_count,
                "requested_dimension": requested_dimension,
                "retained_dimension": int(coordinates.shape[1]),
                "positive_eigenvalue_count": len(positive),
                "eigenvalue_tolerance": float(tolerance),
                "negative_eigenvalue_mass": negative_mass,
                "positive_eigenvalue_mass": positive_mass,
                "coordinate_sign_rule": "largest_absolute_coordinate_positive",
            }
        ),
    )


def build_formal_representation_candidates(
    standardized: StandardizedModalityBlocks,
) -> dict[str, RepresentationMatrixCandidate]:
    """Build the exact ten label-free candidates approved by the protocol."""

    balanced = balance_modality_blocks(standardized)
    candidates = build_global_representation_candidates(balanced)
    candidates.update(build_per_modality_representation_candidates(standardized))
    late = build_late_affinity_candidate(balanced)
    candidates[late.representation_id] = late
    validate_formal_representation_registry(candidates)
    return candidates


def build_diagnostic_representation_registry(
) -> dict[str, DiagnosticRepresentationAdapter]:
    """Return the two frozen, permanently ineligible diagnostic adapters."""

    registry = {
        "diagnostic_umap": DiagnosticRepresentationAdapter(
            adapter_id="diagnostic_umap",
            family="diagnostic_umap",
            backend="optional_umap_learn",
        ),
        "diagnostic_legacy_rcl_guided": DiagnosticRepresentationAdapter(
            adapter_id="diagnostic_legacy_rcl_guided",
            family="diagnostic_legacy_rcl_guided",
            backend="frozen_external_coordinates",
        ),
    }
    if tuple(registry) != DIAGNOSTIC_REPRESENTATION_IDS:
        raise RuntimeError("diagnostic representation registry drifted")
    return registry


def _adapt_diagnostic_coordinates(
    *,
    adapter: DiagnosticRepresentationAdapter,
    case_ids: Sequence[str],
    coordinates: Any,
    source_artifact_sha256: str,
    extra_parameters: Mapping[str, Any],
) -> RepresentationMatrixCandidate:
    ids = _canonical_case_ids(case_ids)
    if not isinstance(source_artifact_sha256, str) or not _SHA256.fullmatch(
        source_artifact_sha256
    ):
        raise ValueError("source_artifact_sha256 must be a lowercase SHA-256")
    matrix = np.asarray(coordinates, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(ids)
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("diagnostic coordinates must be a finite case matrix")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    parameters = {
        "diagnostic_only": True,
        "formal_eligible": False,
        "selection_eligible": False,
        "backend": adapter.backend,
        **dict(extra_parameters),
    }
    return RepresentationMatrixCandidate(
        representation_id=adapter.adapter_id,
        family=adapter.family,
        case_ids=ids,
        matrix=matrix,
        feature_names=tuple(
            f"diagnostic{index + 1:03d}" for index in range(matrix.shape[1])
        ),
        formal_eligible=False,
        source_matrix_sha256=source_artifact_sha256,
        parameters=MappingProxyType(parameters),
    )


def build_umap_diagnostic_candidate(
    source: RepresentationMatrixCandidate,
    *,
    embedding_backend: Callable[[np.ndarray], np.ndarray] | None = None,
) -> RepresentationMatrixCandidate:
    """Run a real or injected UMAP backend and force diagnostic-only status."""

    if type(source) is not RepresentationMatrixCandidate:
        raise ValueError("UMAP diagnostic source must be a representation candidate")
    backend_name = "injected_umap_backend"
    backend = embedding_backend
    if backend is None:
        try:
            from umap import UMAP
        except ImportError as error:
            raise RuntimeError(
                "diagnostic UMAP backend unavailable; umap-learn is not installed"
            ) from error
        reducer = UMAP(n_components=2, random_state=42, transform_seed=42)
        backend = reducer.fit_transform
        backend_name = "umap-learn"
    if not callable(backend):
        raise ValueError("UMAP embedding backend must be callable")
    coordinates = backend(np.array(source.matrix, dtype=float, copy=True))
    registered = build_diagnostic_representation_registry()["diagnostic_umap"]
    adapter = DiagnosticRepresentationAdapter(
        adapter_id=registered.adapter_id,
        family=registered.family,
        backend=backend_name,
    )
    return _adapt_diagnostic_coordinates(
        adapter=adapter,
        case_ids=source.case_ids,
        coordinates=coordinates,
        source_artifact_sha256=source.matrix_sha256,
        extra_parameters={
            "source_representation_id": source.representation_id,
            "random_state": 42,
        },
    )


def adapt_legacy_rcl_guided_candidate(
    *,
    case_ids: Sequence[str],
    coordinates: Any,
    source_artifact_sha256: str,
) -> RepresentationMatrixCandidate:
    """Wrap frozen RCL-guided coordinates without making them selectable."""

    adapter = build_diagnostic_representation_registry()[
        "diagnostic_legacy_rcl_guided"
    ]
    return _adapt_diagnostic_coordinates(
        adapter=adapter,
        case_ids=case_ids,
        coordinates=coordinates,
        source_artifact_sha256=source_artifact_sha256,
        extra_parameters={
            "uses_rcl_guidance": True,
            "source_policy": "frozen_external_diagnostic_only",
        },
    )


def validate_formal_representation_registry(
    candidates: Mapping[str, RepresentationMatrixCandidate],
) -> Mapping[str, RepresentationMatrixCandidate]:
    """Reject diagnostics, missing candidates, extras, and ordering drift."""

    if not isinstance(candidates, Mapping):
        raise ValueError("formal representation registry must be a mapping")
    diagnostic = [
        key
        for key, candidate in candidates.items()
        if isinstance(candidate, RepresentationMatrixCandidate)
        and not candidate.formal_eligible
    ]
    if diagnostic:
        raise ValueError(
            "diagnostic-only representation cannot enter formal selection: "
            + diagnostic[0]
        )
    if tuple(candidates) != FORMAL_REPRESENTATION_IDS or len(candidates) != 10:
        raise ValueError("formal representation registry must contain the exact ten")
    for representation_id, candidate in candidates.items():
        if (
            type(candidate) is not RepresentationMatrixCandidate
            or candidate.representation_id != representation_id
            or not candidate.formal_eligible
        ):
            raise ValueError("formal representation candidate schema is invalid")
    return candidates


def derive_representation_fit_scope(
    *,
    dataset_id: str,
    stage: str,
    split_records: Sequence[Mapping[str, Any]],
    split_manifest_sha256: str,
    held_out_fault_type: str | None = None,
) -> RepresentationFitScope:
    """Derive the only candidate population allowed to fit a representation."""

    if dataset_id not in FORMAL_DATASETS:
        raise ValueError("representation fit-scope dataset is outside the protocol")
    if not isinstance(split_manifest_sha256, str) or not _SHA256.fullmatch(
        split_manifest_sha256
    ):
        raise ValueError("split_manifest_sha256 must be a lowercase SHA-256")
    if stage not in {"ordinary", "strict_lofo"}:
        raise ValueError("representation fit stage must be ordinary or strict_lofo")
    if stage == "ordinary" and held_out_fault_type is not None:
        raise ValueError("ordinary representation fit cannot hold out a fault type")
    if stage == "strict_lofo" and (
        not isinstance(held_out_fault_type, str)
        or not held_out_fault_type.strip()
        or held_out_fault_type != held_out_fault_type.strip()
    ):
        raise ValueError("strict LOFO representation fit requires a held-out type")
    if isinstance(split_records, (str, bytes, bytearray)) or not isinstance(
        split_records, Sequence
    ):
        raise ValueError("split records must be an ordered sequence")
    required_fields = {"case_id", "fault_type", "incident_id", "split"}
    normalized: list[tuple[str, str, str, str]] = []
    for index, record in enumerate(split_records):
        if not isinstance(record, Mapping) or set(record) != required_fields:
            raise ValueError(f"split record {index} fields are invalid")
        values = tuple(record[field] for field in (
            "case_id",
            "fault_type",
            "incident_id",
            "split",
        ))
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in values
        ):
            raise ValueError(f"split record {index} contains a non-canonical value")
        case_id, fault_type, incident_id, split = values
        if split not in {"outer_train", "outer_test"}:
            raise ValueError("split membership must be outer_train or outer_test")
        normalized.append((case_id, fault_type, incident_id, split))
    if not normalized:
        raise ValueError("split records must not be empty")
    case_ids = [record[0] for record in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("split case IDs must be unique")
    normalized.sort(key=lambda record: record[0])
    outer_train = tuple(
        case_id
        for case_id, _fault_type, _incident_id, split in normalized
        if split == "outer_train"
    )
    outer_test = tuple(
        case_id
        for case_id, _fault_type, _incident_id, split in normalized
        if split == "outer_test"
    )
    if not outer_train or not outer_test:
        raise ValueError("split records require non-empty outer_train and outer_test")
    if stage == "ordinary":
        fit_ids = outer_train
        test_only = outer_test
        unused_outer_test: tuple[str, ...] = ()
        label_access = "none"
    else:
        fit_ids = tuple(
            case_id
            for case_id, fault_type, _incident_id, split in normalized
            if split == "outer_train" and fault_type != held_out_fault_type
        )
        test_only = tuple(
            case_id
            for case_id, fault_type, _incident_id, _split in normalized
            if fault_type == held_out_fault_type
        )
        unused_outer_test = tuple(
            case_id
            for case_id, fault_type, _incident_id, split in normalized
            if split == "outer_test" and fault_type != held_out_fault_type
        )
        label_access = "lofo_fold_construction_only"
        if not fit_ids or not test_only:
            raise ValueError("strict LOFO fit scope requires fit and held-out cases")
    return RepresentationFitScope(
        dataset_id=dataset_id,
        stage=stage,
        held_out_fault_type=held_out_fault_type,
        fit_case_ids=fit_ids,
        original_outer_train_case_ids=outer_train,
        original_outer_test_case_ids=outer_test,
        test_only_case_ids=test_only,
        unused_outer_test_case_ids=unused_outer_test,
        split_manifest_sha256=split_manifest_sha256,
        label_access=label_access,
    )


def assert_representation_fit_population(
    scope: RepresentationFitScope,
    standardizer: ModalityStandardizer,
) -> None:
    """Reject a transform fitted on any case outside the frozen scope."""

    if type(scope) is not RepresentationFitScope:
        raise ValueError("fit scope uses an invalid schema")
    if type(standardizer) is not ModalityStandardizer:
        raise ValueError("standardizer uses an invalid schema")
    if (
        standardizer.fit_case_ids != scope.fit_case_ids
        or standardizer.fit_population_sha256 != scope.fit_population_sha256
    ):
        raise ValueError("representation fit population does not match frozen scope")


def fit_standardized_representation_scope(
    *,
    scope: RepresentationFitScope,
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> FittedStandardizedRepresentation:
    """Fit and transform only the exact candidate IDs in a frozen scope."""

    if type(scope) is not RepresentationFitScope:
        raise ValueError("fit scope uses an invalid schema")
    standardizer = fit_modality_standardizer(
        fit_case_ids=scope.fit_case_ids,
        case_modalities=case_modalities,
        feature_names_by_modality=feature_names_by_modality,
    )
    assert_representation_fit_population(scope, standardizer)
    standardized = transform_modality_blocks(
        artifact=standardizer,
        case_modalities=case_modalities,
        case_ids=scope.fit_case_ids,
    )
    if standardized.case_ids != scope.fit_case_ids:
        raise ValueError("standardized matrix row population drifted")
    return FittedStandardizedRepresentation(
        scope=scope,
        standardizer=standardizer,
        standardized=standardized,
    )


def _representation_numerical_health(
    candidate: RepresentationMatrixCandidate,
) -> RepresentationNumericalHealth:
    matrix = np.asarray(candidate.matrix, dtype=float)
    is_matrix = matrix.ndim == 2
    row_count = int(matrix.shape[0]) if is_matrix else 0
    output_dimension = int(matrix.shape[1]) if is_matrix else 0
    feature_name_count = len(candidate.feature_names)
    shape_consistent = (
        is_matrix
        and row_count == len(candidate.case_ids)
        and output_dimension == feature_name_count
        and row_count > 0
        and output_dimension > 0
    )
    finite = np.isfinite(matrix)
    non_finite_count = int(matrix.size - int(finite.sum()))
    all_finite = bool(finite.all())
    if is_matrix and matrix.size and all_finite:
        variances = matrix.var(axis=0, ddof=0)
        zero_variance_count = int(np.sum(variances == 0.0))
        numerical_rank = int(np.linalg.matrix_rank(matrix))
        minimum = float(matrix.min())
        maximum = float(matrix.max())
        maximum_absolute = float(np.abs(matrix).max())
    else:
        zero_variance_count = output_dimension
        numerical_rank = 0
        minimum = None
        maximum = None
        maximum_absolute = None
    passed = (
        shape_consistent
        and all_finite
        and zero_variance_count == 0
        and numerical_rank > 0
    )
    return RepresentationNumericalHealth(
        row_count=row_count,
        output_dimension=output_dimension,
        feature_name_count=feature_name_count,
        shape_consistent=shape_consistent,
        all_finite=all_finite,
        non_finite_value_count=non_finite_count,
        zero_variance_dimension_count=zero_variance_count,
        numerical_rank=numerical_rank,
        minimum_value=minimum,
        maximum_value=maximum,
        maximum_absolute_value=maximum_absolute,
        passed=passed,
    )


def build_representation_manifests(
    *,
    scope: RepresentationFitScope,
    feature_inventory: FeatureInventory,
    fitted: FittedStandardizedRepresentation,
    candidates: Mapping[str, RepresentationMatrixCandidate],
) -> dict[str, RepresentationManifest]:
    """Build deterministic manifests for the exact ten formal matrices."""

    if type(scope) is not RepresentationFitScope:
        raise ValueError("representation manifest scope uses an invalid schema")
    if type(feature_inventory) is not FeatureInventory:
        raise ValueError("representation feature inventory uses an invalid schema")
    if type(fitted) is not FittedStandardizedRepresentation:
        raise ValueError("fitted representation uses an invalid schema")
    if fitted.scope != scope:
        raise ValueError("fitted representation scope does not match manifest scope")
    if feature_inventory.dataset_id != scope.dataset_id:
        raise ValueError("feature inventory dataset does not match fit scope")
    assert_representation_fit_population(scope, fitted.standardizer)
    if fitted.standardized.case_ids != scope.fit_case_ids:
        raise ValueError("standardized fit population does not match frozen scope")
    if any(
        feature_inventory.eligible_features_by_modality[modality]
        != fitted.standardizer.feature_names_by_modality[modality]
        for modality in MODALITIES
    ):
        raise ValueError("feature inventory does not match fitted feature boundary")
    validate_formal_representation_registry(candidates)

    balanced = balance_modality_blocks(fitted.standardized)
    modality_records: dict[str, RepresentationModalityManifest] = {}
    for modality in MODALITIES:
        standardizer = fitted.standardizer
        balance = balanced.contribution_audit.records_by_modality[modality]
        modality_records[modality] = RepresentationModalityManifest(
            modality=modality,
            input_feature_names=standardizer.feature_names_by_modality[modality],
            retained_feature_names=(
                standardizer.retained_feature_names_by_modality[modality]
            ),
            dropped_zero_variance_features=(
                standardizer.dropped_zero_variance_by_modality[modality]
            ),
            input_dimension=len(
                standardizer.feature_names_by_modality[modality]
            ),
            retained_dimension=len(
                standardizer.retained_feature_names_by_modality[modality]
            ),
            observed_case_count=(
                standardizer.observed_case_count_by_modality[modality]
            ),
            standardization_means=standardizer.means_by_modality[modality],
            standardization_scales=standardizer.scales_by_modality[modality],
            block_multiplier=balance.block_multiplier,
            raw_median_pairwise_distance=balance.raw_median_pairwise_distance,
            scaled_median_pairwise_distance=(
                balance.scaled_median_pairwise_distance
            ),
            active=balance.active,
            inactive_reason=balance.inactive_reason,
        )
    frozen_modalities = MappingProxyType(modality_records)

    manifests: dict[str, RepresentationManifest] = {}
    for representation_id in FORMAL_REPRESENTATION_IDS:
        candidate = candidates[representation_id]
        if (
            candidate.case_ids != scope.fit_case_ids
            or candidate.fit_population_sha256 != scope.fit_population_sha256
        ):
            raise ValueError(
                f"representation fit population drifted: {representation_id}"
            )
        health = _representation_numerical_health(candidate)
        if not health.passed:
            raise ValueError(
                f"representation numerical health failed: {representation_id}"
            )
        parameters = json.loads(
            json.dumps(
                dict(candidate.parameters),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        manifests[representation_id] = RepresentationManifest(
            dataset_id=scope.dataset_id,
            stage=scope.stage,
            held_out_fault_type=scope.held_out_fault_type,
            representation_id=representation_id,
            family=candidate.family,
            formal_eligible=candidate.formal_eligible,
            fit_case_ids=scope.fit_case_ids,
            fit_population_sha256=scope.fit_population_sha256,
            fit_scope_sha256=scope.scope_sha256,
            split_manifest_sha256=scope.split_manifest_sha256,
            feature_inventory_sha256=feature_inventory.inventory_sha256,
            feature_source_artifact_sha256=(
                feature_inventory.source_artifact_sha256
            ),
            feature_policy_version=feature_inventory.policy_version,
            feature_exclusions=feature_inventory.exclusions,
            modalities=frozen_modalities,
            reduction_parameters=MappingProxyType(parameters),
            row_count=health.row_count,
            output_dimension=health.output_dimension,
            source_matrix_sha256=candidate.source_matrix_sha256,
            matrix_sha256=candidate.matrix_sha256,
            numerical_health=health,
        )
    return manifests


def inventory_feature_names(
    *,
    dataset_id: str,
    feature_names_by_modality: Mapping[str, Sequence[str]],
    source_artifact_sha256: str,
) -> FeatureInventory:
    """Classify current features without allowing exclusions into a matrix."""

    if dataset_id not in FORMAL_DATASETS:
        raise ValueError("feature inventory dataset is outside the formal protocol")
    if not isinstance(source_artifact_sha256, str) or not _SHA256.fullmatch(
        source_artifact_sha256
    ):
        raise ValueError("source_artifact_sha256 must be a lowercase SHA-256")
    raw = _validate_feature_layout(feature_names_by_modality)
    eligible: dict[str, tuple[str, ...]] = {}
    exclusions: list[FeatureExclusion] = []
    for modality in MODALITIES:
        kept: list[str] = []
        for feature_name in raw[modality]:
            reason = forbidden_feature_reason(feature_name)
            if reason is None:
                kept.append(feature_name)
            else:
                exclusions.append(
                    FeatureExclusion(
                        modality=modality,
                        feature_name=feature_name,
                        reason=reason,
                    )
                )
        eligible[modality] = tuple(kept)
    raw_count = sum(len(raw[modality]) for modality in MODALITIES)
    eligible_count = sum(len(eligible[modality]) for modality in MODALITIES)
    return FeatureInventory(
        dataset_id=dataset_id,
        modalities=MODALITIES,
        source_artifact_sha256=source_artifact_sha256,
        raw_features_by_modality=MappingProxyType(raw),
        eligible_features_by_modality=MappingProxyType(eligible),
        exclusions=tuple(exclusions),
        raw_feature_count=raw_count,
        eligible_feature_count=eligible_count,
        excluded_feature_count=raw_count - eligible_count,
    )


def validate_clustering_matrix_features(
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Hard-fail if a forbidden or malformed feature can enter clustering."""

    canonical = _validate_feature_layout(feature_names_by_modality)
    for modality in MODALITIES:
        for feature_name in canonical[modality]:
            reason = forbidden_feature_reason(feature_name)
            if reason is not None:
                raise ValueError(
                    f"forbidden clustering feature ({reason}): "
                    f"{modality}.{feature_name}"
                )
    return canonical


__all__ = [
    "FEATURE_INVENTORY_SCHEMA_VERSION",
    "FEATURE_POLICY_VERSION",
    "DIAGNOSTIC_REPRESENTATION_IDS",
    "FORMAL_DATASETS",
    "FORMAL_REPRESENTATION_IDS",
    "DIRECT_FUSION_SCHEMA_VERSION",
    "GLOBAL_REPRESENTATION_IDS",
    "LATE_AFFINITY_REPRESENTATION_ID",
    "MODALITY_BALANCE_SCHEMA_VERSION",
    "MODALITY_STANDARDIZER_SCHEMA_VERSION",
    "MODALITIES",
    "PER_MODALITY_REPRESENTATION_IDS",
    "REPRESENTATION_MANIFEST_SCHEMA_VERSION",
    "REPRESENTATION_MATRIX_SCHEMA_VERSION",
    "BalancedModalityBlocks",
    "DiagnosticRepresentationAdapter",
    "FeatureExclusion",
    "FeatureInventory",
    "FittedStandardizedRepresentation",
    "FusedRepresentationMatrix",
    "LateAffinityGeometry",
    "ModalityContributionAudit",
    "ModalityContributionRecord",
    "ModalityStandardizer",
    "RepresentationFitScope",
    "RepresentationManifest",
    "RepresentationMatrixCandidate",
    "RepresentationModalityManifest",
    "RepresentationNumericalHealth",
    "StandardizedModalityBlocks",
    "adapt_legacy_rcl_guided_candidate",
    "assert_representation_fit_population",
    "balance_modality_blocks",
    "build_diagnostic_representation_registry",
    "build_global_representation_candidates",
    "build_formal_representation_candidates",
    "build_late_affinity_candidate",
    "build_per_modality_representation_candidates",
    "build_representation_manifests",
    "build_umap_diagnostic_candidate",
    "coverage_normalized_late_distance",
    "derive_representation_fit_scope",
    "fit_modality_standardizer",
    "fit_standardized_representation_scope",
    "fuse_balanced_blocks",
    "forbidden_feature_reason",
    "inventory_feature_names",
    "transform_modality_blocks",
    "validate_clustering_matrix_features",
    "validate_formal_representation_registry",
]
