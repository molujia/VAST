"""Immutable schemas and protocol matrix for combined active learning 2.0."""

from __future__ import annotations

import hashlib
import json
import math
import re
import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from statistics import fmean
from typing import Any, ClassVar, TypeVar, get_args, get_origin, get_type_hints


ACTIVE_LEARNING_SEEDS = (41, 42, 43)
DOWNSTREAM_RCL_SEED = 42
ANNOTATION_BUDGET = 30
ORDINARY_MODE = "ordinary_query_only_t1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
NO_REPRESENTATION_DEPENDENCY_SHA256 = hashlib.sha256(
    b"rcl-active-learning-combined-2.0:no-representation-dependency"
).hexdigest()


class DatasetId(str, Enum):
    RCABENCH = "rcabench"
    AIOPS22_PRE = "aiops2022_pre"


class ApprovalState(str, Enum):
    IMPLEMENTATION = "implementation"
    FORMAL_FIRST_STAGE = "formal_first_stage"
    AWAITING_OWNER_CLUSTER_PARAMETER_DECISION = (
        "awaiting_owner_cluster_parameter_decision"
    )
    AWAITING_OWNER_BEST_ANALYSIS_APPROVAL = (
        "awaiting_owner_best_analysis_approval"
    )
    OWNER_APPROVED_BEST_ANALYSIS = "owner_approved_best_analysis"
    COMPLETE = "complete"


T = TypeVar("T", bound="TypedRecord")


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, TypedRecord):
        return value.to_dict()
    if is_dataclass(value):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    return value


def _coerce(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, getattr(types, "UnionType", object)):
        if value is None and type(None) in args:
            return None
        failures: list[Exception] = []
        for candidate in args:
            if candidate is not type(None):
                try:
                    return _coerce(candidate, value)
                except (TypeError, ValueError) as exc:
                    failures.append(exc)
        if failures:
            raise ValueError(
                f"value does not match any allowed union member: {value!r}"
            ) from failures[-1]
        return value
    if origin is tuple:
        item_type = args[0] if args else Any
        return tuple(_coerce(item_type, item) for item in value)
    if origin is list:
        item_type = args[0] if args else Any
        return [_coerce(item_type, item) for item in value]
    if origin in (dict, Mapping):
        key_type, item_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce(key_type, key): _coerce(item_type, item)
            for key, item in dict(value).items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, TypedRecord):
        return annotation.from_dict(value)
    return value


class TypedRecord:
    """Small standard-library JSON round-trip layer for frozen dataclasses."""

    schema_name: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _primitive(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls: type[T], payload: Mapping[str, Any]) -> T:
        hints = get_type_hints(cls)
        unknown = set(payload).difference(hints)
        if unknown:
            raise ValueError(
                f"{cls.__name__} contains unknown fields: {', '.join(sorted(unknown))}"
            )
        values = {
            name: _coerce(hints[name], value)
            for name, value in dict(payload).items()
            if name in hints
        }
        return cls(**values)


def _require_text(value: str, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256.fullmatch(str(value)):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _require_seed(seed: int) -> None:
    if int(seed) not in ACTIVE_LEARNING_SEEDS:
        raise ValueError(
            f"active-learning seed must be one of {ACTIVE_LEARNING_SEEDS}"
        )


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        _primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HashBundle(TypedRecord):
    code_sha256: str
    data_sha256: str
    representation_sha256: str
    query_plan_sha256: str
    rcl_control_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_sha256(getattr(self, field.name), field.name)


@dataclass(frozen=True)
class RepresentationCandidate(TypedRecord):
    representation_id: str
    family: str
    formal_eligible: bool
    fit_population_sha256: str
    matrix_sha256: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.representation_id, "representation_id")
        _require_text(self.family, "family")
        _require_sha256(self.fit_population_sha256, "fit_population_sha256")
        _require_sha256(self.matrix_sha256, "matrix_sha256")


@dataclass(frozen=True)
class ClusteringGrid(TypedRecord):
    clusterer_id: str
    dataset_id: DatasetId
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.clusterer_id, "clusterer_id")
        if self.clusterer_id not in {"kmeans", "dbscan", "hdbscan", "mutual_knn"}:
            raise ValueError(f"unsupported clusterer_id: {self.clusterer_id}")
        if not self.parameters:
            raise ValueError("clustering grid parameters must be non-empty")


@dataclass(frozen=True)
class StructureGateConfig(TypedRecord):
    target_cluster_count: int
    min_cluster_count: int
    max_cluster_count: int
    min_effective_support: int = 5
    min_effective_coverage: float = 0.70
    max_largest_effective_cluster_share: float = 0.30
    quota_policy: str = "equal_weight_round_robin"
    complete_first_coverage_round: bool = True
    residual_start_after_first_round: bool = True
    max_residual_queries: int = 2

    def __post_init__(self) -> None:
        if not (
            2 <= self.min_cluster_count <= self.target_cluster_count <= self.max_cluster_count
        ):
            raise ValueError("invalid cluster-count band")
        if self.max_cluster_count - self.min_cluster_count > 4:
            raise ValueError("cluster-count band may float by at most two around target")
        if self.min_effective_support < 1:
            raise ValueError("minimum effective support must be positive")
        if not 0.0 <= self.min_effective_coverage <= 1.0:
            raise ValueError("effective coverage threshold must be in [0, 1]")
        if not 0.0 < self.max_largest_effective_cluster_share < 1.0:
            raise ValueError("largest-cluster threshold must be in (0, 1)")
        if self.quota_policy != "equal_weight_round_robin":
            raise ValueError("combined 2.0 requires equal-weight round-robin quotas")
        if not self.complete_first_coverage_round:
            raise ValueError("combined 2.0 requires a complete first coverage round")
        if not self.residual_start_after_first_round:
            raise ValueError("residual queries may begin only after the first round")
        if self.max_residual_queries != 2:
            raise ValueError("combined 2.0 residual-query allowance is exactly two")


@dataclass(frozen=True)
class StructureGateResult(TypedRecord):
    dataset_id: DatasetId
    clusterer_id: str
    active_learning_seed: int
    effective_cluster_count: int
    effective_coverage: float
    largest_effective_cluster_share: float
    passed: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_seed(self.active_learning_seed)
        if self.effective_cluster_count < 0:
            raise ValueError("effective_cluster_count must be non-negative")
        if not 0.0 <= self.effective_coverage <= 1.0:
            raise ValueError("effective_coverage must be in [0, 1]")
        if not 0.0 <= self.largest_effective_cluster_share <= 1.0:
            raise ValueError("largest_effective_cluster_share must be in [0, 1]")
        if self.passed and self.rejection_reasons:
            raise ValueError("passing structure result cannot contain rejection reasons")
        if not self.passed and not self.rejection_reasons:
            raise ValueError("failed structure result must contain rejection reasons")


@dataclass(frozen=True)
class SelectorPlan(TypedRecord):
    plan_id: str
    dataset_id: DatasetId
    method_id: str
    selector_id: str
    active_learning_seed: int
    budget: int
    selected_case_ids: tuple[str, ...]
    geometry_sha256: str
    quota_sha256: str
    plan_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.plan_id, "plan_id")
        _require_seed(self.active_learning_seed)
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError(f"selector budget must be {ANNOTATION_BUDGET}")
        if len(self.selected_case_ids) != self.budget:
            raise ValueError("selected case count must equal budget")
        if len(set(self.selected_case_ids)) != len(self.selected_case_ids):
            raise ValueError("selected case IDs must be unique")
        for name in ("geometry_sha256", "quota_sha256", "plan_sha256"):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class OrdinaryUnit(TypedRecord):
    unit_id: str
    dataset_id: DatasetId
    arm_id: str
    method_id: str
    selector_id: str
    active_learning_seed: int
    budget: int
    rcl_seed: int
    evaluation_mode: str
    hashes: HashBundle
    record_role: str = "fresh_unit"
    snapshot_action: str = "none"
    execution_allowed: bool = True
    representation_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.unit_id, "unit_id")
        _require_seed(self.active_learning_seed)
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError(f"ordinary budget must be {ANNOTATION_BUDGET}")
        if self.rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError(f"downstream RCL seed must remain {DOWNSTREAM_RCL_SEED}")
        if self.evaluation_mode != ORDINARY_MODE:
            raise ValueError("ordinary evaluation is T1-only")
        if (
            self.record_role != "fresh_unit"
            or self.snapshot_action != "none"
            or not self.execution_allowed
        ):
            raise ValueError("ordinary units must be fresh executable non-snapshot units")


@dataclass(frozen=True)
class StrictLofoUnit(TypedRecord):
    unit_id: str
    dataset_id: DatasetId
    held_out_fault_type: str
    method_id: str
    selector_id: str
    active_learning_seed: int
    budget: int
    rcl_seed: int
    fold_sha256: str
    hashes: HashBundle
    record_role: str = "fresh_unit"
    snapshot_action: str = "none"
    execution_allowed: bool = True

    def __post_init__(self) -> None:
        _require_text(self.held_out_fault_type, "held_out_fault_type")
        _require_seed(self.active_learning_seed)
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError(f"LOFO budget must be {ANNOTATION_BUDGET}")
        if self.rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError(f"downstream RCL seed must remain {DOWNSTREAM_RCL_SEED}")
        _require_sha256(self.fold_sha256, "fold_sha256")
        if (
            self.record_role != "fresh_unit"
            or self.snapshot_action != "none"
            or not self.execution_allowed
        ):
            raise ValueError("LOFO units must be fresh executable non-snapshot units")


@dataclass(frozen=True)
class SnapshotReference(TypedRecord):
    """Immutable historical DBSCAN score evidence that can never be executed."""

    reference_id: str
    dataset_id: DatasetId
    method_id: str
    source_active_learning_seeds: tuple[int, ...]
    mean_top135: float
    reported_mean_top135: float
    source_evidence_sha256: str
    record_role: str = "reuse_only_external_reference"
    snapshot_action: str = "reuse_reference"
    execution_allowed: bool = False
    rerun_forbidden: bool = True

    def __post_init__(self) -> None:
        _require_text(self.reference_id, "reference_id")
        if self.method_id != "dbscan":
            raise ValueError("snapshot reference method must be dbscan")
        if self.source_active_learning_seeds != (42, 43, 44):
            raise ValueError("snapshot reference seeds must remain 42/43/44")
        if not 0.0 <= float(self.mean_top135) <= 1.0:
            raise ValueError("snapshot mean_top135 must be in [0, 1]")
        if round(float(self.mean_top135), 6) != float(self.reported_mean_top135):
            raise ValueError("reported snapshot mean must be the six-decimal source mean")
        _require_sha256(self.source_evidence_sha256, "source_evidence_sha256")
        if (
            self.record_role != "reuse_only_external_reference"
            or self.snapshot_action != "reuse_reference"
            or self.execution_allowed
            or not self.rerun_forbidden
        ):
            raise ValueError(
                "snapshot references are reuse-only external evidence and cannot execute"
            )


@dataclass(frozen=True)
class ExecutionManifest(TypedRecord):
    """Typed boundary between fresh execution units and historical references."""

    schema_version: str
    manifest_id: str
    stage: str
    units: tuple[OrdinaryUnit | StrictLofoUnit, ...]
    snapshot_references: tuple[SnapshotReference, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "rcl-active-learning-combined-2.0-execution-v1":
            raise ValueError("unsupported execution manifest schema version")
        _require_text(self.manifest_id, "manifest_id")
        if self.stage not in {"ordinary", "strict_lofo", "smoke"}:
            raise ValueError("unsupported execution manifest stage")
        if any(not isinstance(unit, (OrdinaryUnit, StrictLofoUnit)) for unit in self.units):
            raise ValueError("execution manifests require typed fresh execution units")
        if any(
            not isinstance(reference, SnapshotReference)
            for reference in self.snapshot_references
        ):
            raise ValueError("snapshot references must use the typed reuse-only schema")
        unit_ids = [unit.unit_id for unit in self.units]
        reference_ids = [reference.reference_id for reference in self.snapshot_references]
        all_ids = [*unit_ids, *reference_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("execution manifest record IDs must be unique")
        if self.stage == "ordinary" and any(
            not isinstance(unit, OrdinaryUnit) for unit in self.units
        ):
            raise ValueError("ordinary manifest may contain only ordinary units")
        if self.stage == "strict_lofo" and any(
            not isinstance(unit, StrictLofoUnit) for unit in self.units
        ):
            raise ValueError("strict-LOFO manifest may contain only strict-LOFO units")


@dataclass(frozen=True)
class DiagnosticRecord(TypedRecord):
    unit_id: str
    metric_id: str
    value: float
    direction: str
    label_access: str

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value)):
            raise ValueError("diagnostic value must be finite")
        if self.direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("invalid diagnostic direction")
        if self.label_access not in {"label_free", "post_selection_ground_truth"}:
            raise ValueError("invalid diagnostic label-access phase")


@dataclass(frozen=True)
class AggregateRecord(TypedRecord):
    aggregate_id: str
    dataset_id: DatasetId
    method_id: str
    selector_id: str
    mean_top135: float
    active_learning_seeds: tuple[int, ...]
    qualified: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.active_learning_seeds != ACTIVE_LEARNING_SEEDS:
            raise ValueError("aggregate must use active-learning seeds 41/42/43")
        if not math.isfinite(float(self.mean_top135)):
            raise ValueError("mean_top135 must be finite")
        if self.qualified and self.rejection_reasons:
            raise ValueError("qualified aggregate cannot have rejection reasons")


@dataclass(frozen=True)
class RankingRecord(TypedRecord):
    board: str
    method_id: str
    qualified: bool
    rank: int | None
    tie_break_values: tuple[float, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.board not in {"clustering_only", "all_methods"}:
            raise ValueError("invalid ranking board")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be positive")
        if len(self.tie_break_values) != 4:
            raise ValueError("ranking requires exactly four tie-break values")
        if self.qualified and self.rejection_reasons:
            raise ValueError("qualified ranking record cannot have rejection reasons")


@dataclass(frozen=True)
class ApprovalRecord(TypedRecord):
    state: ApprovalState
    owner_approval_sha256: str | None

    def __post_init__(self) -> None:
        if self.state == ApprovalState.OWNER_APPROVED_BEST_ANALYSIS:
            if self.owner_approval_sha256 is None:
                raise ValueError("owner-approved state requires approval evidence")
            _require_sha256(self.owner_approval_sha256, "owner_approval_sha256")
        elif self.owner_approval_sha256 is not None:
            _require_sha256(self.owner_approval_sha256, "owner_approval_sha256")


@dataclass(frozen=True)
class ArmSpec(TypedRecord):
    arm_id: str
    acquisition_family: str
    method_id: str
    selector_id: str
    clusterer_id: str | None

    def __post_init__(self) -> None:
        _require_text(self.arm_id, "arm_id")
        expected = f"{self.method_id}.{self.selector_id}"
        if self.arm_id != expected:
            raise ValueError(f"arm_id must equal {expected}")
        if self.acquisition_family == "clustered":
            if self.clusterer_id != self.method_id:
                raise ValueError("clustered arm must identify its own clusterer")
            if self.method_id not in {"kmeans", "dbscan", "hdbscan", "mutual_knn"}:
                raise ValueError("unsupported clustered method")
            if self.selector_id not in {"center", "boundary", "within_cluster_random"}:
                raise ValueError("unsupported clustered selector")
        elif self.acquisition_family == "independent":
            if self.clusterer_id is not None:
                raise ValueError("independent arm must bypass clustering")
            if self.method_id not in {
                "knn_fault_mode",
                "falcon_hybrid",
                "facility_location",
            }:
                raise ValueError("unsupported independent method")
        elif self.acquisition_family == "global_random":
            if self.clusterer_id is not None:
                raise ValueError("global random must bypass clustering")
            if self.method_id != "global_random" or self.selector_id != "global_random":
                raise ValueError("global-random identifiers are immutable")
        else:
            raise ValueError("unsupported acquisition family")


@dataclass(frozen=True)
class ProtocolManifest(TypedRecord):
    schema_version: str
    datasets: tuple[DatasetId, ...]
    active_learning_seeds: tuple[int, ...]
    downstream_rcl_seed: int
    budget: int
    ordinary_mode: str
    lofo_fault_types: Mapping[str, tuple[str, ...]]
    structure_gates: Mapping[str, StructureGateConfig]
    arms: tuple[ArmSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "rcl-active-learning-combined-2.0-protocol-v1":
            raise ValueError("unsupported combined protocol schema version")
        if self.datasets != (DatasetId.RCABENCH, DatasetId.AIOPS22_PRE):
            raise ValueError("combined protocol datasets are immutable")
        if self.active_learning_seeds != ACTIVE_LEARNING_SEEDS:
            raise ValueError("combined protocol active-learning seeds are immutable")
        if self.downstream_rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError("combined protocol RCL seed is immutable")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("combined protocol budget is immutable")
        if self.ordinary_mode != ORDINARY_MODE:
            raise ValueError("combined protocol is ordinary T1-only")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("duplicate ordinary arm")
        if len(self.arms) != 16:
            raise ValueError("combined protocol requires exactly 16 ordinary arms")
        for dataset in self.datasets:
            if dataset.value not in self.lofo_fault_types:
                raise ValueError(f"missing LOFO fault types for {dataset.value}")
            if dataset.value not in self.structure_gates:
                raise ValueError(f"missing structure gates for {dataset.value}")


@dataclass(frozen=True)
class AmendedStructurePolicy(TypedRecord):
    dataset_id: DatasetId
    target_cluster_count: int
    min_cluster_count: int
    max_cluster_count: int
    min_effective_support: int = 5
    min_mean_effective_coverage: float = 0.50
    largest_cluster_required_seed_count: int = 2
    parameter_extension_requires_owner: bool = True

    def __post_init__(self) -> None:
        if not (
            2 <= self.min_cluster_count
            <= self.target_cluster_count
            <= self.max_cluster_count
        ):
            raise ValueError("invalid amended cluster-count policy")
        if self.min_effective_support != 5:
            raise ValueError("amended effective-cluster support must remain five")
        if self.min_mean_effective_coverage != 0.50:
            raise ValueError("amended mean effective-coverage threshold must be 0.50")
        if self.largest_cluster_required_seed_count != 2:
            raise ValueError("amended largest-cluster rule requires two of three seeds")
        if not self.parameter_extension_requires_owner:
            raise ValueError("parameter-grid extension requires owner approval")


def amended_structure_policy(dataset_id: DatasetId | str) -> AmendedStructurePolicy:
    dataset = DatasetId(dataset_id)
    if dataset == DatasetId.RCABENCH:
        return AmendedStructurePolicy(dataset, 15, 11, 20)
    return AmendedStructurePolicy(dataset, 10, 6, 15)


@dataclass(frozen=True)
class AmendedArmSpec(TypedRecord):
    arm_id: str
    acquisition_family: str
    method_id: str
    selector_id: str
    representation_id: str | None
    clusterer_id: str | None

    def __post_init__(self) -> None:
        _require_text(self.arm_id, "arm_id")
        from rcl_study.combined_active_learning_representation import (
            FORMAL_REPRESENTATION_IDS,
        )

        if self.acquisition_family == "global_random":
            if (
                self.arm_id != "global_random.global_random"
                or self.method_id != "global_random"
                or self.selector_id != "global_random"
                or self.representation_id is not None
                or self.clusterer_id is not None
            ):
                raise ValueError("amended global random must bypass representation and clustering")
            return
        if self.representation_id not in FORMAL_REPRESENTATION_IDS:
            raise ValueError("amended arm requires an approved formal representation")
        expected = f"{self.representation_id}.{self.method_id}.{self.selector_id}"
        if self.arm_id != expected:
            raise ValueError(f"amended arm_id must equal {expected}")
        if self.acquisition_family == "clustered":
            if self.clusterer_id != self.method_id:
                raise ValueError("clustered amended arm must identify its clusterer")
            if self.method_id not in {"kmeans", "dbscan", "hdbscan", "mutual_knn"}:
                raise ValueError("unsupported amended clustered method")
            if self.selector_id not in {"center", "boundary", "within_cluster_random"}:
                raise ValueError("unsupported amended clustered selector")
        elif self.acquisition_family == "independent":
            if self.clusterer_id is not None:
                raise ValueError("independent amended arm must bypass clustering")
            if self.method_id not in {
                "knn_fault_mode",
                "falcon_hybrid",
                "facility_location",
            } or self.selector_id != self.method_id:
                raise ValueError("unsupported amended independent method")
        else:
            raise ValueError("unsupported amended acquisition family")


@dataclass(frozen=True)
class AmendedProtocolManifest(TypedRecord):
    schema_version: str
    datasets: tuple[DatasetId, ...]
    active_learning_seeds: tuple[int, ...]
    downstream_rcl_seed: int
    budget: int
    ordinary_mode: str
    candidate_population: str
    representation_ids: tuple[str, ...]
    structure_policies: Mapping[str, AmendedStructurePolicy]
    arms: tuple[AmendedArmSpec, ...]

    def __post_init__(self) -> None:
        from rcl_study.combined_active_learning_representation import (
            FORMAL_REPRESENTATION_IDS,
        )

        if self.schema_version != "rcl-active-learning-combined-2.0-protocol-v2":
            raise ValueError("unsupported amended protocol schema version")
        if self.datasets != (DatasetId.RCABENCH, DatasetId.AIOPS22_PRE):
            raise ValueError("amended protocol datasets are immutable")
        if self.active_learning_seeds != ACTIVE_LEARNING_SEEDS:
            raise ValueError("amended active-learning seeds are immutable")
        if self.downstream_rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError("amended downstream RCL seed is immutable")
        if self.budget != ANNOTATION_BUDGET or self.ordinary_mode != ORDINARY_MODE:
            raise ValueError("amended protocol is budget-30 ordinary T1 only")
        if self.candidate_population != "full_observable_outer_train_not_sampled":
            raise ValueError("amended protocol requires complete observable outer_train")
        if self.representation_ids != FORMAL_REPRESENTATION_IDS:
            raise ValueError("amended formal representations drifted")
        if set(self.structure_policies) != {dataset.value for dataset in self.datasets}:
            raise ValueError("amended structure policies are incomplete")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(self.arms) != 151 or len(arm_ids) != len(set(arm_ids)):
            raise ValueError("amended protocol requires exactly 151 unique arms")


def amended_protocol_manifest() -> AmendedProtocolManifest:
    from rcl_study.combined_active_learning_representation import (
        FORMAL_REPRESENTATION_IDS,
    )

    clustered = tuple(
        AmendedArmSpec(
            arm_id=f"{representation}.{clusterer}.{selector}",
            acquisition_family="clustered",
            method_id=clusterer,
            selector_id=selector,
            representation_id=representation,
            clusterer_id=clusterer,
        )
        for representation in FORMAL_REPRESENTATION_IDS
        for clusterer in ("kmeans", "dbscan", "hdbscan", "mutual_knn")
        for selector in ("center", "boundary", "within_cluster_random")
    )
    independent = tuple(
        AmendedArmSpec(
            arm_id=f"{representation}.{method}.{method}",
            acquisition_family="independent",
            method_id=method,
            selector_id=method,
            representation_id=representation,
            clusterer_id=None,
        )
        for representation in FORMAL_REPRESENTATION_IDS
        for method in ("knn_fault_mode", "falcon_hybrid", "facility_location")
    )
    random = AmendedArmSpec(
        arm_id="global_random.global_random",
        acquisition_family="global_random",
        method_id="global_random",
        selector_id="global_random",
        representation_id=None,
        clusterer_id=None,
    )
    return AmendedProtocolManifest(
        schema_version="rcl-active-learning-combined-2.0-protocol-v2",
        datasets=(DatasetId.RCABENCH, DatasetId.AIOPS22_PRE),
        active_learning_seeds=ACTIVE_LEARNING_SEEDS,
        downstream_rcl_seed=DOWNSTREAM_RCL_SEED,
        budget=ANNOTATION_BUDGET,
        ordinary_mode=ORDINARY_MODE,
        candidate_population="full_observable_outer_train_not_sampled",
        representation_ids=FORMAL_REPRESENTATION_IDS,
        structure_policies={
            dataset.value: amended_structure_policy(dataset)
            for dataset in (DatasetId.RCABENCH, DatasetId.AIOPS22_PRE)
        },
        arms=(*clustered, *independent, random),
    )


def amended_ordinary_unit_ids(protocol: AmendedProtocolManifest) -> tuple[str, ...]:
    if not isinstance(protocol, AmendedProtocolManifest):
        raise ValueError("amended ordinary units require protocol v2")
    values = []
    for dataset in protocol.datasets:
        for seed in protocol.active_learning_seeds:
            for arm in protocol.arms:
                if arm.acquisition_family == "global_random":
                    identity = "global_random.global_random"
                else:
                    identity = (
                        f"{arm.representation_id}.{arm.method_id}.{arm.selector_id}"
                    )
                values.append(f"ordinary.{dataset.value}.{identity}.seed{seed}")
    if len(values) != 906 or len(values) != len(set(values)):
        raise ValueError("amended ordinary matrix must contain exactly 906 units")
    return tuple(values)


@dataclass(frozen=True)
class GroundTruthClusterCap(TypedRecord):
    candidate_count: int
    largest_fault_type_count: int
    largest_root_cause_service_count: int
    reference_largest_count: int
    max_allowed_largest_effective_cluster_size: int
    max_allowed_largest_effective_cluster_share: float

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("ground-truth cap requires candidates")
        expected_reference = max(
            self.largest_fault_type_count,
            self.largest_root_cause_service_count,
        )
        if self.reference_largest_count != expected_reference:
            raise ValueError("ground-truth cap reference count drifted")
        expected_size = math.floor(1.15 * expected_reference)
        expected_share = 1.15 * expected_reference / self.candidate_count
        if self.max_allowed_largest_effective_cluster_size != expected_size:
            raise ValueError("ground-truth cap integer size drifted")
        if not math.isclose(
            self.max_allowed_largest_effective_cluster_share,
            expected_share,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("ground-truth cap share drifted")


def freeze_ground_truth_cluster_cap(
    *,
    candidate_case_ids: tuple[str, ...],
    fault_type_by_case: Mapping[str, str],
    root_cause_services_by_case: Mapping[str, tuple[str, ...]],
) -> GroundTruthClusterCap:
    case_ids = tuple(candidate_case_ids)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("ground-truth cap requires unique candidate case IDs")
    if set(fault_type_by_case) != set(case_ids):
        raise ValueError("fault-type cap inventory must cover the candidate pool")
    if set(root_cause_services_by_case) != set(case_ids):
        raise ValueError("root-cause cap inventory must cover the candidate pool")
    fault_counts = Counter(str(fault_type_by_case[case_id]) for case_id in case_ids)
    service_counts: Counter[str] = Counter()
    for case_id in case_ids:
        service_counts.update(
            {
                str(service)
                for service in root_cause_services_by_case[case_id]
                if str(service).strip()
            }
        )
    largest_fault = max(fault_counts.values(), default=0)
    largest_service = max(service_counts.values(), default=0)
    reference = max(largest_fault, largest_service)
    return GroundTruthClusterCap(
        candidate_count=len(case_ids),
        largest_fault_type_count=largest_fault,
        largest_root_cause_service_count=largest_service,
        reference_largest_count=reference,
        max_allowed_largest_effective_cluster_size=math.floor(1.15 * reference),
        max_allowed_largest_effective_cluster_share=(
            1.15 * reference / len(case_ids)
        ),
    )


@dataclass(frozen=True)
class SeedStructureObservation(TypedRecord):
    active_learning_seed: int
    effective_cluster_count: int
    effective_coverage: float
    largest_effective_cluster_size: int
    candidate_count: int
    residual_or_noise_count: int

    def __post_init__(self) -> None:
        _require_seed(self.active_learning_seed)
        if self.effective_cluster_count < 0 or self.candidate_count < 1:
            raise ValueError("invalid structure observation size")
        if not 0.0 <= self.effective_coverage <= 1.0:
            raise ValueError("effective coverage must be in [0, 1]")
        effective_count = round(self.effective_coverage * self.candidate_count)
        if not 0 <= self.largest_effective_cluster_size <= effective_count:
            raise ValueError("largest effective cluster exceeds effective coverage")
        if self.residual_or_noise_count != self.candidate_count - effective_count:
            raise ValueError("residual/noise count disagrees with effective coverage")


@dataclass(frozen=True)
class StructureTripletAssessment(TypedRecord):
    dataset_id: DatasetId
    policy: AmendedStructurePolicy
    observations: tuple[SeedStructureObservation, ...]
    ground_truth_cap: GroundTruthClusterCap
    status: str
    counts_legal: bool
    illegal_count_by_seed: Mapping[int, int]
    mean_effective_coverage: float
    mean_coverage_passed: bool
    largest_cluster_passing_seed_count: int
    largest_cluster_passed: bool
    extension_search_allowed: bool
    terminal_method_rejection: bool


def assess_structure_triplet(
    *,
    dataset_id: DatasetId | str,
    observations: tuple[SeedStructureObservation, ...],
    ground_truth_cap: GroundTruthClusterCap,
) -> StructureTripletAssessment:
    dataset = DatasetId(dataset_id)
    policy = amended_structure_policy(dataset)
    if tuple(item.active_learning_seed for item in observations) != ACTIVE_LEARNING_SEEDS:
        raise ValueError("structure assessment requires ordered seeds 41/42/43")
    if any(item.candidate_count != ground_truth_cap.candidate_count for item in observations):
        raise ValueError("structure observation candidate count drifted")
    illegal = {
        item.active_learning_seed: item.effective_cluster_count
        for item in observations
        if not (
            policy.min_cluster_count
            <= item.effective_cluster_count
            <= policy.max_cluster_count
        )
    }
    mean_coverage = fmean(item.effective_coverage for item in observations)
    mean_coverage_passed = mean_coverage > policy.min_mean_effective_coverage
    largest_passing = sum(
        item.largest_effective_cluster_size
        <= ground_truth_cap.max_allowed_largest_effective_cluster_size
        for item in observations
    )
    largest_passed = largest_passing >= policy.largest_cluster_required_seed_count
    if illegal:
        status = "awaiting_owner_cluster_parameter_decision"
    elif mean_coverage_passed and largest_passed:
        status = "usable"
    else:
        status = "rejected_structure"
    return StructureTripletAssessment(
        dataset_id=dataset,
        policy=policy,
        observations=observations,
        ground_truth_cap=ground_truth_cap,
        status=status,
        counts_legal=not illegal,
        illegal_count_by_seed=illegal,
        mean_effective_coverage=mean_coverage,
        mean_coverage_passed=mean_coverage_passed,
        largest_cluster_passing_seed_count=largest_passing,
        largest_cluster_passed=largest_passed,
        extension_search_allowed=False,
        terminal_method_rejection=False,
    )


@dataclass(frozen=True)
class UnitProvenanceInputs(TypedRecord):
    code_sha256: str
    rcl_control_sha256: str
    data_sha256_by_dataset: Mapping[str, str]
    representation_sha256_by_dataset: Mapping[str, str]
    query_plan_sha256_by_unit: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_sha256(self.code_sha256, "code_sha256")
        _require_sha256(self.rcl_control_sha256, "rcl_control_sha256")
        for dataset, digest in self.data_sha256_by_dataset.items():
            _require_sha256(digest, f"data_sha256_by_dataset.{dataset}")
        for dataset, digest in self.representation_sha256_by_dataset.items():
            _require_sha256(digest, f"representation_sha256_by_dataset.{dataset}")
        for unit_id, digest in self.query_plan_sha256_by_unit.items():
            _require_sha256(digest, f"query_plan_sha256_by_unit.{unit_id}")


@dataclass(frozen=True)
class AmendedUnitProvenanceInputs(TypedRecord):
    code_sha256: str
    rcl_control_sha256: str
    data_sha256_by_dataset: Mapping[str, str]
    representation_sha256_by_dataset_and_id: Mapping[str, Mapping[str, str]]
    query_plan_sha256_by_unit: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_sha256(self.code_sha256, "code_sha256")
        _require_sha256(self.rcl_control_sha256, "rcl_control_sha256")
        expected_datasets = {dataset.value for dataset in DatasetId}
        if set(self.data_sha256_by_dataset) != expected_datasets:
            raise ValueError("amended data provenance must cover both datasets")
        if set(self.representation_sha256_by_dataset_and_id) != expected_datasets:
            raise ValueError("amended representation provenance must cover both datasets")
        from rcl_study.combined_active_learning_representation import (
            FORMAL_REPRESENTATION_IDS,
        )

        for dataset, digest in self.data_sha256_by_dataset.items():
            _require_sha256(digest, f"data_sha256_by_dataset.{dataset}")
        for dataset, values in self.representation_sha256_by_dataset_and_id.items():
            if set(values) != set(FORMAL_REPRESENTATION_IDS):
                raise ValueError(
                    f"amended representation provenance is incomplete: {dataset}"
                )
            for representation_id, digest in values.items():
                _require_sha256(
                    digest,
                    f"representation_sha256_by_dataset_and_id.{dataset}.{representation_id}",
                )
        for unit_id, digest in self.query_plan_sha256_by_unit.items():
            _require_sha256(digest, f"query_plan_sha256_by_unit.{unit_id}")


def build_amended_ordinary_units(
    protocol: AmendedProtocolManifest,
    provenance: AmendedUnitProvenanceInputs,
) -> tuple[OrdinaryUnit, ...]:
    expected_ids = amended_ordinary_unit_ids(protocol)
    query_ids = set(provenance.query_plan_sha256_by_unit)
    if query_ids != set(expected_ids):
        missing = sorted(set(expected_ids) - query_ids)
        extra = sorted(query_ids - set(expected_ids))
        raise ValueError(
            "amended query-plan provenance mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    protocol_sha256 = semantic_sha256(protocol)
    units: list[OrdinaryUnit] = []
    for dataset in protocol.datasets:
        for seed in protocol.active_learning_seeds:
            for arm in protocol.arms:
                if arm.acquisition_family == "global_random":
                    identity = "global_random.global_random"
                    representation_sha256 = NO_REPRESENTATION_DEPENDENCY_SHA256
                else:
                    identity = (
                        f"{arm.representation_id}.{arm.method_id}.{arm.selector_id}"
                    )
                    representation_sha256 = (
                        provenance.representation_sha256_by_dataset_and_id[
                            dataset.value
                        ][str(arm.representation_id)]
                    )
                unit_id = f"ordinary.{dataset.value}.{identity}.seed{seed}"
                units.append(
                    OrdinaryUnit(
                        unit_id=unit_id,
                        dataset_id=dataset,
                        arm_id=arm.arm_id,
                        method_id=arm.method_id,
                        selector_id=arm.selector_id,
                        active_learning_seed=seed,
                        budget=protocol.budget,
                        rcl_seed=protocol.downstream_rcl_seed,
                        evaluation_mode=protocol.ordinary_mode,
                        hashes=HashBundle(
                            code_sha256=provenance.code_sha256,
                            data_sha256=provenance.data_sha256_by_dataset[
                                dataset.value
                            ],
                            representation_sha256=representation_sha256,
                            query_plan_sha256=(
                                provenance.query_plan_sha256_by_unit[unit_id]
                            ),
                            rcl_control_sha256=provenance.rcl_control_sha256,
                            protocol_sha256=protocol_sha256,
                        ),
                        representation_id=arm.representation_id,
                    )
                )
    if len(units) != 906 or len({unit.unit_id for unit in units}) != 906:
        raise ValueError("amended ordinary unit expansion drifted from 906")
    return tuple(units)


@dataclass(frozen=True)
class AmendedExecutionManifest(TypedRecord):
    schema_version: str
    manifest_id: str
    stage: str
    units: tuple[OrdinaryUnit, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "rcl-active-learning-combined-2.0-execution-v2":
            raise ValueError("unsupported amended execution manifest schema")
        _require_text(self.manifest_id, "manifest_id")
        if self.stage not in {"ordinary", "smoke"}:
            raise ValueError("amended execution manifest forbids LOFO and T2 stages")
        if any(not isinstance(unit, OrdinaryUnit) for unit in self.units):
            raise ValueError("amended execution manifest accepts ordinary units only")
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("amended execution unit IDs must be unique")
        if self.stage == "ordinary" and len(self.units) != 906:
            raise ValueError("amended ordinary execution manifest requires 906 units")


def default_protocol_manifest() -> ProtocolManifest:
    cluster_arms = tuple(
        ArmSpec(
            arm_id=f"{clusterer}.{selector}",
            acquisition_family="clustered",
            method_id=clusterer,
            selector_id=selector,
            clusterer_id=clusterer,
        )
        for clusterer in ("kmeans", "dbscan", "hdbscan", "mutual_knn")
        for selector in ("center", "boundary", "within_cluster_random")
    )
    independent_arms = tuple(
        ArmSpec(
            arm_id=f"{method}.{method}",
            acquisition_family="independent",
            method_id=method,
            selector_id=method,
            clusterer_id=None,
        )
        for method in ("knn_fault_mode", "falcon_hybrid", "facility_location")
    )
    global_random = ArmSpec(
        arm_id="global_random.global_random",
        acquisition_family="global_random",
        method_id="global_random",
        selector_id="global_random",
        clusterer_id=None,
    )
    return ProtocolManifest(
        schema_version="rcl-active-learning-combined-2.0-protocol-v1",
        datasets=(DatasetId.RCABENCH, DatasetId.AIOPS22_PRE),
        active_learning_seeds=ACTIVE_LEARNING_SEEDS,
        downstream_rcl_seed=DOWNSTREAM_RCL_SEED,
        budget=ANNOTATION_BUDGET,
        ordinary_mode=ORDINARY_MODE,
        lofo_fault_types={
            DatasetId.RCABENCH.value: (
                "NetworkDelay",
                "HTTPResponsePatchBody",
                "HTTPResponseDelay",
                "HTTPRequestAbort",
                "PodKill",
                "JVMMemoryStress",
                "NetworkBandwidth",
            ),
            DatasetId.AIOPS22_PRE.value: (
                "node 磁盘空间消耗",
                "k8s容器读io负载",
                "k8s容器网络丢包",
            ),
        },
        structure_gates={
            DatasetId.RCABENCH.value: StructureGateConfig(15, 13, 17),
            DatasetId.AIOPS22_PRE.value: StructureGateConfig(10, 8, 12),
        },
        arms=(*cluster_arms, *independent_arms, global_random),
    )


def ordinary_unit_ids(protocol: ProtocolManifest) -> tuple[str, ...]:
    return tuple(
        f"ordinary.{dataset.value}.{arm.method_id}.{arm.selector_id}.seed{seed}"
        for dataset in protocol.datasets
        for seed in protocol.active_learning_seeds
        for arm in protocol.arms
    )


def build_ordinary_units(
    protocol: ProtocolManifest,
    provenance: UnitProvenanceInputs,
) -> tuple[OrdinaryUnit, ...]:
    expected_ids = ordinary_unit_ids(protocol)
    query_ids = set(provenance.query_plan_sha256_by_unit)
    missing_queries = sorted(set(expected_ids).difference(query_ids))
    extra_queries = sorted(query_ids.difference(expected_ids))
    if missing_queries:
        raise ValueError(
            "missing query-plan provenance: " + ", ".join(missing_queries[:3])
        )
    if extra_queries:
        raise ValueError("unexpected query-plan provenance: " + ", ".join(extra_queries[:3]))

    protocol_sha256 = semantic_sha256(protocol)
    units: list[OrdinaryUnit] = []
    for dataset in protocol.datasets:
        if dataset.value not in provenance.data_sha256_by_dataset:
            raise ValueError(f"missing data provenance for {dataset.value}")
        if dataset.value not in provenance.representation_sha256_by_dataset:
            raise ValueError(f"missing representation provenance for {dataset.value}")
        for seed in protocol.active_learning_seeds:
            for arm in protocol.arms:
                unit_id = (
                    f"ordinary.{dataset.value}.{arm.method_id}."
                    f"{arm.selector_id}.seed{seed}"
                )
                units.append(
                    OrdinaryUnit(
                        unit_id=unit_id,
                        dataset_id=dataset,
                        arm_id=arm.arm_id,
                        method_id=arm.method_id,
                        selector_id=arm.selector_id,
                        active_learning_seed=seed,
                        budget=protocol.budget,
                        rcl_seed=protocol.downstream_rcl_seed,
                        evaluation_mode=protocol.ordinary_mode,
                        hashes=HashBundle(
                            code_sha256=provenance.code_sha256,
                            data_sha256=provenance.data_sha256_by_dataset[dataset.value],
                            representation_sha256=(
                                NO_REPRESENTATION_DEPENDENCY_SHA256
                                if arm.acquisition_family == "global_random"
                                else provenance.representation_sha256_by_dataset[
                                    dataset.value
                                ]
                            ),
                            query_plan_sha256=provenance.query_plan_sha256_by_unit[unit_id],
                            rcl_control_sha256=provenance.rcl_control_sha256,
                            protocol_sha256=protocol_sha256,
                        ),
                    )
                )
    if len(units) != 96 or len({unit.unit_id for unit in units}) != 96:
        raise ValueError("ordinary matrix must expand to exactly 96 unique units")
    return tuple(units)


def validate_completion_for_resume(
    unit: OrdinaryUnit | StrictLofoUnit,
    completion: Mapping[str, Any],
) -> None:
    if completion.get("status") != "complete":
        raise ValueError("completion status is not complete")
    if completion.get("unit_id") != unit.unit_id:
        raise ValueError("completion unit_id mismatch")
    if completion.get("record_role") != "fresh_unit":
        raise ValueError("completion record role is not fresh_unit")
    if completion.get("snapshot_action") != "none":
        raise ValueError("completion attempts a snapshot action")
    observed = completion.get("hashes")
    if not isinstance(observed, Mapping):
        raise ValueError("completion is missing immutable hashes")
    expected = unit.hashes.to_dict()
    for field_name, expected_value in expected.items():
        if observed.get(field_name) != expected_value:
            raise ValueError(f"stale completion hash: {field_name}")
    if set(observed) != set(expected):
        raise ValueError("stale completion hash: unexpected hash fields")


__all__ = [
    "ACTIVE_LEARNING_SEEDS",
    "ANNOTATION_BUDGET",
    "DOWNSTREAM_RCL_SEED",
    "NO_REPRESENTATION_DEPENDENCY_SHA256",
    "AggregateRecord",
    "AmendedArmSpec",
    "AmendedExecutionManifest",
    "AmendedProtocolManifest",
    "AmendedStructurePolicy",
    "AmendedUnitProvenanceInputs",
    "ApprovalRecord",
    "ApprovalState",
    "ArmSpec",
    "ClusteringGrid",
    "DatasetId",
    "DiagnosticRecord",
    "ExecutionManifest",
    "HashBundle",
    "GroundTruthClusterCap",
    "OrdinaryUnit",
    "ProtocolManifest",
    "RankingRecord",
    "RepresentationCandidate",
    "SelectorPlan",
    "SeedStructureObservation",
    "SnapshotReference",
    "StrictLofoUnit",
    "StructureGateConfig",
    "StructureGateResult",
    "StructureTripletAssessment",
    "UnitProvenanceInputs",
    "build_ordinary_units",
    "build_amended_ordinary_units",
    "amended_ordinary_unit_ids",
    "amended_protocol_manifest",
    "amended_structure_policy",
    "assess_structure_triplet",
    "default_protocol_manifest",
    "ordinary_unit_ids",
    "freeze_ground_truth_cluster_cap",
    "semantic_sha256",
    "validate_completion_for_resume",
]
