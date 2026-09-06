"""Strict leave-one-fault-out fold construction for combined study 2.0."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from rcl_study.combined_active_learning_baselines import (
    IndependentQueryPlan,
    build_global_random_query_plan,
)
from rcl_study.combined_active_learning_clustering import (
    EffectiveClusterPartition,
    StructuralConfigurationScore,
    build_effective_cluster_partition,
    dbscan_parameter_grid,
    evaluate_structure_gate,
    fit_native_kmeans,
    hdbscan_parameter_grid,
    kmeans_parameter_grid,
    mutual_knn_parameter_grid,
    rank_structural_configurations,
    run_dbscan_grid,
    run_hdbscan_grid,
    run_mutual_knn_grid,
)
from rcl_study.combined_active_learning_representation import (
    FORMAL_REPRESENTATION_IDS,
    FittedStandardizedRepresentation,
    RepresentationFitScope,
    RepresentationMatrixCandidate,
    build_formal_representation_candidates,
    fit_standardized_representation_scope,
)
from rcl_study.combined_active_learning_rcl import (
    DatasetOrdinaryStrategySelection,
    MethodOrdinaryQualification,
)
from rcl_study.combined_active_learning_schemas import (
    ACTIVE_LEARNING_SEEDS,
    ANNOTATION_BUDGET,
    DOWNSTREAM_RCL_SEED,
    DatasetId,
    default_protocol_manifest,
)


FROZEN_LOFO_FAULT_TYPES = {
    dataset_id: tuple(fault_types)
    for dataset_id, fault_types in default_protocol_manifest().lofo_fault_types.items()
}
CANDIDATE_SOURCE = "non-held-out original outer_train only"
TEST_SOURCE = "held-out type from original outer_train and outer_test"
UNUSED_SOURCE = "non-held-out original outer_test"
FROZEN_CLUSTERER_IDS = ("kmeans", "dbscan", "hdbscan", "mutual_knn")
LOFO_LABEL_ACCESS = "none_after_strict_lofo_fold_freeze"
CROSS_DATASET_ORDER = ("rcabench", "aiops2022_pre")


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class OriginalSplitCase:
    case_id: str
    original_split: str
    fault_type: str

    def __post_init__(self) -> None:
        if not self.case_id or self.case_id != self.case_id.strip():
            raise ValueError("LOFO case ID must be canonical and non-empty")
        if self.original_split not in {"outer_train", "outer_test"}:
            raise ValueError("LOFO original split must be outer_train or outer_test")
        if not self.fault_type or self.fault_type != self.fault_type.strip():
            raise ValueError("LOFO fault type must be canonical and non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "original_split": self.original_split,
            "fault_type": self.fault_type,
        }


@dataclass(frozen=True)
class StrictLofoFold:
    dataset_id: str
    held_out_fault_type: str
    original_inventory_sha256: str
    fault_type_normalization_sha256: str
    original_record_count: int
    candidate_case_ids: tuple[str, ...]
    held_out_outer_train_case_ids: tuple[str, ...]
    held_out_outer_test_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    unused_outer_test_case_ids: tuple[str, ...]
    candidate_source: str
    test_source: str
    unused_source: str

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if self.held_out_fault_type not in FROZEN_LOFO_FAULT_TYPES[self.dataset_id]:
            raise ValueError("held-out fault type is outside the frozen LOFO set")
        for values, name in (
            (self.candidate_case_ids, "candidate case IDs"),
            (self.held_out_outer_train_case_ids, "held-out outer-train case IDs"),
            (self.held_out_outer_test_case_ids, "held-out outer-test case IDs"),
            (self.test_case_ids, "test case IDs"),
            (self.unused_outer_test_case_ids, "unused outer-test case IDs"),
        ):
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise ValueError(f"LOFO {name} must be sorted and unique")
        if len(self.candidate_case_ids) < ANNOTATION_BUDGET:
            raise ValueError("LOFO candidate pool cannot fill budget 30")
        if not self.test_case_ids:
            raise ValueError("LOFO test set must contain the held-out fault type")
        if set(self.test_case_ids) != {
            *self.held_out_outer_train_case_ids,
            *self.held_out_outer_test_case_ids,
        }:
            raise ValueError("LOFO test set does not contain every held-out case")
        populations = (
            set(self.candidate_case_ids),
            set(self.test_case_ids),
            set(self.unused_outer_test_case_ids),
        )
        if any(
            populations[left].intersection(populations[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("LOFO candidate, test, and unused sets must be disjoint")
        if sum(len(values) for values in populations) != self.original_record_count:
            raise ValueError("LOFO partition does not exhaust the original inventory")
        if (
            self.candidate_source != CANDIDATE_SOURCE
            or self.test_source != TEST_SOURCE
            or self.unused_source != UNUSED_SOURCE
        ):
            raise ValueError("LOFO source-role contract drifted")
        _require_sha256(self.original_inventory_sha256, "original inventory SHA-256")
        _require_sha256(
            self.fault_type_normalization_sha256,
            "fault-type normalization SHA-256",
        )

    def _fold_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-strict-lofo-fold-v1",
            "dataset_id": self.dataset_id,
            "held_out_fault_type": self.held_out_fault_type,
            "original_inventory_sha256": self.original_inventory_sha256,
            "fault_type_normalization_sha256": (
                self.fault_type_normalization_sha256
            ),
            "original_record_count": self.original_record_count,
            "candidate_case_ids": list(self.candidate_case_ids),
            "held_out_outer_train_case_ids": list(
                self.held_out_outer_train_case_ids
            ),
            "held_out_outer_test_case_ids": list(
                self.held_out_outer_test_case_ids
            ),
            "test_case_ids": list(self.test_case_ids),
            "unused_outer_test_case_ids": list(self.unused_outer_test_case_ids),
            "candidate_source": self.candidate_source,
            "test_source": self.test_source,
            "unused_source": self.unused_source,
        }

    @property
    def fold_sha256(self) -> str:
        return _semantic_sha256(self._fold_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._fold_dict(), "fold_sha256": self.fold_sha256}


@dataclass(frozen=True)
class FrozenLofoRepresentationRefit:
    fold: StrictLofoFold
    scope: RepresentationFitScope
    frozen_representation_id: str
    frozen_representation_family: str
    fitted: FittedStandardizedRepresentation
    candidate: RepresentationMatrixCandidate
    label_access: str = LOFO_LABEL_ACCESS

    def __post_init__(self) -> None:
        if type(self.fold) is not StrictLofoFold:
            raise ValueError("LOFO representation refit requires a strict fold")
        if type(self.scope) is not RepresentationFitScope:
            raise ValueError("LOFO representation refit scope is invalid")
        if type(self.fitted) is not FittedStandardizedRepresentation:
            raise ValueError("LOFO fitted representation is invalid")
        if type(self.candidate) is not RepresentationMatrixCandidate:
            raise ValueError("LOFO representation candidate is invalid")
        if (
            self.scope.dataset_id != self.fold.dataset_id
            or self.scope.stage != "strict_lofo"
            or self.scope.held_out_fault_type != self.fold.held_out_fault_type
            or self.scope.fit_case_ids != self.fold.candidate_case_ids
            or self.scope.test_only_case_ids != self.fold.test_case_ids
            or self.scope.unused_outer_test_case_ids
            != self.fold.unused_outer_test_case_ids
        ):
            raise ValueError("LOFO representation scope does not match the fold")
        if self.fitted.scope != self.scope:
            raise ValueError("LOFO standardized fit does not match the fold scope")
        if (
            self.candidate.case_ids != self.fold.candidate_case_ids
            or self.candidate.fit_population_sha256
            != self.scope.fit_population_sha256
        ):
            raise ValueError("LOFO representation candidate population drifted")
        if (
            self.candidate.representation_id != self.frozen_representation_id
            or self.candidate.family != self.frozen_representation_family
            or not self.candidate.formal_eligible
        ):
            raise ValueError("LOFO frozen representation family drifted")
        if self.label_access != LOFO_LABEL_ACCESS:
            raise ValueError("LOFO representation label-access contract drifted")

    def _refit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-lofo-refit-v1",
            "dataset_id": self.fold.dataset_id,
            "held_out_fault_type": self.fold.held_out_fault_type,
            "fold_sha256": self.fold.fold_sha256,
            "fit_scope_sha256": self.scope.scope_sha256,
            "fit_population_sha256": self.scope.fit_population_sha256,
            "frozen_representation_id": self.frozen_representation_id,
            "frozen_representation_family": self.frozen_representation_family,
            "representation_matrix_sha256": self.candidate.matrix_sha256,
            "source_matrix_sha256": self.candidate.source_matrix_sha256,
            "row_count": len(self.candidate.case_ids),
            "output_dimension": int(self.candidate.matrix.shape[1]),
            "label_access": self.label_access,
        }

    @property
    def refit_sha256(self) -> str:
        return _semantic_sha256(self._refit_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._refit_dict(), "refit_sha256": self.refit_sha256}


@dataclass(frozen=True)
class LofoClusteringReselection:
    refit: FrozenLofoRepresentationRefit
    clusterer_id: str
    frozen_grid_size: int
    frozen_grid_sha256: str
    structure_gate_sha256: str
    attempted_fit_count: int
    configuration_scores: tuple[StructuralConfigurationScore, ...]
    fit_rejection_reasons: tuple[str, ...]
    selected_configuration_id: str | None
    selected_configuration: Mapping[str, Any] | None
    selected_score: StructuralConfigurationScore | None
    selected_partitions: tuple[EffectiveClusterPartition, ...]
    selected_geometries: tuple[Any, ...]
    status: str
    hard_stop: bool
    failure_reason: str | None
    label_access: str = LOFO_LABEL_ACCESS

    def __post_init__(self) -> None:
        if type(self.refit) is not FrozenLofoRepresentationRefit:
            raise ValueError("LOFO clustering requires a frozen representation refit")
        if self.clusterer_id not in FROZEN_CLUSTERER_IDS:
            raise ValueError("LOFO clusterer is outside the frozen protocol")
        if self.frozen_grid_size < 1:
            raise ValueError("LOFO frozen clustering grid must be non-empty")
        if self.attempted_fit_count != (
            self.frozen_grid_size * len(ACTIVE_LEARNING_SEEDS)
        ):
            raise ValueError("LOFO clustering did not attempt the complete frozen grid")
        _require_sha256(self.frozen_grid_sha256, "frozen grid SHA-256")
        _require_sha256(self.structure_gate_sha256, "structure gate SHA-256")
        if self.label_access != LOFO_LABEL_ACCESS:
            raise ValueError("LOFO clustering label-access contract drifted")
        if self.status == "selected":
            if (
                self.hard_stop
                or self.failure_reason is not None
                or self.selected_configuration_id is None
                or self.selected_configuration is None
                or self.selected_score is None
                or not self.selected_score.eligible
                or self.selected_score.rank != 1
                or self.selected_score.configuration_id
                != self.selected_configuration_id
                or tuple(
                    partition.active_learning_seed
                    for partition in self.selected_partitions
                )
                != ACTIVE_LEARNING_SEEDS
                or tuple(
                    geometry.active_learning_seed
                    for geometry in self.selected_geometries
                )
                != ACTIVE_LEARNING_SEEDS
            ):
                raise ValueError("selected LOFO clustering evidence is inconsistent")
            if not all(
                evaluate_structure_gate(partition).passed
                for partition in self.selected_partitions
            ):
                raise ValueError("selected LOFO clustering violates a structure gate")
        elif self.status == "structural_failure":
            if (
                not self.hard_stop
                or self.failure_reason
                != "no_frozen_grid_configuration_passed"
                or self.selected_configuration_id is not None
                or self.selected_configuration is not None
                or self.selected_score is not None
                or self.selected_partitions
                or self.selected_geometries
            ):
                raise ValueError("failed LOFO clustering evidence is inconsistent")
        else:
            raise ValueError("LOFO clustering status is invalid")

    def _selection_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-lofo-clustering-v1",
            "dataset_id": self.refit.fold.dataset_id,
            "held_out_fault_type": self.refit.fold.held_out_fault_type,
            "fold_sha256": self.refit.fold.fold_sha256,
            "representation_refit_sha256": self.refit.refit_sha256,
            "representation_matrix_sha256": self.refit.candidate.matrix_sha256,
            "clusterer_id": self.clusterer_id,
            "active_learning_seeds": list(ACTIVE_LEARNING_SEEDS),
            "frozen_grid_size": self.frozen_grid_size,
            "frozen_grid_sha256": self.frozen_grid_sha256,
            "structure_gate_sha256": self.structure_gate_sha256,
            "attempted_fit_count": self.attempted_fit_count,
            "fit_rejection_reasons": list(self.fit_rejection_reasons),
            "configuration_scores": [
                score.to_dict() for score in self.configuration_scores
            ],
            "selected_configuration_id": self.selected_configuration_id,
            "selected_configuration": (
                dict(self.selected_configuration)
                if self.selected_configuration is not None
                else None
            ),
            "selected_geometry_sha256_by_seed": {
                str(geometry.active_learning_seed): geometry.geometry_sha256
                for geometry in self.selected_geometries
            },
            "selected_partition_sha256_by_seed": {
                str(partition.active_learning_seed): partition.partition_sha256
                for partition in self.selected_partitions
            },
            "status": self.status,
            "hard_stop": self.hard_stop,
            "failure_reason": self.failure_reason,
            "label_access": self.label_access,
        }

    @property
    def reselection_sha256(self) -> str:
        return _semantic_sha256(self._selection_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._selection_dict(),
            "reselection_sha256": self.reselection_sha256,
        }


@dataclass(frozen=True)
class LofoGlobalRandomSeedResult:
    unit_id: str
    dataset_id: str
    held_out_fault_type: str
    fold_sha256: str
    query_plan_sha256: str
    execution_request_sha256: str
    runner_output_sha256: str
    active_learning_seed: int
    rcl_seed: int
    budget: int
    selected_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    evaluation_scope: str = "strict_lofo_held_out_fault_type_only"

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if not self.unit_id or not self.held_out_fault_type:
            raise ValueError("LOFO random result identity must be non-empty")
        for value, name in (
            (self.fold_sha256, "LOFO random fold SHA-256"),
            (self.query_plan_sha256, "LOFO random query plan SHA-256"),
            (self.execution_request_sha256, "LOFO execution request SHA-256"),
            (self.runner_output_sha256, "LOFO runner output SHA-256"),
        ):
            _require_sha256(value, name)
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("LOFO random active-learning seed is outside the protocol")
        if self.rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError("LOFO random downstream RCL seed must remain 42")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("LOFO random budget must remain 30")
        if (
            len(self.selected_case_ids) != self.budget
            or len(set(self.selected_case_ids)) != self.budget
            or not self.test_case_ids
            or set(self.selected_case_ids).intersection(self.test_case_ids)
        ):
            raise ValueError("LOFO random selected and test populations are invalid")
        if self.evaluation_scope != "strict_lofo_held_out_fault_type_only":
            raise ValueError("LOFO random evaluation scope drifted")
        for value in (self.hit_at_1, self.hit_at_3, self.hit_at_5):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("LOFO random Hit metric must be finite and in [0, 1]")

    @property
    def top135(self) -> float:
        return (self.hit_at_1 + self.hit_at_3 + self.hit_at_5) / 3.0

    def _result_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-lofo-seed-result-v1",
            "unit_id": self.unit_id,
            "dataset_id": self.dataset_id,
            "held_out_fault_type": self.held_out_fault_type,
            "fold_sha256": self.fold_sha256,
            "query_plan_sha256": self.query_plan_sha256,
            "execution_request_sha256": self.execution_request_sha256,
            "runner_output_sha256": self.runner_output_sha256,
            "active_learning_seed": self.active_learning_seed,
            "rcl_seed": self.rcl_seed,
            "budget": self.budget,
            "selected_case_ids": list(self.selected_case_ids),
            "test_case_ids": list(self.test_case_ids),
            "evaluation_scope": self.evaluation_scope,
            "metrics": {
                "Hit@1": self.hit_at_1,
                "Hit@3": self.hit_at_3,
                "Hit@5": self.hit_at_5,
                "TOP135": self.top135,
            },
        }

    @property
    def result_sha256(self) -> str:
        return _semantic_sha256(self._result_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._result_dict(), "result_sha256": self.result_sha256}


@dataclass(frozen=True)
class LofoMethodSeedResult:
    dataset_id: str
    held_out_fault_type: str
    fold_sha256: str
    method_id: str
    selector_id: str
    query_plan_sha256: str
    active_learning_seed: int
    rcl_seed: int
    budget: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if self.held_out_fault_type not in FROZEN_LOFO_FAULT_TYPES[self.dataset_id]:
            raise ValueError("LOFO method result fault type is outside the frozen set")
        if (
            not self.method_id
            or self.method_id == "global_random"
            or not self.selector_id
            or self.method_id != self.method_id.strip()
            or self.selector_id != self.selector_id.strip()
        ):
            raise ValueError("LOFO method result strategy identity is invalid")
        _require_sha256(self.fold_sha256, "LOFO method fold SHA-256")
        _require_sha256(self.query_plan_sha256, "LOFO method query plan SHA-256")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("LOFO method active-learning seed is outside the protocol")
        if self.rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError("LOFO method downstream RCL seed must remain 42")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("LOFO method budget must remain 30")
        for value in (self.hit_at_1, self.hit_at_3, self.hit_at_5):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("LOFO method Hit metric must be finite and in [0, 1]")

    @property
    def top135(self) -> float:
        return (self.hit_at_1 + self.hit_at_3 + self.hit_at_5) / 3.0

    def _result_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-lofo-method-seed-v1",
            "dataset_id": self.dataset_id,
            "held_out_fault_type": self.held_out_fault_type,
            "fold_sha256": self.fold_sha256,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "query_plan_sha256": self.query_plan_sha256,
            "active_learning_seed": self.active_learning_seed,
            "rcl_seed": self.rcl_seed,
            "budget": self.budget,
            "metrics": {
                "Hit@1": self.hit_at_1,
                "Hit@3": self.hit_at_3,
                "Hit@5": self.hit_at_5,
                "TOP135": self.top135,
            },
        }

    @property
    def result_sha256(self) -> str:
        return _semantic_sha256(self._result_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._result_dict(), "result_sha256": self.result_sha256}


@dataclass(frozen=True)
class LofoSeedMacroAverage:
    dataset_id: str
    method_id: str
    selector_id: str
    active_learning_seed: int
    held_out_fault_types: tuple[str, ...]
    method_macro_top135: float
    global_random_macro_top135: float

    def __post_init__(self) -> None:
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("LOFO macro active-learning seed is outside the protocol")
        if self.held_out_fault_types != FROZEN_LOFO_FAULT_TYPES[self.dataset_id]:
            raise ValueError("LOFO macro fault-type order drifted")
        for value in (
            self.method_macro_top135,
            self.global_random_macro_top135,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("LOFO macro TOP135 must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "active_learning_seed": self.active_learning_seed,
            "held_out_fault_types": list(self.held_out_fault_types),
            "method_macro_top135": self.method_macro_top135,
            "global_random_macro_top135": self.global_random_macro_top135,
        }


@dataclass(frozen=True)
class LofoDatasetQualification:
    dataset_id: str
    method_id: str
    selector_id: str
    held_out_fault_types: tuple[str, ...]
    active_learning_seeds: tuple[int, ...]
    per_seed: tuple[LofoSeedMacroAverage, ...]
    method_mean_top135: float
    global_random_mean_top135: float
    absolute_improvement: float
    relative_improvement: float | None
    qualified: bool
    method_result_sha256_by_key: tuple[tuple[str, str], ...]
    global_random_result_sha256_by_key: tuple[tuple[str, str], ...]
    comparison_rule: str = "strictly_greater_than_matched_global_random"

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if self.held_out_fault_types != FROZEN_LOFO_FAULT_TYPES[self.dataset_id]:
            raise ValueError("LOFO qualification fault-type order drifted")
        if self.active_learning_seeds != ACTIVE_LEARNING_SEEDS:
            raise ValueError("LOFO qualification seed order drifted")
        if tuple(item.active_learning_seed for item in self.per_seed) != (
            ACTIVE_LEARNING_SEEDS
        ):
            raise ValueError("LOFO qualification per-seed evidence is incomplete")
        if self.comparison_rule != "strictly_greater_than_matched_global_random":
            raise ValueError("LOFO qualification comparison rule drifted")
        expected_delta = self.method_mean_top135 - self.global_random_mean_top135
        if not math.isclose(self.absolute_improvement, expected_delta, abs_tol=1e-15):
            raise ValueError("LOFO qualification improvement was not recomputed")
        if self.qualified != (
            self.method_mean_top135 > self.global_random_mean_top135
        ):
            raise ValueError("LOFO qualification must use strict improvement")
        expected_count = len(self.held_out_fault_types) * len(
            self.active_learning_seeds
        )
        for records, name in (
            (self.method_result_sha256_by_key, "method"),
            (self.global_random_result_sha256_by_key, "global random"),
        ):
            if len(records) != expected_count or len({key for key, _ in records}) != (
                expected_count
            ):
                raise ValueError(f"LOFO {name} result hashes are incomplete")
            for _key, digest in records:
                _require_sha256(digest, f"LOFO {name} result SHA-256")

    def _qualification_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-lofo-qualification-v1",
            "dataset_id": self.dataset_id,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "held_out_fault_types": list(self.held_out_fault_types),
            "active_learning_seeds": list(self.active_learning_seeds),
            "aggregation_order": "fault_type_macro_then_active_learning_seed_mean",
            "per_seed": [item.to_dict() for item in self.per_seed],
            "method_mean_top135": self.method_mean_top135,
            "global_random_mean_top135": self.global_random_mean_top135,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "qualified": self.qualified,
            "comparison_rule": self.comparison_rule,
            "method_result_sha256_by_key": dict(
                self.method_result_sha256_by_key
            ),
            "global_random_result_sha256_by_key": dict(
                self.global_random_result_sha256_by_key
            ),
        }

    @property
    def qualification_sha256(self) -> str:
        return _semantic_sha256(self._qualification_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._qualification_dict(),
            "qualification_sha256": self.qualification_sha256,
        }


@dataclass(frozen=True)
class CombinedMethodQualification:
    method_id: str
    ordinary_qualified: bool
    lofo_evaluated: bool
    qualified: bool
    selected_selector_by_dataset: tuple[tuple[str, str], ...]
    ordinary_qualification_sha256: str
    lofo_qualification_sha256_by_dataset: tuple[tuple[str, str], ...]
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.method_id or self.method_id == "global_random":
            raise ValueError("combined qualification method identity is invalid")
        if tuple(dataset for dataset, _selector in self.selected_selector_by_dataset) != (
            CROSS_DATASET_ORDER
        ):
            raise ValueError("combined qualification selector dataset order drifted")
        _require_sha256(
            self.ordinary_qualification_sha256,
            "ordinary qualification SHA-256",
        )
        lofo_datasets = tuple(
            dataset for dataset, _digest in self.lofo_qualification_sha256_by_dataset
        )
        if self.lofo_evaluated:
            if lofo_datasets != CROSS_DATASET_ORDER:
                raise ValueError("combined qualification requires both LOFO datasets")
            for _dataset, digest in self.lofo_qualification_sha256_by_dataset:
                _require_sha256(digest, "LOFO qualification SHA-256")
        elif self.lofo_qualification_sha256_by_dataset:
            raise ValueError("pruned combined qualification cannot contain LOFO hashes")
        if self.qualified != (
            self.ordinary_qualified and self.lofo_evaluated and not self.failure_reasons
        ):
            raise ValueError("combined cross-dataset qualification decision drifted")
        if not self.ordinary_qualified and self.lofo_evaluated:
            raise ValueError("ordinary-unqualified method must be pruned before LOFO")

    def _qualification_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-cross-dataset-gates-v1",
            "method_id": self.method_id,
            "ordinary_qualified": self.ordinary_qualified,
            "lofo_evaluated": self.lofo_evaluated,
            "qualified": self.qualified,
            "selected_selector_by_dataset": dict(
                self.selected_selector_by_dataset
            ),
            "ordinary_qualification_sha256": (
                self.ordinary_qualification_sha256
            ),
            "lofo_qualification_sha256_by_dataset": dict(
                self.lofo_qualification_sha256_by_dataset
            ),
            "failure_reasons": list(self.failure_reasons),
            "gate_order": [
                "ordinary_snapshot_threshold_on_both_datasets",
                "strict_lofo_over_matched_global_random_on_both_datasets",
            ],
        }

    @property
    def qualification_sha256(self) -> str:
        return _semantic_sha256(self._qualification_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._qualification_dict(),
            "qualification_sha256": self.qualification_sha256,
        }


@dataclass(frozen=True)
class LofoGlobalRandomComparisonReference:
    method_id: str
    shared_execution_key: str
    unit_id: str
    query_plan_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.method_id
            or self.method_id == "global_random"
            or not self.shared_execution_key
            or not self.unit_id
        ):
            raise ValueError("LOFO random comparison identity is invalid")
        _require_sha256(self.query_plan_sha256, "shared query plan SHA-256")
        _require_sha256(self.result_sha256, "shared result SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "method_id": self.method_id,
            "shared_execution_key": self.shared_execution_key,
            "unit_id": self.unit_id,
            "query_plan_sha256": self.query_plan_sha256,
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True)
class SharedLofoGlobalRandomExecution:
    fold: StrictLofoFold
    query_plan: IndependentQueryPlan
    result: LofoGlobalRandomSeedResult
    surviving_method_ids: tuple[str, ...]
    comparison_references: tuple[LofoGlobalRandomComparisonReference, ...]
    shared_execution_key: str
    execution_count: int = 1

    def __post_init__(self) -> None:
        if type(self.fold) is not StrictLofoFold:
            raise ValueError("shared LOFO random execution requires a strict fold")
        if type(self.query_plan) is not IndependentQueryPlan:
            raise ValueError("shared LOFO random query plan is invalid")
        if type(self.result) is not LofoGlobalRandomSeedResult:
            raise ValueError("shared LOFO random result is invalid")
        if (
            self.query_plan.method_id != "global_random"
            or self.query_plan.selector_id != "global_random"
            or self.query_plan.dataset_id != self.fold.dataset_id
            or self.query_plan.selected_case_ids != self.result.selected_case_ids
            or self.query_plan.plan_sha256 != self.result.query_plan_sha256
            or self.result.fold_sha256 != self.fold.fold_sha256
            or self.result.test_case_ids != self.fold.test_case_ids
        ):
            raise ValueError("shared LOFO random plan or result identity drifted")
        if (
            not self.surviving_method_ids
            or tuple(sorted(self.surviving_method_ids))
            != self.surviving_method_ids
            or len(set(self.surviving_method_ids))
            != len(self.surviving_method_ids)
            or "global_random" in self.surviving_method_ids
        ):
            raise ValueError("shared LOFO random methods must be sorted and unique")
        expected = tuple(
            LofoGlobalRandomComparisonReference(
                method_id=method_id,
                shared_execution_key=self.shared_execution_key,
                unit_id=self.result.unit_id,
                query_plan_sha256=self.query_plan.plan_sha256,
                result_sha256=self.result.result_sha256,
            )
            for method_id in self.surviving_method_ids
        )
        if self.comparison_references != expected:
            raise ValueError("LOFO comparisons do not share one random result")
        if self.execution_count != 1:
            raise ValueError("LOFO global random must execute once per fold and seed")

    def _execution_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-shared-lofo-random-v1",
            "shared_execution_key": self.shared_execution_key,
            "execution_count": self.execution_count,
            "fold_sha256": self.fold.fold_sha256,
            "query_plan_sha256": self.query_plan.plan_sha256,
            "result_sha256": self.result.result_sha256,
            "surviving_method_ids": list(self.surviving_method_ids),
            "comparison_references": [
                reference.to_dict() for reference in self.comparison_references
            ],
        }

    @property
    def shared_execution_sha256(self) -> str:
        return _semantic_sha256(self._execution_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._execution_dict(),
            "shared_execution_sha256": self.shared_execution_sha256,
        }


def _coerce_record(value: OriginalSplitCase | Mapping[str, Any]) -> OriginalSplitCase:
    if type(value) is OriginalSplitCase:
        return value
    if not isinstance(value, Mapping):
        raise ValueError("LOFO original records must be typed cases or objects")
    return OriginalSplitCase(
        case_id=str(value.get("case_id", "")),
        original_split=str(value.get("original_split", value.get("split", ""))),
        fault_type=str(value.get("fault_type", "")),
    )


def _normalizer(
    dataset_id: str,
    fault_type_codebook: Mapping[str, str] | None,
) -> tuple[Any, str]:
    if dataset_id == "aiops2022_pre":
        identity = {
            "schema_version": "aiops22-pre-direct-fault-type-v1",
            "transform": "preserve_raw_fault_type",
        }
        return lambda value: value, _semantic_sha256(identity)
    if not isinstance(fault_type_codebook, Mapping) or not fault_type_codebook:
        raise ValueError("RCABench strict LOFO requires its frozen fault-type codebook")
    codebook = {str(code): str(name) for code, name in fault_type_codebook.items()}
    if any(not code or not name for code, name in codebook.items()):
        raise ValueError("RCABench fault-type codebook is invalid")

    def normalize(value: str) -> str:
        prefix = "fault_type_code:"
        if not value.startswith(prefix):
            raise ValueError("RCABench inventory fault type is not codebook encoded")
        code = value[len(prefix) :]
        if code not in codebook:
            raise ValueError("RCABench fault-type codebook lacks an inventory code")
        return codebook[code]

    identity = {
        "schema_version": "rcabench-fault-type-codebook-projection-v1",
        "code_to_raw_type": dict(sorted(codebook.items())),
    }
    return normalize, _semantic_sha256(identity)


def build_strict_lofo_fold(
    *,
    dataset_id: str,
    held_out_fault_type: str,
    original_records: Sequence[OriginalSplitCase | Mapping[str, Any]],
    original_inventory_sha256: str,
    fault_type_codebook: Mapping[str, str] | None = None,
) -> StrictLofoFold:
    """Partition the original split into strict candidate/test/unused roles."""

    DatasetId(dataset_id)
    _require_sha256(original_inventory_sha256, "original inventory SHA-256")
    if held_out_fault_type not in FROZEN_LOFO_FAULT_TYPES[dataset_id]:
        raise ValueError("held-out fault type is outside the frozen LOFO set")
    if isinstance(original_records, (str, bytes)) or not original_records:
        raise ValueError("LOFO original records must be non-empty")
    records = tuple(sorted((_coerce_record(row) for row in original_records), key=lambda row: row.case_id))
    case_ids = tuple(record.case_id for record in records)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("LOFO original case IDs must be unique")
    normalize, normalization_sha256 = _normalizer(
        dataset_id,
        fault_type_codebook,
    )

    normalized = tuple((record, normalize(record.fault_type)) for record in records)
    held_train = tuple(
        record.case_id
        for record, fault_type in normalized
        if record.original_split == "outer_train"
        and fault_type == held_out_fault_type
    )
    held_test = tuple(
        record.case_id
        for record, fault_type in normalized
        if record.original_split == "outer_test"
        and fault_type == held_out_fault_type
    )
    candidates = tuple(
        record.case_id
        for record, fault_type in normalized
        if record.original_split == "outer_train"
        and fault_type != held_out_fault_type
    )
    test = tuple(sorted((*held_train, *held_test)))
    unused = tuple(
        record.case_id
        for record, fault_type in normalized
        if record.original_split == "outer_test"
        and fault_type != held_out_fault_type
    )
    return StrictLofoFold(
        dataset_id=dataset_id,
        held_out_fault_type=held_out_fault_type,
        original_inventory_sha256=original_inventory_sha256,
        fault_type_normalization_sha256=normalization_sha256,
        original_record_count=len(records),
        candidate_case_ids=candidates,
        held_out_outer_train_case_ids=held_train,
        held_out_outer_test_case_ids=held_test,
        test_case_ids=test,
        unused_outer_test_case_ids=unused,
        candidate_source=CANDIDATE_SOURCE,
        test_source=TEST_SOURCE,
        unused_source=UNUSED_SOURCE,
    )


def validate_strict_lofo_fold(payload: Mapping[str, Any]) -> StrictLofoFold:
    """Recompute a serialized strict-LOFO fold seal and invariants."""

    if not isinstance(payload, Mapping):
        raise ValueError("strict LOFO fold must be an object")
    identity = dict(payload)
    declared = identity.pop("fold_sha256", None)
    if declared != _semantic_sha256(identity):
        raise ValueError("strict LOFO fold seal drifted")
    schema_version = identity.pop("schema_version", None)
    if schema_version != "combined-active-learning-strict-lofo-fold-v1":
        raise ValueError("strict LOFO fold schema drifted")
    for name in (
        "candidate_case_ids",
        "held_out_outer_train_case_ids",
        "held_out_outer_test_case_ids",
        "test_case_ids",
        "unused_outer_test_case_ids",
    ):
        identity[name] = tuple(identity.get(name, ()))
    try:
        return StrictLofoFold(**identity)
    except TypeError as exc:
        raise ValueError("strict LOFO fold field allowlist drifted") from exc


def refit_frozen_lofo_representation(
    *,
    fold: StrictLofoFold,
    frozen_representation_id: str,
    frozen_representation_family: str,
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> FrozenLofoRepresentationRefit:
    """Refit one already-selected representation on the strict candidate pool."""

    if type(fold) is not StrictLofoFold:
        raise ValueError("LOFO representation refit requires a strict fold")
    if frozen_representation_id not in FORMAL_REPRESENTATION_IDS:
        raise ValueError("LOFO refit requires a formal frozen representation")
    if not isinstance(frozen_representation_family, str) or not (
        frozen_representation_family.strip()
    ):
        raise ValueError("LOFO frozen representation family is invalid")
    scope = RepresentationFitScope(
        dataset_id=fold.dataset_id,
        stage="strict_lofo",
        held_out_fault_type=fold.held_out_fault_type,
        fit_case_ids=fold.candidate_case_ids,
        original_outer_train_case_ids=tuple(
            sorted(
                (
                    *fold.candidate_case_ids,
                    *fold.held_out_outer_train_case_ids,
                )
            )
        ),
        original_outer_test_case_ids=tuple(
            sorted(
                (
                    *fold.held_out_outer_test_case_ids,
                    *fold.unused_outer_test_case_ids,
                )
            )
        ),
        test_only_case_ids=fold.test_case_ids,
        unused_outer_test_case_ids=fold.unused_outer_test_case_ids,
        split_manifest_sha256=fold.original_inventory_sha256,
        label_access="lofo_fold_construction_only",
    )
    fitted = fit_standardized_representation_scope(
        scope=scope,
        case_modalities=case_modalities,
        feature_names_by_modality=feature_names_by_modality,
    )
    candidate = build_formal_representation_candidates(
        fitted.standardized
    )[frozen_representation_id]
    if candidate.family != frozen_representation_family:
        raise ValueError("LOFO frozen representation family does not match its ID")
    return FrozenLofoRepresentationRefit(
        fold=fold,
        scope=scope,
        frozen_representation_id=frozen_representation_id,
        frozen_representation_family=frozen_representation_family,
        fitted=fitted,
        candidate=candidate,
    )


def _frozen_clustering_grid(
    clusterer_id: str,
    dataset_id: str,
) -> tuple[Any, ...]:
    if clusterer_id == "kmeans":
        return kmeans_parameter_grid(dataset_id)
    if clusterer_id == "dbscan":
        return dbscan_parameter_grid()
    if clusterer_id == "hdbscan":
        return hdbscan_parameter_grid()
    if clusterer_id == "mutual_knn":
        return mutual_knn_parameter_grid()
    raise ValueError("LOFO clusterer is outside the frozen protocol")


def _configuration_id(clusterer_id: str, configuration: Any) -> str:
    return f"{clusterer_id}:{_semantic_sha256(configuration.to_dict())}"


def _grid_rows_for_seed(
    *,
    clusterer_id: str,
    dataset_id: str,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
) -> tuple[tuple[Any, Any | None, str | None], ...]:
    if clusterer_id == "kmeans":
        rows = []
        for configuration in kmeans_parameter_grid(dataset_id):
            try:
                result = fit_native_kmeans(
                    dataset_id=dataset_id,
                    candidate=candidate,
                    active_learning_seed=active_learning_seed,
                    cluster_count=configuration.cluster_count,
                    n_init=configuration.n_init,
                )
            except ValueError as exc:
                rows.append((configuration, None, str(exc)))
            else:
                rows.append((configuration, result, None))
        return tuple(rows)
    if clusterer_id == "dbscan":
        attempts = run_dbscan_grid(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
        ).attempts
    elif clusterer_id == "hdbscan":
        attempts = run_hdbscan_grid(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
        ).attempts
    elif clusterer_id == "mutual_knn":
        attempts = run_mutual_knn_grid(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
        ).attempts
    else:
        raise ValueError("LOFO clusterer is outside the frozen protocol")
    return tuple(
        (
            attempt.configuration,
            attempt.result,
            attempt.rejection_reason,
        )
        for attempt in attempts
    )


def reselect_lofo_clustering_parameters(
    *,
    refit: FrozenLofoRepresentationRefit,
    clusterer_id: str,
) -> LofoClusteringReselection:
    """Search only the frozen grid and apply unchanged label-free gates."""

    if type(refit) is not FrozenLofoRepresentationRefit:
        raise ValueError("LOFO clustering requires a frozen representation refit")
    if clusterer_id not in FROZEN_CLUSTERER_IDS:
        raise ValueError("LOFO clusterer is outside the frozen protocol")
    frozen_grid = _frozen_clustering_grid(clusterer_id, refit.fold.dataset_id)
    configurations = {
        _configuration_id(clusterer_id, configuration): configuration
        for configuration in frozen_grid
    }
    if len(configurations) != len(frozen_grid):
        raise RuntimeError("LOFO frozen clustering grid contains duplicates")
    expected_configuration_ids = tuple(configurations)
    geometries_by_configuration: dict[str, dict[int, Any]] = {
        configuration_id: {} for configuration_id in expected_configuration_ids
    }
    fit_rejections: list[str] = []
    attempted_fit_count = 0
    for seed in ACTIVE_LEARNING_SEEDS:
        rows = _grid_rows_for_seed(
            clusterer_id=clusterer_id,
            dataset_id=refit.fold.dataset_id,
            candidate=refit.candidate,
            active_learning_seed=seed,
        )
        attempted_fit_count += len(rows)
        observed_configuration_ids = tuple(
            _configuration_id(clusterer_id, configuration)
            for configuration, _result, _reason in rows
        )
        if observed_configuration_ids != expected_configuration_ids:
            raise RuntimeError("LOFO clustering runner drifted from the frozen grid")
        for configuration, result, rejection_reason in rows:
            configuration_id = _configuration_id(clusterer_id, configuration)
            if result is None:
                if not rejection_reason:
                    raise RuntimeError("LOFO clustering rejection lacks a reason")
                fit_rejections.append(
                    f"{configuration_id}:seed{seed}:{rejection_reason}"
                )
            else:
                if rejection_reason is not None:
                    raise RuntimeError("LOFO fitted clustering has a rejection reason")
                geometries_by_configuration[configuration_id][seed] = result

    partitions_by_configuration: dict[
        str, tuple[EffectiveClusterPartition, ...]
    ] = {}
    ordered_geometries_by_configuration: dict[str, tuple[Any, ...]] = {}
    for configuration_id in expected_configuration_ids:
        geometries_by_seed = geometries_by_configuration[configuration_id]
        if tuple(sorted(geometries_by_seed)) != ACTIVE_LEARNING_SEEDS:
            continue
        geometries = tuple(
            geometries_by_seed[seed] for seed in ACTIVE_LEARNING_SEEDS
        )
        partitions = tuple(
            build_effective_cluster_partition(
                dataset_id=refit.fold.dataset_id,
                clusterer_id=clusterer_id,
                active_learning_seed=geometry.active_learning_seed,
                case_ids=geometry.case_ids,
                raw_labels=geometry.labels,
                source_geometry_sha256=geometry.geometry_sha256,
            )
            for geometry in geometries
        )
        ordered_geometries_by_configuration[configuration_id] = geometries
        partitions_by_configuration[configuration_id] = partitions

    scores = (
        rank_structural_configurations(partitions_by_configuration)
        if partitions_by_configuration
        else ()
    )
    selected_score = next((score for score in scores if score.eligible), None)
    grid_sha256 = _semantic_sha256(
        {
            "schema_version": "combined-active-learning-frozen-grid-v1",
            "dataset_id": refit.fold.dataset_id,
            "clusterer_id": clusterer_id,
            "configurations": [
                configuration.to_dict() for configuration in frozen_grid
            ],
        }
    )
    structure_gate_sha256 = _semantic_sha256(
        default_protocol_manifest()
        .structure_gates[refit.fold.dataset_id]
        .to_dict()
    )
    if selected_score is None:
        return LofoClusteringReselection(
            refit=refit,
            clusterer_id=clusterer_id,
            frozen_grid_size=len(frozen_grid),
            frozen_grid_sha256=grid_sha256,
            structure_gate_sha256=structure_gate_sha256,
            attempted_fit_count=attempted_fit_count,
            configuration_scores=tuple(scores),
            fit_rejection_reasons=tuple(fit_rejections),
            selected_configuration_id=None,
            selected_configuration=None,
            selected_score=None,
            selected_partitions=(),
            selected_geometries=(),
            status="structural_failure",
            hard_stop=True,
            failure_reason="no_frozen_grid_configuration_passed",
        )
    selected_id = selected_score.configuration_id
    return LofoClusteringReselection(
        refit=refit,
        clusterer_id=clusterer_id,
        frozen_grid_size=len(frozen_grid),
        frozen_grid_sha256=grid_sha256,
        structure_gate_sha256=structure_gate_sha256,
        attempted_fit_count=attempted_fit_count,
        configuration_scores=tuple(scores),
        fit_rejection_reasons=tuple(fit_rejections),
        selected_configuration_id=selected_id,
        selected_configuration=MappingProxyType(
            dict(configurations[selected_id].to_dict())
        ),
        selected_score=selected_score,
        selected_partitions=partitions_by_configuration[selected_id],
        selected_geometries=ordered_geometries_by_configuration[selected_id],
        status="selected",
        hard_stop=False,
        failure_reason=None,
    )


def _lofo_hit_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"LOFO random metric {name} must be finite and in [0, 1]")
    return float(value)


def execute_shared_lofo_global_random(
    *,
    fold: StrictLofoFold,
    active_learning_seed: int,
    surviving_method_ids: Sequence[str],
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> SharedLofoGlobalRandomExecution:
    """Execute one random baseline and share it across all method comparisons."""

    if type(fold) is not StrictLofoFold:
        raise ValueError("shared LOFO random execution requires a strict fold")
    if active_learning_seed not in ACTIVE_LEARNING_SEEDS:
        raise ValueError("LOFO random active-learning seed is outside the protocol")
    if isinstance(surviving_method_ids, (str, bytes)):
        raise ValueError("LOFO random comparisons require a method sequence")
    methods = tuple(surviving_method_ids)
    if not methods:
        raise ValueError("LOFO random sharing requires at least one method")
    if (
        len(set(methods)) != len(methods)
        or any(
            not isinstance(method_id, str)
            or not method_id
            or method_id != method_id.strip()
            or method_id == "global_random"
            for method_id in methods
        )
    ):
        raise ValueError("LOFO random comparison methods must be canonical and unique")
    if not callable(runner):
        raise ValueError("LOFO random execution requires a callable runner")
    ordered_methods = tuple(sorted(methods))
    query_plan = build_global_random_query_plan(
        dataset_id=fold.dataset_id,
        eligible_case_ids=fold.candidate_case_ids,
        active_learning_seed=active_learning_seed,
    )
    unit_id = (
        f"strict_lofo.global_random.{fold.dataset_id}."
        f"seed{active_learning_seed}.{fold.fold_sha256[:12]}"
    )
    request_identity = {
        "schema_version": "combined-active-learning-lofo-runner-request-v1",
        "unit_id": unit_id,
        "dataset_id": fold.dataset_id,
        "held_out_fault_type": fold.held_out_fault_type,
        "fold_sha256": fold.fold_sha256,
        "query_plan_sha256": query_plan.plan_sha256,
        "candidate_pool_sha256": query_plan.candidate_pool_sha256,
        "active_learning_seed": active_learning_seed,
        "rcl_seed": DOWNSTREAM_RCL_SEED,
        "budget": ANNOTATION_BUDGET,
        "selected_case_ids": list(query_plan.selected_case_ids),
        "test_case_ids": list(fold.test_case_ids),
        "evaluation_scope": "strict_lofo_held_out_fault_type_only",
        "fresh_training_required": True,
    }
    request = {
        **request_identity,
        "execution_request_sha256": _semantic_sha256(request_identity),
    }
    raw = runner(deepcopy(request))
    if not isinstance(raw, Mapping):
        raise ValueError("LOFO random runner result must be an object")
    checked = deepcopy(dict(raw))
    expected_fields = {
        "schema_version",
        "unit_id",
        "dataset_id",
        "held_out_fault_type",
        "fold_sha256",
        "query_plan_sha256",
        "execution_request_sha256",
        "active_learning_seed",
        "rcl_seed",
        "budget",
        "selected_case_ids",
        "test_case_ids",
        "evaluation_scope",
        "metrics",
    }
    if set(checked) != expected_fields:
        raise ValueError("LOFO random runner result field allowlist drifted")
    if checked.get("selected_case_ids") != request["selected_case_ids"]:
        raise ValueError("LOFO random runner selected-case identity drifted")
    if checked.get("test_case_ids") != request["test_case_ids"]:
        raise ValueError("LOFO random runner test-only identity drifted")
    expected_identity = {
        "schema_version": "combined-active-learning-lofo-runner-result-v1",
        "unit_id": unit_id,
        "dataset_id": fold.dataset_id,
        "held_out_fault_type": fold.held_out_fault_type,
        "fold_sha256": fold.fold_sha256,
        "query_plan_sha256": query_plan.plan_sha256,
        "execution_request_sha256": request["execution_request_sha256"],
        "active_learning_seed": active_learning_seed,
        "rcl_seed": DOWNSTREAM_RCL_SEED,
        "budget": ANNOTATION_BUDGET,
        "evaluation_scope": "strict_lofo_held_out_fault_type_only",
    }
    for name, expected in expected_identity.items():
        if checked.get(name) != expected:
            raise ValueError(f"LOFO random runner identity drifted for {name}")
    metrics = checked.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "Hit@1",
        "Hit@3",
        "Hit@5",
    }:
        raise ValueError("LOFO random runner metrics must be Hit@1/3/5")
    result = LofoGlobalRandomSeedResult(
        unit_id=unit_id,
        dataset_id=fold.dataset_id,
        held_out_fault_type=fold.held_out_fault_type,
        fold_sha256=fold.fold_sha256,
        query_plan_sha256=query_plan.plan_sha256,
        execution_request_sha256=request["execution_request_sha256"],
        runner_output_sha256=_semantic_sha256(checked),
        active_learning_seed=active_learning_seed,
        rcl_seed=DOWNSTREAM_RCL_SEED,
        budget=ANNOTATION_BUDGET,
        selected_case_ids=query_plan.selected_case_ids,
        test_case_ids=fold.test_case_ids,
        hit_at_1=_lofo_hit_metric(metrics, "Hit@1"),
        hit_at_3=_lofo_hit_metric(metrics, "Hit@3"),
        hit_at_5=_lofo_hit_metric(metrics, "Hit@5"),
    )
    shared_execution_key = unit_id
    references = tuple(
        LofoGlobalRandomComparisonReference(
            method_id=method_id,
            shared_execution_key=shared_execution_key,
            unit_id=unit_id,
            query_plan_sha256=query_plan.plan_sha256,
            result_sha256=result.result_sha256,
        )
        for method_id in ordered_methods
    )
    return SharedLofoGlobalRandomExecution(
        fold=fold,
        query_plan=query_plan,
        result=result,
        surviving_method_ids=ordered_methods,
        comparison_references=references,
        shared_execution_key=shared_execution_key,
    )


def aggregate_and_qualify_lofo_dataset(
    *,
    method_results: Sequence[LofoMethodSeedResult],
    global_random_results: Sequence[LofoGlobalRandomSeedResult],
) -> LofoDatasetQualification:
    """Macro-average strict LOFO and apply the matched random strict gate."""

    if isinstance(method_results, (str, bytes)) or not method_results:
        raise ValueError("LOFO aggregation requires method results")
    if isinstance(global_random_results, (str, bytes)) or not (
        global_random_results
    ):
        raise ValueError("LOFO aggregation requires global-random results")
    methods = tuple(method_results)
    randoms = tuple(global_random_results)
    if any(type(item) is not LofoMethodSeedResult for item in methods):
        raise ValueError("LOFO method result schema is invalid")
    if any(type(item) is not LofoGlobalRandomSeedResult for item in randoms):
        raise ValueError("LOFO global-random result schema is invalid")
    strategies = {
        (item.dataset_id, item.method_id, item.selector_id) for item in methods
    }
    if len(strategies) != 1:
        raise ValueError("LOFO aggregation requires one dataset-specific strategy")
    dataset_id, method_id, selector_id = next(iter(strategies))
    fault_types = FROZEN_LOFO_FAULT_TYPES[dataset_id]
    expected_keys = tuple(
        (fault_type, seed)
        for seed in ACTIVE_LEARNING_SEEDS
        for fault_type in fault_types
    )
    method_by_key = {
        (item.held_out_fault_type, item.active_learning_seed): item
        for item in methods
    }
    random_by_key = {
        (item.held_out_fault_type, item.active_learning_seed): item
        for item in randoms
        if item.dataset_id == dataset_id
    }
    expected_key_set = set(expected_keys)
    if (
        len(methods) != len(expected_keys)
        or len(method_by_key) != len(expected_keys)
        or set(method_by_key) != expected_key_set
        or len(randoms) != len(expected_keys)
        or len(random_by_key) != len(expected_keys)
        or set(random_by_key) != expected_key_set
    ):
        raise ValueError("LOFO aggregation requires the complete frozen fold-by-seed matrix")
    for key in expected_keys:
        method_item = method_by_key[key]
        random_item = random_by_key[key]
        if method_item.fold_sha256 != random_item.fold_sha256:
            raise ValueError("LOFO method result lacks its matched random fold")

    per_seed = tuple(
        LofoSeedMacroAverage(
            dataset_id=dataset_id,
            method_id=method_id,
            selector_id=selector_id,
            active_learning_seed=seed,
            held_out_fault_types=fault_types,
            method_macro_top135=math.fsum(
                method_by_key[(fault_type, seed)].top135
                for fault_type in fault_types
            )
            / len(fault_types),
            global_random_macro_top135=math.fsum(
                random_by_key[(fault_type, seed)].top135
                for fault_type in fault_types
            )
            / len(fault_types),
        )
        for seed in ACTIVE_LEARNING_SEEDS
    )
    method_mean = math.fsum(
        item.method_macro_top135 for item in per_seed
    ) / len(per_seed)
    random_mean = math.fsum(
        item.global_random_macro_top135 for item in per_seed
    ) / len(per_seed)
    improvement = method_mean - random_mean
    relative_improvement = improvement / random_mean if random_mean > 0.0 else None
    key_names = tuple(
        (f"{fault_type}|seed{seed}", (fault_type, seed))
        for fault_type, seed in expected_keys
    )
    return LofoDatasetQualification(
        dataset_id=dataset_id,
        method_id=method_id,
        selector_id=selector_id,
        held_out_fault_types=fault_types,
        active_learning_seeds=ACTIVE_LEARNING_SEEDS,
        per_seed=per_seed,
        method_mean_top135=method_mean,
        global_random_mean_top135=random_mean,
        absolute_improvement=improvement,
        relative_improvement=relative_improvement,
        qualified=method_mean > random_mean,
        method_result_sha256_by_key=tuple(
            (name, method_by_key[key].result_sha256) for name, key in key_names
        ),
        global_random_result_sha256_by_key=tuple(
            (name, random_by_key[key].result_sha256) for name, key in key_names
        ),
    )


def qualify_combined_method_across_datasets(
    *,
    ordinary_qualification: MethodOrdinaryQualification,
    ordinary_selections: Sequence[DatasetOrdinaryStrategySelection],
    lofo_qualifications: Sequence[LofoDatasetQualification],
) -> CombinedMethodQualification:
    """Apply ordinary pruning before the two-dataset strict-LOFO gate."""

    if type(ordinary_qualification) is not MethodOrdinaryQualification:
        raise ValueError("combined qualification requires an ordinary decision")
    if isinstance(ordinary_selections, (str, bytes)) or len(
        ordinary_selections
    ) != len(CROSS_DATASET_ORDER):
        raise ValueError("combined qualification requires both ordinary selections")
    if any(
        type(selection) is not DatasetOrdinaryStrategySelection
        for selection in ordinary_selections
    ):
        raise ValueError("combined ordinary selection schema is invalid")
    selections_by_dataset = {
        selection.dataset_id: selection for selection in ordinary_selections
    }
    if set(selections_by_dataset) != set(CROSS_DATASET_ORDER):
        raise ValueError("combined qualification requires both ordinary datasets")
    if any(
        selection.method_id != ordinary_qualification.method_id
        for selection in ordinary_selections
    ):
        raise ValueError("combined ordinary selections must share one method")
    expected_selection_hashes = tuple(
        (
            dataset_id,
            selections_by_dataset[dataset_id].selection_sha256,
        )
        for dataset_id in CROSS_DATASET_ORDER
    )
    if (
        ordinary_qualification.selection_sha256_by_dataset
        != expected_selection_hashes
        or ordinary_qualification.qualified
        != all(
            selections_by_dataset[dataset_id].passed
            for dataset_id in CROSS_DATASET_ORDER
        )
    ):
        raise ValueError("combined ordinary qualification evidence drifted")
    selected_selectors = tuple(
        (
            dataset_id,
            selections_by_dataset[dataset_id].selected_selector_id,
        )
        for dataset_id in CROSS_DATASET_ORDER
    )
    lofo_items = tuple(lofo_qualifications)
    if not ordinary_qualification.qualified:
        if lofo_items:
            raise ValueError("ordinary-unqualified method must be pruned before LOFO")
        return CombinedMethodQualification(
            method_id=ordinary_qualification.method_id,
            ordinary_qualified=False,
            lofo_evaluated=False,
            qualified=False,
            selected_selector_by_dataset=selected_selectors,
            ordinary_qualification_sha256=(
                ordinary_qualification.qualification_sha256
            ),
            lofo_qualification_sha256_by_dataset=(),
            failure_reasons=tuple(
                f"ordinary:{reason}"
                for reason in ordinary_qualification.failure_reasons
            ),
        )
    if len(lofo_items) != len(CROSS_DATASET_ORDER) or any(
        type(item) is not LofoDatasetQualification for item in lofo_items
    ):
        raise ValueError("ordinary-qualified method requires both LOFO datasets")
    lofo_by_dataset = {item.dataset_id: item for item in lofo_items}
    if set(lofo_by_dataset) != set(CROSS_DATASET_ORDER):
        raise ValueError("ordinary-qualified method requires both LOFO datasets")
    for dataset_id in CROSS_DATASET_ORDER:
        item = lofo_by_dataset[dataset_id]
        selection = selections_by_dataset[dataset_id]
        if item.method_id != ordinary_qualification.method_id:
            raise ValueError("LOFO qualification method differs from ordinary")
        if item.selector_id != selection.selected_selector_id:
            raise ValueError("LOFO must use the dataset ordinary-best selector")
    failure_reasons = tuple(
        (
            f"{dataset_id} LOFO mean TOP135 "
            f"{lofo_by_dataset[dataset_id].method_mean_top135:.6f} is not "
            "strictly greater than matched global random "
            f"{lofo_by_dataset[dataset_id].global_random_mean_top135:.6f}"
        )
        for dataset_id in CROSS_DATASET_ORDER
        if not lofo_by_dataset[dataset_id].qualified
    )
    return CombinedMethodQualification(
        method_id=ordinary_qualification.method_id,
        ordinary_qualified=True,
        lofo_evaluated=True,
        qualified=not failure_reasons,
        selected_selector_by_dataset=selected_selectors,
        ordinary_qualification_sha256=(
            ordinary_qualification.qualification_sha256
        ),
        lofo_qualification_sha256_by_dataset=tuple(
            (
                dataset_id,
                lofo_by_dataset[dataset_id].qualification_sha256,
            )
            for dataset_id in CROSS_DATASET_ORDER
        ),
        failure_reasons=failure_reasons,
    )


__all__ = [
    "CANDIDATE_SOURCE",
    "CombinedMethodQualification",
    "CROSS_DATASET_ORDER",
    "FROZEN_LOFO_FAULT_TYPES",
    "FrozenLofoRepresentationRefit",
    "LOFO_LABEL_ACCESS",
    "LofoClusteringReselection",
    "LofoGlobalRandomComparisonReference",
    "LofoGlobalRandomSeedResult",
    "LofoDatasetQualification",
    "LofoMethodSeedResult",
    "LofoSeedMacroAverage",
    "OriginalSplitCase",
    "SharedLofoGlobalRandomExecution",
    "StrictLofoFold",
    "TEST_SOURCE",
    "UNUSED_SOURCE",
    "build_strict_lofo_fold",
    "aggregate_and_qualify_lofo_dataset",
    "execute_shared_lofo_global_random",
    "refit_frozen_lofo_representation",
    "qualify_combined_method_across_datasets",
    "reselect_lofo_clustering_parameters",
    "validate_strict_lofo_fold",
]
