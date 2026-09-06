"""Fail-closed bridge from combined 2.0 query plans to the RCL runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from rcl_study.combined_active_learning_acquisition import ClusterQueryPlan
from rcl_study.combined_active_learning_baselines import IndependentQueryPlan
from rcl_study.combined_active_learning_schemas import (
    ACTIVE_LEARNING_SEEDS,
    ANNOTATION_BUDGET,
    DOWNSTREAM_RCL_SEED,
    DatasetId,
    ORDINARY_MODE,
    OrdinaryUnit,
    StrictLofoUnit,
)
from rcl_study.query_active_deepening import first_target_rank

if TYPE_CHECKING:
    from rcl_study.combined_active_learning_lofo import StrictLofoFold


FIXED_QUERY_SELECTOR_ID = "fixed_budget_set"
T1_ONLY_SCOPE = "T1_only"
STRICT_LOFO_SCOPE = "strict_lofo_held_out_fault_type_only"
ORDINARY_SNAPSHOT_THRESHOLDS = (
    ("rcabench", 0.637289),
    ("aiops2022_pre", 0.788735),
)
EXPECTED_ORDINARY_SELECTORS = {
    "kmeans": ("center", "boundary", "within_cluster_random"),
    "dbscan": ("center", "boundary", "within_cluster_random"),
    "hdbscan": ("center", "boundary", "within_cluster_random"),
    "mutual_knn": ("center", "boundary", "within_cluster_random"),
    "knn_fault_mode": ("knn_fault_mode",),
    "falcon_hybrid": ("falcon_hybrid",),
    "facility_location": ("facility_location",),
    "global_random": ("global_random",),
}


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _population_sha256(values: Sequence[str]) -> str:
    return _semantic_sha256(list(values))


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_posix_root(value: str) -> PurePosixPath:
    root = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or not root.is_absolute()
        or any(part in {".", ".."} for part in value.split("/"))
        or str(root) != value
    ):
        raise ValueError("RCL output root must be a canonical absolute POSIX path")
    return root


@dataclass(frozen=True)
class FrozenRclControl:
    """The complete ordinary-stage downstream randomness/control identity."""

    split_manifest_sha256: str
    model_config_sha256: str
    runner_sha256: str
    split_seed: int = DOWNSTREAM_RCL_SEED
    model_seed: int = DOWNSTREAM_RCL_SEED
    initialization_seed: int = DOWNSTREAM_RCL_SEED
    training_seed: int = DOWNSTREAM_RCL_SEED
    outer_test_ratio: float = 0.30
    inner_validation_ratio: float = 0.20
    normal_policy: str = "fault_only"
    fresh_training_required: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.split_manifest_sha256, "split manifest SHA-256"),
            (self.model_config_sha256, "model config SHA-256"),
            (self.runner_sha256, "runner SHA-256"),
        ):
            _require_sha256(value, name)
        if {
            self.split_seed,
            self.model_seed,
            self.initialization_seed,
            self.training_seed,
        } != {DOWNSTREAM_RCL_SEED}:
            raise ValueError("all downstream RCL randomness must use seed 42")
        if (
            not math.isclose(self.outer_test_ratio, 0.30)
            or not math.isclose(self.inner_validation_ratio, 0.20)
        ):
            raise ValueError("ordinary RCL split ratios drifted")
        if self.normal_policy != "fault_only":
            raise ValueError("ordinary RCL normal policy drifted")
        if self.fresh_training_required is not True:
            raise ValueError("ordinary RCL must start from a fresh model state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-rcl-control-v1",
            "split_manifest_sha256": self.split_manifest_sha256,
            "model_config_sha256": self.model_config_sha256,
            "runner_sha256": self.runner_sha256,
            "split_seed": self.split_seed,
            "model_seed": self.model_seed,
            "initialization_seed": self.initialization_seed,
            "training_seed": self.training_seed,
            "outer_test_ratio": self.outer_test_ratio,
            "inner_validation_ratio": self.inner_validation_ratio,
            "normal_policy": self.normal_policy,
            "fresh_training_required": self.fresh_training_required,
            "state_inputs": {
                "model_state": None,
                "checkpoint": None,
                "optimizer": None,
                "resume": None,
                "warm_start": None,
            },
        }

    @property
    def control_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())


@dataclass(frozen=True)
class Budget30RclBridge:
    """Immutable, label-free projection of one query plan into one RCL unit."""

    unit_id: str
    dataset_id: str
    arm_id: str
    method_id: str
    source_selector_id: str
    active_learning_seed: int
    budget: int
    selected_case_ids: tuple[str, ...]
    query_plan_type: str
    query_plan_sha256: str
    query_plan_reference_sha256: str
    output_root: str
    code_sha256: str
    data_sha256: str
    representation_sha256: str
    protocol_sha256: str
    control: FrozenRclControl

    def __post_init__(self) -> None:
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("RCL bridge requires the frozen budget 30")
        if len(self.selected_case_ids) != self.budget or len(
            set(self.selected_case_ids)
        ) != self.budget:
            raise ValueError("RCL bridge selected cases must be 30 unique IDs")
        if any(not case_id for case_id in self.selected_case_ids):
            raise ValueError("RCL bridge case IDs must be non-empty")
        if self.query_plan_type not in {"cluster", "independent"}:
            raise ValueError("RCL bridge query-plan type drifted")
        for value, name in (
            (self.query_plan_sha256, "query plan SHA-256"),
            (self.query_plan_reference_sha256, "query plan reference SHA-256"),
            (self.code_sha256, "code SHA-256"),
            (self.data_sha256, "data SHA-256"),
            (self.representation_sha256, "representation SHA-256"),
            (self.protocol_sha256, "protocol SHA-256"),
        ):
            _require_sha256(value, name)
        _require_posix_root(self.output_root)

    def _bridge_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-budget30-rcl-bridge-v1",
            "unit_id": self.unit_id,
            "dataset_id": self.dataset_id,
            "arm_id": self.arm_id,
            "method_id": self.method_id,
            "source_selector_id": self.source_selector_id,
            "active_learning_seed": self.active_learning_seed,
            "budget": self.budget,
            "selected_case_ids": list(self.selected_case_ids),
            "query_plan_type": self.query_plan_type,
            "query_plan_sha256": self.query_plan_sha256,
            "query_plan_reference_sha256": self.query_plan_reference_sha256,
            "output_root": self.output_root,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "representation_sha256": self.representation_sha256,
            "protocol_sha256": self.protocol_sha256,
            "rcl_control": self.control.to_dict(),
            "rcl_control_sha256": self.control.control_sha256,
        }

    @property
    def bridge_sha256(self) -> str:
        return _semantic_sha256(self._bridge_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._bridge_dict(), "bridge_sha256": self.bridge_sha256}


@dataclass(frozen=True)
class StrictLofoRclBridge:
    """Sealed strict-LOFO projection with explicit population boundaries."""

    unit_id: str
    dataset_id: str
    held_out_fault_type: str
    method_id: str
    source_selector_id: str
    active_learning_seed: int
    budget: int
    selected_case_ids: tuple[str, ...]
    training_window_ids: tuple[str, ...]
    query_plan_type: str
    query_plan_sha256: str
    query_plan_reference_sha256: str
    fold: StrictLofoFold
    output_root: str
    code_sha256: str
    data_sha256: str
    representation_sha256: str
    protocol_sha256: str
    control: FrozenRclControl

    def __post_init__(self) -> None:
        from rcl_study.combined_active_learning_lofo import StrictLofoFold

        DatasetId(self.dataset_id)
        if type(self.fold) is not StrictLofoFold:
            raise ValueError("strict-LOFO bridge requires a typed sealed fold")
        if (
            self.fold.dataset_id != self.dataset_id
            or self.fold.held_out_fault_type != self.held_out_fault_type
        ):
            raise ValueError("strict-LOFO bridge fold identity drifted")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("strict-LOFO bridge active-learning seed drifted")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("strict-LOFO bridge requires the frozen budget 30")
        if (
            len(self.selected_case_ids) != self.budget
            or len(set(self.selected_case_ids)) != self.budget
            or any(not case_id for case_id in self.selected_case_ids)
        ):
            raise ValueError("strict-LOFO selected cases must be 30 unique IDs")
        if (
            not self.training_window_ids
            or len(self.training_window_ids) != len(set(self.training_window_ids))
            or any(not case_id for case_id in self.training_window_ids)
        ):
            raise ValueError("strict-LOFO training population must be non-empty and unique")
        if (
            not set(self.fold.candidate_case_ids).issubset(self.training_window_ids)
            or set(self.training_window_ids).intersection(self.fold.test_case_ids)
            or set(self.training_window_ids).intersection(
                self.fold.unused_outer_test_case_ids
            )
        ):
            raise ValueError("strict-LOFO signed training population drifted")
        if not set(self.selected_case_ids).issubset(
            self.fold.candidate_case_ids
        ) or not set(self.selected_case_ids).issubset(self.training_window_ids):
            raise ValueError("selected cases must belong to the strict candidate population")
        if set(self.selected_case_ids).intersection(
            (*self.fold.test_case_ids, *self.fold.unused_outer_test_case_ids)
        ):
            raise ValueError("selected cases leaked into strict test or unused populations")
        if self.query_plan_type not in {"cluster", "independent"}:
            raise ValueError("strict-LOFO query-plan type drifted")
        if type(self.control) is not FrozenRclControl:
            raise ValueError("strict-LOFO bridge requires frozen RCL controls")
        for value, name in (
            (self.query_plan_sha256, "query plan SHA-256"),
            (self.query_plan_reference_sha256, "query plan reference SHA-256"),
            (self.code_sha256, "code SHA-256"),
            (self.data_sha256, "data SHA-256"),
            (self.representation_sha256, "representation SHA-256"),
            (self.protocol_sha256, "protocol SHA-256"),
        ):
            _require_sha256(value, name)
        _require_posix_root(self.output_root)

    def _bridge_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-strict-lofo-rcl-bridge-v1",
            "unit_id": self.unit_id,
            "dataset_id": self.dataset_id,
            "held_out_fault_type": self.held_out_fault_type,
            "method_id": self.method_id,
            "source_selector_id": self.source_selector_id,
            "active_learning_seed": self.active_learning_seed,
            "budget": self.budget,
            "selected_case_ids": list(self.selected_case_ids),
            "training_window_ids": list(self.training_window_ids),
            "query_plan_type": self.query_plan_type,
            "query_plan_sha256": self.query_plan_sha256,
            "query_plan_reference_sha256": self.query_plan_reference_sha256,
            "strict_lofo_fold": self.fold.to_dict(),
            "fold_sha256": self.fold.fold_sha256,
            "output_root": self.output_root,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "representation_sha256": self.representation_sha256,
            "protocol_sha256": self.protocol_sha256,
            "rcl_control": self.control.to_dict(),
            "rcl_control_sha256": self.control.control_sha256,
        }

    @property
    def bridge_sha256(self) -> str:
        return _semantic_sha256(self._bridge_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._bridge_dict(), "bridge_sha256": self.bridge_sha256}


@dataclass(frozen=True)
class OrdinarySeedResult:
    """Sealed ordinary T1 metrics for one active-learning seed."""

    dataset_id: str
    arm_id: str
    method_id: str
    selector_id: str
    active_learning_seed: int
    rcl_seed: int
    budget: int
    evaluation_scope: str
    evaluation_case_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    top135: float
    query_plan_sha256: str
    bridge_sha256: str
    runner_query_plan_sha256: str
    runner_result_sha256: str
    code_sha256: str
    data_sha256: str
    representation_sha256: str
    protocol_sha256: str
    rcl_control_sha256: str

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("ordinary result active-learning seed drifted")
        if self.rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError("ordinary result RCL seed must be 42")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("ordinary result budget must be 30")
        if self.evaluation_scope != T1_ONLY_SCOPE:
            raise ValueError("ordinary result must be T1-only")
        if (
            isinstance(self.evaluation_case_count, bool)
            or not isinstance(self.evaluation_case_count, int)
            or self.evaluation_case_count < 1
        ):
            raise ValueError("ordinary T1 metric denominator must be positive")
        hits = (self.hit_at_1, self.hit_at_3, self.hit_at_5)
        if (
            not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in hits)
            or not self.hit_at_1 <= self.hit_at_3 <= self.hit_at_5
        ):
            raise ValueError("ordinary T1 metrics must be finite, bounded, and monotonic")
        expected_top135 = sum(hits) / 3.0
        if not math.isfinite(self.top135) or not math.isclose(
            self.top135,
            expected_top135,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("ordinary TOP135 must equal the arithmetic Hit@1/3/5 mean")
        for value, name in (
            (self.query_plan_sha256, "source query plan SHA-256"),
            (self.bridge_sha256, "bridge SHA-256"),
            (self.runner_query_plan_sha256, "runner query plan SHA-256"),
            (self.runner_result_sha256, "runner result SHA-256"),
            (self.code_sha256, "code SHA-256"),
            (self.data_sha256, "data SHA-256"),
            (self.representation_sha256, "representation SHA-256"),
            (self.protocol_sha256, "protocol SHA-256"),
            (self.rcl_control_sha256, "RCL control SHA-256"),
        ):
            _require_sha256(value, name)

    def _result_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-ordinary-seed-result-v1",
            "dataset_id": self.dataset_id,
            "arm_id": self.arm_id,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "active_learning_seed": self.active_learning_seed,
            "rcl_seed": self.rcl_seed,
            "budget": self.budget,
            "evaluation_scope": self.evaluation_scope,
            "evaluation_case_count": self.evaluation_case_count,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "top135": self.top135,
            "query_plan_sha256": self.query_plan_sha256,
            "bridge_sha256": self.bridge_sha256,
            "runner_query_plan_sha256": self.runner_query_plan_sha256,
            "runner_result_sha256": self.runner_result_sha256,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "representation_sha256": self.representation_sha256,
            "protocol_sha256": self.protocol_sha256,
            "rcl_control_sha256": self.rcl_control_sha256,
        }

    @property
    def result_sha256(self) -> str:
        return _semantic_sha256(self._result_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._result_dict(), "result_sha256": self.result_sha256}


def _strict_result_metric(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(
            f"strict-LOFO metrics field {name} must be finite and in [0, 1]"
        )
    return float(value)


@dataclass(frozen=True)
class StrictLofoSeedResult:
    """Typed and sealed metrics plus population evidence for one LOFO seed."""

    unit_id: str
    dataset_id: str
    held_out_fault_type: str
    method_id: str
    selector_id: str
    active_learning_seed: int
    rcl_seed: int
    budget: int
    evaluation_scope: str
    evaluation_case_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    top135: float
    selected_case_ids: tuple[str, ...]
    training_window_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    unused_outer_test_case_ids: tuple[str, ...]
    fold_sha256: str
    query_plan_sha256: str
    bridge_sha256: str
    runner_query_plan_sha256: str
    runner_result_sha256: str
    code_sha256: str
    data_sha256: str
    representation_sha256: str
    protocol_sha256: str
    rcl_control_sha256: str
    selected_population_sha256: str
    training_population_sha256: str
    test_population_sha256: str
    unused_outer_test_population_sha256: str

    def __post_init__(self) -> None:
        DatasetId(self.dataset_id)
        if not self.unit_id or not self.held_out_fault_type:
            raise ValueError("strict-LOFO result identity must be non-empty")
        if self.active_learning_seed not in ACTIVE_LEARNING_SEEDS:
            raise ValueError("strict-LOFO result active-learning seed drifted")
        if self.rcl_seed != DOWNSTREAM_RCL_SEED:
            raise ValueError("strict-LOFO result RCL seed must be 42")
        if self.budget != ANNOTATION_BUDGET:
            raise ValueError("strict-LOFO result budget must be 30")
        if self.evaluation_scope != STRICT_LOFO_SCOPE:
            raise ValueError("strict-LOFO result evaluation scope drifted")
        if (
            type(self.evaluation_case_count) is not int
            or self.evaluation_case_count <= 0
            or self.evaluation_case_count != len(self.test_case_ids)
            or not self.test_case_ids
        ):
            raise ValueError("strict-LOFO metric denominator must equal the test population")
        populations = (
            self.selected_case_ids,
            self.training_window_ids,
            self.test_case_ids,
            self.unused_outer_test_case_ids,
        )
        if any(
            type(value) is not str or not value or value != value.strip()
            for values in populations
            for value in values
        ):
            raise ValueError(
                "strict-LOFO result population IDs must be canonical strings"
            )
        if any(len(values) != len(set(values)) for values in populations):
            raise ValueError("strict-LOFO result populations must be unique")
        if len(self.selected_case_ids) != self.budget or not set(
            self.selected_case_ids
        ).issubset(self.training_window_ids):
            raise ValueError("strict-LOFO selected population drifted")
        if (
            set(self.training_window_ids).intersection(self.test_case_ids)
            or set(self.training_window_ids).intersection(
                self.unused_outer_test_case_ids
            )
            or set(self.test_case_ids).intersection(
                self.unused_outer_test_case_ids
            )
        ):
            raise ValueError("strict-LOFO training/test/unused populations overlap")
        hits = tuple(
            _strict_result_metric(value, name)
            for name, value in (
                ("hit_at_1", self.hit_at_1),
                ("hit_at_3", self.hit_at_3),
                ("hit_at_5", self.hit_at_5),
            )
        )
        top135 = _strict_result_metric(self.top135, "top135")
        if not hits[0] <= hits[1] <= hits[2]:
            raise ValueError("strict-LOFO metrics must be monotonic")
        if not math.isclose(
            top135,
            sum(hits) / 3.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("strict-LOFO TOP135 must equal the Hit@1/3/5 mean")
        hash_fields = (
            (self.fold_sha256, "fold SHA-256"),
            (self.query_plan_sha256, "source query plan SHA-256"),
            (self.bridge_sha256, "bridge SHA-256"),
            (self.runner_query_plan_sha256, "runner query plan SHA-256"),
            (self.runner_result_sha256, "runner result SHA-256"),
            (self.code_sha256, "code SHA-256"),
            (self.data_sha256, "data SHA-256"),
            (self.representation_sha256, "representation SHA-256"),
            (self.protocol_sha256, "protocol SHA-256"),
            (self.rcl_control_sha256, "RCL control SHA-256"),
            (self.selected_population_sha256, "selected population SHA-256"),
            (self.training_population_sha256, "training population SHA-256"),
            (self.test_population_sha256, "test population SHA-256"),
            (
                self.unused_outer_test_population_sha256,
                "unused outer-test population SHA-256",
            ),
        )
        for value, name in hash_fields:
            _require_sha256(value, name)
        expected_populations = (
            (self.selected_case_ids, self.selected_population_sha256, "selected"),
            (self.training_window_ids, self.training_population_sha256, "training"),
            (self.test_case_ids, self.test_population_sha256, "test"),
            (
                self.unused_outer_test_case_ids,
                self.unused_outer_test_population_sha256,
                "unused outer-test",
            ),
        )
        for values, declared, name in expected_populations:
            if _population_sha256(values) != declared:
                raise ValueError(f"strict-LOFO {name} population hash drifted")

    def _result_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-strict-lofo-seed-result-v1",
            "unit_id": self.unit_id,
            "dataset_id": self.dataset_id,
            "held_out_fault_type": self.held_out_fault_type,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "active_learning_seed": self.active_learning_seed,
            "rcl_seed": self.rcl_seed,
            "budget": self.budget,
            "evaluation_scope": self.evaluation_scope,
            "evaluation_case_count": self.evaluation_case_count,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "top135": self.top135,
            "selected_case_ids": list(self.selected_case_ids),
            "training_window_ids": list(self.training_window_ids),
            "test_case_ids": list(self.test_case_ids),
            "unused_outer_test_case_ids": list(self.unused_outer_test_case_ids),
            "fold_sha256": self.fold_sha256,
            "query_plan_sha256": self.query_plan_sha256,
            "bridge_sha256": self.bridge_sha256,
            "runner_query_plan_sha256": self.runner_query_plan_sha256,
            "runner_result_sha256": self.runner_result_sha256,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "representation_sha256": self.representation_sha256,
            "protocol_sha256": self.protocol_sha256,
            "rcl_control_sha256": self.rcl_control_sha256,
            "selected_population_sha256": self.selected_population_sha256,
            "training_population_sha256": self.training_population_sha256,
            "test_population_sha256": self.test_population_sha256,
            "unused_outer_test_population_sha256": (
                self.unused_outer_test_population_sha256
            ),
        }

    @property
    def result_sha256(self) -> str:
        return _semantic_sha256(self._result_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._result_dict(), "result_sha256": self.result_sha256}


@dataclass(frozen=True)
class OrdinaryThreeSeedAggregate:
    """Exact arithmetic aggregate over active-learning seeds 41/42/43."""

    dataset_id: str
    arm_id: str
    method_id: str
    selector_id: str
    active_learning_seeds: tuple[int, ...]
    result_sha256_by_seed: tuple[tuple[int, str], ...]
    top135_by_seed: tuple[tuple[int, float], ...]
    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float
    mean_top135: float
    evaluation_case_count: int
    rcl_seed: int
    budget: int

    def _aggregate_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-ordinary-3seed-mean-v1",
            "dataset_id": self.dataset_id,
            "arm_id": self.arm_id,
            "method_id": self.method_id,
            "selector_id": self.selector_id,
            "active_learning_seeds": list(self.active_learning_seeds),
            "result_sha256_by_seed": dict(self.result_sha256_by_seed),
            "top135_by_seed": dict(self.top135_by_seed),
            "mean_hit_at_1": self.mean_hit_at_1,
            "mean_hit_at_3": self.mean_hit_at_3,
            "mean_hit_at_5": self.mean_hit_at_5,
            "mean_top135": self.mean_top135,
            "evaluation_case_count": self.evaluation_case_count,
            "rcl_seed": self.rcl_seed,
            "budget": self.budget,
        }

    @property
    def aggregate_sha256(self) -> str:
        return _semantic_sha256(self._aggregate_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._aggregate_dict(),
            "aggregate_sha256": self.aggregate_sha256,
        }


@dataclass(frozen=True)
class DatasetOrdinaryStrategySelection:
    """Dataset-specific best ordinary strategy and snapshot-gate decision."""

    dataset_id: str
    method_id: str
    selected_arm_id: str
    selected_selector_id: str
    mean_top135: float
    threshold: float
    margin: float
    passed: bool
    tied_best_selector_ids: tuple[str, ...]
    aggregate_sha256_by_selector: tuple[tuple[str, str], ...]

    def _selection_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-dataset-best-ordinary-v1",
            "dataset_id": self.dataset_id,
            "method_id": self.method_id,
            "selected_arm_id": self.selected_arm_id,
            "selected_selector_id": self.selected_selector_id,
            "mean_top135": self.mean_top135,
            "threshold": self.threshold,
            "margin": self.margin,
            "passed": self.passed,
            "tied_best_selector_ids": list(self.tied_best_selector_ids),
            "aggregate_sha256_by_selector": dict(
                self.aggregate_sha256_by_selector
            ),
        }

    @property
    def selection_sha256(self) -> str:
        return _semantic_sha256(self._selection_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._selection_dict(),
            "selection_sha256": self.selection_sha256,
        }


@dataclass(frozen=True)
class MethodOrdinaryQualification:
    """Two-dataset ordinary snapshot qualification for one method."""

    method_id: str
    qualified: bool
    failure_reasons: tuple[str, ...]
    selection_sha256_by_dataset: tuple[tuple[str, str], ...]

    def _qualification_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "combined-active-learning-ordinary-qualification-v1",
            "method_id": self.method_id,
            "qualified": self.qualified,
            "failure_reasons": list(self.failure_reasons),
            "selection_sha256_by_dataset": dict(
                self.selection_sha256_by_dataset
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


def _query_plan_identity(
    query_plan: ClusterQueryPlan | IndependentQueryPlan,
) -> tuple[str, str, str, tuple[str, ...], str, str, str]:
    if type(query_plan) is ClusterQueryPlan:
        query_plan.__post_init__()
        return (
            "cluster",
            query_plan.clusterer_id,
            query_plan.selector_id,
            query_plan.selected_case_ids,
            query_plan.plan_sha256,
            _semantic_sha256(query_plan.to_dict()),
            query_plan.representation_matrix_sha256,
        )
    if type(query_plan) is IndependentQueryPlan:
        query_plan.__post_init__()
        return (
            "independent",
            query_plan.method_id,
            query_plan.selector_id,
            query_plan.selected_case_ids,
            query_plan.plan_sha256,
            _semantic_sha256(query_plan.to_dict()),
            query_plan.representation_dependency_sha256,
        )
    raise ValueError("RCL bridge requires a frozen rich query plan")


def build_budget30_rcl_bridge(
    *,
    unit: OrdinaryUnit,
    query_plan: ClusterQueryPlan | IndependentQueryPlan,
    control: FrozenRclControl,
    output_root: str,
) -> Budget30RclBridge:
    """Bind one ordinary unit to an immutable rich budget-30 query plan."""

    if type(unit) is not OrdinaryUnit:
        raise ValueError("RCL bridge requires a typed ordinary unit")
    if type(control) is not FrozenRclControl:
        raise ValueError("RCL bridge requires frozen RCL controls")
    (
        plan_type,
        method_id,
        selector_id,
        selected_case_ids,
        plan_sha256,
        reference_sha256,
        representation_dependency_sha256,
    ) = _query_plan_identity(query_plan)
    if unit.hashes.query_plan_sha256 != plan_sha256:
        raise ValueError("ordinary unit query-plan hash drifted")
    if unit.hashes.representation_sha256 != representation_dependency_sha256:
        raise ValueError("ordinary unit representation dependency drifted")
    if unit.hashes.rcl_control_sha256 != control.control_sha256:
        raise ValueError("ordinary unit RCL control hash drifted")
    expected_arm_id = (
        f"{unit.representation_id}.{method_id}.{selector_id}"
        if unit.representation_id is not None
        else f"{method_id}.{selector_id}"
    )
    if (
        unit.dataset_id.value != query_plan.dataset_id
        or unit.method_id != method_id
        or unit.selector_id != selector_id
        or unit.arm_id != expected_arm_id
    ):
        raise ValueError("ordinary unit method or selector identity drifted")
    if (
        unit.active_learning_seed != query_plan.active_learning_seed
        or unit.budget != query_plan.budget
        or unit.budget != ANNOTATION_BUDGET
        or unit.rcl_seed != DOWNSTREAM_RCL_SEED
        or unit.evaluation_mode != ORDINARY_MODE
    ):
        raise ValueError("ordinary unit query-plan protocol drifted")
    root = _require_posix_root(output_root)
    unit_output_root = str(root / unit.unit_id)
    return Budget30RclBridge(
        unit_id=unit.unit_id,
        dataset_id=unit.dataset_id.value,
        arm_id=unit.arm_id,
        method_id=unit.method_id,
        source_selector_id=unit.selector_id,
        active_learning_seed=unit.active_learning_seed,
        budget=unit.budget,
        selected_case_ids=tuple(selected_case_ids),
        query_plan_type=plan_type,
        query_plan_sha256=plan_sha256,
        query_plan_reference_sha256=reference_sha256,
        output_root=unit_output_root,
        code_sha256=unit.hashes.code_sha256,
        data_sha256=unit.hashes.data_sha256,
        representation_sha256=unit.hashes.representation_sha256,
        protocol_sha256=unit.hashes.protocol_sha256,
        control=control,
    )


def build_strict_lofo_rcl_bridge(
    *,
    unit: StrictLofoUnit,
    fold: StrictLofoFold,
    query_plan: ClusterQueryPlan | IndependentQueryPlan,
    control: FrozenRclControl,
    training_window_ids: Sequence[str],
    output_root: str,
) -> StrictLofoRclBridge:
    """Bind a strict unit, sealed fold, and rich query plan fail closed."""

    from rcl_study.combined_active_learning_lofo import StrictLofoFold

    if type(unit) is not StrictLofoUnit:
        raise ValueError("strict-LOFO bridge requires a typed strict unit")
    if type(fold) is not StrictLofoFold:
        raise ValueError("strict-LOFO bridge requires a typed sealed fold")
    if type(control) is not FrozenRclControl:
        raise ValueError("strict-LOFO bridge requires frozen RCL controls")
    (
        plan_type,
        method_id,
        selector_id,
        selected_case_ids,
        plan_sha256,
        reference_sha256,
        representation_dependency_sha256,
    ) = _query_plan_identity(query_plan)
    if unit.hashes.query_plan_sha256 != plan_sha256:
        raise ValueError("strict-LOFO unit query-plan hash drifted")
    if unit.hashes.representation_sha256 != representation_dependency_sha256:
        raise ValueError("strict-LOFO unit representation dependency drifted")
    if unit.hashes.rcl_control_sha256 != control.control_sha256:
        raise ValueError("strict-LOFO unit RCL control hash drifted")
    if unit.fold_sha256 != fold.fold_sha256:
        raise ValueError("strict-LOFO unit fold seal drifted")
    if (
        unit.dataset_id.value != fold.dataset_id
        or unit.dataset_id.value != query_plan.dataset_id
        or unit.held_out_fault_type != fold.held_out_fault_type
        or unit.method_id != method_id
        or unit.selector_id != selector_id
    ):
        raise ValueError("strict-LOFO unit method, selector, or fold identity drifted")
    if (
        unit.active_learning_seed != query_plan.active_learning_seed
        or unit.budget != query_plan.budget
        or unit.budget != ANNOTATION_BUDGET
        or unit.rcl_seed != DOWNSTREAM_RCL_SEED
    ):
        raise ValueError("strict-LOFO unit query-plan protocol drifted")
    selected = tuple(selected_case_ids)
    if not set(selected).issubset(fold.candidate_case_ids) or set(
        selected
    ).intersection((*fold.test_case_ids, *fold.unused_outer_test_case_ids)):
        raise ValueError("query plan violates the strict candidate population")
    root = _require_posix_root(output_root)
    return StrictLofoRclBridge(
        unit_id=unit.unit_id,
        dataset_id=unit.dataset_id.value,
        held_out_fault_type=unit.held_out_fault_type,
        method_id=unit.method_id,
        source_selector_id=unit.selector_id,
        active_learning_seed=unit.active_learning_seed,
        budget=unit.budget,
        selected_case_ids=selected,
        training_window_ids=tuple(training_window_ids),
        query_plan_type=plan_type,
        query_plan_sha256=plan_sha256,
        query_plan_reference_sha256=reference_sha256,
        fold=fold,
        output_root=str(root / unit.unit_id),
        code_sha256=unit.hashes.code_sha256,
        data_sha256=unit.hashes.data_sha256,
        representation_sha256=unit.hashes.representation_sha256,
        protocol_sha256=unit.hashes.protocol_sha256,
        control=control,
    )


def _validate_frozen_rcl_control(
    payload: Mapping[str, Any],
    *,
    declared_sha256: str,
) -> FrozenRclControl:
    if not isinstance(payload, Mapping):
        raise ValueError("RCL control must be an object")
    checked = deepcopy(dict(payload))
    if checked.pop("schema_version", None) != (
        "combined-active-learning-rcl-control-v1"
    ):
        raise ValueError("RCL control schema drifted")
    expected_fields = {
        "split_manifest_sha256",
        "model_config_sha256",
        "runner_sha256",
        "split_seed",
        "model_seed",
        "initialization_seed",
        "training_seed",
        "outer_test_ratio",
        "inner_validation_ratio",
        "normal_policy",
        "fresh_training_required",
        "state_inputs",
    }
    if set(checked) != expected_fields:
        raise ValueError("RCL control field allowlist drifted")
    state_inputs = checked.pop("state_inputs")
    if state_inputs != {
        "model_state": None,
        "checkpoint": None,
        "optimizer": None,
        "resume": None,
        "warm_start": None,
    }:
        raise ValueError("RCL control state inputs drifted")
    try:
        control = FrozenRclControl(**checked)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RCL control is invalid: {exc}") from exc
    if control.control_sha256 != declared_sha256:
        raise ValueError("RCL control seal drifted")
    return control


def validate_budget30_rcl_bridge(
    payload: Mapping[str, Any],
) -> Budget30RclBridge:
    """Rebuild one serialized bridge only after every nested seal validates."""

    if not isinstance(payload, Mapping):
        raise ValueError("budget-30 RCL bridge must be an object")
    checked = deepcopy(dict(payload))
    declared = checked.pop("bridge_sha256", None)
    if declared != _semantic_sha256(checked):
        raise ValueError("budget-30 RCL bridge seal drifted")
    if checked.pop("schema_version", None) != (
        "combined-active-learning-budget30-rcl-bridge-v1"
    ):
        raise ValueError("budget-30 RCL bridge schema drifted")
    expected_fields = {
        "unit_id",
        "dataset_id",
        "arm_id",
        "method_id",
        "source_selector_id",
        "active_learning_seed",
        "budget",
        "selected_case_ids",
        "query_plan_type",
        "query_plan_sha256",
        "query_plan_reference_sha256",
        "output_root",
        "code_sha256",
        "data_sha256",
        "representation_sha256",
        "protocol_sha256",
        "rcl_control",
        "rcl_control_sha256",
    }
    if set(checked) != expected_fields:
        raise ValueError("budget-30 RCL bridge field allowlist drifted")
    control_sha256 = checked.pop("rcl_control_sha256")
    _require_sha256(control_sha256, "RCL control SHA-256")
    control = _validate_frozen_rcl_control(
        checked.pop("rcl_control"),
        declared_sha256=control_sha256,
    )
    selected_case_ids = checked.get("selected_case_ids")
    if not isinstance(selected_case_ids, list) or any(
        not isinstance(case_id, str) for case_id in selected_case_ids
    ):
        raise ValueError("budget-30 RCL bridge case IDs must be a list of strings")
    checked["selected_case_ids"] = tuple(selected_case_ids)
    try:
        bridge = Budget30RclBridge(control=control, **checked)
    except TypeError as exc:
        raise ValueError("budget-30 RCL bridge fields are invalid") from exc
    if bridge.bridge_sha256 != declared:
        raise ValueError("budget-30 RCL bridge did not round trip")
    return bridge


def validate_strict_lofo_rcl_bridge(
    payload: Mapping[str, Any],
) -> StrictLofoRclBridge:
    """Rebuild a serialized strict bridge after nested fold/control validation."""

    from rcl_study.combined_active_learning_lofo import validate_strict_lofo_fold

    if not isinstance(payload, Mapping):
        raise ValueError("strict-LOFO RCL bridge must be an object")
    checked = deepcopy(dict(payload))
    declared = checked.pop("bridge_sha256", None)
    if declared != _semantic_sha256(checked):
        raise ValueError("strict-LOFO RCL bridge seal drifted")
    if checked.pop("schema_version", None) != (
        "combined-active-learning-strict-lofo-rcl-bridge-v1"
    ):
        raise ValueError("strict-LOFO RCL bridge schema drifted")
    expected_fields = {
        "unit_id",
        "dataset_id",
        "held_out_fault_type",
        "method_id",
        "source_selector_id",
        "active_learning_seed",
        "budget",
        "selected_case_ids",
        "training_window_ids",
        "query_plan_type",
        "query_plan_sha256",
        "query_plan_reference_sha256",
        "strict_lofo_fold",
        "fold_sha256",
        "output_root",
        "code_sha256",
        "data_sha256",
        "representation_sha256",
        "protocol_sha256",
        "rcl_control",
        "rcl_control_sha256",
    }
    if set(checked) != expected_fields:
        raise ValueError("strict-LOFO RCL bridge field allowlist drifted")
    fold = validate_strict_lofo_fold(checked.pop("strict_lofo_fold"))
    if checked.pop("fold_sha256") != fold.fold_sha256:
        raise ValueError("strict-LOFO bridge fold seal drifted")
    control_sha256 = checked.pop("rcl_control_sha256")
    _require_sha256(control_sha256, "RCL control SHA-256")
    control = _validate_frozen_rcl_control(
        checked.pop("rcl_control"),
        declared_sha256=control_sha256,
    )
    for name in ("selected_case_ids", "training_window_ids"):
        values = checked.get(name)
        if not isinstance(values, list) or any(
            not isinstance(case_id, str) for case_id in values
        ):
            raise ValueError(
                f"strict-LOFO bridge {name} must be a list of strings"
            )
        checked[name] = tuple(values)
    try:
        bridge = StrictLofoRclBridge(
            fold=fold,
            control=control,
            **checked,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strict-LOFO RCL bridge fields are invalid: {exc}") from exc
    if bridge.bridge_sha256 != declared:
        raise ValueError("strict-LOFO RCL bridge did not round trip")
    return bridge


def project_authority_query_only_unit(
    bridge: Budget30RclBridge,
) -> dict[str, Any]:
    """Project the allowlisted payload accepted by the authority RCL runner."""

    if type(bridge) is not Budget30RclBridge:
        raise ValueError("authority projection requires a budget-30 RCL bridge")
    selector_config = {
        "schema_version": "combined-active-learning-fixed-budget-selector-v1",
        "source_plan_type": bridge.query_plan_type,
        "source_method_id": bridge.method_id,
        "source_selector_id": bridge.source_selector_id,
        "selected_case_ids": list(bridge.selected_case_ids),
        "query_plan_sha256": bridge.query_plan_sha256,
        "query_plan_reference_sha256": bridge.query_plan_reference_sha256,
    }
    control = bridge.control
    return {
        "unit_id": bridge.unit_id,
        "output_root": bridge.output_root,
        "canonical_dataset_id": bridge.dataset_id,
        "method_id": bridge.arm_id,
        "selector_id": FIXED_QUERY_SELECTOR_ID,
        "selector_config": selector_config,
        "selector_config_sha256": _semantic_sha256(selector_config),
        "active_learning_seed": bridge.active_learning_seed,
        "split_seed": control.split_seed,
        "model_seed": control.model_seed,
        "initialization_seed": control.initialization_seed,
        "training_seed": control.training_seed,
        "evaluation_scope": T1_ONLY_SCOPE,
        "budget": bridge.budget,
        "selected_case_ids": list(bridge.selected_case_ids),
        "query_plan_sha256": bridge.query_plan_sha256,
        "query_plan_reference_sha256": bridge.query_plan_reference_sha256,
        "split_manifest_sha256": control.split_manifest_sha256,
        "model_config_sha256": control.model_config_sha256,
        "runner_sha256": control.runner_sha256,
        "outer_test_ratio": control.outer_test_ratio,
        "inner_val_ratio": control.inner_validation_ratio,
        "normal_policy": control.normal_policy,
        "fresh_training_required": control.fresh_training_required,
        "state_inputs": control.to_dict()["state_inputs"],
        "rcl_control_sha256": control.control_sha256,
        "bridge_sha256": bridge.bridge_sha256,
    }


def project_strict_lofo_query_only_unit(
    bridge: StrictLofoRclBridge,
) -> dict[str, Any]:
    """Project one strict fold to the dedicated real-backend allowlist."""

    if type(bridge) is not StrictLofoRclBridge:
        raise ValueError("strict backend projection requires a strict-LOFO bridge")
    fold = bridge.fold
    selector_config = {
        "schema_version": "combined-active-learning-fixed-budget-selector-v1",
        "source_plan_type": bridge.query_plan_type,
        "source_method_id": bridge.method_id,
        "source_selector_id": bridge.source_selector_id,
        "selected_case_ids": list(bridge.selected_case_ids),
        "query_plan_sha256": bridge.query_plan_sha256,
        "query_plan_reference_sha256": bridge.query_plan_reference_sha256,
    }
    control = bridge.control
    return {
        "unit_id": bridge.unit_id,
        "output_root": bridge.output_root,
        "canonical_dataset_id": bridge.dataset_id,
        "held_out_fault_type": bridge.held_out_fault_type,
        "method_id": bridge.method_id,
        "source_selector_id": bridge.source_selector_id,
        "selector_id": FIXED_QUERY_SELECTOR_ID,
        "selector_config": selector_config,
        "selector_config_sha256": _semantic_sha256(selector_config),
        "active_learning_seed": bridge.active_learning_seed,
        "split_seed": control.split_seed,
        "model_seed": control.model_seed,
        "initialization_seed": control.initialization_seed,
        "training_seed": control.training_seed,
        "evaluation_scope": STRICT_LOFO_SCOPE,
        "budget": bridge.budget,
        "selected_case_ids": list(bridge.selected_case_ids),
        "training_window_ids": list(bridge.training_window_ids),
        "training_population_sha256": _population_sha256(
            bridge.training_window_ids
        ),
        "query_plan_sha256": bridge.query_plan_sha256,
        "query_plan_reference_sha256": bridge.query_plan_reference_sha256,
        "fold_sha256": fold.fold_sha256,
        "strict_lofo_fold": fold.to_dict(),
        "original_inventory_sha256": fold.original_inventory_sha256,
        "fault_type_normalization_sha256": (
            fold.fault_type_normalization_sha256
        ),
        "candidate_case_ids": list(fold.candidate_case_ids),
        "held_out_outer_train_case_ids": list(
            fold.held_out_outer_train_case_ids
        ),
        "held_out_outer_test_case_ids": list(fold.held_out_outer_test_case_ids),
        "test_case_ids": list(fold.test_case_ids),
        "unused_outer_test_case_ids": list(fold.unused_outer_test_case_ids),
        "candidate_population_sha256": _population_sha256(
            fold.candidate_case_ids
        ),
        "test_population_sha256": _population_sha256(fold.test_case_ids),
        "unused_outer_test_population_sha256": _population_sha256(
            fold.unused_outer_test_case_ids
        ),
        "split_manifest_sha256": control.split_manifest_sha256,
        "model_config_sha256": control.model_config_sha256,
        "runner_sha256": control.runner_sha256,
        "outer_test_ratio": control.outer_test_ratio,
        "inner_val_ratio": control.inner_validation_ratio,
        "normal_policy": control.normal_policy,
        "fresh_training_required": control.fresh_training_required,
        "state_inputs": control.to_dict()["state_inputs"],
        "rcl_control_sha256": control.control_sha256,
        "bridge_sha256": bridge.bridge_sha256,
    }


def validate_strict_lofo_backend_payload(
    payload: Mapping[str, Any],
    *,
    bridge: StrictLofoRclBridge,
) -> dict[str, Any]:
    """Require an exact strict projection before dispatching the real backend."""

    if not isinstance(payload, Mapping):
        raise ValueError("strict-LOFO backend payload must be an object")
    expected = project_strict_lofo_query_only_unit(bridge)
    candidate = deepcopy(dict(payload))
    if set(candidate) != set(expected) or candidate != expected:
        raise ValueError("strict-LOFO backend payload identity drifted")
    if (
        candidate["evaluation_scope"] != STRICT_LOFO_SCOPE
        or candidate["fresh_training_required"] is not True
        or candidate["state_inputs"]
        != {
            "model_state": None,
            "checkpoint": None,
            "optimizer": None,
            "resume": None,
            "warm_start": None,
        }
    ):
        raise ValueError("strict-LOFO backend must start from fresh training")
    return candidate


def validate_ordinary_t1_backend_payload(
    payload: Mapping[str, Any],
    *,
    bridge: Budget30RclBridge,
) -> dict[str, Any]:
    """Validate the exact T1-only payload before authority-runner dispatch."""

    if not isinstance(payload, Mapping):
        raise ValueError("ordinary backend payload must be an object")
    if payload.get("evaluation_scope") != T1_ONLY_SCOPE:
        raise ValueError("ordinary T2 planning is forbidden")
    expected = project_authority_query_only_unit(bridge)
    candidate = deepcopy(dict(payload))
    if set(candidate) != set(expected) or candidate != expected:
        raise ValueError("ordinary T1 backend payload identity drifted")
    forbidden = {
        "training_annotations",
        "fault_type",
        "root_cause_ids",
        "cluster_labels",
        "representation_matrix",
        "held_out_fault_type",
        "oracle_full",
    }
    if forbidden.intersection(candidate):
        raise ValueError("ordinary backend payload contains forbidden information")
    return candidate


def _validate_ordinary_t1_runner_result(
    result: Mapping[str, Any],
    *,
    bridge: Budget30RclBridge,
    backend_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("ordinary T1 runner result must be an object")
    checked = deepcopy(dict(result))
    metrics = checked.get("metrics")
    evaluation_status = checked.get("evaluation_status_by_view")
    paths = checked.get("view_result_paths")
    partitions = checked.get("partitions")
    evidence = checked.get("per_case_ranking_evidence_by_view")
    if (
        checked.get("evaluation_scope") != T1_ONLY_SCOPE
        or not isinstance(metrics, Mapping)
        or set(metrics) != {"T1", "T2"}
        or metrics.get("T2") != {"status": "not_requested"}
        or evaluation_status != {"T1": "completed", "T2": "not_requested"}
        or not isinstance(paths, Mapping)
        or set(paths) != {"T1"}
        or not isinstance(partitions, Mapping)
        or set(partitions) != {"T1"}
        or not isinstance(evidence, Mapping)
        or set(evidence) != {"T1"}
    ):
        raise ValueError("ordinary T2 output is forbidden")

    expected_identity = {
        "schema_version": (
            "rcl-query-active-deepening-current-method-unit-result-v1"
        ),
        "unit_id": bridge.unit_id,
        "canonical_dataset_id": bridge.dataset_id,
        "method_id": bridge.arm_id,
        "selector_id": FIXED_QUERY_SELECTOR_ID,
        "selector_config_sha256": backend_payload["selector_config_sha256"],
        "active_learning_seed": bridge.active_learning_seed,
        "training_seed": bridge.control.training_seed,
        "split_seed": bridge.control.split_seed,
        "evaluation_scope": T1_ONLY_SCOPE,
        "budget": bridge.budget,
    }
    for name, expected in expected_identity.items():
        if checked.get(name) != expected:
            raise ValueError(f"ordinary T1 runner identity drifted for {name}")
    expected_selected = list(bridge.selected_case_ids)
    for name in ("queried_case_ids", "selected_case_ids"):
        if checked.get(name) != expected_selected:
            raise ValueError("ordinary T1 runner selected-case identity drifted")
    selector_selected = checked.get("selected_case_ids_from_selector")
    if not isinstance(selector_selected, list) or any(
        not isinstance(case_id, str) for case_id in selector_selected
    ):
        raise ValueError("ordinary T1 runner selected-case identity drifted")
    dataset_prefix = f"{bridge.dataset_id}::"
    normalized_selector_ids = []
    for case_id in selector_selected:
        if case_id.startswith(dataset_prefix):
            normalized_selector_ids.append(case_id.removeprefix(dataset_prefix))
        elif "::" in case_id:
            raise ValueError("ordinary T1 runner selected-case identity drifted")
        else:
            normalized_selector_ids.append(case_id)
    if normalized_selector_ids != expected_selected:
        raise ValueError("ordinary T1 runner selected-case identity drifted")
    _require_sha256(
        checked.get("query_plan_sha256"),
        "runner training query plan SHA-256",
    )
    return checked


def run_ordinary_query_only_t1(
    *,
    bridge: Budget30RclBridge,
    run_root: str,
    runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one ordinary query-only T1 unit through the authority runner."""

    if type(bridge) is not Budget30RclBridge:
        raise ValueError("ordinary T1 execution requires a budget-30 RCL bridge")
    root = _require_posix_root(run_root)
    if PurePosixPath(bridge.output_root).parent != root:
        raise ValueError("ordinary T1 run root differs from the bridge output root")
    backend = validate_ordinary_t1_backend_payload(
        project_authority_query_only_unit(bridge),
        bridge=bridge,
    )
    if runner is None:
        from rcl_study.query_active_real_execution import (
            run_deepening_query_only_t1_t2_unit,
        )

        runner = run_deepening_query_only_t1_t2_unit
    raw_result = runner(deepcopy(backend), Path(run_root))
    return _validate_ordinary_t1_runner_result(
        raw_result,
        bridge=bridge,
        backend_payload=backend,
    )


def _require_strict_ranking_strings(
    value: Any,
    *,
    field_name: str,
) -> list[str]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(
            type(item) is not str or not item or item != item.strip()
            for item in value
        )
    ):
        raise ValueError(
            f"strict-LOFO ranking evidence {field_name} must contain canonical strings"
        )
    return list(value)


def _strict_lofo_metrics_from_ranking_evidence(
    evidence: Any,
    *,
    expected_case_ids: Sequence[str],
) -> dict[str, Any]:
    """Recompute count-authoritative strict metrics from ranking rows."""

    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise ValueError("strict-LOFO ranking evidence must be an ordered sequence")
    expected = [str(case_id) for case_id in expected_case_ids]
    by_case: dict[str, Mapping[str, Any]] = {}
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("strict-LOFO ranking evidence row must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_case:
            raise ValueError("strict-LOFO ranking evidence case IDs are invalid")
        by_case[case_id] = row
    if len(by_case) != len(expected) or set(by_case) != set(expected):
        raise ValueError(
            "strict-LOFO ranking evidence must exactly cover the test population"
        )
    hit_counts = {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0}
    for case_id in expected:
        row = by_case[case_id]
        targets = _require_strict_ranking_strings(
            row.get("targets"),
            field_name="targets",
        )
        ranking = _require_strict_ranking_strings(
            row.get("ranking"),
            field_name="ranking",
        )
        first_rank = first_target_rank(ranking, targets)
        for cutoff, name in ((1, "hit_at_1"), (3, "hit_at_3"), (5, "hit_at_5")):
            hit_counts[name] += int(bool(first_rank and first_rank <= cutoff))
    denominator = len(expected)
    return {
        **{name: count / denominator for name, count in hit_counts.items()},
        "denominator": denominator,
    }


def _validate_strict_lofo_runner_result(
    result: Mapping[str, Any],
    *,
    bridge: StrictLofoRclBridge,
    backend_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("strict-LOFO runner result must be an object")
    checked = deepcopy(dict(result))
    expected_fields = {
        "schema_version",
        "unit_id",
        "canonical_dataset_id",
        "held_out_fault_type",
        "method_id",
        "source_selector_id",
        "selector_id",
        "selector_config_sha256",
        "active_learning_seed",
        "training_seed",
        "split_seed",
        "evaluation_scope",
        "budget",
        "runner_query_plan_sha256",
        "selected_case_ids",
        "selected_case_ids_from_selector",
        "training_window_ids",
        "test_case_ids",
        "unused_outer_test_case_ids",
        "selected_population_sha256",
        "training_population_sha256",
        "test_population_sha256",
        "unused_outer_test_population_sha256",
        "partitions",
        "metrics",
        "per_case_ranking_evidence",
        "view_result_paths",
        "evaluation_status_by_view",
        "result_path",
        "run_root",
    }
    if set(checked) != expected_fields:
        raise ValueError("strict-LOFO runner result field allowlist drifted")
    expected_identity = {
        "schema_version": "rcl-query-active-strict-lofo-unit-result-v1",
        "unit_id": bridge.unit_id,
        "canonical_dataset_id": bridge.dataset_id,
        "held_out_fault_type": bridge.held_out_fault_type,
        "method_id": bridge.method_id,
        "source_selector_id": bridge.source_selector_id,
        "selector_id": FIXED_QUERY_SELECTOR_ID,
        "selector_config_sha256": backend_payload["selector_config_sha256"],
        "active_learning_seed": bridge.active_learning_seed,
        "training_seed": bridge.control.training_seed,
        "split_seed": bridge.control.split_seed,
        "evaluation_scope": STRICT_LOFO_SCOPE,
        "budget": bridge.budget,
    }
    for name, expected in expected_identity.items():
        if checked.get(name) != expected:
            raise ValueError(f"strict-LOFO runner identity drifted for {name}")
    selected = checked.get("selected_case_ids")
    training = checked.get("training_window_ids")
    test = checked.get("test_case_ids")
    unused = checked.get("unused_outer_test_case_ids")
    populations = {
        "selected": selected,
        "training": training,
        "test": test,
        "unused outer-test": unused,
    }
    for name, values in populations.items():
        if (
            not isinstance(values, list)
            or any(
                type(value) is not str
                or not value
                or value != value.strip()
                for value in values
            )
            or len(values) != len(set(values))
        ):
            raise ValueError(f"strict-LOFO runner {name} population is invalid")
    if selected != list(bridge.selected_case_ids):
        raise ValueError("strict-LOFO runner selected-case identity drifted")
    selector_selected = checked.get("selected_case_ids_from_selector")
    if not isinstance(selector_selected, list) or any(
        type(value) is not str or not value or value != value.strip()
        for value in selector_selected
    ):
        raise ValueError("strict-LOFO runner selector population is invalid")
    prefix = f"{bridge.dataset_id}::"
    normalized = [
        value.removeprefix(prefix) if value.startswith(prefix) else value
        for value in selector_selected
    ]
    if normalized != selected:
        raise ValueError("strict-LOFO runner selected-case identity drifted")
    fold = bridge.fold
    if training != backend_payload.get("training_window_ids"):
        raise ValueError(
            "strict-LOFO runner training population differs from signed authority"
        )
    if not set(fold.candidate_case_ids).issubset(training):
        raise ValueError("strict-LOFO runner training population lost candidates")
    if (
        set(training).intersection(fold.test_case_ids)
        or set(training).intersection(fold.unused_outer_test_case_ids)
    ):
        raise ValueError("strict-LOFO runner training population leaked")
    if test != list(fold.test_case_ids) or unused != list(
        fold.unused_outer_test_case_ids
    ):
        raise ValueError("strict-LOFO runner test/unused population drifted")
    expected_hashes = {
        "selected_population_sha256": _population_sha256(selected),
        "training_population_sha256": _population_sha256(training),
        "test_population_sha256": _population_sha256(test),
        "unused_outer_test_population_sha256": _population_sha256(unused),
    }
    for name, expected in expected_hashes.items():
        if checked.get(name) != expected:
            raise ValueError(f"strict-LOFO runner {name} drifted")
    _require_sha256(
        checked.get("runner_query_plan_sha256"),
        "strict-LOFO runner query plan SHA-256",
    )
    if checked.get("partitions") != {"strict_lofo": test}:
        raise ValueError("strict-LOFO runner partitions drifted")
    metrics = checked.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"strict_lofo"}:
        raise ValueError("strict-LOFO runner metrics drifted")
    rebuilt_metrics = _strict_lofo_metrics_from_ranking_evidence(
        checked.get("per_case_ranking_evidence"),
        expected_case_ids=test,
    )
    declared_metrics = metrics.get("strict_lofo")
    if not isinstance(declared_metrics, Mapping):
        raise ValueError("strict-LOFO runner metrics must be an object")
    if (
        isinstance(declared_metrics.get("denominator"), bool)
        or declared_metrics.get("denominator")
        != rebuilt_metrics["denominator"]
    ):
        raise ValueError(
            "strict-LOFO ranking evidence denominator differs from metrics"
        )
    for name in ("hit_at_1", "hit_at_3", "hit_at_5"):
        value = declared_metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), rebuilt_metrics[name], rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ValueError(
                f"strict-LOFO ranking evidence differs from metrics for {name}"
            )
    if checked.get("evaluation_status_by_view") != {
        "strict_lofo": "completed"
    }:
        raise ValueError("strict-LOFO runner evaluation status drifted")
    paths = checked.get("view_result_paths")
    if not isinstance(paths, Mapping) or set(paths) != {"strict_lofo"}:
        raise ValueError("strict-LOFO runner view paths drifted")
    return checked


def run_strict_lofo_query_only(
    *,
    bridge: StrictLofoRclBridge,
    run_root: str,
    runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one strict fold through the dedicated real backend."""

    if type(bridge) is not StrictLofoRclBridge:
        raise ValueError("strict-LOFO execution requires a strict bridge")
    root = _require_posix_root(run_root)
    if PurePosixPath(bridge.output_root).parent != root:
        raise ValueError("strict-LOFO run root differs from the bridge output root")
    backend = validate_strict_lofo_backend_payload(
        project_strict_lofo_query_only_unit(bridge),
        bridge=bridge,
    )
    if runner is None:
        from rcl_study.query_active_real_execution import (
            run_strict_lofo_query_only_unit,
        )

        runner = run_strict_lofo_query_only_unit
    raw_result = runner(deepcopy(backend), Path(run_root))
    return _validate_strict_lofo_runner_result(
        raw_result,
        bridge=bridge,
        backend_payload=backend,
    )


def _t1_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"ordinary T1 metric {name} must be finite and in [0, 1]")
    return float(value)


def build_strict_lofo_seed_result(
    *,
    bridge: StrictLofoRclBridge,
    runner_result: Mapping[str, Any],
) -> StrictLofoSeedResult:
    """Extract and seal one held-out-only strict-LOFO result."""

    backend = project_strict_lofo_query_only_unit(bridge)
    checked = _validate_strict_lofo_runner_result(
        runner_result,
        bridge=bridge,
        backend_payload=backend,
    )
    metrics = checked["metrics"]["strict_lofo"]
    if not isinstance(metrics, Mapping):
        raise ValueError("strict-LOFO metrics must be an object")
    hit_at_1 = _t1_metric(metrics, "hit_at_1")
    hit_at_3 = _t1_metric(metrics, "hit_at_3")
    hit_at_5 = _t1_metric(metrics, "hit_at_5")
    denominator = metrics.get("denominator")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator != len(bridge.fold.test_case_ids)
    ):
        raise ValueError("strict-LOFO metric denominator drifted")
    return StrictLofoSeedResult(
        unit_id=bridge.unit_id,
        dataset_id=bridge.dataset_id,
        held_out_fault_type=bridge.held_out_fault_type,
        method_id=bridge.method_id,
        selector_id=bridge.source_selector_id,
        active_learning_seed=bridge.active_learning_seed,
        rcl_seed=bridge.control.training_seed,
        budget=bridge.budget,
        evaluation_scope=STRICT_LOFO_SCOPE,
        evaluation_case_count=denominator,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        top135=(hit_at_1 + hit_at_3 + hit_at_5) / 3.0,
        selected_case_ids=tuple(checked["selected_case_ids"]),
        training_window_ids=tuple(checked["training_window_ids"]),
        test_case_ids=tuple(checked["test_case_ids"]),
        unused_outer_test_case_ids=tuple(
            checked["unused_outer_test_case_ids"]
        ),
        fold_sha256=bridge.fold.fold_sha256,
        query_plan_sha256=bridge.query_plan_sha256,
        bridge_sha256=bridge.bridge_sha256,
        runner_query_plan_sha256=checked["runner_query_plan_sha256"],
        runner_result_sha256=_semantic_sha256(checked),
        code_sha256=bridge.code_sha256,
        data_sha256=bridge.data_sha256,
        representation_sha256=bridge.representation_sha256,
        protocol_sha256=bridge.protocol_sha256,
        rcl_control_sha256=bridge.control.control_sha256,
        selected_population_sha256=checked["selected_population_sha256"],
        training_population_sha256=checked["training_population_sha256"],
        test_population_sha256=checked["test_population_sha256"],
        unused_outer_test_population_sha256=checked[
            "unused_outer_test_population_sha256"
        ],
    )


def validate_strict_lofo_seed_result(
    payload: Mapping[str, Any],
) -> StrictLofoSeedResult:
    """Recompute a serialized strict result seal and all population hashes."""

    if not isinstance(payload, Mapping):
        raise ValueError("strict-LOFO seed result must be an object")
    evaluation_case_count = payload.get("evaluation_case_count")
    if type(evaluation_case_count) is not int or evaluation_case_count <= 0:
        raise ValueError(
            "strict-LOFO metric denominator must be a positive integer"
        )
    identity = deepcopy(dict(payload))
    declared = identity.pop("result_sha256", None)
    if declared != _semantic_sha256(identity):
        raise ValueError("strict-LOFO seed result seal drifted")
    if identity.pop("schema_version", None) != (
        "combined-active-learning-strict-lofo-seed-result-v1"
    ):
        raise ValueError("strict-LOFO seed result schema drifted")
    for name in (
        "selected_case_ids",
        "training_window_ids",
        "test_case_ids",
        "unused_outer_test_case_ids",
    ):
        values = identity.get(name)
        if not isinstance(values, list) or any(
            type(value) is not str or not value or value != value.strip()
            for value in values
        ):
            raise ValueError(
                f"strict-LOFO {name} must contain canonical strings"
            )
        identity[name] = tuple(values)
    try:
        return StrictLofoSeedResult(**identity)
    except TypeError as exc:
        raise ValueError("strict-LOFO seed result field allowlist drifted") from exc


def build_ordinary_seed_result(
    *,
    bridge: Budget30RclBridge,
    runner_result: Mapping[str, Any],
) -> OrdinarySeedResult:
    """Extract and seal Hit@1/3/5 plus recomputed TOP135 for one seed."""

    backend = project_authority_query_only_unit(bridge)
    checked = _validate_ordinary_t1_runner_result(
        runner_result,
        bridge=bridge,
        backend_payload=backend,
    )
    t1 = checked["metrics"]["T1"]
    if not isinstance(t1, Mapping):
        raise ValueError("ordinary T1 metrics must be an object")
    hit_at_1 = _t1_metric(t1, "hit_at_1")
    hit_at_3 = _t1_metric(t1, "hit_at_3")
    hit_at_5 = _t1_metric(t1, "hit_at_5")
    denominator = t1.get("denominator")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
    ):
        raise ValueError("ordinary T1 metric denominator must be a positive integer")
    return OrdinarySeedResult(
        dataset_id=bridge.dataset_id,
        arm_id=bridge.arm_id,
        method_id=bridge.method_id,
        selector_id=bridge.source_selector_id,
        active_learning_seed=bridge.active_learning_seed,
        rcl_seed=bridge.control.training_seed,
        budget=bridge.budget,
        evaluation_scope=T1_ONLY_SCOPE,
        evaluation_case_count=denominator,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        top135=(hit_at_1 + hit_at_3 + hit_at_5) / 3.0,
        query_plan_sha256=bridge.query_plan_sha256,
        bridge_sha256=bridge.bridge_sha256,
        runner_query_plan_sha256=checked["query_plan_sha256"],
        runner_result_sha256=_semantic_sha256(checked),
        code_sha256=bridge.code_sha256,
        data_sha256=bridge.data_sha256,
        representation_sha256=bridge.representation_sha256,
        protocol_sha256=bridge.protocol_sha256,
        rcl_control_sha256=bridge.control.control_sha256,
    )


def validate_ordinary_seed_result(payload: Mapping[str, Any]) -> OrdinarySeedResult:
    """Recompute a serialized per-seed result seal and typed invariants."""

    if not isinstance(payload, Mapping):
        raise ValueError("ordinary seed result must be an object")
    identity = deepcopy(dict(payload))
    declared = identity.pop("result_sha256", None)
    if declared != _semantic_sha256(identity):
        raise ValueError("ordinary seed result seal drifted")
    schema_version = identity.pop("schema_version", None)
    if schema_version != "combined-active-learning-ordinary-seed-result-v1":
        raise ValueError("ordinary seed result schema drifted")
    try:
        return OrdinarySeedResult(**identity)
    except TypeError as exc:
        raise ValueError("ordinary seed result field allowlist drifted") from exc


def aggregate_ordinary_three_seed_results(
    results: Sequence[OrdinarySeedResult | Mapping[str, Any]],
) -> OrdinaryThreeSeedAggregate:
    """Compute the exact arithmetic mean over seeds 41, 42, and 43."""

    if isinstance(results, (str, bytes)) or len(results) != 3:
        raise ValueError("ordinary aggregate requires exactly seeds 41/42/43")
    checked = tuple(
        result
        if type(result) is OrdinarySeedResult
        else validate_ordinary_seed_result(result)
        for result in results
    )
    if {result.active_learning_seed for result in checked} != set(
        ACTIVE_LEARNING_SEEDS
    ):
        raise ValueError("ordinary aggregate requires exactly seeds 41/42/43")
    ordered = tuple(sorted(checked, key=lambda item: item.active_learning_seed))
    shared_identity = {
        (
            result.dataset_id,
            result.arm_id,
            result.method_id,
            result.selector_id,
            result.rcl_seed,
            result.budget,
            result.evaluation_scope,
            result.evaluation_case_count,
            result.code_sha256,
            result.data_sha256,
            result.representation_sha256,
            result.protocol_sha256,
            result.rcl_control_sha256,
        )
        for result in ordered
    }
    if len(shared_identity) != 1:
        raise ValueError("three-seed results must belong to the same ordinary arm")

    def mean(name: str) -> float:
        return float(sum(getattr(result, name) for result in ordered) / 3.0)

    first = ordered[0]
    return OrdinaryThreeSeedAggregate(
        dataset_id=first.dataset_id,
        arm_id=first.arm_id,
        method_id=first.method_id,
        selector_id=first.selector_id,
        active_learning_seeds=ACTIVE_LEARNING_SEEDS,
        result_sha256_by_seed=tuple(
            (result.active_learning_seed, result.result_sha256)
            for result in ordered
        ),
        top135_by_seed=tuple(
            (result.active_learning_seed, result.top135) for result in ordered
        ),
        mean_hit_at_1=mean("hit_at_1"),
        mean_hit_at_3=mean("hit_at_3"),
        mean_hit_at_5=mean("hit_at_5"),
        mean_top135=mean("top135"),
        evaluation_case_count=first.evaluation_case_count,
        rcl_seed=first.rcl_seed,
        budget=first.budget,
    )


def select_dataset_best_ordinary_strategy(
    aggregates: Sequence[OrdinaryThreeSeedAggregate],
) -> DatasetOrdinaryStrategySelection:
    """Select one method's best mean-TOP135 strategy on one dataset."""

    if isinstance(aggregates, (str, bytes)) or not aggregates:
        raise ValueError("ordinary strategy candidates must be non-empty")
    if any(type(item) is not OrdinaryThreeSeedAggregate for item in aggregates):
        raise ValueError("ordinary strategy candidates must be typed aggregates")
    scopes = {(item.dataset_id, item.method_id) for item in aggregates}
    if len(scopes) != 1:
        raise ValueError("ordinary strategy candidates must cover one dataset and method")
    dataset_id, method_id = next(iter(scopes))
    DatasetId(dataset_id)
    if method_id not in EXPECTED_ORDINARY_SELECTORS:
        raise ValueError("ordinary strategy method is unsupported")
    expected_selectors = set(EXPECTED_ORDINARY_SELECTORS[method_id])
    actual_selectors = [item.selector_id for item in aggregates]
    if (
        len(actual_selectors) != len(set(actual_selectors))
        or set(actual_selectors) != expected_selectors
    ):
        raise ValueError("ordinary method requires its complete selector set")
    for item in aggregates:
        if item.arm_id != f"{method_id}.{item.selector_id}":
            raise ValueError("ordinary strategy arm identity drifted")
        if (
            item.active_learning_seeds != ACTIVE_LEARNING_SEEDS
            or item.rcl_seed != DOWNSTREAM_RCL_SEED
            or item.budget != ANNOTATION_BUDGET
            or not math.isfinite(item.mean_top135)
            or not 0.0 <= item.mean_top135 <= 1.0
        ):
            raise ValueError("ordinary strategy aggregate protocol drifted")
    best_score = max(item.mean_top135 for item in aggregates)
    tied = tuple(
        sorted(
            item.selector_id
            for item in aggregates
            if item.mean_top135 == best_score
        )
    )
    selected_selector = tied[0]
    selected = next(
        item for item in aggregates if item.selector_id == selected_selector
    )
    threshold = dict(ORDINARY_SNAPSHOT_THRESHOLDS)[dataset_id]
    margin = best_score - threshold
    return DatasetOrdinaryStrategySelection(
        dataset_id=dataset_id,
        method_id=method_id,
        selected_arm_id=selected.arm_id,
        selected_selector_id=selected_selector,
        mean_top135=best_score,
        threshold=threshold,
        margin=margin,
        passed=best_score >= threshold,
        tied_best_selector_ids=tied,
        aggregate_sha256_by_selector=tuple(
            sorted(
                (item.selector_id, item.aggregate_sha256)
                for item in aggregates
            )
        ),
    )


def qualify_ordinary_method_across_datasets(
    selections: Sequence[DatasetOrdinaryStrategySelection],
) -> MethodOrdinaryQualification:
    """Require the selected strategy to meet the snapshot gate on both datasets."""

    if isinstance(selections, (str, bytes)) or len(selections) != 2:
        raise ValueError("ordinary qualification requires both datasets")
    if any(
        type(selection) is not DatasetOrdinaryStrategySelection
        for selection in selections
    ):
        raise ValueError("ordinary qualification requires typed selections")
    by_dataset = {selection.dataset_id: selection for selection in selections}
    if set(by_dataset) != {name for name, _ in ORDINARY_SNAPSHOT_THRESHOLDS}:
        raise ValueError("ordinary qualification requires both datasets")
    methods = {selection.method_id for selection in selections}
    if len(methods) != 1:
        raise ValueError("ordinary qualification selections must share one method")
    failure_reasons = tuple(
        (
            f"{dataset_id} mean TOP135 {selection.mean_top135:.6f} "
            f"is below {selection.threshold:.6f}"
        )
        for dataset_id, _threshold in ORDINARY_SNAPSHOT_THRESHOLDS
        if not (selection := by_dataset[dataset_id]).passed
    )
    return MethodOrdinaryQualification(
        method_id=next(iter(methods)),
        qualified=not failure_reasons,
        failure_reasons=failure_reasons,
        selection_sha256_by_dataset=tuple(
            (
                dataset_id,
                by_dataset[dataset_id].selection_sha256,
            )
            for dataset_id, _threshold in ORDINARY_SNAPSHOT_THRESHOLDS
        ),
    )


__all__ = [
    "Budget30RclBridge",
    "DatasetOrdinaryStrategySelection",
    "EXPECTED_ORDINARY_SELECTORS",
    "FIXED_QUERY_SELECTOR_ID",
    "FrozenRclControl",
    "MethodOrdinaryQualification",
    "ORDINARY_SNAPSHOT_THRESHOLDS",
    "OrdinarySeedResult",
    "OrdinaryThreeSeedAggregate",
    "STRICT_LOFO_SCOPE",
    "StrictLofoRclBridge",
    "StrictLofoSeedResult",
    "T1_ONLY_SCOPE",
    "build_budget30_rcl_bridge",
    "build_ordinary_seed_result",
    "build_strict_lofo_rcl_bridge",
    "build_strict_lofo_seed_result",
    "aggregate_ordinary_three_seed_results",
    "project_authority_query_only_unit",
    "project_strict_lofo_query_only_unit",
    "qualify_ordinary_method_across_datasets",
    "run_ordinary_query_only_t1",
    "run_strict_lofo_query_only",
    "select_dataset_best_ordinary_strategy",
    "validate_budget30_rcl_bridge",
    "validate_ordinary_seed_result",
    "validate_ordinary_t1_backend_payload",
    "validate_strict_lofo_backend_payload",
    "validate_strict_lofo_rcl_bridge",
    "validate_strict_lofo_seed_result",
]
