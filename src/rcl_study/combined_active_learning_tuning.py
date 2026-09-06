"""Frozen, label-free AIOps22 HDBSCAN tuning and orchestration protocol.

The module contains immutable tuning/state schemas and guarded JSON
publication helpers, but never executes HDBSCAN or downstream RCL. Numerical
fitting and center-plan construction are dependency-injected and validated at
the protocol boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import fmean
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


METHOD_IDENTITY = "global_pca_dim32 + HDBSCAN + center"
DATASET_ID = "aiops2022_pre"
CANDIDATE_POOL_SIZE = 169
BUDGET = 30
ACQUISITION_SEEDS = (41, 42, 43)
DOWNSTREAM_SEED = 42
EVALUATION_PROTOCOL = "outer_test_guided"
EFFECTIVE_SUPPORT = 5
METRIC = "euclidean"
ALPHA = 1.0
CLUSTER_SELECTION_EPSILON = 0.0
ALLOW_SINGLE_CLUSTER = False
STORE_CENTERS = "medoid"
TARGET_EFFECTIVE_CLUSTER_COUNT = 10
MIN_EFFECTIVE_CLUSTER_COUNT = 6
MAX_EFFECTIVE_CLUSTER_COUNT = 15
MIN_MEAN_EFFECTIVE_COVERAGE = 0.50
MAX_LARGEST_EFFECTIVE_CLUSTER_SHARE = 0.50
MAX_CONFIGURATIONS_PER_BATCH = 12
MAX_SEED_UNITS_PER_BATCH = 36
SCHEMA_VERSION = "combined-active-learning-limited-hdbscan-tuning-v1"
TUNING_RESULT_SCHEMA_VERSION = "combined-active-learning-tuning-batch-result-v1"
TUNING_STATE_SCHEMA_VERSION = "combined-active-learning-tuning-state-v1"
TUNING_SENTINEL_SCHEMA_VERSION = "combined-active-learning-tuning-sentinel-v1"
TUNING_OWNER_OVERRIDE_SCHEMA_VERSION = (
    "combined-active-learning-tuning-owner-override-v1"
)
TUNING_INDEX_SCHEMA_VERSION = "combined-active-learning-tuning-batch-index-v1"
FROZEN_FOLLOWUP_RELATIVE_PARTS = (
    "outputs",
    "combined_active_learning_2_0",
    "followup",
    "aiops22_hdbscan_tuning_v1",
)
TARGET_TOP135 = Decimal("0.830247")
PARAMETER_DISTANCE_DEFINITION = (
    "abs(min_cluster_size-5)+abs(min_samples-3)+"
    "selection_method_changed+max_cluster_size_enabled"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASE_METRIC_DECIMAL_PLACES = 36
_METRIC_DECIMAL_PRECISION = 48
_DERIVED_METRIC_DECIMAL_PLACES = (
    _BASE_METRIC_DECIMAL_PLACES + _METRIC_DECIMAL_PRECISION
)
_AGGREGATE_REASSOCIATION_TOLERANCE = Decimal(1).scaleb(
    1 - _METRIC_DECIMAL_PRECISION
)


def _is_frozen_float(value: Any, expected: float) -> bool:
    return type(value) is float and math.isfinite(value) and value == expected


def _configuration_identifier(
    *,
    round_id: str,
    cluster_selection_method: str,
    min_cluster_size: int,
    min_samples: int,
    max_cluster_size: Optional[int],
) -> str:
    maximum = "none" if max_cluster_size is None else str(max_cluster_size)
    return (
        "hdbscan.{round_id}.{method}.mcs{mcs:02d}.ms{ms:02d}.max{maximum}."
        "metric-euclidean.alpha1.eps0.single0.centers-medoid"
    ).format(
        round_id=round_id.lower(),
        method=cluster_selection_method,
        mcs=min_cluster_size,
        ms=min_samples,
        maximum=maximum,
    )


@dataclass(frozen=True)
class TuningConfiguration:
    configuration_id: str
    round_id: str
    cluster_selection_method: str
    min_cluster_size: int
    min_samples: int
    max_cluster_size: Optional[int]
    metric: str = METRIC
    alpha: float = ALPHA
    cluster_selection_epsilon: float = CLUSTER_SELECTION_EPSILON
    allow_single_cluster: bool = ALLOW_SINGLE_CLUSTER
    store_centers: str = STORE_CENTERS

    def __post_init__(self) -> None:
        if self.round_id not in ("A", "B"):
            raise ValueError("tuning round_id must be A or B")
        expected_method = "leaf" if self.round_id == "A" else "eom"
        if self.cluster_selection_method != expected_method:
            raise ValueError("cluster selection method does not match tuning round")
        if type(self.min_cluster_size) is not int or self.min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be a positive integer")
        if type(self.min_samples) is not int or self.min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")
        if self.round_id == "A" and self.max_cluster_size is not None:
            raise ValueError("Round A max_cluster_size must be None")
        if self.round_id == "B" and (
            type(self.max_cluster_size) is not int or self.max_cluster_size <= 0
        ):
            raise ValueError("Round B max_cluster_size must be a positive integer")
        if not (
            _is_frozen_float(self.alpha, ALPHA)
            and _is_frozen_float(
                self.cluster_selection_epsilon, CLUSTER_SELECTION_EPSILON
            )
        ):
            raise ValueError(
                "alpha and cluster_selection_epsilon must be finite canonical float controls"
            )
        fixed_controls = (
            self.metric == METRIC,
            self.allow_single_cluster is ALLOW_SINGLE_CLUSTER,
            self.store_centers == STORE_CENTERS,
        )
        if not all(fixed_controls):
            raise ValueError("configuration drifted from frozen HDBSCAN controls")
        expected_id = _configuration_identifier(
            round_id=self.round_id,
            cluster_selection_method=self.cluster_selection_method,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            max_cluster_size=self.max_cluster_size,
        )
        if self.configuration_id != expected_id:
            raise ValueError("configuration_id does not match canonical parameters")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "round_id": self.round_id,
            "cluster_selection_method": self.cluster_selection_method,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "max_cluster_size": self.max_cluster_size,
            "metric": self.metric,
            "alpha": self.alpha,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "allow_single_cluster": self.allow_single_cluster,
            "store_centers": self.store_centers,
        }


def _make_configuration(
    *,
    round_id: str,
    cluster_selection_method: str,
    min_cluster_size: int,
    min_samples: int,
    max_cluster_size: Optional[int],
) -> TuningConfiguration:
    return TuningConfiguration(
        configuration_id=_configuration_identifier(
            round_id=round_id,
            cluster_selection_method=cluster_selection_method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
        ),
        round_id=round_id,
        cluster_selection_method=cluster_selection_method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=max_cluster_size,
    )


def generate_round_a_grid() -> Tuple[TuningConfiguration, ...]:
    """Return the exact 6 by 5 Leaf grid in deterministic order."""

    return tuple(
        _make_configuration(
            round_id="A",
            cluster_selection_method="leaf",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            max_cluster_size=None,
        )
        for min_cluster_size in (3, 4, 5, 6, 7, 8)
        for min_samples in (1, 2, 3, 4, 5)
    )


def generate_round_b_grid() -> Tuple[TuningConfiguration, ...]:
    """Return the exact 3 by 4 by 5 conditional EOM grid."""

    return tuple(
        _make_configuration(
            round_id="B",
            cluster_selection_method="eom",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
        )
        for min_cluster_size in (3, 5, 8)
        for min_samples in (1, 2, 3, 5)
        for max_cluster_size in (15, 20, 25, 30, 40)
    )


@dataclass(frozen=True)
class TuningContract:
    method_identity: str
    dataset_id: str
    candidate_pool_size: int
    budget: int
    acquisition_seeds: Tuple[int, int, int]
    downstream_seed: int
    metric: str
    evaluation_protocol: str
    effective_support: int
    alpha: float
    cluster_selection_epsilon: float
    allow_single_cluster: bool
    store_centers: str
    round_a_configurations: Tuple[TuningConfiguration, ...]
    round_b_configurations: Tuple[TuningConfiguration, ...]
    round_b_role: str = "conditional_frozen_candidate_pool"

    def __post_init__(self) -> None:
        if not (
            _is_frozen_float(self.alpha, ALPHA)
            and _is_frozen_float(
                self.cluster_selection_epsilon, CLUSTER_SELECTION_EPSILON
            )
        ):
            raise ValueError(
                "alpha and cluster_selection_epsilon must be finite canonical float controls"
            )
        if (
            self.method_identity != METHOD_IDENTITY
            or self.dataset_id != DATASET_ID
            or self.candidate_pool_size != CANDIDATE_POOL_SIZE
            or self.budget != BUDGET
            or self.acquisition_seeds != ACQUISITION_SEEDS
            or self.downstream_seed != DOWNSTREAM_SEED
            or self.metric != METRIC
            or self.evaluation_protocol != EVALUATION_PROTOCOL
            or self.effective_support != EFFECTIVE_SUPPORT
            or self.allow_single_cluster is not ALLOW_SINGLE_CLUSTER
            or self.store_centers != STORE_CENTERS
            or self.round_a_configurations != generate_round_a_grid()
            or self.round_b_configurations != generate_round_b_grid()
            or self.round_b_role != "conditional_frozen_candidate_pool"
        ):
            raise ValueError("tuning contract drifted from frozen controls")

    def controls_dict(self) -> Dict[str, Any]:
        return {
            "method_identity": self.method_identity,
            "dataset_id": self.dataset_id,
            "candidate_pool_size": self.candidate_pool_size,
            "budget": self.budget,
            "acquisition_seeds": list(self.acquisition_seeds),
            "downstream_seed": self.downstream_seed,
            "metric": self.metric,
            "evaluation_protocol": self.evaluation_protocol,
            "effective_support": self.effective_support,
            "alpha": self.alpha,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "allow_single_cluster": self.allow_single_cluster,
            "store_centers": self.store_centers,
            "target_effective_cluster_count": TARGET_EFFECTIVE_CLUSTER_COUNT,
            "effective_cluster_count_range": [
                MIN_EFFECTIVE_CLUSTER_COUNT,
                MAX_EFFECTIVE_CLUSTER_COUNT,
            ],
            "mean_effective_coverage_strictly_greater_than": (
                MIN_MEAN_EFFECTIVE_COVERAGE
            ),
            "largest_effective_cluster_share_at_most": (
                MAX_LARGEST_EFFECTIVE_CLUSTER_SHARE
            ),
            "parameter_distance_definition": PARAMETER_DISTANCE_DEFINITION,
        }


def build_default_contract() -> TuningContract:
    return TuningContract(
        method_identity=METHOD_IDENTITY,
        dataset_id=DATASET_ID,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        budget=BUDGET,
        acquisition_seeds=ACQUISITION_SEEDS,
        downstream_seed=DOWNSTREAM_SEED,
        metric=METRIC,
        evaluation_protocol=EVALUATION_PROTOCOL,
        effective_support=EFFECTIVE_SUPPORT,
        alpha=ALPHA,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        allow_single_cluster=ALLOW_SINGLE_CLUSTER,
        store_centers=STORE_CENTERS,
        round_a_configurations=generate_round_a_grid(),
        round_b_configurations=generate_round_b_grid(),
    )


@dataclass(frozen=True)
class StructureSeedSummary:
    seed: int
    effective_cluster_count: int
    effective_coverage: float
    largest_effective_cluster_share: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("structure summary seed must be an integer")
        if (
            type(self.effective_cluster_count) is not int
            or self.effective_cluster_count < 0
        ):
            raise ValueError("effective cluster count must be a non-negative integer")
        for field_name, value in (
            ("effective coverage", self.effective_coverage),
            ("largest effective cluster share", self.largest_effective_cluster_share),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("{} must be finite and in [0,1]".format(field_name))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "effective_cluster_count": self.effective_cluster_count,
            "effective_coverage": float(self.effective_coverage),
            "largest_effective_cluster_share": float(
                self.largest_effective_cluster_share
            ),
        }


@dataclass(frozen=True)
class CandidateStructureEvaluation:
    configuration: TuningConfiguration
    seed_summaries: Tuple[StructureSeedSummary, ...]
    eligible: bool
    rejection_reasons: Tuple[str, ...]
    mean_absolute_cluster_distance: float
    mean_effective_coverage: float
    maximum_largest_effective_cluster_share: float
    parameter_distance: int

    def __post_init__(self) -> None:
        _validate_structure_evaluation(self)


def _parameter_distance(configuration: TuningConfiguration) -> int:
    return (
        abs(configuration.min_cluster_size - 5)
        + abs(configuration.min_samples - 3)
        + (0 if configuration.cluster_selection_method == "leaf" else 1)
        + (0 if configuration.max_cluster_size is None else 1)
    )


def _canonical_structure_components(
    configuration: TuningConfiguration,
    seed_summaries: Sequence[StructureSeedSummary],
) -> Tuple[
    Tuple[StructureSeedSummary, ...],
    bool,
    Tuple[str, ...],
    float,
    float,
    float,
    int,
]:
    if not isinstance(configuration, TuningConfiguration):
        raise ValueError("canonical structure evaluation requires a tuning configuration")
    summaries = tuple(seed_summaries)
    if (
        len(summaries) != len(ACQUISITION_SEEDS)
        or any(not isinstance(item, StructureSeedSummary) for item in summaries)
        or tuple(item.seed for item in summaries) != ACQUISITION_SEEDS
    ):
        raise ValueError(
            "canonical structure evaluation requires the exact acquisition seeds"
        )

    reasons = []
    for summary in summaries:
        if not (
            MIN_EFFECTIVE_CLUSTER_COUNT
            <= summary.effective_cluster_count
            <= MAX_EFFECTIVE_CLUSTER_COUNT
        ):
            reasons.append(
                "seed_{}.effective_cluster_count={} outside [{},{}]".format(
                    summary.seed,
                    summary.effective_cluster_count,
                    MIN_EFFECTIVE_CLUSTER_COUNT,
                    MAX_EFFECTIVE_CLUSTER_COUNT,
                )
            )

    mean_coverage = float(
        sum(
            (Decimal(str(float(item.effective_coverage))) for item in summaries),
            Decimal(0),
        )
        / Decimal(len(summaries))
    )
    if not mean_coverage > MIN_MEAN_EFFECTIVE_COVERAGE:
        reasons.append(
            "mean_effective_coverage={:.6f} must be > {:.6f}".format(
                mean_coverage, MIN_MEAN_EFFECTIVE_COVERAGE
            )
        )

    for summary in summaries:
        share = float(summary.largest_effective_cluster_share)
        if share > MAX_LARGEST_EFFECTIVE_CLUSTER_SHARE:
            reasons.append(
                "seed_{}.largest_effective_cluster_share={:.6f} exceeds {:.6f}".format(
                    summary.seed, share, MAX_LARGEST_EFFECTIVE_CLUSTER_SHARE
                )
            )
    return (
        summaries,
        not reasons,
        tuple(reasons),
        fmean(
            abs(item.effective_cluster_count - TARGET_EFFECTIVE_CLUSTER_COUNT)
            for item in summaries
        ),
        mean_coverage,
        max(float(item.largest_effective_cluster_share) for item in summaries),
        _parameter_distance(configuration),
    )


def _validate_structure_evaluation(
    evaluation: CandidateStructureEvaluation,
) -> None:
    if not isinstance(evaluation, CandidateStructureEvaluation):
        raise ValueError("canonical structure evaluation has the wrong type")
    if type(evaluation.seed_summaries) is not tuple:
        raise ValueError("canonical structure evaluation summaries must be immutable")
    expected = _canonical_structure_components(
        evaluation.configuration, evaluation.seed_summaries
    )
    observed = (
        evaluation.seed_summaries,
        evaluation.eligible,
        evaluation.rejection_reasons,
        evaluation.mean_absolute_cluster_distance,
        evaluation.mean_effective_coverage,
        evaluation.maximum_largest_effective_cluster_share,
        evaluation.parameter_distance,
    )
    if (
        type(evaluation.eligible) is not bool
        or type(evaluation.rejection_reasons) is not tuple
        or type(evaluation.mean_absolute_cluster_distance) is not float
        or type(evaluation.mean_effective_coverage) is not float
        or type(evaluation.maximum_largest_effective_cluster_share) is not float
        or type(evaluation.parameter_distance) is not int
        or observed != expected
    ):
        raise ValueError(
            "canonical structure evaluation fields do not match recomputed values"
        )


def assess_candidate_structure(
    configuration: TuningConfiguration,
    seed_summaries: Sequence[StructureSeedSummary],
) -> CandidateStructureEvaluation:
    """Apply the complete label-free three-seed structure gate."""

    (
        summaries,
        eligible,
        reasons,
        mean_distance,
        mean_coverage,
        maximum_share,
        parameter_distance,
    ) = _canonical_structure_components(configuration, seed_summaries)
    return CandidateStructureEvaluation(
        configuration=configuration,
        seed_summaries=summaries,
        eligible=eligible,
        rejection_reasons=reasons,
        mean_absolute_cluster_distance=mean_distance,
        mean_effective_coverage=mean_coverage,
        maximum_largest_effective_cluster_share=maximum_share,
        parameter_distance=parameter_distance,
    )


def label_free_sort_key(
    evaluation: CandidateStructureEvaluation,
) -> Tuple[float, float, float, int, str]:
    """Rank without labels, ground truth, or downstream RCL results."""

    if not isinstance(evaluation, CandidateStructureEvaluation):
        raise ValueError("label-free ranking requires a structure evaluation")
    _validate_structure_evaluation(evaluation)
    return (
        evaluation.mean_absolute_cluster_distance,
        -evaluation.mean_effective_coverage,
        evaluation.maximum_largest_effective_cluster_share,
        evaluation.parameter_distance,
        evaluation.configuration.configuration_id,
    )


def rank_structure_evaluations(
    evaluations: Iterable[CandidateStructureEvaluation],
) -> Tuple[CandidateStructureEvaluation, ...]:
    return tuple(sorted(tuple(evaluations), key=label_free_sort_key))


def evaluate_configuration_structures(
    contract: TuningContract,
    configurations: Sequence[TuningConfiguration],
    *,
    fit_fn: Callable[..., StructureSeedSummary]
) -> Tuple[CandidateStructureEvaluation, ...]:
    """Call an injected fit adapter for every configuration/seed unit."""

    if contract != build_default_contract():
        raise ValueError("fit adapter requires the frozen tuning contract")
    allowed = {
        item.configuration_id
        for item in contract.round_a_configurations + contract.round_b_configurations
    }
    observed = set()
    evaluations = []
    for configuration in tuple(configurations):
        if (
            not isinstance(configuration, TuningConfiguration)
            or configuration.configuration_id not in allowed
            or configuration.configuration_id in observed
        ):
            raise ValueError("fit adapter configuration set is invalid")
        observed.add(configuration.configuration_id)
        summaries = []
        for seed in contract.acquisition_seeds:
            summary = fit_fn(
                configuration=configuration,
                seed=seed,
                effective_support=contract.effective_support,
            )
            if not isinstance(summary, StructureSeedSummary):
                raise ValueError("fit adapter must return StructureSeedSummary")
            if summary.seed != seed:
                raise ValueError("fit adapter returned the wrong seed")
            summaries.append(summary)
        evaluations.append(assess_candidate_structure(configuration, summaries))
    return tuple(evaluations)


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_pool_digest(pool_case_ids: Sequence[str]) -> str:
    encoded = json.dumps(
        list(pool_case_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _query_plan_triple_payload(
    query_plans: Sequence[Tuple[int, Sequence[str]]],
) -> Dict[str, Any]:
    return {
        "acquisition_seeds": list(ACQUISITION_SEEDS),
        "plans": [
            {"seed": seed, "case_ids": list(plan)} for seed, plan in query_plans
        ],
    }


def _query_plan_triple_digest(
    query_plans: Sequence[Tuple[int, Sequence[str]]],
) -> str:
    return _semantic_sha256(_query_plan_triple_payload(query_plans))


@dataclass(frozen=True)
class TuningCandidate:
    evaluation: CandidateStructureEvaluation
    ordered_pool_case_ids: Tuple[str, ...]
    ordered_pool_sha256: str
    query_plans: Tuple[Tuple[int, Tuple[str, ...]], ...]
    query_plan_triple_hash: str

    @property
    def configuration(self) -> TuningConfiguration:
        return self.evaluation.configuration


def build_tuning_candidate(
    evaluation: CandidateStructureEvaluation,
    *,
    pool_case_ids: Sequence[str],
    ordered_pool_sha256: str,
    center_plan_fn: Callable[..., Sequence[str]]
) -> TuningCandidate:
    """Generate and validate the three injected medoid-center query plans."""

    if not isinstance(evaluation, CandidateStructureEvaluation):
        raise ValueError("query plans require a canonical structure evaluation")
    _validate_structure_evaluation(evaluation)
    if not evaluation.eligible:
        raise ValueError("query plans may be built only for an eligible configuration")
    pool = tuple(pool_case_ids)
    if (
        len(pool) != CANDIDATE_POOL_SIZE
        or any(not isinstance(case_id, str) or not case_id for case_id in pool)
        or len(set(pool)) != CANDIDATE_POOL_SIZE
    ):
        raise ValueError("query planning requires the complete 169 unique case pool")
    if (
        not isinstance(ordered_pool_sha256, str)
        or not _SHA256.fullmatch(ordered_pool_sha256)
        or ordered_pool_sha256 != _ordered_pool_digest(pool)
    ):
        raise ValueError("ordered pool SHA-256 does not match the canonical case order")
    pool_set = set(pool)
    plans = []
    for seed in ACQUISITION_SEEDS:
        plan = tuple(
            center_plan_fn(
                configuration=evaluation.configuration,
                seed=seed,
                pool_case_ids=pool,
                budget=BUDGET,
            )
        )
        if (
            len(plan) != BUDGET
            or any(not isinstance(case_id, str) or not case_id for case_id in plan)
            or len(set(plan)) != BUDGET
        ):
            raise ValueError(
                "each center query plan must contain exactly 30 unique string case IDs"
            )
        if any(case_id not in pool_set for case_id in plan):
            raise ValueError("center query plan contains a case outside the full candidate pool")
        plans.append((seed, plan))
    query_plans = tuple(plans)
    return TuningCandidate(
        evaluation=evaluation,
        ordered_pool_case_ids=pool,
        ordered_pool_sha256=ordered_pool_sha256,
        query_plans=query_plans,
        query_plan_triple_hash=_query_plan_triple_digest(query_plans),
    )


def _validate_candidate(candidate: TuningCandidate) -> None:
    if not isinstance(candidate, TuningCandidate):
        raise ValueError("candidate validation requires tuning candidates")
    _validate_structure_evaluation(candidate.evaluation)
    if not candidate.evaluation.eligible:
        raise ValueError("tuning candidate must retain an eligible evaluation")
    pool = candidate.ordered_pool_case_ids
    if (
        len(pool) != CANDIDATE_POOL_SIZE
        or any(not isinstance(case_id, str) or not case_id for case_id in pool)
        or len(set(pool)) != CANDIDATE_POOL_SIZE
        or not _SHA256.fullmatch(str(candidate.ordered_pool_sha256))
        or _ordered_pool_digest(pool) != candidate.ordered_pool_sha256
    ):
        raise ValueError("tuning candidate ordered pool SHA-256 is invalid")
    if tuple(seed for seed, _plan in candidate.query_plans) != ACQUISITION_SEEDS:
        raise ValueError("tuning candidate plans must contain the exact acquisition seeds")
    pool_set = set(pool)
    for _seed, plan in candidate.query_plans:
        if (
            len(plan) != BUDGET
            or any(not isinstance(case_id, str) or not case_id for case_id in plan)
            or len(set(plan)) != BUDGET
        ):
            raise ValueError("each tuning candidate plan must contain 30 unique cases")
        if any(case_id not in pool_set for case_id in plan):
            raise ValueError("tuning candidate plan contains a case outside its ordered pool")
    if candidate.query_plan_triple_hash != _query_plan_triple_digest(
        candidate.query_plans
    ):
        raise ValueError("tuning candidate query plan triple hash is invalid")


@dataclass(frozen=True)
class DuplicatePlanMapping:
    duplicate_configuration_id: str
    canonical_configuration_id: str
    query_plan_triple_hash: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "duplicate_configuration_id": self.duplicate_configuration_id,
            "canonical_configuration_id": self.canonical_configuration_id,
            "query_plan_triple_hash": self.query_plan_triple_hash,
        }


@dataclass(frozen=True)
class DeduplicationResult:
    source_candidates: Tuple[TuningCandidate, ...]
    canonical_candidates: Tuple[TuningCandidate, ...]
    duplicate_mappings: Tuple[DuplicatePlanMapping, ...]


def _candidate_sort_key(candidate: TuningCandidate) -> Tuple[Any, ...]:
    round_order = 0 if candidate.configuration.round_id == "A" else 1
    return (round_order,) + label_free_sort_key(candidate.evaluation)


def deduplicate_query_plan_triples(
    candidates: Iterable[TuningCandidate],
) -> DeduplicationResult:
    """Keep the highest label-free-ranked configuration for each plan triple."""

    supplied = tuple(candidates)
    if any(not isinstance(item, TuningCandidate) for item in supplied):
        raise ValueError("plan deduplication requires tuning candidates")
    for candidate in supplied:
        _validate_candidate(candidate)
    pool_hashes = {candidate.ordered_pool_sha256 for candidate in supplied}
    if len(pool_hashes) > 1:
        raise ValueError("plan deduplication requires one ordered pool identity")
    ordered = tuple(sorted(supplied, key=_candidate_sort_key))
    ids = [item.configuration.configuration_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("plan deduplication received duplicate configuration IDs")
    by_hash = {}
    canonical = []
    duplicates = []
    for candidate in ordered:
        existing = by_hash.get(candidate.query_plan_triple_hash)
        if existing is None:
            by_hash[candidate.query_plan_triple_hash] = candidate
            canonical.append(candidate)
            continue
        if candidate.query_plans != existing.query_plans:
            raise ValueError("query plan triple hash collision")
        duplicates.append(
            DuplicatePlanMapping(
                duplicate_configuration_id=candidate.configuration.configuration_id,
                canonical_configuration_id=existing.configuration.configuration_id,
                query_plan_triple_hash=candidate.query_plan_triple_hash,
            )
        )
    return DeduplicationResult(ordered, tuple(canonical), tuple(duplicates))


@dataclass(frozen=True)
class BatchConfigurationReference:
    configuration_id: str
    query_plan_triple_hash: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "configuration_id": self.configuration_id,
            "query_plan_triple_hash": self.query_plan_triple_hash,
        }


@dataclass(frozen=True)
class TuningBatch:
    batch_id: str
    round_id: str
    configuration_references: Tuple[BatchConfigurationReference, ...]
    seed_unit_count: int

    @property
    def configuration_ids(self) -> Tuple[str, ...]:
        return tuple(item.configuration_id for item in self.configuration_references)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "round_id": self.round_id,
            "configuration_ids": list(self.configuration_ids),
            "configuration_references": [
                item.to_dict() for item in self.configuration_references
            ],
            "seed_unit_count": self.seed_unit_count,
        }


def build_batches(
    canonical_candidates: Sequence[TuningCandidate],
) -> Tuple[TuningBatch, ...]:
    """Build deterministic, round-separated batches capped at 12/36."""

    supplied = tuple(canonical_candidates)
    if any(not isinstance(item, TuningCandidate) for item in supplied):
        raise ValueError("batch input must contain tuning candidates")
    for candidate in supplied:
        _validate_candidate(candidate)
    candidates = tuple(sorted(supplied, key=_candidate_sort_key))
    ids = [item.configuration.configuration_id for item in candidates]
    if (
        any(not item.evaluation.eligible for item in candidates)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("batch input must be unique eligible canonical candidates")
    batches = []
    for round_id in ("A", "B"):
        in_round = [item for item in candidates if item.configuration.round_id == round_id]
        for start in range(0, len(in_round), MAX_CONFIGURATIONS_PER_BATCH):
            chunk = tuple(in_round[start : start + MAX_CONFIGURATIONS_PER_BATCH])
            configuration_ids = tuple(
                item.configuration.configuration_id for item in chunk
            )
            seed_unit_count = len(configuration_ids) * len(ACQUISITION_SEEDS)
            if seed_unit_count > MAX_SEED_UNITS_PER_BATCH:
                raise ValueError("tuning batch exceeded the frozen seed-unit cap")
            batches.append(
                TuningBatch(
                    batch_id="round-{}-batch-{:03d}".format(
                        round_id.lower(), start // MAX_CONFIGURATIONS_PER_BATCH + 1
                    ),
                    round_id=round_id,
                    configuration_references=tuple(
                        BatchConfigurationReference(
                            configuration_id=item.configuration.configuration_id,
                            query_plan_triple_hash=item.query_plan_triple_hash,
                        )
                        for item in chunk
                    ),
                    seed_unit_count=seed_unit_count,
                )
            )
    return tuple(batches)


@dataclass(frozen=True)
class ExcludedConfiguration:
    configuration_id: str
    rejection_reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class TuningProvenance:
    structure_inputs: str = "label_free_effective_cluster_summaries"
    query_plan_inputs: str = "complete_case_pool_and_medoid_centers"
    ground_truth_used_for_ranking: bool = False
    rcl_used_for_ranking: bool = False
    case_labels_used_for_ranking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure_inputs": self.structure_inputs,
            "query_plan_inputs": self.query_plan_inputs,
            "ground_truth_used_for_ranking": self.ground_truth_used_for_ranking,
            "rcl_used_for_ranking": self.rcl_used_for_ranking,
            "case_labels_used_for_ranking": self.case_labels_used_for_ranking,
        }


def _candidate_plan_dict(candidate: TuningCandidate) -> Dict[str, Any]:
    return {
        "configuration_id": candidate.configuration.configuration_id,
        "ordered_pool_sha256": candidate.ordered_pool_sha256,
        "query_plan_triple_hash": candidate.query_plan_triple_hash,
        "query_plans": [
            {"seed": seed, "case_ids": list(plan)}
            for seed, plan in candidate.query_plans
        ],
    }


@dataclass(frozen=True)
class TuningManifest:
    contract: TuningContract
    eligible_count: int
    excluded: Tuple[ExcludedConfiguration, ...]
    duplicate_mappings: Tuple[DuplicatePlanMapping, ...]
    canonical_candidates: Tuple[TuningCandidate, ...]
    ordered_pool_sha256: Optional[str]
    batches: Tuple[TuningBatch, ...]
    provenance: TuningProvenance
    schema_version: str = SCHEMA_VERSION
    immutable: bool = True
    mutation_policy: str = "frozen_no_overwrite"

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.immutable is not True
            or self.mutation_policy != "frozen_no_overwrite"
        ):
            raise ValueError("tuning manifest immutability controls drifted")
        if self.canonical_candidates:
            if (
                not isinstance(self.ordered_pool_sha256, str)
                or not _SHA256.fullmatch(self.ordered_pool_sha256)
                or any(
                    candidate.ordered_pool_sha256 != self.ordered_pool_sha256
                    for candidate in self.canonical_candidates
                )
            ):
                raise ValueError("tuning manifest ordered pool identity drifted")
        elif self.ordered_pool_sha256 is not None:
            raise ValueError("empty tuning manifest cannot invent an ordered pool identity")

    def _body_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "immutable": self.immutable,
            "mutation_policy": self.mutation_policy,
            "ordered_pool_sha256": self.ordered_pool_sha256,
            "controls": self.contract.controls_dict(),
            "grid": {
                "round_a": len(self.contract.round_a_configurations),
                "round_b": len(self.contract.round_b_configurations),
                "total": len(self.contract.round_a_configurations)
                + len(self.contract.round_b_configurations),
            },
            "configurations": {
                "round_a": [
                    item.to_dict() for item in self.contract.round_a_configurations
                ],
                "round_b": [
                    item.to_dict() for item in self.contract.round_b_configurations
                ],
            },
            "structure_summary": {
                "eligible_count": self.eligible_count,
                "excluded_count": len(self.excluded),
                "duplicate_count": len(self.duplicate_mappings),
                "excluded": [item.to_dict() for item in self.excluded],
                "duplicates": [
                    item.to_dict() for item in self.duplicate_mappings
                ],
            },
            "canonical_candidates": [
                _candidate_plan_dict(item) for item in self.canonical_candidates
            ],
            "batches": [item.to_dict() for item in self.batches],
            "evaluation_protocol": self.contract.evaluation_protocol,
            "provenance": self.provenance.to_dict(),
            "round_b_policy": {
                "role": self.contract.round_b_role,
                "activation": "only_after_round_a_decision",
                "rcl_early_stop_state_machine_implemented": True,
            },
        }

    @property
    def manifest_sha256(self) -> str:
        return _semantic_sha256(self._body_dict())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._body_dict()
        payload["manifest_sha256"] = _semantic_sha256(payload)
        return payload

    @staticmethod
    def validate_payload(
        payload: Mapping[str, Any],
        *,
        trusted_expected_sha256: Optional[str] = None
    ) -> str:
        """Validate round-tripped semantics, internal hashes, and an optional trust anchor."""

        if not isinstance(payload, Mapping):
            raise ValueError("tuning manifest payload must be a mapping")
        body = dict(payload)
        manifest_sha256 = body.pop("manifest_sha256", None)
        if (
            not isinstance(manifest_sha256, str)
            or not _SHA256.fullmatch(manifest_sha256)
            or _semantic_sha256(body) != manifest_sha256
        ):
            raise ValueError("tuning manifest SHA-256 is invalid")
        if trusted_expected_sha256 is not None and (
            not isinstance(trusted_expected_sha256, str)
            or not _SHA256.fullmatch(trusted_expected_sha256)
            or trusted_expected_sha256 != manifest_sha256
        ):
            raise ValueError("trusted manifest SHA-256 does not match the payload")

        expected_body_fields = {
            "schema_version",
            "immutable",
            "mutation_policy",
            "ordered_pool_sha256",
            "controls",
            "grid",
            "configurations",
            "structure_summary",
            "canonical_candidates",
            "batches",
            "evaluation_protocol",
            "provenance",
            "round_b_policy",
        }
        if set(body) != expected_body_fields:
            raise ValueError("tuning manifest frozen contract fields are invalid")

        expected_contract = build_default_contract()
        expected_grid = {"round_a": 30, "round_b": 60, "total": 90}
        expected_configurations = {
            "round_a": [
                item.to_dict() for item in expected_contract.round_a_configurations
            ],
            "round_b": [
                item.to_dict() for item in expected_contract.round_b_configurations
            ],
        }
        if (
            body.get("schema_version") != SCHEMA_VERSION
            or body.get("immutable") is not True
            or body.get("mutation_policy") != "frozen_no_overwrite"
            or body.get("evaluation_protocol") != EVALUATION_PROTOCOL
        ):
            raise ValueError("tuning manifest frozen controls are invalid")
        if (
            body.get("controls") != expected_contract.controls_dict()
            or body.get("grid") != expected_grid
            or body.get("configurations") != expected_configurations
        ):
            raise ValueError("tuning manifest frozen contract is invalid")
        if body.get("provenance") != TuningProvenance().to_dict():
            raise ValueError("tuning manifest label-free provenance is invalid")
        expected_round_b_policy = {
            "role": "conditional_frozen_candidate_pool",
            "activation": "only_after_round_a_decision",
            "rcl_early_stop_state_machine_implemented": True,
        }
        if body.get("round_b_policy") != expected_round_b_policy:
            raise ValueError("tuning manifest Round B policy is invalid")

        all_configurations = (
            expected_contract.round_a_configurations
            + expected_contract.round_b_configurations
        )
        configuration_rounds = {
            item.configuration_id: item.round_id for item in all_configurations
        }
        all_configuration_ids = set(configuration_rounds)
        ordered_pool_sha256 = body.get("ordered_pool_sha256")
        records = body.get("canonical_candidates")
        if not isinstance(records, list):
            raise ValueError("tuning manifest canonical candidates are missing")
        if records:
            if (
                not isinstance(ordered_pool_sha256, str)
                or not _SHA256.fullmatch(ordered_pool_sha256)
            ):
                raise ValueError("tuning manifest ordered pool SHA-256 is invalid")
        elif ordered_pool_sha256 is not None:
            raise ValueError("empty tuning manifest ordered pool SHA-256 is invalid")

        plan_hashes = {}
        canonical_ids = []
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "configuration_id",
                "ordered_pool_sha256",
                "query_plan_triple_hash",
                "query_plans",
            }:
                raise ValueError("tuning manifest candidate record must be canonical")
            configuration_id = record.get("configuration_id")
            if (
                not isinstance(configuration_id, str)
                or configuration_id not in all_configuration_ids
                or configuration_id in plan_hashes
                or record.get("ordered_pool_sha256") != ordered_pool_sha256
            ):
                raise ValueError("tuning manifest candidate identity is invalid")
            query_plans_payload = record.get("query_plans")
            if not isinstance(query_plans_payload, list) or len(
                query_plans_payload
            ) != len(ACQUISITION_SEEDS):
                raise ValueError("tuning manifest candidate plans are invalid")
            query_plans = []
            for expected_seed, plan_record in zip(
                ACQUISITION_SEEDS, query_plans_payload
            ):
                if not isinstance(plan_record, Mapping) or set(plan_record) != {
                    "seed",
                    "case_ids",
                }:
                    raise ValueError("tuning manifest candidate plan is invalid")
                case_ids = plan_record.get("case_ids")
                if (
                    plan_record.get("seed") != expected_seed
                    or not isinstance(case_ids, list)
                    or len(case_ids) != BUDGET
                    or any(
                        not isinstance(case_id, str) or not case_id
                        for case_id in case_ids
                    )
                    or len(set(case_ids)) != BUDGET
                ):
                    raise ValueError("tuning manifest candidate plan is invalid")
                query_plans.append((expected_seed, tuple(case_ids)))
            expected_triple_hash = _query_plan_triple_digest(query_plans)
            if record.get("query_plan_triple_hash") != expected_triple_hash:
                raise ValueError("tuning manifest query plan triple hash is invalid")
            canonical_ids.append(configuration_id)
            plan_hashes[configuration_id] = expected_triple_hash

        summary = body.get("structure_summary")
        if not isinstance(summary, Mapping) or set(summary) != {
            "eligible_count",
            "excluded_count",
            "duplicate_count",
            "excluded",
            "duplicates",
        }:
            raise ValueError("tuning manifest structure summary is invalid")
        excluded = summary.get("excluded")
        duplicates = summary.get("duplicates")
        if not isinstance(excluded, list) or not isinstance(duplicates, list):
            raise ValueError("tuning manifest structure summary is invalid")

        excluded_ids = []
        for record in excluded:
            if not isinstance(record, Mapping) or set(record) != {
                "configuration_id",
                "rejection_reasons",
            }:
                raise ValueError("tuning manifest structure summary exclusion is invalid")
            configuration_id = record.get("configuration_id")
            reasons = record.get("rejection_reasons")
            if (
                not isinstance(configuration_id, str)
                or configuration_id not in all_configuration_ids
                or not isinstance(reasons, list)
                or not reasons
                or any(not isinstance(reason, str) or not reason for reason in reasons)
            ):
                raise ValueError("tuning manifest structure summary exclusion is invalid")
            excluded_ids.append(configuration_id)

        duplicate_ids = []
        for record in duplicates:
            if not isinstance(record, Mapping) or set(record) != {
                "duplicate_configuration_id",
                "canonical_configuration_id",
                "query_plan_triple_hash",
            }:
                raise ValueError("tuning manifest structure summary duplicate is invalid")
            duplicate_id = record.get("duplicate_configuration_id")
            canonical_id = record.get("canonical_configuration_id")
            if (
                not isinstance(duplicate_id, str)
                or duplicate_id not in all_configuration_ids
                or not isinstance(canonical_id, str)
                or canonical_id not in plan_hashes
                or record.get("query_plan_triple_hash") != plan_hashes[canonical_id]
            ):
                raise ValueError("tuning manifest structure summary duplicate is invalid")
            duplicate_ids.append(duplicate_id)

        eligible_ids = canonical_ids + duplicate_ids
        if (
            type(summary.get("eligible_count")) is not int
            or type(summary.get("excluded_count")) is not int
            or type(summary.get("duplicate_count")) is not int
            or summary.get("eligible_count") != len(eligible_ids)
            or summary.get("excluded_count") != len(excluded_ids)
            or summary.get("duplicate_count") != len(duplicate_ids)
            or len(eligible_ids) != len(set(eligible_ids))
            or len(excluded_ids) != len(set(excluded_ids))
            or set(eligible_ids) & set(excluded_ids)
            or set(eligible_ids) | set(excluded_ids) != all_configuration_ids
        ):
            raise ValueError("tuning manifest structure summary is inconsistent")

        batches = body.get("batches")
        if not isinstance(batches, list):
            raise ValueError("tuning manifest batches are missing")
        referenced_ids = []
        referenced_ids_by_round = {"A": [], "B": []}
        observed_round_sequence = []
        round_batch_counts = {"A": 0, "B": 0}
        for batch in batches:
            if not isinstance(batch, Mapping) or set(batch) != {
                "batch_id",
                "round_id",
                "configuration_ids",
                "configuration_references",
                "seed_unit_count",
            }:
                raise ValueError("tuning manifest batch shape is invalid")
            round_id = batch.get("round_id")
            configuration_ids = batch.get("configuration_ids")
            references = batch.get("configuration_references")
            if (
                round_id not in ("A", "B")
                or not isinstance(configuration_ids, list)
                or any(
                    not isinstance(configuration_id, str)
                    for configuration_id in configuration_ids
                )
                or len(configuration_ids) != len(set(configuration_ids))
                or not isinstance(references, list)
                or not 1 <= len(configuration_ids) <= MAX_CONFIGURATIONS_PER_BATCH
                or len(references) != len(configuration_ids)
                or type(batch.get("seed_unit_count")) is not int
                or batch.get("seed_unit_count")
                != len(configuration_ids) * len(ACQUISITION_SEEDS)
            ):
                raise ValueError("tuning manifest batch shape is invalid")
            if any(
                configuration_rounds.get(configuration_id) != round_id
                for configuration_id in configuration_ids
            ):
                raise ValueError("tuning manifest batch round is inconsistent")
            round_batch_counts[round_id] += 1
            observed_round_sequence.append(round_id)
            expected_batch_id = "round-{}-batch-{:03d}".format(
                round_id.lower(), round_batch_counts[round_id]
            )
            if batch.get("batch_id") != expected_batch_id:
                raise ValueError("tuning manifest batch identity is inconsistent")
            for configuration_id, reference in zip(configuration_ids, references):
                if (
                    not isinstance(reference, Mapping)
                    or set(reference) != {
                        "configuration_id",
                        "query_plan_triple_hash",
                    }
                    or reference.get("configuration_id") != configuration_id
                    or plan_hashes.get(configuration_id)
                    != reference.get("query_plan_triple_hash")
                ):
                    raise ValueError("tuning manifest batch plan reference is invalid")
                referenced_ids.append(configuration_id)
                referenced_ids_by_round[round_id].append(configuration_id)
        if observed_round_sequence != sorted(
            observed_round_sequence, key=lambda round_id: 0 if round_id == "A" else 1
        ):
            raise ValueError(
                "tuning manifest round sequence must contain every Round A batch before Round B"
            )
        for round_id in ("A", "B"):
            expected_round_ids = [
                configuration_id
                for configuration_id in canonical_ids
                if configuration_rounds[configuration_id] == round_id
            ]
            if referenced_ids_by_round[round_id] != expected_round_ids:
                raise ValueError(
                    "tuning manifest batch configuration references drifted from the frozen per-round label-free order"
                )
        if referenced_ids != canonical_ids:
            raise ValueError("tuning manifest batches do not cover canonical candidates")
        return manifest_sha256


def build_tuning_manifest(
    contract: TuningContract,
    evaluations: Sequence[CandidateStructureEvaluation],
    deduplication: DeduplicationResult,
) -> TuningManifest:
    """Freeze the complete 90-configuration structural decision manifest."""

    if contract != build_default_contract():
        raise ValueError("manifest requires the frozen tuning contract")
    all_configurations = contract.round_a_configurations + contract.round_b_configurations
    expected_ids = {item.configuration_id for item in all_configurations}
    evaluation_tuple = tuple(evaluations)
    if any(
        not isinstance(item, CandidateStructureEvaluation)
        for item in evaluation_tuple
    ):
        raise ValueError("manifest requires typed structure evaluations")
    for evaluation in evaluation_tuple:
        _validate_structure_evaluation(evaluation)
    observed_ids = [item.configuration.configuration_id for item in evaluation_tuple]
    if (
        len(evaluation_tuple) != len(all_configurations)
        or len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != expected_ids
    ):
        raise ValueError("manifest requires one structure evaluation for all 90 configurations")
    if not isinstance(deduplication, DeduplicationResult):
        raise ValueError("manifest requires a typed deduplication result")

    evaluation_by_id = {
        item.configuration.configuration_id: item for item in evaluation_tuple
    }
    eligible_ids = {
        configuration_id
        for configuration_id, evaluation in evaluation_by_id.items()
        if evaluation.eligible
    }
    candidate_groups = (
        deduplication.source_candidates,
        deduplication.canonical_candidates,
    )
    for candidates in candidate_groups:
        if any(not isinstance(item, TuningCandidate) for item in candidates):
            raise ValueError("manifest deduplication result contains an invalid candidate")
        for candidate in candidates:
            _validate_candidate(candidate)
            expected_evaluation = evaluation_by_id.get(
                candidate.configuration.configuration_id
            )
            if candidate.evaluation != expected_evaluation:
                raise ValueError("manifest candidate evaluation is stale or contradictory")
    source_ids = [
        item.configuration.configuration_id
        for item in deduplication.source_candidates
    ]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != eligible_ids:
        raise ValueError("manifest deduplication result does not cover eligible candidates")
    if deduplicate_query_plan_triples(
        deduplication.source_candidates
    ) != deduplication:
        raise ValueError("manifest deduplication result is stale or contradictory")

    pool_hashes = {
        item.ordered_pool_sha256 for item in deduplication.source_candidates
    }
    if len(pool_hashes) > 1:
        raise ValueError("manifest candidates disagree on ordered pool identity")
    ordered_pool_sha256 = next(iter(pool_hashes)) if pool_hashes else None

    excluded = tuple(
        ExcludedConfiguration(
            item.configuration.configuration_id, item.rejection_reasons
        )
        for item in sorted(
            (evaluation for evaluation in evaluation_tuple if not evaluation.eligible),
            key=lambda evaluation: evaluation.configuration.configuration_id,
        )
    )
    manifest = TuningManifest(
        contract=contract,
        eligible_count=len(eligible_ids),
        excluded=excluded,
        duplicate_mappings=deduplication.duplicate_mappings,
        canonical_candidates=deduplication.canonical_candidates,
        ordered_pool_sha256=ordered_pool_sha256,
        batches=build_batches(deduplication.canonical_candidates),
        provenance=TuningProvenance(),
    )
    TuningManifest.validate_payload(manifest.to_dict())
    return manifest


def _is_within(path: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _absolute_posix_path(value: Any) -> PurePosixPath:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("tuning output path must be absolute and normalized")
    return path


def _validate_lexical_allowed_followup_root(
    value: Any,
    *,
    workspace: PurePosixPath,
    first_stage: PurePosixPath,
    outputs: PurePosixPath,
) -> Optional[PurePosixPath]:
    if value is None:
        return None
    allowed = _absolute_posix_path(value)
    if (
        allowed in (workspace, outputs, first_stage)
        or not _is_within(allowed, workspace)
        or not _is_within(allowed, outputs)
        or not _is_within(allowed, first_stage)
    ):
        raise ValueError(
            "allowed follow-up root must be a strict descendant of the "
            "workspace, output root, and first-stage root"
        )
    return allowed


def validate_tuning_output_path(
    target: Any,
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> PurePosixPath:
    """Lexically reject first-stage overwrite and all output-root escapes."""

    workspace = _absolute_posix_path(isolated_workspace_root)
    first_stage = _absolute_posix_path(first_stage_root)
    outputs = _absolute_posix_path(output_root)
    candidate = _absolute_posix_path(target)
    allowed = _validate_lexical_allowed_followup_root(
        allowed_followup_root,
        workspace=workspace,
        first_stage=first_stage,
        outputs=outputs,
    )
    allowed_first_stage_overlap = (
        allowed is not None
        and candidate != allowed
        and _is_within(candidate, allowed)
    )
    if (
        not _is_within(outputs, workspace)
        or not _is_within(first_stage, workspace)
        or not _is_within(candidate, outputs)
        or candidate == outputs
        or (
            _is_within(candidate, first_stage)
            and not allowed_first_stage_overlap
        )
    ):
        raise ValueError(
            "tuning output path must stay below output_root and outside first-stage root"
        )
    return candidate


def _real_path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("tuning write path may not contain a symlink component")


def _resolve_real_directory(value: Any, field_name: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be an absolute normalized path".format(field_name))
    _reject_symlink_components(path)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("{} must be an existing real directory".format(field_name)) from exc
    if not resolved.is_dir():
        raise ValueError("{} must be an existing real directory".format(field_name))
    return resolved


def _resolve_allowed_followup_root(
    value: Any,
    *,
    workspace: Path,
    first_stage: Path,
    outputs: Path,
) -> Optional[Path]:
    if value is None:
        return None
    allowed = _resolve_real_directory(value, "allowed follow-up root")
    if (
        allowed in (workspace, outputs, first_stage)
        or not _real_path_is_within(allowed, workspace)
        or not _real_path_is_within(allowed, outputs)
        or not _real_path_is_within(allowed, first_stage)
    ):
        raise ValueError(
            "allowed follow-up root must be a strict descendant of the "
            "workspace, output root, and first-stage root"
        )
    return allowed


def validate_frozen_tuning_followup_root(
    followup_root: Any,
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any = None,
    output_root: Any = None,
) -> Path:
    workspace = _resolve_real_directory(
        isolated_workspace_root, "isolated workspace root"
    )
    followup = _resolve_real_directory(followup_root, "follow-up root")
    expected = workspace.joinpath(*FROZEN_FOLLOWUP_RELATIVE_PARTS)
    if followup != expected:
        raise ValueError(
            "follow-up root must equal the frozen workspace-relative tuning root"
        )
    if (first_stage_root is None) != (output_root is None):
        raise ValueError(
            "follow-up root validation requires both protected output roots"
        )
    if first_stage_root is not None:
        outputs = _resolve_real_directory(output_root, "tuning output root")
        first_stage = _resolve_real_directory(first_stage_root, "first-stage root")
        _resolve_allowed_followup_root(
            followup,
            workspace=workspace,
            first_stage=first_stage,
            outputs=outputs,
        )
    return followup


def validate_tuning_write_path(
    target: Any,
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> Path:
    """Validate a real exclusive-create target without writing any file.

    All existing roots and every existing target component must be real paths,
    never symlinks. The target parent must already exist so that callers can
    subsequently use an exclusive-create primitive without a path-resolution
    race introduced by directory creation.
    """

    workspace = _resolve_real_directory(
        isolated_workspace_root, "isolated workspace root"
    )
    outputs = _resolve_real_directory(output_root, "tuning output root")
    first_stage = _resolve_real_directory(first_stage_root, "first-stage root")
    allowed = _resolve_allowed_followup_root(
        allowed_followup_root,
        workspace=workspace,
        first_stage=first_stage,
        outputs=outputs,
    )
    raw_target = Path(target)
    if not raw_target.is_absolute() or ".." in raw_target.parts:
        raise ValueError("tuning write path must be absolute and normalized")
    _reject_symlink_components(raw_target)
    if raw_target.exists() or raw_target.is_symlink():
        raise ValueError("tuning write path target must not already exist")
    parent = _resolve_real_directory(raw_target.parent, "tuning target parent")
    candidate = parent / raw_target.name
    allowed_first_stage_overlap = (
        allowed is not None
        and candidate != allowed
        and _real_path_is_within(candidate, allowed)
    )
    if (
        not _real_path_is_within(outputs, workspace)
        or not _real_path_is_within(first_stage, workspace)
        or not _real_path_is_within(candidate, outputs)
        or candidate == outputs
        or (
            _real_path_is_within(candidate, first_stage)
            and not allowed_first_stage_overlap
        )
    ):
        raise ValueError(
            "tuning write path must stay below output_root and outside first-stage root"
        )
    return candidate


def _metric_decimal(
    value: Any,
    field_name: str,
    *,
    max_decimal_places: int = _BASE_METRIC_DECIMAL_PLACES
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("{} must be finite and in [0,1]".format(field_name))
    text = str(value)
    canonical_text = re.compile(
        r"^(?:0(?:\.[0-9]{{1,{0}}})?|1(?:\.0{{1,{0}}})?)$".format(
            max_decimal_places
        )
    )
    if (
        len(text) > max_decimal_places + 2
        or not canonical_text.fullmatch(text)
    ):
        raise ValueError(
            "{} must be finite and in [0,1] using canonical decimal notation with "
            "at most {} decimal places".format(field_name, max_decimal_places)
        )
    try:
        metric = Decimal(text)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("{} must be finite and in [0,1]".format(field_name)) from exc
    if not metric.is_finite() or not Decimal(0) <= metric <= Decimal(1):
        raise ValueError("{} must be finite and in [0,1]".format(field_name))
    return metric


def _decimal_mean(values: Sequence[Decimal]) -> Decimal:
    supplied = tuple(values)
    with localcontext() as context:
        context.prec = _METRIC_DECIMAL_PRECISION
        return sum(supplied, Decimal(0)) / Decimal(len(supplied))


@dataclass(frozen=True)
class TuningSeedScore:
    """One immutable outer-test score for a frozen configuration and seed."""

    configuration_id: str
    seed: int
    hit_at_1: Decimal
    hit_at_3: Decimal
    hit_at_5: Decimal
    mrr: Decimal
    top135: Decimal
    protocol: str

    def __post_init__(self) -> None:
        if not isinstance(self.configuration_id, str) or not self.configuration_id:
            raise ValueError("tuning seed score requires a configuration_id")
        if self.seed not in ACQUISITION_SEEDS or type(self.seed) is not int:
            raise ValueError("tuning seed score seed must be one of 41, 42, and 43")
        if self.protocol != EVALUATION_PROTOCOL:
            raise ValueError("tuning seed score protocol must be outer_test_guided")
        for name in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr"):
            object.__setattr__(self, name, _metric_decimal(getattr(self, name), name))
        object.__setattr__(
            self,
            "top135",
            _metric_decimal(
                self.top135,
                "top135",
                max_decimal_places=_DERIVED_METRIC_DECIMAL_PLACES,
            ),
        )
        expected_top135 = _decimal_mean(
            (self.hit_at_1, self.hit_at_3, self.hit_at_5)
        )
        if self.top135 != expected_top135:
            raise ValueError("TOP135 must equal the Decimal mean of Hit@1, Hit@3, and Hit@5")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "seed": self.seed,
            "Hit@1": format(self.hit_at_1, "f"),
            "Hit@3": format(self.hit_at_3, "f"),
            "Hit@5": format(self.hit_at_5, "f"),
            "MRR": format(self.mrr, "f"),
            "TOP135": format(self.top135, "f"),
            "protocol": self.protocol,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TuningSeedScore":
        if not isinstance(payload, Mapping) or set(payload) != {
            "configuration_id",
            "seed",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "MRR",
            "TOP135",
            "protocol",
        }:
            raise ValueError("tuning seed score payload is not canonical")
        score = TuningSeedScore(
            configuration_id=payload["configuration_id"],
            seed=payload["seed"],
            hit_at_1=payload["Hit@1"],
            hit_at_3=payload["Hit@3"],
            hit_at_5=payload["Hit@5"],
            mrr=payload["MRR"],
            top135=payload["TOP135"],
            protocol=payload["protocol"],
        )
        if score.to_dict() != dict(payload):
            raise ValueError("tuning seed score payload is not canonical")
        return score


@dataclass(frozen=True)
class TuningThreeSeedAggregate:
    configuration_id: str
    seed_scores: Tuple[TuningSeedScore, ...]
    mean_hit_at_1: Decimal = field(init=False)
    mean_hit_at_3: Decimal = field(init=False)
    mean_hit_at_5: Decimal = field(init=False)
    mean_mrr: Decimal = field(init=False)
    mean_top135: Decimal = field(init=False)

    def __post_init__(self) -> None:
        scores = tuple(sorted(tuple(self.seed_scores), key=lambda score: score.seed))
        if (
            not isinstance(self.configuration_id, str)
            or not self.configuration_id
            or len(scores) != len(ACQUISITION_SEEDS)
            or any(not isinstance(score, TuningSeedScore) for score in scores)
            or tuple(score.seed for score in scores) != ACQUISITION_SEEDS
            or any(score.configuration_id != self.configuration_id for score in scores)
        ):
            raise ValueError(
                "three-seed aggregate requires exactly one score for seeds 41, 42, and 43 for one configuration"
            )
        protocols = {score.protocol for score in scores}
        if protocols != {EVALUATION_PROTOCOL}:
            raise ValueError(
                "three-seed aggregate requires outer_test_guided scores for "
                "seeds 41, 42, and 43"
            )
        object.__setattr__(self, "seed_scores", scores)
        for output_name, score_name in (
            ("mean_hit_at_1", "hit_at_1"),
            ("mean_hit_at_3", "hit_at_3"),
            ("mean_hit_at_5", "hit_at_5"),
            ("mean_mrr", "mrr"),
            ("mean_top135", "top135"),
        ):
            object.__setattr__(
                self,
                output_name,
                _decimal_mean(tuple(getattr(score, score_name) for score in scores)),
            )
        recomputed_mean_top135 = _decimal_mean(
            (self.mean_hit_at_1, self.mean_hit_at_3, self.mean_hit_at_5)
        )
        if (
            abs(self.mean_top135 - recomputed_mean_top135)
            > _AGGREGATE_REASSOCIATION_TOLERANCE
        ):
            raise ValueError("mean TOP135 does not equal the mean of aggregate Hit@1/3/5")

    @property
    def protocol(self) -> str:
        return self.seed_scores[0].protocol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "seed_scores": [score.to_dict() for score in self.seed_scores],
            "mean_hit_at_1": format(self.mean_hit_at_1, "f"),
            "mean_hit_at_3": format(self.mean_hit_at_3, "f"),
            "mean_hit_at_5": format(self.mean_hit_at_5, "f"),
            "mean_mrr": format(self.mean_mrr, "f"),
            "mean_top135": format(self.mean_top135, "f"),
            "protocol": self.protocol,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TuningThreeSeedAggregate":
        if not isinstance(payload, Mapping) or set(payload) != {
            "configuration_id",
            "seed_scores",
            "mean_hit_at_1",
            "mean_hit_at_3",
            "mean_hit_at_5",
            "mean_mrr",
            "mean_top135",
            "protocol",
        } or not isinstance(payload.get("seed_scores"), list):
            raise ValueError("three-seed aggregate payload is not canonical")
        aggregate = aggregate_three_seed_scores(
            tuple(TuningSeedScore.from_dict(item) for item in payload["seed_scores"])
        )
        if aggregate.to_dict() != dict(payload):
            raise ValueError("three-seed aggregate payload is not canonical")
        return aggregate


def aggregate_three_seed_scores(
    scores: Sequence[TuningSeedScore],
) -> TuningThreeSeedAggregate:
    supplied = tuple(scores)
    if not supplied or not isinstance(supplied[0], TuningSeedScore):
        raise ValueError(
            "three-seed aggregate requires exactly one score for seeds 41, 42, and 43 for one configuration"
        )
    return TuningThreeSeedAggregate(supplied[0].configuration_id, supplied)


def aggregate_passes_target(aggregate: TuningThreeSeedAggregate) -> bool:
    if not isinstance(aggregate, TuningThreeSeedAggregate):
        raise ValueError("target decision requires a three-seed aggregate")
    return aggregate.mean_top135 >= TARGET_TOP135


def tuning_batch_from_manifest_record(record: Mapping[str, Any]) -> TuningBatch:
    if not isinstance(record, Mapping) or set(record) != {
        "batch_id",
        "round_id",
        "configuration_ids",
        "configuration_references",
        "seed_unit_count",
    }:
        raise ValueError("tuning manifest batch record is not canonical")
    references = record.get("configuration_references")
    if not isinstance(references, list):
        raise ValueError("tuning manifest batch record is not canonical")
    typed_references = []
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != {
            "configuration_id",
            "query_plan_triple_hash",
        }:
            raise ValueError("tuning manifest batch record is not canonical")
        typed_references.append(
            BatchConfigurationReference(
                reference["configuration_id"], reference["query_plan_triple_hash"]
            )
        )
    batch = TuningBatch(
        batch_id=record.get("batch_id"),
        round_id=record.get("round_id"),
        configuration_references=tuple(typed_references),
        seed_unit_count=record.get("seed_unit_count"),
    )
    if batch.to_dict() != dict(record):
        raise ValueError("tuning manifest batch record is not canonical")
    return batch


@dataclass(frozen=True)
class BatchEvaluation:
    batch_id: str
    round_id: str
    configuration_ids: Tuple[str, ...]
    aggregates: Tuple[TuningThreeSeedAggregate, ...]
    passing_configuration_ids: Tuple[str, ...] = field(init=False)
    winner_configuration_id: Optional[str] = field(init=False)
    schema_version: str = TUNING_RESULT_SCHEMA_VERSION
    immutable: bool = True

    def __post_init__(self) -> None:
        aggregates = tuple(sorted(tuple(self.aggregates), key=lambda item: item.configuration_id))
        if (
            not isinstance(self.batch_id, str)
            or not self.batch_id
            or self.round_id not in ("A", "B")
            or type(self.configuration_ids) is not tuple
            or not self.configuration_ids
            or len(self.configuration_ids) != len(set(self.configuration_ids))
            or any(not isinstance(item, TuningThreeSeedAggregate) for item in aggregates)
            or {item.configuration_id for item in aggregates} != set(self.configuration_ids)
            or len(aggregates) != len(self.configuration_ids)
            or any(item.protocol != EVALUATION_PROTOCOL for item in aggregates)
            or self.schema_version != TUNING_RESULT_SCHEMA_VERSION
            or self.immutable is not True
        ):
            raise ValueError(
                "complete batch evaluation must cover every configuration exactly once with outer_test_guided three-seed scores"
            )
        object.__setattr__(self, "aggregates", aggregates)
        passing = tuple(
            sorted(
                item.configuration_id
                for item in aggregates
                if aggregate_passes_target(item)
            )
        )
        object.__setattr__(self, "passing_configuration_ids", passing)
        if passing:
            by_id = {item.configuration_id: item for item in aggregates}
            winner = min(
                passing,
                key=lambda configuration_id: (
                    -by_id[configuration_id].mean_top135,
                    -by_id[configuration_id].mean_mrr,
                    -by_id[configuration_id].mean_hit_at_1,
                    configuration_id,
                ),
            )
        else:
            winner = None
        object.__setattr__(self, "winner_configuration_id", winner)

    def _body_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "immutable": self.immutable,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "target_top135": str(TARGET_TOP135),
            "batch_id": self.batch_id,
            "round_id": self.round_id,
            "configuration_ids": list(self.configuration_ids),
            "aggregates": [item.to_dict() for item in self.aggregates],
            "passing_configuration_ids": list(self.passing_configuration_ids),
            "winner_configuration_id": self.winner_configuration_id,
        }

    @property
    def batch_result_sha256(self) -> str:
        return _semantic_sha256(self._body_dict())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._body_dict()
        payload["batch_result_sha256"] = _semantic_sha256(payload)
        return payload

    @staticmethod
    def validate_payload(
        payload: Mapping[str, Any], batch: TuningBatch
    ) -> "BatchEvaluation":
        if not isinstance(payload, Mapping):
            raise ValueError("complete batch result payload is missing")
        body = dict(payload)
        observed_hash = body.pop("batch_result_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != _semantic_sha256(body):
            raise ValueError("complete batch result SHA-256 is invalid")
        if set(body) != {
            "schema_version",
            "immutable",
            "evaluation_protocol",
            "target_top135",
            "batch_id",
            "round_id",
            "configuration_ids",
            "aggregates",
            "passing_configuration_ids",
            "winner_configuration_id",
        } or not isinstance(body.get("aggregates"), list):
            raise ValueError("complete batch result payload is not canonical")
        result = BatchEvaluation(
            batch_id=body.get("batch_id"),
            round_id=body.get("round_id"),
            configuration_ids=tuple(body.get("configuration_ids", ())),
            aggregates=tuple(
                TuningThreeSeedAggregate.from_dict(item)
                for item in body.get("aggregates", ())
            ),
            schema_version=body.get("schema_version"),
            immutable=body.get("immutable"),
        )
        if (
            result.batch_id != batch.batch_id
            or result.round_id != batch.round_id
            or result.configuration_ids != batch.configuration_ids
            or result.to_dict() != dict(payload)
        ):
            raise ValueError("complete batch result contradicts the frozen batch")
        return result


def evaluate_tuning_batch(
    batch: TuningBatch, scores: Sequence[TuningSeedScore]
) -> BatchEvaluation:
    if not isinstance(batch, TuningBatch):
        raise ValueError("complete batch evaluation requires a frozen tuning batch")
    supplied = tuple(scores)
    if any(not isinstance(score, TuningSeedScore) for score in supplied):
        raise ValueError("complete batch evaluation requires typed seed scores")
    expected_units = {
        (configuration_id, seed)
        for configuration_id in batch.configuration_ids
        for seed in ACQUISITION_SEEDS
    }
    observed_units = [(score.configuration_id, score.seed) for score in supplied]
    if (
        len(observed_units) != len(expected_units)
        or len(observed_units) != len(set(observed_units))
        or set(observed_units) != expected_units
        or any(score.protocol != EVALUATION_PROTOCOL for score in supplied)
    ):
        raise ValueError(
            "complete batch evaluation must contain every frozen configuration and seed exactly once under outer_test_guided"
        )
    aggregates = tuple(
        aggregate_three_seed_scores(
            tuple(
                score
                for score in supplied
                if score.configuration_id == configuration_id
            )
        )
        for configuration_id in batch.configuration_ids
    )
    return BatchEvaluation(
        batch.batch_id, batch.round_id, batch.configuration_ids, aggregates
    )


@dataclass(frozen=True)
class TuningDecisionSentinel:
    outcome: str
    manifest_sha256: str
    completed_batch_ids: Tuple[str, ...]
    unlaunched_batch_ids: Tuple[str, ...]
    winning_batch_id: Optional[str]
    winner_configuration_id: Optional[str]
    analysis_allowed: bool
    terminal_transition_index: int
    previous_state_sha256: Optional[str]
    terminal_batch_result_sha256: Optional[str]
    terminal_state_core_sha256: str
    evaluation_protocol: str = EVALUATION_PROTOCOL
    immutable: bool = True
    fail_closed: bool = True
    schema_version: str = TUNING_SENTINEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        common_valid = (
            self.outcome in ("passed", "exhausted")
            and isinstance(self.manifest_sha256, str)
            and bool(_SHA256.fullmatch(self.manifest_sha256))
            and type(self.completed_batch_ids) is tuple
            and type(self.unlaunched_batch_ids) is tuple
            and all(
                isinstance(batch_id, str) and bool(batch_id)
                for batch_id in self.completed_batch_ids
            )
            and all(
                isinstance(batch_id, str) and bool(batch_id)
                for batch_id in self.unlaunched_batch_ids
            )
            and len(self.completed_batch_ids) == len(set(self.completed_batch_ids))
            and len(self.unlaunched_batch_ids) == len(set(self.unlaunched_batch_ids))
            and not set(self.completed_batch_ids) & set(self.unlaunched_batch_ids)
            and type(self.terminal_transition_index) is int
            and self.terminal_transition_index >= 0
            and self.terminal_transition_index == len(self.completed_batch_ids)
            and (
                (
                    self.terminal_transition_index == 0
                    and self.previous_state_sha256 is None
                    and self.terminal_batch_result_sha256 is None
                )
                or (
                    self.terminal_transition_index > 0
                    and isinstance(self.previous_state_sha256, str)
                    and bool(_SHA256.fullmatch(self.previous_state_sha256))
                    and isinstance(self.terminal_batch_result_sha256, str)
                    and bool(_SHA256.fullmatch(self.terminal_batch_result_sha256))
                )
            )
            and isinstance(self.terminal_state_core_sha256, str)
            and bool(_SHA256.fullmatch(self.terminal_state_core_sha256))
            and self.evaluation_protocol == EVALUATION_PROTOCOL
            and self.immutable is True
            and self.fail_closed is True
            and self.schema_version == TUNING_SENTINEL_SCHEMA_VERSION
        )
        passed_valid = (
            self.outcome == "passed"
            and self.analysis_allowed is True
            and bool(self.completed_batch_ids)
            and isinstance(self.winning_batch_id, str)
            and self.winning_batch_id == self.completed_batch_ids[-1]
            and isinstance(self.winner_configuration_id, str)
            and bool(self.winner_configuration_id)
        )
        exhausted_valid = (
            self.outcome == "exhausted"
            and self.analysis_allowed is False
            and self.winning_batch_id is None
            and self.winner_configuration_id is None
            and not self.unlaunched_batch_ids
        )
        if not common_valid or not (passed_valid or exhausted_valid):
            raise ValueError("tuning decision sentinel is not semantically complete")

    def _body_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "immutable": self.immutable,
            "fail_closed": self.fail_closed,
            "outcome": self.outcome,
            "manifest_sha256": self.manifest_sha256,
            "evaluation_protocol": self.evaluation_protocol,
            "analysis_allowed": self.analysis_allowed,
            "completed_batch_ids": list(self.completed_batch_ids),
            "unlaunched_batch_ids": list(self.unlaunched_batch_ids),
            "winning_batch_id": self.winning_batch_id,
            "winner_configuration_id": self.winner_configuration_id,
            "terminal_transition_index": self.terminal_transition_index,
            "previous_state_sha256": self.previous_state_sha256,
            "terminal_batch_result_sha256": self.terminal_batch_result_sha256,
            "terminal_state_core_sha256": self.terminal_state_core_sha256,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._body_dict()
        payload["sentinel_sha256"] = _semantic_sha256(payload)
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TuningDecisionSentinel":
        if not isinstance(payload, Mapping):
            raise ValueError("tuning decision sentinel is missing")
        body = dict(payload)
        observed_hash = body.pop("sentinel_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != _semantic_sha256(body):
            raise ValueError("tuning decision sentinel SHA-256 is invalid")
        if set(body) != {
            "schema_version",
            "immutable",
            "fail_closed",
            "outcome",
            "manifest_sha256",
            "evaluation_protocol",
            "analysis_allowed",
            "completed_batch_ids",
            "unlaunched_batch_ids",
            "winning_batch_id",
            "winner_configuration_id",
            "terminal_transition_index",
            "previous_state_sha256",
            "terminal_batch_result_sha256",
            "terminal_state_core_sha256",
        }:
            raise ValueError("tuning decision sentinel shape is invalid")
        sentinel = TuningDecisionSentinel(
            outcome=body.get("outcome"),
            manifest_sha256=body.get("manifest_sha256"),
            completed_batch_ids=tuple(body.get("completed_batch_ids", ())),
            unlaunched_batch_ids=tuple(body.get("unlaunched_batch_ids", ())),
            winning_batch_id=body.get("winning_batch_id"),
            winner_configuration_id=body.get("winner_configuration_id"),
            analysis_allowed=body.get("analysis_allowed"),
            terminal_transition_index=body.get("terminal_transition_index"),
            previous_state_sha256=body.get("previous_state_sha256"),
            terminal_batch_result_sha256=body.get(
                "terminal_batch_result_sha256"
            ),
            terminal_state_core_sha256=body.get("terminal_state_core_sha256"),
            evaluation_protocol=body.get("evaluation_protocol"),
            immutable=body.get("immutable"),
            fail_closed=body.get("fail_closed"),
            schema_version=body.get("schema_version"),
        )
        if sentinel.to_dict() != dict(payload):
            raise ValueError("tuning decision sentinel is not canonical")
        return sentinel


_OWNER_OVERRIDE_SELECTION_RULE = (
    "descending_mean_top135_then_mrr_then_hit1_then_configuration_id"
)


@dataclass(frozen=True)
class TuningOwnerOverride:
    manifest_sha256: str
    exhausted_state_sha256: str
    exhausted_sentinel_sha256: str
    accepted_configuration_id: str
    accepted_aggregate: TuningThreeSeedAggregate
    owner_decision_id: str
    owner_decision_date: str
    source_outcome: str = "exhausted"
    snapshot_target_top135: Decimal = TARGET_TOP135
    snapshot_target_met: bool = False
    selection_rule: str = _OWNER_OVERRIDE_SELECTION_RULE
    analysis_allowed: bool = True
    evaluation_protocol: str = EVALUATION_PROTOCOL
    immutable: bool = True
    schema_version: str = TUNING_OWNER_OVERRIDE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        valid = (
            isinstance(self.manifest_sha256, str)
            and bool(_SHA256.fullmatch(self.manifest_sha256))
            and isinstance(self.exhausted_state_sha256, str)
            and bool(_SHA256.fullmatch(self.exhausted_state_sha256))
            and isinstance(self.exhausted_sentinel_sha256, str)
            and bool(_SHA256.fullmatch(self.exhausted_sentinel_sha256))
            and isinstance(self.accepted_configuration_id, str)
            and bool(self.accepted_configuration_id)
            and isinstance(self.accepted_aggregate, TuningThreeSeedAggregate)
            and self.accepted_configuration_id
            == self.accepted_aggregate.configuration_id
            and self.accepted_aggregate.mean_top135 < TARGET_TOP135
            and isinstance(self.owner_decision_id, str)
            and bool(self.owner_decision_id)
            and isinstance(self.owner_decision_date, str)
            and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.owner_decision_date))
            and self.source_outcome == "exhausted"
            and self.snapshot_target_top135 == TARGET_TOP135
            and self.snapshot_target_met is False
            and self.selection_rule == _OWNER_OVERRIDE_SELECTION_RULE
            and self.analysis_allowed is True
            and self.evaluation_protocol == EVALUATION_PROTOCOL
            and self.immutable is True
            and self.schema_version == TUNING_OWNER_OVERRIDE_SCHEMA_VERSION
        )
        if not valid:
            raise ValueError("tuning owner override is not semantically complete")

    def _body_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "immutable": self.immutable,
            "source_outcome": self.source_outcome,
            "manifest_sha256": self.manifest_sha256,
            "exhausted_state_sha256": self.exhausted_state_sha256,
            "exhausted_sentinel_sha256": self.exhausted_sentinel_sha256,
            "evaluation_protocol": self.evaluation_protocol,
            "analysis_allowed": self.analysis_allowed,
            "accepted_configuration_id": self.accepted_configuration_id,
            "accepted_aggregate": self.accepted_aggregate.to_dict(),
            "snapshot_target_top135": format(self.snapshot_target_top135, "f"),
            "snapshot_target_met": self.snapshot_target_met,
            "selection_rule": self.selection_rule,
            "owner_decision_id": self.owner_decision_id,
            "owner_decision_date": self.owner_decision_date,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._body_dict()
        payload["owner_override_sha256"] = _semantic_sha256(payload)
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TuningOwnerOverride":
        if not isinstance(payload, Mapping):
            raise ValueError("tuning owner override is missing")
        body = dict(payload)
        observed_hash = body.pop("owner_override_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != _semantic_sha256(body):
            raise ValueError("tuning owner override SHA-256 is invalid")
        expected_fields = {
            "schema_version",
            "immutable",
            "source_outcome",
            "manifest_sha256",
            "exhausted_state_sha256",
            "exhausted_sentinel_sha256",
            "evaluation_protocol",
            "analysis_allowed",
            "accepted_configuration_id",
            "accepted_aggregate",
            "snapshot_target_top135",
            "snapshot_target_met",
            "selection_rule",
            "owner_decision_id",
            "owner_decision_date",
        }
        if set(body) != expected_fields:
            raise ValueError("tuning owner override shape is invalid")
        override = TuningOwnerOverride(
            manifest_sha256=body.get("manifest_sha256"),
            exhausted_state_sha256=body.get("exhausted_state_sha256"),
            exhausted_sentinel_sha256=body.get("exhausted_sentinel_sha256"),
            accepted_configuration_id=body.get("accepted_configuration_id"),
            accepted_aggregate=TuningThreeSeedAggregate.from_dict(
                body.get("accepted_aggregate")
            ),
            owner_decision_id=body.get("owner_decision_id"),
            owner_decision_date=body.get("owner_decision_date"),
            source_outcome=body.get("source_outcome"),
            snapshot_target_top135=_metric_decimal(
                body.get("snapshot_target_top135"), "snapshot_target_top135"
            ),
            snapshot_target_met=body.get("snapshot_target_met"),
            selection_rule=body.get("selection_rule"),
            analysis_allowed=body.get("analysis_allowed"),
            evaluation_protocol=body.get("evaluation_protocol"),
            immutable=body.get("immutable"),
            schema_version=body.get("schema_version"),
        )
        if override.to_dict() != dict(payload):
            raise ValueError("tuning owner override is not canonical")
        return override


def _completed_batch_from_payload(payload: Mapping[str, Any]) -> BatchEvaluation:
    if not isinstance(payload, Mapping):
        raise ValueError("owner override requires canonical completed batches")
    body = dict(payload)
    observed_hash = body.pop("batch_result_sha256", None)
    if not isinstance(observed_hash, str) or observed_hash != _semantic_sha256(body):
        raise ValueError("owner override completed batch SHA-256 is invalid")
    expected_fields = {
        "schema_version",
        "immutable",
        "evaluation_protocol",
        "target_top135",
        "batch_id",
        "round_id",
        "configuration_ids",
        "aggregates",
        "passing_configuration_ids",
        "winner_configuration_id",
    }
    if set(body) != expected_fields or not isinstance(body.get("aggregates"), list):
        raise ValueError("owner override completed batch shape is invalid")
    result = BatchEvaluation(
        batch_id=body.get("batch_id"),
        round_id=body.get("round_id"),
        configuration_ids=tuple(body.get("configuration_ids", ())),
        aggregates=tuple(
            TuningThreeSeedAggregate.from_dict(item)
            for item in body.get("aggregates", ())
        ),
        schema_version=body.get("schema_version"),
        immutable=body.get("immutable"),
    )
    if (
        body.get("evaluation_protocol") != EVALUATION_PROTOCOL
        or body.get("target_top135") != format(TARGET_TOP135, "f")
        or result.to_dict() != dict(payload)
        or result.passing_configuration_ids
        or result.winner_configuration_id is not None
    ):
        raise ValueError("owner override requires a canonical non-passing completed batch")
    return result


def build_exhausted_owner_override(
    exhausted_state_payload: Mapping[str, Any],
    completed_batch_payloads: Sequence[Mapping[str, Any]],
    *,
    owner_decision_id: str,
    owner_decision_date: str,
) -> TuningOwnerOverride:
    if not isinstance(exhausted_state_payload, Mapping):
        raise ValueError("owner override requires an exhausted state")
    state = dict(exhausted_state_payload)
    observed_state_hash = state.pop("state_sha256", None)
    if (
        not isinstance(observed_state_hash, str)
        or observed_state_hash != _semantic_sha256(state)
        or state.get("status") != "exhausted"
        or state.get("analysis_allowed") is not False
        or state.get("winner_configuration_id") is not None
        or state.get("unlaunched_batch_ids") != []
    ):
        raise ValueError("owner override requires a trusted exhausted state")
    sentinel = TuningDecisionSentinel.from_dict(state.get("sentinel"))
    if (
        sentinel.outcome != "exhausted"
        or sentinel.analysis_allowed is not False
        or sentinel.manifest_sha256 != state.get("manifest_sha256")
    ):
        raise ValueError("owner override requires a trusted exhausted sentinel")
    completed = tuple(
        _completed_batch_from_payload(payload) for payload in completed_batch_payloads
    )
    expected_results = state.get("completed_batch_results")
    if (
        not completed
        or not isinstance(expected_results, list)
        or [item.batch_id for item in completed] != state.get("completed_batch_ids")
        or [
            {
                "batch_id": item.batch_id,
                "batch_result_sha256": item.to_dict()["batch_result_sha256"],
            }
            for item in completed
        ]
        != expected_results
    ):
        raise ValueError("owner override completed batches contradict exhausted state")
    aggregates = tuple(
        aggregate for result in completed for aggregate in result.aggregates
    )
    if len({item.configuration_id for item in aggregates}) != len(aggregates):
        raise ValueError("owner override completed configurations must be unique")
    accepted = min(
        aggregates,
        key=lambda item: (
            -item.mean_top135,
            -item.mean_mrr,
            -item.mean_hit_at_1,
            item.configuration_id,
        ),
    )
    return TuningOwnerOverride(
        manifest_sha256=sentinel.manifest_sha256,
        exhausted_state_sha256=observed_state_hash,
        exhausted_sentinel_sha256=sentinel.to_dict()["sentinel_sha256"],
        accepted_configuration_id=accepted.configuration_id,
        accepted_aggregate=accepted,
        owner_decision_id=owner_decision_id,
        owner_decision_date=owner_decision_date,
    )


def _tuning_state_core_payload(
    *,
    manifest_sha256: str,
    status: str,
    next_batch_id: Optional[str],
    completed_batch_results: Tuple[Tuple[str, str], ...],
    unlaunched_batch_ids: Tuple[str, ...],
    analysis_allowed: bool,
    winner_configuration_id: Optional[str],
    frozen_action: str,
    transition_index: int,
    previous_state_sha256: Optional[str],
    schema_version: str = TUNING_STATE_SCHEMA_VERSION,
    immutable: bool = True,
    mutation_policy: str = "frozen_no_overwrite",
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "immutable": immutable,
        "mutation_policy": mutation_policy,
        "manifest_sha256": manifest_sha256,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "status": status,
        "frozen_action": frozen_action,
        "transition_index": transition_index,
        "previous_state_sha256": previous_state_sha256,
        "next_batch_id": next_batch_id,
        "completed_batch_ids": [item[0] for item in completed_batch_results],
        "completed_batch_results": [
            {"batch_id": batch_id, "batch_result_sha256": result_hash}
            for batch_id, result_hash in completed_batch_results
        ],
        "unlaunched_batch_ids": list(unlaunched_batch_ids),
        "analysis_allowed": analysis_allowed,
        "winner_configuration_id": winner_configuration_id,
    }


@dataclass(frozen=True)
class TuningRunState:
    manifest_sha256: str
    status: str
    next_batch_id: Optional[str]
    completed_batch_results: Tuple[Tuple[str, str], ...]
    unlaunched_batch_ids: Tuple[str, ...]
    analysis_allowed: bool
    winner_configuration_id: Optional[str]
    sentinel: Optional[TuningDecisionSentinel]
    frozen_action: str
    transition_index: int
    previous_state_sha256: Optional[str]
    schema_version: str = TUNING_STATE_SCHEMA_VERSION
    immutable: bool = True
    mutation_policy: str = "frozen_no_overwrite"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest_sha256, str)
            or not _SHA256.fullmatch(self.manifest_sha256)
            or self.status not in ("awaiting_batch_execution", "passed", "exhausted")
            or type(self.completed_batch_results) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or not _SHA256.fullmatch(item[1])
                for item in self.completed_batch_results
            )
            or len(self.completed_batch_ids) != len(set(self.completed_batch_ids))
            or type(self.unlaunched_batch_ids) is not tuple
            or any(
                not isinstance(batch_id, str) or not batch_id
                for batch_id in self.unlaunched_batch_ids
            )
            or type(self.transition_index) is not int
            or self.transition_index != len(self.completed_batch_results)
            or (
                self.transition_index == 0
                and self.previous_state_sha256 is not None
            )
            or (
                self.transition_index > 0
                and (
                    not isinstance(self.previous_state_sha256, str)
                    or not _SHA256.fullmatch(self.previous_state_sha256)
                )
            )
            or self.schema_version != TUNING_STATE_SCHEMA_VERSION
            or self.immutable is not True
            or self.mutation_policy != "frozen_no_overwrite"
        ):
            raise ValueError("tuning run state is invalid")
        awaiting_valid = (
            self.status == "awaiting_batch_execution"
            and isinstance(self.next_batch_id, str)
            and self.next_batch_id in self.unlaunched_batch_ids
            and self.analysis_allowed is False
            and self.winner_configuration_id is None
            and self.sentinel is None
            and self.frozen_action == "awaiting_external_batch_execution"
        )
        terminal_valid = (
            self.status in ("passed", "exhausted")
            and self.next_batch_id is None
            and isinstance(self.sentinel, TuningDecisionSentinel)
            and self.sentinel.outcome == self.status
            and self.sentinel.analysis_allowed is self.analysis_allowed
            and self.sentinel.winner_configuration_id
            == self.winner_configuration_id
            and self.sentinel.terminal_transition_index == self.transition_index
            and self.sentinel.previous_state_sha256 == self.previous_state_sha256
            and self.sentinel.terminal_state_core_sha256 == self.state_core_sha256
            and self.sentinel.terminal_batch_result_sha256
            == (
                self.completed_batch_results[-1][1]
                if self.completed_batch_results
                else None
            )
            and self.frozen_action == "no_further_batch_launch"
        )
        if not (awaiting_valid or terminal_valid):
            raise ValueError("tuning run state terminal/awaiting semantics are invalid")

    @property
    def completed_batch_ids(self) -> Tuple[str, ...]:
        return tuple(item[0] for item in self.completed_batch_results)

    def _core_dict(self) -> Dict[str, Any]:
        return _tuning_state_core_payload(
            manifest_sha256=self.manifest_sha256,
            status=self.status,
            next_batch_id=self.next_batch_id,
            completed_batch_results=self.completed_batch_results,
            unlaunched_batch_ids=self.unlaunched_batch_ids,
            analysis_allowed=self.analysis_allowed,
            winner_configuration_id=self.winner_configuration_id,
            frozen_action=self.frozen_action,
            transition_index=self.transition_index,
            previous_state_sha256=self.previous_state_sha256,
            schema_version=self.schema_version,
            immutable=self.immutable,
            mutation_policy=self.mutation_policy,
        )

    @property
    def state_core_sha256(self) -> str:
        return _semantic_sha256(self._core_dict())

    def _body_dict(self) -> Dict[str, Any]:
        payload = self._core_dict()
        payload["state_core_sha256"] = self.state_core_sha256
        payload["sentinel"] = (
            None if self.sentinel is None else self.sentinel.to_dict()
        )
        return payload

    def to_dict(self) -> Dict[str, Any]:
        payload = self._body_dict()
        payload["state_sha256"] = _semantic_sha256(payload)
        return payload


def _validated_manifest_batches(
    manifest_payload: Mapping[str, Any], trusted_expected_sha256: str
) -> Tuple[str, Tuple[TuningBatch, ...]]:
    trusted = TuningManifest.validate_payload(
        manifest_payload, trusted_expected_sha256=trusted_expected_sha256
    )
    return trusted, tuple(
        tuning_batch_from_manifest_record(record)
        for record in manifest_payload["batches"]
    )


def _validate_tuning_state_payload(
    payload: Mapping[str, Any],
    *,
    manifest_sha256: str,
    batches: Tuple[TuningBatch, ...]
) -> TuningRunState:
    if not isinstance(payload, Mapping):
        raise ValueError("tuning state payload is missing")
    body = dict(payload)
    observed_hash = body.pop("state_sha256", None)
    if not isinstance(observed_hash, str) or observed_hash != _semantic_sha256(body):
        raise ValueError("tuning state SHA-256 is invalid")
    expected_fields = {
        "schema_version",
        "immutable",
        "mutation_policy",
        "manifest_sha256",
        "evaluation_protocol",
        "status",
        "frozen_action",
        "transition_index",
        "previous_state_sha256",
        "next_batch_id",
        "completed_batch_ids",
        "completed_batch_results",
        "unlaunched_batch_ids",
        "analysis_allowed",
        "winner_configuration_id",
        "state_core_sha256",
        "sentinel",
    }
    results_payload = body.get("completed_batch_results")
    if set(body) != expected_fields or not isinstance(results_payload, list):
        raise ValueError("tuning state payload shape is invalid")
    completed_results = []
    for record in results_payload:
        if not isinstance(record, Mapping) or set(record) != {
            "batch_id",
            "batch_result_sha256",
        }:
            raise ValueError("tuning state completed batch record is invalid")
        completed_results.append(
            (record.get("batch_id"), record.get("batch_result_sha256"))
        )
    raw_sentinel = body.get("sentinel")
    sentinel = (
        None
        if raw_sentinel is None
        else TuningDecisionSentinel.from_dict(raw_sentinel)
    )
    state = TuningRunState(
        manifest_sha256=body.get("manifest_sha256"),
        status=body.get("status"),
        next_batch_id=body.get("next_batch_id"),
        completed_batch_results=tuple(completed_results),
        unlaunched_batch_ids=tuple(body.get("unlaunched_batch_ids", ())),
        analysis_allowed=body.get("analysis_allowed"),
        winner_configuration_id=body.get("winner_configuration_id"),
        sentinel=sentinel,
        frozen_action=body.get("frozen_action"),
        transition_index=body.get("transition_index"),
        previous_state_sha256=body.get("previous_state_sha256"),
        schema_version=body.get("schema_version"),
        immutable=body.get("immutable"),
        mutation_policy=body.get("mutation_policy"),
    )
    all_batch_ids = tuple(batch.batch_id for batch in batches)
    completed_count = len(state.completed_batch_ids)
    semantic_valid = (
        body.get("evaluation_protocol") == EVALUATION_PROTOCOL
        and state.manifest_sha256 == manifest_sha256
        and state.completed_batch_ids == all_batch_ids[:completed_count]
        and state.unlaunched_batch_ids == all_batch_ids[completed_count:]
        and body.get("completed_batch_ids") == list(state.completed_batch_ids)
        and body.get("state_core_sha256") == state.state_core_sha256
        and state.transition_index == completed_count
    )
    if state.status == "awaiting_batch_execution":
        semantic_valid = semantic_valid and bool(state.unlaunched_batch_ids) and (
            state.next_batch_id == state.unlaunched_batch_ids[0]
        )
    elif state.status == "exhausted":
        semantic_valid = semantic_valid and completed_count == len(all_batch_ids)
    else:
        semantic_valid = semantic_valid and bool(completed_count)
    if state.sentinel is not None:
        semantic_valid = semantic_valid and (
            state.sentinel.manifest_sha256 == manifest_sha256
            and state.sentinel.completed_batch_ids == state.completed_batch_ids
            and state.sentinel.unlaunched_batch_ids == state.unlaunched_batch_ids
        )
    if not semantic_valid or state.to_dict() != dict(payload):
        raise ValueError("tuning state contradicts the trusted manifest")
    return state


def initialize_tuning_state(
    manifest_payload: Mapping[str, Any], *, trusted_expected_sha256: str
) -> TuningRunState:
    trusted, batches = _validated_manifest_batches(
        manifest_payload, trusted_expected_sha256
    )
    batch_ids = tuple(batch.batch_id for batch in batches)
    if batch_ids:
        return TuningRunState(
            manifest_sha256=trusted,
            status="awaiting_batch_execution",
            next_batch_id=batch_ids[0],
            completed_batch_results=(),
            unlaunched_batch_ids=batch_ids,
            analysis_allowed=False,
            winner_configuration_id=None,
            sentinel=None,
            frozen_action="awaiting_external_batch_execution",
            transition_index=0,
            previous_state_sha256=None,
        )
    empty_core_sha256 = _semantic_sha256(
        _tuning_state_core_payload(
            manifest_sha256=trusted,
            status="exhausted",
            next_batch_id=None,
            completed_batch_results=(),
            unlaunched_batch_ids=(),
            analysis_allowed=False,
            winner_configuration_id=None,
            frozen_action="no_further_batch_launch",
            transition_index=0,
            previous_state_sha256=None,
        )
    )
    sentinel = TuningDecisionSentinel(
        outcome="exhausted",
        manifest_sha256=trusted,
        completed_batch_ids=(),
        unlaunched_batch_ids=(),
        winning_batch_id=None,
        winner_configuration_id=None,
        analysis_allowed=False,
        terminal_transition_index=0,
        previous_state_sha256=None,
        terminal_batch_result_sha256=None,
        terminal_state_core_sha256=empty_core_sha256,
    )
    return TuningRunState(
        manifest_sha256=trusted,
        status="exhausted",
        next_batch_id=None,
        completed_batch_results=(),
        unlaunched_batch_ids=(),
        analysis_allowed=False,
        winner_configuration_id=None,
        sentinel=sentinel,
        frozen_action="no_further_batch_launch",
        transition_index=0,
        previous_state_sha256=None,
    )


def advance_tuning_state(
    manifest_payload: Mapping[str, Any],
    state_payload: Mapping[str, Any],
    completed_batch_result_payload: Mapping[str, Any],
    *,
    trusted_expected_sha256: str,
    trusted_expected_state_sha256: str,
    trusted_expected_batch_result_sha256: str
) -> TuningRunState:
    trusted, batches = _validated_manifest_batches(
        manifest_payload, trusted_expected_sha256
    )
    if (
        not isinstance(trusted_expected_state_sha256, str)
        or not _SHA256.fullmatch(trusted_expected_state_sha256)
        or not isinstance(state_payload, Mapping)
        or state_payload.get("state_sha256") != trusted_expected_state_sha256
    ):
        raise ValueError("tuning state does not match the trusted state SHA-256")
    if (
        not isinstance(trusted_expected_batch_result_sha256, str)
        or not _SHA256.fullmatch(trusted_expected_batch_result_sha256)
        or not isinstance(completed_batch_result_payload, Mapping)
        or completed_batch_result_payload.get("batch_result_sha256")
        != trusted_expected_batch_result_sha256
    ):
        raise ValueError(
            "completed batch result does not match the trusted batch result SHA-256"
        )
    state = _validate_tuning_state_payload(
        state_payload, manifest_sha256=trusted, batches=batches
    )
    batches_by_id = {batch.batch_id: batch for batch in batches}
    if not isinstance(completed_batch_result_payload, Mapping):
        raise ValueError("completed batch result is missing")
    result_batch_id = completed_batch_result_payload.get("batch_id")
    batch = batches_by_id.get(result_batch_id)
    if batch is None:
        raise ValueError("completed result refers to a later batch outside the manifest")
    completed_hashes = dict(state.completed_batch_results)
    result_body = dict(completed_batch_result_payload)
    observed_result_hash = result_body.pop("batch_result_sha256", None)
    if (
        not isinstance(observed_result_hash, str)
        or observed_result_hash != _semantic_sha256(result_body)
    ):
        raise ValueError("completed batch result SHA-256 is invalid")
    if result_batch_id in completed_hashes:
        if completed_hashes[result_batch_id] == observed_result_hash:
            return state
        raise ValueError("a completed batch result cannot be rerun or overwritten")
    result = BatchEvaluation.validate_payload(completed_batch_result_payload, batch)
    if state.status != "awaiting_batch_execution":
        raise ValueError("terminal tuning state rejects every later batch result")
    if result.batch_id != state.next_batch_id:
        raise ValueError("only the next manifest-ordered batch may be completed")

    completed = state.completed_batch_results + (
        (result.batch_id, result.batch_result_sha256),
    )
    transition_index = state.transition_index + 1
    previous_state_sha256 = trusted_expected_state_sha256
    all_batch_ids = tuple(item.batch_id for item in batches)
    remaining = all_batch_ids[len(completed):]
    if result.winner_configuration_id is not None:
        passed_core_sha256 = _semantic_sha256(
            _tuning_state_core_payload(
                manifest_sha256=trusted,
                status="passed",
                next_batch_id=None,
                completed_batch_results=completed,
                unlaunched_batch_ids=remaining,
                analysis_allowed=True,
                winner_configuration_id=result.winner_configuration_id,
                frozen_action="no_further_batch_launch",
                transition_index=transition_index,
                previous_state_sha256=previous_state_sha256,
            )
        )
        sentinel = TuningDecisionSentinel(
            outcome="passed",
            manifest_sha256=trusted,
            completed_batch_ids=tuple(item[0] for item in completed),
            unlaunched_batch_ids=remaining,
            winning_batch_id=result.batch_id,
            winner_configuration_id=result.winner_configuration_id,
            analysis_allowed=True,
            terminal_transition_index=transition_index,
            previous_state_sha256=previous_state_sha256,
            terminal_batch_result_sha256=result.batch_result_sha256,
            terminal_state_core_sha256=passed_core_sha256,
        )
        return TuningRunState(
            manifest_sha256=trusted,
            status="passed",
            next_batch_id=None,
            completed_batch_results=completed,
            unlaunched_batch_ids=remaining,
            analysis_allowed=True,
            winner_configuration_id=result.winner_configuration_id,
            sentinel=sentinel,
            frozen_action="no_further_batch_launch",
            transition_index=transition_index,
            previous_state_sha256=previous_state_sha256,
        )
    if remaining:
        return TuningRunState(
            manifest_sha256=trusted,
            status="awaiting_batch_execution",
            next_batch_id=remaining[0],
            completed_batch_results=completed,
            unlaunched_batch_ids=remaining,
            analysis_allowed=False,
            winner_configuration_id=None,
            sentinel=None,
            frozen_action="awaiting_external_batch_execution",
            transition_index=transition_index,
            previous_state_sha256=previous_state_sha256,
        )
    exhausted_core_sha256 = _semantic_sha256(
        _tuning_state_core_payload(
            manifest_sha256=trusted,
            status="exhausted",
            next_batch_id=None,
            completed_batch_results=completed,
            unlaunched_batch_ids=(),
            analysis_allowed=False,
            winner_configuration_id=None,
            frozen_action="no_further_batch_launch",
            transition_index=transition_index,
            previous_state_sha256=previous_state_sha256,
        )
    )
    sentinel = TuningDecisionSentinel(
        outcome="exhausted",
        manifest_sha256=trusted,
        completed_batch_ids=tuple(item[0] for item in completed),
        unlaunched_batch_ids=(),
        winning_batch_id=None,
        winner_configuration_id=None,
        analysis_allowed=False,
        terminal_transition_index=transition_index,
        previous_state_sha256=previous_state_sha256,
        terminal_batch_result_sha256=result.batch_result_sha256,
        terminal_state_core_sha256=exhausted_core_sha256,
    )
    return TuningRunState(
        manifest_sha256=trusted,
        status="exhausted",
        next_batch_id=None,
        completed_batch_results=completed,
        unlaunched_batch_ids=(),
        analysis_allowed=False,
        winner_configuration_id=None,
        sentinel=sentinel,
        frozen_action="no_further_batch_launch",
        transition_index=transition_index,
        previous_state_sha256=previous_state_sha256,
    )


def validate_analysis_launch_guard(
    sentinel_payload: Any,
    *,
    trusted_expected_manifest_sha256: str,
    trusted_expected_sentinel_sha256: str,
    owner_override_payload: Any = None,
    trusted_expected_owner_override_sha256: Optional[str] = None,
    isolated_workspace_root: Any = None,
    first_stage_root: Any = None,
    output_root: Any = None,
    allowed_followup_root: Any = None,
) -> str:
    if not isinstance(sentinel_payload, Mapping):
        if any(
            value is None
            for value in (
                isolated_workspace_root,
                first_stage_root,
                output_root,
            )
        ):
            raise ValueError(
                "analysis launch guard requires protected roots for a sentinel file"
            )
        sentinel_payload = read_tuning_json(
            sentinel_payload,
            isolated_workspace_root=isolated_workspace_root,
            first_stage_root=first_stage_root,
            output_root=output_root,
            allowed_followup_root=allowed_followup_root,
        )
    if (
        not isinstance(sentinel_payload, Mapping)
        or sentinel_payload.get("sentinel_sha256")
        != trusted_expected_sentinel_sha256
    ):
        raise ValueError("analysis launch guard rejected the trusted sentinel SHA-256")
    try:
        sentinel = TuningDecisionSentinel.from_dict(sentinel_payload)
    except (ValueError, TypeError) as exc:
        raise ValueError("analysis launch guard rejected an invalid sentinel") from exc
    if sentinel.manifest_sha256 != trusted_expected_manifest_sha256:
        raise ValueError("analysis launch guard rejected the trusted manifest SHA-256")
    passed = (
        sentinel.outcome != "passed"
        or sentinel.analysis_allowed is not True
        or sentinel.evaluation_protocol != EVALUATION_PROTOCOL
        or sentinel.immutable is not True
        or sentinel.fail_closed is not True
        or not sentinel.winner_configuration_id
    ) is False
    if passed:
        return sentinel.winner_configuration_id
    if sentinel.outcome != "exhausted" or sentinel.analysis_allowed is not False:
        raise ValueError(
            "analysis launch guard requires a complete passed sentinel or trusted owner override"
        )
    if (
        not isinstance(owner_override_payload, Mapping)
        or not isinstance(trusted_expected_owner_override_sha256, str)
        or owner_override_payload.get("owner_override_sha256")
        != trusted_expected_owner_override_sha256
    ):
        raise ValueError(
            "analysis launch guard requires a complete passed sentinel or trusted owner override"
        )
    try:
        owner_override = TuningOwnerOverride.from_dict(owner_override_payload)
    except (ValueError, TypeError) as exc:
        raise ValueError("analysis launch guard rejected an invalid owner override") from exc
    if (
        owner_override.manifest_sha256 != trusted_expected_manifest_sha256
        or owner_override.exhausted_sentinel_sha256
        != trusted_expected_sentinel_sha256
        or owner_override.source_outcome != sentinel.outcome
        or owner_override.analysis_allowed is not True
        or owner_override.snapshot_target_met is not False
        or owner_override.evaluation_protocol != EVALUATION_PROTOCOL
    ):
        raise ValueError("analysis launch guard rejected the trusted owner override")
    return owner_override.accepted_configuration_id


def validate_tuning_read_path(
    target: Any,
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> Path:
    workspace = _resolve_real_directory(
        isolated_workspace_root, "isolated workspace root"
    )
    outputs = _resolve_real_directory(output_root, "tuning output root")
    first_stage = _resolve_real_directory(first_stage_root, "first-stage root")
    allowed = _resolve_allowed_followup_root(
        allowed_followup_root,
        workspace=workspace,
        first_stage=first_stage,
        outputs=outputs,
    )
    raw_target = Path(target)
    if not raw_target.is_absolute() or ".." in raw_target.parts:
        raise ValueError("tuning manifest read path must be absolute and normalized")
    _reject_symlink_components(raw_target)
    try:
        candidate = raw_target.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("tuning manifest read path must be an existing real file") from exc
    allowed_first_stage_overlap = (
        allowed is not None
        and candidate != allowed
        and _real_path_is_within(candidate, allowed)
    )
    if (
        not candidate.is_file()
        or not _real_path_is_within(outputs, workspace)
        or not _real_path_is_within(first_stage, workspace)
        or not _real_path_is_within(candidate, outputs)
        or (
            _real_path_is_within(candidate, first_stage)
            and not allowed_first_stage_overlap
        )
    ):
        raise ValueError(
            "tuning manifest read path must stay below output_root and outside first-stage root"
        )
    return candidate


def read_tuning_json(
    target: Any,
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> Mapping[str, Any]:
    path = validate_tuning_read_path(
        target,
        isolated_workspace_root=isolated_workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
        allowed_followup_root=allowed_followup_root,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("tuning JSON input must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("tuning JSON input must contain an object")
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload, Mapping):
        raise ValueError("canonical JSON output requires an object")
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical JSON output contains an unsupported value") from exc


def _fsync_directory(directory: Path) -> None:
    descriptor = None
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        # Directory fsync is not supported by every local filesystem (notably
        # some Windows filesystems); the atomic hard-link remains authoritative.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_canonical_json_exclusive(
    target: Any,
    payload: Mapping[str, Any],
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> Path:
    """Atomically publish canonical JSON without ever replacing a target."""

    candidate = validate_tuning_write_path(
        target,
        isolated_workspace_root=isolated_workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
        allowed_followup_root=allowed_followup_root,
    )
    encoded = _canonical_json_bytes(payload)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb", prefix=".tuning-json-", dir=candidate.parent, delete=False
        ) as temporary:
            temporary_name = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        if _resolve_real_directory(
            Path(target).parent, "tuning target parent"
        ) != candidate.parent:
            raise ValueError("tuning target parent changed during publication")
        try:
            os.link(temporary_name, candidate)
        except FileExistsError as exc:
            raise ValueError("tuning write path target must not already exist") from exc
        _fsync_directory(candidate.parent)
    finally:
        if temporary_name is not None:
            try:
                temporary_name.unlink()
            except FileNotFoundError:
                pass
    return candidate


@dataclass(frozen=True)
class CanonicalJsonPublication:
    path: Path
    status: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or self.status not in ("created", "reused")
            or not isinstance(self.canonical_sha256, str)
            or not _SHA256.fullmatch(self.canonical_sha256)
        ):
            raise ValueError("canonical JSON publication result is invalid")


def _validate_tuning_publication_path(
    target: Any,
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> Path:
    workspace = _resolve_real_directory(
        isolated_workspace_root, "isolated workspace root"
    )
    outputs = _resolve_real_directory(output_root, "tuning output root")
    first_stage = _resolve_real_directory(first_stage_root, "first-stage root")
    allowed = _resolve_allowed_followup_root(
        allowed_followup_root,
        workspace=workspace,
        first_stage=first_stage,
        outputs=outputs,
    )
    raw_target = Path(target)
    if not raw_target.is_absolute() or ".." in raw_target.parts:
        raise ValueError("tuning publication path must be absolute and normalized")
    _reject_symlink_components(raw_target)
    parent = _resolve_real_directory(raw_target.parent, "tuning target parent")
    candidate = parent / raw_target.name
    allowed_first_stage_overlap = (
        allowed is not None
        and candidate != allowed
        and _real_path_is_within(candidate, allowed)
    )
    if (
        not _real_path_is_within(outputs, workspace)
        or not _real_path_is_within(first_stage, workspace)
        or not _real_path_is_within(candidate, outputs)
        or candidate == outputs
        or (
            _real_path_is_within(candidate, first_stage)
            and not allowed_first_stage_overlap
        )
    ):
        raise ValueError(
            "tuning publication path must stay below output_root and outside first-stage root"
        )
    if candidate.exists() and not candidate.is_file():
        raise ValueError("tuning publication target must be a regular file")
    return candidate


def preflight_canonical_json_idempotent(
    target: Any,
    payload: Mapping[str, Any],
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> Optional[CanonicalJsonPublication]:
    candidate = _validate_tuning_publication_path(
        target,
        isolated_workspace_root=isolated_workspace_root,
        first_stage_root=first_stage_root,
        output_root=output_root,
        allowed_followup_root=allowed_followup_root,
    )
    expected_bytes = _canonical_json_bytes(payload)
    semantic_sha256 = _semantic_sha256(payload)
    if not candidate.exists():
        return None
    try:
        observed_bytes = candidate.read_bytes()
        observed_payload = json.loads(observed_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "tuning write path target must not already exist: existing canonical JSON "
            "publication conflicts with expected content"
        ) from exc
    if (
        observed_bytes != expected_bytes
        or not isinstance(observed_payload, Mapping)
        or _semantic_sha256(observed_payload) != semantic_sha256
    ):
        raise ValueError("existing canonical JSON publication has different content")
    return CanonicalJsonPublication(candidate, "reused", semantic_sha256)


def write_canonical_json_idempotent(
    target: Any,
    payload: Mapping[str, Any],
    *,
    isolated_workspace_root: Any,
    first_stage_root: Any,
    output_root: Any,
    allowed_followup_root: Any = None,
) -> CanonicalJsonPublication:
    roots = {
        "isolated_workspace_root": isolated_workspace_root,
        "first_stage_root": first_stage_root,
        "output_root": output_root,
        "allowed_followup_root": allowed_followup_root,
    }
    existing = preflight_canonical_json_idempotent(target, payload, **roots)
    if existing is not None:
        return existing
    try:
        path = write_canonical_json_exclusive(target, payload, **roots)
        return CanonicalJsonPublication(path, "created", _semantic_sha256(payload))
    except ValueError as exc:
        raced = preflight_canonical_json_idempotent(target, payload, **roots)
        if raced is not None:
            return raced
        raise exc


def build_batch_execution_index(
    manifest_payload: Mapping[str, Any], *, trusted_expected_sha256: str
) -> Dict[str, Any]:
    trusted, batches = _validated_manifest_batches(
        manifest_payload, trusted_expected_sha256
    )
    body = {
        "schema_version": TUNING_INDEX_SCHEMA_VERSION,
        "immutable": True,
        "mutation_policy": "frozen_no_overwrite",
        "execution_mode": "frozen_batches_only_no_rcl_launch",
        "manifest_sha256": trusted,
        "method_identity": METHOD_IDENTITY,
        "acquisition_seeds": list(ACQUISITION_SEEDS),
        "budget": BUDGET,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "round_id": batch.round_id,
                "configuration_ids": list(batch.configuration_ids),
                "seed_unit_count": batch.seed_unit_count,
                "launch_status": "frozen_not_launched",
            }
            for batch in batches
        ],
    }
    body["batch_execution_index_sha256"] = _semantic_sha256(body)
    return body


__all__ = [
    "ACQUISITION_SEEDS",
    "ALPHA",
    "ALLOW_SINGLE_CLUSTER",
    "BatchEvaluation",
    "BatchConfigurationReference",
    "CanonicalJsonPublication",
    "BUDGET",
    "CANDIDATE_POOL_SIZE",
    "CLUSTER_SELECTION_EPSILON",
    "CandidateStructureEvaluation",
    "DATASET_ID",
    "DOWNSTREAM_SEED",
    "DeduplicationResult",
    "DuplicatePlanMapping",
    "EFFECTIVE_SUPPORT",
    "EVALUATION_PROTOCOL",
    "ExcludedConfiguration",
    "FROZEN_FOLLOWUP_RELATIVE_PARTS",
    "METHOD_IDENTITY",
    "METRIC",
    "PARAMETER_DISTANCE_DEFINITION",
    "STORE_CENTERS",
    "StructureSeedSummary",
    "TARGET_TOP135",
    "TuningBatch",
    "TuningCandidate",
    "TuningConfiguration",
    "TuningContract",
    "TuningManifest",
    "TuningProvenance",
    "TuningDecisionSentinel",
    "TuningOwnerOverride",
    "TuningRunState",
    "TuningSeedScore",
    "TuningThreeSeedAggregate",
    "advance_tuning_state",
    "aggregate_passes_target",
    "aggregate_three_seed_scores",
    "assess_candidate_structure",
    "build_exhausted_owner_override",
    "build_batches",
    "build_default_contract",
    "build_tuning_candidate",
    "build_tuning_manifest",
    "build_batch_execution_index",
    "deduplicate_query_plan_triples",
    "evaluate_configuration_structures",
    "evaluate_tuning_batch",
    "generate_round_a_grid",
    "generate_round_b_grid",
    "label_free_sort_key",
    "rank_structure_evaluations",
    "read_tuning_json",
    "preflight_canonical_json_idempotent",
    "tuning_batch_from_manifest_record",
    "validate_analysis_launch_guard",
    "validate_frozen_tuning_followup_root",
    "validate_tuning_output_path",
    "validate_tuning_read_path",
    "validate_tuning_write_path",
    "write_canonical_json_exclusive",
    "write_canonical_json_idempotent",
]
