"""Fixed-winner selector, semantic, and AULC analysis primitives.

The winner follow-up is append-only.  It preserves the owner-accepted tuning
center plan as the traversal anchor, derives boundary and within-cluster-random
plans from the same HDBSCAN geometry and allocation, and opens labels only in
later analysis functions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from rcl_study.combined_active_learning_acquisition import (
    MAX_RESIDUAL_QUERIES,
    QUOTA_POLICY,
    ClusterAllocationStep,
    ClusterQueryPlan,
    ClusterQueryRecord,
    MatchedClusterAllocation,
    build_hdbscan_selector_queues,
    build_within_cluster_random_queues,
    assert_matched_selector_contracts,
)
from rcl_study.combined_active_learning_clustering import (
    EffectiveClusterPartition,
    HDBSCANClusteringResult,
    build_effective_cluster_partition,
)
from rcl_study.combined_active_learning_representation import (
    BalancedModalityBlocks,
    RepresentationMatrixCandidate,
    fuse_balanced_blocks,
)
from rcl_study.combined_active_learning_rcl import (
    Budget30RclBridge,
    validate_budget30_rcl_bridge,
)
from rcl_study.combined_active_learning_schemas import (
    ACTIVE_LEARNING_SEEDS,
    ANNOTATION_BUDGET,
    semantic_sha256,
)
from rcl_study.combined_active_learning_tuning_structure import (
    FittedHDBSCANGeometry,
)


WINNER_SELECTORS = ("center", "boundary", "within_cluster_random")
AULC_BUDGETS = (8, 16, 24, 30)
AULC_GAP_TRIGGER = 0.02


@dataclass(frozen=True)
class WinnerMatchedPlanBundle:
    native_result: HDBSCANClusteringResult
    partition: EffectiveClusterPartition
    allocation: MatchedClusterAllocation
    plans: Mapping[str, ClusterQueryPlan]


@dataclass(frozen=True)
class JaccardClusterMatch:
    reference_label: int | None
    target_label: int | None
    intersection_size: int
    union_size: int
    jaccard: float


@dataclass(frozen=True)
class AulcTriggerDecision:
    triggered: bool
    seed_winner_changed: bool
    winner_by_seed: tuple[tuple[int, str], ...]
    mean_top135_by_selector: tuple[tuple[str, float], ...]
    top_two_mean_gap: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class WinnerSelectorFreeze:
    manifest: Mapping[str, Any]
    new_units: tuple[Mapping[str, Any], ...]

    @property
    def new_unit_count(self) -> int:
        return len(self.new_units)

    @property
    def aiops_center_reuse_count(self) -> int:
        return len(self.manifest["aiops2022_pre"]["center_reuse_unit_ids"])

    @property
    def rcabench_reuse_count(self) -> int:
        return len(self.manifest["rcabench"]["reused_unit_ids"])


def to_native_hdbscan_result(
    *,
    fitted_geometry: FittedHDBSCANGeometry,
    candidate: RepresentationMatrixCandidate,
    dataset_id: str,
    active_learning_seed: int,
) -> HDBSCANClusteringResult:
    """Project the tuning fitter result into the common native HDBSCAN type."""

    if type(fitted_geometry) is not FittedHDBSCANGeometry:
        raise ValueError("winner conversion requires fitted HDBSCAN geometry")
    if type(candidate) is not RepresentationMatrixCandidate:
        raise ValueError("winner conversion requires a formal representation")
    if active_learning_seed not in ACTIVE_LEARNING_SEEDS:
        raise ValueError("winner conversion seed must be 41, 42, or 43")
    if (
        tuple(candidate.case_ids) != fitted_geometry.case_ids
        or candidate.matrix_sha256
        != fitted_geometry.representation_matrix_sha256
        or candidate.representation_id != "global_pca_dim32"
    ):
        raise ValueError("winner geometry and representation drifted")
    labels = tuple(int(value) for value in fitted_geometry.labels)
    non_noise_labels = tuple(sorted({label for label in labels if label >= 0}))
    non_noise_count = sum(label >= 0 for label in labels)
    largest = max(
        (sum(label == current for label in labels) for current in non_noise_labels),
        default=0,
    )
    configuration = fitted_geometry.configuration
    return HDBSCANClusteringResult(
        dataset_id=dataset_id,
        active_learning_seed=active_learning_seed,
        min_cluster_size=configuration.min_cluster_size,
        min_samples=configuration.min_samples,
        cluster_selection_method=configuration.cluster_selection_method,
        representation_id=candidate.representation_id,
        representation_matrix_sha256=candidate.matrix_sha256,
        case_ids=tuple(candidate.case_ids),
        labels=labels,
        membership_strengths=tuple(
            float(value) for value in fitted_geometry.membership_strengths
        ),
        medoid_cluster_labels=tuple(
            int(value) for value in fitted_geometry.medoid_cluster_labels
        ),
        medoids=np.asarray(fitted_geometry.medoids, dtype=float),
        noise_case_ids=tuple(
            case_id
            for case_id, label in zip(candidate.case_ids, labels)
            if label < 0
        ),
        non_noise_cluster_count=len(non_noise_labels),
        non_noise_case_count=non_noise_count,
        noise_count=len(labels) - non_noise_count,
        non_noise_coverage=non_noise_count / len(labels),
        largest_non_noise_cluster_share=largest / len(labels),
        backend=fitted_geometry.backend,
    )


def _anchored_allocation(
    *,
    partition: EffectiveClusterPartition,
    frozen_center_case_ids: Sequence[str],
) -> tuple[MatchedClusterAllocation, tuple[str, ...]]:
    selected = tuple(frozen_center_case_ids)
    if (
        len(selected) != ANNOTATION_BUDGET
        or len(set(selected)) != ANNOTATION_BUDGET
        or not set(selected).issubset(partition.case_ids)
    ):
        raise ValueError("frozen tuning center plan must contain 30 unique pool cases")
    effective_labels = {cluster.raw_label for cluster in partition.effective_clusters}
    if len(effective_labels) < 2:
        raise ValueError("winner selector analysis requires at least two clusters")
    label_by_case = dict(zip(partition.case_ids, partition.effective_labels))
    first_round = tuple(label_by_case[case_id] for case_id in selected[: len(effective_labels)])
    if (
        any(label not in effective_labels for label in first_round)
        or len(set(first_round)) != len(effective_labels)
        or set(first_round) != effective_labels
    ):
        raise ValueError("frozen center plan does not begin with one full cluster round")
    expected_residual = min(
        MAX_RESIDUAL_QUERIES,
        len(partition.residual_case_ids),
        ANNOTATION_BUDGET - len(first_round),
    )
    residual_slice = selected[
        len(first_round) : len(first_round) + expected_residual
    ]
    residual_set = set(partition.residual_case_ids)
    if any(case_id not in residual_set for case_id in residual_slice):
        raise ValueError("frozen center plan residual positions drifted")
    remaining = selected[len(first_round) + expected_residual :]
    if any(label_by_case[case_id] not in effective_labels for case_id in remaining):
        raise ValueError("frozen center plan has residual cases after the residual block")
    effective_sequence = tuple(
        label_by_case[case_id]
        for case_id in selected
        if label_by_case[case_id] in effective_labels
    )
    expected_sequence = tuple(
        first_round[index % len(first_round)]
        for index in range(len(effective_sequence))
    )
    if effective_sequence != expected_sequence:
        raise ValueError("frozen center plan is not equal-weight round-robin")

    occurrences: Counter[int] = Counter()
    steps: list[ClusterAllocationStep] = []
    for query_index, case_id in enumerate(selected):
        raw_label = label_by_case[case_id]
        if raw_label in effective_labels:
            round_index = occurrences[raw_label]
            occurrences[raw_label] += 1
            steps.append(
                ClusterAllocationStep(
                    query_index=query_index,
                    allocation_kind="effective_cluster",
                    raw_label=raw_label,
                    effective_round_index=round_index,
                )
            )
        else:
            steps.append(
                ClusterAllocationStep(
                    query_index=query_index,
                    allocation_kind="residual",
                    raw_label=None,
                    effective_round_index=None,
                )
            )
    allocation = MatchedClusterAllocation(
        dataset_id=partition.dataset_id,
        clusterer_id=partition.clusterer_id,
        active_learning_seed=partition.active_learning_seed,
        partition_sha256=partition.partition_sha256,
        budget=ANNOTATION_BUDGET,
        quota_policy=QUOTA_POLICY,
        cluster_visit_order=first_round,
        effective_cluster_quotas=tuple(sorted(occurrences.items())),
        residual_quota=expected_residual,
        steps=tuple(steps),
    )
    return allocation, residual_slice


def build_tuning_anchored_matched_hdbscan_plans(
    *,
    fitted_geometry: FittedHDBSCANGeometry,
    candidate: RepresentationMatrixCandidate,
    active_learning_seed: int,
    frozen_center_case_ids: Sequence[str],
) -> WinnerMatchedPlanBundle:
    """Keep the accepted center plan and match two alternative selectors to it."""

    native = to_native_hdbscan_result(
        fitted_geometry=fitted_geometry,
        candidate=candidate,
        dataset_id="aiops2022_pre",
        active_learning_seed=active_learning_seed,
    )
    partition = build_effective_cluster_partition(
        dataset_id=native.dataset_id,
        clusterer_id=native.clusterer_id,
        active_learning_seed=active_learning_seed,
        case_ids=native.case_ids,
        raw_labels=native.labels,
        source_geometry_sha256=native.geometry_sha256,
    )
    allocation, residual_case_ids = _anchored_allocation(
        partition=partition,
        frozen_center_case_ids=frozen_center_case_ids,
    )
    queues = build_hdbscan_selector_queues(
        partition=partition,
        result=native,
        candidate=candidate,
    )
    queues["within_cluster_random"] = build_within_cluster_random_queues(
        partition=partition,
        representation_matrix_sha256=candidate.matrix_sha256,
    )
    contracts = assert_matched_selector_contracts(
        partition=partition,
        allocation=allocation,
        selector_queues=queues,
    )
    raw_label_by_case = dict(zip(partition.case_ids, partition.raw_labels))
    residual_rank = {case_id: index for index, case_id in enumerate(residual_case_ids, 1)}
    plans: dict[str, ClusterQueryPlan] = {}
    for selector_id in WINNER_SELECTORS:
        contract = contracts[selector_id]
        input_sha256 = semantic_sha256(
            {
                "role": "tuning-center-anchored-winner-selector-plan",
                "selector_id": selector_id,
                "anchor_center_case_ids": list(frozen_center_case_ids),
                "allocation_sha256": allocation.allocation_sha256,
                "queues_sha256": contract.queues_sha256,
            }
        )
        offsets: Counter[int] = Counter()
        residual_offset = 0
        records: list[ClusterQueryRecord] = []
        for step in allocation.steps:
            if step.allocation_kind == "effective_cluster":
                assert step.raw_label is not None
                source = queues[selector_id].queue_for(step.raw_label)[
                    offsets[step.raw_label]
                ]
                offsets[step.raw_label] += 1
                records.append(
                    ClusterQueryRecord(
                        query_index=step.query_index,
                        case_id=source.case_id,
                        raw_label=source.raw_label,
                        selector_id=selector_id,
                        rank_within_cluster=source.rank_within_cluster,
                        selector_score=source.selector_score,
                        score_definition=source.score_definition,
                        score_components=source.score_components,
                        reason=source.reason,
                        is_core=source.is_core,
                        is_border=source.is_border,
                        is_noise=source.is_noise,
                        is_residual=False,
                        active_learning_seed=active_learning_seed,
                        input_sha256=input_sha256,
                    )
                )
                continue
            case_id = residual_case_ids[residual_offset]
            residual_offset += 1
            score = int(
                semantic_sha256(
                    {
                        "role": "tuning-center-anchored-shared-residual",
                        "partition_sha256": partition.partition_sha256,
                        "case_id": case_id,
                    }
                )[:13],
                16,
            ) / float((16**13) - 1)
            records.append(
                ClusterQueryRecord(
                    query_index=step.query_index,
                    case_id=case_id,
                    raw_label=int(raw_label_by_case[case_id]),
                    selector_id=selector_id,
                    rank_within_cluster=residual_rank[case_id],
                    selector_score=score,
                    score_definition="frozen_tuning_center_shared_residual_order",
                    score_components=(("sha256_order_score", score),),
                    reason="matched_to_frozen_tuning_center_residual",
                    is_core=None,
                    is_border=None,
                    is_noise=case_id in set(partition.noise_case_ids),
                    is_residual=True,
                    active_learning_seed=active_learning_seed,
                    input_sha256=input_sha256,
                )
            )
        plans[selector_id] = ClusterQueryPlan(
            dataset_id=partition.dataset_id,
            clusterer_id=partition.clusterer_id,
            selector_id=selector_id,
            active_learning_seed=active_learning_seed,
            budget=ANNOTATION_BUDGET,
            representation_matrix_sha256=candidate.matrix_sha256,
            geometry_sha256=native.geometry_sha256,
            partition_sha256=partition.partition_sha256,
            quota_sha256=allocation.quota_sha256,
            allocation_sha256=allocation.allocation_sha256,
            queues_sha256=contract.queues_sha256,
            case_sets_sha256=contract.case_sets_sha256,
            input_sha256=input_sha256,
            records=tuple(records),
        )
    if plans["center"].selected_case_ids != tuple(frozen_center_case_ids):
        raise ValueError("common center queue cannot reproduce the accepted tuning plan")
    return WinnerMatchedPlanBundle(native, partition, allocation, plans)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def build_winner_selector_freeze(
    *,
    aiops_bundles: Mapping[int, WinnerMatchedPlanBundle],
    accepted_configuration_id: str,
    tuning_manifest_sha256: str,
    tuning_sentinel_sha256: str,
    owner_override_sha256: str,
    aiops_center_reuse_unit_ids: Mapping[int, str],
    rcabench_reuse_unit_ids: Sequence[str],
) -> WinnerSelectorFreeze:
    """Freeze the matched winner comparison and its exact reuse/run split."""

    if (
        not isinstance(accepted_configuration_id, str)
        or not accepted_configuration_id
        or set(aiops_bundles) != set(ACTIVE_LEARNING_SEEDS)
        or set(aiops_center_reuse_unit_ids) != set(ACTIVE_LEARNING_SEEDS)
    ):
        raise ValueError("winner selector freeze identity or seed coverage drifted")
    trust = {
        "tuning_manifest_sha256": _require_sha256(
            tuning_manifest_sha256, "tuning manifest SHA-256"
        ),
        "tuning_sentinel_sha256": _require_sha256(
            tuning_sentinel_sha256, "tuning sentinel SHA-256"
        ),
        "owner_override_sha256": _require_sha256(
            owner_override_sha256, "owner override SHA-256"
        ),
    }
    rcabench_reuse = tuple(rcabench_reuse_unit_ids)
    expected_rcabench = {
        f"ordinary.rcabench.global_pca_dim32.hdbscan.{selector}.seed{seed}"
        for selector in WINNER_SELECTORS
        for seed in ACTIVE_LEARNING_SEEDS
    }
    if len(rcabench_reuse) != 9 or set(rcabench_reuse) != expected_rcabench:
        raise ValueError("RCABench winner reuse must cover exactly three selectors and seeds")
    center_reuse = {
        int(seed): str(aiops_center_reuse_unit_ids[seed])
        for seed in ACTIVE_LEARNING_SEEDS
    }
    if len(set(center_reuse.values())) != 3 or any(
        not unit_id for unit_id in center_reuse.values()
    ):
        raise ValueError("AIOps22 tuning-center reuse unit IDs are invalid")

    plans: list[dict[str, Any]] = []
    new_units: list[dict[str, Any]] = []
    geometry_hashes: set[str] = set()
    for selector_id in WINNER_SELECTORS:
        for seed in ACTIVE_LEARNING_SEEDS:
            bundle = aiops_bundles[seed]
            if type(bundle) is not WinnerMatchedPlanBundle:
                raise ValueError("winner selector freeze requires typed plan bundles")
            if set(bundle.plans) != set(WINNER_SELECTORS):
                raise ValueError("winner selector bundle does not contain three selectors")
            plan = bundle.plans[selector_id]
            if (
                plan.dataset_id != "aiops2022_pre"
                or plan.clusterer_id != "hdbscan"
                or plan.selector_id != selector_id
                or plan.active_learning_seed != seed
                or plan.budget != ANNOTATION_BUDGET
                or len(plan.selected_case_ids) != len(set(plan.selected_case_ids))
            ):
                raise ValueError("winner selector plan controls drifted")
            geometry_hashes.add(plan.geometry_sha256)
            encoded = plan.to_dict()
            plans.append(encoded)
            if selector_id == "center":
                continue
            new_units.append(
                {
                    "unit_id": (
                        "winner.aiops2022_pre.global_pca_dim32.hdbscan."
                        f"{selector_id}.seed{seed}"
                    ),
                    "dataset_id": "aiops2022_pre",
                    "representation_id": "global_pca_dim32",
                    "clusterer_id": "hdbscan",
                    "selector_id": selector_id,
                    "active_learning_seed": seed,
                    "downstream_seed": 42,
                    "budget": ANNOTATION_BUDGET,
                    "evaluation_protocol": "outer_test_guided",
                    "query_plan_sha256": plan.plan_sha256,
                }
            )
    if len(geometry_hashes) != 1 or len(new_units) != 6:
        raise ValueError("winner AIOps22 geometry or missing-unit split drifted")
    manifest: dict[str, Any] = {
        "schema_version": "combined-active-learning-winner-selector-freeze-v1",
        "immutable": True,
        "method_identity": "global_pca_dim32+hdbscan",
        "accepted_configuration_id": accepted_configuration_id,
        "evaluation_protocol": "outer_test_guided",
        "snapshot_target_met": False,
        "analysis_allowed": True,
        **trust,
        "aiops2022_pre": {
            "geometry_sha256": next(iter(geometry_hashes)),
            "plans": plans,
            "center_reuse_unit_ids": {
                str(seed): center_reuse[seed] for seed in ACTIVE_LEARNING_SEEDS
            },
            "new_unit_ids": [unit["unit_id"] for unit in new_units],
        },
        "rcabench": {
            "configuration": {
                "cluster_selection_method": "leaf",
                "min_cluster_size": 5,
                "min_samples": 5,
            },
            "reused_unit_ids": list(rcabench_reuse),
        },
        "new_rcl_unit_count": len(new_units),
        "reused_rcl_unit_count": 3 + len(rcabench_reuse),
        "ground_truth_read": False,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    return WinnerSelectorFreeze(manifest=manifest, new_units=tuple(new_units))


def build_winner_rcl_bridges(
    *,
    freeze: WinnerSelectorFreeze,
    reference_bridge_payload_by_seed: Mapping[int, Mapping[str, Any]],
    output_root: str,
    code_sha256: str,
) -> Mapping[str, Budget30RclBridge]:
    """Bind only the six missing AIOps22 selector units to frozen RCL controls."""

    if type(freeze) is not WinnerSelectorFreeze:
        raise ValueError("winner RCL bridge construction requires a typed freeze")
    if set(reference_bridge_payload_by_seed) != set(ACTIVE_LEARNING_SEEDS):
        raise ValueError("winner RCL references must cover seeds 41, 42, and 43")
    root = PurePosixPath(output_root)
    if (
        not output_root
        or "\\" in output_root
        or not root.is_absolute()
        or str(root) != output_root
        or any(part in {".", ".."} for part in output_root.split("/"))
    ):
        raise ValueError("winner RCL output root must be a canonical POSIX path")
    execution_code_sha256 = _require_sha256(
        code_sha256, "winner execution code SHA-256"
    )
    references = {
        seed: validate_budget30_rcl_bridge(reference_bridge_payload_by_seed[seed])
        for seed in ACTIVE_LEARNING_SEEDS
    }
    for seed, reference in references.items():
        if (
            reference.dataset_id != "aiops2022_pre"
            or reference.method_id != "hdbscan"
            or reference.source_selector_id != "center"
            or reference.active_learning_seed != seed
            or reference.budget != ANNOTATION_BUDGET
            or reference.control.training_seed != 42
        ):
            raise ValueError("winner RCL reference bridge controls drifted")
    if (
        len({item.control.control_sha256 for item in references.values()}) != 1
        or len({item.data_sha256 for item in references.values()}) != 1
    ):
        raise ValueError("winner RCL reference controls or data hashes disagree")

    plan_records = freeze.manifest["aiops2022_pre"]["plans"]
    if not isinstance(plan_records, list):
        raise ValueError("winner selector plans are missing from the freeze")
    plan_by_selector_seed = {
        (record.get("selector_id"), record.get("active_learning_seed")): record
        for record in plan_records
        if isinstance(record, Mapping)
    }
    if len(plan_by_selector_seed) != 9:
        raise ValueError("winner selector freeze must contain nine unique plans")

    bridges: dict[str, Budget30RclBridge] = {}
    for unit in freeze.new_units:
        selector_id = unit["selector_id"]
        seed = unit["active_learning_seed"]
        plan = plan_by_selector_seed.get((selector_id, seed))
        if not isinstance(plan, Mapping):
            raise ValueError("winner RCL unit has no frozen query plan")
        selected = plan.get("selected_case_ids")
        plan_sha256 = plan.get("plan_sha256")
        if (
            plan_sha256 != unit.get("query_plan_sha256")
            or not isinstance(selected, list)
            or len(selected) != ANNOTATION_BUDGET
            or len(set(selected)) != ANNOTATION_BUDGET
        ):
            raise ValueError("winner RCL query plan identity drifted")
        reference = references[seed]
        unit_id = unit["unit_id"]
        bridge = Budget30RclBridge(
            unit_id=unit_id,
            dataset_id="aiops2022_pre",
            arm_id=f"global_pca_dim32.hdbscan.{selector_id}",
            method_id="hdbscan",
            source_selector_id=selector_id,
            active_learning_seed=seed,
            budget=ANNOTATION_BUDGET,
            selected_case_ids=tuple(selected),
            query_plan_type="cluster",
            query_plan_sha256=plan_sha256,
            query_plan_reference_sha256=semantic_sha256(dict(plan)),
            output_root=str(root / "rcl" / unit_id),
            code_sha256=execution_code_sha256,
            data_sha256=reference.data_sha256,
            representation_sha256=plan["representation_matrix_sha256"],
            protocol_sha256=freeze.manifest["manifest_sha256"],
            control=reference.control,
        )
        bridges[unit_id] = bridge
    if len(bridges) != 6:
        raise ValueError("winner RCL bridge expansion must contain six units")
    return bridges


def _semantic_category_summary(
    values: Sequence[str],
    *,
    pool_counts: Counter[str],
    pool_size: int,
) -> dict[str, Any]:
    counts = Counter(values)
    support = len(values)
    if support < 1:
        return {
            "support": 0,
            "purity": None,
            "normalized_entropy": None,
            "dominant": [],
        }
    probabilities = [count / support for count in counts.values()]
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    normalizer = math.log(len(pool_counts)) if len(pool_counts) > 1 else 0.0
    dominant = []
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]:
        pool_share = pool_counts[category] / pool_size
        share = count / support
        dominant.append(
            {
                "category": category,
                "count": count,
                "share": share,
                "pool_share": pool_share,
                "lift_vs_pool": share / pool_share if pool_share > 0.0 else None,
            }
        )
    return {
        "support": support,
        "unique_category_count": len(counts),
        "purity": max(counts.values()) / support,
        "normalized_entropy": entropy / normalizer if normalizer > 0.0 else 0.0,
        "dominant": dominant,
    }


def backproject_global_pca_modalities(
    *,
    candidate: RepresentationMatrixCandidate,
    balanced: BalancedModalityBlocks,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Back-project retained global-PCA scores into balanced modality blocks."""

    if (
        type(candidate) is not RepresentationMatrixCandidate
        or type(balanced) is not BalancedModalityBlocks
        or candidate.family != "global_pca"
        or candidate.parameters.get("reduction") != "global_pca"
        or tuple(candidate.case_ids) != tuple(balanced.case_ids)
    ):
        raise ValueError("global PCA backprojection inputs are incompatible")
    fused = fuse_balanced_blocks(balanced)
    if candidate.source_matrix_sha256 != fused.matrix_sha256:
        raise ValueError("global PCA source matrix differs from balanced fusion")
    centered = np.asarray(fused.matrix, dtype=float) - np.asarray(
        fused.matrix, dtype=float
    ).mean(axis=0, keepdims=True)
    _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    retained = int(candidate.matrix.shape[1])
    if retained < 1 or retained > right.shape[0]:
        raise ValueError("global PCA retained dimension is invalid")
    components = right[:retained].copy()
    for index in range(retained):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    reconstructed_scores = centered @ components.T
    if not np.allclose(
        reconstructed_scores,
        np.asarray(candidate.matrix, dtype=float),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("global PCA score reconstruction drifted")
    backprojected = np.asarray(candidate.matrix, dtype=float) @ components
    blocks = {
        modality: backprojected[:, slice(*fused.modality_slices[modality])]
        for modality in ("metric", "log", "trace", "topology", "time")
    }
    names = {
        modality: tuple(balanced.feature_names_by_modality[modality])
        for modality in ("metric", "log", "trace", "topology", "time")
    }
    if any(blocks[modality].shape[1] != len(names[modality]) for modality in blocks):
        raise ValueError("global PCA modality slice dimensions drifted")
    return blocks, names


def _semantic_multilabel_summary(
    values: Sequence[Sequence[str]],
    *,
    pool_case_counts: Counter[str],
    pool_size: int,
) -> dict[str, Any]:
    canonical = [tuple(sorted(set(items))) for items in values]
    support = len(canonical)
    if support < 1:
        return {
            "support": 0,
            "multi_label": True,
            "purity": None,
            "normalized_entropy": None,
            "dominant": [],
        }
    if any(not items for items in canonical):
        raise ValueError("multi-label semantic category must not be empty")
    counts = Counter(item for items in canonical for item in items)
    occurrence_count = sum(counts.values())
    probabilities = [count / occurrence_count for count in counts.values()]
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    normalizer = math.log(len(pool_case_counts)) if len(pool_case_counts) > 1 else 0.0
    dominant = []
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]:
        pool_share = pool_case_counts[category] / pool_size
        share = count / support
        dominant.append(
            {
                "category": category,
                "count": count,
                "share": share,
                "pool_share": pool_share,
                "lift_vs_pool": share / pool_share if pool_share > 0.0 else None,
            }
        )
    return {
        "support": support,
        "multi_label": True,
        "unique_category_count": len(counts),
        "mean_labels_per_case": occurrence_count / support,
        "purity": max(counts.values()) / support,
        "normalized_entropy": entropy / normalizer if normalizer > 0.0 else 0.0,
        "dominant": dominant,
    }


def derive_entity_layer(root_cause_services: Sequence[str]) -> str:
    """Map one case's deduplicated root-cause entities to a coarse layer."""

    services = tuple(sorted(set(str(value).strip() for value in root_cause_services)))
    if not services or any(not value for value in services):
        raise ValueError("root-cause services must be non-empty")
    layers = set()
    for value in services:
        lowered = value.lower()
        prefix = lowered.split(":", 1)[0]
        if prefix in {"host", "node", "machine", "vm"}:
            layers.add("infrastructure_host")
        elif any(token in lowered for token in ("mysql", "postgres", "mongodb", "redis", "database", "db:")):
            layers.add("datastore")
        elif prefix in {"service", "pod", "container"}:
            layers.add("application_service")
        else:
            layers.add("other_entity")
    return next(iter(layers)) if len(layers) == 1 else "mixed:" + "+".join(sorted(layers))


def derive_topology_role(root_cause_services: Sequence[str]) -> str:
    """Derive a topology role from frozen root-cause entity names."""

    services = tuple(sorted(set(str(value).strip() for value in root_cause_services)))
    if not services or any(not value for value in services):
        raise ValueError("root-cause services must be non-empty")
    roles = set()
    for value in services:
        lowered = value.lower()
        prefix = lowered.split(":", 1)[0]
        if prefix in {"host", "node", "machine", "vm"}:
            roles.add("infrastructure_host")
        elif any(token in lowered for token in ("mysql", "postgres", "mongodb", "redis", "database", "db:")):
            roles.add("datastore")
        elif any(token in lowered for token in ("frontend", "ui-dashboard", "gateway", "ingress")):
            roles.add("entry_or_user_interface")
        elif prefix in {"service", "pod", "container"}:
            roles.add("internal_service")
        else:
            roles.add("other_topology_role")
    return next(iter(roles)) if len(roles) == 1 else "mixed:" + "+".join(sorted(roles))


def _time_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "support": 0,
            "minimum_epoch_seconds": None,
            "q1_epoch_seconds": None,
            "median_epoch_seconds": None,
            "q3_epoch_seconds": None,
            "maximum_epoch_seconds": None,
            "span_seconds": None,
        }
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("cluster timestamps must be finite epoch seconds")
    minimum = float(array.min())
    maximum = float(array.max())
    return {
        "support": len(values),
        "minimum_epoch_seconds": minimum,
        "q1_epoch_seconds": float(np.quantile(array, 0.25)),
        "median_epoch_seconds": float(np.median(array)),
        "q3_epoch_seconds": float(np.quantile(array, 0.75)),
        "maximum_epoch_seconds": maximum,
        "span_seconds": maximum - minimum,
    }


def _modal_contribution_summary(
    indices: Sequence[int],
    *,
    blocks: Mapping[str, np.ndarray],
    feature_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    norms: dict[str, float] = {}
    top_features: dict[str, list[dict[str, Any]]] = {}
    for modality in ("metric", "log", "trace", "topology", "time"):
        matrix = np.asarray(blocks[modality], dtype=float)
        names = tuple(feature_names[modality])
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("backprojected modality block shape drifted")
        shift = matrix[np.asarray(indices, dtype=int)].mean(axis=0)
        norms[modality] = float(np.linalg.norm(shift))
        top_features[modality] = [
            {"feature": names[index], "mean_shift": float(shift[index])}
            for index in sorted(
                range(len(names)),
                key=lambda index: (-abs(float(shift[index])), names[index]),
            )[:5]
        ]
    total = math.fsum(norms.values())
    shares = {
        modality: (value / total if total > 0.0 else 0.0)
        for modality, value in norms.items()
    }
    largest = min(shares, key=lambda modality: (-shares[modality], modality))
    return {
        "definition": "l2_norm_of_cluster_mean_retained_pca_backprojection_by_balanced_modality",
        "causal_claim": False,
        "norm_by_modality": norms,
        "share_by_modality": shares,
        "largest_modality": largest,
        "top_backprojected_features_by_modality": top_features,
    }


def _agreement_metrics(true_values: Sequence[str], cluster_labels: Sequence[int]) -> dict[str, float | int | None]:
    if len(true_values) != len(cluster_labels):
        raise ValueError("semantic labels and clusters have different lengths")
    if len(true_values) < 2 or len(set(cluster_labels)) < 2 or len(set(true_values)) < 2:
        return {
            "case_count": len(true_values),
            "homogeneity": None,
            "completeness": None,
            "nmi": None,
            "ari": None,
        }
    return {
        "case_count": len(true_values),
        "homogeneity": float(homogeneity_score(true_values, cluster_labels)),
        "completeness": float(completeness_score(true_values, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(true_values, cluster_labels)),
        "ari": float(adjusted_rand_score(true_values, cluster_labels)),
    }


def build_partition_semantic_report(
    *,
    dataset_id: str,
    seed: int,
    case_ids: Sequence[str],
    effective_labels: Sequence[int],
    fault_type_by_case: Mapping[str, str],
    root_cause_services_by_case: Mapping[str, Sequence[str]],
    entity_layer_by_case: Mapping[str, str],
    topology_role_by_case: Mapping[str, str],
    timestamp_by_case: Mapping[str, float],
    backprojected_blocks_by_modality: Mapping[str, np.ndarray],
    feature_names_by_modality: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Describe one frozen partition after labels are opened for evaluation."""

    ids = tuple(case_ids)
    labels = tuple(int(value) for value in effective_labels)
    modalities = {"metric", "log", "trace", "topology", "time"}
    if (
        not dataset_id
        or seed not in ACTIVE_LEARNING_SEEDS
        or not ids
        or len(ids) != len(set(ids))
        or len(labels) != len(ids)
        or set(fault_type_by_case) != set(ids)
        or set(root_cause_services_by_case) != set(ids)
        or set(entity_layer_by_case) != set(ids)
        or set(topology_role_by_case) != set(ids)
        or set(timestamp_by_case) != set(ids)
        or set(backprojected_blocks_by_modality) != modalities
        or set(feature_names_by_modality) != modalities
    ):
        raise ValueError("semantic report inputs do not cover one frozen candidate pool")
    for modality in modalities:
        matrix = np.asarray(backprojected_blocks_by_modality[modality], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
            raise ValueError("semantic report modality evidence is invalid")

    fault_values = [str(fault_type_by_case[case_id]) for case_id in ids]
    root_case_sets = [
        tuple(sorted(set(root_cause_services_by_case[case_id]))) for case_id in ids
    ]
    root_values = [" | ".join(items) for items in root_case_sets]
    layer_values = [str(entity_layer_by_case[case_id]) for case_id in ids]
    topology_role_values = [str(topology_role_by_case[case_id]) for case_id in ids]
    if any(
        not value
        for value in fault_values + root_values + layer_values + topology_role_values
    ):
        raise ValueError("semantic report ground truth contains an empty category")
    pool_counts = {
        "fault_type": Counter(fault_values),
        "root_cause_service_set": Counter(root_values),
        "entity_layer": Counter(layer_values),
        "topology_role": Counter(topology_role_values),
    }
    root_service_pool_case_counts = Counter(
        service for services in root_case_sets for service in services
    )
    values_by_name = {
        "fault_type": fault_values,
        "root_cause_service_set": root_values,
        "entity_layer": layer_values,
        "topology_role": topology_role_values,
    }
    members: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label >= 0:
            members.setdefault(label, []).append(index)
    if len(members) < 2:
        raise ValueError("semantic report requires at least two effective clusters")

    def card_for(indices: Sequence[int], raw_label: int | None) -> dict[str, Any]:
        card = {
            "raw_label": raw_label,
            "support": len(indices),
            "candidate_share": len(indices) / len(ids),
            "time": _time_summary([float(timestamp_by_case[ids[index]]) for index in indices]),
            "modal_contribution": _modal_contribution_summary(
                indices,
                blocks=backprojected_blocks_by_modality,
                feature_names=feature_names_by_modality,
            ),
        }
        for name, values in values_by_name.items():
            if name == "root_cause_service_set":
                card["root_cause_service"] = _semantic_multilabel_summary(
                    [root_case_sets[index] for index in indices],
                    pool_case_counts=root_service_pool_case_counts,
                    pool_size=len(ids),
                )
            else:
                card[name] = _semantic_category_summary(
                    [values[index] for index in indices],
                    pool_counts=pool_counts[name],
                    pool_size=len(ids),
                )
        return card

    cards = [card_for(indices, label) for label, indices in sorted(members.items())]
    residual_indices = [index for index, label in enumerate(labels) if label < 0]
    semantic_metrics: dict[str, Any] = {}
    effective_indices = [index for index, label in enumerate(labels) if label >= 0]
    for name, values in values_by_name.items():
        semantic_metrics[name] = {
            "effective_cases_only": _agreement_metrics(
                [values[index] for index in effective_indices],
                [labels[index] for index in effective_indices],
            ),
            "all_cases_residual_as_one_cluster": _agreement_metrics(values, labels),
        }
    report: dict[str, Any] = {
        "schema_version": "combined-active-learning-winner-partition-semantics-v1",
        "dataset_id": dataset_id,
        "active_learning_seed": seed,
        "candidate_count": len(ids),
        "effective_cluster_count": len(members),
        "effective_support": len(effective_indices),
        "effective_coverage": len(effective_indices) / len(ids),
        "residual_support": len(residual_indices),
        "residual_share": len(residual_indices) / len(ids),
        "cluster_cards": cards,
        "residual_card": card_for(residual_indices, None) if residual_indices else None,
        "global_label_agreement": semantic_metrics,
        "ground_truth_read": True,
        "used_for_selection": False,
    }
    report["report_sha256"] = semantic_sha256(report)
    return report


def build_selector_score_report(
    *,
    dataset_id: str,
    score_by_selector_and_seed: Mapping[str, Mapping[int, Mapping[str, float]]],
    global_random_by_seed: Mapping[int, Mapping[str, float]],
    snapshot_mean_top135: float | None,
) -> dict[str, Any]:
    """Summarize complete winner-selector metrics against matched controls."""

    metric_names = ("top1", "top3", "top5", "mrr", "top135")
    if not dataset_id or set(score_by_selector_and_seed) != set(WINNER_SELECTORS):
        raise ValueError("selector score report requires exactly three winner selectors")

    def checked_scores(
        values: Mapping[str, float], *, context: str
    ) -> dict[str, float]:
        if set(values) != set(metric_names):
            raise ValueError(f"{context} must contain complete ranking metrics")
        output = {}
        for name in metric_names:
            value = values[name]
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{context} {name} must be finite in [0, 1]")
            output[name] = float(value)
        expected_top135 = math.fsum(
            output[name] for name in ("top1", "top3", "top5")
        ) / 3.0
        if not math.isclose(
            output["top135"], expected_top135, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{context} TOP135 disagrees with Hit@1/3/5")
        return output

    if set(global_random_by_seed) != set(ACTIVE_LEARNING_SEEDS):
        raise ValueError("selector score report requires three global-random seeds")
    random = {
        seed: checked_scores(
            global_random_by_seed[seed], context=f"global random seed {seed}"
        )
        for seed in ACTIVE_LEARNING_SEEDS
    }
    selectors = {}
    for selector_id in WINNER_SELECTORS:
        values = score_by_selector_and_seed[selector_id]
        if set(values) != set(ACTIVE_LEARNING_SEEDS):
            raise ValueError("selector score report requires seeds 41, 42, and 43")
        selectors[selector_id] = {
            seed: checked_scores(
                values[seed], context=f"{selector_id} seed {seed}"
            )
            for seed in ACTIVE_LEARNING_SEEDS
        }

    def means(values_by_seed: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
        return {
            name: math.fsum(values_by_seed[seed][name] for seed in ACTIVE_LEARNING_SEEDS)
            / len(ACTIVE_LEARNING_SEEDS)
            for name in metric_names
        }

    random_mean = means(random)
    selector_means = {}
    for selector_id in WINNER_SELECTORS:
        mean = means(selectors[selector_id])
        selector_means[selector_id] = {
            **mean,
            "standard_deviation": {
                name: float(
                    np.std(
                        [selectors[selector_id][seed][name] for seed in ACTIVE_LEARNING_SEEDS],
                        ddof=1,
                    )
                )
                for name in metric_names
            },
            "delta_vs_global_random": {
                name: mean[name] - random_mean[name] for name in metric_names
            },
            "delta_vs_snapshot_top135": (
                None
                if snapshot_mean_top135 is None
                else mean["top135"] - float(snapshot_mean_top135)
            ),
            "strictly_above_global_random": mean["top135"] > random_mean["top135"],
        }
    best_selector = min(
        WINNER_SELECTORS,
        key=lambda selector_id: (-selector_means[selector_id]["top135"], selector_id),
    )
    trigger = evaluate_aulc_trigger(
        {
            selector_id: {
                seed: selectors[selector_id][seed]["top135"]
                for seed in ACTIVE_LEARNING_SEEDS
            }
            for selector_id in WINNER_SELECTORS
        }
    )
    report: dict[str, Any] = {
        "schema_version": "combined-active-learning-winner-selector-score-report-v1",
        "dataset_id": dataset_id,
        "metrics": list(metric_names),
        "selector_seed_scores": selectors,
        "selector_means": selector_means,
        "global_random_seed_scores": random,
        "global_random_mean": random_mean,
        "snapshot_mean_top135": snapshot_mean_top135,
        "best_selector_by_mean_top135": best_selector,
        "seed_winner_by_top135": [
            {"seed": seed, "selector_id": selector_id}
            for seed, selector_id in trigger.winner_by_seed
        ],
        "aulc_trigger": {
            "triggered": trigger.triggered,
            "seed_winner_changed": trigger.seed_winner_changed,
            "top_two_mean_gap": trigger.top_two_mean_gap,
            "mean_top135_by_selector": [
                {"selector_id": selector_id, "mean_top135": value}
                for selector_id, value in trigger.mean_top135_by_selector
            ],
            "reasons": list(trigger.reasons),
        },
    }
    report["report_sha256"] = semantic_sha256(report)
    return report


def first_matching_target_mrr(rows: Sequence[Mapping[str, Any]]) -> float:
    """Compute MRR from the first ranked item matching any case target."""

    if not rows:
        raise ValueError("cannot compute MRR for an empty case sequence")
    reciprocal_ranks: list[float] = []
    for row in rows:
        ranking = row.get("ranking")
        targets = row.get("targets")
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            raise ValueError("ranking must be a sequence")
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise ValueError("targets must be a sequence")
        target_set = {str(target) for target in targets if str(target)}
        if not target_set:
            raise ValueError("targets must not be empty")
        rank = next(
            (
                index
                for index, candidate in enumerate(ranking, start=1)
                if str(candidate) in target_set
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return math.fsum(reciprocal_ranks) / len(reciprocal_ranks)


def maximum_weight_jaccard_alignment(
    *,
    reference_clusters: Mapping[int, Sequence[str]],
    target_clusters: Mapping[int, Sequence[str]],
) -> tuple[JaccardClusterMatch, ...]:
    """Align two clusterings by maximum total case-set Jaccard overlap."""

    reference = tuple(sorted((int(label), frozenset(items)) for label, items in reference_clusters.items()))
    target = tuple(sorted((int(label), frozenset(items)) for label, items in target_clusters.items()))
    if any(not values for _, values in reference + target):
        raise ValueError("Jaccard alignment clusters must be non-empty")
    if not reference:
        return tuple(JaccardClusterMatch(None, label, 0, len(values), 0.0) for label, values in target)
    if not target:
        return tuple(JaccardClusterMatch(label, None, 0, len(values), 0.0) for label, values in reference)
    weights = np.zeros((len(reference), len(target)), dtype=float)
    intersections = np.zeros_like(weights, dtype=int)
    unions = np.zeros_like(weights, dtype=int)
    for row, (_reference_label, left) in enumerate(reference):
        for column, (_target_label, right) in enumerate(target):
            intersection = len(left.intersection(right))
            union = len(left.union(right))
            intersections[row, column] = intersection
            unions[row, column] = union
            weights[row, column] = intersection / union if union else 0.0
    row_indices, column_indices = linear_sum_assignment(-weights)
    matched_reference: set[int] = set()
    matched_target: set[int] = set()
    results: list[JaccardClusterMatch] = []
    for row, column in zip(row_indices, column_indices):
        if weights[row, column] <= 0.0:
            continue
        matched_reference.add(row)
        matched_target.add(column)
        results.append(
            JaccardClusterMatch(
                reference_label=reference[row][0],
                target_label=target[column][0],
                intersection_size=int(intersections[row, column]),
                union_size=int(unions[row, column]),
                jaccard=float(weights[row, column]),
            )
        )
    results.extend(
        JaccardClusterMatch(label, None, 0, len(values), 0.0)
        for index, (label, values) in enumerate(reference)
        if index not in matched_reference
    )
    results.extend(
        JaccardClusterMatch(None, label, 0, len(values), 0.0)
        for index, (label, values) in enumerate(target)
        if index not in matched_target
    )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.reference_label is None,
                item.reference_label if item.reference_label is not None else math.inf,
                item.target_label if item.target_label is not None else math.inf,
            ),
        )
    )


def evaluate_aulc_trigger(
    top135_by_selector_and_seed: Mapping[str, Mapping[int, float]],
) -> AulcTriggerDecision:
    """Apply the frozen seed-winner-change or mean-gap-below-0.02 rule."""

    if set(top135_by_selector_and_seed) != set(WINNER_SELECTORS):
        raise ValueError("AULC trigger requires exactly the three winner selectors")
    normalized: dict[str, dict[int, float]] = {}
    for selector_id in WINNER_SELECTORS:
        values = top135_by_selector_and_seed[selector_id]
        if set(values) != set(ACTIVE_LEARNING_SEEDS):
            raise ValueError("AULC trigger requires seeds 41, 42, and 43")
        normalized[selector_id] = {}
        for seed, value in values.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("AULC trigger TOP135 values must be in [0, 1]")
            normalized[selector_id][seed] = float(value)
    winner_by_seed = tuple(
        (
            seed,
            min(
                WINNER_SELECTORS,
                key=lambda selector_id: (-normalized[selector_id][seed], selector_id),
            ),
        )
        for seed in ACTIVE_LEARNING_SEEDS
    )
    seed_changed = len({selector for _seed, selector in winner_by_seed}) > 1
    means = tuple(
        sorted(
            (
                (selector_id, math.fsum(normalized[selector_id].values()) / 3.0)
                for selector_id in WINNER_SELECTORS
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    gap = means[0][1] - means[1][1]
    close = gap < AULC_GAP_TRIGGER
    reasons = tuple(
        reason
        for condition, reason in (
            (seed_changed, "budget30_selector_winner_changes_across_seeds"),
            (close, "budget30_top_two_mean_top135_gap_below_0.02"),
        )
        if condition
    ) or ("frozen_aulc_trigger_not_met",)
    return AulcTriggerDecision(
        triggered=seed_changed or close,
        seed_winner_changed=seed_changed,
        winner_by_seed=winner_by_seed,
        mean_top135_by_selector=means,
        top_two_mean_gap=gap,
        reasons=reasons,
    )


def normalized_trapezoid_aulc(mrr_by_budget: Mapping[int, float]) -> float:
    """Return trapezoidal MRR area normalized by the 8-to-30 budget span."""

    if set(mrr_by_budget) != set(AULC_BUDGETS):
        raise ValueError("MRR AULC requires exactly budgets 8, 16, 24, and 30")
    values = []
    for budget in AULC_BUDGETS:
        value = mrr_by_budget[budget]
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("MRR AULC values must be in [0, 1]")
        values.append(float(value))
    area = math.fsum(
        (right_budget - left_budget) * (left_value + right_value) / 2.0
        for left_budget, right_budget, left_value, right_value in zip(
            AULC_BUDGETS,
            AULC_BUDGETS[1:],
            values,
            values[1:],
        )
    )
    return area / (AULC_BUDGETS[-1] - AULC_BUDGETS[0])


def strict_nested_prefixes(
    selected_case_ids: Sequence[str],
) -> dict[int, tuple[str, ...]]:
    """Return the preregistered 8/16/24/30 prefixes of one frozen plan."""

    selected = tuple(selected_case_ids)
    if (
        len(selected) != ANNOTATION_BUDGET
        or len(set(selected)) != ANNOTATION_BUDGET
        or any(not case_id for case_id in selected)
    ):
        raise ValueError("AULC prefixes require one frozen plan with 30 unique cases")
    return {budget: selected[:budget] for budget in AULC_BUDGETS}


__all__ = [
    "AULC_BUDGETS",
    "AULC_GAP_TRIGGER",
    "AulcTriggerDecision",
    "JaccardClusterMatch",
    "WINNER_SELECTORS",
    "WinnerMatchedPlanBundle",
    "WinnerSelectorFreeze",
    "build_winner_selector_freeze",
    "backproject_global_pca_modalities",
    "build_tuning_anchored_matched_hdbscan_plans",
    "build_winner_rcl_bridges",
    "build_partition_semantic_report",
    "build_selector_score_report",
    "derive_entity_layer",
    "derive_topology_role",
    "evaluate_aulc_trigger",
    "first_matching_target_mrr",
    "maximum_weight_jaccard_alignment",
    "normalized_trapezoid_aulc",
    "strict_nested_prefixes",
    "to_native_hdbscan_result",
]
