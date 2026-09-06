from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


class CalibrationValidationError(ValueError):
    """Raised when development calibration leaks or profile selection drifts."""


DATASETS = ("rcabench", "aiops2022_pre")
ARMS = ("oser_meta", "mm_dro", "cope_gate")
EXPECTED_EXCLUSIONS = {
    "rcabench": (
        "NetworkDelay",
        "HTTPResponsePatchBody",
        "HTTPResponseDelay",
        "HTTPRequestAbort",
        "PodKill",
        "JVMMemoryStress",
        "NetworkBandwidth",
    ),
    "aiops2022_pre": (
        "node 磁盘空间消耗",
        "k8s容器读io负载",
        "k8s容器网络丢包",
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _finite_unit(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CalibrationValidationError(f"{field} must be finite in [0,1]")
    return result


def _validate_bundle_hash(bundle: Mapping[str, Any]) -> None:
    identity = {key: deepcopy(value) for key, value in bundle.items() if key != "calibration_bundle_sha256"}
    if bundle.get("calibration_bundle_sha256") != _semantic_hash(identity):
        raise CalibrationValidationError("calibration bundle hash drift")


def build_calibration_bundle(
    pools_by_dataset: Mapping[str, Mapping[str, Any]],
    queried_labels_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    formal_training_case_ids_by_dataset: Mapping[str, Iterable[Any]],
) -> dict[str, Any]:
    if tuple(pools_by_dataset) != DATASETS or tuple(queried_labels_by_dataset) != DATASETS:
        raise CalibrationValidationError("calibration requires exactly both decision datasets in order")
    if tuple(formal_training_case_ids_by_dataset) != DATASETS:
        raise CalibrationValidationError("formal training ledgers must cover both decision datasets")
    datasets: dict[str, dict[str, Any]] = {}
    total_cost = 0
    total_overlap = 0
    for dataset_id in DATASETS:
        pool = dict(pools_by_dataset[dataset_id])
        if pool.get("schema_version") != "conservative-lofo-union-excluded-calibration-v1":
            raise CalibrationValidationError(f"unexpected calibration pool schema for {dataset_id}")
        if str(pool.get("dataset_id")) != dataset_id:
            raise CalibrationValidationError(f"calibration pool dataset drift for {dataset_id}")
        excluded = tuple(str(value) for value in pool.get("excluded_fault_types", ()))
        if excluded != EXPECTED_EXCLUSIONS[dataset_id]:
            raise CalibrationValidationError(f"calibration union exclusion drift for {dataset_id}")
        queried = tuple(str(value) for value in pool.get("queried_development_case_ids", ()))
        admitted = tuple(str(value) for value in pool.get("admitted_candidate_case_ids", ()))
        excluded_case_ids = tuple(str(value) for value in pool.get("excluded_case_ids", ()))
        if len(queried) != 30 or len(queried) != len(set(queried)):
            raise CalibrationValidationError(f"{dataset_id} must contribute exactly 30 unique labels")
        if int(pool.get("development_label_cost", -1)) != 30:
            raise CalibrationValidationError(f"{dataset_id} calibration label cost drift")
        if not set(queried) <= set(admitted) or set(admitted) & set(excluded_case_ids):
            raise CalibrationValidationError(f"{dataset_id} queried cases violate union exclusion")
        if not _valid_sha256(pool.get("calibration_pool_sha256")):
            raise CalibrationValidationError(f"{dataset_id} calibration pool hash is invalid")
        raw_labels = dict(queried_labels_by_dataset[dataset_id])
        if set(raw_labels) != set(queried):
            raise CalibrationValidationError(f"{dataset_id} queried label coverage drift")
        labels: dict[str, dict[str, str]] = {}
        for case_id in queried:
            raw = dict(raw_labels[case_id])
            if set(raw) != {"root_cause", "fault_type"}:
                raise CalibrationValidationError(f"{dataset_id} label is incomplete for {case_id}")
            label = {
                "root_cause": str(raw["root_cause"]).strip(),
                "fault_type": str(raw["fault_type"]).strip(),
            }
            if not all(label.values()):
                raise CalibrationValidationError(f"{dataset_id} label is empty for {case_id}")
            if label["fault_type"] in set(excluded):
                raise CalibrationValidationError(
                    f"{dataset_id} queried label contains an excluded fault type"
                )
            labels[case_id] = label
        formal = {str(value) for value in formal_training_case_ids_by_dataset[dataset_id]}
        overlap = formal & set(queried)
        if overlap:
            raise CalibrationValidationError(
                f"{dataset_id} calibration formal training overlap: {sorted(overlap)[:5]}"
            )
        total_overlap += len(overlap)
        total_cost += 30
        dataset_identity = {
            "dataset_id": dataset_id,
            "calibration_pool_sha256": str(pool["calibration_pool_sha256"]),
            "excluded_fault_types": excluded,
            "excluded_case_ids": excluded_case_ids,
            "queried_case_ids": queried,
            "labels_by_case": labels,
            "queried_labels_sha256": _semantic_hash(labels),
            "development_label_cost": 30,
            "formal_training_overlap_count": 0,
        }
        datasets[dataset_id] = {
            **dataset_identity,
            "dataset_calibration_sha256": _semantic_hash(dataset_identity),
        }
    identity = {
        "schema_version": "conservative-lofo-calibration-bundle-v1",
        "dataset_ids": DATASETS,
        "datasets": datasets,
        "development_label_cost": total_cost,
        "formal_training_overlap_count": total_overlap,
        "screening_target_union_excluded": True,
    }
    return {**identity, "calibration_bundle_sha256": _semantic_hash(identity)}


def build_nested_queried_type_folds(
    bundle: Mapping[str, Any], dataset_id: str
) -> tuple[dict[str, Any], ...]:
    _validate_bundle_hash(bundle)
    dataset = str(dataset_id)
    if dataset not in DATASETS:
        raise CalibrationValidationError(f"unknown calibration dataset: {dataset}")
    record = dict(bundle["datasets"][dataset])
    queried = tuple(str(value) for value in record.get("queried_case_ids", ()))
    labels = {str(key): dict(value) for key, value in dict(record.get("labels_by_case", {})).items()}
    if set(queried) != set(labels) or len(queried) != 30:
        raise CalibrationValidationError("nested pseudo-LOFO label membership drift")
    groups: dict[str, list[str]] = {}
    for case_id in queried:
        fault_type = str(labels[case_id].get("fault_type", "")).strip()
        groups.setdefault(fault_type, []).append(case_id)
    if len(groups) < 2 or "" in groups:
        raise CalibrationValidationError("nested pseudo-LOFO requires at least two queried types")
    folds = []
    for fault_type in sorted(groups):
        query = tuple(case_id for case_id in queried if case_id in set(groups[fault_type]))
        support = tuple(case_id for case_id in queried if case_id not in set(query))
        identity = {
            "schema_version": "conservative-lofo-calibration-pseudo-fold-v1",
            "dataset_id": dataset,
            "calibration_bundle_sha256": str(bundle["calibration_bundle_sha256"]),
            "pseudo_held_out_fault_type": fault_type,
            "support_case_ids": support,
            "query_case_ids": query,
        }
        folds.append({**identity, "pseudo_fold_sha256": _semantic_hash(identity)})
    return tuple(folds)


def build_calibration_profile_result(
    *,
    arm: str,
    profile_id: str,
    calibration_bundle_sha256: str,
    dataset_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arm_id = str(arm)
    if arm_id not in ARMS:
        raise CalibrationValidationError(f"unknown calibration arm: {arm_id}")
    profile = str(profile_id).strip()
    if not profile:
        raise CalibrationValidationError("profile_id must be nonempty")
    if not _valid_sha256(calibration_bundle_sha256):
        raise CalibrationValidationError("calibration bundle hash is invalid")
    if tuple(dataset_metrics) != DATASETS:
        raise CalibrationValidationError("profile result must cover both datasets in order")
    normalized: dict[str, dict[str, Any]] = {}
    ratios = []
    raw_mrr = []
    hit1 = []
    for dataset_id in DATASETS:
        raw = dict(dataset_metrics[dataset_id])
        baseline = _finite_unit(raw.get("baseline_inner_mrr"), "baseline_inner_mrr")
        candidate = _finite_unit(raw.get("profile_inner_mrr"), "profile_inner_mrr")
        candidate_hit1 = _finite_unit(
            raw.get("profile_inner_hit_at_1"), "profile_inner_hit_at_1"
        )
        fold_count = int(raw.get("pseudo_lofo_fold_count", 0))
        if fold_count <= 0:
            raise CalibrationValidationError("pseudo_lofo_fold_count must be positive")
        ratio = candidate / max(baseline, 1e-12)
        normalized[dataset_id] = {
            "baseline_inner_mrr": baseline,
            "profile_inner_mrr": candidate,
            "profile_inner_hit_at_1": candidate_hit1,
            "pseudo_lofo_fold_count": fold_count,
            "normalized_inner_mrr": ratio,
        }
        ratios.append(ratio)
        raw_mrr.append(candidate)
        hit1.append(candidate_hit1)
    identity = {
        "schema_version": "conservative-lofo-calibration-profile-result-v1",
        "arm": arm_id,
        "profile_id": profile,
        "calibration_bundle_sha256": str(calibration_bundle_sha256),
        "dataset_metrics": normalized,
        "mean_dataset_ratio_inner_mrr": sum(ratios) / len(ratios),
        "mean_raw_inner_mrr": sum(raw_mrr) / len(raw_mrr),
        "mean_hit_at_1": sum(hit1) / len(hit1),
    }
    return {**identity, "profile_result_sha256": _semantic_hash(identity)}


def _validate_profile_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != "conservative-lofo-calibration-profile-result-v1":
        raise CalibrationValidationError("unexpected profile result schema")
    identity = {key: deepcopy(value) for key, value in result.items() if key != "profile_result_sha256"}
    if result.get("profile_result_sha256") != _semantic_hash(identity):
        raise CalibrationValidationError("profile result hash drift")
    for field in ("mean_dataset_ratio_inner_mrr", "mean_raw_inner_mrr", "mean_hit_at_1"):
        value = float(result.get(field, float("nan")))
        if not math.isfinite(value):
            raise CalibrationValidationError(f"profile result {field} is non-finite")


def select_shared_arm_profile(
    arm: str,
    frozen_profiles: Sequence[Mapping[str, Any]],
    profile_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arm_id = str(arm)
    if arm_id not in ARMS:
        raise CalibrationValidationError(f"unknown calibration arm: {arm_id}")
    profiles = tuple(deepcopy(dict(value)) for value in frozen_profiles)
    if len(profiles) != 4:
        raise CalibrationValidationError(f"{arm_id} must have exactly four frozen profiles")
    profile_by_id = {str(row.get("profile_id", "")): row for row in profiles}
    if "" in profile_by_id or len(profile_by_id) != 4:
        raise CalibrationValidationError(f"{arm_id} frozen profile IDs are invalid")
    results = tuple(deepcopy(dict(value)) for value in profile_results)
    if len(results) != 4:
        raise CalibrationValidationError(f"{arm_id} must have exactly four profile results")
    for result in results:
        _validate_profile_result(result)
        if str(result.get("arm")) != arm_id:
            raise CalibrationValidationError("profile result arm drift")
    result_by_id = {str(result.get("profile_id", "")): result for result in results}
    if set(result_by_id) != set(profile_by_id) or len(result_by_id) != 4:
        raise CalibrationValidationError("profile result IDs differ from frozen profiles")
    bundle_hashes = {str(result["calibration_bundle_sha256"]) for result in results}
    if len(bundle_hashes) != 1:
        raise CalibrationValidationError("profile results do not share one calibration bundle")
    ranking = sorted(
        (
            {
                "profile_id": profile_id,
                "mean_dataset_ratio_inner_mrr": float(result_by_id[profile_id]["mean_dataset_ratio_inner_mrr"]),
                "mean_raw_inner_mrr": float(result_by_id[profile_id]["mean_raw_inner_mrr"]),
                "mean_hit_at_1": float(result_by_id[profile_id]["mean_hit_at_1"]),
                "profile_result_sha256": str(result_by_id[profile_id]["profile_result_sha256"]),
            }
            for profile_id in profile_by_id
        ),
        key=lambda row: (
            -row["mean_dataset_ratio_inner_mrr"],
            -row["mean_raw_inner_mrr"],
            -row["mean_hit_at_1"],
            row["profile_id"],
        ),
    )
    selected_id = ranking[0]["profile_id"]
    selected_profile = deepcopy(profile_by_id[selected_id])
    identity = {
        "schema_version": "conservative-lofo-shared-arm-profile-v1",
        "arm": arm_id,
        "calibration_bundle_sha256": next(iter(bundle_hashes)),
        "frozen_profiles_sha256": _semantic_hash(profiles),
        "selected_profile_id": selected_id,
        "selected_profile": selected_profile,
        "selected_profile_sha256": _semantic_hash(selected_profile),
        "profile_ranking": ranking,
        "selection_rule": "mean_dataset_ratio_inner_mrr_then_raw_mrr_then_hit_at_1_then_profile_id",
        "dataset_specific_overrides": False,
        "held_out_fold_overrides": False,
    }
    return {**identity, "selection_sha256": _semantic_hash(identity)}


def build_selected_profile_registry(
    profiles_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    results_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    calibration_bundle_sha256: str,
) -> dict[str, Any]:
    if tuple(profiles_by_arm) != ARMS or tuple(results_by_arm) != ARMS:
        raise CalibrationValidationError("selected registry requires exactly all three arms in order")
    if not _valid_sha256(calibration_bundle_sha256):
        raise CalibrationValidationError("selected registry calibration hash is invalid")
    selections = {
        arm: select_shared_arm_profile(arm, profiles_by_arm[arm], results_by_arm[arm])
        for arm in ARMS
    }
    if any(
        selection["calibration_bundle_sha256"] != calibration_bundle_sha256
        for selection in selections.values()
    ):
        raise CalibrationValidationError("selected profile uses another calibration bundle")
    identity = {
        "schema_version": "conservative-lofo-selected-profile-registry-v1",
        "calibration_bundle_sha256": calibration_bundle_sha256,
        "development_label_cost": 60,
        "arm_ids": ARMS,
        "selections": selections,
        "dataset_specific_overrides": False,
        "held_out_fold_overrides": False,
    }
    registry = {**identity, "selected_profile_registry_sha256": _semantic_hash(identity)}
    validate_selected_profile_registry(
        registry,
        profiles_by_arm,
        calibration_bundle_sha256=calibration_bundle_sha256,
    )
    return registry


def validate_selected_profile_registry(
    registry: Mapping[str, Any],
    profiles_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    calibration_bundle_sha256: str,
) -> dict[str, Any]:
    if registry.get("schema_version") != "conservative-lofo-selected-profile-registry-v1":
        raise CalibrationValidationError("unexpected selected profile registry schema")
    if str(registry.get("calibration_bundle_sha256")) != str(calibration_bundle_sha256):
        raise CalibrationValidationError("selected registry calibration bundle drift")
    if int(registry.get("development_label_cost", -1)) != 60:
        raise CalibrationValidationError("selected registry development label cost drift")
    if bool(registry.get("dataset_specific_overrides", True)):
        raise CalibrationValidationError("dataset-specific profile overrides are forbidden")
    if bool(registry.get("held_out_fold_overrides", True)):
        raise CalibrationValidationError("held-out-fold profile overrides are forbidden")
    if tuple(registry.get("arm_ids", ())) != ARMS or tuple(profiles_by_arm) != ARMS:
        raise CalibrationValidationError("selected registry arm scope drift")
    selections = dict(registry.get("selections", {}))
    if tuple(selections) != ARMS:
        raise CalibrationValidationError("selected registry must contain one selection per arm")
    for arm in ARMS:
        selection = dict(selections[arm])
        if bool(selection.get("dataset_specific_overrides", True)):
            raise CalibrationValidationError("dataset-specific profile override detected")
        if bool(selection.get("held_out_fold_overrides", True)):
            raise CalibrationValidationError("held-out-fold profile override detected")
        if str(selection.get("arm")) != arm:
            raise CalibrationValidationError("selected profile arm drift")
        if str(selection.get("calibration_bundle_sha256")) != calibration_bundle_sha256:
            raise CalibrationValidationError("selected profile calibration drift")
        profile_by_id = {
            str(dict(profile).get("profile_id", "")): deepcopy(dict(profile))
            for profile in profiles_by_arm[arm]
        }
        if len(profile_by_id) != 4:
            raise CalibrationValidationError(f"{arm} frozen profile count drift")
        selected_id = str(selection.get("selected_profile_id", ""))
        if selected_id not in profile_by_id or selection.get("selected_profile") != profile_by_id[selected_id]:
            raise CalibrationValidationError("selected profile does not match frozen table")
        if selection.get("selected_profile_sha256") != _semantic_hash(profile_by_id[selected_id]):
            raise CalibrationValidationError("selected profile hash drift")
        identity = {key: deepcopy(value) for key, value in selection.items() if key != "selection_sha256"}
        if selection.get("selection_sha256") != _semantic_hash(identity):
            raise CalibrationValidationError("selected profile selection hash drift")
    identity = {
        key: deepcopy(value)
        for key, value in registry.items()
        if key != "selected_profile_registry_sha256"
    }
    if registry.get("selected_profile_registry_sha256") != _semantic_hash(identity):
        raise CalibrationValidationError("selected profile registry hash drift")
    return {
        "valid": True,
        "selected_arm_count": len(selections),
        "development_label_cost": 60,
        "selected_profile_registry_sha256": registry["selected_profile_registry_sha256"],
    }


__all__ = [
    "ARMS",
    "DATASETS",
    "CalibrationValidationError",
    "build_calibration_bundle",
    "build_calibration_profile_result",
    "build_nested_queried_type_folds",
    "build_selected_profile_registry",
    "select_shared_arm_profile",
    "validate_selected_profile_registry",
]
