from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


DATASETS = ("rcabench", "aiops2022_pre")
ARMS = ("oser_meta", "mm_dro", "cope_gate")


class FormalCalibrationValidationError(ValueError):
    """Raised when formal calibration ownership, membership, or hashes drift."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_sha(value: Any, field: str) -> str:
    text = str(value).lower()
    if not _valid_sha(text):
        raise FormalCalibrationValidationError(f"{field} must be a SHA-256 value")
    return text


def _require_unique_ids(values: Sequence[Any], field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or "" in normalized or len(normalized) != len(set(normalized)):
        raise FormalCalibrationValidationError(f"{field} must be unique and nonempty")
    return normalized


def _validate_semantic_hash(value: Mapping[str, Any], hash_field: str, message: str) -> None:
    identity = {key: deepcopy(item) for key, item in value.items() if key != hash_field}
    if value.get(hash_field) != _hash(identity):
        raise FormalCalibrationValidationError(message)


def build_nested_calibration_folds(
    dataset_id: str,
    queried_case_ids: Sequence[str],
    labels_by_case: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], ...]:
    dataset = str(dataset_id).strip()
    if dataset not in DATASETS:
        raise FormalCalibrationValidationError(f"unknown formal calibration dataset: {dataset}")
    queried = _require_unique_ids(queried_case_ids, "queried_case_ids")
    if len(queried) != 30:
        raise FormalCalibrationValidationError(f"{dataset} must contain exactly 30 queried labels")
    raw_labels = {str(key): dict(value) for key, value in labels_by_case.items()}
    if set(raw_labels) != set(queried):
        raise FormalCalibrationValidationError(f"{dataset} label coverage must equal queried cases")
    labels: dict[str, dict[str, str]] = {}
    groups: dict[str, list[str]] = {}
    for case_id in queried:
        row = raw_labels[case_id]
        if set(row) != {"root_cause", "fault_type"}:
            raise FormalCalibrationValidationError(f"{dataset} label fields drifted for {case_id}")
        root_cause = str(row["root_cause"]).strip()
        fault_type = str(row["fault_type"]).strip()
        if not root_cause or not fault_type:
            raise FormalCalibrationValidationError(f"{dataset} label is empty for {case_id}")
        labels[case_id] = {"root_cause": root_cause, "fault_type": fault_type}
        groups.setdefault(fault_type, []).append(case_id)
    if len(groups) < 2:
        raise FormalCalibrationValidationError(
            f"{dataset} nested pseudo-LOFO requires at least two queried fault types"
        )
    folds = []
    for index, fault_type in enumerate(sorted(groups)):
        validation_set = set(groups[fault_type])
        validation = tuple(case_id for case_id in queried if case_id in validation_set)
        support = tuple(case_id for case_id in queried if case_id not in validation_set)
        identity = {
            "schema_version": "conservative-lofo-formal-pseudo-fold-v1",
            "dataset_id": dataset,
            "fold_id": f"fold-{index:03d}",
            "pseudo_held_out_fault_type": fault_type,
            "support_case_ids": support,
            "validation_case_ids": validation,
        }
        folds.append({**identity, "fold_sha256": _hash(identity)})
    return tuple(folds)


def _validate_source_manifest(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if manifest.get("schema_version") != "conservative-lofo-calibration-manifest-v1":
        raise FormalCalibrationValidationError("unexpected calibration manifest schema")
    _validate_semantic_hash(manifest, "manifest_sha256", "calibration manifest hash drift")
    if tuple(manifest.get("dataset_ids", ())) != DATASETS or tuple(
        manifest.get("arm_ids", ())
    ) != ARMS:
        raise FormalCalibrationValidationError("formal calibration dataset/arm registry drift")
    if int(manifest.get("development_label_cost", -1)) != 60:
        raise FormalCalibrationValidationError("formal calibration development label cost drift")
    units = tuple(dict(value) for value in manifest.get("units", ()))
    if int(manifest.get("expected_unit_count", -1)) != 12 or len(units) != 12:
        raise FormalCalibrationValidationError("formal calibration requires exactly 12 units")
    unit_ids = []
    arm_counts: Counter[str] = Counter()
    for unit in units:
        if unit.get("schema_version") != "conservative-lofo-calibration-unit-v1":
            raise FormalCalibrationValidationError("unexpected calibration unit schema")
        _validate_semantic_hash(unit, "unit_sha256", "calibration unit hash drift")
        unit_id = str(unit.get("unit_id", "")).strip()
        arm = str(unit.get("arm", "")).strip()
        profile_id = str(unit.get("profile_id", "")).strip()
        profile = dict(unit.get("profile", {}))
        if not unit_id or arm not in ARMS or not profile_id:
            raise FormalCalibrationValidationError("calibration unit identity is incomplete")
        if profile.get("profile_id") != profile_id or unit.get("profile_sha256") != _hash(profile):
            raise FormalCalibrationValidationError("calibration unit profile hash drift")
        forbidden = {
            "dataset_profile_overrides",
            "held_out_fold_overrides",
            "fold_profile_overrides",
            "checkpoint_root",
        }
        if forbidden & set(unit):
            raise FormalCalibrationValidationError("dataset/fold/checkpoint override is forbidden")
        if tuple(unit.get("consumed_screening_target_fault_types", ())) != ():
            raise FormalCalibrationValidationError("screening target consumption is forbidden")
        unit_ids.append(unit_id)
        arm_counts[arm] += 1
    if len(unit_ids) != len(set(unit_ids)) or arm_counts != Counter({arm: 4 for arm in ARMS}):
        raise FormalCalibrationValidationError("formal calibration unit registry must be 4 profiles per arm")
    return units


def _bind_authority_score_bundles(
    dataset_id: str,
    folds: Sequence[Mapping[str, Any]],
    raw_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    bindings = {str(key): dict(value) for key, value in raw_bindings.items()}
    expected_ids = tuple(str(fold["fold_id"]) for fold in folds)
    if tuple(bindings) != expected_ids:
        raise FormalCalibrationValidationError(
            f"{dataset_id} authority bundle fold registry drift"
        )
    bound_folds = []
    for raw_fold in folds:
        fold = dict(raw_fold)
        fold_id = str(fold["fold_id"])
        binding = bindings[fold_id]
        support = tuple(str(value) for value in fold["support_case_ids"])
        evaluation = tuple(str(value) for value in fold["validation_case_ids"])
        if (
            str(binding.get("fold_id", "")) != fold_id
            or str(binding.get("held_out_fault_type", ""))
            != str(fold["pseudo_held_out_fault_type"])
            or tuple(str(value) for value in binding.get("support_case_ids", ())) != support
            or tuple(str(value) for value in binding.get("evaluation_case_ids", ()))
            != evaluation
        ):
            raise FormalCalibrationValidationError(
                f"{dataset_id} authority bundle membership drift for {fold_id}"
            )
        relative_path = str(binding.get("authority_score_bundle_path", "")).replace("\\", "/")
        parsed_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or parsed_path.suffix != ".json"
        ):
            raise FormalCalibrationValidationError(
                f"{dataset_id} authority bundle path is invalid for {fold_id}"
            )
        score_hashes = dict(binding.get("authority_score_artifact_sha256", {}))
        if set(score_hashes) != {"support", "evaluation"}:
            raise FormalCalibrationValidationError(
                f"{dataset_id} authority score artifact registry drift for {fold_id}"
            )
        authority_fields = {
            "authority_score_bundle_path": relative_path,
            "authority_score_bundle_sha256": _require_sha(
                binding.get("authority_score_bundle_sha256"),
                f"{dataset_id}.{fold_id}.authority_score_bundle_sha256",
            ),
            "authority_model_sha256": _require_sha(
                binding.get("authority_model_sha256"),
                f"{dataset_id}.{fold_id}.authority_model_sha256",
            ),
            "authority_feature_order_sha256": _require_sha(
                binding.get("authority_feature_order_sha256"),
                f"{dataset_id}.{fold_id}.authority_feature_order_sha256",
            ),
            "authority_score_artifact_sha256": {
                role: _require_sha(
                    score_hashes[role],
                    f"{dataset_id}.{fold_id}.{role}_score_artifact_sha256",
                )
                for role in ("support", "evaluation")
            },
        }
        identity = {
            key: deepcopy(value)
            for key, value in fold.items()
            if key != "fold_sha256"
        }
        identity.update(authority_fields)
        bound_folds.append({**identity, "fold_sha256": _hash(identity)})
    return tuple(bound_folds)


def build_formal_calibration_execution(
    *,
    calibration_manifest: Mapping[str, Any],
    datasets_by_id: Mapping[str, Mapping[str, Any]],
    code_sha256: str,
    output_root: str,
) -> dict[str, Any]:
    units = _validate_source_manifest(calibration_manifest)
    code_hash = _require_sha(code_sha256, "code_sha256")
    root = str(output_root).strip().rstrip("/")
    if not root or not (
        PurePosixPath(root).is_absolute() or PureWindowsPath(root).is_absolute()
    ):
        raise FormalCalibrationValidationError("output_root must be an absolute path")
    if tuple(datasets_by_id) != DATASETS:
        raise FormalCalibrationValidationError("formal calibration requires both datasets in order")
    datasets: dict[str, dict[str, Any]] = {}
    for dataset_id in DATASETS:
        raw = dict(datasets_by_id[dataset_id])
        if str(raw.get("dataset_id", "")) != dataset_id:
            raise FormalCalibrationValidationError("formal calibration dataset identity drift")
        queried = _require_unique_ids(raw.get("queried_case_ids", ()), "queried_case_ids")
        if len(queried) != 30:
            raise FormalCalibrationValidationError(f"{dataset_id} must contain exactly 30 queried labels")
        fit_case_ids = _require_unique_ids(raw.get("fit_case_ids", ()), "fit_case_ids")
        if not set(queried) <= set(fit_case_ids):
            raise FormalCalibrationValidationError("queried cases must belong to the union-excluded fit pool")
        screening = {str(value) for value in raw.get("screening_target_case_ids", ())}
        consumed_screening = screening & (set(fit_case_ids) | set(queried))
        if consumed_screening:
            raise FormalCalibrationValidationError("screening target case consumption is forbidden")
        formal = {str(value) for value in raw.get("formal_training_case_ids", ())}
        if formal & set(queried):
            raise FormalCalibrationValidationError("formal training overlap with development labels")
        membership_folds = build_nested_calibration_folds(
            dataset_id, queried, dict(raw.get("labels_by_case", {}))
        )
        folds = _bind_authority_score_bundles(
            dataset_id,
            membership_folds,
            dict(raw.get("authority_score_bundles_by_fold", {})),
        )
        coverage = Counter(
            case_id for fold in folds for case_id in fold["validation_case_ids"]
        )
        if set(coverage) != set(queried) or set(coverage.values()) != {1}:
            raise FormalCalibrationValidationError("validation coverage must include each queried case once")
        for field in ("query_plan_sha256", "cases_sha256", "transform_sha256"):
            _require_sha(raw.get(field), f"{dataset_id}.{field}")
        identity = {
            "dataset_id": dataset_id,
            "queried_case_ids": queried,
            "fit_case_ids": fit_case_ids,
            "labels_by_case": {
                case_id: {
                    "root_cause": str(raw["labels_by_case"][case_id]["root_cause"]).strip(),
                    "fault_type": str(raw["labels_by_case"][case_id]["fault_type"]).strip(),
                }
                for case_id in queried
            },
            "screening_target_case_ids": tuple(sorted(screening)),
            "formal_training_case_ids": tuple(sorted(formal)),
            "query_plan_sha256": str(raw["query_plan_sha256"]),
            "cases_sha256": str(raw["cases_sha256"]),
            "transform_sha256": str(raw["transform_sha256"]),
            "folds": folds,
            "pseudo_lofo_fold_count": len(folds),
            "authority_score_bundle_count": len(folds),
            "validation_coverage_count": len(coverage),
            "screening_target_consumption_count": 0,
            "formal_training_overlap_count": 0,
        }
        datasets[dataset_id] = {
            **identity,
            "dataset_execution_sha256": _hash(identity),
        }
    execution_units = []
    for source in units:
        unit_id = str(source["unit_id"])
        checkpoint_root = f"{root}/checkpoints/{unit_id}"
        identity = {
            "schema_version": "conservative-lofo-formal-calibration-unit-v1",
            "unit_id": unit_id,
            "unit_sha256": str(source["unit_sha256"]),
            "arm": str(source["arm"]),
            "profile_id": str(source["profile_id"]),
            "profile": deepcopy(source["profile"]),
            "profile_sha256": str(source["profile_sha256"]),
            "calibration_bundle_sha256": str(calibration_manifest["calibration_bundle_sha256"]),
            "checkpoint_root": checkpoint_root,
            "dataset_fold_sha256": {
                dataset: tuple(fold["fold_sha256"] for fold in datasets[dataset]["folds"])
                for dataset in DATASETS
            },
            "dataset_authority_score_bundle_sha256": {
                dataset: tuple(
                    fold["authority_score_bundle_sha256"]
                    for fold in datasets[dataset]["folds"]
                )
                for dataset in DATASETS
            },
            "dataset_specific_profile_overrides": False,
            "fold_specific_profile_overrides": False,
        }
        execution_units.append({**identity, "execution_unit_sha256": _hash(identity)})
    identity = {
        "schema_version": "conservative-lofo-formal-calibration-execution-v1",
        "manifest_sha256": str(calibration_manifest["manifest_sha256"]),
        "calibration_bundle_sha256": str(calibration_manifest["calibration_bundle_sha256"]),
        "code_sha256": code_hash,
        "output_root": root,
        "dataset_ids": DATASETS,
        "arm_ids": ARMS,
        "datasets": datasets,
        "development_label_cost": 60,
        "formal_training_overlap_count": 0,
        "screening_target_consumption_count": 0,
        "expected_unit_count": 12,
        "units": execution_units,
        "dataset_specific_profile_overrides": False,
        "fold_specific_profile_overrides": False,
    }
    result = {**identity, "execution_sha256": _hash(identity)}
    validate_formal_calibration_execution(result)
    return result


def validate_formal_calibration_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(value)
    if execution.get("schema_version") != "conservative-lofo-formal-calibration-execution-v1":
        raise FormalCalibrationValidationError("unexpected formal calibration execution schema")
    if tuple(execution.get("dataset_ids", ())) != DATASETS or tuple(
        execution.get("arm_ids", ())
    ) != ARMS:
        raise FormalCalibrationValidationError("formal calibration dataset/arm drift")
    if execution.get("dataset_specific_profile_overrides") is not False or execution.get(
        "fold_specific_profile_overrides"
    ) is not False:
        raise FormalCalibrationValidationError("dataset/fold-specific profile override is forbidden")
    datasets = dict(execution.get("datasets", {}))
    if len(datasets) != len(DATASETS) or set(datasets) != set(DATASETS):
        raise FormalCalibrationValidationError("formal calibration dataset registry drift")
    total_coverage = 0
    for dataset_id in DATASETS:
        dataset = dict(datasets[dataset_id])
        queried = tuple(str(value) for value in dataset.get("queried_case_ids", ()))
        if len(queried) != 30 or len(set(queried)) != 30:
            raise FormalCalibrationValidationError(f"{dataset_id} must contain exactly 30 labels")
        coverage: Counter[str] = Counter()
        fold_hashes = []
        authority_bundle_hashes = []
        for fold in dataset.get("folds", ()):
            row = dict(fold)
            support = set(str(value) for value in row.get("support_case_ids", ()))
            validation = set(str(value) for value in row.get("validation_case_ids", ()))
            if support & validation:
                raise FormalCalibrationValidationError("pseudo-fold support/validation overlap")
            if support | validation != set(queried):
                raise FormalCalibrationValidationError("pseudo-fold membership does not cover cohort")
            coverage.update(str(value) for value in row.get("validation_case_ids", ()))
            _validate_semantic_hash(row, "fold_sha256", "pseudo-fold hash drift")
            fold_hashes.append(str(row["fold_sha256"]))
            relative_path = str(row.get("authority_score_bundle_path", "")).replace("\\", "/")
            parsed_path = PurePosixPath(relative_path)
            if (
                not relative_path
                or parsed_path.is_absolute()
                or ".." in parsed_path.parts
                or parsed_path.suffix != ".json"
            ):
                raise FormalCalibrationValidationError("authority score bundle path drift")
            authority_bundle_hashes.append(
                _require_sha(
                    row.get("authority_score_bundle_sha256"),
                    "authority_score_bundle_sha256",
                )
            )
            _require_sha(row.get("authority_model_sha256"), "authority_model_sha256")
            _require_sha(
                row.get("authority_feature_order_sha256"),
                "authority_feature_order_sha256",
            )
            score_hashes = dict(row.get("authority_score_artifact_sha256", {}))
            if set(score_hashes) != {"support", "evaluation"}:
                raise FormalCalibrationValidationError("authority score artifact hash drift")
            for role in ("support", "evaluation"):
                _require_sha(score_hashes[role], f"authority_{role}_score_artifact_sha256")
        if set(coverage) != set(queried) or set(coverage.values()) != {1}:
            raise FormalCalibrationValidationError("validation coverage must equal each case once")
        screening = set(str(value) for value in dataset.get("screening_target_case_ids", ()))
        if screening & (
            set(str(value) for value in dataset.get("fit_case_ids", ())) | set(queried)
        ):
            raise FormalCalibrationValidationError("screening target case consumption is forbidden")
        if set(str(value) for value in dataset.get("formal_training_case_ids", ())) & set(queried):
            raise FormalCalibrationValidationError("formal training overlap with development labels")
        if int(dataset.get("validation_coverage_count", -1)) != 30:
            raise FormalCalibrationValidationError("validation coverage count drift")
        if (
            int(dataset.get("authority_score_bundle_count", -1)) != len(fold_hashes)
            or len(authority_bundle_hashes) != len(set(authority_bundle_hashes))
        ):
            raise FormalCalibrationValidationError("authority score bundle coverage drift")
        total_coverage += len(coverage)
        _validate_semantic_hash(
            dataset, "dataset_execution_sha256", "dataset execution hash drift"
        )
    units = tuple(dict(value) for value in execution.get("units", ()))
    if int(execution.get("expected_unit_count", -1)) != 12 or len(units) != 12:
        raise FormalCalibrationValidationError("formal calibration requires exactly 12 units")
    expected_root = str(execution.get("output_root", "")).rstrip("/")
    checkpoint_roots = []
    for unit in units:
        unexpected_override_keys = {
            str(key) for key in unit if "override" in str(key).lower()
        } - {"dataset_specific_profile_overrides", "fold_specific_profile_overrides"}
        if unexpected_override_keys:
            raise FormalCalibrationValidationError("unit dataset/fold override is forbidden")
        if unit.get("dataset_specific_profile_overrides") is not False or unit.get(
            "fold_specific_profile_overrides"
        ) is not False:
            raise FormalCalibrationValidationError("unit dataset/fold override is forbidden")
        unit_id = str(unit.get("unit_id", ""))
        expected_checkpoint = f"{expected_root}/checkpoints/{unit_id}"
        if unit.get("checkpoint_root") != expected_checkpoint:
            raise FormalCalibrationValidationError("unit checkpoint root ownership drift")
        checkpoint_roots.append(str(unit["checkpoint_root"]))
        expected_folds = {
            dataset: tuple(fold["fold_sha256"] for fold in datasets[dataset]["folds"])
            for dataset in DATASETS
        }
        raw_fold_hashes = unit.get("dataset_fold_sha256")
        if not isinstance(raw_fold_hashes, Mapping):
            raise FormalCalibrationValidationError("unit pseudo-fold binding drift")
        raw_fold_hashes = dict(raw_fold_hashes)
        if len(raw_fold_hashes) != len(DATASETS) or set(raw_fold_hashes) != set(DATASETS):
            raise FormalCalibrationValidationError("unit pseudo-fold binding drift")
        normalized_fold_hashes = {
            dataset: tuple(str(value) for value in raw_fold_hashes[dataset])
            for dataset in DATASETS
        }
        if normalized_fold_hashes != expected_folds:
            raise FormalCalibrationValidationError("unit pseudo-fold binding drift")
        expected_authority = {
            dataset: tuple(
                fold["authority_score_bundle_sha256"]
                for fold in datasets[dataset]["folds"]
            )
            for dataset in DATASETS
        }
        raw_authority = unit.get("dataset_authority_score_bundle_sha256")
        if not isinstance(raw_authority, Mapping):
            raise FormalCalibrationValidationError("unit authority bundle binding drift")
        normalized_authority = {
            dataset: tuple(str(value) for value in raw_authority.get(dataset, ()))
            for dataset in DATASETS
        }
        if normalized_authority != expected_authority:
            raise FormalCalibrationValidationError("unit authority bundle binding drift")
        _validate_semantic_hash(unit, "execution_unit_sha256", "execution unit hash drift")
    if len(checkpoint_roots) != len(set(checkpoint_roots)):
        raise FormalCalibrationValidationError("checkpoint root must be unique per unit")
    if int(execution.get("development_label_cost", -1)) != 60 or total_coverage != 60:
        raise FormalCalibrationValidationError("formal calibration sixty-label coverage drift")
    if int(execution.get("formal_training_overlap_count", -1)) != 0 or int(
        execution.get("screening_target_consumption_count", -1)
    ) != 0:
        raise FormalCalibrationValidationError("formal calibration forbidden overlap drift")
    _validate_semantic_hash(execution, "execution_sha256", "formal calibration execution hash drift")
    return {
        "valid": True,
        "dataset_count": 2,
        "unit_count": 12,
        "development_label_cost": 60,
        "validation_coverage_count": 60,
    }


def _unit_interval(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalCalibrationValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FormalCalibrationValidationError(f"{field} must be finite in [0,1]")
    return result


def build_formal_unit_metrics(
    *,
    unit: Mapping[str, Any],
    dataset_metrics: Mapping[str, Mapping[str, Any]],
    fold_audits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if unit.get("schema_version") != "conservative-lofo-formal-calibration-unit-v1":
        raise FormalCalibrationValidationError("unexpected formal calibration unit schema")
    _validate_semantic_hash(unit, "execution_unit_sha256", "execution unit hash drift")
    if tuple(dataset_metrics) != DATASETS or tuple(fold_audits) != DATASETS:
        raise FormalCalibrationValidationError("unit metrics must cover both datasets in order")
    normalized_metrics: dict[str, dict[str, Any]] = {}
    normalized_audits: dict[str, tuple[dict[str, Any], ...]] = {}
    invocation_count = 0
    checkpoint_roots = []
    for dataset_id in DATASETS:
        raw = dict(dataset_metrics[dataset_id])
        metric = {
            "baseline_inner_mrr": _unit_interval(raw.get("baseline_inner_mrr"), "baseline_inner_mrr"),
            "profile_inner_mrr": _unit_interval(raw.get("profile_inner_mrr"), "profile_inner_mrr"),
            "profile_inner_hit_at_1": _unit_interval(raw.get("profile_inner_hit_at_1"), "profile_inner_hit_at_1"),
            "pseudo_lofo_fold_count": int(raw.get("pseudo_lofo_fold_count", 0)),
        }
        expected_hashes = tuple(unit["dataset_fold_sha256"][dataset_id])
        audits = tuple(dict(value) for value in fold_audits[dataset_id])
        if metric["pseudo_lofo_fold_count"] != len(expected_hashes) or len(audits) != len(
            expected_hashes
        ):
            raise FormalCalibrationValidationError("unit pseudo-fold metric coverage drift")
        for audit, expected_hash in zip(audits, expected_hashes):
            if audit.get("fold_sha256") != expected_hash:
                raise FormalCalibrationValidationError("unit fold audit hash drift")
            count = int(audit.get("actual_trainer_invocation_count", 0))
            if count != 1:
                raise FormalCalibrationValidationError("each pseudo-fold must invoke the trainer once")
            if int(audit.get("held_out_fitting_overlap_count", -1)) != 0:
                raise FormalCalibrationValidationError("held-out fitting overlap is forbidden")
            checkpoint = str(audit.get("checkpoint_root", ""))
            if not checkpoint.startswith(str(unit["checkpoint_root"]) + "/"):
                raise FormalCalibrationValidationError("fold checkpoint root ownership drift")
            checkpoint_roots.append(checkpoint)
            invocation_count += count
        normalized_metrics[dataset_id] = metric
        normalized_audits[dataset_id] = audits
    if len(checkpoint_roots) != len(set(checkpoint_roots)):
        raise FormalCalibrationValidationError("fold checkpoint roots must be unique")
    identity = {
        "schema_version": "conservative-lofo-calibration-unit-metrics-v1",
        "unit_id": str(unit["unit_id"]),
        "unit_sha256": str(unit["unit_sha256"]),
        "execution_unit_sha256": str(unit["execution_unit_sha256"]),
        "calibration_bundle_sha256": str(unit["calibration_bundle_sha256"]),
        "arm": str(unit["arm"]),
        "profile_id": str(unit["profile_id"]),
        "consumed_screening_target_fault_types": (),
        "dataset_metrics": normalized_metrics,
        "fold_audits": normalized_audits,
        "actual_trainer_invocation_count": invocation_count,
        "held_out_fitting_overlap_count": 0,
    }
    return {**identity, "metrics_sha256": _hash(identity)}


__all__ = [
    "ARMS",
    "DATASETS",
    "FormalCalibrationValidationError",
    "build_formal_calibration_execution",
    "build_formal_unit_metrics",
    "build_nested_calibration_folds",
    "validate_formal_calibration_execution",
]
