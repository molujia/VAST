from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


class OrchestrationValidationError(ValueError):
    """Raised when an initial-screening manifest or unit evidence is unsafe."""


DATASETS = ("rcabench", "aiops2022_pre")
ARMS = ("baseline", "oser_meta", "mm_dro", "cope_gate")
ENHANCED_ARMS = ARMS[1:]
EXPECTED_TARGETS = {
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


def _canonical(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _valid_hash(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _validate_input_hashes(record: Mapping[str, Any], context: str) -> None:
    for field in (
        "membership_sha256",
        "query_plan_sha256",
        "query_event_history_sha256",
        "label_ledger_sha256",
        "base_score_sha256",
        "authority_score_bundle_sha256",
    ):
        if not _valid_hash(record.get(field)):
            raise OrchestrationValidationError(f"{context} {field} is invalid")


def _unit_id(kind: str, dataset: str, arm: str, target: str | None) -> str:
    target_token = "ordinary" if target is None else _hash(target)[:12]
    return f"{kind}__{dataset}__{target_token}__{arm}"


def _build_unit(
    *,
    kind: str,
    dataset_id: str,
    arm: str,
    target: str | None,
    source: Mapping[str, Any],
    registry: Mapping[str, Any],
    code_sha256: str,
    output_root: str,
) -> dict[str, Any]:
    unit_id = _unit_id(kind, dataset_id, arm, target)
    if arm == "baseline":
        profile_id = "unchanged_pairwise_linear"
        profile_sha = str(source["base_score_sha256"])
        registry_sha = None
        profile = None
    else:
        selection = dict(registry["selections"][arm])
        profile_id = str(selection["selected_profile_id"])
        profile_sha = str(selection["selected_profile_sha256"])
        registry_sha = str(registry["selected_profile_registry_sha256"])
        profile = deepcopy(selection.get("selected_profile"))
    unit_root = str(PurePosixPath(output_root) / "units" / unit_id)
    identity = {
        "schema_version": "conservative-lofo-initial-screen-unit-v1",
        "unit_id": unit_id,
        "kind": kind,
        "dataset_id": dataset_id,
        "held_out_fault_type": target,
        "arm": arm,
        "seed": 42,
        "budget": 30,
        "membership_sha256": str(source["membership_sha256"]),
        "query_plan_sha256": str(source["query_plan_sha256"]),
        "query_event_history_sha256": str(source["query_event_history_sha256"]),
        "label_ledger_sha256": str(source["label_ledger_sha256"]),
        "base_score_sha256": str(source["base_score_sha256"]),
        "authority_score_bundle_sha256": str(source["authority_score_bundle_sha256"]),
        "code_sha256": str(code_sha256),
        "selected_profile_registry_sha256": registry_sha,
        "selected_profile_id": profile_id,
        "selected_profile_sha256": profile_sha,
        "profile": profile,
        "checkpoint_owner": f"{unit_id}::checkpoint",
        "checkpoint_path": str(PurePosixPath(output_root) / "checkpoints" / unit_id / arm),
        "resource_class": "cpu" if arm == "baseline" else "gpu",
        "output_root": unit_root,
        "expected_artifacts": ("result.json", "per_case.json", "mechanism.json", "COMPLETED.json"),
        "mutable_model_state_shared": False,
    }
    return {**identity, "unit_fingerprint_sha256": _hash(identity)}


def build_initial_screen_manifest(
    *,
    targets_by_dataset: Mapping[str, Sequence[Any]],
    selected_profile_registry: Mapping[str, Any],
    lofo_inputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ordinary_inputs: Mapping[str, Mapping[str, Any]],
    code_sha256: str,
    output_root: str,
) -> dict[str, Any]:
    targets = {dataset: tuple(str(value) for value in values) for dataset, values in targets_by_dataset.items()}
    if targets != EXPECTED_TARGETS:
        raise OrchestrationValidationError("initial screening target inventory drift")
    if not _valid_hash(code_sha256):
        raise OrchestrationValidationError("initial screening code hash is invalid")
    registry = deepcopy(dict(selected_profile_registry))
    if registry.get("schema_version") != "conservative-lofo-selected-profile-registry-v1":
        raise OrchestrationValidationError("selected profile registry schema drift")
    if not _valid_hash(registry.get("selected_profile_registry_sha256")):
        raise OrchestrationValidationError("selected profile registry hash is invalid")
    if tuple(dict(registry.get("selections", {}))) != ENHANCED_ARMS:
        raise OrchestrationValidationError("selected profile registry arm scope drift")
    for arm in ENHANCED_ARMS:
        selection = dict(registry["selections"][arm])
        if not str(selection.get("selected_profile_id", "")).strip() or not _valid_hash(selection.get("selected_profile_sha256")):
            raise OrchestrationValidationError(f"selected profile is invalid for {arm}")
    if tuple(lofo_inputs) != DATASETS or tuple(ordinary_inputs) != DATASETS:
        raise OrchestrationValidationError("initial screening input dataset scope drift")
    units = []
    for dataset in DATASETS:
        if tuple(lofo_inputs[dataset]) != EXPECTED_TARGETS[dataset]:
            raise OrchestrationValidationError(f"LOFO input target scope drift for {dataset}")
        for target in EXPECTED_TARGETS[dataset]:
            source = dict(lofo_inputs[dataset][target])
            _validate_input_hashes(source, f"{dataset}/{target}")
            for arm in ARMS:
                units.append(
                    _build_unit(
                        kind="strict_lofo",
                        dataset_id=dataset,
                        arm=arm,
                        target=target,
                        source=source,
                        registry=registry,
                        code_sha256=code_sha256,
                        output_root=output_root,
                    )
                )
        ordinary_source = dict(ordinary_inputs[dataset])
        _validate_input_hashes(ordinary_source, f"{dataset}/ordinary")
        for arm in ARMS:
            units.append(
                _build_unit(
                    kind="ordinary",
                    dataset_id=dataset,
                    arm=arm,
                    target=None,
                    source=ordinary_source,
                    registry=registry,
                    code_sha256=code_sha256,
                    output_root=output_root,
                )
            )
    identity = {
        "schema_version": "conservative-lofo-initial-screen-manifest-v1",
        "output_root": str(output_root),
        "seed": 42,
        "budget": 30,
        "dataset_ids": DATASETS,
        "arm_ids": ARMS,
        "targets_by_dataset": targets,
        "selected_profile_registry_sha256": registry["selected_profile_registry_sha256"],
        "code_sha256": str(code_sha256),
        "expected_lofo_unit_count": 40,
        "expected_ordinary_unit_count": 8,
        "expected_unit_count": 48,
        "units": units,
        "full_lofo_authorized": False,
        "exhaustive_all_type_lofo": False,
        "automatic_follow_on": False,
        "terminal_state": "pre_full_user_review_required",
    }
    manifest = {**identity, "manifest_sha256": _hash(identity)}
    validate_initial_screen_manifest(manifest)
    return manifest


def validate_initial_screen_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "conservative-lofo-initial-screen-manifest-v1":
        raise OrchestrationValidationError("unexpected initial-screen manifest schema")
    if bool(manifest.get("full_lofo_authorized", True)) or bool(manifest.get("exhaustive_all_type_lofo", True)):
        raise OrchestrationValidationError("exhaustive or full LOFO work is not authorized")
    if bool(manifest.get("automatic_follow_on", True)):
        raise OrchestrationValidationError("automatic follow-on is forbidden")
    for key, value in manifest.items():
        lowered = str(key).lower()
        if ("exhaustive" in lowered or "full_launcher" in lowered) and bool(value):
            raise OrchestrationValidationError("exhaustive launcher field is forbidden")
    if int(manifest.get("seed", -1)) != 42 or int(manifest.get("budget", -1)) != 30:
        raise OrchestrationValidationError("manifest seed or budget drift")
    if tuple(manifest.get("dataset_ids", ())) != DATASETS or tuple(manifest.get("arm_ids", ())) != ARMS:
        raise OrchestrationValidationError("manifest dataset or arm scope drift")
    raw_targets = dict(manifest.get("targets_by_dataset", {}))
    normalized_targets = {
        dataset: tuple(str(value) for value in raw_targets.get(dataset, ()))
        for dataset in DATASETS
    }
    if normalized_targets != EXPECTED_TARGETS:
        raise OrchestrationValidationError("manifest target inventory drift")
    units = tuple(dict(value) for value in manifest.get("units", ()))
    if len(units) != 48 or int(manifest.get("expected_unit_count", -1)) != 48:
        raise OrchestrationValidationError("manifest must contain exactly 48 units")
    lofo = [unit for unit in units if unit.get("kind") == "strict_lofo"]
    ordinary = [unit for unit in units if unit.get("kind") == "ordinary"]
    if len(lofo) != 40 or len(ordinary) != 8:
        raise OrchestrationValidationError("manifest must contain exactly 40 LOFO and 8 ordinary units")
    if len({unit.get("unit_id") for unit in units}) != 48:
        raise OrchestrationValidationError("manifest unit IDs are not unique")
    if len({unit.get("output_root") for unit in units}) != 48:
        raise OrchestrationValidationError("manifest output roots are not isolated")
    if len({unit.get("checkpoint_path") for unit in units}) != 48:
        raise OrchestrationValidationError("manifest checkpoint paths are not isolated")
    if len({unit.get("checkpoint_owner") for unit in units}) != 48:
        raise OrchestrationValidationError("manifest checkpoint ownership is not isolated")
    for unit in units:
        for field in (
            "membership_sha256",
            "query_plan_sha256",
            "query_event_history_sha256",
            "label_ledger_sha256",
            "base_score_sha256",
            "authority_score_bundle_sha256",
            "code_sha256",
            "selected_profile_sha256",
        ):
            if not _valid_hash(unit.get(field)):
                raise OrchestrationValidationError(f"unit hash is invalid: {unit.get('unit_id')} {field}")
        if int(unit.get("seed", -1)) != 42 or int(unit.get("budget", -1)) != 30:
            raise OrchestrationValidationError("unit seed or budget drift")
        identity = {key: deepcopy(value) for key, value in unit.items() if key != "unit_fingerprint_sha256"}
        if unit.get("unit_fingerprint_sha256") != _hash(identity):
            raise OrchestrationValidationError(f"unit fingerprint/hash drift: {unit.get('unit_id')}")
    for dataset, targets in EXPECTED_TARGETS.items():
        for target in targets:
            matched = [unit for unit in lofo if unit["dataset_id"] == dataset and unit["held_out_fault_type"] == target]
            if tuple(unit["arm"] for unit in matched) != ARMS:
                raise OrchestrationValidationError(f"LOFO arm coverage drift for {dataset}/{target}")
            if len({unit["query_plan_sha256"] for unit in matched}) != 1:
                raise OrchestrationValidationError("matched arms do not share one query plan")
            if len({unit["authority_score_bundle_sha256"] for unit in matched}) != 1:
                raise OrchestrationValidationError("matched arms do not share one authority bundle")
    identity = {key: deepcopy(value) for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _hash(identity):
        raise OrchestrationValidationError("manifest hash drift")
    return {"valid": True, "lofo_unit_count": 40, "ordinary_unit_count": 8, "unit_count": 48}


def _validate_unit_identity(unit: Mapping[str, Any]) -> None:
    identity = {key: deepcopy(value) for key, value in unit.items() if key != "unit_fingerprint_sha256"}
    if unit.get("unit_fingerprint_sha256") != _hash(identity):
        raise OrchestrationValidationError("unit fingerprint hash drift")


def build_unit_completion(unit: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    _validate_unit_identity(unit)
    raw = deepcopy(dict(evidence))
    identity = {
        "schema_version": "conservative-lofo-initial-screen-unit-completed-v1",
        "unit_id": str(unit["unit_id"]),
        "unit_fingerprint_sha256": str(unit["unit_fingerprint_sha256"]),
        **raw,
    }
    completion = {**identity, "completion_sha256": _hash(identity)}
    validate_unit_completion(unit, completion)
    return completion


def validate_unit_completion(unit: Mapping[str, Any], completion: Mapping[str, Any]) -> dict[str, Any]:
    _validate_unit_identity(unit)
    if completion.get("status") != "completed":
        raise OrchestrationValidationError(f"unit failed or is not completed: {unit.get('unit_id')}")
    if completion.get("schema_version") != "conservative-lofo-initial-screen-unit-completed-v1":
        raise OrchestrationValidationError("unexpected unit completion schema")
    if str(completion.get("unit_id")) != str(unit.get("unit_id")) or completion.get("unit_fingerprint_sha256") != unit.get("unit_fingerprint_sha256"):
        raise OrchestrationValidationError("unit completion ownership drift")
    if int(completion.get("seed", -1)) != 42 or int(completion.get("seed", -1)) != int(unit["seed"]):
        raise OrchestrationValidationError("unit completion seed drift")
    if int(completion.get("budget", -1)) != 30 or int(completion.get("budget", -1)) != int(unit["budget"]):
        raise OrchestrationValidationError("unit completion budget drift")
    for field in (
        "query_plan_sha256",
        "query_event_history_sha256",
        "label_ledger_sha256",
        "code_sha256",
        "selected_profile_sha256",
        "checkpoint_owner",
    ):
        if completion.get(field) != unit.get(field):
            label = "history" if field == "query_event_history_sha256" else field
            raise OrchestrationValidationError(f"unit completion {label} drift")
    if int(completion.get("held_out_fitting_overlap_count", -1)) != 0:
        raise OrchestrationValidationError("unit completion held-out overlap is nonzero")
    if int(completion.get("selected_case_count", -1)) != 30:
        raise OrchestrationValidationError("unit completion selected budget drift")
    if completion.get("membership_sha256") != unit.get("membership_sha256"):
        raise OrchestrationValidationError("unit completion membership drift")
    if completion.get("base_score_contract_sha256") != unit.get("base_score_sha256"):
        raise OrchestrationValidationError("unit completion base-score contract drift")
    if completion.get("authority_score_bundle_sha256") != unit.get(
        "authority_score_bundle_sha256"
    ):
        raise OrchestrationValidationError("unit completion authority bundle drift")
    if not _valid_hash(completion.get("actual_base_score_sha256")):
        raise OrchestrationValidationError("unit completion actual base-score hash is invalid")
    if not _valid_hash(completion.get("checkpoint_sha256")):
        raise OrchestrationValidationError("unit completion checkpoint hash is invalid")
    expected_checkpoint = str(PurePosixPath(str(unit["checkpoint_path"])) / "checkpoint.json")
    if str(completion.get("checkpoint_path")) != expected_checkpoint:
        raise OrchestrationValidationError("unit completion checkpoint path drift")
    if not bool(completion.get("expected_artifacts_present", False)):
        raise OrchestrationValidationError("unit completion misses expected artifacts")
    per_case = tuple(completion.get("per_case", ()))
    if not per_case:
        raise OrchestrationValidationError("unit completion lacks per-case evidence")
    case_ids = []
    for raw in per_case:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        case_ids.append(case_id)
        candidates = int(row.get("candidate_count", 0))
        ranking = int(row.get("ranking_count", -1))
        if not case_id or candidates <= 0 or ranking != candidates or not bool(row.get("finite_scores", False)):
            raise OrchestrationValidationError("unit per-case ranking is not candidate-complete and finite")
        candidate_ids = tuple(str(value) for value in row.get("candidate_ids", ()))
        ranking_ids = tuple(str(value) for value in row.get("ranking", ()))
        targets = tuple(str(value) for value in row.get("targets", ()))
        base_scores = tuple(float(value) for value in row.get("base_scores", ()))
        gates = tuple(float(value) for value in row.get("gates", ()))
        residuals = tuple(float(value) for value in row.get("residuals", ()))
        final_scores = tuple(float(value) for value in row.get("final_scores", ()))
        if (
            len(candidate_ids) != candidates
            or len(set(candidate_ids)) != candidates
            or len(ranking_ids) != candidates
            or set(ranking_ids) != set(candidate_ids)
            or not targets
            or not set(targets) <= set(candidate_ids)
            or any(len(values) != candidates for values in (base_scores, gates, residuals, final_scores))
        ):
            raise OrchestrationValidationError("unit per-case detailed ranking evidence drift")
        profile = dict(unit.get("profile") or {})
        cap = float(profile.get("residual_cap", 0.0 if unit.get("arm") == "baseline" else 1.0))
        for base, gate, residual, final in zip(base_scores, gates, residuals, final_scores):
            if not all(math.isfinite(value) for value in (base, gate, residual, final)):
                raise OrchestrationValidationError("unit per-case score is non-finite")
            if not 0.0 <= gate <= 1.0 or abs(residual) > cap + 1e-9:
                raise OrchestrationValidationError("unit per-case gate or residual cap drift")
            if abs(final - (base + gate * residual)) > 1e-12:
                raise OrchestrationValidationError("unit per-case residual composition drift")
    if len(case_ids) != len(set(case_ids)):
        raise OrchestrationValidationError("unit per-case evidence contains duplicate IDs")
    identity = {key: deepcopy(value) for key, value in completion.items() if key != "completion_sha256"}
    if completion.get("completion_sha256") != _hash(identity):
        raise OrchestrationValidationError("unit completion hash drift")
    return {"valid": True, "unit_id": unit["unit_id"], "case_count": len(per_case)}


def resume_initial_screen(
    manifest: Mapping[str, Any], completions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_initial_screen_manifest(manifest)
    unit_by_id = {str(unit["unit_id"]): dict(unit) for unit in manifest["units"]}
    completed_by_id = {}
    for raw in completions:
        completion = dict(raw)
        unit_id = str(completion.get("unit_id", ""))
        if completion.get("status") == "failed":
            raise OrchestrationValidationError(f"initial screen contains a failed unit: {unit_id}")
        if unit_id not in unit_by_id or unit_id in completed_by_id:
            raise OrchestrationValidationError("initial screen completion membership drift")
        validate_unit_completion(unit_by_id[unit_id], completion)
        completed_by_id[unit_id] = completion
    pending = tuple(unit_id for unit_id in unit_by_id if unit_id not in completed_by_id)
    identity = {
        "schema_version": "conservative-lofo-initial-screen-progress-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_unit_ids": tuple(completed_by_id),
        "pending_unit_ids": pending,
        "completed_unit_count": len(completed_by_id),
        "pending_unit_count": len(pending),
        "total_unit_count": 48,
        "failure_count": 0,
    }
    return {**identity, "progress_sha256": _hash(identity)}


def finalize_initial_screen(
    manifest: Mapping[str, Any], completions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(tuple(completions)) != 48:
        raise OrchestrationValidationError("premature initial-screen completion")
    progress = resume_initial_screen(manifest, completions)
    if progress["pending_unit_count"] != 0:
        raise OrchestrationValidationError("premature initial-screen completion")
    completion_hashes = {
        str(completion["unit_id"]): str(completion["completion_sha256"])
        for completion in completions
    }
    identity = {
        "schema_version": "conservative-lofo-initial-screen-completed-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_unit_count": 48,
        "failure_count": 0,
        "unit_completion_sha256": completion_hashes,
        "all_done_eligible": True,
        "automatic_follow_on": False,
    }
    return {**identity, "completed_sha256": _hash(identity)}


def build_pre_full_terminal_state(
    manifest: Mapping[str, Any], completed_sha256: str
) -> dict[str, Any]:
    validate_initial_screen_manifest(manifest)
    if not _valid_hash(completed_sha256):
        raise OrchestrationValidationError("completed evidence hash is invalid")
    identity = {
        "schema_version": "conservative-lofo-pre-full-terminal-state-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "initial_screen_completed_sha256": str(completed_sha256),
        "state": "pre_full_user_review_required",
        "full_lofo_authorized": False,
        "automatic_follow_on": False,
        "exhaustive_manifest_created": False,
        "exhaustive_launcher_created": False,
    }
    return {**identity, "terminal_state_sha256": _hash(identity)}


__all__ = [
    "OrchestrationValidationError",
    "build_initial_screen_manifest",
    "build_pre_full_terminal_state",
    "build_unit_completion",
    "finalize_initial_screen",
    "resume_initial_screen",
    "validate_initial_screen_manifest",
    "validate_unit_completion",
]
