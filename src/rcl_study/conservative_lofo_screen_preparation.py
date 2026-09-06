from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from .conservative_lofo_orchestration import (
    ARMS,
    DATASETS,
    ENHANCED_ARMS,
    EXPECTED_TARGETS,
    build_initial_screen_manifest,
    validate_initial_screen_manifest,
)
from .conservative_lofo_authority_scores import validate_authority_score_bundle


REGISTRY_SHA256 = "341c76bc250ddf580afaf81fc36b1536f2d2391616817386dba4950a4576a069"


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


def _valid_hash(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _registry(value: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(value))
    if source.get("schema_version") != "conservative-lofo-selected-profile-registry-v1":
        raise ValueError("selected-profile registry schema drift")
    identity = {key: deepcopy(item) for key, item in source.items() if key != "selected_profile_registry_sha256"}
    if source.get("selected_profile_registry_sha256") != _hash(identity):
        raise ValueError("selected-profile registry hash drift")
    if source["selected_profile_registry_sha256"] != REGISTRY_SHA256:
        raise ValueError("selected-profile registry is not the frozen Task 13 authority")
    selections = dict(source.get("selections", {}))
    if tuple(selections) != ENHANCED_ARMS:
        raise ValueError("selected-profile registry arm scope drift")
    for arm in ENHANCED_ARMS:
        selection = dict(selections[arm])
        profile = dict(selection.get("selected_profile", {}))
        if selection.get("selected_profile_id") != profile.get("profile_id"):
            raise ValueError(f"selected-profile ID drift for {arm}")
        if selection.get("selected_profile_sha256") != _hash(profile):
            raise ValueError(f"selected-profile hash drift for {arm}")
    return source


def _request(value: Mapping[str, Any], dataset: str, target: str | None) -> dict[str, Any]:
    source = deepcopy(dict(value))
    if source.get("schema_version") != "conservative-lofo-materialized-screen-request-v1":
        raise ValueError("materialized request schema drift")
    identity = {key: deepcopy(item) for key, item in source.items() if key != "request_sha256"}
    if source.get("request_sha256") != _hash(identity):
        raise ValueError("materialized request hash drift")
    expected_kind = "ordinary" if target is None else "strict_lofo"
    if source.get("dataset_id") != dataset or source.get("kind") != expected_kind:
        raise ValueError("materialized request dataset/kind drift")
    observed_target = source.get("held_out_fault_type")
    if observed_target != target:
        raise ValueError("materialized request target drift")
    if int(source.get("seed", -1)) != 42 or int(source.get("budget", -1)) != 30:
        raise ValueError("materialized request seed/budget drift")
    if "base_training_steps" in source:
        raise ValueError("Torch substitute base training is forbidden")
    selected = tuple(str(item) for item in source.get("selected_case_ids", ()))
    fit_ids = tuple(str(item) for item in source.get("fit_case_ids", ()))
    evaluation = tuple(str(item) for item in source.get("evaluation_case_ids", ()))
    if len(selected) != 30 or len(set(selected)) != 30 or fit_ids != selected:
        raise ValueError("materialized request must fit exactly thirty selected cases")
    if not evaluation or set(selected) & set(evaluation):
        raise ValueError("materialized request evaluation overlap")
    for field in (
        "membership_sha256",
        "query_plan_sha256",
        "query_event_history_sha256",
        "label_ledger_sha256",
        "base_score_contract_sha256",
        "authority_score_bundle_sha256",
        "authority_model_sha256",
        "authority_feature_order_sha256",
    ):
        if not _valid_hash(source.get(field)):
            raise ValueError(f"materialized request {field} drift")
    bundle = deepcopy(dict(source.get("authority_score_bundle", {})))
    validate_authority_score_bundle(bundle)
    expected_kind = "ordinary" if target is None else "strict_lofo"
    if (
        bundle.get("dataset_id") != dataset
        or bundle.get("fold_kind") != expected_kind
        or bundle.get("held_out_fault_type") != target
        or bundle.get("query_plan_sha256") != source["query_plan_sha256"]
        or bundle.get("authority_score_bundle_sha256")
        != source["authority_score_bundle_sha256"]
        or bundle.get("model_sha256") != source["authority_model_sha256"]
        or bundle.get("feature_order_sha256")
        != source["authority_feature_order_sha256"]
        or dict(bundle.get("score_artifact_sha256", {}))
        != dict(source.get("authority_score_artifact_sha256", {}))
        or tuple(bundle.get("support_case_ids", ())) != selected
        or tuple(bundle.get("evaluation_case_ids", ())) != evaluation
    ):
        raise ValueError("materialized request authority bundle drift")
    relative = str(source.get("authority_score_bundle_path", "")).replace("\\", "/")
    parsed = PurePosixPath(relative)
    if not relative or parsed.is_absolute() or ".." in parsed.parts or parsed.suffix != ".json":
        raise ValueError("materialized request authority bundle path drift")
    return source


def _scope(requests_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    if tuple(requests_by_dataset) != DATASETS:
        raise ValueError("materialized request dataset scope drift")
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        source = dict(requests_by_dataset[dataset])
        expected_keys = EXPECTED_TARGETS[dataset] + ("ordinary",)
        if tuple(source) != expected_keys:
            raise ValueError(f"materialized request target scope drift for {dataset}")
        normalized[dataset] = {}
        for key in expected_keys:
            target = None if key == "ordinary" else key
            normalized[dataset][key] = _request(source[key], dataset, target)
    return normalized


def _manifest_inputs(
    requests: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, str]]]:
    lofo: dict[str, dict[str, dict[str, str]]] = {}
    ordinary: dict[str, dict[str, str]] = {}
    fields = {
        "membership_sha256": "membership_sha256",
        "query_plan_sha256": "query_plan_sha256",
        "query_event_history_sha256": "query_event_history_sha256",
        "label_ledger_sha256": "label_ledger_sha256",
        "base_score_sha256": "base_score_contract_sha256",
        "authority_score_bundle_sha256": "authority_score_bundle_sha256",
    }
    for dataset in DATASETS:
        lofo[dataset] = {}
        for target in EXPECTED_TARGETS[dataset]:
            source = requests[dataset][target]
            lofo[dataset][target] = {output: str(source[input_name]) for output, input_name in fields.items()}
        source = requests[dataset]["ordinary"]
        ordinary[dataset] = {output: str(source[input_name]) for output, input_name in fields.items()}
    return lofo, ordinary


def prepare_materialized_screen_matrix(
    *,
    requests_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selected_profile_registry: Mapping[str, Any],
    code_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Write one immutable 48-unit matrix from already materialized real folds."""

    if not _valid_hash(code_sha256):
        raise ValueError("formal screen code SHA-256 is invalid")
    registry = _registry(selected_profile_registry)
    requests = _scope(requests_by_dataset)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_path_by_key: dict[tuple[str, str | None], Path] = {}
    for dataset in DATASETS:
        for key, source in requests[dataset].items():
            target = None if key == "ordinary" else key
            token = "ordinary" if target is None else _hash(target)[:12]
            path = root / "fold_inputs" / f"{dataset}__{token}.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != source:
                raise ValueError("formal screen root contains a conflicting fold input")
            _write(path, source)
            authority_path = (root / str(source["authority_score_bundle_path"])).resolve()
            if not authority_path.is_relative_to(root):
                raise ValueError("authority score bundle escapes formal root")
            bundle = source["authority_score_bundle"]
            if authority_path.exists() and json.loads(
                authority_path.read_text(encoding="utf-8")
            ) != bundle:
                raise ValueError("formal screen root contains a conflicting authority bundle")
            _write(authority_path, bundle)
            request_path_by_key[(dataset, target)] = path
    lofo, ordinary = _manifest_inputs(requests)
    manifest = build_initial_screen_manifest(
        targets_by_dataset=EXPECTED_TARGETS,
        selected_profile_registry=registry,
        lofo_inputs=lofo,
        ordinary_inputs=ordinary,
        code_sha256=str(code_sha256),
        output_root=str(root),
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("formal screen root contains a conflicting manifest")
    _write(manifest_path, manifest)
    for raw_unit in manifest["units"]:
        unit = dict(raw_unit)
        request_path = request_path_by_key[(unit["dataset_id"], unit["held_out_fault_type"])]
        request_source = requests[unit["dataset_id"]][
            "ordinary" if unit["held_out_fault_type"] is None else unit["held_out_fault_type"]
        ]
        input_identity = {
            "schema_version": "conservative-lofo-real-unit-execution-input-v1",
            "unit_id": unit["unit_id"],
            "unit_fingerprint_sha256": unit["unit_fingerprint_sha256"],
            "training_request_path": str(request_path),
            "training_request_sha256": request_source["request_sha256"],
            "authority_score_bundle_sha256": request_source[
                "authority_score_bundle_sha256"
            ],
            "selected_profile_registry_sha256": registry["selected_profile_registry_sha256"],
            "arm": unit["arm"],
            "profile": unit.get("profile"),
        }
        execution_input = {**input_identity, "input_sha256": _hash(input_identity)}
        _write(root / "unit_inputs" / f"{unit['unit_id']}.json", execution_input)
    progress_identity = {
        "schema_version": "conservative-lofo-initial-screen-progress-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "phase": "prepared",
        "current_unit": None,
        "completed_unit_count": 0,
        "pending_unit_count": 48,
        "total_unit_count": 48,
        "failure_count": 0,
        "tqdm_progress": "0/48",
    }
    _write(root / "progress.json", {**progress_identity, "progress_sha256": _hash(progress_identity)})
    (root / "heartbeat").write_text("prepared\n", encoding="utf-8")
    validate_initial_screen_manifest(manifest)
    summary_identity = {
        "schema_version": "conservative-lofo-formal-screen-preparation-v1",
        "status": "prepared",
        "output_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_profile_registry_sha256": registry["selected_profile_registry_sha256"],
        "fold_input_count": 12,
        "authority_score_bundle_count": 12,
        "lofo_unit_count": 40,
        "ordinary_unit_count": 8,
        "unit_input_count": 48,
        "full_lofo_authorized": False,
        "automatic_follow_on": False,
    }
    summary = {**summary_identity, "preparation_sha256": _hash(summary_identity)}
    _write(root / "preparation.json", summary)
    return summary


__all__ = ["prepare_materialized_screen_matrix"]
