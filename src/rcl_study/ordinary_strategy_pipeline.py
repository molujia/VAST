"""Manifest, orchestration, resume, and real bridge for ordinary strategies."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans

from .ordinary_query_engine import (
    BUDGET,
    ROUND_SIZES,
    STRATEGY_IDS,
    MappingAnnotationOracle,
    build_strategy_registry,
    run_ordinary_query_engine,
)
from .ordinary_strategy_scoring import (
    score_ranking_rows,
    validate_count_authoritative_view,
)


MANIFEST_SCHEMA_VERSION = "ordinary-strategy-manifest-v1"
UNIT_RESULT_SCHEMA_VERSION = "ordinary-strategy-unit-result-v1"
CANDIDATE_CONTEXT_SCHEMA_VERSION = "ordinary-strategy-candidate-context-v1"
PIPELINE_COMPLETION_SCHEMA_VERSION = "ordinary-strategy-completion-v1"
FORMAL_DATASETS = ("aiops2022_pre", "rcabench")
FUSION_IDS = (
    "masked_early",
    "coverage_normalized_late_affinity",
    "shared_state_alignment",
)
ACTIVE_LEARNING_SEEDS = (42, 43, 44)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path | str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("invalid JSON artifact: %s" % path) from error
    if not isinstance(value, Mapping):
        raise ValueError("JSON artifact must contain a mapping")
    return dict(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(".%s.%d.tmp" % (target.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _require_sha(value: Any, field: str) -> str:
    text = str(value)
    if _SHA_RE.fullmatch(text) is None:
        raise ValueError("%s must be a lowercase SHA-256" % field)
    return text


def _safe_id(value: Any) -> str:
    text = str(value).strip()
    if not text or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError("unit/run identity is not path-safe")
    return text


def _path_under(path: Path | str, root: Path | str) -> bool:
    candidate = Path(path).resolve()
    authority = Path(root).resolve()
    try:
        candidate.relative_to(authority)
    except ValueError:
        return False
    return True


def _unit_paths(output_root: Path, unit_id: str) -> Dict[str, str]:
    namespace = output_root / "units" / unit_id
    return {
        "namespace_root": str(namespace),
        "transform_root": str(namespace / "transform"),
        "selector_root": str(namespace / "selector"),
        "temp_root": str(namespace / "tmp"),
        "checkpoint_root": str(namespace / "checkpoints"),
        "score_root": str(namespace / "scores"),
        "marker_root": str(namespace / "markers"),
        "output_root": str(namespace / "backend"),
    }


def _build_unit(
    *,
    phase: str,
    output_root: Path,
    dataset_id: str,
    strategy_id: str,
    active_learning_seed: int,
    fusion: Mapping[str, Any] | None,
    dataset_input: Mapping[str, Any] | None,
    code_sha256: str,
) -> Dict[str, Any]:
    fusion_id = str((fusion or {}).get("fusion_id", "smoke_fusion"))
    unit_id = "%s.%s.%s.seed%d.%s" % (
        phase,
        dataset_id,
        strategy_id,
        int(active_learning_seed),
        fusion_id,
    )
    unit = {
        "unit_id": _safe_id(unit_id),
        "phase": phase,
        "protocol": "ordinary",
        "study_mode": "query_only",
        "dataset_id": dataset_id,
        "canonical_dataset_id": dataset_id,
        "strategy_id": strategy_id,
        "strategy_config": deepcopy(
            build_strategy_registry()["strategy_configs"].get(strategy_id, {})
        ),
        "active_learning_seed": int(active_learning_seed),
        "training_seed": 42,
        "split_seed": 42,
        "budget": BUDGET,
        "round_sizes": list(ROUND_SIZES),
        "fusion_id": fusion_id,
        "code_sha256": code_sha256,
    }
    if fusion is not None:
        unit.update(
            {
                "fusion_artifact_path": str(fusion["path"]),
                "fusion_artifact_sha256": _require_sha(
                    fusion["sha256"], "fusion_artifact_sha256"
                ),
            }
        )
    if dataset_input is not None:
        unit.update(
            {
                "inventory_path": str(dataset_input["inventory_path"]),
                "inventory_sha256": _require_sha(
                    dataset_input["inventory_sha256"], "inventory_sha256"
                ),
                "candidate_context_path": str(dataset_input["candidate_context_path"]),
                "candidate_context_sha256": _require_sha(
                    dataset_input["candidate_context_sha256"],
                    "candidate_context_sha256",
                ),
            }
        )
    unit.update(_unit_paths(output_root, unit["unit_id"]))
    unit["unit_fingerprint"] = unit_fingerprint(unit)
    return unit


def unit_fingerprint(unit: Mapping[str, Any]) -> str:
    keys = (
        "unit_id",
        "phase",
        "protocol",
        "study_mode",
        "dataset_id",
        "strategy_id",
        "strategy_config",
        "active_learning_seed",
        "training_seed",
        "split_seed",
        "budget",
        "round_sizes",
        "fusion_id",
        "fusion_artifact_sha256",
        "inventory_sha256",
        "candidate_context_sha256",
        "code_sha256",
    )
    return semantic_sha256({key: unit.get(key) for key in keys})


def seal_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(manifest))
    result.pop("manifest_sha256", None)
    for unit in result.get("units", ()):  # keep fingerprints synchronized after bounded edits
        unit["unit_fingerprint"] = unit_fingerprint(unit)
    result["manifest_sha256"] = semantic_sha256(result)
    return result


def _base_manifest(
    *,
    run_id: str,
    phase: str,
    output_root: Path | str,
    code_sha256: str,
    units: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    root = Path(output_root).resolve()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": _safe_id(run_id),
        "phase": phase,
        "protocol": "ordinary",
        "study_mode": "query_only",
        "output_root": str(root),
        "split_seed": 42,
        "budget": BUDGET,
        "round_sizes": list(ROUND_SIZES),
        "active_learning_seeds": list(ACTIVE_LEARNING_SEEDS),
        "max_gpus": 24,
        "code_sha256": _require_sha(code_sha256, "code_sha256"),
        "units": [deepcopy(dict(unit)) for unit in units],
    }
    return seal_manifest(manifest)


def _validated_dataset_inputs(dataset_inputs: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(dataset_inputs, Mapping) or set(dataset_inputs) != set(FORMAL_DATASETS):
        raise ValueError("dataset inputs must contain rcabench and aiops2022_pre")
    return {str(key): value for key, value in dataset_inputs.items()}


def build_fusion_screen_manifest(
    *,
    run_id: str,
    output_root: Path | str,
    dataset_inputs: Any,
    code_sha256: str,
) -> Dict[str, Any]:
    inputs = _validated_dataset_inputs(dataset_inputs)
    root = Path(output_root).resolve()
    units = []
    for dataset_id in FORMAL_DATASETS:
        fusions = inputs[dataset_id].get("fusion_artifacts")
        if not isinstance(fusions, Mapping) or set(fusions) != set(FUSION_IDS):
            raise ValueError("fusion screen requires exactly three fusion artifacts")
        for fusion_id in FUSION_IDS:
            fusion = {"fusion_id": fusion_id, **dict(fusions[fusion_id])}
            for seed in ACTIVE_LEARNING_SEEDS:
                units.append(
                    _build_unit(
                        phase="fusion_screen",
                        output_root=root,
                        dataset_id=dataset_id,
                        strategy_id="kmeans_coverage",
                        active_learning_seed=seed,
                        fusion=fusion,
                        dataset_input=inputs[dataset_id],
                        code_sha256=code_sha256,
                    )
                )
    return _base_manifest(
        run_id=run_id,
        phase="fusion_screen",
        output_root=root,
        code_sha256=code_sha256,
        units=units,
    )


def build_strategy_screen_manifest(
    *,
    run_id: str,
    output_root: Path | str,
    dataset_inputs: Any,
    frozen_fusion_by_dataset: Any,
    code_sha256: str,
) -> Dict[str, Any]:
    inputs = _validated_dataset_inputs(dataset_inputs)
    if not isinstance(frozen_fusion_by_dataset, Mapping) or set(frozen_fusion_by_dataset) != set(FORMAL_DATASETS):
        raise ValueError("strategy screen requires one frozen fusion per dataset")
    root = Path(output_root).resolve()
    units = []
    for dataset_id in FORMAL_DATASETS:
        fusion = dict(frozen_fusion_by_dataset[dataset_id])
        if fusion.get("fusion_id") not in FUSION_IDS:
            raise ValueError("frozen fusion ID is invalid")
        for strategy_id in STRATEGY_IDS:
            for seed in ACTIVE_LEARNING_SEEDS:
                units.append(
                    _build_unit(
                        phase="strategy_screen",
                        output_root=root,
                        dataset_id=dataset_id,
                        strategy_id=strategy_id,
                        active_learning_seed=seed,
                        fusion=fusion,
                        dataset_input=inputs[dataset_id],
                        code_sha256=code_sha256,
                    )
                )
    return _base_manifest(
        run_id=run_id,
        phase="strategy_screen",
        output_root=root,
        code_sha256=code_sha256,
        units=units,
    )


def build_smoke_manifest(
    *,
    run_id: str,
    output_root: Path | str,
    unit_specs: Any,
    code_sha256: str,
) -> Dict[str, Any]:
    if isinstance(unit_specs, (str, bytes)) or not isinstance(unit_specs, Sequence) or not unit_specs:
        raise ValueError("smoke manifest requires unit specs")
    root = Path(output_root).resolve()
    units = []
    for raw in unit_specs:
        unit = _build_unit(
            phase="smoke",
            output_root=root,
            dataset_id=str(raw["dataset_id"]),
            strategy_id=str(raw["strategy_id"]),
            active_learning_seed=int(raw["active_learning_seed"]),
            fusion=None,
            dataset_input=None,
            code_sha256=code_sha256,
        )
        units.append(unit)
    return _base_manifest(
        run_id=run_id,
        phase="smoke",
        output_root=root,
        code_sha256=code_sha256,
        units=units,
    )


def validate_ordinary_strategy_manifest(manifest: Any) -> Dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("ordinary strategy manifest must be a mapping")
    result = deepcopy(dict(manifest))
    if result.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("ordinary strategy manifest schema drifted")
    phase = str(result.get("phase", ""))
    if phase not in {"fusion_screen", "strategy_screen", "final", "smoke"}:
        raise ValueError("ordinary strategy manifest phase is invalid")
    if result.get("protocol") != "ordinary" or result.get("study_mode") != "query_only":
        raise ValueError("ordinary query-only protocol is required")
    if result.get("budget") != BUDGET or result.get("round_sizes") != list(ROUND_SIZES):
        raise ValueError("ordinary strategy manifest budget/schedule drifted")
    _require_sha(result.get("code_sha256"), "code_sha256")
    units = result.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("ordinary strategy manifest contains no units")
    expected_count = {"fusion_screen": 18, "strategy_screen": 42}.get(phase)
    if expected_count is not None and len(units) != expected_count:
        raise ValueError("formal manifest unit count drifted")
    root = Path(str(result.get("output_root", ""))).resolve()
    unit_ids = []
    namespace_roots = []
    checkpoint_roots = []
    for unit in units:
        if not isinstance(unit, Mapping):
            raise ValueError("manifest unit must be a mapping")
        unit_id = _safe_id(unit.get("unit_id"))
        unit_ids.append(unit_id)
        dataset_id = str(unit.get("dataset_id", ""))
        if dataset_id not in FORMAL_DATASETS:
            raise ValueError("formal dataset must be rcabench or aiops2022_pre")
        if unit.get("strategy_id") not in STRATEGY_IDS:
            raise ValueError("formal strategy is not registered")
        if unit.get("budget") != BUDGET or unit.get("round_sizes") != list(ROUND_SIZES):
            raise ValueError("unit budget/schedule drifted")
        if unit.get("protocol") != "ordinary" or unit.get("study_mode") != "query_only":
            raise ValueError("unit protocol drifted")
        namespace = str(unit.get("namespace_root", ""))
        checkpoint = str(unit.get("checkpoint_root", ""))
        if not _path_under(namespace, root) or not _path_under(checkpoint, namespace):
            raise ValueError("unit paths must remain isolated under output root")
        namespace_roots.append(str(Path(namespace).resolve()))
        checkpoint_roots.append(str(Path(checkpoint).resolve()))
        if phase != "smoke":
            for field in (
                "fusion_artifact_sha256",
                "inventory_sha256",
                "candidate_context_sha256",
            ):
                _require_sha(unit.get(field), field)
        if unit.get("unit_fingerprint") != unit_fingerprint(unit):
            raise ValueError("unit fingerprint drifted")
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("unit IDs must be unique")
    if len(namespace_roots) != len(set(namespace_roots)) or len(checkpoint_roots) != len(set(checkpoint_roots)):
        raise ValueError("unit paths must be isolated")
    declared = result.pop("manifest_sha256", None)
    if declared != semantic_sha256(result):
        raise ValueError("manifest SHA-256 drifted")
    result["manifest_sha256"] = declared
    return result


def select_usable_gpus(
    gpu_rows: Any,
    *,
    max_gpus: int = 24,
    minimum_free_mb: int = 8000,
    maximum_utilization: float = 50.0,
) -> list[Dict[str, Any]]:
    if isinstance(gpu_rows, (str, bytes)) or not isinstance(gpu_rows, Sequence):
        raise ValueError("GPU rows must be a sequence")
    if not 1 <= int(max_gpus) <= 24:
        raise ValueError("max_gpus must be in [1, 24]")
    selected = []
    for source in gpu_rows:
        if not isinstance(source, Mapping):
            raise ValueError("GPU row must be a mapping")
        row = {
            "gpu_id": int(source["gpu_id"]),
            "memory_free_mb": int(source["memory_free_mb"]),
            "utilization_percent": float(source["utilization_percent"]),
        }
        if row["memory_free_mb"] >= int(minimum_free_mb) and row["utilization_percent"] <= float(maximum_utilization):
            selected.append(row)
    selected.sort(key=lambda row: (row["utilization_percent"], -row["memory_free_mb"], row["gpu_id"]))
    return selected[: int(max_gpus)]


def _validated_artifact_sha(artifact: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(artifact))
    declared = str(payload.pop("artifact_sha256", ""))
    if declared != semantic_sha256(payload):
        raise ValueError("artifact SHA-256 drifted")
    return declared


def build_candidate_context_artifact(
    dataset_id: Any,
    *,
    inventory_records: Any,
    window_rows: Any,
) -> Dict[str, Any]:
    dataset = str(dataset_id)
    if dataset not in FORMAL_DATASETS:
        raise ValueError("candidate context dataset is invalid")
    if isinstance(inventory_records, (str, bytes)) or not isinstance(inventory_records, Sequence):
        raise ValueError("inventory records must be a sequence")
    if isinstance(window_rows, (str, bytes)) or not isinstance(window_rows, Sequence):
        raise ValueError("window rows must be a sequence")
    windows = {}
    for source in window_rows:
        if not isinstance(source, Mapping) or set(source) != {"case_id", "start_ts", "targets"}:
            raise ValueError("window context fields drifted")
        case_id = str(source["case_id"])
        targets = source["targets"]
        start = float(source["start_ts"])
        if case_id in windows or not case_id or not math.isfinite(start):
            raise ValueError("window context identity/time is invalid")
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence) or not targets:
            raise ValueError("window context targets are invalid")
        windows[case_id] = {"start_ts": start, "targets": [str(value) for value in targets]}
    observable = []
    annotations = {}
    for record in inventory_records:
        if not isinstance(record, Mapping):
            raise ValueError("inventory record is invalid")
        if str(record.get("split")) != "outer_train":
            continue
        case_id = str(record.get("case_id", ""))
        if case_id not in windows or case_id in annotations:
            raise ValueError("candidate context coverage drifted")
        incident_id = str(record.get("incident_id", ""))
        fault_type = str(record.get("fault_type", ""))
        if not incident_id or not fault_type:
            raise ValueError("candidate context inventory labels are incomplete")
        observable.append(
            {
                "case_id": case_id,
                "timestamp": windows[case_id]["start_ts"],
                "incident_id": incident_id,
            }
        )
        annotations[case_id] = {
            "root_cause": _canonical_json(windows[case_id]["targets"]),
            "fault_type": fault_type,
        }
    observable.sort(key=lambda row: row["case_id"])
    if len(observable) < BUDGET:
        raise ValueError("candidate context cannot fill budget 30")
    payload = {
        "schema_version": CANDIDATE_CONTEXT_SCHEMA_VERSION,
        "dataset_id": dataset,
        "split_seed": 42,
        "observable_candidates": observable,
        "private_annotation_oracle": annotations,
        "label_firewall": "private_annotation_oracle_never_enters_selector_candidates",
    }
    payload["artifact_sha256"] = semantic_sha256(payload)
    return payload


def _label_free_uncertainty(matrix: np.ndarray) -> np.ndarray:
    distinct = np.unique(matrix, axis=0).shape[0]
    if distinct <= 1:
        return np.zeros(matrix.shape[0], dtype=float)
    cluster_count = min(14, distinct, matrix.shape[0])
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(matrix)
    centers = np.vstack([matrix[labels == label].mean(axis=0) for label in range(cluster_count)])
    distances = np.linalg.norm(matrix - centers[labels], axis=1)
    maximum = float(distances.max())
    return np.zeros_like(distances) if maximum <= 0.0 else distances / maximum


def merge_fusion_and_candidate_context(
    fusion_artifact: Any,
    candidate_context: Any,
) -> Dict[str, Any]:
    if not isinstance(fusion_artifact, Mapping) or not isinstance(candidate_context, Mapping):
        raise ValueError("fusion/context artifacts must be mappings")
    _validated_artifact_sha(fusion_artifact)
    _validated_artifact_sha(candidate_context)
    dataset = str(candidate_context.get("dataset_id", ""))
    fusion_dataset = str(
        fusion_artifact.get(
            "canonical_dataset_id",
            fusion_artifact.get("dataset_id", ""),
        )
    )
    if fusion_dataset != dataset:
        raise ValueError("fusion/context dataset drifted")
    context_by_case = {
        str(row["case_id"]): row
        for row in candidate_context.get("observable_candidates", ())
    }
    embedding_by_case = {
        str(row["case_id"]): row
        for row in fusion_artifact.get("case_embeddings", ())
    }
    if set(context_by_case) != set(embedding_by_case):
        raise ValueError("fusion/context case coverage drifted")
    case_ids = sorted(context_by_case)
    matrix = np.asarray([embedding_by_case[case_id]["embedding"] for case_id in case_ids], dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("fusion embeddings must be a finite matrix")
    uncertainties = _label_free_uncertainty(matrix)
    candidates = []
    for index, case_id in enumerate(case_ids):
        context = context_by_case[case_id]
        candidates.append(
            {
                "case_id": case_id,
                "embedding": [float(value) for value in matrix[index]],
                "timestamp": float(context["timestamp"]),
                "incident_id": str(context["incident_id"]),
                "uncertainty": float(uncertainties[index]),
            }
        )
    annotations = deepcopy(dict(candidate_context.get("private_annotation_oracle") or {}))
    if set(annotations) != set(case_ids):
        raise ValueError("private annotation oracle coverage drifted")
    return {
        "dataset_id": dataset,
        "fusion_id": str(fusion_artifact.get("fusion_id", "")),
        "candidates": candidates,
        "private_annotations": annotations,
        "fusion_artifact_sha256": str(fusion_artifact["artifact_sha256"]),
        "candidate_context_sha256": str(candidate_context["artifact_sha256"]),
    }


def _validate_unit_result(unit: Mapping[str, Any], result: Any) -> Dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("unit result must be a mapping")
    value = deepcopy(dict(result))
    required = {
        "schema_version",
        "status",
        "unit_id",
        "dataset_id",
        "strategy_id",
        "active_learning_seed",
        "selected_case_ids",
        "views",
        "query_plan_sha256",
    }
    if not required.issubset(value):
        raise ValueError("unit result is incomplete")
    if value["schema_version"] != UNIT_RESULT_SCHEMA_VERSION or value["status"] != "success":
        raise ValueError("unit result did not report success")
    for field in ("unit_id", "dataset_id", "strategy_id", "active_learning_seed"):
        if value[field] != unit[field]:
            raise ValueError("unit result identity drifted")
    selected = [str(case_id) for case_id in value["selected_case_ids"]]
    if len(selected) != BUDGET or len(selected) != len(set(selected)):
        raise ValueError("unit result selected budget is invalid")
    if not isinstance(value["views"], Mapping) or set(value["views"]) != {"T1", "T2"}:
        raise ValueError("unit result views are incomplete")
    value["views"] = {
        view: validate_count_authoritative_view(value["views"][view])
        for view in ("T1", "T2")
    }
    _require_sha(value["query_plan_sha256"], "query_plan_sha256")
    return value


def _existing_unit_result(unit: Mapping[str, Any], manifest_sha256: str) -> Dict[str, Any] | None:
    path = Path(str(unit["namespace_root"])) / "unit-result.json"
    if not path.is_file():
        return None
    wrapper = _read_json(path)
    if wrapper.get("unit_fingerprint") != unit_fingerprint(unit) or wrapper.get("manifest_sha256") != manifest_sha256:
        raise ValueError("stale unit result cannot be resumed")
    return _validate_unit_result(unit, wrapper.get("result"))


def _progress_payload(
    *,
    phase: str,
    total: int,
    completed: int,
    failures: int,
    recent_unit: str,
    selected_gpu_ids: Sequence[int],
    manifest_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "ordinary-strategy-progress-v1",
        "phase": phase,
        "total": int(total),
        "completed": int(completed),
        "failures": int(failures),
        "recent_unit": recent_unit,
        "selected_gpu_ids": [int(value) for value in selected_gpu_ids],
        "manifest_sha256": manifest_sha256,
    }


def format_tqdm_progress_line(progress: Mapping[str, Any]) -> str:
    total = max(1, int(progress["total"]))
    completed = int(progress["completed"])
    width = 20
    filled = min(width, int(width * completed / float(total)))
    return "[%s%s] %d/%d %5.1f%% phase=%s unit=%s failures=%d" % (
        "=" * filled,
        "." * (width - filled),
        completed,
        total,
        100.0 * completed / float(total),
        progress["phase"],
        progress["recent_unit"],
        int(progress["failures"]),
    )


def _write_progress(root: Path, progress: Mapping[str, Any]) -> None:
    _write_json_atomic(root / "progress.json", progress)
    print(format_tqdm_progress_line(progress), flush=True)


def execute_ordinary_strategy_manifest(
    manifest: Any,
    unit_executor: Callable[[Mapping[str, Any], Path], Mapping[str, Any]],
    *,
    gpu_rows: Sequence[Mapping[str, Any]],
    max_workers: int | None = None,
) -> Dict[str, Any]:
    frozen = validate_ordinary_strategy_manifest(manifest)
    root = Path(frozen["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    completed_marker = root / "COMPLETED.json"
    all_done_marker = root / "all.done"
    if completed_marker.exists() != all_done_marker.exists():
        raise ValueError("stale top-level completion marker detected")
    if completed_marker.is_file():
        completion = _read_json(completed_marker)
        if completion.get("manifest_sha256") != frozen["manifest_sha256"]:
            raise ValueError("stale completion belongs to another manifest")
    if (root / ".failed").exists() or (root / "pipeline.failed.json").exists():
        raise ValueError("stale failure marker requires a new output path")
    selected_gpus = select_usable_gpus(gpu_rows, max_gpus=int(frozen["max_gpus"]))
    selected_gpu_ids = [row["gpu_id"] for row in selected_gpus]
    results = {}
    pending = []
    for unit in frozen["units"]:
        existing = _existing_unit_result(unit, frozen["manifest_sha256"])
        if existing is None:
            pending.append(unit)
        else:
            results[unit["unit_id"]] = existing
    if completed_marker.is_file() and pending:
        raise ValueError("stale completion has missing unit results")
    if not pending:
        return _read_json(completed_marker)

    worker_limit = int(max_workers or len(pending))
    gpu_limit = len(selected_gpu_ids) if selected_gpu_ids else 1
    worker_count = max(1, min(worker_limit, len(pending), gpu_limit))
    progress_lock = threading.Lock()
    failures: Dict[str, str] = {}
    progress = _progress_payload(
        phase="running",
        total=len(frozen["units"]),
        completed=len(results),
        failures=0,
        recent_unit="",
        selected_gpu_ids=selected_gpu_ids,
        manifest_sha256=frozen["manifest_sha256"],
    )
    _write_progress(root, progress)

    def invoke(index: int, source_unit: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
        unit = deepcopy(dict(source_unit))
        unit["assigned_gpu_id"] = None if not selected_gpu_ids else selected_gpu_ids[index % len(selected_gpu_ids)]
        for path_field in (
            "namespace_root",
            "transform_root",
            "selector_root",
            "temp_root",
            "checkpoint_root",
            "score_root",
            "marker_root",
            "output_root",
        ):
            Path(str(unit[path_field])).mkdir(parents=True, exist_ok=True)
        raw = unit_executor(unit, root)
        validated = _validate_unit_result(source_unit, raw)
        wrapper = {
            "schema_version": "ordinary-strategy-unit-wrapper-v1",
            "unit_id": source_unit["unit_id"],
            "unit_fingerprint": unit_fingerprint(source_unit),
            "manifest_sha256": frozen["manifest_sha256"],
            "assigned_gpu_id": unit["assigned_gpu_id"],
            "result": validated,
        }
        _write_json_atomic(Path(str(source_unit["namespace_root"])) / "unit-result.json", wrapper)
        _write_json_atomic(Path(str(source_unit["marker_root"])) / "success.json", {"status": "success", "unit_id": source_unit["unit_id"]})
        return str(source_unit["unit_id"]), validated

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_by_unit = {
            pool.submit(invoke, index, unit): unit for index, unit in enumerate(pending)
        }
        for future in as_completed(future_by_unit):
            unit = future_by_unit[future]
            try:
                unit_id, result = future.result()
                results[unit_id] = result
            except Exception as error:  # fail closed after independent workers settle
                unit_id = str(unit["unit_id"])
                failures[unit_id] = "%s: %s" % (type(error).__name__, error)
                marker_root = Path(str(unit["marker_root"]))
                marker_root.mkdir(parents=True, exist_ok=True)
                _write_json_atomic(marker_root / "failed.json", {"status": "failed", "unit_id": unit_id, "error": failures[unit_id]})
            with progress_lock:
                progress = _progress_payload(
                    phase="running" if not failures else "failed",
                    total=len(frozen["units"]),
                    completed=len(results),
                    failures=len(failures),
                    recent_unit=str(unit["unit_id"]),
                    selected_gpu_ids=selected_gpu_ids,
                    manifest_sha256=frozen["manifest_sha256"],
                )
                _write_progress(root, progress)
    if failures:
        failure_payload = {
            "schema_version": "ordinary-strategy-failure-v1",
            "status": "failed",
            "manifest_sha256": frozen["manifest_sha256"],
            "completed": len(results),
            "failures": failures,
        }
        _write_json_atomic(root / "pipeline.failed.json", failure_payload)
        (root / ".failed").write_text(semantic_sha256(failure_payload) + "\n", encoding="utf-8")
        if completed_marker.exists():
            completed_marker.unlink()
        if all_done_marker.exists():
            all_done_marker.unlink()
        raise RuntimeError("ordinary strategy pipeline failed for %d unit(s)" % len(failures))
    completion = {
        "schema_version": PIPELINE_COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "run_id": frozen["run_id"],
        "phase": frozen["phase"],
        "manifest_sha256": frozen["manifest_sha256"],
        "total": len(frozen["units"]),
        "completed": len(results),
        "failures": 0,
        "selected_gpu_ids": selected_gpu_ids,
        "unit_result_sha256": {
            unit_id: semantic_sha256(result) for unit_id, result in sorted(results.items())
        },
    }
    completion["completion_sha256"] = semantic_sha256(completion)
    _write_json_atomic(completed_marker, completion)
    all_done_marker.write_text(completion["completion_sha256"] + "\n", encoding="utf-8")
    _write_progress(
        root,
        _progress_payload(
            phase="complete",
            total=len(frozen["units"]),
            completed=len(results),
            failures=0,
            recent_unit="",
            selected_gpu_ids=selected_gpu_ids,
            manifest_sha256=frozen["manifest_sha256"],
        ),
    )
    return completion


def build_status_snapshot(run_root: Path | str) -> Dict[str, Any]:
    root = Path(run_root)
    progress = _read_json(root / "progress.json") if (root / "progress.json").is_file() else {}
    if (root / "COMPLETED.json").is_file() and (root / "all.done").is_file():
        completion = "complete"
    elif (root / ".failed").is_file() or (root / "pipeline.failed.json").is_file():
        completion = "failed"
    else:
        completion = "running" if progress else "not_started"
    return {
        "completion": completion,
        "progress": progress,
        "paths": {
            "progress": str(root / "progress.json"),
            "completed": str(root / "COMPLETED.json"),
            "all_done": str(root / "all.done"),
            "failed": str(root / "pipeline.failed.json"),
        },
    }


def status_command(run_root: Path | str) -> str:
    root = str(Path(run_root))
    return "python -m json.tool %s/progress.json; test -f %s/pipeline.failed.json && python -m json.tool %s/pipeline.failed.json; test -f %s/all.done && echo ALL_DONE" % (
        root,
        root,
        root,
        root,
    )


def _query_diagnostics(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    first_round = {}
    queried_types = []
    proxy_mode_by_case = {}
    proxy_mode_sources = Counter()
    proxy_fields = (
        "proxy_mode_id",
        "predicted_fault_type",
        "component_id",
        "timestamp_chunk",
    )
    for event in events:
        if event.get("event_type") != "selection_commit":
            continue
        score_by_case = {
            str(row["case_id"]): row
            for row in event.get("candidate_scores", ())
        }
        for case_id in event.get("case_ids", ()):
            case_id = str(case_id)
            if case_id in proxy_mode_by_case:
                raise ValueError("query diagnostics contain duplicate selection")
            score = score_by_case.get(case_id)
            if score is None:
                raise ValueError("selected query lacks candidate score evidence")
            source = "unpartitioned"
            mode = "__unpartitioned__"
            for field in proxy_fields:
                value = score.get(field)
                if value is not None and str(value):
                    source = field
                    mode = "%s:%s" % (field, value)
                    break
            proxy_mode_by_case[case_id] = mode
            proxy_mode_sources[source] += 1
    contingency = defaultdict(Counter)
    for event in events:
        if event.get("event_type") != "annotation_reveal":
            continue
        round_index = int(event["round_index"])
        for reveal in event["reveals"]:
            case_id = str(reveal["case_id"])
            if case_id not in proxy_mode_by_case:
                raise ValueError("query reveal lacks committed proxy evidence")
            fault_type = str(reveal["annotation"]["fault_type"])
            queried_types.append(fault_type)
            first_round.setdefault(fault_type, round_index)
            contingency[proxy_mode_by_case[case_id]][fault_type] += 1
    pair_count = sum(sum(counts.values()) for counts in contingency.values())
    if pair_count != len(proxy_mode_by_case):
        raise ValueError("query proxy/type evidence is incomplete")
    purity_hits = sum(max(counts.values()) for counts in contingency.values())
    return {
        "queried_fault_types": queried_types,
        "first_round_by_fault_type": first_round,
        "proxy_type_agreement": (
            0.0 if pair_count == 0 else purity_hits / float(pair_count)
        ),
        "proxy_type_agreement_definition": "selected_query_cluster_purity",
        "proxy_type_pair_count": pair_count,
        "proxy_mode_source_counts": dict(sorted(proxy_mode_sources.items())),
    }


def _count_views_from_backend(result: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(result.get("views"), Mapping) and set(result["views"]) == {"T1", "T2"}:
        return {
            view: validate_count_authoritative_view(result["views"][view])
            for view in ("T1", "T2")
        }
    evidence = result.get("per_case_ranking_evidence_by_view")
    partitions = result.get("partitions")
    if not isinstance(evidence, Mapping) or not isinstance(partitions, Mapping):
        raise ValueError("backend result lacks count-authoritative ranking evidence")
    views = {}
    for view in ("T1", "T2"):
        rows = []
        for source in evidence.get(view, ()):
            rows.append(
                {
                    "case_id": str(source["case_id"]),
                    "targets": list(source["targets"]),
                    "ranking": list(source["ranking"]),
                }
            )
        views[view] = score_ranking_rows(
            rows,
            expected_case_ids=[str(value) for value in partitions[view]],
        )
    return views


def run_ordinary_strategy_unit(
    unit: Mapping[str, Any],
    run_root: Path,
    *,
    backend_executor: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    fusion = _read_json(unit["fusion_artifact_path"])
    context = _read_json(unit["candidate_context_path"])
    if str(fusion.get("artifact_sha256")) != str(unit["fusion_artifact_sha256"]):
        raise ValueError("unit fusion artifact SHA drifted")
    if str(context.get("artifact_sha256")) != str(unit["candidate_context_sha256"]):
        raise ValueError("unit candidate context SHA drifted")
    merged = merge_fusion_and_candidate_context(fusion, context)
    oracle = MappingAnnotationOracle(merged["private_annotations"])
    selection = run_ordinary_query_engine(
        merged["candidates"],
        annotation_oracle=oracle,
        strategy_id=unit["strategy_id"],
        active_learning_seed=int(unit["active_learning_seed"]),
        config=dict(unit.get("strategy_config") or {}),
    )
    namespace = Path(str(unit["namespace_root"]))
    namespace.mkdir(parents=True, exist_ok=True)
    event_path = namespace / "query-events.json"
    _write_json_atomic(event_path, {"events": selection["events"], "query_plan_sha256": selection["query_plan_sha256"]})
    fixed_unit = deepcopy(dict(unit))
    fixed_unit.update(
        {
            "selector_id": "fixed_budget_set",
            "selector_config": {},
            "selector_config_sha256": semantic_sha256({}),
            "selected_case_ids": list(selection["selected_case_ids"]),
            "method_id": "ordinary_multimodal_active_learning",
            "output_root": str(namespace / "backend"),
        }
    )
    if backend_executor is None:
        from .query_active_real_execution import run_deepening_query_only_t1_t2_unit

        backend_executor = run_deepening_query_only_t1_t2_unit
    backend = backend_executor(fixed_unit, Path(run_root))
    views = _count_views_from_backend(backend)
    diagnostics = _query_diagnostics(selection["events"])
    result = {
        "schema_version": UNIT_RESULT_SCHEMA_VERSION,
        "status": "success",
        "unit_id": unit["unit_id"],
        "dataset_id": unit["dataset_id"],
        "strategy_id": unit["strategy_id"],
        "active_learning_seed": int(unit["active_learning_seed"]),
        "selected_case_ids": list(selection["selected_case_ids"]),
        "views": views,
        "query_plan_sha256": selection["query_plan_sha256"],
        "annotation_cost": int(selection["annotation_cost"]),
        "query_event_log_path": str(event_path.resolve()),
        "fusion_id": str(unit.get("fusion_id", "")),
        "fusion_artifact_sha256": str(unit.get("fusion_artifact_sha256", "")),
        "queried_fault_types": diagnostics["queried_fault_types"],
        "first_round_by_fault_type": diagnostics["first_round_by_fault_type"],
        "proxy_type_agreement": diagnostics["proxy_type_agreement"],
        "proxy_type_agreement_definition": diagnostics[
            "proxy_type_agreement_definition"
        ],
        "proxy_type_pair_count": diagnostics["proxy_type_pair_count"],
        "proxy_mode_source_counts": diagnostics[
            "proxy_mode_source_counts"
        ],
        "backend_result_path": str(Path(fixed_unit["output_root"]) / "metrics.json"),
    }
    return _validate_unit_result(unit, result)


__all__ = [
    "ACTIVE_LEARNING_SEEDS",
    "CANDIDATE_CONTEXT_SCHEMA_VERSION",
    "FORMAL_DATASETS",
    "FUSION_IDS",
    "MANIFEST_SCHEMA_VERSION",
    "PIPELINE_COMPLETION_SCHEMA_VERSION",
    "UNIT_RESULT_SCHEMA_VERSION",
    "build_candidate_context_artifact",
    "build_fusion_screen_manifest",
    "build_smoke_manifest",
    "build_status_snapshot",
    "build_strategy_screen_manifest",
    "execute_ordinary_strategy_manifest",
    "format_tqdm_progress_line",
    "merge_fusion_and_candidate_context",
    "run_ordinary_strategy_unit",
    "seal_manifest",
    "select_usable_gpus",
    "semantic_sha256",
    "status_command",
    "unit_fingerprint",
    "validate_ordinary_strategy_manifest",
]
