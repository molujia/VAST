from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FinalRCLExecutionError(RuntimeError):
    """Raised when resumable final execution cannot be trusted."""


FINAL_STAGE_KINDS = (
    "representation_hdbscan",
    "cvae_neural_pool",
    "augmented_base_ranker",
    "oser_checkpoint",
    "ranking",
    "unit_aggregate",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_CLASSES = ("cpu", "gpu")
_ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)
_DATASETS = ("rcabench", "aiops2022_pre")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, context: str) -> str:
    result = str(value).strip()
    if not _SHA256.fullmatch(result):
        raise FinalRCLExecutionError(f"{context} must be SHA-256")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise FinalRCLExecutionError(f"invalid {context}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise FinalRCLExecutionError(f"{context} must be a JSON object")
    return dict(payload)


def build_stage_spec(
    *,
    stage_kind: str,
    owner_id: str,
    input_identity: Mapping[str, Any],
    dependency_manifest_sha256s: Sequence[Any],
    code_sha256: str,
    config_sha256: str,
    resource_class: str,
) -> dict[str, Any]:
    kind = str(stage_kind)
    owner = str(owner_id).strip()
    resource = str(resource_class)
    if kind not in FINAL_STAGE_KINDS or not owner:
        raise FinalRCLExecutionError("stage kind or owner is invalid")
    if resource not in _RESOURCE_CLASSES:
        raise FinalRCLExecutionError("stage resource class must be cpu or gpu")
    if not isinstance(input_identity, Mapping) or not input_identity:
        raise FinalRCLExecutionError("stage input identity must be a nonempty mapping")
    dependencies = tuple(
        _sha256(value, "stage dependency manifest")
        for value in dependency_manifest_sha256s
    )
    if len(dependencies) != len(set(dependencies)):
        raise FinalRCLExecutionError("stage dependency manifests must be unique")
    identity = {
        "schema_version": "final-rcl-stage-spec-v1",
        "stage_kind": kind,
        "owner_id": owner,
        "resource_class": resource,
        "input_identity": deepcopy(dict(input_identity)),
        "input_identity_sha256": _semantic_hash(input_identity),
        "dependency_manifest_sha256s": dependencies,
        "code_sha256": _sha256(code_sha256, "stage code"),
        "config_sha256": _sha256(config_sha256, "stage config"),
    }
    content_sha = _semantic_hash(identity)
    return {
        **identity,
        "stage_content_sha256": content_sha,
        "stage_id": f"{owner}--{kind}--{content_sha[:12]}",
    }


def _validate_stage_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = deepcopy(dict(value))
    supplied_content_sha = spec.pop("stage_content_sha256", None)
    supplied_stage_id = spec.pop("stage_id", None)
    if spec.get("schema_version") != "final-rcl-stage-spec-v1":
        raise FinalRCLExecutionError("stage spec schema drifted")
    expected_content_sha = _semantic_hash(spec)
    if supplied_content_sha != expected_content_sha:
        raise FinalRCLExecutionError("stage content identity drifted")
    expected_stage_id = (
        f"{spec.get('owner_id')}--{spec.get('stage_kind')}--{expected_content_sha[:12]}"
    )
    if supplied_stage_id != expected_stage_id:
        raise FinalRCLExecutionError("stage ID drifted")
    _sha256(spec.get("input_identity_sha256"), "stage input identity")
    if spec.get("input_identity_sha256") != _semantic_hash(spec.get("input_identity")):
        raise FinalRCLExecutionError("stage input identity hash drifted")
    if spec.get("stage_kind") not in FINAL_STAGE_KINDS:
        raise FinalRCLExecutionError("unknown stage kind")
    if spec.get("resource_class") not in _RESOURCE_CLASSES:
        raise FinalRCLExecutionError("unknown stage resource class")
    for value in spec.get("dependency_manifest_sha256s", ()):
        _sha256(value, "stage dependency manifest")
    return {
        **spec,
        "stage_content_sha256": supplied_content_sha,
        "stage_id": supplied_stage_id,
    }


def validate_completed_stage(
    *, stage_dir: Path, expected_spec: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    root = Path(stage_dir).resolve()
    manifest_path = root / "stage-manifest.json"
    marker_path = root / "stage.done"
    manifest = _read_json(manifest_path, "completed stage manifest")
    supplied_manifest_sha = manifest.pop("manifest_sha256", None)
    if supplied_manifest_sha != _semantic_hash(manifest):
        raise FinalRCLExecutionError("completed stage manifest hash drifted")
    if (
        manifest.get("schema_version") != "final-rcl-completed-stage-v1"
        or manifest.get("status") != "completed"
    ):
        raise FinalRCLExecutionError("completed stage status drifted")
    spec = _validate_stage_spec(dict(manifest.get("spec", {})))
    if expected_spec is not None:
        expected = _validate_stage_spec(expected_spec)
        if spec["stage_content_sha256"] != expected["stage_content_sha256"]:
            raise FinalRCLExecutionError("completed stage input/dependency identity mismatched")
    outputs = manifest.get("outputs")
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence) or not outputs:
        raise FinalRCLExecutionError("completed stage output closure is empty")
    names: set[str] = set()
    relative_paths: set[str] = set()
    for raw in outputs:
        if not isinstance(raw, Mapping):
            raise FinalRCLExecutionError("completed stage output reference is invalid")
        name = str(raw.get("name", "")).strip()
        relative = str(raw.get("relative_path", "")).strip()
        if not name or name in names or not relative or relative in relative_paths:
            raise FinalRCLExecutionError("completed stage output identity is duplicate")
        output_path = (root / relative).resolve()
        try:
            output_path.relative_to(root)
        except ValueError as exc:
            raise FinalRCLExecutionError("completed stage output escapes stage root") from exc
        if (
            not output_path.is_file()
            or output_path.stat().st_size != raw.get("bytes")
            or _file_sha256(output_path) != raw.get("file_sha256")
        ):
            raise FinalRCLExecutionError("completed stage output hash drifted")
        names.add(name)
        relative_paths.add(relative)
    marker = _read_json(marker_path, "stage success marker")
    if marker != {
        "schema_version": "final-rcl-stage-success-marker-v1",
        "stage_content_sha256": spec["stage_content_sha256"],
        "manifest_sha256": supplied_manifest_sha,
    }:
        raise FinalRCLExecutionError("stage success marker drifted")
    return {
        "valid": True,
        "stage_kind": spec["stage_kind"],
        "owner_id": spec["owner_id"],
        "stage_id": spec["stage_id"],
        "stage_content_sha256": spec["stage_content_sha256"],
        "manifest_sha256": supplied_manifest_sha,
        "manifest_path": str(manifest_path),
        "success_marker_path": str(marker_path),
        "outputs": tuple(deepcopy(outputs)),
    }


def _record_reuse_rejection(stage_dir: Path, error: Exception) -> None:
    rejection = {
        "schema_version": "final-rcl-stage-reuse-rejection-v1",
        "observed_at": _now(),
        "error_type": type(error).__name__,
        "reason": str(error),
    }
    _atomic_json(
        Path(stage_dir) / "reuse_rejections" / f"{time.time_ns()}-{uuid.uuid4().hex}.json",
        rejection,
    )


def run_or_reuse_stage(
    *,
    stage_dir: Path,
    expected_spec: Mapping[str, Any],
    producer: Callable[[Path], Mapping[str, Path]],
) -> dict[str, Any]:
    root = Path(stage_dir).resolve()
    spec = _validate_stage_spec(expected_spec)
    had_prior_evidence = (root / "stage-manifest.json").exists() or (
        root / "stage.done"
    ).exists()
    try:
        audit = validate_completed_stage(stage_dir=root, expected_spec=spec)
    except FinalRCLExecutionError as exc:
        if had_prior_evidence:
            _record_reuse_rejection(root, exc)
    else:
        return {**audit, "decision": "reused"}

    root.mkdir(parents=True, exist_ok=True)
    produced = producer(root)
    if not isinstance(produced, Mapping) or not produced:
        raise FinalRCLExecutionError("stage producer returned no output files")
    output_refs = []
    seen_paths: set[str] = set()
    for raw_name, raw_path in sorted(produced.items()):
        name = str(raw_name).strip()
        output_path = Path(raw_path).resolve()
        try:
            relative = output_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise FinalRCLExecutionError("stage producer output escapes stage root") from exc
        if not name or not output_path.is_file() or relative in seen_paths:
            raise FinalRCLExecutionError("stage producer output closure is invalid")
        seen_paths.add(relative)
        output_refs.append(
            {
                "name": name,
                "relative_path": relative,
                "bytes": output_path.stat().st_size,
                "file_sha256": _file_sha256(output_path),
            }
        )
    manifest_identity = {
        "schema_version": "final-rcl-completed-stage-v1",
        "status": "completed",
        "completed_at": _now(),
        "spec": spec,
        "outputs": tuple(output_refs),
    }
    manifest_sha = _semantic_hash(manifest_identity)
    _atomic_json(
        root / "stage-manifest.json",
        {**manifest_identity, "manifest_sha256": manifest_sha},
    )
    _atomic_json(
        root / "stage.done",
        {
            "schema_version": "final-rcl-stage-success-marker-v1",
            "stage_content_sha256": spec["stage_content_sha256"],
            "manifest_sha256": manifest_sha,
        },
    )
    audit = validate_completed_stage(stage_dir=root, expected_spec=spec)
    return {
        **audit,
        "decision": "recomputed" if had_prior_evidence else "computed",
    }


def _validate_execution_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = deepcopy(dict(value))
    supplied_hash = manifest.pop("manifest_sha256", None)
    if supplied_hash != _semantic_hash(manifest):
        raise FinalRCLExecutionError("execution manifest hash drifted")
    units = manifest.get("units")
    if (
        manifest.get("schema_version") != "final-rcl-six-unit-execution-v1"
        or manifest.get("unit_count") != 6
        or manifest.get("seed") != 42
        or isinstance(units, (str, bytes))
        or not isinstance(units, Sequence)
        or len(units) != 6
    ):
        raise FinalRCLExecutionError("execution manifest identity drifted")
    expected_keys = {(dataset, arm) for dataset in _DATASETS for arm in _ARMS}
    observed_keys = {(unit.get("dataset_id"), unit.get("arm_id")) for unit in units}
    if observed_keys != expected_keys:
        raise FinalRCLExecutionError("execution manifest six-unit coverage drifted")
    output_roots = [str(unit.get("unit_output_root", "")) for unit in units]
    log_paths = [str(unit.get("log_path", "")) for unit in units]
    if (
        len(set(output_roots)) != 6
        or len(set(log_paths)) != 6
        or any(unit.get("resource_class") not in _RESOURCE_CLASSES for unit in units)
    ):
        raise FinalRCLExecutionError("unit output/log/resource ownership drifted")
    for key in ("code_sha256", "config_sha256", "input_sha256", "registry_sha256"):
        _sha256(manifest.get(key), key)
    for field in ("max_cpu_workers", "max_gpu_workers"):
        raw = manifest.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise FinalRCLExecutionError("execution worker limits must be positive integers")
    if not isinstance(manifest.get("require_aggregate_report"), bool):
        raise FinalRCLExecutionError("aggregate-report requirement must be boolean")
    aggregate_path = Path(str(manifest.get("aggregate_report_path", ""))).resolve()
    expected_aggregate_path = Path(str(manifest.get("run_root", ""))).resolve() / "aggregate-report.json"
    if aggregate_path != expected_aggregate_path:
        raise FinalRCLExecutionError("aggregate-report path drifted")
    return {**manifest, "manifest_sha256": supplied_hash}


def build_final_execution_manifest(
    *,
    registry: Mapping[str, Any],
    run_root: Path,
    tmux_session: str,
    code_sha256: str,
    config_sha256: str,
    input_sha256: str,
    max_cpu_workers: int,
    max_gpu_workers: int,
    require_aggregate_report: bool = False,
) -> dict[str, Any]:
    formal = deepcopy(dict(registry))
    units = formal.get("units")
    if (
        formal.get("schema_version") != "final-rcl-formal-registry-v1"
        or formal.get("unit_count") != 6
        or formal.get("seed") != 42
        or formal.get("clusterer_id") != "hdbscan"
        or formal.get("outer_router_enabled") is not False
        or not isinstance(units, Sequence)
        or len(units) != 6
    ):
        raise FinalRCLExecutionError("formal registry is not the frozen six-unit authority")
    run_path = Path(run_root).resolve()
    session = str(tmux_session).strip()
    if not session:
        raise FinalRCLExecutionError("tmux session must be nonempty")
    normalized_units = []
    for raw in units:
        unit = deepcopy(dict(raw))
        unit_id = str(unit.get("unit_id", "")).strip()
        arm = str(unit.get("arm_id", ""))
        if not unit_id or arm not in _ARMS:
            raise FinalRCLExecutionError("formal unit identity drifted")
        unit_root = run_path / "units" / unit_id
        normalized_units.append(
            {
                **unit,
                "resource_class": "cpu" if arm == "hdbscan_query_pairwise" else "gpu",
                "unit_output_root": str(unit_root),
                "log_path": str(run_path / "logs" / f"{unit_id}.log"),
                "final_stage_dir": str(unit_root / "unit_aggregate"),
            }
        )
    identity = {
        "schema_version": "final-rcl-six-unit-execution-v1",
        "created_at": _now(),
        "run_root": str(run_path),
        "tmux_session": session,
        "seed": 42,
        "unit_count": 6,
        "registry_sha256": _sha256(formal.get("registry_sha256"), "formal registry"),
        "code_sha256": _sha256(code_sha256, "execution code"),
        "config_sha256": _sha256(config_sha256, "execution config"),
        "input_sha256": _sha256(input_sha256, "execution input"),
        "max_cpu_workers": max_cpu_workers,
        "max_gpu_workers": max_gpu_workers,
        "require_aggregate_report": bool(require_aggregate_report),
        "aggregate_report_path": str(run_path / "aggregate-report.json"),
        "units": deepcopy(normalized_units),
        "all_done_path": str(run_path / "all.done"),
    }
    manifest = {**identity, "manifest_sha256": _semantic_hash(identity)}
    _validate_execution_manifest(manifest)
    manifest_path = run_path / "run-manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path, "existing execution manifest")
        existing.pop("created_at", None)
        candidate = deepcopy(manifest)
        candidate.pop("created_at", None)
        existing.pop("manifest_sha256", None)
        candidate.pop("manifest_sha256", None)
        if _semantic_hash(existing) != _semantic_hash(candidate):
            raise FinalRCLExecutionError("run root already belongs to a different execution")
        return _read_json(manifest_path, "existing execution manifest")
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    for unit in normalized_units:
        unit_root = Path(unit["unit_output_root"])
        unit_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            unit_root / "status.json",
            {
                "schema_version": "final-rcl-unit-status-v1",
                "unit_id": unit["unit_id"],
                "state": "pending",
                "updated_at": _now(),
            },
        )
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_run_manifest(run_root: Path) -> dict[str, Any]:
    manifest = _read_json(Path(run_root) / "run-manifest.json", "execution manifest")
    return _validate_execution_manifest(manifest)


def _valid_unit_aggregate(unit: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        audit = validate_completed_stage(stage_dir=Path(unit["final_stage_dir"]))
    except FinalRCLExecutionError:
        return None
    if audit["stage_kind"] != "unit_aggregate" or audit["owner_id"] != unit["unit_id"]:
        return None
    return audit


def read_final_run_status(
    *, run_root: Path, tmux_alive: bool | None = None
) -> dict[str, Any]:
    manifest = _load_run_manifest(run_root)
    completed = []
    running = []
    failed = []
    pending = []
    durations = []
    validated_stage_count = 0
    for unit in manifest["units"]:
        unit_root = Path(unit["unit_output_root"])
        for stage_kind in FINAL_STAGE_KINDS:
            stage_dir = unit_root / stage_kind
            if stage_dir.is_dir():
                try:
                    validate_completed_stage(stage_dir=stage_dir)
                except FinalRCLExecutionError:
                    pass
                else:
                    validated_stage_count += 1
        aggregate = _valid_unit_aggregate(unit)
        if aggregate is not None:
            completed.append(unit["unit_id"])
            status_path = unit_root / "status.json"
            if status_path.is_file():
                status = _read_json(status_path, "unit status")
                duration = status.get("duration_seconds")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                    if math.isfinite(float(duration)) and float(duration) >= 0.0:
                        durations.append(float(duration))
            continue
        failure_path = unit_root / "failure.json"
        if failure_path.is_file():
            failure = _read_json(failure_path, "unit failure")
            failed.append(failure)
            continue
        status_path = unit_root / "status.json"
        state = (
            _read_json(status_path, "unit status").get("state")
            if status_path.is_file()
            else "pending"
        )
        if state == "running":
            running.append(unit["unit_id"])
        else:
            pending.append(unit["unit_id"])
    remaining = len(running) + len(pending) + len(failed)
    worker_capacity = manifest["max_cpu_workers"] + manifest["max_gpu_workers"]
    eta = (
        math.fsum(durations) / len(durations) * remaining / worker_capacity
        if durations and remaining
        else (0.0 if not remaining else None)
    )
    all_done_path = Path(manifest["all_done_path"])
    marker_valid = False
    if all_done_path.is_file():
        try:
            marker = _read_json(all_done_path, "all.done marker")
            aggregate_valid = True
            if manifest["require_aggregate_report"]:
                aggregate = _validated_aggregate_report(
                    Path(manifest["aggregate_report_path"])
                )
                aggregate_valid = (
                    marker.get("aggregate_report_sha256")
                    == aggregate["aggregate_sha256"]
                    and marker.get("aggregate_report_file_sha256")
                    == _file_sha256(Path(manifest["aggregate_report_path"]))
                )
            marker_valid = (
                marker.get("schema_version") == "final-rcl-all-done-v1"
                and marker.get("execution_manifest_sha256") == manifest["manifest_sha256"]
                and set(marker.get("completed_unit_ids", ())) == set(completed)
                and len(completed) == 6
                and aggregate_valid
            )
        except FinalRCLExecutionError:
            marker_valid = False
    return {
        "schema_version": "final-rcl-progress-v1",
        "run_root": manifest["run_root"],
        "tmux_session": manifest["tmux_session"],
        "tmux_alive": tmux_alive,
        "expected": 6,
        "completed": len(completed),
        "running": len(running),
        "failed": len(failed),
        "pending": len(pending),
        "completed_unit_ids": tuple(completed),
        "running_unit_ids": tuple(running),
        "pending_unit_ids": tuple(pending),
        "validated_stage_count": validated_stage_count,
        "recent_failures": tuple(failed[-5:]),
        "estimated_remaining_seconds": eta,
        "all_done_path": str(all_done_path),
        "all_done_valid": marker_valid,
    }


def _validated_aggregate_report(path: Path) -> dict[str, Any]:
    report = _read_json(path, "aggregate report")
    supplied = report.pop("aggregate_sha256", None)
    if (
        supplied != _semantic_hash(report)
        or report.get("schema_version") != "final-rcl-matched-comparison-v1"
        or report.get("seed") != 42
        or report.get("unit_count") != 6
        or set(dict(report.get("datasets", {}))) != set(_DATASETS)
    ):
        raise FinalRCLExecutionError("aggregate report identity drifted")
    return {**report, "aggregate_sha256": supplied}


def finalize_final_run(run_root: Path) -> dict[str, Any]:
    manifest = _load_run_manifest(run_root)
    status = read_final_run_status(run_root=run_root)
    if status["completed"] != 6 or status["failed"] != 0:
        raise FinalRCLExecutionError(
            "all.done requires six validated unit aggregates and zero failures"
        )
    aggregate_fields: dict[str, Any] = {}
    if manifest["require_aggregate_report"]:
        aggregate_path = Path(manifest["aggregate_report_path"])
        aggregate = _validated_aggregate_report(aggregate_path)
        aggregate_fields = {
            "aggregate_report_sha256": aggregate["aggregate_sha256"],
            "aggregate_report_file_sha256": _file_sha256(aggregate_path),
        }
    marker = {
        "schema_version": "final-rcl-all-done-v1",
        "execution_manifest_sha256": manifest["manifest_sha256"],
        "completed_unit_ids": tuple(status["completed_unit_ids"]),
        "completed_at": _now(),
        **aggregate_fields,
    }
    _atomic_json(Path(manifest["all_done_path"]), marker)
    return marker


def run_final_execution(
    *,
    execution_manifest: Mapping[str, Any],
    unit_worker: Callable[[Mapping[str, Any], Path, Path], Mapping[str, Any]],
    auto_finalize: bool = True,
) -> dict[str, Any]:
    manifest = _validate_execution_manifest(execution_manifest)
    persisted = _load_run_manifest(Path(manifest["run_root"]))
    if persisted["manifest_sha256"] != manifest["manifest_sha256"]:
        raise FinalRCLExecutionError("persisted execution manifest differs from request")
    cpu_slots = threading.Semaphore(manifest["max_cpu_workers"])
    gpu_slots = threading.Semaphore(manifest["max_gpu_workers"])
    reused_unit_count = 0
    pending_units = []
    for unit in manifest["units"]:
        if _valid_unit_aggregate(unit) is not None:
            reused_unit_count += 1
        else:
            pending_units.append(unit)

    def execute(unit: Mapping[str, Any]) -> None:
        resource = unit["resource_class"]
        semaphore = gpu_slots if resource == "gpu" else cpu_slots
        unit_root = Path(unit["unit_output_root"])
        log_path = Path(unit["log_path"])
        started = time.monotonic()
        with semaphore:
            failure_path = unit_root / "failure.json"
            _atomic_json(
                unit_root / "status.json",
                {
                    "schema_version": "final-rcl-unit-status-v1",
                    "unit_id": unit["unit_id"],
                    "state": "running",
                    "resource_class": resource,
                    "started_at": _now(),
                    "updated_at": _now(),
                },
            )
            try:
                result = unit_worker(unit, unit_root, log_path)
                audit = _valid_unit_aggregate(unit)
                if audit is None:
                    raise FinalRCLExecutionError(
                        "unit worker returned without a validated unit aggregate"
                    )
                if result.get("manifest_sha256") != audit["manifest_sha256"]:
                    raise FinalRCLExecutionError("unit worker aggregate identity drifted")
            except Exception as exc:  # retained as durable experiment evidence
                _atomic_json(
                    failure_path,
                    {
                        "schema_version": "final-rcl-unit-failure-v1",
                        "unit_id": unit["unit_id"],
                        "failed_at": _now(),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                _atomic_json(
                    unit_root / "status.json",
                    {
                        "schema_version": "final-rcl-unit-status-v1",
                        "unit_id": unit["unit_id"],
                        "state": "failed",
                        "updated_at": _now(),
                        "duration_seconds": time.monotonic() - started,
                    },
                )
                return
            if failure_path.exists():
                failure_path.unlink()
            _atomic_json(
                unit_root / "status.json",
                {
                    "schema_version": "final-rcl-unit-status-v1",
                    "unit_id": unit["unit_id"],
                    "state": "completed",
                    "updated_at": _now(),
                    "duration_seconds": time.monotonic() - started,
                    "aggregate_manifest_sha256": audit["manifest_sha256"],
                },
            )

    max_workers = manifest["max_cpu_workers"] + manifest["max_gpu_workers"]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(execute, unit) for unit in pending_units]
        for future in as_completed(futures):
            future.result()
    status = read_final_run_status(run_root=Path(manifest["run_root"]))
    status["reused_unit_count"] = reused_unit_count
    if not isinstance(auto_finalize, bool):
        raise FinalRCLExecutionError("auto_finalize must be boolean")
    if status["completed"] == 6 and status["failed"] == 0 and auto_finalize:
        finalize_final_run(Path(manifest["run_root"]))
        status = read_final_run_status(run_root=Path(manifest["run_root"]))
        status["reused_unit_count"] = reused_unit_count
    return status


__all__ = [
    "FINAL_STAGE_KINDS",
    "FinalRCLExecutionError",
    "build_final_execution_manifest",
    "build_stage_spec",
    "finalize_final_run",
    "read_final_run_status",
    "run_final_execution",
    "run_or_reuse_stage",
    "validate_completed_stage",
]
