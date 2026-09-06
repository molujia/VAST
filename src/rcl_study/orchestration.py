"""Fail-closed, resumable orchestration primitives for detached RCL runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .artifacts import sha256_file


RUN_MANIFEST_SCHEMA_VERSION = "rcl-experiment-run-manifest-v1"
PROGRESS_SCHEMA_VERSION = "rcl-experiment-progress-v1"
UNIT_MANIFEST_SCHEMA_VERSION = "rcl-experiment-unit-attempt-v1"
COMPLETION_SCHEMA_VERSION = "rcl-experiment-completion-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JSON_REPLACE_ATTEMPTS = 5
_JSON_REPLACE_BACKOFF_SECONDS = 0.02


def _utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(
        float(epoch), timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_component(value: Any, context: str) -> str:
    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or not _SAFE_COMPONENT.fullmatch(text)
    ):
        raise ValueError("%s contains an unsafe path component: %r" % (context, text))
    return text


def _require_sha256(value: Any, context: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError("%s must be a lowercase SHA-256 digest" % context)
    return text


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        last_error: PermissionError | None = None
        for attempt in range(_JSON_REPLACE_ATTEMPTS):
            try:
                temporary.replace(path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                if attempt + 1 >= _JSON_REPLACE_ATTEMPTS:
                    break
                time.sleep(_JSON_REPLACE_BACKOFF_SECONDS)
        if last_error is not None:
            raise last_error
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a JSON object" % context)
    return dict(payload)


def _semantic_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_semantic_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    input_sha256: str
    config_sha256: str
    code_sha256: str
    case_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        _safe_component(self.unit_id, "unit_id")
        _require_sha256(self.input_sha256, "input_sha256")
        _require_sha256(self.config_sha256, "config_sha256")
        _require_sha256(self.code_sha256, "code_sha256")
        normalized = tuple(str(case_id) for case_id in self.case_ids)
        if not normalized or any(not case_id for case_id in normalized):
            raise ValueError("UnitSpec case_ids must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("UnitSpec case_ids must be unique")
        object.__setattr__(self, "case_ids", normalized)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["case_ids"] = list(self.case_ids)
        return payload


@dataclass(frozen=True)
class UnitExecution:
    unit_id: str
    attempt_id: str
    attempt_dir: Path
    result_path: Path
    skipped: bool


class ProgressTracker:
    """Atomically persist progress and append a compact tqdm-style log."""

    def __init__(
        self,
        root: Path,
        *,
        total_units: int,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.root = Path(root)
        self.total_units = int(total_units)
        if self.total_units < 0:
            raise ValueError("total_units must be non-negative")
        self.clock = clock or time.time
        self.progress_path = self.root / "progress.json"
        self.log_path = self.root / "logs" / "run.log"

    def _read(self) -> Dict[str, Any]:
        return _read_json(self.progress_path, "progress")

    def _line(self, state: Mapping[str, Any]) -> str:
        total = int(state["total_units"])
        completed = int(state["completed_units"])
        ratio = float(completed) / float(total) if total else 1.0
        width = 20
        filled = min(width, max(0, int(round(ratio * width))))
        bar = "=" * filled + "." * (width - filled)
        return (
            "[%s] %d/%d %5.1f%% phase=%s unit=%s failures=%d"
            % (
                bar,
                completed,
                total,
                ratio * 100.0,
                state.get("phase", ""),
                state.get("current_unit", "") or "-",
                int(state.get("failure_count", 0)),
            )
        )

    def _commit(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _write_json(self.progress_path, state)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(self._line(state) + "\n")
        return state

    def initialize(self, phase: str) -> Dict[str, Any]:
        now = float(self.clock())
        state = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "phase": str(phase),
            "completed_units": 0,
            "total_units": self.total_units,
            "current_unit": "",
            "latest_unit": "",
            "failure_count": 0,
            "started_at_epoch": now,
            "started_at_utc": _utc_iso(now),
            "last_update_epoch": now,
            "last_update_utc": _utc_iso(now),
            "elapsed_seconds": 0.0,
        }
        return self._commit(state)

    def transition(
        self, phase: str, *, current_unit: str = ""
    ) -> Dict[str, Any]:
        state = self._read()
        now = float(self.clock())
        state.update(
            {
                "phase": str(phase),
                "current_unit": str(current_unit),
                "last_update_epoch": now,
                "last_update_utc": _utc_iso(now),
                "elapsed_seconds": max(
                    0.0, now - float(state["started_at_epoch"])
                ),
            }
        )
        return self._commit(state)

    def advance(
        self,
        unit_id: str,
        *,
        failed: bool = False,
    ) -> Dict[str, Any]:
        state = self._read()
        now = float(self.clock())
        completed = int(state["completed_units"])
        if not failed:
            completed = min(self.total_units, completed + 1)
        state.update(
            {
                "completed_units": completed,
                "current_unit": str(unit_id),
                "latest_unit": str(unit_id),
                "failure_count": int(state.get("failure_count", 0))
                + int(bool(failed)),
                "last_update_epoch": now,
                "last_update_utc": _utc_iso(now),
                "elapsed_seconds": max(
                    0.0, now - float(state["started_at_epoch"])
                ),
            }
        )
        return self._commit(state)


def create_run_root(
    base_root: Path,
    *,
    stage: str,
    method_id: str,
    canonical_dataset_id: str,
    candidate_id: str,
    seed: int,
) -> Path:
    components = [
        _safe_component(stage, "stage"),
        _safe_component(method_id, "method_id"),
        _safe_component(canonical_dataset_id, "canonical_dataset_id"),
        _safe_component(candidate_id, "candidate_id"),
        "seed-%d" % int(seed),
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "run-%s-%s" % (stamp, uuid.uuid4().hex[:12])
    root = Path(base_root).joinpath(*components, run_id)
    root.mkdir(parents=True, exist_ok=False)
    return root


def initialize_run(
    root: Path,
    *,
    stage: str,
    method_id: str,
    canonical_dataset_id: str,
    candidate_id: str,
    seed: int,
    expected_units: Sequence[UnitSpec],
    config_sha256: str,
    code_sha256: str,
) -> Dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_digest = _require_sha256(config_sha256, "config_sha256")
    code_digest = _require_sha256(code_sha256, "code_sha256")
    unit_ids = [spec.unit_id for spec in expected_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("expected unit IDs must be unique")
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": root.name,
        "stage": _safe_component(stage, "stage"),
        "method_id": _safe_component(method_id, "method_id"),
        "canonical_dataset_id": _safe_component(
            canonical_dataset_id, "canonical_dataset_id"
        ),
        "candidate_id": _safe_component(candidate_id, "candidate_id"),
        "seed": int(seed),
        "config_sha256": config_digest,
        "code_sha256": code_digest,
        "expected_unit_count": len(expected_units),
        "expected_units": [spec.to_dict() for spec in expected_units],
    }
    manifest["contract_sha256"] = _semantic_sha256(manifest)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path, "run manifest")
        if existing != manifest:
            raise ValueError("existing run manifest does not match requested contract")
    else:
        _write_json(manifest_path, manifest)
    tracker = ProgressTracker(root, total_units=len(expected_units))
    if not tracker.progress_path.exists():
        tracker.initialize("initialized")
    return manifest


def _spec_matches(manifest: Mapping[str, Any], spec: UnitSpec) -> bool:
    return (
        str(manifest.get("unit_id", "")) == spec.unit_id
        and str(manifest.get("input_sha256", "")) == spec.input_sha256
        and str(manifest.get("config_sha256", "")) == spec.config_sha256
        and str(manifest.get("code_sha256", "")) == spec.code_sha256
        and tuple(str(item) for item in manifest.get("case_ids", []))
        == spec.case_ids
    )


def _valid_completed_attempt(
    root: Path,
    spec: UnitSpec,
) -> Optional[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    unit_dir = Path(root) / "units" / spec.unit_id
    for attempt_dir in sorted(
        unit_dir.glob("attempt-*"), reverse=True
    ):
        done_path = attempt_dir / ".done"
        manifest_path = attempt_dir / "unit_manifest.json"
        result_path = attempt_dir / "result.json"
        if not (done_path.is_file() and manifest_path.is_file() and result_path.is_file()):
            continue
        try:
            manifest = _read_json(manifest_path, "unit manifest")
            if (
                str(manifest.get("schema_version", ""))
                != UNIT_MANIFEST_SCHEMA_VERSION
                or str(manifest.get("status", "")) != "complete"
                or not _spec_matches(manifest, spec)
                or str(manifest.get("result_sha256", ""))
                != sha256_file(result_path)
                or done_path.read_text(encoding="utf-8").strip()
                != sha256_file(manifest_path)
            ):
                continue
            result = _read_json(result_path, "unit result")
            if str(result.get("unit_id", "")) != spec.unit_id:
                continue
            if tuple(str(item) for item in result.get("case_ids", [])) != spec.case_ids:
                continue
        except (OSError, ValueError):
            continue
        return attempt_dir, manifest, result
    return None


def _next_attempt_dir(unit_dir: Path) -> Path:
    existing = []
    for path in unit_dir.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if match:
            existing.append(int(match.group(1)))
    attempt_number = max(existing, default=0) + 1
    attempt_dir = unit_dir / ("attempt-%04d" % attempt_number)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir


def execute_unit(
    root: Path,
    spec: UnitSpec,
    runner: Callable[[], Mapping[str, Any]],
    *,
    tracker: Optional[ProgressTracker] = None,
) -> UnitExecution:
    root = Path(root)
    existing = _valid_completed_attempt(root, spec)
    if existing is not None:
        attempt_dir = existing[0]
        if tracker is not None:
            tracker.advance(spec.unit_id)
        return UnitExecution(
            unit_id=spec.unit_id,
            attempt_id=attempt_dir.name,
            attempt_dir=attempt_dir,
            result_path=attempt_dir / "result.json",
            skipped=True,
        )

    unit_dir = root / "units" / spec.unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(unit_dir)
    started = time.time()
    base_manifest = {
        "schema_version": UNIT_MANIFEST_SCHEMA_VERSION,
        "unit_id": spec.unit_id,
        "attempt_id": attempt_dir.name,
        "status": "running",
        "input_sha256": spec.input_sha256,
        "config_sha256": spec.config_sha256,
        "code_sha256": spec.code_sha256,
        "case_ids": list(spec.case_ids),
        "started_at_epoch": started,
        "started_at_utc": _utc_iso(started),
    }
    _write_json(attempt_dir / "unit_manifest.json", base_manifest)
    try:
        raw_result = runner()
        if not isinstance(raw_result, Mapping):
            raise ValueError("unit runner must return a mapping")
        result = dict(raw_result)
        result_path = attempt_dir / "result.json"
        _write_json(result_path, result)
        finished = time.time()
        complete_manifest = dict(base_manifest)
        complete_manifest.update(
            {
                "status": "complete",
                "result_sha256": sha256_file(result_path),
                "finished_at_epoch": finished,
                "finished_at_utc": _utc_iso(finished),
                "elapsed_seconds": max(0.0, finished - started),
            }
        )
        manifest_path = attempt_dir / "unit_manifest.json"
        _write_json(manifest_path, complete_manifest)
        (attempt_dir / ".done").write_text(
            sha256_file(manifest_path) + "\n", encoding="utf-8"
        )
        if tracker is not None:
            tracker.advance(spec.unit_id)
        return UnitExecution(
            unit_id=spec.unit_id,
            attempt_id=attempt_dir.name,
            attempt_dir=attempt_dir,
            result_path=result_path,
            skipped=False,
        )
    except Exception as exc:
        finished = time.time()
        failed_manifest = dict(base_manifest)
        failed_manifest.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "finished_at_epoch": finished,
                "finished_at_utc": _utc_iso(finished),
                "elapsed_seconds": max(0.0, finished - started),
            }
        )
        _write_json(attempt_dir / "unit_manifest.json", failed_manifest)
        _write_json(
            attempt_dir / "error.json",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        (attempt_dir / ".failed").write_text(
            "%s: %s\n" % (type(exc).__name__, exc),
            encoding="utf-8",
        )
        if tracker is not None:
            tracker.advance(spec.unit_id, failed=True)
        raise


def _validate_units(
    root: Path,
    *,
    expected_units: Sequence[UnitSpec],
    expected_case_ids: Set[str],
    required_metrics: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    root = Path(root)
    manifest = _read_json(root / "manifest.json", "run manifest")
    expected_payload = [spec.to_dict() for spec in expected_units]
    if manifest.get("expected_units") != expected_payload:
        raise ValueError("run manifest expected unit contract mismatch")
    if int(manifest.get("expected_unit_count", -1)) != len(expected_units):
        raise ValueError("run manifest expected unit count mismatch")
    results = []
    covered_cases = set()
    for spec in expected_units:
        valid = _valid_completed_attempt(root, spec)
        if valid is None:
            raise ValueError(
                "required unit is missing, stale, corrupt, or failed: %s"
                % spec.unit_id
            )
        result = valid[2]
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("unit %s metrics must be an object" % spec.unit_id)
        for metric_name in required_metrics:
            value = metrics.get(metric_name)
            if isinstance(value, bool):
                raise ValueError(
                    "unit %s metric %s must be finite"
                    % (spec.unit_id, metric_name)
                )
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "unit %s metric %s must be finite"
                    % (spec.unit_id, metric_name)
                )
            if not math.isfinite(number):
                raise ValueError(
                    "unit %s metric %s must be finite"
                    % (spec.unit_id, metric_name)
                )
        covered_cases.update(str(case_id) for case_id in result["case_ids"])
        results.append(result)
    if covered_cases != {str(case_id) for case_id in expected_case_ids}:
        raise ValueError("validated unit case coverage mismatch")
    historical_failed_attempt_count = len(
        list((root / "units").rglob(".failed"))
    )
    return results, sorted(covered_cases), historical_failed_attempt_count


def finalize_run(
    root: Path,
    *,
    expected_units: Sequence[UnitSpec],
    expected_case_ids: Set[str],
    required_metrics: Sequence[str],
) -> Dict[str, Any]:
    root = Path(root)
    for marker_name in ("COMPLETED.json", "all.done"):
        marker = root / marker_name
        if marker.exists():
            marker.unlink()
    results, covered_cases, failed_count = _validate_units(
        root,
        expected_units=expected_units,
        expected_case_ids=expected_case_ids,
        required_metrics=required_metrics,
    )
    results_path = root / "results" / "validated_units.json"
    _write_json(
        results_path,
        {
            "unit_count": len(results),
            "units": results,
        },
    )
    now = time.time()
    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "run_manifest_sha256": sha256_file(root / "manifest.json"),
        "validated_results_sha256": sha256_file(results_path),
        "validated_unit_ids": sorted(spec.unit_id for spec in expected_units),
        "validated_case_ids": covered_cases,
        "historical_failed_attempt_count": failed_count,
        "required_metrics": list(required_metrics),
        "completed_at_epoch": now,
        "completed_at_utc": _utc_iso(now),
    }
    completion_path = root / "COMPLETED.json"
    _write_json(completion_path, completion)
    (root / "all.done").write_text(
        sha256_file(completion_path) + "\n", encoding="utf-8"
    )
    progress_path = root / "progress.json"
    if progress_path.exists():
        tracker = ProgressTracker(root, total_units=len(expected_units))
        tracker.transition("complete")
    return completion


def validate_run_completion(
    root: Path,
    *,
    expected_units: Sequence[UnitSpec],
    expected_case_ids: Set[str],
    required_metrics: Sequence[str],
) -> Dict[str, Any]:
    root = Path(root)
    completion_path = root / "COMPLETED.json"
    all_done_path = root / "all.done"
    if not completion_path.is_file():
        raise ValueError("COMPLETED.json success record is missing")
    if not all_done_path.is_file():
        raise ValueError("all.done success marker is missing")
    results, covered_cases, failed_count = _validate_units(
        root,
        expected_units=expected_units,
        expected_case_ids=expected_case_ids,
        required_metrics=required_metrics,
    )
    completion = _read_json(completion_path, "completion record")
    if (
        str(completion.get("schema_version", ""))
        != COMPLETION_SCHEMA_VERSION
        or str(completion.get("status", "")) != "complete"
    ):
        raise ValueError("completion record is invalid")
    if (
        all_done_path.read_text(encoding="utf-8").strip()
        != sha256_file(completion_path)
    ):
        raise ValueError("all.done completion hash mismatch")
    results_path = root / "results" / "validated_units.json"
    if (
        not results_path.is_file()
        or str(completion.get("validated_results_sha256", ""))
        != sha256_file(results_path)
    ):
        raise ValueError("validated results hash mismatch")
    if str(completion.get("run_manifest_sha256", "")) != sha256_file(
        root / "manifest.json"
    ):
        raise ValueError("run manifest hash mismatch")
    if completion.get("validated_case_ids") != covered_cases:
        raise ValueError("completion case coverage mismatch")
    if completion.get("validated_unit_ids") != sorted(
        spec.unit_id for spec in expected_units
    ):
        raise ValueError("completion unit coverage mismatch")
    if int(completion.get("historical_failed_attempt_count", -1)) != failed_count:
        raise ValueError("completion historical failure count mismatch")
    if completion.get("required_metrics") != list(required_metrics):
        raise ValueError("completion required metrics mismatch")
    if len(results) != len(expected_units):
        raise ValueError("completion result count mismatch")
    return completion


def _default_tmux_probe(session: str) -> bool:
    completed = subprocess.run(
        ["tmux", "has-session", "-t", str(session)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _tail_text_file(path: Path, limit: int) -> List[str]:
    count = max(0, int(limit))
    candidate = Path(path)
    if count == 0 or not candidate.is_file():
        return []
    with candidate.open(
        "r", encoding="utf-8", errors="replace"
    ) as handle:
        return [
            line.rstrip("\r\n")
            for line in deque(handle, maxlen=count)
        ]


def collect_status(
    root: Path,
    *,
    tmux_session: str = "",
    tmux_probe: Optional[Callable[[str], bool]] = None,
    recent_log_lines: int = 20,
    clock: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    root = Path(root)
    log_limit = max(0, int(recent_log_lines))
    progress_path = root / "progress.json"
    progress = (
        _read_json(progress_path, "progress")
        if progress_path.is_file()
        else {}
    )
    now = float((clock or time.time)())
    last_update = progress.get("last_update_epoch")
    if last_update is not None:
        progress["last_activity_age_seconds"] = max(
            0.0, now - float(last_update)
        )
    unit_progress: Dict[str, Any] = {}
    current_unit = str(progress.get("current_unit", ""))
    if (
        current_unit
        and Path(current_unit).name == current_unit
        and current_unit not in {".", ".."}
    ):
        detail_path = (
            root / "unit_progress" / (current_unit + ".json")
        )
        if detail_path.is_file():
            unit_progress = _read_json(
                detail_path, "unit progress"
            )
            detail_update = unit_progress.get("last_update_epoch")
            if detail_update is not None:
                unit_progress["last_activity_age_seconds"] = max(
                    0.0, now - float(detail_update)
                )
    log_path = root / "logs" / "run.log"
    recent = _tail_text_file(log_path, log_limit)
    environment_unit_log: Optional[Path] = None
    recent_environment_unit_log_lines: List[str] = []
    model_train_log: Optional[Path] = None
    recent_model_train_log_lines: List[str] = []
    if (
        current_unit
        and Path(current_unit).name == current_unit
        and current_unit not in {".", ".."}
    ):
        current_unit_root = (
            root / "optimization_units" / current_unit
        )
        candidate_log = (
            current_unit_root / "logs" / "environment-unit.log"
        )
        if candidate_log.is_file():
            environment_unit_log = candidate_log
            recent_environment_unit_log_lines = _tail_text_file(
                candidate_log, log_limit
            )
        train_logs = [
            path
            for path in current_unit_root.rglob("train_log.jsonl")
            if path.is_file()
        ]
        if train_logs:
            model_train_log = max(
                train_logs,
                key=lambda path: (path.stat().st_mtime, path.as_posix()),
            )
            recent_model_train_log_lines = _tail_text_file(
                model_train_log, log_limit
            )
    active_units: List[Dict[str, Any]] = []
    active_root = root / "active_units"
    if active_root.is_dir():
        for marker_path in sorted(active_root.glob("*.json")):
            try:
                marker = _read_json(marker_path, "active unit")
            except (OSError, ValueError):
                continue
            unit_id = str(marker.get("unit_id", ""))
            if (
                not unit_id
                or Path(unit_id).name != unit_id
                or unit_id in {".", ".."}
                or marker_path.name != unit_id + ".json"
            ):
                continue
            active_unit_root = root / "optimization_units" / unit_id
            active_log = (
                active_unit_root / "logs" / "environment-unit.log"
            )
            detail_path = root / "unit_progress" / (unit_id + ".json")
            detail = {}
            if detail_path.is_file():
                try:
                    detail = _read_json(
                        detail_path, "active unit progress"
                    )
                except (OSError, ValueError):
                    detail = {}
            train_logs = [
                path
                for path in active_unit_root.rglob("train_log.jsonl")
                if path.is_file()
            ]
            active_train_log = (
                max(
                    train_logs,
                    key=lambda path: (
                        path.stat().st_mtime,
                        path.as_posix(),
                    ),
                )
                if train_logs
                else None
            )
            started_at_epoch = marker.get("started_at_epoch")
            age = (
                max(0.0, now - float(started_at_epoch))
                if started_at_epoch is not None
                else None
            )
            active_units.append(
                {
                    "unit_id": unit_id,
                    "environment": str(marker.get("environment", "")),
                    "started_at_epoch": started_at_epoch,
                    "active_age_seconds": age,
                    "unit_progress": detail,
                    "environment_unit_log": str(active_log),
                    "recent_environment_unit_log_lines": (
                        _tail_text_file(active_log, log_limit)
                    ),
                    "model_train_log": (
                        str(active_train_log)
                        if active_train_log is not None
                        else ""
                    ),
                    "recent_model_train_log_lines": (
                        _tail_text_file(active_train_log, log_limit)
                        if active_train_log is not None
                        else []
                    ),
                }
            )
    failure_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob(".failed")
        if path.is_file()
    )
    probe = tmux_probe or _default_tmux_probe
    alive = bool(probe(tmux_session)) if tmux_session else False
    completion_path = root / "COMPLETED.json"
    all_done_path = root / "all.done"
    return {
        "root": str(root),
        "tmux": {
            "session": str(tmux_session),
            "alive": alive,
        },
        "progress": progress,
        "unit_progress": unit_progress,
        "recent_log_lines": recent,
        "environment_unit_log": (
            str(environment_unit_log)
            if environment_unit_log is not None
            else ""
        ),
        "recent_environment_unit_log_lines": (
            recent_environment_unit_log_lines
        ),
        "model_train_log": (
            str(model_train_log) if model_train_log is not None else ""
        ),
        "recent_model_train_log_lines": recent_model_train_log_lines,
        "active_units": active_units,
        "failures": {
            "count": len(failure_paths),
            "paths": failure_paths,
        },
        "completion": {
            "completed_json": str(completion_path),
            "completed_json_exists": completion_path.is_file(),
            "all_done": str(all_done_path),
            "all_done_exists": all_done_path.is_file(),
        },
    }


def validate_expected_unit_closure(
    root: Path,
    *,
    expected_unit_ids: Sequence[str],
    units_directory: str = "units",
) -> Dict[str, Any]:
    """Validate one direct result/manifest/.done hash chain per expected unit."""

    root = Path(root)
    directory_name = _safe_component(
        units_directory,
        "units_directory",
    )
    unit_ids = tuple(
        _safe_component(value, "expected unit ID")
        for value in expected_unit_ids
    )
    if not unit_ids or len(unit_ids) != len(set(unit_ids)):
        raise ValueError(
            "expected unit IDs must be non-empty and unique"
        )
    units_root = root / directory_name
    active_failures = sorted(
        path.as_posix()
        for path in units_root.rglob(".failed")
        if path.is_file()
    )
    if active_failures:
        raise ValueError(
            "expected unit closure contains active failed markers: %s"
            % active_failures
        )
    result_hashes: Dict[str, str] = {}
    manifest_hashes: Dict[str, str] = {}
    for unit_id in unit_ids:
        unit_root = units_root / unit_id
        result_path = unit_root / "result.json"
        manifest_path = unit_root / "unit_manifest.json"
        done_path = unit_root / ".done"
        if not (
            result_path.is_file()
            and manifest_path.is_file()
            and done_path.is_file()
        ):
            raise ValueError(
                "required unit is missing, stale, corrupt, or failed: %s"
                % unit_id
            )
        try:
            result = _read_json(result_path, "unit result")
            manifest = _read_json(manifest_path, "unit manifest")
            result_hash = sha256_file(result_path)
            manifest_hash = sha256_file(manifest_path)
            valid = (
                str(result.get("unit_id", "")) == unit_id
                and str(result.get("status", "")) == "complete"
                and str(manifest.get("unit_id", "")) == unit_id
                and str(manifest.get("status", "")) == "complete"
                and str(manifest.get("result_sha256", ""))
                == result_hash
                and done_path.read_text(encoding="utf-8").strip()
                == manifest_hash
            )
        except (OSError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                "required unit is missing, stale, corrupt, or failed: %s"
                % unit_id
            )
        result_hashes[unit_id] = result_hash
        manifest_hashes[unit_id] = manifest_hash
    return {
        "validated_unit_ids": sorted(unit_ids),
        "unit_result_sha256": {
            unit_id: result_hashes[unit_id]
            for unit_id in sorted(result_hashes)
        },
        "unit_manifest_sha256": {
            unit_id: manifest_hashes[unit_id]
            for unit_id in sorted(manifest_hashes)
        },
    }


__all__ = [
    "COMPLETION_SCHEMA_VERSION",
    "PROGRESS_SCHEMA_VERSION",
    "ProgressTracker",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "UNIT_MANIFEST_SCHEMA_VERSION",
    "UnitExecution",
    "UnitSpec",
    "collect_status",
    "create_run_root",
    "execute_unit",
    "finalize_run",
    "initialize_run",
    "sha256_file",
    "validate_expected_unit_closure",
    "validate_run_completion",
]
