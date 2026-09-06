"""Hash-addressed per-case caching and fail-closed rcabench aggregation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .artifacts import (
    AUDIT_SCHEMA_VERSION,
    SERIES_SCHEMA_VERSION,
    sha256_file,
    validate_audit,
    validate_canonical_series_file,
    validate_canonical_series_rows,
)
from .rcabench import CANONICAL_DATASET_ID, canonicalize_case


CASE_CACHE_SCHEMA_VERSION = "rcabench-case-cache-v1"
RUN_MANIFEST_SCHEMA_VERSION = "rcabench-canonical-preprocessing-v1"
RUN_COMPLETION_SCHEMA_VERSION = "rcabench-canonical-completion-v1"

_SOURCE_FILENAMES = (
    "normal_metrics.parquet",
    "abnormal_metrics.parquet",
    "normal_metrics_sum.parquet",
    "abnormal_metrics_sum.parquet",
    "normal_metrics_histogram.parquet",
    "abnormal_metrics_histogram.parquet",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_COUNT_NAMES = (
    "input_file_count",
    "existing_file_count",
    "empty_file_count",
    "missing_file_count",
    "exact_duplicates_removed",
    "conflicting_series_count",
    "conflicting_observation_count",
    "non_finite_row_count",
    "unresolved_service_row_count",
    "output_series_count",
    "output_row_count",
)


class DatasetCanonicalizationError(RuntimeError):
    """Raised after one or more cases fail canonical preprocessing."""


def _require_sha256(value: str, context: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError("%s must be a lowercase SHA-256 digest" % context)
    return text


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _read_json(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    if not isinstance(payload, Mapping):
        raise ValueError("%s must be a JSON object" % context)
    return dict(payload)


def _read_jsonl(path: Path, context: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("line %d is not an object" % line_number)
                rows.append(dict(payload))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (context, path, exc))
    return rows


def _case_slug(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id)).strip("._")
    if not readable:
        readable = "case"
    suffix = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:12]
    return "%s-%s" % (readable[:80], suffix)


def source_checksums(case_dir: Path) -> Dict[str, str]:
    case_dir = Path(case_dir)
    checksums: Dict[str, str] = {}
    for filename in _SOURCE_FILENAMES:
        path = case_dir / filename
        if path.is_file():
            checksums[filename] = sha256_file(path)
    return checksums


def _cache_key(
    case_id: str,
    checksums: Mapping[str, str],
    config_sha256: str,
    code_sha256: str,
) -> str:
    return _json_sha256(
        {
            "schema_version": CASE_CACHE_SCHEMA_VERSION,
            "case_id": str(case_id),
            "source_checksums": dict(checksums),
            "config_sha256": config_sha256,
            "code_sha256": code_sha256,
        }
    )


def _validate_case_attempt(
    attempt_dir: Path,
    *,
    case_id: str,
    cache_key: str,
    checksums: Mapping[str, str],
    config_sha256: str,
    code_sha256: str,
    require_completion: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = attempt_dir / "manifest.json"
    rows_path = attempt_dir / "rows.jsonl"
    audit_path = attempt_dir / "audit.json"
    manifest = _read_json(manifest_path, "case-cache manifest")
    if manifest.get("schema_version") != CASE_CACHE_SCHEMA_VERSION:
        raise ValueError("case-cache manifest schema_version mismatch")
    expected = {
        "case_id": str(case_id),
        "cache_key": cache_key,
        "source_checksums": dict(checksums),
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError("case-cache manifest %s mismatch" % field)
    if sha256_file(rows_path) != str(manifest.get("rows_sha256", "")):
        raise ValueError("case-cache rows hash mismatch")
    if sha256_file(audit_path) != str(manifest.get("audit_sha256", "")):
        raise ValueError("case-cache audit hash mismatch")

    rows = _read_jsonl(rows_path, "case-cache rows")
    audit = _read_json(audit_path, "case-cache audit")
    summary = validate_canonical_series_rows(rows, CANONICAL_DATASET_ID)
    if summary["case_count"] != 1 or {row["case_id"] for row in rows} != {case_id}:
        raise ValueError("case-cache rows case coverage mismatch")
    if int(audit.get("output_row_count", -1)) != summary["row_count"]:
        raise ValueError("case-cache audit output_row_count mismatch")
    if int(audit.get("output_series_count", -1)) != summary["series_count"]:
        raise ValueError("case-cache audit output_series_count mismatch")

    if require_completion:
        completion_path = attempt_dir / "COMPLETED.json"
        all_done_path = attempt_dir / "all.done"
        completion = _read_json(completion_path, "case-cache completion")
        manifest_sha256 = sha256_file(manifest_path)
        if completion.get("status") != "complete":
            raise ValueError("case-cache completion status mismatch")
        if completion.get("manifest_sha256") != manifest_sha256:
            raise ValueError("case-cache completion manifest hash mismatch")
        if all_done_path.read_text(encoding="utf-8").strip() != manifest_sha256:
            raise ValueError("case-cache all.done manifest hash mismatch")
    return manifest, rows, audit


def _load_current_case_cache(
    key_root: Path,
    *,
    case_id: str,
    cache_key: str,
    checksums: Mapping[str, str],
    config_sha256: str,
    code_sha256: str,
) -> Optional[Tuple[Path, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]]:
    pointer_path = key_root / "current.json"
    if not pointer_path.is_file():
        return None
    try:
        pointer = _read_json(pointer_path, "case-cache pointer")
        attempt_name = str(pointer["attempt"])
        if not re.fullmatch(r"attempt-[0-9]{4}", attempt_name):
            raise ValueError("case-cache pointer attempt is unsafe")
        attempt_dir = key_root / attempt_name
        manifest, rows, audit = _validate_case_attempt(
            attempt_dir,
            case_id=case_id,
            cache_key=cache_key,
            checksums=checksums,
            config_sha256=config_sha256,
            code_sha256=code_sha256,
            require_completion=True,
        )
        if pointer.get("manifest_sha256") != sha256_file(
            attempt_dir / "manifest.json"
        ):
            raise ValueError("case-cache pointer manifest hash mismatch")
        return attempt_dir, manifest, rows, audit
    except (KeyError, OSError, ValueError):
        return None


def _next_attempt_dir(key_root: Path) -> Path:
    existing = [
        int(path.name.rsplit("-", 1)[1])
        for path in key_root.glob("attempt-[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    ]
    attempt_number = max(existing, default=0) + 1
    attempt_dir = key_root / ("attempt-%04d" % attempt_number)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir


def _load_or_build_case(
    *,
    case_id: str,
    case_dir: Path,
    cache_root: Path,
    config_sha256: str,
    code_revision: str,
    code_sha256: str,
) -> Dict[str, Any]:
    checksums = source_checksums(case_dir)
    source_sha256 = _json_sha256(checksums)
    cache_key = _cache_key(
        case_id,
        checksums,
        config_sha256,
        code_sha256,
    )
    key_root = cache_root / _case_slug(case_id) / cache_key
    key_root.mkdir(parents=True, exist_ok=True)
    cached = _load_current_case_cache(
        key_root,
        case_id=case_id,
        cache_key=cache_key,
        checksums=checksums,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
    )
    if cached is not None:
        attempt_dir, manifest, rows, audit = cached
        return {
            "cache_hit": True,
            "attempt_dir": attempt_dir,
            "manifest": manifest,
            "rows": rows,
            "audit": audit,
            "source_checksums": checksums,
            "source_sha256": source_sha256,
        }

    attempt_dir = _next_attempt_dir(key_root)
    result = canonicalize_case(case_id, case_dir)
    rows = result["rows"]
    audit = result["audit"]
    validate_canonical_series_rows(rows, CANONICAL_DATASET_ID)
    rows_path = attempt_dir / "rows.jsonl"
    audit_path = attempt_dir / "audit.json"
    _write_jsonl(rows_path, rows)
    _write_json(audit_path, audit)
    manifest = {
        "schema_version": CASE_CACHE_SCHEMA_VERSION,
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "case_id": case_id,
        "case_dir": str(Path(case_dir).resolve()),
        "cache_key": cache_key,
        "source_checksums": checksums,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "code_revision": code_revision,
        "code_sha256": code_sha256,
        "rows_sha256": sha256_file(rows_path),
        "audit_sha256": sha256_file(audit_path),
        "row_count": len(rows),
        "series_count": int(audit["output_series_count"]),
    }
    manifest_path = attempt_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _validate_case_attempt(
        attempt_dir,
        case_id=case_id,
        cache_key=cache_key,
        checksums=checksums,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        require_completion=False,
    )
    manifest_sha256 = sha256_file(manifest_path)
    _write_json(
        attempt_dir / "COMPLETED.json",
        {
            "schema_version": CASE_CACHE_SCHEMA_VERSION,
            "status": "complete",
            "manifest_sha256": manifest_sha256,
        },
    )
    (attempt_dir / "all.done").write_text(
        manifest_sha256 + "\n", encoding="utf-8"
    )
    _write_json(
        key_root / "current.json",
        {
            "attempt": attempt_dir.name,
            "manifest_sha256": manifest_sha256,
        },
    )
    return {
        "cache_hit": False,
        "attempt_dir": attempt_dir,
        "manifest": manifest,
        "rows": rows,
        "audit": audit,
        "source_checksums": checksums,
        "source_sha256": source_sha256,
    }


def _compact_case_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain only metadata needed after one case has been persisted."""

    attempt_dir = Path(result["attempt_dir"])
    manifest = dict(result["manifest"])
    audit = dict(result["audit"])
    return {
        "cache_hit": bool(result["cache_hit"]),
        "attempt_dir": attempt_dir,
        "manifest_sha256": sha256_file(attempt_dir / "manifest.json"),
        "rows_path": attempt_dir / "rows.jsonl",
        "rows_sha256": str(manifest["rows_sha256"]),
        "row_count": int(manifest["row_count"]),
        "series_count": int(manifest["series_count"]),
        "audit_path": attempt_dir / "audit.json",
        "audit_sha256": str(manifest["audit_sha256"]),
        "audit_counts": {
            name: int(audit.get(name, 0))
            for name in _AUDIT_COUNT_NAMES
        },
        "source_checksums": dict(result["source_checksums"]),
        "source_sha256": str(result["source_sha256"]),
    }


def _load_or_build_case_compact(
    *,
    case_id: str,
    case_dir: Path,
    cache_root: Path,
    config_sha256: str,
    code_revision: str,
    code_sha256: str,
) -> Dict[str, Any]:
    result = _load_or_build_case(
        case_id=case_id,
        case_dir=case_dir,
        cache_root=cache_root,
        config_sha256=config_sha256,
        code_revision=code_revision,
        code_sha256=code_sha256,
    )
    return _compact_case_result(result)


def _emit_progress(
    callback: Optional[Callable[[Mapping[str, Any]], None]],
    *,
    phase: str,
    completed: int,
    total: int,
    current_item: str = "",
    cache_hit: Optional[bool] = None,
) -> None:
    if callback is None:
        return
    payload: Dict[str, Any] = {
        "phase": str(phase),
        "completed": int(completed),
        "total": int(total),
        "current_item": str(current_item),
    }
    if cache_hit is not None:
        payload["cache_hit"] = bool(cache_hit)
    callback(payload)


def _concatenate_case_rows(
    case_results: Mapping[str, Mapping[str, Any]],
    destination: Path,
    *,
    progress_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
) -> Tuple[str, int]:
    """Concatenate sorted per-case JSONL shards with bounded memory."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    digest = hashlib.sha256()
    case_ids = sorted(case_results)
    row_count = 0
    with temporary.open("wb") as target:
        for index, case_id in enumerate(case_ids, start=1):
            result = case_results[case_id]
            rows_path = Path(result["rows_path"])
            if sha256_file(rows_path) != str(result["rows_sha256"]):
                raise ValueError(
                    "case rows hash drifted before aggregation: %s"
                    % case_id
                )
            with rows_path.open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
            row_count += int(result["row_count"])
            _emit_progress(
                progress_callback,
                phase="aggregate_series",
                completed=index,
                total=len(case_ids),
                current_item=case_id,
            )
    temporary.replace(destination)
    return digest.hexdigest(), row_count


def _validate_run_files(
    output_root: Path,
    *,
    expected_case_ids: Optional[Set[str]] = None,
    expected_config_sha256: Optional[str] = None,
    expected_code_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    manifest = _read_json(manifest_path, "rcabench run manifest")
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("rcabench run manifest schema_version mismatch")
    if manifest.get("canonical_dataset_id") != CANONICAL_DATASET_ID:
        raise ValueError("rcabench run manifest canonical_dataset_id mismatch")
    case_ids = {str(case_id) for case_id in manifest.get("case_ids", [])}
    if not case_ids:
        raise ValueError("rcabench run manifest case_ids must not be empty")
    if expected_case_ids is not None and case_ids != set(expected_case_ids):
        raise ValueError("rcabench run manifest case coverage mismatch")
    config_sha256 = _require_sha256(
        str(manifest.get("config_sha256", "")), "manifest config_sha256"
    )
    code_sha256 = _require_sha256(
        str(manifest.get("code_sha256", "")), "manifest code_sha256"
    )
    if (
        expected_config_sha256 is not None
        and config_sha256 != expected_config_sha256
    ):
        raise ValueError("rcabench run manifest config hash mismatch")
    if expected_code_sha256 is not None and code_sha256 != expected_code_sha256:
        raise ValueError("rcabench run manifest code hash mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("rcabench run manifest artifacts must be an object")
    series_entry = artifacts.get("series")
    audit_entry = artifacts.get("audit")
    if not isinstance(series_entry, Mapping) or not isinstance(
        audit_entry, Mapping
    ):
        raise ValueError("rcabench run manifest series and audit are required")
    series_path = output_root / str(series_entry.get("path", ""))
    audit_path = output_root / str(audit_entry.get("path", ""))
    if (
        str(series_entry.get("path")) != "series.jsonl"
        or not series_path.is_file()
    ):
        raise ValueError("series artifact path is invalid")
    if str(audit_entry.get("path")) != "audit.json" or not audit_path.is_file():
        raise ValueError("audit artifact path is invalid")
    if sha256_file(series_path) != str(series_entry.get("sha256", "")):
        raise ValueError("series artifact hash mismatch")
    if sha256_file(audit_path) != str(audit_entry.get("sha256", "")):
        raise ValueError("audit artifact hash mismatch")

    audit = _read_json(audit_path, "aggregate canonical audit")
    summary = validate_canonical_series_file(
        series_path, CANONICAL_DATASET_ID
    )
    validated_audit = validate_audit(audit, CANONICAL_DATASET_ID)
    if summary["sha256"] != str(series_entry.get("sha256", "")):
        raise ValueError("series artifact hash mismatch")
    if set(summary["case_ids"]) != case_ids:
        raise ValueError("aggregate canonical series case coverage mismatch")
    if summary["row_count"] != int(series_entry.get("row_count", -1)):
        raise ValueError("series artifact row_count mismatch")
    if summary["row_count"] != int(validated_audit["output_row_count"]):
        raise ValueError("aggregate audit output_row_count mismatch")
    if summary["series_count"] != int(validated_audit["output_series_count"]):
        raise ValueError("aggregate audit output_series_count mismatch")
    if summary["case_count"] != int(validated_audit["input_case_count"]):
        raise ValueError("aggregate audit input_case_count mismatch")
    if validated_audit["config_sha256"] != config_sha256:
        raise ValueError("aggregate audit config hash mismatch")
    if validated_audit.get("code_sha256") != code_sha256:
        raise ValueError("aggregate audit code hash mismatch")
    if not validated_audit["passed"]:
        raise ValueError("aggregate audit did not pass")
    return manifest


def validate_completion(
    output_root: Path,
    *,
    expected_case_ids: Optional[Set[str]] = None,
    expected_config_sha256: Optional[str] = None,
    expected_code_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    if any(output_root.rglob("*.failed")) or any(
        output_root.rglob("*.failed.json")
    ):
        raise ValueError("rcabench run contains a failure marker")
    completion_path = output_root / "COMPLETED.json"
    all_done_path = output_root / "all.done"
    if not completion_path.is_file():
        raise ValueError("COMPLETED.json success record is missing")
    if not all_done_path.is_file():
        raise ValueError("all.done success marker is missing")
    manifest = _validate_run_files(
        output_root,
        expected_case_ids=expected_case_ids,
        expected_config_sha256=expected_config_sha256,
        expected_code_sha256=expected_code_sha256,
    )
    manifest_sha256 = sha256_file(output_root / "manifest.json")
    completion = _read_json(completion_path, "rcabench run completion")
    if completion.get("schema_version") != RUN_COMPLETION_SCHEMA_VERSION:
        raise ValueError("rcabench run completion schema_version mismatch")
    if completion.get("status") != "complete":
        raise ValueError("rcabench run completion status mismatch")
    if completion.get("manifest_sha256") != manifest_sha256:
        raise ValueError("rcabench run completion manifest hash mismatch")
    if all_done_path.read_text(encoding="utf-8").strip() != manifest_sha256:
        raise ValueError("rcabench run all.done manifest hash mismatch")
    return manifest


def _aggregate_audit(
    *,
    cases: Mapping[str, Path],
    case_results: Mapping[str, Mapping[str, Any]],
    config_sha256: str,
    code_revision: str,
    code_sha256: str,
    environment_manifest_path: str,
    environment_manifest_sha256: str,
) -> Dict[str, Any]:
    totals = {
        name: sum(
            int(
                case_results[case_id]["audit_counts"].get(
                    name, 0
                )
            )
            for case_id in sorted(cases)
        )
        for name in _AUDIT_COUNT_NAMES
    }
    flattened_checksums = {
        "%s/%s" % (case_id, filename): digest
        for case_id in sorted(cases)
        for filename, digest in sorted(
            case_results[case_id]["source_checksums"].items()
        )
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "input_file_count": totals["input_file_count"],
        "input_case_count": len(cases),
        "admitted_case_count": len(cases),
        "excluded_case_count": 0,
        "exact_duplicates_removed": totals["exact_duplicates_removed"],
        "conflicting_series_count": totals["conflicting_series_count"],
        "non_finite_row_count": totals["non_finite_row_count"],
        "unresolved_service_row_count": totals["unresolved_service_row_count"],
        "output_series_count": totals["output_series_count"],
        "output_row_count": totals["output_row_count"],
        "source_checksums": flattened_checksums,
        "code_revision": code_revision,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "environment_manifest_path": environment_manifest_path,
        "environment_manifest_sha256": environment_manifest_sha256,
        "passed": True,
        "cache_hits": sum(
            int(bool(result["cache_hit"])) for result in case_results.values()
        ),
        "cache_misses": sum(
            int(not bool(result["cache_hit"])) for result in case_results.values()
        ),
        "existing_file_count": totals["existing_file_count"],
        "empty_file_count": totals["empty_file_count"],
        "missing_file_count": totals["missing_file_count"],
        "conflicting_observation_count": totals[
            "conflicting_observation_count"
        ],
        "case_audit_refs": {
            case_id: {
                "path": str(
                    Path(case_results[case_id]["audit_path"]).resolve()
                ),
                "sha256": str(
                    case_results[case_id]["audit_sha256"]
                ),
            }
            for case_id in sorted(cases)
        },
    }


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output_root must be new or empty: %s" % output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def canonicalize_dataset(
    cases: Mapping[str, Path],
    *,
    output_root: Path,
    cache_root: Path,
    config: Mapping[str, Any],
    code_revision: str,
    code_sha256: str,
    environment_manifest_path: str,
    environment_manifest_sha256: str,
    progress_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
    case_workers: int = 1,
) -> Dict[str, Any]:
    """Canonicalize a case map using hash-valid caches and aggregate safely."""

    if not cases:
        raise ValueError("cases must not be empty")
    canonical_cases = {str(case_id): Path(path) for case_id, path in cases.items()}
    invalid_case_ids = [
        case_id
        for case_id in canonical_cases
        if not case_id.startswith(CANONICAL_DATASET_ID + "::")
    ]
    if invalid_case_ids:
        raise ValueError("non-canonical rcabench case_id: %s" % invalid_case_ids[0])
    code_sha256 = _require_sha256(code_sha256, "code_sha256")
    environment_manifest_sha256 = _require_sha256(
        environment_manifest_sha256, "environment_manifest_sha256"
    )
    if not str(code_revision).strip():
        raise ValueError("code_revision must be non-empty")
    if not str(environment_manifest_path).strip():
        raise ValueError("environment_manifest_path must be non-empty")
    workers = int(case_workers)
    if workers < 1:
        raise ValueError("case_workers must be at least 1")
    config_payload = dict(config)
    config_sha256 = _json_sha256(config_payload)
    output_root = Path(output_root)
    cache_root = Path(cache_root)
    _prepare_output_root(output_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "config.snapshot.json",
        {
            "config": config_payload,
            "config_sha256": config_sha256,
            "code_revision": code_revision,
            "code_sha256": code_sha256,
        },
    )

    results: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    completed_cases = 0

    def record_failure(
        case_id: str, case_dir: Path, exc: Exception
    ) -> None:
        case_dir = canonical_cases[case_id]
        checksums = source_checksums(case_dir)
        failure = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "canonical_dataset_id": CANONICAL_DATASET_ID,
            "case_id": case_id,
            "case_dir": str(case_dir.resolve()),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "source_checksums": checksums,
            "source_sha256": _json_sha256(checksums),
            "config_sha256": config_sha256,
            "code_revision": code_revision,
            "code_sha256": code_sha256,
        }
        failures.append(failure)
        _write_json(
            output_root
            / "failures"
            / ("%s.failed.json" % _case_slug(case_id)),
            failure,
        )

    sorted_case_ids = sorted(canonical_cases)
    if workers == 1:
        for case_id in sorted_case_ids:
            case_dir = canonical_cases[case_id]
            try:
                result = _load_or_build_case_compact(
                    case_id=case_id,
                    case_dir=case_dir,
                    cache_root=cache_root,
                    config_sha256=config_sha256,
                    code_revision=code_revision,
                    code_sha256=code_sha256,
                )
                results[case_id] = result
                completed_cases += 1
                _emit_progress(
                    progress_callback,
                    phase="canonicalize_cases",
                    completed=completed_cases,
                    total=len(sorted_case_ids),
                    current_item=case_id,
                    cache_hit=bool(result["cache_hit"]),
                )
            except Exception as exc:
                record_failure(case_id, case_dir, exc)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            future_cases = {
                pool.submit(
                    _load_or_build_case_compact,
                    case_id=case_id,
                    case_dir=canonical_cases[case_id],
                    cache_root=cache_root,
                    config_sha256=config_sha256,
                    code_revision=code_revision,
                    code_sha256=code_sha256,
                ): case_id
                for case_id in sorted_case_ids
            }
            for future in as_completed(future_cases):
                case_id = future_cases[future]
                case_dir = canonical_cases[case_id]
                try:
                    result = future.result()
                    results[case_id] = result
                    completed_cases += 1
                    _emit_progress(
                        progress_callback,
                        phase="canonicalize_cases",
                        completed=completed_cases,
                        total=len(sorted_case_ids),
                        current_item=case_id,
                        cache_hit=bool(result["cache_hit"]),
                    )
                except Exception as exc:
                    record_failure(case_id, case_dir, exc)

    if failures:
        run_failure = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "canonical_dataset_id": CANONICAL_DATASET_ID,
            "status": "failed",
            "failed_case_ids": [failure["case_id"] for failure in failures],
            "failure_count": len(failures),
            "config_sha256": config_sha256,
            "code_revision": code_revision,
            "code_sha256": code_sha256,
        }
        _write_json(output_root / "run.failed.json", run_failure)
        raise DatasetCanonicalizationError(
            "rcabench canonicalization failed for: %s"
            % ", ".join(run_failure["failed_case_ids"])
        )

    audit = _aggregate_audit(
        cases=canonical_cases,
        case_results=results,
        config_sha256=config_sha256,
        code_revision=code_revision,
        code_sha256=code_sha256,
        environment_manifest_path=environment_manifest_path,
        environment_manifest_sha256=environment_manifest_sha256,
    )
    series_path = output_root / "series.jsonl"
    audit_path = output_root / "audit.json"
    series_sha256, series_row_count = _concatenate_case_rows(
        results,
        series_path,
        progress_callback=progress_callback,
    )
    if series_row_count != int(audit["output_row_count"]):
        raise DatasetCanonicalizationError(
            "streamed series row count does not match aggregate audit"
        )
    _write_json(audit_path, audit)
    flattened_source_checksums = dict(audit["source_checksums"])
    source_sha256 = _json_sha256(flattened_source_checksums)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "canonical_dataset_id": CANONICAL_DATASET_ID,
        "case_ids": sorted(canonical_cases),
        "source_sha256": source_sha256,
        "source_checksums": flattened_source_checksums,
        "config_sha256": config_sha256,
        "code_revision": code_revision,
        "code_sha256": code_sha256,
        "environment_manifest_path": environment_manifest_path,
        "environment_manifest_sha256": environment_manifest_sha256,
        "artifacts": {
            "series": {
                "path": "series.jsonl",
                "schema_version": SERIES_SCHEMA_VERSION,
                "sha256": series_sha256,
                "row_count": series_row_count,
            },
            "audit": {
                "path": "audit.json",
                "schema_version": AUDIT_SCHEMA_VERSION,
                "sha256": sha256_file(audit_path),
                "row_count": 1,
            },
        },
        "case_cache_entries": {
            case_id: {
                "attempt_dir": str(results[case_id]["attempt_dir"].resolve()),
                "manifest_sha256": str(
                    results[case_id]["manifest_sha256"]
                ),
                "cache_hit": bool(results[case_id]["cache_hit"]),
                "source_sha256": results[case_id]["source_sha256"],
            }
            for case_id in sorted(results)
        },
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)

    try:
        _emit_progress(
            progress_callback,
            phase="validate_aggregate",
            completed=0,
            total=1,
            current_item=str(series_path),
        )
        manifest_sha256 = sha256_file(manifest_path)
        _write_json(
            output_root / "COMPLETED.json",
            {
                "schema_version": RUN_COMPLETION_SCHEMA_VERSION,
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "status": "complete",
                "manifest_sha256": manifest_sha256,
            },
        )
        (output_root / "all.done").write_text(
            manifest_sha256 + "\n", encoding="utf-8"
        )
        validate_completion(
            output_root,
            expected_case_ids=set(canonical_cases),
            expected_config_sha256=config_sha256,
            expected_code_sha256=code_sha256,
        )
        _emit_progress(
            progress_callback,
            phase="complete",
            completed=1,
            total=1,
            current_item=str(output_root),
        )
    except Exception as exc:
        for marker_name in ("COMPLETED.json", "all.done"):
            marker_path = output_root / marker_name
            if marker_path.exists():
                marker_path.unlink()
        _write_json(
            output_root / "run.failed.json",
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "canonical_dataset_id": CANONICAL_DATASET_ID,
                "status": "failed_validation",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "config_sha256": config_sha256,
                "code_revision": code_revision,
                "code_sha256": code_sha256,
            },
        )
        raise DatasetCanonicalizationError(
            "rcabench aggregate validation failed: %s" % exc
        )

    return {
        "manifest": manifest,
        "output_root": str(output_root.resolve()),
        "cache_hits": int(audit["cache_hits"]),
        "cache_misses": int(audit["cache_misses"]),
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "code_revision": code_revision,
        "code_sha256": code_sha256,
    }


__all__ = [
    "CASE_CACHE_SCHEMA_VERSION",
    "RUN_COMPLETION_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "DatasetCanonicalizationError",
    "canonicalize_dataset",
    "sha256_file",
    "source_checksums",
    "validate_completion",
]
