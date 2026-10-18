"""Append-only, hash-chained demo-search evidence. No training or transfer code.

An externally retained ``head_hash`` detects truncation on resume. A local hash
chain is not a signature against an actor able to rewrite the entire chain; keep
the manifest and head receipt outside the ledger when stronger integrity is needed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .metrics import _ids, evaluate_case


def _bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamps must be timezone-aware ISO strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamps must include their timezone")
    return parsed


@dataclass(frozen=True)
class SearchManifest:
    dataset: str
    search_run_id: str
    eligible_case_ids: tuple[str, ...]
    excluded_semantic_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    case_manifest_hash: str
    code_revision: str
    config_hash: str
    preprocessing_hash: str
    checkpoint_hash: str
    catalog_hash: str
    data_hash: str
    seed: int = 42

    def __post_init__(self):
        _ids([self.dataset], "dataset")
        _ids([self.search_run_id], "search_run_id")
        _ids([self.code_revision], "code_revision")
        for field in ("eligible_case_ids", "excluded_semantic_ids", "candidate_ids"):
            values = _ids(getattr(self, field), field, allow_empty=field != "candidate_ids")
            object.__setattr__(self, field, values)
        if set(self.eligible_case_ids) & set(self.excluded_semantic_ids):
            raise ValueError("The independent semantic-evaluation set is not search-eligible")
        if type(self.seed) is not int or self.seed != 42:
            raise ValueError("The approved quick-start order uses frozen seed 42")
        for field in ("case_manifest_hash", "config_hash", "preprocessing_hash",
                      "checkpoint_hash", "catalog_hash", "data_hash"):
            if not isinstance(getattr(self, field), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict:
        return json.loads(_bytes(asdict(self)))


class SearchLedger:
    """Single-writer persistent ledger; stale/concurrent writers are rejected.

The supplied manifest must already be produced without looking at predictions.
Its declared order is preserved exactly; this class cannot prove that provenance.
Each append describes an actual completed attempt, including failures. Replays
and retries never replace earlier attempts. Selection/rejection are extra events.
"""

    def __init__(self, path: Path, manifest: SearchManifest, *, resume: bool = False,
                 expected_head_hash: str | None = None):
        self.path = Path(path)
        self.manifest = manifest
        self._records = []
        self._raw = b""
        if resume:
            self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            envelope = self._envelope({"event": "manifest", "schema_version": "vast-v3-demo-ledger-v1",
                                       "manifest": manifest.to_dict()})
            raw = _bytes(envelope) + b"\n"
            with self.path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            self._raw = raw
            self._records.append(envelope)
        if expected_head_hash is not None and self.head_hash != expected_head_hash:
            raise ValueError("Ledger head differs from the externally retained receipt")

    @property
    def head_hash(self) -> str:
        return self._records[-1]["record_sha256"] if self._records else "0" * 64

    @property
    def attempts(self) -> tuple[dict, ...]:
        # Return copies: callers cannot mutate the in-memory source of future hashes.
        return tuple(json.loads(_bytes(row["payload"])) for row in self._records
                     if row["payload"]["event"] == "attempt")

    def _envelope(self, payload: dict) -> dict:
        body = {"sequence": len(self._records), "previous_sha256": self.head_hash,
                "payload": payload}
        return {**body, "record_sha256": _hash(body)}

    def _load(self):
        raw = self.path.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("Empty or partial ledger; preserve it for investigation")
        for line in raw.splitlines():
            try:
                envelope = json.loads(line)
                body = {key: envelope[key] for key in ("sequence", "previous_sha256", "payload")}
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("Malformed ledger record") from exc
            if envelope != {**body, "record_sha256": _hash(body)}:
                raise ValueError("Ledger record was altered")
            if body["sequence"] != len(self._records) or body["previous_sha256"] != self.head_hash:
                raise ValueError("Broken ledger sequence/hash chain")
            payload = body["payload"]
            if not self._records:
                expected = {"event": "manifest", "schema_version": "vast-v3-demo-ledger-v1",
                            "manifest": self.manifest.to_dict()}
                if payload != expected:
                    raise ValueError("Frozen manifest identity changed")
            elif payload.get("event") == "attempt":
                expected = self._make_attempt(
                    case_id=payload.get("case_id"), stage=payload.get("stage"),
                    status=payload.get("status"), started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"), scores=payload.get("candidate_scores"),
                    roots=payload.get("true_roots") or None, repeat_of=payload.get("repeat_of"),
                    error_reason=payload.get("error_reason"), evidence_paths=payload.get("evidence_paths", ()))
                if payload != expected:
                    raise ValueError("Attempt contents disagree with canonical evaluation/identity")
            elif payload.get("event") in ("selected", "rejected"):
                self._validate_decision(payload["event"], payload.get("case_id"),
                                        payload.get("reason"), payload.get("evidence_paths", ()))
                _time(payload.get("recorded_at"))
            else:
                raise ValueError("Unknown ledger event")
            self._records.append(envelope)
        self._raw = raw

    @contextmanager
    def _writer_lock(self):
        lock_path = self.path.with_name(self.path.name + ".lock")
        try:
            handle = lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError("Ledger is locked; never remove another writer's lock automatically") from exc
        try:
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            handle.close()
            lock_path.unlink()

    def _append(self, payload: dict) -> dict:
        with self._writer_lock():
            if self.path.read_bytes() != self._raw:
                raise ValueError("Ledger changed since this writer loaded it; resume afresh")
            envelope = self._envelope(payload)
            raw = _bytes(envelope) + b"\n"
            with self.path.open("ab") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            self._raw += raw
            self._records.append(envelope)
        return json.loads(_bytes(payload))

    def _search_attempts(self) -> list[dict]:
        return [row for row in self.attempts if row["stage"] != "export_replay"]

    def _make_attempt(self, *, case_id: str, stage: str, status: str,
                      started_at: str, finished_at: str, scores=None, roots=None,
                      repeat_of: int | None = None, error_reason: str | None = None,
                      evidence_paths=()) -> dict:
        if case_id not in self.manifest.eligible_case_ids:
            raise ValueError("Case is outside the frozen eligible pool")
        if stage not in {"search", "raw_retry", "export_replay"} or status not in {"scored", "error"}:
            raise ValueError("Unknown attempt stage/status")
        started, finished = _time(started_at), _time(finished_at)
        if finished < started:
            raise ValueError("Attempt finished before it started")
        evidence = list(_ids(evidence_paths, "evidence paths", allow_empty=True))
        attempts = list(self.attempts)
        prior_search = self._search_attempts()
        prior_case = [row for row in prior_search if row["case_id"] == case_id]
        if stage == "search":
            seen = list(dict.fromkeys(row["case_id"] for row in prior_search))
            if prior_case or repeat_of is not None or len(seen) >= len(self.manifest.eligible_case_ids):
                raise ValueError("Repeated cases require explicit raw_retry links")
            if self.manifest.eligible_case_ids[len(seen)] != case_id:
                raise ValueError("Search order differs from the frozen eligible manifest")
        else:
            if type(repeat_of) is not int or not 1 <= repeat_of <= len(attempts):
                raise ValueError("A retry/replay requires a valid repeat_of attempt index")
            previous = attempts[repeat_of - 1]
            if previous["case_id"] != case_id:
                raise ValueError("Retry/replay link belongs to a different case")
            if stage == "raw_retry":
                if not prior_case or previous != prior_case[-1] or previous["status"] != "error":
                    raise ValueError("Raw retry must link to this case's latest failed raw attempt")
            elif self.summary()["selected_case_id"] != case_id:
                raise ValueError("Only the selected raw TOP1 case may enter export replay")
        if status == "scored":
            if error_reason:
                raise ValueError("A scored attempt cannot have an execution error")
            result = evaluate_case(case_id, self.manifest.candidate_ids, scores, roots or ())
            metric = result.to_dict()
            numeric_scores = dict(zip(result.ranking, result.scores))
        else:
            if not isinstance(error_reason, str) or not error_reason.strip():
                raise ValueError("Error attempts require an explicit reason")
            if scores is not None or roots is not None:
                raise ValueError("Error attempts must not masquerade as scored predictions")
            metric = {"ranking": [], "true_roots": [], "root_ranks": {}, "first_root_rank": None,
                      **{f"hit_at_{k}": None for k in range(1, 6)}}
            numeric_scores = None
        first_scored = {}
        for row in prior_search:
            if row["status"] == "scored":
                first_scored.setdefault(row["case_id"], row)
        if status == "scored" and stage != "export_replay":
            first_scored.setdefault(case_id, metric)
        cumulative = [sum(row[f"hit_at_{k}"] for row in first_scored.values()) for k in range(1, 6)]
        identity = self.manifest.to_dict()
        for field in ("eligible_case_ids", "excluded_semantic_ids", "candidate_ids", "seed"):
            identity.pop(field)
        return {"event": "attempt", **identity, "attempt_index": len(attempts) + 1,
                "case_id": case_id, "stage": stage, "repeat_of": repeat_of,
                "started_at": started_at, "finished_at": finished_at,
                "duration_seconds": (finished - started).total_seconds(), "status": status,
                "candidate_scores": numeric_scores, "predicted_candidates": metric["ranking"],
                "predicted_top5": metric["ranking"][:5], "true_roots": metric["true_roots"],
                "root_ranks": metric["root_ranks"], "first_root_rank": metric["first_root_rank"],
                **{f"hit_at_{k}": metric[f"hit_at_{k}"] for k in range(1, 6)},
                "cumulative_hits": cumulative, "cumulative_unique_scored_cases": len(first_scored),
                "error_reason": error_reason, "selection_status": "unreviewed",
                "rejection_reason": None, "evidence_paths": evidence,
                "tie_break": "descending_score_then_ascending_canonical_id"}

    def append_attempt(self, *, case_id: str, stage: str, status: str,
                       started_at: str, finished_at: str, scores=None, roots=None,
                       repeat_of: int | None = None, error_reason: str | None = None,
                       evidence_paths=()) -> dict:
        return self._append(self._make_attempt(
            case_id=case_id, stage=stage, status=status, started_at=started_at,
            finished_at=finished_at, scores=scores, roots=roots, repeat_of=repeat_of,
            error_reason=error_reason, evidence_paths=evidence_paths))

    def _validate_decision(self, event, case_id, reason, evidence_paths):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Selection/rejection needs an explicit evidence-based reason")
        _ids(evidence_paths, "decision evidence paths")
        successes = [row for row in self._search_attempts()
                     if row["case_id"] == case_id and row["status"] == "scored" and row["hit_at_1"] == 1]
        if not successes:
            raise ValueError("Selection/rejection requires an actual raw TOP1 result")
        decisions = [row["payload"] for row in self._records if row["payload"]["event"] in ("selected", "rejected")]
        if any(row["case_id"] == case_id and row["event"] == "rejected" for row in decisions):
            raise ValueError("A rejected case remains rejected; do not rewrite the search history")
        selected = self.summary()["selected_case_id"]
        if event == "selected" and selected is not None:
            raise ValueError("Reject the current selection explicitly before selecting another case")
        if event == "selected":
            rejected = {row["case_id"] for row in decisions if row["event"] == "rejected"}
            earlier = [row for row in self._search_attempts()
                       if row["attempt_index"] < successes[0]["attempt_index"]
                       and row["status"] == "scored" and row["hit_at_1"] == 1
                       and row["case_id"] not in rejected]
            if earlier:
                raise ValueError("An earlier TOP1 requires a recorded evidence-based rejection first")

    def _decision(self, event, case_id, reason, evidence_paths):
        self._validate_decision(event, case_id, reason, evidence_paths)
        return self._append({"event": event, "case_id": case_id, "reason": reason,
                             "evidence_paths": list(evidence_paths),
                             "recorded_at": datetime.now(timezone.utc).isoformat()})

    def select_case(self, case_id: str, *, reason: str, evidence_paths) -> dict:
        return self._decision("selected", case_id, reason, evidence_paths)

    def reject_case(self, case_id: str, *, reason: str, evidence_paths) -> dict:
        return self._decision("rejected", case_id, reason, evidence_paths)

    def summary(self) -> dict:
        attempts = list(self.attempts)
        search = [row for row in attempts if row["stage"] != "export_replay"]
        scored = [row for row in search if row["status"] == "scored"]
        first = next((row for row in scored if row["hit_at_1"]), None)
        selected = None
        for envelope in self._records:
            row = envelope["payload"]
            if row["event"] == "selected":
                selected = row["case_id"]
            elif row["event"] == "rejected" and selected == row["case_id"]:
                selected = None
        return {"dataset": self.manifest.dataset, "search_run_id": self.manifest.search_run_id,
                "total_attempts": len(attempts), "search_attempts": len(search),
                "attempted_unique_cases": len({row["case_id"] for row in search}),
                "scored_unique_cases": len({row["case_id"] for row in scored}),
                "scored_attempts": sum(row["status"] == "scored" for row in attempts),
                "errors": sum(row["status"] == "error" for row in attempts),
                "retries": sum(row["stage"] == "raw_retry" for row in attempts),
                "export_replay_attempts": sum(row["stage"] == "export_replay" for row in attempts),
                "first_top1_index": first["attempt_index"] if first else None,
                "first_top1_case_id": first["case_id"] if first else None,
                "selected_case_id": selected,
                "eligible_pool_exhausted": len({row["case_id"] for row in search}) == len(self.manifest.eligible_case_ids),
                "benchmark_estimate": False,
                "demo_requirement_met": False, "transfer_authorized": False,
                "completion_note": "Selection alone is not package replay equivalence or transfer approval",
                "head_hash": self.head_hash}

    def export_reports(self, directory: Path) -> tuple[Path, Path]:
        """Render every attempt and all decisions; never rewrite ledger records."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path, markdown_path = directory / "attempts.csv", directory / "summary.md"
        if self.path.resolve() in (csv_path.resolve(), markdown_path.resolve()):
            raise ValueError("Reports must not overwrite the ledger")
        fields = ["dataset", "search_run_id", "attempt_index", "case_id", "case_manifest_hash",
                  "stage", "repeat_of", "status", "first_root_rank", "predicted_candidates",
                  "predicted_top5", "true_roots", "root_ranks", *[f"hit_at_{k}" for k in range(1, 6)],
                  "cumulative_hits", "cumulative_unique_scored_cases", "error_reason", "started_at",
                  "finished_at", "duration_seconds", "code_revision", "config_hash", "preprocessing_hash",
                  "checkpoint_hash", "catalog_hash", "data_hash", "tie_break", "evidence_paths",
                  "selection_status", "rejection_reason", "decision_reason", "decision_evidence_paths"]
        decisions = {row["payload"]["case_id"]: row["payload"] for row in self._records
                     if row["payload"]["event"] in ("selected", "rejected")}
        report_rows = []
        for attempt in self.attempts:
            decision = decisions.get(attempt["case_id"])
            report_rows.append({**attempt,
                "selection_status": decision["event"] if decision else "unreviewed",
                "rejection_reason": decision["reason"] if decision and decision["event"] == "rejected" else None,
                "decision_reason": decision["reason"] if decision else None,
                "decision_evidence_paths": decision["evidence_paths"] if decision else []})
        def cell(value):
            if isinstance(value, (list, dict)):
                return _bytes(value).decode("utf-8")
            return "" if value is None else str(value)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report_rows:
                values = {field: cell(row[field]) for field in fields}
                writer.writerow({key: "'" + value if value.startswith(("=", "+", "-", "@")) else value
                                 for key, value in values.items()})
        summary = self.summary()
        lines = ["# Quick-start search audit", "",
                 "This stopped search is not an unbiased benchmark performance estimate.", "",
                 "Selection alone does not establish replay equivalence or authorize transfer.", "",
                 "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", "",
                 "## Every executed attempt", "", "| Attempt | Case | Stage | Status | Root ranks | First root rank | Top5 | Hits 1-5 | Selection | Reason/error |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        def md(value):
            return cell(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
        for row in report_rows:
            values = [row["attempt_index"], row["case_id"], row["stage"], row["status"],
                      row["root_ranks"], row["first_root_rank"], row["predicted_top5"],
                      [row[f"hit_at_{k}"] for k in range(1, 6)], row["selection_status"],
                      row["error_reason"] or row["decision_reason"]]
            lines.append("| " + " | ".join(md(value) for value in values) + " |")
        lines += ["", "## Selection and rejection events", ""]
        for envelope in self._records:
            row = envelope["payload"]
            if row["event"] in ("selected", "rejected"):
                lines.append(f"- {md(row['case_id'])}: {row['event']}; {md(row['reason'])}; evidence {md(row['evidence_paths'])}")
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return csv_path, markdown_path
