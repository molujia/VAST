"""Durable ordered acquisition plans, distinct from the paid-label ledger.

The immutable plan is checkpoint-bound before the first answer. LabelStore's
write-ahead ledger remains authoritative for payment and answers; trace cursors
may lag it by one persisted transition if the second atomic write is interrupted.
Digests detect accidental modification, not a malicious rewrite of every file.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..acquisition.propcover import seeded_case_priority


SCHEMA = "vast-v3-ordered-query-trace/1"
PROGRESS_FIELDS = frozenset({"accessed_ids", "completed_ids", "complete", "plan_hash"})


def _plan_hash(record):
    immutable = {key: value for key, value in record.items() if key not in PROGRESS_FIELDS}
    return hashlib.sha256(json.dumps(immutable, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class OrderedQueryTrace:
    def __init__(self, path, *, identity, pool_ids, schedule, stages, seed):
        self.path = Path(path)
        self.identity = dict(identity)
        self.pool_ids = sorted(pool_ids)
        self.schedule, self.stages, self.seed = schedule, stages, seed
        self.rounds = []
        self._recovered_from_checkpoint = False

    @property
    def plan_hashes(self):
        return [record["plan_hash"] for record in self.rounds]

    @property
    def query_order(self):
        return [cid for record in self.rounds for cid in record["completed_ids"]]

    def state_dict(self):
        return {"schema": SCHEMA, "identity": self.identity, "pool_ids": self.pool_ids,
                "budget": self.schedule.budget, "rounds": deepcopy(self.rounds)}

    def save(self):
        descriptor, name = tempfile.mkstemp(prefix=".query-trace-", suffix=".tmp", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.state_dict(), handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load_state(self, stored):
        if (stored["schema"] != SCHEMA or stored["identity"] != self.identity
                or stored["pool_ids"] != self.pool_ids or stored["budget"] != self.schedule.budget):
            raise ValueError("query-trace identity mismatch")
        self.rounds = deepcopy(stored["rounds"])
        self._validate_plans()

    def load(self, *, checkpoint_trace):
        """Recover a missing/one-plan-behind sidecar only from exact saved plans.

        Checkpoint precedes sidecar publication. Neither sorted L nor the label
        ledger is ever used to fabricate a missing immutable acquisition plan.
        """
        if checkpoint_trace is None:
            raise ValueError("checkpoint lacks ordered query-trace; historical order cannot be invented")
        try:
            self._load_state(checkpoint_trace)
            saved_rounds, saved_hashes = deepcopy(self.rounds), self.plan_hashes
            if not self.path.is_file():
                self._recovered_from_checkpoint = True
                return
            self._load_state(json.loads(self.path.read_text(encoding="utf-8")))
            if (self.plan_hashes != saved_hashes[:len(self.rounds)]
                    or not 0 <= len(saved_rounds) - len(self.rounds) <= 1):
                raise ValueError("query-trace does not match checkpoint query plan binding")
            if len(self.rounds) < len(saved_rounds):
                if saved_rounds[-1]["accessed_ids"]:
                    raise ValueError("query-trace lost an already accessed query plan")
                self.rounds.append(saved_rounds[-1])
                self._validate_plans()
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid ordered query-trace") from exc

    def _validate_plans(self):
        expected = [(index, stage) for index, stage in enumerate(self.stages) if stage["query_count"]]
        if not isinstance(self.rounds, list) or len(self.rounds) > len(expected):
            raise ValueError("query-trace has unexpected rounds")
        previous = []
        for record, (index, stage) in zip(self.rounds, expected):
            if record["plan_hash"] != _plan_hash(record):
                raise ValueError("query-trace plan hash mismatch")
            version = sum(item["epochs"] for item in self.stages[:index]) * self.schedule.update_slots(len(self.pool_ids))
            if (record["stage_index"] != index or record["round_index"] != stage["round_index"]
                    or record["stage"] != stage["stage"]
                    or record["encoder_version"] != version or record["update_version"] != version
                    or record["labeled_ids"] != sorted(previous) or record["spent_before"] != len(previous)
                    or record["remaining_budget"] != self.schedule.budget - len(previous)
                    or record["query_count"] != stage["query_count"]):
                raise ValueError("query-trace round/encoder/budget identity mismatch")
            selected = record["selected_ids"]
            selector = record["selector"]
            if (not isinstance(selected, list) or len(selected) != stage["query_count"]
                    or len(set(selected)) != len(selected)
                    or not set(selected) <= set(self.pool_ids) - set(previous)
                    or selector["name"] != "PropCover" or selector["seed"] != self.seed
                    or selector["implementation"] != "vast_v3.acquisition.propcover.PropCover/1"
                    or selector["priority"] != list(seeded_case_priority(self.pool_ids, self.seed))
                    or len(selector["gains"]) != len(selected)):
                raise ValueError("query-trace selection/selector mismatch")
            accessed, completed = record["accessed_ids"], record["completed_ids"]
            if (not isinstance(accessed, list) or not isinstance(completed, list)
                    or accessed != selected[:len(accessed)] or completed != selected[:len(completed)]
                    or not 0 <= len(accessed) - len(completed) <= 1
                    or type(record["complete"]) is not bool
                    or record["complete"] != (completed == selected)):
                raise ValueError("query-trace completion is not an ordered prefix")
            if previous and not self.rounds[stage["round_index"] - 1]["complete"]:
                raise ValueError("query-trace has a new plan before the previous round completed")
            previous.extend(selected)

    def append(self, *, stage_index, update_version, labeled_ids, spent, selection, selector):
        stage = self.stages[stage_index]
        if selection.encoder_version != str(update_version):
            raise ValueError("query plan must use the current encoder version")
        record = {"stage_index": stage_index, "stage": stage["stage"], "round_index": stage["round_index"],
                  "encoder_version": update_version, "update_version": update_version,
                  "labeled_ids": sorted(labeled_ids), "spent_before": spent,
                  "remaining_budget": self.schedule.budget - spent, "query_count": stage["query_count"],
                  "selected_ids": list(selection.case_ids),
                  "selector": {"name": "PropCover", "implementation": "vast_v3.acquisition.propcover.PropCover/1",
                               "seed": self.seed, "priority": list(selector.priority),
                               "gains": list(selection.gains), "base_coverage": selection.base_coverage,
                               "final_coverage": selection.final_coverage,
                               "reference_occurrences": selection.reference_occurrences,
                               "cache_stats": selection.cache_stats},
                  "accessed_ids": [], "completed_ids": [], "complete": False}
        record["plan_hash"] = _plan_hash(record)
        self.rounds.append(record)
        self._validate_plans()
        return record

    def record_for_stage(self, stage_index):
        return next((record for record in self.rounds if record["stage_index"] == stage_index), None)

    def reconcile(self, ledger, *, binding, checkpoint_binding=None, loop_state=None):
        """Validate ordered payment/answer prefixes, then advance only lagging cursors."""
        self._validate_plans()
        if checkpoint_binding is not None and checkpoint_binding != self.plan_hashes:
            raise ValueError("query-trace does not match checkpoint query plan binding")
        if not isinstance(binding, list) or binding != self.plan_hashes[:len(binding)]:
            raise ValueError("query-trace does not match label ledger query plan binding")
        selected = [cid for record in self.rounds for cid in record["selected_ids"]]
        paid, revealed = set(ledger["accessed"]), set(ledger["revealed"])
        bound_ids = {cid for record in self.rounds[:len(binding)] for cid in record["selected_ids"]}
        if (not paid <= bound_ids or paid != set(selected[:len(paid)])
                or revealed != set(selected[:len(revealed)]) or not revealed <= paid
                or not 0 <= len(paid) - len(revealed) <= 1):
            raise ValueError("query-trace and label ledger do not share an ordered access prefix")
        old_paid = sum(len(record["accessed_ids"]) for record in self.rounds)
        old_done = sum(len(record["completed_ids"]) for record in self.rounds)
        max_lag = len(selected) if self._recovered_from_checkpoint else 1
        if not 0 <= len(paid) - old_paid <= max_lag or not 0 <= len(revealed) - old_done <= max_lag:
            raise ValueError("query-trace completion conflicts with durable label ledger")
        if loop_state is not None:
            completed_stages = loop_state["query_stages"]
            expected_stages = [index for index, stage in enumerate(self.stages)
                               if stage["query_count"] and index < loop_state["stage_index"]]
            current = loop_state["stage_index"]
            if current in completed_stages:
                expected_stages.append(current)
            if completed_stages != expected_stages:
                raise ValueError("query-trace checkpoint query stage history mismatch")
            for record in self.rounds:
                if record["stage_index"] > current:
                    raise ValueError("query-trace plan is ahead of checkpoint stage")
            for index in completed_stages:
                record = self.record_for_stage(index)
                if record is None or not set(record["selected_ids"]) <= revealed:
                    raise ValueError("query-trace lacks checkpoint-completed answers")
        for record in self.rounds:
            record["accessed_ids"] = [cid for cid in record["selected_ids"] if cid in paid]
            record["completed_ids"] = [cid for cid in record["selected_ids"] if cid in revealed]
            record["complete"] = record["completed_ids"] == record["selected_ids"]
        self._recovered_from_checkpoint = False

