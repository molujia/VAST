"""Training-run progress and full-schedule acceptance, separate from learning."""
from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import tempfile
import time

import torch

from ..data.repository import file_sha256
from ..data.serialization import strict_json_loads
from .checkpoint import inspect_checkpoint
from .loop import Trainer
from .query_trace import OrderedQueryTrace


RUN_SCHEMA = 'vast-v3-demo-training-run/1'
RUN_PURPOSE = 'budgeted-training-not-smoke'


def read_json(path):
    return strict_json_loads(Path(path).read_bytes())


def atomic_json(path, payload):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_json(path, payload):
    """A matching completed preparation artifact is immutable across resumes."""
    path = Path(path)
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f'frozen artifact identity mismatch: {path.name}')
    else:
        atomic_json(path, payload)


class RunProgress:
    def __init__(self, directory, *, resume=False):
        self.directory = Path(directory)
        marker = {'schema': RUN_SCHEMA, 'purpose': RUN_PURPOSE}
        if resume:
            if not self.directory.is_dir() or not (self.directory / 'run.json').is_file():
                raise ValueError('resume requires an existing versioned training run')
            if read_json(self.directory / 'run.json') != marker:
                raise ValueError('run identity/purpose differs; smoke is not training initialization')
        else:
            try:
                self.directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError('existing attempt requires explicit resume; choose a new run directory') from exc
            atomic_json(self.directory / 'run.json', marker)
        self.started = time.perf_counter()

    def update(self, phase, **values):
        status = {'schema': 'vast-v3-training-progress/1', 'purpose': RUN_PURPOSE,
                  'phase': phase, 'complete': False, 'unix_time': time.time(),
                  'invocation_elapsed_seconds': time.perf_counter() - self.started, **values}
        # Append log before publishing the current progress pointer.
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(status, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(self.directory / 'status.json', status)
        print(json.dumps(status, sort_keys=True, allow_nan=False), flush=True)
        return status


class ProgressTrainer(Trainer):
    """Observe existing Trainer operations; never add forwards, queries or RNG."""
    def __init__(self, *args, reporter, policy_guard=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reporter = reporter
        self.policy_guard = policy_guard
        self._reported_phase = None
        self._phase_cases = 0

    def _case(self, cid):
        case = super()._case(cid)
        if self._reported_phase is not None:
            self._phase_cases += 1
            self.reporter.update(self._reported_phase, case_id=cid, cases_visited=self._phase_cases,
                                 pool_size=self.pool_size, updates=self.state['updates'])
        return case

    def _eligible(self, epoch, weights):
        self.reporter.update('eligibility', global_epoch=epoch, pool_size=self.pool_size,
                             updates=self.state['updates'])
        self._reported_phase, self._phase_cases = 'eligibility', 0
        try:
            result = super()._eligible(epoch, weights)
        finally:
            self._reported_phase = None
        self.reporter.update('eligibility_complete', global_epoch=epoch,
                             eligible_counts={name: len(ids) for name, ids in result.items()})
        return result

    def _query(self, count, *, stages):
        self.reporter.update('acquisition', query_count=count, labels_spent=self.labels.spent,
                             stage_index=self.state['stage_index'], updates=self.state['updates'])
        self._reported_phase, self._phase_cases = 'acquisition', 0
        try:
            return super()._query(count, stages=stages)
        finally:
            self._reported_phase = None

    def _save(self, stages, *, complete=False):
        if self.policy_guard is not None:
            self.policy_guard()
        super()._save(stages, complete=complete)
        self.reporter.update('training', stage_index=self.state['stage_index'],
            epoch=self.state['epoch'], slot=self.state['slot'], updates=self.state['updates'],
            labels_spent=self.labels.spent, schedule_finished=complete)


def inspect_final_training_checkpoint(path, *, expected_identity, config, permitted_case_ids,
                                      expected_label_state=None, expected_query_access_sha256=None,
                                      expected_checkpoint_sha256=None):
    """Read-only final gate, including the complete paid-answer ledger.

    At training completion supply the current paid LabelStore.state_dict().
    Independent verification instead requires BOTH checkpoint and journal hashes
    from an already trusted frozen final-model manifest, plus its identity and
    config. Never derive these expected hashes from the files being inspected.
    Neither path queries labels or opens the curator/evaluation store.
    """
    if expected_identity.get('config_hash') != config.identity:
        raise ValueError('final training configuration identity mismatch')
    if (expected_identity.get('purpose') != RUN_PURPOSE or config.variant != 'A3'
            or config.regime != 'closed_loop' or config.seed != 42):
        raise ValueError('final training identity/purpose requires seed42 A3 closed_loop')
    ids = tuple(permitted_case_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('final training requires a unique complete permitted pool')
    path = Path(path)
    checkpoint_digest = file_sha256(path)
    if expected_checkpoint_sha256 is not None and checkpoint_digest != expected_checkpoint_sha256:
        raise ValueError('frozen checkpoint identity checksum mismatch')
    payload = inspect_checkpoint(path, expected_identity=expected_identity, require_complete=True)
    if any((value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()
           for value in payload['model'].values()):
        raise ValueError('final checkpoint contains nonfinite model weights')
    schedule = config.schedule()
    stages = schedule.stages(config.regime, pool_size=len(ids))
    epoch_count = sum(stage['epochs'] for stage in stages)
    slots = schedule.update_slots(len(ids))
    updates = epoch_count * slots
    progress, extra = payload['progress'], payload['extra_state']
    loop = extra.get('loop', {})
    if (progress['stage'] != 'finished' or progress['epoch'] != 0 or progress['slot'] != 0
            or progress['encoder_version'] != updates or loop.get('updates') != updates
            or loop.get('stage_index') != len(stages) or loop.get('epoch') != 0
            or loop.get('slot') != 0 or loop.get('stream') is not None):
        raise ValueError('checkpoint has not finished the complete scheduled stages/updates')
    expected_epochs = [(stage['stage'], epoch) for stage in stages for epoch in range(stage['epochs'])]
    epochs = loop.get('epochs', [])
    if (len(epochs) != epoch_count or loop.get('epoch_totals') or loop.get('epoch_counts')
            or [(row['stage'], row['epoch']) for row in epochs] != expected_epochs
            or [row['global_epoch'] for row in epochs] != list(range(epoch_count))
            or any(row['update_slots'] != slots for row in epochs)):
        raise ValueError('checkpoint full schedule epoch evidence is incomplete')
    query_sets = payload['query_sets']
    paid_count = min(schedule.budget, len(ids))
    if (set(query_sets['L']) | set(query_sets['U']) != set(ids)
            or len(query_sets['L']) != paid_count):
        raise ValueError('final query population/budget differs from permitted training pool')
    journal_path, trace_path = path.parent / 'query-access.json', path.parent / 'query-trace.json'
    if expected_label_state is None and (expected_query_access_sha256 is None or expected_checkpoint_sha256 is None):
        raise ValueError('final paid label ledger requires expected paid state or trusted frozen checkpoint/journal hashes')
    journal_bytes = journal_path.read_bytes()
    journal_digest = sha256(journal_bytes).hexdigest()
    if expected_query_access_sha256 is not None and journal_digest != expected_query_access_sha256:
        raise ValueError('final paid label ledger frozen checksum mismatch')
    journal, trace_payload = strict_json_loads(journal_bytes), read_json(trace_path)
    ledger = journal['labels']
    if expected_label_state is not None and ledger != expected_label_state:
        raise ValueError('final paid label ledger differs from the already-paid label store state')
    if (journal['identity'] != expected_identity or ledger['schema'] != 'vast-v3-query-access/1'
            or ledger['budget'] != schedule.budget or ledger['max_roots'] != config.effective_max_roots
            or ledger['blocked_reason'] or len(ledger['accessed']) != paid_count
            or set(ledger['accessed']) != set(query_sets['L'])
            or set(ledger['revealed']) != set(query_sets['L'])):
        raise ValueError('final paid label ledger is incomplete or exceeds budget')
    if trace_payload != extra.get('query_trace'):
        raise ValueError('final query trace must exactly match checkpoint completion evidence')
    trace = OrderedQueryTrace(trace_path, identity=expected_identity, pool_ids=ids,
                              schedule=schedule, stages=stages, seed=config.seed)
    trace.load(checkpoint_trace=extra.get('query_trace'))
    trace.reconcile(ledger, binding=journal.get('query_plans'),
                    checkpoint_binding=extra.get('query_plans'), loop_state=loop)
    if (len(trace.rounds) != len(schedule.query_batches(len(ids)))
            or any(not row['complete'] for row in trace.rounds)
            or len(trace.query_order) != paid_count):
        raise ValueError('final ordered query trace is incomplete')
    return {'schedule_epochs': epoch_count, 'updates': updates, 'labels_spent': paid_count,
            'query_order': trace.query_order, 'checkpoint_sha256': checkpoint_digest,
            'query_access_sha256': journal_digest, 'query_trace_sha256': file_sha256(trace_path)}
