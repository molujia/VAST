"""Frozen, label-blind cached training subset. Never a production population."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from pathlib import Path
import time

import torch

from ..data.case_cache import ObservableCaseCache, BINDING_SCHEMA, data_code_hash
from ..data.repository import file_sha256, manifest_to_dict
from ..data.serialization import exact_fields, strict_json_loads
from ..runtime import RuntimeConfig, canonical_hash
from .run_control import RunProgress, ProgressTrainer, atomic_json, read_json


QUICKRUN_PURPOSE = 'cached-subset-budgeted-quickstart-not-production'
SNAPSHOT_SCHEMA = 'vast-v3-cached-training-subset/1'
QUICKRUN_PROFILE = 'cached-subset-b30-16epochs-v1'


def quickrun_runtime_config(dataset):
    return RuntimeConfig(dataset=dataset, warmup_epochs=1, nonfinal_epochs=1, final_epochs=10,
                         numeric_execution='metric-scalar-trace-bucket-v1', numeric_batch_size=32,
                         numeric_max_batch_samples=32768)


class ReadOnlyCaseCache(ObservableCaseCache):
    """Existing cache only: no directory creation and no cache-miss fallback."""
    def __init__(self, root, *, max_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('cache max_bytes must be positive')
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError('quickrun requires an existing observation cache')
        self.max_bytes = max_bytes
        self._stats = dict(hits=0, misses=0, writes=0, bytes_written=0)

    def load(self, binding):
        result = super().load(binding)
        if result is None:
            raise ValueError('frozen cache entry missing; adapter fallback is forbidden')
        return result

    def store(self, *args, **kwargs):
        raise ValueError('quickrun observation cache is read-only')


def _parent(repository):
    if (repository.identity_kind != 'content-pinned' or repository.case_cache is None
            or repository._pins is None or repository._cache_code_hash != data_code_hash()):
        raise ValueError('snapshot requires the original content-pinned repository and data code')
    return {'manifest_sha256': canonical_hash(manifest_to_dict(repository.manifest)),
            'repository_identity_sha256': repository.identity_sha256,
            'source_pins_sha256': canonical_hash(repository._pins),
            'data_code_sha256': repository._cache_code_hash,
            'static_topology_sha256': (repository.static_topology.identity_sha256
                                       if repository.static_topology is not None else None)}


def _binding(repository, cid):
    return {'schema': BINDING_SCHEMA, 'repository_identity_sha256': repository.identity_sha256,
            'repository_identity_kind': repository.identity_kind, 'case_id': cid,
            'window': asdict(repository._windows[cid]), 'catalog': asdict(repository.manifest.catalog),
            'source_sha256': {str(p): repository._pins[str(p)] for p in repository._files[cid]},
            'code_sha256': repository._cache_code_hash,
            'static_topology_sha256': (repository.static_topology.identity_sha256
                                       if repository.static_topology is not None else None)}


def _inventory(cache):
    if (cache.root / '.write.lock').exists() or any(cache.root.glob('.*.tmp')):
        raise ValueError('snapshot requires a quiescent cache without writer lock or partial files')
    paths = tuple(sorted(cache.root.glob('*.json.gz')))
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('snapshot cache entries must be regular files')
    return paths


def _snapshot_identity(snapshot):
    return canonical_hash({k: v for k, v in snapshot.items() if k != 'identity_sha256'})


def create_snapshot(repository):
    """Freeze every currently published entry, without reading any labels."""
    parent = _parent(repository)
    entries, seen = [], set()
    allowed = set(repository.manifest.split.train_case_ids)
    paths = _inventory(repository.case_cache)
    for path in paths:
        try:
            envelope = strict_json_loads(gzip.decompress(path.read_bytes()))
            cid = envelope['payload']['binding']['case_id']
            if not isinstance(cid, str) or cid not in allowed or cid in seen:
                raise ValueError('snapshot contains duplicate, test, or foreign case identity')
            binding = _binding(repository, cid)
            if path.name != repository.case_cache._path(binding).name:
                raise ValueError('cache filename/parent binding identity mismatch')
            if repository.case_cache.load(binding) is None:
                raise ValueError('cache entry disappeared during snapshot')
            entries.append({'case_id': cid, 'filename': path.name, 'sha256': file_sha256(path)})
            seen.add(cid)
        except (OSError, EOFError, KeyError, TypeError) as exc:
            raise ValueError(f'malformed snapshot cache entry: {path.name}') from exc
    entries.sort(key=lambda row: row['case_id'])
    result = {'schema': SNAPSHOT_SCHEMA, 'purpose': QUICKRUN_PURPOSE,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset': repository.manifest.split.dataset, 'parent': parent,
              'train_case_ids': [e['case_id'] for e in entries], 'entries': entries}
    result['identity_sha256'] = _snapshot_identity(result)
    validate_snapshot(result, repository)
    return result


def validate_snapshot(snapshot, repository):
    """Fail closed before fitting, querying, or loading models."""
    exact_fields(snapshot, ('schema', 'purpose', 'created_at_utc', 'dataset', 'parent',
                            'train_case_ids', 'entries', 'identity_sha256'), 'cached snapshot')
    if (snapshot['schema'] != SNAPSHOT_SCHEMA or snapshot['purpose'] != QUICKRUN_PURPOSE
            or snapshot['dataset'] != repository.manifest.split.dataset
            or canonical_hash(snapshot['parent']) != canonical_hash(_parent(repository))
            or snapshot['identity_sha256'] != _snapshot_identity(snapshot)):
        raise ValueError('cached snapshot parent/schema/identity mismatch')
    try:
        timestamp = datetime.fromisoformat(snapshot['created_at_utc'])
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError('snapshot time must be UTC')
    except (ValueError, TypeError) as exc:
        raise ValueError('snapshot requires a valid UTC creation time') from exc
    ids = snapshot['train_case_ids']
    if (not isinstance(ids, list) or any(not isinstance(cid, str) or not cid for cid in ids)
            or len(ids) <= 30 or len(set(ids)) != len(ids) or ids != sorted(ids)
            or not set(ids) <= set(repository.manifest.split.train_case_ids)):
        raise ValueError('cached subset requires >30 unique sorted original train IDs; no test/foreign IDs')
    entries = snapshot['entries']
    if not isinstance(entries, list) or len(entries) != len(ids):
        raise ValueError('snapshot cache entries must exactly cover the frozen subset')
    filenames = []
    for cid, entry in zip(ids, entries):
        exact_fields(entry, ('case_id', 'filename', 'sha256'), 'snapshot entry')
        binding = _binding(repository, cid)
        path = repository.case_cache._path(binding)
        if entry['case_id'] != cid or entry['filename'] != path.name:
            raise ValueError('snapshot case/cache filename identity mismatch')
        if (path.is_symlink() or not path.is_file() or file_sha256(path) != entry['sha256']):
            raise ValueError('snapshot cache entry missing or checksum mismatch')
        if repository.case_cache.load(binding) is None:
            raise ValueError('snapshot cache entry missing')
        filenames.append(path.name)
    if set(filenames) != {p.name for p in _inventory(repository.case_cache)}:
        raise ValueError('snapshot must bind all published training cache entries')
    return tuple(ids)


class CachedTrainingCases(Mapping):
    """Training view, leaving the complete parent manifest/repository untouched."""
    def __init__(self, repository, snapshot):
        self.repository = repository
        self.snapshot = deepcopy(snapshot)
        self._ids = validate_snapshot(self.snapshot, repository)
        self._entries = {row['case_id']: row for row in self.snapshot['entries']}
        # Preserve repository's raw/stat/inventory checks, but close its miss path.
        repository.case_cache = ReadOnlyCaseCache(repository.case_cache.root,
                                                  max_bytes=repository.case_cache.max_bytes)

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def __getitem__(self, cid):
        if cid not in self._entries:
            raise KeyError(cid)
        entry = self._entries[cid]
        path = self.repository.case_cache.root / entry['filename']
        if (path.is_symlink() or not path.is_file() or file_sha256(path) != entry['sha256']):
            raise ValueError('frozen cache entry missing or checksum mismatch; no adapter fallback')
        return self.repository[cid]

    def case_audit(self, cid):
        if cid not in self._entries:
            raise KeyError(cid)
        return self.repository.case_audit(cid)


class QuickrunProgress(RunProgress):
    def __init__(self, directory, *, resume=False):
        self.directory = Path(directory)
        marker = {'schema': 'vast-v3-cached-quickrun/1', 'purpose': QUICKRUN_PURPOSE,
                  'formal_experiment': False}
        if resume:
            if not (self.directory / 'run.json').is_file() or read_json(self.directory / 'run.json') != marker:
                raise ValueError('resume requires the original cached quickrun identity/purpose')
        else:
            try:
                self.directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError('existing quickrun requires explicit resume') from exc
            atomic_json(self.directory / 'run.json', marker)
        self.started = time.perf_counter()

    def update(self, phase, **values):
        return super().update(phase, **{'purpose': QUICKRUN_PURPOSE, 'eligible_for_demo': False,
            'formal_experiment': False, 'eligible_for_quickstart': False, 'quickstart_ready': False, **values})


def validate_training_subset(repository, config, subset, provenance):
    if (type(subset) is not CachedTrainingCases or subset.repository is not repository
            or canonical_hash(subset.snapshot) != canonical_hash(provenance)
            or config != quickrun_runtime_config(config.dataset)):
        raise ValueError('quickrun subset/profile provenance identity mismatch')
    validate_snapshot(provenance, repository)
    return {'training_purpose': QUICKRUN_PURPOSE,
            'subset_snapshot_hash': provenance['identity_sha256'],
            'quickrun_profile_hash': canonical_hash({'profile': QUICKRUN_PROFILE, 'config': config.to_dict()}),
            'subset_case_ids_hash': canonical_hash(list(subset))}


def validate_quickrun_invocation(invocation, identity):
    if (invocation.get('schema') != 'vast-v3-quickrun-invocation/1'
            or invocation.get('purpose') != QUICKRUN_PURPOSE
            or invocation.get('standard_pool_hash') != identity['manifest_hash']
            or any(invocation.get(key) != identity[key] for key in (
                'manifest_hash', 'config_hash', 'code_hash', 'text_lock_hash', 'data_hash',
                'curator_hash', 'static_topology_hash', 'subset_snapshot_hash',
                'quickrun_profile_hash', 'subset_case_ids_hash',
                'convolution_policy', 'execution_policy_hash', 'effective_backend_sha256'))):
        raise ValueError('quickrun invocation input identity mismatch')


class QuickrunTrainer(ProgressTrainer):
    """Observe actual U backward and Adam updates without extra forwards/RNG.

    Scalar-loss hooks prove the U term participates in backward; optimizer hooks
    establish actual parameter changes on those same steps. This does not claim
    that changes from a mixed-objective step are attributable exclusively to U.
    Evidence lives in loop state and is checkpointed/restored by the real Trainer.
    """
    def run(self, **kwargs):
        original = self.objectives.case_losses
        pending = []
        before = []

        def evidence():
            return self.state.setdefault('quickrun_u_evidence', {
                'backward_terms': 0, 'optimizer_steps': 0, 'parameter_update_steps': 0, 'case_ids': []})

        def observed_losses(case, **options):
            values = original(case, **options)
            value = values.get('U_soft')
            if torch.is_grad_enabled() and value is not None and value.requires_grad:
                def backward(gradient):
                    if torch.isfinite(gradient).all() and torch.count_nonzero(gradient).item():
                        record = evidence()
                        record['backward_terms'] += 1
                        record['case_ids'] = sorted(set(record['case_ids']) | {case.case_id})
                        pending.append(case.case_id)
                value.register_hook(backward)
            return values

        def before_step(optimizer, args, kwargs):
            before.clear()
            if pending:
                before.extend((p, p.detach().clone()) for group in optimizer.param_groups
                              for p in group['params'] if p.grad is not None)

        def after_step(optimizer, args, kwargs):
            if pending:
                record = evidence()
                record['optimizer_steps'] += 1
                if any(not torch.equal(p.detach(), old) for p, old in before):
                    record['parameter_update_steps'] += 1
            pending.clear()
            before.clear()

        self.objectives.case_losses = observed_losses
        pre = self.optimizer.register_step_pre_hook(before_step)
        post = self.optimizer.register_step_post_hook(after_step)
        try:
            return super().run(**kwargs)
        finally:
            self.objectives.case_losses = original
            pre.remove()
            post.remove()


def inspect_quickrun_u_evidence(path, *, config, permitted_case_ids):
    if config != quickrun_runtime_config(config.dataset) or len(permitted_case_ids) <= 30:
        raise ValueError('quickstart acceptance requires the exact B30/16-epoch cached profile')
    payload = torch.load(path, weights_only=True, map_location='cpu')
    loop = payload['extra_state']['loop']
    evidence = loop.get('quickrun_u_evidence', {})
    u_epochs = [row for row in loop['epochs'] if row['weights']['U_soft'] > 0]
    counts = [row['U_soft_count'] for row in u_epochs]
    eligible = [row['eligible_counts']['U_soft'] for row in u_epochs]
    ready = (len(u_epochs) == 5 and all(count > 0 for count in counts)
             and all(count > 0 for count in eligible)
             and all(row['U_soft_mean'] is not None for row in u_epochs)
             and all(type(evidence.get(key)) is int and evidence[key] > 0 for key in
                     ('backward_terms', 'optimizer_steps', 'parameter_update_steps'))
             and evidence.get('backward_terms') == sum(counts)
             and evidence.get('parameter_update_steps', 0) <= evidence.get('optimizer_steps', 0)
             and evidence.get('optimizer_steps', 0) <= sum(row['update_slots'] for row in u_epochs)
             and bool(evidence.get('case_ids'))
             and set(evidence.get('case_ids', [])) <= set(payload['query_sets']['U']))
    return {'u_training_verified': bool(ready), 'u_training_evidence': evidence,
            'u_active_epochs': len(u_epochs), 'u_loss_count': sum(counts)}
