"""Optional, content-bound observable JSON cache; no labels or model tensors.

Published entries are immutable. Corruption, orphan partial files, contention,
and capacity exhaustion are explicit errors, never silent eviction or fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .serialization import exact_fields, observable_from_dict, observable_to_dict, strict_json_loads


SCHEMA = 'vast-v3-observable-cache/1'
BINDING_SCHEMA = 'vast-v3-observable-cache-binding/1'
_BINDING_FIELDS = ('schema', 'repository_identity_sha256', 'repository_identity_kind', 'case_id',
                   'window', 'catalog', 'source_sha256', 'code_sha256', 'static_topology_sha256')


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def data_code_hash():
    """Conservatively bind all installed data-module names and source bytes."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def _audit(value):
    if isinstance(value, dict):
        forbidden = {'ground_truth', 'roots', 'root', 'targets', 'target', 'labels', 'label',
                     'raw_roots', 'canonical_roots', 'root_labels', 'root_ids',
                     'fault_type', 'injection', 'positive_ids', 'positive_types', 'is_positive',
                     'metadata_json', 'conclusion'}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('cache audit keys must be strings')
            if key.lower() in forbidden or key.lower().startswith(('root_cause', 'fault_', 'positive_')):
                raise ValueError('label-bearing cache audit field is forbidden')
            _audit(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _audit(item)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError('cache audit accepts plain JSON only, not tensors or objects')


def _binding(binding):
    exact_fields(binding, _BINDING_FIELDS, 'cache binding')
    if binding['schema'] != BINDING_SCHEMA or binding['repository_identity_kind'] != 'content-pinned':
        raise ValueError('cache requires a content-pinned binding')
    if not isinstance(binding['case_id'], str) or not binding['case_id']:
        raise ValueError('cache case identity must be nonempty')
    hashes = [binding['repository_identity_sha256'], binding['code_sha256']]
    if binding['static_topology_sha256'] is not None:
        hashes.append(binding['static_topology_sha256'])
    sources = binding['source_sha256']
    if not isinstance(sources, dict) or not sources or any(not isinstance(p, str) or not p for p in sources):
        raise ValueError('cache source checksum map must be nonempty')
    hashes.extend(sources.values())
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
           for value in hashes):
        raise ValueError('cache binding requires SHA-256 identities')
    return strict_json_loads(_bytes(binding))


class ObservableCaseCache:
    """A dedicated local directory with an explicit compressed-byte quota.

    Repository input validation is required before load. The quota covers
    published gzip entries and in-flight temporary bytes, not decoded memory.
    ``stats`` is operational provenance only, separate from case/audit content.
    """

    def __init__(self, root, *, max_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('cache max_bytes capacity must be a positive integer')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._stats = dict(hits=0, misses=0, writes=0, bytes_written=0)

    @property
    def stats(self):
        return dict(self._stats)

    def _path(self, binding):
        return self.root/(_digest(binding)+'.json.gz')

    def _read(self, binding):
        path = self._path(binding)
        key = path.name.removesuffix('.json.gz')
        if any(self.root.glob(f'.{key}.*.tmp')):
            raise ValueError(f'incomplete cache partial for {key}; inspect writer/partial before retry')
        if path.is_symlink():
            raise ValueError(f'cache entry must not be a symbolic link: {path}')
        if not path.exists():
            return None
        try:
            envelope = strict_json_loads(gzip.decompress(path.read_bytes()))
            exact_fields(envelope, ('schema', 'payload', 'payload_sha256'), 'cache envelope')
            if envelope['schema'] != SCHEMA or _digest(envelope['payload']) != envelope['payload_sha256']:
                raise ValueError('cache payload checksum/schema mismatch')
            payload = envelope['payload']
            exact_fields(payload, ('binding', 'case', 'audit'), 'cache payload')
            if payload['binding'] != binding:
                raise ValueError('cache input identity mismatch')
            case = observable_from_dict(payload['case'])
            if not isinstance(payload['audit'], dict):
                raise ValueError('cache audit must be an object')
            _audit(payload['audit'])
            if case.case_id != binding['case_id']:
                raise ValueError('cache case identity mismatch')
            window = binding['window']
            if any(getattr(case, name) != window[name] for name in ('reference_start', 'incident_start', 'incident_end')):
                raise ValueError('cache case window mismatch')
            return case, payload['audit'], payload
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'invalid observable cache entry {path}: {exc}') from exc

    def load(self, binding):
        checked = _binding(binding)
        result = self._read(checked)
        self._stats['hits' if result is not None else 'misses'] += 1
        return None if result is None else result[:2]

    @contextmanager
    def _writer(self):
        path = self.root/'.write.lock'
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError(f'cache writer lock busy: {path}; verify live writer before recovery') from exc
        try:
            with os.fdopen(descriptor, 'w', encoding='ascii') as handle:
                handle.write(str(os.getpid())); handle.flush(); os.fsync(handle.fileno())
            yield
        finally:
            path.unlink()

    def store(self, binding, case, audit):
        checked = _binding(binding)
        _audit(audit)
        if not isinstance(audit, dict) or case.case_id != checked['case_id']:
            raise ValueError('cache audit/case identity mismatch')
        payload = {'binding': checked, 'case': observable_to_dict(case), 'audit': strict_json_loads(_bytes(audit))}
        envelope = {'schema': SCHEMA, 'payload': payload, 'payload_sha256': _digest(payload)}
        compressed = gzip.compress(_bytes(envelope), mtime=0)
        destination = self._path(checked)
        with self._writer():
            existing = self._read(checked)
            if existing is not None:
                if existing[2] != payload:
                    raise ValueError('cache key already contains different observable data/audit')
                return
            files = tuple(self.root.glob('*.json.gz')) + tuple(self.root.glob('.*.tmp'))
            if any(path.is_symlink() or not path.is_file() for path in files):
                raise ValueError('cache storage inventory contains a nonregular entry')
            used = sum(path.stat().st_size for path in files)
            if used+len(compressed) > self.max_bytes:
                raise ValueError(f'cache capacity exceeded: {used}+{len(compressed)} > {self.max_bytes}; no entries evicted')
            descriptor, name = tempfile.mkstemp(prefix='.'+destination.name.removesuffix('.json.gz')+'.',
                                                suffix='.tmp', dir=self.root)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, 'wb') as handle:
                    handle.write(compressed); handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, destination)
                if os.name != 'nt':
                    directory = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                if temporary.exists():
                    temporary.unlink()
        self._stats['writes'] += 1
        self._stats['bytes_written'] += len(compressed)
