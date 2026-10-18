"""Lazy model-facing repository with a separate source-identity audit.

Construction scans file metadata, never telemetry contents. Content-pinned
identities require a complete externally frozen SHA-256 map. Without that map
the identity is explicitly metadata-bound; content digests are recorded on
first materialization. This distinction must survive checkpoint provenance.
"""
from __future__ import annotations

from collections.abc import Mapping, Iterator
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re

from . import adapters, case_cache as cache_module
from .manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from .schema import Entity, ObservableCase
from .serialization import exact_fields, strict_json_loads


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _validate_manifest(manifest):
    ids = [w.case_id for w in manifest.windows]
    if (len(ids) != len(set(ids)) or set(ids) != set(manifest.split.train_case_ids) | set(manifest.split.test_case_ids)
            or any(w.dataset != manifest.split.dataset for w in manifest.windows)):
        raise ValueError('frozen manifest population/dataset mismatch')
    if manifest.split.dataset not in ('rcabench', 'aiops2022_pre'):
        raise ValueError('unsupported dataset')


def manifest_to_dict(manifest: CuratedManifest) -> dict:
    _validate_manifest(manifest)
    # audit is intentionally not serialized: it may contain curator-only data.
    return json.loads(json.dumps({'schema': 'vast-v3-public-manifest/1',
        'windows': [asdict(w) for w in manifest.windows], 'catalog': asdict(manifest.catalog),
        'split': asdict(manifest.split)}, allow_nan=False))


def manifest_from_dict(payload: Mapping) -> CuratedManifest:
    exact_fields(payload, ('schema', 'windows', 'catalog', 'split'), 'public manifest')
    if payload['schema'] != 'vast-v3-public-manifest/1':
        raise ValueError('unsupported public manifest schema')
    catalog = payload['catalog']; split = payload['split']
    exact_fields(catalog, (f.name for f in fields(CandidateCatalog)), 'catalog')
    exact_fields(split, (f.name for f in fields(FrozenSplit)), 'split')
    entities = []
    for row in catalog['entities']:
        exact_fields(row, (f.name for f in fields(Entity)), 'entity')
        if type(row['candidate']) is not bool:
            raise ValueError('candidate must be boolean')
        entities.append(Entity(**row))
    result = CuratedManifest(tuple(WindowSpec.from_dict(w) for w in payload['windows']),
        CandidateCatalog(tuple(entities), catalog['version'], catalog['source_sha256']), FrozenSplit(**split), {})
    _validate_manifest(result)
    return result


def _inside(root, candidate):
    path = candidate.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('source path must stay within raw root')
    return path


def _stat(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


class DatasetRepository(Mapping[str, ObservableCase]):
    @classmethod
    def from_curated(cls, manifest: CuratedManifest | str | Path, raw_root: str | Path, *,
                     expected_manifest_sha256: str | None = None,
                     expected_source_sha256: Mapping[str, str] | None = None,
                     max_cached_cases: int = 2, static_topology=None, case_cache=None):
        if not isinstance(manifest, CuratedManifest):
            raw = Path(manifest).read_bytes()
            if expected_manifest_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
                raise ValueError('public manifest checksum mismatch')
            manifest = manifest_from_dict(strict_json_loads(raw))
        elif expected_manifest_sha256 is not None and _digest(manifest_to_dict(manifest)) != expected_manifest_sha256:
            raise ValueError('public manifest checksum mismatch')
        return cls(manifest, raw_root, expected_source_sha256, max_cached_cases,
                   static_topology=static_topology, case_cache=case_cache)

    def __init__(self, manifest, raw_root, expected_source_sha256=None, max_cached_cases=2, *, static_topology=None,
                 case_cache=None):
        _validate_manifest(manifest)
        self.static_topology = static_topology
        if static_topology is not None:
            from .static_topology import StaticTopology
            if type(static_topology) is not StaticTopology:
                raise ValueError('static topology must be a validated StaticTopology artifact')
            static_topology.validate_manifest(manifest)
        if type(max_cached_cases) is not int or max_cached_cases < 1:
            raise ValueError('raw case cache capacity must be a positive integer')
        self.max_cached_cases = max_cached_cases
        self.manifest = manifest
        self.raw_root = Path(raw_root).resolve()
        if case_cache is not None:
            if not isinstance(case_cache, cache_module.ObservableCaseCache):
                raise ValueError('case_cache must be an ObservableCaseCache')
            if case_cache.root.is_relative_to(self.raw_root):
                raise ValueError('observable cache must be outside the raw root')
        self.case_cache = case_cache
        self._cache_code_hash = cache_module.data_code_hash() if case_cache is not None else None
        self._windows = {w.case_id: w for w in manifest.windows}
        self._cases, self._audits, self._files, self._stats = OrderedDict(), {}, {}, {}
        self._source_digests = {}
        # Resolve every path first, so traversal is rejected independently of
        # whether another case has missing modalities.
        bases = {w.case_id: _inside(self.raw_root, self.raw_root / (w.source_id if w.dataset == 'rcabench' else w.day))
                 for w in manifest.windows}
        for key, window in self._windows.items():
            base = bases[key]
            if window.dataset == 'rcabench':
                paths = [base / 'env.json', *(base / f'{phase}_{kind}.parquet'
                    for phase in ('normal', 'abnormal') for kind in ('metrics', 'metrics_sum', 'metrics_histogram', 'logs', 'traces'))]
            else:
                base = base / 'cloudbed'
                groups = [sorted((base / 'metric' / scope).glob('*.csv')) for scope in ('container', 'node')]
                groups.append(sorted((base / 'log/all').glob('log_filebeat-*.csv')))
                if not all(groups):
                    raise FileNotFoundError(f'required AIOps raw modality files absent: {base}')
                paths = [p for group in groups for p in group] + [base / 'trace/all/trace_jaeger-span.csv']
            self._files[key] = tuple(_inside(self.raw_root, path) for path in paths)
            for path in self._files[key]:
                if not path.is_file():
                    raise FileNotFoundError(f'required raw source file absent: {path}')
                self._stats[str(path)] = _stat(path)
        self._pins = None
        if expected_source_sha256 is not None:
            self._pins = {str(Path(p).resolve()): digest for p, digest in expected_source_sha256.items()}
            if set(self._pins) != set(self._stats) or any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in self._pins.values()):
                raise ValueError('source checksum map must cover exactly every required file')
        self._static_source = None
        if static_topology is not None:
            self._static_source = _inside(self.raw_root, self.raw_root / static_topology.source_file)
            if str(self._static_source) not in self._stats:
                raise ValueError('static topology source absent from repository inventory')
            if self._pins is not None and self._pins[str(self._static_source)] != static_topology.source_sha256:
                raise ValueError('static topology source checksum mismatch')
        self.identity_kind = 'content-pinned' if self._pins is not None else 'metadata-bound'
        self._update_identity()

    def _update_identity(self):
        self.identity_sha256 = _digest({'manifest': manifest_to_dict(self.manifest), 'raw_root': str(self.raw_root),
            'identity_kind': self.identity_kind, 'sources': self._pins if self._pins is not None else self._stats,
            **({'static_topology_sha256': self.static_topology.identity_sha256} if self.static_topology is not None else {})})

    def _check_static_source(self):
        if self._static_source is None:
            return
        path = self._static_source
        self._check_stats((path,))
        if str(path) not in self._source_digests:
            self._source_digests[str(path)] = file_sha256(path)
            self._check_stats((path,))
        if self._source_digests[str(path)] != self.static_topology.source_sha256:
            raise ValueError('static topology source checksum mismatch')

    def pin_source_content(self):
        """Explicit expensive curator operation: hash each unique source once.

        Call before creating any checkpoint or loading cases. Unlike normal
        construction this reads all required source bytes, but not their rows.
        Formal runtime must use this or a complete expected_source_sha256 map.
        """
        if self._audits:
            raise ValueError('content identity must be frozen before loading any case')
        paths = tuple(Path(p) for p in self.source_files)
        self._check_stats(paths)
        for path in paths:
            if str(path) not in self._source_digests:
                digest = file_sha256(path)
                self._check_stats((path,))
                self._source_digests[str(path)] = digest
        self._check_stats(paths)
        self._check_static_source()
        if self._pins is not None and self._pins != self._source_digests:
            raise ValueError('raw source checksum mismatch')
        self._pins = dict(self._source_digests)
        self.identity_kind = 'content-pinned'
        self._update_identity()
        return dict(self._pins)

    def __len__(self):
        return len(self._windows)

    def __iter__(self) -> Iterator[str]:
        return iter(self._windows)

    @property
    def source_files(self):
        return tuple(sorted(self._stats))

    @property
    def loaded_case_ids(self):
        return tuple(self._cases)

    def _check_stats(self, paths):
        if any(not p.is_file() or _stat(p) != self._stats[str(p)] for p in paths):
            raise ValueError('raw source changed after repository identity freeze')

    def __getitem__(self, case_id):
        window = self._windows[case_id]
        if self.case_cache is not None and self.identity_kind != 'content-pinned':
            raise ValueError('observable cache requires content-pinned repository before case access')
        paths = self._files[case_id]
        self._check_stats(paths)
        self._check_static_source()
        if window.dataset == 'aiops2022_pre':
            base = self.raw_root / window.day / 'cloudbed'
            current = {p.resolve() for scope in ('container', 'node') for p in (base / 'metric' / scope).glob('*.csv')}
            current.update(p.resolve() for p in (base / 'log/all').glob('log_filebeat-*.csv'))
            current.add((base / 'trace/all/trace_jaeger-span.csv').resolve())
            if current != set(paths):
                raise ValueError('raw source file inventory changed after identity freeze')
        if case_id in self._cases:
            self._cases.move_to_end(case_id)
            return self._cases[case_id]
        for path in paths:
            if str(path) not in self._source_digests:
                digest = file_sha256(path)
                self._check_stats((path,))
                self._source_digests[str(path)] = digest
        digests = {str(path): self._source_digests[str(path)] for path in paths}
        if self._pins is not None and any(self._pins[p] != sha for p, sha in digests.items()):
            raise ValueError('raw source checksum mismatch')
        binding = None
        cached = None
        if self.case_cache is not None:
            binding = {'schema': cache_module.BINDING_SCHEMA,
                'repository_identity_sha256': self.identity_sha256, 'repository_identity_kind': self.identity_kind,
                'case_id': case_id, 'window': asdict(window), 'catalog': asdict(self.manifest.catalog),
                'source_sha256': digests, 'code_sha256': self._cache_code_hash,
                'static_topology_sha256': self.static_topology.identity_sha256 if self.static_topology is not None else None}
            cached = self.case_cache.load(binding)
        if cached is None:
            loaded = (adapters.load_rcabench_case(self.raw_root / window.source_id, window=window, catalog=self.manifest.catalog)
                      if window.dataset == 'rcabench' else
                      adapters.load_aiops22_case(self.raw_root, window=window, catalog=self.manifest.catalog))
            if self.static_topology is not None:
                loaded = self.static_topology.apply(loaded, window)
        else:
            loaded = adapters.LoadedCase(*cached)
        self._check_stats(paths)
        if loaded.case.case_id != case_id or loaded.case.candidate_ids != self.manifest.catalog.candidate_ids:
            raise ValueError('loaded case/catalog identity mismatch')
        audit = {**loaded.audit, 'source_sha256': digests,
            'case_source_sha256': _digest(digests), 'repository_identity_sha256': self.identity_sha256,
            'repository_identity_kind': self.identity_kind,
            'context_only_entity_ids': sorted(e.id for e in loaded.case.entities if not e.candidate)}
        if self.case_cache is not None and cached is None:
            self.case_cache.store(binding, loaded.case, audit)
        self._cases[case_id] = loaded.case
        if len(self._cases) > self.max_cached_cases:
            self._cases.popitem(last=False)
        self._audits[case_id] = audit
        return loaded.case

    def case_audit(self, case_id):
        if case_id not in self._audits:
            raise ValueError('case has not been loaded; audit is not a model input')
        return deepcopy(self._audits[case_id])
