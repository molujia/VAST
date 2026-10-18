"""Frozen public dataset metadata and explicitly curator-only root mapping.

Nothing in this module samples a split or estimates a fault boundary. Public
window objects contain only approved intervals and provenance, never labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable, Mapping

from .schema import Entity


AIOPS_MAPPING_VERSION = 'aiops-historical-normalize-service/1'
AIOPS_MAPPING_SOURCE = 'VAST/src/vast/_historical/half_supervise/src/nexusrcl_rebuild/datasets/common.py:normalize_service_name'
AIOPS_MAPPING_SOURCE_SHA256 = '9d19b2a478f328815a648b9ab69233501896bbb22ed4948fc9dba38e81e282c8'


def resolve_aiops_owner(catalog, raw: str, category: str = 'service') -> str:
    """AIOps-only inherited mapping; node/host identity is never normalized.

    These exact four patterns reproduce the frozen historical implementation.
    They are not added to the dataset-agnostic catalog resolver.
    """
    _text(raw, 'AIOps owner')
    if category == 'host':
        return catalog.resolve(raw, category)
    if category != 'service':
        raise ValueError('unsupported AIOps entity category')
    name = raw.removeprefix('service:')
    for pattern in (r'^(.*?service)(?:\d+)?-\d+$', r'^(.*?)(?:\d+)-\d+$',
                    r'^(.*?)-\d+$', r'^(.*?service)\d+$'):
        match = re.match(pattern, name)
        if match:
            name = match.group(1)
            break
    return catalog.resolve(name, category)


def _sha256(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('a lowercase SHA-256 source digest is required')
    return value


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    return value


def _strict_fields(data: Mapping, cls):
    expected = {f.name for f in fields(cls)}
    if set(data) != expected:
        raise ValueError(f'{cls.__name__} fields differ: {sorted(set(data) ^ expected)}')


@dataclass(frozen=True)
class CandidateCatalog:
    entities: tuple[Entity, ...]
    version: str
    source_sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'entities', tuple(self.entities))
        _text(self.version, 'catalog version')
        _sha256(self.source_sha256)
        ids = [e.id for e in self.entities]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('frozen catalog must be nonempty and unique')
        for e in self.entities:
            if e.category not in ('service', 'host') or not e.candidate or not e.id.startswith(e.category + ':'):
                raise ValueError('catalog must contain canonical service/host candidates')
            _text(e.id.split(':', 1)[1], 'entity name')

    @property
    def candidate_ids(self):
        return tuple(e.id for e in self.entities)

    def resolve(self, raw: str, category: str = 'service') -> str:
        """Resolve known names/pods; unknown names remain context-only entities."""
        _text(raw, 'observation owner')
        if category not in ('service', 'host'):
            raise ValueError('unsupported entity category')
        name = raw.removeprefix(category + ':')
        exact = category + ':' + name
        if exact in self.candidate_ids or category == 'host':
            return exact
        # Only documented Kubernetes replica suffixes are removed, not arbitrary
        # application suffixes; longest canonical match prevents prefix aliases.
        for entity in sorted(self.entities, key=lambda e: len(e.id), reverse=True):
            base = entity.id.removeprefix('service:')
            if entity.category == 'service' and re.fullmatch(re.escape(base) + r'-(?:\d+|[0-9a-f]{8,10}-[a-z0-9]{5})', name):
                return entity.id
        name = re.sub(r'-(?:\d+|[0-9a-f]{8,10}-[a-z0-9]{5})$', '', name)
        return 'service:' + name


@dataclass(frozen=True)
class WindowSpec:
    case_id: str
    dataset: str
    source_id: str
    day: str
    reference_start: float
    reference_end: float
    incident_start: float
    incident_end: float
    reference_evidence: str
    source_sha256: str
    reference_missing: bool = False

    def __post_init__(self):
        for field in ('case_id', 'dataset', 'source_id', 'day', 'reference_evidence'):
            _text(getattr(self, field), field)
        _sha256(self.source_sha256)
        numbers = (self.reference_start, self.reference_end, self.incident_start, self.incident_end)
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in numbers):
            raise ValueError('window boundaries must be finite seconds')
        if not self.reference_start <= self.reference_end <= self.incident_start < self.incident_end:
            raise ValueError('declared reference and incident windows must not overlap')
        object.__setattr__(self, 'reference_missing', self.reference_start == self.reference_end)

    def includes(self, timestamp: float) -> bool:
        return (self.reference_start <= timestamp < self.reference_end or
                self.incident_start <= timestamp < self.incident_end)

    @classmethod
    def from_dict(cls, data: Mapping):
        _strict_fields(data, cls)
        return cls(**data)


@dataclass(frozen=True)
class FrozenSplit:
    dataset: str
    train_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    source_sha256: str

    def __post_init__(self):
        _text(self.dataset, 'dataset')
        _sha256(self.source_sha256)
        for name in ('train_case_ids', 'test_case_ids'):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if not values or len(values) != len(set(values)) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError('frozen split populations must be nonempty unique case IDs')
        if set(self.train_case_ids) & set(self.test_case_ids):
            raise ValueError('frozen train/test split overlaps')

    @classmethod
    def from_dict(cls, data: Mapping, *, source_sha256: str):
        if set(data) != {'dataset', 'outer_train_case_ids', 'outer_test_case_ids'}:
            raise ValueError('unexpected frozen split fields')
        return cls(data['dataset'], tuple(data['outer_train_case_ids']), tuple(data['outer_test_case_ids']), source_sha256)


def load_frozen_split(path: str | Path, *, expected_sha256: str, expected_counts: tuple[int, int]) -> FrozenSplit:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha256(expected_sha256):
        raise ValueError('frozen split checksum mismatch')
    split = FrozenSplit.from_dict(json.loads(raw), source_sha256=expected_sha256)
    if (len(split.train_case_ids), len(split.test_case_ids)) != expected_counts:
        raise ValueError('frozen split population drift')
    return split


def canonical_roots(catalog: CandidateCatalog, granularity: str, raw_identifiers: Iterable[str], *, complete: bool = True) -> tuple[str, ...]:
    """CURATOR ONLY: map the complete joint root set, never broaden host labels."""
    if not complete:
        raise ValueError('incomplete root annotation is not admissible')
    category = {'service': 'service', 'pod': 'service', 'container': 'service', 'node': 'host', 'host': 'host'}.get(granularity)
    if category is None:
        raise ValueError('unsupported root granularity')
    roots = set()
    for raw in raw_identifiers:
        if granularity == 'container' and '.' in raw:
            raw = raw.rsplit('.', 1)[1]
        roots.add(catalog.resolve(raw, category))
    if not roots or not roots <= set(catalog.candidate_ids):
        raise ValueError('complete root set is empty or outside frozen catalog')
    return tuple(x for x in catalog.candidate_ids if x in roots)


def curate_frozen_windows(rows: Iterable[Mapping], *, dataset: str, source_sha256: str):
    """CURATOR ONLY: project frozen rows to label-free, provenance-bound windows.

    AIOps prior-fault exclusion uses frozen boundaries only. Their historical
    label-dependent derivation is inherited by explicit user approval dated
    2026-10-15; this routine never reads fault type or recomputes durations.
    The source order and complete population are retained, including zero refs.
    """
    _sha256(source_sha256)
    if dataset not in ('rcabench', 'aiops2022_pre'):
        raise ValueError('unknown frozen dataset')
    faults, normals = [], defaultdict(list)
    for row in rows:
        kind = row.get('window_kind')
        if kind == 'fault':
            faults.append(row)
        elif kind == 'normal':
            normals[_text(row.get('day'), 'day')].append(row)
        else:
            raise ValueError('unknown frozen window kind')
    ids = [_text(row.get('window_id'), 'window_id') for row in faults]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('frozen fault population empty or duplicate')
    windows, interval_sources, reference_counts, durations = {}, Counter(), Counter(), Counter()
    days = defaultdict(list)
    for row in faults:
        days[_text(row.get('day'), 'day')].append(row)
    for day, group in days.items():
        previous_end = float('-inf')
        for row in sorted(group, key=lambda r: (float(r['start_ts']), float(r['end_ts']), r['window_id'])):
            start, end = float(row['start_ts']), float(row['end_ts'])
            if dataset == 'rcabench':
                if len(normals[day]) != 1:
                    raise ValueError('RCABench case must have exactly one frozen reference window')
                normal = normals[day][0]
                ref_start, ref_end = float(normal['start_ts']), float(normal['end_ts'])
                evidence = 'paired frozen normal window; no inferred health interval'
            else:
                metadata = row.get('metadata_json', '{}')
                metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
                interval_source = metadata.get('interval_source')
                if interval_source not in ('run_table_exact', 'run_table_level_median', 'run_table_type_median'):
                    raise ValueError('unknown historical interval provenance')
                interval_sources[interval_source] += 1
                ref_start, ref_end = min(start, max(start - 300., previous_end)), start
                if previous_end >= start:
                    reason = 'prior frozen fault overlap' if previous_end > start else 'prior frozen fault boundary'
                    evidence = f'reference missing: {reason}; no healthy segment invented'
                elif previous_end > start - 300:
                    evidence = 'up to 300s preceding fault, clipped at prior frozen fault end'
                else:
                    evidence = '300s preceding fault, nonoverlapping prior frozen fault intervals'
                length = ref_end - ref_start
                reference_counts['zero' if length == 0 else 'full_300' if length == 300 else 'shortened'] += 1
            window = WindowSpec(row['window_id'], dataset, _text(row.get('source_id'), 'source_id'), day,
                                ref_start, ref_end, start, end, evidence, source_sha256)
            windows[window.case_id] = window
            durations[str(end - start)] += 1
            previous_end = max(previous_end, end)
    audit = {'schema': 'vast-v3-curator-window-audit/1', 'dataset': dataset,
             'source_sha256': source_sha256, 'case_count': len(windows),
             'inherited_window_approval_date': '2026-10-15',
             'interval_source_counts': dict(interval_sources), 'reference_counts': dict(reference_counts),
             'incident_duration_seconds_counts': dict(sorted(durations.items(), key=lambda pair: float(pair[0]))),
             'limitations': ['Frozen IDs are curator lookup keys, not model features.',
                            'Inherited AIOps boundaries include label-dependent historical median sources.']}
    return tuple(windows[case_id] for case_id in ids), audit


def _csv_rows(path: str | Path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError('empty curator CSV') from exc
        if len(header) != len(set(header)):
            raise ValueError('duplicate curator CSV columns')
        for values in reader:
            if len(values) != len(header):
                raise ValueError(f'curator CSV width mismatch at line {reader.line_num}')
            yield dict(zip(header, values))


def load_frozen_windows(path: str | Path, *, dataset: str, expected_sha256: str, expected_count: int,
                        expected_interval_sources: Mapping[str, int] | None = None):
    """CURATOR ONLY: verify the existing frozen source; never modify it."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != _sha256(expected_sha256):
        raise ValueError('frozen windows checksum mismatch')
    windows, audit = curate_frozen_windows(_csv_rows(path), dataset=dataset, source_sha256=expected_sha256)
    if len(windows) != expected_count:
        raise ValueError('frozen window population drift')
    if expected_interval_sources is not None and audit['interval_source_counts'] != dict(expected_interval_sources):
        raise ValueError('frozen interval provenance population drift')
    return windows, audit


@dataclass(frozen=True)
class CuratedManifest:
    windows: tuple[WindowSpec, ...]
    catalog: CandidateCatalog
    split: FrozenSplit
    audit: dict


def _ids_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def build_curator_manifest(*, dataset: str, windows_path: str | Path, expected_windows_sha256: str,
                           entity_catalog_path: str | Path, expected_catalog_ids_sha256: str,
                           train_pool_path: str | Path, expected_train_sha256: str,
                           test_rankings_path: str | Path, expected_test_ids_sha256: str,
                           expected_counts: tuple[int, int], expected_entity_counts: Mapping[str, int],
                           expected_interval_sources: Mapping[str, int] | None = None) -> CuratedManifest:
    """Read-only CURATOR entry point; project old artifacts to IDs/provenance.

    Old features, is_positive, ranking order and targets are not accessed or
    returned. Train/test case order is inherited, never randomized here.
    """
    windows, audit = load_frozen_windows(windows_path, dataset=dataset, expected_sha256=expected_windows_sha256,
                                         expected_count=sum(expected_counts), expected_interval_sources=expected_interval_sources)
    entities = {}
    for row in _csv_rows(entity_catalog_path):
        entity_id, category = _text(row.get('entity_id'), 'entity_id'), _text(row.get('entity_type'), 'entity_type')
        if entity_id in entities and entities[entity_id] != category:
            raise ValueError('catalog entity has conflicting categories')
        entities[entity_id] = category
    ids = tuple(sorted(entities))
    if _ids_digest(ids) != _sha256(expected_catalog_ids_sha256):
        raise ValueError('catalog identity checksum mismatch')
    if dict(Counter(entities.values())) != dict(expected_entity_counts):
        raise ValueError('catalog category population drift')
    catalog = CandidateCatalog(tuple(Entity(i, entities[i]) for i in ids), f'{dataset}-frozen-standard-v1', expected_catalog_ids_sha256)
    raw_train = Path(train_pool_path).read_bytes()
    if hashlib.sha256(raw_train).hexdigest() != _sha256(expected_train_sha256):
        raise ValueError('frozen training pool checksum mismatch')
    train = json.loads(raw_train)
    if train.get('dataset_id') != dataset or train.get('ground_truth_included') is not False:
        raise ValueError('training pool dataset/label-boundary mismatch')
    train_ids = tuple(train['case_ids'])
    if train.get('case_count') != len(train_ids):
        raise ValueError('training pool declared population drift')
    test_ids = []
    with Path(test_rankings_path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                raise ValueError('empty line in frozen test identity source')
            test_ids.append(_text(json.loads(line).get('case_id'), 'test case_id'))
    if _ids_digest(test_ids) != _sha256(expected_test_ids_sha256):
        raise ValueError('frozen test identity checksum mismatch')
    identity = hashlib.sha256((expected_train_sha256 + expected_test_ids_sha256).encode()).hexdigest()
    split = FrozenSplit(dataset, train_ids, tuple(test_ids), identity)
    if (len(train_ids), len(test_ids)) != expected_counts or set(train_ids) | set(test_ids) != {w.case_id for w in windows}:
        raise ValueError('frozen split/window population mismatch')
    audit.update({'train_count': len(train_ids), 'test_count': len(test_ids),
                  'entity_counts': dict(expected_entity_counts), 'catalog_ids_sha256': expected_catalog_ids_sha256,
                  'train_pool_sha256': expected_train_sha256, 'test_ids_sha256': expected_test_ids_sha256,
                  'source_paths': {'windows': str(windows_path), 'catalog': str(entity_catalog_path),
                                   'train_pool': str(train_pool_path), 'test_ids': str(test_rankings_path)},
                  'telemetry_parser_validated': False})
    return CuratedManifest(windows, catalog, split, audit)


def build_standard_manifest(dataset: str) -> CuratedManifest:
    """Read approved standard sources from explicitly configured local paths.

    Set VAST_<DATASET>_{WINDOWS,CATALOG,TRAIN_POOL,TEST_IDS}, with DATASET
    uppercased. TEST_IDS remains the historical JSONL source from which only
    case_id is projected. Relocating files never changes the frozen checks.
    """
    configs = {
        'rcabench': ('54de9844a5316e3a17720b7d9ba51695882652b1eb7c2647208d8909c3e374e1',
                     'db0e5629d7dd1be27e96bbfd825c5e454c1717563c1103c77f880e82a9ea082f',
                     '5a1b36dc958f26f8c0661938c8b767b57480c216900be688048225e84a8be913',
                     '184a33a4efbfadcbdfa6da104112c8fe647288007a4e1690f777c38a84b62013', (995, 427), {'service': 54}, None),
        'aiops2022_pre': ('8fd706fbb74db95230e3dd89f4f5a18fb272e5c2d02cfc3b9b35e1969964a84d',
                         '626de6e032509f2402a3e55a69766924bf4dc0e8430fb62b57bc4953b9708b59',
                         'b6af98a6f361685ef6415fac638ff70ab3330f271201a4eb3a4b4044f10d4276',
                         '51107cc4df777e825311f3e89b171b46e6fe16cad82403ef5acd61f7c8bbcb29', (169, 72), {'service': 10, 'host': 6},
                         {'run_table_exact': 147, 'run_table_level_median': 88, 'run_table_type_median': 6}),
    }
    if dataset not in configs:
        raise ValueError('unknown standard dataset')
    win_sha, cat_sha, train_sha, test_sha, counts, entity_counts, interval_sources = configs[dataset]
    prefix = f'VAST_{dataset.upper()}_'
    sources = {name: os.environ.get(prefix + name) for name in ('WINDOWS', 'CATALOG', 'TRAIN_POOL', 'TEST_IDS')}
    missing = [prefix + name for name, value in sources.items() if value is None or not value.strip()]
    if missing:
        raise ValueError('Missing or empty standard dataset source configuration: ' + ', '.join(missing) +
                         '. Set each environment variable to its approved frozen source file path.')
    return build_curator_manifest(dataset=dataset, windows_path=Path(sources['WINDOWS']), expected_windows_sha256=win_sha,
        entity_catalog_path=Path(sources['CATALOG']), expected_catalog_ids_sha256=cat_sha,
        train_pool_path=Path(sources['TRAIN_POOL']), expected_train_sha256=train_sha,
        test_rankings_path=Path(sources['TEST_IDS']),
        expected_test_ids_sha256=test_sha, expected_counts=counts, expected_entity_counts=entity_counts,
        expected_interval_sources=interval_sources)
