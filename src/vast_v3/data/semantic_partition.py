"""Pure, prediction-blind proportional semantic reservation for E06.

Inputs are an independent frozen split, actual test IDs and already-produced
joint strata. This module reads no files, telemetry, root sets, models or
predictions, and cannot certify how the upstream evidence was produced. Its
identity binds all supplied provenance without using provenance to select cases.
The historical ID-only demo-partition/1 interface remains unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
import re
from typing import Iterable, Mapping

from ..encoders.fragments import FAMILIES
from .manifests import FrozenSplit


SCHEMA = 'vast-v3-semantic-partition/2'
ALGORITHM = 'hamilton-sha256-priority-v1'
SEED = 42
SEMANTIC_COUNT = 30


def _bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hash(value) -> str:
    return sha256(_bytes(value)).hexdigest()


def _text(value, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError(f'{name} must be UTF-8 encodable text without surrogate code points') from exc
    return value


def _digest(value, name: str) -> str:
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _collection(values, name: str) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{name} must be a collection, not a scalar')
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(f'{name} must be a collection') from exc


def _ids(values, name: str) -> tuple[str, ...]:
    values = _collection(values, name)
    for value in values:
        _text(value, name)
    if len(values) != len(set(values)):
        raise ValueError(f'{name} contains duplicate case IDs')
    return values


def _stratum(fault_type, presence, support) -> dict:
    # JSON preserves integer 1 versus string "1". Booleans are rejected before
    # this key is constructed, avoiding Python's equality of True and 1.
    return {'fault_type': fault_type, 'presence': list(presence), 'support': support}


def _substream(dataset: str, purpose: str) -> str:
    return _hash({'schema': SCHEMA, 'seed': SEED, 'dataset': dataset, 'purpose': purpose})


def _priority(seed: str, **fields) -> str:
    return _hash({'substream_seed': seed, **fields})


def _order(ids: Iterable[str], seed: str, **fields) -> tuple[str, ...]:
    return tuple(sorted(ids, key=lambda cid: (_priority(seed, case_id=cid, **fields), cid)))


@dataclass(frozen=True, slots=True)
class CaseStratum:
    """One precomputed case descriptor, not an expert causal annotation.

    ``presence`` follows FAMILIES exactly. ``source_sha256`` binds the source
    projection and ``evidence_sha256`` binds its observable profile evidence;
    their contents and observation semantics remain the producer's obligation.
    Even missing evidence needs an explicit, hash-bound missing-evidence record.
    """

    case_id: str
    fault_type: int | str
    presence: tuple[bool, ...]
    support: str
    source_sha256: str
    evidence_sha256: str
    profile_version: str
    preprocessing_sha256: str

    def __post_init__(self):
        _text(self.case_id, 'case_id')
        if type(self.fault_type) not in (int, str):
            raise ValueError('fault_type must be an integer or nonempty string, never bool')
        if type(self.fault_type) is str:
            _text(self.fault_type, 'fault_type')
        presence = _collection(self.presence, 'presence')
        if len(presence) != len(FAMILIES) or any(type(bit) is not bool for bit in presence):
            raise ValueError('presence requires exactly five booleans in FAMILIES order')
        object.__setattr__(self, 'presence', presence)
        if type(self.support) is not str or self.support not in ('observed', 'prior-only', 'missing'):
            raise ValueError('support must be observed, prior-only or missing')
        _text(self.profile_version, 'profile_version')
        for name in ('source_sha256', 'evidence_sha256', 'preprocessing_sha256'):
            _digest(getattr(self, name), name)

    def to_dict(self) -> dict:
        return {'case_id': self.case_id,
                **_stratum(self.fault_type, self.presence, self.support),
                'source_sha256': self.source_sha256, 'evidence_sha256': self.evidence_sha256,
                'profile_version': self.profile_version,
                'preprocessing_sha256': self.preprocessing_sha256}


@dataclass(frozen=True, slots=True)
class _StratumQuota:
    fault_type: int | str
    presence: tuple[bool, ...]
    support: str
    population: int
    base_quota: int
    remainder: int
    quota: int
    stratum_id: str
    tie_priority: str

    def to_dict(self) -> dict:
        return {**_stratum(self.fault_type, self.presence, self.support),
                'population': self.population, 'base_quota': self.base_quota,
                'remainder': self.remainder, 'quota': self.quota,
                'stratum_id': self.stratum_id, 'tie_priority': self.tie_priority}


@dataclass(frozen=True, slots=True)
class SemanticPartition:
    """Verified immutable snapshot; every construction checks all three inputs.

    Split population order is canonicalized because IDs are set membership, not
    sampling priority. The source checksum still binds the exact source bytes.
    No caller-owned mutable container or FrozenSplit reference is retained.
    """

    split: InitVar[FrozenSplit]
    actual_test_case_ids: InitVar[Iterable[str]]
    strata: tuple[CaseStratum, ...]
    dataset: str = field(init=False)
    split_source_sha256: str = field(init=False)
    train_case_ids: tuple[str, ...] = field(init=False)
    test_case_ids: tuple[str, ...] = field(init=False)
    seed: int = field(default=SEED, init=False)
    tie_seed: str = field(init=False)
    within_seed: str = field(init=False)
    semantic_seed: str = field(init=False)
    search_seed: str = field(init=False)
    quotas: tuple[_StratumQuota, ...] = field(init=False)
    semantic_case_ids: tuple[str, ...] = field(init=False)
    eligible_case_ids: tuple[str, ...] = field(init=False)

    def __post_init__(self, split: FrozenSplit, actual_test_case_ids: Iterable[str]):
        if not isinstance(split, FrozenSplit):
            raise ValueError('an independently verified FrozenSplit is required')
        # Revalidate and snapshot rather than trusting a caller-owned reference.
        frozen = FrozenSplit(_text(split.dataset, 'dataset'),
            tuple(sorted(_ids(split.train_case_ids, 'frozen training population'))),
            tuple(sorted(_ids(split.test_case_ids, 'frozen test population'))),
            _digest(split.source_sha256, 'split source'))
        population = len(frozen.test_case_ids)
        if population < SEMANTIC_COUNT:
            raise ValueError('at least 30 frozen test cases are required')
        actual = _ids(actual_test_case_ids, 'actual test population')
        expected = set(frozen.test_case_ids)
        if set(actual) != expected:
            raise ValueError('actual test population differs from the complete frozen test population')
        records = _collection(self.strata, 'strata')
        if any(type(record) is not CaseStratum for record in records):
            raise ValueError('strata must contain only complete CaseStratum records')
        record_ids = _ids((record.case_id for record in records), 'stratum population')
        if set(record_ids) != expected:
            raise ValueError('stratum population differs from the complete frozen test population')
        records = tuple(sorted(records, key=lambda record: record.case_id))
        for name in ('dataset', 'train_case_ids', 'test_case_ids'):
            object.__setattr__(self, name, getattr(frozen, name))
        object.__setattr__(self, 'split_source_sha256', frozen.source_sha256)
        object.__setattr__(self, 'strata', records)
        for name, purpose in [('tie_seed', 'stratum-tie'), ('within_seed', 'within-stratum'),
                              ('semantic_seed', 'semantic-presentation'), ('search_seed', 'demo-search')]:
            object.__setattr__(self, name, _substream(frozen.dataset, purpose))

        groups = defaultdict(list)
        for record in records:
            groups[(record.fault_type, record.presence, record.support)].append(record.case_id)
        rows = []
        for (fault_type, presence, support), members in groups.items():
            base, remainder = divmod(SEMANTIC_COUNT * len(members), population)
            stratum_id = _hash(_stratum(fault_type, presence, support))
            rows.append(_StratumQuota(fault_type, presence, support, len(members), base,
                remainder, base, stratum_id, _priority(self.tie_seed, stratum_id=stratum_id)))
        remaining_seats = SEMANTIC_COUNT - sum(row.base_quota for row in rows)
        ranked = sorted(rows, key=lambda row: (-row.remainder, row.tie_priority, row.stratum_id))
        awarded = {row.stratum_id for row in ranked[:remaining_seats]}
        quotas, selected = [], []
        for row in sorted(rows, key=lambda row: row.stratum_id):
            quota = row.base_quota + int(row.stratum_id in awarded)
            quotas.append(_StratumQuota(row.fault_type, row.presence, row.support,
                row.population, row.base_quota, row.remainder, quota, row.stratum_id, row.tie_priority))
            members = groups[(row.fault_type, row.presence, row.support)]
            selected.extend(_order(members, self.within_seed, stratum_id=row.stratum_id)[:quota])
        object.__setattr__(self, 'quotas', tuple(quotas))
        object.__setattr__(self, 'semantic_case_ids', _order(selected, self.semantic_seed))
        object.__setattr__(self, 'eligible_case_ids', _order(expected - set(selected), self.search_seed))

    def _payload(self) -> dict:
        return {'schema': SCHEMA, 'algorithm': ALGORITHM, 'dataset': self.dataset,
                'families': list(FAMILIES), 'seed': self.seed, 'semantic_count': SEMANTIC_COUNT,
                'split_source_sha256': self.split_source_sha256,
                'train_case_ids': list(self.train_case_ids), 'test_case_ids': list(self.test_case_ids),
                'strata': [record.to_dict() for record in self.strata],
                'quotas': [quota.to_dict() for quota in self.quotas],
                'substreams': {'stratum-tie': self.tie_seed, 'within-stratum': self.within_seed,
                               'semantic-presentation': self.semantic_seed, 'demo-search': self.search_seed},
                'semantic_case_ids': list(self.semantic_case_ids),
                'eligible_case_ids': list(self.eligible_case_ids),
                'exclusion_reasons': {cid: 'reserved-independent-semantic-evaluation'
                                      for cid in self.semantic_case_ids}}

    @property
    def identity(self) -> str:
        """Digest of the complete canonical input snapshot and derived partition."""
        return _hash(self._payload())

    def to_dict(self) -> dict:
        payload = self._payload()
        return {**payload, 'manifest_sha256': _hash(payload)}

    @classmethod
    def from_dict(cls, data: Mapping, *, split: FrozenSplit,
                  actual_test_case_ids: Iterable[str], strata: Iterable[CaseStratum]) -> SemanticPartition:
        """Recompute using independent inputs; never authorize edits by self-hash.

        This verifies identity, not a timestamp or freeze receipt. The caller is
        responsible for freezing these inputs before any prediction or search.
        """
        expected = build_semantic_partition(split, actual_test_case_ids=actual_test_case_ids, strata=strata)
        try:
            matches = isinstance(data, Mapping) and _bytes(dict(data)) == _bytes(expected.to_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid semantic partition serialization') from exc
        if not matches:
            raise ValueError('semantic partition differs from independently supplied inputs or derived identity')
        return expected


def build_semantic_partition(split: FrozenSplit, *, actual_test_case_ids: Iterable[str],
                             strata: Iterable[CaseStratum]) -> SemanticPartition:
    """Reserve exactly 30 proportionally; order the full complement independently.

    Missing, duplicate, extra and training records are blockers. Rare strata may
    receive zero seats; all nonempty strata and their quotas remain in the audit.
    """
    return SemanticPartition(split=split, actual_test_case_ids=actual_test_case_ids, strata=strata)
