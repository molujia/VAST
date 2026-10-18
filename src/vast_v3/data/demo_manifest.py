"""Model-blind semantic holdout and demo order from public frozen split IDs.

No case observations, labels, predictions, root counts or outcomes enter this
module. Generation does not read files or run inference. Callers must durably
freeze the returned identity before prediction and revalidate the actual pool
when binding the already-frozen model assets to a search ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Iterable, Mapping

from .manifests import FrozenSplit


SCHEMA = 'vast-v3-demo-partition/1'
ALGORITHM = 'sha256-priority-v1'
SEED = 42
SEMANTIC_COUNT = 30


def _bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hash(value) -> str:
    return sha256(_bytes(value)).hexdigest()


def _ids(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{name} must be a collection of case IDs')
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f'{name} must be a collection of case IDs') from exc
    if any(not isinstance(cid, str) or not cid.strip() for cid in values):
        raise ValueError(f'{name} requires nonempty string case IDs')
    if len(values) != len(set(values)):
        raise ValueError(f'{name} contains duplicate case IDs')
    return values


def _substream(dataset: str, purpose: str) -> str:
    return _hash({'schema': SCHEMA, 'seed': SEED, 'dataset': dataset, 'purpose': purpose})


def _order(ids: Iterable[str], substream_seed: str) -> tuple[str, ...]:
    # Hash priorities fix the full algorithm independently of Python RNG state
    # or container iteration order. IDs are keys only, never parsed as features.
    return tuple(sorted(ids, key=lambda cid: (
        _hash({'substream_seed': substream_seed, 'case_id': cid}), cid)))


@dataclass(frozen=True)
class DemoManifest:
    """Deterministic partition; use build_demo_manifest to check a current pool."""

    split: FrozenSplit
    seed: int = field(default=SEED, init=False)
    semantic_seed: str = field(init=False)
    search_seed: str = field(init=False)
    semantic_case_ids: tuple[str, ...] = field(init=False)
    eligible_case_ids: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.split, FrozenSplit):
            raise ValueError('a verified public FrozenSplit is required')
        _ids(self.split.train_case_ids, 'frozen training population')
        test_ids = _ids(self.split.test_case_ids, 'frozen test population')
        if len(test_ids) < SEMANTIC_COUNT:
            raise ValueError('at least 30 frozen test cases are required for semantic evaluation')
        semantic_seed = _substream(self.split.dataset, 'semantic-evaluation')
        search_seed = _substream(self.split.dataset, 'demo-search')
        semantic = _order(test_ids, semantic_seed)[:SEMANTIC_COUNT]
        remaining = set(test_ids) - set(semantic)
        object.__setattr__(self, 'semantic_seed', semantic_seed)
        object.__setattr__(self, 'search_seed', search_seed)
        object.__setattr__(self, 'semantic_case_ids', semantic)
        object.__setattr__(self, 'eligible_case_ids', _order(remaining, search_seed))

    def _payload(self) -> dict:
        return {'schema': SCHEMA, 'algorithm': ALGORITHM,
                'dataset': self.split.dataset, 'split_source_sha256': self.split.source_sha256,
                'train_case_ids': list(self.split.train_case_ids),
                'test_case_ids': list(self.split.test_case_ids),
                'seed': self.seed, 'semantic_count': SEMANTIC_COUNT,
                'semantic_seed': self.semantic_seed, 'search_seed': self.search_seed,
                'semantic_case_ids': list(self.semantic_case_ids),
                'eligible_case_ids': list(self.eligible_case_ids),
                'exclusion_reasons': {cid: 'reserved-independent-semantic-evaluation'
                                      for cid in self.semantic_case_ids}}

    @property
    def identity(self) -> str:
        """SHA-256 of the entire canonical payload, not just selected case IDs."""
        return _hash(self._payload())

    def to_dict(self) -> dict:
        payload = self._payload()
        return {**payload, 'manifest_sha256': _hash(payload)}

    @classmethod
    def from_dict(cls, data: Mapping, *, split: FrozenSplit,
                  actual_test_case_ids: Iterable[str]) -> DemoManifest:
        """Rebuild against an independently supplied split and live pool IDs.

        A newly self-hashed edited manifest cannot change the partition, source,
        order, seed, algorithm or exclusions. This is not a cryptographic proof
        that the external frozen split itself predates model prediction.
        """
        expected = build_demo_manifest(split, actual_test_case_ids=actual_test_case_ids)
        try:
            matches = isinstance(data, Mapping) and _bytes(dict(data)) == _bytes(expected.to_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid demo manifest serialization') from exc
        if not matches:
            raise ValueError('demo manifest differs from the frozen model-blind partition or identity')
        return expected


def build_demo_manifest(split: FrozenSplit, *, actual_test_case_ids: Iterable[str]) -> DemoManifest:
    """Reserve 30 semantic cases first, then freeze the entire remaining order.

    Actual IDs must exactly cover the frozen test pool; missing, extra, repeated
    or training IDs are blockers, never silently dropped. A mapping is iterated
    for keys only. No fixed dataset population count is assumed.
    """
    result = DemoManifest(split)
    actual = _ids(actual_test_case_ids, 'actual test population')
    expected = set(split.test_case_ids)
    if set(actual) != expected:
        missing, extra = expected - set(actual), set(actual) - expected
        raise ValueError(f'actual test population differs from frozen split: '
                         f'{len(missing)} missing, {len(extra)} extra')
    return result


def bind_search_manifest(partition: DemoManifest, *, split: FrozenSplit,
                         actual_test_case_ids: Iterable[str], search_run_id: str,
                         candidate_ids: Iterable[str], code_revision: str,
                         config_hash: str, preprocessing_hash: str, checkpoint_hash: str,
                         catalog_hash: str, data_hash: str):
    """Bind a revalidated partition to the existing ledger's complete identity.

    The caller must verify artifact hashes and final training completion before
    calling; this pure metadata function never opens a checkpoint or label store.
    """
    from ..evaluation.search_ledger import SearchManifest

    if not isinstance(partition, DemoManifest):
        raise ValueError('a frozen DemoManifest is required')
    verified = DemoManifest.from_dict(partition.to_dict(), split=split,
                                     actual_test_case_ids=actual_test_case_ids)
    if not verified.eligible_case_ids:
        raise ValueError('empty demo-search pool after reserving 30 semantic cases')
    return SearchManifest(dataset=split.dataset, search_run_id=search_run_id,
        eligible_case_ids=verified.eligible_case_ids,
        excluded_semantic_ids=verified.semantic_case_ids,
        candidate_ids=_ids(candidate_ids, 'candidate population'),
        case_manifest_hash=verified.identity, code_revision=code_revision,
        config_hash=config_hash, preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash, catalog_hash=catalog_hash, data_hash=data_hash,
        seed=verified.seed)
