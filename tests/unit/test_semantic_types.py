"""Synthetic-only tests for the sealed, test-only semantic type boundary."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import inspect
import json
import random

import numpy as np
import pytest
import torch

from vast_v3.data.curation import CuratedLabelRecord, CuratedLabels, curate_root_records
from vast_v3.data.manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from vast_v3.data.repository import _digest, manifest_to_dict
from vast_v3.data.schema import Entity


MANIFEST_HASH = 'a' * 64
SEALED_HASH = 'b' * 64
EVIDENCE_HASH = 'c' * 64


def api():
    try:
        return importlib.import_module('vast_v3.data.semantic_types')
    except ModuleNotFoundError:
        pytest.fail('The semantic type projection API is not implemented yet')


def record(case_id='case-a', fault_type='cpu', evidence_sha256=EVIDENCE_HASH):
    return api().SemanticTypeRecord(case_id, fault_type, evidence_sha256)


def projection(records=None, **overrides):
    return api().SemanticTypeProjection(**{
        'records': [record()] if records is None else records,
        'manifest_sha256': MANIFEST_HASH,
        'sealed_identity_sha256': SEALED_HASH,
        **overrides,
    })


def manifest():
    catalog = CandidateCatalog((Entity('service:root-a', 'service'),
                                Entity('service:root-b', 'service')), 'fixture-v1', 'd' * 64)
    ids = ('train-secret', 'test-b', 'test-a')
    windows = tuple(WindowSpec(cid, 'rcabench', 'source-secret-' + cid, '2022-05-03',
        0., 2., 3., 8., 'synthetic frozen interval', 'e' * 64) for cid in ids)
    split = FrozenSplit('rcabench', ('train-secret',), ('test-b', 'test-a'), 'f' * 64)
    return CuratedManifest(windows, catalog, split, {})


def curated_records(frozen=None):
    frozen = manifest() if frozen is None else frozen
    types = {'train-secret': 'training-type-secret', 'test-a': 1, 'test-b': '1'}
    return [CuratedLabelRecord(w.case_id, w.source_id, 'service', ('root-a',),
        types[w.case_id], '/evidence-secret/' + w.case_id + '.json', EVIDENCE_HASH,
        True, 'synthetic complete record') for w in frozen.windows]


def curated():
    frozen = manifest()
    return curate_root_records(frozen, curated_records(frozen))


def export(source):
    assert hasattr(source, 'test_type_projection'), 'CuratedLabels.test_type_projection is missing'
    return source.test_type_projection()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def test_curator_exports_exactly_all_frozen_test_rows_and_only_allowed_fields():
    source = curated()
    result = export(source)
    assert tuple(row.case_id for row in result.records) == ('test-a', 'test-b')
    assert result.records[0].fault_type == 1 and type(result.records[0].fault_type) is int
    assert result.records[1].fault_type == '1' and type(result.records[1].fault_type) is str
    assert result.manifest_sha256 == _digest(manifest_to_dict(source._manifest))
    assert result.sealed_identity_sha256 == source.identity_sha256
    data = result.to_dict()
    assert set(data) == {'schema', 'manifest_sha256', 'sealed_identity_sha256', 'records', 'identity_sha256'}
    assert data['schema'] == 'vast-v3-semantic-types/1'
    assert all(set(row) == {'case_id', 'fault_type', 'evidence_sha256'} for row in data['records'])
    serialized = json.dumps(data)
    for forbidden in ('train-secret', 'training-type-secret', 'root-a', 'source-secret',
                      'evidence-secret', 'raw_roots', 'canonical_roots', 'audit'):
        assert forbidden not in serialized


def test_curator_does_not_accept_a_caller_selected_subset():
    source = curated()
    export(source)
    assert set(inspect.signature(CuratedLabels.test_type_projection).parameters) == {'self'}
    with pytest.raises(TypeError):
        source.test_type_projection(('test-a',))
    with pytest.raises(TypeError):
        source.test_type_projection(case_ids=('test-a',))


def test_curator_does_not_query_labels_or_change_the_paid_budget(monkeypatch):
    source = curated()
    store = source.query_store(budget=2)
    before = store.state_dict()

    def forbidden(*args, **kwargs):
        pytest.fail('Type projection must not use a root-label store')

    monkeypatch.setattr(source, 'query_store', forbidden)
    monkeypatch.setattr(source, 'evaluation_store', forbidden)
    assert len(export(source).records) == 2
    assert store.state_dict() == before


def test_empty_projection_is_legal_without_changing_frozen_split_policy():
    assert projection([]).records == ()
    assert projection([]).to_dict()['records'] == []


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


@pytest.mark.parametrize('field,value', [
    ('case_id', ''), ('case_id', ' \t'), ('case_id', 1), ('case_id', True),
    ('case_id', b'a'), ('case_id', StringSubclass('a')), ('case_id', '\ud800'),
    ('fault_type', ''), ('fault_type', ' \n'), ('fault_type', True), ('fault_type', False),
    ('fault_type', 1.0), ('fault_type', None), ('fault_type', []),
    ('fault_type', IntSubclass(1)), ('fault_type', StringSubclass('cpu')),
    ('fault_type', '\udfff'), ('evidence_sha256', 'A' * 64),
    ('evidence_sha256', 'a' * 63), ('evidence_sha256', 'a' * 65),
    ('evidence_sha256', 'g' * 64), ('evidence_sha256', 'a' * 64 + '\n'),
    ('evidence_sha256', True), ('evidence_sha256', StringSubclass('a' * 64)),
])
def test_record_strictly_rejects_invalid_types_text_and_hashes(field, value):
    args = {'case_id': 'case-a', 'fault_type': 'cpu', 'evidence_sha256': EVIDENCE_HASH}
    args[field] = value
    with pytest.raises(ValueError):
        api().SemanticTypeRecord(**args)


@pytest.mark.parametrize('fault_type', [0, -1, 1, '1', ' CPU ', '故障🔥', 'e\u0301'])
def test_record_preserves_exact_valid_text_and_integer_values(fault_type):
    row = record('案例🧪', fault_type)
    assert row.case_id == '案例🧪'
    assert row.fault_type == fault_type and type(row.fault_type) is type(fault_type)
    assert projection([row]).to_dict()['records'][0]['fault_type'] == fault_type


@pytest.mark.parametrize('field', ['manifest_sha256', 'sealed_identity_sha256'])
@pytest.mark.parametrize('value', [None, True, 1, 'a' * 63, 'A' * 64, 'g' * 64,
                                  'a' * 64 + '\n', StringSubclass('a' * 64)])
def test_projection_requires_exact_lowercase_sha256_strings(field, value):
    with pytest.raises(ValueError):
        projection([], **{field: value})


@pytest.mark.parametrize('records', [None, 1, 'case-a', b'case-a', [{}], [object()]])
def test_projection_rejects_non_record_collections(records):
    with pytest.raises(ValueError):
        api().SemanticTypeProjection(records, MANIFEST_HASH, SEALED_HASH)


def test_projection_rejects_duplicate_case_ids():
    with pytest.raises(ValueError, match='duplicate'):
        projection([record('same', 1), record('same', '1')])


def test_projection_identity_binds_schema_hashes_and_every_record_field():
    result = projection([record('案例🧪', '故障🔥')])
    payload = result.to_dict()
    claimed = payload.pop('identity_sha256')
    assert claimed == result.identity_sha256 == digest(payload)
    assert digest({**payload, 'schema': 'another-schema'}) != claimed
    for field in ('manifest_sha256', 'sealed_identity_sha256'):
        assert projection(result.records, **{field: 'd' * 64}).identity_sha256 != claimed
    for changes in ({'case_id': 'different'}, {'fault_type': 'different'},
                    {'evidence_sha256': 'd' * 64}):
        assert projection([replace(result.records[0], **changes)]).identity_sha256 != claimed
    assert projection([record(fault_type=1)]).identity_sha256 != projection([record(fault_type='1')]).identity_sha256


def test_projection_snapshots_inputs_is_frozen_and_returns_fresh_json():
    row = record()
    rows = [row]
    result = projection(rows)
    original = result.to_dict()
    assert type(result.records) is tuple
    assert result.records[0] is not row
    rows.clear()
    object.__setattr__(row, 'fault_type', 'caller-mutation')
    assert result.to_dict() == original
    assert deepcopy(result) == result
    with pytest.raises(FrozenInstanceError):
        result.manifest_sha256 = 'd' * 64
    with pytest.raises(FrozenInstanceError):
        result.records[0].fault_type = 'changed'
    first, second = result.to_dict(), result.to_dict()
    assert first is not second and first['records'] is not second['records']
    assert first['records'][0] is not second['records'][0]
    first['records'][0]['fault_type'] = 'changed'
    first['records'].append({'case_id': 'extra'})
    first['manifest_sha256'] = 'd' * 64
    assert result.to_dict() == second == original


def test_projection_revalidates_copied_records():
    row = record()
    object.__setattr__(row, 'fault_type', True)
    with pytest.raises(ValueError):
        projection([row])


def test_projection_order_is_canonical_and_does_not_touch_global_rng():
    source = curated()
    rows = [record('z'), record('a', 1)]
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = projection(rows)
    second = projection(reversed(rows))
    exported = export(source)
    assert first == second and first.to_dict() == second.to_dict()
    assert tuple(row.case_id for row in first.records) == ('a', 'z')
    assert exported == export(curate_root_records(manifest(), reversed(curated_records())))
    assert random.getstate() == py_state
    actual_np = np.random.get_state()
    assert actual_np[0] == np_state[0] and np.array_equal(actual_np[1], np_state[1])
    assert actual_np[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize('drift', ['test-record', 'train-record', 'manifest', 'roots', 'audit'])
def test_curator_rejects_content_drift_after_sealing(drift):
    source = curated()
    export(source)
    if drift in ('test-record', 'train-record'):
        target = 'test-a' if drift == 'test-record' else 'train-secret'
        source._records = tuple(replace(row, fault_type='changed') if row.case_id == target else row
                                for row in source._records)
    elif drift == 'manifest':
        object.__setattr__(source._manifest.split, 'source_sha256', '0' * 64)
    elif drift == 'roots':
        source._roots['test-a'] = ('service:root-b',)
    else:
        source.audit['complete'] = False
    with pytest.raises(ValueError, match='sealed.*identity|identity.*sealed'):
        export(source)


def test_root_changes_affect_provenance_but_not_exported_membership_or_types():
    original = export(curated())
    rows = [replace(row, raw_roots=('root-a', 'root-b'), cosource_confirmation='synthetic co-source')
            for row in curated_records()]
    changed = export(curate_root_records(manifest(), rows))
    assert changed.records == original.records
    assert changed.manifest_sha256 == original.manifest_sha256
    assert changed.sealed_identity_sha256 != original.sealed_identity_sha256
    assert changed.identity_sha256 != original.identity_sha256


@pytest.mark.parametrize('target', ['test-a', 'train-secret'])
def test_source_evidence_change_changes_projection_identity(target):
    original = export(curated())
    rows = [replace(row, evidence_sha256='0' * 64) if row.case_id == target else row
            for row in curated_records()]
    changed = export(curate_root_records(manifest(), rows))
    assert changed.identity_sha256 != original.identity_sha256
    assert changed.sealed_identity_sha256 != original.sealed_identity_sha256
    assert [(row.case_id, row.fault_type) for row in changed.records] == [
        (row.case_id, row.fault_type) for row in original.records]
    if target == 'train-secret':
        assert changed.records == original.records
