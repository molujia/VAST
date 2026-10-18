"""Synthetic-only tests for the approved proportional semantic partition."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import inspect
import json
import random

import pytest

from vast_v3.data.manifests import FrozenSplit


FAMILIES = ('revisit', 'mixed_branch_convergence', 'branch', 'convergence', 'chain')
PRESENCE = (True, False, False, False, True)


def api():
    try:
        return importlib.import_module('vast_v3.data.semantic_partition')
    except ModuleNotFoundError:
        pytest.fail('The proportional semantic partition core is not implemented yet')


def split(count=72, dataset='synthetic-fixture'):
    return FrozenSplit(dataset, ('train-b', 'train-a'),
                       tuple(f'test-{index:04}' for index in range(count)), 'a' * 64)


def record(fixture_case_id, fault_type='cpu', **overrides):
    values = dict(case_id=fixture_case_id, fault_type=fault_type, presence=PRESENCE,
                  support='observed', source_sha256='b' * 64,
                  evidence_sha256='c' * 64, profile_version='observable-fixture/1',
                  preprocessing_sha256='d' * 64)
    values.update(overrides)
    return api().CaseStratum(**values)


def strata(frozen, sizes=None):
    if sizes is None:
        return [record(cid, index % 3) for index, cid in enumerate(frozen.test_case_ids)]
    result, start = [], 0
    for fault_type, size in enumerate(sizes):
        result.extend(record(cid, fault_type) for cid in frozen.test_case_ids[start:start + size])
        start += size
    assert start == len(frozen.test_case_ids)
    return result


def build(frozen=None, records=None, actual=None):
    frozen = frozen or split()
    return api().build_semantic_partition(frozen,
        actual_test_case_ids=frozen.test_case_ids if actual is None else actual,
        strata=strata(frozen) if records is None else records)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def priority(seed, **fields):
    return digest({'substream_seed': seed, **fields})


def restore(data, frozen=None, records=None, actual=None):
    frozen = frozen or split()
    return api().SemanticPartition.from_dict(data, split=frozen,
        actual_test_case_ids=frozen.test_case_ids if actual is None else actual,
        strata=strata(frozen) if records is None else records)


def rehash(data):
    data.pop('manifest_sha256')
    data['manifest_sha256'] = digest(data)
    return data


@pytest.mark.parametrize('count', [30, 31, 72, 427])
def test_exact_thirty_and_complete_complement_for_variable_population(count):
    frozen = split(count)
    result = build(frozen)
    assert result.seed == 42
    assert len(result.semantic_case_ids) == 30
    assert len(result.eligible_case_ids) == count - 30
    assert not set(result.semantic_case_ids) & set(result.eligible_case_ids)
    assert set(result.semantic_case_ids) | set(result.eligible_case_ids) == set(frozen.test_case_ids)
    assert sum(q.population for q in result.quotas) == count
    assert sum(q.quota for q in result.quotas) == 30


def test_hand_calculated_hamilton_quotas_without_minimum_seats():
    # N=72: bases [12, 9, 7, 0], remainders [36, 12, 66, 30].
    # Size 19 and size 30 receive the two remaining seats; size 1 gets none.
    frozen = split()
    result = build(frozen, strata(frozen, [30, 22, 19, 1]))
    rows = {q.fault_type: (q.population, q.base_quota, q.remainder, q.quota)
            for q in result.quotas}
    assert rows == {0: (30, 12, 36, 13), 1: (22, 9, 12, 9),
                    2: (19, 7, 66, 8), 3: (1, 0, 30, 0)}


def test_rare_strata_can_receive_zero_seats_and_are_recorded():
    frozen = split(100)
    result = build(frozen, strata(frozen, [97, 1, 1, 1]))
    assert sorted((q.population, q.quota) for q in result.quotas) == [(1, 0), (1, 0), (1, 1), (97, 29)]
    # The final fractional seat ties between the rare singleton strata.
    rare = sorted((q for q in result.quotas if q.population == 1),
                  key=lambda q: (q.tie_priority, q.stratum_id))
    assert [q.quota for q in rare] == [1, 0, 0]


def test_more_than_thirty_strata_use_seeded_ties_and_report_every_zero():
    frozen = split(40)
    result = build(frozen, strata(frozen, [1] * 40))
    assert len(result.quotas) == 40
    assert sum(q.quota == 0 for q in result.quotas) == 10
    ranked = sorted(result.quotas, key=lambda q: (q.tie_priority, q.stratum_id))
    assert [q.quota for q in ranked] == [1] * 30 + [0] * 10
    assert all(q.base_quota == 0 and q.remainder == 30 for q in ranked)


def test_all_joint_dimensions_are_distinct_and_missing_support_not_dropped():
    frozen = split(60)
    variants = [dict(fault_type=1), dict(fault_type='1'),
                dict(fault_type=1, presence=(False,) * 5),
                dict(fault_type=1, support='prior-only'),
                dict(fault_type=1, support='missing', presence=(False,) * 5)]
    records = [record(cid, **variants[index % 5]) for index, cid in enumerate(frozen.test_case_ids)]
    result = build(frozen, records)
    assert len(result.quotas) == 5
    assert {q.support for q in result.quotas} == {'observed', 'prior-only', 'missing'}
    assert {type(q.fault_type) for q in result.quotas} == {int, str}
    assert all(q.population == 12 and q.quota == 6 for q in result.quotas)
    assert result.to_dict()['families'] == list(FAMILIES)


def test_recorded_substreams_reproduce_selection_presentation_and_search():
    frozen = split()
    records = strata(frozen)
    result = build(frozen, records)
    data = result.to_dict()
    assert data['schema'] == 'vast-v3-semantic-partition/2'
    assert data['algorithm'] == 'hamilton-sha256-priority-v1'
    assert len(set(data['substreams'].values())) == 4
    for purpose, seed in data['substreams'].items():
        assert seed == digest({'schema': data['schema'], 'seed': 42,
                               'dataset': frozen.dataset, 'purpose': purpose})
    chosen = []
    for q in result.quotas:
        key = {'fault_type': q.fault_type, 'presence': list(q.presence), 'support': q.support}
        assert q.stratum_id == digest(key)
        assert q.tie_priority == priority(data['substreams']['stratum-tie'], stratum_id=q.stratum_id)
        members = [r.case_id for r in records if (r.fault_type, r.presence, r.support) ==
                   (q.fault_type, q.presence, q.support)]
        ordered = sorted(members, key=lambda cid: (
            priority(data['substreams']['within-stratum'], stratum_id=q.stratum_id, case_id=cid), cid))
        chosen.extend(ordered[:q.quota])
    order = lambda ids, purpose: tuple(sorted(ids, key=lambda cid: (
        priority(data['substreams'][purpose], case_id=cid), cid)))
    assert result.semantic_case_ids == order(chosen, 'semantic-presentation')
    assert result.eligible_case_ids == order(set(frozen.test_case_ids) - set(chosen), 'demo-search')


def test_input_order_and_global_rng_do_not_change_any_identity():
    frozen = split()
    records = strata(frozen)
    random.seed(871)
    before_rng = random.getstate()
    first = build(frozen, records, set(frozen.test_case_ids))
    assert random.getstate() == before_rng
    random.seed(33)
    second = build(replace(frozen, train_case_ids=tuple(reversed(frozen.train_case_ids)),
                           test_case_ids=tuple(reversed(frozen.test_case_ids))),
                   list(reversed(records)), reversed(frozen.test_case_ids))
    assert first.to_dict() == second.to_dict()


def test_dataset_names_separate_all_substreams():
    first = build(split(dataset='one')).to_dict()['substreams']
    second = build(split(dataset='two')).to_dict()['substreams']
    assert all(first[key] != second[key] for key in first)


@pytest.mark.parametrize('actual', [None, (), 'test-0000', ['test-0000'] * 72,
    [f'test-{i:04}' for i in range(71)], [f'test-{i:04}' for i in range(73)],
    [f'test-{i:04}' for i in range(71)] + ['train-a'], [' '] * 72, [1] * 72])
def test_actual_ids_must_be_complete_unique_test_only_strings(actual):
    frozen = split()
    with pytest.raises(ValueError):
        api().build_semantic_partition(frozen, actual_test_case_ids=actual, strata=strata(frozen))


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'train', 'mapping', 'none', 'scalar'])
def test_strata_must_cover_exactly_the_complete_unique_test_population(change):
    frozen = split()
    records = strata(frozen)
    if change == 'missing':
        records.pop()
    elif change == 'extra':
        records.append(record('unknown'))
    elif change == 'duplicate':
        records[-1] = records[0]
    elif change == 'train':
        records[-1] = record('train-a')
    elif change == 'mapping':
        records[0] = {'case_id': frozen.test_case_ids[0]}
    elif change == 'none':
        records = None
    else:
        records = 'test-0000'
    with pytest.raises(ValueError):
        api().build_semantic_partition(frozen, actual_test_case_ids=frozen.test_case_ids, strata=records)


@pytest.mark.parametrize('field,value', [
    ('case_id', ''), ('case_id', ' '), ('case_id', 1),
    ('fault_type', True), ('fault_type', False), ('fault_type', 1.0), ('fault_type', None),
    ('fault_type', ''), ('fault_type', ' '), ('fault_type', []),
    ('presence', [True] * 4), ('presence', [True] * 6), ('presence', [1] * 5),
    ('presence', '00000'), ('presence', None),
    ('support', 'prior_only'), ('support', None), ('support', []),
    ('source_sha256', 'B' * 64), ('source_sha256', 'z' * 64), ('source_sha256', ''),
    ('evidence_sha256', 'c' * 63), ('evidence_sha256', None),
    ('profile_version', ''), ('profile_version', 1),
    ('preprocessing_sha256', 'd' * 65), ('preprocessing_sha256', {}),
])
def test_stratum_record_rejects_malformed_fields(field, value):
    with pytest.raises(ValueError):
        record('test-0000', **{field: value})


@pytest.mark.parametrize('field', ['case_id', 'fault_type', 'profile_version'])
@pytest.mark.parametrize('escaped', [r'"\ud800"', r'"prefix\udfff"'])
def test_stratum_text_rejects_unencodable_surrogates_at_construction(field, escaped):
    with pytest.raises(ValueError, match='UTF-8'):
        record('test-0000', **{field: json.loads(escaped)})


@pytest.mark.parametrize('field', ['dataset', 'train_case_ids', 'test_case_ids', 'actual_test_case_ids'])
def test_partition_rejects_surrogates_in_independent_text_inputs(field):
    frozen = split()
    records = strata(frozen)
    invalid = json.loads(r'"\ud800"')
    actual = frozen.test_case_ids
    if field == 'dataset':
        frozen = replace(frozen, dataset=invalid)
    elif field == 'train_case_ids':
        frozen = replace(frozen, train_case_ids=(invalid,))
    elif field == 'test_case_ids':
        frozen = replace(frozen, test_case_ids=(invalid,) + frozen.test_case_ids[1:])
    else:
        actual = (invalid,) + frozen.test_case_ids[1:]
    with pytest.raises(ValueError, match='UTF-8'):
        build(frozen, records, actual)


def test_ordinary_unicode_text_roundtrips_without_normalization():
    frozen = FrozenSplit('数据集-\U0001f52c', ('训练案例',),
                         tuple(f'案例-\U0001f680-{i:03}' for i in range(31)), 'a' * 64)
    records = [record(cid, fault_type='故障-\U0001f525', profile_version='观测版本-é-e\u0301/1')
               for cid in frozen.test_case_ids]
    result = build(frozen, records)
    data = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert restore(data, frozen, records) == result
    assert data['dataset'] == '数据集-\U0001f52c'
    assert data['strata'][0]['profile_version'] == '观测版本-é-e\u0301/1'
    assert len(result.semantic_case_ids) == 30


def test_fewer_than_thirty_is_a_blocker_and_split_is_required():
    with pytest.raises(ValueError, match='30'):
        build(split(29))
    with pytest.raises(ValueError):
        api().build_semantic_partition({}, actual_test_case_ids=(), strata=())


@pytest.mark.parametrize('field,value', [
    ('source_sha256', 'e' * 64), ('evidence_sha256', 'e' * 64),
    ('profile_version', 'observable-fixture/2'), ('preprocessing_sha256', 'e' * 64),
])
def test_source_metadata_binds_identity_but_never_changes_selection(field, value):
    frozen = split()
    records = strata(frozen)
    original = build(frozen, records)
    records[0] = replace(records[0], **{field: value})
    changed = build(frozen, records)
    assert changed.identity != original.identity
    assert changed.semantic_case_ids == original.semantic_case_ids
    assert changed.eligible_case_ids == original.eligible_case_ids
    assert changed.quotas == original.quotas
    with pytest.raises(ValueError):
        restore(original.to_dict(), frozen, records)


def test_full_split_source_and_training_population_bind_identity_only():
    frozen = split()
    original = build(frozen)
    for changed in [replace(frozen, source_sha256='f' * 64),
                    replace(frozen, train_case_ids=('different-training-id',))]:
        result = build(changed)
        assert result.identity != original.identity
        assert result.semantic_case_ids == original.semantic_case_ids
        with pytest.raises(ValueError):
            restore(original.to_dict(), changed)


def test_mutable_inputs_and_exports_cannot_change_frozen_partition():
    frozen = split()
    bits = list(PRESENCE)
    records = [record(cid, presence=bits) for cid in frozen.test_case_ids]
    actual = list(frozen.test_case_ids)
    result = build(frozen, records, actual)
    identity = result.identity
    bits.clear()
    records.clear()
    actual.clear()
    assert result.strata[0].presence == PRESENCE
    data = result.to_dict()
    data['strata'][0]['presence'].clear()
    data['quotas'].clear()
    data['semantic_case_ids'].clear()
    data['substreams'].clear()
    assert result.identity == identity
    with pytest.raises(FrozenInstanceError):
        result.seed = 7
    with pytest.raises(FrozenInstanceError):
        result.strata[0].support = 'missing'
    with pytest.raises(FrozenInstanceError):
        result.quotas[0].quota = 0


def test_manifest_hash_and_roundtrip_require_independent_inputs():
    frozen = split()
    result = build(frozen)
    data = result.to_dict()
    manifest_hash = data.pop('manifest_sha256')
    assert manifest_hash == result.identity == digest(data)
    assert len(data['strata']) == len(frozen.test_case_ids)
    assert data['exclusion_reasons'] == {cid: 'reserved-independent-semantic-evaluation'
                                        for cid in result.semantic_case_ids}
    data['manifest_sha256'] = manifest_hash
    assert restore(json.loads(json.dumps(data))) == result
    with pytest.raises(ValueError):
        restore(data, actual=frozen.test_case_ids[:-1])
    with pytest.raises(TypeError):
        api().SemanticPartition.from_dict(data)


@pytest.mark.parametrize('field,value', [
    ('seed', True), ('seed', 43), ('semantic_count', 29), ('schema', 'unknown'),
    ('algorithm', 'unknown'), ('split_source_sha256', 'f' * 64),
    ('train_case_ids', []), ('families', list(reversed(FAMILIES))),
    ('exclusion_reasons', {}), ('predictions', {}), ('strata', []), ('quotas', []),
    ('substreams', {}),
])
def test_rehashed_manifest_tampering_is_not_authorized(field, value):
    data = build().to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        restore(rehash(data))


@pytest.mark.parametrize('field', ['semantic_case_ids', 'eligible_case_ids', 'test_case_ids', 'strata', 'quotas'])
def test_rehashed_order_tampering_is_not_authorized(field):
    data = build().to_dict()
    data[field].reverse()
    with pytest.raises(ValueError):
        restore(rehash(data))


def test_rehashed_case_metadata_and_quota_tampering_are_not_authorized():
    for container, field, value in [('strata', 'evidence_sha256', 'f' * 64),
                                   ('quotas', 'quota', 99), ('quotas', 'remainder', True)]:
        data = build().to_dict()
        data[container][0][field] = value
        with pytest.raises(ValueError):
            restore(rehash(data))


def test_api_does_not_accept_outcomes_or_read_actual_pool_values():
    class IDsOnly(dict):
        def __getitem__(self, key):
            raise AssertionError('actual case values must not be read')

        def items(self):
            raise AssertionError('actual case values must not be read')

        def values(self):
            raise AssertionError('actual case values must not be read')

    frozen = split()
    assert len(build(frozen, actual=IDsOnly.fromkeys(frozen.test_case_ids)).semantic_case_ids) == 30
    assert set(inspect.signature(api().build_semantic_partition).parameters) == {
        'split', 'actual_test_case_ids', 'strata'}
    for field in ('true_roots', 'root_counts', 'predictions', 'scores', 'model',
                  'checkpoint', 'exclusions', 'seed', 'semantic_count'):
        with pytest.raises(TypeError):
            api().build_semantic_partition(frozen, actual_test_case_ids=frozen.test_case_ids,
                                           strata=strata(frozen), **{field: None})
