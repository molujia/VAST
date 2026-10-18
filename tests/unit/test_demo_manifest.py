"""Model-blind partitions of synthetic frozen test IDs; no real cases loaded."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import inspect
import json
import random

import pytest

from vast_v3.data.manifests import FrozenSplit


def api():
    try:
        return importlib.import_module('vast_v3.data.demo_manifest')
    except ModuleNotFoundError:
        pytest.fail('The label-blind demo manifest generator is not implemented yet')


def split(count=37, dataset='synthetic-fixture'):
    return FrozenSplit(dataset, ('train-b', 'train-a'),
                       tuple(f'test-{index:04}' for index in range(count)), 'a' * 64)


def build(frozen=None, actual=None):
    frozen = frozen or split()
    return api().build_demo_manifest(frozen, actual_test_case_ids=(
        frozen.test_case_ids if actual is None else actual))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind(partition, frozen=None, actual=None, **overrides):
    frozen = frozen or split()
    values = dict(split=frozen, actual_test_case_ids=(
        frozen.test_case_ids if actual is None else actual),
        search_run_id='synthetic-demo-only', candidate_ids=('service:a', 'host:b'),
        code_revision='fixture-code', config_hash='1' * 64,
        preprocessing_hash='2' * 64, checkpoint_hash='3' * 64,
        catalog_hash='4' * 64, data_hash='5' * 64)
    values.update(overrides)
    return api().bind_search_manifest(partition, **values)


@pytest.mark.parametrize('count', [30, 31, 37, 72, 427])
def test_reserves_exactly_thirty_and_preserves_complete_variable_pool(count):
    frozen = split(count)
    result = build(frozen)
    assert len(result.semantic_case_ids) == 30
    assert len(result.eligible_case_ids) == count - 30
    assert not set(result.semantic_case_ids) & set(result.eligible_case_ids)
    assert set(result.semantic_case_ids) | set(result.eligible_case_ids) == set(frozen.test_case_ids)
    assert not (set(result.semantic_case_ids) | set(result.eligible_case_ids)) & set(frozen.train_case_ids)


def test_recorded_independent_substreams_reproduce_both_orders():
    frozen = split()
    result = build(frozen)
    record = result.to_dict()
    assert record['seed'] == 42
    assert record['algorithm'] == 'sha256-priority-v1'
    assert record['semantic_seed'] != record['search_seed']
    for purpose, key in [('semantic-evaluation', 'semantic_seed'), ('demo-search', 'search_seed')]:
        assert record[key] == digest({'schema': record['schema'], 'seed': 42,
                                      'dataset': frozen.dataset, 'purpose': purpose})
    order = lambda ids, key: tuple(sorted(ids, key=lambda cid: (
        digest({'substream_seed': record[key], 'case_id': cid}), cid)))
    assert result.semantic_case_ids == order(frozen.test_case_ids, 'semantic_seed')[:30]
    remaining = set(frozen.test_case_ids) - set(result.semantic_case_ids)
    assert result.eligible_case_ids == order(remaining, 'search_seed')


def test_generation_is_global_rng_and_actual_iteration_order_independent():
    frozen = split()
    random.seed(99)
    state = random.getstate()
    first = build(frozen, list(reversed(frozen.test_case_ids)))
    assert random.getstate() == state
    random.seed(200)
    second = build(frozen, set(frozen.test_case_ids))
    assert first.to_dict() == second.to_dict()


def test_dataset_names_separate_substreams():
    first = build(split(dataset='dataset-one')).to_dict()
    second = build(split(dataset='dataset-two')).to_dict()
    assert first['semantic_seed'] != second['semantic_seed']
    assert first['search_seed'] != second['search_seed']


def test_generator_reads_only_public_ids_and_exposes_no_outcome_input():
    class IdOnlyPool(dict):
        def __getitem__(self, key):
            raise AssertionError('Case values, predictions and labels must not be read')

        def values(self):
            raise AssertionError('Case values must not be read')

        def items(self):
            raise AssertionError('Case values must not be read')

    frozen = split()
    result = build(frozen, IdOnlyPool.fromkeys(frozen.test_case_ids))
    assert len(result.semantic_case_ids) == 30
    parameters = inspect.signature(api().build_demo_manifest).parameters
    assert set(parameters) == {'split', 'actual_test_case_ids'}
    for field in ('true_roots', 'predictions', 'scores', 'exclusions'):
        with pytest.raises(TypeError):
            api().build_demo_manifest(frozen, actual_test_case_ids=frozen.test_case_ids,
                                      **{field: {}})


@pytest.mark.parametrize('actual', [
    (), ('test-0000',) * 37, ('unknown',), 'test-0000',
    tuple(f'test-{i:04}' for i in range(36)),
    tuple(f'test-{i:04}' for i in range(38)),
    tuple(f'test-{i:04}' for i in range(36)) + ('train-a',),
])
def test_actual_pool_must_match_complete_frozen_test_population(actual):
    with pytest.raises(ValueError):
        build(actual=actual)


def test_fewer_than_thirty_test_cases_are_a_blocker_not_a_smaller_holdout():
    with pytest.raises(ValueError, match='30'):
        build(split(29))


def test_manifest_hash_covers_complete_identity_and_records_every_exclusion():
    result = build()
    record = result.to_dict()
    expected_hash = record.pop('manifest_sha256')
    assert expected_hash == result.identity == digest(record)
    assert record['train_case_ids'] == ['train-b', 'train-a']
    assert record['test_case_ids'] == list(split().test_case_ids)
    assert record['split_source_sha256'] == 'a' * 64
    assert record['exclusion_reasons'] == {
        cid: 'reserved-independent-semantic-evaluation' for cid in result.semantic_case_ids}
    changed_source = build(replace(split(), source_sha256='b' * 64))
    changed_train = build(replace(split(), train_case_ids=('other-training-case',)))
    assert changed_source.identity != result.identity
    assert changed_train.identity != result.identity
    # Metadata changes cannot retrospectively change the model-blind selection.
    assert changed_source.semantic_case_ids == result.semantic_case_ids


def test_manifest_and_exported_containers_cannot_mutate_frozen_identity():
    result = build()
    before = result.identity
    with pytest.raises(FrozenInstanceError):
        result.seed = 99
    record = result.to_dict()
    record['eligible_case_ids'].clear()
    record['exclusion_reasons'].clear()
    assert result.identity == before
    assert len(result.eligible_case_ids) == 7


def test_roundtrip_revalidates_frozen_source_and_current_actual_pool():
    frozen = split()
    result = build(frozen)
    restored = api().DemoManifest.from_dict(json.loads(json.dumps(result.to_dict())),
        split=frozen, actual_test_case_ids=list(reversed(frozen.test_case_ids)))
    assert restored == result
    with pytest.raises(ValueError):
        api().DemoManifest.from_dict(result.to_dict(), split=frozen,
            actual_test_case_ids=frozen.test_case_ids[:-1])
    with pytest.raises(ValueError):
        api().DemoManifest.from_dict(result.to_dict(),
            split=replace(frozen, source_sha256='f' * 64), actual_test_case_ids=frozen.test_case_ids)


@pytest.mark.parametrize('field,value', [
    ('seed', 43), ('seed', True), ('semantic_count', 29),
    ('schema', 'unknown'), ('algorithm', 'unknown'),
    ('semantic_seed', 'f' * 64), ('search_seed', 'f' * 64),
    ('split_source_sha256', 'f' * 64), ('train_case_ids', ['different']),
    ('exclusion_reasons', {}), ('predictions', {}),
])
def test_recomputed_self_hash_does_not_authorize_manifest_tampering(field, value):
    frozen = split()
    record = build(frozen).to_dict()
    record[field] = value
    record.pop('manifest_sha256')
    record['manifest_sha256'] = digest(record)
    with pytest.raises(ValueError):
        api().DemoManifest.from_dict(record, split=frozen, actual_test_case_ids=frozen.test_case_ids)


@pytest.mark.parametrize('field', ['semantic_case_ids', 'eligible_case_ids', 'test_case_ids'])
def test_recomputed_self_hash_does_not_authorize_case_reordering(field):
    frozen = split()
    record = build(frozen).to_dict()
    record[field].reverse()
    record.pop('manifest_sha256')
    record['manifest_sha256'] = digest(record)
    with pytest.raises(ValueError):
        api().DemoManifest.from_dict(record, split=frozen, actual_test_case_ids=frozen.test_case_ids)


def test_search_binding_uses_frozen_partition_and_all_supplied_version_hashes(tmp_path):
    result = build()
    manifest = bind(result)
    assert manifest.case_manifest_hash == result.identity
    assert manifest.eligible_case_ids == result.eligible_case_ids
    assert manifest.excluded_semantic_ids == result.semantic_case_ids
    assert manifest.candidate_ids == ('service:a', 'host:b')
    assert manifest.config_hash == '1' * 64
    assert manifest.preprocessing_hash == '2' * 64
    assert manifest.checkpoint_hash == '3' * 64
    assert manifest.catalog_hash == '4' * 64
    assert manifest.data_hash == '5' * 64
    assert manifest.code_revision == 'fixture-code'
    from vast_v3.evaluation.search_ledger import SearchLedger
    ledger = SearchLedger(tmp_path / 'attempts.jsonl', manifest)
    ledger.append_attempt(case_id=result.eligible_case_ids[0], stage='search', status='error',
        started_at='2026-10-16T00:00:00Z', finished_at='2026-10-16T00:00:01Z',
        error_reason='synthetic fixture only')
    assert ledger.attempts[0]['case_manifest_hash'] == result.identity


def test_search_binding_rejects_drift_after_partition_was_generated():
    result = build()
    with pytest.raises(ValueError):
        bind(result, actual=split().test_case_ids[:-1])
    with pytest.raises(ValueError):
        bind(result, frozen=replace(split(), source_sha256='f' * 64))


def test_search_binding_does_not_silently_accept_an_empty_eligible_pool():
    frozen = split(30)
    with pytest.raises(ValueError, match='empty'):
        bind(build(frozen), frozen=frozen)


@pytest.mark.parametrize('field', ['config_hash', 'preprocessing_hash', 'checkpoint_hash',
                                   'catalog_hash', 'data_hash'])
def test_search_binding_requires_each_version_hash(field):
    with pytest.raises(ValueError):
        bind(build(), **{field: ''})


@pytest.mark.parametrize('candidate_ids', ['ab', None])
def test_search_binding_rejects_a_scalar_instead_of_candidate_id_collection(candidate_ids):
    with pytest.raises(ValueError):
        bind(build(), candidate_ids=candidate_ids)
