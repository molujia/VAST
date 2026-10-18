"""Synthetic raw-observation disk cache; never benchmark data or root labels."""
from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import os

import pytest

from vast_v3.data import adapters, repository
from vast_v3.data.manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from vast_v3.data.schema import Entity
from vast_v3.data.serialization import serialize_observable


def cache_api():
    assert importlib.util.find_spec('vast_v3.data.case_cache'), 'observable disk cache is not implemented'
    return importlib.import_module('vast_v3.data.case_cache')


def frozen_manifest():
    catalog = CandidateCatalog((Entity('service:a', 'service'), Entity('service:b', 'service'),
                                Entity('host:h', 'host')), 'cache-fixture/1', 'a'*64)
    ids = ('train-b', 'train-a', 'test')
    windows = tuple(WindowSpec(cid, 'aiops2022_pre', cid, 'day', 0., 2., 3., 8.,
                               'synthetic reference', 'b'*64) for cid in ids)
    return CuratedManifest(windows, catalog, FrozenSplit('aiops2022_pre', ids[:2], ids[2:], 'c'*64), {})


def write_sources(root):
    base = root/'day/cloudbed'
    rows = {'metric/container/cpu.csv': 'timestamp,cmdb_id,kpi_name,value\n1,h.a-0,cpu,2\n4,h.a-0,cpu,9\n4,h.a-0,cpu,11\n',
            'metric/node/cpu.csv': 'timestamp,cmdb_id,kpi_name,value\n1,h,cpu,3\n',
            'log/all/log_filebeat-one.csv': 'timestamp,cmdb_id,value\n4,a-0,first\n4,a-0,second\n',
            'trace/all/trace_jaeger-span.csv': 'timestamp,cmdb_id,span_id,trace_id,parent_span,duration\n2500,a-0,p,t,,1\n4100,external-0,c,t,p,2\n'}
    for relative, text in rows.items():
        path = base/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return base


def setup_cache(tmp_path, *, capacity=10_000_000, pinned=True, **kwargs):
    api = cache_api()
    raw = tmp_path/'raw'
    write_sources(raw)
    cache = api.ObservableCaseCache(tmp_path/'cache', max_bytes=capacity)
    repo = repository.DatasetRepository.from_curated(frozen_manifest(), raw, case_cache=cache,
                                                     max_cached_cases=1, **kwargs)
    pins = repo.pin_source_content() if pinned else None
    return repo, cache, pins


def fresh(repo, cache, pins, **kwargs):
    return repository.DatasetRepository.from_curated(kwargs.pop('manifest', repo.manifest), repo.raw_root,
        expected_source_sha256=pins, case_cache=cache, max_cached_cases=1, **kwargs)


def test_lossless_cache_survives_lru_and_new_repository_without_adapter(tmp_path, monkeypatch):
    repo, cache, pins = setup_cache(tmp_path)
    expected = repo['train-b']; expected_audit = repo.case_audit('train-b')
    uncached = repository.DatasetRepository.from_curated(repo.manifest, repo.raw_root,
                                                        expected_source_sha256=pins)
    assert serialize_observable(uncached['train-b']) == serialize_observable(expected)
    assert uncached.case_audit('train-b') == expected_audit
    repo['train-a']
    assert repo.loaded_case_ids == ('train-a',)
    monkeypatch.setattr(adapters, 'load_aiops22_case', lambda *a, **k: pytest.fail('cache hit parsed raw CSV'))
    assert serialize_observable(repo['train-b']) == serialize_observable(expected)
    cold = fresh(repo, cache, pins)
    assert cold['train-b'] == expected
    assert cold.case_audit('train-b') == expected_audit
    assert cache.stats['hits'] == 2 and cache.stats['writes'] == 2
    assert [log.text for log in expected.logs] == ['first', 'second']
    assert any(not entity.candidate for entity in expected.entities)
    assert any(series.modality == 'trace' for series in expected.numeric)
    assert not any('cache' in key for key in expected_audit)


def test_default_repository_never_creates_disk_cache(tmp_path):
    write_sources(tmp_path/'raw')
    repo = repository.DatasetRepository.from_curated(frozen_manifest(), tmp_path/'raw')
    repo['train-b']
    assert not (tmp_path/'cache').exists()


def test_cache_requires_content_pin_before_any_case_access(tmp_path):
    repo, cache, _ = setup_cache(tmp_path, pinned=False)
    with pytest.raises(ValueError, match='content-pinned'):
        repo['train-b']
    assert not list(cache.root.glob('*.json.gz'))
    repo.pin_source_content()
    assert repo['train-b'].case_id == 'train-b'


@pytest.mark.parametrize('mutation', ['source', 'inventory', 'wrong_pin'])
def test_cache_hit_cannot_bypass_source_validation(tmp_path, mutation):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']
    other = fresh(repo, cache, pins)
    path = repo.raw_root/'day/cloudbed/metric/node/cpu.csv'
    if mutation == 'source':
        path.write_text(path.read_text()+'4,h,cpu,5\n')
    elif mutation == 'inventory':
        path.with_name('added.csv').write_text(path.read_text())
    else:
        other = fresh(repo, cache, {p: 'f'*64 for p in pins})
    with pytest.raises(ValueError, match='changed|checksum'):
        other['train-b']


def test_new_process_detects_same_stat_but_different_source_content(tmp_path):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']
    path = repo.raw_root/'day/cloudbed/metric/node/cpu.csv'; before = path.stat()
    path.write_bytes(path.read_bytes().replace(b'1,h,cpu,3', b'1,h,cpu,8'))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='checksum'):
        fresh(repo, cache, pins)['train-b']


@pytest.mark.parametrize('changed', ['window', 'catalog', 'code'])
def test_cache_key_binds_semantic_input_and_data_code(tmp_path, monkeypatch, changed):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']
    manifest = repo.manifest
    if changed == 'window':
        manifest = replace(manifest, windows=(replace(manifest.windows[0], incident_end=7.),)+manifest.windows[1:])
    elif changed == 'catalog':
        manifest = replace(manifest, catalog=replace(manifest.catalog, version='cache-fixture/2'))
    else:
        monkeypatch.setattr(cache_api(), 'data_code_hash', lambda: 'd'*64)
    calls = []
    original = adapters.load_aiops22_case
    def tracked(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(adapters, 'load_aiops22_case', tracked)
    fresh(repo, cache, pins, manifest=manifest)['train-b']
    assert calls == [1] and len(list(cache.root.glob('*.json.gz'))) == 2


def test_cache_key_binds_static_prior_and_does_not_apply_it_twice(tmp_path, monkeypatch):
    from vast_v3.data.static_topology import StaticTopology, build_static_topology
    api = cache_api(); raw = tmp_path/'raw'; base = write_sources(raw)
    trace = base/'trace/all/trace_jaeger-span.csv'
    trace.write_text('timestamp,cmdb_id,span_id,trace_id,parent_span,duration\n'
                     '500,a-0,p,t,,1\n600,b-0,c,t,p,2\n4100,b-0,i,t,p,3\n')
    cache = api.ObservableCaseCache(tmp_path/'cache', max_bytes=100_000)
    repo = repository.DatasetRepository.from_curated(frozen_manifest(), raw, case_cache=cache)
    pins = repo.pin_source_content(); unaugmented = repo['train-b']
    prior = build_static_topology(repo.manifest, raw)
    augmented_repo = fresh(repo, cache, pins, static_topology=prior)
    augmented = augmented_repo['train-b']; audit = augmented_repo.case_audit('train-b')
    assert augmented.relations != unaugmented.relations
    assert len(list(cache.root.glob('*.json.gz'))) == 2
    monkeypatch.setattr(StaticTopology, 'apply', lambda *a: pytest.fail('cached prior applied twice'))
    second = fresh(repo, cache, pins, static_topology=prior)
    assert second['train-b'] == augmented and second.case_audit('train-b') == audit


def test_repeated_publication_is_idempotent_and_stats_are_not_mutable(tmp_path):
    repo, cache, _ = setup_cache(tmp_path)
    case = repo['train-b']; path, = cache.root.glob('*.json.gz')
    before = path.read_bytes(); envelope = json.loads(gzip.decompress(before))
    cache.store(envelope['payload']['binding'], case, repo.case_audit('train-b'))
    assert path.read_bytes() == before and cache.stats['writes'] == 1
    stats = cache.stats; stats['hits'] = 99
    assert cache.stats['hits'] == 0


@pytest.mark.parametrize('damage', ['gzip', 'payload_hash', 'unexpected_case_label', 'audit_label'])
def test_corrupt_or_label_bearing_entry_is_explicit_error_not_reload(tmp_path, monkeypatch, damage):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']
    path, = cache.root.glob('*.json.gz')
    if damage == 'gzip':
        path.write_bytes(b'incomplete gzip')
    else:
        envelope = json.loads(gzip.decompress(path.read_bytes()))
        if damage == 'payload_hash':
            envelope['payload']['case']['logs'][0]['text'] = 'changed'
        else:
            target = envelope['payload']['case'] if damage == 'unexpected_case_label' else envelope['payload']['audit']
            target['roots'] = ['service:a']
            raw = json.dumps(envelope['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            envelope['payload_sha256'] = hashlib.sha256(raw).hexdigest()
        path.write_bytes(gzip.compress(json.dumps(envelope).encode(), mtime=0))
    monkeypatch.setattr(adapters, 'load_aiops22_case', lambda *a, **k: pytest.fail('corrupt cache silently rebuilt'))
    with pytest.raises(ValueError, match='cache|fields|label'):
        fresh(repo, cache, pins)['train-b']


@pytest.mark.parametrize('field', ['raw_roots', 'canonical_roots', 'root_labels', 'root_ids'])
def test_store_rejects_nested_root_label_aliases_before_publication(tmp_path, field):
    repo, cache, _ = setup_cache(tmp_path)
    case = repo['train-b']; path, = cache.root.glob('*.json.gz')
    binding = json.loads(gzip.decompress(path.read_bytes()))['payload']['binding']
    other = cache_api().ObservableCaseCache(tmp_path/'label-audit', max_bytes=100_000)
    with pytest.raises(ValueError, match='label'):
        other.store(binding, case, {'nested': [{field: ['service:a']}]})
    assert not list(other.root.glob('*.json.gz'))


@pytest.mark.parametrize('field', ['raw_roots', 'canonical_roots', 'root_labels', 'root_ids'])
def test_hot_load_rejects_nested_root_label_aliases_with_valid_payload_hash(tmp_path, monkeypatch, field):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']; path, = cache.root.glob('*.json.gz')
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    envelope['payload']['audit']['nested'] = [{field: ['service:a']}]
    raw = json.dumps(envelope['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    envelope['payload_sha256'] = hashlib.sha256(raw).hexdigest()
    path.write_bytes(gzip.compress(json.dumps(envelope).encode(), mtime=0))
    monkeypatch.setattr(adapters, 'load_aiops22_case', lambda *a, **k: pytest.fail('label cache silently rebuilt'))
    with pytest.raises(ValueError, match='label'):
        fresh(repo, cache, pins)['train-b']


def test_orphan_partial_entry_is_not_a_cache_miss(tmp_path, monkeypatch):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']; path, = cache.root.glob('*.json.gz')
    path.rename(cache.root/f'.{path.name.removesuffix(".json.gz")}.interrupted.tmp')
    monkeypatch.setattr(adapters, 'load_aiops22_case', lambda *a, **k: pytest.fail('orphan silently ignored'))
    with pytest.raises(ValueError, match='partial|incomplete'):
        fresh(repo, cache, pins)['train-b']


def test_capacity_fails_without_evicting_existing_entries(tmp_path):
    repo, cache, pins = setup_cache(tmp_path)
    repo['train-b']; path, = cache.root.glob('*.json.gz'); saved = path.read_bytes()
    small = cache_api().ObservableCaseCache(cache.root, max_bytes=path.stat().st_size+8)
    with pytest.raises(ValueError, match='capacity'):
        fresh(repo, small, pins)['train-a']
    assert path.read_bytes() == saved
    assert len(list(cache.root.glob('*.json.gz'))) == 1
    assert fresh(repo, small, pins)['train-b'].case_id == 'train-b'


def test_writer_lock_is_exclusive_and_does_not_delete_another_writer_lock(tmp_path):
    repo, cache, _ = setup_cache(tmp_path)
    lock = cache.root/'.write.lock'; lock.write_text('another writer')
    with pytest.raises(ValueError, match='lock|busy'):
        repo['train-b']
    assert lock.read_text() == 'another writer'
    assert not list(cache.root.glob('*.json.gz'))


def test_failed_atomic_publish_leaves_no_visible_entry_and_can_retry(tmp_path, monkeypatch):
    repo, cache, _ = setup_cache(tmp_path)
    original = os.replace
    def interrupted(*args):
        raise OSError('synthetic publish interruption')
    monkeypatch.setattr(cache_api().os, 'replace', interrupted)
    with pytest.raises(OSError, match='interruption'):
        repo['train-b']
    assert not list(cache.root.glob('*.json.gz'))
    assert not list(cache.root.glob('*.tmp')) and not (cache.root/'.write.lock').exists()
    monkeypatch.setattr(cache_api().os, 'replace', original)
    assert repo['train-b'].case_id == 'train-b'
    assert len(list(cache.root.glob('*.json.gz'))) == 1


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, None])
def test_cache_requires_explicit_positive_byte_capacity(tmp_path, limit):
    with pytest.raises(ValueError, match='capacity|max_bytes'):
        cache_api().ObservableCaseCache(tmp_path/'cache', max_bytes=limit)


def test_cache_cannot_be_written_under_raw_root(tmp_path):
    api = cache_api(); write_sources(tmp_path/'raw')
    cache = api.ObservableCaseCache(tmp_path/'raw/cache', max_bytes=10000)
    with pytest.raises(ValueError, match='raw root'):
        repository.DatasetRepository.from_curated(frozen_manifest(), tmp_path/'raw', case_cache=cache)
