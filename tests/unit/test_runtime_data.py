"""Synthetic runtime/curator fixtures; no benchmark observations or labels."""
from dataclasses import asdict, replace
import hashlib
import importlib
import json

import pytest

from vast_v3.data.manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from vast_v3.data.schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation


def api(name):
    try:
        return importlib.import_module('vast_v3.data.' + name)
    except ModuleNotFoundError:
        pytest.fail(f'{name} runtime boundary is not implemented')


def manifest(dataset='rcabench'):
    catalog = CandidateCatalog((Entity('service:a', 'service'), Entity('service:b', 'service'),
                                Entity('host:h', 'host')), 'fixture-v1', 'a' * 64)
    windows = tuple(WindowSpec(i, dataset, i, '2022-05-03', 0., 2., 3., 8.,
                               'synthetic frozen interval', 'b' * 64) for i in ('train', 'test'))
    return CuratedManifest(windows, catalog, FrozenSplit(dataset, ('train',), ('test',), 'c' * 64), {})


def case():
    return ObservableCase('train', manifest().catalog.entities, 0., 3., 8.,
        (NumericSeries('metric', 'service:a', 'cpu', (1., 4.), (2., 9.)),
         NumericSeries('trace', 'call:ab', 'duration_native', (4.,), (5.,))),
        (LogRecord('service:b', 4., 'timeout', 2),),
        (Relation('call:ab', 'service:a', 'service:b', 'call', 4., 4., active_intervals=((4., 4.),)),))


def test_observable_roundtrip_is_strict_and_preserves_native_ownership():
    s = api('serialization')
    data = s.serialize_observable(case())
    assert s.deserialize_observable(data) == case()
    assert s.observable_from_dict(s.observable_to_dict(case())) == case()
    assert 'fault_type' not in data and 'source_files' not in data


@pytest.mark.parametrize('location', ['case', 'entity', 'numeric', 'log', 'relation'])
def test_observable_rejects_labels_at_every_level(location):
    s = api('serialization')
    payload = s.observable_to_dict(case())
    target = payload if location == 'case' else payload[{'entity': 'entities', 'numeric': 'numeric',
                                                       'log': 'logs', 'relation': 'relations'}[location]][0]
    target['roots'] = ['service:a']
    with pytest.raises(ValueError, match='fields'):
        s.observable_from_dict(payload)


@pytest.mark.parametrize('change', [
    lambda p: p.update(incident_end=float('inf')),
    lambda p: p['entities'][0].update(candidate='false'),
    lambda p: p['numeric'][0].update(observed=[1, True]),
    lambda p: p['logs'][0].update(frequency=1.5),
])
def test_observable_rejects_json_type_coercion_and_nonfinite(change):
    s = api('serialization'); payload = s.observable_to_dict(case()); change(payload)
    with pytest.raises(ValueError):
        s.observable_from_dict(payload)


def test_duplicate_json_keys_are_not_silently_overwritten():
    with pytest.raises(ValueError, match='duplicate'):
        api('serialization').deserialize_observable('{"case_id":"one","case_id":"two"}')


def write_aiops(root, day='2022-05-03'):
    base = root / day / 'cloudbed'
    files = {'metric/container/cpu.csv': 'timestamp,cmdb_id,kpi_name,value\n1,h.a-0,cpu,2\n4,h.a-0,cpu,9\n',
             'metric/node/cpu.csv': 'timestamp,cmdb_id,kpi_name,value\n1,h,cpu,3\n',
             'log/all/log_filebeat-one.csv': 'timestamp,cmdb_id,value\n4,a-0,timeout\n',
             'trace/all/trace_jaeger-span.csv': 'timestamp,cmdb_id,span_id,trace_id,parent_span,duration\n4000,a-0,p,t,,1\n4100,b-0,c,t,p,2\n'}
    for relative, content in files.items():
        path = base / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    return base


def test_repository_is_lazy_mapping_preserving_population_and_source_identity(tmp_path):
    r = api('repository'); write_aiops(tmp_path)
    repo = r.DatasetRepository.from_curated(manifest('aiops2022_pre'), tmp_path)
    assert tuple(repo) == ('train', 'test') and len(repo) == 2
    assert repo.loaded_case_ids == ()
    assert len(repo.identity_sha256) == 64
    loaded = repo['train']
    assert loaded.case_id == 'train' and loaded.candidate_ids == manifest().catalog.candidate_ids
    assert repo.loaded_case_ids == ('train',)
    assert len(repo.case_audit('train')['case_source_sha256']) == 64
    assert repo['train'] is loaded
    with pytest.raises(KeyError):
        repo['absent']


def test_repository_manifest_path_roundtrip_and_strict_checksum(tmp_path):
    r = api('repository'); write_aiops(tmp_path)
    path = tmp_path / 'public.json'
    path.write_text(json.dumps(r.manifest_to_dict(manifest('aiops2022_pre'))))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    repo = r.DatasetRepository.from_curated(path, tmp_path, expected_manifest_sha256=sha)
    assert len(repo) == 2
    with pytest.raises(ValueError, match='checksum'):
        r.DatasetRepository.from_curated(path, tmp_path, expected_manifest_sha256='f' * 64)
    payload = json.loads(path.read_text()); payload['roots'] = {'train': ['service:a']}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='fields'):
        r.DatasetRepository.from_curated(path, tmp_path)


def test_repository_blocks_source_mutation_and_bad_content_pin(tmp_path):
    r = api('repository'); base = write_aiops(tmp_path)
    repo = r.DatasetRepository.from_curated(manifest('aiops2022_pre'), tmp_path)
    path = base / 'metric/node/cpu.csv'; path.write_text(path.read_text() + '4,h,cpu,8\n')
    with pytest.raises(ValueError, match='source.*changed'):
        repo['train']
    pins = {str(p): 'f' * 64 for p in tmp_path.rglob('*.csv')}
    pinned = r.DatasetRepository.from_curated(manifest('aiops2022_pre'), tmp_path, expected_source_sha256=pins)
    with pytest.raises(ValueError, match='checksum'):
        pinned['train']


def test_repository_rejects_population_drift_and_traversal(tmp_path):
    r = api('repository')
    bad = replace(manifest(), windows=manifest().windows[:1])
    with pytest.raises(ValueError, match='population'):
        r.DatasetRepository.from_curated(bad, tmp_path)
    bad = replace(manifest(), windows=(replace(manifest().windows[0], source_id='../outside'), manifest().windows[1]))
    with pytest.raises(ValueError, match='within raw root'):
        r.DatasetRepository.from_curated(bad, tmp_path)


def records():
    c = api('curation')
    return [c.CuratedLabelRecord(i, i, 'service', ('a',), 'synthetic-fault', 'fixture.json', 'd' * 64,
                                 True, 'synthetic complete record', '') for i in ('train', 'test')]


def test_curator_separates_train_query_and_test_evaluation_and_seals_roundtrip():
    c = api('curation'); curated = c.curate_root_records(manifest(), records(), mapping_version='fixture-v1')
    store = curated.query_store(budget=1)
    assert store.query(case()) == frozenset({'service:a'}) and store.spent == 1
    with pytest.raises(ValueError, match='absent'):
        curated.query_store(budget=2).query(replace(case(), case_id='test'))
    assert curated.evaluation_store()['test'] == ('service:a',)
    with pytest.raises(KeyError):
        curated.evaluation_store()['train']
    sealed = curated.to_dict()
    assert c.CuratedLabels.from_dict(sealed, manifest()).identity_sha256 == curated.identity_sha256
    sealed['records'][0]['canonical_roots'] = ['service:b']
    with pytest.raises(ValueError, match='mapping|identity'):
        c.CuratedLabels.from_dict(sealed, manifest())


@pytest.mark.parametrize('changes', [
    {'complete': False}, {'raw_roots': ('unmapped',)}, {'source_id': 'different'},
    {'raw_roots': ('a', 'b')},
])
def test_curator_blocks_incomplete_unmapped_mismatched_unconfirmed_roots(changes):
    c = api('curation'); rows = records(); rows[0] = replace(rows[0], **changes)
    with pytest.raises(ValueError):
        c.curate_root_records(manifest(), rows, mapping_version='fixture-v1')


def test_curator_reports_joint_root_distribution_and_keeps_host_unexpanded():
    c = api('curation'); rows = records()
    rows[0] = replace(rows[0], raw_roots=('a', 'b'), cosource_confirmation='expert-2026-10-15')
    rows[1] = replace(rows[1], granularity='node', raw_roots=('h',))
    curated = c.curate_root_records(manifest(), rows, mapping_version='fixture-v1')
    assert curated.audit['root_cardinality_counts'] == {'1': 1, '2': 1}
    assert curated.evaluation_store()['test'] == ('host:h',)
    with pytest.raises(ValueError, match='population'):
        c.curate_root_records(manifest(), rows[:1], mapping_version='fixture-v1')


def test_raw_rcabench_curator_reads_injection_not_prediction_exports(tmp_path):
    c = api('curation')
    for i in ('train', 'test'):
        path = tmp_path / i; path.mkdir()
        (path / 'injection.json').write_text(json.dumps({'ground_truth': {'service': ['a']}, 'fault_type': 'memory'}))
        (path / 'prediction.json').write_text(json.dumps({'targets': ['b']}))
    curated = c.curate_raw_labels(manifest(), tmp_path)
    assert curated.evaluation_store()['test'] == ('service:a',)
    assert curated.audit['case_count'] == 2


def test_raw_aiops_curator_matches_original_timestamp_not_inherited_interval(tmp_path):
    c = api('curation'); m = manifest('aiops2022_pre')
    windows = tuple(replace(w, source_id=f'hd1:{w.day}:{t}:pod:a2-0:fault') for w, t in zip(m.windows, (10, 20)))
    m = replace(m, windows=windows)
    path = tmp_path / 'groundtruth'; path.mkdir()
    (path / 'groundtruth-2022-05-03.json').write_text(json.dumps({'timestamp': [10, 20], 'level': ['pod', 'pod'],
        'cmdb_id': ['a2-0', 'a2-0'], 'failure_type': ['fault', 'fault']}))
    curated = c.curate_raw_labels(m, tmp_path)
    assert curated.evaluation_store()['test'] == ('service:a',)
    assert [w.incident_start for w in m.windows] == [3., 3.]


def test_aiops_historical_mapping_is_shared_by_observations_and_curator_only():
    from vast_v3.data.adapters import aiops22_rows_to_case
    c = api('curation'); m = manifest('aiops2022_pre')
    rows = [replace(r, granularity='pod', raw_roots=('a2-0',)) for r in records()]
    curated = c.curate_root_records(m, rows)
    loaded = aiops22_rows_to_case(m.windows[0], m.catalog, metric_rows=[
        {'timestamp': 1, 'cmdb_id': 'h.a2-0', 'kpi_name': 'cpu', 'value': 3, '_scope': 'container'}])
    assert loaded.case.numeric[0].owner == 'service:a'
    assert curated.evaluation_store()['test'] == ('service:a',)
    assert m.catalog.resolve('a2-0') == 'service:a2'  # generic mapping is not broadened
    assert loaded.audit['entity_mapping_version'] == curated.audit['entity_mapping_version']


def test_repository_hashes_shared_files_once_and_bounds_raw_case_cache(tmp_path, monkeypatch):
    r = api('repository'); write_aiops(tmp_path); m = manifest('aiops2022_pre')
    m = replace(m, windows=(*m.windows, replace(m.windows[0], case_id='train2', source_id='train2')),
                split=replace(m.split, train_case_ids=('train', 'train2')))
    original = r.file_sha256; calls = []
    def counted(path):
        calls.append(str(path)); return original(path)
    monkeypatch.setattr(r, 'file_sha256', counted)
    repo = r.DatasetRepository.from_curated(m, tmp_path, max_cached_cases=2)
    repo['train']; repo['test']; repo['train2']
    assert len(calls) == len(set(calls)) == 4
    assert repo.loaded_case_ids == ('test', 'train2')
    repo['train']
    assert len(calls) == 4 and repo.loaded_case_ids == ('train2', 'train')
    assert repo.case_audit('test')['case_source_sha256'] == repo.case_audit('train')['case_source_sha256']


def test_repository_rejects_added_source_file_after_identity_freeze(tmp_path):
    r = api('repository'); base = write_aiops(tmp_path)
    repo = r.DatasetRepository.from_curated(manifest('aiops2022_pre'), tmp_path)
    (base / 'metric/node/extra.csv').write_text('timestamp,cmdb_id,kpi_name,value\n1,h,cpu,4\n')
    with pytest.raises(ValueError, match='source.*changed'):
        repo['train']


def test_curator_rejects_conflicting_evidence_hash_for_same_file():
    c = api('curation'); rows = records(); rows[1] = replace(rows[1], evidence_sha256='f' * 64)
    with pytest.raises(ValueError, match='evidence'):
        c.curate_root_records(manifest(), rows)


def test_rcabench_original_fault_type_integer_enum_is_preserved(tmp_path):
    c = api('curation')
    for case_id in ('train', 'test'):
        folder = tmp_path / case_id; folder.mkdir()
        (folder / 'injection.json').write_text(json.dumps({'ground_truth': {'service': ['a', 'b']}, 'fault_type': 7}))
    curated = c.curate_raw_labels(manifest(), tmp_path)
    assert all(r['fault_type'] == 7 for r in curated.to_dict()['records'])
    assert curated.audit['root_cardinality_by_fault_type'] == {'7': {'2': 2}}


def test_explicit_source_pinning_reuses_hashes_and_precedes_case_loading(tmp_path, monkeypatch):
    r = api('repository'); write_aiops(tmp_path); calls = []; original = r.file_sha256
    def counted(path):
        calls.append(str(path)); return original(path)
    monkeypatch.setattr(r, 'file_sha256', counted)
    repo = r.DatasetRepository.from_curated(manifest('aiops2022_pre'), tmp_path)
    before = repo.identity_sha256
    digests = repo.pin_source_content()
    assert repo.identity_kind == 'content-pinned' and repo.identity_sha256 != before
    assert len(digests) == len(calls) == 4
    repo['train']; repo['test']
    assert len(calls) == 4
    with pytest.raises(ValueError, match='before loading'):
        repo.pin_source_content()


def test_repository_records_context_only_nodes_outside_observable_fields(tmp_path):
    r = api('repository'); base = write_aiops(tmp_path)
    path = base / 'log/all/log_filebeat-one.csv'
    path.write_text(path.read_text() + '4,context-0,synthetic\n')
    repo = r.DatasetRepository.from_curated(manifest('aiops2022_pre'), tmp_path)
    assert any(e.id == 'service:context' and not e.candidate for e in repo['train'].entities)
    assert repo.case_audit('train')['context_only_entity_ids'] == ['service:context']


def test_rcabench_lazy_repository_loads_native_parquet_without_label_files(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    r = api('repository')
    columns = {'metrics': ['time', 'service_name', 'metric', 'value'],
        'metrics_sum': ['time', 'service_name', 'metric', 'value'],
        'metrics_histogram': ['time', 'service_name', 'metric', 'count', 'sum', 'min', 'max'],
        'logs': ['time', 'service_name', 'message'],
        'traces': ['time', 'service_name', 'trace_id', 'span_id', 'parent_span_id', 'duration']}
    for key in ('train', 'test'):
        folder = tmp_path / key; folder.mkdir()
        (folder / 'env.json').write_text('{}')
        (folder / 'injection.json').write_text('NOT JSON: learner must not read this')
        for phase in ('normal', 'abnormal'):
            for kind, names in columns.items():
                values = {name: pa.array([], type=pa.string()) for name in names}
                if kind == 'metrics':
                    values = {'time': [1_000_000_000 if phase == 'normal' else 4_000_000_000],
                              'service_name': ['a'], 'metric': ['cpu'], 'value': [2.]}
                pq.write_table(pa.table(values), folder / f'{phase}_{kind}.parquet')
    repo = r.DatasetRepository.from_curated(manifest(), tmp_path)
    repo.pin_source_content()
    assert repo['train'].numeric[0].times == (1., 4.)
    assert repo['test'].case_id == 'test'
    assert all('injection' not in path for path in repo.source_files)


@pytest.mark.parametrize('dataset', ['rcabench', 'aiops2022_pre'])
@pytest.mark.parametrize('raw,alias,used', [
    ('legacy-owner', 'service:a', True),
    ('a', 'service:b', True),
    ('unused-legacy', 'service:a', False),
])
def test_curator_alias_cannot_override_frozen_telemetry_mapping(dataset, raw, alias, used):
    c = api('curation'); m = manifest(dataset)
    rows = [replace(r, raw_roots=(raw,)) for r in records()] if used else records()
    with pytest.raises(ValueError, match='alias.*frozen'):
        c.curate_root_records(m, rows, aliases={raw: alias}, mapping_version='declared-alias-v1')


@pytest.mark.parametrize('dataset,raw,granularity,expected', [
    ('rcabench', 'a-0', 'service', 'service:a'),
    ('aiops2022_pre', 'a2-0', 'pod', 'service:a'),
    ('aiops2022_pre', 'h', 'node', 'host:h'),
])
def test_matching_alias_is_only_a_versioned_declaration(dataset, raw, granularity, expected):
    c = api('curation'); m = manifest(dataset)
    rows = [replace(r, granularity=granularity, raw_roots=(raw,)) for r in records()]
    curated = c.curate_root_records(m, rows, aliases={raw: expected}, mapping_version='matching-alias-v1')
    assert curated.evaluation_store()['test'] == (expected,)
    assert curated.to_dict()['aliases'] == {raw: expected}
    assert curated.mapping_version == 'matching-alias-v1'
    assert c.CuratedLabels.from_dict(curated.to_dict(), m).identity_sha256 == curated.identity_sha256


def test_matching_host_alias_does_not_allow_service_label_category_change():
    c = api('curation')
    rows = [replace(r, raw_roots=('h',)) for r in records()]
    with pytest.raises(ValueError, match='category'):
        c.curate_root_records(manifest(), rows, aliases={'h': 'host:h'}, mapping_version='host-alias-v1')
