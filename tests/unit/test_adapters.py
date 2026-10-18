"""Synthetic source records only; no benchmark telemetry is distributed here."""

import importlib
import json
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from vast_v3.data.schema import Entity


def api():
    try:
        return (importlib.import_module('vast_v3.data.adapters'),
                importlib.import_module('vast_v3.data.manifests'))
    except ModuleNotFoundError:
        pytest.fail('Native observation adapters and frozen manifests are not implemented')


def catalog():
    _, m = api()
    return m.CandidateCatalog((Entity('service:a', 'service'), Entity('service:b', 'service'),
                              Entity('host:h', 'host')), 'fixture-catalog-v1', 'a'*64)


def window(dataset='rcabench'):
    _, m = api()
    return m.WindowSpec('fixture', dataset, 'fixture-source', '2022-05-03',
                        0., 3., 4., 8., 'explicit synthetic windows; not inferred', 'b'*64)


def test_native_rcabench_keeps_full_catalog_context_and_directed_edge_series():
    a, _ = api()
    metrics = [{'time': 0.25, 'metric': 'cpu', 'value': 1., 'service_name': 'a',
                '_modality': 'metrics', '_source': 'normal_metrics.parquet:1'},
               {'time': 4.125, 'metric': 'cpu', 'value': float('nan'), 'service_name': 'a',
                '_modality': 'metrics', '_source': 'abnormal_metrics.parquet:1'}]
    traces = [{'time': 0.5, 'trace_id': 't1', 'span_id': 'parent', 'parent_span_id': '',
               'service_name': 'a', 'duration': 10, '_source': 'traces:1'},
              {'time': 0.625, 'trace_id': 't1', 'span_id': 'child', 'parent_span_id': 'parent',
               'service_name': 'ctx', 'duration': 20, '_source': 'traces:2'},
              {'time': 5.125, 'trace_id': 't2', 'span_id': 'parent', 'parent_span_id': '',
               'service_name': 'a', 'duration': 30, '_source': 'traces:3'},
              {'time': 5.25, 'trace_id': 't2', 'span_id': 'child', 'parent_span_id': 'parent',
               'service_name': 'ctx', 'duration': 40, '_source': 'traces:4'}]
    loaded = a.rcabench_rows_to_case(window(), catalog(), metric_rows=metrics,
        log_rows=[{'time': 4.125, 'service_name': 'a', 'message': 'synthetic timeout'}],
        trace_rows=traces, timestamp_unit='s')
    case = loaded.case
    assert case.candidate_ids == ('service:a', 'service:b', 'host:h')
    assert any(e.id == 'service:ctx' and not e.candidate for e in case.entities)
    metric = next(s for s in case.numeric if s.modality == 'metric')
    assert metric.times == (0.25, 4.125)
    assert metric.observed == (True, False)
    edge = next(r for r in case.relations if r.kind == 'call')
    trace = next(s for s in case.numeric if s.modality == 'trace')
    assert (edge.source, edge.target) == ('service:a', 'service:ctx')
    assert trace.owner == edge.id
    assert trace.times == (0.625, 5.25)
    assert trace.values == (20., 40.)
    assert edge.active_intervals == ((0.625, 0.625), (5.25, 5.25))
    assert not edge.is_active(2., 3.)
    assert loaded.audit['trace_parentless_records'] == 2
    assert loaded.audit['trace_duration_unit'] == 'unresolved_native_unit'
    assert not {'roots', 'fault_type', 'positive_ids'} & asdict(case).keys()


def test_reference_gap_and_outside_records_are_explicitly_excluded():
    a, _ = api()
    rows = [{'time': t, 'metric': 'cpu', 'value': 1, 'service_name': 'a'}
            for t in [-1, 0, 2, 3.5, 4, 7, 8]]
    loaded = a.rcabench_rows_to_case(window(), catalog(), metric_rows=rows, timestamp_unit='s')
    assert loaded.case.numeric[0].times == (0., 2., 4., 7.)
    assert loaded.audit['outside_declared_windows'] == 3


def test_histograms_keep_each_value_column_as_its_own_semantic_channel():
    a, _ = api()
    row = {'time': 1, 'metric': 'latency', 'service_name': 'a', '_modality': 'metrics_histogram',
           'count': 2, 'sum': 6, 'min': 1, 'max': 5}
    loaded = a.rcabench_rows_to_case(window(), catalog(), metric_rows=[row], timestamp_unit='s')
    assert {s.channel.rsplit('/', 1)[-1] for s in loaded.case.numeric} == {'count', 'sum', 'min', 'max'}


def test_two_native_pod_series_are_not_mixed_before_encoding():
    a, _ = api()
    rows = [{'time': t, 'metric': 'cpu', 'service_name': 'a', 'attr.k8s.pod.name': pod, 'value': v}
            for pod, v in [('pod-one', 1), ('pod-two', 9)] for t in [0, 1, 2]]
    loaded = a.rcabench_rows_to_case(window(), catalog(), metric_rows=rows, timestamp_unit='s')
    assert len(loaded.case.numeric) == 2
    assert loaded.case.numeric[0].channel == loaded.case.numeric[1].channel
    assert {s.values for s in loaded.case.numeric} == {(1., 1., 1.), (9., 9., 9.)}


def test_aiops_trace_ms_and_metric_log_seconds_are_separate():
    a, _ = api()
    loaded = a.aiops22_rows_to_case(window('aiops2022_pre'), catalog(),
        metric_rows=[{'timestamp': '1.25', 'cmdb_id': 'h.a-0', 'kpi_name': 'cpu', 'value': '3', '_scope': 'container'}],
        log_rows=[{'timestamp': '4.5', 'cmdb_id': 'a-0', 'value': 'synthetic failure'}],
        trace_rows=[{'timestamp': '4000', 'cmdb_id': 'a-0', 'span_id': 'p', 'trace_id': 't', 'parent_span': '', 'duration': '2'},
                    {'timestamp': '4125', 'cmdb_id': 'b-0', 'span_id': 'c', 'trace_id': 't', 'parent_span': 'p', 'duration': '8'}])
    assert next(s for s in loaded.case.numeric if s.modality == 'metric').times == (1.25,)
    assert next(s for s in loaded.case.numeric if s.modality == 'trace').times == (4.125,)
    assert loaded.case.logs[0].time == 4.5
    assert any(r.kind == 'hosted_on' and r.source == 'service:a' and r.target == 'host:h'
               for r in loaded.case.relations)


@pytest.mark.parametrize('bad', [
    {'time': 1, 'metric': 'cpu', 'value': 1},
    {'time': float('nan'), 'metric': 'cpu', 'value': 1, 'service_name': 'a'},
    {'time': 1, 'metric': 'cpu', 'value': 'not-numeric', 'service_name': 'a'},
    {'time': 1, 'metric': 'cpu', 'value': 1, 'service_name': 'a', 'ground_truth': 'a'},
])
def test_invalid_observation_records_never_silently_skip(bad):
    a, _ = api()
    with pytest.raises(ValueError):
        a.rcabench_rows_to_case(window(), catalog(), metric_rows=[bad], timestamp_unit='s')


def test_unknown_parent_is_disclosed_without_inventing_an_edge():
    a, _ = api()
    result = a.rcabench_rows_to_case(window(), catalog(), trace_rows=[
        {'time': 4, 'trace_id': 't', 'span_id': 's', 'parent_span_id': 'missing', 'service_name': 'b', 'duration': 1}], timestamp_unit='s')
    assert not result.case.relations
    assert result.audit['trace_unresolved_parent_records'] == 1


def test_conflicting_span_ownership_is_an_error():
    a, _ = api()
    traces = [{'time': 1, 'trace_id': 't', 'span_id': 'same', 'parent_span_id': '', 'service_name': s, 'duration': 1}
              for s in ['a', 'b']]
    with pytest.raises(ValueError, match='span'):
        a.rcabench_rows_to_case(window(), catalog(), trace_rows=traces, timestamp_unit='s')


def test_strict_csv_reader_rejects_the_known_truncation_shape(tmp_path):
    a, _ = api()
    path = tmp_path / 'fixture.csv'
    path.write_text('timestamp,cmdb_id,value\n1,a,2\n2,a\n', encoding='utf-8')
    with pytest.raises(ValueError, match='width'):
        list(a.strict_csv_rows(path, required_columns=('timestamp', 'cmdb_id', 'value')))


def test_missing_raw_files_fail_instead_of_looking_like_empty_modality(tmp_path):
    a, _ = api()
    with pytest.raises(FileNotFoundError):
        a.load_aiops22_case(tmp_path, window=window('aiops2022_pre'), catalog=catalog())
    with pytest.raises(FileNotFoundError):
        a.load_rcabench_case(tmp_path, window=window(), catalog=catalog())


def test_timestamp_conversion_requires_explicit_numeric_units():
    a, _ = api()
    assert a.native_seconds(4125, unit='ms') == 4.125
    assert a.native_seconds(datetime(1970, 1, 1, 0, 0, 1, 250000, tzinfo=timezone.utc), unit='datetime') == 1.25
    with pytest.raises(ValueError):
        a.native_seconds(4125, unit='infer')
    with pytest.raises(ValueError):
        a.native_seconds(datetime(1970, 1, 1), unit='datetime')


def test_frozen_split_preserves_supplied_order_and_rejects_population_drift(tmp_path):
    _, m = api()
    path = tmp_path / 'split.json'
    payload = {'dataset': 'fixture', 'outer_train_case_ids': ['b', 'a'], 'outer_test_case_ids': ['c']}
    path.write_text(json.dumps(payload), encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    split = m.load_frozen_split(path, expected_sha256=digest, expected_counts=(2, 1))
    assert split.train_case_ids == ('b', 'a')
    assert split.test_case_ids == ('c',)
    with pytest.raises(ValueError):
        m.load_frozen_split(path, expected_sha256='f'*64, expected_counts=(2, 1))
    with pytest.raises(ValueError):
        m.load_frozen_split(path, expected_sha256=digest, expected_counts=(3, 1))
    payload['outer_test_case_ids'] = ['a']
    with pytest.raises(ValueError):
        m.FrozenSplit.from_dict(payload, source_sha256=digest)


def test_roots_are_curator_only_joint_complete_and_host_not_expanded():
    _, m = api()
    assert m.canonical_roots(catalog(), 'service', ['a', 'b'], complete=True) == ('service:a', 'service:b')
    assert m.canonical_roots(catalog(), 'node', ['h'], complete=True) == ('host:h',)
    with pytest.raises(ValueError):
        m.canonical_roots(catalog(), 'service', ['unmapped'], complete=True)
    with pytest.raises(ValueError):
        m.canonical_roots(catalog(), 'service', ['a'], complete=False)


def test_public_window_manifest_rejects_embedded_labels_and_missing_provenance():
    _, m = api()
    payload = asdict(window())
    assert m.WindowSpec.from_dict(payload) == window()
    with pytest.raises(ValueError):
        m.WindowSpec.from_dict({**payload, 'fault_type': 'not-for-learner'})
    with pytest.raises(ValueError):
        m.WindowSpec.from_dict({**payload, 'reference_evidence': ''})


def test_curator_keeps_frozen_fault_boundaries_and_explicit_missing_reference():
    _, m = api()
    rows = [
        {'window_id': 'one', 'source_id': 'one', 'window_kind': 'fault', 'day': '2022-05-03',
         'start_ts': '500', 'end_ts': '900', 'metadata_json': json.dumps({'interval_source': 'run_table_exact', 'fault_type': 'curator-only'})},
        {'window_id': 'two', 'source_id': 'two', 'window_kind': 'fault', 'day': '2022-05-03',
         'start_ts': '800', 'end_ts': '860', 'metadata_json': json.dumps({'interval_source': 'run_table_level_median'})},
        {'window_id': 'three', 'source_id': 'three', 'window_kind': 'fault', 'day': '2022-05-03',
         'start_ts': '1000', 'end_ts': '1300', 'metadata_json': json.dumps({'interval_source': 'run_table_type_median'})},
    ]
    windows, audit = m.curate_frozen_windows(rows, dataset='aiops2022_pre', source_sha256='c'*64)
    assert [(w.incident_start, w.incident_end) for w in windows] == [(500, 900), (800, 860), (1000, 1300)]
    assert [(w.reference_start, w.reference_end) for w in windows] == [(200, 500), (800, 800), (900, 1000)]
    assert windows[1].reference_missing
    assert 'overlap' in windows[1].reference_evidence
    assert audit['reference_counts'] == {'full_300': 1, 'zero': 1, 'shortened': 1}
    assert audit['interval_source_counts'] == {'run_table_exact': 1, 'run_table_level_median': 1, 'run_table_type_median': 1}
    assert all('fault_type' not in asdict(w) for w in windows)


def test_frozen_rcabench_reference_is_paired_by_case_not_arbitrary_normal():
    _, m = api()
    rows = [
        {'window_id': 'normal', 'source_id': 'normal', 'window_kind': 'normal', 'day': 'raw-case', 'start_ts': '0', 'end_ts': '3'},
        {'window_id': 'case', 'source_id': 'raw-case', 'window_kind': 'fault', 'day': 'raw-case', 'start_ts': '4', 'end_ts': '8'},
    ]
    windows, audit = m.curate_frozen_windows(rows, dataset='rcabench', source_sha256='c'*64)
    assert len(windows) == 1
    assert (windows[0].reference_start, windows[0].reference_end) == (0, 3)
    with pytest.raises(ValueError, match='reference'):
        m.curate_frozen_windows(rows[1:], dataset='rcabench', source_sha256='c'*64)


def test_curated_window_loader_checks_hash_count_and_interval_source_counts(tmp_path):
    _, m = api()
    path = tmp_path / 'windows.csv'
    path.write_text('window_id,source_id,window_kind,day,start_ts,end_ts,metadata_json\na,a,fault,2022-05-03,400,500,"{""interval_source"":""run_table_exact""}"\n', encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result, audit = m.load_frozen_windows(path, dataset='aiops2022_pre', expected_sha256=digest,
        expected_count=1, expected_interval_sources={'run_table_exact': 1})
    assert len(result) == 1
    for changes in [{'expected_sha256': 'e'*64}, {'expected_count': 2}, {'expected_interval_sources': {'run_table_exact': 2}}]:
        kwargs = dict(dataset='aiops2022_pre', expected_sha256=digest, expected_count=1, expected_interval_sources={'run_table_exact': 1})
        kwargs.update(changes)
        with pytest.raises(ValueError):
            m.load_frozen_windows(path, **kwargs)


def test_missing_value_field_is_not_a_missing_measurement():
    a, _ = api()
    with pytest.raises(ValueError, match='value'):
        a.aiops22_rows_to_case(window('aiops2022_pre'), catalog(), metric_rows=[
            {'timestamp': 1, 'cmdb_id': 'h.a-0', 'kpi_name': 'cpu', '_scope': 'container'}])
    with pytest.raises(ValueError, match='duration'):
        a.rcabench_rows_to_case(window(), catalog(), timestamp_unit='s', trace_rows=[
            {'time': 1, 'trace_id': 't', 'span_id': 's', 'parent_span_id': '', 'service_name': 'a'}])


def test_curator_manifest_builds_complete_frozen_population_without_old_features(tmp_path):
    _, m = api()
    windows = tmp_path / 'windows.csv'
    windows.write_text('window_id,source_id,window_kind,day,start_ts,end_ts,metadata_json\na,a,fault,d,400,500,"{""interval_source"":""run_table_exact""}"\nb,b,fault,d,900,1000,"{""interval_source"":""run_table_exact""}"\n', encoding='utf-8')
    entities = tmp_path / 'entity_features.csv'
    entities.write_text('entity_id,entity_type,is_positive,old_feature\nservice:a,service,1,999\nhost:h,host,0,-999\nservice:a,service,0,123\n', encoding='utf-8')
    train = tmp_path / 'pool.json'
    train.write_text(json.dumps({'dataset_id': 'aiops2022_pre', 'case_count': 1, 'case_ids': ['a'], 'ground_truth_included': False}), encoding='utf-8')
    test = tmp_path / 'rankings.jsonl'
    test.write_text(json.dumps({'case_id': 'b', 'ranking': ['unused'], 'targets': ['not-read']})+'\n', encoding='utf-8')
    compact_hash = lambda values: hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
    kwargs = dict(dataset='aiops2022_pre', windows_path=windows,
        expected_windows_sha256=hashlib.sha256(windows.read_bytes()).hexdigest(),
        entity_catalog_path=entities, expected_catalog_ids_sha256=compact_hash(['host:h', 'service:a']),
        train_pool_path=train, expected_train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        test_rankings_path=test, expected_test_ids_sha256=compact_hash(['b']),
        expected_counts=(1, 1), expected_entity_counts={'service': 1, 'host': 1},
        expected_interval_sources={'run_table_exact': 2})
    manifest = m.build_curator_manifest(**kwargs)
    assert manifest.split.train_case_ids == ('a',)
    assert manifest.split.test_case_ids == ('b',)
    assert manifest.catalog.candidate_ids == ('host:h', 'service:a')
    assert [w.case_id for w in manifest.windows] == ['a', 'b']
    assert 'targets' not in repr(manifest)
    with pytest.raises(ValueError, match='catalog'):
        m.build_curator_manifest(**{**kwargs, 'expected_catalog_ids_sha256': 'f'*64})
    test.write_text(json.dumps({'case_id': 'not-in-window', 'targets': []})+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='population'):
        m.build_curator_manifest(**{**kwargs, 'expected_test_ids_sha256': compact_hash(['not-in-window'])})


def test_standard_manifest_rejects_nonstandard_dataset_before_accessing_server():
    _, m = api()
    with pytest.raises(ValueError, match='dataset'):
        m.build_standard_manifest('unknown')


@pytest.mark.parametrize('kind,missing', [
    ('traces', 'parent_span_id'), ('metrics', 'value'),
    ('metrics_sum', 'value'), ('metrics_histogram', 'count'), ('logs', 'message'),
])
def test_empty_parquet_still_requires_structural_columns(tmp_path, kind, missing):
    import pyarrow as pa
    import pyarrow.parquet as pq
    a, _ = api()
    fields = {
        'traces': ('time', 'trace_id', 'span_id', 'parent_span_id', 'service_name', 'duration'),
        'metrics': ('time', 'metric', 'service_name', 'value'),
        'metrics_sum': ('time', 'metric', 'service_name', 'value'),
        'metrics_histogram': ('time', 'metric', 'service_name', 'count', 'sum', 'min', 'max'),
        'logs': ('time', 'service_name', 'message'),
    }
    (tmp_path / 'env.json').write_text('{}', encoding='utf-8')
    for phase in ('normal', 'abnormal'):
        for source, names in fields.items():
            columns = {name: pa.array([], type=pa.string()) for name in names
                       if not (phase == 'abnormal' and source == kind and name == missing)}
            pq.write_table(pa.table(columns), tmp_path / f'{phase}_{source}.parquet')
    with pytest.raises(ValueError, match='required columns'):
        a.load_rcabench_case(tmp_path, window=window(), catalog=catalog())


def test_outside_parent_metadata_does_not_import_outside_measurements_or_entities():
    a, _ = api()
    traces = [
        {'time': 3.9, 'trace_id': 't', 'span_id': 'p', 'parent_span_id': '', 'service_name': 'a', 'duration': 99},
        {'time': 4.1, 'trace_id': 't', 'span_id': 'c', 'parent_span_id': 'p', 'service_name': 'b', 'duration': 7},
        {'time': 100., 'trace_id': 'other', 'span_id': 'x', 'parent_span_id': '', 'service_name': 'irrelevant', 'duration': 999},
    ]
    loaded = a.rcabench_rows_to_case(window(), catalog(), trace_rows=traces, timestamp_unit='s')
    assert loaded.case.candidate_ids == catalog().candidate_ids
    assert len(loaded.case.entities) == len(catalog().entities)
    assert [(s.times, s.values) for s in loaded.case.numeric] == [((4.1,), (7.,))]
    assert loaded.audit['outside_declared_windows'] == 2
