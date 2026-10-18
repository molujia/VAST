"""Synthetic-only regressions for the approved static AIOps topology policy."""
from dataclasses import replace
import csv
import hashlib
import importlib

import pytest

from vast_v3.data import adapters
from vast_v3.data.manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from vast_v3.data.schema import Entity
from vast_v3.data.repository import DatasetRepository


HEADER = 'timestamp,cmdb_id,span_id,trace_id,duration,type,status_code,operation_name,parent_span\n'
PARENT = '1000,a-0,p,t,1,server,0,operation,\n'
CHILD = '1500,b-0,c,t,2,server,0,operation,p\n'
TAIL = '2000,b-0,last,t,3,server,0,operation'


def trace_reader():
    assert hasattr(adapters, 'aiops_trace_csv_rows'), 'AIOps-only audited tail reader is required'
    return adapters.aiops_trace_csv_rows


def test_exact_aiops_unterminated_tail_is_quarantined_with_audit(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_bytes((HEADER + PARENT + CHILD + TAIL).encode())
    before = path.read_bytes()
    audit = {}
    rows = list(trace_reader()(path, audit=audit))
    assert [r['span_id'] for r in rows] == ['p', 'c']
    record, = audit['trace_tail_quarantine']
    assert record['line_number'] == 4
    assert record['expected_columns'] == 9 and record['observed_columns'] == 8
    assert record['missing_column'] == 'parent_span'
    assert record['file_size_bytes'] == len(before)
    assert record['row_sha256'] == hashlib.sha256(TAIL.encode()).hexdigest()
    assert record['reason'] == 'unterminated_final_row_missing_parent_span'
    assert path.read_bytes() == before


@pytest.mark.parametrize('suffix', [TAIL + '\n', TAIL + '\n' + PARENT,
                                    TAIL.rsplit(',', 1)[0], TAIL + ',p,extra',
                                    'bad,b-0,last,t,3,server,0,operation',
                                    '2000,b-0,last,t,broken,server,0,operation'])
def test_other_aiops_csv_malformations_still_block(tmp_path, suffix):
    path = tmp_path / 'trace.csv'
    path.write_text(HEADER + PARENT + suffix, encoding='utf-8')
    with pytest.raises((ValueError, csv.Error)):
        list(trace_reader()(path, audit={}))


def test_default_csv_reader_never_quarantines_and_complete_tail_is_retained(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text(HEADER + PARENT + TAIL, encoding='utf-8')
    with pytest.raises(ValueError, match='width'):
        list(adapters.strict_csv_rows(path))
    path.write_text(HEADER + PARENT + CHILD.rstrip('\n'), encoding='utf-8')
    audit = {}
    assert len(list(trace_reader()(path, audit=audit))) == 2
    assert audit['trace_tail_quarantine'] == []


def test_same_width_tail_under_different_header_is_not_quarantined(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text(HEADER.replace('parent_span\n', 'unexpected\n') + PARENT + TAIL, encoding='utf-8')
    with pytest.raises(ValueError):
        list(trace_reader()(path, audit={}))


@pytest.mark.parametrize('operation', ['oper"ation', '"operation"junk', 'op""eration'])
def test_tail_quarantine_rejects_additional_quoting_damage(tmp_path, operation):
    path = tmp_path / 'trace.csv'
    path.write_bytes((HEADER + PARENT + TAIL.replace('operation', operation)).encode())
    audit = {}
    with pytest.raises((ValueError, csv.Error)):
        list(trace_reader()(path, audit=audit))
    assert audit['trace_tail_quarantine'] == []


@pytest.mark.parametrize('operation', ['"oper,ation"', '"oper""ation"', '""'])
def test_tail_quarantine_allows_valid_quoted_fields(tmp_path, operation):
    path = tmp_path / 'trace.csv'
    path.write_bytes((HEADER + PARENT + TAIL.replace('operation', operation)).encode())
    audit = {}
    assert len(list(trace_reader()(path, audit=audit))) == 1
    assert len(audit['trace_tail_quarantine']) == 1


def topology_api():
    try:
        return importlib.import_module('vast_v3.data.static_topology')
    except ModuleNotFoundError:
        pytest.fail('Identity-bound static topology artifact is required')


def manifest():
    catalog = CandidateCatalog((Entity('service:a', 'service'), Entity('service:b', 'service'),
                                Entity('host:h', 'host')), 'synthetic-static-v1', 'a' * 64)
    windows = (
        WindowSpec('later', 'aiops2022_pre', 'later', 'day', 10., 12., 13., 15., 'declared reference', 'b' * 64),
        WindowSpec('test', 'aiops2022_pre', 'test', 'test-day', -10., -8., -7., -5., 'declared reference', 'b' * 64),
        WindowSpec('first', 'aiops2022_pre', 'first', 'day', 0., 2., 3., 5., 'declared reference', 'b' * 64),
    )
    return CuratedManifest(windows, catalog,
        FrozenSplit('aiops2022_pre', ('later', 'first'), ('test',), 'c' * 64), {})


def write_raw(root, *, tail=True):
    for day in ('day', 'test-day'):
        base = root / day / 'cloudbed'
        files = {
            'metric/container/cpu.csv': 'timestamp,cmdb_id,kpi_name,value\n1,h.a-0,cpu,2\n11,h.a-0,cpu,3\n14,h.a-0,cpu,4\n',
            'metric/node/cpu.csv': 'timestamp,cmdb_id,kpi_name,value\n1,h,cpu,1\n11,h,cpu,2\n',
            'log/all/log_filebeat-one.csv': 'timestamp,cmdb_id,value\n14,b-0,timeout\n',
            'trace/all/trace_jaeger-span.csv': HEADER + PARENT + CHILD
                + '3500,context-0,incident,t,8,server,0,operation,p\n' + (TAIL if tail else ''),
        }
        for relative, value in files.items():
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value.encode())
    return root / 'day/cloudbed/trace/all/trace_jaeger-span.csv'


def test_static_artifact_uses_only_earliest_permitted_training_reference(tmp_path, monkeypatch):
    s = topology_api(); source = write_raw(tmp_path)
    original = adapters.aiops_trace_csv_rows; opened = []
    def counted(path, **kwargs):
        opened.append(path)
        yield from original(path, **kwargs)
    monkeypatch.setattr(adapters, 'aiops_trace_csv_rows', counted)
    artifact = s.build_static_topology(manifest(), tmp_path)
    assert opened == [source]
    assert artifact.reference_window.case_id == 'first'
    edge, = artifact.relations
    assert (edge.source, edge.target, edge.active_intervals) == ('service:a', 'service:b', ((1.5, 1.5),))
    assert artifact.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    payload = artifact.to_dict()
    assert payload['trace_tail_quarantine'][0]['line_number'] == 5
    assert s.StaticTopology.from_dict(payload).identity_sha256 == artifact.identity_sha256
    assert 'context' not in repr(payload['entities'])
    assert not {'roots', 'labels', 'fault_type'} & payload.keys()


def test_static_artifact_rejects_test_permission_and_empty_first_reference(tmp_path):
    s = topology_api(); source = write_raw(tmp_path)
    with pytest.raises(ValueError, match='permitted'):
        s.build_static_topology(manifest(), tmp_path, permitted_case_ids=('test',))
    with pytest.raises(ValueError, match='observed call'):
        s.build_static_topology(manifest(), tmp_path, permitted_case_ids=('later',))
    source.write_text(HEADER + PARENT, encoding='utf-8')
    with pytest.raises(ValueError, match='observed call'):
        s.build_static_topology(manifest(), tmp_path)


@pytest.mark.parametrize('permitted', ['first', {'first': True}, [True], [1], ['first', 'first'], []])
def test_invalid_permission_container_is_rejected(tmp_path, permitted):
    s = topology_api(); write_raw(tmp_path)
    with pytest.raises(ValueError, match='permitted'):
        s.build_static_topology(manifest(), tmp_path, permitted_case_ids=permitted)


def test_static_entities_are_only_catalog_and_observed_call_endpoints(tmp_path):
    s = topology_api(); source = write_raw(tmp_path, tail=False)
    source.write_bytes(source.read_bytes() + b'1700,unrelated-0,orphan,t,1,server,0,operation,\n')
    artifact = s.build_static_topology(manifest(), tmp_path)
    assert {e.id for e in artifact.entities} == set(manifest().catalog.candidate_ids)


def test_static_artifact_strict_roundtrip_and_identity_tamper_detection(tmp_path):
    s = topology_api(); write_raw(tmp_path)
    artifact = s.build_static_topology(manifest(), tmp_path)
    payload = artifact.to_dict()
    payload['source_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='identity'):
        s.StaticTopology.from_dict(payload)
    payload = artifact.to_dict(); payload['roots'] = ['service:a']
    with pytest.raises(ValueError, match='fields'):
        s.StaticTopology.from_dict(payload)


def test_repository_inherits_structure_without_inventing_trace_values(tmp_path):
    s = topology_api(); write_raw(tmp_path)
    artifact = s.build_static_topology(manifest(), tmp_path)
    plain = DatasetRepository.from_curated(manifest(), tmp_path)
    repo = DatasetRepository.from_curated(manifest(), tmp_path, static_topology=artifact)
    assert repo.identity_sha256 != plain.identity_sha256
    case = repo['later']
    edge, = (r for r in case.relations if r.kind == 'call')
    assert edge.id == artifact.relations[0].id
    assert edge.active_intervals == ((10., 12.), (13., 15.))
    assert edge.is_active(10., 11.) and edge.is_active(14., 15.)
    assert not edge.is_active(12.2, 12.3)
    assert not edge.is_active(12., 13.)
    assert not edge.is_active(15., 16.)
    assert edge.is_active(11., 12.) and edge.is_active(13., 14.)
    assert not [x for x in case.numeric if x.modality == 'trace']
    assert case.numeric == plain['later'].numeric and case.logs == plain['later'].logs
    assert not [r for r in plain['later'].relations if r.kind == 'call']
    audit = repo.case_audit('later')['static_topology']
    assert audit['identity_sha256'] == artifact.identity_sha256
    assert audit['observed_call_intervals'] == {}
    assert audit['semantics'] == 'inherited_static_structure_not_contemporaneous_call_evidence'
    assert audit['inherited_relation_ids'] == [edge.id]


def test_existing_call_owner_is_extended_not_duplicated_and_raw_intervals_audited(tmp_path):
    s = topology_api(); write_raw(tmp_path)
    artifact = s.build_static_topology(manifest(), tmp_path)
    repo = DatasetRepository.from_curated(manifest(), tmp_path, static_topology=artifact)
    case = repo['first']
    edges = [r for r in case.relations if (r.source, r.target) == ('service:a', 'service:b')]
    assert len(edges) == 1 and edges[0].active_intervals == ((0., 2.), (3., 5.))
    trace = next(n for n in case.numeric if n.owner == edges[0].id)
    assert trace.times == (1.5,) and trace.values == (2.,)
    audit = repo.case_audit('first')['static_topology']
    assert audit['observed_call_intervals'][edges[0].id] == [[1.5, 1.5]]
    assert audit['observed_trace_value_count'] == 2  # includes the separate incident-only edge
    other = next(r for r in case.relations if r.target == 'service:context')
    assert other.active_intervals == ((3.5, 3.5),)


def test_static_repository_construction_is_lazy_and_rejects_manifest_mismatch(tmp_path, monkeypatch):
    s = topology_api(); write_raw(tmp_path)
    m = manifest(); artifact = s.build_static_topology(m, tmp_path)
    import vast_v3.data.repository as repository
    def forbidden(*args, **kwargs):
        pytest.fail('Repository construction must not read telemetry')
    monkeypatch.setattr(repository, 'file_sha256', forbidden)
    monkeypatch.setattr(adapters, 'load_aiops22_case', forbidden)
    repo = DatasetRepository.from_curated(m, tmp_path, static_topology=artifact)
    assert repo.loaded_case_ids == ()
    changed = replace(m, catalog=replace(m.catalog, version='changed'))
    with pytest.raises(ValueError, match='manifest|catalog'):
        DatasetRepository.from_curated(changed, tmp_path, static_topology=artifact)


@pytest.mark.parametrize('operation', ['materialize', 'pin'])
def test_static_source_hash_is_checked_even_before_loading_its_case(tmp_path, operation):
    s = topology_api(); source = write_raw(tmp_path)
    artifact = s.build_static_topology(manifest(), tmp_path)
    source.write_bytes(source.read_bytes().replace(b'operation', b'changedop'))
    repo = DatasetRepository.from_curated(manifest(), tmp_path, static_topology=artifact)
    with pytest.raises(ValueError, match='static.*checksum'):
        repo['test'] if operation == 'materialize' else repo.pin_source_content()
