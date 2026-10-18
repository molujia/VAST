"""Synthetic regressions for the documented RCABench resource-owner rules."""
import pytest

from vast_v3.data.adapters import rcabench_rows_to_case
from vast_v3.data.manifests import CandidateCatalog, WindowSpec
from vast_v3.data.schema import Entity


def load(attributes, *, metric='cpu'):
    catalog = CandidateCatalog((Entity('service:a', 'service'), Entity('service:b', 'service')),
                               'synthetic', 'a' * 64)
    window = WindowSpec('fixture', 'rcabench', 'fixture', '2022-05-01', 0., 3., 4., 8.,
                        'synthetic explicit window', 'b' * 64)
    row = {'time': 1., 'metric': metric, 'value': 2., 'service_name': '', **attributes}
    return rcabench_rows_to_case(window, catalog, metric_rows=[row], timestamp_unit='s')


@pytest.mark.parametrize('attributes', [
    {'attr.k8s.replicaset.name': 'a-56b9db98d8'},
    {'attr.k8s.deployment.name': 'a'},
    {'attr.k8s.statefulset.name': 'a'},
    {'attr.k8s.pod.name': 'a-56b9db98d8-p9rq8'},
    {'attr.k8s.pod.name': 'ts0/a-56b9db98d8-p9rq8'},
])
def test_empty_service_name_uses_documented_resource_owner(attributes):
    result = load(attributes)
    assert result.case.numeric[0].owner == 'service:a'
    assert result.case.candidate_ids == ('service:a', 'service:b')
    assert result.audit['metric_resource_owner_fallback_records'] == 1


def test_direct_service_name_keeps_priority_over_resource_fields():
    result = load({'service_name': 'a', 'attr.k8s.pod.name': 'b-56b9db98d8-p9rq8'})
    assert result.case.numeric[0].owner == 'service:a'
    assert result.audit['metric_resource_owner_fallback_records'] == 0


def test_namespace_only_indicator_is_audited_not_attached_to_a_fake_service():
    result = load({'attr.k8s.namespace.name': 'ts0'}, metric='k8s.namespace.phase')
    assert not result.case.numeric
    assert result.case.candidate_ids == ('service:a', 'service:b')
    assert len(result.case.entities) == 2
    assert result.audit['metric_namespace_only_records'] == 1


def test_unknown_resource_service_stays_context_only():
    result = load({'attr.k8s.pod.name': 'loadgenerator-8465b66847-7qxhm'})
    assert result.case.numeric[0].owner == 'service:loadgenerator'
    assert not next(e for e in result.case.entities if e.id == 'service:loadgenerator').candidate


def test_unexplained_missing_owner_still_blocks():
    with pytest.raises(ValueError, match='owner'):
        load({})


def test_namespace_metadata_does_not_excuse_ownerless_service_metric():
    with pytest.raises(ValueError, match='owner'):
        load({'attr.k8s.namespace.name': 'ts0'})


def test_hubble_endpoint_owner_is_caller_not_an_exporter_service():
    result = load({'service_name': 'b', 'attr.source': 'ts0/a-56b9db98d8-p9rq8',
                   'attr.destination': 'ts0/b-56b9db98d8-p9rq8'}, metric='hubble_http_requests_total')
    assert result.case.numeric[0].owner == 'service:a'
    assert result.case.numeric[0].channel == 'metrics/hubble_http_requests_total/value'
    assert result.audit['metric_hubble_caller_records'] == 1


def test_missing_hubble_endpoint_falls_back_without_inventing_pair():
    result = load({'service_name': 'b', 'attr.source': 'ts0/a-56b9db98d8-p9rq8'},
                  metric='hubble_http_requests_total')
    assert result.case.numeric[0].owner == 'service:b'
    assert result.audit['metric_hubble_caller_records'] == 0


@pytest.mark.parametrize('source,target', [('ts0/a-56b9db98d8-p9rq8', 'ts0/'),
                                          ('ts0/', 'ts0/a-56b9db98d8-p9rq8')])
def test_hubble_namespace_without_endpoint_is_not_a_resolved_pair(source, target):
    result = load({'service_name': 'b', 'attr.source': source, 'attr.destination': target},
                  metric='hubble_http_requests_total')
    assert result.case.numeric[0].owner == 'service:b'
    assert result.audit['metric_hubble_caller_records'] == 0
