"""CLI wiring only: the data layer separately tests real synthetic CSVs."""
import json
from types import SimpleNamespace

import pytest

import vast_v3.cli.smoke as cli


def helper():
    assert hasattr(cli, 'prepare_observation_repository'), 'approved static topology not wired into smoke'
    return cli.prepare_observation_repository


def test_aiops_smoke_persists_prior_before_repository_load(tmp_path, monkeypatch):
    prepare = helper()
    manifest = SimpleNamespace(split=SimpleNamespace(dataset='aiops2022_pre'))
    payload = {'identity_sha256': 'a' * 64, 'fixture': 'synthetic-prior'}
    prior = SimpleNamespace(to_dict=lambda: payload, identity_sha256='a' * 64)
    events = []
    monkeypatch.setattr(cli, 'build_static_topology', lambda m, root: prior)

    def repository(m, root, **kwargs):
        assert m is manifest
        assert kwargs == {'static_topology': prior}
        assert json.loads((tmp_path / 'attempt.static-topology.json').read_text()) == payload
        events.append('repository')
        return 'repository-fixture'

    monkeypatch.setattr(cli.DatasetRepository, 'from_curated', repository)
    result = prepare(manifest, 'synthetic-root', tmp_path / 'attempt',
                     lambda phase, **kw: events.append(phase))
    assert result == 'repository-fixture'
    assert events == ['freeze_static_topology', 'repository']


def test_rcabench_smoke_does_not_read_or_create_aiops_prior(tmp_path, monkeypatch):
    prepare = helper()
    manifest = SimpleNamespace(split=SimpleNamespace(dataset='rcabench'))
    monkeypatch.setattr(cli, 'build_static_topology', lambda *a, **k: pytest.fail('unexpected topology read'))
    monkeypatch.setattr(cli.DatasetRepository, 'from_curated', lambda *a, **k: k)
    assert prepare(manifest, 'synthetic-root', tmp_path / 'attempt', lambda *a, **k: None) == {}
    assert not (tmp_path / 'attempt.static-topology.json').exists()


def test_topology_error_is_not_silently_replaced_with_empty_prior(tmp_path, monkeypatch):
    prepare = helper()
    manifest = SimpleNamespace(split=SimpleNamespace(dataset='aiops2022_pre'))

    def fail(*args):
        raise ValueError('synthetic missing reference')

    monkeypatch.setattr(cli, 'build_static_topology', fail)
    monkeypatch.setattr(cli.DatasetRepository, 'from_curated', lambda *a, **k: pytest.fail('loaded after failed prior'))
    with pytest.raises(ValueError, match='missing reference'):
        prepare(manifest, 'synthetic-root', tmp_path / 'attempt', lambda *a, **k: None)
