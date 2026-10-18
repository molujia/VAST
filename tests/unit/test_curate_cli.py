"""Curator-only CLI using the existing tiny on-disk raw annotation fixtures."""
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
import os

import pytest

from test_runtime_data import manifest as fixture_manifest, write_aiops
from vast_v3.data.curation import CuratedLabels
from vast_v3.data.repository import DatasetRepository, manifest_to_dict
from vast_v3.data.schema import LabelStore


def cli():
    assert importlib.util.find_spec('vast_v3.cli.curate') is not None, 'explicit sealed curator CLI missing'
    from vast_v3.cli import curate
    return curate


def raw_fixture(root, dataset):
    root.mkdir()
    manifest = fixture_manifest(dataset)
    if dataset == 'rcabench':
        for case_id in ('train', 'test'):
            folder = root / case_id
            folder.mkdir()
            (folder / 'injection.json').write_text(json.dumps({
                'ground_truth': {'service': ['a', 'b']}, 'fault_type': 7}))
            (folder / 'prediction.json').write_text(json.dumps({'targets': ['unrelated-prediction']}))
    else:
        write_aiops(root)
        manifest = replace(manifest, windows=tuple(
            replace(window, source_id=f'hd1:{window.day}:{timestamp}:pod:a2-0:fault')
            for window, timestamp in zip(manifest.windows, (10, 20))))
        labels = root / 'groundtruth'
        labels.mkdir()
        (labels / 'groundtruth-2022-05-03.json').write_text(json.dumps({
            'timestamp': [10, 20], 'level': ['pod', 'pod'], 'cmdb_id': ['a2-0', 'a2-0'],
            'failure_type': ['fault', 'fault']}))
    return manifest


def test_curator_cli_exposes_only_fixed_population_and_explicit_paths(capsys):
    api = cli()
    with pytest.raises(SystemExit) as result:
        api.main(['--help'])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ('--dataset', '--raw-root', '--output-dir'):
        assert flag in help_text
    for flag in ('--manifest', '--case-id', '--roots', '--mapping', '--resume', '--force', '--budget'):
        assert flag not in help_text


@pytest.mark.parametrize('dataset', ['rcabench', 'aiops2022_pre'])
def test_curator_raw_roundtrip_keeps_public_inputs_clean_and_never_queries_or_loads_observations(
        tmp_path, monkeypatch, capsys, dataset):
    api = cli()
    raw = tmp_path / 'raw'
    manifest = raw_fixture(raw, dataset)
    monkeypatch.setattr(api, 'build_standard_manifest', lambda requested: manifest)
    monkeypatch.setattr(DatasetRepository, 'from_curated', lambda *a, **kw: pytest.fail('curation loaded observations'))
    monkeypatch.setattr(CuratedLabels, 'query_store', lambda *a, **kw: pytest.fail('curation opened training labels'))
    monkeypatch.setattr(CuratedLabels, 'evaluation_store', lambda *a, **kw: pytest.fail('curation opened evaluation labels'))
    monkeypatch.setattr(LabelStore, 'query', lambda *a, **kw: pytest.fail('curation bought a label'))
    output = tmp_path / 'curator-v001'
    assert api.main(['--dataset', dataset, '--raw-root', str(raw), '--output-dir', str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == json.loads((output / 'curator-summary.json').read_text())
    assert summary['complete'] is True
    assert summary['counts'] == {'cases': 2, 'train': 1, 'test': 1, 'candidates': 3}
    assert set(summary['artifacts']) == {'public_manifest', 'sealed_labels'}
    for artifact in summary['artifacts'].values():
        from pathlib import Path
        path = Path(artifact['path'])
        assert path.parent == output and artifact['sha256'] == sha256(path.read_bytes()).hexdigest()
    public = json.loads((output / 'public-manifest.json').read_text())
    sealed = json.loads((output / 'sealed-labels.json').read_text())
    assert public == manifest_to_dict(manifest)
    restored = CuratedLabels.from_dict(sealed, manifest)
    assert summary['sealed_identity_sha256'] == restored.identity_sha256
    assert 'root_cardinality_counts' in sealed['audit']
    assert 'root_cardinality_by_fault_type' in sealed['audit']
    assert not {'roots', 'records', 'fault_type', 'audit'} & public.keys()
    assert not {'root_cardinality_counts', 'root_cardinality_by_fault_type', 'records', 'roots'} & summary.keys()
    assert 'canonical_roots' not in json.dumps(summary) and 'a2-0' not in json.dumps(summary)
    assert sorted(path.name for path in output.iterdir()) == [
        'curator-summary.json', 'public-manifest.json', 'sealed-labels.json']
    if os.name != 'nt':
        assert output.stat().st_mode & 0o077 == 0
        assert all(path.stat().st_mode & 0o077 == 0 for path in output.iterdir())


def test_existing_attempt_is_rejected_before_any_frozen_source_or_label_read(tmp_path, monkeypatch, capsys):
    api = cli()
    raw = tmp_path / 'raw'
    raw.mkdir()
    output = tmp_path / 'existing'
    output.mkdir()
    original = output / 'sealed-labels.json'
    original.write_bytes(b'preserve this attempt')
    monkeypatch.setattr(api, 'build_standard_manifest', lambda *a: pytest.fail('overwriting attempt read sources'))
    assert api.main(['--dataset', 'rcabench', '--raw-root', str(raw), '--output-dir', str(output)]) == 1
    assert original.read_bytes() == b'preserve this attempt'
    assert set(output.iterdir()) == {original}
    assert json.loads(capsys.readouterr().out)['complete'] is False


def test_invalid_raw_labels_never_publish_completion_or_dump_answers(tmp_path, monkeypatch, capsys):
    api = cli()
    raw = tmp_path / 'raw'
    manifest = raw_fixture(raw, 'rcabench')
    path = raw / 'train' / 'injection.json'
    path.write_text(json.dumps({'ground_truth': {'service': ['DO_NOT_ECHO_THIS_ROOT']}, 'fault_type': 7}))
    monkeypatch.setattr(api, 'build_standard_manifest', lambda requested: manifest)
    output = tmp_path / 'failed-attempt'
    assert api.main(['--dataset', 'rcabench', '--raw-root', str(raw), '--output-dir', str(output)]) == 1
    captured = capsys.readouterr()
    assert 'DO_NOT_ECHO_THIS_ROOT' not in captured.out + captured.err
    assert json.loads(captured.out)['complete'] is False
    assert not (output / 'curator-summary.json').exists()
    assert not (output / 'sealed-labels.json').exists()
    assert (output / 'curator-error.json').is_file()


def test_output_inside_raw_root_is_rejected_without_writing_raw_data(tmp_path, capsys):
    api = cli()
    raw = tmp_path / 'raw'
    raw.mkdir()
    output = raw / 'curator-output'
    assert api.main(['--dataset', 'rcabench', '--raw-root', str(raw), '--output-dir', str(output)]) == 1
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)['complete'] is False


def test_atomic_publication_never_replaces_a_preexisting_target(tmp_path):
    api = cli()
    path = tmp_path / 'artifact.json'
    path.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        api.atomic_new_json(path, {'replacement': True})
    assert path.read_bytes() == b'preserve'
    assert list(tmp_path.iterdir()) == [path]
