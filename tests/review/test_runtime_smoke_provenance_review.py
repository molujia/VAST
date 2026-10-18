"""Independent synthetic review probes: no network, roots, or real model load."""

import json

import pytest
import torch

from vast_v3.data.preprocessing import Preprocessor
from vast_v3.data.manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from vast_v3.data.repository import manifest_to_dict
from vast_v3.data.schema import Entity, NumericSeries, ObservableCase


def test_failed_model_smoke_keeps_source_audit_and_text_lock(tmp_path, monkeypatch):
    import vast_v3.cli.smoke as cli
    import vast_v3.smoke as smoke

    case = ObservableCase(
        'review-train', (Entity('service:a', 'service'),), 0., 3., 6.,
        (NumericSeries('metric', 'service:a', 'cpu', (0., 1., 2., 3., 4., 5.),
                       (0., 1., 0., 2., 4., 3.)),), (), (),
    )
    encoder = lambda texts: torch.zeros((len(texts), 384))
    pre = Preprocessor.fit([case], permitted_case_ids={case.case_id})
    audit = {'case_source_sha256': 'a' * 64,
             'source_sha256': {'synthetic-input.parquet': 'b' * 64},
             'repository_identity_sha256': 'c' * 64,
             'repository_identity_kind': 'metadata-bound'}
    lock = {'repository': 'synthetic-test-only', 'revision': 'd' * 40, 'files': {}}
    lock_path = tmp_path / 'fixture-lock.json'
    lock_path.write_text(json.dumps(lock), encoding='utf-8')
    manifest = CuratedManifest(
        (WindowSpec(case.case_id, 'rcabench', 'synthetic-source', 'synthetic-day',
                    0., 3., 3., 6., 'declared synthetic reference', 'e' * 64),
         WindowSpec('review-test', 'rcabench', 'synthetic-test-source', 'synthetic-test-day',
                    10., 13., 13., 16., 'declared synthetic reference', 'e' * 64)),
        CandidateCatalog(case.entities, 'synthetic-review-v1', 'f' * 64),
        FrozenSplit('rcabench', (case.case_id,), ('review-test',), 'e' * 64),
        {},
    )

    class Repository:
        def __getitem__(self, case_id):
            assert case_id == case.case_id
            return case

        def case_audit(self, case_id):
            assert case_id == case.case_id
            return audit

    monkeypatch.setattr(cli, 'build_standard_manifest', lambda dataset: manifest)
    monkeypatch.setattr(cli.DatasetRepository, 'from_curated', lambda *a, **k: Repository())
    monkeypatch.setattr(cli, 'FrozenSentenceEncoder', lambda *a, **k: encoder)
    monkeypatch.setattr(cli.Preprocessor, 'fit', lambda *a, **k: pre)

    def fail_construct(*args, **kwargs):
        raise RuntimeError('synthetic model construction failure')

    monkeypatch.setattr(smoke, 'build_objectives', fail_construct)
    output = tmp_path / 'attempt'
    argv = ['--dataset', 'rcabench', '--raw-root', str(tmp_path / 'unused-raw'),
            '--text-directory', str(tmp_path / 'unused-model'),
            '--text-lock', str(lock_path), '--output', str(output)]
    with pytest.raises(RuntimeError, match='synthetic model construction failure'):
        cli.main(argv)

    report = json.loads((output / 'smoke-report.json').read_text())
    assert report['complete'] is False
    assert report['error_type'] == 'RuntimeError'
    assert json.loads(output.with_name('attempt.preflight.json').read_text())['phase'] == 'error'
    assert json.loads(output.with_name('attempt.source-audit.json').read_text()) == audit
    assert json.loads(output.with_name('attempt.text-lock.json').read_text()) == lock
    assert json.loads(output.with_name('attempt.manifest.json').read_text()) == manifest_to_dict(manifest)

    original = (output / 'smoke-report.json').read_bytes()
    with pytest.raises(ValueError, match='exists'):
        cli.main(argv)
    assert (output / 'smoke-report.json').read_bytes() == original
