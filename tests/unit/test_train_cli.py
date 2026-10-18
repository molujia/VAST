"""Local synthetic training-entrypoint checks; no dataset or model download."""
from collections.abc import Mapping
from dataclasses import replace
import importlib.util
import json

import pytest
import torch

from vast_v3.data.curation import CuratedLabelRecord, CuratedLabels, curate_root_records
from vast_v3.data.manifests import CandidateCatalog, CuratedManifest, FrozenSplit, WindowSpec
from vast_v3.data.preprocessing import Preprocessor
from vast_v3.data.schema import Entity, NumericSeries, ObservableCase
from vast_v3.runtime import RuntimeConfig


def train_api():
    assert importlib.util.find_spec('vast_v3.cli.train') is not None, 'budgeted training CLI missing'
    from vast_v3.cli import train
    return train


class SyntheticRepository(Mapping):
    identity_kind = 'content-pinned'
    identity_sha256 = 'a' * 64
    static_topology = None

    def __init__(self, count=3):
        ids = tuple(f'c{i}' for i in range(count))
        entities = (Entity('service:a', 'service'), Entity('service:b', 'service'))
        self.manifest = CuratedManifest(
            tuple(WindowSpec(cid, 'rcabench', cid, 'synthetic', 0., 3., 3., 5.,
                             'synthetic reference', 'b' * 64) for cid in (*ids, 'test')),
            CandidateCatalog(entities, 'synthetic/1', 'c' * 64),
            FrozenSplit('rcabench', ids, ('test',), 'd' * 64), {})
        base = ObservableCase('c0', entities, 0., 3., 5.,
            (NumericSeries('metric', 'service:a', 'cpu', (0., 1., 2., 3., 4.), (0., 1., 0., 3., 2.)),
             NumericSeries('metric', 'service:b', 'cpu', (0., 1., 2., 3., 4.), (1., 0., 1., 2., 3.))))
        self.cases = {cid: replace(base, case_id=cid) for cid in ids}
        self.reads = []

    def __len__(self):
        return len(self.cases) + 1

    def __iter__(self):
        return iter((*self.cases, 'test'))

    def __getitem__(self, key):
        assert key != 'test', 'test observation reached training/fitting'
        self.reads.append(key)
        return self.cases[key]

    def labels(self):
        return curate_root_records(self.manifest, [
            CuratedLabelRecord(w.case_id, w.source_id, 'service', ('a',), 'synthetic',
                               'synthetic-labels', 'e' * 64, True, 'synthetic complete root')
            for w in self.manifest.windows])


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def invoke(path, repo=None, config=None, **kwargs):
    api = train_api()
    repo = repo or SyntheticRepository()
    config = config or RuntimeConfig(dataset='rcabench', budget=2, query_batch_size=1,
        warmup_epochs=1, nonfinal_epochs=1, final_epochs=1, u_disabled_epochs=0, u_ramp_epochs=0)
    return api.run_training(repository=repo, curated_labels=repo.labels(), config=config,
        text_encoder=lambda texts: torch.zeros(len(texts), 384), text_lock={'synthetic': 'fixed'},
        run_dir=path, code_hash='f' * 64, **kwargs)


def test_cli_exposes_inputs_but_no_schedule_smoke_or_test_selection(capsys):
    api = train_api()
    with pytest.raises(SystemExit) as result:
        api.main(['--help'])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ('--manifest', '--sealed-labels', '--raw-root', '--source-pins', '--text-lock',
                 '--text-directory', '--run-dir', '--resume', '--case-cache-root', '--max-updates'):
        assert flag in help_text
    for flag in ('--epochs', '--config', '--case-id', '--roots', '--checkpoint', '--seed', '--budget'):
        assert flag not in help_text


def test_real_objectives_full_short_schedule_resume_preserves_weights_and_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(CuratedLabels, 'evaluation_store',
                        lambda self: pytest.fail('test labels reached training entrypoint'))
    full = invoke(tmp_path / 'full')
    paused = invoke(tmp_path / 'partial', max_updates=2)
    assert not paused['complete'] and paused['phase'] == 'diagnostic_pause'
    assert paused['updates'] == 2 and paused['labels_spent'] == 1
    assert not (tmp_path / 'partial' / 'final-model.json').exists()
    pre_bytes = (tmp_path / 'partial' / 'preprocessing.json').read_bytes()
    monkeypatch.setattr(Preprocessor, 'fit_repository',
                        lambda *a, **kw: pytest.fail('resume refitted frozen preprocessing'))
    resumed = invoke(tmp_path / 'partial', resume=True)
    assert resumed['complete'] and resumed['updates'] == full['updates'] == 3
    assert resumed['labels_spent'] == 2
    assert not resumed['eligible_for_demo']  # shortened synthetic configuration, not production B30
    assert (tmp_path / 'partial' / 'preprocessing.json').read_bytes() == pre_bytes
    a = torch.load(tmp_path / 'full' / 'checkpoints' / 'latest.pt', weights_only=True)
    b = torch.load(tmp_path / 'partial' / 'checkpoints' / 'latest.pt', weights_only=True)
    assert a['extra_state']['query_trace'] == b['extra_state']['query_trace']
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    events = [json.loads(line) for line in (tmp_path / 'partial' / 'events.jsonl').read_text().splitlines()]
    assert {'fit_reference', 'eligibility', 'acquisition', 'training', 'diagnostic_pause', 'complete'} <= {
        row['phase'] for row in events}


def test_entrypoint_never_overwrites_existing_attempt(tmp_path):
    path = tmp_path / 'run'
    invoke(path, max_updates=1)
    before = (path / 'checkpoints' / 'latest.pt').read_bytes()
    with pytest.raises(ValueError, match='resume|existing'):
        invoke(path)
    assert (path / 'checkpoints' / 'latest.pt').read_bytes() == before


@pytest.mark.parametrize('drift', ['source', 'code', 'text', 'preprocessing', 'smoke'])
def test_resume_rejects_identity_or_smoke_before_observation_or_query(tmp_path, monkeypatch, drift):
    path = tmp_path / 'run'
    invoke(path, max_updates=1)
    repo = SyntheticRepository()
    if drift == 'source':
        repo.identity_sha256 = '9' * 64
    elif drift == 'preprocessing':
        artifact = json.loads((path / 'preprocessing.json').read_text())
        artifact['preprocessing']['fitted_case_ids'].append('test')
        (path / 'preprocessing.json').write_text(json.dumps(artifact))
    elif drift == 'smoke':
        artifact = json.loads((path / 'run.json').read_text())
        artifact['purpose'] = 'label-free-smoke'
        (path / 'run.json').write_text(json.dumps(artifact))
    monkeypatch.setattr(CuratedLabels, 'query_store', lambda *a, **kw: pytest.fail('invalid resume read roots'))
    api = train_api()
    config = RuntimeConfig(dataset='rcabench', budget=2, query_batch_size=1,
        warmup_epochs=1, nonfinal_epochs=1, final_epochs=1, u_disabled_epochs=0, u_ramp_epochs=0)
    with pytest.raises(ValueError, match='identity|preprocessing|purpose'):
        api.run_training(repository=repo, curated_labels=repo.labels(), config=config,
            text_encoder=lambda texts: torch.zeros(len(texts), 384),
            text_lock={'synthetic': 'changed' if drift == 'text' else 'fixed'},
            run_dir=path, code_hash=('0' if drift == 'code' else 'f') * 64, resume=True)
    assert repo.reads == []


def test_final_gate_rejects_stage_complete_and_missing_schedule_evidence(tmp_path):
    api = train_api()
    result = invoke(tmp_path / 'run')
    final = json.loads((tmp_path / 'run' / 'final-model.json').read_text())
    checkpoint = tmp_path / 'run' / 'checkpoints' / 'latest.pt'
    original = torch.load(checkpoint, weights_only=True)
    for field, value in [('stage', 'smoke'), ('encoder_version', 1)]:
        payload = dict(original)
        payload['progress'] = {**original['progress'], field: value}
        torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match='finished|schedule|updates'):
            api.inspect_final_training_checkpoint(checkpoint, expected_identity=final['identity'],
                config=RuntimeConfig.from_dict(final['config']), permitted_case_ids=('c0', 'c1', 'c2'))
    assert result['complete']


def test_default_b30_170_epoch_schedule_completes_with_thirty_paid_queries(tmp_path, monkeypatch):
    api = train_api()
    from test_ordered_query_trace import SyntheticObjectives

    def lightweight_objectives(config, pre, **kwargs):
        # Replace only expensive neural heads; real Trainer/schedule/ledger/RNG remain in scope.
        torch.manual_seed(config.seed)
        obj = SyntheticObjectives(pre.fitted_case_ids)
        obj.model.frontend.preprocessing = pre
        obj.max_roots = config.effective_max_roots
        return obj

    monkeypatch.setattr(api, 'build_objectives', lightweight_objectives)
    config = RuntimeConfig(dataset='rcabench')
    pause = invoke(tmp_path / 'b30', repo=SyntheticRepository(31), config=config, max_updates=181)
    assert not pause['complete'] and pause['labels_spent'] == 5
    result = invoke(tmp_path / 'b30', repo=SyntheticRepository(31), config=config, resume=True)
    assert result['complete'] and result['updates'] == 170 * 8
    assert result['labels_spent'] == 30
    final = json.loads((tmp_path / 'b30' / 'final-model.json').read_text())
    assert final['schedule_epochs'] == 170 and len(final['query_order']) == 30
    journal = json.loads((tmp_path / 'b30' / 'checkpoints' / 'query-access.json').read_text())
    assert len(journal['labels']['accessed']) == len(journal['labels']['revealed']) == 30
    trace = json.loads((tmp_path / 'b30' / 'checkpoints' / 'query-trace.json').read_text())
    assert len(trace['rounds']) == 6 and all(row['complete'] for row in trace['rounds'])


@pytest.mark.parametrize('value', [0, -1, True])
def test_diagnostic_update_limit_must_be_positive(tmp_path, value):
    with pytest.raises(ValueError, match='max_updates'):
        invoke(tmp_path / 'run', max_updates=value)


def test_resume_rejects_unbound_production_invocation_before_training(tmp_path, monkeypatch):
    path = tmp_path / 'run'
    invoke(path, max_updates=1)
    (path / 'invocation.json').write_text('{}')
    monkeypatch.setattr(CuratedLabels, 'query_store', lambda *a, **kw: pytest.fail('unbound invocation reached roots'))
    with pytest.raises(ValueError, match='invocation.*identity'):
        invoke(path, resume=True)


def test_final_gate_checks_config_identity_even_when_schedule_is_unchanged(tmp_path):
    api = train_api()
    invoke(tmp_path / 'run')
    final = json.loads((tmp_path / 'run' / 'final-model.json').read_text())
    config = replace(RuntimeConfig.from_dict(final['config']), log_recovery_radius_steps=3)
    with pytest.raises(ValueError, match='configuration identity'):
        api.inspect_final_training_checkpoint(tmp_path / 'run' / 'checkpoints' / 'latest.pt',
            expected_identity=final['identity'], config=config, permitted_case_ids=('c0', 'c1', 'c2'))


def test_final_gate_rejects_nonfinite_model_weights(tmp_path):
    api = train_api()
    invoke(tmp_path / 'run')
    final = json.loads((tmp_path / 'run' / 'final-model.json').read_text())
    checkpoint = tmp_path / 'run' / 'checkpoints' / 'latest.pt'
    payload = torch.load(checkpoint, weights_only=True)
    first = next(value for value in payload['model'].values() if value.numel() and value.is_floating_point())
    first.flatten()[0] = torch.nan
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='nonfinite'):
        api.inspect_final_training_checkpoint(checkpoint, expected_identity=final['identity'],
            config=RuntimeConfig.from_dict(final['config']), permitted_case_ids=('c0', 'c1', 'c2'))


def test_main_rejects_changed_standard_manifest_before_offline_model_or_raw_reads(tmp_path, monkeypatch):
    api = train_api()
    from vast_v3.data.repository import manifest_to_dict
    repo = SyntheticRepository()
    public = manifest_to_dict(repo.manifest)
    public['windows'][0]['incident_end'] = 7
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(public))
    monkeypatch.setattr(api, 'build_standard_manifest', lambda dataset: repo.manifest)
    monkeypatch.setattr(api.DatasetRepository, 'from_curated', lambda *a, **kw: pytest.fail('bad manifest reached raw input'))
    with pytest.raises(ValueError, match='complete approved frozen'):
        api.main(['--dataset', 'rcabench', '--manifest', str(manifest), '--raw-root', str(tmp_path / 'raw'),
            '--sealed-labels', str(tmp_path / 'sealed.json'), '--text-directory', str(tmp_path / 'text'),
            '--text-lock', str(tmp_path / 'lock.json'), '--case-cache-root', str(tmp_path / 'cache'),
            '--case-cache-max-bytes', '10000', '--run-dir', str(tmp_path / 'run')])


@pytest.mark.parametrize('mutation', ['empty_answer', 'changed_answer', 'missing_manifest', 'changed_manifest'])
def test_final_acceptance_binds_paid_store_not_only_ledger_case_ids(tmp_path, monkeypatch, mutation):
    api = train_api()
    run = api.ProgressTrainer.run

    def corrupt_after_training(trainer, **kwargs):
        result = run(trainer, **kwargs)
        path = trainer.directory / 'query-access.json'
        journal = json.loads(path.read_text())
        ledger = journal['labels']
        first = next(iter(ledger['revealed']))
        if mutation == 'empty_answer':
            ledger['revealed'][first] = []
        elif mutation == 'changed_answer':
            ledger['revealed'][first] = ['service:b']
        elif mutation == 'missing_manifest':
            del ledger['manifest_hash']
        else:
            ledger['manifest_hash'] = '0' * 64
        path.write_text(json.dumps(journal))
        trainer.labels.query = lambda *a, **kw: pytest.fail('final gate must not query more labels')
        return result

    monkeypatch.setattr(api.ProgressTrainer, 'run', corrupt_after_training)
    with pytest.raises(ValueError, match='paid label ledger'):
        invoke(tmp_path / 'run')
    assert not (tmp_path / 'run' / 'final-model.json').exists()


def test_independent_final_inspection_requires_trusted_paid_ledger_evidence(tmp_path):
    api = train_api()
    invoke(tmp_path / 'run')
    final = json.loads((tmp_path / 'run' / 'final-model.json').read_text())
    with pytest.raises(ValueError, match='paid label ledger.*expected'):
        api.inspect_final_training_checkpoint(tmp_path / 'run' / 'checkpoints' / 'latest.pt',
            expected_identity=final['identity'], config=RuntimeConfig.from_dict(final['config']),
            permitted_case_ids=('c0', 'c1', 'c2'))


def test_frozen_journal_digest_verifies_independently_without_curator_or_label_query(tmp_path, monkeypatch):
    import inspect
    api = train_api()
    assert 'expected_query_access_sha256' in inspect.signature(api.inspect_final_training_checkpoint).parameters
    invoke(tmp_path / 'run')
    final = json.loads((tmp_path / 'run' / 'final-model.json').read_text())
    monkeypatch.setattr(CuratedLabels, 'query_store', lambda *a, **kw: pytest.fail('independent gate reopened curator'))
    from vast_v3.data.schema import LabelStore
    monkeypatch.setattr(LabelStore, 'query', lambda *a, **kw: pytest.fail('independent gate bought labels'))
    kwargs = dict(expected_identity=final['identity'], config=RuntimeConfig.from_dict(final['config']),
                  permitted_case_ids=('c0', 'c1', 'c2'),
                  expected_query_access_sha256=final['query_access_sha256'],
                  expected_checkpoint_sha256=final['checkpoint_sha256'])
    checkpoint = tmp_path / 'run' / 'checkpoints' / 'latest.pt'
    verified = api.inspect_final_training_checkpoint(checkpoint, **kwargs)
    assert verified['labels_spent'] == 2
    journal_path = checkpoint.parent / 'query-access.json'
    journal = json.loads(journal_path.read_text())
    journal['labels']['manifest_hash'] = '0' * 64
    journal_path.write_text(json.dumps(journal))
    with pytest.raises(ValueError, match='paid label ledger'):
        api.inspect_final_training_checkpoint(checkpoint, **kwargs)


def test_independent_final_inspection_binds_checkpoint_and_journal_to_one_frozen_manifest(tmp_path):
    api = train_api()
    invoke(tmp_path / 'run')
    final = json.loads((tmp_path / 'run' / 'final-model.json').read_text())
    checkpoint = tmp_path / 'run' / 'checkpoints' / 'latest.pt'
    kwargs = dict(expected_identity=final['identity'], config=RuntimeConfig.from_dict(final['config']),
                  permitted_case_ids=('c0', 'c1', 'c2'),
                  expected_query_access_sha256=final['query_access_sha256'])
    with pytest.raises(ValueError, match='paid label ledger.*expected'):
        api.inspect_final_training_checkpoint(checkpoint, **kwargs)
    payload = torch.load(checkpoint, weights_only=True)
    first = next(value for value in payload['model'].values() if value.numel() and value.is_floating_point())
    first.flatten()[0] += 0.1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='frozen checkpoint identity'):
        api.inspect_final_training_checkpoint(checkpoint, expected_checkpoint_sha256=final['checkpoint_sha256'], **kwargs)

