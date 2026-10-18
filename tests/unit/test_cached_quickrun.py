"""Local synthetic cached-subset contract; no real-data readiness claim."""
from dataclasses import asdict, replace
import gzip
import importlib.util
import json
import inspect

import pytest
import torch

from test_case_cache import frozen_manifest, write_sources
from vast_v3.data import adapters
from vast_v3.data.case_cache import ObservableCaseCache
from vast_v3.data.repository import DatasetRepository, file_sha256
from vast_v3.runtime import RuntimeConfig, canonical_hash


def api():
    assert importlib.util.find_spec('vast_v3.training.quickrun'), 'cached quickrun is missing'
    from vast_v3.training import quickrun
    return quickrun


def cached_pool(tmp_path, count=33, publish=31):
    raw = tmp_path / 'raw'
    write_sources(raw)
    base = frozen_manifest()
    ids = tuple(f'c{i:03d}' for i in range(count))
    windows = tuple(replace(base.windows[0], case_id=cid, source_id=cid) for cid in (*ids, 'test'))
    manifest = replace(base, windows=windows, split=replace(base.split, train_case_ids=ids, test_case_ids=('test',)))
    cache = ObservableCaseCache(tmp_path / 'cache', max_bytes=100_000_000)
    repo = DatasetRepository.from_curated(manifest, raw, case_cache=cache)
    repo.pin_source_content()
    for cid in ids[:publish]:
        repo[cid]
    return repo


def rehash(snapshot):
    snapshot['identity_sha256'] = canonical_hash({k: v for k, v in snapshot.items() if k != 'identity_sha256'})
    return snapshot


def test_snapshot_freezes_all_published_train_ids_and_parent_identities(tmp_path, monkeypatch):
    q = api()
    repo = cached_pool(tmp_path)
    monkeypatch.setattr(adapters, 'load_aiops22_case', lambda *a, **k: pytest.fail('snapshot decoded raw input'))
    snapshot = q.create_snapshot(repo)
    assert snapshot['schema'] == 'vast-v3-cached-training-subset/1'
    assert snapshot['dataset'] == 'aiops2022_pre'
    assert snapshot['parent']['repository_identity_sha256'] == repo.identity_sha256
    assert snapshot['parent']['data_code_sha256'] == repo._cache_code_hash
    assert snapshot['train_case_ids'] == sorted(repo.manifest.split.train_case_ids[:31])
    assert snapshot['created_at_utc'].endswith('+00:00')
    assert len(snapshot['entries']) == 31
    assert q.validate_snapshot(snapshot, repo) == tuple(snapshot['train_case_ids'])
    for entry in snapshot['entries']:
        assert entry['sha256'] == file_sha256(repo.case_cache.root / entry['filename'])


@pytest.mark.parametrize('mutation', ['duplicate', 'test', 'foreign', 'unordered', 'small', 'sha', 'filename',
                                     'identity', 'parent', 'schema', 'time', 'missing_field', 'omit_entry'])
def test_snapshot_rejects_invalid_scope_and_bindings(tmp_path, mutation):
    q = api()
    repo = cached_pool(tmp_path)
    snapshot = q.create_snapshot(repo)
    if mutation in ('duplicate', 'test', 'foreign'):
        snapshot['train_case_ids'][-1] = snapshot['train_case_ids'][0] if mutation == 'duplicate' else mutation
    elif mutation == 'unordered':
        snapshot['train_case_ids'].reverse()
    elif mutation == 'small':
        snapshot['train_case_ids'] = snapshot['train_case_ids'][:30]
        snapshot['entries'] = snapshot['entries'][:30]
    elif mutation in ('sha', 'filename'):
        snapshot['entries'][0]['sha256' if mutation == 'sha' else 'filename'] = '0' * 64
    elif mutation == 'identity':
        snapshot['identity_sha256'] = '0' * 64
    elif mutation == 'parent':
        snapshot['parent']['repository_identity_sha256'] = '0' * 64
    elif mutation == 'schema':
        snapshot['schema'] = 'unknown'
    elif mutation == 'time':
        snapshot['created_at_utc'] = 'not-a-date'
    elif mutation == 'missing_field':
        del snapshot['dataset']
    else:
        snapshot['entries'].pop()
    if mutation != 'identity':
        rehash(snapshot)
    with pytest.raises(ValueError):
        q.validate_snapshot(snapshot, repo)


@pytest.mark.parametrize('corruption', ['missing', 'gzip', 'test', 'foreign', 'duplicate', 'partial', 'lock'])
def test_snapshot_generation_fails_closed(tmp_path, corruption):
    q = api()
    repo = cached_pool(tmp_path, publish=30 if corruption == 'missing' else 31)
    path = next(repo.case_cache.root.glob('*.json.gz'))
    if corruption == 'gzip':
        path.write_bytes(b'broken gzip')
    elif corruption == 'test':
        repo['test']
    elif corruption in ('foreign', 'duplicate'):
        envelope = json.loads(gzip.decompress(path.read_bytes()))
        if corruption == 'foreign':
            envelope['payload']['binding']['repository_identity_sha256'] = '0' * 64
        extra = repo.case_cache.root / ('f' * 64 + '.json.gz')
        extra.write_bytes(gzip.compress(json.dumps(envelope).encode()))
    elif corruption in ('partial', 'lock'):
        (repo.case_cache.root / ('.interrupted.tmp' if corruption == 'partial' else '.write.lock')).write_text('busy')
    with pytest.raises(ValueError):
        q.create_snapshot(repo)


@pytest.mark.parametrize('corruption', [None, 'missing', 'corrupt'])
def test_cache_only_view_never_falls_back_or_writes(tmp_path, monkeypatch, corruption):
    q = api()
    repo = cached_pool(tmp_path)
    snapshot = q.create_snapshot(repo)
    view = q.CachedTrainingCases(repo, snapshot)
    assert tuple(view) == tuple(snapshot['train_case_ids'])
    monkeypatch.setattr(adapters, 'load_aiops22_case', lambda *a, **k: pytest.fail('adapter fallback'))
    monkeypatch.setattr(ObservableCaseCache, 'store', lambda *a, **k: pytest.fail('cache write'))
    entry = snapshot['entries'][0]
    path = repo.case_cache.root / entry['filename']
    if corruption == 'missing':
        path.unlink()
    elif corruption == 'corrupt':
        path.write_bytes(b'broken')
    if corruption:
        with pytest.raises(ValueError):
            view[entry['case_id']]
    else:
        assert view[entry['case_id']].case_id == entry['case_id']
        assert view[entry['case_id']].case_id == entry['case_id']
        with pytest.raises(KeyError):
            view['test']
        with pytest.raises(KeyError):
            view['c032']


def test_quickrun_config_is_exact_six_round_b30_16_epochs_without_changing_default():
    q = api()
    for dataset in ('rcabench', 'aiops2022_pre'):
        config = q.quickrun_runtime_config(dataset)
        assert config == replace(RuntimeConfig(dataset=dataset), warmup_epochs=1, nonfinal_epochs=1,
                                 final_epochs=10, numeric_execution='metric-scalar-trace-bucket-v1')
        assert sum(s['epochs'] for s in config.schedule().stages('closed_loop', pool_size=31)) == 16
        assert config.schedule().query_batches(31) == (5, 5, 5, 5, 5, 5)
        assert sum(s['epochs'] for s in RuntimeConfig(dataset=dataset).schedule().stages('closed_loop', pool_size=31)) == 170


def rc_pool(tmp_path):
    from test_train_cli import SyntheticRepository
    from vast_v3.data.case_cache import BINDING_SCHEMA
    fixture = SyntheticRepository(33)
    raw = tmp_path / 'raw'
    for window in fixture.manifest.windows:
        base = raw / window.source_id
        base.mkdir(parents=True)
        for name in ('env.json', *(f'{phase}_{kind}.parquet' for phase in ('normal', 'abnormal')
                for kind in ('metrics', 'metrics_sum', 'metrics_histogram', 'logs', 'traces'))):
            (base / name).write_text('synthetic source inventory only')
    cache = ObservableCaseCache(tmp_path / 'cache', max_bytes=100_000_000)
    repo = DatasetRepository.from_curated(fixture.manifest, raw, case_cache=cache)
    pins = repo.pin_source_content()
    for window in fixture.manifest.windows[:31]:
        binding = {'schema': BINDING_SCHEMA, 'repository_identity_sha256': repo.identity_sha256,
            'repository_identity_kind': repo.identity_kind, 'case_id': window.case_id, 'window': asdict(window),
            'catalog': asdict(repo.manifest.catalog), 'source_sha256': {str(p): pins[str(p)] for p in repo._files[window.case_id]},
            'code_sha256': repo._cache_code_hash, 'static_topology_sha256': None}
        cache.store(binding, fixture.cases[window.case_id], {})
    return repo, fixture.labels()


def light_objectives(config, pre, **kwargs):
    from test_ordered_query_trace import SyntheticObjectives
    class DifferentiableU(SyntheticObjectives):
        def case_losses(self, case, *, epoch, roots, seed, components):
            values = super().case_losses(case, epoch=epoch, roots=roots, seed=seed, components=components)
            if 'U_soft' in components:
                values['U_soft'] = (self.model.weight - 3).square()
            return values
    torch.manual_seed(config.seed)
    obj = DifferentiableU(pre.fitted_case_ids)
    obj.model.frontend.preprocessing = pre
    obj.max_roots = config.effective_max_roots
    return obj


def invoke_cached(path, repo, labels, snapshot, **kwargs):
    from vast_v3.cli import train
    q = api()
    assert 'training_subset' in inspect.signature(train.run_training).parameters, 'explicit subset route missing'
    return train.run_training(repository=repo, curated_labels=labels,
        config=q.quickrun_runtime_config('rcabench'), text_encoder=lambda texts: torch.zeros(len(texts), 384),
        text_lock={'synthetic': 'fixed'}, run_dir=path, code_hash='f' * 64,
        training_subset=q.CachedTrainingCases(repo, snapshot), quickrun_provenance=snapshot,
        convolution_policy='fp32-no-cudnn-tf32-v1', **kwargs)


def test_real_trainer_cached_subset_resume_and_u_gate(tmp_path, monkeypatch):
    from vast_v3.cli import train
    from vast_v3.data.curation import CuratedLabels
    from vast_v3.data.preprocessing import Preprocessor
    q = api()
    repo, labels = rc_pool(tmp_path)
    snapshot = q.create_snapshot(repo)
    monkeypatch.setattr(train, 'build_objectives', light_objectives)
    monkeypatch.setattr(adapters, 'load_rcabench_case', lambda *a, **k: pytest.fail('raw fallback'))
    monkeypatch.setattr(CuratedLabels, 'evaluation_store', lambda *a, **k: pytest.fail('test roots'))
    full = invoke_cached(tmp_path / 'full', repo, labels, snapshot)
    paused = invoke_cached(tmp_path / 'part', repo, labels, snapshot, max_updates=93)
    assert not paused['complete'] and not paused['eligible_for_quickstart']
    monkeypatch.setattr(Preprocessor, 'fit_repository', lambda *a, **k: pytest.fail('resume refit'))
    resumed = invoke_cached(tmp_path / 'part', repo, labels, snapshot, resume=True)
    assert full['complete'] and resumed['complete'] and resumed['schedule_epochs'] == 16
    assert resumed['labels_spent'] == 30 and resumed['updates'] == 128
    assert not resumed['eligible_for_demo'] and not resumed['eligible_for_quickstart']  # no CLI provenance
    a, b = [torch.load(tmp_path / name / 'checkpoints/latest.pt', weights_only=True) for name in ('full', 'part')]
    assert a['extra_state']['query_trace'] == b['extra_state']['query_trace']
    assert a['extra_state']['loop']['quickrun_u_evidence'] == b['extra_state']['loop']['quickrun_u_evidence']
    assert b['extra_state']['loop']['quickrun_u_evidence']['parameter_update_steps'] > 0
    assert set(b['query_sets']['L']) <= set(snapshot['train_case_ids'])
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    final = json.loads((tmp_path / 'part/final-model.json').read_text())
    assert final['purpose'] == q.QUICKRUN_PURPOSE
    assert final['identity']['subset_snapshot_hash'] == snapshot['identity_sha256']
    pre = json.loads((tmp_path / 'part/preprocessing.json').read_text())
    assert pre['preprocessing']['fitted_case_ids'] == snapshot['train_case_ids']


def test_resume_rejects_different_snapshot_before_label_store(tmp_path, monkeypatch):
    from vast_v3.cli import train
    from vast_v3.data.curation import CuratedLabels
    q = api()
    repo, labels = rc_pool(tmp_path)
    snapshot = q.create_snapshot(repo)
    monkeypatch.setattr(train, 'build_objectives', light_objectives)
    invoke_cached(tmp_path / 'part', repo, labels, snapshot, max_updates=1)
    changed = json.loads(json.dumps(snapshot))
    changed['created_at_utc'] = '2026-01-01T00:00:00+00:00'
    rehash(changed)
    monkeypatch.setattr(CuratedLabels, 'query_store', lambda *a, **k: pytest.fail('changed subset opened label store'))
    with pytest.raises(ValueError, match='identity'):
        invoke_cached(tmp_path / 'part', repo, labels, changed, resume=True)


def cli_api():
    assert importlib.util.find_spec('vast_v3.cli.quickrun'), 'cached quickrun CLI missing'
    from vast_v3.cli import quickrun
    return quickrun


def cli_inputs(tmp_path, monkeypatch):
    q = cli_api()
    repo, labels = rc_pool(tmp_path)
    from vast_v3.data.repository import manifest_to_dict
    for name, value in [('manifest', manifest_to_dict(repo.manifest)), ('labels', labels.to_dict()),
                         ('pins', repo._pins), ('lock', {'synthetic': 'fixed'})]:
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    monkeypatch.setattr(q, 'build_standard_manifest', lambda _: repo.manifest)
    args = ['--dataset', 'rcabench', '--manifest', str(tmp_path / 'manifest.json'),
            '--raw-root', str(tmp_path / 'raw'), '--sealed-labels', str(tmp_path / 'labels.json'),
            '--source-pins', str(tmp_path / 'pins.json'), '--text-lock', str(tmp_path / 'lock.json'),
            '--text-directory', str(tmp_path / 'text'), '--case-cache-root', str(tmp_path / 'cache'),
            '--case-cache-max-bytes', '100000000', '--snapshot', str(tmp_path / 'subset.json'),
            '--run-dir', str(tmp_path / 'run'), '--threads', '1']
    return q, args


def test_snapshot_cli_does_not_load_text_or_create_run(tmp_path, monkeypatch):
    q, args = cli_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(q, 'FrozenSentenceEncoder', lambda *a, **k: pytest.fail('snapshot loaded text model'))
    assert q.main(['snapshot', *args]) == 0
    assert (tmp_path / 'subset.json').is_file() and not (tmp_path / 'run').exists()
    before = (tmp_path / 'subset.json').read_bytes()
    assert q.main(['snapshot', *args]) == 0
    assert (tmp_path / 'subset.json').read_bytes() == before


def test_quickrun_help_does_not_offer_rebuilding_original_inputs(capsys):
    q = cli_api()
    with pytest.raises(SystemExit) as result:
        q.main(['--help'])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert 'otherwise freeze' not in help_text
    assert 'original static topology' in help_text
    assert '--epochs' not in help_text and '--budget' not in help_text


def test_quickrun_cli_rejects_changed_standard_manifest_before_source_access(tmp_path, monkeypatch):
    q, args = cli_inputs(tmp_path, monkeypatch)
    path = tmp_path / 'manifest.json'
    value = json.loads(path.read_text())
    value['windows'][0]['incident_end'] = 9
    path.write_text(json.dumps(value))
    monkeypatch.setattr(q.DatasetRepository, 'from_curated', lambda *a, **k: pytest.fail('changed manifest read sources'))
    with pytest.raises(ValueError, match='complete approved frozen'):
        q.main(['snapshot', *args])
    assert not (tmp_path / 'subset.json').exists()


@pytest.mark.parametrize('has_u', [False, True])
def test_cli_quickstart_gate_requires_actual_u_backward_and_updates(tmp_path, monkeypatch, has_u):
    from vast_v3.cli import train
    from test_ordered_query_trace import SyntheticObjectives
    q, args = cli_inputs(tmp_path, monkeypatch)
    assert q.main(['snapshot', *args]) == 0
    monkeypatch.setattr(q, 'FrozenSentenceEncoder', lambda *a, **k: lambda texts: torch.zeros(len(texts), 384))
    def objectives(config, pre, **kwargs):
        if has_u:
            return light_objectives(config, pre, **kwargs)
        obj = SyntheticObjectives(pre.fitted_case_ids)
        obj.model.frontend.preprocessing = pre
        obj.max_roots = config.effective_max_roots
        return obj
    monkeypatch.setattr(train, 'build_objectives', objectives)
    assert q.main(['run', *args]) == 0
    final = json.loads((tmp_path / 'run/final-model.json').read_text())
    assert final['eligible_for_demo'] is False
    assert final['formal_experiment'] is False
    assert final['eligible_for_quickstart'] is has_u
    assert final['quickstart_ready'] is has_u
    assert final['schedule_epochs'] == 16 and final['labels_spent'] == 30
    assert final['purpose'] == api().QUICKRUN_PURPOSE
    assert len(json.loads((tmp_path / 'run/checkpoints/query-trace.json').read_text())['rounds']) == 6
    assert json.loads((tmp_path / 'run/run.json').read_text())['purpose'] == api().QUICKRUN_PURPOSE


def test_quickrun_marker_and_status_explicitly_disclaim_formal_experiment(tmp_path):
    q = api()
    reporter = q.QuickrunProgress(tmp_path / 'run')
    assert json.loads((tmp_path / 'run/run.json').read_text())['formal_experiment'] is False
    assert reporter.update('training')['formal_experiment'] is False
    resumed = q.QuickrunProgress(tmp_path / 'run', resume=True)
    assert resumed.update('diagnostic_pause')['formal_experiment'] is False


def test_real_neural_objectives_observer_preserves_parameters_rng_and_resume(tmp_path):
    from types import SimpleNamespace
    from test_ranker import setup
    from vast_v3.data.schema import LabelStore
    from vast_v3.training.loop import Trainer
    from vast_v3.training.objectives import TrainingObjectives
    from vast_v3.training.schedule import TrainingSchedule
    q = api()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    def make(cls, path):
        torch.manual_seed(123)
        case, model = setup()
        cases = {f'c{i}': replace(case, case_id=f'c{i}') for i in range(2)}
        labels = LabelStore({cid: ('a',) for cid in cases}, budget=1, max_roots=2)
        schedule = TrainingSchedule(budget=1, query_batch_size=1, warmup_epochs=1,
            nonfinal_epochs=1, final_epochs=1, u_disabled_epochs=0, u_ramp_epochs=0)
        extra = {'reporter': SimpleNamespace(update=lambda *a, **k: None)} if cls is q.QuickrunTrainer else {}
        return cls(TrainingObjectives(model, max_roots=2), cases, labels, schedule,
            checkpoint_dir=path, identity=dict(data_hash='synthetic', config_hash='unit',
                preprocessing_hash='pre', code_hash='code'), **extra)
    try:
        original = make(Trainer, tmp_path / 'ordinary')
        original.run()
        original_rng = torch.get_rng_state().clone()
        observed = make(q.QuickrunTrainer, tmp_path / 'observed')
        observed.run(max_updates=1)
        resumed = make(q.QuickrunTrainer, tmp_path / 'observed')
        resumed.run(resume=True)
        assert torch.equal(torch.get_rng_state(), original_rng)
        for name, value in original.objectives.state_dict().items():
            torch.testing.assert_close(resumed.objectives.state_dict()[name], value, rtol=0, atol=0)
        evidence = resumed.state['quickrun_u_evidence']
        assert evidence['backward_terms'] > 0 and evidence['parameter_update_steps'] > 0
    finally:
        torch.set_num_threads(previous)
