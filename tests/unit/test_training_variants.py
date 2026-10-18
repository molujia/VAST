"""Executable A0--A6 controls and exact disabled-path contracts."""
from dataclasses import replace
from collections.abc import Mapping
import random

import pytest
import torch

from test_ranker import setup
from vast_v3.data.schema import LabelStore
from vast_v3.training.loop import Trainer
from vast_v3.training.objectives import TrainingObjectives
from vast_v3.training.schedule import TrainingSchedule


IDENTITY = dict(data_hash='data', config_hash='config', preprocessing_hash='pre', code_hash='code')


@pytest.fixture(autouse=True)
def single_threaded_fixture():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('variant,expected', [
    ('A0', {'pn'}), ('A1', {'pn', 'local_recovery'}),
    ('A2', {'pn', 'local_recovery', 'process_recovery', 'invariance'}),
    ('A3', set(TrainingObjectives.names)), ('A4', set(TrainingObjectives.names)),
    ('A5', set(TrainingObjectives.names)), ('A6', set(TrainingObjectives.names)-{'U_soft'}),
])
def test_variant_components_and_fixed_loss_weights(variant, expected):
    schedule = TrainingSchedule(observation_weight=.6, source_weight=.7, u_weight=.8,
                                u_disabled_epochs=0, u_ramp_epochs=0)
    weights = schedule.loss_weights(variant, epoch=0, phase='joint', u_enabled=True)
    assert {k for k, v in weights.items() if v} == expected
    assert weights['local_recovery'] == pytest.approx(.6 if variant == 'A1' else 0 if variant == 'A0' else .2)
    assert weights['source'] == (.7 if 'source' in expected else 0)
    warm = schedule.loss_weights(variant, epoch=0, phase='warmup', u_enabled=True)
    assert warm['pn'] == warm['source'] == warm['U_soft'] == 0
    assert warm['local_recovery'] == weights['local_recovery']


class LOnlyMapping(Mapping):
    def __init__(self, case):
        self.case = case
        self.reads = []

    def __iter__(self):
        return iter(('x', 'u'))

    def __len__(self):
        return 2

    def __getitem__(self, key):
        self.reads.append(key)
        assert key == 'x', 'A0 read a U observation'
        return self.case


def test_a0_skips_warmup_and_never_reads_unlabeled_case_objects(tmp_path):
    case, model = setup()
    pool = LOnlyMapping(case)
    objectives = TrainingObjectives(model, variant='A0')
    schedule = TrainingSchedule(budget=1, warmup_epochs=2, final_epochs=1)
    labels = LabelStore({'x': ('a',)}, budget=1)
    trainer = Trainer(objectives, pool, labels, schedule, checkpoint_dir=tmp_path,
                      identity=IDENTITY, pool_size=9)
    result = trainer.run(regime='fixed_list', fixed_query_ids=['x'])
    assert result['complete'] and result['updates'] == 3
    assert all(row['stage'] == 'joint' for row in result['epochs'])
    assert result['epochs'][0]['pn_count'] == 3
    assert all(result['epochs'][0][name+'_count'] == 0 for name in objectives.names if name != 'pn')
    assert pool.reads and set(pool.reads) == {'x'}
    assert all(p.grad is None for p in objectives.decoder.parameters())


@pytest.mark.parametrize('variant,components', [('A0', set()), ('A1', {'process_recovery', 'invariance', 'source', 'U_soft'}),
                                               ('A2', {'source', 'U_soft'}), ('A6', {'U_soft'})])
def test_disabled_components_do_not_touch_case_or_roots(variant, components):
    _, model = setup()
    objectives = TrainingObjectives(model, variant=variant)
    class Forbidden:
        def __iter__(self):
            raise AssertionError('disabled branch read roots')
        def __getattribute__(self, name):
            raise AssertionError('disabled branch read case')
    values = objectives.case_losses(Forbidden(), epoch=0, roots=Forbidden(), components=components)
    assert all(value is None for value in values.values())


@pytest.mark.parametrize('name', ['observation_weight', 'source_weight', 'u_weight'])
@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_loss_weights_must_be_finite_nonnegative(name, value):
    with pytest.raises(ValueError, match=name):
        TrainingSchedule(**{name: value})


def short_schedule(**kwargs):
    return TrainingSchedule(budget=1, warmup_epochs=1, final_epochs=1,
                            u_disabled_epochs=0, u_ramp_epochs=0, **kwargs)


def test_zero_source_weights_are_exactly_a2_in_parameters_rng_and_streams(tmp_path):
    results = []
    for variant in ('A2', 'A3'):
        torch.manual_seed(50); random.seed(50)
        case, model = setup()
        obj = TrainingObjectives(model, variant=variant)
        schedule = short_schedule(source_weight=0., u_weight=0.) if variant == 'A3' else short_schedule()
        trainer = Trainer(obj, {'x': case, 'u': replace(case, case_id='u')},
                          LabelStore({'x': ('a',)}, budget=1), schedule,
                          checkpoint_dir=tmp_path/variant, identity=IDENTITY)
        result = trainer.run(regime='fixed_list', fixed_query_ids=['x'])
        assert not obj._libraries
        assert all(p.grad is None for p in obj.decoder.parameters())
        results.append((obj.state_dict(), torch.get_rng_state(), random.getstate(), result['epochs']))
    for name in results[0][0]:
        assert torch.equal(results[0][0][name], results[1][0][name]), name
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2:] == results[1][2:]


@pytest.mark.parametrize('change', ['variant', 'weight', 'seed', 'regime', 'fixed_list'])
def test_resume_rejects_training_configuration_drift(tmp_path, change):
    case, model = setup()
    pool = {'x': case, 'u': replace(case, case_id='u')}
    original = Trainer(TrainingObjectives(model, variant='A2'), pool,
                       LabelStore({'x': ('a',), 'u': ('b',)}, budget=1), short_schedule(),
                       checkpoint_dir=tmp_path, identity=IDENTITY)
    original.run(regime='fixed_list', fixed_query_ids=['x'], max_updates=1)
    _, second_model = setup()
    # A drifted list needs matching preprocessing so identity, not leakage,
    # remains the reason the checkpoint cannot load.
    if change == 'fixed_list':
        second_model.frontend.preprocessing = replace(second_model.frontend.preprocessing, fitted_case_ids=('u',))
    changed = Trainer(TrainingObjectives(second_model, variant='A3' if change == 'variant' else 'A2'), pool,
                      LabelStore({'x': ('a',), 'u': ('b',)}, budget=1),
                      short_schedule(observation_weight=.3) if change == 'weight' else short_schedule(),
                      checkpoint_dir=tmp_path, identity=IDENTITY, seed=43 if change == 'seed' else 42)
    with pytest.raises(ValueError, match='identity'):
        changed.run(regime='closed_loop' if change == 'regime' else 'fixed_list', resume=True,
                    fixed_query_ids=['u'] if change == 'fixed_list' else ['x'])


def test_fixed_list_interruption_resumes_remaining_paid_list(tmp_path):
    from vast_v3.data.preprocessing import Preprocessor
    case, model = setup(); other = replace(case, case_id='u')
    model.frontend.preprocessing = Preprocessor.fit([case, other], permitted_case_ids={'x', 'u'})
    pool = {'x': case, 'u': other}
    schedule = TrainingSchedule(budget=2, warmup_epochs=0, final_epochs=1)
    first = Trainer(TrainingObjectives(model, variant='A0'), pool,
                    LabelStore({'x': ('a',), 'u': ('b',)}, budget=2), schedule,
                    checkpoint_dir=tmp_path, identity=IDENTITY)
    persist = first._persist_query
    def stop_after_first_answer(state):
        persist(state)
        if set(state['revealed']) == {'x'}:
            raise RuntimeError('simulated interruption after durable first answer')
    first._persist_query = stop_after_first_answer
    with pytest.raises(RuntimeError, match='interruption'):
        first.run(regime='fixed_list', fixed_query_ids=['x', 'u'])
    restored = Trainer(TrainingObjectives(model, variant='A0'), pool,
                       LabelStore({'x': ('a',), 'u': ('b',)}, budget=2), schedule,
                       checkpoint_dir=tmp_path, identity=IDENTITY)
    result = restored.run(regime='fixed_list', fixed_query_ids=['x', 'u'], resume=True)
    assert result['complete'] and set(result['queried_ids']) == {'x', 'u'}
    assert restored.labels.spent == 2


def test_fixed_list_rejects_preprocessing_fitted_on_U_before_query(tmp_path):
    case, model = setup()
    model.frontend.preprocessing = replace(model.frontend.preprocessing, fitted_case_ids=('x', 'u'))
    labels = LabelStore({'x': ('a',)}, budget=1)
    trainer = Trainer(TrainingObjectives(model, variant='A2'), {'x': case, 'u': replace(case, case_id='u')},
                      labels, short_schedule(), checkpoint_dir=tmp_path, identity=IDENTITY)
    with pytest.raises(ValueError, match='preprocessing.*fixed.*L'):
        trainer.run(regime='fixed_list', fixed_query_ids=['x'])
    assert labels.spent == 0


def test_a1_local_log_targets_require_log_context_even_if_metrics_are_visible():
    case, model = setup()
    objectives = TrainingObjectives(model, variant='A1')
    output = model(case, seed=42)
    assert output.front.local_context_mask[0, 0]
    assert not output.front.log_context_mask[0, 0]
    for modality, width in [('text', 384), ('log_frequency', 1)]:
        record = dict(modality=modality, unit='a', index=0, bins=[0], target=torch.ones(1, width))
        assert objectives._recovery(output, [record], process=False) is None


def test_a5_keeps_true_source_completion_but_omits_wrong_source_comparison():
    torch.manual_seed(51)
    case, model = setup()
    full = TrainingObjectives(model, variant='A3', max_roots=2)
    no_wrong = TrainingObjectives(setup()[1], variant='A5', max_roots=2)
    no_wrong.load_state_dict(full.state_dict())
    prepared = full.views.prepare(case, epoch=0)
    from vast_v3.training.objectives import keyed_seed
    output = full.model(prepared.case, seed=keyed_seed(42, 'x', 0, 'fragments'))
    records = full._targets(case, prepared, output)
    errors, _ = full._source_errors(output, records, (frozenset({0}),))
    values = no_wrong.case_losses(case, epoch=0, roots={'a'}, components={'source'})
    assert values['source'] is not None
    torch.testing.assert_close(values['source'], errors[0])
    assert full.case_losses(case, epoch=0, roots={'a'}, components={'source'})['source'] > errors[0]


def test_a4_bypass_is_bias_free_visible_h_and_does_not_perturb_shared_initialization():
    built = []
    for variant in ('A3', 'A4'):
        torch.manual_seed(52)
        case, model = setup()
        obj = TrainingObjectives(model, variant=variant)
        built.append((obj, torch.get_rng_state()))
    full, bypass = built[0][0], built[1][0]
    for name, value in full.state_dict().items():
        assert torch.equal(value, bypass.state_dict()[name]), name
    assert torch.equal(built[0][1], built[1][1])
    assert hasattr(bypass, 'bypass_heads')
    assert all(head.bias is None for head in bypass.bypass_heads.values())
    with torch.no_grad():
        for parameter in full.decoder.parameters():
            parameter.zero_()
        for parameter in bypass.decoder.parameters():
            parameter.zero_()
    for obj in (full, bypass):
        obj.case_losses(case, epoch=0, roots={'a'}, components={'source'})['source'].backward()
    assert full.model.frontend.fusion[0].weight.grad.abs().sum() == 0
    assert bypass.model.frontend.fusion[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('regime', ['all_label', 'lofo'])
def test_all_label_regimes_reject_undelivered_pool_slots_before_reading_roots(tmp_path, regime):
    case, model = setup()
    labels = LabelStore({'x': ('a',)}, budget=1)
    trainer = Trainer(TrainingObjectives(model, variant='A0'), {'x': case}, labels,
                      short_schedule(), checkpoint_dir=tmp_path, identity=IDENTITY, pool_size=2)
    with pytest.raises(ValueError, match='complete permitted pool'):
        trainer.run(regime=regime)
    assert labels.spent == 0


@pytest.mark.parametrize('variant', ['A1', 'A2', 'A6'])
def test_warmup_has_no_label_decoder_or_rank_head_gradients(tmp_path, variant):
    case, model = setup()
    objectives = TrainingObjectives(model, variant=variant)
    labels = LabelStore({'x': ('a',)}, budget=1)
    trainer = Trainer(objectives, {'x': case}, labels, short_schedule(),
                      checkpoint_dir=tmp_path, identity=IDENTITY)
    trainer.run(max_updates=1)
    assert labels.spent == 0
    assert all(p.grad is None for p in model.rank.parameters())
    assert all(p.grad is None for p in objectives.decoder.parameters())
    assert model.frontend.metric_cnn[0].weight.grad.abs().sum() > 0
    if variant == 'A1':
        assert all(p.grad is None for p in objectives.process_heads.parameters())
        assert all(p.grad is None for p in model.scope.parameters())


def test_a6_U_keeps_observation_loss_but_never_constructs_source_library(tmp_path):
    case, model = setup()
    objectives = TrainingObjectives(model, variant='A6')
    trainer = Trainer(objectives, {'x': case, 'u': replace(case, case_id='u')},
                      LabelStore({'x': ('a',)}, budget=1), short_schedule(),
                      checkpoint_dir=tmp_path, identity=IDENTITY)
    result = trainer.run(regime='fixed_list', fixed_query_ids=['x'])
    assert not objectives._libraries
    assert result['epochs'][-1]['source_count'] == 1
    assert result['epochs'][-1]['U_soft_count'] == 0
    assert result['epochs'][-1]['local_recovery_count'] == 2


def test_a2_full_resume_matches_uninterrupted_parameters_and_epoch_records(tmp_path):
    runs = []
    for interrupted in (False, True):
        torch.manual_seed(53)
        case, model = setup()
        obj = TrainingObjectives(model, variant='A2')
        schedule = short_schedule()
        pool = {'x': case, 'u': replace(case, case_id='u')}
        trainer = Trainer(obj, pool, LabelStore({'x': ('a',)}, budget=1), schedule,
                          checkpoint_dir=tmp_path/str(interrupted), identity=IDENTITY)
        if interrupted:
            trainer.run(regime='fixed_list', fixed_query_ids=['x'], max_updates=1)
            obj = TrainingObjectives(setup()[1], variant='A2')
            trainer = Trainer(obj, pool, LabelStore({'x': ('a',)}, budget=1), schedule,
                              checkpoint_dir=tmp_path/str(interrupted), identity=IDENTITY)
        result = trainer.run(regime='fixed_list', fixed_query_ids=['x'], resume=interrupted)
        runs.append((obj.state_dict(), result['epochs']))
    for name in runs[0][0]:
        assert torch.equal(runs[0][0][name], runs[1][0][name]), name
    assert runs[0][1] == runs[1][1]


def test_zero_source_equivalence_is_independent_of_python_hash_seed(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = """
import hashlib, sys, torch
sys.path.insert(0, 'tests/unit')
from test_ranker import setup
from dataclasses import replace
from vast_v3.data.schema import LabelStore
from vast_v3.training.objectives import TrainingObjectives
from vast_v3.training.schedule import TrainingSchedule
from vast_v3.training.loop import Trainer
torch.set_num_threads(1)
torch.manual_seed(59)
case, model = setup()
variant = sys.argv[2]
objectives = TrainingObjectives(model, variant=variant)
weights = dict(source_weight=0., u_weight=0.) if variant == 'A3' else {}
schedule = TrainingSchedule(budget=1, warmup_epochs=1, final_epochs=1, **weights)
trainer = Trainer(objectives, {'x': case, 'u': replace(case, case_id='u')},
    LabelStore({'x': ('a',)}, budget=1), schedule, checkpoint_dir=sys.argv[1],
    identity=dict(data_hash='d', config_hash='c', preprocessing_hash='p', code_hash='s'))
trainer.run(regime='fixed_list', fixed_query_ids=['x'])
digest = hashlib.sha256()
for name, value in objectives.state_dict().items():
    digest.update(name.encode()); digest.update(value.numpy().tobytes())
print(digest.hexdigest())
"""
    hashes = []
    workspace = Path(__file__).resolve().parents[2]
    for variant, hash_seed in [('A2', '1'), ('A3', '2')]:
        env = {**os.environ, 'PYTHONHASHSEED': hash_seed,
               'PYTHONPATH': str(workspace/'src')}
        result = subprocess.run([sys.executable, '-c', script, str(tmp_path/variant), variant],
                                cwd=workspace, env=env, capture_output=True, text=True, check=True)
        hashes.append(result.stdout.strip())
    assert hashes[0] == hashes[1]


def test_resume_validates_fixed_list_journal_before_loading_revealed_roots(tmp_path):
    case, model = setup()
    pool = {'x': case, 'u': replace(case, case_id='u')}
    labels = LabelStore({'x': ('a',), 'u': ('b',)}, budget=1)
    original = Trainer(TrainingObjectives(model, variant='A0'), pool, labels,
                       short_schedule(), checkpoint_dir=tmp_path, identity=IDENTITY)
    def interruption_before_answer(cid):
        raise RuntimeError('stopped before answer')
    original._access = interruption_before_answer
    with pytest.raises(RuntimeError, match='before answer'):
        original.run(regime='fixed_list', fixed_query_ids=['x'])
    # A conflicting persisted access ledger must not reveal an outside-list
    # answer to the newly constructed learner before the scope check runs.
    outside = LabelStore({'x': ('a',), 'u': ('b',)}, budget=1)
    outside.query(pool['u'])
    original._persist_query(outside.state_dict())
    resumed_labels = LabelStore({'x': ('a',), 'u': ('b',)}, budget=1)
    resumed = Trainer(TrainingObjectives(setup()[1], variant='A0'), pool, resumed_labels,
                      short_schedule(), checkpoint_dir=tmp_path, identity=IDENTITY)
    with pytest.raises(ValueError, match='frozen fixed list'):
        resumed.run(regime='fixed_list', fixed_query_ids=['x'], resume=True)
    assert not resumed_labels.queried and not resumed_labels.pending
