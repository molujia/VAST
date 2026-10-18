"""Synthetic review probes for control boundaries; production code is unchanged."""

from dataclasses import replace

import pytest
import torch

from vast_v3.data.schema import Entity, LabelStore, NumericSeries, ObservableCase, Relation
from vast_v3.data.preprocessing import Preprocessor
from vast_v3.encoders.frontends import NativeFrontend
from vast_v3.models.ranker import VASTModel
from vast_v3.training.loop import Trainer
from vast_v3.training.objectives import TrainingObjectives
from vast_v3.training.schedule import TrainingSchedule


IDENTITY = dict(data_hash='review-data', config_hash='review-config',
                preprocessing_hash='review-pre', code_hash='review-code')


def setup():
    case = ObservableCase('x', (Entity('a', 'service'), Entity('b', 'service')),
        0., 3., 6.,
        (NumericSeries('metric', 'a', 'cpu', (0., 1., 2., 3., 4.), (0., 1., 0., 3., 2.)),
         NumericSeries('metric', 'b', 'cpu', (0., 1., 2., 3., 4.), (0., 1., 0., 1., 3.))),
        (), (Relation('ab', 'a', 'b', 'call', 0., 6.),))
    pre = Preprocessor.fit([case], permitted_case_ids={'x'})
    front = NativeFrontend(pre, ('service',), ('call',), ('cpu',),
                           text_encoder=lambda texts: torch.zeros(len(texts), 384))
    return case, VASTModel(front)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trainer(directory, *, mask_seed=42, budget=1, labels=None, case_count=2):
    torch.manual_seed(61)
    case, model = setup()
    pool = {'x': case}
    pool.update({f'u{i}': replace(case, case_id=f'u{i}') for i in range(case_count - 1)})
    labels = labels or LabelStore({cid: ('a',) for cid in pool}, budget=budget)
    objective = TrainingObjectives(model, variant='A3', seed=mask_seed)
    schedule = TrainingSchedule(budget=budget, query_batch_size=budget,
                                warmup_epochs=0, nonfinal_epochs=1, final_epochs=1,
                                u_disabled_epochs=0, u_ramp_epochs=0)
    return Trainer(objective, pool, labels, schedule, checkpoint_dir=directory,
                   identity=IDENTITY, seed=42)


def test_closed_loop_rejects_an_incoming_ledger_over_schedule_budget(tmp_path):
    run = trainer(tmp_path)
    labels = LabelStore({'x': ('a',), 'u0': ('a',)}, budget=2)
    labels.query(run.cases['x'])
    labels.query(run.cases['u0'])
    run.labels = labels
    with pytest.raises(ValueError, match='budget|ledger|queried'):
        run.run()


def test_fixed_list_rejects_outside_pending_reservation_before_answer(tmp_path):
    run = trainer(tmp_path)
    def interrupt_reservation(state):
        raise RuntimeError('interrupted before answer')
    with pytest.raises(RuntimeError, match='before answer'):
        run.labels.query(run.cases['u0'], persist=interrupt_reservation)
    assert run.labels.pending == {'u0'} and not run.labels.queried
    with pytest.raises(ValueError, match='frozen fixed list'):
        run.run(regime='fixed_list', fixed_query_ids=['x'])
    assert not run.labels.queried, 'fixed-list validation revealed an outside-list root first'


def test_resume_binds_the_mask_seed_not_just_trainer_seed(tmp_path):
    run = trainer(tmp_path, mask_seed=42, case_count=5)
    run.run(regime='fixed_list', fixed_query_ids=['x'], max_updates=1)
    resumed = trainer(tmp_path, mask_seed=999, case_count=5)
    with pytest.raises(ValueError, match='identity|seed'):
        resumed.run(regime='fixed_list', fixed_query_ids=['x'], resume=True)


def test_closed_loop_mid_batch_crash_resumes_exact_paid_queries_and_weights(tmp_path):
    full = trainer(tmp_path / 'full', budget=3, case_count=4)
    expected = full.run()
    interrupted = trainer(tmp_path / 'resume', budget=3, case_count=4)
    persist = interrupted._persist_query
    def stop_after_first_answer(state):
        persist(state)
        if len(state['revealed']) == 1:
            raise RuntimeError('interruption after first answer')
    interrupted._persist_query = stop_after_first_answer
    with pytest.raises(RuntimeError, match='interruption'):
        interrupted.run()
    resumed = trainer(tmp_path / 'resume', budget=3, case_count=4)
    actual = resumed.run(resume=True)
    assert actual == expected
    assert resumed.labels.spent == 3
    for name, value in full.objectives.state_dict().items():
        torch.testing.assert_close(resumed.objectives.state_dict()[name], value, rtol=0, atol=0)
