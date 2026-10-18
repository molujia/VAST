"""Bounded synthetic review probes for resume and durable label accounting."""

from dataclasses import replace

import pytest
import torch

from test_frontend_contract_review import fixture_case, front
from vast_v3.data.schema import LabelStore
from vast_v3.models.ranker import VASTModel
from vast_v3.training.loop import Trainer
from vast_v3.training.objectives import TrainingObjectives
from vast_v3.training.schedule import TrainingSchedule


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_trainer(directory, *, seed=17, count=5, bad_roots=False):
    torch.manual_seed(seed)
    case = replace(fixture_case(), logs=())
    pool = {f"c{i}": replace(case, case_id=f"c{i}") for i in range(count)}
    # Fixed-list learners share preprocessing fitted only on their delivered L.
    # The remaining pool objects are transform-only, including on resume.
    model = VASTModel(front(pool['c0']))
    objectives = TrainingObjectives(model, max_roots=2)
    labels = LabelStore({cid: (("a", "b") if bad_roots else ("a",)) for cid in pool},
                        budget=1, max_roots=1 if bad_roots else 2)
    schedule = TrainingSchedule(budget=1, query_batch_size=1, warmup_epochs=0,
                                nonfinal_epochs=1, final_epochs=1,
                                u_disabled_epochs=0, u_ramp_epochs=0)
    return Trainer(objectives, pool, labels, schedule, checkpoint_dir=directory,
                   identity=dict(data_hash="review-data", config_hash="review-config",
                                 preprocessing_hash="review-pre", code_hash="review-code"))


def test_mid_epoch_resume_matches_uninterrupted_weights_and_reports(tmp_path):
    full = make_trainer(tmp_path / "full")
    assert full.objectives.model.frontend.preprocessing.fitted_case_ids == ('c0',)
    expected = full.run(regime="fixed_list", fixed_query_ids=["c0"])
    partial = make_trainer(tmp_path / "resumed")
    paused = partial.run(regime="fixed_list", fixed_query_ids=["c0"], max_updates=1)
    assert not paused["complete"] and partial.state["slot"] == 1
    cached = dict(partial.objectives._libraries)
    resumed = make_trainer(tmp_path / "resumed", seed=999)
    actual = resumed.run(regime="fixed_list", fixed_query_ids=["c0"], resume=True)
    assert actual == expected
    assert resumed.objectives._libraries == cached
    for name, value in full.objectives.state_dict().items():
        torch.testing.assert_close(resumed.objectives.state_dict()[name], value, rtol=0, atol=0)


def test_failed_label_access_has_durable_checkpoint_record(tmp_path):
    trainer = make_trainer(tmp_path, count=2, bad_roots=True)
    with pytest.raises(ValueError, match="prior"):
        trainer.run()
    assert trainer.labels.spent == 1
    # The resident store correctly charges/blocks. A restart must not erase the
    # access because the round-level checkpoint was postponed until success.
    assert (tmp_path / "latest.pt").exists(), "accessed labels have no durable training snapshot"
