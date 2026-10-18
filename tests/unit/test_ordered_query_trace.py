"""Synthetic bounded Trainer tests; none is a real-data or formal B30 run."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from vast_v3.data.schema import Entity, LabelStore, ObservableCase
from vast_v3.training.loop import Trainer
from vast_v3.training.schedule import TrainingSchedule
from vast_v3.training.variants import NAMES


class SyntheticModel(torch.nn.Module):
    def __init__(self, case_ids):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.frontend = SimpleNamespace(preprocessing=SimpleNamespace(
            identity="synthetic-preprocessing", fitted_case_ids=tuple(case_ids)))
        self.encodings = []

    def forward(self, case, *, seed):
        self.encodings.append(case.case_id)
        # Deliberate global-RNG consumption tests the post-selection checkpoint.
        torch.rand(())
        return SimpleNamespace(unmarked=torch.empty(0, 2), fragments=())


class SyntheticObjectives(torch.nn.Module):
    names = NAMES
    variant = "A3"
    max_roots = 1

    def __init__(self, case_ids):
        super().__init__()
        self.model = SyntheticModel(case_ids)
        self.views = SimpleNamespace(seed=42)
        self._library_epoch = None
        self._libraries = {}

    def case_losses(self, case, *, epoch, roots, seed, components):
        target = (int(case.case_id[1:]) + 1) / 10 + torch.rand(())
        return {name: ((self.model.weight - target).square() if name in components
                       and name in {"pn", "invariance"} else None) for name in self.names}


def make_trainer(path, *, budget=5, count=7, batch_size=5):
    torch.manual_seed(717)
    case = ObservableCase("c0", (Entity("root", "service"),), 0., 1., 2.)
    pool = {f"c{i}": replace(case, case_id=f"c{i}") for i in range(count)}
    labels = LabelStore({cid: ("root",) for cid in pool}, budget=budget, max_roots=1)
    schedule = TrainingSchedule(budget=budget, query_batch_size=batch_size,
                                warmup_epochs=0, nonfinal_epochs=1, final_epochs=1,
                                u_disabled_epochs=0, u_ramp_epochs=0)
    return Trainer(SyntheticObjectives(pool), pool, labels, schedule, checkpoint_dir=path,
                   identity=dict(data_hash="synthetic-data", config_hash="synthetic-config",
                                 preprocessing_hash="synthetic-pre", code_hash="synthetic-code"))


def read_trace(directory):
    path = directory / "query-trace.json"
    assert path.is_file(), "full ordered query plan must be durable before label access"
    return json.loads(path.read_text(encoding="utf-8"))


def test_plan_is_frozen_before_first_answer_and_contains_round_audit(tmp_path):
    trainer = make_trainer(tmp_path)
    access = trainer._access
    seen = []

    def check_before_access(cid):
        trace = read_trace(tmp_path)
        record = trace["rounds"][0]
        if cid not in trainer.labels.queried:
            assert record["selected_ids"][len(seen)] == cid
            seen.append(cid)
        assert len(record["selected_ids"]) == 5
        assert len(trainer.objectives.model.encodings) == 7
        assert record["round_index"] == 0
        assert record["stage_index"] == 0
        assert record["encoder_version"] == record["update_version"] == 0
        assert record["labeled_ids"] == []
        assert record["spent_before"] == 0 and record["remaining_budget"] == 5
        assert record["selector"]["name"] == "PropCover"
        assert record["selector"]["seed"] == 42
        return access(cid)

    trainer._access = check_before_access
    result = trainer.run()
    trace = read_trace(tmp_path)
    assert trace["schema"] == "vast-v3-ordered-query-trace/1"
    assert trace["identity"] == trainer.identity
    assert trace["rounds"][0]["selected_ids"] == seen
    assert trace["rounds"][0]["completed_ids"] == seen
    assert trace["rounds"][0]["complete"] is True
    assert result["query_order"] == seen
    assert seen != sorted(seen)


@pytest.mark.parametrize("point", ["before_first", "before_middle", "reserved_middle",
                                   "before_answer_commit", "after_answer_commit", "after_last"])
def test_mid_batch_resume_reuses_plan_and_matches_full_weights(tmp_path, monkeypatch, point):
    full = make_trainer(tmp_path / "full")
    expected = full.run()
    partial = make_trainer(tmp_path / "resumed")
    original_access, original_persist = partial._access, partial._persist_query

    def interrupt_access(cid):
        if cid not in partial.labels.queried:
            if point == "before_first" and partial.labels.spent == 0:
                raise InterruptedError(point)
            if point == "before_middle" and partial.labels.spent == 2:
                raise InterruptedError(point)
        return original_access(cid)

    def interrupt_persist(state):
        paid, completed = len(state["accessed"]), len(state["revealed"])
        if point == "before_answer_commit" and completed == 3:
            raise InterruptedError(point)
        original_persist(state)
        if ((point == "reserved_middle" and paid == 3 and completed == 2)
                or (point == "after_answer_commit" and completed == 3)
                or (point == "after_last" and completed == 5)):
            raise InterruptedError(point)

    partial._access, partial._persist_query = interrupt_access, interrupt_persist
    with pytest.raises(InterruptedError, match=point):
        partial.run()
    frozen = read_trace(tmp_path / "resumed")["rounds"][0]["selected_ids"]
    resumed = make_trainer(tmp_path / "resumed")

    def forbidden_reselection(*args, **kwargs):
        pytest.fail("resume must consume frozen selection, not rerun acquisition")

    monkeypatch.setattr("vast_v3.training.loop.PropCover.select", forbidden_reselection)
    actual = resumed.run(resume=True)
    assert actual == expected
    assert resumed.labels.spent == 5
    assert resumed.objectives.model.encodings == []
    assert actual["query_order"] == frozen == expected["query_order"]
    assert read_trace(tmp_path / "resumed") == read_trace(tmp_path / "full")
    for name, value in full.objectives.state_dict().items():
        torch.testing.assert_close(resumed.objectives.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("encoder_version", 100), ("update_version", 100),
                                        ("selected_ids", ["c0"]), ("remaining_budget", 50),
                                        ("labeled_ids", ["c6"]), ("round_index", 1)])
def test_resume_rejects_tampered_query_plan_before_root_access(tmp_path, field, value):
    trainer = make_trainer(tmp_path)
    trainer.run(max_updates=1)
    path = tmp_path / "query-trace.json"
    trace = read_trace(tmp_path)
    trace["rounds"][0][field] = value
    path.write_text(json.dumps(trace), encoding="utf-8")
    resumed = make_trainer(tmp_path)
    resumed._access = lambda cid: pytest.fail("tampered query trace reached root access")
    with pytest.raises(ValueError, match="query.trace|query plan"):
        resumed.run(resume=True)


def test_finished_legacy_ledger_cannot_invent_missing_selection_order(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.run()
    path = tmp_path / "query-trace.json"
    if path.exists():
        path.unlink()
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=True)
    checkpoint["extra_state"].pop("query_trace", None)
    checkpoint["extra_state"].pop("query_plans", None)
    torch.save(checkpoint, tmp_path / "latest.pt")
    resumed = make_trainer(tmp_path)
    with pytest.raises(ValueError, match="query.trace|ordered.*trace"):
        resumed.run(resume=True)


@pytest.mark.parametrize("budget,count,expected", [(0, 7, []), (2, 7, [2]),
                                                  (9, 3, [3]), (7, 9, [5, 2]),
                                                  (30, 31, [5, 5, 5, 5, 5, 5])])
def test_synthetic_budget_boundaries_preserve_round_order(tmp_path, budget, count, expected):
    trainer = make_trainer(tmp_path, budget=budget, count=count)
    result = trainer.run()
    trace = read_trace(tmp_path)
    assert [len(record["selected_ids"]) for record in trace["rounds"]] == expected
    order = []
    for record in trace["rounds"]:
        assert record["labeled_ids"] == sorted(order)
        assert record["spent_before"] == len(order)
        assert record["remaining_budget"] == budget - len(order)
        assert record["completed_ids"] == record["selected_ids"]
        order.extend(record["selected_ids"])
    assert result["query_order"] == order
    assert len(order) == len(set(order)) == trainer.labels.spent == min(budget, count)


def test_empty_schedule_persists_completion_before_return_and_resume(tmp_path):
    from vast_v3.training.checkpoint import inspect_checkpoint

    trainer = make_trainer(tmp_path, budget=0)
    result = trainer.run()
    assert result["complete"] and result["updates"] == 0
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["complete"] is True
    payload = inspect_checkpoint(tmp_path / "latest.pt", expected_identity=trainer.identity,
                                 require_complete=True)
    assert payload["progress"]["stage"] == "finished"
    assert payload["progress"]["stage_complete"] is True
    assert trainer.labels.spent == 0 and read_trace(tmp_path)["rounds"] == []
    resumed = make_trainer(tmp_path, budget=0)
    assert resumed.run(resume=True) == result
    assert json.loads((tmp_path / "progress.json").read_text(encoding="utf-8")) == progress


@pytest.mark.parametrize("after_reservation", [False, True])
def test_trace_cursor_can_lag_one_durable_label_write(tmp_path, monkeypatch, after_reservation):
    full = make_trainer(tmp_path / "full")
    expected = full.run()
    partial = make_trainer(tmp_path / "resumed")
    from vast_v3.training.query_trace import OrderedQueryTrace
    save = OrderedQueryTrace.save

    def interrupted_save(trace):
        access_path = trace.path.parent / "query-access.json"
        if access_path.exists():
            ledger = json.loads(access_path.read_text(encoding="utf-8"))["labels"]
            if len(ledger["accessed"]) == 3 and len(ledger["revealed"]) == (2 if after_reservation else 3):
                raise InterruptedError("between ledger and trace writes")
        return save(trace)

    with monkeypatch.context() as scoped:
        scoped.setattr(OrderedQueryTrace, "save", interrupted_save)
        with pytest.raises(InterruptedError, match="between ledger"):
            partial.run()
    resumed = make_trainer(tmp_path / "resumed")
    assert resumed.run(resume=True) == expected
    torch.testing.assert_close(resumed.objectives.model.weight, full.objectives.model.weight, rtol=0, atol=0)


def test_plan_checkpoint_recovers_interrupted_trace_sidecar_write(tmp_path, monkeypatch):
    full = make_trainer(tmp_path / "full")
    expected = full.run()
    partial = make_trainer(tmp_path / "resumed")
    from vast_v3.training.query_trace import OrderedQueryTrace
    save = OrderedQueryTrace.save

    def interrupted_save(trace):
        if trace.rounds:
            raise InterruptedError("persisting frozen plan sidecar")
        return save(trace)

    with monkeypatch.context() as scoped:
        scoped.setattr(OrderedQueryTrace, "save", interrupted_save)
        with pytest.raises(InterruptedError, match="frozen plan"):
            partial.run()
    assert partial.labels.spent == 0
    resumed = make_trainer(tmp_path / "resumed")
    monkeypatch.setattr("vast_v3.training.loop.PropCover.select",
                        lambda *args, **kwargs: pytest.fail("durable plan was reselected"))
    assert resumed.run(resume=True) == expected
    torch.testing.assert_close(resumed.objectives.model.weight, full.objectives.model.weight, rtol=0, atol=0)


def test_missing_sidecar_recovers_only_from_checkpointed_exact_trace(tmp_path):
    trainer = make_trainer(tmp_path)
    expected = trainer.run()
    original = read_trace(tmp_path)
    (tmp_path / "query-trace.json").unlink()
    resumed = make_trainer(tmp_path)
    assert resumed.run(resume=True) == expected
    assert read_trace(tmp_path) == original


def test_mid_second_round_resume_preserves_both_ordered_batches(tmp_path, monkeypatch):
    full = make_trainer(tmp_path / "full", budget=7, count=9)
    expected = full.run()
    partial = make_trainer(tmp_path / "resumed", budget=7, count=9)
    persist = partial._persist_query

    def interrupt(state):
        persist(state)
        if len(state["revealed"]) == 6:
            raise InterruptedError("second round")

    partial._persist_query = interrupt
    with pytest.raises(InterruptedError, match="second round"):
        partial.run()
    resumed = make_trainer(tmp_path / "resumed", budget=7, count=9)
    monkeypatch.setattr("vast_v3.training.loop.PropCover.select",
                        lambda *args, **kwargs: pytest.fail("second batch was reselected"))
    assert resumed.run(resume=True) == expected
    assert resumed.labels.spent == 7
    assert read_trace(tmp_path / "resumed") == read_trace(tmp_path / "full")
    torch.testing.assert_close(resumed.objectives.model.weight, full.objectives.model.weight, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["identity", "completion", "selector", "binding", "rehash_order"])
def test_trace_and_ledger_bindings_reject_inconsistent_mutations(tmp_path, mutation):
    trainer = make_trainer(tmp_path)
    trainer.run(max_updates=1)
    trace = read_trace(tmp_path)
    path = tmp_path / "query-trace.json"
    record = trace["rounds"][0]
    if mutation == "identity":
        trace["identity"]["code_hash"] = "unrelated-code"
    elif mutation == "completion":
        record["completed_ids"].reverse()
    elif mutation == "selector":
        record["selector"]["seed"] = 99
    elif mutation == "binding":
        path = tmp_path / "query-access.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        trace.pop("query_plans")
    else:
        from vast_v3.training.query_trace import _plan_hash
        record["selected_ids"].reverse()
        record["accessed_ids"] = list(record["selected_ids"])
        record["completed_ids"] = list(record["selected_ids"])
        record["plan_hash"] = _plan_hash(record)
    path.write_text(json.dumps(trace), encoding="utf-8")
    resumed = make_trainer(tmp_path)
    resumed._access = lambda cid: pytest.fail("inconsistent trace reached root access")
    with pytest.raises(ValueError, match="query.trace|query plan"):
        resumed.run(resume=True)


def test_mid_batch_missing_sidecar_uses_checkpoint_plan_not_ledger_sort(tmp_path):
    full = make_trainer(tmp_path / "full")
    expected = full.run()
    partial = make_trainer(tmp_path / "resumed")
    persist = partial._persist_query

    def interrupt(state):
        persist(state)
        if len(state["revealed"]) == 3:
            raise InterruptedError("after third answer")

    partial._persist_query = interrupt
    with pytest.raises(InterruptedError):
        partial.run()
    (tmp_path / "resumed" / "query-trace.json").unlink()
    resumed = make_trainer(tmp_path / "resumed")
    assert resumed.run(resume=True) == expected
    assert read_trace(tmp_path / "resumed") == read_trace(tmp_path / "full")
    torch.testing.assert_close(resumed.objectives.model.weight, full.objectives.model.weight, rtol=0, atol=0)


def test_resume_rejects_nonprefix_paid_access_before_resolving_pending_label(tmp_path):
    trainer = make_trainer(tmp_path)
    persist = trainer._persist_query

    def interrupt(state):
        persist(state)
        if len(state["accessed"]) == 1:
            raise InterruptedError("first reservation")

    trainer._persist_query = interrupt
    with pytest.raises(InterruptedError):
        trainer.run()
    trace = read_trace(tmp_path)
    path = tmp_path / "query-access.json"
    access = json.loads(path.read_text(encoding="utf-8"))
    access["labels"]["accessed"] = [trace["rounds"][0]["selected_ids"][1]]
    path.write_text(json.dumps(access), encoding="utf-8")
    resumed = make_trainer(tmp_path)
    resumed._access = lambda cid: pytest.fail("nonprefix ledger reached a pending answer")
    with pytest.raises(ValueError, match="ordered access prefix"):
        resumed.run(resume=True)


def test_real_objectives_closed_loop_query_interrupt_matches_parameters(tmp_path):
    from test_ranker import setup
    from vast_v3.training.objectives import TrainingObjectives

    def real_trainer(path):
        torch.manual_seed(717)
        case, model = setup()
        pool = {f"c{i}": replace(case, case_id=f"c{i}") for i in range(3)}
        labels = LabelStore({cid: ("a",) for cid in pool}, budget=2, max_roots=2)
        schedule = TrainingSchedule(budget=2, query_batch_size=2, warmup_epochs=0,
                                    nonfinal_epochs=1, final_epochs=1, u_disabled_epochs=0,
                                    u_ramp_epochs=0)
        return Trainer(TrainingObjectives(model, max_roots=2), pool, labels, schedule,
                       checkpoint_dir=path, identity=dict(data_hash="synthetic", config_hash="unit",
                                                        preprocessing_hash="pre", code_hash="code"))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        full = real_trainer(tmp_path / "full")
        expected = full.run()
        partial = real_trainer(tmp_path / "resumed")
        persist = partial._persist_query

        def interrupt(state):
            persist(state)
            if len(state["revealed"]) == 1:
                raise InterruptedError("real objectives synthetic fixture")

        partial._persist_query = interrupt
        with pytest.raises(InterruptedError):
            partial.run()
        resumed = real_trainer(tmp_path / "resumed")
        assert resumed.run(resume=True) == expected
        assert resumed.labels.spent == 2
        assert read_trace(tmp_path / "resumed") == read_trace(tmp_path / "full")
        for name, value in full.objectives.state_dict().items():
            torch.testing.assert_close(resumed.objectives.state_dict()[name], value, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
