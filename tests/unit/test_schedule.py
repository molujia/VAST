import importlib

import pytest


def api():
    try:
        return importlib.import_module("vast_v3.training.schedule")
    except ModuleNotFoundError:
        pytest.fail("Training schedule implementation is missing")


def test_reference_query_batches_and_last_round_duration():
    schedule = api().TrainingSchedule()
    assert schedule.query_batches(100) == (5, 5, 5, 5, 5, 5)
    assert schedule.query_batches(12) == (5, 5, 2)
    assert schedule.query_batches(0) == ()
    assert schedule.round_epochs(0, 6) == 20
    assert schedule.round_epochs(5, 6) == 50
    assert api().TrainingSchedule(budget=7).query_batches(100) == (5, 2)


def test_five_disabled_then_five_ramping_u_epochs():
    schedule = api().TrainingSchedule()
    assert [schedule.u_weight_at(epoch) for epoch in range(5)] == [0.0] * 5
    assert [schedule.u_weight_at(epoch) for epoch in range(5, 10)] == pytest.approx([.02, .04, .06, .08, .1])
    assert schedule.u_weight_at(10) == .1
    assert schedule.u_weight_at(49, enabled=False) == 0
    with pytest.raises(ValueError):
        schedule.u_weight_at(-1)


def test_fixed_controls_do_not_inherit_sequential_rounds():
    schedule = api().TrainingSchedule()
    closed = schedule.stages("closed_loop", pool_size=100)
    assert [s["epochs"] for s in closed] == [20, 20, 20, 20, 20, 20, 50]
    assert [s["query_count"] for s in closed] == [0, 5, 5, 5, 5, 5, 5]
    for regime in ["fixed_list", "all_label", "lofo"]:
        stages = schedule.stages(regime, pool_size=100)
        assert [s["epochs"] for s in stages] == [20, 50]
        assert stages[-1]["query_count"] == (30 if regime == "fixed_list" else 0)
        assert stages[-1]["u_enabled"] == (regime == "fixed_list")
    a0 = schedule.stages("fixed_list", pool_size=100, a0=True)
    assert len(a0) == 1 and a0[0]["phase"] == "joint"
    assert not a0[0]["u_enabled"]


def test_update_slots_use_pool_not_each_component_population():
    schedule = api().TrainingSchedule()
    assert [schedule.update_slots(n) for n in [0, 1, 4, 5, 100]] == [0, 1, 1, 2, 25]


def test_independent_streams_have_no_cross_component_sampling_effect():
    cls = api().IndependentEligibleStreams
    streams = cls({"pn": ["a", "b", "c", "d", "e"], "obs": list("uvwxyz")}, seed=13)
    reference = cls({"pn": ["a", "b", "c", "d", "e"]}, seed=13)
    assert streams.draw("pn") == reference.draw("pn")
    for _ in range(20):
        streams.draw("obs")
    assert streams.draw("pn") == reference.draw("pn")
    assert streams.draw("pn") == reference.draw("pn")


def test_small_and_empty_streams_draw_up_to_four_then_reshuffle():
    streams = api().IndependentEligibleStreams({"small": ["a", "b"], "empty": [], "five": list("abcde")}, seed=4)
    assert set(streams.draw("small")) == {"a", "b"}
    assert set(streams.draw("small")) == {"a", "b"}
    assert streams.draw("empty") == ()
    first, tail, again = [streams.draw("five") for _ in range(3)]
    assert [len(first), len(tail), len(again)] == [4, 1, 4]
    assert set(first + tail) == set("abcde")
    assert streams.draw_all().keys() == {"small", "empty", "five"}


def test_stream_state_exactly_restores_next_draws():
    cls = api().IndependentEligibleStreams
    streams = cls({"src": list("abcdef"), "u": list("uvwxyz")}, seed=8)
    streams.draw_all()
    state = streams.state_dict()
    expected = [streams.draw_all() for _ in range(7)]
    restored = cls({"src": list("abcdef"), "u": list("uvwxyz")}, seed=99)
    restored.load_state_dict(state)
    assert [restored.draw_all() for _ in range(7)] == expected
    wrong = cls({"src": list("abcdefg"), "u": list("uvwxyz")}, seed=8)
    with pytest.raises(ValueError, match="population"):
        wrong.load_state_dict(state)


def test_invalid_schedule_and_duplicate_stream_population_fail():
    for kwargs in [{"budget": -1}, {"query_batch_size": 0}, {"case_batch_size": 0}, {"u_ramp_epochs": -1}]:
        with pytest.raises(ValueError):
            api().TrainingSchedule(**kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        api().IndependentEligibleStreams({"pn": ["a", "a"]}, seed=3)
