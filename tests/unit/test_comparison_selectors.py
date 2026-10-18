"""Approved label-free controls: hand calculations and descriptor boundaries."""
import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vast_v3.acquisition.propcover import CaseTokens, seeded_case_priority
from vast_v3.data.preprocessing import Preprocessor, Scale
from vast_v3.data.schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation
from vast_v3.encoders.fragments import EventRef, Fragment, FragmentEdge


@pytest.fixture
def api():
    class Module:
        def __getattr__(self, name):
            try:
                return getattr(importlib.import_module("vast_v3.acquisition.comparison_selectors"), name)
            except ModuleNotFoundError as error:
                pytest.fail(f"approved comparison selectors are not implemented: {error}")
    return Module()


def scope_case(name, vectors, provenance=None):
    tensor = torch.tensor(vectors, dtype=torch.float64).reshape(-1, 2)
    return CaseTokens(name, tensor, ("chain",) * len(tensor),
                      tuple(provenance or [f"p{i}" for i in range(len(tensor))]), "frozen-v1")


def symptoms(api, name, vectors, kinds=None, provenance=None, observed=None):
    tensor = torch.tensor(vectors, dtype=torch.float64).reshape(-1, 2)
    return api.SymptomCaseTokens(name, tensor, tuple(kinds or ["node"] * len(tensor)),
                                 tuple(provenance or [f"p{i}" for i in range(len(tensor))]),
                                 "frozen-v1", tuple(observed or [True] * len(tensor)))


def test_time_spread_starts_with_medoid_then_farthest_first(api):
    s = api.TimeSpread({"left": 0., "middle": 2., "right": 10., "absent": None},
                       priority=("absent", "left", "right", "middle"))
    assert s.select(batch_size=3).case_ids == ("middle", "right", "left")
    assert s.select(("middle",), batch_size=2).case_ids == ("right", "left")
    assert s.times == {"left": 0., "middle": .2, "right": 1., "absent": None}


def test_time_constant_axis_and_missing_l_use_common_priority(api):
    s = api.TimeSpread({"a": 5., "b": 5., "c": None}, priority=("c", "b", "a"))
    assert s.select(batch_size=3).case_ids == ("b", "c", "a")
    assert s.select(("c",), batch_size=2).case_ids == ("b", "a")
    assert s.select(("a",), batch_size=2).gains == (0., 0.)


def test_time_duplicate_instants_add_no_new_distance(api):
    s = api.TimeSpread({"a": 0., "b": 0., "c": 1.}, priority=("b", "a", "c"))
    result = s.select(("a",), batch_size=2)
    assert result.case_ids == ("c", "b")
    assert result.gains == (1., 0.)


def test_symptom_mass_filters_before_category_case_occurrence_balancing(api):
    a = symptoms(api, "a", [[2, 0], [2, 0], [0, 0], [0, 3], [99, 99]],
                 ["node", "node", "edge", "edge", "edge"],
                 ["same", "same", "zero", "edge", "hidden"], [True, True, True, True, False])
    b = symptoms(api, "b", [[2, 0], [2, 0]])
    s = api.SymptomCover([a, b])
    weights = dict(zip(s.occurrence_keys, s.weights.tolist()))
    assert weights == {("a", "edge", "edge"): .5, ("a", "node", "same"): .25,
                       ("b", "node", "p0"): .125, ("b", "node", "p1"): .125}
    assert s.reference_count == 4
    assert s.coverage(("a",)) == pytest.approx(1.)


def test_symptom_categories_do_not_cross_match_and_bundle_wins(api):
    a = symptoms(api, "a", [[1, 0]], ["node"])
    b = symptoms(api, "b", [[1, 0]], ["edge"])
    both = symptoms(api, "both", [[1, 0], [1, 0]], ["node", "edge"])
    s = api.SymptomCover([a, b, both], priority=("a", "b", "both"))
    assert s.coverage(("a",)) == .5
    assert s.coverage(()) == 0.
    assert s.select(batch_size=1).case_ids == ("both",)
    result = s.select(("both",), batch_size=2)
    assert result.case_ids == ("a", "b")
    assert result.gains == (0., 0.)


def test_symptom_snapshot_does_not_mutate_when_input_tensor_changes(api):
    a = symptoms(api, "a", [[1, 0]])
    b = symptoms(api, "b", [[-1, 0]])
    s = api.SymptomCover([a, b])
    a.tokens.fill_(99)
    assert s.coverage(("a",)) == .5
    with pytest.raises(ValueError, match="provenance"):
        api.SymptomCover([symptoms(api, "bad", [[1, 0], [0, 1]], provenance=["x", "x"])])


def test_symptom_extraction_only_reads_local_observed_and_owner_local_edge_states(api):
    # Deliberately no h, SCOPE, topology, recovery, root, or fault-type fields.
    front = SimpleNamespace(local=torch.tensor([[[3., 0.], [9., 9.]]]),
                            local_mask=torch.tensor([[True, False]]),
                            edge_states=torch.tensor([[[0., 4.], [8., 8.]]]),
                            edge_mask=torch.tensor([[True, False]]), edge_owner_ids=("call-owner",))
    got = api.symptom_case_from_frontend("x", front, entity_ids=("entity",), encoder_version="v1")
    assert got.kinds == ("node", "edge")
    assert got.tokens.tolist() == [[3., 0.], [0., 4.]]
    assert len(set(got.provenance)) == 2


def test_pooled_averages_real_provenance_not_duplicate_enumerations(api):
    s = api.PooledMedoid([scope_case("a", [[1, 0], [1, 0], [0, 1]], ["x", "x", "y"]),
                          scope_case("zero", [[1, 0], [-1, 0]]), scope_case("empty", [])])
    assert s.pooled["a"].tolist() == pytest.approx([2 ** -.5, 2 ** -.5])
    assert s.pooled["zero"] is None and s.pooled["empty"] is None
    assert s.objective(()) == 1.
    assert s.objective(("zero",)) == 1.
    assert s.objective(("a",)) == pytest.approx(0., abs=1e-15)


def test_pooled_empty_l_uses_distance_one_and_not_priority_sampling(api):
    s = api.PooledMedoid([scope_case("bad", [[-1, 0]]), scope_case("a", [[1, 0]]),
                          scope_case("b", [[1, 0]]), scope_case("empty", [])],
                         priority=("empty", "bad", "a", "b"))
    result = s.select(batch_size=1)
    assert result.case_ids == ("a",)
    assert result.base_objective == 1.
    assert result.final_objective == pytest.approx(1 / 3)
    assert result.gains == pytest.approx((2 / 3,))


def test_pooled_keeps_l_and_uses_float64_objective(api):
    s = api.PooledMedoid([scope_case("l", [[-1, 0]]), scope_case("a", [[1, 0]]),
                          scope_case("b", [[0, 1]]), scope_case("empty", [])],
                         priority=("empty", "l", "b", "a"))
    result = s.select(("l",), batch_size=1)
    assert "l" not in result.case_ids
    assert all(out != "l" for out, _, _ in result.swaps)
    assert result.final_objective == pytest.approx(s.objective(("l", *result.case_ids)))
    assert all(vector is None or vector.dtype == torch.float64 for vector in s.pooled.values())


def test_pooled_best_single_swap_improves_greedy_and_cannot_remove_l(api):
    r = 3 ** .5 / 2
    vectors = {"p0": [1, 0], "p30": [r, .5], "p60": [.5, r],
               "p120": [-.5, r], "p150": [-r, .5], "p180": [-1, 0]}
    s = api.PooledMedoid([scope_case(k, [v]) for k, v in vectors.items()],
                         priority=("p60", "p120", "p150", "p30", "p0", "p180"))
    result = s.select(batch_size=2)
    assert set(result.case_ids) == {"p30", "p150"}
    assert result.final_objective == pytest.approx((2 - 3 ** .5) / 6)
    assert result.swaps
    assert len(result.swaps) <= 100 and all(gain > 0 for _, _, gain in result.swaps)
    assert sum(result.gains) + sum(gain for _, _, gain in result.swaps) == pytest.approx(
        result.base_objective - result.final_objective)
    locked = s.select(("p60",), batch_size=1)
    assert locked.case_ids == ("p150",)
    assert locked.final_objective == pytest.approx((7 - 3 * 3 ** .5) / 24)
    assert not locked.swaps


def test_pooled_cluster_assignments_ignore_invalid_centers_and_use_priority_ties(api):
    s = api.PooledMedoid([scope_case("a", [[1, 0]]), scope_case("b", [[-1, 0]]),
                          scope_case("mid", [[0, 1]]), scope_case("empty", [])],
                         priority=("b", "a", "empty", "mid"))
    assert s.assignments(("empty",)) == {"a": None, "b": None, "mid": None}
    assert s.assignments(("a", "b", "empty")) == {"a": "a", "b": "b", "mid": "b"}


def motif_fixture():
    case = ObservableCase("x", (Entity("a", "service"), Entity("b", "service"), Entity("c", "host")),
                          0., 2., 5.,
                          (NumericSeries("metric", "a", "cpu", (2.1, 2.9), (0., 4.)),
                           NumericSeries("metric", "b", "cpu", (3.1,), (0.,)),
                           NumericSeries("trace", "ab", "duration", (2.1, 3.1), (4., 0.)),
                           NumericSeries("trace", "cb", "duration", (2.1, 3.1), (-4., -4.))),
                          (LogRecord("a", 2.2, "visible symptom", 3),),
                          (Relation("ab", "a", "b", "call", 0., 5.),
                           Relation("cb", "c", "b", "call", 0., 5.)))
    pre = Preprocessor({("service", "cpu"): Scale(0., 1., 3),
                        ("service->service", "duration"): Scale(0., 1., 3),
                        ("host->service", "duration"): Scale(0., 1., 3),
                        ("service", "__log_frequency__"): Scale(0., 1., 3)},
                       1., ("x",), 1)
    events = (EventRef(0, 2, 2., 3., 0, "a2"), EventRef(1, 3, 3., 4., 0, "b3"),
              EventRef(2, 2, 2., 3., 1, "c2"))
    fragment = Fragment(events, (FragmentEdge(0, 1, 0, 0), FragmentEdge(2, 1, 0, 1)),
                        "convergence", "f")
    return case, pre, fragment


def test_motif_reference_deviation_is_not_first_difference_or_healthy_missing(api):
    case, pre, fragment = motif_fixture()
    desc = api.motif_case_from_fragments(case, (fragment,), pre,
                                        metric_channels=("missing", "cpu"), trace_channels=("duration",))
    code = next(iter(desc.codes))
    assert '"high"' in code and '"low"' in code and '"reference-band"' in code
    assert '"unknown"' in code
    assert '"missing","unknown"' in code
    assert "visible symptom" not in code and "service->service" not in code
    assert desc.preprocessing_identity == pre.identity


def test_motif_trace_stays_with_owner_and_its_two_endpoint_bins(api):
    case, pre, fragment = motif_fixture()
    kwargs = dict(metric_channels=("cpu",), trace_channels=("duration",))
    original = api.motif_code(case, fragment, pre, **kwargs)
    swapped = replace(fragment, edges=(replace(fragment.edges[0], owner=1), replace(fragment.edges[1], owner=0)))
    assert api.motif_code(case, swapped, pre, **kwargs) != original
    masked, _ = case.mask_blocks({("edge", "ab", 3)}, delta=1.)
    assert api.motif_code(masked, fragment, pre, **kwargs) != original


def test_motif_anonymous_canonicalization_transports_attributes_and_shift(api):
    case, pre, fragment = motif_fixture()
    kwargs = dict(metric_channels=("cpu",), trace_channels=("duration",))
    original = api.motif_code(case, fragment, pre, **kwargs)
    order = (2, 0, 1)
    positions = {old: new for new, old in enumerate(order)}
    permuted = replace(fragment, events=tuple(fragment.events[i] for i in order),
                       edges=tuple(replace(e, source=positions[e.source], target=positions[e.target])
                                   for e in reversed(fragment.edges)))
    assert api.motif_code(case, permuted, pre, **kwargs) == original
    shifted = replace(fragment, events=tuple(replace(e, start=e.start + 100, end=e.end + 100)
                                             for e in fragment.events))
    assert api.motif_code(case.shift_time(100), shifted, pre, **kwargs) == original


@pytest.mark.parametrize("bounds,expected", [((0., .9), "[0,1)"), ((1., 1.9), "[1,2)"),
    ((2., 3.9), "[2,4)"), ((4., 8.), "[4,8]"), ((.9, 1.), "unknown"),
    ((-1., .5), "unknown"), ((1., 2.), "unknown"), ((8., 8.1), "unknown")])
def test_motif_delay_entire_supported_interval_must_fit(api, bounds, expected):
    a = EventRef(0, 0, 0., 0., 0, "a")
    b = EventRef(1, 1, bounds[0], bounds[1], 0, "b")
    assert api.delay_bucket(a, b, delta=1.) == expected


def test_motif_coverage_is_distinct_unweighted_codes_and_respects_l(api):
    s = api.MotifCover([api.MotifCase("a", frozenset(("x", "y")), "p"),
                        api.MotifCase("b", frozenset(("y", "z")), "p"),
                        api.MotifCase("c", frozenset(("z",)), "p")], priority=("c", "a", "b"))
    assert s.select(batch_size=2).case_ids == ("a", "c")
    result = s.select(("a",), batch_size=2)
    assert result.case_ids == ("c", "b") and result.gains == (1., 0.)


@pytest.mark.parametrize("kind", ["time", "symptom", "pooled", "motif"])
def test_shared_priority_empty_representations_budget_and_no_label_dependency(api, kind):
    ids = ("a", "b", "c")
    if kind == "time":
        s = api.TimeSpread(dict.fromkeys(ids), seed=17)
    elif kind == "symptom":
        s = api.SymptomCover([symptoms(api, i, []) for i in ids], seed=17)
    elif kind == "pooled":
        s = api.PooledMedoid([scope_case(i, []) for i in ids], seed=17)
    else:
        s = api.MotifCover([api.MotifCase(i, frozenset(), "p") for i in ids], seed=17)
    assert s.priority == seeded_case_priority(ids, 17)
    assert s.select(batch_size=30).case_ids == s.priority
    assert s.select_budgeted(s.priority[:1], total_budget=2, batch_size=30).case_ids == s.priority[1:2]
    assert s.select_budgeted((), total_budget=0).case_ids == ()
    with pytest.raises(ValueError, match="exceeds"):
        s.select_budgeted(s.priority[:2], total_budget=1)
    with pytest.raises(ValueError, match="partition"):
        s.select(("a",), ("a", "b", "c"))
    # There is no label store, root, type, or validation-score argument at all.
    with pytest.raises(TypeError):
        s.select(root_labels={"a": "hidden"})


@pytest.mark.parametrize("cls,args", [("TimeSpread", ({},)), ("SymptomCover", ([],)),
                                      ("PooledMedoid", ([],)), ("MotifCover", ([],))])
def test_empty_pool_is_a_valid_zero_query(api, cls, args):
    assert getattr(api, cls)(*args).select().case_ids == ()
