"""Case-level facility location over provenance-distinct fragment occurrences."""

import importlib

import pytest
import torch


@pytest.fixture
def api():
    class Module:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            try:
                return getattr(importlib.import_module(self.name), name)
            except ModuleNotFoundError as error:
                pytest.fail(f"acquisition implementation is missing: {error}")
    return Module("vast_v3.acquisition.propcover"), Module("vast_v3.acquisition.baselines")


def case(api, name, vectors, families=None, provenance=None, version="v1"):
    vectors = torch.tensor(vectors, dtype=torch.float32).reshape(-1, 2)
    count = vectors.shape[0]
    return api[0].CaseTokens(name, vectors, tuple(families or ["chain"] * count),
                             tuple(provenance or [f"p-{i}" for i in range(count)]), version)


def test_family_case_occurrence_mass_is_exact(api):
    a = case(api, "a", [[1, 0], [1, 0], [0, 1]], ["chain", "chain", "branch"])
    b = case(api, "b", [[1, 0]])
    selector = api[0].PropCover([a, b], seed=7)
    weights = dict(zip(selector.occurrence_keys, selector.weights.tolist()))
    assert weights == {("a", "p-0"): .125, ("a", "p-1"): .125,
                       ("a", "p-2"): .5, ("b", "p-0"): .25}
    assert sum(weights.values()) == 1.


def test_exact_provenance_dedup_does_not_merge_equal_vectors(api):
    a = case(api, "a", [[1, 0], [1, 0], [1, 0], [1, 0]],
             ["chain", "chain", "chain", "branch"], ["same", "same", "other", "branch"])
    b = case(api, "b", [[1, 0]], provenance=["same"])
    selector = api[0].PropCover([a, b])
    assert len(selector.occurrence_keys) == 4
    assert len({key[0] for key in selector.occurrence_keys}) == 2


def test_gain_selects_whole_case_not_an_individual_fragment(api):
    cases = [case(api, "a", [[1, 0]]), case(api, "b", [[0, 1]]),
             case(api, "bundle", [[1, 0], [0, 1]])]
    selector = api[0].PropCover(cases, priority=("a", "b", "bundle"))
    selected = selector.select((), batch_size=1, remaining_budget=1)
    assert selected.case_ids == ("bundle",)
    assert selected.gains == pytest.approx((1.,))
    assert selector.coverage(()) == 0.
    assert selector.coverage(("a",)) < selector.coverage(("bundle",))


def test_conditional_coverage_uses_labeled_baseline_and_fills_zero_gains(api):
    cases = [case(api, "a", [[1, 0]]), case(api, "b", [[0, 1]]),
             case(api, "bundle", [[1, 0], [0, 1]])]
    selector = api[0].PropCover(cases, priority=("b", "bundle", "a"))
    selected = selector.select(("bundle",), batch_size=5, remaining_budget=2)
    assert selected.case_ids == ("b", "a")
    assert selected.gains == (0., 0.)
    assert selected.base_coverage == pytest.approx(1.)
    assert selected.final_coverage == pytest.approx(1.)
    assert selector.objective(("a",), labeled_ids=("bundle",)) == 0.


def test_empty_reference_inventory_has_no_weights_but_spends_remaining_budget(api):
    selector = api[0].PropCover([case(api, "a", []), case(api, "b", []),
                                case(api, "c", [])], priority=("c", "a", "b"))
    assert selector.weights.numel() == 0
    assert selector.occurrence_keys == ()
    selected = selector.select(("a",), batch_size=5, remaining_budget=1)
    assert selected.case_ids == ("c",)
    assert selected.gains == (0.,)
    assert selected.reference_occurrences == 0
    assert selected.final_coverage == 0.


def test_empty_case_has_zero_cached_coverage_not_a_synthetic_token(api):
    selector = api[0].PropCover([case(api, "empty", []), case(api, "real", [[1, 0]])],
                               priority=("empty", "real"))
    assert selector.coverage(("empty",)) == 0.
    selected = selector.select((), batch_size=2, remaining_budget=2)
    assert selected.case_ids == ("real", "empty")
    assert selected.gains == (1., 0.)


def test_acquisition_never_recomputes_distances_after_cache_construction(api):
    selector = api[0].PropCover([case(api, "a", [[1, 0]]), case(api, "b", [[0, 1]])],
                               similarity_chunk_size=1)
    before = dict(selector.cache_stats)
    first = selector.select((), batch_size=1, remaining_budget=2)
    selector.select(first.case_ids, batch_size=1, remaining_budget=1)
    selector.objective(("a",))
    assert selector.cache_stats == before
    assert before["kernel_pairs"] == 4


def test_mixed_encoder_versions_are_rejected(api):
    with pytest.raises(ValueError, match="encoder version"):
        api[0].PropCover([case(api, "a", [[1, 0]], version="v1"),
                          case(api, "b", [[0, 1]], version="v2")])


def test_refresh_requires_complete_l_and_u_inventory_and_preserves_priority(api):
    selector = api[0].PropCover([case(api, "a", [[1, 0]]), case(api, "b", [[-1, 0]])], seed=9)
    assert selector.coverage(("a",)) == .5
    priority = selector.priority
    with pytest.raises(ValueError, match="complete permitted pool"):
        selector.refresh([case(api, "a", [[1, 0]], version="v2")])
    selector.refresh([case(api, "b", [[1, 0]], version="v2"),
                      case(api, "a", [[1, 0]], version="v2")])
    assert selector.encoder_version == "v2"
    assert selector.coverage(("a",)) == 1.
    assert selector.priority == priority


def test_total_budget_counts_already_queried_initial_cases(api):
    selector = api[0].PropCover([case(api, str(i), []) for i in range(5)], seed=5)
    chosen = selector.select_budgeted(("0", "1"), total_budget=3, batch_size=5)
    assert len(chosen.case_ids) == 1
    assert not set(chosen.case_ids) & {"0", "1"}
    with pytest.raises(ValueError, match="exceeds"):
        selector.select_budgeted(("0", "1"), total_budget=1)


def test_token_snapshot_is_frozen_and_nonunit_tokens_are_rejected(api):
    first, second = case(api, "a", [[1, 0]]), case(api, "b", [[-1, 0]])
    selector = api[0].PropCover([first, second])
    first.tokens.fill_(100.)
    assert selector.coverage(("a",)) == .5
    with pytest.raises(ValueError, match="unit"):
        api[0].PropCover([case(api, "zero", [[0, 0]])])


def test_conflicting_duplicate_provenance_is_an_error(api):
    with pytest.raises(ValueError, match="provenance"):
        api[0].PropCover([case(api, "a", [[1, 0], [0, 1]], provenance=["same", "same"])])


def test_partial_or_overlapping_l_u_partitions_are_rejected(api):
    selector = api[0].PropCover([case(api, "a", []), case(api, "b", [])])
    with pytest.raises(ValueError, match="partition"):
        selector.select(("a",), ("a", "b"), batch_size=1, remaining_budget=1)
    with pytest.raises(ValueError, match="partition"):
        selector.select((), ("a",), batch_size=1, remaining_budget=1)


def test_random_baseline_uses_persisted_label_blind_priority(api):
    baseline = api[1].RandomCaseSelector(("c", "a", "b"), seed=42)
    reordered = api[1].RandomCaseSelector(("b", "c", "a"), seed=42)
    assert baseline.priority == reordered.priority
    assert baseline.select((), batch_size=2, remaining_budget=2) == baseline.priority[:2]
    assert baseline.select((baseline.priority[0],), batch_size=5,
                           remaining_budget=1) == baseline.priority[1:2]
