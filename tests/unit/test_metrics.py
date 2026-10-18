"""Synthetic metric fixtures; these are not benchmark results."""

import importlib
import math

import pytest


def api():
    try:
        return importlib.import_module("vast_v3.evaluation.metrics")
    except ModuleNotFoundError:
        pytest.fail("The metric contract is not implemented yet")


def test_canonical_ties_full_roots_and_cumulative_hits():
    result = api().evaluate_case("fixture", ["c", "b", "a"],
                                 {"a": 2.0, "b": 2.0, "c": 1.0}, ["c", "b"])
    assert result.ranking == ("a", "b", "c")
    assert result.root_ranks == {"b": 2, "c": 3}
    assert result.first_root_rank == 2
    assert result.hits == (0, 1, 1, 1, 1)
    assert result.to_dict()["true_roots"] == ["b", "c"]


def test_aggregate_fixed_denominator_exact_mrr_and_multiroot_recovery():
    metric = api()
    catalog = list("abcdefgh")
    scores = {entity: 8 - index for index, entity in enumerate(catalog)}
    cases = [metric.evaluate_case("one", catalog, scores, ["a"]),
             metric.evaluate_case("two", catalog, scores, ["c", "e"])]
    summary = metric.aggregate_metrics(cases, expected_case_ids=["one", "two"])
    assert [summary[f"H{k}"] for k in range(1, 6)] == [1, 1, 2, 2, 2]
    assert summary["N"] == 2
    assert summary["TOP135"] == pytest.approx(5 / 6)
    assert summary["Avg@5"] == pytest.approx(0.8)
    assert summary["MRR"] == pytest.approx(2 / 3)
    assert summary["RootRecall@3"] == pytest.approx(0.75)
    assert summary["AllRoots@3"] == pytest.approx(0.5)
    assert summary["AllRoots@5"] == 1.0


def test_mrr_bounds_are_bounds_not_midpoint_estimates():
    bounds = api().mrr_bounds_from_hits(4, [1, 2, 2, 2, 3])
    assert bounds == {"MRR_lower": pytest.approx(1.7 / 4),
                      "MRR_upper": pytest.approx((1.7 + 1 / 6) / 4)}


@pytest.mark.parametrize("catalog,scores,roots", [
    (["a", "b"], {"a": 1.0}, ["a"]),
    (["a"], {"a": 1.0, "b": 0.0}, ["a"]),
    (["a", "a"], {"a": 1.0}, ["a"]),
    (["a"], {"a": 1.0}, []),
    (["a"], {"a": 1.0}, ["b"]),
    (["a"], {"a": 1.0}, ["a", "a"]),
    (["a"], {"a": math.nan}, ["a"]),
    (["a"], {"a": math.inf}, ["a"]),
    ([], {}, ["a"]),
])
def test_invalid_candidate_or_root_contract_is_an_error(catalog, scores, roots):
    with pytest.raises(ValueError):
        api().evaluate_case("fixture", catalog, scores, roots)


@pytest.mark.parametrize("observed,expected", [([], ["missing"]),
    (["one"], ["one", "two"]), (["one", "one"], ["one"]),
    (["one"], ["different"]), (["one"], ["one", "one"])])
def test_aggregate_cannot_silently_skip_or_duplicate_incidents(observed, expected):
    metric = api()
    results = [metric.evaluate_case(case, ["a"], {"a": 1}, ["a"])
               for case in observed]
    with pytest.raises(ValueError):
        metric.aggregate_metrics(results, expected_case_ids=expected)


@pytest.mark.parametrize("n,hits", [(0, [0]*5), (2, [0, 2, 1, 2, 2]),
    (2, [1, 1, 2, 2, 3]), (2, [1, 1, 2, 2]), (2, [0.5, 1, 1, 1, 1])])
def test_invalid_cumulative_counts_are_rejected(n, hits):
    with pytest.raises(ValueError):
        api().mrr_bounds_from_hits(n, hits)


def test_aggregate_revalidates_tampered_case_metrics():
    metric = api()
    case = metric.evaluate_case("one", ["a", "b"], {"a": 2, "b": 1}, ["b"])
    case.root_ranks["b"] = 1
    with pytest.raises(ValueError):
        metric.aggregate_metrics([case], expected_case_ids=["one"])
