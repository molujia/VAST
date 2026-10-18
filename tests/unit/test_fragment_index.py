"""Exact inventory counts and integer-rank bijections, independently of RNG."""

from collections import Counter, defaultdict
import importlib
import itertools
import random

import pytest

from vast_v3.encoders.fragments import EventRef, FragmentEdge, FAMILIES, structural_family


@pytest.fixture
def index_type():
    try:
        return importlib.import_module("vast_v3.encoders.fragment_index").FragmentIndex
    except ModuleNotFoundError:
        def missing(*args, **kwargs):
            pytest.fail("exact fragment index implementation is missing")
        return missing


def inventory(entities, endpoints):
    events = tuple(EventRef(e, 0, 0., 1., 0, f"event-{i:03}")
                   for i, e in enumerate(entities))
    # Each edge has distinct typed provenance, including equal endpoints.
    edges = tuple(FragmentEdge(s, t, i % 3, i, bool(i % 2))
                  for i, (s, t) in enumerate(endpoints))
    return events, edges


def brute_force(events, edges, maximum):
    family_counts = Counter()
    strata = defaultdict(set)
    for size in range(1, maximum + 1):
        for indices in itertools.combinations(range(len(edges)), size):
            remaining = set(indices)
            reached = {remaining.pop()}
            while remaining:
                extra = {j for j in remaining if any(
                    {edges[i].source, edges[i].target}
                    & {edges[j].source, edges[j].target} for i in reached)}
                if not extra:
                    break
                reached.update(extra)
                remaining.difference_update(extra)
            if remaining:
                continue
            vertices = sorted({v for i in indices
                               for v in (edges[i].source, edges[i].target)})
            local = {v: k for k, v in enumerate(vertices)}
            selected = tuple(FragmentEdge(local[edges[i].source], local[edges[i].target],
                                          edges[i].relation, edges[i].owner, edges[i].response)
                             for i in indices)
            family = structural_family(tuple(events[v] for v in vertices), selected)
            family_counts[family] += 1
            for anchor in {events[v].entity for v in vertices}:
                strata[anchor, family].add(indices)
    return {f: family_counts[f] for f in FAMILIES}, strata


def assert_bijection(index_type, events, edges, maximum):
    index = index_type(events, edges, max_edges=maximum)
    expected_counts, strata = brute_force(events, edges, maximum)
    assert index.family_counts == expected_counts
    assert index.total_count == sum(expected_counts.values())
    anchors = {a for a, _ in strata}
    assert index.anchor_family_counts == {
        a: {f: len(strata[a, f]) for f in FAMILIES} for a in anchors}
    requests = {key: list(reversed(range(len(values)))) for key, values in strata.items()}
    decoded = index.unrank_many(requests)
    assert set(decoded) == set(requests)
    for key, expected in strata.items():
        assert len(decoded[key]) == len(set(decoded[key]))
        assert set(decoded[key]) == expected
        if expected:
            assert index.unrank(*key, 0) == decoded[key][-1]
    return index


def test_chain_has_exact_inventory_and_rank_bijection(index_type):
    events, edges = inventory([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    index = assert_bijection(index_type, events, edges, 3)
    assert index.total_count == 6


@pytest.mark.parametrize("maximum", [1, 2, 3])
def test_loops_parallel_directions_and_revisit(index_type, maximum):
    events, edges = inventory([0, 1, 0, 2],
        [(0, 0), (0, 1), (0, 1), (1, 0), (1, 2), (2, 2), (0, 3), (2, 3)])
    assert_bijection(index_type, events, edges, maximum)


def test_all_families_have_independent_anchor_strata(index_type):
    # A->B, A->C, D->C creates mixed branch/convergence; another occurrence
    # of A creates revisit. Nothing filters these observable structures.
    events, edges = inventory([0, 1, 2, 3, 0],
        [(0, 1), (0, 2), (3, 2), (1, 3), (3, 4)])
    index = assert_bijection(index_type, events, edges, 3)
    assert all(index.family_counts[f] > 0 for f in FAMILIES)


@pytest.mark.parametrize("seed", range(50))
def test_small_random_graphs_exhaustively_match_bruteforce(index_type, seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 8)
    events, edges = inventory([rng.randrange(4) for _ in range(n)],
        [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randrange(0, 13))])
    for maximum in (1, 2, 3):
        assert_bijection(index_type, events, edges, maximum)


def test_entity_renaming_does_not_change_rank_order(index_type):
    events, edges = inventory([2, 8, 2, 3, 8],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (3, 1)])
    moved = {2: 901, 8: 4, 3: 70}
    renamed = tuple(EventRef(moved[e.entity], e.bin, e.start, e.end, e.category, e.provenance)
                    for e in events)
    first = index_type(events, edges)
    second = index_type(renamed, edges)
    for a, counts in first.anchor_family_counts.items():
        for f, n in counts.items():
            assert [first.unrank(a, f, r) for r in range(n)] == [
                second.unrank(moved[a], f, r) for r in range(n)]


def test_batch_preserves_requested_order_and_duplicate_ranks(index_type):
    events, edges = inventory([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)])
    index = index_type(events, edges)
    requests = {(0, "chain"): [2, 0, 2, 1], (0, "branch"): [3, 0], (99, "chain"): []}
    result = index.unrank_many(requests)
    assert result[99, "chain"] == []
    for (a, f), ranks in requests.items():
        assert result[a, f] == [index.unrank(a, f, r) for r in ranks]


def test_empty_inventory_and_invalid_ranks(index_type):
    index = index_type((), ())
    assert index.total_count == 0
    assert index.family_counts == dict.fromkeys(FAMILIES, 0)
    assert index.anchor_family_counts == {}
    assert index.unrank_many({}) == {}
    with pytest.raises(IndexError):
        index.unrank(0, "chain", 0)
    with pytest.raises(ValueError):
        index.unrank(0, "unknown", 0)
    with pytest.raises(ValueError):
        index_type((), (), max_edges=4)
    events, edges = inventory([0, 1], [(0, 1)])
    index = index_type(events, edges)
    for rank in (-1, 1):
        with pytest.raises(IndexError):
            index.unrank(0, "chain", rank)


def test_coarse_block_boundaries_do_not_skip_candidates(index_type):
    events, edges = inventory(list(range(25)), [(0, i) for i in range(1, 25)])
    index = assert_bijection(index_type, events, edges, 3)
    assert index.total_count == 2324
