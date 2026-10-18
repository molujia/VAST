"""Count-only construction preserves the frozen exact-direct rank oracle."""
from pathlib import Path
import random
import runpy

import pytest

from vast_v3.encoders.fragment_index import FragmentIndex
from vast_v3.encoders.fragments import EventRef, FragmentEdge
import vast_v3.encoders.direct_sampling as direct


FrozenFragmentIndex = runpy.run_path(str(Path(__file__).parents[1] / 'oracles' /
    'fragment_index_pre_count_kernel.py'))['FragmentIndex']


def inventory(seed, *, n=8, edge_count=14):
    rng = random.Random(seed)
    events = tuple(EventRef(rng.randrange(5), 0, 0., 1., 0, f'event-{i:04}') for i in range(n))
    edges = tuple(FragmentEdge(rng.randrange(n), rng.randrange(n), i % 3, i, bool(i % 2))
                  for i in range(edge_count))
    return events, edges


def assert_same_index(current, frozen, *, exhaustive=True):
    assert current.family_counts == frozen.family_counts
    assert current.anchor_family_counts == frozen.anchor_family_counts
    assert current.total_count == frozen.total_count
    assert [(b.start, b.units) for b in current._blocks] == [(b.start, b.units) for b in frozen._blocks]
    assert {key: (w.cumulative, w.blocks) for key, w in current._weights.items()} == {
        key: (w.cumulative, w.blocks) for key, w in frozen._weights.items()}
    assert list(current._units()) == list(frozen._units())
    requests = {}
    rng = random.Random(713)
    for anchor, counts in frozen.anchor_family_counts.items():
        for family, count in counts.items():
            if count:
                ranks = list(range(count)) if exhaustive else [0, count - 1, *rng.choices(range(count), k=8)]
                requests[anchor, family] = list(reversed(ranks))
    assert current.unrank_many(requests) == frozen.unrank_many(requests)


@pytest.mark.parametrize('max_edges', (1, 2, 3))
def test_index_construction_does_not_build_rank_descriptors(monkeypatch, max_edges):
    events, edges = inventory(42)
    expected = FrozenFragmentIndex(events, edges, max_edges=max_edges)
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('counting must not construct rank descriptors')
        patch.setattr(FragmentIndex, '_descriptors', forbidden)
        actual = FragmentIndex(events, edges, max_edges=max_edges)
    assert_same_index(actual, expected)


@pytest.mark.parametrize('seed', range(60))
def test_count_kernel_preserves_every_rank_on_loops_parallel_edges_and_revisits(seed):
    events, edges = inventory(seed)
    for max_edges in (1, 2, 3):
        assert_same_index(FragmentIndex(events, edges, max_edges),
                          FrozenFragmentIndex(events, edges, max_edges))


def test_block_boundaries_and_new_mask_inventories_match_frozen_index():
    n = 52
    events = tuple(EventRef(i % 11, 0, 0., 1., 0, f'event-{i:04}') for i in range(n))
    full = tuple(FragmentEdge(s, t, i % 3, i) for i, (s, t) in enumerate(
        [(0, i) for i in range(1, n)] + [(i, (i + 1) % n) for i in range(n)]))
    for mask in range(4):
        edges = tuple(e for i, e in enumerate(full) if (i + mask) % 5)
        actual, expected = FragmentIndex(events, edges), FrozenFragmentIndex(events, edges)
        assert len(actual._blocks) > 3
        assert_same_index(actual, expected, exhaustive=False)


@pytest.mark.parametrize('seed', (0, 1, 42, 928))
def test_entire_direct_sampling_audit_and_fragments_match_frozen_index(monkeypatch, seed):
    events, edges = inventory(203, n=15, edge_count=43)
    for max_edges in (1, 2, 3):
        for anchor_cap, case_cap in ((0, 0), (2, 3), (8, 128)):
            kwargs = dict(delta=1., seed=seed, max_edges=max_edges,
                          per_anchor_cap=anchor_cap, case_cap=case_cap)
            with monkeypatch.context() as patch:
                patch.setattr(direct, 'FragmentIndex', FrozenFragmentIndex)
                expected = direct.extract_fragments_direct(events, edges, **kwargs)
            actual = direct.extract_fragments_direct(events, edges, **kwargs,
                                                    sampling_audit=expected.sampling_audit)
            assert actual == expected


def test_empty_index_uses_unchanged_zero_counts():
    assert_same_index(FragmentIndex((), ()), FrozenFragmentIndex((), ()))
