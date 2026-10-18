"""Persisted independent exact-direct review probes on synthetic inventories.

These tests characterize existing behavior; they are not evidence of a new
production fix. Exact finite RNG tapes avoid statistical/flaky uniformity gates.
The index oracle enumerates only small test graphs, never production inventories.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import combinations, product
import json
import math
import random
from types import SimpleNamespace

import pytest

from vast_v3.encoders import direct_sampling as sampler
from vast_v3.encoders.fragment_index import FragmentIndex
from vast_v3.encoders.fragments import EventRef, FragmentEdge, FAMILIES


class _RandomTape:
    """A fully enumerated integer RNG path, with its actual draw bounds."""

    def __init__(self, stops, draws):
        self.remaining = iter(zip(stops, draws))

    def randrange(self, stop):
        expected_stop, value = next(self.remaining)
        assert stop == expected_stop
        return value


@pytest.mark.parametrize("size,count", [(n, k) for n in range(8) for k in range(n + 1)])
def test_floyd_every_finite_rng_tape_gives_exact_uniform_subsets(monkeypatch, size, count):
    stops = tuple(range(size - count + 1, size + 1))
    hits = Counter()
    for draws in product(*(range(stop) for stop in stops)):
        tape = _RandomTape(stops, draws)
        # Only the entropy source is replaced; the real Floyd implementation
        # must map every equiprobable path to its selected rank subset.
        monkeypatch.setattr(sampler, "random", SimpleNamespace(Random=lambda seed: tape))
        ranks = sampler._uniform_ranks(size, count, seed=0)
        assert ranks == sorted(set(ranks)) and len(ranks) == count
        assert all(0 <= rank < size for rank in ranks)
        assert next(tape.remaining, None) is None
        hits[tuple(ranks)] += 1
    assert set(hits) == set(combinations(range(size), count))
    assert set(hits.values()) == {math.factorial(count)}


def _inventory(entities, endpoints):
    events = tuple(EventRef(entity, 0, 0., 1., 0, f"event-{i:03}")
                   for i, entity in enumerate(entities))
    # Distinct typed owner provenance keeps parallel directed edges distinct.
    edges = tuple(FragmentEdge(source, target, i % 3, i, bool(i % 2))
                  for i, (source, target) in enumerate(endpoints))
    return events, edges


@pytest.mark.parametrize("seed", range(12))
def test_nonmonotone_entity_owner_and_input_order_changes_replay_exact_audit(seed):
    rng = random.Random(seed)
    endpoints = [(0, 0), (0, 1), (0, 1), (1, 2), (2, 3), (4, 3), (0, 4), (5, 4)]
    endpoints += [(rng.randrange(6), rng.randrange(6)) for _ in range(6)]
    events, edges = _inventory((4, 0, 4, 1, 3, 2), endpoints)
    owners = {edge.owner: f"owner-{edge.owner:03}" for edge in edges}
    first = sampler.extract_fragments_direct(events, edges, delta=1., seed=seed,
        per_anchor_cap=3, case_cap=7, owner_keys=owners)

    entity_names = {0: 91, 1: 3, 2: 72, 3: 18, 4: 6}
    order = list(range(len(events)))
    rng.shuffle(order)
    positions = {old: new for new, old in enumerate(order)}
    new_owner_names = [1000 + 13 * i for i in range(len(owners))]
    rng.shuffle(new_owner_names)
    owner_names = dict(zip(owners, new_owner_names))
    moved_events = tuple(replace(events[i], entity=entity_names[events[i].entity],
                                 start=events[i].start - 128., end=events[i].end - 128.)
                         for i in order)
    moved_edges = [replace(edge, source=positions[edge.source], target=positions[edge.target],
                           owner=owner_names[edge.owner]) for edge in edges]
    rng.shuffle(moved_edges)
    moved = sampler.extract_fragments_direct(moved_events, moved_edges, delta=1., seed=seed,
        per_anchor_cap=3, case_cap=7,
        owner_keys={owner_names[owner]: key for owner, key in owners.items()},
        sampling_audit=first.sampling_audit)

    assert first.sampling_audit == moved.sampling_audit
    assert [f.provenance for f in first.fragments] == [f.provenance for f in moved.fragments]
    assert moved.per_anchor_candidates == {
        entity_names[anchor]: count for anchor, count in first.per_anchor_candidates.items()}


_AUDIT_FIELDS = ("schema", "seed", "inventory_sha256", "caps", "family_counts",
                 "anchors", "case", "final_provenance", "audit_sha256")
_NESTED_MUTATIONS = {
    "anchor_rank": ("anchors", 0, "ranks", "chain", 0),
    "anchor_quota": ("anchors", 0, "quotas", "chain"),
    "case_rank": ("case", "ranks", "chain", 0),
    "case_population": ("case", "populations", "chain"),
    "family_count": ("family_counts", "chain"),
}
_AUDIT_MUTATIONS = (["missing_" + key for key in _AUDIT_FIELDS]
                    + list(_NESTED_MUTATIONS) + ["extra_field", "bad_hash", "fake_provenance"])


@pytest.mark.parametrize("mutation", _AUDIT_MUTATIONS)
def test_strict_audit_rejects_seventeen_missing_or_tampered_records(mutation):
    events, edges = _inventory((0, 1), [(0, 1)] * 3)
    options = dict(delta=1., seed=7, per_anchor_cap=1, case_cap=1, max_edges=1)
    original = sampler.extract_fragments_direct(events, edges, **options).sampling_audit
    changed = deepcopy(original)
    if mutation.startswith("missing_"):
        del changed[mutation.removeprefix("missing_")]
    elif mutation in _NESTED_MUTATIONS:
        path = _NESTED_MUTATIONS[mutation]
        node = changed
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] += 1
    elif mutation == "extra_field":
        changed["unknown"] = "extra"
    elif mutation == "bad_hash":
        changed["audit_sha256"] = "0" * 64
    else:
        changed["final_provenance"].append("not-a-fragment")
    if "audit_sha256" in changed and mutation != "bad_hash":
        # A valid checksum cannot make altered sampling semantics acceptable.
        payload = {key: value for key, value in changed.items() if key != "audit_sha256"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        changed["audit_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    with pytest.raises(ValueError, match="audit"):
        sampler.extract_fragments_direct(events, edges, **options, sampling_audit=changed)


def _independent_oracle(events, edges):
    """Vertex reachability and family predicates, independent of index helpers."""
    family_counts = Counter()
    strata = defaultdict(set)
    for size in (1, 2, 3):
        for indices in combinations(range(len(edges)), size):
            selected = [edges[i] for i in indices]
            vertices = {vertex for edge in selected for vertex in (edge.source, edge.target)}
            reached = {min(vertices)}
            for _ in range(size):
                for edge in selected:
                    if edge.source in reached or edge.target in reached:
                        reached.update((edge.source, edge.target))
            if reached != vertices:
                continue
            anchors = {events[vertex].entity for vertex in vertices}
            if len(anchors) < len(vertices):
                family = "revisit"
            else:
                outgoing, incoming = defaultdict(set), defaultdict(set)
                for edge in selected:
                    outgoing[edge.source].add(edge.target)
                    incoming[edge.target].add(edge.source)
                branch = any(len(neighbors) > 1 for neighbors in outgoing.values())
                convergence = any(len(neighbors) > 1 for neighbors in incoming.values())
                family = ("mixed_branch_convergence" if branch and convergence else
                          "branch" if branch else "convergence" if convergence else "chain")
            family_counts[family] += 1
            for anchor in anchors:
                strata[anchor, family].add(indices)
    return family_counts, strata


def _assert_index_bijection(events, edges):
    expected_counts, strata = _independent_oracle(events, edges)
    index = FragmentIndex(events, edges)
    assert index.family_counts == {family: expected_counts[family] for family in FAMILIES}
    assert index.total_count == sum(expected_counts.values())
    assert index.anchor_family_counts == {
        anchor: {family: len(strata.get((anchor, family), ())) for family in FAMILIES}
        for anchor in {anchor for anchor, _ in strata}}
    # Reverse requests exercise output order as well as exact set membership.
    requests = {key: list(reversed(range(len(candidates)))) for key, candidates in strata.items()}
    decoded = index.unrank_many(requests)
    assert set(decoded) == set(requests)
    for key, expected in strata.items():
        assert len(decoded[key]) == len(set(decoded[key]))
        assert set(decoded[key]) == expected
        assert index.unrank(*key, 0) == decoded[key][-1]
    return index


@pytest.mark.parametrize("seed", range(100))
def test_small_multigraph_counts_and_every_anchor_rank_match_independent_oracle(seed):
    rng = random.Random(16917 + seed)
    size = rng.randrange(2, 8)
    entities = [rng.choice((0, 7, 2, 91)) for _ in range(size)]
    endpoints = [(rng.randrange(size), rng.randrange(size)) for _ in range(rng.randrange(4, 17))]
    _assert_index_bijection(*_inventory(entities, endpoints))


def test_cross_block_parallel_loop_and_revisit_rank_bijection():
    endpoints = [(0, 0), (0, 1), (0, 1), (1, 0), (1, 2), (2, 3), (4, 3), (0, 4), (5, 4)]
    # Twenty-five extra incident typed edges force more than 256 enumeration
    # units while retaining loops, parallel edges, revisit, and all families.
    endpoints += [(0, i % 6) if i % 2 else (i % 6, 0) for i in range(25)]
    index = _assert_index_bijection(*_inventory((0, 1, 0, 2, 3, 4), endpoints))
    assert len(index._blocks) > 1
    assert all(index.family_counts[family] > 0 for family in FAMILIES)
