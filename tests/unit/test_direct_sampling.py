"""Exact direct sampler distribution/identity contracts, using synthetic graphs."""
from collections import Counter
from dataclasses import replace
import importlib
import hashlib
import json
import random

import pytest

from vast_v3.encoders.fragments import (EventRef, FragmentEdge, extract_fragments,
                                        _balanced_select, _seed, FAMILIES)


def direct():
    try:
        return importlib.import_module('vast_v3.encoders.direct_sampling').extract_fragments_direct
    except ModuleNotFoundError:
        pytest.fail('approved exact direct sampler is not implemented')


def graph():
    events = tuple(EventRef(i, 0, 0., 1., 0, f'event-{i}') for i in range(6))
    edges = tuple(FragmentEdge(0, i, 0, i) for i in range(1, 6))
    return events, edges


def test_uncapped_direct_matches_exhaustive_inventory_and_all_counts():
    fn = direct()
    rng = random.Random(182)
    for _ in range(40):
        events = tuple(EventRef(rng.randrange(4), 0, 0., 1., 0, f'e-{i}') for i in range(5))
        edges = tuple(FragmentEdge(rng.randrange(5), rng.randrange(5), rng.randrange(2), i)
                      for i in range(7))
        a = extract_fragments(events, edges, delta=1., per_anchor_cap=1000, case_cap=1000)
        b = fn(events, edges, delta=1., per_anchor_cap=1000, case_cap=1000)
        assert b.fragments == a.fragments
        for name in ('candidate_count', 'per_anchor_candidates', 'per_anchor_retained',
                     'family_candidates', 'family_retained'):
            assert getattr(b, name) == getattr(a, name)
        assert b.sampling_marks == {}


def test_coupled_renaming_time_shift_and_owner_permutation_replay_exactly():
    fn = direct(); events, edges = graph()
    owners = {i: f'owner-{i}' for i in range(1, 6)}
    a = fn(events, edges, delta=1., seed=17, per_anchor_cap=2, case_cap=4, owner_keys=owners)
    order = (4, 2, 0, 5, 3, 1); inv = {old: new for new, old in enumerate(order)}
    names = {0: 99, 1: 50, 2: 73, 3: 11, 4: 128, 5: 39}
    moved = tuple(replace(events[i], entity=names[events[i].entity], start=100., end=101.) for i in order)
    moved_edges = tuple(replace(e, source=inv[e.source], target=inv[e.target], owner=100 + e.owner)
                        for e in reversed(edges))
    b = fn(moved, moved_edges, delta=1., seed=17, per_anchor_cap=2, case_cap=4,
           owner_keys={100 + i: value for i, value in owners.items()}, sampling_audit=a.sampling_audit)
    assert [x.provenance for x in a.fragments] == [x.provenance for x in b.fragments]
    assert a.sampling_audit == b.sampling_audit


def test_changed_inventory_seed_or_cap_cannot_replay_old_audit():
    fn = direct(); events, edges = graph()
    a = fn(events, edges, delta=1., seed=5)
    for kwargs in ({'seed': 6}, {'case_cap': 2}, {'links': edges[:-1]}):
        args = {'events': events, 'links': edges, 'delta': 1., 'seed': 5,
                'sampling_audit': a.sampling_audit, **kwargs}
        with pytest.raises(ValueError, match='audit'):
            fn(**args)
    tampered = json.loads(json.dumps(a.sampling_audit)); tampered['seed'] = True
    with pytest.raises(ValueError, match='audit'):
        fn(events, edges, delta=1., seed=5, sampling_audit=tampered)


def test_empty_and_zero_caps_have_no_synthetic_fragment():
    fn = direct()
    assert fn((), (), delta=1.).candidate_count == 0
    events, edges = graph()
    for caps in ({'case_cap': 0}, {'per_anchor_cap': 0}):
        result = fn(events, edges, delta=1., **caps)
        assert not result.fragments and result.candidate_count > 0


def test_uniform_one_family_anchor_then_case_does_not_bias_by_edge_order():
    fn = direct(); events, edges = graph()
    # Retain all anchor choices so the final case sampler selects one of five
    # one-edge chain fragments. This is a statistical diagnostic, not a proof.
    hits = Counter()
    for seed in range(1000):
        result = fn(events, edges, delta=1., seed=seed, max_edges=1,
                    per_anchor_cap=10, case_cap=1)
        hits[result.fragments[0].provenance] += 1
    assert len(hits) == 5
    assert all(145 <= n <= 255 for n in hits.values())


def test_caps_and_quota_match_reference_for_every_seed():
    fn = direct(); events, edges = graph()
    full = extract_fragments(events, edges, delta=1., per_anchor_cap=1000, case_cap=1000)
    for seed in range(30):
        a = extract_fragments(events, edges, delta=1., seed=seed, per_anchor_cap=2, case_cap=100)
        b = fn(events, edges, delta=1., seed=seed, per_anchor_cap=2, case_cap=100)
        assert b.per_anchor_retained == a.per_anchor_retained
        # Audit includes every anchor's precise population and quota, without
        # a per-candidate random-mark dictionary.
        assert b.sampling_audit['schema'] == 'vast-v3-exact-direct/1'
        assert b.sampling_marks == {}
        assert all(v <= 2 for v in b.per_anchor_retained.values())
        for anchor in range(6):
            key = hashlib.sha256(json.dumps(sorted(e.provenance for e in events if e.entity == anchor)).encode()).hexdigest()
            group = [f for f in full if anchor in {e.entity for e in f.events}]
            selected = _balanced_select(group, 2, dict.fromkeys((f.provenance for f in group), .5),
                                        _seed(seed, 'anchor/' + key))
            counts = {f: sum(x.family == f for x in selected) for f in FAMILIES}
            audited = next(x for x in b.sampling_audit['anchors'] if x['anchor_key'] == key)
            assert audited['quotas'] == counts


def test_no_exhaustive_enumeration_is_called(monkeypatch):
    fn = direct(); events, edges = graph()
    import vast_v3.encoders.fragments as reference
    def forbidden(*args, **kwargs):
        raise AssertionError('the direct sampler must not enumerate the whole population')
    monkeypatch.setattr(reference, 'extract_fragments', forbidden)
    assert fn(events, edges, delta=1., seed=1).candidate_count > 0
