"""Parameter-free direct-selection caching never stores trainable state."""

import copy
import importlib
import json

import pytest

from vast_v3.encoders.fragments import EventRef, FragmentEdge
from vast_v3.encoders.direct_sampling import extract_fragments_direct


@pytest.fixture
def cache_type():
    try:
        return importlib.import_module("vast_v3.encoders.fragment_cache").DirectSamplingCache
    except ModuleNotFoundError:
        def missing(*args, **kwargs):
            pytest.fail("direct selection skeleton cache implementation is missing")
        return missing


def fixture_inventory():
    events = tuple(EventRef(entity, i, float(i), float(i) + .25, i % 2, f"event-{i}")
                   for i, entity in enumerate((20, 4, 20, 7, 4)))
    edges = (FragmentEdge(0, 1, 0, 0), FragmentEdge(1, 2, 1, 1),
             FragmentEdge(0, 3, 0, 2), FragmentEdge(2, 3, 1, 0),
             FragmentEdge(3, 4, 0, 1), FragmentEdge(0, 1, 0, 2, True))
    return events, edges, {0: "owner-a", 1: "owner-b", 2: "owner-c"}


def extract(events, edges, owners, **kwargs):
    return extract_fragments_direct(events, edges, delta=1., seed=19,
                                    per_anchor_cap=2, case_cap=4, owner_keys=owners, **kwargs)


def test_cache_miss_then_hit_matches_independent_uncached_result(cache_type, monkeypatch):
    events, edges, owners = fixture_inventory()
    reference = extract(events, edges, owners)
    cache = cache_type()
    first = extract(events, edges, owners, sampling_cache=cache)
    assert first == reference
    assert cache.stats["misses"] == 1 and cache.stats["hits"] == 0
    module = importlib.import_module("vast_v3.encoders.direct_sampling")
    def cannot_reindex(*args, **kwargs):
        pytest.fail("a cache hit must not rebuild the candidate index")
    monkeypatch.setattr(module, "FragmentIndex", cannot_reindex)
    second = extract(events, edges, owners, sampling_cache=cache,
                     sampling_audit=first.sampling_audit)
    assert second == reference
    assert cache.stats["hits"] == 1
    assert "cache" not in json.dumps(second.sampling_audit)


def test_shift_rename_category_bin_and_owner_reindex_rebind_current_records(cache_type, monkeypatch):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    original = extract(events, edges, owners, sampling_cache=cache)
    rename = {20: 1, 4: 92, 7: 6}
    event_order = (3, 1, 4, 0, 2)
    inverse = {old: new for new, old in enumerate(event_order)}
    moved_events = tuple(EventRef(rename[events[i].entity], events[i].bin + 7,
                                   events[i].start + 64., events[i].end + 64.,
                                   events[i].category + 3, events[i].provenance)
                         for i in event_order)
    owner_order = {0: 2, 1: 0, 2: 1}
    moved_edges = tuple(FragmentEdge(inverse[e.source], inverse[e.target], e.relation,
                                     owner_order[e.owner], e.response) for e in reversed(edges))
    moved_owners = {owner_order[a]: value for a, value in owners.items()}
    reference = extract(moved_events, moved_edges, moved_owners,
                        sampling_audit=original.sampling_audit)
    module = importlib.import_module("vast_v3.encoders.direct_sampling")
    monkeypatch.setattr(module, "FragmentIndex", lambda *args, **kwargs:
                        pytest.fail("transported identical inventory should hit cache"))
    cached = extract(moved_events, moved_edges, moved_owners, sampling_cache=cache,
                     sampling_audit=original.sampling_audit)
    assert cached == reference
    assert cached.sampling_audit == original.sampling_audit
    assert cache.stats["hits"] == 1
    assert all(e.start >= 64. and e.category >= 3 and e.bin >= 7
               for fragment in cached for e in fragment.events)


def test_eligibility_boundary_change_is_a_miss_even_with_same_input_lengths(cache_type):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    first = extract(events, edges, owners, sampling_cache=cache)
    changed = tuple(EventRef(e.entity, e.bin, 100. if i == 1 else e.start,
                             100.25 if i == 1 else e.end, e.category, e.provenance)
                    for i, e in enumerate(events))
    reference = extract(changed, edges, owners)
    second = extract(changed, edges, owners, sampling_cache=cache)
    assert second == reference
    assert second.candidate_count != first.candidate_count
    assert cache.stats["misses"] == 2 and cache.stats["hits"] == 0


@pytest.mark.parametrize("mutation", ["entity_partition", "owner_identity", "relation", "response", "isolated_event"])
def test_semantic_inventory_changes_miss(cache_type, mutation):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    extract(events, edges, owners, sampling_cache=cache)
    if mutation == "entity_partition":
        e = events[0]
        events = (EventRef(99, e.bin, e.start, e.end, e.category, e.provenance),) + events[1:]
    elif mutation == "owner_identity":
        owners = {**owners, 0: "different-owner"}
    elif mutation in ("relation", "response"):
        e = edges[0]
        edges = (FragmentEdge(e.source, e.target, e.relation + (mutation == "relation"),
                              e.owner, not e.response if mutation == "response" else e.response),) + edges[1:]
    else:
        events += (EventRef(20, 9, 9., 9.25, 0, "isolated-observation"),)
    assert extract(events, edges, owners, sampling_cache=cache) == extract(events, edges, owners)
    assert cache.stats["misses"] == 2


def test_seed_and_caps_are_in_cache_identity(cache_type):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    for kwargs in ({}, {"seed": 2}, {"per_anchor_cap": 1}, {"case_cap": 1}, {"max_edges": 2}):
        baseline = dict(delta=1., seed=1, per_anchor_cap=2, case_cap=4, owner_keys=owners)
        baseline.update(kwargs)
        assert extract_fragments_direct(events, edges, sampling_cache=cache, **baseline) == \
            extract_fragments_direct(events, edges, **baseline)
    assert cache.stats["misses"] == 5 and cache.stats["hits"] == 0


@pytest.mark.parametrize("mutation", ["ranks", "family_counts", "final_provenance", "audit_sha256", "seed"])
def test_supplied_audit_tampering_is_rejected_on_hit(cache_type, mutation):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    first = extract(events, edges, owners, sampling_cache=cache)
    audit = copy.deepcopy(first.sampling_audit)
    if mutation == "ranks":
        audit["case"]["ranks"]["chain"].append(999)
    elif mutation == "family_counts":
        audit["family_counts"]["chain"] += 1
    elif mutation == "final_provenance":
        audit["final_provenance"] = []
    elif mutation == "seed":
        audit["seed"] += 1
    else:
        audit["audit_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit"):
        extract(events, edges, owners, sampling_cache=cache, sampling_audit=audit)


def test_mutating_returned_audit_cannot_poison_cached_payload(cache_type):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    expected = extract(events, edges, owners)
    first = extract(events, edges, owners, sampling_cache=cache)
    first.sampling_audit["family_counts"]["chain"] = -100
    second = extract(events, edges, owners, sampling_cache=cache)
    assert second == expected
    second.sampling_audit["case"]["ranks"]["chain"].append(999)
    assert extract(events, edges, owners, sampling_cache=cache) == expected


def test_lru_entry_eviction_and_exact_serialized_byte_accounting(cache_type):
    cache = cache_type(max_bytes=10000, max_entries=2)
    value = {"unicode": "观察", "ranks": [1, 4]}
    assert cache.put("a", value)
    expected_bytes = len("a".encode()) + len(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                         allow_nan=False).encode())
    assert cache.stats["bytes"] == expected_bytes
    cache.put("b", {"ranks": []})
    assert cache.get("a") == value  # a becomes most recently used.
    cache.put("c", {"ranks": [2]})
    assert cache.get("b") is None
    assert cache.get("a") == value
    assert cache.stats["entries"] == 2 and cache.stats["evictions"] == 1
    snapshot = cache.stats
    snapshot["bytes"] = -1
    assert cache.stats["bytes"] > 0
    cache.clear()
    assert cache.stats["bytes"] == cache.stats["entries"] == 0


def test_byte_eviction_oversized_and_disabled_caches(cache_type):
    cache = cache_type(max_bytes=15, max_entries=8)
    cache.put("a", {"x": 1})
    cache.put("b", {"x": 2})  # Each entry costs eight encoded bytes.
    assert cache.stats["evictions"] == 1 and cache.stats["bytes"] == 8
    assert not cache.put("oversized", {"x": "a" * 100})
    assert cache.stats["oversize_skips"] == 1
    assert cache.stats["bytes"] <= 15
    for kwargs in ({"max_bytes": 0}, {"max_entries": 0}):
        disabled = cache_type(**kwargs)
        assert not disabled.put("x", {"x": 1})
        assert disabled.get("x") is None


def test_cache_payload_is_json_only_and_capacity_is_validated(cache_type):
    for kwargs in ({"max_bytes": -1}, {"max_entries": -1}, {"max_entries": 1.5}, {"max_bytes": True}):
        with pytest.raises(ValueError):
            cache_type(**kwargs)
    cache = cache_type()
    with pytest.raises((TypeError, ValueError)):
        cache.put("opaque", {"activation": object()})
    with pytest.raises(ValueError):
        cache.put("nan", {"x": float("nan")})


@pytest.mark.parametrize("caps", [{"per_anchor_cap": 0}, {"case_cap": 0}, {}])
def test_empty_or_zero_output_cache_hits_preserve_all_counts(cache_type, caps):
    events, edges, owners = fixture_inventory()
    if not caps:
        events, edges, owners = (), (), {}
    cache = cache_type()
    args = dict(delta=1., seed=5, owner_keys=owners, **caps)
    expected = extract_fragments_direct(events, edges, **args)
    first = extract_fragments_direct(events, edges, sampling_cache=cache, **args)
    second = extract_fragments_direct(events, edges, sampling_cache=cache,
                                     sampling_audit=first.sampling_audit, **args)
    assert first == second == expected
    assert second.fragments == ()
    assert cache.stats["hits"] == cache.stats["misses"] == 1


def test_direct_recompute_after_eviction_is_identical(cache_type):
    events, edges, owners = fixture_inventory()
    cache = cache_type(max_entries=1)
    args = dict(delta=1., owner_keys=owners, per_anchor_cap=2, case_cap=4)
    first = extract_fragments_direct(events, edges, seed=1, sampling_cache=cache, **args)
    extract_fragments_direct(events, edges, seed=2, sampling_cache=cache, **args)
    again = extract_fragments_direct(events, edges, seed=1, sampling_cache=cache,
                                    sampling_audit=first.sampling_audit, **args)
    assert again == first
    assert cache.stats["misses"] == 3 and cache.stats["evictions"] == 2


def test_cache_lookup_does_not_bypass_conflicting_provenance_validation(cache_type):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    extract(events, edges, owners, sampling_cache=cache)
    e = events[0]
    conflicting = EventRef(e.entity, e.bin, e.start, e.end, e.category + 1, e.provenance)
    with pytest.raises(ValueError, match="conflicting"):
        extract(events + (conflicting,), edges, owners, sampling_cache=cache)


def test_changed_temporal_order_with_same_eligible_edges_rebinds_intervals(cache_type):
    events, edges, owners = fixture_inventory()
    cache = cache_type()
    first = extract(events, edges, owners, sampling_cache=cache)
    overlap = tuple(EventRef(e.entity, e.bin, 0., 1., e.category, e.provenance) for e in events)
    expected = extract(overlap, edges, owners)
    result = extract(overlap, edges, owners, sampling_cache=cache, sampling_audit=first.sampling_audit)
    assert result == expected
    assert cache.stats["hits"] == 1
    assert all(e.start == 0. and e.end == 1. for fragment in result for e in fragment.events)
