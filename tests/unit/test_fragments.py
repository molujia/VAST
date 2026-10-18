"""Observable-fragment extraction contract, without root-label inputs."""

import importlib

import pytest


@pytest.fixture
def api():
    class Module:
        def __getattr__(self, name):
            try:
                return getattr(importlib.import_module("vast_v3.encoders.fragments"), name)
            except ModuleNotFoundError as error:
                pytest.fail(f"fragment implementation is missing: {error}")
    return Module()


def event(api, entity, time, name=None, category=0):
    return api.EventRef(entity, time, float(time), float(time) + 0.25,
                        category, name or f"observation-{entity}-{time}")


def test_connected_edge_sets_are_exactly_deduplicated(api):
    events = tuple(event(api, i, i) for i in range(4))
    links = tuple(api.FragmentEdge(i, i + 1, 0, i) for i in range(3))
    result = api.extract_fragments(events, links + (links[0],), delta=1.0)
    assert result.candidate_count == 6
    assert sorted(len(fragment.edges) for fragment in result.fragments) == [1, 1, 1, 2, 2, 3]
    assert len({fragment.provenance for fragment in result}) == 6


def test_family_priority_and_full_paths_preserve_shared_convergence(api):
    events = (event(api, 0, 0), event(api, 1, 0),
              event(api, 2, 1), event(api, 3, 2))
    edges = (api.FragmentEdge(0, 2, 0, 0), api.FragmentEdge(1, 2, 0, 1),
             api.FragmentEdge(2, 3, 0, 2))
    fragment = api.Fragment(events, edges, "convergence", "shared")
    assert api.structural_family(fragment.events, fragment.edges) == "convergence"
    assert set(api.ordered_paths(fragment)) == {(0, 2), (1, 2)}


def test_revisit_has_priority_over_branch(api):
    events = (event(api, 0, 0), event(api, 0, 1), event(api, 2, 2))
    edges = (api.FragmentEdge(0, 1, 0), api.FragmentEdge(0, 2, 0))
    assert api.structural_family(events, edges) == "revisit"


def test_unknown_order_stops_paths_without_deleting_edges(api):
    events = (event(api, 0, 0), event(api, 1, 0), event(api, 2, 1))
    edges = (api.FragmentEdge(0, 1, 0, 0), api.FragmentEdge(1, 2, 0, 1))
    fragment = api.Fragment(events, edges, "chain", "partial")
    assert api.ordered_paths(fragment) == ((1,),)
    assert len(fragment.edges) == 2


def test_horizon_and_backward_time_do_not_connect(api):
    events = (event(api, 0, 0), event(api, 1, 20), event(api, 2, 2))
    result = api.extract_fragments(events, (api.FragmentEdge(0, 1, 0),
                                           api.FragmentEdge(2, 0, 0)), delta=1.0)
    assert result.fragments == ()
    assert result.candidate_count == 0


def test_different_time_occurrences_remain_distinct(api):
    events = (event(api, 0, 0), event(api, 1, 1),
              event(api, 0, 2), event(api, 1, 3))
    result = api.extract_fragments(events, (api.FragmentEdge(0, 1, 0, 0),
                                           api.FragmentEdge(2, 3, 0, 0)), delta=1.0)
    assert len(result.fragments) == 2


def test_family_balanced_case_cap_and_coupled_permutation(api):
    events = tuple(event(api, i, i) for i in range(7))
    links = tuple(api.FragmentEdge(0, i, 0, i - 1) for i in range(1, 7))
    first = api.extract_fragments(events, links, delta=1.0, seed=41,
                                 per_anchor_cap=2, case_cap=4)
    assert len(first.fragments) == 4
    assert set(fragment.family for fragment in first) == {"chain", "branch"}
    assert max(first.per_anchor_retained.values()) <= 2
    order = (4, 6, 2, 0, 3, 1, 5)
    inverse = {old: new for new, old in enumerate(order)}
    renamed = tuple(api.EventRef(20 + e.entity, e.bin, e.start + 100,
                                 e.end + 100, e.category, e.provenance)
                    for e in (events[i] for i in order))
    reversed_links = tuple(api.FragmentEdge(inverse[e.source], inverse[e.target],
                                           e.relation, e.owner, e.response)
                           for e in reversed(links))
    second = api.extract_fragments(renamed, reversed_links, delta=1.0, seed=41,
                                  per_anchor_cap=2, case_cap=4,
                                  sampling_marks=first.sampling_marks)
    assert {f.provenance for f in first} == {f.provenance for f in second}


def test_enumeration_safety_is_explicit_error_not_silent_sampling(api):
    events = tuple(event(api, i, i) for i in range(4))
    links = tuple(api.FragmentEdge(0, i, 0, i) for i in range(1, 4))
    with pytest.raises(api.EnumerationLimitError, match="enumeration"):
        api.extract_fragments(events, links, delta=1.0, max_enumerated=2)


def test_empty_inventory_has_no_synthetic_fragment(api):
    result = api.extract_fragments((), (), delta=1.0)
    assert result.fragments == ()
    assert result.sampling_marks == {}
    assert result.candidate_count == 0


def test_invalid_interval_is_not_silently_used(api):
    with pytest.raises(ValueError, match="interval"):
        api.EventRef(0, 0, 3.0, 2.0, 0, "bad")


def test_case_reservoir_has_its_own_coupled_random_marks(api):
    events = tuple(event(api, i, i) for i in range(5))
    links = tuple(api.FragmentEdge(0, i, 0, i - 1) for i in range(1, 5))
    complete = api.extract_fragments(events, links, delta=1., max_edges=1)
    default = api.extract_fragments(events, links, delta=1., max_edges=1, case_cap=1)
    target = next(f for f in complete if f.provenance != default.fragments[0].provenance)
    marks = {"case/" + f.provenance: (0. if f == target else 1.) for f in complete}
    selected = api.extract_fragments(events, links, delta=1., max_edges=1,
                                     case_cap=1, sampling_marks=marks)
    assert selected.fragments == (target,)
    assert any(key.startswith("anchor/") for key in selected.sampling_marks)


def test_stable_owner_keys_transport_sampling_through_owner_reindexing(api):
    events = tuple(event(api, i, i) for i in range(5))
    links = tuple(api.FragmentEdge(0, i, 0, i - 1) for i in range(1, 5))
    owner_keys = {index: f"measurement-{index}" for index in range(4)}
    first = api.extract_fragments(events, links, delta=1., seed=7, case_cap=3,
                                  owner_keys=owner_keys)
    owner_permutation = {0: 3, 1: 2, 2: 0, 3: 1}
    moved_links = tuple(api.FragmentEdge(edge.source, edge.target, edge.relation,
                                        owner_permutation[edge.owner]) for edge in links)
    moved_keys = {owner_permutation[index]: key for index, key in owner_keys.items()}
    second = api.extract_fragments(events, moved_links, delta=1., seed=7, case_cap=3,
                                   owner_keys=moved_keys, sampling_marks=first.sampling_marks)
    assert {f.provenance for f in first} == {f.provenance for f in second}


def test_extraction_reports_family_counts_before_and_after_sampling(api):
    events = tuple(event(api, i, i) for i in range(4))
    links = tuple(api.FragmentEdge(0, i, 0, i - 1) for i in range(1, 4))
    result = api.extract_fragments(events, links, delta=1., case_cap=2)
    assert result.family_candidates["chain"] == 3
    assert result.family_candidates["branch"] == 4
    assert sum(result.family_retained.values()) == 2
    assert result.family_retained["chain"] == result.family_retained["branch"] == 1
