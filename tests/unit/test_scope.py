"""Shared candidate-conditioned SCOPE, including ownership and symmetry."""

import importlib

import pytest
import torch


@pytest.fixture
def api():
    class Module:
        def __init__(self, module):
            self.module = module

        def __getattr__(self, name):
            try:
                return getattr(importlib.import_module(self.module), name)
            except ModuleNotFoundError as error:
                pytest.fail(f"SCOPE implementation is missing: {error}")
    return Module("vast_v3.encoders.fragments"), Module("vast_v3.encoders.scope")


def convergence(api, *, overlapping=False):
    f, _ = api
    end = 2.5 if overlapping else 0.25
    events = (f.EventRef(0, 0, 0.0, end, 0, "a"),
              f.EventRef(1, 0, 0.0, end, 0, "b"),
              f.EventRef(2, 2, 2.0, 2.25, 0, "c"))
    return f.Fragment(events, (f.FragmentEdge(0, 2, 0, 0),
                               f.FragmentEdge(1, 2, 0, 1)),
                      "convergence", "abc")


def tensors(nodes=3, times=4, edges=2):
    torch.manual_seed(7)
    return (torch.zeros(nodes, times, 64), torch.zeros(nodes, times, 64),
            torch.randn(edges, times, 64), torch.ones(nodes, times, 3),
            torch.ones(nodes, times, 3), torch.ones(edges, times, dtype=torch.bool))


def encoder(api):
    torch.manual_seed(31)
    return api[1].ScopeEncoder(entity_types=2, relation_types=2)


def test_order_two_signature_distinguishes_increment_order(api):
    _, module = api
    a = torch.tensor([[1., 0.], [0., 1.]])
    forward = module.signature_order_two(a)
    reverse = module.signature_order_two(a.flip(0))
    assert forward.shape == (7,)
    assert torch.equal(forward[1:3], reverse[1:3])
    assert not torch.equal(forward[3:], reverse[3:])
    assert torch.equal(forward, torch.tensor([1., 1., 1., .5, 1., 0., .5]))


def test_candidate_mark_precedes_canonical_owner_binding(api):
    model = encoder(api)
    fragment = convergence(api)
    inputs = tensors()
    swapped = list(inputs)
    swapped[2] = inputs[2].flip(0)
    unmarked, _ = model(fragment, *inputs)
    unmarked_swapped, _ = model(fragment, *swapped)
    marked, role = model(fragment, *inputs, candidate=0)
    marked_swapped, _ = model(fragment, *swapped, candidate=0)
    torch.testing.assert_close(unmarked, unmarked_swapped, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(marked, marked_swapped, rtol=1e-5, atol=1e-6)
    assert role.shape == (32,)
    assert marked.shape == (64,)
    torch.testing.assert_close(marked.norm(), torch.tensor(1.))


def test_permutation_renaming_and_time_shift_preserve_encoding(api):
    f, _ = api
    model = encoder(api)
    fragment = convergence(api)
    inputs = list(tensors())
    inputs[0] = torch.randn_like(inputs[0])
    inputs[1] = torch.randn_like(inputs[1])
    inputs[4] = torch.rand_like(inputs[4])
    expected, expected_role = model(fragment, *inputs, candidate=0)
    order = (2, 0, 1)
    inverse = {old: new for new, old in enumerate(order)}
    entity_perm = {0: 1, 1: 2, 2: 0}
    events = tuple(f.EventRef(entity_perm[e.entity], e.bin, e.start + 73,
                             e.end + 73, e.category, e.provenance)
                   for e in (fragment.events[i] for i in order))
    edges = tuple(f.FragmentEdge(inverse[e.source], inverse[e.target], e.relation,
                                e.owner, e.response) for e in reversed(fragment.edges))
    moved = f.Fragment(events, edges, fragment.family, "renamed-metadata")
    reordered = list(inputs)
    for index in (0, 1, 3, 4):
        reordered[index] = inputs[index][torch.tensor([2, 0, 1])]
    actual, actual_role = model(moved, *reordered, candidate=entity_perm[0])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_role, expected_role)


def test_unknown_order_keeps_owner_numeric_skip(api):
    model = encoder(api)
    fragment = convergence(api, overlapping=True)
    inputs = tensors()
    marked, _ = model(fragment, *inputs, candidate=0)
    changed = list(inputs)
    changed[2] = inputs[2].clone()
    changed[2][0] += 4
    other, _ = model(fragment, *changed, candidate=0)
    assert not torch.allclose(marked, other)


def test_same_entity_matrix_preserves_shared_node_vs_distinct_node(api):
    f, _ = api
    model = encoder(api)
    events = (f.EventRef(0, 0, 0., .25, 0, "a"),
              f.EventRef(1, 1, 1., 1.25, 0, "b"),
              f.EventRef(2, 2, 2., 2.25, 0, "c1"),
              f.EventRef(2, 3, 3., 3.25, 0, "c2"))
    edges = (f.FragmentEdge(0, 1, 0, 0), f.FragmentEdge(0, 2, 0, 1),
             f.FragmentEdge(1, 3, 0, 2))
    shared = f.Fragment(events, edges, "revisit", "shared")
    distinct_events = events[:3] + (f.EventRef(3, 3, 3., 3.25, 0, "d"),)
    distinct = f.Fragment(distinct_events, edges, "branch", "distinct")
    inputs = list(tensors(nodes=4, times=4, edges=3))
    inputs[2].zero_()
    z_shared, r_shared = model(shared, *inputs)
    z_distinct, r_distinct = model(distinct, *inputs)
    assert not torch.allclose(r_shared, r_distinct)
    assert not torch.allclose(z_shared, z_distinct)


def test_tied_mappings_reduce_after_nonlinear_token_mlp(api):
    model = encoder(api)
    captured = []
    hook = model.token_mlp.register_forward_hook(lambda _module, _args, output: captured.append(output))
    try:
        result, _ = model(convergence(api), *tensors())
    finally:
        hook.remove()
    assert len(captured) == 2
    mean = torch.stack(captured).mean(0)
    torch.testing.assert_close(result, mean / mean.norm())


def test_zero_norm_uses_unit_fallback_without_new_fragment(api):
    model = encoder(api)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    fragment = convergence(api)
    result, _ = model(fragment, *tensors())
    expected = torch.zeros(64)
    expected[0] = 1
    assert torch.equal(result, expected)
    assert fragment.provenance == "abc"
    assert fragment.family == "convergence"


def test_empty_fragment_is_not_encoded_as_occurrence(api):
    f, _ = api
    with pytest.raises(ValueError, match="empty"):
        encoder(api)(f.Fragment((), (), "chain", "empty"), *tensors())


def test_owner_skip_backpropagates_only_available_owner_records(api):
    model = encoder(api)
    inputs = list(tensors(edges=3))
    inputs[2].requires_grad_()
    inputs[5][1] = False
    z, _ = model(convergence(api), *inputs, candidate=0)
    z[5].backward()
    assert inputs[2].grad[0].abs().sum() > 0
    assert inputs[2].grad[1].abs().sum() == 0
    assert inputs[2].grad[2].abs().sum() == 0
