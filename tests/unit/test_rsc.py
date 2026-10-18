import importlib
import time

import pytest
import torch


def api():
    try:
        return importlib.import_module("vast_v3.models.rsc")
    except ModuleNotFoundError:
        pytest.fail("RSC implementation is missing")


def decoder(**kwargs):
    torch.manual_seed(71)
    return api().RSCDecoder(entity_types=2, relation_types=2,
                            metric_channels=3, trace_channels=2, **kwargs)


def inputs():
    return (torch.randn(5, 10, 64), torch.ones(5, 10, dtype=torch.bool),
            torch.tensor([0, 0, 1, 0, 0]),
            [(0, 1, 0), (1, 2, 1), (2, 3, 0)],
            torch.ones(3, 10, dtype=torch.bool),
            torch.tensor([[True, False, False, False, False]]))


def test_unreachable_states_and_all_heads_are_exact_zero():
    model = decoder()
    args = inputs()
    states = model(*args)
    assert states.shape == (1, 5, 10, 64)
    assert torch.count_nonzero(states[:, 4]) == 0
    assert torch.count_nonzero(states[:, 3]) > 0
    assert all(torch.count_nonzero(x[:, 4]) == 0
               for x in model.predict_nodes(states).values())
    edge = model.predict_edges(states, [(4, 4, 0)])
    assert edge.shape == (1, 1, 10, 2)
    assert torch.count_nonzero(edge) == 0


def test_zero_driver_and_source_removal_cannot_be_bypassed():
    model = decoder()
    h, visible, types, relations, active, sources = inputs()
    for hs, vs, ss in [(torch.zeros_like(h), visible, sources),
                       (h, torch.zeros_like(visible), sources),
                       (h, visible, torch.zeros_like(sources))]:
        states = model(hs, vs, types, relations, active, ss)
        assert torch.count_nonzero(states) == 0
        assert all(torch.count_nonzero(x) == 0 for x in model.predict_nodes(states).values())
        assert torch.count_nonzero(model.predict_edges(states, relations)) == 0


def test_driver_is_shared_and_hidden_values_do_not_leak():
    model = decoder()
    h = torch.randn(3, 2, 64)
    visible = torch.tensor([[True, True], [False, False], [False, False]])
    sources = torch.tensor([[False, True, True]])
    args = (visible, torch.zeros(3, dtype=torch.long), [],
            torch.empty(0, 2, dtype=torch.bool), sources)
    first = model(h, *args)
    h[1:] = float("nan")
    second = model(h, *args)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, 1], first[:, 2])


def test_driver_widths_bias_freedom_and_exact_residual_injection():
    model = decoder()
    assert model.driver[0].weight.shape == (16, 64)
    assert model.driver[2].weight.shape == (8, 16)
    assert model.layers == 3
    assert model.lags == (0, 1, 2, 4, 8)
    assert all(module.bias is None for module in model.modules()
               if isinstance(module, torch.nn.Linear))
    with torch.no_grad():
        model.self_transform.weight.zero_()
        for module in model.relation_transforms:
            module.weight.zero_()
    args = inputs()
    b = model.driver(args[0].mean(0))
    expected = model.source_injections[0](b)
    states = model(*args)
    torch.testing.assert_close(states[0, 0], expected)
    assert torch.count_nonzero(states[:, 1:]) == 0


def test_inactive_edges_and_no_more_than_three_hops():
    model = decoder()
    h, visible, types, _, _, sources = inputs()
    relations = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0)]
    active = torch.ones(4, 10, dtype=torch.bool)
    states = model(h, visible, types, relations, active, sources)
    assert torch.count_nonzero(states[:, 3]) > 0
    assert torch.count_nonzero(states[:, 4]) == 0
    active[0] = False
    states = model(h, visible, types, relations, active, sources)
    assert torch.count_nonzero(states[:, 1:]) == 0


def test_attention_is_normalized_over_valid_incoming_edge_lag_pairs():
    model = decoder(layers=1)
    with torch.no_grad():
        model.self_transform.weight.zero_()
        for layer in model.attention.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
        model.relation_transforms[0].weight.copy_(torch.eye(64))
    h = torch.randn(2, 3, 64)
    h[:, 1:] = 0
    types = torch.tensor([0, 0])
    states = model(h, torch.ones(2, 3, dtype=torch.bool), types,
                   [(0, 1, 0)], torch.ones(1, 3, dtype=torch.bool),
                   torch.tensor([[True, False]]))
    signal = model.source_injections[0](model.driver(h.mean(0)))[0]
    # The number of valid lag positions at t=0,1,2 is 1,2,3.
    for t, denominator in enumerate([1, 2, 3]):
        torch.testing.assert_close(states[0, 1, t], torch.tanh(signal / denominator))


def test_rsc_is_node_permutation_equivariant():
    model = decoder()
    h, visible, types, relations, active, sources = inputs()
    expected = model(h, visible, types, relations, active, sources)
    p = torch.tensor([4, 2, 0, 3, 1])
    inverse = torch.argsort(p)
    mapped = [(int(inverse[u]), int(inverse[v]), r) for u, v, r in relations]
    result = model(h[p], visible[p], types[p], mapped, active, sources[:, p])
    torch.testing.assert_close(result, expected[:, p])


def test_u_library_retains_singletons_and_legal_unique_d1_pairs():
    scores = torch.arange(54.0, requires_grad=True)
    sets = api().unlabeled_source_library(scores, K_max=2, seed=40)
    assert sets[:54] == tuple(frozenset([i]) for i in range(54))
    assert len(sets) == 70
    assert len(set(sets)) == len(sets)
    assert all(len(s) == 2 for s in sets[54:])
    assert sets[54] == frozenset([52, 53])
    assert sets == api().unlabeled_source_library(scores, K_max=2, seed=40)
    assert len(api().unlabeled_source_library(scores, K_max=1)) == 54


def test_u_library_small_support_and_general_cap():
    library = api().unlabeled_source_library(torch.arange(4.0), K_max=3)
    assert len(library) == 4 + 6 + 4
    assert all(1 <= len(s) <= 3 for s in library)
    all_sets = api().unlabeled_source_library(torch.arange(4.0))
    assert len(all_sets) == 15
    assert api().unlabeled_source_library(torch.empty(0)) == ()
    with pytest.raises(ValueError, match="K_max"):
        api().unlabeled_source_library(torch.arange(4.0), K_max=0)


def test_library_large_catalog_does_not_enumerate_power_set():
    start = time.monotonic()
    library = api().unlabeled_source_library(torch.arange(10000.0), K_max=2)
    assert len(library) == 10016
    assert time.monotonic() - start < 5


def test_random_library_chooses_cardinality_before_members():
    # Prefix quota zero exposes the prescribed two-stage random proposal law.
    counts = {k: 0 for k in range(2, 7)}
    for seed in range(1000):
        sets = api().unlabeled_source_library(torch.zeros(6), seed=seed,
                                             max_extras=1, max_prefixes=0)
        counts[len(sets[-1])] += 1
    assert all(150 < count < 250 for count in counts.values())


def test_l_library_only_equal_cardinality_replacements_and_prior_conflict():
    library = api().labeled_source_library(6, {1, 3}, K_max=2, seed=7)
    assert library[0] == frozenset({1, 3})
    assert len(library) == 15
    assert len(set(library)) == 15
    assert all(len(s) == 2 for s in library)
    assert all(s - {1, 3} for s in library[1:])
    assert api().labeled_source_library(2, {0, 1}, K_max=2) == (frozenset({0, 1}),)
    with pytest.raises(ValueError, match="prior|K_max"):
        api().labeled_source_library(5, {0, 1, 2}, K_max=2)


def test_source_set_mask_keeps_membership_discrete():
    mask = api().source_set_mask((frozenset({0}), frozenset({1, 2})), 3)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, False, False], [False, True, True]]


def test_numeric_gradient_has_only_the_shared_driver_entry_path():
    model = decoder()
    h, visible, types, relations, active, sources = inputs()
    h.requires_grad_()
    visible[4] = False
    states = model(h, visible, types, relations, active, sources)
    model.predict_nodes(states)["metric"][:, 2].square().mean().backward()
    assert h.grad[visible].abs().sum() > 0
    assert h.grad[~visible].eq(0).all()
    # Every visible entity contributes equally to the only numerical pool.
    torch.testing.assert_close(h.grad[0], h.grad[3])
    assert model.driver[0].weight.grad.abs().sum() > 0
    assert model.relation_transforms[1].weight.grad.abs().sum() > 0
    assert model.attention[0].weight.grad.abs().sum() > 0


def test_rank_library_handles_default_large_cap_without_enumeration():
    start = time.monotonic()
    sets = api().unlabeled_source_library(torch.arange(1000.0), seed=3)
    assert len(sets) == 1016
    assert [len(s) for s in sets[1000:1008]] == list(range(2, 10))
    assert time.monotonic() - start < 5


def test_empty_hypothesis_axis_is_preserved():
    model = decoder()
    h, visible, types, relations, active, _ = inputs()
    states = model(h, visible, types, relations, active, torch.empty(0, 5, dtype=torch.bool))
    assert states.shape == (0, 5, 10, 64)
    assert model.predict_edges(states, []).shape == (0, 0, 10, 2)


def test_l_library_rejects_empty_root_set():
    with pytest.raises(ValueError, match="empty|root|source"):
        api().labeled_source_library(3, set(), K_max=2)
