"""Pure canonical caching, checked against the frozen B30 Policy-r1 oracle."""

import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
import inspect
from itertools import permutations, product
from pathlib import Path
import random
import sys
import textwrap

import pytest
import torch

from vast_v3.encoders.fragments import EventRef, Fragment, FragmentEdge
from vast_v3.encoders.scope import ScopeEncoder


_SPEC = importlib.util.spec_from_file_location(
    "vast_v3.encoders.scope_b30_policy_r1_oracle",
    Path(__file__).parents[1] / "oracles" / "scope_b30_policy_r1.py")
_ORACLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ORACLE)


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cache_type():
    try:
        from vast_v3.encoders.structural_cache import StructuralCache
    except ModuleNotFoundError:
        pytest.fail("pure structural cache is not implemented")
    return StructuralCache


def sample():
    return Fragment((EventRef(0, 0, 0., .25, 0, "a"),
                     EventRef(1, 1, 1., 1.25, 1, "b"),
                     EventRef(2, 2, 2., 2.25, 0, "c")),
                    (FragmentEdge(0, 1, 0, 0), FragmentEdge(1, 2, 1, 1)),
                    "chain", "sample")


def test_warm_canonical_request_does_not_serialize(monkeypatch):
    model = ScopeEncoder(width=4, relation_types=2)
    fragment = sample()
    expected = model._canonical(fragment, 0)

    def cannot_serialize(*args, **kwargs):
        pytest.fail("warm canonical request must not enter _serialize")

    monkeypatch.setattr(model, "_serialize", cannot_serialize)
    assert model._canonical(fragment, 0) == expected


def test_original_numerical_methods_and_uncached_algorithm_are_unchanged():
    for name in ("_serialize", "_edge_orders", "_edge_slot", "forward", "_canonical"):
        current = "_canonical_uncached" if name == "_canonical" else name
        expected = ast.parse(textwrap.dedent(inspect.getsource(getattr(_ORACLE.ScopeEncoder, name))))
        actual = ast.parse(textwrap.dedent(inspect.getsource(getattr(ScopeEncoder, current))))
        actual.body[0].name = expected.body[0].name
        assert ast.dump(actual) == ast.dump(expected)


def test_frozen_oracle_is_byte_exact_b30_policy_r1_source():
    source = Path(_ORACLE.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "266159502d8d43a5ddceca855609aa4f01324fed6e4c98c7520356ab22b960ae")


def test_cache_defaults_hit_miss_clear_and_lru(cache_type):
    cache = cache_type(max_entries=2)
    assert cache.stats["max_bytes"] == 16 * 1024 * 1024
    assert cache.get((1,)) is None
    assert cache.put((1,), ((4,), ((0,),)))
    assert cache.put((2,), ((5,), ((0,),)))
    assert cache.get((1,)) == ((4,), ((0,),))
    assert cache.put((3,), ((6,), ((0,),)))
    assert cache.get((2,)) is None
    assert cache.get((1,)) == ((4,), ((0,),))
    assert cache.stats["evictions"] == 1
    assert cache.stats["entries"] == 2
    before = cache.stats
    cache.clear()
    assert cache.stats["entries"] == cache.stats["bytes"] == 0
    assert cache.stats["hits"] == before["hits"]
    assert cache.get((1,)) is None
    assert cache_type().stats["max_entries"] == 2048


@pytest.mark.parametrize("kwargs", [{"max_entries": 0}, {"max_bytes": 0}, {"max_bytes": 1}])
def test_disabled_or_tiny_cache_does_not_retain_entries(cache_type, kwargs):
    cache = cache_type(**kwargs)
    assert not cache.put((1,), (2,))
    assert cache.get((1,)) is None
    assert cache.stats["entries"] == cache.stats["bytes"] == 0


@pytest.mark.parametrize("kwargs", [{"max_entries": -1}, {"max_bytes": -1},
                                    {"max_bytes": 1.5}, {"max_entries": True}])
def test_capacity_validation(cache_type, kwargs):
    with pytest.raises(ValueError, match="nonnegative integers"):
        cache_type(**kwargs)


def recursive_size(value):
    return sys.getsizeof(value) + (sum(recursive_size(item) for item in value)
                                  if type(value) is tuple else 0)


def test_byte_budget_includes_payload_entry_and_actual_ordered_dict_growth(cache_type):
    cache = cache_type(max_bytes=2300, max_entries=500)
    for index in range(120):
        assert cache.put((index,), ((index,) * 4, ((0, 1),)))
        entries = cache._entries
        required = sys.getsizeof(entries) + sum(
            recursive_size(key) + recursive_size(entry) for key, entry in entries.items())
        assert cache.stats["bytes"] >= required
        assert cache.stats["bytes"] <= cache.max_bytes
        assert 0 < cache.stats["entries"] <= cache.max_entries
    assert cache.stats["evictions"] > 0


def test_oversize_bypass_preserves_hot_entries_and_replacement_accounting(cache_type):
    cache = cache_type(max_bytes=1600)
    assert cache.put((1,), (3,))
    original = cache.stats["bytes"]
    assert not cache.put((2,), tuple(range(1000)))
    assert cache.get((1,)) == (3,)
    assert cache.stats["bytes"] == original
    assert not cache.put((1,), tuple(range(1000)))
    assert cache.get((1,)) == (3,)
    assert cache.put((1,), (4, 5))
    assert cache.get((1,)) == (4, 5)
    assert cache.stats["entries"] == 1
    assert cache.stats["bytes"] > original


@pytest.mark.parametrize("bad", [[1], (1, [2]), 1., "key", True, torch.zeros(1)])
def test_cache_rejects_any_non_int_tuple_reference(cache_type, bad):
    cache = cache_type()
    with pytest.raises(TypeError, match="int.*tuple"):
        cache.put((1,), bad)
    with pytest.raises(TypeError, match="int.*tuple"):
        cache.put(bad, (1,))
    with pytest.raises(TypeError, match="int.*tuple"):
        cache.get(bad)
    assert cache.stats["entries"] == 0


def assert_immutable(value):
    assert type(value) in (int, tuple)
    if type(value) is tuple:
        for item in value:
            assert_immutable(item)


def test_cache_is_instance_local_deepcopy_independent_and_tensor_free():
    model = ScopeEncoder(width=4)
    model._canonical(sample(), 0)
    other = ScopeEncoder(width=4)
    assert other.structural_cache.stats["entries"] == 0
    cloned = copy.deepcopy(model)
    assert cloned.structural_cache is not model.structural_cache
    assert cloned.structural_cache._entries is not model.structural_cache._entries
    for key, entry in model.structural_cache._entries.items():
        assert_immutable(key)
        assert_immutable(entry)
    cloned.structural_cache.clear()
    assert model.structural_cache.stats["entries"] == 1


def test_disabled_encoder_recomputes_canonical(monkeypatch):
    model = ScopeEncoder(width=4, structural_cache_max_entries=0)
    calls = []
    original = model._serialize

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_serialize", record)
    first = model._canonical(sample(), 0)
    second = model._canonical(sample(), 0)
    assert first == second
    assert len(calls) == 12
    assert model.structural_cache.stats["entries"] == 0


def test_encoder_eviction_recomputes_exact_ordered_mappings():
    model = ScopeEncoder(width=4, structural_cache_max_entries=1)
    fragment = sample()
    expected = model._canonical(fragment, None)
    model._canonical(fragment, 0)
    assert model._canonical(fragment, None) == expected
    assert model.structural_cache.stats["misses"] == 3
    assert model.structural_cache.stats["evictions"] == 2


def test_schema_version_is_part_of_key(monkeypatch):
    from vast_v3.encoders import scope
    model = ScopeEncoder(width=4)
    first = model._canonical_key(sample(), 0)
    monkeypatch.setattr(scope, "_CANONICAL_CACHE_SCHEMA", scope._CANONICAL_CACHE_SCHEMA + 1)
    assert model._canonical_key(sample(), 0) != first


def test_canonical_key_tracks_every_structural_input_in_original_positions():
    model = ScopeEncoder(width=4, relation_types=3)
    fragment = sample()
    key = model._canonical_key(fragment, 0)
    assert_immutable(key)
    variants = [
        replace(fragment, events=(replace(fragment.events[0], category=1),) + fragment.events[1:]),
        replace(fragment, events=fragment.events + (EventRef(3, 3, 3., 3.25, 0, "d"),)),
        replace(fragment, events=(replace(fragment.events[0], entity=1),) + fragment.events[1:]),
        replace(fragment, events=(replace(fragment.events[0], end=1.),) + fragment.events[1:]),
        replace(fragment, edges=fragment.edges + (fragment.edges[0],)),
        replace(fragment, edges=tuple(reversed(fragment.edges))),
        replace(fragment, edges=(replace(fragment.edges[0], source=2), fragment.edges[1])),
        replace(fragment, edges=(replace(fragment.edges[0], target=2), fragment.edges[1])),
        replace(fragment, edges=(replace(fragment.edges[0], relation=1), fragment.edges[1])),
        replace(fragment, edges=(replace(fragment.edges[0], response=True), fragment.edges[1])),
    ]
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=3)
    for changed in variants:
        assert model._canonical_key(changed, 0) != key
        assert model._canonical(changed, 0) == oracle._canonical(changed, 0)
    assert model._canonical_key(fragment, None) != key
    assert model._canonical_key(fragment, 1) != key
    assert model._canonical_key(fragment, 0, (0, 1)) != key
    assert model._canonical_key(fragment, 0, (1, 0)) != model._canonical_key(fragment, 0, (0, 1))
    assert model._canonical_key(fragment, 0, branch=frozenset((0,))) != key
    assert ScopeEncoder(width=4, entity_types=3)._canonical_key(fragment, 0) != key
    assert ScopeEncoder(width=4, relation_types=2)._canonical_key(fragment, 0) != key


def permute_fragment(fragment, order):
    inverse = {old: new for new, old in enumerate(order)}
    return replace(fragment, events=tuple(fragment.events[i] for i in order),
                   edges=tuple(replace(edge, source=inverse[edge.source], target=inverse[edge.target])
                               for edge in fragment.edges))


def test_small_graphs_all_permutations_and_marks_match_ordered_oracle_mappings():
    model = ScopeEncoder(width=4, relation_types=2)
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=2)
    for n in range(1, 5):
        for shared, overlapping, response in product((False, True), repeat=3):
            events = tuple(EventRef(0 if shared else i, i, 0. if overlapping else float(i),
                                    4. if overlapping else i + .25, i % 2, str(i)) for i in range(n))
            edges = tuple(FragmentEdge(i, min(i + 1, n - 1), i % 2, i, response)
                          for i in range(min(3, n)))
            fragment = Fragment(events, edges, "chain", "small")
            for order in permutations(range(n)):
                moved = permute_fragment(fragment, order)
                for candidate, endpoint, branch in ((None, None, frozenset()),
                                                     (0, (0, n - 1), frozenset((0, n - 1)))):
                    expected = oracle._canonical(moved, candidate, endpoint, branch)
                    assert model._canonical(moved, candidate, endpoint, branch) == expected
                    assert model._canonical(moved, candidate, endpoint, branch) == expected


def test_response_uses_int_not_bool_and_parallel_edges_preserve_multiplicity():
    model = ScopeEncoder(width=4, relation_types=3)
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=3)
    fragment = sample()
    variants = [replace(fragment, edges=(replace(fragment.edges[0], response=value),))
                for value in (False, True, 2, -1, 1.9)]
    assert model._canonical_key(variants[1], None) != model._canonical_key(variants[2], None)
    assert model._canonical_key(variants[1], None) == model._canonical_key(variants[4], None)
    variants.append(replace(fragment, edges=(fragment.edges[0], fragment.edges[0])))
    for changed in variants:
        assert model._canonical(changed, None) == oracle._canonical(changed, None)


def test_float_order_is_strict_and_not_quantized():
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    fragment = sample()
    import math
    before = replace(fragment, events=(replace(fragment.events[0], end=math.nextafter(1., 0.)),)
                     + fragment.events[1:])
    touching = replace(fragment, events=(replace(fragment.events[0], end=1.),) + fragment.events[1:])
    overlapping = replace(fragment, events=(replace(fragment.events[0], end=1.1),) + fragment.events[1:])
    assert model._canonical_key(before, None) != model._canonical_key(touching, None)
    assert model._canonical_key(touching, None) == model._canonical_key(overlapping, None)
    for changed in (before, touching, overlapping):
        assert model._canonical(changed, None) == oracle._canonical(changed, None)


def test_padding_endpoint_and_branch_sentinel_marks_match_oracle():
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    fragment = sample()
    for endpoint, branch in ((None, frozenset()), ((-1, 0), frozenset()),
                             (None, frozenset((-1,))), ((-1, 0), frozenset((-1,)))):
        assert model._canonical(fragment, None, endpoint, branch) == oracle._canonical(
            fragment, None, endpoint, branch)


def test_mutable_nested_inventory_rebuilds_value_key():
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    original = sample()
    events, edges = list(original.events), list(original.edges)
    fragment = replace(original, events=events, edges=edges)
    old_key = model._canonical_key(fragment, 0)
    model._canonical(fragment, 0)
    events[0] = replace(events[0], category=1)
    edges[0] = replace(edges[0], response=True)
    assert model._canonical_key(fragment, 0) != old_key
    assert model._canonical(fragment, 0) == oracle._canonical(fragment, 0)
    assert model.structural_cache.stats["misses"] == 2


@pytest.mark.parametrize("warm", [False, True])
def test_fractional_category_canonical_preserves_old_cold_and_hot_semantics(warm):
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    fragment = sample()
    changed = replace(fragment, events=(replace(fragment.events[0], category=.5),) + fragment.events[1:])
    if warm:
        model._canonical(fragment, 0)
    expected = oracle._canonical(changed, 0)
    assert model._canonical(changed, 0) == expected


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("field,value,error", [("source", .5, KeyError),
                                               ("target", 1.5, KeyError),
                                               ("relation", .5, TypeError)])
def test_noninteger_edge_indices_preserve_original_exceptions_on_cold_and_hot_requests(warm, field, value, error):
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    fragment = sample()
    changed = replace(fragment, edges=(replace(fragment.edges[0], **{field: value}), fragment.edges[1]))
    if warm:
        model._canonical(fragment, 0)
    with pytest.raises(error) as expected:
        oracle._canonical(changed, 0)
    with pytest.raises(error) as actual:
        model._canonical(changed, 0)
    assert actual.value.args == expected.value.args


@pytest.mark.parametrize("field", ["entity_types", "relation_types"])
def test_noninteger_schema_indices_do_not_hit_integer_schema_entries(field):
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    fragment = sample()
    model._canonical(fragment, 0)
    setattr(model, field, float(getattr(model, field)))
    setattr(oracle, field, float(getattr(oracle, field)))
    with pytest.raises(TypeError) as expected:
        oracle._canonical(fragment, 0)
    with pytest.raises(TypeError) as actual:
        model._canonical(fragment, 0)
    assert actual.value.args == expected.value.args


@pytest.mark.parametrize("field", ["category", "source", "target", "relation", "entity_types", "relation_types"])
def test_numpy_integer_structural_indices_conservatively_bypass(field):
    import numpy as np
    model = ScopeEncoder(width=4)
    oracle = _ORACLE.ScopeEncoder(width=4)
    fragment = sample()
    expected = model._canonical(fragment, 0)
    before = model.structural_cache.stats
    if field in ("entity_types", "relation_types"):
        setattr(model, field, np.int64(getattr(model, field)))
        setattr(oracle, field, np.int64(getattr(oracle, field)))
    elif field == "category":
        fragment = replace(fragment, events=(replace(fragment.events[0], category=np.int64(0)),)
                           + fragment.events[1:])
    else:
        edge = fragment.edges[0]
        fragment = replace(fragment, edges=(replace(edge, **{field: np.int64(getattr(edge, field))}),)
                           + fragment.edges[1:])
    assert model._canonical_key(fragment, 0) is None
    assert model._canonical(fragment, 0) == oracle._canonical(fragment, 0) == expected
    assert model.structural_cache.stats == before


def inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(123)
    return (torch.randn(6, 6, 4, generator=generator, dtype=dtype),
            torch.randn(6, 6, 4, generator=generator, dtype=dtype),
            torch.randn(4, 6, 4, generator=generator, dtype=dtype),
            torch.ones(6, 6, 3, dtype=dtype),
            torch.rand(6, 6, 3, generator=generator, dtype=dtype),
            torch.ones(4, 6, dtype=torch.bool))


@pytest.mark.parametrize("warm", [False, True])
def test_fractional_category_full_forward_preserves_old_cold_and_hot_semantics(warm):
    model = ScopeEncoder(width=4, relation_types=2)
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=2)
    oracle.load_state_dict(model.state_dict())
    fragment = sample()
    changed = replace(fragment, events=(replace(fragment.events[0], category=.5),) + fragment.events[1:])
    data = inputs()
    if warm:
        model(fragment, *data, candidate=0)
    actual, actual_role = model(changed, *data, candidate=0)
    expected, expected_role = oracle(changed, *data, candidate=0)
    assert torch.equal(actual_role, expected_role)
    assert torch.equal(actual, expected)


def test_omitted_metadata_changes_hit_but_forward_rebinds_current_data(monkeypatch):
    model = ScopeEncoder(width=4, relation_types=2)
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=2)
    oracle.load_state_dict(model.state_dict())
    fragment = sample()
    moved = replace(fragment, events=tuple(replace(event, entity=event.entity + 3, bin=event.bin + 1,
                                                  start=100. + 2 * event.start, end=100. + 2 * event.end,
                                                  provenance="renamed-" + event.provenance)
                                          for event in fragment.events),
                    edges=tuple(replace(edge, owner=edge.owner + 2) for edge in fragment.edges),
                    family="branch", provenance="different")
    assert model._canonical_key(fragment, 0) == model._canonical_key(moved, 3)
    data = inputs()
    before = model(fragment, *data, candidate=0)

    def cannot_serialize(*args, **kwargs):
        pytest.fail("omitted numerical metadata must reuse only the canonical structure")

    monkeypatch.setattr(model, "_serialize", cannot_serialize)
    after = model(moved, *data, candidate=3)
    expected = oracle(moved, *data, candidate=3)
    assert torch.equal(after[0], expected[0])
    assert torch.equal(after[1], expected[1])
    assert torch.equal(before[1], after[1])
    assert not torch.equal(before[0], after[0])
    changed = list(data)
    changed[2] = data[2] + 3
    changed_output = model(moved, *changed, candidate=3)[0]
    assert torch.equal(changed_output, oracle(moved, *changed, candidate=3)[0])
    assert not torch.equal(changed_output, after[0])


@pytest.mark.parametrize("field", ["owner", "delay"])
def test_owner_and_delay_independently_rebind_on_warm_forward(field, monkeypatch):
    model = ScopeEncoder(width=4, relation_types=2)
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=2)
    oracle.load_state_dict(model.state_dict())
    fragment = sample()
    if field == "owner":
        moved = replace(fragment, edges=tuple(replace(edge, owner=edge.owner + 2) for edge in fragment.edges))
    else:
        moved = replace(fragment, events=tuple(replace(event, start=3 * event.start, end=3 * event.end)
                                               for event in fragment.events))
    data = inputs()
    before = model(fragment, *data, candidate=0)

    def cannot_serialize(*args, **kwargs):
        pytest.fail("changing a numerical field must leave the structure cache hot")

    monkeypatch.setattr(model, "_serialize", cannot_serialize)
    actual = model(moved, *data, candidate=0)
    expected = oracle(moved, *data, candidate=0)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert not torch.equal(actual[0], before[0])


def test_encoder_construction_uses_identical_rng_and_parameters_to_oracle():
    torch_state = torch.random.get_rng_state()
    model = ScopeEncoder(width=4)
    after_current = torch.random.get_rng_state()
    torch.random.set_rng_state(torch_state)
    oracle = _ORACLE.ScopeEncoder(width=4)
    assert torch.equal(after_current, torch.random.get_rng_state())
    assert all(torch.equal(value, oracle.state_dict()[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["chain", "symmetric", "parallel", "shared", "overlap"])
def test_exact_forward_input_and_all_parameter_gradients_two_backwards_update_reload(dtype, kind):
    fragment = sample()
    candidate = 0
    if kind == "symmetric":
        fragment = replace(fragment, events=(fragment.events[0],
                                             replace(fragment.events[1], category=0, start=0., end=.25),
                                             fragment.events[2]),
                           edges=(FragmentEdge(0, 2, 0, 0), FragmentEdge(1, 2, 0, 1)))
        candidate = None
    elif kind == "parallel":
        fragment = replace(fragment, edges=(fragment.edges[0], replace(fragment.edges[0], owner=1)))
    elif kind == "shared":
        fragment = replace(fragment, events=fragment.events[:2] + (replace(fragment.events[2], entity=0),))
    elif kind == "overlap":
        fragment = replace(fragment, events=tuple(replace(event, start=0., end=3.) for event in fragment.events))
    model = ScopeEncoder(width=4, relation_types=2).to(dtype=dtype)
    oracle = _ORACLE.ScopeEncoder(width=4, relation_types=2).to(dtype=dtype)
    initial = copy.deepcopy(model.state_dict())
    oracle.load_state_dict(initial)
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in oracle.state_dict().items()}
    base = inputs(dtype)
    for step in range(4):
        if step == 2:
            with torch.no_grad():
                for current, reference in zip(model.parameters(), oracle.parameters()):
                    if current.grad is not None:
                        current.add_(current.grad, alpha=-.01)
                        reference.add_(reference.grad, alpha=-.01)
            model.eval()
            oracle.eval()
        elif step == 3:
            model.load_state_dict(initial)
            oracle.load_state_dict(initial)
            model.train()
            oracle.train()
        current_inputs = [value.clone().requires_grad_(index < 5) for index, value in enumerate(base)]
        reference_inputs = [value.clone().requires_grad_(index < 5) for index, value in enumerate(base)]
        model.zero_grad(set_to_none=True)
        oracle.zero_grad(set_to_none=True)
        actual = model(fragment, *current_inputs, candidate=candidate)
        expected = oracle(fragment, *reference_inputs, candidate=candidate)
        assert all(torch.equal(left, right) for left, right in zip(actual, expected))
        (actual[0][1] + actual[1].square().sum() * .01).backward()
        (expected[0][1] + expected[1].square().sum() * .01).backward()
        for current, reference in zip(current_inputs[:5], reference_inputs[:5]):
            assert (current.grad is None) == (reference.grad is None)
            if current.grad is not None:
                assert torch.equal(current.grad, reference.grad)
        for (name, current), (reference_name, reference) in zip(model.named_parameters(), oracle.named_parameters()):
            assert name == reference_name
            assert (current.grad is None) == (reference.grad is None), name
            if current.grad is not None:
                assert torch.equal(current.grad, reference.grad), name
    assert model.structural_cache.stats["hits"] > 0
    for key, entry in model.structural_cache._entries.items():
        assert_immutable(key)
        assert_immutable(entry)


def test_cache_does_not_change_rng_checkpoint_or_dtype_lifecycle():
    model = ScopeEncoder(width=4)
    fragment = sample()
    before = copy.deepcopy(model.state_dict())
    torch_state, python_state = torch.random.get_rng_state(), random.getstate()
    expected = model._canonical(fragment, 0)
    for mode in (True, False):
        model.train(mode)
        model.to(dtype=torch.float64)
        model.load_state_dict(before)
        assert model._canonical(fragment, 0) == expected
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert set(model.state_dict()) == set(before)
    assert not model._buffers
    assert "structural_cache" not in model._modules
    assert model.structural_cache.stats["entries"] == 1
    assert model.structural_cache.stats["hits"] == 2
