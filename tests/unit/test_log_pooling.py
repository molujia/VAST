"""Ordered log pooling preserves the legacy native-time, weighted contract."""
from collections import Counter
import importlib
import math
import random

import pytest
import torch

from vast_v3.data.preprocessing import Preprocessor
from vast_v3.data.schema import Entity, LogRecord, ObservableCase
from vast_v3.encoders.frontends import NativeFrontend


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(373)
    yield
    torch.set_num_threads(previous)


def pooling():
    name = 'vast_v3.encoders.log_pooling'
    assert importlib.util.find_spec(name) is not None, 'ordered log pooling is not implemented'
    return importlib.import_module(name).pool_encoded_logs


def legacy(encoded, owners, times, frequencies, *, entity_count, bins,
           reference_start, delta, radius_steps):
    zero = encoded.new_zeros(encoded.shape[-1])
    states, masks, recovery, context = [], [], [], []
    for owner in range(entity_count):
        row, mask, recrow, ctx = [], [], [], []
        for b in range(bins):
            observed = [j for j, (v, t) in enumerate(zip(owners, times))
                        if v == owner and math.floor((t-reference_start)/delta) == b]
            query = reference_start+(b+.5)*delta
            recovered = [j for j, (v, t) in enumerate(zip(owners, times))
                         if v == owner and abs(t-query) <= radius_steps*delta]
            row.append(sum((encoded[j]*frequencies[j] for j in observed), zero)
                       / sum(frequencies[j] for j in observed) if observed else zero)
            recrow.append(sum((encoded[j]*frequencies[j] for j in recovered), zero)
                          / sum(frequencies[j] for j in recovered) if recovered else zero)
            mask.append(bool(observed)); ctx.append(bool(recovered))
        states.append(torch.stack(row)); masks.append(mask)
        recovery.append(torch.stack(recrow)); context.append(ctx)
    return (torch.stack(states), torch.tensor(masks, device=encoded.device),
            torch.stack(recovery), torch.tensor(context, device=encoded.device))


def graph_counts(*outputs):
    seen, counts, stack = set(), Counter(), [x.grad_fn for x in outputs]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn); counts[type(fn).__name__] += 1
        stack.extend(parent for parent, _ in fn.next_functions)
    return counts


def fixture_records(size, origin=1_700_000_000.):
    rng = random.Random(887)
    owners = [rng.randrange(3) for _ in range(size)]
    times = [origin+rng.randrange(64)/4 for _ in range(size)]
    frequencies = [rng.randrange(1, 21) for _ in range(size)]
    return owners, times, frequencies


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('radius', [0., 1.25, 2., 50.])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_weighted_native_pooling_matches_legacy_values_and_gradients(dtype, radius, device):
    fn = pooling()
    owners, times, frequencies = fixture_records(257)
    kwargs = dict(entity_count=4, bins=8, reference_start=1_700_000_000., delta=2., radius_steps=radius)
    data = torch.randn(257, 64, dtype=dtype, device=device)
    left, right = data.clone().requires_grad_(), data.clone().requires_grad_()
    expected = legacy(left, owners, times, frequencies, **kwargs)
    actual = fn(right, owners, times, frequencies, **kwargs)
    atol, rtol = ((2e-6, 2e-5) if dtype == torch.float32 else (1e-12, 1e-11))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        assert a.device == b.device and a.dtype == b.dtype
    weight_a, weight_b = torch.randn_like(actual[0]), torch.randn_like(actual[2])
    loss_a = (expected[0]*weight_a).sum() + (expected[2]*weight_b).sum()
    loss_b = (actual[0]*weight_a).sum() + (actual[2]*weight_b).sum()
    loss_a.backward(); loss_b.backward()
    torch.testing.assert_close(left.grad, right.grad, atol=atol, rtol=rtol)


def test_closed_float_boundaries_match_original_predicate_and_empty_owners():
    fn = pooling()
    origin, delta = 1_700_000_000., .1
    center = origin+(3+.5)*delta
    radius = 1.25*delta
    times = [center-radius, math.nextafter(center-radius, -math.inf),
             center+radius, math.nextafter(center+radius, math.inf), center]
    owners, frequencies = [0]*len(times), [1, 2, 10_000_001, 3, 1]
    encoded = torch.randn(len(times), 64)
    kwargs = dict(entity_count=2, bins=8, reference_start=origin, delta=delta, radius_steps=1.25)
    expected = legacy(encoded, owners, times, frequencies, **kwargs)
    actual = fn(encoded, owners, times, frequencies, **kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert not actual[1][1].any() and not actual[3][1].any()


def test_empty_logs_return_zero_states_and_false_masks():
    output = pooling()(torch.empty(0, 64, dtype=torch.float64), [], [], [],
                       entity_count=3, bins=8, reference_start=0., delta=1., radius_steps=2.)
    assert output[0].shape == output[2].shape == (3, 8, 64)
    assert output[0].dtype == output[2].dtype == torch.float64
    assert output[1].dtype == output[3].dtype == torch.bool
    assert all(not value.count_nonzero() for value in output)


def test_all_unassociated_logs_leave_mlp_gradient_none_and_adam_momentum_untouched():
    frontend = NativeFrontend(Preprocessor({}, 1., (), 0), ('service',), ('call',), ('cpu',),
                              text_encoder=lambda texts: torch.ones(len(texts), 384),
                              log_recovery_radius_steps=0.)
    entities = (Entity('a', 'service'),)
    optimizer = torch.optim.Adam(frontend.parameters(), lr=.001)
    warmup = ObservableCase('warmup', entities, -1., 0., 1.,
                            logs=(LogRecord('a', 0., 'observed warmup'),))
    output = frontend(warmup)
    output.local.square().sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in frontend.log_mlp.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in frontend.log_mlp.parameters()]
    timestamp = math.nextafter(1., -math.inf)
    assert timestamp < 1. and math.floor(timestamp-(-1.)) == 2
    case = ObservableCase('rounding', entities, -1., 0., 1.,
                          logs=(LogRecord('a', timestamp, 'visible but unassociated'),))
    output = frontend(case)
    assert not output.availability.any() and not output.log_context_mask.any()
    (output.h.square().sum()+output.local.square().sum()+output.recovery_local.square().sum()).backward()
    assert all(parameter.grad is None for parameter in frontend.log_mlp.parameters())
    optimizer.step()
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(frontend.log_mlp.parameters(), before))


@pytest.mark.parametrize('branch', ['observed', 'recovery'])
def test_empty_pooling_branch_does_not_connect_mlp_through_other_branch(branch):
    radius = 2. if branch == 'observed' else 0.
    timestamp = math.nextafter(1., -math.inf) if branch == 'observed' else 0.
    frontend = NativeFrontend(Preprocessor({}, 1., (), 0), ('service',), ('call',), ('cpu',),
                              text_encoder=lambda texts: torch.ones(len(texts), 384),
                              log_recovery_radius_steps=radius)
    case = ObservableCase('one-empty-branch', (Entity('a', 'service'),), -1., 0., 1.,
                          logs=(LogRecord('a', timestamp, 'visible'),))
    output = frontend(case)
    if branch == 'observed':
        assert not output.availability.any() and output.log_context_mask.any()
        loss = output.h.square().sum()+output.local.square().sum()
    else:
        assert output.availability.any() and not output.log_context_mask.any()
        loss = output.recovery_local.square().sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in frontend.log_mlp.parameters())


def test_pooling_graph_does_not_grow_per_record():
    fn = pooling()
    sizes = []
    for count in (16, 512):
        owners, times, frequencies = fixture_records(count, origin=0.)
        output = fn(torch.randn(count, 64, requires_grad=True), owners, times, frequencies,
                    entity_count=4, bins=8, reference_start=0., delta=2., radius_steps=2.)
        nodes = graph_counts(output[0], output[2])
        assert nodes['SelectBackward0'] == 0
        assert nodes['IndexSelectBackward0'] == 0, 'forward gathers must have an ordered transpose backward'
        sizes.append(sum(nodes.values()))
    assert max(sizes) < 32 and sizes[0] == sizes[1]


def test_frontend_graph_does_not_grow_per_log_record():
    frontend = NativeFrontend(Preprocessor({}, 1., (), 0), ('service',), ('call',), ('cpu',),
                              text_encoder=lambda texts: torch.ones(len(texts), 384))
    sizes = []
    for count in (16, 128):
        case = ObservableCase('pool', (Entity('a', 'service'),), 0., 2., 4.,
                              logs=tuple(LogRecord('a', (i % 16)/4, 'visible text', 1+i % 3)
                                         for i in range(count)))
        output = frontend(case)
        sizes.append(sum(graph_counts(output.h, output.local, output.recovery_local).values()))
    assert sizes[1]-sizes[0] < 32, f'per-record autograd graph growth: {sizes}'


def test_ordered_pooling_gradcheck_and_gradgradcheck():
    fn = pooling()
    owners, times, frequencies = [0, 1, 0, 0], [.25, 1.5, 2.5, 2.5], [1, 7, 3, 2]
    def operation(values):
        result = fn(values, owners, times, frequencies, entity_count=3, bins=4,
                    reference_start=0., delta=1., radius_steps=1.25)
        return result[0], result[2]
    values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(operation, (values,))
    assert torch.autograd.gradgradcheck(operation, (values,))


def test_frontend_outputs_and_trainable_parameter_gradients_match_legacy(monkeypatch):
    import vast_v3.encoders.frontends as module
    fn = pooling()
    frontend = NativeFrontend(Preprocessor({}, 1., (), 0), ('service',), ('call',), ('cpu',),
                              text_encoder=lambda texts: torch.tensor([[len(text)/20]*384 for text in texts]))
    case = ObservableCase('weighted', (Entity('a', 'service'), Entity('b', 'service')), 0., 4., 8.,
                          logs=tuple(LogRecord('a' if i % 3 else 'b', (i % 32)/4,
                                               'native '+('x'*(i % 7)), 1+i % 9) for i in range(127)))
    monkeypatch.setattr(module, 'pool_encoded_logs', legacy)
    expected = frontend(case)
    monkeypatch.setattr(module, 'pool_encoded_logs', fn)
    actual = frontend(case)
    for name in ('h', 'local', 'recovery_local', 'availability', 'log_context_mask'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=2e-6, rtol=2e-5)
    parameters = tuple(frontend.log_mlp.parameters()) + tuple(frontend.local_projection.parameters())
    def loss(output):
        return output.h.square().sum()+output.local.square().sum()+.71*output.recovery_local.square().sum()
    gradients_a = torch.autograd.grad(loss(expected), parameters)
    gradients_b = torch.autograd.grad(loss(actual), parameters)
    for a, b in zip(gradients_a, gradients_b):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_pooling_replays_forward_and_backward_bitwise_without_global_policy_change():
    fn = pooling()
    owners, times, frequencies = fixture_records(4096, origin=0.)
    original = torch.randn(4096, 64, device='cuda')
    policy = torch.are_deterministic_algorithms_enabled()
    previous = None
    for _ in range(3):
        values = original.clone().requires_grad_()
        result = fn(values, owners, times, frequencies, entity_count=4, bins=8,
                    reference_start=0., delta=2., radius_steps=2.)
        loss = result[0].square().sum()+result[2].square().sum()
        loss.backward()
        actual = result[0].detach(), result[2].detach(), values.grad
        if previous is not None:
            assert all(torch.equal(a, b) for a, b in zip(actual, previous))
        previous = actual
        assert torch.are_deterministic_algorithms_enabled() == policy
