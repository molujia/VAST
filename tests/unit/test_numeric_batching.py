"""Length-bucket execution against an immutable, pre-change scalar oracle."""
from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from vast_v3.data.preprocessing import Preprocessor, Scale
from vast_v3.data.schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation
from vast_v3.encoders.frontends import NativeFrontend


ATOL = 2e-5
RTOL = 2e-4
ORACLE_PATH = Path(__file__).parents[1] / 'oracles' / 'frontends_scalar_v1.py'
ORACLE_SHA256 = 'fd5048aaee589f3d81969afc92bbf37f245206b1764db70cb023ef99d537d37f'
APPROXIMATE_FIELDS = {'h', 'local', 'edge_states', 'recovery_local', 'edge_recovery_states'}


def scalar_class():
    # Package-qualified shim resolves the unchanged relative imports in the oracle.
    name = 'vast_v3.encoders._frozen_frontends_scalar_v1_test'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ORACLE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].NativeFrontend


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def preprocessing():
    return Preprocessor({(category, channel): Scale(.25, 1.75, 20)
                         for category in ('service', 'service->service')
                         for channel in ('cpu', 'latency', 'memory')}, 2., ('fit-only',), 1)


def make_model(kind=NativeFrontend, device='cpu', **kwargs):
    return kind(preprocessing(), ('service',), ('call',), ('cpu', 'latency', 'memory', 'unfitted'),
                text_encoder=lambda texts: torch.arange(384).float().repeat(len(texts), 1) / 384,
                **kwargs).to(device)


def series(length, *, modality='metric', owner='a', channel='cpu', offset=0.):
    times = tuple(7.5 * j / max(length, 1) for j in range(length))
    return NumericSeries(modality, owner, channel, times,
                         tuple(((j * 3) % 11) / 7. + offset for j in range(length)))


def make_case(numeric=(), *, relations=True, logs=True):
    return ObservableCase('numeric-batching', (Entity('a', 'service'), Entity('b', 'service', False)),
                          0., 4., 8., tuple(numeric),
                          (LogRecord('a', 1., 'a request_id=xyz failed', 2),
                           LogRecord('b', 5., 'b pending', 1)) if logs else (),
                          (Relation('ab', 'a', 'b', 'call', 0., 8.),
                           Relation('ba', 'b', 'a', 'call', 2., 7.)) if relations else ())


def error_summary(actual, expected):
    if not actual.numel():
        return (0., 0., 0.)
    difference = (actual - expected).abs()
    return (difference.max().item(),
            (difference / expected.abs().clamp_min(1e-12)).max().item(),
            (difference / (ATOL + RTOL * expected.abs())).max().item())


def assert_equivalent(case, *, device='cpu', batch_size=32, max_samples=32768,
                      mode='length-bucket-v1'):
    torch.manual_seed(7301)
    expected_model = make_model(scalar_class(), device)
    actual_model = make_model(device=device, numeric_execution=mode,
                              numeric_batch_size=batch_size, numeric_max_batch_samples=max_samples)
    actual_model.load_state_dict(expected_model.state_dict())
    frozen_case = copy.deepcopy(case)
    frozen_preprocessor = copy.deepcopy(actual_model.preprocessing.to_dict())
    frozen_parameters = {name: value.detach().clone() for name, value in actual_model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    atol, rtol = (0., 0.) if mode == 'scalar-v1' else (ATOL, RTOL)
    expected = expected_model(case)
    actual = actual_model(case)
    output_errors = []
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, torch.Tensor):
            assert a.dtype == b.dtype and a.device == b.device and a.shape == b.shape, field.name
            if field.name in APPROXIMATE_FIELDS:
                output_errors.append(error_summary(a, b))
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=field.name)
            else:
                assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name

    # Every differentiable frontend branch contributes to the same scalar loss.
    def loss(output):
        return sum((index + 1) * getattr(output, name).square().mean()
                   for index, name in enumerate(sorted(APPROXIMATE_FIELDS))
                   if getattr(output, name).numel())
    a_loss, b_loss = loss(actual), loss(expected)
    torch.testing.assert_close(a_loss, b_loss, atol=atol, rtol=rtol)
    a_loss.backward()
    b_loss.backward()
    gradient_errors = []
    expected_parameters = dict(expected_model.named_parameters())
    for name, parameter in actual_model.named_parameters():
        other = expected_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            gradient_errors.append(error_summary(parameter.grad, other.grad))
            torch.testing.assert_close(parameter.grad, other.grad, atol=atol, rtol=rtol, msg=name)
    assert case == frozen_case
    assert actual_model.preprocessing.to_dict() == frozen_preprocessor
    for name, value in actual_model.state_dict().items():
        assert torch.equal(value, frozen_parameters[name]), name
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py_rng
    now_np = np.random.get_state()
    assert now_np[0] == np_rng[0] and np.array_equal(now_np[1], np_rng[1]) and now_np[2:] == np_rng[2:]
    if cuda_rng is not None:
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    maxima = lambda errors: tuple(max((error[i] for error in errors), default=0.) for i in range(3))
    print(f'\n{device} output(max_abs,max_rel,max_tolerance_ratio)={maxima(output_errors)} '
          f'gradient={maxima(gradient_errors)} loss={error_summary(a_loss, b_loss)}')
    return actual, actual_model


def cnn_shapes(model, case):
    shapes = {'metric': [], 'trace': []}
    handles = [getattr(model, name + '_cnn').register_forward_pre_hook(
        lambda _module, args, name=name: shapes[name].append(tuple(args[0].shape)))
               for name in shapes]
    try:
        model(case)
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def test_oracle_is_exact_frozen_source_and_default_path_is_bitwise_equal():
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256
    torch.manual_seed(31)
    expected = make_model(scalar_class())
    actual = make_model()
    actual.load_state_dict(expected.state_dict())
    case = make_case((series(3), series(2, modality='trace', owner='ab', channel='latency')))
    a, b = actual(case), expected(case)
    for field in fields(a):
        av, bv = getattr(a, field.name), getattr(b, field.name)
        assert torch.equal(av, bv) if isinstance(av, torch.Tensor) else av == bv
    assert actual.numeric_execution == 'scalar-v1'
    assert actual.numeric_batch_size == 32
    assert actual.numeric_max_batch_samples == 32768


def test_same_length_series_reduce_calls_without_mixing_modalities():
    case = make_case([series(3, offset=i / 10) for i in range(35)] +
                     [series(3, modality='trace', owner='ab', channel='latency') for _ in range(4)])
    baseline = cnn_shapes(make_model(scalar_class()), case)
    batched = cnn_shapes(make_model(numeric_execution='length-bucket-v1'), case)
    assert len(baseline['metric']) == 35 and len(baseline['trace']) == 4
    assert batched == {'metric': [(32, 3, 3), (3, 3, 3)], 'trace': [(4, 3, 3)]}
    assert_equivalent(case)


def test_constructor_modes_do_not_change_initialization_or_rng():
    snapshots = []
    for kind, kwargs in ((scalar_class(), {}), (NativeFrontend, {}),
                         (NativeFrontend, {'numeric_execution': 'scalar-v1'}),
                         (NativeFrontend, {'numeric_execution': 'length-bucket-v1'})):
        torch.manual_seed(8172)
        model = make_model(kind, **kwargs)
        snapshots.append((copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()))
    for state, rng in snapshots[1:]:
        assert state.keys() == snapshots[0][0].keys()
        assert all(torch.equal(value, snapshots[0][0][name]) for name, value in state.items())
        assert torch.equal(rng, snapshots[0][1])


def test_custom_series_count_cap_is_honored():
    case = make_case([series(3) for _ in range(7)], logs=False)
    model = make_model(numeric_execution='length-bucket-v1', numeric_batch_size=3)
    assert cnn_shapes(model, case)['metric'] == [(3, 3, 3), (3, 3, 3), (1, 3, 3)]
    assert_equivalent(case, batch_size=3)


@pytest.mark.parametrize('lengths,max_samples,expected', [
    ([1, 2, 3, 2, 1, 3], 32768, [(2, 3, 1), (2, 3, 2), (2, 3, 3)]),
    ([4] * 5, 8, [(2, 3, 4), (2, 3, 4), (1, 3, 4)]),
    ([4] * 5, 7, [(1, 3, 4)] * 5),
    ([9, 9, 2, 2], 8, [(1, 3, 9), (1, 3, 9), (2, 3, 2)]),
    ([16384, 16384, 16384], 32768, [(2, 3, 16384), (1, 3, 16384)]),
    ([32769, 32769], 32768, [(1, 3, 32769), (1, 3, 32769)]),
])
def test_exact_native_length_and_sample_limits(lengths, max_samples, expected):
    case = make_case([series(length) for length in lengths], logs=False)
    actual = make_model(numeric_execution='length-bucket-v1', numeric_max_batch_samples=max_samples)
    assert cnn_shapes(actual, case)['metric'] == expected
    assert_equivalent(case, max_samples=max_samples)


def rich_case():
    # Interleaved lengths force reordering during CNN execution. Duplicate series,
    # channels and timestamps must retain multiplicity and last-write context.
    first = NumericSeries('trace', 'ab', 'latency', (0., 1., 1., 4., 6.), (1., 2., 2., 3., 4.))
    last = replace(first, observed=(False,) * 5)
    return make_case((series(3), first, series(1, owner='b', channel='memory'),
                      series(2), first, series(3, owner='b'),
                      series(2, modality='trace', owner='ba', channel='memory'), last,
                      series(3, channel='unsupported'), series(2, channel='unfitted'),
                      series(0), NumericSeries('metric', 'a', 'cpu', (-3., -2., 0., 3., 7., 8., 9.),
                                              (999., 999., 1., 3., 2., 999., 999.))))


def test_explicit_scalar_retains_exact_outputs_loss_and_gradients():
    assert_equivalent(rich_case(), mode='scalar-v1')


def test_numeric_series_permutations_match_same_input_scalar_oracle():
    case = rich_case()
    visible = NumericSeries('trace', 'ba', 'latency', (0., .1, .2), (1., 2., 3.))
    withheld = NumericSeries('trace', 'ba', 'latency', (0., .1), (1., 2.), (False, False))
    case = replace(case, numeric=case.numeric + (visible, withheld))
    count = len(case.numeric)
    permutations = (tuple(reversed(range(count))),
                    tuple(range(0, count, 2)) + tuple(range(1, count, 2)))
    for order in permutations:
        assert sorted(order) == list(range(count))
        permuted = replace(case, numeric=tuple(case.numeric[index] for index in order))
        assert permuted.numeric != case.numeric
        # Each permutation is compared with its own scalar result; repeated
        # channels may legitimately change last-write context across inputs.
        assert_equivalent(permuted)


def test_entity_relation_catalog_permutations_match_same_input_scalar_oracle():
    case = rich_case()
    for reverse_entities, reverse_relations in ((True, False), (False, True), (True, True)):
        permuted = replace(case,
                           entities=case.entities[::-1] if reverse_entities else case.entities,
                           relations=case.relations[::-1] if reverse_relations else case.relations)
        # Native owner IDs stay fixed while positional owner tables are rebuilt.
        # This checks oracle equivalence, not invariance to catalog permutations.
        assert permuted.numeric == case.numeric
        actual, _ = assert_equivalent(permuted)
        assert actual.edge_owner_ids == tuple(relation.id for relation in permuted.relations)
        assert actual.candidate_indices == ((1,) if reverse_entities else (0,))


@pytest.mark.parametrize('view', ['original', 'whole-block-mask', 'all-masked', 'shifted', 'empty-trace', 'empty-all'])
def test_all_outputs_and_complete_gradients_match_frozen_scalar(view):
    case = rich_case()
    if view == 'whole-block-mask':
        case, _ = case.mask_blocks({('node', 'a', 1), ('edge', 'ab', 2)}, delta=2.)
    elif view == 'all-masked':
        case = replace(case, numeric=tuple(replace(s, observed=(False,) * len(s.times)) for s in case.numeric), logs=())
    elif view == 'shifted':
        case = case.shift_time(256.)
    elif view == 'empty-trace':
        case = replace(case, numeric=tuple(s for s in case.numeric if s.modality == 'metric'), relations=())
    elif view == 'empty-all':
        case = replace(case, numeric=(), logs=(), relations=())
    actual, _ = assert_equivalent(case)
    if view == 'original':
        assert actual.edge_context_mask[0].any()
        assert actual.trace_channel_context[(0, 0, 'latency')] is False
        assert actual.edge_owner_ids == ('ab', 'ba')


def test_native_crop_is_before_bucketing_and_does_not_read_outside_values():
    outside = NumericSeries('metric', 'a', 'cpu', (-2., 0., 2., 4., 8.), (999., 1., 2., 3., 999.))
    inside = NumericSeries('metric', 'b', 'cpu', (0., 2., 4.), (1., 2., 3.))
    case = make_case((outside, inside), relations=False, logs=False)
    model = make_model(numeric_execution='length-bucket-v1')
    assert cnn_shapes(model, case)['metric'] == [(2, 3, 3)]
    actual, _ = assert_equivalent(case)
    assert torch.equal(actual.local[0], actual.local[1])


def test_duplicate_max_ties_keep_first_series_gradient():
    # The first length bucket also contains the final series. Restoring original
    # order is essential: in bin 1, memory must win the max tie before latency.
    case = make_case((NumericSeries('metric', 'a', 'cpu', (0., .1, .2), (1., 1., 1.)),
                      NumericSeries('metric', 'a', 'memory', (2., 2.1), (1., 1.)),
                      NumericSeries('metric', 'a', 'latency', (2., 2.1, 2.2), (1., 1., 1.))), logs=False)
    expected = make_model(scalar_class())
    actual = make_model(numeric_execution='length-bucket-v1')
    with torch.no_grad():
        for parameter in expected.metric_cnn.parameters():
            parameter.zero_()
        expected.metric_channel.weight.fill_(1.)
        expected.metric_pool.weight.zero_()
        expected.metric_pool.weight[:, 64:] = torch.eye(64)
        expected.metric_pool.bias.zero_()
    actual.load_state_dict(expected.state_dict())
    for model in (expected, actual):
        output = model._numeric(case, 'metric', ('a', 'b'), {'a': 'service', 'b': 'service'}, 4)[0]
        output[0, 1].sum().backward()
    assert torch.equal(actual.metric_channel.weight.grad, expected.metric_channel.weight.grad)
    assert torch.equal(actual.metric_channel.weight.grad[2], torch.ones(64))
    assert not actual.metric_channel.weight.grad[1].any()


def test_cross_bucket_duplicate_channel_context_keeps_last_write():
    first = NumericSeries('trace', 'ab', 'latency', (0., .1, .2), (1., 2., 3.))
    middle = NumericSeries('trace', 'ab', 'latency', (0., .1), (1., 2.), (False, False))
    case = make_case((first, middle, first), logs=False)
    actual, _ = assert_equivalent(case)
    assert actual.trace_channel_context[(0, 0, 'latency')] is True
    torch.testing.assert_close(actual.edge_coverage[0, 0], torch.tensor(6 / 8), rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [
    {'numeric_execution': 'unknown'}, {'numeric_batch_size': 0}, {'numeric_batch_size': -1},
    {'numeric_batch_size': 1.5}, {'numeric_batch_size': True}, {'numeric_max_batch_samples': 0},
    {'numeric_max_batch_samples': -1}, {'numeric_max_batch_samples': 2.5},
    {'numeric_max_batch_samples': True},
])
def test_invalid_numeric_execution_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match='numeric'):
        make_model(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA equivalence requires a CUDA runtime')
@pytest.mark.parametrize('case_kind', ['rich', 'over-batch-limit', 'sample-limit'])
def test_cuda_outputs_gradients_and_rng(case_kind):
    case = rich_case() if case_kind == 'rich' else make_case(
        [series(7) for _ in range(35)] if case_kind == 'over-batch-limit' else
        [series(16384), series(16384), series(32769)], logs=False)
    assert_equivalent(case, device='cuda')
