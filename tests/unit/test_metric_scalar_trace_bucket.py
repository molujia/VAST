"""Hybrid dispatch preserves scalar metric graph order and batches only traces.

The frozen scalar source is the independent oracle. These synthetic fixtures
are correctness diagnostics, never training objectives or speed benchmarks.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import fields, replace
import hashlib
from itertools import permutations

import pytest
import torch

from test_numeric_batching import (
    APPROXIMATE_FIELDS, ATOL, ORACLE_PATH, ORACLE_SHA256, RTOL,
    assert_equivalent, cnn_shapes, make_case, make_model, rich_case,
    scalar_class, series,
)
from vast_v3.data.schema import NumericSeries
from vast_v3.encoders.frontends import NativeFrontend
from vast_v3.runtime import RuntimeConfig, build_objectives
from vast_v3.training.checkpoint import load_checkpoint, save_checkpoint


MODE = 'metric-scalar-trace-bucket-v1'
MODES = ('scalar-v1', 'length-bucket-v1', MODE)
DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]
# Every declared repeat is evaluated; no best-run selection or retry budget.
GPU_REPEAT_SEEDS = (1601, 1602, 1603)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def backend_state():
    return dict(cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                dtype=torch.get_default_dtype())


@contextmanager
def diagnostic_policy(device):
    """Approved CUDA policy changes this one flag only, scoped to the fixture."""
    before = backend_state()
    try:
        if device == 'cuda':
            torch.backends.cudnn.allow_tf32 = False
        expected = {**before, 'cudnn_allow_tf32': False} if device == 'cuda' else before
        assert backend_state() == expected
        yield
        assert backend_state() == expected
    finally:
        torch.backends.cudnn.allow_tf32 = before['cudnn_allow_tf32']
    assert backend_state() == before


def compare_outputs(actual, expected, *, exact=False):
    for field in fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, torch.Tensor):
            assert (left.shape, left.dtype, left.device) == (right.shape, right.dtype, right.device), field.name
            if field.name in APPROXIMATE_FIELDS and not exact:
                torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL, msg=field.name)
            else:
                assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


def compare_gradients(actual, expected, *, exact=False):
    reference = dict(expected.named_parameters())
    assert dict(actual.named_parameters()).keys() == reference.keys()
    for name, value in actual.named_parameters():
        other = reference[name]
        assert (value.grad is None) == (other.grad is None), name
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad,
                atol=0. if exact else ATOL, rtol=0. if exact else RTOL, msg=name)


def test_explicit_hybrid_mode_is_supported_without_changing_default_or_schema():
    config = RuntimeConfig(dataset='rcabench', numeric_execution=MODE)
    model = make_model(numeric_execution=MODE)
    assert model.numeric_execution == config.numeric_execution == MODE
    assert RuntimeConfig.from_dict(config.to_dict()) == config
    scalar = RuntimeConfig(dataset='rcabench')
    assert scalar.numeric_execution == make_model().numeric_execution == 'scalar-v1'
    assert scalar.schema == config.schema == 'vast-v3-runtime/3'
    assert scalar.to_dict().keys() == config.to_dict().keys()
    assert (config.numeric_batch_size, config.numeric_max_batch_samples) == (32, 32768)
    configs = [replace(config, numeric_execution=mode) for mode in MODES]
    assert len({item.identity for item in configs}) == 3
    assert all(item.schedule() == scalar.schedule() for item in configs)


def test_frontend_accepts_hybrid_independently_of_runtime():
    assert make_model(numeric_execution=MODE).numeric_execution == MODE


def test_hybrid_constructor_and_factory_keep_initialization_rng_and_backend_state():
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256
    before = backend_state()
    snapshots = []
    for kind, options in [(scalar_class(), {})] + [
            (NativeFrontend, {'numeric_execution': mode}) for mode in MODES]:
        torch.manual_seed(8172)
        model = make_model(kind, **options)
        snapshots.append((copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()))
    for state, rng in snapshots[1:]:
        assert state.keys() == snapshots[0][0].keys()
        assert all(torch.equal(value, snapshots[0][0][name]) for name, value in state.items())
        assert torch.equal(rng, snapshots[0][1])
    config = RuntimeConfig(dataset='rcabench', numeric_execution=MODE,
                           numeric_batch_size=7, numeric_max_batch_samples=111)
    pre = model.preprocessing
    objectives = [build_objectives(replace(config, numeric_execution=mode), pre,
                  text_encoder=lambda texts: torch.zeros(len(texts), 384)) for mode in MODES]
    for obj in objectives[1:]:
        assert obj.state_dict().keys() == objectives[0].state_dict().keys()
        assert all(torch.equal(value, objectives[0].state_dict()[name])
                   for name, value in obj.state_dict().items())
    front = objectives[-1].model.frontend
    assert (front.numeric_execution, front.numeric_batch_size, front.numeric_max_batch_samples) == (MODE, 7, 111)
    assert backend_state() == before


@pytest.mark.parametrize('training', [True, False], ids=['train', 'eval'])
@pytest.mark.parametrize('context', ['grad', 'no_grad', 'inference'])
def test_modality_schedule_is_fixed_in_all_training_and_grad_contexts(training, context):
    case = make_case((series(3, offset=.1),
        series(2, modality='trace', owner='ab', channel='latency'),
        series(1, offset=.2), series(3, owner='b', offset=.3),
        series(4, modality='trace', owner='ba', channel='latency'),
        series(2, offset=.4), series(2, modality='trace', owner='ba', channel='memory')))
    model = make_model(numeric_execution=MODE).train(training)
    oracle = make_model(scalar_class()).train(training)
    oracle.load_state_dict(model.state_dict())
    shapes, inputs = {'metric': [], 'trace': []}, []
    handles = [getattr(model, modality + '_cnn').register_forward_pre_hook(
        lambda _, args, modality=modality: shapes[modality].append(tuple(args[0].shape)))
        for modality in shapes]
    handles.append(model.metric_cnn.register_forward_pre_hook(
        lambda _, args: inputs.append(args[0].detach().clone())))
    manager = {'grad': torch.enable_grad, 'no_grad': torch.no_grad, 'inference': torch.inference_mode}[context]
    before = backend_state()
    try:
        with manager():
            actual, expected = model(case), oracle(case)
            compare_outputs(actual, expected)
            assert actual.h.requires_grad == (context == 'grad')
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == {'metric': [(1, 3, 3), (1, 3, 1), (1, 3, 3), (1, 3, 2)],
                      'trace': [(2, 3, 2), (1, 3, 4)]}
    prepared = list(model._numeric_inputs(case, 'metric', ('a', 'b'), {'a': 'service', 'b': 'service'}))
    assert len(prepared) == len(inputs)
    assert all(torch.equal(value, torch.tensor(record[4], dtype=torch.float32).unsqueeze(0))
               for value, record in zip(inputs, prepared))
    assert backend_state() == before


def test_metric_preparation_cnn_yield_and_pooling_remain_interleaved(monkeypatch):
    lengths = (3, 1, 3, 2)
    case = make_case(tuple(NumericSeries('metric', 'a', 'cpu',
        tuple(j / 10 for j in range(length)), tuple(float(index + j) for j in range(length)))
        for index, length in enumerate(lengths)), logs=False)
    model = make_model(numeric_execution=MODE)
    events = []
    original_inputs, original_encoded = model._numeric_inputs, model._numeric_encoded
    original_mean = torch.Tensor.mean

    def prepared(*args, **kwargs):
        for index, record in enumerate(original_inputs(*args, **kwargs)):
            events.append(('prepare', index))
            yield record

    def encoded(*args, **kwargs):
        for index, record in enumerate(original_encoded(*args, **kwargs)):
            events.append(('yield', index))
            yield record

    def pooled(value, *args, **kwargs):
        if value.ndim == 2 and value.shape[-1] == 64:
            events.append(('pool', None))
        return original_mean(value, *args, **kwargs)

    monkeypatch.setattr(model, '_numeric_inputs', prepared)
    monkeypatch.setattr(model, '_numeric_encoded', encoded)
    monkeypatch.setattr(torch.Tensor, 'mean', pooled)
    handle = model.metric_cnn.register_forward_pre_hook(
        lambda _, args: events.append(('cnn', args[0].shape[-1])))
    try:
        model._numeric(case, 'metric', ('a', 'b'), {'a': 'service', 'b': 'service'}, 4)
    finally:
        handle.remove()
    expected = []
    for index, length in enumerate(lengths):
        expected += [('prepare', index), ('cnn', length), ('yield', index), ('pool', None), ('pool', None)]
    # Final channel pooling follows native per-series pooling. The exact prefix
    # rules out list(inputs), bucket-size-one emulation, and length reordering.
    assert events[:len(expected)] == expected


@pytest.mark.parametrize('lengths,max_samples,batch_size,expected', [
    ([3] * 35, 32768, 32, [(32, 3, 3), (3, 3, 3)]),
    ([1, 2, 3, 2, 1, 3], 32768, 32, [(2, 3, 1), (2, 3, 2), (2, 3, 3)]),
    ([3] * 7, 32768, 3, [(3, 3, 3), (3, 3, 3), (1, 3, 3)]),
    ([4] * 5, 8, 32, [(2, 3, 4), (2, 3, 4), (1, 3, 4)]),
    ([4] * 5, 7, 32, [(1, 3, 4)] * 5),
    ([9, 9, 2, 2], 8, 32, [(1, 3, 9), (1, 3, 9), (2, 3, 2)]),
    ([16384] * 3, 32768, 32, [(2, 3, 16384), (1, 3, 16384)]),
    ([32769] * 2, 32768, 32, [(1, 3, 32769)] * 2),
])
def test_trace_caps_native_lengths_and_full_frontend_gradients(lengths, max_samples, batch_size, expected):
    case = make_case([series(3), series(1), series(3)] + [series(length,
        modality='trace', owner='ab', channel='latency') for length in lengths], logs=False)
    model = make_model(numeric_execution=MODE, numeric_batch_size=batch_size,
                       numeric_max_batch_samples=max_samples)
    shapes = cnn_shapes(model, case)
    assert shapes['metric'] == [(1, 3, 3), (1, 3, 1), (1, 3, 3)]
    assert shapes['trace'] == expected
    assert_equivalent(case, mode=MODE, batch_size=batch_size, max_samples=max_samples)


@pytest.mark.parametrize('view', ['original', 'whole-block-mask', 'all-masked', 'shifted',
                                 'empty-trace', 'empty-metric', 'empty-all'])
def test_ragged_duplicates_masks_contexts_and_every_output_gradient(view):
    case = rich_case()
    if view == 'whole-block-mask':
        case, _ = case.mask_blocks({('node', 'a', 1), ('edge', 'ab', 2)}, delta=2.)
    elif view == 'all-masked':
        case = replace(case, numeric=tuple(replace(s, observed=(False,) * len(s.times)) for s in case.numeric), logs=())
    elif view == 'shifted':
        case = case.shift_time(256.)
    elif view == 'empty-trace':
        case = replace(case, numeric=tuple(s for s in case.numeric if s.modality == 'metric'), relations=())
    elif view == 'empty-metric':
        case = replace(case, numeric=tuple(s for s in case.numeric if s.modality == 'trace'))
    elif view == 'empty-all':
        case = replace(case, numeric=(), logs=(), relations=())
    actual, _ = assert_equivalent(case, mode=MODE)
    if view == 'original':
        assert actual.trace_channel_context[(0, 0, 'latency')] is False
        assert actual.edge_context_mask[0].any()


@pytest.mark.parametrize('catalog', ['entities', 'relations', 'both', 'series-reversed', 'series-interleaved'])
def test_owner_catalog_and_series_permutations_use_same_input_oracle(catalog):
    case = rich_case()
    if catalog in ('entities', 'both'):
        case = replace(case, entities=case.entities[::-1])
    if catalog in ('relations', 'both'):
        case = replace(case, relations=case.relations[::-1])
    if catalog == 'series-reversed':
        case = replace(case, numeric=case.numeric[::-1])
    elif catalog == 'series-interleaved':
        case = replace(case, numeric=case.numeric[::2] + case.numeric[1::2])
    actual, _ = assert_equivalent(case, mode=MODE)
    assert actual.edge_owner_ids == tuple(r.id for r in case.relations)
    assert actual.candidate_indices == ((1,) if catalog in ('entities', 'both') else (0,))


def test_trace_native_crop_last_write_context_and_duplicate_coverage():
    first = NumericSeries('trace', 'ab', 'latency', (-2., 0., .1, .1, 8.), (999., 1., 2., 2., 999.))
    middle = NumericSeries('trace', 'ab', 'latency', (0., .1), (1., 2.), (False, False))
    case = make_case((first, middle, first), logs=False)
    model = make_model(numeric_execution=MODE)
    assert cnn_shapes(model, case) == {'metric': [], 'trace': [(2, 3, 3), (1, 3, 2)]}
    actual, _ = assert_equivalent(case, mode=MODE)
    assert actual.trace_channel_context[(0, 0, 'latency')] is True
    assert torch.equal(actual.edge_coverage[0, 0], torch.tensor(6 / 8))


def test_trace_duplicate_max_ties_preserve_first_channel_gradient():
    case = make_case((NumericSeries('trace', 'ab', 'cpu', (0., .1, .2), (1., 1., 1.)),
        NumericSeries('trace', 'ab', 'memory', (2., 2.1), (1., 1.)),
        NumericSeries('trace', 'ab', 'latency', (2., 2.1, 2.2), (1., 1., 1.))), logs=False)
    oracle, actual = make_model(scalar_class()), make_model(numeric_execution=MODE)
    with torch.no_grad():
        for parameter in oracle.trace_cnn.parameters():
            parameter.zero_()
        oracle.trace_channel.weight.fill_(1.)
        oracle.trace_pool.weight.zero_()
        oracle.trace_pool.weight[:, 64:] = torch.eye(64)
        oracle.trace_pool.bias.zero_()
    actual.load_state_dict(oracle.state_dict())
    for model in (oracle, actual):
        output = model._numeric(case, 'trace', ('ab', 'ba'),
            {'ab': 'service->service', 'ba': 'service->service'}, 4)[0]
        output[0, 1].sum().backward()
    compare_gradients(actual, oracle, exact=True)
    assert torch.equal(actual.trace_channel.weight.grad[2], torch.ones(64))
    assert not actual.trace_channel.weight.grad[1].any()


@pytest.mark.parametrize('saved_mode,requested_mode', list(permutations(MODES, 2)))
def test_checkpoint_modes_cannot_cross_resume_before_any_state_or_rng_mutation(tmp_path, saved_mode, requested_mode):
    config = RuntimeConfig(dataset='rcabench', numeric_execution=saved_mode)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    identity = dict(data_hash='synthetic', preprocessing_hash='synthetic',
                    code_hash='unchanged-code', config_hash=config.identity)
    path = save_checkpoint(tmp_path / 'checkpoint.pt', model=model, optimizer=optimizer,
        identity=identity, progress=dict(stage='synthetic', epoch=0, slot=1,
            encoder_version=0, stage_complete=True), query_sets={'L': [], 'U': []})
    target = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.Adam(target.parameters())
    before = copy.deepcopy(target.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='config_hash'):
        load_checkpoint(path, model=target, optimizer=target_optimizer,
            expected_identity={**identity, 'config_hash': replace(config, numeric_execution=requested_mode).identity})
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())
    assert target_optimizer.state_dict()['state'] == {}
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('second_cotangent', ['nonzero', 'zero', 'disconnected'])
def test_shared_metric_views_preserve_exact_parameter_accumulation(device, second_cotangent):
    """A shared module's second view must cover nonzero, zero, and None grads."""
    with diagnostic_policy(device):
        torch.manual_seed(7351)
        oracle = make_model(scalar_class(), device)
        actual = make_model(device=device, numeric_execution=MODE)
        actual.load_state_dict(oracle.state_dict())
        case = make_case((series(7), series(3, owner='b'), series(7, channel='memory')), logs=False)
        views = (case, case.shift_time(128.))
        captured = []
        for model in (oracle, actual):
            outputs = [model(view) for view in views]
            first, second = outputs[0].recovery_local, outputs[1].recovery_local
            first.retain_grad()
            second.retain_grad()
            # This is a declared synthetic cotangent, not a formal loss change.
            loss = first.square().sum()
            if second_cotangent != 'disconnected':
                loss = loss + (second.square().sum() if second_cotangent == 'nonzero' else second.sum() * 0.)
            loss.backward()
            assert first.grad is not None and first.grad.abs().max() > 0
            if second_cotangent == 'disconnected':
                assert second.grad is None
            else:
                assert second.grad is not None
                assert bool(second.grad.abs().max() > 0) == (second_cotangent == 'nonzero')
            captured.append(outputs)
        for left, right in zip(captured[1], captured[0]):
            compare_outputs(left, right, exact=True)
        compare_gradients(actual, oracle, exact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA equivalence requires a CUDA runtime')
@pytest.mark.parametrize('case_kind', ['rich', 'trace-count-cap', 'trace-sample-cap'])
def test_cuda_complete_frontends_and_gradients(case_kind):
    case = rich_case() if case_kind == 'rich' else make_case(
        [series(7)] + [series(length, modality='trace', owner='ab', channel='latency')
                      for length in ([7] * 35 if case_kind == 'trace-count-cap' else [16384, 16384, 32769])],
        logs=False)
    with diagnostic_policy('cuda'):
        assert_equivalent(case, device='cuda', mode=MODE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='fixed high-dynamic fixture requires CUDA')
@pytest.mark.parametrize('seed', GPU_REPEAT_SEEDS)
@pytest.mark.parametrize('second_cotangent', ['nonzero', 'zero', 'disconnected'])
def test_cuda_fixed_high_dynamic_near_cancellation_shared_metric(seed, second_cotangent):
    """Three fixed seeds, two high-dynamic views, all metric gradients exact.

    Identical views receive opposite 1e4-scale cotangents with a fixed small
    residual. The metric path is isolated so trace rounding cannot perturb its
    upstream derivative. Trace batching is separately tested end-to-end above.
    """
    with diagnostic_policy('cuda'):
        torch.manual_seed(seed)
        oracle = make_model(scalar_class(), 'cuda')
        actual = make_model(device='cuda', numeric_execution=MODE)
        actual.load_state_dict(oracle.state_dict())
        samples = tuple(((-1.) ** j) * (10000. + j / 10) for j in range(21))
        case = make_case(tuple(NumericSeries('metric', owner, 'cpu',
            tuple(7.5 * j / 21 for j in range(21)), samples)
            for owner in ('a', 'b', 'a', 'b')), logs=False)
        results = []
        for model in (oracle, actual):
            values = [model._numeric(case, 'metric', ('a', 'b'),
                {'a': 'service', 'b': 'service'}, 4)[0] for _ in range(2)]
            first, second = values
            first.retain_grad()
            second.retain_grad()
            upstream = torch.linspace(-10000., 10000., first.numel(), device='cuda').reshape_as(first)
            loss = (first * upstream).sum()
            if second_cotangent == 'nonzero':
                loss = loss + (second * (-upstream + .03125)).sum()
            elif second_cotangent == 'zero':
                loss = loss + (second * torch.zeros_like(upstream)).sum()
            loss.backward()
            assert torch.equal(first.grad, upstream)
            if second_cotangent == 'disconnected':
                assert second.grad is None
            else:
                expected = -upstream + .03125 if second_cotangent == 'nonzero' else torch.zeros_like(upstream)
                assert torch.equal(second.grad, expected)
            results.append(values)
        assert all(torch.equal(a, b) for a, b in zip(*results))
        compare_gradients(actual, oracle, exact=True)
