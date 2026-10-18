"""A1 control-only changes against the frozen, same-mode B30 policy frontend.

The cotangents below are synthetic correctness diagnostics, not training losses
or timing claims. CUDA skips are explicitly not evidence of GPU equivalence.
"""
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
from torch.utils._python_dispatch import TorchDispatchMode

from test_numeric_batching import make_case, make_model, rich_case, series
from vast_v3.data.schema import LogRecord, NumericSeries
from vast_v3.encoders.frontends import NativeFrontend


MODE = 'metric-scalar-trace-bucket-v1'
MODES = ('scalar-v1', 'length-bucket-v1', MODE)
ORACLE_PATH = Path(__file__).parents[1] / 'oracles' / 'frontends_b30_policy_r1.py'
ORACLE_SHA256 = 'f6ae94b5f0b7081e0b0ebd04179ecb9519df43162c835e1d43f4dd0e01bf7bd2'
SCALAR_ORACLE_SHA256 = 'fd5048aaee589f3d81969afc92bbf37f245206b1764db70cb023ef99d537d37f'
DIFFERENTIABLE = ('h', 'local', 'edge_states', 'recovery_local', 'edge_recovery_states')
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable; CPU is not GPU evidence')


def oracle_class():
    name = 'vast_v3.encoders._frozen_frontends_b30_policy_r1_test'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ORACLE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].NativeFrontend


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def repeated_logs_case():
    case = rich_case()
    frequencies = (2, 1, 2, 7, 1, 7, 16777217, 16777216)
    logs = tuple(LogRecord('a' if i % 2 else 'b', .25 + i * .75,
                           f'record {i} request_id=hidden', frequency)
                 for i, frequency in enumerate(frequencies))
    # Out-of-window records must not populate the per-forward frequency memo.
    return replace(case, logs=(LogRecord('a', -1., 'before', 101),) + logs +
                   (LogRecord('b', 8., 'after', 103),))


def visible_frequencies(case):
    return [float(log.frequency) for log in case.logs
            if case.reference_start <= log.time < case.incident_end]


class ScalarCalls(TorchDispatchMode):
    """Observe real ATen scalar extraction, including bool(tensor), not .item alone."""

    def __init__(self):
        super().__init__()
        self.reads = []
        self.scalar_log1p = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.reads.append((args[0].dtype, args[0].device.type))
        if func == torch.ops.aten.log1p.default and args[0].ndim == 0:
            self.scalar_log1p.append((args[0].dtype, args[0].device.type))
        return func(*args, **(kwargs or {}))


def assert_exact(left, right, name='value'):
    if isinstance(right, torch.Tensor):
        assert (left.shape, left.dtype, left.device) == (right.shape, right.dtype, right.device), name
        assert torch.equal(left, right), name
    elif isinstance(right, dict):
        assert left.keys() == right.keys(), name
        for key in right:
            assert_exact(left[key], right[key], f'{name}.{key}')
    elif isinstance(right, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), name
        for i, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f'{name}[{i}]')
    else:
        assert left == right, name


def compare_outputs(actual, expected):
    assert [field.name for field in fields(actual)] == [field.name for field in fields(expected)]
    for field in fields(expected):
        assert_exact(getattr(actual, field.name), getattr(expected, field.name), field.name)


def capture_neural_inputs(model):
    calls = []
    handles = []
    for name, module in model.named_modules():
        if not name:
            continue
        def capture(_module, args, name=name):
            calls.append((name, type(_module).__name__, tuple(arg.detach().clone() for arg in args)))
        handles.append(module.register_forward_pre_hook(capture))
    return calls, handles


def diagnostic_loss(output):
    return sum((i + 1) * getattr(output, name).square().sum()
               for i, name in enumerate(DIFFERENTIABLE) if getattr(output, name).numel())


def assert_training_equivalent(cases, *, mode=MODE, device='cpu', second_cotangent='nonzero'):
    torch.manual_seed(5107)
    expected = make_model(oracle_class(), device, numeric_execution=mode)
    actual = make_model(device=device, numeric_execution=mode)
    actual.load_state_dict(expected.state_dict())
    original_cases = copy.deepcopy(cases)
    preprocessor = copy.deepcopy(actual.preprocessing.to_dict())
    cpu_rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else None
    records = []
    for model in (expected, actual):
        calls, handles = capture_neural_inputs(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        try:
            outputs = [model(case) for case in cases]
        finally:
            for handle in handles:
                handle.remove()
        loss = diagnostic_loss(outputs[0])
        for output in outputs[1:]:
            if second_cotangent == 'nonzero':
                loss = loss + diagnostic_loss(output)
            elif second_cotangent == 'zero':
                loss = loss + diagnostic_loss(output) * 0.
        loss.backward()
        gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                     for name, parameter in model.named_parameters()}
        optimizer.step()
        records.append((outputs, loss.detach(), gradients, copy.deepcopy(model.state_dict()),
                        copy.deepcopy(optimizer.state_dict()), calls))
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert random.getstate() == py_rng
        now_np = np.random.get_state()
        assert now_np[0] == np_rng[0] and np.array_equal(now_np[1], np_rng[1]) and now_np[2:] == np_rng[2:]
        if cuda_rng is not None:
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    for actual_output, expected_output in zip(records[1][0], records[0][0]):
        compare_outputs(actual_output, expected_output)
    for i, name in enumerate(('loss', 'gradients-including-None', 'Adam-parameters', 'Adam-state',
                              'neural-call-order-shapes-and-inputs'), start=1):
        assert_exact(records[1][i], records[0][i], name)
    assert cases == original_cases
    assert actual.preprocessing.to_dict() == preprocessor
    return records[1][0], actual


def test_frozen_b30_oracle_and_older_scalar_oracle_are_immutable():
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256
    assert hashlib.sha256(ORACLE_PATH.with_name('frontends_scalar_v1.py').read_bytes()).hexdigest() == SCALAR_ORACLE_SHA256


def test_no_logs_has_no_control_tensor_scalar_reads():
    case = replace(rich_case(), logs=())
    old, new = make_model(oracle_class(), numeric_execution=MODE), make_model(numeric_execution=MODE)
    with ScalarCalls() as baseline:
        old(case)
    with ScalarCalls() as optimized:
        new(case)
    assert sum(dtype == torch.bool for dtype, _ in baseline.reads) > 0
    assert not any(dtype == torch.bool for dtype, _ in optimized.reads)
    # CPU one_hot checks integer min/max internally; those are not Python
    # decisions on telemetry or relation masks and remain once after hoisting.
    assert optimized.reads == [(torch.int64, 'cpu')] * 2


def test_frequency_scalar_log1p_is_unique_per_forward_and_never_cached_across_views():
    case = repeated_logs_case()
    expected_count = len(set(visible_frequencies(case)))
    model = make_model(numeric_execution=MODE)
    for view in (case, case.shift_time(128.), case):
        with ScalarCalls() as observed:
            model(view)
        assert len(observed.scalar_log1p) == expected_count
        # The finite-embedding validation remains one legitimate bool extraction.
        assert observed.reads == ([(torch.bool, 'cpu')] +
                                  [(torch.get_default_dtype(), 'cpu')] * expected_count +
                                  [(torch.int64, 'cpu')] * 2)


def test_topology_one_hot_is_once_per_forward(monkeypatch):
    calls = []
    original = torch.nn.functional.one_hot
    def observed(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'one_hot', observed)
    model = make_model(numeric_execution=MODE)
    case = replace(rich_case(), logs=())
    for _ in range(2):
        calls.clear()
        model(case)
        assert calls == [(len(case.entities),)]


@pytest.mark.parametrize('modality,owners,categories', [
    ('metric', ('a', 'b'), {'a': 'service', 'b': 'service'}),
    ('trace', ('ab', 'ba'), {'ab': 'service->service', 'ba': 'service->service'}),
])
def test_numeric_control_masks_are_optional_and_match_converted_observations(modality, owners, categories):
    case = make_case((series(3, channel='unfitted'),
        NumericSeries('metric', 'b', 'cpu', (1., 3.), (float('nan'), 2.)),
        series(3, modality='trace', owner='ab', channel='unfitted'),
        NumericSeries('trace', 'ba', 'latency', (1., 3.), (float('inf'), 2.))), logs=False)
    model = make_model(numeric_execution=MODE)
    default = model._numeric(case, modality, owners, categories, 4)
    controls = [['sentinel']]
    with ScalarCalls() as observed:
        enriched = model._numeric(case, modality, owners, categories, 4, control_masks=controls)
    assert len(default) == len(enriched) == 6
    assert_exact(default, enriched)
    assert controls[0] == ['sentinel']
    assert controls[1:] == default[1].tolist()
    assert controls[1:] == [[False] * 4, [False, True, False, False]]
    assert observed.reads == []


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('view', ['rich', 'missing', 'raw', 'masked', 'all-masked', 'shifted',
                                 'no-metric', 'no-trace', 'no-relations', 'permuted'])
def test_same_mode_all_outputs_loss_gradients_adam_rng_and_neural_inputs(mode, view):
    case = repeated_logs_case()
    if view == 'missing':
        case = replace(case, numeric=(), logs=(), relations=())
    elif view == 'raw':
        case = make_case((series(4, channel='unfitted'),
            NumericSeries('metric', 'b', 'cpu', (-1., 0., 3., 8.),
                          (99., float('nan'), float('inf'), 99.)),
            series(4, modality='trace', owner='ab', channel='unfitted')), logs=False)
    elif view == 'masked':
        case, _ = case.mask_blocks({('node', 'a', 1), ('edge', 'ab', 2)}, delta=2.)
    elif view == 'all-masked':
        case = replace(case, numeric=tuple(replace(s, observed=(False,) * len(s.times)) for s in case.numeric), logs=())
    elif view == 'shifted':
        case = case.shift_time(256.)
    elif view == 'no-metric':
        case = replace(case, numeric=tuple(s for s in case.numeric if s.modality == 'trace'))
    elif view in ('no-trace', 'no-relations'):
        case = replace(case, numeric=tuple(s for s in case.numeric if s.modality == 'metric'),
                       relations=() if view == 'no-relations' else case.relations)
    elif view == 'permuted':
        case = replace(case, entities=case.entities[::-1], relations=case.relations[::-1], numeric=case.numeric[::-1])
    outputs, _ = assert_training_equivalent((case,), mode=mode)
    if view == 'rich':
        assert outputs[0].trace_channel_context[(0, 0, 'latency')] is False
        assert outputs[0].edge_context_mask[0].any()


@pytest.mark.parametrize('second_cotangent', ['nonzero', 'zero', 'disconnected'])
def test_shared_module_multiple_views_preserve_accumulation(second_cotangent):
    case = repeated_logs_case()
    masked, _ = case.mask_blocks({('node', 'a', 1), ('edge', 'ab', 2)}, delta=2.)
    assert_training_equivalent((case, masked.shift_time(256.)), second_cotangent=second_cotangent)


@pytest.mark.parametrize('modality,owner', [('metric', 'a'), ('trace', 'ab')])
def test_duplicate_max_ties_and_context_last_write_remain_exact(modality, owner):
    first = NumericSeries(modality, owner, 'memory', (2., 2.1), (1., 1.))
    second = NumericSeries(modality, owner, 'latency', (2., 2.1, 2.2), (1., 1., 1.))
    case = make_case((first, second, replace(first, observed=(False, False))), logs=False)
    expected = make_model(oracle_class(), numeric_execution=MODE)
    actual = make_model(numeric_execution=MODE)
    with torch.no_grad():
        for parameter in getattr(expected, modality + '_cnn').parameters():
            parameter.zero_()
        getattr(expected, modality + '_channel').weight.fill_(1.)
        pool = getattr(expected, modality + '_pool')
        pool.weight.zero_()
        pool.weight[:, 64:] = torch.eye(64)
        pool.bias.zero_()
    actual.load_state_dict(expected.state_dict())
    owners = ('a', 'b') if modality == 'metric' else ('ab', 'ba')
    categories = dict.fromkeys(owners, 'service' if modality == 'metric' else 'service->service')
    results = []
    for model in (expected, actual):
        result = model._numeric(case, modality, owners, categories, 4)
        result[0][0, 1].sum().backward()
        results.append(result)
    assert_exact(results[1], results[0])
    assert results[1][4][(0, 1, 'memory')] is False
    gradient = getattr(actual, modality + '_channel').weight.grad
    assert torch.equal(gradient[2], torch.ones(64))
    assert not gradient[1].any()
    for (name, parameter), (other_name, other) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other_name
        assert_exact(parameter.grad, other.grad, name)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_frequency_memo_keeps_original_default_dtype_expression_and_float_keys(dtype):
    case = repeated_logs_case()
    case = replace(case, logs=case.logs + (LogRecord('a', 2., 'large A', 2**53),
                                          LogRecord('b', 3., 'large B', 2**53 + 1)))
    models = (make_model(oracle_class(), numeric_execution=MODE), make_model(numeric_execution=MODE))
    before = torch.get_default_dtype()
    snapshots = []
    class StopBeforeMLP(Exception):
        pass
    # Capturing the original MLP input isolates default-dtype frequency behavior.
    # A float64 default otherwise promotes extras and hits the original float32
    # MLP dtype mismatch; do not cast away that behavior in production or tests.
    try:
        torch.set_default_dtype(dtype)
        for model in models:
            def capture(_module, args):
                snapshots.append(args[0].detach().clone())
                raise StopBeforeMLP
            handle = model.log_mlp.register_forward_pre_hook(capture)
            try:
                with ScalarCalls() as observed, pytest.raises(StopBeforeMLP):
                    model(case)
            finally:
                handle.remove()
        frequencies = visible_frequencies(case)
        assert len(observed.scalar_log1p) == len(set(frequencies))
        assert all(item == (dtype, 'cpu') for item in observed.scalar_log1p)
        expected = torch.tensor([torch.log1p(torch.tensor(float(f))).item() for f in frequencies])
        assert_exact(snapshots[1], snapshots[0])
        assert snapshots[1].dtype == dtype
        assert torch.equal(snapshots[1][:, 384], expected)
    finally:
        torch.set_default_dtype(before)


@pytest.mark.parametrize('bad_kind', ['shape', 'nonfinite', 'exception'])
def test_frequency_work_stays_after_text_encoding_and_validation(bad_kind):
    events = []
    def encoder(texts):
        events.append('text')
        if bad_kind == 'exception':
            raise RuntimeError('text failed')
        return torch.full((len(texts), 383 if bad_kind == 'shape' else 384), float('nan'))
    model = make_model(numeric_execution=MODE)
    model.text_encoder = encoder
    error = RuntimeError if bad_kind == 'exception' else ValueError
    with ScalarCalls() as observed, pytest.raises(error):
        model(repeated_logs_case())
    assert events == ['text']
    assert observed.scalar_log1p == []
    assert observed.reads == ([(torch.bool, 'cpu')] if bad_kind == 'nonfinite' else [])


@CUDA
def test_cuda_no_logs_has_no_aten_local_scalar_dense_reads():
    model = make_model(device='cuda', numeric_execution=MODE)
    with ScalarCalls() as observed:
        model(replace(rich_case(), logs=()))
    assert observed.reads == []


@CUDA
@pytest.mark.parametrize('default_device', ['cpu', 'cuda'])
def test_cuda_logs_keep_only_finite_validation_and_unique_frequency_reads(default_device):
    model = make_model(device='cuda', numeric_execution=MODE)
    case = repeated_logs_case()
    with torch.device(default_device), ScalarCalls() as observed:
        model(case)
    count = len(set(visible_frequencies(case)))
    assert observed.reads == [(torch.bool, 'cuda')] + [(torch.get_default_dtype(), default_device)] * count
    assert observed.scalar_log1p == [(torch.get_default_dtype(), default_device)] * count


@CUDA
@pytest.mark.parametrize('logs', [False, True])
def test_cuda_same_mode_outputs_gradients_adam_rng_and_neural_inputs(logs):
    case = repeated_logs_case()
    if not logs:
        case = replace(case, logs=())
    assert_training_equivalent((case, case.shift_time(256.)), device='cuda')
