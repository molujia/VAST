"""Complete hybrid training against the immutable scalar frontend oracle."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from reviewed_cpu_pn_adam_exception import assert_initial_adam_tensor_close
from test_numeric_batching_objectives import fixture
from vast_v3.runtime import RuntimeConfig, build_objectives
from vast_v3.training.checkpoint import load_checkpoint, save_checkpoint

try:
    from test_numeric_batching import ORACLE_PATH, ORACLE_SHA256, scalar_class
except ModuleNotFoundError:
    # A targeted integration-only invocation does not add tests/unit to sys.path.
    _spec = importlib.util.spec_from_file_location(
        "_numeric_batching_oracle_helpers",
        Path(__file__).parents[1] / "unit" / "test_numeric_batching.py",
    )
    _helpers = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_helpers)
    ORACLE_PATH, ORACLE_SHA256 = _helpers.ORACLE_PATH, _helpers.ORACLE_SHA256
    scalar_class = _helpers.scalar_class


ATOL, RTOL = 2e-5, 2e-4
HYBRID = "metric-scalar-trace-bucket-v1"
FRONT_FLOAT_FIELDS = {"h", "local", "edge_states", "recovery_local", "edge_recovery_states"}


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@pytest.fixture(autouse=True)
def _isolated_rng_and_single_cpu_thread():
    previous_threads, previous_rng = torch.get_num_threads(), _rng_state()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        _restore_rng(previous_rng)


def _backend_policy():
    return {
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }


@contextmanager
def _cuda_diagnostic_policy(device):
    before = _backend_policy()
    try:
        if device == "cuda":
            torch.backends.cudnn.allow_tf32 = False
        expected = dict(before)
        if device == "cuda":
            expected["cudnn_tf32"] = False
        assert _backend_policy() == expected
        yield
        assert _backend_policy() == expected
    finally:
        # No matmul, precision, determinism, benchmark, or other flag is changed.
        if device == "cuda":
            torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        assert _backend_policy() == before


def _compare(actual, expected, label, *, exact=False):
    """Compare every nested field; metadata and nonfloating tensors stay exact."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), label
        assert (actual.shape, actual.dtype, actual.device) == (
            expected.shape, expected.dtype, expected.device
        ), label
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
        if exact or not expected.is_floating_point():
            assert torch.equal(actual, expected), label
        else:
            torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL,
                                       msg=lambda detail: label + "\n" + detail)
        return float((actual.detach() - expected.detach()).abs().max()) if (
            expected.numel() and expected.is_floating_point()
        ) else 0.
    if isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected), label
        return 0.
    if is_dataclass(expected):
        assert is_dataclass(actual), label
        names = tuple(field.name for field in fields(expected))
        assert tuple(field.name for field in fields(actual)) == names, label
        return max((_compare(getattr(actual, name), getattr(expected, name),
                             label + "." + name, exact=exact)
                    for name in names), default=0.)
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), label
        return max((_compare(actual[key], value, f"{label}[{key!r}]", exact=exact)
                    for key, value in expected.items()), default=0.)
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), label
        return max((_compare(left, right, f"{label}[{index}]", exact=exact)
                    for index, (left, right) in enumerate(zip(actual, expected))), default=0.)
    assert type(actual) is type(expected) and actual == expected, label
    return 0.


def _compare_outputs(actual, expected, label, *, exact=False):
    assert len(actual) == len(expected), label
    maximum = 0.
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert tuple(field.name for field in fields(left)) == tuple(
            field.name for field in fields(right)
        )
        for field in fields(right):
            a, b = getattr(left, field.name), getattr(right, field.name)
            path = f"{label}[{index}].{field.name}"
            if field.name == "front":
                assert tuple(f.name for f in fields(a)) == tuple(f.name for f in fields(b))
                for front_field in fields(b):
                    name = front_field.name
                    maximum = max(maximum, _compare(
                        getattr(a, name), getattr(b, name), path + "." + name,
                        exact=exact or name not in FRONT_FLOAT_FIELDS,
                    ))
            else:
                maximum = max(maximum, _compare(a, b, path, exact=exact))
    return maximum


def _build(config, pre, text, device, *, oracle=False):
    objectives = build_objectives(config, pre, text_encoder=text, device=device)
    if oracle:
        assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256
        assert ORACLE_SHA256 == "fd5048aaee589f3d81969afc92bbf37f245206b1764db70cb023ef99d537d37f"
        current = objectives.model.frontend
        # Replacing the implementation must not advance the initialization RNG.
        with torch.random.fork_rng(devices=[]):
            frozen = scalar_class()(
                pre, current.entity_categories, current.relation_categories, current.channels,
                text_encoder=text, log_recovery_radius_steps=config.log_recovery_radius_steps,
            ).to(device)
        frozen.load_state_dict(current.state_dict(), strict=True)
        _compare(frozen.state_dict(), current.state_dict(), "oracle initialization", exact=True)
        objectives.model.frontend = frozen
    return objectives


def _cases_and_jobs(branch):
    first, pre, text = fixture()
    second = replace(first, case_id=first.case_id + "-second", numeric=tuple(
        replace(series, values=tuple(value * (1. + .03 * index) + .07 for value in series.values))
        for index, series in enumerate(first.numeric)
    ))
    cases = (first, second)
    jobs = []
    for epoch in (0, 1):
        for index, case in enumerate(cases):
            roots = frozenset({"a" if index == 0 else "b"})
            if branch == "U" or (branch == "mixed" and (epoch + index) % 2):
                roots = None
            jobs.append((case, epoch, roots))
    return cases, jobs, pre, text


def _shared_probes(scalar, hybrid, cases, jobs):
    probes, masks = list(cases), set()
    for case, epoch, _ in jobs:
        left, right = scalar.views.prepare(case, epoch=epoch), hybrid.views.prepare(case, epoch=epoch)
        assert left == right
        assert left.blocks and left.targets
        masks.add(left.blocks)
        probes.append(left.case)
    assert len(masks) > 1, "multiple distinct shared masked views must be exercised"
    return tuple(probes)


def _outputs(objectives, probes, seed=42):
    with torch.no_grad():
        return tuple(objectives.model(case, seed=seed + index) for index, case in enumerate(probes))


def _accumulate(objectives, jobs, *, seed=42):
    values, memberships = [], []
    observed = set()
    for case, epoch, roots in jobs:
        losses = objectives.case_losses(case, epoch=epoch, roots=roots, seed=seed)
        assert tuple(losses) == objectives.names
        expected = {"local_recovery", "process_recovery", "invariance"}
        expected |= {"U_soft"} if roots is None else {"pn", "source"}
        assert {name for name, value in losses.items() if value is not None} == expected
        observed.update(expected)
        values.append({name: value.detach().clone() if value is not None else None
                       for name, value in losses.items()})
        memberships.append(objectives.library_membership(case.case_id, epoch)
                           if roots is None else None)
        sum(value for value in losses.values() if value is not None).backward()
    assert {"local_recovery", "process_recovery", "invariance"} <= observed
    return values, memberships


def _gradients(objectives):
    return {name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in objectives.named_parameters()}


def _draw_rng():
    return {
        "python": random.random(),
        "numpy": np.random.random(4),
        "cpu": torch.rand(4),
        "seed": torch.randint(1, 100000, ()).item(),
        "cuda": [torch.rand(4, device=f"cuda:{index}")
                 for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }


def _reload_exact(tmp_path, objectives, optimizer, config, pre, text, device, probes, *, oracle):
    objectives.eval()
    identity = dict(data_hash="hybrid-synthetic-two-cases", preprocessing_hash=pre.identity,
                    config_hash=config.identity, code_hash="hybrid-objective-regression")
    saved_rng = _rng_state()
    path = save_checkpoint(
        tmp_path / (config.numeric_execution + ".pt"), model=objectives, optimizer=optimizer,
        identity=identity, progress=dict(stage="hybrid-regression", epoch=2, slot=0,
                                        encoder_version=1, stage_complete=True),
        query_sets={"L": [], "U": [case.case_id for case in probes[:2]]},
    )
    _compare(_rng_state(), saved_rng, "saving preserves RNG", exact=True)
    expected_draws = _draw_rng()
    expected_outputs = _outputs(objectives, probes, seed=expected_draws["seed"])
    restored = _build(config, pre, text, device, oracle=oracle)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
    # Building reseeds RNG. Advance every stream again before loading so this
    # exercises real restoration, not a comparison of an untouched RNG snapshot.
    # Distinct seeds also cover the second checkpoint, whose saved RNG has
    # already advanced by the first checkpoint's replayed draws.
    random.seed(config.seed + 1001)
    np.random.seed(config.seed + 1001)
    torch.manual_seed(config.seed + 1001)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 1001)
    _draw_rng()
    disturbed = _rng_state()
    assert disturbed["python"] != saved_rng["python"]
    assert not np.array_equal(disturbed["numpy"][1], saved_rng["numpy"][1])
    assert not torch.equal(disturbed["cpu"], saved_rng["cpu"])
    assert all(not torch.equal(a, b) for a, b in zip(disturbed["cuda"], saved_rng["cuda"]))
    load_checkpoint(path, model=restored, optimizer=restored_optimizer,
                    expected_identity=identity, require_complete=True,
                    # Match Trainer and the real-data probe: CPU deserialization
                    # preserves Adam's CPU step counters while load_state_dict
                    # places model tensors/moments on their owning devices.
                    restore_rng=True, map_location="cpu")
    _compare(_rng_state(), saved_rng, "restored RNG state", exact=True)
    _compare(restored.state_dict(), objectives.state_dict(), "restored full model", exact=True)
    _compare(restored_optimizer.state_dict(), optimizer.state_dict(), "restored Adam", exact=True)
    actual_draws = _draw_rng()
    _compare(actual_draws, expected_draws, "restored RNG draws", exact=True)
    restored.eval()
    _compare_outputs(_outputs(restored, probes, seed=actual_draws["seed"]), expected_outputs,
                     config.numeric_execution + " full output replay", exact=True)
    return restored, restored_optimizer


def _resume_step_exact(original, restored, optimizer, restored_optimizer, jobs, probes):
    # A fresh epoch rebuilds keyed view/library caches on both sides of resume.
    resumed_jobs = [(case, epoch + 3, roots) for case, epoch, roots in jobs]
    values, gradients, clipped, norms = [], [], [], []
    for obj, opt in ((original, optimizer), (restored, restored_optimizer)):
        obj.train()
        opt.zero_grad(set_to_none=True)
        values.append(_accumulate(obj, resumed_jobs, seed=917))
        gradients.append(_gradients(obj))
        norms.append(torch.nn.utils.clip_grad_norm_(obj.parameters(), 5., error_if_nonfinite=True))
        clipped.append(_gradients(obj))
        opt.step()
        obj.eval()
    _compare(values[1], values[0], "resumed losses and libraries", exact=True)
    _compare(gradients[1], gradients[0], "resumed raw gradients", exact=True)
    _compare(norms[1], norms[0], "resumed clipping norm", exact=True)
    _compare(clipped[1], clipped[0], "resumed clipped gradients", exact=True)
    _compare(restored.state_dict(), original.state_dict(), "resumed model update", exact=True)
    _compare(restored_optimizer.state_dict(), optimizer.state_dict(), "resumed Adam update", exact=True)
    _compare_outputs(_outputs(restored, probes), _outputs(original, probes),
                     "resumed complete output", exact=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable; CPU does not establish CUDA equivalence"))])
@pytest.mark.parametrize("branch", ["PN-source", "U", "mixed"])
def test_complete_objectives_accumulation_adam_and_exact_resume(tmp_path, device, branch):
    config = RuntimeConfig(dataset="rcabench", numeric_execution=HYBRID)
    # The reviewed exception is confined to this unchanged synthetic fixture.
    assert config.seed == 42
    scalar_config = replace(config, numeric_execution="scalar-v1")
    cases, jobs, pre, text = _cases_and_jobs(branch)
    reviewed_exceptions = []
    with _cuda_diagnostic_policy(device):
        scalar = _build(scalar_config, pre, text, device, oracle=True)
        hybrid = _build(config, pre, text, device)
        hybrid.load_state_dict(scalar.state_dict(), strict=True)
        _compare(hybrid.state_dict(), scalar.state_dict(), "identical initial state", exact=True)
        probes = _shared_probes(scalar, hybrid, cases, jobs)
        maxima = {"initial_full_output": _compare_outputs(
            _outputs(hybrid, probes), _outputs(scalar, probes), "initial complete output")}
        optimizers = [torch.optim.Adam(obj.parameters(), lr=.001) for obj in (scalar, hybrid)]
        values = [_accumulate(obj, jobs) for obj in (scalar, hybrid)]
        maxima["all_losses"] = _compare(values[1][0], values[0][0], "all complete objective losses")
        _compare(values[1][1], values[0][1], "source library membership", exact=True)
        maxima["raw_gradients"] = _compare(_gradients(hybrid), _gradients(scalar), "all raw gradients")
        norms = [torch.nn.utils.clip_grad_norm_(obj.parameters(), 5., error_if_nonfinite=True)
                 for obj in (scalar, hybrid)]
        maxima["clip_norm"] = _compare(norms[1], norms[0], "global clipping norm")
        maxima["clipped_gradients"] = _compare(_gradients(hybrid), _gradients(scalar),
                                              "all clipped gradients")
        for optimizer in optimizers:
            optimizer.step()
        actual_state, expected_state = hybrid.state_dict(), scalar.state_dict()
        assert actual_state.keys() == expected_state.keys(), "post-Adam full state keys"
        maxima["post_Adam_full_state"] = 0.
        for name, expected in expected_state.items():
            actual = actual_state[name]
            if name == "model.frontend.trace_cnn.2.weight":
                reviewed = assert_initial_adam_tensor_close(
                    actual, expected, tensor_name=name, stage="first-post-Adam-full-state",
                    device=device, branch=branch,
                )
                if reviewed is not None:
                    reviewed_exceptions.append(reviewed)
                difference = float((actual.detach() - expected.detach()).abs().max())
            else:
                difference = _compare(actual, expected, f"post-Adam full state[{name!r}]")
            maxima["post_Adam_full_state"] = max(maxima["post_Adam_full_state"], difference)
        maxima["post_Adam_optimizer"] = _compare(optimizers[1].state_dict(), optimizers[0].state_dict(),
                                                 "post-Adam optimizer state")
        scalar.eval()
        hybrid.eval()
        maxima["post_Adam_full_output"] = _compare_outputs(
            _outputs(hybrid, probes), _outputs(scalar, probes), "post-Adam complete output")
        loaded = [
            _reload_exact(tmp_path, scalar, optimizers[0], scalar_config, pre, text, device,
                          probes, oracle=True),
            _reload_exact(tmp_path, hybrid, optimizers[1], config, pre, text, device,
                          probes, oracle=False),
        ]
        maxima["reloaded_full_output"] = _compare_outputs(
            _outputs(loaded[1][0], probes), _outputs(loaded[0][0], probes),
            "both-reloaded scalar versus hybrid complete output")
        # Supplemental CPU observation only: the approved CUDA contract is the
        # exact checkpoint state/RNG/full-forward replay above, not bitwise
        # equality of a future backward/Adam trajectory.
        if device == "cpu":
            for original, optimizer, (restored, restored_optimizer) in zip(
                    (scalar, hybrid), optimizers, loaded):
                _resume_step_exact(original, restored, optimizer, restored_optimizer, jobs, probes)
            maxima["resumed_full_output"] = _compare_outputs(
                _outputs(loaded[1][0], probes), _outputs(loaded[0][0], probes),
                "both-resumed scalar versus hybrid complete output")
        print({"device": device, "branch": branch, "cases": len(cases), "backwards_per_step": len(jobs),
               "atol": ATOL, "rtol": RTOL, "measured_max_abs": maxima,
               "strict_numeric_pass": not reviewed_exceptions,
               "reviewed_numerical_exceptions": reviewed_exceptions,
               "all_later_checks_completed": True})
    # Deliberately deferred until every optimizer/output/reload/resume assertion
    # and backend-policy restoration have finished. Never hide a later failure.
    if reviewed_exceptions:
        assert len(reviewed_exceptions) == 1
        pytest.xfail(
            "reviewed user-accepted numerical exception: CPU PN-source first Adam "
            "model.frontend.trace_cnn.2.weight[63,23,2]; "
            f"abs={reviewed_exceptions[0]['absolute_difference']:.17g} <= 3.43e-5, "
            f"ratio={reviewed_exceptions[0]['tolerance_ratio']:.12g} <= 1.64; "
            "all later optimizer/output/checkpoint/exact-resume checks passed; "
            "not strict equivalence or CUDA/real-probe acceptance"
        )
