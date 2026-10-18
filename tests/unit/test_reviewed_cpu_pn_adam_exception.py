"""The reviewed numerical exception is one CPU fixture element, never a tolerance policy."""
import importlib.util
from pathlib import Path

import pytest
import torch


INDEX = (63, 23, 2)
CONTEXT = dict(tensor_name="model.frontend.trace_cnn.2.weight",
               stage="first-post-Adam-full-state", device="cpu", branch="PN-source")


def helper():
    path = Path(__file__).parents[1] / "integration" / "reviewed_cpu_pn_adam_exception.py"
    assert path.is_file(), "reviewed CPU PN Adam exception helper is not implemented"
    spec = importlib.util.spec_from_file_location("reviewed_cpu_pn_adam_exception_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.assert_initial_adam_tensor_close


def known_pair():
    expected = torch.zeros(64, 64, 3)
    actual = expected.clone()
    expected[INDEX] = .00445557851344347
    actual[INDEX] = .004421328194439411
    return actual, expected


def test_only_known_single_element_returns_reviewed_diagnostic():
    compare = helper()
    actual, expected = known_pair()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    diagnostic = compare(actual, expected, **CONTEXT)
    assert diagnostic["tensor_name"] == CONTEXT["tensor_name"]
    assert diagnostic["index"] == INDEX
    assert diagnostic["absolute_difference"] == pytest.approx(3.425031900405884e-5)
    assert diagnostic["tolerance_ratio"] == pytest.approx(1.639468193)
    assert "Mismatched elements: 1 / 12288" in diagnostic["strict_assertion"]


@pytest.mark.parametrize("change", [
    {"tensor_name": "model.frontend.metric_cnn.2.weight"},
    {"tensor_name": "model.frontend.trace_cnn.0.weight"},
    {"stage": "resumed-model-update"},
    {"stage": "post-Adam-optimizer-state"},
    {"device": "cuda"},
    {"branch": "U"},
    {"branch": "mixed"},
])
def test_wrong_scope_does_not_receive_exception(change):
    compare = helper()
    actual, expected = known_pair()
    with pytest.raises(AssertionError):
        compare(actual, expected, **(CONTEXT | change))


def test_different_single_coordinate_does_not_receive_exception():
    compare = helper()
    actual, expected = known_pair()
    actual[0, 0, 0], expected[0, 0, 0] = actual[INDEX].clone(), expected[INDEX].clone()
    actual[INDEX] = expected[INDEX]
    with pytest.raises(AssertionError):
        compare(actual, expected, **CONTEXT)


def test_second_out_of_tolerance_element_is_not_hidden():
    compare = helper()
    actual, expected = known_pair()
    actual[0, 0, 0] = 2.01e-5
    with pytest.raises(AssertionError):
        compare(actual, expected, **CONTEXT)


@pytest.mark.parametrize("reference,difference", [(.01, 3.44e-5), (0., 3.29e-5)])
def test_absolute_and_ratio_limits_are_independently_enforced(reference, difference):
    compare = helper()
    actual, expected = known_pair()
    expected[INDEX] = reference
    actual[INDEX] = reference - difference
    with pytest.raises(AssertionError):
        compare(actual, expected, **CONTEXT)


@pytest.mark.parametrize("side", ["actual", "expected"])
@pytest.mark.parametrize("index", [INDEX, (0, 0, 0)])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_every_element_on_both_sides_must_be_finite(side, index, value):
    compare = helper()
    actual, expected = known_pair()
    {"actual": actual, "expected": expected}[side][index] = value
    with pytest.raises(AssertionError, match="finite"):
        compare(actual, expected, **CONTEXT)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_tensor_structure_mismatch_is_not_accepted(mismatch):
    compare = helper()
    actual, expected = known_pair()
    actual = actual.reshape(-1) if mismatch == "shape" else actual.double()
    with pytest.raises(AssertionError):
        compare(actual, expected, **CONTEXT)


@pytest.mark.parametrize("context", [CONTEXT, CONTEXT | {"branch": "U", "device": "cuda"}])
def test_strict_pass_returns_no_exception_even_without_approved_scope(context):
    compare = helper()
    _, expected = known_pair()
    actual = expected.clone()
    actual[INDEX] += 1e-5
    assert compare(actual, expected, **context) is None


def test_other_elements_keep_original_tolerance_when_exception_is_used():
    compare = helper()
    actual, expected = known_pair()
    actual[0, 0, 0] = 1.99e-5
    assert compare(actual, expected, **CONTEXT)["index"] == INDEX


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tensors_cannot_be_approved_by_spoofing_cpu_context():
    compare = helper()
    actual, expected = known_pair()
    with pytest.raises(AssertionError):
        compare(actual.cuda(), expected.cuda(), **CONTEXT)
