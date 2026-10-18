import importlib

import pytest
import torch
import torch.nn.functional as F


def api():
    try:
        return importlib.import_module("vast_v3.training.losses")
    except ModuleNotFoundError:
        pytest.fail("Loss implementation is missing")


def test_class_normalized_bce_preserves_class_not_candidate_mass():
    scores = torch.tensor([1.0, -2.0, -3.0, 4.0], requires_grad=True)
    targets = torch.tensor([True, False, False, False])
    expected = (F.softplus(-scores[0]) + F.softplus(scores[1:]).mean()) / 2
    torch.testing.assert_close(api().class_normalized_bce(scores, targets), expected)
    torch.testing.assert_close(api().class_normalized_bce(scores, torch.ones(4, dtype=torch.bool)),
                               F.softplus(-scores).mean())
    assert api().class_normalized_bce(torch.empty(0), torch.empty(0, dtype=torch.bool)) is None


def record(modality, unit, prediction, eligible=None):
    return dict(modality=modality, unit=unit, prediction=prediction,
                target=torch.zeros(prediction.shape[-2:]),
                eligible=(torch.ones(prediction.shape[-2], dtype=torch.bool)
                          if eligible is None else eligible))


def test_recovery_averages_dimension_then_entry_unit_modality():
    records = [record("metric", "node0", torch.tensor([[1.0], [1.0], [1.0]])),
               record("metric", "node1", torch.tensor([[3.0]])),
               record("text", "node0", torch.tensor([[0.0, 2.0]]))]
    # metric = (.5 + 2.5)/2 = 1.5; text = (0 + 1.5)/2 = .75.
    torch.testing.assert_close(api().hierarchical_recovery_loss(records), torch.tensor(1.125))


def test_recovery_masks_have_shared_hypothesis_eligibility_and_no_zero_fakes():
    prediction = torch.tensor([[[1.0], [float("nan")]],
                               [[3.0], [float("nan")]]], requires_grad=True)
    rec = record("trace", (0, 1, "call"), prediction, torch.tensor([True, False]))
    result = api().hierarchical_recovery_loss([rec])
    torch.testing.assert_close(result, torch.tensor([.5, 2.5]))
    result.sum().backward()
    assert prediction.grad[:, 1].eq(0).all()
    rec["eligible"] = torch.tensor([False, False])
    assert api().hierarchical_recovery_loss([rec]) is None
    assert api().hierarchical_recovery_loss([]) is None
    rec["eligible"] = torch.tensor([[True, False], [False, True]])
    with pytest.raises(ValueError, match="shared|eligible"):
        api().hierarchical_recovery_loss([rec])


def test_ineligible_units_and_modalities_do_not_dilute_denominator():
    good = record("metric", 0, torch.tensor([[2.0]]))
    absent = record("metric", 1, torch.tensor([[999.0]]), torch.tensor([False]))
    absent_modality = record("trace", (0, 1), torch.tensor([[999.0]]), torch.tensor([False]))
    torch.testing.assert_close(api().hierarchical_recovery_loss([good, absent, absent_modality]),
                               torch.tensor(1.5))
    with pytest.raises(ValueError, match="duplicate|once"):
        api().hierarchical_recovery_loss([good, good])


def test_l_source_loss_softplus_and_empty_wrong_library():
    truth = torch.tensor(2.0, requires_grad=True)
    wrong = torch.tensor([3.0, 4.0], requires_grad=True)
    torch.testing.assert_close(api().labeled_source_loss(truth, wrong),
                               truth + F.softplus(truth - wrong).mean())
    assert api().labeled_source_loss(truth, torch.empty(0)) is truth
    assert api().labeled_source_loss(None, None) is None


def test_u_logsumexp_retains_score_and_error_gradients():
    scores = torch.tensor([.3, -.7, 1.1], dtype=torch.float64, requires_grad=True)
    sets = torch.eye(3, dtype=torch.bool)
    errors = torch.tensor([.1, .8, 2.0], dtype=torch.float64, requires_grad=True)
    temperature = .7
    loss = api().unlabeled_source_loss(scores, sets, errors, T_E=temperature)
    q = torch.softmax(scores - .5, 0)
    revised = torch.softmax(torch.log(q) - errors / temperature, 0)
    loss.backward()
    torch.testing.assert_close(scores.grad, temperature * (q - revised))
    torch.testing.assert_close(errors.grad, revised)
    assert torch.count_nonzero(scores.grad) > 0


def test_u_multisource_cardinality_penalty_and_numerical_stability():
    scores = torch.tensor([10000.0, 10001.0], requires_grad=True)
    sets = torch.tensor([[True, False], [False, True], [True, True]])
    errors = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
    loss = api().unlabeled_source_loss(scores, sets, errors, lambda_c=.5)
    assert torch.isfinite(loss)
    assert abs(loss.item() - 2.0) < .01
    loss.backward()
    assert scores.grad is not None and errors.grad is not None


def test_u_empty_and_invalid_inputs_are_explicit():
    assert api().unlabeled_source_loss(torch.zeros(2), torch.empty(0, 2, dtype=torch.bool),
                                        torch.empty(0)) is None
    assert api().unlabeled_source_loss(torch.zeros(2), torch.eye(2, dtype=torch.bool), None) is None
    with pytest.raises(ValueError, match="temperature|T_E"):
        api().unlabeled_source_loss(torch.zeros(2), torch.eye(2, dtype=torch.bool), torch.ones(2), T_E=0)


def test_eligible_case_means_and_fixed_observation_component_weights():
    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    torch.testing.assert_close(api().eligible_mean([a, None, b]), torch.tensor(2.0))
    assert api().eligible_mean([None, None]) is None
    torch.testing.assert_close(api().observation_loss(a, None, b), torch.tensor(4 / 3))
    assert api().observation_loss(None, None, None) is None


def test_nonfinite_target_is_missing_but_nonfinite_prediction_is_error():
    rec = record("metric", 0, torch.tensor([[2.0], [9.0]], requires_grad=True))
    rec["target"][1] = float("nan")
    torch.testing.assert_close(api().hierarchical_recovery_loss([rec]), torch.tensor(1.5))
    rec["prediction"] = torch.tensor([[float("nan")], [9.0]])
    with pytest.raises(ValueError, match="finite"):
        api().hierarchical_recovery_loss([rec])


def test_recovery_fixed_targets_do_not_receive_gradients():
    prediction = torch.tensor([[2.0]], requires_grad=True)
    rec = record("text", 0, prediction)
    rec["target"].requires_grad_()
    api().hierarchical_recovery_loss([rec]).backward()
    assert prediction.grad.item() == 1
    assert rec["target"].grad is None


def test_nonfinite_cardinality_penalty_is_rejected():
    with pytest.raises(ValueError, match="lambda_c|penalty"):
        api().unlabeled_source_loss(torch.zeros(2), torch.eye(2, dtype=torch.bool),
                                    torch.ones(2), lambda_c=float("nan"))


def test_partial_missing_channels_average_only_observed_coordinates():
    prediction = torch.tensor([[[1.0, float("nan")], [1.0, 3.0], [float("nan"), float("nan")]],
                               [[2.0, float("nan")], [2.0, 4.0], [float("nan"), float("nan")]]],
                              requires_grad=True)
    rec = record("metric", 0, prediction)
    rec["target"] = torch.tensor([[0.0, float("nan")], [0.0, 0.0], [float("nan"), float("nan")]])
    result = api().hierarchical_recovery_loss([rec])
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
    result.sum().backward()
    assert prediction.grad[:, 0, 1].eq(0).all()
    assert prediction.grad[:, 2].eq(0).all()
    assert prediction.grad[:, 0, 0].gt(0).all()
