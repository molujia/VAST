"""Representation diagnostics on tiny synthetic vectors only."""

import importlib

import pytest


def api():
    try:
        return importlib.import_module("vast_v3.evaluation.diagnostics")
    except ModuleNotFoundError:
        pytest.fail("The diagnostic contract is not implemented yet")


def test_matched_representation_change_reports_repeat_baseline():
    report = api().representation_change(
        {"affected": [1, 0], "control": [0, 1]},
        {"control": [0, 1], "affected": [0, 1]},
        unchanged_repeat={"affected": [1, 0], "control": [0, 1]},
        affected_ids=["affected"],
    )
    assert report["affected_count"] == 1
    assert report["unaffected_count"] == 1
    assert report["rows"][0]["relative_l2"] == pytest.approx(2**0.5)
    assert report["max_repeat_absolute_l2"] == 0
    assert report["unaffected_mean_absolute_l2"] == 0
    assert report["supports_causal_claim"] is False


def test_zero_norm_is_defined_without_fake_observations():
    report = api().representation_change({"zero": [0, 0]}, {"zero": [0, 0]},
        unchanged_repeat={"zero": [0, 0]}, affected_ids=[])
    assert report["rows"][0]["relative_l2"] == 0
    assert report["affected_mean_absolute_l2"] is None


@pytest.mark.parametrize("changed,repeat,affected", [
    ({"different": [1, 0]}, {"one": [1, 0]}, []),
    ({"one": [1]}, {"one": [1, 0]}, []),
    ({"one": [float("nan"), 0]}, {"one": [1, 0]}, []),
    ({"one": [1, 0]}, {"one": [1, 0]}, ["missing"]),
])
def test_diagnostic_requires_exact_matching_observations(changed, repeat, affected):
    with pytest.raises(ValueError):
        api().representation_change({"one": [1, 0]}, changed,
                                     unchanged_repeat=repeat, affected_ids=affected)


def test_source_completion_gap_keeps_ineligible_masks_out_of_denominator():
    result = api().source_completion_gap([
        {"case_id": "one", "mask_id": "m1", "true_error": 0.2,
         "wrong_errors": [0.5, 0.7], "eligible": True},
        {"case_id": "one", "mask_id": "m2", "true_error": None,
         "wrong_errors": [], "eligible": False, "reason": "no context"},
        {"case_id": "two", "mask_id": "m1", "true_error": 0.5,
         "wrong_errors": [0.3], "eligible": True},
    ], expected_case_ids=["one", "two", "three"])
    assert result["eligible_case_count"] == 2
    assert result["total_case_count"] == 3
    assert result["eligible_mask_count"] == 2
    assert result["ineligible_mask_count"] == 1
    assert result["mean_case_gap"] == pytest.approx(0.1)
    assert result["positive_case_fraction"] == 0.5
    assert result["cases_without_eligible_masks"] == ["three"]
    assert result["localization_accuracy"] is None


def test_source_gap_empty_eligibility_is_not_zero_error():
    result = api().source_completion_gap([], expected_case_ids=["one"])
    assert result["mean_case_gap"] is None
    assert result["positive_case_fraction"] is None


@pytest.mark.parametrize("rows", [
    [{"case_id": "one", "mask_id": "m", "true_error": 0.1,
      "wrong_errors": [], "eligible": True}],
    [{"case_id": "other", "mask_id": "m", "true_error": 0.1,
      "wrong_errors": [0.2], "eligible": True}],
    [{"case_id": "one", "mask_id": "m", "true_error": -1,
      "wrong_errors": [0.2], "eligible": True}],
])
def test_source_gap_invalid_inputs_are_not_silently_dropped(rows):
    with pytest.raises(ValueError):
        api().source_completion_gap(rows, expected_case_ids=["one"])


def test_source_gap_duplicate_masks_cannot_double_weight_an_incident():
    row = {"case_id": "one", "mask_id": "m", "true_error": 0.1,
           "wrong_errors": [0.2], "eligible": True}
    with pytest.raises(ValueError):
        api().source_completion_gap([row, row], expected_case_ids=["one"])
