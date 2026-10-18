"""Observation-matched interface diagnostics, not claims of causal correctness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .metrics import _finite, _ids


def _vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(_finite(value, "embedding coordinate") for value in values)
    if not vector:
        raise ValueError("An embedding vector must not be empty")
    return vector


def representation_change(reference: Mapping[str, Sequence[float]],
                          changed: Mapping[str, Sequence[float]], *,
                          unchanged_repeat: Mapping[str, Sequence[float]],
                          affected_ids: Sequence[str]) -> dict:
    """Compare already inverse-permuted, provenance-matched vectors to a repeat.

The caller must carry observation IDs/random marks through perturbations. This
function rejects nonmatching inventories instead of deleting difficult matches.
"""
    inventory = _ids(tuple(reference), "reference inventory", allow_empty=True)
    if set(changed) != set(inventory) or set(unchanged_repeat) != set(inventory):
        raise ValueError("All diagnostic inventories must match exactly")
    affected = set(_ids(affected_ids, "affected IDs", allow_empty=True))
    if not affected <= set(inventory):
        raise ValueError("Affected IDs must belong to the matched inventory")
    rows = []
    for identity in sorted(inventory):
        base = _vector(reference[identity])
        other = _vector(changed[identity])
        repeat = _vector(unchanged_repeat[identity])
        if len(base) != len(other) or len(base) != len(repeat):
            raise ValueError("Matched embedding dimensions must agree")
        norm = math.sqrt(sum(value * value for value in base))
        absolute = math.dist(base, other)
        repeat_error = math.dist(base, repeat)
        rows.append({"observation_id": identity, "affected": identity in affected,
                     "absolute_l2": absolute, "relative_l2": absolute / max(norm, 1e-12),
                     "repeat_absolute_l2": repeat_error,
                     "repeat_relative_l2": repeat_error / max(norm, 1e-12)})
    def mean_for(flag):
        selected = [row["absolute_l2"] for row in rows if row["affected"] == flag]
        return sum(selected) / len(selected) if selected else None
    return {"rows": rows, "matched_count": len(rows), "affected_count": len(affected),
            "unaffected_count": len(rows) - len(affected),
            "affected_mean_absolute_l2": mean_for(True),
            "unaffected_mean_absolute_l2": mean_for(False),
            "max_repeat_absolute_l2": max((row["repeat_absolute_l2"] for row in rows),
                                          default=None),
            "supports_causal_claim": False}


def source_completion_gap(rows: Sequence[Mapping], *,
                          expected_case_ids: Sequence[str]) -> dict:
    """Compute mean_wrong E(R)-E(Y), averaging masks then incidents.

Inputs must be post-fit held-out diagnostic records with shared mask/hypothesis
contracts. This arithmetic helper does not authorize reading evaluation roots,
prove the frozen checkpoint's provenance, or compute localization accuracy.
"""
    expected = _ids(expected_case_ids, "expected_case_ids")
    gaps = {case: [] for case in expected}
    seen = set()
    eligible_masks = 0
    excluded = []
    for row in rows:
        case = row.get("case_id")
        mask = row.get("mask_id")
        if case not in gaps or not isinstance(mask, str) or not mask.strip():
            raise ValueError("Diagnostic case/mask must identify the frozen population")
        key = (case, mask)
        if key in seen:
            raise ValueError("Duplicate diagnostic case/mask")
        seen.add(key)
        if type(row.get("eligible")) is not bool:
            raise ValueError("Eligibility must be explicit")
        if not row["eligible"]:
            if not row.get("reason") or row.get("true_error") is not None or row.get("wrong_errors"):
                raise ValueError("Ineligible masks require a reason and no fabricated errors")
            excluded.append({"case_id": case, "mask_id": mask, "reason": row["reason"]})
            continue
        true = _finite(row.get("true_error"), "true-source error")
        wrong = tuple(_finite(value, "wrong-source error") for value in row.get("wrong_errors", ()))
        if true < 0 or not wrong or any(value < 0 for value in wrong):
            raise ValueError("Eligible masks require nonnegative true/wrong errors")
        gaps[case].append(sum(wrong) / len(wrong) - true)
        eligible_masks += 1
    case_gaps = {case: sum(values) / len(values) for case, values in gaps.items() if values}
    count = len(case_gaps)
    return {"case_gaps": case_gaps, "total_case_count": len(expected),
            "eligible_case_count": count, "eligible_mask_count": eligible_masks,
            "ineligible_mask_count": len(excluded), "ineligible_masks": excluded,
            "cases_without_eligible_masks": [case for case in expected if not gaps[case]],
            "mean_case_gap": sum(case_gaps.values()) / count if count else None,
            "positive_case_fraction": sum(gap > 0 for gap in case_gaps.values()) / count if count else None,
            "localization_accuracy": None, "supports_causal_claim": False}
