"""Case-equal supervision weights for queried and pseudo fault cases."""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd


def prepare_case_balanced_pseudo_frame(
    training_frame: pd.DataFrame,
    *,
    pseudo_to_query_loss_ratio: float = 0.25,
) -> pd.DataFrame:
    ratio = float(pseudo_to_query_loss_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(
            "pseudo_to_query_loss_ratio must be finite and non-negative"
        )
    required = {
        "window_id",
        "label",
        "label_source",
        "sample_weight",
    }
    missing = sorted(required.difference(training_frame.columns))
    if missing:
        raise ValueError(
            "case-balanced frame missing columns: %s" % missing
        )
    frame = training_frame[
        training_frame["label"].astype(int) >= 0
    ].copy()
    if frame.empty:
        raise ValueError("case-balanced training frame must not be empty")
    frame["window_id"] = frame["window_id"].astype(str)

    case_types: Dict[str, str] = {}
    pseudo_confidence: Dict[str, float] = {}
    for window_id, group in frame.groupby("window_id", sort=False):
        if int((group["label"].astype(int) == 1).sum()) <= 0:
            raise ValueError(
                "case %s has no positive target mass" % window_id
            )
        sources = set(group["label_source"].astype(str))
        has_query = any(
            source in {"queried", "oracle", "authoritative"}
            for source in sources
        )
        has_pseudo = any(source.startswith("pseudo_") for source in sources)
        if has_query == has_pseudo:
            raise ValueError(
                "case %s has ambiguous supervision source" % window_id
            )
        if has_query:
            case_types[str(window_id)] = "queried"
            continue
        case_types[str(window_id)] = "pseudo"
        if "pseudo_case_confidence" in group.columns:
            values = {
                float(value)
                for value in group["pseudo_case_confidence"].tolist()
            }
        else:
            values = {
                float(value)
                for value in group["sample_weight"].tolist()
            }
        if len(values) != 1:
            raise ValueError(
                "pseudo case confidence must be constant within a case"
            )
        confidence = next(iter(values))
        if not math.isfinite(confidence) or confidence <= 0.0:
            raise ValueError("pseudo case confidence must be positive")
        pseudo_confidence[str(window_id)] = confidence

    query_ids = sorted(
        case_id
        for case_id, case_type in case_types.items()
        if case_type == "queried"
    )
    pseudo_ids = sorted(
        case_id
        for case_id, case_type in case_types.items()
        if case_type == "pseudo"
    )
    if not query_ids:
        raise ValueError(
            "case-balanced pseudo training requires queried fault cases"
        )
    case_weights = {case_id: 1.0 for case_id in query_ids}
    target_pseudo_mass = ratio * float(len(query_ids))
    confidence_mass = sum(
        pseudo_confidence[case_id] for case_id in pseudo_ids
    )
    if pseudo_ids:
        if confidence_mass <= 0.0:
            raise ValueError("pseudo confidence mass must be positive")
        scale = target_pseudo_mass / confidence_mass
        for case_id in pseudo_ids:
            case_weights[case_id] = pseudo_confidence[case_id] * scale

    frame["case_loss_weight"] = frame["window_id"].map(case_weights)
    counts = frame.groupby("window_id")["window_id"].transform("count")
    frame["sample_weight"] = frame["case_loss_weight"] / counts.astype(float)
    query_mass = sum(case_weights[case_id] for case_id in query_ids)
    pseudo_mass = sum(case_weights[case_id] for case_id in pseudo_ids)
    diagnostics = {
        "schema_version": "case-balanced-pseudo-diagnostics-v1",
        "query_case_count": len(query_ids),
        "pseudo_case_count": len(pseudo_ids),
        "query_case_loss_mass": query_mass,
        "pseudo_case_loss_mass": pseudo_mass,
        "configured_pseudo_to_query_loss_ratio": ratio,
        "effective_pseudo_to_query_loss_ratio": (
            pseudo_mass / query_mass if query_mass > 0.0 else 0.0
        ),
        "case_equal_before_confidence": True,
        "candidate_row_count_invariant": True,
    }
    frame.attrs["case_balance_diagnostics"] = diagnostics
    return frame.reset_index(drop=True)


__all__ = ["prepare_case_balanced_pseudo_frame"]
