from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .service_continuous_pairwise_bridge import (
    ServiceContinuityPairwiseBridge,
    fit_service_continuity_pairwise_bridge,
)


class FinalPairwiseValidationError(ValueError):
    """Raised when final weighted pairwise supervision ownership drifts."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def fit_final_augmented_pairwise_base(
    *,
    dataset_id: str,
    training_mode: str,
    supervised_case_limit: int,
    real_training_rows: Sequence[Mapping[str, Any]],
    real_targets_by_case: Mapping[str, Sequence[Any]],
    synthetic_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    supervision_authority_sha256: str,
    generation_disposition: Mapping[str, Any],
) -> ServiceContinuityPairwiseBridge:
    """Fit the weighted augmented authority ranker for either supervision mode."""

    mode = str(training_mode)
    if mode not in {"query_only", "oracle_full"}:
        raise FinalPairwiseValidationError("invalid pairwise supervision mode")
    if (
        isinstance(supervised_case_limit, bool)
        or not isinstance(supervised_case_limit, int)
        or supervised_case_limit <= 0
        or len(real_targets_by_case) != supervised_case_limit
    ):
        raise FinalPairwiseValidationError("pairwise supervised cardinality drifted")
    if not _SHA256.fullmatch(str(supervision_authority_sha256)):
        raise FinalPairwiseValidationError("supervision authority SHA-256 drifted")
    fitted = fit_service_continuity_pairwise_bridge(
        dataset_id=dataset_id,
        arm_id="proxy_mode_cvae_compatible",
        augmentation_enabled=True,
        real_training_rows=real_training_rows,
        real_targets_by_case=real_targets_by_case,
        synthetic_rows=synthetic_rows,
        feature_columns=feature_columns,
        query_plan_sha256=supervision_authority_sha256,
        random_state=42,
        generation_disposition=generation_disposition,
        query_budget=supervised_case_limit,
    )
    audit_identity = {
        key: value
        for key, value in fitted.training_audit.items()
        if key != "training_audit_sha256"
    }
    audit_identity.update(
        {
            "schema_version": "final-rcl-augmented-pairwise-training-audit-v1",
            "training_mode": mode,
            "supervised_case_limit": supervised_case_limit,
            "supervision_authority_sha256": str(supervision_authority_sha256),
        }
    )
    return ServiceContinuityPairwiseBridge(
        bridge=fitted.bridge,
        training_audit={
            **audit_identity,
            "training_audit_sha256": _semantic_hash(audit_identity),
        },
    )


__all__ = ["FinalPairwiseValidationError", "fit_final_augmented_pairwise_base"]
