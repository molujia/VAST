from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class CVAEProvenanceValidationError(ValueError):
    """Raised when a generated CVAE row cannot be sealed provenance-first."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = {
    "cvae_arbitrary",
    "cvae_compatible",
    "proxy_mode_cvae_compatible",
}
_PARTITIONS = {"mechanism", "propagation", "context"}


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash(value: Any, context: str) -> str:
    result = str(value).strip()
    if not _SHA256.fullmatch(result):
        raise CVAEProvenanceValidationError(f"{context} SHA-256 is invalid")
    return result


def _nonempty(value: Any, context: str) -> str:
    result = str(value).strip()
    if not result:
        raise CVAEProvenanceValidationError(f"{context} must be nonempty")
    return result


def _source_posterior(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != _PARTITIONS:
        raise CVAEProvenanceValidationError("source posterior partition closure drift")
    result = {}
    for partition in sorted(_PARTITIONS):
        entry = value[partition]
        if not isinstance(entry, Mapping) or set(entry) != {
            "mu_sha256",
            "logvar_sha256",
        }:
            raise CVAEProvenanceValidationError(
                f"source posterior {partition} closure drift"
            )
        result[partition] = {
            "mu_sha256": _hash(entry["mu_sha256"], f"{partition} posterior mean"),
            "logvar_sha256": _hash(
                entry["logvar_sha256"], f"{partition} posterior log-variance"
            ),
        }
    return result


def _interpolation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "same_fault_type_partner_case_id",
        "alpha",
        "mechanism_latent_sha256",
        "propagation_latent_sha256",
        "context_latent_sha256",
    }:
        raise CVAEProvenanceValidationError("latent interpolation closure drift")
    partner_raw = value["same_fault_type_partner_case_id"]
    partner = None if partner_raw is None else _nonempty(partner_raw, "partner case")
    alpha_raw = value["alpha"]
    if alpha_raw is None:
        alpha = None
    elif (
        isinstance(alpha_raw, bool)
        or not isinstance(alpha_raw, (int, float))
        or not math.isfinite(float(alpha_raw))
        or not 0.0 <= float(alpha_raw) <= 1.0
    ):
        raise CVAEProvenanceValidationError("interpolation alpha must be in [0, 1]")
    else:
        alpha = float(alpha_raw)
    if (partner is None) != (alpha is None):
        raise CVAEProvenanceValidationError(
            "partner and interpolation alpha must be jointly absent or present"
        )
    return {
        "same_fault_type_partner_case_id": partner,
        "alpha": alpha,
        "mechanism_latent_sha256": _hash(
            value["mechanism_latent_sha256"], "mechanism latent"
        ),
        "propagation_latent_sha256": _hash(
            value["propagation_latent_sha256"], "propagation latent"
        ),
        "context_latent_sha256": _hash(
            value["context_latent_sha256"], "context latent"
        ),
    }


def _target_condition(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "target_service_id",
        "target_context_sha256",
        "target_context_mask_sha256",
    }:
        raise CVAEProvenanceValidationError("target condition closure drift")
    return {
        "target_service_id": _nonempty(
            value["target_service_id"], "target service ID"
        ),
        "target_context_sha256": _hash(
            value["target_context_sha256"], "target context"
        ),
        "target_context_mask_sha256": _hash(
            value["target_context_mask_sha256"], "target context mask"
        ),
    }


def _compatibility(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "score",
        "compatibility_sha256",
        "target_weight_plan_sha256",
    }:
        raise CVAEProvenanceValidationError("compatibility closure drift")
    score_raw = value["score"]
    if (
        isinstance(score_raw, bool)
        or not isinstance(score_raw, (int, float))
        or not math.isfinite(float(score_raw))
        or not 0.0 <= float(score_raw) <= 1.0
    ):
        raise CVAEProvenanceValidationError("compatibility score must be in [0, 1]")
    return {
        "score": float(score_raw),
        "compatibility_sha256": _hash(
            value["compatibility_sha256"], "compatibility"
        ),
        "target_weight_plan_sha256": _hash(
            value["target_weight_plan_sha256"], "target weight plan"
        ),
    }


def _activity(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CVAEProvenanceValidationError("activity evidence must be a mapping")
    status = str(value.get("status", "")).strip()
    if status != "mechanistically_active":
        raise CVAEProvenanceValidationError(
            "mechanistically inactive CVAE generation cannot be admitted"
        )
    return {
        "status": status,
        "activity_sha256": _hash(value.get("activity_sha256"), "activity evidence"),
    }


def build_cvae_row_provenance(
    *,
    arm_id: str,
    checkpoint_sha256: str,
    profile_id: str,
    profile_sha256: str,
    source_case_id: str,
    queried_label_sha256: str,
    source_posterior: Mapping[str, Mapping[str, Any]],
    latent_seed: int,
    interpolation: Mapping[str, Any],
    target_condition: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    decoded_state_sha256: str,
    training_weight: float,
    activity_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    arm = str(arm_id).strip()
    if arm not in _ARMS:
        raise CVAEProvenanceValidationError("unknown CVAE arm")
    if isinstance(latent_seed, bool) or not isinstance(latent_seed, int) or latent_seed < 0:
        raise CVAEProvenanceValidationError("latent seed must be nonnegative integer")
    if (
        isinstance(training_weight, bool)
        or not isinstance(training_weight, (int, float))
        or not math.isfinite(float(training_weight))
        or not 0.0 < float(training_weight) <= 1.0
    ):
        raise CVAEProvenanceValidationError("training weight must be in (0, 1]")
    identity = {
        "schema_version": "service-continuous-cvae-provenance-v1",
        "arm_id": arm,
        "checkpoint_sha256": _hash(checkpoint_sha256, "checkpoint"),
        "profile_id": _nonempty(profile_id, "profile ID"),
        "profile_sha256": _hash(profile_sha256, "profile"),
        "source_case_id": _nonempty(source_case_id, "source case ID"),
        "queried_label_sha256": _hash(queried_label_sha256, "queried label"),
        "source_posterior": _source_posterior(source_posterior),
        "latent_seed": latent_seed,
        "interpolation": _interpolation(interpolation),
        "target_condition": _target_condition(target_condition),
        "compatibility": _compatibility(compatibility),
        "decoded_state_sha256": _hash(decoded_state_sha256, "decoded state"),
        "training_weight": float(training_weight),
        "activity_evidence": _activity(deepcopy(dict(activity_evidence))),
    }
    return {**identity, "provenance_sha256": _semantic_hash(identity)}


__all__ = ["CVAEProvenanceValidationError", "build_cvae_row_provenance"]
