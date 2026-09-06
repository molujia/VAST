from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any


class CVAEActivityValidationError(ValueError):
    """Raised when a CVAE activity audit cannot be reconstructed."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THRESHOLDS = {
    "minimum_reconstruction_improvement": 1e-4,
    "minimum_mean_kl": 1e-6,
    "minimum_mean_posterior_variance": 1e-4,
    "maximum_source_copy_rate": 0.95,
    "minimum_target_context_delta": 1e-5,
    "minimum_unique_generated_states": 2,
    "mechanism_radius_tolerance": 1e-6,
}


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numbers(value: Any, context: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise CVAEActivityValidationError(f"{context} must be a nonempty sequence")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CVAEActivityValidationError(f"{context} must be numeric")
        result.append(float(item))
    return result


def _hashes(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise CVAEActivityValidationError(f"{context} must be a nonempty sequence")
    result = [str(item).strip() for item in value]
    if any(not _SHA256.fullmatch(item) for item in result):
        raise CVAEActivityValidationError(f"{context} must contain SHA-256 values")
    return result


def build_cvae_activity_audit(
    *,
    profile_id: str,
    gradient_norms: Sequence[Any],
    reconstruction_history: Sequence[Any],
    kl_history: Sequence[Any],
    posterior_logvars: Sequence[Any],
    source_state_sha256s: Sequence[Any],
    generated_state_sha256s: Sequence[Any],
    target_context_deltas: Sequence[Any],
    mechanism_latent_distances: Sequence[Any],
    sampling_radius: float,
) -> dict[str, Any]:
    profile = str(profile_id).strip()
    if not profile:
        raise CVAEActivityValidationError("profile_id must be nonempty")
    gradients = _numbers(gradient_norms, "gradient_norms")
    reconstruction = _numbers(reconstruction_history, "reconstruction_history")
    kl_values = _numbers(kl_history, "kl_history")
    logvars = _numbers(posterior_logvars, "posterior_logvars")
    sources = _hashes(source_state_sha256s, "source_state_sha256s")
    generated = _hashes(generated_state_sha256s, "generated_state_sha256s")
    target_deltas = _numbers(target_context_deltas, "target_context_deltas")
    mechanism_distances = _numbers(
        mechanism_latent_distances, "mechanism_latent_distances"
    )
    if len(sources) != len(generated):
        raise CVAEActivityValidationError(
            "source and generated state ledgers must align"
        )
    if (
        isinstance(sampling_radius, bool)
        or not isinstance(sampling_radius, (int, float))
        or not math.isfinite(float(sampling_radius))
        or float(sampling_radius) <= 0.0
    ):
        raise CVAEActivityValidationError("sampling_radius must be positive finite")
    radius = float(sampling_radius)

    finite_gradients = all(math.isfinite(value) for value in gradients)
    finite_objectives = all(
        math.isfinite(value)
        for values in (reconstruction, kl_values, logvars, target_deltas)
        for value in values
    )
    finite_distances = all(math.isfinite(value) for value in mechanism_distances)
    reconstruction_improvement = (
        reconstruction[0] - reconstruction[-1] if finite_objectives else 0.0
    )
    mean_kl = (
        math.fsum(kl_values) / len(kl_values) if finite_objectives else 0.0
    )
    posterior_variances = (
        [math.exp(min(20.0, max(-60.0, value))) for value in logvars]
        if finite_objectives
        else [0.0]
    )
    mean_posterior_variance = math.fsum(posterior_variances) / len(
        posterior_variances
    )
    source_copy_count = sum(
        source_hash == generated_hash
        for source_hash, generated_hash in zip(sources, generated)
    )
    source_copy_rate = source_copy_count / len(generated)
    target_context_delta_mean = (
        math.fsum(abs(value) for value in target_deltas) / len(target_deltas)
        if finite_objectives
        else 0.0
    )
    unique_generated_count = len(set(generated))
    maximum_mechanism_distance = (
        max(mechanism_distances) if finite_distances else float("inf")
    )

    reconstruction_active = (
        finite_objectives
        and reconstruction_improvement
        > _THRESHOLDS["minimum_reconstruction_improvement"]
    )
    kl_active = finite_objectives and mean_kl > _THRESHOLDS["minimum_mean_kl"]
    posterior_active = (
        finite_objectives
        and mean_posterior_variance
        > _THRESHOLDS["minimum_mean_posterior_variance"]
    )
    target_sensitive = (
        finite_objectives
        and target_context_delta_mean
        > _THRESHOLDS["minimum_target_context_delta"]
    )
    mechanism_radius_respected = (
        finite_distances
        and maximum_mechanism_distance
        <= radius + _THRESHOLDS["mechanism_radius_tolerance"]
    )

    rejection_reasons = []
    if not finite_gradients:
        rejection_reasons.append("non_finite_gradients")
    if not finite_objectives:
        rejection_reasons.append("non_finite_objectives")
    if not reconstruction_active:
        rejection_reasons.append("reconstruction_not_improved")
    if not kl_active:
        rejection_reasons.append("kl_inactive")
    if not posterior_active:
        rejection_reasons.append("posterior_collapsed")
    if source_copy_rate >= _THRESHOLDS["maximum_source_copy_rate"]:
        rejection_reasons.append("source_copy_only")
    if not target_sensitive:
        rejection_reasons.append("target_context_insensitive")
    if unique_generated_count < _THRESHOLDS["minimum_unique_generated_states"]:
        rejection_reasons.append("generated_state_diversity_absent")
    if not mechanism_radius_respected:
        rejection_reasons.append("mechanism_radius_violated")

    finite_gradient_values = [value for value in gradients if math.isfinite(value)]
    identity = {
        "schema_version": "service-continuous-cvae-activity-audit-v1",
        "profile_id": profile,
        "status": (
            "mechanistically_active" if not rejection_reasons else "mechanistically_inactive"
        ),
        "rejection_reasons": rejection_reasons,
        "thresholds": dict(_THRESHOLDS),
        "finite_gradients": finite_gradients,
        "finite_gradient_count": len(finite_gradient_values),
        "non_finite_gradient_count": len(gradients) - len(finite_gradient_values),
        "maximum_finite_gradient_norm": (
            max(finite_gradient_values) if finite_gradient_values else None
        ),
        "finite_objectives": finite_objectives,
        "reconstruction_improvement": reconstruction_improvement,
        "reconstruction_improved": reconstruction_active,
        "mean_kl": mean_kl,
        "kl_active": kl_active,
        "mean_posterior_variance": mean_posterior_variance,
        "posterior_variance_active": posterior_active,
        "source_copy_count": source_copy_count,
        "source_copy_rate": source_copy_rate,
        "target_context_delta_mean": target_context_delta_mean,
        "target_context_sensitive": target_sensitive,
        "unique_generated_state_count": unique_generated_count,
        "generated_state_diversity_nonzero": (
            unique_generated_count >= _THRESHOLDS["minimum_unique_generated_states"]
        ),
        "sampling_radius": radius,
        "maximum_mechanism_latent_distance": (
            maximum_mechanism_distance if finite_distances else None
        ),
        "mechanism_radius_respected": mechanism_radius_respected,
    }
    return {**identity, "activity_sha256": _semantic_hash(identity)}


__all__ = ["CVAEActivityValidationError", "build_cvae_activity_audit"]
