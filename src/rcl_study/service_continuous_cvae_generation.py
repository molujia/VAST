from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from .service_continuous_cvae import FactorizedConditionalVAE, sample_posterior
from .service_continuous_cvae_provenance import build_cvae_row_provenance


class CVAEGenerationValidationError(ValueError):
    """Raised when conditional-VAE generation violates label or latent safety."""


_ARMS = ("cvae_arbitrary", "cvae_compatible")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_TENSORS = (
    "mechanism",
    "mechanism_mask",
    "propagation",
    "propagation_mask",
    "context",
    "context_mask",
)


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ids(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CVAEGenerationValidationError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise CVAEGenerationValidationError(
            f"{context} must contain unique nonempty IDs"
        )
    return result


def _tensor(value: Any, context: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] != 1
        or not torch.is_floating_point(value)
        or not torch.isfinite(value).all()
    ):
        raise CVAEGenerationValidationError(
            f"{context} must be one finite floating-point row"
        )
    return value


def _validated_label(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "root_cause",
        "fault_type",
        "label_source",
        "budget_cost",
    }:
        raise CVAEGenerationValidationError(f"{context} queried label closure drift")
    label = deepcopy(dict(value))
    ownership = (label["label_source"], label["budget_cost"])
    if (
        not str(label["root_cause"]).strip()
        or not str(label["fault_type"]).strip()
        or ownership not in {("queried_budget", 1), ("oracle_full", 0)}
    ):
        raise CVAEGenerationValidationError(
            f"{context} requires query-owned or oracle-full supervision"
        )
    return label


def _source_case(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CVAEGenerationValidationError(f"{context} must be a mapping")
    case_id = str(value.get("case_id", "")).strip()
    if not case_id:
        raise CVAEGenerationValidationError(f"{context} case_id must be nonempty")
    label = _validated_label(value.get("queried_label"), context)
    observations = _ids(value.get("observation_case_ids"), f"{context} observations")
    label_ids = _ids(value.get("label_case_ids"), f"{context} label IDs")
    if case_id not in observations or set(label_ids) != {case_id}:
        raise CVAEGenerationValidationError(
            f"{context} cannot borrow labels from an unqueried neighbor"
        )
    tensors = {name: _tensor(value.get(name), f"{context}.{name}") for name in _SOURCE_TENSORS}
    for base in ("mechanism", "propagation", "context"):
        if tensors[base].shape != tensors[f"{base}_mask"].shape:
            raise CVAEGenerationValidationError(f"{context}.{base} mask shape drift")
        if bool(torch.any((tensors[f"{base}_mask"] < 0.0) | (tensors[f"{base}_mask"] > 1.0))):
            raise CVAEGenerationValidationError(f"{context}.{base} mask range drift")
    return {
        "case_id": case_id,
        "queried_label": label,
        "observation_case_ids": observations,
        "label_case_ids": label_ids,
        **tensors,
    }


def _profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "profile_id",
        "sampling_radius",
        "samples_per_target",
        "profile_sha256",
    }:
        raise CVAEGenerationValidationError("CVAE generation profile closure drift")
    profile_id = str(value.get("profile_id", "")).strip()
    radius = value.get("sampling_radius")
    samples = value.get("samples_per_target")
    profile_hash = str(value.get("profile_sha256", ""))
    if (
        not profile_id
        or isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius))
        or float(radius) <= 0.0
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
        or not _SHA256.fullmatch(profile_hash)
    ):
        raise CVAEGenerationValidationError("CVAE generation profile values drift")
    return {
        "profile_id": profile_id,
        "sampling_radius": float(radius),
        "samples_per_target": samples,
        "profile_sha256": profile_hash,
    }


def _targets(value: Any, arm_id: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise CVAEGenerationValidationError("target_contexts must be a nonempty sequence")
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise CVAEGenerationValidationError(f"target_contexts[{index}] must be a mapping")
        target = str(raw.get("target_service_id", "")).strip()
        target_context = _tensor(raw.get("target_context"), f"target {target} context")
        target_mask = _tensor(raw.get("target_context_mask"), f"target {target} mask")
        weight = raw.get("target_weight")
        compatibility = raw.get("compatibility")
        weight_hash = str(raw.get("target_weight_plan_sha256", ""))
        compatibility_hash = str(raw.get("compatibility_sha256", ""))
        if (
            not target
            or target in seen
            or target_context.shape != target_mask.shape
            or bool(torch.any((target_mask < 0.0) | (target_mask > 1.0)))
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not 0.0 < float(weight) <= 1.0
            or isinstance(compatibility, bool)
            or not isinstance(compatibility, (int, float))
            or not 0.0 <= float(compatibility) <= 1.0
            or not _SHA256.fullmatch(weight_hash)
            or not _SHA256.fullmatch(compatibility_hash)
        ):
            raise CVAEGenerationValidationError("target context or weight-plan drift")
        seen.add(target)
        result.append(
            {
                "target_service_id": target,
                "target_context": target_context,
                "target_context_mask": target_mask,
                "target_weight": float(weight),
                "compatibility": float(compatibility),
                "target_weight_plan_sha256": weight_hash,
                "compatibility_sha256": compatibility_hash,
            }
        )
    if not math.isclose(math.fsum(row["target_weight"] for row in result), 1.0, abs_tol=1e-8):
        raise CVAEGenerationValidationError("target weights must sum to one")
    if arm_id == "cvae_arbitrary":
        first = result[0]["target_weight"]
        if any(not math.isclose(row["target_weight"], first, abs_tol=1e-8) for row in result):
            raise CVAEGenerationValidationError(
                "cvae_arbitrary requires equal normalized target mass"
            )
    return result


def _tensor_hash(value: Tensor) -> str:
    return _semantic_hash(
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.detach().cpu().tolist(),
        }
    )


def _bounded_mechanism_sample(
    mu: Tensor, logvar: Tensor, *, radius: float, seed: int
) -> tuple[Tensor, float]:
    proposed = sample_posterior(mu, logvar, seed=seed)
    delta = proposed - mu
    distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(radius / distance.clamp_min(1e-12), max=1.0)
    bounded = mu + delta * scale
    bounded_distance = float(torch.linalg.vector_norm(bounded - mu).item())
    return bounded, bounded_distance


def _encode(
    model: FactorizedConditionalVAE,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    return model(
        mechanism=source["mechanism"],
        mechanism_mask=source["mechanism_mask"],
        propagation=source["propagation"],
        propagation_mask=source["propagation_mask"],
        context=source["context"],
        context_mask=source["context_mask"],
        target_context=target["target_context"],
        target_context_mask=target["target_context_mask"],
        sample=False,
    )


def _decode(
    model: FactorizedConditionalVAE,
    *,
    mechanism_z: Tensor,
    propagation_z: Tensor,
    context_z: Tensor,
    target: Mapping[str, Any],
) -> dict[str, Tensor]:
    target_embedding = model.target_context_encoder(
        torch.cat(
            (
                target["target_context"] * target["target_context_mask"],
                target["target_context_mask"],
            ),
            dim=1,
        )
    )
    decoded = model.decoder(
        torch.cat((mechanism_z, propagation_z, context_z, target_embedding), dim=1)
    )
    mechanism_end = model.dimensions["mechanism_dim"]
    propagation_end = mechanism_end + model.dimensions["propagation_dim"]
    result = {
        "mechanism": decoded[:, :mechanism_end],
        "propagation": decoded[:, mechanism_end:propagation_end],
        "context": decoded[:, propagation_end:],
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise CVAEGenerationValidationError("decoded CVAE state must remain finite")
    return result


def generate_cvae_service_continuity(
    *,
    model: FactorizedConditionalVAE,
    arm_id: str,
    source_case: Mapping[str, Any],
    target_contexts: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    sample_seed: int,
    checkpoint_sha256: str,
    activity_evidence: Mapping[str, Any],
    same_fault_type_partner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arm = str(arm_id).strip()
    if arm not in _ARMS or not isinstance(model, FactorizedConditionalVAE):
        raise CVAEGenerationValidationError("unknown CVAE generation arm or model")
    source = _source_case(source_case, "source case")
    frozen_profile = _profile(profile)
    targets = _targets(target_contexts, arm)
    seed = int(sample_seed)
    if seed < 0:
        raise CVAEGenerationValidationError("sample_seed must be nonnegative")

    partner = None
    if same_fault_type_partner is not None:
        partner = _source_case(same_fault_type_partner, "same-fault partner")
        if partner["case_id"] == source["case_id"]:
            raise CVAEGenerationValidationError("same-fault partner must be a different query")
        if partner["queried_label"]["fault_type"] != source["queried_label"]["fault_type"]:
            raise CVAEGenerationValidationError(
                "cross-fault labeled interpolation is forbidden"
            )

    source_encoded = _encode(model, source, targets[0])["posterior"]
    source_posterior_identity = {
        partition: {
            "mu_sha256": _tensor_hash(source_encoded[partition]["mu"]),
            "logvar_sha256": _tensor_hash(source_encoded[partition]["logvar"]),
        }
        for partition in ("mechanism", "propagation", "context")
    }
    partner_encoded = (
        _encode(model, partner, targets[0])["posterior"] if partner is not None else None
    )
    source_mechanism_mu = source_encoded["mechanism"]["mu"]
    interpolation_center = source_mechanism_mu
    if partner_encoded is not None:
        interpolation_center = 0.5 * (
            source_mechanism_mu + partner_encoded["mechanism"]["mu"]
        )
    # The total displacement, including optional within-type interpolation,
    # remains bounded around the source posterior mean.
    mechanism_logvar = source_encoded["mechanism"]["logvar"]
    center_delta = interpolation_center - source_mechanism_mu
    center_distance = torch.linalg.vector_norm(center_delta, dim=1, keepdim=True)
    center_scale = torch.clamp(
        frozen_profile["sampling_radius"] / center_distance.clamp_min(1e-12),
        max=1.0,
    )
    bounded_center = source_mechanism_mu + center_delta * center_scale

    rows = []
    for target_index, target in enumerate(targets):
        for sample_index in range(frozen_profile["samples_per_target"]):
            row_seed = seed + target_index * 1009 + sample_index * 17
            proposed_mechanism = sample_posterior(
                bounded_center, mechanism_logvar, seed=row_seed
            )
            delta = proposed_mechanism - source_mechanism_mu
            distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            scale = torch.clamp(
                frozen_profile["sampling_radius"] / distance.clamp_min(1e-12),
                max=1.0,
            )
            mechanism_z = source_mechanism_mu + delta * scale
            mechanism_distance = float(
                torch.linalg.vector_norm(mechanism_z - source_mechanism_mu).item()
            )
            propagation_z = sample_posterior(
                source_encoded["propagation"]["mu"],
                source_encoded["propagation"]["logvar"],
                seed=row_seed + 1,
            )
            context_z = sample_posterior(
                source_encoded["context"]["mu"],
                source_encoded["context"]["logvar"],
                seed=row_seed + 2,
            )
            decoded = _decode(
                model,
                mechanism_z=mechanism_z,
                propagation_z=propagation_z,
                context_z=context_z,
                target=target,
            )
            decoded_identity = {
                name: _tensor_hash(value) for name, value in decoded.items()
            }
            row_identity = {
                "schema_version": "service-continuous-cvae-synthetic-row-v1",
                "arm_id": arm,
                "source_case_id": source["case_id"],
                "target_service_id": target["target_service_id"],
                "sample_index": sample_index,
                "latent_seed": row_seed,
                "profile_id": frozen_profile["profile_id"],
                "profile_sha256": frozen_profile["profile_sha256"],
                "mechanism_latent_distance": mechanism_distance,
                "mechanism_latent_sha256": _tensor_hash(mechanism_z),
                "propagation_latent_sha256": _tensor_hash(propagation_z),
                "context_latent_sha256": _tensor_hash(context_z),
                "decoded_state_sha256": _semantic_hash(decoded_identity),
                "target_weight": target["target_weight"],
                "compatibility": target["compatibility"],
                "provisional_training_weight": target["target_weight"]
                / frozen_profile["samples_per_target"],
                "target_weight_plan_sha256": target[
                    "target_weight_plan_sha256"
                ],
                "compatibility_sha256": target["compatibility_sha256"],
                "same_fault_type_partner_used": partner is not None,
                "same_fault_type_partner_case_id": (
                    partner["case_id"] if partner is not None else None
                ),
                "query_budget_cost": 0,
            }
            provenance = build_cvae_row_provenance(
                arm_id=arm,
                checkpoint_sha256=checkpoint_sha256,
                profile_id=frozen_profile["profile_id"],
                profile_sha256=frozen_profile["profile_sha256"],
                source_case_id=source["case_id"],
                queried_label_sha256=_semantic_hash(source["queried_label"]),
                source_posterior=source_posterior_identity,
                latent_seed=row_seed,
                interpolation={
                    "same_fault_type_partner_case_id": (
                        partner["case_id"] if partner is not None else None
                    ),
                    "alpha": 0.5 if partner is not None else None,
                    "mechanism_latent_sha256": row_identity[
                        "mechanism_latent_sha256"
                    ],
                    "propagation_latent_sha256": row_identity[
                        "propagation_latent_sha256"
                    ],
                    "context_latent_sha256": row_identity["context_latent_sha256"],
                },
                target_condition={
                    "target_service_id": target["target_service_id"],
                    "target_context_sha256": _tensor_hash(
                        target["target_context"]
                    ),
                    "target_context_mask_sha256": _tensor_hash(
                        target["target_context_mask"]
                    ),
                },
                compatibility={
                    "score": target["compatibility"],
                    "compatibility_sha256": target["compatibility_sha256"],
                    "target_weight_plan_sha256": target[
                        "target_weight_plan_sha256"
                    ],
                },
                decoded_state_sha256=row_identity["decoded_state_sha256"],
                training_weight=row_identity["provisional_training_weight"],
                activity_evidence=activity_evidence,
            )
            rows.append(
                {
                    **row_identity,
                    "synthetic_label": {
                        "root_cause": target["target_service_id"],
                        "fault_type": source["queried_label"]["fault_type"],
                        "label_source_case_id": source["case_id"],
                        "label_source": "queried_budget_transfer",
                    },
                    "decoded_mechanism": decoded["mechanism"],
                    "decoded_propagation": decoded["propagation"],
                    "decoded_context": decoded["context"],
                    "provenance": provenance,
                    "synthetic_row_sha256": _semantic_hash(row_identity),
                }
            )

    label_case_ids = [source["case_id"]]
    if partner is not None:
        label_case_ids.append(partner["case_id"])
    observation_ids = list(source["observation_case_ids"])
    observation_only = [
        case_id for case_id in observation_ids if case_id not in set(label_case_ids)
    ]
    audit_identity = {
        "schema_version": "service-continuous-cvae-generation-bundle-v1",
        "arm_id": arm,
        "source_case_id": source["case_id"],
        "same_fault_type_partner_required": False,
        "same_fault_type_partner_used": partner is not None,
        "same_fault_type_partner_case_id": (
            partner["case_id"] if partner is not None else None
        ),
        "synthetic_row_count": len(rows),
        "label_access_audit": {
            "label_case_ids": label_case_ids,
            "observation_only_case_ids": observation_only,
            "borrowed_label_count": 0,
        },
        "synthetic_row_sha256s": [row["synthetic_row_sha256"] for row in rows],
    }
    return {
        **audit_identity,
        "synthetic_rows": rows,
        "bundle_sha256": _semantic_hash(audit_identity),
    }


__all__ = [
    "CVAEGenerationValidationError",
    "generate_cvae_service_continuity",
]
