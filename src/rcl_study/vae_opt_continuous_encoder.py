from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .service_continuous_cvae import FactorizedConditionalVAE


class VAELatentEncoderError(ValueError):
    """Raised when Version B latent encoding violates its contract."""


_FACTORS = ("mechanism", "propagation", "context")


def encode_factorized_batch(
    model: FactorizedConditionalVAE, batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Encode a normalized candidate batch to concatenated factor posteriors."""

    if not isinstance(model, FactorizedConditionalVAE):
        raise VAELatentEncoderError("model must be FactorizedConditionalVAE")
    mus = []
    logvars = []
    for factor in _FACTORS:
        value = batch.get(factor)
        mask = batch.get(f"{factor}_mask")
        if (
            not isinstance(value, Tensor)
            or not isinstance(mask, Tensor)
            or value.ndim != 2
            or value.shape != mask.shape
        ):
            raise VAELatentEncoderError(f"{factor} tensor layout drift")
        encoder = getattr(model, f"{factor}_encoder")
        mu, logvar = encoder(value, mask)
        mus.append(mu)
        logvars.append(logvar)
    result = {"mu": torch.cat(mus, dim=1), "logvar": torch.cat(logvars, dim=1)}
    expected = sum(
        int(model.dimensions[f"{factor}_latent_width"]) for factor in _FACTORS
    )
    if result["mu"].shape[1] != expected or not all(
        torch.isfinite(value).all() for value in result.values()
    ):
        raise VAELatentEncoderError("latent output dimension or finiteness drift")
    return result


def load_latent_encoder_checkpoint(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> tuple[FactorizedConditionalVAE, dict[str, Any], dict[str, Any]]:
    """Load the Version A encoder and its fold-local normalization authority."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise VAELatentEncoderError("checkpoint does not exist")
    payload = torch.load(path, map_location=torch.device(device))
    if not isinstance(payload, Mapping) or payload.get("schema_version") not in {
        "vae-opt-discrete-cvae-checkpoint-v1",
        "service-continuous-cvae-checkpoint-v1",
    }:
        raise VAELatentEncoderError("unsupported CVAE checkpoint schema")
    dimensions = payload.get("model_dimensions")
    if not isinstance(dimensions, Mapping):
        profile = dict(payload.get("profile", {}))
        normalization = dict(payload.get("normalization", {}))
        dimensions = {
            "mechanism_dim": len(normalization.get("mechanism", {}).get("mean", [])),
            "propagation_dim": len(
                normalization.get("propagation", {}).get("mean", [])
            ),
            "context_dim": len(normalization.get("context", {}).get("mean", [])),
            "target_context_dim": len(
                normalization.get("context", {}).get("mean", [])
            ),
            "hidden_width": int(profile.get("hidden_width", 128)),
            "mechanism_latent_width": int(
                profile.get("mechanism_latent_width", 16)
            ),
            "propagation_latent_width": int(
                profile.get("propagation_latent_width", 16)
            ),
            "context_latent_width": int(profile.get("context_latent_width", 16)),
        }
    model = FactorizedConditionalVAE(**{key: int(value) for key, value in dimensions.items()})
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device)).eval()
    return model, dict(payload["normalization"]), dict(payload)


def encode_records_with_normalization(
    *,
    model: FactorizedConditionalVAE,
    records: Sequence[Mapping[str, Any]],
    normalization: Mapping[str, Any],
    device: str = "cpu",
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Apply train-only normalization and encode case/candidate records."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise VAELatentEncoderError("records must be nonempty")
    tensors: dict[str, list[list[float]]] = {factor: [] for factor in _FACTORS}
    masks: dict[str, list[list[float]]] = {factor: [] for factor in _FACTORS}
    keys: list[tuple[str, str]] = []
    for raw in records:
        case_id = str(raw.get("case_id", "")).strip()
        candidate_id = str(raw.get("candidate_id", "")).strip()
        state = raw.get("state")
        mask = raw.get("mask")
        if (
            not case_id
            or not candidate_id
            or (case_id, candidate_id) in set(keys)
            or not isinstance(state, Mapping)
            or not isinstance(mask, Mapping)
        ):
            raise VAELatentEncoderError("record ownership drift")
        keys.append((case_id, candidate_id))
        for factor in _FACTORS:
            values = [float(value) for value in state[factor]]
            observed = [float(value) for value in mask[factor]]
            authority = dict(normalization.get(factor, {}))
            mean = [float(value) for value in authority.get("mean", [])]
            scale = [float(value) for value in authority.get("scale", [])]
            if (
                len(values) != len(observed)
                or len(values) != len(mean)
                or len(values) != len(scale)
                or any(value not in (0.0, 1.0) for value in observed)
                or any(not math.isfinite(value) for value in values + mean + scale)
                or any(value <= 0.0 for value in scale)
            ):
                raise VAELatentEncoderError("record/normalization width drift")
            tensors[factor].append(
                [((value - center) / width) * present for value, center, width, present in zip(values, mean, scale, observed)]
            )
            masks[factor].append(observed)
    torch_device = torch.device(device)
    batch = {
        factor: torch.tensor(tensors[factor], dtype=torch.float32, device=torch_device)
        for factor in _FACTORS
    }
    batch.update(
        {
            f"{factor}_mask": torch.tensor(
                masks[factor], dtype=torch.float32, device=torch_device
            )
            for factor in _FACTORS
        }
    )
    with torch.no_grad():
        encoded = encode_factorized_batch(model, batch)
    mu = encoded["mu"].detach().cpu().numpy()
    logvar = encoded["logvar"].detach().cpu().numpy()
    return {
        key: {"mu": mu[index].copy(), "logvar": logvar[index].copy()}
        for index, key in enumerate(keys)
    }


def sample_latent_candidate_cases(
    *,
    encoded_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    supervised_case_ids: Sequence[Any],
    candidate_ids_by_case: Mapping[str, Sequence[Any]],
    targets_by_case: Mapping[str, Sequence[Any]],
    posterior_samples: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Build one mean case plus K posterior-sampled candidate-complete cases."""

    if (
        isinstance(posterior_samples, bool)
        or not isinstance(posterior_samples, int)
        or posterior_samples <= 0
    ):
        raise VAELatentEncoderError("posterior_samples must be positive")
    rows: list[dict[str, Any]] = []
    case_ids: list[str] = []
    expanded_targets: dict[str, list[str]] = {}
    rng = np.random.default_rng(int(seed))
    for raw_case_id in supervised_case_ids:
        case_id = str(raw_case_id)
        candidates = [str(value) for value in candidate_ids_by_case[case_id]]
        targets = [str(value) for value in targets_by_case[case_id]]
        if not candidates or not set(targets) <= set(candidates):
            raise VAELatentEncoderError("latent case candidate/target drift")
        variants = [(case_id, None)] + [
            (f"latent:{case_id}:{index:02d}", index)
            for index in range(posterior_samples)
        ]
        for variant_id, sample_index in variants:
            case_ids.append(variant_id)
            expanded_targets[variant_id] = list(targets)
            for candidate_id in candidates:
                encoded = encoded_rows[(case_id, candidate_id)]
                mu = np.asarray(encoded["mu"], dtype=float)
                logvar = np.asarray(encoded["logvar"], dtype=float)
                if mu.shape != (48,) or logvar.shape != (48,):
                    raise VAELatentEncoderError("Version B requires 48-dim posteriors")
                latent = (
                    mu
                    if sample_index is None
                    else mu
                    + rng.standard_normal(48)
                    * np.exp(0.5 * np.clip(logvar, -12.0, 8.0))
                )
                rows.append(
                    {
                        "case_id": variant_id,
                        "candidate_id": candidate_id,
                        "latent": [float(value) for value in latent],
                        "source_case_id": case_id,
                        "sample_index": sample_index,
                    }
                )
    return {
        "rows": rows,
        "case_ids": case_ids,
        "targets_by_case": expanded_targets,
        "posterior_samples": posterior_samples,
        "latent_dimension": 48,
    }


__all__ = [
    "VAELatentEncoderError",
    "encode_factorized_batch",
    "encode_records_with_normalization",
    "load_latent_encoder_checkpoint",
    "sample_latent_candidate_cases",
]
