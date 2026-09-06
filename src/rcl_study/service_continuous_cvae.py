from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class ConditionalVAEValidationError(ValueError):
    """Raised when a feature/state CVAE input violates its frozen contract."""


def _validate_tensor(value: Tensor, context: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ConditionalVAEValidationError(f"{context} must be a rank-2 tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ConditionalVAEValidationError(
            f"{context} must contain finite floating-point values"
        )
    return value


def _validate_mask(mask: Tensor, reference: Tensor, context: str) -> Tensor:
    value = _validate_tensor(mask, context)
    if value.shape != reference.shape:
        raise ConditionalVAEValidationError(f"{context} shape mismatch")
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise ConditionalVAEValidationError(f"{context} must be within [0, 1]")
    return value


def masked_reconstruction_loss(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    predicted = _validate_tensor(prediction, "prediction")
    expected = _validate_tensor(target, "target")
    if predicted.shape != expected.shape:
        raise ConditionalVAEValidationError("prediction and target shape mismatch")
    observed = _validate_mask(mask, expected, "reconstruction mask")
    denominator = observed.sum()
    if not bool(denominator > 0.0):
        raise ConditionalVAEValidationError(
            "masked reconstruction requires at least one observed coordinate"
        )
    return (((predicted - expected) ** 2) * observed).sum() / denominator


def sample_posterior(mu: Tensor, logvar: Tensor, *, seed: int | None = None) -> Tensor:
    mean = _validate_tensor(mu, "posterior mean")
    raw_logvar = _validate_tensor(logvar, "posterior log-variance")
    if mean.shape != raw_logvar.shape:
        raise ConditionalVAEValidationError("posterior parameter shape mismatch")
    bounded_logvar = raw_logvar.clamp(min=-12.0, max=8.0)
    generator = None
    if seed is not None:
        normalized_seed = int(seed)
        if normalized_seed < 0:
            raise ConditionalVAEValidationError("posterior seed must be nonnegative")
        generator = torch.Generator(device=mean.device)
        generator.manual_seed(normalized_seed)
    epsilon = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    sample = mean + torch.exp(0.5 * bounded_logvar) * epsilon
    if not torch.isfinite(sample).all():
        raise ConditionalVAEValidationError("posterior sample must remain finite")
    return sample


class _FactorEncoder(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, latent_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width * 2, hidden_width),
            nn.SiLU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_width, latent_width)
        self.logvar = nn.Linear(hidden_width, latent_width)

    def forward(self, value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.network(torch.cat((value * mask, mask), dim=1))
        return self.mu(hidden), self.logvar(hidden).clamp(min=-12.0, max=8.0)


class FactorizedConditionalVAE(nn.Module):
    """Conditional VAE over observable mechanism, propagation, and context states."""

    def __init__(
        self,
        *,
        mechanism_dim: int,
        propagation_dim: int,
        context_dim: int,
        target_context_dim: int,
        hidden_width: int,
        mechanism_latent_width: int,
        propagation_latent_width: int,
        context_latent_width: int,
    ) -> None:
        super().__init__()
        dimensions = {
            "mechanism_dim": mechanism_dim,
            "propagation_dim": propagation_dim,
            "context_dim": context_dim,
            "target_context_dim": target_context_dim,
            "hidden_width": hidden_width,
            "mechanism_latent_width": mechanism_latent_width,
            "propagation_latent_width": propagation_latent_width,
            "context_latent_width": context_latent_width,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        ):
            raise ConditionalVAEValidationError(
                "CVAE dimensions and widths must be positive integers"
            )
        self.dimensions = dict(dimensions)
        self.mechanism_encoder = _FactorEncoder(
            mechanism_dim, hidden_width, mechanism_latent_width
        )
        self.propagation_encoder = _FactorEncoder(
            propagation_dim, hidden_width, propagation_latent_width
        )
        self.context_encoder = _FactorEncoder(
            context_dim, hidden_width, context_latent_width
        )
        self.target_context_encoder = nn.Sequential(
            nn.Linear(target_context_dim * 2, hidden_width),
            nn.SiLU(),
            nn.LayerNorm(hidden_width),
        )
        decoder_input_width = (
            mechanism_latent_width
            + propagation_latent_width
            + context_latent_width
            + hidden_width
        )
        decoder_output_width = mechanism_dim + propagation_dim + context_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_width, hidden_width),
            nn.SiLU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, decoder_output_width),
        )

    def _validate_inputs(self, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
        dimensions = {
            "mechanism": self.dimensions["mechanism_dim"],
            "propagation": self.dimensions["propagation_dim"],
            "context": self.dimensions["context_dim"],
            "target_context": self.dimensions["target_context_dim"],
        }
        validated: dict[str, Tensor] = {}
        batch_size: int | None = None
        for name, width in dimensions.items():
            value = _validate_tensor(values[name], name)
            mask = _validate_mask(values[f"{name}_mask"], value, f"{name}_mask")
            if value.shape[1] != width:
                raise ConditionalVAEValidationError(f"{name} width mismatch")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ConditionalVAEValidationError("CVAE batch-size mismatch")
            validated[name] = value
            validated[f"{name}_mask"] = mask
        return validated

    @staticmethod
    def _partition(
        encoder: _FactorEncoder,
        value: Tensor,
        mask: Tensor,
        *,
        sample: bool,
        seed: int | None,
    ) -> dict[str, Tensor]:
        mu, logvar = encoder(value, mask)
        z = sample_posterior(mu, logvar, seed=seed) if sample else mu
        return {"mu": mu, "logvar": logvar, "z": z}

    def forward(
        self,
        *,
        mechanism: Tensor,
        mechanism_mask: Tensor,
        propagation: Tensor,
        propagation_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        target_context: Tensor,
        target_context_mask: Tensor,
        sample: bool = True,
        sample_seed: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(sample, bool):
            raise ConditionalVAEValidationError("sample must be boolean")
        values = self._validate_inputs(
            {
                "mechanism": mechanism,
                "mechanism_mask": mechanism_mask,
                "propagation": propagation,
                "propagation_mask": propagation_mask,
                "context": context,
                "context_mask": context_mask,
                "target_context": target_context,
                "target_context_mask": target_context_mask,
            }
        )
        base_seed = None if sample_seed is None else int(sample_seed)
        posterior = {
            "mechanism": self._partition(
                self.mechanism_encoder,
                values["mechanism"],
                values["mechanism_mask"],
                sample=sample,
                seed=base_seed,
            ),
            "propagation": self._partition(
                self.propagation_encoder,
                values["propagation"],
                values["propagation_mask"],
                sample=sample,
                seed=None if base_seed is None else base_seed + 1,
            ),
            "context": self._partition(
                self.context_encoder,
                values["context"],
                values["context_mask"],
                sample=sample,
                seed=None if base_seed is None else base_seed + 2,
            ),
        }
        target_embedding = self.target_context_encoder(
            torch.cat(
                (
                    values["target_context"] * values["target_context_mask"],
                    values["target_context_mask"],
                ),
                dim=1,
            )
        )
        decoded = self.decoder(
            torch.cat(
                (
                    posterior["mechanism"]["z"],
                    posterior["propagation"]["z"],
                    posterior["context"]["z"],
                    target_embedding,
                ),
                dim=1,
            )
        )
        mechanism_end = self.dimensions["mechanism_dim"]
        propagation_end = mechanism_end + self.dimensions["propagation_dim"]
        reconstruction = {
            "mechanism": decoded[:, :mechanism_end],
            "propagation": decoded[:, mechanism_end:propagation_end],
            "context": decoded[:, propagation_end:],
        }
        if not all(torch.isfinite(value).all() for value in reconstruction.values()):
            raise ConditionalVAEValidationError("CVAE reconstruction must remain finite")
        return {
            "posterior": posterior,
            "reconstruction": reconstruction,
            "target_context_embedding": target_embedding,
        }


__all__ = [
    "ConditionalVAEValidationError",
    "FactorizedConditionalVAE",
    "masked_reconstruction_loss",
    "sample_posterior",
]
