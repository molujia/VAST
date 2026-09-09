from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .service_continuous_cvae import FactorizedConditionalVAE
from .service_continuous_cvae_training import compute_unlabeled_objectives


class VAEOptimizedTrainingError(ValueError):
    """Raised when the optimized CVAE training contract is invalid."""


def kl_weight_at_step(
    step: int, *, final_weight: float = 0.0125, warmup_steps: int = 50
) -> float:
    """Return the frozen linear KL warm-up weight for an optimizer step."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise VAEOptimizedTrainingError("step must be a nonnegative integer")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps <= 0
    ):
        raise VAEOptimizedTrainingError("warmup_steps must be a positive integer")
    if (
        isinstance(final_weight, bool)
        or not isinstance(final_weight, (int, float))
        or not math.isfinite(float(final_weight))
        or float(final_weight) < 0.0
    ):
        raise VAEOptimizedTrainingError("final_weight must be finite and nonnegative")
    return float(final_weight) * min(float(step) / float(warmup_steps), 1.0)


def mechanism_triplet_loss(
    mechanism_mu: Tensor,
    fault_types: Sequence[Any],
    *,
    margin: float = 0.5,
) -> Tensor:
    """Average deterministic triplets over repeated supervised fault types."""

    if not isinstance(mechanism_mu, Tensor) or mechanism_mu.ndim != 2:
        raise VAEOptimizedTrainingError("mechanism_mu must be a rank-2 tensor")
    if not torch.is_floating_point(mechanism_mu) or not torch.isfinite(
        mechanism_mu
    ).all():
        raise VAEOptimizedTrainingError("mechanism_mu must be finite floating point")
    labels = [str(value).strip() for value in fault_types]
    if (
        len(labels) != mechanism_mu.shape[0]
        or any(not value for value in labels)
        or isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) <= 0.0
    ):
        raise VAEOptimizedTrainingError("triplet labels or margin are invalid")
    terms = []
    for anchor, anchor_label in enumerate(labels):
        positives = [
            index
            for index, label in enumerate(labels)
            if index != anchor and label == anchor_label
        ]
        negatives = [
            index for index, label in enumerate(labels) if label != anchor_label
        ]
        for positive in positives:
            for negative in negatives:
                terms.append(
                    F.triplet_margin_loss(
                        mechanism_mu[anchor : anchor + 1],
                        mechanism_mu[positive : positive + 1],
                        mechanism_mu[negative : negative + 1],
                        margin=float(margin),
                        reduction="mean",
                    )
                )
    return (
        torch.stack(terms).mean()
        if terms
        else mechanism_mu.sum() * 0.0
    )


def train_kl_annealed_cvae(
    *,
    model: FactorizedConditionalVAE,
    batch: Mapping[str, Tensor],
    profile: Mapping[str, Any],
    optimizer_steps: int,
    supervised_mechanism: Tensor,
    supervised_mechanism_mask: Tensor,
    supervised_fault_types: Sequence[Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Train one CVAE with KL warm-up and a supervised latent triplet term."""

    if not isinstance(model, FactorizedConditionalVAE):
        raise VAEOptimizedTrainingError("model must be FactorizedConditionalVAE")
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps <= 0
    ):
        raise VAEOptimizedTrainingError("optimizer_steps must be positive")
    required = {
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "masked_reconstruction_weight",
        "kl_weight_final",
        "kl_warmup_steps",
        "target_context_weight",
        "cycle_consistency_weight",
        "triplet_margin",
        "triplet_weight",
    }
    if not isinstance(profile, Mapping) or not required <= set(profile):
        raise VAEOptimizedTrainingError("optimized training profile closure drift")
    if (
        not isinstance(supervised_mechanism, Tensor)
        or not isinstance(supervised_mechanism_mask, Tensor)
        or supervised_mechanism.ndim != 2
        or supervised_mechanism.shape != supervised_mechanism_mask.shape
        or len(supervised_fault_types) != supervised_mechanism.shape[0]
    ):
        raise VAEOptimizedTrainingError("supervised mechanism layout drift")

    torch.manual_seed(int(seed))
    if supervised_mechanism.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    gradient_norms: list[float] = []
    model.train()
    for step in range(optimizer_steps):
        effective_kl = kl_weight_at_step(
            step,
            final_weight=float(profile["kl_weight_final"]),
            warmup_steps=int(profile["kl_warmup_steps"]),
        )
        weights = {
            "masked_reconstruction": float(
                profile["masked_reconstruction_weight"]
            ),
            "kl": effective_kl,
            "target_context": float(profile["target_context_weight"]),
            "cycle_consistency": float(profile["cycle_consistency_weight"]),
        }
        optimizer.zero_grad(set_to_none=True)
        objectives = compute_unlabeled_objectives(
            model=model,
            batch=batch,
            weights=weights,
            sample_seed=int(seed) + step,
        )
        mechanism_mu, _ = model.mechanism_encoder(
            supervised_mechanism, supervised_mechanism_mask
        )
        triplet = mechanism_triplet_loss(
            mechanism_mu,
            supervised_fault_types,
            margin=float(profile["triplet_margin"]),
        )
        total = objectives["total"] + float(profile["triplet_weight"]) * triplet
        if not torch.isfinite(total):
            raise VAEOptimizedTrainingError("optimized CVAE loss became non-finite")
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(profile["gradient_clip_norm"])
        )
        gradient_norms.append(float(gradient.detach().cpu().item()))
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "kl_weight": effective_kl,
                "masked_reconstruction": float(
                    objectives["components"]["masked_reconstruction"]
                    .detach()
                    .cpu()
                    .item()
                ),
                "kl": float(
                    objectives["components"]["kl"].detach().cpu().item()
                ),
                "triplet": float(triplet.detach().cpu().item()),
                "total": float(total.detach().cpu().item()),
            }
        )

    model.eval()
    final_weights = {
        "masked_reconstruction": float(profile["masked_reconstruction_weight"]),
        "kl": float(profile["kl_weight_final"]),
        "target_context": float(profile["target_context_weight"]),
        "cycle_consistency": float(profile["cycle_consistency_weight"]),
    }
    with torch.no_grad():
        final = compute_unlabeled_objectives(
            model=model,
            batch=batch,
            weights=final_weights,
            sample_seed=int(seed),
        )
    return {
        "model": model,
        "history": history,
        "gradient_norms": gradient_norms,
        "final_components": {
            name: float(value.detach().cpu().item())
            for name, value in final["components"].items()
        },
        "optimizer_step_count": optimizer_steps,
        "final_kl_weight": float(profile["kl_weight_final"]),
        "triplet_active_step_count": sum(row["triplet"] > 0.0 for row in history),
    }


__all__ = [
    "VAEOptimizedTrainingError",
    "kl_weight_at_step",
    "mechanism_triplet_loss",
    "train_kl_annealed_cvae",
]
