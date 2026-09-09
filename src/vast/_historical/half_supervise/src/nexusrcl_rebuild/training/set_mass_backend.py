"""Fixed linear within-case set-mass ranker for partial supervision."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch

from .case_balanced_pseudo import prepare_case_balanced_pseudo_frame


def set_mass_loss(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.dim() != 1 or target_mask.shape != logits.shape:
        raise ValueError("set-mass logits and target mask must be aligned")
    target_mask = target_mask.to(dtype=torch.bool, device=logits.device)
    if not bool(target_mask.any()):
        raise ValueError("set-mass case requires at least one target")
    log_probabilities = torch.log_softmax(logits, dim=0)
    return -torch.logsumexp(log_probabilities[target_mask], dim=0)


class SetMassLinearClassifier:
    def __init__(
        self,
        coefficients: np.ndarray,
        *,
        training_diagnostics: Dict[str, Any],
    ) -> None:
        self.coef_ = np.asarray(coefficients, dtype=float).reshape(1, -1)
        self.intercept_ = np.asarray([0.0], dtype=float)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.training_diagnostics = dict(training_diagnostics)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        logits = values.dot(self.coef_[0])
        logits = np.clip(logits, -60.0, 60.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])


def fit_set_mass_linear(
    training_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    epochs: int = 200,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    seed: int = 42,
    pseudo_to_query_loss_ratio: float = 0.25,
) -> SetMassLinearClassifier:
    columns = tuple(str(value) for value in feature_columns)
    if not columns:
        raise ValueError("set-mass backend requires feature columns")
    missing = sorted(set(columns).difference(training_frame.columns))
    if missing:
        raise ValueError(
            "set-mass training frame missing features: %s" % missing
        )
    if (
        int(epochs) != 200
        or not math.isclose(float(learning_rate), 0.01)
        or not math.isclose(float(weight_decay), 0.0001)
        or int(seed) != 42
    ):
        raise ValueError("set_mass_linear_v1 hyperparameters are frozen")
    frame = prepare_case_balanced_pseudo_frame(
        training_frame,
        pseudo_to_query_loss_ratio=pseudo_to_query_loss_ratio,
    )
    values = (
        frame[list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(values).all():
        raise ValueError("set-mass features must be finite")
    torch.manual_seed(int(seed))
    features = torch.tensor(values, dtype=torch.float32)
    coefficients = torch.nn.Parameter(
        torch.zeros(len(columns), dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        [coefficients],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    groups = []
    queried_multi_positive = 0
    for window_id, group in frame.groupby("window_id", sort=False):
        indices = torch.tensor(group.index.to_numpy(), dtype=torch.long)
        targets = torch.tensor(
            group["label"].astype(int).to_numpy() == 1,
            dtype=torch.bool,
        )
        weight_values = {
            float(value) for value in group["case_loss_weight"].tolist()
        }
        if len(weight_values) != 1:
            raise ValueError("case loss weight must be constant per case")
        source = str(group["label_source"].iloc[0])
        if source == "queried" and int(targets.sum()) > 1:
            queried_multi_positive += 1
        groups.append(
            (
                str(window_id),
                indices,
                targets,
                next(iter(weight_values)),
                source,
            )
        )
    total_case_weight = sum(float(group[3]) for group in groups)
    if total_case_weight <= 0.0:
        raise ValueError("set-mass total case weight must be positive")
    loss_history = []
    nonzero_gradient_epochs = 0
    pseudo_nonzero_gradient_epochs = 0
    for _epoch in range(int(epochs)):
        logits = features.matmul(coefficients)
        weighted_losses = []
        pseudo_weighted_losses = []
        for _case_id, indices, targets, case_weight, source in groups:
            weighted_loss = float(case_weight) * set_mass_loss(
                logits.index_select(0, indices),
                targets,
            )
            weighted_losses.append(weighted_loss)
            if source.startswith("pseudo_"):
                pseudo_weighted_losses.append(weighted_loss)
        loss = torch.stack(weighted_losses).sum() / total_case_weight
        if not torch.isfinite(loss):
            raise FloatingPointError("set-mass training loss is non-finite")
        optimizer.zero_grad()
        if pseudo_weighted_losses:
            pseudo_loss = (
                torch.stack(pseudo_weighted_losses).sum()
                / total_case_weight
            )
            pseudo_gradient = torch.autograd.grad(
                pseudo_loss,
                coefficients,
                retain_graph=True,
            )[0]
            if bool(torch.count_nonzero(pseudo_gradient)):
                pseudo_nonzero_gradient_epochs += 1
        loss.backward()
        if (
            coefficients.grad is not None
            and bool(torch.count_nonzero(coefficients.grad))
        ):
            nonzero_gradient_epochs += 1
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    balance = dict(frame.attrs["case_balance_diagnostics"])
    diagnostics = {
        "schema_version": "set-mass-linear-training-v1",
        "backend": "set_mass_linear_v1",
        "architecture": "single_linear_entity_scorer",
        "objective": "negative_log_probability_mass_of_target_set",
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "feature_columns": list(columns),
        "case_count": len(groups),
        "queried_multi_positive_case_count": queried_multi_positive,
        "nonzero_gradient_epoch_count": nonzero_gradient_epochs,
        "pseudo_nonzero_gradient_epoch_count": (
            pseudo_nonzero_gradient_epochs
        ),
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        **balance,
    }
    return SetMassLinearClassifier(
        coefficients.detach().cpu().numpy(),
        training_diagnostics=diagnostics,
    )


__all__ = [
    "SetMassLinearClassifier",
    "fit_set_mass_linear",
    "set_mass_loss",
]
