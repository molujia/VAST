from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class OSERValidationError(ValueError):
    """Raised when OSER supervision, numerics, or episode ownership drift."""


_PAIR_INDEX_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def _upper_triangle_indices(size: int, device: torch.device) -> torch.Tensor:
    key = (int(size), device.type, device.index)
    cached = _PAIR_INDEX_CACHE.get(key)
    if cached is None:
        cached = torch.triu_indices(size, size, offset=1, device=device)
        _PAIR_INDEX_CACHE[key] = cached
    return cached


def _finite_nonnegative(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OSERValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise OSERValidationError(f"{field_name} must be finite and nonnegative")
    return result


def _normalized_budget_keys(
    budget_case_keys: Sequence[Any], maximum_budget: int
) -> tuple[str, ...]:
    try:
        limit = int(maximum_budget)
    except (TypeError, ValueError) as exc:
        raise OSERValidationError("maximum_budget must be an integer") from exc
    keys = tuple(str(value).strip() for value in budget_case_keys)
    if limit <= 0 or not keys or "" in keys or len(keys) != len(set(keys)):
        raise OSERValidationError("budget supervision keys must be unique and nonempty")
    if len(keys) > limit:
        raise OSERValidationError("budget supervision exceeds maximum_budget")
    return keys


def build_oser_episodes(
    queried_label_records: Sequence[Mapping[str, Any]],
    budget_case_keys: Sequence[Any],
    maximum_budget: int = 30,
) -> dict[str, Any]:
    """Build queried-type pseudo-LOFO episodes from budget-owned labels only."""

    budget_keys = _normalized_budget_keys(budget_case_keys, maximum_budget)
    records: dict[str, str] = {}
    for raw in queried_label_records:
        row = dict(raw)
        case_key = str(row.get("case_key", "")).strip()
        fault_type = str(row.get("fault_type", "")).strip()
        if not case_key or not fault_type or case_key in records:
            raise OSERValidationError("queried label records must be unique and complete")
        if case_key not in set(budget_keys):
            raise OSERValidationError("fault type is not owned by budget supervision")
        records[case_key] = fault_type
    if set(records) != set(budget_keys):
        raise OSERValidationError("budget supervision lacks a queried fault-type label")

    by_type: dict[str, list[str]] = defaultdict(list)
    for case_key in budget_keys:
        by_type[records[case_key]].append(case_key)
    fault_types = tuple(sorted(by_type))
    common = {
        "schema_version": "conservative-lofo-oser-episodes-v1",
        "budget_case_keys": budget_keys,
        "budget_case_count": len(budget_keys),
        "maximum_budget": int(maximum_budget),
        "queried_fault_types": fault_types,
        "queried_type_count": len(fault_types),
    }
    if len(fault_types) < 2:
        return {
            **common,
            "status": "fallback",
            "fallback_reason": "insufficient_episode_groups",
            "episodes": (),
        }

    episodes = []
    for fault_type in fault_types:
        support = tuple(
            case_key for case_key in budget_keys if records[case_key] != fault_type
        )
        outer = tuple(by_type[fault_type])
        case_weight = 1.0 / len(outer)
        episodes.append(
            {
                "query_fault_type": fault_type,
                "support_case_keys": support,
                "outer_case_keys": outer,
                "outer_case_weights": {
                    case_key: case_weight for case_key in outer
                },
                "episode_weight": 1.0 / len(fault_types),
            }
        )
    return {**common, "status": "ready", "fallback_reason": None, "episodes": episodes}


class OSERMetaResidualModel(nn.Module):
    """Frozen observable-state backbone plus a small meta-adapted residual."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_cap: float,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) <= 0:
            raise OSERValidationError("OSER dimensions must be positive")
        cap = _finite_nonnegative(residual_cap, "residual_cap")
        if cap <= 0.0 or cap > 1.0:
            raise OSERValidationError("residual_cap must lie in (0,1]")
        if int(seed) != 42:
            raise OSERValidationError("OSER initialization seed must be 42")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_cap = cap
        self.seed = 42
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.state_backbone = nn.Linear(self.input_dim, self.hidden_dim)
            self.adapter = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.residual_head = nn.Linear(self.hidden_dim, 1)
            self.gate_head = nn.Linear(self.hidden_dim, 1)
        for parameter in self.state_backbone.parameters():
            parameter.requires_grad_(False)

    def trainable_parameter_map(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def forward_with_parameters(
        self,
        states: torch.Tensor,
        parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.as_tensor(states, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise OSERValidationError("OSER state tensor shape drift")
        if not torch.isfinite(values).all():
            raise OSERValidationError("OSER state tensor is non-finite")
        owned = self.trainable_parameter_map()
        supplied = owned if parameters is None else dict(parameters)
        if set(supplied) != set(owned):
            raise OSERValidationError("OSER adapted parameter ownership drift")
        frozen = torch.tanh(
            F.linear(
                values,
                self.state_backbone.weight.detach(),
                self.state_backbone.bias.detach(),
            )
        )
        adapted = torch.tanh(
            F.linear(frozen, supplied["adapter.weight"], supplied["adapter.bias"])
        )
        residual = self.residual_cap * torch.tanh(
            F.linear(
                adapted,
                supplied["residual_head.weight"],
                supplied["residual_head.bias"],
            ).squeeze(-1)
        )
        gate = torch.sigmoid(
            F.linear(
                adapted,
                supplied["gate_head.weight"],
                supplied["gate_head.bias"],
            ).squeeze(-1)
        )
        return residual, gate

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_with_parameters(states)


def _normalized_case(
    case_key: str,
    case: Mapping[str, Any],
    model: OSERMetaResidualModel,
) -> dict[str, Any]:
    states = torch.as_tensor(case.get("states"), dtype=torch.float32)
    base_scores = torch.as_tensor(case.get("base_scores"), dtype=torch.float32)
    evidence = torch.as_tensor(case.get("evidence_supported"), dtype=torch.bool)
    if states.ndim != 2 or states.shape[1] != model.input_dim:
        raise OSERValidationError(f"OSER states are invalid for {case_key}")
    candidate_count = states.shape[0]
    if (
        base_scores.shape != (candidate_count,)
        or evidence.shape != (candidate_count,)
        or not torch.isfinite(states).all()
        or not torch.isfinite(base_scores).all()
    ):
        raise OSERValidationError(f"OSER candidate tensors drifted for {case_key}")
    positives = tuple(int(value) for value in case.get("positive_indices", ()))
    if (
        not positives
        or len(positives) != len(set(positives))
        or any(value < 0 or value >= candidate_count for value in positives)
    ):
        raise OSERValidationError(f"OSER positives are invalid for {case_key}")
    candidate_ids = tuple(
        str(value)
        for value in case.get(
            "candidate_ids", tuple(str(index) for index in range(candidate_count))
        )
    )
    if (
        len(candidate_ids) != candidate_count
        or "" in candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise OSERValidationError(f"OSER candidate IDs drifted for {case_key}")
    return {
        "states": states,
        "base_scores": base_scores,
        "evidence_supported": evidence,
        "positive_indices": positives,
        "candidate_ids": candidate_ids,
    }


def _case_terms(
    model: OSERMetaResidualModel,
    case_key: str,
    case: Mapping[str, Any],
    parameters: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = _normalized_case(case_key, case, model)
    return _case_terms_from_normalized(model, normalized, parameters)


def _case_terms_from_normalized(
    model: OSERMetaResidualModel,
    normalized: Mapping[str, Any],
    parameters: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    residual, declared_gate = model.forward_with_parameters(
        normalized["states"], parameters
    )
    gate = declared_gate * normalized["evidence_supported"].to(declared_gate.dtype)
    final_scores = normalized["base_scores"] + gate * residual
    positive_set = set(normalized["positive_indices"])
    negative_indices = [
        index for index in range(len(final_scores)) if index not in positive_set
    ]
    if not negative_indices:
        ranking = final_scores.sum() * 0.0
    else:
        positive_tensor = torch.as_tensor(
            normalized["positive_indices"], dtype=torch.long, device=final_scores.device
        )
        negative_tensor = torch.as_tensor(
            negative_indices, dtype=torch.long, device=final_scores.device
        )
        differences = (
            final_scores[positive_tensor, None] - final_scores[None, negative_tensor]
        ).reshape(-1)
        ranking = F.softplus(-differences).mean()

    pair_indices = _upper_triangle_indices(len(final_scores), final_scores.device)
    left_indices, right_indices = pair_indices[0], pair_indices[1]
    base_differences = (
        normalized["base_scores"][left_indices]
        - normalized["base_scores"][right_indices]
    )
    non_tied = base_differences.detach() != 0.0
    if bool(non_tied.any().item()):
        directions = torch.sign(base_differences.detach()[non_tied])
        final_differences = (
            final_scores[left_indices[non_tied]] - final_scores[right_indices[non_tied]]
        )
        distillation = F.softplus(-directions * final_differences).mean()
    else:
        distillation = final_scores.sum() * 0.0
    return {
        "ranking": ranking,
        "distillation": distillation,
        "residual_magnitude": residual.square().mean(),
        "gate_sparsity": gate.mean(),
        "residual": residual,
        "gate": gate,
    }


def _batched_case_term_vectors(
    model: OSERMetaResidualModel,
    normalized_cases: Mapping[str, Mapping[str, Any]],
    case_keys: Sequence[str],
    parameters: Mapping[str, torch.Tensor],
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Build the base-parameter terms as compact batched autograd graphs."""

    if batch_size <= 0:
        raise OSERValidationError("OSER batch size must be positive")
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for position, case_key in enumerate(case_keys):
        if case_key not in normalized_cases:
            raise OSERValidationError(f"OSER case is absent: {case_key}")
        grouped[len(normalized_cases[case_key]["base_scores"])].append(
            (position, case_key)
        )
    ordered: dict[str, list[torch.Tensor | None]] = {
        name: [None] * len(case_keys)
        for name in (
            "ranking",
            "distillation",
            "residual_magnitude",
            "gate_sparsity",
        )
    }
    for candidate_count, members in grouped.items():
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            normalized = [normalized_cases[case_key] for _, case_key in chunk]
            states = torch.stack([row["states"] for row in normalized])
            base_scores = torch.stack([row["base_scores"] for row in normalized])
            evidence = torch.stack([row["evidence_supported"] for row in normalized])
            residual, declared_gate = model.forward_with_parameters(
                states.reshape(-1, model.input_dim), parameters
            )
            residual = residual.reshape(len(chunk), candidate_count)
            declared_gate = declared_gate.reshape(len(chunk), candidate_count)
            gate = declared_gate * evidence.to(declared_gate.dtype)
            final_scores = base_scores + gate * residual

            positive_mask = torch.zeros_like(evidence, dtype=torch.bool)
            for row_index, row in enumerate(normalized):
                positive_mask[row_index, list(row["positive_indices"])] = True
            negative_mask = ~positive_mask
            ranking_pair_mask = positive_mask[:, :, None] & negative_mask[:, None, :]
            ranking_losses = F.softplus(
                -(final_scores[:, :, None] - final_scores[:, None, :])
            )
            ranking_counts = ranking_pair_mask.sum(dim=(1, 2))
            ranking = torch.where(
                ranking_counts > 0,
                (ranking_losses * ranking_pair_mask).sum(dim=(1, 2))
                / ranking_counts.clamp_min(1).to(ranking_losses.dtype),
                final_scores.sum(dim=1) * 0.0,
            )

            pair_indices = _upper_triangle_indices(candidate_count, final_scores.device)
            left_indices, right_indices = pair_indices[0], pair_indices[1]
            base_differences = (
                base_scores[:, left_indices] - base_scores[:, right_indices]
            )
            non_tied = base_differences.detach() != 0.0
            directions = torch.sign(base_differences.detach())
            final_differences = (
                final_scores[:, left_indices] - final_scores[:, right_indices]
            )
            distillation_losses = F.softplus(-directions * final_differences)
            distillation_counts = non_tied.sum(dim=1)
            distillation = torch.where(
                distillation_counts > 0,
                (distillation_losses * non_tied).sum(dim=1)
                / distillation_counts.clamp_min(1).to(distillation_losses.dtype),
                final_scores.sum(dim=1) * 0.0,
            )
            chunk_vectors = {
                "ranking": ranking,
                "distillation": distillation,
                "residual_magnitude": residual.square().mean(dim=1),
                "gate_sparsity": gate.mean(dim=1),
            }
            for row_index, (position, _) in enumerate(chunk):
                for term_name, vector in chunk_vectors.items():
                    ordered[term_name][position] = vector[row_index]
    if any(value is None for values in ordered.values() for value in values):
        raise OSERValidationError("OSER batched term construction is incomplete")
    return {
        term_name: torch.stack([value for value in values if value is not None])
        for term_name, values in ordered.items()
    }


def _mean_case_term(
    model: OSERMetaResidualModel,
    cases: Mapping[str, Mapping[str, Any]],
    case_keys: Sequence[str],
    parameters: Mapping[str, torch.Tensor],
    term_name: str,
    case_weights: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    if not case_keys:
        raise OSERValidationError("OSER objective case set must be nonempty")
    terms = []
    weights = []
    for case_key in case_keys:
        if case_key not in cases:
            raise OSERValidationError(f"OSER case is absent: {case_key}")
        terms.append(_case_terms(model, case_key, cases[case_key], parameters)[term_name])
        weights.append(
            1.0
            if case_weights is None
            else _finite_nonnegative(case_weights.get(case_key), "case weight")
        )
    denominator = sum(weights)
    if denominator <= 0.0:
        raise OSERValidationError("OSER case weights must have positive mass")
    return sum(term * weight for term, weight in zip(terms, weights)) / denominator


def _mean_cached_case_term(
    case_terms: Mapping[str, Mapping[str, torch.Tensor]],
    case_keys: Sequence[str],
    term_name: str,
    case_weights: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    if not case_keys:
        raise OSERValidationError("OSER objective case set must be nonempty")
    terms = []
    weights = []
    for case_key in case_keys:
        if case_key not in case_terms:
            raise OSERValidationError(f"OSER case is absent: {case_key}")
        terms.append(case_terms[case_key][term_name])
        weights.append(
            1.0
            if case_weights is None
            else _finite_nonnegative(case_weights.get(case_key), "case weight")
        )
    denominator = sum(weights)
    if denominator <= 0.0:
        raise OSERValidationError("OSER case weights must have positive mass")
    return sum(term * weight for term, weight in zip(terms, weights)) / denominator


def _mean_case_term_vector(
    term_vector: torch.Tensor,
    case_positions: Mapping[str, int],
    case_keys: Sequence[str],
    case_weights: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    if not case_keys:
        raise OSERValidationError("OSER objective case set must be nonempty")
    try:
        positions = [case_positions[case_key] for case_key in case_keys]
    except KeyError as exc:
        raise OSERValidationError(f"OSER case is absent: {exc.args[0]}") from exc
    weights = [
        1.0
        if case_weights is None
        else _finite_nonnegative(case_weights.get(case_key), "case weight")
        for case_key in case_keys
    ]
    denominator = sum(weights)
    if denominator <= 0.0:
        raise OSERValidationError("OSER case weights must have positive mass")
    position_tensor = torch.as_tensor(
        positions, dtype=torch.long, device=term_vector.device
    )
    weight_tensor = torch.as_tensor(
        weights, dtype=term_vector.dtype, device=term_vector.device
    )
    selected = term_vector.index_select(0, position_tensor)
    return (selected * weight_tensor).sum() / denominator


def compute_oser_joint_objective(
    model: OSERMetaResidualModel,
    cases: Mapping[str, Mapping[str, Any]],
    episode_artifact: Mapping[str, Any],
    inner_learning_rate: float,
    lambda_meta: float,
    lambda_distillation: float = 0.10,
    lambda_residual: float = 0.10,
    lambda_gate: float = 0.01,
) -> dict[str, Any]:
    """Compute OSER's ordinary plus second-order pseudo-LOFO objective."""

    if episode_artifact.get("status") != "ready":
        raise OSERValidationError("OSER meta objective requires ready episode groups")
    inner_rate = _finite_nonnegative(inner_learning_rate, "inner_learning_rate")
    meta_weight = _finite_nonnegative(lambda_meta, "lambda_meta")
    distill_weight = _finite_nonnegative(lambda_distillation, "lambda_distillation")
    residual_weight = _finite_nonnegative(lambda_residual, "lambda_residual")
    gate_weight = _finite_nonnegative(lambda_gate, "lambda_gate")
    if inner_rate <= 0.0:
        raise OSERValidationError("inner_learning_rate must be positive")

    case_keys = tuple(str(value) for value in episode_artifact.get("budget_case_keys", ()))
    if set(case_keys) != set(str(value) for value in cases):
        raise OSERValidationError("OSER cases do not match budget episode supervision")
    case_weights_raw = episode_artifact.get("case_weights")
    case_weights = None
    if case_weights_raw is not None:
        if not isinstance(case_weights_raw, Mapping) or set(
            str(key) for key in case_weights_raw
        ) != set(case_keys):
            raise OSERValidationError("OSER case-weight ownership drift")
        case_weights = {
            str(key): _finite_nonnegative(value, "case weight")
            for key, value in case_weights_raw.items()
        }
        if any(value <= 0.0 for value in case_weights.values()):
            raise OSERValidationError("OSER case weights must be positive")
    parameters = model.trainable_parameter_map()
    normalized_cases = {
        case_key: _normalized_case(case_key, cases[case_key], model)
        for case_key in case_keys
    }
    case_positions = {case_key: index for index, case_key in enumerate(case_keys)}
    base_term_vectors = _batched_case_term_vectors(
        model, normalized_cases, case_keys, parameters
    )
    ordinary = _mean_case_term_vector(
        base_term_vectors["ranking"], case_positions, case_keys, case_weights
    )
    distillation = _mean_case_term_vector(
        base_term_vectors["distillation"], case_positions, case_keys, case_weights
    )
    residual = _mean_case_term_vector(
        base_term_vectors["residual_magnitude"], case_positions, case_keys, case_weights
    )
    gate = _mean_case_term_vector(
        base_term_vectors["gate_sparsity"], case_positions, case_keys, case_weights
    )

    meta_terms = []
    episodes = tuple(episode_artifact.get("episodes", ()))
    for episode in episodes:
        support_keys = tuple(str(value) for value in episode.get("support_case_keys", ()))
        outer_keys = tuple(str(value) for value in episode.get("outer_case_keys", ()))
        if set(support_keys) & set(outer_keys):
            raise OSERValidationError("OSER support/query episode overlap")
        inner_loss = _mean_case_term_vector(
            base_term_vectors["ranking"], case_positions, support_keys, case_weights
        )
        gradients = torch.autograd.grad(
            inner_loss,
            tuple(parameters.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        adapted = {
            name: parameter - inner_rate * gradient
            for (name, parameter), gradient in zip(parameters.items(), gradients)
        }
        outer_terms = {
            case_key: _case_terms_from_normalized(
                model, normalized_cases[case_key], adapted
            )
            for case_key in outer_keys
        }
        outer_loss = _mean_cached_case_term(
            outer_terms,
            outer_keys,
            "ranking",
            case_weights=dict(episode.get("outer_case_weights", {})),
        )
        episode_weight = _finite_nonnegative(
            episode.get("episode_weight"), "episode_weight"
        )
        meta_terms.append(outer_loss * episode_weight)
    if not meta_terms:
        raise OSERValidationError("OSER ready artifact has no episodes")
    meta = sum(meta_terms)
    components = {
        "ordinary_ranking": ordinary,
        "meta_ranking": meta,
        "base_order_distillation": distillation,
        "residual_magnitude": residual,
        "gate_sparsity": gate,
    }
    if not all(torch.isfinite(value) for value in components.values()):
        raise OSERValidationError("OSER objective component is non-finite")
    loss = (
        ordinary
        + meta_weight * meta
        + distill_weight * distillation
        + residual_weight * residual
        + gate_weight * gate
    )
    if not torch.isfinite(loss):
        raise OSERValidationError("OSER objective is non-finite")
    return {
        "loss": loss,
        "components": components,
        "component_values": {
            name: float(value.detach().cpu()) for name, value in components.items()
        },
        "higher_order_inner_update": True,
        "inner_update_count": len(episodes),
        "adapted_parameter_names": tuple(parameters),
    }


def build_oser_fallback_rows(
    candidate_ids_by_case: Mapping[str, Sequence[Any]],
    episode_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    reason = str(episode_artifact.get("fallback_reason", ""))
    if episode_artifact.get("status") != "fallback" or reason != "insufficient_episode_groups":
        raise OSERValidationError("OSER fallback rows require insufficient episode groups")
    rows = []
    for case_key, candidate_ids in candidate_ids_by_case.items():
        for candidate_id in candidate_ids:
            rows.append(
                {
                    "case_id": str(case_key),
                    "candidate_id": str(candidate_id),
                    "arm": "oser_meta",
                    "residual": 0.0,
                    "gate": 0.0,
                    "evidence_status": "missing",
                    "evidence_value": None,
                }
            )
    return {
        "rows": rows,
        "fallback_reason": reason,
        "mechanism_active": False,
    }


def build_oser_residual_rows(
    model: OSERMetaResidualModel,
    cases: Mapping[str, Mapping[str, Any]],
    episode_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if episode_artifact.get("status") != "ready":
        candidate_ids = {
            str(case_key): _normalized_case(str(case_key), case, model)["candidate_ids"]
            for case_key, case in cases.items()
        }
        return build_oser_fallback_rows(candidate_ids, episode_artifact)["rows"]
    parameters = model.trainable_parameter_map()
    rows = []
    for case_key, case in cases.items():
        normalized = _normalized_case(str(case_key), case, model)
        residual, gate = model.forward_with_parameters(normalized["states"], parameters)
        for index, candidate_id in enumerate(normalized["candidate_ids"]):
            supported = bool(normalized["evidence_supported"][index].item())
            declared_gate = float(gate[index].detach().cpu()) if supported else 0.0
            rows.append(
                {
                    "case_id": str(case_key),
                    "candidate_id": candidate_id,
                    "arm": "oser_meta",
                    "residual": float(residual[index].detach().cpu()),
                    "gate": declared_gate,
                    "evidence_status": "supported" if supported else "missing",
                    "evidence_value": declared_gate if supported else None,
                }
            )
    return rows


def audit_oser_mechanism(
    residual_rows: Sequence[Mapping[str, Any]],
    episode_artifact: Mapping[str, Any],
    gradient_l1: Any,
) -> dict[str, Any]:
    try:
        gradient = float(gradient_l1)
    except (TypeError, ValueError):
        gradient = float("nan")
    finite_rows = []
    supported_active_count = 0
    for row in residual_rows:
        try:
            residual = float(row.get("residual"))
            gate = float(row.get("gate"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(residual) and math.isfinite(gate):
            finite_rows.append((residual, gate))
            if (
                str(row.get("evidence_status", "")) == "supported"
                and abs(residual) > 1e-12
                and gate > 1e-12
            ):
                supported_active_count += 1
    reasons = []
    if episode_artifact.get("status") != "ready":
        reasons.append(str(episode_artifact.get("fallback_reason", "episode_groups_not_ready")))
    if not math.isfinite(gradient) or gradient <= 0.0:
        reasons.append("nonpositive_or_nonfinite_gradient")
    if supported_active_count == 0:
        reasons.append("always_off_or_zero_residual")
    if len(finite_rows) != len(residual_rows):
        reasons.append("nonfinite_residual_rows")
    return {
        "mechanism_active": not reasons,
        "inactivity_reasons": tuple(reasons),
        "episode_group_count": int(episode_artifact.get("queried_type_count", 0)),
        "finite_residual_count": len(finite_rows),
        "residual_row_count": len(residual_rows),
        "supported_active_count": supported_active_count,
        "gradient_l1": gradient,
    }


__all__ = [
    "OSERMetaResidualModel",
    "OSERValidationError",
    "audit_oser_mechanism",
    "build_oser_episodes",
    "build_oser_fallback_rows",
    "build_oser_residual_rows",
    "compute_oser_joint_objective",
]
