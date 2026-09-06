from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .conservative_lofo_oser import (
    OSERMetaResidualModel,
    audit_oser_mechanism,
    build_oser_residual_rows,
    compute_oser_joint_objective,
)
from .conservative_lofo_residual import (
    ConservativeResidualOutput,
    apply_conservative_residual,
)


class FinalOSERValidationError(ValueError):
    """Raised when final CVAE/OSER ownership or execution drifts."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OSER_PROFILE_SHA256 = (
    "ef4f2c0229e8549cc2b85d33bf6478d10870606929cf156a4e99a60d70eb581a"
)


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


def _ids(value: Sequence[Any], context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalOSERValidationError(f"{context} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if not result or "" in result or len(result) != len(set(result)):
        raise FinalOSERValidationError(f"{context} must be unique and nonempty")
    return result


def _positive_weight(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise FinalOSERValidationError(f"{context} must be positive finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalOSERValidationError(f"{context} must be positive finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FinalOSERValidationError(f"{context} must be positive finite")
    return result


def build_final_oser_family_episodes(
    *,
    real_label_records: Sequence[Mapping[str, Any]],
    supervised_case_ids: Sequence[Any],
    synthetic_rows: Sequence[Mapping[str, Any]],
    training_mode: str,
    supervised_case_limit: int,
) -> dict[str, Any]:
    """Build pseudo-LOFO episodes with real-only outer queries and family support."""

    mode = str(training_mode)
    if mode not in {"query_only", "oracle_full"}:
        raise FinalOSERValidationError("invalid OSER training mode")
    real_ids = _ids(supervised_case_ids, "supervised real case IDs")
    if (
        isinstance(supervised_case_limit, bool)
        or not isinstance(supervised_case_limit, int)
        or supervised_case_limit <= 0
        or len(real_ids) != supervised_case_limit
    ):
        raise FinalOSERValidationError("supervised cardinality differs from its limit")
    labels: dict[str, str] = {}
    for raw in real_label_records:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        fault_type = str(row.get("fault_type", "")).strip()
        if not case_id or not fault_type or case_id in labels:
            raise FinalOSERValidationError("real label records are incomplete or duplicate")
        labels[case_id] = fault_type
    if tuple(labels) != real_ids:
        raise FinalOSERValidationError("real label order or membership drifted")

    child_keys_by_parent: dict[str, list[str]] = defaultdict(list)
    child_weights_by_parent: dict[str, list[float]] = defaultdict(list)
    child_rows = []
    seen_children: set[str] = set()
    for raw in synthetic_rows:
        row = dict(raw)
        row_hash = str(row.get("synthetic_row_sha256", "")).strip()
        parent = str(row.get("source_case_id", "")).strip()
        weight = _positive_weight(
            row.get("final_training_weight"), "synthetic family weight"
        )
        if (
            not _SHA256.fullmatch(row_hash)
            or row_hash in seen_children
            or parent not in labels
            or row.get("query_budget_cost") != 0
        ):
            raise FinalOSERValidationError("synthetic family ownership drifted")
        child_key = f"synthetic:{row_hash}"
        seen_children.add(row_hash)
        child_keys_by_parent[parent].append(child_key)
        child_weights_by_parent[parent].append(weight)
        child_rows.append(
            {
                "case_key": child_key,
                "parent_case_id": parent,
                "fault_type": labels[parent],
                "training_weight": weight,
            }
        )
    for parent, weights in child_weights_by_parent.items():
        if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-9):
            raise FinalOSERValidationError(
                f"synthetic descendants for {parent} do not have total mass 1.0"
            )

    training_keys = (*real_ids, *(row["case_key"] for row in child_rows))
    case_weights = {case_id: 1.0 for case_id in real_ids}
    case_weights.update(
        {row["case_key"]: row["training_weight"] for row in child_rows}
    )
    by_type: dict[str, list[str]] = defaultdict(list)
    for case_id in real_ids:
        by_type[labels[case_id]].append(case_id)
    fault_types = tuple(sorted(by_type))
    family_audit = {
        "family_count": len(real_ids),
        "synthetic_family_count": len(child_keys_by_parent),
        "synthetic_descendant_count": len(child_rows),
        "split_family_count": 0,
        "family_members_sha256": _semantic_hash(
            {
                parent: (parent, *child_keys_by_parent.get(parent, ()))
                for parent in real_ids
            }
        ),
    }
    common = {
        "schema_version": "final-rcl-oser-family-episodes-v1",
        "training_mode": mode,
        "supervised_case_limit": supervised_case_limit,
        "supervised_real_case_keys": real_ids,
        "supervised_real_case_count": len(real_ids),
        "synthetic_case_keys": tuple(row["case_key"] for row in child_rows),
        "synthetic_case_count": len(child_rows),
        "training_case_keys": training_keys,
        "budget_case_keys": training_keys,
        "budget_case_count": len(training_keys),
        "maximum_budget": supervised_case_limit,
        "case_weights": case_weights,
        "queried_fault_types": fault_types,
        "queried_type_count": len(fault_types),
        "parent_family_isolation_audit": family_audit,
    }
    if len(fault_types) < 2:
        identity = {
            **common,
            "status": "fallback",
            "fallback_reason": "insufficient_episode_groups",
            "episodes": (),
        }
        return {**identity, "episode_sha256": _semantic_hash(identity)}

    episodes = []
    for fault_type in fault_types:
        hidden_real = tuple(by_type[fault_type])
        hidden_set = set(hidden_real)
        support = []
        quarantined = []
        for parent in real_ids:
            family = (parent, *child_keys_by_parent.get(parent, ()))
            if parent in hidden_set:
                quarantined.extend(family[1:])
            else:
                support.extend(family)
        episodes.append(
            {
                "query_fault_type": fault_type,
                "support_case_keys": tuple(support),
                "outer_case_keys": hidden_real,
                "outer_case_weights": {
                    case_id: 1.0 / len(hidden_real) for case_id in hidden_real
                },
                "quarantined_hidden_descendant_keys": tuple(quarantined),
                "episode_weight": 1.0 / len(fault_types),
            }
        )
    identity = {
        **common,
        "status": "ready",
        "fallback_reason": None,
        "episodes": tuple(episodes),
    }
    return {**identity, "episode_sha256": _semantic_hash(identity)}


def _profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = dict(value)
    expected = {
        "profile_id": "oser-p02",
        "profile_sha256": _OSER_PROFILE_SHA256,
        "state_width": 32,
        "residual_cap": 0.05,
        "gate_threshold": 0.5,
        "lambda_meta": 0.5,
        "inner_updates": 1,
        "inner_learning_rate": 0.05,
        "training_steps": 30,
    }
    if any(profile.get(key) != expected_value for key, expected_value in expected.items()):
        raise FinalOSERValidationError("frozen oser-p02 profile drifted")
    return profile


def _state_payload(model: OSERMetaResidualModel) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in sorted(model.state_dict().items())
    }


@dataclass
class FinalOSERFit:
    model: OSERMetaResidualModel
    episode_artifact: Mapping[str, Any]
    profile: Mapping[str, Any]
    audit: Mapping[str, Any]


def fit_final_oser(
    *,
    training_cases: Mapping[str, Mapping[str, Any]],
    episode_artifact: Mapping[str, Any],
    profile: Mapping[str, Any],
    state_transform_sha256: str,
    seed: int,
) -> FinalOSERFit:
    frozen = _profile(profile)
    if int(seed) != 42 or not _SHA256.fullmatch(str(state_transform_sha256)):
        raise FinalOSERValidationError("OSER seed or state-transform identity drifted")
    episodes = dict(episode_artifact)
    supplied_hash = episodes.pop("episode_sha256", None)
    if supplied_hash != _semantic_hash(episodes):
        raise FinalOSERValidationError("OSER episode artifact hash drifted")
    episodes["episode_sha256"] = supplied_hash
    training_keys = tuple(str(key) for key in episode_artifact.get("training_case_keys", ()))
    if tuple(training_cases) != training_keys:
        raise FinalOSERValidationError("OSER training case order or membership drifted")
    first = next(iter(training_cases.values()), None)
    if not isinstance(first, Mapping):
        raise FinalOSERValidationError("OSER training cases are empty")
    states = first.get("states")
    if not isinstance(states, Sequence) or not states or not isinstance(states[0], Sequence):
        raise FinalOSERValidationError("OSER state rows are empty")
    input_dim = len(states[0])
    model = OSERMetaResidualModel(
        input_dim=input_dim,
        hidden_dim=int(frozen["state_width"]),
        residual_cap=float(frozen["residual_cap"]),
        seed=42,
    )
    objective_values: dict[str, float] = {}
    gradient_l1 = 0.0
    step_count = 0
    if episode_artifact.get("status") == "ready":
        optimizer = torch.optim.Adam(model.trainable_parameter_map().values(), lr=0.01)
        for _ in range(int(frozen["training_steps"])):
            optimizer.zero_grad(set_to_none=True)
            objective = compute_oser_joint_objective(
                model=model,
                cases=training_cases,
                episode_artifact=episode_artifact,
                inner_learning_rate=float(frozen["inner_learning_rate"]),
                lambda_meta=float(frozen["lambda_meta"]),
            )
            objective["loss"].backward()
            gradient_l1 = math.fsum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in model.trainable_parameter_map().values()
                if parameter.grad is not None
            )
            optimizer.step()
            step_count += 1
        objective_values = dict(objective["component_values"])
    rows = build_oser_residual_rows(model, training_cases, episode_artifact)
    mechanism = audit_oser_mechanism(rows, episode_artifact, gradient_l1)
    state_payload = _state_payload(model)
    checkpoint_identity = {
        "schema_version": "final-rcl-oser-checkpoint-v1",
        "profile_id": frozen["profile_id"],
        "profile_sha256": frozen["profile_sha256"],
        "episode_sha256": episode_artifact["episode_sha256"],
        "state_transform_sha256": str(state_transform_sha256),
        "training_case_keys": training_keys,
        "training_case_weights": dict(episode_artifact.get("case_weights", {})),
        "model_state": state_payload,
    }
    checkpoint_sha256 = _semantic_hash(checkpoint_identity)
    audit_identity = {
        "schema_version": "final-rcl-oser-training-audit-v1",
        "profile_id": frozen["profile_id"],
        "profile_sha256": frozen["profile_sha256"],
        "training_mode": episode_artifact.get("training_mode"),
        "supervised_case_limit": episode_artifact.get("supervised_case_limit"),
        "real_training_case_count": episode_artifact.get("supervised_real_case_count"),
        "synthetic_training_case_count": episode_artifact.get("synthetic_case_count"),
        "objective_step_count": step_count,
        "objective_components": objective_values,
        "gradient_l1": gradient_l1,
        "mechanism_activity": mechanism,
        "fallback_reason": episode_artifact.get("fallback_reason"),
        "state_transform_sha256": str(state_transform_sha256),
        "checkpoint_sha256": checkpoint_sha256,
        "parent_family_isolation": dict(
            episode_artifact.get("parent_family_isolation_audit", {})
        ),
        "outer_query_synthetic_count": sum(
            str(case_key).startswith("synthetic:")
            for episode in episode_artifact.get("episodes", ())
            for case_key in episode.get("outer_case_keys", ())
        ),
    }
    audit = {**audit_identity, "audit_sha256": _semantic_hash(audit_identity)}
    return FinalOSERFit(
        model=model,
        episode_artifact=dict(episode_artifact),
        profile=frozen,
        audit=audit,
    )


def score_final_oser(
    *,
    fitted: FinalOSERFit,
    inference_cases: Mapping[str, Mapping[str, Any]],
    base_score_artifact: Mapping[str, Any],
    artifact_role: str,
) -> ConservativeResidualOutput:
    rows = build_oser_residual_rows(
        fitted.model, inference_cases, fitted.episode_artifact
    )
    profile = {
        "arm": "oser_meta",
        "profile_id": fitted.profile["profile_id"],
        "residual_cap": fitted.profile["residual_cap"],
        "gate_threshold": fitted.profile["gate_threshold"],
    }
    checkpoint = {
        "owner_arm": "oser_meta",
        "checkpoint_sha256": fitted.audit["checkpoint_sha256"],
        "query_plan_sha256": base_score_artifact["query_plan_sha256"],
    }
    return apply_conservative_residual(
        base_artifact=base_score_artifact,
        residual_rows=rows,
        selected_profile=profile,
        checkpoint_ref=checkpoint,
        artifact_role=artifact_role,
    )


__all__ = [
    "FinalOSERFit",
    "FinalOSERValidationError",
    "build_final_oser_family_episodes",
    "fit_final_oser",
    "score_final_oser",
]
