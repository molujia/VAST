from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

    from .service_continuous_cvae import FactorizedConditionalVAE


class ProxyModeCVAEValidationError(ValueError):
    """Raised when proxy-mode support or generation violates its contract."""


_FACTORS = ("mechanism", "propagation", "context")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "fault_type",
        "root_cause",
        "rootcause",
        "ground_truth",
        "groundtruth",
        "label",
        "labels",
        "injection_type",
        "injection_target",
    }
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


def _ids(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProxyModeCVAEValidationError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if (
        (not allow_empty and not result)
        or any(not item for item in result)
        or len(result) != len(set(result))
    ):
        raise ProxyModeCVAEValidationError(
            f"{context} must contain unique nonempty IDs"
        )
    return result


def _finite_fraction(value: Any, context: str, *, open_left: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProxyModeCVAEValidationError(f"{context} must be numeric")
    result = float(value)
    lower_valid = result > 0.0 if open_left else result >= 0.0
    if not math.isfinite(result) or not lower_valid or result > 1.0:
        raise ProxyModeCVAEValidationError(f"{context} must be in [0, 1]")
    return result


def _aggregate_case_embeddings(
    records: Sequence[Mapping[str, Any]], fit_case_ids: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    by_case: dict[str, list[Mapping[str, Any]]] = {
        case_id: [] for case_id in fit_case_ids
    }
    dimensions: dict[str, int] = {}
    seen_keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != {
            "case_id",
            "candidate_id",
            "state",
            "mask",
        }:
            leaked = (
                set(raw) & _FORBIDDEN_LABEL_FIELDS
                if isinstance(raw, Mapping)
                else set()
            )
            detail = "forbidden label field" if leaked else "record closure drift"
            raise ProxyModeCVAEValidationError(
                f"proxy-mode record {index} has {detail}"
            )
        case_id = str(raw["case_id"]).strip()
        candidate_id = str(raw["candidate_id"]).strip()
        if case_id not in by_case:
            raise ProxyModeCVAEValidationError(
                "held-out or non-fit case entered proxy-mode fitting"
            )
        key = (case_id, candidate_id)
        if not candidate_id or key in seen_keys:
            raise ProxyModeCVAEValidationError(
                "proxy-mode record identity is empty or duplicated"
            )
        seen_keys.add(key)
        state = raw["state"]
        mask = raw["mask"]
        if (
            not isinstance(state, Mapping)
            or set(state) != set(_FACTORS)
            or not isinstance(mask, Mapping)
            or set(mask) != set(_FACTORS)
        ):
            raise ProxyModeCVAEValidationError("proxy-mode factor closure drift")
        for factor in _FACTORS:
            values = np.asarray(state[factor], dtype=float)
            masks = np.asarray(mask[factor], dtype=float)
            if (
                values.ndim != 1
                or values.size == 0
                or masks.shape != values.shape
                or not np.isfinite(values).all()
                or not np.isin(masks, (0.0, 1.0)).all()
            ):
                raise ProxyModeCVAEValidationError(
                    f"proxy-mode {factor} state/mask drift"
                )
            previous = dimensions.setdefault(factor, int(values.size))
            if previous != int(values.size):
                raise ProxyModeCVAEValidationError(
                    f"proxy-mode {factor} dimension drift"
                )
        by_case[case_id].append(raw)
    if any(not rows for rows in by_case.values()):
        raise ProxyModeCVAEValidationError(
            "every fit case must contribute observable proxy-mode rows"
        )

    vectors: list[list[float]] = []
    feature_names: list[str] = []
    for factor in _FACTORS:
        for position in range(dimensions[factor]):
            feature_names.extend(
                (f"{factor}.{position}.masked_mean", f"{factor}.{position}.coverage")
            )
    for case_id in fit_case_ids:
        vector: list[float] = []
        rows = by_case[case_id]
        for factor in _FACTORS:
            values = np.asarray([row["state"][factor] for row in rows], dtype=float)
            masks = np.asarray([row["mask"][factor] for row in rows], dtype=float)
            observed = masks.sum(axis=0)
            means = (values * masks).sum(axis=0) / np.maximum(observed, 1.0)
            coverage = observed / float(len(rows))
            for mean, support in zip(means, coverage):
                vector.extend((float(mean), float(support)))
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ProxyModeCVAEValidationError("proxy-mode case embeddings are invalid")
    return matrix, feature_names


def _dbscan_labels(
    matrix: np.ndarray, *, eps_quantile: float, min_samples: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    count = int(matrix.shape[0])
    minimum = min(int(min_samples), count)
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    ordered = np.sort(distances, axis=1)
    kth = ordered[:, minimum - 1]
    eps = max(float(np.quantile(kth, eps_quantile)), 1e-9)
    neighborhoods = distances <= eps + 1e-12
    core = neighborhoods.sum(axis=1) >= minimum
    labels = np.full(count, -1, dtype=int)
    next_label = 0
    for start in range(count):
        if not core[start] or labels[start] >= 0:
            continue
        labels[start] = next_label
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in np.flatnonzero(neighborhoods[current] & core):
                index = int(neighbor)
                if labels[index] < 0:
                    labels[index] = next_label
                    queue.append(index)
        next_label += 1
    core_indices = np.flatnonzero(core)
    for index in np.flatnonzero(~core):
        candidates = [
            int(core_index)
            for core_index in core_indices
            if neighborhoods[index, core_index]
        ]
        if candidates:
            chosen = min(
                candidates,
                key=lambda item: (distances[index, item], labels[item], item),
            )
            labels[index] = labels[chosen]
    return labels, eps, kth, core


def build_proxy_mode_partition(
    *,
    records: Sequence[Mapping[str, Any]],
    fit_case_ids: Sequence[str],
    queried_case_ids: Sequence[str],
    held_out_case_ids: Sequence[str],
    eps_quantile: float = 0.8,
    min_samples: int = 6,
) -> dict[str, Any]:
    """Fit deterministic fold-local proxy modes without consuming labels."""

    fit_ids = _ids(fit_case_ids, "fit case IDs")
    queried_ids = _ids(queried_case_ids, "queried case IDs")
    held_out_ids = _ids(held_out_case_ids, "held-out case IDs", allow_empty=True)
    if not set(queried_ids) <= set(fit_ids):
        raise ProxyModeCVAEValidationError(
            "queried cases must be inside the fold-local fit set"
        )
    overlap = set(fit_ids) & set(held_out_ids)
    if overlap:
        raise ProxyModeCVAEValidationError(
            "held-out cases are forbidden in proxy-mode fitting"
        )
    quantile = _finite_fraction(eps_quantile, "eps_quantile", open_left=True)
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ProxyModeCVAEValidationError("min_samples must be an integer >= 2")
    matrix, feature_names = _aggregate_case_embeddings(records, fit_ids)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    normalized = (matrix - mean) / scale
    labels, eps, kth, core = _dbscan_labels(
        normalized, eps_quantile=quantile, min_samples=min_samples
    )
    members_by_label: dict[int, list[str]] = defaultdict(list)
    for case_id, label in zip(fit_ids, labels.tolist()):
        if label >= 0:
            members_by_label[int(label)].append(case_id)
    mode_by_label = {
        label: "mode:"
        + hashlib.sha256("\0".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        for label, members in members_by_label.items()
    }
    case_mode_ids = {
        case_id: (
            mode_by_label[int(label)]
            if label >= 0
            else "noise:"
            + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
        )
        for case_id, label in zip(fit_ids, labels.tolist())
    }
    queried_mode_members: dict[str, list[str]] = defaultdict(list)
    for case_id in queried_ids:
        queried_mode_members[case_mode_ids[case_id]].append(case_id)
    normalization_identity = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    identity = {
        "schema_version": "service-continuous-proxy-mode-partition-v1",
        "algorithm": "fold_local_dbscan_case_state",
        "eps_quantile": quantile,
        "resolved_eps": eps,
        "min_samples": int(min_samples),
        "fit_case_ids": fit_ids,
        "queried_case_ids": queried_ids,
        "held_out_case_ids_sha256": _semantic_hash(held_out_ids),
        "held_out_overlap_count": 0,
        "label_fields_consumed": [],
        "feature_names": feature_names,
        "normalization_sha256": _semantic_hash(normalization_identity),
        "case_embedding_sha256": _semantic_hash(normalized.tolist()),
        "k_distance_sha256": _semantic_hash(kth.tolist()),
        "core_case_count": int(core.sum()),
        "noise_case_count": int((labels < 0).sum()),
        "case_mode_ids": case_mode_ids,
        "queried_mode_members": dict(sorted(queried_mode_members.items())),
    }
    return {**identity, "partition_sha256": _semantic_hash(identity)}


def validate_proxy_mode_partition(
    partition: Mapping[str, Any],
    *,
    expected_fit_case_ids: Sequence[str],
    expected_queried_case_ids: Sequence[str],
    expected_held_out_case_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(partition, Mapping):
        raise ProxyModeCVAEValidationError("proxy-mode partition must be a mapping")
    value = deepcopy(dict(partition))
    supplied = str(value.pop("partition_sha256", ""))
    if (
        value.get("schema_version")
        != "service-continuous-proxy-mode-partition-v1"
        or supplied != _semantic_hash(value)
        or value.get("label_fields_consumed") != []
        or int(value.get("held_out_overlap_count", -1)) != 0
        or list(value.get("fit_case_ids", ())) != list(expected_fit_case_ids)
        or list(value.get("queried_case_ids", ())) != list(expected_queried_case_ids)
        or value.get("held_out_case_ids_sha256")
        != _semantic_hash(list(expected_held_out_case_ids))
    ):
        raise ProxyModeCVAEValidationError(
            "proxy-mode partition identity or Strict-LOFO binding drift"
        )
    case_modes = value.get("case_mode_ids")
    queried_members = value.get("queried_mode_members")
    if (
        not isinstance(case_modes, Mapping)
        or set(case_modes) != set(expected_fit_case_ids)
        or not isinstance(queried_members, Mapping)
    ):
        raise ProxyModeCVAEValidationError("proxy-mode membership closure drift")
    flattened = [
        str(case_id)
        for members in queried_members.values()
        for case_id in members
    ]
    if sorted(flattened) != sorted(str(item) for item in expected_queried_case_ids):
        raise ProxyModeCVAEValidationError("queried proxy-mode coverage drift")
    return {**value, "partition_sha256": supplied}


def validate_proxy_mode_support(support: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a support digest identifies its final persisted content."""

    if not isinstance(support, Mapping):
        raise ProxyModeCVAEValidationError("proxy-mode support must be a mapping")
    value = deepcopy(dict(support))
    supplied = str(value.pop("support_sha256", ""))
    members = _ids(
        value.get("queried_member_case_ids"),
        "proxy-mode support queried member case IDs",
    )
    if (
        value.get("schema_version")
        != "service-continuous-proxy-mode-latent-support-v1"
        or members != sorted(members)
        or not _SHA256.fullmatch(supplied)
        or supplied != _semantic_hash(value)
    ):
        raise ProxyModeCVAEValidationError("proxy-mode support identity drift")
    return {**value, "support_sha256": supplied}


def _validated_targets(
    raw_targets: Sequence[Mapping[str, Any]], source_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    from .service_continuous_cvae_generation import _tensor

    if isinstance(raw_targets, (str, bytes)) or not isinstance(raw_targets, Sequence):
        raise ProxyModeCVAEValidationError("target contexts must be a sequence")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise ProxyModeCVAEValidationError(f"target {index} must be a mapping")
        target_id = str(raw.get("target_service_id", "")).strip()
        reasons = list(raw.get("hard_rejection_reasons", ()))
        eligible = raw.get("hard_eligible") is True
        context = _tensor(raw.get("target_context"), f"target {target_id} context")
        context_mask = _tensor(
            raw.get("target_context_mask"), f"target {target_id} context mask"
        )
        compatibility = _finite_fraction(
            raw.get("compatibility"), f"target {target_id} compatibility"
        )
        if (
            not target_id
            or target_id in seen
            or context.shape != context_mask.shape
            or bool(torch.any((context_mask < 0.0) | (context_mask > 1.0)))
            or not _SHA256.fullmatch(str(raw.get("target_weight_plan_sha256", "")))
            or not _SHA256.fullmatch(str(raw.get("compatibility_sha256", "")))
        ):
            raise ProxyModeCVAEValidationError("target context identity drift")
        seen.add(target_id)
        entry = {
            "target_service_id": target_id,
            "target_context": context,
            "target_context_mask": context_mask,
            "target_weight": float(raw.get("target_weight", 0.0)),
            "compatibility": compatibility,
            "target_weight_plan_sha256": str(raw["target_weight_plan_sha256"]),
            "compatibility_sha256": str(raw["compatibility_sha256"]),
            "hard_eligible": eligible,
            "hard_rejection_reasons": [str(reason) for reason in reasons],
        }
        if eligible:
            if reasons or not bool(torch.any(context_mask > 0.0)):
                raise ProxyModeCVAEValidationError(
                    "hard-eligible target has rejection reasons or no context support"
                )
            accepted.append(entry)
        else:
            if not reasons:
                raise ProxyModeCVAEValidationError(
                    "hard-ineligible target must record a rejection reason"
                )
            rejected.append(
                {
                    "source_case_id": source_id,
                    "target_service_id": target_id,
                    "reasons": entry["hard_rejection_reasons"],
                }
            )
    if not accepted:
        raise ProxyModeCVAEValidationError(
            f"source {source_id} has no hard-compatible service context"
        )
    total = math.fsum(max(row["compatibility"], 1e-6) for row in accepted)
    for row in accepted:
        row["target_weight"] = max(row["compatibility"], 1e-6) / total
    return accepted, rejected


def _posterior_vector(posterior: Mapping[str, Mapping[str, Tensor]]) -> tuple[Tensor, Tensor]:
    import torch

    return (
        torch.cat([posterior[factor]["mu"] for factor in _FACTORS], dim=1),
        torch.cat([posterior[factor]["logvar"] for factor in _FACTORS], dim=1),
    )


def _coverage_bank(
    anchors: Mapping[str, tuple[Tensor, Tensor]],
    *,
    desired_count: int,
    radius: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from .service_continuous_cvae import sample_posterior
    from .service_continuous_cvae_generation import _tensor_hash

    def clone_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
        copied = dict(row)
        copied["value"] = row["value"].detach().clone()
        copied["anchor_case_ids"] = list(row["anchor_case_ids"])
        return copied

    ordered_ids = sorted(anchors)
    candidates: list[dict[str, Any]] = []
    for case_id in ordered_ids:
        mu, _ = anchors[case_id]
        candidates.append(
            {
                "value": mu,
                "origin": "queried_posterior_mean",
                "anchor_case_ids": [case_id],
                "latent_seed": seed,
            }
        )
    for left_id, right_id in combinations(ordered_ids, 2):
        left = anchors[left_id][0]
        right = anchors[right_id][0]
        for alpha in (0.25, 0.5, 0.75):
            candidates.append(
                {
                    "value": (1.0 - alpha) * left + alpha * right,
                    "origin": "pair_interpolation",
                    "anchor_case_ids": [left_id, right_id],
                    "interpolation_alpha": alpha,
                    "latent_seed": seed,
                }
            )
    posterior_draws = max(2, int(math.ceil(desired_count / max(len(ordered_ids), 1))))
    for source_index, case_id in enumerate(ordered_ids):
        mu, logvar = anchors[case_id]
        for draw_index in range(posterior_draws):
            latent_seed = seed + source_index * 1009 + draw_index * 17 + 1
            proposed = sample_posterior(mu, logvar, seed=latent_seed)
            delta = proposed - mu
            distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            bounded = mu + delta * torch.clamp(
                radius / distance.clamp_min(1e-12), max=1.0
            )
            candidates.append(
                {
                    "value": bounded,
                    "origin": "bounded_posterior",
                    "anchor_case_ids": [case_id],
                    "latent_seed": latent_seed,
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(_tensor_hash(candidate["value"]), candidate)
    bank = list(unique.values())
    matrix = torch.cat([row["value"] for row in bank], dim=0)
    centroid = torch.mean(
        torch.cat([anchors[case_id][0] for case_id in ordered_ids], dim=0), dim=0
    )
    selected: list[int] = []
    first = max(
        range(len(bank)),
        key=lambda index: (
            float(torch.linalg.vector_norm(matrix[index] - centroid).item()),
            _tensor_hash(bank[index]["value"]),
        ),
    )
    selected.append(first)
    if len(bank) > 1 and desired_count > 1:
        second = max(
            (index for index in range(len(bank)) if index not in selected),
            key=lambda index: (
                float(torch.linalg.vector_norm(matrix[index] - matrix[first]).item()),
                _tensor_hash(bank[index]["value"]),
            ),
        )
        selected.append(second)
    if len(ordered_ids) > 1 and desired_count > 2:
        midpoint_candidates = [
            index
            for index, row in enumerate(bank)
            if row["origin"] == "pair_interpolation"
            and math.isclose(float(row.get("interpolation_alpha", -1.0)), 0.5)
            and index not in selected
        ]
        if midpoint_candidates:
            selected.append(
                min(
                    midpoint_candidates,
                    key=lambda index: (
                        float(torch.linalg.vector_norm(matrix[index] - centroid).item()),
                        _tensor_hash(bank[index]["value"]),
                    ),
                )
            )
    while len(selected) < min(desired_count, len(bank)):
        remaining = [index for index in range(len(bank)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(
                    float(
                        torch.linalg.vector_norm(matrix[index] - matrix[chosen]).item()
                    )
                    for chosen in selected
                ),
                _tensor_hash(bank[index]["value"]),
            ),
        )
        selected.append(next_index)
    chosen_rows = [clone_candidate(bank[index]) for index in selected]
    if chosen_rows and len(chosen_rows) < desired_count:
        chosen_rows.extend(
            clone_candidate(chosen_rows[index % len(chosen_rows)])
            for index in range(desired_count - len(chosen_rows))
        )
    anchor_matrix = torch.cat([anchors[case_id][0] for case_id in ordered_ids], dim=0)
    support_scale = max(
        radius,
        max(
            float(torch.linalg.vector_norm(anchor - centroid).item())
            for anchor in anchor_matrix
        ),
        1e-6,
    )
    for coverage_rank, row in enumerate(chosen_rows):
        nearest = min(
            float(torch.linalg.vector_norm(row["value"] - anchor).item())
            for anchor in anchor_matrix
        )
        row["coverage_rank"] = coverage_rank
        row["valid"] = row["origin"] in {
            "queried_posterior_mean",
            "pair_interpolation",
            "bounded_posterior",
        }
        row["quality"] = max(1e-6, math.exp(-nearest / support_scale))
        row["latent_sha256"] = _tensor_hash(row["value"])
    pairwise = []
    for left, right in combinations(chosen_rows, 2):
        pairwise.append(
            float(torch.linalg.vector_norm(left["value"] - right["value"]).item())
        )
    audit = {
        "schema_version": "service-continuous-proxy-mode-latent-support-v1",
        "queried_member_case_ids": ordered_ids,
        "shared_support": len(ordered_ids) > 1,
        "fallback": (
            None if len(ordered_ids) > 1 else "singleton_mode"
        ),
        "coverage_sampling": "deterministic_farthest_point",
        "candidate_bank_count": len(bank),
        "selected_count": len(chosen_rows),
        "selected_unique_count": len(
            {row["latent_sha256"] for row in chosen_rows}
        ),
        "support_radius": radius,
        "support_scale": support_scale,
        "minimum_selected_pairwise_distance": min(pairwise) if pairwise else 0.0,
        "selected_latent_sha256s": [row["latent_sha256"] for row in chosen_rows],
        "selected_origins": [row["origin"] for row in chosen_rows],
    }
    return chosen_rows, audit


def generate_proxy_mode_cvae_service_continuity(
    *,
    model: FactorizedConditionalVAE,
    source_cases: Sequence[Mapping[str, Any]],
    target_contexts_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    proxy_mode_partition: Mapping[str, Any],
    profile: Mapping[str, Any],
    sample_seed: int,
    checkpoint_sha256: str,
    activity_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate mode-support coverage samples under compatible contexts."""

    from .service_continuous_cvae import FactorizedConditionalVAE
    from .service_continuous_cvae_generation import (
        _decode,
        _encode,
        _profile,
        _source_case,
        _tensor_hash,
    )
    from .service_continuous_cvae_provenance import build_cvae_row_provenance

    if not isinstance(model, FactorizedConditionalVAE):
        raise ProxyModeCVAEValidationError("a factorized conditional VAE is required")
    if isinstance(source_cases, (str, bytes)) or not isinstance(source_cases, Sequence):
        raise ProxyModeCVAEValidationError("source_cases must be a sequence")
    sources = [_source_case(row, "proxy-mode source") for row in source_cases]
    source_ids = [row["case_id"] for row in sources]
    if not sources or len(source_ids) != len(set(source_ids)):
        raise ProxyModeCVAEValidationError("proxy-mode sources must be unique")
    frozen_profile = _profile(profile)
    seed = int(sample_seed)
    if seed < 0 or not _SHA256.fullmatch(str(checkpoint_sha256)):
        raise ProxyModeCVAEValidationError("sample seed or checkpoint hash drift")
    case_modes = dict(proxy_mode_partition.get("case_mode_ids", {}))
    queried_mode_members = dict(
        proxy_mode_partition.get("queried_mode_members", {})
    )
    if any(source_id not in case_modes for source_id in source_ids):
        raise ProxyModeCVAEValidationError(
            "queried source is absent from the proxy-mode partition"
        )
    partition_sha256 = str(
        proxy_mode_partition.get(
            "proxy_partition_sha256",
            proxy_mode_partition.get("partition_sha256", ""),
        )
    )

    valid_targets: dict[str, list[dict[str, Any]]] = {}
    rejected_targets: list[dict[str, Any]] = []
    posteriors: dict[str, tuple[Tensor, Tensor, Mapping[str, Any]]] = {}
    for source in sources:
        source_id = source["case_id"]
        accepted, rejected = _validated_targets(
            target_contexts_by_source.get(source_id, ()), source_id
        )
        valid_targets[source_id] = accepted
        rejected_targets.extend(rejected)
        posterior = _encode(model, source, accepted[0])["posterior"]
        mu, logvar = _posterior_vector(posterior)
        posteriors[source_id] = (mu, logvar, posterior)

    sources_by_mode: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        sources_by_mode[str(case_modes[source_id])].append(source_id)
    mode_samples: dict[str, list[dict[str, Any]]] = {}
    mode_audits: dict[str, dict[str, Any]] = {}
    for mode_offset, (mode_id, members) in enumerate(sorted(sources_by_mode.items())):
        desired = sum(
            len(valid_targets[source_id]) * frozen_profile["samples_per_target"]
            for source_id in members
        )
        selected, audit = _coverage_bank(
            {
                source_id: (posteriors[source_id][0], posteriors[source_id][1])
                for source_id in members
            },
            desired_count=desired,
            radius=frozen_profile["sampling_radius"],
            seed=seed + mode_offset * 1_000_003,
        )
        expected_members = [
            source_id
            for source_id in queried_mode_members.get(mode_id, ())
            if source_id in set(source_ids)
        ]
        audit["queried_member_case_ids"] = sorted(expected_members or members)
        if mode_id.startswith("noise:") and len(members) == 1:
            audit["fallback"] = "noise_singleton"
        audit["support_sha256"] = _semantic_hash(audit)
        mode_samples[mode_id] = selected
        mode_audits[mode_id] = validate_proxy_mode_support(audit)

    rows: list[dict[str, Any]] = []
    mode_offsets = {mode_id: 0 for mode_id in mode_samples}
    for source in sources:
        source_id = source["case_id"]
        mode_id = str(case_modes[source_id])
        posterior = posteriors[source_id][2]
        source_posterior_identity = {
            factor: {
                "mu_sha256": _tensor_hash(posterior[factor]["mu"]),
                "logvar_sha256": _tensor_hash(posterior[factor]["logvar"]),
            }
            for factor in _FACTORS
        }
        samples = mode_samples[mode_id]
        mechanism_width = int(posterior["mechanism"]["mu"].shape[1])
        propagation_width = int(posterior["propagation"]["mu"].shape[1])
        for target in valid_targets[source_id]:
            for sample_index in range(frozen_profile["samples_per_target"]):
                support = samples[mode_offsets[mode_id] % len(samples)]
                mode_offsets[mode_id] += 1
                latent = support["value"]
                decoded = _decode(
                    model,
                    mechanism_z=latent[:, :mechanism_width],
                    propagation_z=latent[
                        :, mechanism_width : mechanism_width + propagation_width
                    ],
                    context_z=latent[:, mechanism_width + propagation_width :],
                    target=target,
                )
                decoded = {
                    factor: value.detach().clone()
                    for factor, value in decoded.items()
                }
                decoded_hashes = {
                    factor: _tensor_hash(decoded[factor]) for factor in _FACTORS
                }
                decoded_sha = _semantic_hash(decoded_hashes)
                provisional_weight = float(support["quality"]) * float(
                    target["compatibility"]
                )
                row_identity = {
                    "schema_version": "service-continuous-proxy-mode-cvae-row-v1",
                    "arm_id": "proxy_mode_cvae_compatible",
                    "source_case_id": source_id,
                    "target_service_id": target["target_service_id"],
                    "sample_index": sample_index,
                    "latent_seed": int(support["latent_seed"]),
                    "profile_id": frozen_profile["profile_id"],
                    "profile_sha256": frozen_profile["profile_sha256"],
                    "proxy_mode_id": mode_id,
                    "proxy_mode_partition_sha256": partition_sha256,
                    "mode_support_sha256": mode_audits[mode_id]["support_sha256"],
                    "mode_member_case_ids": mode_audits[mode_id][
                        "queried_member_case_ids"
                    ],
                    "coverage_rank": int(support["coverage_rank"]),
                    "latent_support_origin": support["origin"],
                    "latent_support_anchor_case_ids": support["anchor_case_ids"],
                    "latent_support_valid": bool(support["valid"]),
                    "latent_support_quality": float(support["quality"]),
                    "latent_sha256": support["latent_sha256"],
                    "decoded_state_sha256": decoded_sha,
                    "target_weight": target["target_weight"],
                    "compatibility": target["compatibility"],
                    "provisional_training_weight": provisional_weight,
                    "target_weight_plan_sha256": target[
                        "target_weight_plan_sha256"
                    ],
                    "compatibility_sha256": target["compatibility_sha256"],
                    "query_budget_cost": 0,
                }
                base_provenance = build_cvae_row_provenance(
                    arm_id="proxy_mode_cvae_compatible",
                    checkpoint_sha256=str(checkpoint_sha256),
                    profile_id=frozen_profile["profile_id"],
                    profile_sha256=frozen_profile["profile_sha256"],
                    source_case_id=source_id,
                    queried_label_sha256=_semantic_hash(source["queried_label"]),
                    source_posterior=source_posterior_identity,
                    latent_seed=int(support["latent_seed"]),
                    interpolation={
                        "same_fault_type_partner_case_id": None,
                        "alpha": None,
                        "mechanism_latent_sha256": _tensor_hash(
                            latent[:, :mechanism_width]
                        ),
                        "propagation_latent_sha256": _tensor_hash(
                            latent[
                                :,
                                mechanism_width : mechanism_width
                                + propagation_width,
                            ]
                        ),
                        "context_latent_sha256": _tensor_hash(
                            latent[:, mechanism_width + propagation_width :]
                        ),
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
                    decoded_state_sha256=decoded_sha,
                    training_weight=min(1.0, provisional_weight),
                    activity_evidence=activity_evidence,
                )
                proxy_provenance_identity = {
                    "schema_version": "service-continuous-proxy-mode-cvae-provenance-v1",
                    "base_cvae_provenance": base_provenance,
                    "proxy_mode_id": mode_id,
                    "proxy_mode_partition_sha256": row_identity[
                        "proxy_mode_partition_sha256"
                    ],
                    "mode_support_sha256": row_identity["mode_support_sha256"],
                    "latent_support_origin": support["origin"],
                    "latent_support_quality": float(support["quality"]),
                    "coverage_rank": int(support["coverage_rank"]),
                }
                rows.append(
                    {
                        **row_identity,
                        "synthetic_label": {
                            "root_cause": target["target_service_id"],
                            "fault_type": source["queried_label"]["fault_type"],
                            "label_source_case_id": source_id,
                            "label_source": (
                                "queried_budget_transfer"
                                if source["queried_label"]["label_source"]
                                == "queried_budget"
                                else "oracle_full_transfer"
                            ),
                        },
                        "decoded_mechanism": decoded["mechanism"],
                        "decoded_propagation": decoded["propagation"],
                        "decoded_context": decoded["context"],
                        "provenance": {
                            **proxy_provenance_identity,
                            "provenance_sha256": _semantic_hash(
                                proxy_provenance_identity
                            ),
                        },
                        "synthetic_row_sha256": _semantic_hash(row_identity),
                    }
                )

    audit_identity = {
        "schema_version": "service-continuous-proxy-mode-cvae-bundle-v1",
        "arm_id": "proxy_mode_cvae_compatible",
        "source_case_ids": source_ids,
        "synthetic_row_count": len(rows),
        "rejected_target_count": len(rejected_targets),
        "rejected_targets": rejected_targets,
        "mode_support_audits": mode_audits,
        "label_access_audit": {
            "label_case_ids": source_ids,
            "proxy_mode_label_fields_consumed": [],
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
    "ProxyModeCVAEValidationError",
    "build_proxy_mode_partition",
    "generate_proxy_mode_cvae_service_continuity",
    "validate_proxy_mode_partition",
    "validate_proxy_mode_support",
]
