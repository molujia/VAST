from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .service_continuous_cvae import FactorizedConditionalVAE
from .service_continuous_cvae_activity import build_cvae_activity_audit
from .service_continuous_cvae_generation import generate_cvae_service_continuity
from .service_continuous_cvae_training import compute_unlabeled_objectives
from .service_continuous_proxy_mode_cvae import (
    generate_proxy_mode_cvae_service_continuity,
    validate_proxy_mode_partition,
)
from .final_rcl_hdbscan_proxy import validate_hdbscan_proxy_partition


class NeuralHandshakeValidationError(ValueError):
    """Raised when the neural execution-plane artifact contract drifts."""


_FACTORS = ("mechanism", "propagation", "context")
_CASE_LOCAL_ARMS = ("cvae_arbitrary", "cvae_compatible")
_ARMS = (*_CASE_LOCAL_ARMS, "proxy_mode_cvae_compatible")


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeuralHandshakeValidationError(f"{context} must be a positive integer")
    return value


def _request_v1(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuralHandshakeValidationError("neural request must be a mapping")
    request = dict(value)
    identity = {key: item for key, item in request.items() if key != "request_sha256"}
    if (
        request.get("schema_version")
        != "service-continuous-neural-handshake-request-v1"
        or request.get("dataset_id") != "rcabench"
        or request.get("held_out_fault_type") not in {"NetworkDelay", None}
        or request.get("training_seed") != 42
        or request.get("request_sha256") != _semantic_hash(identity)
    ):
        raise NeuralHandshakeValidationError("neural request identity drift")
    fit_ids = [str(item) for item in request.get("fit_case_ids", ())]
    held_out_ids = [str(item) for item in request.get("held_out_case_ids", ())]
    records = request.get("pretraining_records")
    plans = request.get("target_plans")
    profile = request.get("profile")
    if (
        not fit_ids
        or len(set(fit_ids)) != len(fit_ids)
        or set(fit_ids) & set(held_out_ids)
        or isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or len(records) != len(fit_ids) * 54
        or not isinstance(plans, Mapping)
        or set(plans) != set(_CASE_LOCAL_ARMS)
        or not isinstance(profile, Mapping)
    ):
        raise NeuralHandshakeValidationError("neural membership or profile closure drift")
    source_id = str(request.get("source_case_id", ""))
    source_root = str(request.get("source_root_service_id", ""))
    if not source_id or source_id not in set(fit_ids) or not source_root:
        raise NeuralHandshakeValidationError("neural source ownership drift")
    _positive_int(request.get("optimizer_steps"), "optimizer_steps")
    _positive_int(request.get("samples_per_target"), "samples_per_target")
    return request


def _ids(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise NeuralHandshakeValidationError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if (
        (not allow_empty and not result)
        or any(not item for item in result)
        or len(result) != len(set(result))
    ):
        raise NeuralHandshakeValidationError(
            f"{context} must contain unique nonempty IDs"
        )
    return result


def _request_v2(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuralHandshakeValidationError("neural request must be a mapping")
    request = dict(value)
    identity = {key: item for key, item in request.items() if key != "request_sha256"}
    if (
        request.get("schema_version")
        != "service-continuous-neural-handshake-request-v2"
        or request.get("dataset_id") != "rcabench"
        or request.get("training_seed") != 42
        or request.get("request_sha256") != _semantic_hash(identity)
    ):
        raise NeuralHandshakeValidationError("neural v2 request identity drift")
    fit_ids = _ids(request.get("fit_case_ids"), "fit case IDs")
    queried_ids = _ids(request.get("queried_case_ids"), "queried case IDs")
    held_out_ids = _ids(
        request.get("held_out_case_ids"), "held-out case IDs", allow_empty=True
    )
    if (
        not queried_ids
        or not set(queried_ids) <= set(fit_ids)
        or set(fit_ids) & set(held_out_ids)
        or set(queried_ids) & set(held_out_ids)
    ):
        raise NeuralHandshakeValidationError("neural v2 membership closure drift")
    expected = _positive_int(
        request.get("expected_candidates_per_case"),
        "expected_candidates_per_case",
    )
    records = request.get("pretraining_records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise NeuralHandshakeValidationError("pretraining records must be a sequence")
    record_keys: set[tuple[str, str]] = set()
    case_counts = {case_id: 0 for case_id in fit_ids}
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "case_id",
            "candidate_id",
            "state",
            "mask",
        }:
            raise NeuralHandshakeValidationError(
                "neural v2 pretraining record closure drift"
            )
        case_id = str(raw.get("case_id", "")).strip()
        candidate_id = str(raw.get("candidate_id", "")).strip()
        key = (case_id, candidate_id)
        if (
            case_id not in case_counts
            or not candidate_id
            or key in record_keys
        ):
            raise NeuralHandshakeValidationError(
                "neural v2 pretraining ownership drift"
            )
        record_keys.add(key)
        case_counts[case_id] += 1
    if len(records) != len(fit_ids) * expected or any(
        count != expected for count in case_counts.values()
    ):
        raise NeuralHandshakeValidationError(
            "neural v2 pretraining coverage drift"
        )
    sources = request.get("source_cases")
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise NeuralHandshakeValidationError("source cases must be a sequence")
    normalized_sources = []
    for raw in sources:
        if not isinstance(raw, Mapping) or set(raw) != {
            "source_case_id",
            "source_root_service_id",
            "queried_label",
            "target_plans",
        }:
            raise NeuralHandshakeValidationError("source case closure drift")
        source_id = str(raw.get("source_case_id", "")).strip()
        source_root = str(raw.get("source_root_service_id", "")).strip()
        plans = raw.get("target_plans")
        if (
            source_id not in set(queried_ids)
            or not source_root
            or (source_id, source_root) not in record_keys
            or not isinstance(plans, Mapping)
            or set(plans) != set(_ARMS)
            or any(
                isinstance(plans[arm], (str, bytes))
                or not isinstance(plans[arm], Sequence)
                or not plans[arm]
                for arm in _ARMS
            )
        ):
            raise NeuralHandshakeValidationError("source ownership or target plan drift")
        for arm in _ARMS:
            for target in plans[arm]:
                target_id = str(dict(target).get("target_service_id", "")).strip()
                if not target_id or (source_id, target_id) not in record_keys:
                    raise NeuralHandshakeValidationError(
                        "source target context is outside the unlabeled pool"
                    )
        normalized_sources.append(dict(raw))
    if [str(row["source_case_id"]) for row in normalized_sources] != queried_ids:
        raise NeuralHandshakeValidationError(
            "source cases must cover all queried cases in order"
        )
    try:
        validate_proxy_mode_partition(
            request.get("proxy_mode_partition"),
            expected_fit_case_ids=fit_ids,
            expected_queried_case_ids=queried_ids,
            expected_held_out_case_ids=held_out_ids,
        )
    except Exception as exc:
        raise NeuralHandshakeValidationError(
            f"proxy-mode partition binding drift: {exc}"
        ) from exc
    if not isinstance(request.get("profile"), Mapping):
        raise NeuralHandshakeValidationError("neural v2 profile closure drift")
    checkpoint_output = request.get("checkpoint_output_path")
    if checkpoint_output is not None and (
        not isinstance(checkpoint_output, str)
        or not checkpoint_output.strip()
        or not Path(checkpoint_output).is_absolute()
    ):
        raise NeuralHandshakeValidationError(
            "checkpoint_output_path must be an absolute path"
        )
    _positive_int(request.get("optimizer_steps"), "optimizer_steps")
    _positive_int(request.get("samples_per_target"), "samples_per_target")
    return request


def _request_v3(value: Any) -> dict[str, Any]:
    """Validate the final HDBSCAN-only proxy-CVAE request."""

    if not isinstance(value, Mapping):
        raise NeuralHandshakeValidationError("neural v3 request must be a mapping")
    request = dict(value)
    identity = {key: item for key, item in request.items() if key != "request_sha256"}
    if (
        request.get("schema_version")
        != "service-continuous-neural-handshake-request-v3"
        or request.get("dataset_id") not in {"rcabench", "aiops2022_pre"}
        or request.get("training_mode") not in {"query_only", "oracle_full"}
        or request.get("training_seed") != 42
        or request.get("request_sha256") != _semantic_hash(identity)
    ):
        raise NeuralHandshakeValidationError("neural v3 request identity drift")
    fit_ids = _ids(request.get("fit_case_ids"), "fit case IDs")
    supervised_ids = _ids(request.get("supervised_case_ids"), "supervised case IDs")
    held_out_ids = _ids(
        request.get("held_out_case_ids"), "held-out case IDs", allow_empty=True
    )
    if (
        not set(supervised_ids) <= set(fit_ids)
        or set(fit_ids) & set(held_out_ids)
        or set(supervised_ids) & set(held_out_ids)
        or (
            request["training_mode"] == "oracle_full"
            and supervised_ids != fit_ids
        )
    ):
        raise NeuralHandshakeValidationError("neural v3 membership closure drift")
    expected = _positive_int(
        request.get("expected_candidates_per_case"),
        "expected_candidates_per_case",
    )
    records = request.get("pretraining_records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise NeuralHandshakeValidationError("v3 pretraining records must be a sequence")
    record_keys: set[tuple[str, str]] = set()
    counts = {case_id: 0 for case_id in fit_ids}
    candidates_by_case = {case_id: [] for case_id in fit_ids}
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "case_id",
            "candidate_id",
            "state",
            "mask",
        }:
            raise NeuralHandshakeValidationError("neural v3 pretraining record drift")
        case_id = str(raw.get("case_id", "")).strip()
        candidate_id = str(raw.get("candidate_id", "")).strip()
        key = (case_id, candidate_id)
        if case_id not in counts or not candidate_id or key in record_keys:
            raise NeuralHandshakeValidationError("neural v3 pretraining ownership drift")
        record_keys.add(key)
        counts[case_id] += 1
        candidates_by_case[case_id].append(candidate_id)
    if len(records) != len(fit_ids) * expected or any(
        count != expected for count in counts.values()
    ):
        raise NeuralHandshakeValidationError("neural v3 pretraining coverage drift")
    candidate_set = set(candidates_by_case[fit_ids[0]])
    if len(candidate_set) != expected or any(
        set(values) != candidate_set for values in candidates_by_case.values()
    ):
        raise NeuralHandshakeValidationError("neural v3 dataset candidate set drift")
    sources = request.get("source_cases")
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise NeuralHandshakeValidationError("neural v3 sources must be a sequence")
    normalized_sources = []
    for raw in sources:
        if not isinstance(raw, Mapping) or set(raw) != {
            "source_case_id",
            "source_root_service_id",
            "queried_label",
            "target_plans",
        }:
            raise NeuralHandshakeValidationError("neural v3 source closure drift")
        source = dict(raw)
        source_id = str(source.get("source_case_id", "")).strip()
        root = str(source.get("source_root_service_id", "")).strip()
        label = source.get("queried_label")
        plans = source.get("target_plans")
        expected_label_source = (
            "queried_budget"
            if request["training_mode"] == "query_only"
            else "oracle_full"
        )
        expected_cost = 1 if request["training_mode"] == "query_only" else 0
        if (
            source_id not in set(supervised_ids)
            or not root
            or (source_id, root) not in record_keys
            or not isinstance(label, Mapping)
            or set(label) != {"root_cause", "fault_type", "label_source", "budget_cost"}
            or label.get("label_source") != expected_label_source
            or label.get("budget_cost") != expected_cost
            or not isinstance(plans, Mapping)
            or set(plans) != {"proxy_mode_cvae_compatible"}
            or isinstance(plans["proxy_mode_cvae_compatible"], (str, bytes))
            or not isinstance(plans["proxy_mode_cvae_compatible"], Sequence)
            or not plans["proxy_mode_cvae_compatible"]
        ):
            raise NeuralHandshakeValidationError("neural v3 source ownership drift")
        for target in plans["proxy_mode_cvae_compatible"]:
            target_id = str(dict(target).get("target_service_id", "")).strip()
            if not target_id or (source_id, target_id) not in record_keys:
                raise NeuralHandshakeValidationError(
                    "neural v3 target is outside the unlabeled pool"
                )
        target_ids = [
            str(dict(target).get("target_service_id", "")).strip()
            for target in plans["proxy_mode_cvae_compatible"]
        ]
        if len(target_ids) != len(set(target_ids)) or set(target_ids) != candidate_set:
            raise NeuralHandshakeValidationError(
                "neural v3 requires one target decision per dataset candidate"
            )
        normalized_sources.append(source)
    if [str(row["source_case_id"]) for row in normalized_sources] != supervised_ids:
        raise NeuralHandshakeValidationError("neural v3 source order drift")
    partition = request.get("proxy_mode_partition")
    try:
        validate_hdbscan_proxy_partition(
            partition,
            expected_dataset_id=str(request["dataset_id"]),
            expected_fit_case_ids=fit_ids,
            expected_queried_case_ids=supervised_ids,
            expected_held_out_case_ids=held_out_ids,
            expected_representation_matrix_sha256=str(
                dict(partition).get("representation_matrix_sha256", "")
            ),
            expected_geometry_sha256=str(dict(partition).get("geometry_sha256", "")),
            expected_active_partition_sha256=str(
                dict(partition).get("active_partition_sha256", "")
            ),
        )
    except Exception as exc:
        raise NeuralHandshakeValidationError(
            f"neural v3 HDBSCAN proxy binding drift: {exc}"
        ) from exc
    if not isinstance(request.get("profile"), Mapping) or request["profile"].get(
        "profile_id"
    ) != "balanced":
        raise NeuralHandshakeValidationError("neural v3 balanced profile drift")
    checkpoint_output = request.get("checkpoint_output_path")
    if checkpoint_output is not None and (
        not isinstance(checkpoint_output, str)
        or not checkpoint_output.strip()
        or not Path(checkpoint_output).is_absolute()
    ):
        raise NeuralHandshakeValidationError(
            "checkpoint_output_path must be an absolute path"
        )
    _positive_int(request.get("optimizer_steps"), "optimizer_steps")
    _positive_int(request.get("samples_per_target"), "samples_per_target")
    return request


def validate_neural_handshake_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuralHandshakeValidationError("neural request must be a mapping")
    if value.get("schema_version") == "service-continuous-neural-handshake-request-v1":
        return _request_v1(value)
    if value.get("schema_version") == "service-continuous-neural-handshake-request-v2":
        return _request_v2(value)
    if value.get("schema_version") == "service-continuous-neural-handshake-request-v3":
        return _request_v3(value)
    raise NeuralHandshakeValidationError("unknown neural request schema")


def _record_tensors(
    request: Mapping[str, Any], device: torch.device
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[tuple[str, str], int],
    dict[str, Any],
]:
    records = request["pretraining_records"]
    raw_values = {factor: [] for factor in _FACTORS}
    raw_masks = {factor: [] for factor in _FACTORS}
    row_index: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise NeuralHandshakeValidationError("neural record must be a mapping")
        case_id = str(raw.get("case_id", ""))
        candidate_id = str(raw.get("candidate_id", ""))
        key = (case_id, candidate_id)
        state = raw.get("state")
        mask = raw.get("mask")
        if (
            not all(key)
            or key in row_index
            or not isinstance(state, Mapping)
            or set(state) != set(_FACTORS)
            or not isinstance(mask, Mapping)
            or set(mask) != set(_FACTORS)
        ):
            raise NeuralHandshakeValidationError("neural record ownership drift")
        row_index[key] = index
        for factor in _FACTORS:
            values = [float(item) for item in state[factor]]
            masks = [float(item) for item in mask[factor]]
            if (
                not values
                or len(values) != len(masks)
                or not all(math.isfinite(item) for item in values)
                or any(item not in (0.0, 1.0) for item in masks)
            ):
                raise NeuralHandshakeValidationError("neural state/mask layout drift")
            raw_values[factor].append(values)
            raw_masks[factor].append(masks)
    normalized: dict[str, torch.Tensor] = {}
    mask_tensors: dict[str, torch.Tensor] = {}
    normalization: dict[str, Any] = {}
    for factor in _FACTORS:
        values = torch.tensor(raw_values[factor], dtype=torch.float32, device=device)
        masks = torch.tensor(raw_masks[factor], dtype=torch.float32, device=device)
        if values.ndim != 2 or masks.shape != values.shape:
            raise NeuralHandshakeValidationError("neural tensor rank drift")
        denominator = masks.sum(dim=0).clamp_min(1.0)
        mean = (values * masks).sum(dim=0) / denominator
        variance = (((values - mean) * masks) ** 2).sum(dim=0) / denominator
        scale = variance.sqrt().clamp_min(1e-6)
        normalized[factor] = ((values - mean) / scale) * masks
        mask_tensors[factor] = masks
        normalization[factor] = {
            "mean": mean.detach().cpu().tolist(),
            "scale": scale.detach().cpu().tolist(),
            "observed_count": denominator.detach().cpu().tolist(),
        }
    return normalized, mask_tensors, row_index, {
        **normalization,
        "normalization_sha256": _semantic_hash(normalization),
    }


def _state_dict_hash(model: FactorizedConditionalVAE) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise NeuralHandshakeValidationError("neural output contains non-finite value")
    return value


def execute_neural_handshake(request: Mapping[str, Any]) -> dict[str, Any]:
    prepared = validate_neural_handshake_request(request)
    request_schema = str(prepared["schema_version"])
    is_v2 = request_schema == "service-continuous-neural-handshake-request-v2"
    is_v3 = request_schema == "service-continuous-neural-handshake-request-v3"
    device = torch.device(str(prepared["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NeuralHandshakeValidationError("requested CUDA device is unavailable")
    normalized, masks, row_index, normalization = _record_tensors(prepared, device)
    profile = dict(prepared["profile"])
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = FactorizedConditionalVAE(
        mechanism_dim=normalized["mechanism"].shape[1],
        propagation_dim=normalized["propagation"].shape[1],
        context_dim=normalized["context"].shape[1],
        target_context_dim=normalized["context"].shape[1],
        hidden_width=int(profile["hidden_width"]),
        mechanism_latent_width=int(profile["mechanism_latent_width"]),
        propagation_latent_width=int(profile["propagation_latent_width"]),
        context_latent_width=int(profile["context_latent_width"]),
    ).to(device)
    initial_checkpoint_sha256 = _state_dict_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    batch = {
        "mechanism": normalized["mechanism"],
        "mechanism_mask": masks["mechanism"],
        "propagation": normalized["propagation"],
        "propagation_mask": masks["propagation"],
        "context": normalized["context"],
        "context_mask": masks["context"],
        "target_context": normalized["context"],
        "target_context_mask": masks["context"],
    }
    weights = {
        "masked_reconstruction": float(profile["masked_reconstruction_weight"]),
        "kl": float(profile["kl_weight"]),
        "target_context": float(profile["target_context_weight"]),
        "cycle_consistency": float(profile["cycle_consistency_weight"]),
    }
    model.train()
    initial = compute_unlabeled_objectives(
        model=model, batch=batch, weights=weights, sample_seed=42
    )
    gradient_norms = []
    for _ in range(int(prepared["optimizer_steps"])):
        optimizer.zero_grad(set_to_none=True)
        objectives = compute_unlabeled_objectives(
            model=model, batch=batch, weights=weights, sample_seed=42
        )
        objectives["total"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(profile["gradient_clip_norm"])
        )
        gradient_norms.append(float(gradient.detach().cpu().item()))
        optimizer.step()
    model.eval()
    final = compute_unlabeled_objectives(
        model=model, batch=batch, weights=weights, sample_seed=42
    )
    checkpoint_sha256 = _state_dict_hash(model)
    if checkpoint_sha256 == initial_checkpoint_sha256:
        raise NeuralHandshakeValidationError("optimizer did not change the checkpoint")

    if request_schema == "service-continuous-neural-handshake-request-v1":
        source_specs = [
            {
                "source_case_id": str(prepared["source_case_id"]),
                "source_root_service_id": str(prepared["source_root_service_id"]),
                "queried_label": dict(prepared["queried_label"]),
                "target_plans": dict(prepared["target_plans"]),
            }
        ]
        result_schema = "service-continuous-neural-handshake-result-v1"
    else:
        source_specs = [dict(row) for row in prepared["source_cases"]]
        result_schema = (
            "service-continuous-neural-handshake-result-v3"
            if is_v3
            else "service-continuous-neural-handshake-result-v2"
        )

    decoded_hashes = []
    source_hashes = []
    target_deltas = []
    posterior_logvars = []
    with torch.no_grad():
        for source_spec in source_specs:
            source_id = str(source_spec["source_case_id"])
            source_root = str(source_spec["source_root_service_id"])
            source_index = row_index[(source_id, source_root)]
            source_inputs = {
                factor: normalized[factor][source_index : source_index + 1]
                for factor in _FACTORS
            }
            source_inputs.update(
                {
                    f"{factor}_mask": masks[factor][source_index : source_index + 1]
                    for factor in _FACTORS
                }
            )
            source_hash = _semantic_hash(
                {factor: _tensor_hash(source_inputs[factor]) for factor in _FACTORS}
            )
            decoded_contexts = []
            activity_arm = (
                "proxy_mode_cvae_compatible" if is_v3 else "cvae_arbitrary"
            )
            for target in source_spec["target_plans"][activity_arm]:
                index = row_index[(source_id, str(target["target_service_id"]))]
                output = model(
                    **source_inputs,
                    target_context=normalized["context"][index : index + 1],
                    target_context_mask=masks["context"][index : index + 1],
                    sample=False,
                )
                decoded = output["reconstruction"]
                decoded_hashes.append(
                    _semantic_hash(
                        {factor: _tensor_hash(decoded[factor]) for factor in _FACTORS}
                    )
                )
                source_hashes.append(source_hash)
                decoded_contexts.append(decoded["context"])
                if not posterior_logvars:
                    posterior_logvars = [
                        float(item)
                        for factor in _FACTORS
                        for item in output["posterior"][factor]["logvar"]
                        .detach()
                        .cpu()
                        .flatten()
                        .tolist()
                    ]
            target_deltas.extend(
                float(torch.linalg.vector_norm(value - decoded_contexts[0]).cpu().item())
                for value in decoded_contexts[1:]
            )
    if not target_deltas:
        target_deltas = [0.0]
    activity = build_cvae_activity_audit(
        profile_id=str(profile["profile_id"]),
        gradient_norms=gradient_norms,
        reconstruction_history=[
            float(initial["components"]["masked_reconstruction"].detach().cpu()),
            float(final["components"]["masked_reconstruction"].detach().cpu()),
        ],
        kl_history=[
            float(initial["components"]["kl"].detach().cpu()),
            float(final["components"]["kl"].detach().cpu()),
        ],
        posterior_logvars=posterior_logvars,
        source_state_sha256s=source_hashes,
        generated_state_sha256s=decoded_hashes,
        target_context_deltas=target_deltas,
        mechanism_latent_distances=[0.0] * len(decoded_hashes),
        sampling_radius=float(profile["sampling_radius"]),
    )
    if activity["status"] != "mechanistically_active":
        raise NeuralHandshakeValidationError(
            "CVAE activity rejection: " + ",".join(activity["rejection_reasons"])
        )

    generation_profile = {
        "profile_id": str(profile["profile_id"]),
        "sampling_radius": float(profile["sampling_radius"]),
        "samples_per_target": int(prepared["samples_per_target"]),
    }
    generation_profile["profile_sha256"] = _semantic_hash(generation_profile)
    result_arms = (
        _ARMS
        if is_v2
        else (("proxy_mode_cvae_compatible",) if is_v3 else _CASE_LOCAL_ARMS)
    )
    arm_rows = {arm_id: [] for arm_id in result_arms}
    proxy_sources = []
    proxy_targets: dict[str, list[dict[str, Any]]] = {}
    for source_offset, source_spec in enumerate(source_specs):
        source_id = str(source_spec["source_case_id"])
        source_root = str(source_spec["source_root_service_id"])
        source_index = row_index[(source_id, source_root)]
        source_inputs = {
            factor: normalized[factor][source_index : source_index + 1]
            for factor in _FACTORS
        }
        source_inputs.update(
            {
                f"{factor}_mask": masks[factor][source_index : source_index + 1]
                for factor in _FACTORS
            }
        )
        source_case = {
            "case_id": source_id,
            "queried_label": dict(source_spec["queried_label"]),
            "observation_case_ids": list(prepared["fit_case_ids"]),
            "label_case_ids": [source_id],
            **source_inputs,
        }
        if is_v2 or is_v3:
            proxy_sources.append(source_case)
        for arm_index, arm_id in enumerate(() if is_v3 else _CASE_LOCAL_ARMS):
            target_contexts = []
            for target in source_spec["target_plans"][arm_id]:
                index = row_index[(source_id, str(target["target_service_id"]))]
                target_contexts.append(
                    {
                        **dict(target),
                        "target_context": normalized["context"][index : index + 1],
                        "target_context_mask": masks["context"][index : index + 1],
                    }
                )
            bundle = generate_cvae_service_continuity(
                model=model,
                arm_id=arm_id,
                source_case=source_case,
                target_contexts=target_contexts,
                profile=generation_profile,
                sample_seed=(
                    42 + source_offset * 1_000_003 + arm_index * 100_003
                ),
                checkpoint_sha256=checkpoint_sha256,
                activity_evidence=activity,
            )
            arm_rows[arm_id].extend(_jsonable(bundle["synthetic_rows"]))
        if is_v2 or is_v3:
            proxy_target_contexts = []
            for target in source_spec["target_plans"]["proxy_mode_cvae_compatible"]:
                index = row_index[(source_id, str(target["target_service_id"]))]
                proxy_target_contexts.append(
                    {
                        **dict(target),
                        "target_context": normalized["context"][index : index + 1],
                        "target_context_mask": masks["context"][index : index + 1],
                    }
                )
            proxy_targets[source_id] = proxy_target_contexts
    proxy_bundle = None
    if is_v2 or is_v3:
        proxy_bundle = generate_proxy_mode_cvae_service_continuity(
            model=model,
            source_cases=proxy_sources,
            target_contexts_by_source=proxy_targets,
            proxy_mode_partition=prepared["proxy_mode_partition"],
            profile=generation_profile,
            sample_seed=42 + len(_CASE_LOCAL_ARMS) * 100_003,
            checkpoint_sha256=checkpoint_sha256,
            activity_evidence=activity,
        )
        arm_rows["proxy_mode_cvae_compatible"].extend(
            _jsonable(proxy_bundle["synthetic_rows"])
        )
    activity_output = {
        **activity,
        "optimizer_step_count": int(prepared["optimizer_steps"]),
        "initial_masked_reconstruction_loss": float(
            initial["components"]["masked_reconstruction"].detach().cpu().item()
        ),
        "final_masked_reconstruction_loss": float(
            final["components"]["masked_reconstruction"].detach().cpu().item()
        ),
        "initial_kl_loss": float(
            initial["components"]["kl"].detach().cpu().item()
        ),
        "final_kl_loss": float(
            final["components"]["kl"].detach().cpu().item()
        ),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "normalization_sha256": normalization["normalization_sha256"],
        "pretraining_case_count": len(prepared["fit_case_ids"]),
        "pretraining_candidate_row_count": len(prepared["pretraining_records"]),
        "source_case_count": len(source_specs),
    }
    if proxy_bundle is not None:
        proxy_partition = dict(prepared["proxy_mode_partition"])
        activity_output["proxy_mode_activity"] = {
            "coverage_sampling": "deterministic_farthest_point",
            "partition_sha256": str(
                proxy_partition.get(
                    "proxy_partition_sha256",
                    proxy_partition.get("partition_sha256", ""),
                )
            ),
            "active_partition_sha256": proxy_partition.get(
                "active_partition_sha256"
            ),
            "mode_count": len(proxy_bundle["mode_support_audits"]),
            "shared_mode_count": sum(
                int(bool(row["shared_support"]))
                for row in proxy_bundle["mode_support_audits"].values()
            ),
            "rejected_target_count": int(proxy_bundle["rejected_target_count"]),
            "rejected_targets": [
                dict(row) for row in proxy_bundle["rejected_targets"]
            ],
            "label_access_audit": dict(proxy_bundle["label_access_audit"]),
            "synthetic_row_count": len(proxy_bundle["synthetic_rows"]),
            "mode_support_audits": proxy_bundle["mode_support_audits"],
        }
    checkpoint_artifact = None
    checkpoint_output = prepared.get("checkpoint_output_path")
    if checkpoint_output is not None:
        checkpoint_path = Path(str(checkpoint_output)).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "schema_version": "service-continuous-cvae-checkpoint-v1",
            "request_sha256": str(prepared["request_sha256"]),
            "state_dict_sha256": checkpoint_sha256,
            "profile": dict(profile),
            "normalization": normalization,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
        }
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(checkpoint_payload, temporary)
        temporary.replace(checkpoint_path)
        checkpoint_artifact = {
            "path": str(checkpoint_path),
            "state_dict_sha256": checkpoint_sha256,
            "file_sha256": _file_hash(checkpoint_path),
        }
    identity = {
        "schema_version": result_schema,
        "status": "completed",
        "request_sha256": prepared["request_sha256"],
        "activity_audit": activity_output,
        "arms": arm_rows,
    }
    if checkpoint_artifact is not None:
        identity["checkpoint_artifact"] = checkpoint_artifact
    return {**identity, "result_sha256": _semantic_hash(identity)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NeuralHandshakeValidationError("JSON root must be an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_neural_handshake(_read_json(args.input.resolve()))
    _write_json_atomic(args.output.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NeuralHandshakeValidationError",
    "execute_neural_handshake",
    "validate_neural_handshake_request",
]
