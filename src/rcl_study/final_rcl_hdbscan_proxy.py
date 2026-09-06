from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence


_DATASET_PARAMETERS = {
    "rcabench": {"min_cluster_size": 5, "min_samples": 5},
    "aiops2022_pre": {"min_cluster_size": 6, "min_samples": 2},
}


class HDBSCANProxyContractError(ValueError):
    """Raised when a final-method proxy partition is not HDBSCAN-owned."""


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


def _ids(value: Sequence[Any], context: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HDBSCANProxyContractError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if (
        (not allow_empty and not result)
        or "" in result
        or len(result) != len(set(result))
    ):
        raise HDBSCANProxyContractError(f"{context} must be unique and nonempty")
    return result


def _sha(value: Any, context: str) -> str:
    result = str(value)
    if len(result) != 64 or result.lower() != result or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise HDBSCANProxyContractError(f"{context} must be a lowercase SHA-256")
    return result


def _mode_id(prefix: str, identity: str) -> str:
    return prefix + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build_hdbscan_proxy_partition(
    *,
    dataset_id: str,
    fit_case_ids: Sequence[Any],
    queried_case_ids: Sequence[Any],
    held_out_case_ids: Sequence[Any],
    raw_labels: Sequence[Any],
    membership_strengths: Sequence[Any],
    representation_matrix_sha256: str,
    geometry_sha256: str,
    active_partition_sha256: str,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> dict[str, Any]:
    """Adapt a verified full-pool native HDBSCAN geometry for CVAE proxy modes."""

    dataset = str(dataset_id)
    if dataset not in _DATASET_PARAMETERS:
        raise HDBSCANProxyContractError("unsupported HDBSCAN proxy dataset")
    expected = _DATASET_PARAMETERS[dataset]
    if (
        isinstance(min_cluster_size, bool)
        or isinstance(min_samples, bool)
        or int(min_cluster_size) != expected["min_cluster_size"]
        or int(min_samples) != expected["min_samples"]
        or str(cluster_selection_method).lower() != "leaf"
    ):
        raise HDBSCANProxyContractError("frozen HDBSCAN parameters drifted")
    fit_ids = _ids(fit_case_ids, "fit case IDs")
    queried_ids = _ids(queried_case_ids, "queried case IDs", allow_empty=True)
    held_out_ids = _ids(held_out_case_ids, "held-out case IDs", allow_empty=True)
    if not set(queried_ids) <= set(fit_ids):
        raise HDBSCANProxyContractError("queried cases must belong to the fit pool")
    if set(fit_ids) & set(held_out_ids):
        raise HDBSCANProxyContractError("held-out cases are forbidden in HDBSCAN fitting")
    labels = list(raw_labels)
    strengths = list(membership_strengths)
    if len(labels) != len(fit_ids) or len(strengths) != len(fit_ids):
        raise HDBSCANProxyContractError("HDBSCAN geometry population drifted")
    normalized_labels: list[int] = []
    normalized_strengths: list[float] = []
    for label, strength in zip(labels, strengths):
        if isinstance(label, bool) or not isinstance(label, int):
            raise HDBSCANProxyContractError("HDBSCAN labels must be integers")
        try:
            probability = float(strength)
        except (TypeError, ValueError) as exc:
            raise HDBSCANProxyContractError("HDBSCAN membership must be numeric") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise HDBSCANProxyContractError("HDBSCAN membership must lie in [0,1]")
        normalized_labels.append(int(label))
        normalized_strengths.append(probability)

    members_by_label: dict[int, list[str]] = defaultdict(list)
    for case_id, label in zip(fit_ids, normalized_labels):
        if label >= 0:
            members_by_label[label].append(case_id)
    mode_by_label = {
        label: _mode_id("mode:", "\0".join(sorted(members)))
        for label, members in members_by_label.items()
    }
    case_mode_ids = {
        case_id: (
            mode_by_label[label]
            if label >= 0
            else _mode_id("noise:", case_id)
        )
        for case_id, label in zip(fit_ids, normalized_labels)
    }
    queried_mode_members: dict[str, list[str]] = defaultdict(list)
    for case_id in queried_ids:
        queried_mode_members[case_mode_ids[case_id]].append(case_id)
    records = [
        {
            "case_id": case_id,
            "raw_label": label,
            "membership_strength": strength,
            "is_noise": label < 0,
            "proxy_mode_id": case_mode_ids[case_id],
        }
        for case_id, label, strength in zip(
            fit_ids, normalized_labels, normalized_strengths
        )
    ]
    identity = {
        "schema_version": "final-rcl-hdbscan-proxy-partition-v1",
        "algorithm": "sklearn.cluster.HDBSCAN",
        "clusterer_id": "hdbscan",
        "dataset_id": dataset,
        "parameters": {
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "cluster_selection_method": "leaf",
            "allow_single_cluster": False,
        },
        "fit_case_ids": fit_ids,
        "queried_case_ids": queried_ids,
        "held_out_case_ids_sha256": _semantic_hash(held_out_ids),
        "held_out_overlap_count": 0,
        "label_fields_consumed": [],
        "representation_matrix_sha256": _sha(
            representation_matrix_sha256, "representation matrix"
        ),
        "geometry_sha256": _sha(geometry_sha256, "geometry"),
        "active_partition_sha256": _sha(
            active_partition_sha256, "active partition"
        ),
        "case_records": records,
        "case_mode_ids": case_mode_ids,
        "queried_mode_members": dict(sorted(queried_mode_members.items())),
        "noise_case_count": sum(label < 0 for label in normalized_labels),
        "non_noise_mode_count": len(mode_by_label),
    }
    return {**identity, "proxy_partition_sha256": _semantic_hash(identity)}


def _geometry_value(geometry: Any, name: str) -> Any:
    if hasattr(geometry, name):
        return getattr(geometry, name)
    configuration = getattr(geometry, "configuration", None)
    if configuration is not None and hasattr(configuration, name):
        return getattr(configuration, name)
    raise HDBSCANProxyContractError(f"HDBSCAN geometry lacks {name}")


def build_hdbscan_proxy_partition_from_geometry(
    *,
    geometry: Any,
    queried_case_ids: Sequence[Any],
    held_out_case_ids: Sequence[Any],
    active_partition_sha256: str,
) -> dict[str, Any]:
    clusterer = str(getattr(geometry, "clusterer_id", "hdbscan")).lower()
    backend = str(getattr(geometry, "backend", "sklearn.cluster.HDBSCAN"))
    if clusterer != "hdbscan" or "hdbscan" not in backend.lower():
        raise HDBSCANProxyContractError("geometry is not owned by native HDBSCAN")
    return build_hdbscan_proxy_partition(
        dataset_id=_geometry_value(geometry, "dataset_id"),
        fit_case_ids=_geometry_value(geometry, "case_ids"),
        queried_case_ids=queried_case_ids,
        held_out_case_ids=held_out_case_ids,
        raw_labels=_geometry_value(geometry, "labels"),
        membership_strengths=_geometry_value(geometry, "membership_strengths"),
        representation_matrix_sha256=_geometry_value(
            geometry, "representation_matrix_sha256"
        ),
        geometry_sha256=_geometry_value(geometry, "geometry_sha256"),
        active_partition_sha256=active_partition_sha256,
        min_cluster_size=_geometry_value(geometry, "min_cluster_size"),
        min_samples=_geometry_value(geometry, "min_samples"),
        cluster_selection_method=_geometry_value(
            geometry, "cluster_selection_method"
        ),
    )


def validate_hdbscan_proxy_partition(
    partition: Mapping[str, Any],
    *,
    expected_dataset_id: str,
    expected_fit_case_ids: Sequence[Any],
    expected_queried_case_ids: Sequence[Any],
    expected_held_out_case_ids: Sequence[Any],
    expected_representation_matrix_sha256: str,
    expected_geometry_sha256: str,
    expected_active_partition_sha256: str,
) -> dict[str, Any]:
    if not isinstance(partition, Mapping):
        raise HDBSCANProxyContractError("proxy partition must be a mapping")
    value = deepcopy(dict(partition))
    if (
        value.get("algorithm") != "sklearn.cluster.HDBSCAN"
        or value.get("clusterer_id") != "hdbscan"
    ):
        raise HDBSCANProxyContractError("proxy partition must be HDBSCAN-owned")
    supplied = value.pop("proxy_partition_sha256", None)
    if supplied != _semantic_hash(value):
        raise HDBSCANProxyContractError("HDBSCAN proxy partition hash mismatch")
    fit_ids = _ids(expected_fit_case_ids, "expected fit case IDs")
    queried_ids = _ids(
        expected_queried_case_ids, "expected queried case IDs", allow_empty=True
    )
    held_out_ids = _ids(
        expected_held_out_case_ids, "expected held-out case IDs", allow_empty=True
    )
    if (
        value.get("schema_version")
        != "final-rcl-hdbscan-proxy-partition-v1"
        or value.get("dataset_id") != str(expected_dataset_id)
        or value.get("fit_case_ids") != fit_ids
        or value.get("queried_case_ids") != queried_ids
        or value.get("held_out_case_ids_sha256") != _semantic_hash(held_out_ids)
        or value.get("held_out_overlap_count") != 0
        or value.get("label_fields_consumed") != []
        or value.get("representation_matrix_sha256")
        != _sha(expected_representation_matrix_sha256, "expected representation")
        or value.get("geometry_sha256")
        != _sha(expected_geometry_sha256, "expected geometry")
        or value.get("active_partition_sha256")
        != _sha(expected_active_partition_sha256, "expected active partition")
    ):
        raise HDBSCANProxyContractError("HDBSCAN proxy partition authority drifted")
    records = list(value.get("case_records", ()))
    if [dict(row).get("case_id") for row in records] != fit_ids:
        raise HDBSCANProxyContractError("HDBSCAN proxy case coverage drifted")
    modes = dict(value.get("case_mode_ids", {}))
    if set(modes) != set(fit_ids):
        raise HDBSCANProxyContractError("HDBSCAN proxy mode coverage drifted")
    noise_ids = [
        str(row["case_id"]) for row in records if bool(row.get("is_noise"))
    ]
    noise_modes = [modes[case_id] for case_id in noise_ids]
    if (
        len(noise_modes) != len(set(noise_modes))
        or any(not mode.startswith("noise:") for mode in noise_modes)
        or int(value.get("noise_case_count", -1)) != len(noise_ids)
    ):
        raise HDBSCANProxyContractError("HDBSCAN noise modes were merged or drifted")
    non_noise_modes = {
        modes[str(row["case_id"])]
        for row in records
        if not bool(row.get("is_noise"))
    }
    if int(value.get("non_noise_mode_count", -1)) != len(non_noise_modes):
        raise HDBSCANProxyContractError("HDBSCAN non-noise mode count drifted")
    return {
        "valid": True,
        "fit_case_count": len(fit_ids),
        "queried_case_count": len(queried_ids),
        "noise_case_count": len(noise_ids),
        "non_noise_mode_count": len(non_noise_modes),
        "proxy_partition_sha256": supplied,
    }


__all__ = [
    "HDBSCANProxyContractError",
    "build_hdbscan_proxy_partition",
    "build_hdbscan_proxy_partition_from_geometry",
    "validate_hdbscan_proxy_partition",
]
