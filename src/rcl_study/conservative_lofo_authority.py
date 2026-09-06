from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class AuthorityValidationError(ValueError):
    """Raised when the frozen conservative-LOFO authority drifts."""


_DATASETS = ("rcabench", "aiops2022_pre")
_ARMS = ("oser_meta", "mm_dro", "cope_gate")
_EXPECTED_DBSCAN = {
    "rcabench": {
        "strategy_id": "dbscan_coverage",
        "freeze_sha256": "7afd25dedb2d8be089766a723b399afad7dee16021d81040ab38fc18088b9fc5",
        "config": {
            "boundary_fraction": 0.2,
            "eps_quantile": 0.8,
            "min_samples": 6,
            "noise_budget_cap": 2,
        },
    },
    "aiops2022_pre": {
        "strategy_id": "dbscan_coverage",
        "freeze_sha256": "f61db753f2e712aa95538786c3e7a092b65f3601ca082abb7afd67f0b6194969",
        "config": {
            "boundary_fraction": 0.2,
            "eps_quantile": 0.8,
            "min_samples": 5,
            "noise_budget_cap": 2,
        },
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityValidationError(f"cannot read authority JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuthorityValidationError(f"authority JSON must be an object: {path}")
    return value


def load_authority_registry(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    target_config = _read_json(root_path / "configs" / "conservative_lofo" / "initial_targets.json")
    profile_config = _read_json(root_path / "configs" / "conservative_lofo" / "model_profiles.json")

    dbscan: dict[str, Any] = {}
    for dataset_id in _DATASETS:
        source = _read_json(
            root_path / "configs" / "active_learning" / f"{dataset_id}.dbscan_coverage.json"
        )
        dbscan[dataset_id] = {
            "strategy_id": source.get("strategy_id"),
            "freeze_sha256": source.get("freeze_sha256"),
            "config": deepcopy(source.get("config")),
        }

    registry = {
        "schema_version": target_config.get("schema_version"),
        "authority_package": target_config.get("authority_package"),
        "datasets": tuple(target_config.get("datasets", ())),
        "split_seed": target_config.get("split_seed"),
        "active_learning_seed": target_config.get("active_learning_seed"),
        "budget": target_config.get("budget"),
        "label_return": target_config.get("label_return"),
        "targets": {
            dataset_id: tuple(target_config.get("targets", {}).get(dataset_id, ()))
            for dataset_id in _DATASETS
        },
        "aliases": deepcopy(target_config.get("aliases", {})),
        "union_exclusion": {
            dataset_id: tuple(target_config.get("union_exclusion", {}).get(dataset_id, ()))
            for dataset_id in _DATASETS
        },
        "include_low_support_targets_in_primary_macro": target_config.get(
            "include_low_support_targets_in_primary_macro"
        ),
        "full_lofo_authorized": target_config.get("full_lofo_authorized"),
        "profile_selection": deepcopy(profile_config.get("profile_selection", {})),
        "profiles": {
            arm: tuple(deepcopy(profile_config.get("arms", {}).get(arm, ()))) for arm in _ARMS
        },
        "dbscan": dbscan,
    }
    validate_authority_registry(registry)
    return registry


def _require_equal(registry: Mapping[str, Any], field: str, expected: Any) -> None:
    if registry.get(field) != expected:
        raise AuthorityValidationError(
            f"{field} drift: expected {expected!r}, observed {registry.get(field)!r}"
        )


def validate_authority_registry(registry: Mapping[str, Any]) -> None:
    _require_equal(registry, "authority_package", "dbscan_active_learning_intermediate_20260831")
    _require_equal(registry, "datasets", _DATASETS)
    _require_equal(registry, "split_seed", 42)
    _require_equal(registry, "active_learning_seed", 42)
    _require_equal(registry, "budget", 30)
    _require_equal(registry, "label_return", ["root_cause", "fault_type"])
    _require_equal(registry, "full_lofo_authorized", False)
    _require_equal(registry, "include_low_support_targets_in_primary_macro", True)

    targets = registry.get("targets")
    if not isinstance(targets, Mapping) or tuple(targets) != _DATASETS:
        raise AuthorityValidationError("targets must contain exactly rcabench and aiops2022_pre")
    if tuple(targets["rcabench"]) != (
        "NetworkDelay",
        "HTTPResponsePatchBody",
        "HTTPResponseDelay",
        "HTTPRequestAbort",
        "PodKill",
        "JVMMemoryStress",
        "NetworkBandwidth",
    ):
        raise AuthorityValidationError("targets.rcabench drift")
    if tuple(targets["aiops2022_pre"]) != (
        "node 磁盘空间消耗",
        "k8s容器读io负载",
        "k8s容器网络丢包",
    ):
        raise AuthorityValidationError("targets.aiops2022_pre drift")

    profiles = registry.get("profiles")
    if not isinstance(profiles, Mapping) or tuple(profiles) != _ARMS:
        raise AuthorityValidationError("profiles must contain exactly the three enhanced arms")
    profile_ids: list[str] = []
    for arm in _ARMS:
        rows = profiles[arm]
        if len(rows) != 4:
            raise AuthorityValidationError(f"profiles.{arm} must contain exactly four profiles")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("profile_id"), str):
                raise AuthorityValidationError(f"profiles.{arm} contains an invalid profile")
            profile_ids.append(row["profile_id"])
    if len(profile_ids) != len(set(profile_ids)):
        raise AuthorityValidationError("profile_id values must be globally unique")

    selection = registry.get("profile_selection")
    if not isinstance(selection, Mapping):
        raise AuthorityValidationError("profile_selection must be an object")
    for field in (
        "dataset_specific_overrides",
        "held_out_fold_overrides",
        "cartesian_expansion_after_calibration",
    ):
        if selection.get(field) is not False:
            raise AuthorityValidationError(f"profile_selection.{field} must be false")

    dbscan = registry.get("dbscan")
    if not isinstance(dbscan, Mapping) or tuple(dbscan) != _DATASETS:
        raise AuthorityValidationError("dbscan must contain exactly the decision datasets")
    for dataset_id, expected in _EXPECTED_DBSCAN.items():
        if dbscan.get(dataset_id) != expected:
            raise AuthorityValidationError(f"dbscan.{dataset_id} drift")


__all__ = [
    "AuthorityValidationError",
    "load_authority_registry",
    "validate_authority_registry",
]
