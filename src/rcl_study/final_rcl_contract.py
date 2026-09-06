from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


DATASETS = ("rcabench", "aiops2022_pre")
ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)
EXPECTED_QUERY_PLAN_SHA256 = {
    "rcabench": "712ac6d9ac60b8f717db074f411fbf645a6eff925b462dca35b27c7dba109b6a",
    "aiops2022_pre": "5a69678c9ac1477adfb9f3f83885ba78b7d1fd0e528cb2f3832f91e14009f0f9",
}
EXPECTED_HDBSCAN = {
    "rcabench": {"min_cluster_size": 5, "min_samples": 5},
    "aiops2022_pre": {"min_cluster_size": 6, "min_samples": 2},
}
EXPECTED_CVAE_GENERATION_SHA256 = (
    "f9cd73c6fef481201fba24e5558da535b5f77c2816009c0e24710a3fa94adbc2"
)
EXPECTED_OSER_SHA256 = (
    "ef4f2c0229e8549cc2b85d33bf6478d10870606929cf156a4e99a60d70eb581a"
)


class FinalRCLContractError(ValueError):
    """Raised when the final HDBSCAN/CVAE/OSER contract drifts."""


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_hash(value: Any, context: str) -> str:
    result = str(value)
    if len(result) != 64 or result.lower() != result or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise FinalRCLContractError(f"{context} must be a lowercase SHA-256")
    return result


def _validated_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(value))
    if config.get("schema_version") != "final-rcl-method-config-v1":
        raise FinalRCLContractError("unexpected final method config schema")
    if config.get("seed") != 42:
        raise FinalRCLContractError("final method seed must be 42")
    split = dict(config.get("split", {}))
    if (
        split.get("fit_population") != "outer_train_only"
        or split.get("evaluation_population") != "outer_test_only"
        or not math.isclose(float(split.get("outer_train_ratio", -1)), 0.7)
        or not math.isclose(float(split.get("outer_test_ratio", -1)), 0.3)
    ):
        raise FinalRCLContractError("final method split contract drifted")

    active = dict(config.get("active_learning", {}))
    if (
        active.get("clusterer_id") != "hdbscan"
        or active.get("representation_id") != "global_pca_dim32"
        or active.get("cluster_selection_method") != "leaf"
        or active.get("selector_id") != "center"
        or active.get("budget") != 30
        or active.get("allow_single_cluster") is not False
    ):
        raise FinalRCLContractError("final method requires the frozen HDBSCAN authority")
    dataset_configs = dict(active.get("datasets", {}))
    if tuple(dataset_configs) != DATASETS:
        raise FinalRCLContractError("final method dataset order or membership drifted")
    for dataset_id in DATASETS:
        dataset = dict(dataset_configs[dataset_id])
        expected = EXPECTED_HDBSCAN[dataset_id]
        if any(dataset.get(key) != expected[key] for key in expected):
            raise FinalRCLContractError(f"{dataset_id} HDBSCAN parameters drifted")
        if dataset.get("query_plan_sha256") != EXPECTED_QUERY_PLAN_SHA256[dataset_id]:
            raise FinalRCLContractError(f"{dataset_id} query plan authority drifted")
        for field in (
            "representation_matrix_sha256",
            "geometry_sha256",
            "partition_sha256",
        ):
            _require_hash(dataset.get(field), f"{dataset_id}.{field}")

    cvae = dict(config.get("cvae", {}))
    if (
        cvae.get("arm_id") != "proxy_mode_cvae_compatible"
        or cvae.get("profile_id") != "balanced"
        or cvae.get("generation_profile_sha256")
        != EXPECTED_CVAE_GENERATION_SHA256
        or cvae.get("target_scope")
        != "all_hard_compatible_dataset_candidates"
        or float(cvae.get("synthetic_mass_per_real_parent", -1)) != 1.0
        or cvae.get("synthetic_query_budget_cost") != 0
    ):
        raise FinalRCLContractError("frozen balanced CVAE contract drifted")
    oser = dict(config.get("oser", {}))
    if (
        oser.get("profile_id") != "oser-p02"
        or oser.get("profile_sha256") != EXPECTED_OSER_SHA256
        or oser.get("state_width") != 32
        or float(oser.get("residual_cap", -1)) != 0.05
        or float(oser.get("gate_threshold", -1)) != 0.5
        or float(oser.get("lambda_meta", -1)) != 0.5
        or oser.get("inner_updates") != 1
    ):
        raise FinalRCLContractError("frozen oser-p02 contract drifted")
    router = dict(config.get("outer_router", {}))
    if router != {"enabled": False, "forbidden_arm_id": "residual_only"}:
        raise FinalRCLContractError("residual_only outer router must remain disabled")
    if tuple(config.get("formal_arms", ())) != ARMS:
        raise FinalRCLContractError("formal arm order or membership drifted")
    return config


def load_final_method_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalRCLContractError(f"cannot read final method config: {source}") from exc
    if not isinstance(payload, Mapping):
        raise FinalRCLContractError("final method config must be a JSON object")
    return _validated_config(payload)


def _unit(dataset_id: str, arm_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = dict(config["active_learning"]["datasets"][dataset_id])
    query_only = arm_id != "oracle_full_cvae_oser"
    complete = arm_id != "hdbscan_query_pairwise"
    return {
        "unit_id": f"{dataset_id}--{arm_id}--seed42",
        "dataset_id": dataset_id,
        "arm_id": arm_id,
        "seed": 42,
        "split": {
            "fit_population": "outer_train_only",
            "evaluation_population": "outer_test_only",
        },
        "supervision": {
            "mode": "query_only" if query_only else "oracle_full",
            "case_count": 30 if query_only else "outer_train",
        },
        "query_plan_sha256": (
            dataset["query_plan_sha256"] if query_only else None
        ),
        "augmentation": "proxy_mode_cvae_compatible" if complete else "none",
        "model": "oser_meta" if complete else "pairwise_linear",
        "cvae_profile_sha256": (
            config["cvae"]["generation_profile_sha256"] if complete else None
        ),
        "oser_profile_sha256": (
            config["oser"]["profile_sha256"] if complete else None
        ),
    }


def build_formal_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _validated_config(config)
    units = [
        _unit(dataset_id, arm_id, frozen)
        for dataset_id in DATASETS
        for arm_id in ARMS
    ]
    identity = {
        "schema_version": "final-rcl-formal-registry-v1",
        "method_id": frozen["method_id"],
        "seed": 42,
        "clusterer_id": "hdbscan",
        "outer_router_enabled": False,
        "unit_count": len(units),
        "units": units,
    }
    return {**identity, "registry_sha256": semantic_sha256(identity)}


def validate_formal_registry(
    registry: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = _validated_config(config)
    value = deepcopy(dict(registry))
    if value.get("clusterer_id") != "hdbscan":
        raise FinalRCLContractError("formal registry must use HDBSCAN")
    units = list(value.get("units", ()))
    if any(dict(unit).get("model") == "residual_only" for unit in units):
        raise FinalRCLContractError("residual_only outer router is forbidden")
    supplied = value.pop("registry_sha256", None)
    if supplied != semantic_sha256(value):
        raise FinalRCLContractError("formal registry semantic hash mismatch")
    expected = build_formal_registry(frozen)
    if registry != expected:
        raise FinalRCLContractError("formal registry differs from frozen matrix")
    return {
        "valid": True,
        "unit_count": 6,
        "registry_sha256": expected["registry_sha256"],
    }


__all__ = [
    "ARMS",
    "DATASETS",
    "FinalRCLContractError",
    "build_formal_registry",
    "load_final_method_config",
    "semantic_sha256",
    "validate_formal_registry",
]
