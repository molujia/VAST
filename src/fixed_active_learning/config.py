"""Fail-closed configuration for the fixed active-learning method."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


_FROZEN = {
    "rcabench": {"min_cluster_size": 5, "min_samples": 5, "alias": "rcabench"},
    "aiops2022_pre": {"min_cluster_size": 6, "min_samples": 2, "alias": "hd1"},
}
_SHA256_CHARS = frozenset("0123456789abcdef")


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_CHARS for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class FixedActiveLearningConfig:
    dataset_id: str
    representation_id: str
    required_modalities: tuple[str, ...]
    modalities: tuple[str, ...]
    pca_dimension: int
    clusterer_id: str
    cluster_selection_method: str
    min_cluster_size: int
    min_samples: int
    selector_id: str
    budget: int
    active_learning_seeds: tuple[int, ...]
    rcl_seed: int
    feature_alias: str
    data_root: Path
    candidate_pool_path: Path
    feature_inventory_path: Path
    reference_plan_dir: Path
    reference_representation_matrix_sha256: str
    data_root_env: str


def _resolve_package_path(config_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty package-relative path")
    root = config_path.resolve().parents[1]
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the handoff package") from exc
    return resolved


def load_fixed_config(
    path: str | Path, *, data_root_override: str | Path | None = None
) -> FixedActiveLearningConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixed configuration must be a JSON object")
    dataset_id = payload.get("dataset_id")
    if dataset_id not in _FROZEN:
        raise ValueError("fixed configuration has an unsupported dataset")
    expected = _FROZEN[dataset_id]
    frozen_fields = {
        "representation_id": "global_pca_dim32",
        "required_modalities": ["metric", "log", "trace"],
        "pca_dimension": 32,
        "clusterer_id": "hdbscan",
        "cluster_selection_method": "leaf",
        "min_cluster_size": expected["min_cluster_size"],
        "min_samples": expected["min_samples"],
        "selector_id": "center",
        "budget": 30,
        "active_learning_seeds": [41, 42, 43],
        "rcl_seed": 42,
        "feature_alias": expected["alias"],
        "candidate_population": "full_observable_outer_train_not_sampled",
        "ground_truth_used": False,
    }
    drift = [
        field for field, expected_value in frozen_fields.items()
        if payload.get(field) != expected_value
    ]
    if drift:
        raise ValueError("frozen configuration drift: " + ", ".join(sorted(drift)))
    modalities = tuple(payload.get("modalities", ()))
    if modalities != ("metric", "log", "trace", "topology", "time"):
        raise ValueError("frozen configuration drift: modalities")
    data_root_env = payload.get("data_root_env")
    if not isinstance(data_root_env, str) or not data_root_env:
        raise ValueError("data_root_env must be declared")
    raw_data_root = data_root_override or os.environ.get(data_root_env) or payload.get(
        "default_data_root"
    )
    if not raw_data_root:
        raw_data_root = config_path.resolve().parents[1] / "EXTERNAL_DATA_ROOT_REQUIRED"
    return FixedActiveLearningConfig(
        dataset_id=dataset_id,
        representation_id=payload["representation_id"],
        required_modalities=tuple(payload["required_modalities"]),
        modalities=modalities,
        pca_dimension=int(payload["pca_dimension"]),
        clusterer_id=payload["clusterer_id"],
        cluster_selection_method=payload["cluster_selection_method"],
        min_cluster_size=int(payload["min_cluster_size"]),
        min_samples=int(payload["min_samples"]),
        selector_id=payload["selector_id"],
        budget=int(payload["budget"]),
        active_learning_seeds=tuple(int(seed) for seed in payload["active_learning_seeds"]),
        rcl_seed=int(payload["rcl_seed"]),
        feature_alias=payload["feature_alias"],
        data_root=Path(raw_data_root).expanduser().resolve(),
        candidate_pool_path=_resolve_package_path(
            config_path, payload["candidate_pool_path"], "candidate_pool_path"
        ),
        feature_inventory_path=_resolve_package_path(
            config_path, payload["feature_inventory_path"], "feature_inventory_path"
        ),
        reference_plan_dir=_resolve_package_path(
            config_path, payload["reference_plan_dir"], "reference_plan_dir"
        ),
        reference_representation_matrix_sha256=_require_sha256(
            payload["reference_representation_matrix_sha256"],
            "reference_representation_matrix_sha256",
        ),
        data_root_env=data_root_env,
    )
