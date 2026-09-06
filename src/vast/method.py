"""Stable metadata access without hiding the verified research implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_METHOD_DESCRIPTOR: dict[str, object] = {
    "name": "VAST",
    "status": "interim",
    "active_learning": "hdbscan",
    "augmentation": "proxy_mode_cvae_compatible",
    "base_ranker": "weighted_pairwise_linear",
    "model_extension": "oser_meta",
    "outer_router": None,
}


def describe_method() -> dict[str, object]:
    """Describe the exact algorithmic composition represented by this snapshot."""

    return dict(_METHOD_DESCRIPTOR)


def load_default_config() -> dict[str, Any]:
    """Load the repository's frozen seed-42 method configuration."""

    repository_root = Path(__file__).resolve().parents[2]
    config_path = (
        repository_root
        / "configs"
        / "final_rcl"
        / "hdbscan_proxy_cvae_oser_seed42.json"
    )
    return json.loads(config_path.read_text(encoding="utf-8"))
