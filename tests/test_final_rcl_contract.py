from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from rcl_study.final_rcl_contract import (
    FinalRCLContractError,
    build_formal_registry,
    load_final_method_config,
    validate_formal_registry,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_rcl/hdbscan_proxy_cvae_oser_seed42.json"


def test_frozen_final_method_config_selects_only_hdbscan_balanced_and_oser_p02() -> None:
    config = load_final_method_config(CONFIG)

    assert config["seed"] == 42
    assert config["active_learning"]["clusterer_id"] == "hdbscan"
    assert config["active_learning"]["budget"] == 30
    assert config["cvae"]["profile_id"] == "balanced"
    assert config["oser"]["profile_id"] == "oser-p02"
    assert config["outer_router"] == {
        "enabled": False,
        "forbidden_arm_id": "residual_only",
    }


def test_formal_registry_contains_exactly_six_owned_units() -> None:
    config = load_final_method_config(CONFIG)
    registry = build_formal_registry(config)
    audit = validate_formal_registry(registry, config)

    assert audit["valid"] is True
    assert registry["unit_count"] == 6
    assert [unit["unit_id"] for unit in registry["units"]] == [
        "rcabench--hdbscan_query_cvae_oser--seed42",
        "rcabench--oracle_full_cvae_oser--seed42",
        "rcabench--hdbscan_query_pairwise--seed42",
        "aiops2022_pre--hdbscan_query_cvae_oser--seed42",
        "aiops2022_pre--oracle_full_cvae_oser--seed42",
        "aiops2022_pre--hdbscan_query_pairwise--seed42",
    ]
    query = registry["units"][0]
    oracle = registry["units"][1]
    control = registry["units"][2]
    assert query["supervision"] == {"mode": "query_only", "case_count": 30}
    assert query["augmentation"] == "proxy_mode_cvae_compatible"
    assert query["model"] == "oser_meta"
    assert oracle["supervision"] == {"mode": "oracle_full", "case_count": "outer_train"}
    assert control["augmentation"] == "none"
    assert control["model"] == "pairwise_linear"


def test_registry_rejects_dbscan_or_residual_only_drift() -> None:
    config = load_final_method_config(CONFIG)
    registry = build_formal_registry(config)

    attacked = deepcopy(registry)
    attacked["clusterer_id"] = "dbscan"
    with pytest.raises(FinalRCLContractError, match="HDBSCAN"):
        validate_formal_registry(attacked, config)

    attacked = deepcopy(registry)
    attacked["units"][0]["model"] = "residual_only"
    with pytest.raises(FinalRCLContractError, match="residual_only"):
        validate_formal_registry(attacked, config)
