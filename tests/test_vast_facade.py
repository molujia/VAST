from __future__ import annotations

from vast import __version__, describe_method, load_default_config


def test_method_descriptor_freezes_selected_components() -> None:
    descriptor = describe_method()

    assert descriptor == {
        "name": "VAST",
        "status": "interim",
        "active_learning": "hdbscan",
        "augmentation": "proxy_mode_cvae_compatible",
        "base_ranker": "weighted_pairwise_linear",
        "model_extension": "oser_meta",
        "outer_router": None,
    }


def test_load_config_reads_frozen_default() -> None:
    config = load_default_config()

    assert config["method_id"] == "hdbscan-proxy-cvae-compatible-oser-p02"
    assert config["active_learning"]["clusterer_id"] == "hdbscan"
    assert config["cvae"]["profile_id"] == "balanced"
    assert config["oser"]["profile_id"] == "oser-p02"
    assert config["outer_router"]["enabled"] is False


def test_interim_version_is_explicit() -> None:
    assert __version__ == "0.1.0.dev0"
