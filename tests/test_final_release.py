"""Publication boundaries and small deterministic behavior checks."""
import json
from pathlib import Path

import pytest

from vast import __version__, describe_method, load_default_config
from scripts.run_historical_pairwise_hdbscan_compare import _derive_metrics_without_package


def test_selected_method_and_scientific_recipe():
    assert __version__ == "1.0.0"
    assert describe_method()["method"] == "B"
    cfg = load_default_config()
    assert (cfg["seed"], cfg["posterior_samples"], cfg["augmentation_mass_per_parent"]) == (42, 8, .25)
    assert cfg["cvae"]["optimizer_steps"] == 150
    assert cfg["oser"]["training_steps"] == 30


def test_metrics_recomputed_from_first_true_root_rank():
    rows = [{"case_id": "one", "targets": ["b"], "ranking": ["a", "b", "c"]},
            {"case_id": "two", "targets": ["c", "a"], "ranking": ["c", "b", "a"]}]
    result = _derive_metrics_without_package(rows)
    assert result["mrr"] == .75
    assert result["hit_counts"] == {"hit_at_1": 1, "hit_at_3": 2, "hit_at_5": 2}
    assert len(result["metrics_sha256"]) == 64
    with pytest.raises(ValueError, match="ownership"):
        _derive_metrics_without_package(rows + rows[:1])


def test_input_identity_tracks_external_bytes(tmp_path):
    from vast.runtime import resolve_inputs
    feature = tmp_path / "features"
    feature.mkdir()
    for name in ("windows.csv", "entity_features.csv", "metadata.json"):
        (feature / name).write_text("initial")
    (tmp_path / "split.json").write_text("{}")
    spec = {"datasets": [{"dataset_id": "rcabench", "feature_dir": "features", "split_manifest": "split.json"}], "regimes": ["oracle_full"]}
    _, before = resolve_inputs(tmp_path / "runtime.json", spec)
    (feature / "windows.csv").write_text("modified")
    _, after = resolve_inputs(tmp_path / "runtime.json", spec)
    assert before != after


def test_output_protects_sources_and_external_inputs(tmp_path,monkeypatch):
    import vast.runtime
    monkeypatch.setattr(vast.runtime,"REPOSITORY_ROOT",tmp_path/"project")
    from vast.runtime import validate_output
    REPOSITORY_ROOT=tmp_path/"project"
    for output in (REPOSITORY_ROOT, REPOSITORY_ROOT / "docs" / "run", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            validate_output(output, [tmp_path / "features"])
    validate_output(tmp_path / "new-run", [tmp_path / "features"])


def test_receipt_rejects_corrupt_committed_bytes(tmp_path):
    from vast.runtime import validate_receipt
    from rcl_study.vae_snapshot_contract import file_sha256, semantic_sha256
    artifact = tmp_path / "item.json"
    artifact.write_text("{}")
    receipt = {"identity_sha256": "id", "metadata": {}, "output_sha256s": {"item.json": file_sha256(artifact)}}
    receipt["artifact_sha256"] = semantic_sha256(receipt)
    receipt["stage_root"] = str(tmp_path)
    validate_receipt(receipt)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="corrupt"):
        validate_receipt(receipt)
