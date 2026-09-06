from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import scripts.run_final_rcl_real_smoke as smoke_module
from scripts.run_final_rcl_real_smoke import (
    _extend_package_path,
    _feature_alias_sources,
    _fixed_geometry,
    _state_schema_from_payload,
    _transfer_feature_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_closure_hashes_the_executed_oser_objective_code() -> None:
    assert "rcl_study/conservative_lofo_oser.py" in getattr(
        smoke_module, "CLOSURE_CODE_RELATIVE_PATHS", ()
    )


def test_final_real_smoke_cli_exposes_bounded_and_dual_environment_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_final_rcl_real_smoke.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--torch-python" in completed.stdout
    assert "--oracle-train-limit" in completed.stdout
    assert "--test-case-limit" in completed.stdout
    assert "--rcabench-feature-dir" in completed.stdout
    assert "--aiops22-feature-dir" in completed.stdout
    assert "--feature-root" not in completed.stdout


def test_feature_alias_sources_bind_each_dataset_to_its_authority_bundle(
    tmp_path: Path,
) -> None:
    rcabench = tmp_path / "combined-formal" / "repaired_feature_artifacts" / "rcabench"
    aiops22 = tmp_path / "combined-features" / "window_feature_artifacts_repaired_trace_v1" / "hd1"

    sources = _feature_alias_sources(
        rcabench_feature_dir=rcabench,
        aiops22_feature_dir=aiops22,
    )

    assert sources == {
        "rcabench": rcabench.resolve(),
        "hd4": rcabench.resolve(),
        "hd1": aiops22.resolve(),
    }


def test_fixed_package_path_precedes_an_already_loaded_namespace(tmp_path: Path) -> None:
    loaded = SimpleNamespace(__path__=[str(tmp_path / "main")])
    extension = tmp_path / "authority"

    _extend_package_path(loaded, extension)
    _extend_package_path(loaded, extension)

    assert loaded.__path__ == [str(extension.resolve()), str(tmp_path / "main")]


def test_aiops22_geometry_is_projected_to_the_frozen_native_contract(
    monkeypatch,
) -> None:
    configuration = SimpleNamespace(
        min_cluster_size=6,
        min_samples=2,
        cluster_selection_method="leaf",
        metric="euclidean",
        max_cluster_size=None,
        allow_single_cluster=False,
    )
    fitted = object()
    native = object()
    candidate = object()
    tuning = ModuleType("rcl_study.combined_active_learning_tuning")
    tuning.generate_round_a_grid = lambda: (configuration,)
    structure = ModuleType("rcl_study.combined_active_learning_tuning_structure")
    structure.fit_configuration_geometry = lambda selected, candidate: fitted
    winner = ModuleType("rcl_study.combined_active_learning_winner_analysis")
    conversion = {}

    def to_native_hdbscan_result(**kwargs):
        conversion.update(kwargs)
        return native

    winner.to_native_hdbscan_result = to_native_hdbscan_result
    monkeypatch.setitem(sys.modules, tuning.__name__, tuning)
    monkeypatch.setitem(sys.modules, structure.__name__, structure)
    monkeypatch.setitem(sys.modules, winner.__name__, winner)

    result = _fixed_geometry("aiops2022_pre", candidate, configuration)

    assert result is native
    assert conversion == {
        "fitted_geometry": fitted,
        "candidate": candidate,
        "dataset_id": "aiops2022_pre",
        "active_learning_seed": 42,
    }


def test_serialized_observable_schema_round_trips_into_state_contract() -> None:
    fields = {
        "metric": ("metric_direction",),
        "log": ("log_intensity",),
        "trace": ("trace_latency",),
        "topology": ("topology_depth",),
        "time": ("relative_onset",),
        "candidate": ("candidate_reachability",),
    }
    payload = {
        "schema_version": "conservative-lofo-observable-state-schema-v1",
        "fields_by_type": fields,
        "feature_order": tuple(value for group in fields.values() for value in group),
        "modality_presence_fields": {
            "metric": "has_metric_signal",
            "log": "has_log_signal",
            "trace": "has_trace_signal",
            "topology": "has_topology_signal",
            "time": "has_time_signal",
            "candidate": "has_candidate_signal",
        },
        "clip_value": 20.0,
        "schema_sha256": "a" * 64,
    }

    schema = _state_schema_from_payload(payload)

    assert schema.fields_by_type == fields
    assert schema.feature_order == payload["feature_order"]


def test_transfer_feature_indices_are_semantic_and_dataset_width_agnostic() -> None:
    identity_names = {
        0: "entity_is_service",
        1: "entity_is_host",
        2: "topo_in_degree",
        3: "topo_out_degree",
        4: "topo_in_weight",
        5: "topo_out_weight",
        6: "topo_cross_degree",
        7: "topo_cross_weight",
        61: "topology_change_count",
        62: "has_log_signal",
        63: "has_metric_signal",
        64: "has_trace_signal",
        65: "modalities_present_count",
    }
    rcabench_names = tuple(
        identity_names.get(index, f"metric_or_embedding_{index:03d}")
        for index in range(122)
    )

    rcabench_symptom, rcabench_identity = _transfer_feature_indices(rcabench_names)

    assert rcabench_identity == tuple(range(0, 8)) + tuple(range(61, 66))
    assert rcabench_symptom == tuple(range(8, 61)) + tuple(range(66, 122))

    aiops22_names = (
        "entity_is_host",
        "has_log_signal",
        "metric_kpi_hit_00",
        "metric_abs_z_max",
        "modalities_present_count",
        "topo_in_degree",
        "trace_error_count",
        "topology_change_count",
        "entity_is_service",
    )

    aiops22_symptom, aiops22_identity = _transfer_feature_indices(aiops22_names)

    assert aiops22_symptom == (2, 3, 6)
    assert aiops22_identity == (0, 1, 4, 5, 7, 8)
    assert sorted(aiops22_symptom + aiops22_identity) == list(range(len(aiops22_names)))
