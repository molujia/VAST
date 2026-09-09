from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vast.method import load_default_config, with_historical_authority

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    return importlib.import_module("rcl_study.vae_snapshot_historical")


@pytest.fixture
def fixture():
    module = adapter()
    cfg = with_historical_authority(load_default_config())
    api = module.load_historical_api(cfg)
    sem = api["sem"]
    columns = ["metric_abs_z_max", "log_error_count", "trace_error_ratio"]
    rows = []
    for case in ("p1", "p2"):
        for i, entity in enumerate(("svc-root", "svc-z", "svc-a")):
            rows.append(dict(dataset="fixture", source_id=case, day="fixture-day", start_ts=0, end_ts=60,
                             window_id=case, entity_id=entity, entity_type="service", entity_index=i,
                             window_kind="fault", metric_abs_z_max=float(3-i), log_error_count=float((0,9,0)[i]),
                             trace_error_ratio=float(i)/10))
    tables = sem.FeatureBundleTables(dataset="fixture", windows=pd.DataFrame([
        dict(window_id=c, window_kind="fault", positive_ids_list=["svc-root"]) for c in ("p1", "p2")
    ]), entity_features=pd.DataFrame(rows), metadata={}, feature_columns=columns)
    plan = sem.QueryPlan(dataset="fixture", normal_cluster_id=-2,
        window_clusters={"p1":0, "p2":1}, queried_window_ids=["p1", "p2"],
        queried_roles={"p1":"frozen", "p2":"frozen"},
        queried_labels={"p1":["svc-root"], "p2":["svc-root"]}, pseudo_labels={}, pseudo_confidence={}, metadata={})
    model_config = api["bounded"]._model_config("fault_only")
    return module, api, tables, plan, model_config


def test_original_preparation_preserves_names_order_rows_and_ownership(fixture):
    module, api, tables, plan, config = fixture
    untouched = tables.entity_features.copy(deep=True)
    prepared = module.prepare_historical(api, tables, plan, config)
    expected = api["sem"].prepare_feature_bundle_tables_for_model(tables, tables.feature_columns, config)
    assert prepared.feature_columns == expected.feature_columns
    assert prepared.feature_columns[:3] == list(tables.feature_columns)
    pd.testing.assert_frame_equal(prepared.prepared_tables.entity_features, expected.entity_features)
    pd.testing.assert_frame_equal(tables.entity_features, untouched)
    assert set(prepared.training_frame.window_id) == set(plan.queried_window_ids)
    assert prepared.preprocessing_population == "real_outer_train_only"


def test_renamed_bridge_changes_historical_hard_negative_selection(fixture):
    module, api, tables, plan, config = fixture
    frame = module.prepare_historical(api, tables, plan, config).training_frame
    cols = tables.feature_columns
    original = api["pairwise"]._build_pairwise_training_arrays(frame, cols, negative_top_k=1)[0]
    names = {c:f"authority_feature_{i:04d}" for i,c in enumerate(cols)}
    renamed = api["pairwise"]._build_pairwise_training_arrays(frame.rename(columns=names), list(names.values()), negative_top_k=1)[0]
    # svc-z has stronger historical anomaly hints; a renamed tie picks svc-a.
    assert not np.array_equal(original, renamed)


def test_children_conserve_directional_parent_mass_and_leave_real_arrays_unchanged(fixture):
    module, api, tables, plan, config = fixture
    prepared = module.prepare_historical(api, tables, plan, config)
    children = []
    for i, size in enumerate((2, 3)):
        frame = prepared.training_frame.query("window_id == 'p1'").head(size).copy()
        frame["window_id"] = f"child-{i}"
        frame["metric_abs_z_max"] += 1000
        children.append({"frame":frame, "parent_id":"p1", "relative_weight":1.0})
    arrays = module.build_enhanced_pairs(api, prepared, children, mass=0.25)
    original = api["pairwise"]._build_pairwise_training_arrays(prepared.training_frame, prepared.feature_columns)
    n = len(original[1])
    np.testing.assert_array_equal(arrays.features[:n], original[0])
    np.testing.assert_array_equal(arrays.weights[:n], original[2])
    audit = arrays.family_audit
    assert audit["p1"]["child_pair_mass"] == pytest.approx(audit["p1"]["real_pair_mass"]*0.25)
    assert audit["p2"]["child_pair_mass"] == 0.0


def test_enhanced_scaler_is_fitted_on_real_pairs_only(fixture):
    module, api, tables, plan, config = fixture
    prepared = module.prepare_historical(api, tables, plan, config)
    child = prepared.training_frame.query("window_id == 'p1'").copy()
    child["window_id"] = "child"
    child.loc[child.label == 1, "metric_abs_z_max"] += 10000
    model, audit = module.fit_prepared_historical(api, prepared, [{"frame":child,"parent_id":"p1","relative_weight":1}], mass=0.25)
    arrays = api["pairwise"]._build_pairwise_training_arrays(prepared.training_frame, prepared.feature_columns)
    from sklearn.preprocessing import StandardScaler
    expected = StandardScaler().fit(arrays[0])
    observed = model.classifier.classifier.named_steps["scaler"]
    np.testing.assert_allclose(observed.scale_, expected.scale_, rtol=0, atol=0)
    assert observed.n_samples_seen_ == len(arrays[0])
    assert model.classifier.classifier.named_steps["logreg"].get_params()["solver"] == "liblinear"


def test_disabled_path_calls_original_fitter_with_unchanged_arguments(fixture, monkeypatch):
    module, api, tables, plan, config = fixture
    calls=[]
    sentinel=object()
    def original(**kwargs):
        calls.append(kwargs)
        return sentinel
    monkeypatch.setattr(api["sem"], "fit_semisupervised_ranker_on_tables", original)
    kwargs=dict(tables=tables, feature_root=ROOT, budget=30, random_state=42, model_config=config)
    assert module.fit_disabled(api, **kwargs) is sentinel
    assert calls == [kwargs]


def test_prepared_original_and_disabled_full_path_have_identical_scores(fixture):
    module, api, tables, plan, config = fixture
    clusters = pd.DataFrame({"window_id":["p1","p2"],"cluster_id":[0,1]})
    clusters.attrs["normal_cluster_id"] = -2
    original, _, _ = module.fit_disabled(api, tables=tables, feature_root=ROOT, budget=2,
        random_state=42, model_config=config, query_plan_override=plan, clustered_windows_override=clusters)
    prepared = module.prepare_historical(api, tables, plan, config)
    fitted, _ = module.fit_prepared_historical(api, prepared, [], mass=0)
    expected = original.score_entity_features(tables.entity_features)
    actual = fitted.score_entity_features(tables.entity_features)
    assert list(original.feature_columns) == list(fitted.feature_columns)
    assert expected.entity_id.tolist() == actual.entity_id.tolist()
    np.testing.assert_allclose(actual.score, expected.score, rtol=0, atol=1e-10)
