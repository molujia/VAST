"""Narrow preparation/pair-array seam around the immutable historical source.

The historical package is loaded under a private name so its relative imports
cannot resolve to the current integrated rcl_study package. A process with a
different nexusrcl_rebuild already loaded must use a fresh worker instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rcl_study.vae_snapshot_contract import SnapshotContractError, file_sha256


def load_historical_api(config: Mapping) -> dict:
    root = Path(config["resolved_historical_source"]).resolve()
    for relative, expected in config["authority"]["historical_source"]["required_file_sha256s"].items():
        if file_sha256(root / relative) != expected:
            raise SnapshotContractError(f"historical source drift: {relative}")
    existing = sys.modules.get("nexusrcl_rebuild")
    if existing is not None and root not in Path(existing.__file__).resolve().parents:
        raise SnapshotContractError("a different historical runtime is loaded; use a fresh process")
    for path in reversed((root/"half_supervise/src", root, root/"self_supervise", root/"metric_AD")):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Same import search paths as the historical comparison runner, private
    # rcl_study package ownership to permit current adapters in the same process.
    package_name = "_vae_snapshot_historical_rcl"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(package_name, root/"rcl_study/__init__.py",
                                                      submodule_search_locations=[str(root/"rcl_study")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    elif Path(sys.modules[package_name].__file__).resolve().parent != root/"rcl_study":
        raise SnapshotContractError("historical package root changed inside a live process")
    sem = importlib.import_module("nexusrcl_rebuild.training.semisupervised")
    bounded = importlib.import_module(package_name + ".semi_supervised_bounded")
    pairwise = importlib.import_module("nexusrcl_rebuild.training.pairwise_backend")
    splits = importlib.import_module("nexusrcl_rebuild.evaluation.splits")
    return dict(sem=sem, bounded=bounded, pairwise=pairwise, splits=splits, source_root=root)


@dataclass
class HistoricalPreparation:
    raw_tables: Any
    prepared_tables: Any
    training_frame: pd.DataFrame
    base_feature_columns: list
    feature_columns: list
    baselines: Any
    query_plan: Any
    config: Any
    preprocessing_population: str = "real_outer_train_only"


def prepare_historical(api: Mapping, tables: Any, query_plan: Any, config: Any) -> HistoricalPreparation:
    sem = api["sem"]
    if (config.model_backend != "pairwise_linear" or config.propagate_mode != "disabled"
            or config.use_graph_context_features or config.use_graph_diffusion or config.gated_onehop_relations
            or config.entity_score_calibration_mode != "none" or config.prototype_rerank_mode != "none"):
        raise SnapshotContractError("historical adapter requires the registered pairwise scoring configuration")
    ids = tables.windows.loc[tables.windows.window_kind == "fault", "window_id"].astype(str).tolist()
    admitted = list(query_plan.queried_window_ids)
    if not set(admitted) <= set(ids) or len(admitted) != len(set(admitted)) or query_plan.pseudo_labels:
        raise SnapshotContractError("historical admitted supervision membership drift")
    if config.supervision_mode == "oracle_full" and set(admitted) != set(ids):
        raise SnapshotContractError("Oracle requires all outer-training fault cases")
    base = sem.resolve_feature_columns(tables.feature_columns, config)
    baselines = sem.build_entity_feature_baselines(tables.entity_features, base, config)
    prepared = sem.prepare_feature_bundle_tables_for_model(tables, base, config, baselines)
    frame = sem.build_training_frame(prepared, query_plan, config)
    if set(frame.window_id.astype(str)) != set(admitted):
        raise SnapshotContractError("historical training frame omitted or added supervision")
    return HistoricalPreparation(tables, prepared, frame, list(base), list(prepared.feature_columns),
                                 baselines, query_plan, config)


def reprepare_primitives(api: Mapping, prepared: HistoricalPreparation, rows: pd.DataFrame) -> pd.DataFrame:
    # Drop previous derived values; historical code recomputes them using frozen
    # baselines and the new candidate-complete case, without another fit.
    derived = set(prepared.feature_columns) - set(prepared.base_feature_columns)
    primitive = rows.drop(columns=[c for c in derived if c in rows], errors="ignore")
    return api["sem"].prepare_entity_features_for_model(primitive, prepared.base_feature_columns,
                                                       prepared.config, prepared.baselines)


@dataclass
class EnhancedPairs:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    real_pair_count: int
    family_audit: dict


def build_enhanced_pairs(api: Mapping, prepared: HistoricalPreparation, children: Sequence[Mapping],
                         *, mass: float = 0.25, feature_columns: Sequence[str] | None = None) -> EnhancedPairs:
    if not np.isfinite(mass) or mass < 0:
        raise SnapshotContractError("descendant pair mass must be finite and nonnegative")
    columns = list(feature_columns or prepared.feature_columns)
    builder = api["pairwise"]._build_pairwise_training_arrays
    def pairs(frame):
        return builder(frame, columns, negative_top_k=prepared.config.pairwise_negative_top_k)
    real_x, real_y, real_w = pairs(prepared.training_frame)
    audit = {}
    for parent, frame in prepared.training_frame.groupby("window_id", sort=False):
        _, _, weights = pairs(frame)
        audit[str(parent)] = dict(real_pair_mass=float(weights.sum()), child_pair_mass=0.0,
                                  accepted_children=0, realized_child_weights={})
    pending = {parent: [] for parent in audit}
    seen = set(audit)
    for child in children:
        parent = str(child["parent_id"])
        if parent not in audit:
            raise SnapshotContractError("child parent has no admitted real supervision")
        frame = child["frame"]
        ids = frame.window_id.astype(str).unique().tolist()
        if len(ids) != 1 or ids[0] in seen:
            raise SnapshotContractError("child must have one unique new case identity")
        seen.add(ids[0])
        relative = float(child.get("relative_weight", 1.0))
        if not np.isfinite(relative) or relative < 0:
            raise SnapshotContractError("invalid relative child weight")
        x, y, w = pairs(frame)
        if not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any():
            raise SnapshotContractError("nonfinite or negative child pair values")
        if w.sum() > 0 and relative > 0 and mass > 0:
            pending[parent].append((ids[0], x, y, w, relative))
    xs, ys, ws = [real_x], [real_y], [real_w]
    for parent, descendants in pending.items():
        relative_total = sum(row[4] for row in descendants)
        target_mass = mass * audit[parent]["real_pair_mass"]
        for case, x, y, w, relative in descendants:
            realized = target_mass * relative / relative_total
            w = w * (realized / w.sum())
            xs.append(x); ys.append(y); ws.append(w)
            audit[parent]["child_pair_mass"] += float(w.sum())
            audit[parent]["accepted_children"] += 1
            audit[parent]["realized_child_weights"][case] = float(w.sum())
    return EnhancedPairs(np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), len(real_y), audit)


def fit_prepared_historical(api: Mapping, prepared: HistoricalPreparation, children: Sequence[Mapping],
                            *, mass: float = 0.25, feature_columns: Sequence[str] | None = None,
                            pair_scaler=None):
    columns = list(feature_columns or prepared.feature_columns)
    arrays = build_enhanced_pairs(api, prepared, children, mass=mass, feature_columns=columns)
    pairwise = api["pairwise"]
    if arrays.real_pair_count == 0:
        classifier = pairwise.ConstantPairwiseClassifier(probability=0.5)
        ranker = pairwise.PairwiseClassifierRanker(classifier, columns)
    else:
        # Scaling belongs solely to original supervised pair differences. Each
        # latent column, when appended, also uses only the real mean variants.
        scaler = (StandardScaler() if pair_scaler is None else pair_scaler).fit(arrays.features[:arrays.real_pair_count])
        logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="liblinear")
        logreg.fit(scaler.transform(arrays.features), arrays.labels, sample_weight=arrays.weights)
        classifier = Pipeline([("scaler", scaler), ("logreg", logreg)])
        classifier.training_diagnostics = {"schema_version":"snapshot-pair-mass-v1", "families":arrays.family_audit}
        ranker = pairwise.PairwiseClassifierRanker(classifier, columns)
    model = api["sem"].SemiSupervisedModel(
        dataset=prepared.raw_tables.dataset, feature_columns=columns,
        base_feature_columns=list(prepared.base_feature_columns), classifier=ranker, adjacency=None,
        diffusion_alpha=prepared.config.diffusion_alpha, diffusion_steps=prepared.config.diffusion_steps,
        query_plan=prepared.query_plan, config=prepared.config, entity_feature_baselines=prepared.baselines,
        training_frame=prepared.training_frame.copy(), metadata={"schema_version":"snapshot-historical-model-v1",
        "selected_feature_columns":columns, "selected_base_feature_columns":prepared.base_feature_columns,
        "scaler_fit_population":"real_supervised_pairs_only", "family_audit":arrays.family_audit})
    return model, arrays.family_audit


def fit_disabled(api: Mapping, **kwargs):
    """Entire enhancement disabled: original full historical call, unchanged."""
    return api["sem"].fit_semisupervised_ranker_on_tables(**kwargs)
