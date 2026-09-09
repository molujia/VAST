import importlib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from test_vae_snapshot_historical import fixture


def api(): return importlib.import_module("rcl_study.vae_snapshot_latent")


def posterior(frame):
    n=len(frame)
    return {"keys":list(zip(frame.window_id.astype(str),frame.entity_id.astype(str))),
            "mu":np.arange(n*48).reshape(n,48)*.01,"logvar":np.full((n,48),-.5)}


def test_historical_features_and_roots_retained_in_all_variants(fixture):
    module=api(); historical,hapi,tables,plan,config=fixture
    prepared=historical.prepare_historical(hapi,tables,plan,config); encoded=posterior(prepared.training_frame)
    bundle=module.build_latent_variants(prepared,encoded,posterior_samples=8)
    assert bundle["feature_columns"]==prepared.feature_columns+module.LATENT_COLUMNS
    pd.testing.assert_frame_equal(bundle["prepared"].training_frame[prepared.training_frame.columns],prepared.training_frame)
    assert len(bundle["children"])==16
    for child in bundle["children"]:
        parent=prepared.training_frame[prepared.training_frame.window_id==child["parent_id"]]
        compare=[c for c in parent if c!="window_id"]
        pd.testing.assert_frame_equal(child["frame"][compare],parent[compare])
        assert child["frame"].entity_id.tolist()==parent.entity_id.tolist()
        assert len(child["frame"][module.LATENT_COLUMNS].columns)==48


def test_sampling_is_deterministic_and_uses_candidate_specific_posteriors(fixture):
    module=api(); historical,hapi,tables,plan,config=fixture
    prepared=historical.prepare_historical(hapi,tables,plan,config); encoded=posterior(prepared.training_frame)
    left=module.build_latent_variants(prepared,encoded,posterior_samples=2)
    right=module.build_latent_variants(prepared,encoded,posterior_samples=2)
    for a,b in zip(left["children"],right["children"]): pd.testing.assert_frame_equal(a["frame"],b["frame"])
    means=left["prepared"].training_frame[module.LATENT_COLUMNS].to_numpy()
    np.testing.assert_array_equal(means,encoded["mu"])
    assert not np.array_equal(means[:3],left["children"][0]["frame"][module.LATENT_COLUMNS])
    shuffled={**encoded,"keys":encoded["keys"][::-1],"mu":encoded["mu"][::-1],"logvar":encoded["logvar"][::-1]}
    pd.testing.assert_frame_equal(module.attach_latents(prepared.training_frame,encoded),module.attach_latents(prepared.training_frame,shuffled))


@pytest.mark.parametrize("k",[0,1,8])
def test_mass_is_independent_of_k_and_scalers_fit_real_pairs_only(fixture,k):
    module=api(); historical,hapi,tables,plan,config=fixture
    prepared=historical.prepare_historical(hapi,tables,plan,config); encoded=posterior(prepared.training_frame)
    bundle=module.build_latent_variants(prepared,encoded,posterior_samples=k)
    model,audit=module.fit_B(hapi,bundle,mass=.25)
    original=hapi["pairwise"]._build_pairwise_training_arrays(prepared.training_frame,prepared.feature_columns,negative_top_k=config.pairwise_negative_top_k)
    combined=hapi["pairwise"]._build_pairwise_training_arrays(bundle["prepared"].training_frame,bundle["feature_columns"],negative_top_k=config.pairwise_negative_top_k)
    np.testing.assert_array_equal(original[0],combined[0][:,:len(prepared.feature_columns)])
    scaler=model.classifier.classifier.named_steps["scaler"]
    np.testing.assert_array_equal(scaler.historical_scaler.scale_,StandardScaler().fit(original[0]).scale_)
    np.testing.assert_array_equal(scaler.latent_scaler.scale_,StandardScaler().fit(combined[0][:,-48:]).scale_)
    assert scaler.historical_scaler is not scaler.latent_scaler
    for parent,row in audit.items(): assert row["child_pair_mass"]==pytest.approx(row["real_pair_mass"]*(.25 if k else 0.))
    assert model.metadata["method"]=="B" and model.metadata["is_historical_control"] is False
    scores=module.score_with_mean_latents(model,tables.entity_features,encoded)
    assert np.isfinite(scores.score).all() and len(scores)==len(tables.entity_features)


def test_missing_or_duplicate_candidate_posteriors_are_rejected(fixture):
    module=api(); historical,hapi,tables,plan,config=fixture
    prepared=historical.prepare_historical(hapi,tables,plan,config); encoded=posterior(prepared.training_frame)
    encoded["keys"][0]=encoded["keys"][1]
    with pytest.raises(ValueError,match="candidate|posterior"):
        module.attach_latents(prepared.training_frame,encoded)
