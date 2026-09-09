"""Version B retains every historical feature and appends 48 posterior values."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler

from rcl_study.vae_snapshot_contract import SnapshotContractError

LATENT_COLUMNS=["snapshot_latent_%02d" % i for i in range(48)]


def _aligned(frame,posterior):
    keys=[tuple(k) for k in posterior["keys"]]
    mu=np.asarray(posterior["mu"],dtype=float); logvar=np.asarray(posterior["logvar"],dtype=float)
    if (len(set(keys))!=len(keys) or mu.shape!=(len(keys),48) or logvar.shape!=mu.shape
            or not np.isfinite(mu).all() or not np.isfinite(logvar).all()):
        raise SnapshotContractError("candidate posterior layout drift")
    index={key:i for i,key in enumerate(keys)}; required=list(zip(frame.window_id.astype(str),frame.entity_id.astype(str)))
    if len(set(required))!=len(required) or not set(required)<=set(index):
        raise SnapshotContractError("candidate posterior membership drift")
    indices=[index[key] for key in required]
    return mu[indices],logvar[indices]


def attach_latents(frame,posterior):
    if set(frame)&set(LATENT_COLUMNS): raise SnapshotContractError("latent features already attached")
    mu,_=_aligned(frame,posterior)
    return pd.concat([frame.copy(),pd.DataFrame(mu,index=frame.index,columns=LATENT_COLUMNS)],axis=1)


def build_latent_variants(prepared,posterior,*,posterior_samples=8,seed=42):
    if type(posterior_samples) is not int or posterior_samples<0 or seed!=42:
        raise SnapshotContractError("invalid posterior sample count or seed")
    frame=prepared.training_frame
    means=attach_latents(frame,posterior)
    children=[]; rng=np.random.default_rng(seed)
    # Same sequential NumPy draws, candidate order and logvar clipping as the
    # established sample_latent_candidate_cases; kept torch-free for controller.
    for case in prepared.query_plan.queried_window_ids:
        parent=frame[frame.window_id.astype(str)==str(case)]
        mu,logvar=_aligned(parent,posterior)
        for sample in range(posterior_samples):
            values=mu+rng.standard_normal(mu.shape)*np.exp(.5*np.clip(logvar,-12.,8.))
            child=pd.concat([parent.copy(),pd.DataFrame(values,index=parent.index,columns=LATENT_COLUMNS)],axis=1)
            child["window_id"]="latent:%s:%02d" % (case,sample)
            children.append({"parent_id":str(case),"frame":child,"relative_weight":1.,"sample_index":sample,
                             "root_policy":"unchanged_real_roots","historical_policy":"unchanged_real_rows"})
    return {"prepared":replace(prepared,training_frame=means),"children":children,
            "feature_columns":list(prepared.feature_columns)+LATENT_COLUMNS,"historical_width":len(prepared.feature_columns),
            "posterior_samples":posterior_samples,"sampling_seed":seed,"method":"B","is_historical_control":False}


class SeparatePairScalers(BaseEstimator,TransformerMixin):
    """Independent owners fitted solely on admitted real mean pair differences."""
    def __init__(self,historical_width): self.historical_width=historical_width

    def fit(self,values,y=None):
        self.historical_scaler=StandardScaler().fit(values[:,:self.historical_width])
        self.latent_scaler=StandardScaler().fit(values[:,self.historical_width:])
        self.n_features_in_=values.shape[1]
        return self

    def transform(self,values):
        return np.concatenate([self.historical_scaler.transform(values[:,:self.historical_width]),
                               self.latent_scaler.transform(values[:,self.historical_width:])],axis=1)


def fit_B(api,bundle,*,mass=.25):
    from rcl_study.vae_snapshot_historical import fit_prepared_historical
    model,audit=fit_prepared_historical(api,bundle["prepared"],bundle["children"],mass=mass,
        feature_columns=bundle["feature_columns"],pair_scaler=SeparatePairScalers(bundle["historical_width"]))
    model.metadata.update(method="B",is_historical_control=False,inference="posterior_mean_no_decoder",
        posterior_samples=bundle["posterior_samples"],latent_scaler_fit_population="real_supervised_mean_pair_differences")
    return model,audit


def score_with_mean_latents(model,raw_candidate_rows,posterior):
    if model.metadata.get("method")!="B": raise SnapshotContractError("mean-latent scoring requires Version B model")
    return model.score_entity_features(attach_latents(raw_candidate_rows,posterior))
