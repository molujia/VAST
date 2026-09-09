"""Candidate-aligned, chunked posterior encoding using frozen CVAE transforms."""
from __future__ import annotations

import numpy as np
from rcl_study.vae_snapshot_contract import SnapshotContractError

FACTORS=("mechanism","propagation","context")


def normalized_tensors(records,normalization,*,device="cpu"):
    import torch
    keys=[(str(row["case_id"]),str(row["candidate_id"])) for row in records]
    if not keys or len(keys)!=len(set(keys)): raise SnapshotContractError("candidate encoding ownership drift")
    tensors={}
    for factor,width in zip(FACTORS,(11,6,10)):
        values=np.asarray([row["state"][factor] for row in records],dtype=np.float32)
        masks=np.asarray([row["mask"][factor] for row in records],dtype=np.float32)
        mean=np.asarray(normalization[factor]["mean"],dtype=np.float32)
        scale=np.asarray(normalization[factor]["scale"],dtype=np.float32)
        if (values.shape!=(len(keys),width) or masks.shape!=values.shape or mean.shape!=(width,) or scale.shape!=(width,)
                or not all(np.isfinite(v).all() for v in (values,masks,mean,scale)) or (scale<=0).any()
                or not np.isin(masks,[0.,1.]).all()): raise SnapshotContractError("candidate normalization drift")
        tensors[factor]=torch.as_tensor(((values-mean)/scale)*masks,device=device)
        tensors[factor+"_mask"]=torch.as_tensor(masks,device=device)
    return keys,tensors


def encode_candidates(model,records,normalization,*,device="cpu",batch_size=1024):
    import torch
    from rcl_study.vae_opt_continuous_encoder import encode_factorized_batch
    if type(batch_size) is not int or batch_size<=0: raise ValueError("encoding batch size must be positive")
    keys,tensors=normalized_tensors(records,normalization,device=device); mus=[]; logvars=[]
    model.eval()
    with torch.no_grad():
        for offset in range(0,len(keys),batch_size):
            posterior=encode_factorized_batch(model,{k:v[offset:offset+batch_size] for k,v in tensors.items()})
            mus.append(posterior["mu"].cpu().numpy()); logvars.append(posterior["logvar"].cpu().numpy())
    return {"keys":keys,"mu":np.concatenate(mus),"logvar":np.concatenate(logvars),"inference":"posterior_mean_no_decoder"}
