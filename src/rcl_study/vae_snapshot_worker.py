"""Small JSON-only bridge to the existing torch interpreter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from rcl_study.vae_snapshot_contract import read_json,file_sha256
from rcl_study.vae_snapshot_cvae_stage import _write


def runtime():
    import torch
    return {"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_threads":torch.get_num_threads(),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--kind",choices=["runtime","cvae","encode","oser","oser_predict"],required=True)
    parser.add_argument("--request",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.kind=="runtime": print(json.dumps(runtime())); return
    request=read_json(args.request); output=args.output; output.mkdir(parents=True,exist_ok=True)
    device=request.get("device","cpu")
    def progress(row): print(json.dumps(row),flush=True)
    if args.kind=="cvae":
        from rcl_study.vae_snapshot_cvae_stage import fit_cvae_stage
        result=fit_cvae_stage(read_json(request["fit_request_path"]),output,progress=progress,device=device)
    elif args.kind=="encode":
        import numpy as np
        from rcl_study.vae_opt_continuous_encoder import load_latent_encoder_checkpoint
        from rcl_study.vae_snapshot_encoding import encode_candidates
        commit=read_json(request["fit_commit_path"])
        if file_sha256(commit["checkpoint_path"])!=commit["checkpoint_sha256"]: raise ValueError("encoding checkpoint hash differs")
        model,norm,_=load_latent_encoder_checkpoint(commit["checkpoint_path"],device=device)
        encoded=encode_candidates(model,read_json(request["records_path"])["records"],norm,device=device)
        target=output/"posterior.npz"
        with (output/"posterior.tmp").open("wb") as stream:
            np.savez(stream,case_ids=np.asarray([k[0] for k in encoded["keys"]]),candidate_ids=np.asarray([k[1] for k in encoded["keys"]]),
                     mu=encoded["mu"],logvar=encoded["logvar"])
        (output/"posterior.tmp").replace(target)
        result={"candidate_rows":len(encoded["keys"]),"latent_width":48,"inference":encoded["inference"]}
    elif args.kind=="oser":
        from rcl_study.vae_snapshot_oser import build_snapshot_episodes,fit_snapshot_oser
        payload=read_json(request["training_request_path"])
        episode=build_snapshot_episodes(payload["labels"],payload["children"],regime=payload["regime"],mass=payload["mass"])
        mapping=episode["child_key_by_id"]
        cases={row["case_id"]:payload["real_cases"][row["case_id"]] for row in payload["labels"]}
        cases.update({mapping[key]:payload["child_cases"][key] for key in mapping})
        _,result=fit_snapshot_oser(cases,episode,payload["profile"],method=payload["method"],unit_id=payload["unit_id"],
            state_transform_sha256=payload["state_transform_sha256"],output_root=output)
    else:
        from rcl_study.vae_snapshot_oser import load_snapshot_oser
        from rcl_study.vae_snapshot_oser_inference import score_label_free_oser
        payload=read_json(request["prediction_request_path"])
        result=score_label_free_oser(load_snapshot_oser(read_json(request["oser_commit_path"])),payload["cases"],payload["base_score_artifact"])
    _write(output/"worker-result.json",result)
    print(json.dumps({"stage":args.kind,"status":"completed","output":str(output)}),flush=True)


if __name__=="__main__": main()
