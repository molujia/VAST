"""Replay owner-supplied B artifacts without fitting any model.

Run from a complete Linux checkout in the controller environment. Reference
pickles/checkpoints must be trusted. All new artifacts remain in --output-root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

def main():
    import numpy as np
    import pandas as pd
    from vast.runtime import prepare,validate_output,validate_receipt
    from rcl_study.vae_snapshot_contract import read_json,file_sha256
    from rcl_study.vae_snapshot_cvae_stage import _write
    from rcl_study.vae_snapshot_pipeline import load_pickle,_posterior,_validate_rankings
    from rcl_study.vae_snapshot_historical import load_historical_api
    from rcl_study.vae_snapshot_latent import score_with_mean_latents
    from rcl_study.vae_snapshot_oser import materialize_oser_cases
    from scripts.run_historical_pairwise_hdbscan_compare import _derive_metrics_without_package,_read_jsonl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    args=p.parse_args(); reference=args.reference_run.resolve(); output=args.output_root.resolve()
    validate_output(output,[reference]); output.mkdir(parents=True,exist_ok=True)
    old_cfg=read_json(reference/"config.frozen.json")
    old_manifest=read_json(reference/"manifest.json")
    spec={"schema_version":"vast-runtime-v1","output_root":str(output/"prepared-run"),
          "worker_python":old_cfg["runtime"]["worker_python"],"datasets":[]}
    for dataset,authority in old_cfg["authority"]["datasets"].items():
        controls=Path(old_cfg["historical_control_run"])/"units"
        oracle=controls/(dataset+"--historical_oracle_full--seed42")
        runtime=read_json(oracle/"runtime-spec.json")
        # Independent frozen membership from the approved registry; never inferred
        # from the newly prepared evaluation frame.
        split=output/(dataset+"-split.json")
        frozen_done=read_json(reference/"units"/(dataset+"--oracle_full--B--seed42")/"unit.done.json")
        if file_sha256(frozen_done["result_path"])!=frozen_done["result_sha256"]: raise ValueError("frozen split result changed")
        frozen_result=read_json(frozen_done["result_path"])
        _write(split,{"outer_train_case_ids":authority["outer_train_case_ids"],
                      "outer_test_case_ids":frozen_result["evaluation_case_ids"]})
        spec["datasets"].append({"dataset_id":dataset,"feature_dir":str(Path(runtime["feature_root"])/runtime["feature_alias"]),
            "split_manifest":str(split),"query_plan":authority["resolved_query_plan_file"],
            "supervision_plans":{"oracle_full":str(oracle/"evaluation/query_plan.json"),
                "query_only":str(controls/(dataset+"--hdbscan_query_historical_pairwise--seed42")/"evaluation/query_plan.json")}})
    spec_file=output/"runtime.json"; _write(spec_file,spec)
    manifest=prepare(spec_file,"formal")
    cfg=read_json(output/"prepared-run/config.frozen.json"); api=load_historical_api(cfg)
    def stage(folder,name,digest=None):
        found=[]
        for path in folder.glob("stages/"+name+"/*/attempt-*/stage.done.json"):
            receipt=read_json(path)
            if digest is not None and receipt["artifact_sha256"]!=digest: continue
            receipt["stage_root"]=str(path.parent); validate_receipt(receipt); found.append(path.parent)
        if len(found)!=1: raise ValueError("reference stage not uniquely identified: "+name)
        return found[0]
    def worker(kind,payload,folder):
        folder.mkdir(parents=True,exist_ok=True); request=folder/"request.json"; _write(request,payload)
        with (folder/"worker.log").open("w") as log:
            subprocess.run([spec["worker_python"],"-m","rcl_study.vae_snapshot_worker","--kind",kind,
                "--request",str(request),"--output",str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=[]
    for unit in manifest["units"]:
        uid=unit["unit_id"]; old_unit=reference/"units"/uid; replay=output/"replay"/uid
        done=read_json(old_unit/"unit.done.json")
        if file_sha256(done["result_path"])!=done["result_sha256"]: raise ValueError("reference result hash differs")
        result=read_json(done["result_path"]); lineage=result["stage_lineage"]
        old_input=stage(reference.parent/"_shared_stages/inputs"/unit["dataset_id"]/unit["regime"],"input",lineage["input"])
        new_input=Path(manifest["inputs"][uid]["stage_root"])
        old=load_pickle(old_input/"prepared.pkl"); new=load_pickle(new_input/"prepared.pkl")
        for field in ("observed_frame","test_raw","test_prepared","test_windows"):
            pd.testing.assert_frame_equal(old[field],new[field],check_exact=True)
        pd.testing.assert_frame_equal(old["prepared"].training_frame,new["prepared"].training_frame,check_exact=True)
        for field in ("labels","fit_ids","evaluation_ids"):
            if old[field]!=new[field]: raise ValueError("prepared membership/order differs: "+field)
        for name in ("encode-training.json","encode-test.json"):
            if read_json(old_input/name)!=read_json(new_input/name): raise ValueError("candidate states differ: "+name)
        old_request=read_json(old_input/"fit-request.json")["identity"]
        new_request=read_json(new_input/"fit-request.json")["identity"]
        if {k:v for k,v in old_request.items() if k!="effective_sources"}!={k:v for k,v in new_request.items() if k!="effective_sources"}:
            raise ValueError("CVAE scientific input differs")
        fit_ref=read_json(old_unit/"shared-fit-reference.json")
        worker("encode",{"fit_commit_path":fit_ref["fit_commit_path"],"records_path":str(new_input/"encode-test.json"),"device":cfg["device"]},replay/"encoding")
        posterior=_posterior(replay/"encoding/posterior.npz")
        old_posterior=_posterior(stage(old_unit,"B_encode_test")/"posterior.npz")
        assert posterior["keys"]==old_posterior["keys"]
        encoding_error=max(float(np.max(np.abs(posterior[k]-old_posterior[k]))) for k in ("mu","logvar"))
        assert encoding_error<=1e-10
        base=stage(old_unit,"base_fit",lineage["base_fit"])
        model=load_pickle(base/"model.pkl")
        scores=score_with_mean_latents(model,new["test_raw"],posterior)
        old_prediction=stage(old_unit,"base_predictions",lineage["base_predictions"])
        old_scores=load_pickle(old_prediction/"test-scores.pkl")
        score_error=float(np.max(np.abs(scores["score"].to_numpy()-old_scores["score"].to_numpy())))
        assert score_error<=1e-10
        oser_preparation=stage(old_unit,"OSER_prepare")
        cases=materialize_oser_cases(new["test_prepared"],scores,read_json(oser_preparation/"state-transform.json"),
            supervised=False,artifact_role="snapshot_OSER_inference")
        prediction_file=replay/"prediction.json"
        _write(prediction_file,{"cases":cases,"base_score_artifact":read_json(old_prediction/"base-score-artifact.json")})
        worker("oser_predict",{"prediction_request_path":str(prediction_file),
            "oser_commit_path":str(stage(old_unit,"OSER_fit",lineage["OSER_fit"])/"oser.done.json")},replay/"oser")
        corrected=read_json(replay/"oser/worker-result.json")
        old_corrected=read_json(stage(old_unit,"OSER_predictions",lineage["corrected_predictions"])/"worker-result.json")
        correction_error=max(abs(v-old_corrected["final_scores_by_case"][c][e]) for c,values in corrected["final_scores_by_case"].items() for e,v in values.items())
        assert correction_error<=1e-10
        scores["score"]=[corrected["final_scores_by_case"][str(r.window_id)][str(r.entity_id)] for r in scores.itertuples()]
        rankings=api["bounded"].build_per_case_rankings(scores,new["test_windows"])
        _validate_rankings(rankings,new["test_raw"],new["evaluation_ids"])
        assert file_sha256(result["rankings_path"])==result["ranking_file_sha256"]
        original=_read_jsonl(Path(result["rankings_path"]))
        assert {r["case_id"]:r["ranking"] for r in rankings}=={r["case_id"]:r["ranking"] for r in original}
        metrics=_derive_metrics_without_package(rankings); assert metrics==result["metrics"]
        row={"unit_id":uid,"cases":len(rankings),"feature_count":len(model.feature_columns),"input_frames_exact":True,
            "encoding_max_abs_error":encoding_error,"base_score_max_abs_error":score_error,"final_score_max_abs_error":correction_error,
            "complete_rankings_equal":True,"metrics":{k:metrics[k] for k in ("hit_at_1","hit_at_3","hit_at_5","top135","mrr")}}
        summary.append(row); print(json.dumps(row),flush=True)
        _write(output/"validation-summary.json",{"status":"completed" if len(summary)==len(manifest["units"]) else "running",
            "models_trained":0,"units":summary})

if __name__=="__main__": main()
