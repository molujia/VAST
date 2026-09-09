from __future__ import annotations

from collections import defaultdict

from dataclasses import replace

import inspect

import json

import os

from pathlib import Path

import pickle

import subprocess

import time

import numpy as np

import pandas as pd

from rcl_study.vae_snapshot_contract import (SnapshotContractError,OUTPUT_NAMESPACE,
    file_sha256,read_json,semantic_sha256)

from rcl_study.vae_snapshot_cvae_stage import _write,build_fit_request,materialize_historical_states

from rcl_study.vae_snapshot_execution import StageStore,controller_runtime,effective_sources,file_lock,now

from rcl_study.vae_snapshot_historical import load_historical_api,prepare_historical,reprepare_primitives

from scripts import run_historical_pairwise_hdbscan_compare as historical

def save_pickle(path,value):
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as stream: pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)

def load_pickle(path):
    with Path(path).open("rb") as stream: return pickle.load(stream)

def _print(event): print(json.dumps(event,ensure_ascii=False),flush=True)

def _run_worker_unlocked(repo,cfg,kind,payload,output,unit_status):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    request=output/"worker-request.json"; _write(request,payload)
    started=time.monotonic()
    with (output/"worker.log").open("a",encoding="utf-8") as log:
        command=[cfg["runtime"]["worker_python"],"-m","rcl_study.vae_snapshot_worker","--kind",kind,
                 "--request",str(request),"--output",str(output)]
        process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            event={"stage":kind,"status":"running","controller_pid":os.getpid(),"worker_pid":process.pid,
                   "updated_at":now(),"elapsed_seconds":time.monotonic()-started,"worker_log":str(output/"worker.log")}
            progress=output/"progress.json"
            if progress.exists(): event["counters"]=read_json(progress)
            _write(unit_status,event)
            time.sleep(2)
        if process.returncode: raise RuntimeError(kind+" worker failed; see "+str(output/"worker.log"))
    return read_json(output/"worker-result.json")

def run_worker(repo,cfg,kind,payload,output,unit_status):
    if kind in ("cvae","encode"):
        _write(unit_status,{"stage":kind,"status":"waiting_CVAE_worker","updated_at":now(),"pid":os.getpid()})
        with file_lock(_cache(repo)/"heavy-cvae.lock"):
            return _run_worker_unlocked(repo,cfg,kind,payload,output,unit_status)
    return _run_worker_unlocked(repo,cfg,kind,payload,output,unit_status)

def _posterior(path):
    with np.load(path,allow_pickle=False) as archive:
        return {"keys":list(zip(archive["case_ids"].tolist(),archive["candidate_ids"].tolist())),
                "mu":archive["mu"].copy(),"logvar":archive["logvar"].copy()}

def _validate_rankings(rankings,frame,expected_ids):
    if len(rankings)!=len(expected_ids) or {r["case_id"] for r in rankings}!=set(expected_ids):
        raise SnapshotContractError("full evaluation case membership drift")
    catalogs={str(c):set(g.entity_id.astype(str)) for c,g in frame.groupby("window_id")}
    for row in rankings:
        if (set(row["ranking"])!=catalogs[row["case_id"]] or len(row["ranking"])!=len(catalogs[row["case_id"]])
                or not all(np.isfinite(float(v["score"])) and np.isfinite(float(v["raw_score"])) for v in row["ranked_entities"])):
            raise SnapshotContractError("full evaluation candidate score closure drift")

def run_unit(repo,manifest,unit):
    if unit["method"] != "B": raise SnapshotContractError("this release implements selected method B only")
    repo=Path(repo); cfg=read_json(Path(manifest["run_root"])/"config.frozen.json"); api=load_historical_api(cfg)
    root=Path(unit["unit_output_root"]); root.mkdir(parents=True,exist_ok=True); status=root/"status.json"
    def progress(event): _write(status,{**event,"unit_id":unit["unit_id"]}); _print({**event,"unit_id":unit["unit_id"]})
    store=StageStore(root,progress=progress)
    try:
        progress({"stage":"input","status":"running","updated_at":now()})
        inputs=input_stage(repo,cfg,api,manifest,unit); material_root=Path(inputs["stage_root"])
        material=load_pickle(material_root/"prepared.pkl"); prepared=material["prepared"]
        device=inputs["metadata"]["device"]
        shared=StageStore(_cache(repo)/"cvae",progress=progress)
        def fit(root):
            run_worker(repo,cfg,"cvae",{"fit_request_path":str(material_root/"fit-request.json"),"device":device},root,status)
            return {"outputs":["checkpoint.pt","fit.done.json"],"metadata":{}}
        cvae=shared.run("fit",{"fit_identity_sha256":inputs["metadata"]["fit_identity_sha256"]},fit)
        fit_root=Path(cvae["stage_root"]); fit_commit=read_json(fit_root/"fit.done.json")
        _write(root/"shared-fit-reference.json",{"fit_identity_sha256":fit_commit["identity_sha256"],
            "fit_commit_path":str(fit_root/"fit.done.json"),"checkpoint_sha256":fit_commit["checkpoint_sha256"],
            "dataset_id":unit["dataset_id"],"regime":unit["regime"],"reused":cvae["reused"]})
        base_dependencies=effective_sources(repo,["rcl_study/vae_snapshot_historical.py","rcl_study/vae_snapshot_latent.py"])
        encoded={}
        for population in ("training","test"):
            def encode(root,population=population):
                result=run_worker(repo,cfg,"encode",{"fit_commit_path":str(fit_root/"fit.done.json"),
                    "records_path":str(material_root/("encode-"+population+".json")),"device":device},root,status)
                return {"outputs":["posterior.npz"],"metadata":result}
            encoded[population]=store.run("B_encode_"+population,{"fit":cvae["artifact_sha256"],
                "records":file_sha256(material_root/("encode-"+population+".json")),
                "source":file_sha256(repo/"src/rcl_study/vae_snapshot_encoding.py")},encode)
        def variants(root):
            from rcl_study.vae_snapshot_latent import build_latent_variants
            bundle=build_latent_variants(prepared,_posterior(Path(encoded["training"]["stage_root"])/"posterior.npz"),
                posterior_samples=cfg["posterior_samples"])
            save_pickle(root/"augmentation.pkl",bundle)
            return {"outputs":["augmentation.pkl"],"metadata":{"accepted_children":len(bundle["children"]),"latent_width":48}}
        augmentation=store.run("B_variants",{"encoding":encoded["training"]["artifact_sha256"],"inputs":inputs["artifact_sha256"],
            "K":cfg["posterior_samples"],"source":file_sha256(repo/"src/rcl_study/vae_snapshot_latent.py")},variants)
        bundle=load_pickle(Path(augmentation["stage_root"])/"augmentation.pkl")
        def base_fit(root):
            from rcl_study.vae_snapshot_latent import fit_B
            model,audit=fit_B(api,bundle,mass=cfg["augmentation_mass_per_parent"])
            save_pickle(root/"model.pkl",model); _write(root/"pair-weight-audit.json",audit)
            return {"outputs":["model.pkl","pair-weight-audit.json"],"metadata":{"feature_count":len(model.feature_columns)}}
        base=store.run("base_fit",{"augmentation":augmentation["artifact_sha256"],"mass":cfg["augmentation_mass_per_parent"],
            "sources":base_dependencies,"runtime":controller_runtime()},base_fit)
        model=load_pickle(Path(base["stage_root"])/"model.pkl")
        audit=read_json(Path(base["stage_root"])/"pair-weight-audit.json")
        descendants=[]
        for child in bundle["children"]:
            parent=child["parent_id"]; child_id=str(child["frame"].window_id.iloc[0]); family=audit[parent]
            weight=family["realized_child_weights"].get(child_id,0.)
            if weight>0:
                descendants.append({**child,"child_id":child_id,"relative_weight":weight/family["real_pair_mass"]})
        return _finish_unit(repo,cfg,api,manifest,unit,root,status,store,inputs,material,model,base,descendants,
                            encoded if unit["method"]=="B" else None,augmentation)
    except BaseException as exc:
        progress({"stage":"unit","status":"failed","error":str(exc),"updated_at":now()})
        raise

def _base_predictions(api,model,material,unit,base,posterior,output):
    if unit["method"]=="A": scores=model.score_entity_features(material["test_raw"])
    else:
        from rcl_study.vae_snapshot_latent import score_with_mean_latents
        scores=score_with_mean_latents(model,material["test_raw"],posterior)
    rankings=api["bounded"].build_per_case_rankings(scores,material["test_windows"])
    _validate_rankings(rankings,material["test_raw"],material["evaluation_ids"])
    identity={"schema_version":"conservative-lofo-base-score-artifact-v1","dataset_id":unit["dataset_id"],
        "artifact_role":"snapshot_pre_OSER_diagnostic","backend_id":"pairwise_linear",
        "query_plan_sha256":semantic_sha256({"regime":unit["regime"],"admitted":material["labels"]}),
        "model_sha256":file_sha256(Path(base["stage_root"])/"model.pkl"),"feature_order_sha256":semantic_sha256(model.feature_columns),
        "feature_columns":list(model.feature_columns),
        "candidate_ids_by_case":{str(c):g.entity_id.astype(str).tolist() for c,g in material["test_raw"].groupby("window_id",sort=False)},
        "targets_by_case":{r["case_id"]:r["targets"] for r in rankings},
        "scores_by_case":{r["case_id"]:{v["entity_id"]:v["score"] for v in r["ranked_entities"]} for r in rankings},
        "rankings_by_case":{r["case_id"]:r["ranking"] for r in rankings}}
    artifact={**identity,"score_artifact_sha256":semantic_sha256(identity)}
    _write(output/"base-score-artifact.json",artifact); _write(output/"base-rankings.json",{"rankings":rankings})
    save_pickle(output/"test-scores.pkl",scores)
    return {"outputs":["base-score-artifact.json","base-rankings.json","test-scores.pkl"],
            "metadata":{"test_case_count":len(rankings),"candidate_count":len(scores)}}

def _oser_training_request(repo,cfg,model,material,descendants,unit,output,status):
    from rcl_study.vae_snapshot_oser import fit_observable_transform,materialize_oser_cases
    transform=fit_observable_transform(material["observed_frame"],material["fit_ids"],material["evaluation_ids"])
    real_frame=model.training_frame
    real_cases=materialize_oser_cases(real_frame,model.score_prepared_entity_features(real_frame),transform,
                                     supervised=True,artifact_role="snapshot_OSER_real_training")
    children=[]; child_cases={}; started=time.monotonic()
    for i,child in enumerate(descendants):
        frame=child["frame"]; key=child["child_id"]
        cases=materialize_oser_cases(frame,model.score_prepared_entity_features(frame),transform,
                                    supervised=True,artifact_role="snapshot_OSER_descendant_training")
        child_cases[key]=cases[key]
        children.append({"parent_id":child["parent_id"],"child_id":key,"relative_weight":child["relative_weight"]})
        if (i+1)%10==0 or i+1==len(descendants):
            _write(status,{"stage":"OSER_prepare","status":"running","updated_at":now(),"pid":os.getpid(),
                "prepared_children":i+1,"total_children":len(descendants),"elapsed_seconds":time.monotonic()-started})
    payload={"labels":[{"case_id":row["case_id"],"fault_type":row["fault_type"]} for row in material["labels"]],
        "children":children,"real_cases":real_cases,"child_cases":child_cases,"regime":unit["regime"],
        "mass":cfg["augmentation_mass_per_parent"],"profile":cfg["oser"],"method":unit["method"],"unit_id":unit["unit_id"],
        "state_transform_sha256":transform["transform_sha256"]}
    _write(output/"training-request.json",payload); _write(output/"state-transform.json",transform)
    return {"outputs":["training-request.json","state-transform.json"],"metadata":{"real_cases":len(real_cases),"child_cases":len(child_cases)}}

def _finish_unit(repo,cfg,api,manifest,unit,root,status,store,inputs,material,model,base,descendants,encoded,augmentation):
    posterior=_posterior(Path(encoded["test"]["stage_root"])/"posterior.npz") if encoded else None
    prediction=store.run("base_predictions",{"base":base["artifact_sha256"],"inputs":inputs["artifact_sha256"],
        "encoding":encoded["test"]["artifact_sha256"] if encoded else None,"runtime":controller_runtime(),
        "source":semantic_sha256(inspect.getsource(_base_predictions))},
        lambda output:_base_predictions(api,model,material,unit,base,posterior,output))
    preparation=store.run("OSER_prepare",{"base":base["artifact_sha256"],"augmentation":augmentation["artifact_sha256"],
        "mass":cfg["augmentation_mass_per_parent"],"method":unit["method"],"profile":cfg["oser"],
        "source":semantic_sha256(inspect.getsource(_oser_training_request)),
        "dependencies":effective_sources(repo,["rcl_study/vae_snapshot_oser.py","rcl_study/conservative_lofo_state.py"])},
        lambda output:_oser_training_request(repo,cfg,model,material,descendants,unit,output,status))
    def fit_oser(output):
        result=run_worker(repo,cfg,"oser",{"training_request_path":str(Path(preparation["stage_root"])/"training-request.json")},output,status)
        return {"outputs":["checkpoint.pt","oser.done.json"],"metadata":{"objective_steps":result["audit"]["objective_step_count"],
            "fallback_reason":result["audit"]["fallback_reason"]}}
    oser=store.run("OSER_fit",{"prepared":preparation["artifact_sha256"],"method":unit["method"],"unit_id":unit["unit_id"],
        "runtime":manifest["worker_runtime"],"sources":effective_sources(repo,["rcl_study/vae_snapshot_oser.py","rcl_study/final_rcl_oser.py",
            "rcl_study/conservative_lofo_oser.py","rcl_study/conservative_lofo_residual.py"])},fit_oser)
    def predict_oser(output):
        from rcl_study.vae_snapshot_oser import materialize_oser_cases
        base_root=Path(prediction["stage_root"])
        cases=materialize_oser_cases(material["test_prepared"],load_pickle(base_root/"test-scores.pkl"),
            read_json(Path(preparation["stage_root"])/"state-transform.json"),supervised=False,artifact_role="snapshot_OSER_inference")
        _write(output/"prediction-request.json",{"cases":cases,"base_score_artifact":read_json(base_root/"base-score-artifact.json")})
        run_worker(repo,cfg,"oser_predict",{"prediction_request_path":str(output/"prediction-request.json"),
            "oser_commit_path":str(Path(oser["stage_root"])/"oser.done.json")},output,status)
        return {"outputs":["worker-result.json"],"metadata":{"test_case_count":len(cases)}}
    corrected=store.run("OSER_predictions",{"base_predictions":prediction["artifact_sha256"],"OSER":oser["artifact_sha256"],
        "source":file_sha256(repo/"src/rcl_study/vae_snapshot_oser.py"),
        "inference_source":file_sha256(repo/"src/rcl_study/vae_snapshot_oser_inference.py")},predict_oser)
    def evaluate(output):
        scores=load_pickle(Path(prediction["stage_root"])/"test-scores.pkl")
        final=read_json(Path(corrected["stage_root"])/"worker-result.json")
        scores["score"]=[final["final_scores_by_case"][str(row.window_id)][str(row.entity_id)] for row in scores.itertuples()]
        rankings=api["bounded"].build_per_case_rankings(scores,material["test_windows"])
        _validate_rankings(rankings,material["test_raw"],material["evaluation_ids"])
        before=read_json(Path(prediction["stage_root"])/"base-rankings.json")["rankings"]
        metrics=historical._derive_metrics_without_package(rankings)
        with (output/"rankings.jsonl").open("w",encoding="utf-8") as stream:
            for row in rankings: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        result={"schema_version":"snapshot-enhanced-result-v1","unit_id":unit["unit_id"],"dataset_id":unit["dataset_id"],
            "regime":unit["regime"],"method":unit["method"],"status":"completed","mode":manifest["mode"],
            "evidence_role":"bounded_debug_only" if manifest["mode"]=="smoke" else "ordinary_full_test",
            "supervised_case_ids":[row["case_id"] for row in material["labels"]],"evaluation_case_ids":material["evaluation_ids"],
            "metrics":metrics,"pre_OSER_metrics":historical._derive_metrics_without_package(before),
            "rankings_path":str((output/"rankings.jsonl").resolve()),"ranking_file_sha256":file_sha256(output/"rankings.jsonl"),
            "internal_OSER":read_json(Path(oser["stage_root"])/"oser.done.json")["audit"],
            "augmentation":augmentation["metadata"],"base_feature_count":len(model.feature_columns),
            "stage_lineage":{"input":inputs["artifact_sha256"],"base_fit":base["artifact_sha256"],"OSER_fit":oser["artifact_sha256"],
                             "base_predictions":prediction["artifact_sha256"],"corrected_predictions":corrected["artifact_sha256"]},
            "automatic_adoption":False,"full_pipeline_lofo":False}
        _write(output/"result.json",result)
        return {"outputs":["rankings.jsonl","result.json"],"metadata":{"test_case_count":len(rankings),"metrics":{k:metrics[k] for k in ("hit_at_1","hit_at_3","hit_at_5","top135","mrr")}}}
    evaluation=store.run("evaluation",{"predictions":corrected["artifact_sha256"],"base":prediction["artifact_sha256"],
        "source":semantic_sha256(inspect.getsource(evaluate)),"historical_ranker":unit["historical_source"]["required_file_sha256s"]},evaluate)
    receipt={"unit_id":unit["unit_id"],"status":"completed","result_path":str(Path(evaluation["stage_root"])/"result.json"),
             "result_sha256":file_sha256(Path(evaluation["stage_root"])/"result.json"),"evaluation_artifact_sha256":evaluation["artifact_sha256"],"updated_at":now()}
    _write(root/"unit.done.json",receipt); _write(status,{**receipt,"stage":"completed"})
    _print({"unit_id":unit["unit_id"],"status":"completed","mode":manifest["mode"],"test_cases":len(material["evaluation_ids"])})
    return receipt

def _cache(repo):
    return Path(os.environ["VAST_CACHE_ROOT"]).resolve()

def input_stage(repo,cfg,api,manifest,unit):
    return manifest["inputs"][unit["unit_id"]]
