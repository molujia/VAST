"""Portable preparation and execution around the tested B numerical stages."""
from __future__ import annotations
from dataclasses import replace
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from vast.method import REPOSITORY_ROOT,load_default_config,with_historical_authority
from rcl_study.vae_snapshot_contract import read_json,file_sha256,semantic_sha256
from rcl_study.vae_snapshot_cvae_stage import _write,build_fit_request,materialize_historical_states
from rcl_study.vae_snapshot_execution import StageStore,controller_runtime,file_lock,now,effective_sources
from rcl_study.vae_snapshot_historical import load_historical_api,prepare_historical
from rcl_study.vae_snapshot_pipeline import save_pickle

def configure_environment(output_root):
    os.environ["VAST_CACHE_ROOT"]=str(Path(output_root).resolve()/"_shared_stages")
    os.environ["PYTHONPATH"]=os.pathsep.join([str(REPOSITORY_ROOT/"src"),str(REPOSITORY_ROOT)])

def source_identity():
    paths=list((REPOSITORY_ROOT/"src").rglob("*.py"))+list((REPOSITORY_ROOT/"scripts").glob("*.py"))
    paths += [REPOSITORY_ROOT/"src/vast/default_config.json", REPOSITORY_ROOT/"docs/provenance/source-inventory.json"]
    return {p.relative_to(REPOSITORY_ROOT).as_posix():file_sha256(p) for p in sorted(paths)}

def resolve_path(spec_file,value):
    return (Path(spec_file).resolve().parent/Path(value)).resolve()

def validate_output(output, protected=()):
    output=Path(output).resolve()
    for source in (REPOSITORY_ROOT,*protected):
        source=Path(source).resolve()
        if output==source or output in source.parents or source in output.parents:
            raise ValueError("output overlaps repository or external input")

def resolve_inputs(spec_file,spec):
    datasets=[]; hashes={}
    for raw in spec["datasets"]:
        dataset=dict(raw)
        for key in ("feature_dir","split_manifest","query_plan"):
            if key in raw: dataset[key]=str(resolve_path(spec_file,raw[key]))
        dataset["supervision_plans"]={k:str(resolve_path(spec_file,v)) for k,v in raw.get("supervision_plans",{}).items()}
        files=[Path(dataset["feature_dir"])/n for n in ("windows.csv","entity_features.csv","metadata.json")]
        files += [Path(dataset[k]) for k in ("split_manifest","query_plan") if k in dataset]
        files += [Path(v) for v in dataset["supervision_plans"].values()]
        hashes.update({str(p):file_sha256(p) for p in files})
        datasets.append(dataset)
    return datasets,hashes

def validate_receipt(receipt):
    sealed={k:v for k,v in receipt.items() if k not in ("artifact_sha256","stage_root","reused","recovered")}
    root=Path(receipt["stage_root"]).resolve()
    if semantic_sha256(sealed)!=receipt["artifact_sha256"] or not receipt["output_sha256s"]:
        raise ValueError("corrupt stage receipt")
    for name,digest in receipt["output_sha256s"].items():
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file() or file_sha256(path)!=digest:
            raise ValueError("corrupt stage artifact; preserve it and investigate before resuming")

@contextmanager
def runner_lock(path):
    import fcntl
    with Path(path).open("a+") as stream:
        try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("a runner already owns this output directory") from None
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def prepare_input(api,cfg,dataset,regime,mode,worker_runtime,cache,sources):
    """Use full historical feature tables and only admitted root/type labels."""
    from rcl_study.query_active_deepening import fault_type_oracle_from_metadata_json
    from fixed_active_learning.plan_contract import validate_query_plan
    feature=Path(dataset["feature_dir"])
    identity={"schema":"vast-B-input-v1","dataset":dataset["dataset_id"],"regime":regime,"mode":mode,
        "files":{name:file_sha256(feature/name) for name in ("windows.csv","entity_features.csv","metadata.json")},
        "split":file_sha256(dataset["split_manifest"]),"config":cfg,"worker_runtime":worker_runtime,"sources":sources}
    if regime=="query_only": identity["query_plan"]=file_sha256(dataset["query_plan"])
    order_path=dataset.get("supervision_plans",{}).get(regime)
    if order_path: identity["supervision_order"]=file_sha256(order_path)
    def produce(root):
        sem=api["sem"]; full=sem.load_feature_bundle_tables(feature.parent,feature.name)
        split=api["splits"].build_nested_chronological_splits(full.windows,outer_test_ratio=.3,inner_val_ratio=.2)
        train=sem.subset_feature_bundle_tables(full,split.outer_train_window_ids)
        test=sem.subset_feature_bundle_tables(full,split.outer_test_window_ids)
        train_ids=train.windows.loc[train.windows.window_kind=="fault","window_id"].astype(str).tolist()
        test_ids=test.windows.loc[test.windows.window_kind=="fault","window_id"].astype(str).tolist()
        expected=read_json(dataset["split_manifest"])
        if set(train_ids)!=set(expected["outer_train_case_ids"]) or set(test_ids)!=set(expected["outer_test_case_ids"]) or set(train_ids)&set(test_ids):
            raise ValueError("independent chronological split membership differs")
        if len(train_ids)!=len(expected["outer_train_case_ids"]) or len(test_ids)!=len(expected["outer_test_case_ids"]): raise ValueError("duplicate split membership")
        if regime=="query_only":
            frozen=read_json(dataset["query_plan"]); validate_query_plan(frozen)
            if frozen["active_learning_seed"]!=42 or frozen["dataset_id"]!=dataset["dataset_id"]: raise ValueError("matching seed-42 HDBSCAN plan required")
            selected=list(frozen["selected_case_ids"])
            if not set(selected)<=set(train_ids): raise ValueError("query supervision is outside outer train")
        else: selected=list(train_ids)
        if order_path:
            original=sem.QueryPlan(**read_json(order_path))
            if list(original.queried_window_ids)!=selected:
                if regime=="query_only" or set(original.queried_window_ids)!=set(train_ids): raise ValueError("supervision plan membership/order differs")
                selected=list(original.queried_window_ids)
        else: original=None
        fit_ids=list(selected if regime=="oracle_full" else train_ids); evaluation=list(test_ids); recipe=dict(cfg["cvae"])
        if mode=="smoke":
            selected=selected[:3]; evaluation=evaluation[:3]; recipe["optimizer_steps"]=5
            if regime=="oracle_full":
                fit_ids=list(selected)
                normal=train.windows.loc[train.windows.window_kind=="normal","window_id"].astype(str).tolist()
                train=sem.subset_feature_bundle_tables(train,normal+fit_ids)
        indexed=train.windows.set_index("window_id")
        labels=[{"case_id":case,"root_ids":list(indexed.loc[case,"positive_ids_list"]),
            "fault_type":fault_type_oracle_from_metadata_json(indexed.loc[case,"metadata_json"])} for case in selected]
        if original is None:
            original=sem.QueryPlan(dataset=full.dataset,normal_cluster_id=-1,window_clusters={},queried_window_ids=selected,
                queried_roles={},queried_labels={},pseudo_labels={},pseudo_confidence={},metadata={"source":"frozen_hdbscan" if regime=="query_only" else "all_outer_train"})
        plan=replace(original,queried_window_ids=selected,queried_labels={r["case_id"]:r["root_ids"] for r in labels},
            queried_roles={case:original.queried_roles.get(case,"frozen_hdbscan") for case in selected})
        prepared=prepare_historical(api,api["bounded"]._strip_label_bearing_tables(train),plan,
            api["bounded"]._model_config("fault_only",oracle_full=regime=="oracle_full"))
        observed=prepared.prepared_tables.entity_features
        observed=observed[observed.window_id.astype(str).isin(fit_ids)].copy(); device=cfg["device"]
        fit_sources=effective_sources(REPOSITORY_ROOT,["rcl_study/vae_snapshot_cvae_stage.py","rcl_study/service_continuous_cvae.py",
            "rcl_study/service_continuous_cvae_training.py","rcl_study/service_continuous_neural_runner.py","rcl_study/vae_opt_discrete_training.py"])
        request=build_fit_request(dataset_id=dataset["dataset_id"],regime=regime,records=materialize_historical_states(observed),
            fit_case_ids=fit_ids,held_out_case_ids=test_ids,supervision=labels,recipe=recipe,
            effective_sources={"files":fit_sources,"historical":cfg["authority"]["historical_source"]["required_file_sha256s"],"runtime":worker_runtime,"device":device})
        stripped=api["bounded"]._strip_label_bearing_tables(test)
        test_raw=stripped.entity_features[stripped.entity_features.window_id.astype(str).isin(evaluation)].copy()
        test_prepared=sem.prepare_entity_features_for_model(test_raw,prepared.base_feature_columns,prepared.config,prepared.baselines)
        test_windows=test.windows[test.windows.window_id.astype(str).isin(evaluation)].copy()
        save_pickle(root/"prepared.pkl",{"prepared":prepared,"observed_frame":observed,"test_raw":test_raw,
            "test_prepared":test_prepared,"test_windows":test_windows,"labels":labels,"fit_ids":fit_ids,"evaluation_ids":evaluation})
        _write(root/"fit-request.json",request)
        _write(root/"encode-training.json",{"records":materialize_historical_states(prepared.training_frame)})
        _write(root/"encode-test.json",{"records":materialize_historical_states(test_prepared)})
        return {"outputs":["prepared.pkl","fit-request.json","encode-training.json","encode-test.json"],
            "metadata":{"fit_identity_sha256":request["identity_sha256"],"device":device,"supervised_case_count":len(selected),"test_case_count":len(evaluation)}}
    return StageStore(cache).run("input",identity,produce)

def prepare(spec_file,mode):
    spec=read_json(spec_file)
    if spec.get("schema_version")!="vast-runtime-v1": raise ValueError("runtime schema must be vast-runtime-v1")
    output=resolve_path(spec_file,spec["output_root"])
    if mode not in ("smoke","formal"): raise ValueError("invalid run mode")
    datasets,input_hashes=resolve_inputs(spec_file,spec)
    validate_output(output,[Path(spec_file),*input_hashes,*[d["feature_dir"] for d in datasets]])
    output.mkdir(parents=True,exist_ok=True); configure_environment(output); cfg=with_historical_authority(load_default_config())
    cfg["runtime"]={"worker_python":str(resolve_path(spec_file,spec["worker_python"]))}; cfg["device"]=spec.get("device","cuda:0")
    runtime=json.loads(subprocess.check_output([cfg["runtime"]["worker_python"],"-m","rcl_study.vae_snapshot_worker","--kind","runtime"],cwd=REPOSITORY_ROOT,text=True))
    if cfg["device"].startswith("cuda") and not runtime["cuda_available"]: raise ValueError("CUDA requested but unavailable; set device=cpu explicitly")
    sources=source_identity(); signature={"spec":spec,"mode":mode,"sources":sources,"inputs":input_hashes,"worker_runtime":runtime,"controller":controller_runtime()}
    identity=semantic_sha256(signature); target=output/"manifest.json"
    if target.is_file():
        manifest=read_json(target)
        if manifest["identity_sha256"]!=identity: raise ValueError("existing run identity differs; use a new output directory")
        if read_json(output/"config.frozen.json")!=cfg: raise ValueError("frozen configuration changed")
        for receipt in manifest["inputs"].values(): validate_receipt(receipt)
        return manifest
    if not datasets or len({d["dataset_id"] for d in datasets})!=len(datasets) or any(d["dataset_id"] not in ("rcabench","aiops2022_pre") for d in datasets): raise ValueError("invalid dataset selection")
    regimes=spec.get("regimes",["query_only","oracle_full"])
    if not regimes or len(set(regimes))!=len(regimes) or any(r not in ("query_only","oracle_full") for r in regimes): raise ValueError("invalid regime selection")
    inputs={}; units=[]; api=load_historical_api(cfg)
    for dataset in datasets:
        for regime in regimes:
            uid=dataset["dataset_id"]+"--"+regime+"--B--seed42"
            inputs[uid]=prepare_input(api,cfg,dataset,regime,mode,runtime,output/"_shared_stages/inputs"/uid,sources)
            units.append({"unit_id":uid,"dataset_id":dataset["dataset_id"],"regime":regime,"method":"B",
                "historical_source":cfg["authority"]["historical_source"],"unit_output_root":str(output/"units"/uid)})
    manifest={"schema_version":"vast-B-execution-v1","identity_sha256":identity,"signature":signature,"run_id":output.name,
        "run_root":str(output),"mode":mode,"worker_runtime":runtime,"inputs":inputs,"units":units,"created_at":now()}
    _write(output/"config.frozen.json",cfg); _write(target,manifest)
    return manifest

def collect(output):
    from scripts.run_historical_pairwise_hdbscan_compare import _read_jsonl,_derive_metrics_without_package
    from rcl_study.vae_snapshot_pipeline import load_pickle,_validate_rankings
    manifest=read_json(Path(output)/"manifest.json"); cells=[]
    cfg=read_json(Path(output)/"config.frozen.json"); load_historical_api(cfg)
    for unit in manifest["units"]:
        receipt=read_json(Path(unit["unit_output_root"])/"unit.done.json")
        if file_sha256(receipt["result_path"])!=receipt["result_sha256"]: raise ValueError("unit result changed")
        value=read_json(receipt["result_path"])
        if value["method"]!="B" or value["mode"]!=manifest["mode"] or file_sha256(value["rankings_path"])!=value["ranking_file_sha256"]: raise ValueError("B ranking artifact closure differs")
        if any(value[k]!=unit[k] for k in ("unit_id","dataset_id","regime")): raise ValueError("result unit ownership differs")
        inputs=manifest["inputs"][unit["unit_id"]]; validate_receipt(inputs)
        material=load_pickle(Path(inputs["stage_root"])/"prepared.pkl")
        rankings=_read_jsonl(Path(value["rankings_path"]))
        _validate_rankings(rankings,material["test_raw"],material["evaluation_ids"])
        targets={str(row.window_id):set(row.positive_ids_list) for row in material["test_windows"].itertuples()}
        if any(set(row["targets"])!=targets[row["case_id"]] for row in rankings): raise ValueError("ranking targets differ from evaluation truth")
        if value["metrics"]!=_derive_metrics_without_package(rankings): raise ValueError("reported metrics differ from rankings")
        cells.append(value)
    result={"schema_version":"vast-B-results-v1","status":"completed","mode":manifest["mode"],"cells":cells,"updated_at":now()}
    _write(Path(output)/"run.done.json",result)
    return result

def run(spec_file,mode,resume=False,prepare_only=False):
    spec=read_json(spec_file); output=resolve_path(spec_file,spec["output_root"])
    datasets,hashes=resolve_inputs(spec_file,spec)
    validate_output(output,[Path(spec_file),*hashes,*[d["feature_dir"] for d in datasets]])
    output.mkdir(parents=True,exist_ok=True)
    with runner_lock(output/"runner.lock"):
        if (output/"manifest.json").exists() and not resume: raise ValueError("existing run requires --resume")
        manifest=prepare(spec_file,mode)
        if prepare_only: return {"status":"prepared","units":len(manifest["units"]),"output":str(output)}
        pending=list(manifest["units"]); active=[]; failed=[]
        try:
            while pending or active:
                while pending and len(active)<2 and not failed:
                    unit=pending.pop(0); folder=Path(unit["unit_output_root"]); folder.mkdir(parents=True,exist_ok=True)
                    log=(folder/"controller.log").open("a",encoding="utf-8")
                    command=[sys.executable,str(REPOSITORY_ROOT/"scripts/run_vast.py"),"unit","--output",str(output),"--unit-id",unit["unit_id"]]
                    process=subprocess.Popen(command,cwd=REPOSITORY_ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); active.append((unit,process,log))
                for item in list(active):
                    unit,process,log=item
                    if process.poll() is not None:
                        log.close(); active.remove(item)
                        if process.returncode: failed.append({"unit":unit["unit_id"],"exit_code":process.returncode})
                _write(output/"heartbeat.json",{"pid":os.getpid(),"status":"running","active_units":[u["unit_id"] for u,_,_ in active],"queued_units":len(pending),"failures":failed,"updated_at":now()})
                if failed and not active: break
                if active or pending: time.sleep(2)
        finally:
            for unit,process,log in active:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
                log.close()
        if failed:
            _write(output/"run.failed.json",{"failures":failed,"updated_at":now()}); raise RuntimeError("B unit failed; valid stages preserved")
        result=collect(output); _write(output/"heartbeat.json",{"status":"completed","updated_at":now()})
        return {"status":result["status"],"units":len(result["cells"]),"output":str(output)}
