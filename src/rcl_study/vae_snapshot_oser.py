"""Internal oser-p02 with versioned descendant mass and separate arm ownership."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

from rcl_study.vae_snapshot_contract import SnapshotContractError,file_sha256,read_json,semantic_sha256
from rcl_study.vae_snapshot_cvae_stage import _write


def build_snapshot_episodes(labels,children,*,regime,mass=.25):
    from rcl_study.final_rcl_oser import build_final_oser_family_episodes
    if not math.isfinite(mass) or mass<0: raise SnapshotContractError("invalid OSER family mass")
    admitted=[str(row["case_id"]) for row in labels]; by_parent=defaultdict(list); seen=set()
    for child in children:
        parent=str(child["parent_id"]); child_id=str(child["child_id"]); weight=float(child.get("relative_weight",1.))
        if parent not in admitted: raise SnapshotContractError("OSER child parent is not admitted")
        if child_id in seen or child_id in admitted or not math.isfinite(weight) or weight<0:
            raise SnapshotContractError("OSER child identity/weight drift")
        seen.add(child_id)
        if weight>0 and mass>0: by_parent[parent].append((child_id,weight))
    rows=[]; mapping={}
    for parent in admitted:
        total=math.fsum(weight for _,weight in by_parent[parent])
        for child_id,weight in by_parent[parent]:
            digest=semantic_sha256({"child_id":child_id,"parent_id":parent})
            mapping[child_id]="synthetic:"+digest
            rows.append({"synthetic_row_sha256":digest,"source_case_id":parent,"query_budget_cost":0,
                         "final_training_weight":weight/total})
    # Reuse original family construction/quarantine with normalized legacy mass;
    # version the returned artifact and change ONLY effective descendant weights.
    episode=build_final_oser_family_episodes(real_label_records=labels,supervised_case_ids=admitted,
        synthetic_rows=rows,training_mode=regime,supervised_case_limit=len(admitted))
    episode.pop("episode_sha256")
    for key in episode["synthetic_case_keys"]: episode["case_weights"][key]*=mass
    episode.update(schema_version="snapshot-oser-family-episodes-v1",augmentation_mass_per_parent=mass,
        child_key_by_id=mapping,internal_component=True,episode_scope="OSER_regularization_only")
    return {**episode,"episode_sha256":semantic_sha256(episode)}


def observable_schema():
    from rcl_study.conservative_lofo_state import ObservableStateSchema
    return ObservableStateSchema(
        metric_fields=("metric_direction","metric_magnitude","metric_duration","metric_sparsity"),
        log_fields=("log_intensity","log_template_change","log_relative_time"),
        trace_fields=("trace_latency","trace_error","trace_earliest_anomaly","trace_hop_lag"),
        topology_fields=("topology_depth","topology_width","topology_direction_consistency"),
        time_fields=("relative_onset","relative_peak","relative_recovery"),
        candidate_fields=("candidate_is_service","candidate_reachability","candidate_source_earliness","candidate_explanation_coverage"),
        modality_presence_fields={m:"has_"+m+"_signal" for m in ("metric","log","trace","topology","time","candidate")})


def observable_rows(frame):
    from rcl_study.vae_snapshot_cvae_stage import materialize_historical_states
    from rcl_study.service_continuous_unlabeled_pool import _MECHANISM_FIELDS,_PROPAGATION_FIELDS,_CONTEXT_VALUE_FIELDS,_PRESENCE_FIELDS
    rows=[]
    for record in materialize_historical_states(frame):
        row={"case_id":record["case_id"],"candidate_id":record["candidate_id"]}
        for factor,names in (("mechanism",_MECHANISM_FIELDS),("propagation",_PROPAGATION_FIELDS),
                             ("context",_CONTEXT_VALUE_FIELDS+_PRESENCE_FIELDS)):
            row.update(zip(names,record["state"][factor]))
        rows.append(row)
    return rows


def fit_observable_transform(real_frame,fit_case_ids,held_out_case_ids):
    from rcl_study.conservative_lofo_state import fit_fold_state_transform
    return fit_fold_state_transform(observable_rows(real_frame),observable_schema(),tuple(fit_case_ids),tuple(held_out_case_ids))


def materialize_oser_cases(frame,scored,transform,*,supervised,artifact_role):
    from rcl_study.conservative_lofo_state import build_candidate_state_rows
    rows=observable_rows(frame); ids=list(dict.fromkeys(row["case_id"] for row in rows))
    artifact=build_candidate_state_rows(rows,observable_schema(),transform,ids,artifact_role)
    states={(str(member["case_id"]),str(member["candidate_id"])):state for member,state in zip(artifact["membership"],artifact["state_rows"])}
    scores={(str(row.window_id),str(row.entity_id)):float(row.score) for row in scored.itertuples()}
    if set(states)!=set(scores) or not all(math.isfinite(v) for v in scores.values()):
        raise SnapshotContractError("OSER state/base-score candidate closure drift")
    result={}
    for case,group in frame.groupby("window_id",sort=False):
        candidate_ids=group.entity_id.astype(str).tolist(); owned=[states[(str(case),c)] for c in candidate_ids]
        result[str(case)]={"candidate_ids":candidate_ids,"states":[row["state_vector"] for row in owned],
            "evidence_supported":[any(int(v) for v in row["state_mask"]) for row in owned],
            "base_scores":[scores[(str(case),c)] for c in candidate_ids]}
        if supervised:
            positives=[i for i,value in enumerate(group.label) if float(value)==1.]
            if not positives: raise SnapshotContractError("OSER supervised root missing")
            result[str(case)]["positive_indices"]=positives
    return result


def fit_snapshot_oser(training_cases,episode,config,*,method,unit_id,state_transform_sha256,output_root):
    import torch
    from rcl_study import final_rcl_oser as original
    if method not in ("A","B"): raise SnapshotContractError("OSER must have a VAE arm owner")
    profile={**config,"profile_sha256":config.get("profile_sha256",original._OSER_PROFILE_SHA256)}
    names=("vae_snapshot_oser.py","final_rcl_oser.py","conservative_lofo_oser.py","conservative_lofo_residual.py")
    identity={"schema_version":"snapshot-internal-oser-fit-v1","method":method,"unit_id":unit_id,"seed":42,
        "training_cases_sha256":semantic_sha256(training_cases),"episode_sha256":episode["episode_sha256"],
        "profile":profile,"state_transform_sha256":state_transform_sha256,"torch":torch.__version__,
        "threads":torch.get_num_threads(),"sources":{name:file_sha256(Path(__file__).parent/name) for name in names}}
    digest=semantic_sha256(identity); output=Path(output_root); checkpoint=output/"checkpoint.pt"; commit=output/"oser.done.json"
    if commit.exists():
        saved=read_json(commit)
        if saved["identity_sha256"]!=digest or file_sha256(checkpoint)!=saved["checkpoint_sha256"]:
            raise SnapshotContractError("immutable OSER identity or content differs")
        payload=torch.load(checkpoint,map_location="cpu")
        if payload["identity_sha256"]!=digest: raise SnapshotContractError("OSER payload identity mismatch")
        model=original.OSERMetaResidualModel(input_dim=payload["input_dim"],hidden_dim=32,residual_cap=.05,seed=42)
        model.load_state_dict(payload["model_state_dict"])
        return original.FinalOSERFit(model,episode,profile,payload["audit"]),{**saved,"reused":True}
    if checkpoint.exists(): raise SnapshotContractError("uncommitted OSER weights require recovery validation")
    output.mkdir(parents=True,exist_ok=True)
    fitted=original.fit_final_oser(training_cases=training_cases,episode_artifact=episode,profile=profile,
        state_transform_sha256=state_transform_sha256,seed=42)
    if (not math.isfinite(fitted.audit["gradient_l1"]) or not all(torch.isfinite(v).all() for v in fitted.model.state_dict().values())):
        raise SnapshotContractError("nonfinite OSER fit")
    payload={"identity_sha256":digest,"input_dim":len(next(iter(training_cases.values()))["states"][0]),
             "episode":episode,"profile":profile,
             "model_state_dict":{k:v.detach().cpu() for k,v in fitted.model.state_dict().items()},"audit":fitted.audit}
    temp=checkpoint.with_suffix(".tmp"); torch.save(payload,temp); temp.replace(checkpoint)
    saved={"identity_sha256":digest,"identity":identity,"method":method,"unit_id":unit_id,"internal_component":True,
        "checkpoint_path":str(checkpoint.resolve()),"checkpoint_sha256":file_sha256(checkpoint),"audit":fitted.audit}
    _write(commit,saved)
    return fitted,{**saved,"reused":False}


def score_snapshot_oser(fitted,inference_cases,base_score_artifact):
    from rcl_study.final_rcl_oser import score_final_oser
    output=score_final_oser(fitted=fitted,inference_cases=inference_cases,base_score_artifact=base_score_artifact,
        artifact_role="snapshot_VAE_internal_OSER").to_dict()
    return {**output,"internal_component":True,"episode_scope":"OSER_regularization_only",
            "score_cap_interpretation":"score_units_only","full_pipeline_lofo":False}


def load_snapshot_oser(commit):
    import torch
    from rcl_study import final_rcl_oser as original
    if file_sha256(commit["checkpoint_path"])!=commit["checkpoint_sha256"]:
        raise SnapshotContractError("OSER checkpoint content differs")
    payload=torch.load(commit["checkpoint_path"],map_location="cpu")
    if payload["identity_sha256"]!=commit["identity_sha256"]: raise SnapshotContractError("OSER checkpoint identity differs")
    model=original.OSERMetaResidualModel(input_dim=payload["input_dim"],hidden_dim=32,residual_cap=.05,seed=42)
    model.load_state_dict(payload["model_state_dict"])
    return original.FinalOSERFit(model,payload["episode"],payload["profile"],payload["audit"])
