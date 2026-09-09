"""Label-free OSER inference; no changes to the committed fitting dependencies.

The legacy residual-row validator required positive_indices even for prediction,
although neither forward nor residual application uses them. Replace only that
row-building seam, retaining the original score_final_oser application function.
"""
from __future__ import annotations

from types import FunctionType

from rcl_study.vae_snapshot_contract import SnapshotContractError


def _label_free_rows(model,cases,episode):
    import torch
    from rcl_study.conservative_lofo_oser import build_oser_fallback_rows
    rows=[]; candidate_map={}; normalized={}
    for key,case in cases.items():
        if set(case)!={"states","base_scores","evidence_supported","candidate_ids"}:
            raise SnapshotContractError("OSER inference must not contain supervision or unknown fields")
        states=torch.as_tensor(case["states"],dtype=torch.float32)
        scores=torch.as_tensor(case["base_scores"],dtype=torch.float32)
        evidence=torch.as_tensor(case["evidence_supported"],dtype=torch.bool)
        ids=[str(value) for value in case["candidate_ids"]]
        if (states.shape!=(len(ids),model.input_dim) or not ids or len(ids)!=len(set(ids)) or "" in ids
                or scores.shape!=(len(ids),) or evidence.shape!=(len(ids),)
                or not torch.isfinite(states).all() or not torch.isfinite(scores).all()):
            raise SnapshotContractError("OSER inference candidate tensors drift")
        candidate_map[str(key)]=ids; normalized[str(key)]=(states,evidence,ids)
    if episode.get("status")!="ready": return build_oser_fallback_rows(candidate_map,episode)["rows"]
    parameters=model.trainable_parameter_map()
    with torch.no_grad():
        for key,(states,evidence,ids) in normalized.items():
            residual,gate=model.forward_with_parameters(states,parameters)
            for i,candidate in enumerate(ids):
                supported=bool(evidence[i].item()); declared=float(gate[i].cpu()) if supported else 0.
                rows.append({"case_id":key,"candidate_id":candidate,"arm":"oser_meta","residual":float(residual[i].cpu()),
                    "gate":declared,"evidence_status":"supported" if supported else "missing","evidence_value":declared if supported else None})
    return rows


def score_label_free_oser(fitted,inference_cases,base_score_artifact):
    from rcl_study.final_rcl_oser import score_final_oser
    # Private function globals avoid monkeypatching the original module or any
    # other arm. Score/profile/checkpoint and residual-cap math remain original.
    original_globals=dict(score_final_oser.__globals__)
    original_globals["build_oser_residual_rows"]=_label_free_rows
    score=FunctionType(score_final_oser.__code__,original_globals,score_final_oser.__name__,score_final_oser.__defaults__,score_final_oser.__closure__)
    result=score(fitted=fitted,inference_cases=inference_cases,base_score_artifact=base_score_artifact,
                 artifact_role="snapshot_VAE_internal_OSER").to_dict()
    return {**result,"internal_component":True,"episode_scope":"OSER_regularization_only","full_pipeline_lofo":False,
            "score_cap_interpretation":"score_units_only","inference_label_access":"none"}
