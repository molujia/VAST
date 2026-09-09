"""Shared immutable CVAE fits; generation, rankers and OSER are downstream.

The old factorized architecture, reconstruction/cycle terms, KL schedule and
triplet implementation are reused. This version makes the specified batch-size
64 effective and commits the fitted checkpoint before any decoding/report work.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Mapping, Sequence

from rcl_study.vae_snapshot_contract import SnapshotContractError, file_sha256, read_json, semantic_sha256

FACTORS = ("mechanism", "propagation", "context")


def materialize_historical_states(frame):
    from rcl_study.service_continuous_unlabeled_pool import OBSERVABLE_STATE_FIELDS, build_unlabeled_state_record
    # Same documented conversion as the calibration script's
    # _rows_from_training_frame. Keep it independent of that script's controller
    # sklearn/HDBSCAN imports so the existing Python 3.8 torch worker can use it.
    records = []
    for source in frame.itertuples(index=False):
        def value(name, default=0.0):
            try:
                number = float(getattr(source, name, default))
                return number if math.isfinite(number) else float(default)
            except (ValueError, TypeError):
                return float(default)
        def ratio(numerator, denominator):
            return 0.0 if abs(denominator) <= 1e-12 else numerator / denominator
        peak, last = abs(value("metric_abs_z_max")), abs(value("metric_last_abs_z_max"))
        active, series = max(0.,value("metric_event_active_timestamp_count")), max(0.,value("metric_series_count"))
        in_degree, out_degree = max(0.,value("topo_in_degree")), max(0.,value("topo_out_degree"))
        in_weight, out_weight = abs(value("topo_in_weight")), abs(value("topo_out_weight"))
        metric_rank, trace_rank = value("rel_win_rank_metric_abs_z_max",.5), value("rel_win_rank_trace_latency_abs_z_max",.5)
        has_metric, has_log, has_trace = [int(value("has_"+m+"_signal") > 0) for m in ("metric","log","trace")]
        metric_mean = value("metric_window_mean")
        row = dict(case_id=str(source.window_id), candidate_id=str(source.entity_id),
            metric_direction=1. if metric_mean > 0 else (-1. if metric_mean < 0 else 0.),
            metric_magnitude=peak, metric_duration=active,
            metric_sparsity=ratio(max(0.,value("metric_anomalous_kpi_count")),max(series,1.)),
            log_intensity=value("log_error_count")+value("log_warn_count"),
            log_template_change=value("log_message_entropy"), log_relative_time=value("rel_win_rank_log_error_count",.5),
            trace_latency=abs(value("trace_latency_abs_z_max")), trace_error=value("trace_error_ratio"),
            trace_earliest_anomaly=1.-max(0.,min(1.,trace_rank)), trace_hop_lag=abs(value("trace_server_latency_abs_z_gap")),
            topology_depth=in_degree, topology_width=out_degree,
            topology_direction_consistency=ratio(out_weight,in_weight+out_weight+1e-12),
            relative_onset=ratio(active,max(max(0.,value("metric_sample_count")),1.)),
            relative_peak=max(0.,min(1.,metric_rank)), relative_recovery=ratio(last,max(peak,1e-12)),
            candidate_is_service=value("entity_is_service"), candidate_reachability=ratio(out_degree,in_degree+out_degree+1e-12),
            candidate_source_earliness=1.-max(0.,min(1.,min(metric_rank,trace_rank))),
            candidate_explanation_coverage=max(0.,min(3.,value("modalities_present_count")))/3.,
            has_metric_signal=has_metric,has_log_signal=has_log,has_trace_signal=has_trace,
            has_topology_signal=1,has_time_signal=int(has_metric or has_trace),has_candidate_signal=1)
        records.append(build_unlabeled_state_record(row))
    return records


def build_fit_request(*, dataset_id, regime, records, fit_case_ids, held_out_case_ids,
                      supervision, recipe, effective_sources):
    from rcl_study.service_continuous_unlabeled_pool import _forbidden_keys
    fixed = {"latent_width_per_factor":16, "hidden_width":128, "activation":"silu",
             "normalization":"layer_norm", "optimizer":"adamw"}
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise SnapshotContractError("unsupported CVAE architecture recipe")
    for key in ("optimizer_steps", "batch_size", "kl_warmup_steps"):
        if type(recipe.get(key)) is not int or recipe[key] <= 0:
            raise SnapshotContractError("invalid CVAE recipe " + key)
    fit, held = list(fit_case_ids), list(held_out_case_ids)
    admitted = [str(row["case_id"]) for row in supervision]
    if (dataset_id not in ("rcabench", "aiops2022_pre") or regime not in ("query_only", "oracle_full")
            or not fit or len(set(fit)) != len(fit) or set(fit) & set(held)
            or not admitted or len(set(admitted)) != len(admitted) or not set(admitted) <= set(fit)
            or (regime == "oracle_full" and admitted != fit)):
        raise SnapshotContractError("CVAE supervision or held-out membership drift")
    keys = [(str(row.get("case_id", "")), str(row.get("candidate_id", ""))) for row in records]
    key_set = set(keys)
    if (len(keys) != len(set(keys)) or {key[0] for key in keys} != set(fit)
            or any(not key[1] for key in keys) or _forbidden_keys(records)):
        raise SnapshotContractError("CVAE observable records contain supervision or incomplete membership")
    for row in records:
        if set(row) != {"case_id", "candidate_id", "state", "mask"}:
            raise SnapshotContractError("CVAE observable field closure drift")
        for factor, width in zip(FACTORS, (11, 6, 10)):
            values, masks = row["state"][factor], row["mask"][factor]
            if (len(values) != width or len(masks) != width or any(x not in (0,1) for x in masks)
                    or not all(math.isfinite(float(x)) for x in values)):
                raise SnapshotContractError("CVAE observable vector shape/mask drift")
    for row in supervision:
        if (not str(row.get("fault_type", "")).strip() or not row.get("root_ids")
                or any((row["case_id"], root) not in key_set for root in row["root_ids"])):
            raise SnapshotContractError("CVAE supervision root/fault type is invalid")
    identity = dict(schema_version="snapshot-cvae-fit-identity-v1", dataset_id=dataset_id, regime=regime,
        records=deepcopy(records), fit_case_ids=fit, held_out_case_ids=held, supervision=deepcopy(supervision),
        recipe=deepcopy(recipe), effective_sources=deepcopy(effective_sources), seed=42,
        batching="seeded_candidate_minibatches_v1", normalization="masked_real_fit_population_v1")
    return {"identity":identity, "identity_sha256":semantic_sha256(identity)}


def compare_fit_identity(expected: Mapping, available: Mapping | None):
    if available is None:
        return {"status":"incomplete", "differing_fields":["fit_identity"]}
    fields = sorted(key for key in set(expected) | set(available)
                    if semantic_sha256(expected.get(key)) != semantic_sha256(available.get(key)))
    if not fields:
        return {"status":"reused", "differing_fields":[]}
    inputs = {"records", "fit_case_ids", "supervision", "normalization", "held_out_case_ids"}
    return {"status":"incompatible-input" if inputs & set(fields) else "incompatible-config", "differing_fields":fields}


def _write(path, payload):
    import json
    import tempfile
    # Multiple controllers may update a shared status file concurrently.
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                                     prefix=path.name+".",suffix=".tmp",delete=False) as stream:
        stream.write(json.dumps(payload,allow_nan=False,sort_keys=True,indent=2)+"\n")
        temporary=Path(stream.name)
    temporary.replace(path)


def import_immutable_fit(request, source_root, reference_root):
    """Import a verified semantic fit by reference; never mutate its tensors.

    Legacy tensor tags alone are insufficient: a legacy artifact must first
    supply equivalent fit identity AND a committed content digest. Incompatible
    old inputs cannot be converted by renaming a schema or copying a marker.
    """
    import torch
    from rcl_study.vae_opt_continuous_encoder import load_latent_encoder_checkpoint
    from rcl_study.service_continuous_neural_runner import _state_dict_hash
    source = Path(source_root).resolve()
    saved = read_json(source / "fit.done.json")
    checkpoint = Path(saved["checkpoint_path"])
    if (saved.get("identity_sha256") != request["identity_sha256"]
            or semantic_sha256(request["identity"]) != request["identity_sha256"]
            or file_sha256(checkpoint) != saved["checkpoint_sha256"]):
        raise SnapshotContractError("immutable import identity or content differs")
    model, normalization, payload = load_latent_encoder_checkpoint(checkpoint)
    if (compare_fit_identity(request["identity"], payload.get("snapshot_fit_identity"))["status"] != "reused"
            or payload.get("snapshot_fit_identity_sha256") != request["identity_sha256"]
            or _state_dict_hash(model) != saved["state_dict_sha256"]
            or semantic_sha256(normalization) != semantic_sha256(saved["normalization"])
            or saved["optimizer_step_count"] != request["identity"]["recipe"]["optimizer_steps"]
            or not all(torch.isfinite(value).all() for value in model.state_dict().values())):
        raise SnapshotContractError("immutable import semantic or tensor validation failed")
    target = Path(reference_root)
    target.mkdir(parents=True, exist_ok=True)
    reference = {**saved, "reused":True, "imported_from":str(source),
                 "conversion_evidence":"snapshot-fit-v1 identity and tensors verified; tensor bytes unchanged"}
    path = target / "fit.reference.json"
    if path.exists() and read_json(path) != reference:
        raise SnapshotContractError("immutable fit reference already differs")
    if not path.exists():
        _write(path, reference)
    return reference


def fit_cvae_stage(request, output_root, *, progress=None, device="cpu"):
    import torch
    from rcl_study.service_continuous_cvae import FactorizedConditionalVAE
    from rcl_study.service_continuous_neural_runner import _record_tensors, _state_dict_hash
    from rcl_study.service_continuous_cvae_training import compute_unlabeled_objectives
    from rcl_study.vae_opt_discrete_training import kl_weight_at_step, mechanism_triplet_loss
    identity = request["identity"]
    if semantic_sha256(identity) != request["identity_sha256"]:
        raise SnapshotContractError("CVAE request identity changed")
    output = Path(output_root)
    checkpoint, commit = output/"checkpoint.pt", output/"fit.done.json"
    if commit.is_file():
        saved = read_json(commit)
        if (saved["identity_sha256"] != request["identity_sha256"] or not checkpoint.is_file()
                or file_sha256(checkpoint) != saved["checkpoint_sha256"]):
            raise SnapshotContractError("immutable CVAE fit identity or content differs")
        return {**saved, "reused":True}
    if checkpoint.exists():
        raise SnapshotContractError("uncommitted CVAE checkpoint requires recovery validation; do not overwrite")
    output.mkdir(parents=True, exist_ok=True)
    normalized, masks, index, normalization = _record_tensors({"pretraining_records":identity["records"]}, torch.device(device))
    recipe = identity["recipe"]
    torch.manual_seed(42)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    model = FactorizedConditionalVAE(mechanism_dim=11, propagation_dim=6, context_dim=10, target_context_dim=10,
        hidden_width=recipe["hidden_width"], mechanism_latent_width=16, propagation_latent_width=16, context_latent_width=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    batch = dict(normalized)
    batch.update({factor+"_mask":masks[factor] for factor in FACTORS})
    batch.update(target_context=normalized["context"], target_context_mask=masks["context"])
    supervised_indices, types = [], []
    for label in identity["supervision"]:
        for root in label["root_ids"]:
            supervised_indices.append(index[(label["case_id"],root)])
            types.append(label["fault_type"])
    generator = torch.Generator(device="cpu").manual_seed(42)
    batch_size = int(recipe["batch_size"])
    history, batch_sizes = [], []
    initial_sha = _state_dict_hash(model)
    model.train()
    for step in range(int(recipe["optimizer_steps"])):
        indices = torch.randperm(len(index), generator=generator)[:batch_size].to(device)
        selected = {key:value[indices] for key,value in batch.items()}
        admitted_indices = torch.randperm(len(supervised_indices), generator=generator)[:batch_size].tolist()
        root_indices = [supervised_indices[i] for i in admitted_indices]
        fault_types = [types[i] for i in admitted_indices]
        weight = kl_weight_at_step(step, final_weight=recipe["kl_weight_final"], warmup_steps=recipe["kl_warmup_steps"])
        objective = compute_unlabeled_objectives(model=model, batch=selected, sample_seed=42+step, weights={
            "masked_reconstruction":recipe["masked_reconstruction_weight"], "kl":weight,
            "target_context":recipe["target_context_weight"], "cycle_consistency":recipe["cycle_consistency_weight"]})
        mu, _ = model.mechanism_encoder(normalized["mechanism"][root_indices], masks["mechanism"][root_indices])
        triplet = mechanism_triplet_loss(mu, fault_types, margin=recipe["triplet_margin"])
        total = objective["total"] + recipe["triplet_weight"]*triplet
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            raise SnapshotContractError("nonfinite CVAE objective")
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"])
        if not torch.isfinite(gradient) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise SnapshotContractError("nonfinite CVAE gradients")
        optimizer.step()
        components = {key:float(value.detach().cpu()) for key,value in objective["components"].items()}
        row = {"step":step+1,"total_steps":int(recipe["optimizer_steps"]),"batch_size":len(indices),
               "kl_weight":weight,"kl_unweighted":components["kl"],"kl_weighted":weight*components["kl"],
               "masked_reconstruction":components["masked_reconstruction"],"gradient_norm":float(gradient.detach().cpu()),
               "triplet":float(triplet.detach().cpu()),"total":float(total.detach().cpu())}
        history.append(row); batch_sizes.append(len(indices))
        _write(output/"progress.json",row)
        if progress is not None:
            progress(row)
    model.eval()
    with torch.no_grad():
        posterior = {factor:getattr(model,factor+"_encoder")(normalized[factor],masks[factor]) for factor in FACTORS}
    diagnostics = {"history":history, "batch_sizes":batch_sizes, "finite_gradients":True,
                   "posterior":{factor:{"mu_std":float(values[0].std().cpu()),"logvar_min":float(values[1].min().cpu()),
                                        "logvar_max":float(values[1].max().cpu())} for factor,values in posterior.items()}}
    state_sha = _state_dict_hash(model)
    # Keep the established encoder-loadable tensor schema. The independently
    # committed snapshot identity owns semantic compatibility, not this legacy tag.
    payload = {"schema_version":"vae-opt-discrete-cvae-checkpoint-v1", "snapshot_fit_identity":identity,
               "snapshot_fit_identity_sha256":request["identity_sha256"], "model_dimensions":dict(model.dimensions),
               "normalization":normalization, "model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
               "state_dict_sha256":state_sha, "profile":recipe, "optimizer_state_dict":optimizer.state_dict(),
               "rng_state":torch.get_rng_state(), "sampling_generator_state":generator.get_state()}
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload,temporary)
    temporary.replace(checkpoint)
    completed = {"schema_version":"snapshot-cvae-fit-commit-v1", "identity_sha256":request["identity_sha256"],
                 "checkpoint_path":str(checkpoint.resolve()), "checkpoint_sha256":file_sha256(checkpoint),
                 "state_dict_sha256":state_sha, "initial_state_dict_sha256":initial_sha,
                 "normalization":normalization, "diagnostics":diagnostics, "rng_seed":42,
                 "optimizer_step_count":int(recipe["optimizer_steps"])}
    _write(commit,completed)
    return {**completed,"reused":False}
