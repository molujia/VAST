"""One observable case, no root reads, no accuracy or demo-training claim."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import torch

from .runtime import build_objectives, canonical_hash, package_code_hash
from .training.checkpoint import save_checkpoint, load_checkpoint


def atomic_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name, dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_smoke(case, preprocessing, config, *, text_encoder, output_dir, data_hash, device='cpu'):
    """Finite engineering check. It never accepts a root-label argument.

    Its isolated one-step weights exercise observation and U paths without
    the expensive complete schedule. Such a checkpoint must not seed a B30 run.
    PN/labeled-source and U-empty branches are covered by synthetic integration
    tests; checking real PN would need a separately accounted label query.
    """
    if config.variant != 'A3':
        raise ValueError('label-free smoke exercises the full A3 implementation only')
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError('smoke output already exists; retain it and choose a new attempt directory')
    if not isinstance(data_hash, str) or not data_hash:
        raise ValueError('smoke requires an explicit input identity')
    destination.mkdir(parents=True)
    report = dict(schema='vast-v3-observable-smoke/2', complete=False, case_id=case.case_id,
                  dataset=config.dataset, root_labels_accessed=0, benchmark_result=False,
                  fragment_sampler=config.fragment_sampler,
                  valid_demo_checkpoint=False, optimizer_steps=0, device=str(device),
                  candidate_count=len(case.candidate_ids), entity_count=len(case.entities),
                  numeric_series_count=len(case.numeric), numeric_samples=sum(len(s.times) for s in case.numeric),
                  log_records=len(case.logs), relation_count=len(case.relations),
                  delta_seconds=preprocessing.delta, components={}, phases={})
    start = time.perf_counter()

    def sync():
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)

    def phase(name):
        report['phase'] = name
        report['elapsed_seconds'] = time.perf_counter() - start
        atomic_json(destination / 'smoke-progress.json', report)

    identity = dict(data_hash=data_hash, config_hash=config.identity, preprocessing_hash=preprocessing.identity,
                    code_hash=package_code_hash(), fragment_sampler=config.fragment_sampler,
                    purpose='label-free-engineering-smoke')
    report['identity'] = identity
    atomic_json(destination / 'runtime-config.json', config.to_dict())
    atomic_json(destination / 'preprocessing.json', preprocessing.to_dict())
    try:
        phase('construct_model')
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        objectives = build_objectives(config, preprocessing, text_encoder=text_encoder, device=device)
        optimizer = torch.optim.Adam(objectives.parameters(), lr=.001)
        report['parameters'] = sum(p.numel() for p in objectives.parameters())

        phase('encode_and_rank')
        tick = time.perf_counter()
        objectives.eval()
        with torch.no_grad():
            first = objectives.model(case, seed=config.seed)
        sync()
        report['phases']['initial_encode_rank_seconds'] = time.perf_counter() - tick
        report['input_conditioning'] = {'preprocessing_schema': preprocessing.to_dict()['schema']}
        for name, value in (('initial_score', first.scores), ('fused_state', first.front.h),
                            ('local_state', first.front.local), ('edge_state', first.front.edge_states)):
            finite = bool(torch.isfinite(value).all())
            maximum = (float(value.detach().abs().max()) if value.numel() else 0.0) if finite else None
            report['input_conditioning'][name+'_max_abs'] = maximum
            report['input_conditioning'][name+'_finite'] = finite
        report['retained_fragments'] = len(first.fragments)
        report['fragment_candidate_count'] = first.fragment_candidates
        report['sampling_mark_count'] = len(first.sampling_marks)
        report['sampling_audit'] = None
        if first.sampling_audit is not None:
            atomic_json(destination / 'sampling-audit.json', first.sampling_audit)
            report['sampling_audit'] = dict(schema=first.sampling_audit['schema'],
                seed=first.sampling_audit['seed'], sha256=canonical_hash(first.sampling_audit),
                retained_fragment_count=len(first.fragments))
        if set(first.candidate_ids) != set(case.candidate_ids) or not torch.isfinite(first.scores).all():
            raise ValueError('full-catalog finite ranking contract failed')
        del first

        phase('observation_and_U_forward_backward')
        tick = time.perf_counter()
        objectives.train()
        names = ('local_recovery', 'process_recovery', 'invariance', 'U_soft')
        losses = objectives.case_losses(case, epoch=0, seed=config.seed, components=set(names))
        total = None
        for name in names:
            value = losses[name]
            report['components'][name] = {'eligible': value is not None,
                                         'loss': float(value.detach()) if value is not None else None}
            if value is None:
                continue
            if not torch.isfinite(value):
                raise FloatingPointError(f'nonfinite {name} smoke loss')
            weight = .1 if name == 'U_soft' else .2 / 3
            term = weight * value
            total = term if total is None else total + term
        if total is None or not total.requires_grad:
            raise ValueError('case has no eligible differentiable observation/U target')
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        report['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(objectives.parameters(), 5., error_if_nonfinite=True))
        optimizer.step()
        sync()
        report['optimizer_steps'] = 1
        library = objectives._libraries.get((case.case_id, 0), ())
        report['U_hypothesis_count'] = len(library)
        report['U_hypothesis_max_cardinality'] = max((len(value) for value in library), default=0)
        report['phases']['observation_U_step_seconds'] = time.perf_counter() - tick

        phase('checkpoint_replay')
        tick = time.perf_counter()
        objectives.eval()
        with torch.no_grad():
            before = objectives.model(case, seed=config.seed)
        save_checkpoint(destination / 'smoke.pt', model=objectives, optimizer=optimizer, identity=identity,
            progress=dict(stage='smoke', epoch=0, slot=1, encoder_version=1, stage_complete=True),
            query_sets={'L': [], 'U': [case.case_id]},
            extra_state={'valid_demo_checkpoint': False, 'sampling_audit': before.sampling_audit})
        restored = build_objectives(config, preprocessing, text_encoder=text_encoder, device=device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
        load_checkpoint(destination / 'smoke.pt', model=restored, optimizer=restored_optimizer,
                        expected_identity=identity, require_complete=True, restore_rng=False, map_location=device)
        restored.eval()
        with torch.no_grad():
            coupling = ({'sampling_audit': before.sampling_audit} if config.fragment_sampler == 'exact-direct-v1'
                        else {'sampling_marks': before.sampling_marks})
            after = restored.model(case, seed=config.seed, **coupling)
        sync()
        error = float((before.scores - after.scores).abs().max())
        if before.candidate_ids != after.candidate_ids or not torch.equal(before.scores, after.scores):
            raise ValueError(f'checkpoint inference differs: {error}')
        if before.sampling_audit != after.sampling_audit:
            raise ValueError('checkpoint sampling audit differs')
        report['checkpoint_sampling_audit_match'] = True
        report['checkpoint_reload_max_abs_error'] = error
        report['scores'] = dict(zip(after.candidate_ids, after.scores.detach().cpu().tolist()))
        report['phases']['checkpoint_replay_seconds'] = time.perf_counter() - tick
        report['complete'] = True
        report['peak_gpu_allocated_bytes'] = (torch.cuda.max_memory_allocated(device)
                                              if torch.device(device).type == 'cuda' else None)
        phase('complete')
        atomic_json(destination / 'smoke-report.json', report)
        return report
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), complete=False)
        phase('error')
        atomic_json(destination / 'smoke-report.json', report)
        raise
