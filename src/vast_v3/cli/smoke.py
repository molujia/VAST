"""Run one real-case smoke entirely where the authorized dataset resides."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from ..data.manifests import build_standard_manifest
from ..data.repository import DatasetRepository, manifest_from_dict, manifest_to_dict
from ..data.serialization import strict_json_loads
from ..data.preprocessing import Preprocessor
from ..data.static_topology import build_static_topology
from ..encoders.text import FrozenSentenceEncoder
from ..runtime import RuntimeConfig, canonical_hash
from ..smoke import atomic_json, run_smoke


def select_smoke_case(split, requested):
    case_id = requested or sorted(split.train_case_ids)[0]
    if case_id not in split.train_case_ids:
        raise ValueError('smoke fitting must use a permitted training case, not a test case')
    return case_id


def prepare_observation_repository(manifest, raw_root, output, update):
    """Persist the approved AIOps structural prior before any case/model load."""
    options = {}
    if manifest.split.dataset == 'aiops2022_pre':
        update('freeze_static_topology')
        prior = build_static_topology(manifest, raw_root)
        atomic_json(output.with_name(output.name + '.static-topology.json'), prior.to_dict())
        options['static_topology'] = prior
    return DatasetRepository.from_curated(manifest, raw_root, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description='One training-case observable smoke; no root reads or benchmark scores.')
    parser.add_argument('--dataset', required=True, choices=('rcabench', 'aiops2022_pre'))
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--manifest', help='Public label-free manifest; otherwise verify the authorized frozen server sources.')
    parser.add_argument('--case-id', help='Default: lexicographically first frozen training ID, independent of outcomes.')
    parser.add_argument('--text-directory', required=True, help='Existing offline MiniLM snapshot directory; no model download.')
    parser.add_argument('--text-lock', required=True, help='Versioned model revision and per-file SHA-256 map.')
    parser.add_argument('--output', required=True, help='New attempt directory; existing attempts are not overwritten.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    output = Path(args.output)
    preflight = output.with_name(output.name + '.preflight.json')
    if output.exists() or preflight.exists():
        raise ValueError('attempt output or preflight exists; use a new attempt name')
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    status = {'dataset': args.dataset, 'complete': False, 'root_labels_accessed': 0,
              'purpose': 'label-free-smoke', 'phase': 'frozen_manifest'}

    def update(phase, **values):
        status.update(phase=phase, elapsed_seconds=time.perf_counter() - started, **values)
        atomic_json(preflight, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    try:
        update('frozen_manifest')
        manifest = (manifest_from_dict(strict_json_loads(Path(args.manifest).read_bytes())) if args.manifest
                    else build_standard_manifest(args.dataset))
        if manifest.split.dataset != args.dataset:
            raise ValueError('runtime and manifest datasets differ')
        atomic_json(output.with_name(output.name + '.manifest.json'), manifest_to_dict(manifest))
        case_id = select_smoke_case(manifest.split, args.case_id)
        repository = prepare_observation_repository(manifest, args.raw_root, output, update)
        update('load_native_telemetry', case_id=case_id)
        case = repository[case_id]
        audit = repository.case_audit(case_id)
        atomic_json(output.with_name(output.name + '.source-audit.json'), audit)
        update('load_offline_text', source_identity=audit['case_source_sha256'],
               entity_count=len(case.entities), numeric_series=len(case.numeric), log_records=len(case.logs))
        lock = strict_json_loads(Path(args.text_lock).read_bytes())
        atomic_json(output.with_name(output.name + '.text-lock.json'), lock)
        encoder = FrozenSentenceEncoder(args.text_directory, lock, device=args.device, staging_parent=output.parent)
        update('fit_smoke_reference_only')
        pre = Preprocessor.fit([case], permitted_case_ids={case_id},
                               text_encoder=encoder, text_identity=canonical_hash(lock))
        update('model_smoke', delta_seconds=pre.delta)
        report = run_smoke(case, pre, RuntimeConfig(dataset=args.dataset), text_encoder=encoder,
                           output_dir=output, data_hash=canonical_hash(audit), device=args.device)
        atomic_json(output / 'source-audit.json', audit)
        atomic_json(output / 'text-lock.json', lock)
        update('complete', complete=True, smoke_report=str(output / 'smoke-report.json'),
               smoke_seconds=report['elapsed_seconds'])
        return 0
    except Exception as exc:
        update('error', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
