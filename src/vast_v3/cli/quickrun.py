"""Explicit cached-subset B30 quickstart. Does not confer production eligibility."""
from __future__ import annotations

from pathlib import Path

import torch

from . import train
from ..data.curation import CuratedLabels
from ..data.manifests import build_standard_manifest
from ..data.repository import DatasetRepository, file_sha256, manifest_from_dict, manifest_to_dict
from ..data.static_topology import StaticTopology
from ..encoders.text import FrozenSentenceEncoder
from ..runtime import canonical_hash, package_code_hash
from ..training.numerical_policy import convolution_policy_context
from ..training.quickrun import (QUICKRUN_PURPOSE, CachedTrainingCases, QuickrunProgress,
    ReadOnlyCaseCache, create_snapshot, quickrun_runtime_config, validate_snapshot,
    validate_training_subset)
from ..training.run_control import freeze_json, read_json


def _parser():
    parser = train._parser()
    parser.description = 'Freeze or train the already-cached original train subset; never production/demo eligible.'
    parser.add_argument('command', choices=('snapshot', 'run'))
    parser.add_argument('--snapshot', required=True, help='Frozen cache inventory outside raw/cache/run directories.')
    for action in parser._actions:
        if action.dest == 'source_pins':
            action.required = True
            action.help = 'Required original complete source pins; no regenerated or partial identity.'
        elif action.dest == 'static_topology':
            action.help = 'Required original static topology for AIOps; never rebuild a different prior.'
    return parser


def _inputs(args):
    manifest = manifest_from_dict(read_json(args.manifest))
    standard = build_standard_manifest(args.dataset)
    if manifest_to_dict(manifest) != manifest_to_dict(standard):
        raise ValueError('public manifest differs from the complete approved frozen standard population')
    curated = CuratedLabels.from_dict(read_json(args.sealed_labels), manifest)
    prior = None
    if args.dataset == 'aiops2022_pre':
        if not args.static_topology:
            raise ValueError('cached AIOps quickrun requires its original static topology')
        prior = StaticTopology.from_dict(read_json(args.static_topology))
        prior.validate_manifest(manifest)
        if set(prior.to_dict()['permitted_case_ids']) != set(manifest.split.train_case_ids):
            raise ValueError('static topology must bind the full permitted training pool')
    elif args.static_topology:
        raise ValueError('RCABench does not use the AIOps static prior')
    pins = read_json(args.source_pins)
    cache = ReadOnlyCaseCache(args.case_cache_root, max_bytes=args.case_cache_max_bytes)
    repository = DatasetRepository.from_curated(manifest, args.raw_root, expected_source_sha256=pins,
                                                static_topology=prior, case_cache=cache)
    # Same complete source verification as the standard entrypoint. Cache-only
    # means no telemetry adapter fallback, not permission to skip provenance.
    repository.pin_source_content()
    return repository, curated, read_json(args.text_lock), pins, standard


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.case_cache_max_bytes < 1 or (args.max_updates is not None and args.max_updates < 1):
        parser.error('threads, cache quota and diagnostic max-updates must be positive')
    if args.command == 'snapshot' and (args.resume or args.max_updates is not None):
        parser.error('snapshot does not accept resume or max-updates')
    roots = [Path(value).resolve() for value in (args.raw_root, args.case_cache_root, args.run_dir)]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError('raw, observation cache and run directories must be separate, non-nested paths')
    snapshot_path = Path(args.snapshot).resolve()
    if any(snapshot_path.is_relative_to(root) for root in roots):
        raise ValueError('snapshot must be outside raw, cache, and versioned run directories')
    if args.command == 'snapshot':
        repository, _, _, _, _ = _inputs(args)
        if snapshot_path.exists():
            validate_snapshot(read_json(snapshot_path), repository)
        else:
            snapshot = create_snapshot(repository)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            freeze_json(snapshot_path, snapshot)
        return 0
    reporter = QuickrunProgress(args.run_dir, resume=args.resume)
    with convolution_policy_context(train.DEMO_CONVOLUTION_POLICY,
            on_drift=lambda policy: train._invalidate_execution_policy(reporter.directory, policy)) as policy:
        try:
            output = reporter.directory
            train._freeze_execution_policy(output, policy)
            reporter.update('validate_original_inputs')
            repository, curated, lock, pins, standard = _inputs(args)
            snapshot = read_json(snapshot_path)
            cases = CachedTrainingCases(repository, snapshot)
            config = quickrun_runtime_config(args.dataset)
            invocation = {'schema': 'vast-v3-quickrun-invocation/1', 'purpose': QUICKRUN_PURPOSE,
                **train._policy_identity(policy), **validate_training_subset(repository, config, cases, snapshot),
                'dataset': args.dataset, 'standard_pool_hash': canonical_hash(manifest_to_dict(standard)),
                'manifest_hash': canonical_hash(manifest_to_dict(repository.manifest)),
                'data_hash': repository.identity_sha256, 'config_hash': config.identity,
                'code_hash': package_code_hash(), 'text_lock_hash': canonical_hash(lock),
                'curator_hash': curated.identity_sha256,
                'static_topology_hash': (repository.static_topology.identity_sha256
                    if repository.static_topology is not None else canonical_hash(None)),
                'raw_root': str(roots[0]), 'sealed_labels_sha256': file_sha256(args.sealed_labels),
                'source_pins_hash': canonical_hash(pins),
                'prior_input_hash': file_sha256(args.static_topology) if args.static_topology else None,
                'snapshot_sha256': file_sha256(snapshot_path), 'device': args.device, 'threads': args.threads}
            freeze_json(output / 'invocation.json', invocation)
            freeze_json(output / 'cached-subset.json', snapshot)
            freeze_json(output / 'source-pins.json', pins)
            if repository.static_topology is not None:
                freeze_json(output / 'static-topology.json', repository.static_topology.to_dict())
            reporter.update('load_offline_text', pool_size=len(cases))
            torch.set_num_threads(args.threads)
            encoder = FrozenSentenceEncoder(args.text_directory, lock, device=args.device, staging_parent=output)
            train._check_run_policy(output, policy)
            train.run_training(repository=repository, curated_labels=curated, config=config,
                text_encoder=encoder, text_lock=lock, run_dir=output, resume=args.resume,
                max_updates=args.max_updates, device=args.device, code_hash=invocation['code_hash'],
                reporter=reporter, convolution_policy=policy['name'], training_subset=cases,
                quickrun_provenance=snapshot)
            return 0
        except Exception as exc:
            reporter.update('error', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
