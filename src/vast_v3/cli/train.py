"""One approved seed42/A3/B30 model per dataset; never smoke or demo search."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import math
from pathlib import Path

import torch

from ..data.case_cache import ObservableCaseCache
from ..data.curation import CuratedLabels
from ..data.manifests import build_standard_manifest
from ..data.preprocessing import Preprocessor, SCHEMA, SCALING_POLICY
from ..data.repository import DatasetRepository, manifest_from_dict, manifest_to_dict, file_sha256
from ..data.static_topology import StaticTopology, build_static_topology
from ..data.subsets import TrainingCases
from ..encoders.text import FrozenSentenceEncoder
from ..runtime import RuntimeConfig, build_objectives, canonical_hash, package_code_hash
from ..training.numerical_policy import (assert_policy_unchanged, backend_state, convolution_policy_context,
                                        policy_artifact)
from ..training.run_control import (RUN_PURPOSE, RunProgress, ProgressTrainer, atomic_json,
    freeze_json, inspect_final_training_checkpoint, read_json)


DEMO_CONVOLUTION_POLICY = 'fp32-no-cudnn-tf32-v1'


def demo_runtime_config(dataset):
    """Approved production execution profile; generic RuntimeConfig stays scalar."""
    return RuntimeConfig(dataset=dataset, numeric_execution='metric-scalar-trace-bucket-v1',
                         numeric_batch_size=32, numeric_max_batch_samples=32768)


def _policy_identity(policy):
    return {'convolution_policy': policy['name'], 'execution_policy_hash': policy['identity_sha256'],
            'effective_backend_sha256': canonical_hash(policy['effective_backend'])}


def _invalidate_execution_policy(output, policy):
    """A detected violation permanently excludes this attempt from resume."""
    output = Path(output)
    marker = output / 'policy-invalid.json'
    if output.is_dir() and not marker.exists():
        atomic_json(marker, {'schema': 'vast-v3-invalid-execution-policy/1',
            'reason': 'detected-execution-policy-drift', 'resume_allowed': False,
            'policy': policy_artifact(policy),
            'observed_backend': policy.get('drifted_backend', backend_state())})


def _check_run_policy(output, policy):
    try:
        assert_policy_unchanged(policy)
    except ValueError:
        _invalidate_execution_policy(output, policy)
        raise


def _freeze_execution_policy(output, policy):
    """Reject policy drift before loading observations or opening the curator.

    The normal Trainer restore still validates the full checkpoint identity and
    contents. This read-only preflight checks the execution binding earlier, before
    even constructing a paid label store or loading offline text assets.
    """
    if (output / 'policy-invalid.json').exists():
        raise ValueError('execution policy invalidated this run; resume is forbidden')
    path, checkpoint = output / 'execution-policy.json', output / 'checkpoints/latest.pt'
    if checkpoint.exists() and not path.exists():
        raise ValueError('checkpoint requires its original frozen execution policy')
    artifact = policy_artifact(policy)
    # JSON types are identity-bearing: Python equality alone accepts False == 0
    # and True == 1, even though their canonical policy digests differ.
    if path.exists() and canonical_hash(read_json(path)) != canonical_hash(artifact):
        raise ValueError('frozen execution policy canonical identity mismatch')
    freeze_json(path, artifact)
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        identity = saved.get('identity', {})
        if any(identity.get(key) != value for key, value in _policy_identity(policy).items()):
            raise ValueError('checkpoint execution policy identity mismatch')


class _FitProgress(Mapping):
    def __init__(self, cases, reporter):
        self.cases, self.reporter, self.visits = cases, reporter, 0

    def __len__(self):
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, cid):
        self.reporter.update('fit_reference', case_id=cid, pass_index=self.visits // len(self.cases) + 1,
                             pass_completed=self.visits % len(self.cases), pool_size=len(self.cases))
        case = self.cases[cid]
        self.visits += 1
        return case


def _validate_preprocessing(pre, train_ids, text_identity):
    if (pre.schema != SCHEMA or pre.scaling_policy != SCALING_POLICY
            or pre.fitted_case_ids != tuple(sorted(train_ids)) or pre.text_model_identity != text_identity
            or not math.isfinite(pre.delta) or pre.delta <= 0 or pre.gap_series_count <= 0
            or not pre.stats or any(not math.isfinite(s.center) or not math.isfinite(s.scale)
                or s.scale <= 0 or s.observations < 1 for s in pre.stats.values())
            or any(not math.isfinite(x) for x in pre.text_reference_mean)):
        raise ValueError('preprocessing identity/scope/statistics do not match the full training reference')


def run_training(*, repository, curated_labels, config, text_encoder, text_lock, run_dir,
                 resume=False, max_updates=None, device='cpu', code_hash=None, reporter=None,
                 convolution_policy='inherited-v1', training_subset=None, quickrun_provenance=None):
    """Compose verified inputs and the existing loop; no evaluation-store access.

    Library callers may provide a declared synthetic configuration for tests.
    Only the CLI verifies the standard population and uses the reference config.
    A shortened configuration can finish its own schedule, never become a B30 model.
    """
    with convolution_policy_context(convolution_policy,
            on_drift=lambda policy: _invalidate_execution_policy(run_dir, policy)) as policy:
        return _run_training(repository=repository, curated_labels=curated_labels, config=config,
            text_encoder=text_encoder, text_lock=text_lock, run_dir=run_dir, resume=resume,
            max_updates=max_updates, device=device, code_hash=code_hash, reporter=reporter,
            policy=policy, training_subset=training_subset, quickrun_provenance=quickrun_provenance)


def _run_training(*, repository, curated_labels, config, text_encoder, text_lock, run_dir,
                  resume, max_updates, device, code_hash, reporter, policy,
                  training_subset=None, quickrun_provenance=None):
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError('max_updates must be a positive diagnostic limit')
    if config.seed != 42 or config.variant != 'A3' or config.regime != 'closed_loop':
        raise ValueError('demo training requires seed42 A3 closed_loop')
    quickrun = training_subset is not None
    if quickrun != (quickrun_provenance is not None):
        raise ValueError('explicit training subset and quickrun provenance must be supplied together')
    subset_identity = {}
    if quickrun:
        from ..training.quickrun import (QUICKRUN_PURPOSE, QuickrunProgress, QuickrunTrainer,
            validate_training_subset, validate_quickrun_invocation, inspect_quickrun_u_evidence)
        subset_identity = validate_training_subset(repository, config, training_subset, quickrun_provenance)
    reporter = reporter or (QuickrunProgress if quickrun else RunProgress)(run_dir, resume=resume)
    output = Path(run_dir)
    try:
        _freeze_execution_policy(output, policy)
        reporter.update('execution_policy', numeric_execution=config.numeric_execution, **_policy_identity(policy))
        if repository.identity_kind != 'content-pinned':
            raise ValueError('training requires content-pinned source identity before case caching')
        if repository.manifest.split.dataset != config.dataset:
            raise ValueError('configuration and manifest dataset identity mismatch')
        cases = training_subset if quickrun else TrainingCases(repository)
        train_ids = tuple(sorted(cases))
        public = manifest_to_dict(repository.manifest)
        text_identity = canonical_hash(text_lock)
        prior = repository.static_topology
        identity = {'purpose': RUN_PURPOSE, **_policy_identity(policy), 'data_hash': repository.identity_sha256,
            'manifest_hash': canonical_hash(public), 'catalog_hash': canonical_hash(asdict(repository.manifest.catalog)),
            'config_hash': config.identity, 'code_hash': code_hash or package_code_hash(),
            'text_lock_hash': text_identity, 'curator_hash': curated_labels.identity_sha256,
            'static_topology_hash': prior.identity_sha256 if prior is not None else canonical_hash(None),
            'authority_hash': config.authority_pdf_sha256, **subset_identity}
        production_verified = False
        quickrun_verified = False
        if (output / 'invocation.json').exists():
            invocation = read_json(output / 'invocation.json')
            if quickrun:
                validate_quickrun_invocation(invocation, identity)
                quickrun_verified = True
            elif (invocation.get('schema') != 'vast-v3-training-invocation/1'
                    or invocation.get('purpose') != RUN_PURPOSE
                    or invocation.get('dataset') != config.dataset
                    or invocation.get('standard_pool_hash') != identity['manifest_hash']
                    or any(invocation.get(key) != identity[key] for key in
                           ('manifest_hash', 'config_hash', 'code_hash', 'text_lock_hash',
                            'convolution_policy', 'execution_policy_hash', 'effective_backend_sha256'))):
                raise ValueError('production invocation input identity mismatch')
            else:
                production_verified = True
        inputs = {'schema': 'vast-v3-training-inputs/1', 'identity': identity,
                  'train_case_ids': list(train_ids), 'config': config.to_dict()}
        freeze_json(output / 'training-inputs.json', inputs)
        freeze_json(output / 'manifest.json', public)
        freeze_json(output / 'text-lock.json', text_lock)
        freeze_json(output / 'runtime-config.json', config.to_dict())
        checkpoint = output / 'checkpoints' / 'latest.pt'
        pre_path = output / 'preprocessing.json'
        if pre_path.exists():
            artifact = read_json(pre_path)
            if artifact.get('schema') != 'vast-v3-frozen-training-preprocessing/1' or artifact['inputs'] != inputs:
                raise ValueError('frozen preprocessing input identity mismatch')
            pre = Preprocessor.from_dict(artifact['preprocessing'])
            if artifact['preprocessing_hash'] != pre.identity:
                raise ValueError('frozen preprocessing identity checksum mismatch')
            _validate_preprocessing(pre, train_ids, text_identity)
            reporter.update('reuse_frozen_preprocessing', preprocessing_hash=pre.identity)
        else:
            if checkpoint.exists():
                raise ValueError('checkpoint requires its original frozen preprocessing; refit is forbidden')
            for index, cid in enumerate(train_ids):
                reporter.update('prepare_cache', case_id=cid, completed_cases=index, pool_size=len(cases))
                cases[cid]
            reporter.update('fit_reference', pool_size=len(cases))
            pre = Preprocessor.fit_repository(_FitProgress(cases, reporter), ordered_case_ids=train_ids,
                permitted_case_ids=set(train_ids), text_encoder=text_encoder, text_identity=text_identity)
            _validate_preprocessing(pre, train_ids, text_identity)
            atomic_json(pre_path, {'schema': 'vast-v3-frozen-training-preprocessing/1', 'inputs': inputs,
                                  'preprocessing': pre.to_dict(), 'preprocessing_hash': pre.identity})
        identity['preprocessing_hash'] = pre.identity
        _check_run_policy(output, policy)
        reporter.update('initialize_training', pool_size=len(cases), preprocessing_hash=pre.identity)
        labels = curated_labels.query_store(budget=config.budget)
        objectives = build_objectives(config, pre, text_encoder=text_encoder, device=device)
        _check_run_policy(output, policy)
        trainer = (QuickrunTrainer if quickrun else ProgressTrainer)(objectives, cases, labels, config.schedule(),
            checkpoint_dir=checkpoint.parent, identity=identity, seed=config.seed, reporter=reporter,
            policy_guard=lambda: _check_run_policy(output, policy))
        result = trainer.run(regime=config.regime, resume=resume and checkpoint.exists(), max_updates=max_updates)
        _check_run_policy(output, policy)
        if not result['complete']:
            return reporter.update('diagnostic_pause', updates=result['updates'], labels_spent=labels.spent,
                                   eligible_for_demo=False, resume_required=True)
        reporter.update('verify_final_schedule', updates=result['updates'], labels_spent=labels.spent)
        verified = inspect_final_training_checkpoint(checkpoint, expected_identity=trainer.identity,
            config=config, permitted_case_ids=train_ids, expected_label_state=labels.state_dict())
        reference = config == demo_runtime_config(config.dataset)
        # This is a schedule gate; production population provenance is separately
        # established by main() and bound in invocation.json before any data load.
        eligible = not quickrun and reference and production_verified and policy['name'] == DEMO_CONVOLUTION_POLICY
        quickstart = {}
        if quickrun:
            evidence = inspect_quickrun_u_evidence(checkpoint, config=config, permitted_case_ids=train_ids)
            ready = quickrun_verified and policy['name'] == DEMO_CONVOLUTION_POLICY and evidence['u_training_verified']
            quickstart = {**evidence, 'formal_experiment': False,
                          'eligible_for_quickstart': ready, 'quickstart_ready': ready}
        final = {'schema': 'vast-v3-frozen-training-model/1', 'purpose': QUICKRUN_PURPOSE if quickrun else RUN_PURPOSE,
                 'identity': trainer.identity, 'config': config.to_dict(), **verified,
                 'eligible_for_demo': eligible, 'preprocessing_sha256': file_sha256(pre_path), **quickstart}
        _check_run_policy(output, policy)
        freeze_json(output / 'final-model.json', final)
        return reporter.update('complete', complete=True, **verified, eligible_for_demo=eligible, **quickstart)
    except Exception as exc:
        reporter.update('error', error_type=type(exc).__name__, error=str(exc))
        raise


def _parser():
    parser = argparse.ArgumentParser(description='One seed42 A3 closed-loop B30 demo-model preparation; no search.')
    parser.add_argument('--dataset', required=True, choices=('rcabench', 'aiops2022_pre'))
    parser.add_argument('--manifest', required=True, help='Public manifest matching the approved frozen standard sources.')
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--sealed-labels', required=True, help='Existing independent curator artifact; no guessed path or raw-label fallback.')
    parser.add_argument('--source-pins', help='Complete absolute-source-path to SHA-256 JSON map; otherwise pin source bytes first.')
    parser.add_argument('--static-topology', help='Existing approved AIOps prior; otherwise freeze the permitted earliest reference.')
    parser.add_argument('--case-cache-root', required=True, help='Explicit server-local observation cache, separate from raw/run directories.')
    parser.add_argument('--case-cache-max-bytes', required=True, type=int)
    parser.add_argument('--text-directory', required=True, help='Existing offline MiniLM snapshot; never downloads.')
    parser.add_argument('--text-lock', required=True)
    parser.add_argument('--run-dir', required=True, help='New versioned run directory; never reused without --resume.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int, help='Diagnostic pause after this many additional updates, never completion.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.case_cache_max_bytes < 1 or (args.max_updates is not None and args.max_updates < 1):
        parser.error('threads, cache quota and diagnostic max-updates must be positive')
    roots = [Path(value).resolve() for value in (args.raw_root, args.case_cache_root, args.run_dir)]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError('raw, observation cache and run directories must be separate, non-nested paths')
    config = demo_runtime_config(args.dataset)
    reporter = RunProgress(args.run_dir, resume=args.resume)
    with convolution_policy_context(DEMO_CONVOLUTION_POLICY,
            on_drift=lambda policy: _invalidate_execution_policy(reporter.directory, policy)) as policy:
        return _main_under_policy(args, roots=roots, config=config, reporter=reporter, policy=policy)


def _main_under_policy(args, *, roots, config, reporter, policy):
    output = reporter.directory
    try:
        _freeze_execution_policy(output, policy)
        reporter.update('execution_policy', numeric_execution=config.numeric_execution, **_policy_identity(policy))
        reporter.update('verify_frozen_manifest')
        manifest = manifest_from_dict(read_json(args.manifest))
        standard = build_standard_manifest(args.dataset)
        if manifest_to_dict(manifest) != manifest_to_dict(standard):
            raise ValueError('public manifest differs from the complete approved frozen standard population')
        lock = read_json(args.text_lock)
        invocation = {'schema': 'vast-v3-training-invocation/1', 'purpose': RUN_PURPOSE, **_policy_identity(policy),
            'dataset': args.dataset, 'standard_pool_hash': canonical_hash(manifest_to_dict(standard)),
            'manifest_hash': canonical_hash(manifest_to_dict(manifest)), 'config_hash': config.identity,
            'raw_root': str(roots[0]), 'code_hash': package_code_hash(), 'text_lock_hash': canonical_hash(lock),
            'sealed_labels_sha256': file_sha256(args.sealed_labels),
            'source_pins_hash': canonical_hash(read_json(args.source_pins)) if args.source_pins else None,
            'prior_input_hash': file_sha256(args.static_topology) if args.static_topology else None,
            'device': args.device, 'threads': args.threads}
        freeze_json(output / 'invocation.json', invocation)
        reporter.update('validate_sealed_curator')
        curated = CuratedLabels.from_dict(read_json(args.sealed_labels), manifest)
        prior_path = output / 'static-topology.json'
        prior = None
        if args.dataset == 'aiops2022_pre':
            reporter.update('prepare_static_topology')
            prior = (StaticTopology.from_dict(read_json(args.static_topology)) if args.static_topology else
                     StaticTopology.from_dict(read_json(prior_path)) if prior_path.exists() else
                     build_static_topology(manifest, args.raw_root))
            prior.validate_manifest(manifest)
            if set(prior.to_dict()['permitted_case_ids']) != set(manifest.split.train_case_ids):
                raise ValueError('static topology must bind the full permitted training pool')
            freeze_json(prior_path, prior.to_dict())
        elif args.static_topology:
            raise ValueError('RCABench does not use the AIOps static prior')
        pins_path = output / 'source-pins.json'
        pins = read_json(args.source_pins) if args.source_pins else read_json(pins_path) if pins_path.exists() else None
        cache = ObservableCaseCache(args.case_cache_root, max_bytes=args.case_cache_max_bytes)
        repository = DatasetRepository.from_curated(manifest, args.raw_root, expected_source_sha256=pins,
                                                   static_topology=prior, case_cache=cache)
        reporter.update('pin_source_content', source_files=len(repository.source_files))
        freeze_json(pins_path, repository.pin_source_content())
        reporter.update('load_offline_text')
        torch.set_num_threads(args.threads)
        encoder = FrozenSentenceEncoder(args.text_directory, lock, device=args.device, staging_parent=output)
        _check_run_policy(output, policy)
        run_training(repository=repository, curated_labels=curated, config=config, text_encoder=encoder,
            text_lock=lock, run_dir=output, resume=args.resume, max_updates=args.max_updates,
            device=args.device, code_hash=invocation['code_hash'], reporter=reporter,
            convolution_policy=policy['name'])
        return 0
    except Exception as exc:
        reporter.update('error', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
