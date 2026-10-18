"""Explicit independent curator preparation; never training or observation IO.

Only the approved standard manifest and existing raw-annotation curator define
the population/mapping. The sealed file and private error audit are curator-only
artifacts. stdout is a counts/hash/path summary, never a dump of root answers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from ..data.curation import CuratedLabels, curate_raw_labels
from ..data.manifests import build_standard_manifest
from ..data.repository import file_sha256, manifest_to_dict


def atomic_new_json(path, payload):
    """Publish a complete private JSON inode without replacing any target.

    POSIX hard-link creation is atomic and fails if the target already exists.
    Windows rename also refuses an existing target (unlike POSIX rename).
    Neither path exposes a partially written file or replaces prior evidence.
    """
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == 'nt':
            os.rename(temporary, path)  # Windows always rejects an existing destination.
        else:
            os.link(temporary, path)
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_sealed_labels(*, dataset, raw_root, output_dir):
    """Create one new audited attempt, keeping the frozen windows/split intact."""
    raw = Path(raw_root).resolve()
    output = Path(output_dir).resolve()
    if not raw.is_dir():
        raise ValueError('raw root must be an existing directory')
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError('curator output and raw root must be separate, non-nested directories')
    # Reserve exclusively before reading any frozen metadata or raw annotation.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        manifest = build_standard_manifest(dataset)
        if manifest.split.dataset != dataset:
            raise ValueError('standard manifest dataset mismatch')
        public = manifest_to_dict(manifest)  # Intentionally strips curator audit.
        curated = curate_raw_labels(manifest, raw)
        sealed = curated.to_dict()
        # Exercise the same strict sealed-artifact validation used by cli.train.
        CuratedLabels.from_dict(sealed, manifest)
        public_path = atomic_new_json(output / 'public-manifest.json', public)
        sealed_path = atomic_new_json(output / 'sealed-labels.json', sealed)
        summary = {'schema': 'vast-v3-curator-preparation/1', 'complete': True,
            'purpose': 'independent-curator-only-not-training', 'dataset': dataset,
            'counts': {'cases': len(manifest.windows), 'train': len(manifest.split.train_case_ids),
                       'test': len(manifest.split.test_case_ids), 'candidates': len(manifest.catalog.candidate_ids)},
            'public_manifest_identity_sha256': sealed['manifest_sha256'],
            'sealed_identity_sha256': curated.identity_sha256,
            'artifacts': {'public_manifest': {'path': str(public_path), 'sha256': file_sha256(public_path)},
                          'sealed_labels': {'path': str(sealed_path), 'sha256': file_sha256(sealed_path)}}}
        # The last atomic publication is the only completed-attempt marker.
        atomic_new_json(output / 'curator-summary.json', summary)
        return summary
    except Exception as exc:
        # Detailed mapping issues can contain root IDs: keep them private here,
        # not in stdout, model features, an observable cache, or search output.
        error = {'schema': 'vast-v3-curator-preparation-error/1', 'complete': False,
                 'purpose': 'independent-curator-only-not-training', 'error_type': type(exc).__name__,
                 'error': str(exc)}
        try:
            atomic_new_json(output / 'curator-error.json', error)
        except OSError:
            pass  # Preserve the original failure; never replace prior evidence.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare a new sealed curator artifact for the approved standard pool; no training or label queries.')
    parser.add_argument('--dataset', required=True, choices=('rcabench', 'aiops2022_pre'))
    parser.add_argument('--raw-root', required=True, help='Existing authorized dataset directory containing original annotations.')
    parser.add_argument('--output-dir', required=True, help='New private curator attempt directory; existing attempts are never overwritten.')
    args = parser.parse_args(argv)
    try:
        summary = prepare_sealed_labels(dataset=args.dataset, raw_root=args.raw_root, output_dir=args.output_dir)
    except Exception as exc:
        print(json.dumps({'schema': 'vast-v3-curator-preparation/1', 'complete': False,
                          'dataset': args.dataset, 'error_type': type(exc).__name__,
                          'output_dir': str(Path(args.output_dir).resolve())}, sort_keys=True), flush=True)
        return 1
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
