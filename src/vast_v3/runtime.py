"""Small explicit runtime contract for the approved R03 implementation.

This is distinct from the historical design JSON: actual parameters and input
identities are bound here; no missing authority hash is silently filled in.

Checkpoint/run identities must include this complete configuration's identity
and package_code_hash(), including for callers that construct Trainer directly.
The sampler is part of that contract; an arbitrary caller-supplied config hash
does not make a standalone Trainer safe to resume across sampler versions.
Acquisition caches are rebuilt per round and are not shared across these runs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .data.preprocessing import Preprocessor
from .encoders.frontends import NativeFrontend
from .models.ranker import VASTModel
from .training.objectives import TrainingObjectives
from .training.schedule import TrainingSchedule
from .training.variants import variant_components


R03_PDF_SHA256 = 'df7f2e642f1679d10f7514d39d84aca213f4bfc581168912c77c0baa32d20127'


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RuntimeConfig:
    dataset: str
    seed: int = 42
    variant: str = 'A3'
    regime: str = 'closed_loop'
    budget: int = 30
    query_batch_size: int = 5
    warmup_epochs: int = 20
    nonfinal_epochs: int = 20
    final_epochs: int = 50
    u_disabled_epochs: int = 5
    u_ramp_epochs: int = 5
    case_batch_size: int = 4
    observation_weight: float = 0.2
    source_weight: float = 0.2
    u_weight: float = 0.1
    log_recovery_radius_steps: float = 2.0
    max_roots: int | None = None
    fragment_sampler: str = 'exact-direct-v1'
    max_enumerated: int = 100000  # Reference enumerate-v1 guard, unused by direct sampling.
    # Keep the verified scalar route until real-data batching acceptance passes.
    numeric_execution: str = 'scalar-v1'
    numeric_batch_size: int = 32
    numeric_max_batch_samples: int = 32768
    schema: str = 'vast-v3-runtime/3'
    authority_pdf_sha256: str = R03_PDF_SHA256
    supplement: str = 'approved-log-context-20261016'

    def __post_init__(self):
        if self.schema != 'vast-v3-runtime/3' or self.authority_pdf_sha256 != R03_PDF_SHA256:
            raise ValueError('unsupported runtime schema or methodology authority')
        if self.fragment_sampler not in ('exact-direct-v1', 'enumerate-v1'):
            raise ValueError('unsupported fragment sampler')
        if self.numeric_execution not in ('scalar-v1', 'length-bucket-v1', 'metric-scalar-trace-bucket-v1'):
            raise ValueError('unsupported numeric execution mode')
        for name in ('numeric_batch_size', 'numeric_max_batch_samples'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.supplement != 'approved-log-context-20261016':
            raise ValueError('unknown implementation supplement')
        if self.dataset not in ('rcabench', 'aiops2022_pre'):
            raise ValueError('unsupported dataset')
        variant_components(self.variant)
        if self.regime not in ('closed_loop', 'fixed_list', 'all_label', 'lofo'):
            raise ValueError('unsupported training regime')
        for name, minimum in (('seed', 0), ('max_enumerated', 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.max_roots is not None and (type(self.max_roots) is not int or self.max_roots < 1):
            raise ValueError('max_roots must be a positive integer or null')
        if self.dataset == 'rcabench' and self.max_roots not in (None, 2):
            raise ValueError('RCABench requires the approved maximum-two-root prior')
        radius = self.log_recovery_radius_steps
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
            raise ValueError('log recovery radius must be finite and nonnegative')
        self.schedule()  # One validator owns the schedule contract.

    @property
    def effective_max_roots(self):
        return 2 if self.dataset == 'rcabench' else self.max_roots

    def schedule(self):
        return TrainingSchedule(**{field.name: getattr(self, field.name) for field in fields(TrainingSchedule)})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError('runtime configuration fields must match the versioned schema exactly')
        return cls(**data)

    @property
    def identity(self):
        return canonical_hash(self.to_dict())


def package_code_hash():
    """Hash installed package source names and bytes, never user data/caches."""
    root = Path(__file__).parent
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def build_objectives(config: RuntimeConfig, preprocessing: Preprocessor, *, text_encoder, device='cpu'):
    """Construct the shared model; callers must supply pinned frozen text assets.

    Tests may inject declared synthetic text encoders. Production CLIs load the
    offline Sentence Transformer before calling this factory and record its hash.
    Vocabulary uses only the already-fitted permitted reference statistics.
    """
    if not math.isfinite(preprocessing.delta) or preprocessing.delta <= 0:
        raise ValueError('preprocessing grid must be finite and positive')
    channels = tuple(sorted({channel for _, channel in preprocessing.stats if channel != '__log_frequency__'}))
    if not channels:
        raise ValueError('no supported reference numeric channel vocabulary')
    random.seed(config.seed)
    np.random.seed(config.seed % (2 ** 32))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    frontend = NativeFrontend(preprocessing, ('service', 'host'), ('call', 'response', 'hosted_on'),
        channels, text_encoder=text_encoder, log_recovery_radius_steps=config.log_recovery_radius_steps,
        numeric_execution=config.numeric_execution, numeric_batch_size=config.numeric_batch_size,
        numeric_max_batch_samples=config.numeric_max_batch_samples)
    model = VASTModel(frontend, fragment_sampler=config.fragment_sampler,
                      max_enumerated=config.max_enumerated)
    return TrainingObjectives(model, variant=config.variant, max_roots=config.effective_max_roots,
                              seed=config.seed).to(device)
