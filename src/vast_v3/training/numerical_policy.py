"""Explicit, reversible execution policy; independent of the learning contract.

The backend fields deliberately match the frozen numeric verification probe.
Only the approved cuDNN TF32 flag is owned; no matmul/determinism/dtype/RNG
settings are changed. Initial/restored values are bookkeeping, not identity.
"""
from contextlib import contextmanager
from copy import deepcopy

import torch

from ..runtime import canonical_hash


CONVOLUTION_POLICIES = ('inherited-v1', 'fp32-no-cudnn-tf32-v1')
_IDENTITY_KEYS = ('schema', 'name', 'version', 'effective_backend')


def backend_state():
    return {'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_fp16_reduced_precision': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'matmul_bf16_reduced_precision': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


def policy_artifact(record):
    """Frozen effective identity, invariant to the caller's initial TF32 flag."""
    payload = {key: deepcopy(record[key]) for key in _IDENTITY_KEYS}
    return {**payload, 'identity_sha256': canonical_hash(payload)}


def assert_policy_unchanged(record):
    if (backend_state() != record['effective_backend']
            or policy_artifact(record)['identity_sha256'] != record['identity_sha256']):
        raise ValueError('execution policy effective backend identity drift')


def _record_detected_drift(record, on_drift):
    """Capture a detected violation before restoring this context's owned flag."""
    try:
        assert_policy_unchanged(record)
    except ValueError:
        record['drift_detected'] = True
        record['drifted_backend'] = backend_state()
        if on_drift is not None:
            on_drift(record)
        return True
    return False


@contextmanager
def convolution_policy_context(name='inherited-v1', *, on_drift=None):
    """Report detected drift on either exit path, without masking body exceptions.

    A callback can durably invalidate a training attempt before flag restoration.
    Checks at these boundaries do not detect arbitrary transient flag changes.
    """
    if not isinstance(name, str) or name not in CONVOLUTION_POLICIES:
        raise ValueError('unknown convolution policy: ' + str(name))
    initial = backend_state()
    record = {'schema': 'vast-v3-convolution-policy/1', 'name': name, 'version': 1,
              'initial_backend': initial, 'restored_backend': None, 'drift_detected': False}
    try:
        if name == 'fp32-no-cudnn-tf32-v1':
            torch.backends.cudnn.allow_tf32 = False
        record['effective_backend'] = backend_state()
        record['identity_sha256'] = policy_artifact(record)['identity_sha256']
        try:
            yield record
        except BaseException:
            _record_detected_drift(record, on_drift)
            raise
        else:
            if _record_detected_drift(record, on_drift):
                raise ValueError('execution policy effective backend identity drift')
    finally:
        if name == 'fp32-no-cudnn-tf32-v1':
            torch.backends.cudnn.allow_tf32 = initial['cudnn_allow_tf32']
        record['restored_backend'] = backend_state()
