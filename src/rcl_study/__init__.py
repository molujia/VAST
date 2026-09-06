"""Shared contracts for the canonical RCL experiment study."""

from .datasets import (
    DatasetResolution,
    DatasetSpec,
    canonical_dataset_ids,
    canonical_output_path,
    resolve_dataset,
    with_dataset_identity,
)
from .artifacts import (
    sha256_file,
    validate_anomaly_event_rows,
    validate_audit,
    validate_canonical_series_rows,
    validate_case_manifest_rows,
    validate_completed_bundle,
)
from .results import recompute_hit_at_k, validate_result_record
from .protocols import validate_case_budget, validate_normal_policy
from .rcabench import canonicalize_case as canonicalize_rcabench_case
from .rcabench import normalize_timestamps as normalize_rcabench_timestamps
from .rcabench_pipeline import canonicalize_dataset as canonicalize_rcabench_dataset
from .rcabench_pipeline import validate_completion as validate_rcabench_completion
from .aiops25 import build_case_manifest as build_aiops25_case_manifest
from .aiops25 import canonicalize_case as canonicalize_aiops25_case
from .legacy_events import build_legacy_events

__all__ = [
    "DatasetResolution",
    "DatasetSpec",
    "canonical_dataset_ids",
    "canonical_output_path",
    "canonicalize_aiops25_case",
    "canonicalize_rcabench_case",
    "canonicalize_rcabench_dataset",
    "normalize_rcabench_timestamps",
    "build_aiops25_case_manifest",
    "build_legacy_events",
    "resolve_dataset",
    "recompute_hit_at_k",
    "sha256_file",
    "validate_anomaly_event_rows",
    "validate_audit",
    "validate_case_budget",
    "validate_normal_policy",
    "validate_canonical_series_rows",
    "validate_case_manifest_rows",
    "validate_completed_bundle",
    "validate_result_record",
    "validate_rcabench_completion",
    "with_dataset_identity",
]
