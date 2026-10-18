"""Evaluation-only observation annotations; never model-forward inputs."""

from .validation import ValidationReport, derive_ordered_paths, validate_annotation

__all__ = ["ValidationReport", "derive_ordered_paths", "validate_annotation"]
