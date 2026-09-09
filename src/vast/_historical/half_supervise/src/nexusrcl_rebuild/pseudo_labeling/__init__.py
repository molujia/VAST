"""Leakage-safe pseudo-label generation and evaluation primitives."""

from .contracts import (
    AnnotationAnswer,
    AnnotationQuery,
    EvidenceRecord,
    FrozenPseudoLabelPrediction,
    PseudoAuditSummary,
    SanitizedEntityFeatureView,
    SanitizedFeatureView,
    StrategyResult,
)
from .calibration import (
    CalibratedConfidence,
    CalibrationObservation,
    CalibrationTargetReport,
    DualCalibrationReport,
    DualOutcomeCalibrator,
    ReliabilityBin,
    ScoredLabelPrediction,
    build_leave_one_out_observations,
    evaluate_dual_calibration,
)
from .oracles import AnnotationOracle, PseudoQualityOracle
from .quality import (
    PseudoQualityReport,
    PseudoQualitySummary,
    audit_pseudo_label_quality,
)
from .query import SharedQueryPlan, build_shared_query_plan
from .sanitizer import is_label_bearing_field, sanitize_feature_views
from .supervision import validate_supervision_request

__all__ = [
    "AnnotationAnswer",
    "AnnotationOracle",
    "AnnotationQuery",
    "CalibratedConfidence",
    "CalibrationObservation",
    "CalibrationTargetReport",
    "DualCalibrationReport",
    "DualOutcomeCalibrator",
    "EvidenceRecord",
    "FrozenPseudoLabelPrediction",
    "PseudoAuditSummary",
    "PseudoQualityOracle",
    "PseudoQualityReport",
    "PseudoQualitySummary",
    "ReliabilityBin",
    "ScoredLabelPrediction",
    "SharedQueryPlan",
    "SanitizedEntityFeatureView",
    "SanitizedFeatureView",
    "StrategyResult",
    "build_shared_query_plan",
    "audit_pseudo_label_quality",
    "build_leave_one_out_observations",
    "evaluate_dual_calibration",
    "is_label_bearing_field",
    "sanitize_feature_views",
    "validate_supervision_request",
]
