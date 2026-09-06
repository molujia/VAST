"""Semi-supervised training components for the NexusRCL rebuild."""

from importlib import import_module


_SEMISUPERVISED_EXPORTS = {
    "FeatureBundleTables",
    "QueryPlan",
    "SemiSupervisedConfig",
    "SemiSupervisedModel",
    "build_query_plan",
    "build_training_frame",
    "fit_semisupervised_ranker",
    "fit_semisupervised_ranker_on_tables",
    "load_feature_bundle_tables",
    "rank_scored_rows",
    "subset_feature_bundle_tables",
}

_PSEUDO_ADAPTER_EXPORTS = {
    "PseudoTrainingAdapterConfig",
    "adapt_matched_pseudo_training_rows",
    "adapt_pseudo_training_rows",
    "attach_frozen_pseudo_result",
    "attach_matched_pseudo_arm",
}

_JOINT_GRAPH_TEACHER_EXPORTS = {
    "JointGraphTeacherConfig",
    "JointGraphTeacherDecision",
    "JointGraphTeacherPool",
    "build_joint_graph_teacher_pool",
}

_JOINT_GRAPH_STUDENT_EXPORTS = {
    "JointGraphCaseSupervision",
    "JointGraphStudentConfig",
    "JointGraphStudentSupervision",
    "ListwiseLossResult",
    "WithinCasePair",
    "apply_encoder_update_scope",
    "build_joint_graph_student_supervision",
    "build_joint_graph_student_supervision_from_arm",
    "build_within_case_pairs",
    "compute_case_normalized_listwise_loss",
}

__all__ = [
    "FeatureBundleTables",
    "QueryPlan",
    "SemiSupervisedConfig",
    "SemiSupervisedModel",
    "build_query_plan",
    "build_training_frame",
    "fit_semisupervised_ranker",
    "fit_semisupervised_ranker_on_tables",
    "load_feature_bundle_tables",
    "rank_scored_rows",
    "subset_feature_bundle_tables",
    "PseudoTrainingAdapterConfig",
    "adapt_matched_pseudo_training_rows",
    "adapt_pseudo_training_rows",
    "attach_frozen_pseudo_result",
    "attach_matched_pseudo_arm",
    "JointGraphTeacherConfig",
    "JointGraphTeacherDecision",
    "JointGraphTeacherPool",
    "build_joint_graph_teacher_pool",
    "JointGraphCaseSupervision",
    "JointGraphStudentConfig",
    "JointGraphStudentSupervision",
    "ListwiseLossResult",
    "WithinCasePair",
    "apply_encoder_update_scope",
    "build_joint_graph_student_supervision",
    "build_joint_graph_student_supervision_from_arm",
    "build_within_case_pairs",
    "compute_case_normalized_listwise_loss",
]


def __getattr__(name):
    if name in _SEMISUPERVISED_EXPORTS:
        module = import_module(".semisupervised", __name__)
    elif name in _PSEUDO_ADAPTER_EXPORTS:
        module = import_module(".pseudo_adapter", __name__)
    elif name in _JOINT_GRAPH_TEACHER_EXPORTS:
        module = import_module(".joint_graph_teacher", __name__)
    elif name in _JOINT_GRAPH_STUDENT_EXPORTS:
        module = import_module(".joint_graph_student", __name__)
    else:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(module, name)
    globals()[name] = value
    return value
