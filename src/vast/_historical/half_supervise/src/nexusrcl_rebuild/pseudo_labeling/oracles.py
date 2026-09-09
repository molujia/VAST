"""Narrow label interfaces for annotation and post-generation auditing."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional, Sequence

from .contracts import (
    AnnotationAnswer,
    AnnotationQuery,
    StrategyResult,
)
from .quality import PseudoQualityReport, audit_pseudo_label_quality


def _immutable_labels(
    authoritative_labels: Mapping[str, Sequence[str]],
) -> Mapping[str, tuple]:
    return MappingProxyType(
        {
            str(window_id): tuple(sorted({str(item) for item in labels if str(item)}))
            for window_id, labels in authoritative_labels.items()
        }
    )


class AnnotationOracle:
    """Expose authoritative labels only for a pre-authorized query plan."""

    def __init__(
        self,
        authoritative_labels: Mapping[str, Sequence[str]],
        authorized_window_ids: Sequence[str],
    ) -> None:
        self.__authoritative_labels = _immutable_labels(authoritative_labels)
        self.__authorized_window_ids = frozenset(str(item) for item in authorized_window_ids)

    def answer(self, query: AnnotationQuery) -> AnnotationAnswer:
        if query.window_id not in self.__authorized_window_ids:
            raise PermissionError("window is not authorized by the persisted query plan")
        if query.window_id not in self.__authoritative_labels:
            raise KeyError("authorized window has no authoritative annotation")
        return AnnotationAnswer(
            query=query,
            authoritative_ids=self.__authoritative_labels[query.window_id],
        )


class PseudoQualityOracle:
    """Return aggregate audit counts without exposing labels to generators."""

    def __init__(self, authoritative_labels: Mapping[str, Sequence[str]]) -> None:
        self.__authoritative_labels = _immutable_labels(authoritative_labels)

    def audit(
        self,
        result: StrategyResult,
        eligible_window_ids: Sequence[str],
        entity_types: Optional[Mapping[str, str]] = None,
        cluster_ids: Optional[Mapping[str, Optional[int]]] = None,
    ) -> PseudoQualityReport:
        return audit_pseudo_label_quality(
            result=result,
            authoritative_labels=self.__authoritative_labels,
            eligible_window_ids=eligible_window_ids,
            entity_types=entity_types,
            cluster_ids=cluster_ids,
        )
