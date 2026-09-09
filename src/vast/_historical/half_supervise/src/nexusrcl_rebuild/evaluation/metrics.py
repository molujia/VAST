"""Ranking metrics for unified service-plus-host RCA evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class RankingMetrics:
    case_count: int
    a_at_k: Mapping[int, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"case_count": self.case_count}
        for k, value in self.a_at_k.items():
            payload["A@%d" % k] = value
        payload["metadata"] = dict(self.metadata)
        return payload


def evaluate_ranked_rows(
    scored_rows: pd.DataFrame,
    windows: pd.DataFrame,
    ks: Sequence[int] = (1, 3, 5),
) -> RankingMetrics:
    fault_windows = windows[windows["window_kind"] == "fault"].copy()
    positive_map = {
        str(row.window_id): set(getattr(row, "positive_ids_list", []))
        for row in fault_windows.itertuples(index=False)
    }
    grouped = scored_rows.groupby("window_id", sort=False)
    hits = {int(k): 0 for k in ks}
    evaluated = 0

    for window_id, positives in positive_map.items():
        if not positives or window_id not in grouped.groups:
            continue
        ranked = (
            grouped.get_group(window_id)
            .sort_values(by=["score", "raw_score"], ascending=False)["entity_id"]
            .astype(str)
            .tolist()
        )
        evaluated += 1
        for k in ks:
            if any(entity_id in positives for entity_id in ranked[: int(k)]):
                hits[int(k)] += 1

    scores = {
        int(k): (float(hits[int(k)]) / float(evaluated) if evaluated else 0.0)
        for k in ks
    }
    return RankingMetrics(
        case_count=evaluated,
        a_at_k=scores,
        metadata={"hits": hits},
    )
