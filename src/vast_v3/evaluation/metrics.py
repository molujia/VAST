"""Canonical, complete-catalog metrics from the VAST-V3 R03 contract.

This module is evaluation-only: callers supply authorized truth after prediction.
No incident is silently dropped and no omitted candidate is an infinite-rank root.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real


def _ids(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of canonical IDs")
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def _finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    candidate_ids: tuple[str, ...]
    ranking: tuple[str, ...]
    scores: tuple[float, ...]  # aligned with ranking, not input catalog order
    true_roots: tuple[str, ...]
    root_ranks: Mapping[str, int]
    first_root_rank: int
    hits: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "candidate_ids": list(self.candidate_ids),
            "ranking": list(self.ranking), "scores": list(self.scores),
            "true_roots": list(self.true_roots), "root_ranks": dict(self.root_ranks),
            "first_root_rank": self.first_root_rank,
            **{f"hit_at_{k}": self.hits[k - 1] for k in range(1, 6)},
            "tie_break": "descending_score_then_ascending_canonical_id",
        }


def evaluate_case(case_id: str, candidate_ids: Sequence[str],
                  scores: Mapping[str, Real], roots: Sequence[str]) -> CaseMetrics:
    """Rank exactly the frozen candidates, then evaluate all jointly true roots."""
    _ids([case_id], "case_id")
    catalog = _ids(candidate_ids, "candidate_ids")
    truth = tuple(sorted(_ids(roots, "roots")))
    if not set(truth) <= set(catalog):
        raise ValueError("Every root must belong to the frozen candidate catalog")
    if not isinstance(scores, Mapping) or set(scores) != set(catalog):
        raise ValueError("Scores must cover exactly the frozen candidate catalog")
    numeric = {entity: _finite(scores[entity], f"score[{entity}]") for entity in catalog}
    ranked = tuple(sorted(catalog, key=lambda entity: (-numeric[entity], entity)))
    positions = {entity: rank for rank, entity in enumerate(ranked, 1)}
    root_ranks = {root: positions[root] for root in truth}
    first = min(root_ranks.values())
    return CaseMetrics(case_id, tuple(sorted(catalog)), ranked,
                       tuple(numeric[entity] for entity in ranked), truth,
                       root_ranks, first, tuple(int(first <= k) for k in range(1, 6)))


def mrr_bounds_from_hits(n: int, hits: Sequence[int]) -> dict[str, float]:
    """Bounds from cumulative H1..H5 only; neither is an exact MRR estimate."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("N must be a positive integer")
    values = tuple(hits)
    if len(values) != 5 or any(isinstance(h, bool) or not isinstance(h, Integral)
                               or not 0 <= h <= n for h in values):
        raise ValueError("H1..H5 must be five integer cumulative counts in [0, N]")
    if tuple(sorted(values)) != values:
        raise ValueError("H1..H5 must be nondecreasing")
    prefix = (0,) + values
    s5 = sum((prefix[k] - prefix[k - 1]) / k for k in range(1, 6))
    return {"MRR_lower": s5 / n, "MRR_upper": (s5 + (n - values[4]) / 6) / n}


def aggregate_metrics(cases: Sequence[CaseMetrics], *,
                      expected_case_ids: Sequence[str]) -> dict:
    """Aggregate the exact frozen incident population; never infer a denominator."""
    expected = _ids(expected_case_ids, "expected_case_ids")
    results = tuple(cases)
    observed = _ids([case.case_id for case in results], "observed case IDs")
    if set(observed) != set(expected):
        raise ValueError("Scored incidents differ from the frozen evaluation population")
    # Revalidate even user-constructed dataclasses; frozen does not make nested maps immutable.
    for case in results:
        rebuilt = evaluate_case(case.case_id, case.candidate_ids,
                                dict(zip(case.ranking, case.scores)), case.true_roots)
        if rebuilt != case:
            raise ValueError("CaseMetrics does not match its full ranking/roots")
    n = len(expected)
    hits = [sum(case.hits[k - 1] for case in results) for k in range(1, 6)]
    summary = {"N": n, **{f"H{k}": hits[k - 1] for k in range(1, 6)},
               **{f"Hit@{k}": hits[k - 1] / n for k in range(1, 6)},
               "TOP135": (hits[0] + hits[2] + hits[4]) / (3 * n),
               "Avg@5": sum(hits) / (5 * n),
               "MRR": sum(1 / case.first_root_rank for case in results) / n,
               **mrr_bounds_from_hits(n, hits)}
    for k in range(1, 6):
        summary[f"RootRecall@{k}"] = sum(
            sum(rank <= k for rank in case.root_ranks.values()) / len(case.true_roots)
            for case in results) / n
        summary[f"AllRoots@{k}"] = sum(
            all(rank <= k for rank in case.root_ranks.values()) for case in results) / n
    return summary
