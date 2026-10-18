"""Frozen-encoder, conditional whole-case fragment facility location.

Occurrences are indexed by case and exact fragment provenance, never by vector
value.  Every refresh includes the entire permitted L/U pool.  Similarities are
chunked once into per-case maxima; greedy queries never revisit token distances.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Collection, Iterable, Sequence

import torch
from torch import Tensor

from ..encoders.fragments import FAMILIES


@dataclass(frozen=True)
class CaseTokens:
    case_id: str
    tokens: Tensor
    families: tuple[str, ...]
    provenance: tuple[str, ...]
    encoder_version: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.encoder_version:
            raise ValueError("case ID and encoder version must be nonempty")
        if self.tokens.ndim != 2 or not self.tokens.is_floating_point():
            raise ValueError("tokens must be floating [occurrences,width] vectors")
        if self.tokens.shape[1] < 1 or len(self.families) != len(self.tokens) or len(self.provenance) != len(self.tokens):
            raise ValueError("token, family and provenance lengths must agree")
        if any(family not in FAMILIES for family in self.families):
            raise ValueError("unknown structural family")
        if any(not item for item in self.provenance):
            raise ValueError("fragment provenance must be nonempty")


@dataclass(frozen=True)
class SelectionResult:
    case_ids: tuple[str, ...]
    gains: tuple[float, ...]
    base_coverage: float
    final_coverage: float
    encoder_version: str
    reference_occurrences: int
    cache_stats: dict[str, int]


def seeded_case_priority(case_ids: Iterable[str], seed: int = 0,
                         priority: Sequence[str] | None = None) -> tuple[str, ...]:
    """A persisted random permutation of public case IDs, independent of labels."""
    ids = tuple(case_ids)
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("permitted case IDs must be unique and nonempty")
    if priority is not None:
        result = tuple(priority)
        if len(result) != len(ids) or set(result) != set(ids):
            raise ValueError("priority must be a permutation of the complete permitted pool")
        return result
    result = sorted(ids)
    random.Random(seed).shuffle(result)
    return tuple(result)


def partition_cases(pool_ids: Collection[str], labeled_ids: Iterable[str],
                    unlabeled_ids: Iterable[str] | None = None) -> tuple[set[str], set[str]]:
    labeled_tuple = tuple(labeled_ids)
    labeled = set(labeled_tuple)
    pool = set(pool_ids)
    if len(labeled) != len(labeled_tuple) or not labeled <= pool:
        raise ValueError("L/U partition has duplicate or unknown labeled IDs")
    unlabeled_tuple = tuple(pool - labeled) if unlabeled_ids is None else tuple(unlabeled_ids)
    unlabeled = set(unlabeled_tuple)
    if len(unlabeled) != len(unlabeled_tuple) or labeled & unlabeled or labeled | unlabeled != pool:
        raise ValueError("L/U partition must be disjoint and cover the complete permitted pool")
    return labeled, unlabeled


def query_count(batch_size: int, remaining_budget: int, unlabeled_count: int) -> int:
    if (not isinstance(batch_size, int) or not isinstance(remaining_budget, int)
            or min(batch_size, remaining_budget) < 0):
        raise ValueError("batch size and remaining budget must be nonnegative integers")
    return min(batch_size, remaining_budget, unlabeled_count)


class PropCover:
    """Exact facility-location selector over one frozen encoder version.

    ``select`` proposes IDs but does not access roots or mutate a query ledger.
    The training loop debits actual complete-case queries.  ``select_budgeted``
    additionally derives remaining budget from all previously queried IDs.
    """

    def __init__(self, cases: Iterable[CaseTokens], *, seed: int = 0,
                 priority: Sequence[str] | None = None,
                 similarity_chunk_size: int = 2048) -> None:
        prepared = tuple(cases)
        if not isinstance(similarity_chunk_size, int) or similarity_chunk_size < 1:
            raise ValueError("similarity_chunk_size must be a positive integer")
        self.case_ids = tuple(sorted(case.case_id for case in prepared))
        self.priority = seeded_case_priority(self.case_ids, seed, priority)
        self.similarity_chunk_size = similarity_chunk_size
        self.refresh(prepared)

    @property
    def weights(self) -> Tensor:
        return self._weights.clone()

    @property
    def cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    def refresh(self, cases: Iterable[CaseTokens]) -> None:
        """Atomically replace all L/U representations and rebuild their maxima."""
        supplied = tuple(cases)
        if len(supplied) != len(self.case_ids) or {case.case_id for case in supplied} != set(self.case_ids):
            raise ValueError("refresh must re-encode the complete permitted pool, including L and U")
        versions = {case.encoder_version for case in supplied}
        if len(versions) > 1:
            raise ValueError("cannot mix encoder versions in one acquisition round")
        widths = {case.tokens.shape[1] for case in supplied}
        if len(widths) > 1:
            raise ValueError("all cases must share one token width")
        dtype = torch.float64 if any(case.tokens.dtype == torch.float64 for case in supplied) else torch.float32
        snapshots = {}
        for case in supplied:
            tokens = case.tokens.detach().to(device="cpu", dtype=dtype).clone()
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError("unit tokens must be finite")
            if len(tokens) and not bool(torch.allclose(tokens.norm(dim=1), tokens.new_ones(len(tokens)), rtol=1e-5, atol=1e-5)):
                raise ValueError("PropCover requires unit tokens; normalize real fragments in SCOPE")
            by_provenance = {}
            for index, provenance in enumerate(case.provenance):
                if provenance in by_provenance:
                    previous = by_provenance[provenance]
                    if case.families[index] != case.families[previous] or not torch.equal(tokens[index], tokens[previous]):
                        raise ValueError("one fragment provenance has conflicting tokens or families")
                else:
                    by_provenance[provenance] = index
            order = [by_provenance[key] for key in sorted(by_provenance)]
            snapshots[case.case_id] = CaseTokens(
                case.case_id, tokens[order], tuple(case.families[index] for index in order),
                tuple(case.provenance[index] for index in order), case.encoder_version)
        family_cases: dict[str, set[str]] = defaultdict(set)
        family_counts = {}
        for case_id, case in snapshots.items():
            family_counts[case_id] = Counter(case.families)
            for family in family_counts[case_id]:
                family_cases[family].add(case_id)
        family_total = len(family_cases)
        keys, families, weight_values, vector_rows = [], [], [], []
        for case_id in self.case_ids:
            case = snapshots[case_id]
            for index, (family, provenance) in enumerate(zip(case.families, case.provenance)):
                keys.append((case_id, provenance))
                families.append(family)
                weight_values.append(1.0 / (family_total * len(family_cases[family]) * family_counts[case_id][family]))
                vector_rows.append(case.tokens[index])
        reference = (torch.stack(vector_rows) if vector_rows else
                     torch.empty((0, next(iter(widths), 64)), dtype=dtype))
        weights = torch.tensor(weight_values, dtype=torch.float64)
        maxima = {}
        kernel_blocks, kernel_pairs = 0, 0
        for case_id in self.case_ids:
            candidate = snapshots[case_id].tokens
            values = torch.zeros(len(reference), dtype=dtype)
            if len(candidate):
                for start in range(0, len(reference), self.similarity_chunk_size):
                    end = min(start + self.similarity_chunk_size, len(reference))
                    similarities = ((1.0 + reference[start:end] @ candidate.T) / 2.0).clamp(0.0, 1.0)
                    values[start:end] = similarities.amax(dim=1)
                    kernel_blocks += 1
                    kernel_pairs += (end - start) * len(candidate)
            maxima[case_id] = values
        # Assignment follows all validation/computation: failed refreshes retain
        # the preceding coherent snapshot, never a mixed-version partial cache.
        self.encoder_version = next(iter(versions), "empty-pool")
        self.occurrence_keys = tuple(keys)
        self.occurrence_families = tuple(families)
        self._weights = weights
        self._maxima = maxima
        self._dtype = dtype
        self._cache_stats = {"kernel_blocks": kernel_blocks, "kernel_pairs": kernel_pairs,
                             "cache_columns": len(maxima), "reference_occurrences": len(keys)}

    def _coverage_vector(self, case_ids: Iterable[str]) -> Tensor:
        result = torch.zeros(len(self.occurrence_keys), dtype=self._dtype)
        for case_id in set(case_ids):
            if case_id not in self._maxima:
                raise ValueError(f"unknown case ID in coverage request: {case_id}")
            result = torch.maximum(result, self._maxima[case_id])
        return result

    def _weighted(self, vector: Tensor) -> float:
        return float(torch.dot(vector.to(torch.float64), self._weights))

    def coverage(self, case_ids: Iterable[str]) -> float:
        return self._weighted(self._coverage_vector(case_ids))

    def objective(self, selected_ids: Iterable[str], *, labeled_ids: Iterable[str] = ()) -> float:
        labeled = set(labeled_ids)
        return self.coverage(labeled | set(selected_ids)) - self.coverage(labeled)

    def select(self, labeled_ids: Iterable[str] = (), unlabeled_ids: Iterable[str] | None = None,
               *, batch_size: int = 5, remaining_budget: int = 30) -> SelectionResult:
        labeled, unlabeled = partition_cases(self.case_ids, labeled_ids, unlabeled_ids)
        count = query_count(batch_size, remaining_budget, len(unlabeled))
        baseline = self._coverage_vector(labeled)
        initial = self._weighted(baseline)
        selected, gains = [], []
        for _ in range(count):
            best_id, best_gain = None, float("-inf")
            # Iterating the persisted priority implements exact gain ties,
            # including all-zero gains, with no root-dependent tie rule.
            for case_id in self.priority:
                if case_id not in unlabeled:
                    continue
                gain = self._weighted(torch.maximum(baseline, self._maxima[case_id]) - baseline)
                if gain > best_gain:
                    best_id, best_gain = case_id, gain
            assert best_id is not None
            selected.append(best_id)
            gains.append(best_gain)
            unlabeled.remove(best_id)
            baseline = torch.maximum(baseline, self._maxima[best_id])
        return SelectionResult(tuple(selected), tuple(gains), initial, self._weighted(baseline),
                               self.encoder_version, len(self.occurrence_keys), self.cache_stats)

    def select_budgeted(self, labeled_ids: Iterable[str], *, total_budget: int,
                        batch_size: int = 5) -> SelectionResult:
        labeled = tuple(labeled_ids)
        if not isinstance(total_budget, int) or total_budget < 0:
            raise ValueError("total budget must be a nonnegative integer")
        partition_cases(self.case_ids, labeled)
        remaining = total_budget - len(labeled)
        if remaining < 0:
            raise ValueError("already queried case count exceeds total budget")
        return self.select(labeled, batch_size=batch_size, remaining_budget=remaining)
