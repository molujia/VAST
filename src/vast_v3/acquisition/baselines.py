"""Label-blind whole-case random acquisition control.

Other named experiment controls require their separately frozen descriptors and
adapter definitions; this module does not silently substitute a new algorithm.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .propcover import partition_cases, query_count, seeded_case_priority


class RandomCaseSelector:
    """Uniform seeded permutation of the complete permitted public-ID pool.

    Selection only proposes IDs. It never accesses labels, trains an encoder,
    mutates a query ledger, or chooses a label-dependent eligible subset.
    """

    def __init__(self, case_ids: Iterable[str], *, seed: int = 0,
                 priority: Sequence[str] | None = None) -> None:
        self.case_ids = tuple(case_ids)
        if any(not isinstance(case_id, str) or not case_id for case_id in self.case_ids):
            raise ValueError('permitted case IDs must be nonempty strings')
        self.priority = seeded_case_priority(self.case_ids, seed, priority)

    def select(self, labeled_ids: Iterable[str] = (), unlabeled_ids: Iterable[str] | None = None,
               *, batch_size: int = 5, remaining_budget: int = 30) -> tuple[str, ...]:
        """Use a caller-accounted remainder; ``select_budgeted`` is safer for ledgers."""
        if isinstance(batch_size, bool) or isinstance(remaining_budget, bool):
            raise ValueError('batch size and remaining budget must be nonnegative integers')
        _, unlabeled = partition_cases(self.case_ids, labeled_ids, unlabeled_ids)
        count = query_count(batch_size, remaining_budget, len(unlabeled))
        return tuple(case_id for case_id in self.priority if case_id in unlabeled)[:count]

    def select_budgeted(self, accessed_ids: Iterable[str], *, total_budget: int,
                        batch_size: int = 5) -> tuple[str, ...]:
        """Debit every root-accessed ID, including pending/failed label answers.

        The caller supplies the complete run access ledger's IDs, not just its
        successfully revealed label IDs. Blocked-ledger enforcement and actual
        label queries remain the caller's responsibility.
        """
        if isinstance(total_budget, bool) or not isinstance(total_budget, int) or total_budget < 0:
            raise ValueError('total budget must be a nonnegative integer')
        accessed = tuple(accessed_ids)
        partition_cases(self.case_ids, accessed)
        remaining = total_budget - len(accessed)
        if remaining < 0:
            raise ValueError('already accessed case count exceeds total budget')
        return self.select(accessed, batch_size=batch_size, remaining_budget=remaining)
