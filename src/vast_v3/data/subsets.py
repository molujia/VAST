"""A lazy training-only view, never an alternate scientific population."""
from __future__ import annotations

from collections.abc import Mapping

from .manifests import FrozenSplit


class TrainingCases(Mapping):
    """Expose exactly frozen train IDs without enumerating/loading test cases."""

    def __init__(self, repository):
        split = repository.manifest.split
        if not isinstance(repository, Mapping) or not isinstance(split, FrozenSplit):
            raise ValueError('training view requires a repository with a frozen split')
        self._repository = repository
        self._ids = tuple(split.train_case_ids)
        self._allowed = frozenset(self._ids)
        if not self._allowed <= set(repository):
            raise ValueError('repository is missing frozen training identities')

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)

    def __contains__(self, key):
        return key in self._allowed

    def __getitem__(self, key):
        if key not in self._allowed:
            raise KeyError(key)
        case = self._repository[key]
        if case.case_id != key:
            raise ValueError('training case identity mismatch')
        return case

    def case_audit(self, key):
        if key not in self._allowed:
            raise KeyError(key)
        return self._repository.case_audit(key)
