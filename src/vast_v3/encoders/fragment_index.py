"""Exact, block-indexed connected fragments without materializing triples.

Inputs are already eligible and provenance-deduplicated in a stable canonical
order. This module owns no random state and does not change the candidate law.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import islice
from typing import Iterator, Mapping, Sequence

from .fragments import EventRef, FragmentEdge, FAMILIES


def _whole_count(ids: Sequence[int], i: int, j: int, low: bool) -> int:
    value = len(ids) - bisect_right(ids, j)
    if low:
        value += bisect_left(ids, j) - bisect_right(ids, i)
    return value


def _prefix_count(ids: Sequence[int], i: int, j: int, low: bool, end: int) -> int:
    value = bisect_right(ids, end) - bisect_right(ids, j) if end > j else 0
    if low and end > i:
        value += bisect_right(ids, min(end, j - 1)) - bisect_right(ids, i)
    return value


def _direction_family(edges: Sequence[tuple[int, int]], revisit: bool) -> int:
    if revisit:
        return 0
    branch = convergence = False
    for position, (source, target) in enumerate(edges):
        for other_source, other_target in edges[:position]:
            branch |= source == other_source and target != other_target
            convergence |= target == other_target and source != other_source
    return 1 if branch and convergence else 2 if branch else 3 if convergence else 4


@dataclass(slots=True)
class _Descriptor:
    prefix: tuple[int, ...]
    family: int
    anchors: frozenset[int]
    count: int = 1
    terms: tuple[tuple[Sequence[int], int], ...] = ()
    low: bool = False
    extra_counts: Mapping[int, int] = field(default_factory=dict)
    extra_lists: Mapping[int, Sequence[int]] = field(default_factory=dict)

    def weight(self, anchor: int) -> int:
        return self.count if anchor in self.anchors else self.extra_counts.get(anchor, 0)

    def decode(self, anchor: int, rank: int, edge_count: int) -> tuple[int, ...]:
        if not self.terms:
            return self.prefix
        terms = self.terms if anchor in self.anchors else ((self.extra_lists[anchor], 1),)
        i, j = self.prefix
        lo, hi = i + 1, edge_count - 1
        while lo < hi:
            middle = (lo + hi) // 2
            count = sum(sign * _prefix_count(ids, i, j, self.low, middle)
                        for ids, sign in terms)
            if count > rank:
                hi = middle
            else:
                lo = middle + 1
        return tuple(sorted((i, j, lo)))


@dataclass(slots=True)
class _Block:
    start: tuple[int, int]
    units: int


@dataclass(slots=True)
class _Weights:
    cumulative: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)


class FragmentIndex:
    """Count exact family/anchor strata and decode zero-based integer ranks.

    A candidate appears once in each *distinct entity* anchor that it contains.
    Entity integers do not determine rank ordering. Stable event/edge ordering
    is the caller's responsibility, including transported owner provenance.

    Construction visits connected pairs, never all triples. Sparse cumulative
    block weights make decoding proportional to touched blocks, not a complete
    new scan per sampled rank. Python integer counts cannot silently overflow.
    """

    _BLOCK_SIZE = 256

    def __init__(self, events: Sequence[EventRef], edges: Sequence[FragmentEdge],
                 max_edges: int = 3):
        if max_edges not in (1, 2, 3):
            raise ValueError("max_edges must be 1, 2 or 3")
        self.events = tuple(events)
        self.edges = tuple(edges)
        self.max_edges = max_edges
        self._endpoints = tuple((edge.source, edge.target) for edge in self.edges)
        self._entities = tuple(event.entity for event in self.events)
        self._incident: dict[int, set[int]] = defaultdict(set)
        self._pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._source: dict[int, list[int]] = defaultdict(list)
        self._target: dict[int, list[int]] = defaultdict(list)
        self._source_entity: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._target_entity: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for position, (source, target) in enumerate(self._endpoints):
            if min(source, target) < 0 or max(source, target) >= len(self.events):
                raise ValueError("edge endpoint is outside the event inventory")
            self._incident[source].add(position)
            self._incident[target].add(position)
            self._pair[source, target].append(position)
            self._source[source].append(position)
            self._target[target].append(position)
            self._source_entity[source][self._entities[target]].append(position)
            self._target_entity[target][self._entities[source]].append(position)
        self._blocks: list[_Block] = []
        self._weights: dict[tuple[int, int], _Weights] = {}
        family_totals = [0] * len(FAMILIES)
        anchor_totals: dict[int, list[int]] = {}
        block_counts = defaultdict(lambda: [0] * len(FAMILIES))
        block_start = None
        units = 0

        def flush() -> None:
            if block_start is None:
                return
            block_id = len(self._blocks)
            self._blocks.append(_Block(block_start, units))
            for anchor, counts in block_counts.items():
                total = anchor_totals.setdefault(anchor, [0] * len(FAMILIES))
                for family, count in enumerate(counts):
                    if count:
                        total[family] += count
                        weights = self._weights.setdefault((anchor, family), _Weights())
                        previous = weights.cumulative[-1] if weights.cumulative else 0
                        weights.cumulative.append(previous + count)
                        weights.blocks.append(block_id)

        for unit in self._units():
            if block_start is None:
                block_start = unit
            units += 1
            self._count_unit(unit, block_counts, family_totals)
            if units == self._BLOCK_SIZE:
                flush()
                block_counts = defaultdict(lambda: [0] * len(FAMILIES))
                block_start = None
                units = 0
        flush()
        self.family_counts = dict(zip(FAMILIES, family_totals))
        self.anchor_family_counts = {
            anchor: dict(zip(FAMILIES, counts)) for anchor, counts in anchor_totals.items()}
        self.total_count = sum(family_totals)

    def _count_unit(self, unit: tuple[int, int], block_counts, family_totals) -> None:
        """Accumulate exact block weights without constructing rank descriptors.

        The candidate partition is the same as ``_descriptors`` below. Only
        integer sums are needed here; prefix terms and per-descriptor anchor
        maps are built later, for the small set of blocks actually decoded.
        Unit/block boundaries and the rank-decoding traversal are unchanged.
        """
        i, j = unit
        endpoints = (self._endpoints[i],) if j == -1 else (self._endpoints[i], self._endpoints[j])
        vertices = tuple(sorted({v for edge in endpoints for v in edge}))
        anchors = frozenset(self._entities[v] for v in vertices)
        revisit = len(anchors) < len(vertices)
        # These anchors occur in every candidate represented by this unit.
        # Reuse their mutable counter rows instead of repeated dict lookups.
        anchor_counts = tuple(block_counts[anchor] for anchor in anchors)

        def add(family: int, count: int) -> None:
            family_totals[family] += count
            for counts in anchor_counts:
                counts[family] += count

        add(_direction_family(endpoints, revisit), 1)
        if j == -1 or self.max_edges < 3:
            return
        first_endpoints = set(endpoints[0])
        for source in vertices:
            for target in vertices:
                ids = self._pair.get((source, target))
                if not ids:
                    continue
                low = source not in first_endpoints and target not in first_endpoints
                count = _whole_count(ids, i, j, low)
                if count:
                    add(_direction_family(endpoints + ((source, target),), revisit), count)

        for outward in (True, False):
            entity_lists = self._source_entity if outward else self._target_entity
            for vertex in vertices:
                groups = entity_lists.get(vertex)
                if not groups:
                    continue
                low = vertex not in first_endpoints
                old_count = 0
                for anchor in anchors:
                    ids = groups.get(anchor)
                    if ids:
                        old_count += _whole_count(ids, i, j, low)
                for other in vertices:
                    key = (vertex, other) if outward else (other, vertex)
                    ids = self._pair.get(key)
                    if ids:
                        old_count -= _whole_count(ids, i, j, low)
                if old_count:
                    add(0, old_count)
                new_count = 0
                family = -1
                for anchor, ids in groups.items():
                    if anchor not in anchors:
                        count = _whole_count(ids, i, j, low)
                        if count:
                            if family == -1:
                                edge = (vertex, -1) if outward else (-1, vertex)
                                family = _direction_family(endpoints + (edge,), revisit)
                            block_counts[anchor][family] += count
                            new_count += count
                if new_count:
                    add(family, new_count)

    def _units(self, start: tuple[int, int] | None = None) -> Iterator[tuple[int, int]]:
        first, minimum_neighbor = start if start is not None else (0, -1)
        for i in range(first, len(self.edges)):
            if i != first or minimum_neighbor == -1:
                yield i, -1
            if self.max_edges == 1:
                continue
            source, target = self._endpoints[i]
            for j in sorted(self._incident[source] | self._incident[target]):
                if j > i and (i != first or j >= minimum_neighbor):
                    yield i, j

    def _descriptors(self, unit: tuple[int, int]) -> Iterator[_Descriptor]:
        i, j = unit
        prefix = (i,) if j == -1 else (i, j)
        endpoints = tuple(self._endpoints[index] for index in prefix)
        vertices = tuple(sorted({v for edge in endpoints for v in edge}))
        anchors = frozenset(self._entities[v] for v in vertices)
        revisit = len(anchors) < len(vertices)
        yield _Descriptor(prefix, _direction_family(endpoints, revisit), anchors)
        if j == -1 or self.max_edges < 3:
            return
        first_endpoints = set(endpoints[0])
        for source in vertices:
            for target in vertices:
                ids = self._pair.get((source, target))
                if not ids:
                    continue
                low = source not in first_endpoints and target not in first_endpoints
                count = _whole_count(ids, i, j, low)
                if count:
                    family = _direction_family(endpoints + ((source, target),), revisit)
                    yield _Descriptor(prefix, family, anchors, count, ((ids, 1),), low)

        for outward in (True, False):
            all_lists = self._source if outward else self._target
            entity_lists = self._source_entity if outward else self._target_entity
            for vertex in vertices:
                groups = entity_lists.get(vertex)
                if not groups:
                    continue
                low = vertex not in first_endpoints
                old_lists = [groups[a] for a in anchors if a in groups]
                internal = [self._pair[key] for other in vertices
                            if (key := ((vertex, other) if outward else (other, vertex))) in self._pair]
                old_terms = tuple((ids, 1) for ids in old_lists) + tuple((ids, -1) for ids in internal)
                old_count = sum(sign * _whole_count(ids, i, j, low) for ids, sign in old_terms)
                if old_count:
                    yield _Descriptor(prefix, 0, anchors, old_count, old_terms, low)
                extra_counts = {}
                for anchor, ids in groups.items():
                    if anchor not in anchors:
                        count = _whole_count(ids, i, j, low)
                        if count:
                            extra_counts[anchor] = count
                count = sum(extra_counts.values())
                if count:
                    new_terms = ((all_lists[vertex], 1),) + tuple((ids, -1) for ids in old_lists)
                    edge = (vertex, -1) if outward else (-1, vertex)
                    family = _direction_family(endpoints + (edge,), revisit)
                    yield _Descriptor(prefix, family, anchors, count, new_terms, low,
                                      extra_counts, groups)

    def unrank(self, anchor: int, family: str, rank: int) -> tuple[int, ...]:
        return self.unrank_many({(anchor, family): [rank]})[anchor, family][0]

    def unrank_many(self, requests: Mapping[tuple[int, str], Sequence[int]]) -> dict[
            tuple[int, str], list[tuple[int, ...]]]:
        """Decode all requests with one scan of each touched coarse block.

        Input rank order and duplicate requests are preserved. An empty request
        is legal even for an empty anchor/family stratum.
        """
        result = {key: [None] * len(ranks) for key, ranks in requests.items()}
        by_block: dict[int, dict[tuple[int, int], list[tuple[int, tuple[int, str], int]]]] = defaultdict(
            lambda: defaultdict(list))
        for key, ranks in requests.items():
            anchor, family = key
            if family not in FAMILIES:
                raise ValueError(f"unknown structural family: {family}")
            family_index = FAMILIES.index(family)
            weights = self._weights.get((anchor, family_index))
            count = weights.cumulative[-1] if weights is not None else 0
            for position, rank in enumerate(ranks):
                if not isinstance(rank, int) or not 0 <= rank < count:
                    raise IndexError(f"rank {rank!r} is outside stratum of size {count}")
                assert weights is not None
                slot = bisect_right(weights.cumulative, rank)
                previous = weights.cumulative[slot - 1] if slot else 0
                by_block[weights.blocks[slot]][anchor, family_index].append((rank - previous, key, position))
        for block_id, strata in by_block.items():
            block = self._blocks[block_id]
            states = {key: [sorted(targets), 0, 0] for key, targets in strata.items()}
            for unit in islice(self._units(block.start), block.units):
                for descriptor in self._descriptors(unit):
                    for (anchor, family), state in states.items():
                        targets, position, consumed = state
                        if family != descriptor.family or position == len(targets):
                            continue
                        weight = descriptor.weight(anchor)
                        while position < len(targets) and targets[position][0] < consumed + weight:
                            rank, key, output_position = targets[position]
                            result[key][output_position] = descriptor.decode(anchor, rank - consumed, len(self.edges))
                            position += 1
                        state[1], state[2] = position, consumed + weight
            if any(state[1] != len(state[0]) for state in states.values()):
                raise RuntimeError("fragment block counts and rank decoding disagree")
        return result
