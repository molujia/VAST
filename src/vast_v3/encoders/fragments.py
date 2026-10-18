"""Exact observable fragment inventory and coupled family-balanced sampling.

Input links are already observed-active typed relations.  Their source/target
indices address the input event sequence, whereas output indices are local to
each fragment.  Provenance and sampling marks are metadata, never model inputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Iterator, Mapping, Sequence


FAMILIES = ("revisit", "mixed_branch_convergence", "branch", "convergence", "chain")


@dataclass(frozen=True)
class EventRef:
    entity: int
    bin: int
    start: float
    end: float
    category: int
    provenance: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.end < self.start:
            raise ValueError("event interval must be finite with start <= end")
        if min(self.entity, self.bin, self.category) < 0:
            raise ValueError("entity, bin, and category indices must be nonnegative")
        if not self.provenance:
            raise ValueError("event provenance must be nonempty")


@dataclass(frozen=True)
class FragmentEdge:
    source: int
    target: int
    relation: int
    owner: int = -1
    response: bool = False

    def __post_init__(self) -> None:
        if min(self.source, self.target, self.relation) < 0 or self.owner < -1:
            raise ValueError("invalid edge index")


@dataclass(frozen=True)
class Fragment:
    events: tuple[EventRef, ...]
    edges: tuple[FragmentEdge, ...]
    family: str
    provenance: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown structural family: {self.family}")
        if any(max(edge.source, edge.target) >= len(self.events) for edge in self.edges):
            raise ValueError("edge endpoint is outside the fragment event inventory")


@dataclass(frozen=True)
class ExtractionResult:
    fragments: tuple[Fragment, ...]
    candidate_count: int
    per_anchor_candidates: dict[int, int]
    per_anchor_retained: dict[int, int]
    sampling_marks: dict[str, float]
    family_candidates: dict[str, int]
    family_retained: dict[str, int]

    def __iter__(self) -> Iterator[Fragment]:
        return iter(self.fragments)

    def __len__(self) -> int:
        return len(self.fragments)


class EnumerationLimitError(RuntimeError):
    """Exact enumeration exceeded its explicit safety budget; no partial result."""


def structural_family(events: Sequence[EventRef], edges: Sequence[FragmentEdge]) -> str:
    if len({event.entity for event in events}) < len(events):
        return "revisit"
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        outgoing[edge.source].add(edge.target)
        incoming[edge.target].add(edge.source)
    branch = any(len(group) > 1 for group in outgoing.values())
    convergence = any(len(group) > 1 for group in incoming.values())
    if branch and convergence:
        return "mixed_branch_convergence"
    if branch:
        return "branch"
    return "convergence" if convergence else "chain"


def ordered_paths(fragment: Fragment) -> tuple[tuple[int, ...], ...]:
    """All maximal source-to-sink edge paths in the resolved-time DAG.

    A-C, B-C, C-D gives A-C-D and B-C-D.  Overlapping intervals do not
    establish order and are excluded only from signatures, not from fragments.
    """
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: set[int] = set()
    for index, edge in enumerate(fragment.edges):
        if fragment.events[edge.source].end < fragment.events[edge.target].start:
            outgoing[edge.source].append(index)
            incoming.add(edge.target)
    paths: list[tuple[int, ...]] = []

    def visit(vertex: int, path: tuple[int, ...]) -> None:
        if vertex not in outgoing:
            if path:
                paths.append(path)
            return
        for edge_index in outgoing[vertex]:
            visit(fragment.edges[edge_index].target, path + (edge_index,))

    for vertex in sorted(set(outgoing) - incoming):
        visit(vertex, ())
    return tuple(paths)


def _seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def _link_key(edge: FragmentEdge, events: Sequence[EventRef],
              owner_keys: Mapping[int, str] | None = None) -> tuple:
    owner_identity = ("index", edge.owner)
    if owner_keys is not None:
        if edge.owner == -1:
            owner_identity = ("absent", "")
        else:
            if edge.owner not in owner_keys or not owner_keys[edge.owner]:
                raise ValueError("owner_keys must cover every observed numeric owner")
            owner_identity = ("provenance", owner_keys[edge.owner])
    return (events[edge.source].provenance, events[edge.target].provenance,
            edge.relation, owner_identity, edge.response)


def _balanced_select(fragments: Sequence[Fragment], cap: int,
                     marks: Mapping[str, float], seed: int) -> tuple[Fragment, ...]:
    groups = {family: [] for family in FAMILIES}
    for fragment in fragments:
        groups[fragment.family].append(fragment)
    quotas = {family: 0 for family in FAMILIES}
    rng = random.Random(seed)
    remaining = min(cap, len(fragments))
    while remaining:
        available = [family for family in FAMILIES if quotas[family] < len(groups[family])]
        rng.shuffle(available)
        for family in available:
            if not remaining:
                break
            quotas[family] += 1
            remaining -= 1
    selected = []
    for family in FAMILIES:
        # Continuous random priorities implement uniform reservoir selection.
        # Provenance only breaks exact numeric mark ties; no learned ID feature.
        group = sorted(groups[family], key=lambda item: (marks[item.provenance], item.provenance))
        selected.extend(group[:quotas[family]])
    return tuple(sorted(selected, key=lambda item: item.provenance))


def extract_fragments(
    events: Sequence[EventRef],
    links: Sequence[FragmentEdge],
    *,
    delta: float,
    seed: int = 0,
    per_anchor_cap: int = 8,
    case_cap: int = 128,
    max_edges: int = 3,
    max_enumerated: int = 100_000,
    sampling_marks: Mapping[str, float] | None = None,
    owner_keys: Mapping[int, str] | None = None,
) -> ExtractionResult:
    """Enumerate all eligible connected edge sets, then apply both stated caps.

    No adjacency subsampling is performed.  ``max_enumerated`` is an explicit
    fail-fast resource guard, not a scientific truncation rule.  Stable event
    provenance and owner keys must be transported with coupled permutation
    views, as must any externally provided sampling marks.  ``owner_keys``
    supplies stable metadata when owner array indices themselves are permuted.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if not 1 <= max_edges <= 3 or min(per_anchor_cap, case_cap) < 0 or max_enumerated < 1:
        raise ValueError("invalid fragment caps or enumeration limit")
    by_provenance: dict[str, EventRef] = {}
    for event in events:
        previous = by_provenance.setdefault(event.provenance, event)
        if previous != event:
            raise ValueError("one observation provenance has conflicting event records")
    unique_events = tuple(by_provenance[key] for key in sorted(by_provenance))
    event_index = {event.provenance: i for i, event in enumerate(unique_events)}
    eligible: dict[tuple, FragmentEdge] = {}
    for edge in links:
        if max(edge.source, edge.target) >= len(events):
            raise ValueError("observed relation endpoint is outside event inventory")
        source, target = events[edge.source], events[edge.target]
        # Overlap is uncertain order; disjoint backward relations are not a
        # temporally supported propagation transition in this directed view.
        if target.end < source.start or target.start - source.end > 8 * delta:
            continue
        remapped = FragmentEdge(event_index[source.provenance], event_index[target.provenance],
                                edge.relation, edge.owner, edge.response)
        eligible[_link_key(remapped, unique_events, owner_keys)] = remapped
    edges = tuple(eligible[key] for key in sorted(eligible))
    incident: dict[int, set[int]] = defaultdict(set)
    for index, edge in enumerate(edges):
        incident[edge.source].add(index)
        incident[edge.target].add(index)
    discovered: set[frozenset[int]] = set()
    pending: list[frozenset[int]] = []

    def add(edge_set: frozenset[int]) -> None:
        if edge_set in discovered:
            return
        if len(discovered) >= max_enumerated:
            raise EnumerationLimitError(
                f"exact fragment enumeration exceeded max_enumerated={max_enumerated}; "
                f"eligible_links={len(edges)}, completed_candidates={len(discovered)}; "
                "no partially sampled inventory was returned"
            )
        discovered.add(edge_set)
        pending.append(edge_set)

    for index in range(len(edges)):
        add(frozenset((index,)))
    cursor = 0
    while cursor < len(pending):
        edge_set = pending[cursor]
        cursor += 1
        if len(edge_set) == max_edges:
            continue
        neighbors: set[int] = set()
        for edge_index in edge_set:
            edge = edges[edge_index]
            neighbors.update(incident[edge.source])
            neighbors.update(incident[edge.target])
        for extra in sorted(neighbors - edge_set):
            add(edge_set | {extra})
    candidates: list[Fragment] = []
    for edge_set in discovered:
        members = sorted({vertex for index in edge_set
                          for vertex in (edges[index].source, edges[index].target)})
        local_index = {old: new for new, old in enumerate(members)}
        selected_events = tuple(unique_events[index] for index in members)
        selected_edges = tuple(FragmentEdge(local_index[edges[index].source],
                                           local_index[edges[index].target],
                                           edges[index].relation, edges[index].owner,
                                           edges[index].response)
                               for index in sorted(edge_set))
        evidence = [sorted(event.provenance for event in selected_events),
                    sorted(_link_key(edge, selected_events, owner_keys) for edge in selected_edges)]
        provenance = hashlib.sha256(json.dumps(evidence, separators=(",", ":")).encode()).hexdigest()
        candidates.append(Fragment(selected_events, selected_edges,
                                   structural_family(selected_events, selected_edges), provenance))
    candidates.sort(key=lambda item: item.provenance)
    marks: dict[str, float] = {}

    def stage_marks(stage: str, group: Sequence[Fragment]) -> dict[str, float]:
        current = {}
        for fragment in group:
            key = stage + "/" + fragment.provenance
            mark = (sampling_marks[key] if sampling_marks is not None and key in sampling_marks
                    else random.Random(_seed(seed, key)).random())
            if not math.isfinite(mark):
                raise ValueError("sampling marks must be finite")
            marks[key] = current[fragment.provenance] = float(mark)
        return current
    anchored: dict[int, list[Fragment]] = defaultdict(list)
    for fragment in candidates:
        for entity in {event.entity for event in fragment.events}:
            anchored[entity].append(fragment)
    retained: dict[str, Fragment] = {}
    counts = {}
    for entity, group in anchored.items():
        anchor_observations = sorted(event.provenance for event in unique_events if event.entity == entity)
        anchor_key = hashlib.sha256(json.dumps(anchor_observations).encode()).hexdigest()
        stage = "anchor/" + anchor_key
        selected = _balanced_select(group, per_anchor_cap, stage_marks(stage, group), _seed(seed, stage))
        counts[entity] = len(selected)
        retained.update((fragment.provenance, fragment) for fragment in selected)
    retained_values = tuple(retained.values())
    final = _balanced_select(retained_values, case_cap, stage_marks("case", retained_values), _seed(seed, "case"))
    return ExtractionResult(final, len(candidates),
                            {entity: len(group) for entity, group in anchored.items()}, counts, marks,
                            {family: sum(f.family == family for f in candidates) for family in FAMILIES},
                            {family: sum(f.family == family for f in final) for family in FAMILIES})
