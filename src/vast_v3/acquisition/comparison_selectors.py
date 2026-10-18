"""Approved frozen-pool controls; no labels, model updates, or query side effects.

Definitions and scope: docs/acquisition-controls.md.
Descriptors must come from the same permitted, frozen observation pipeline.
These selectors propose complete case IDs; a caller must debit its query ledger.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
import json
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .propcover import CaseTokens, partition_cases, query_count, seeded_case_priority
from ..data.preprocessing import Preprocessor
from ..data.schema import ObservableCase
from ..encoders.fragments import EventRef, Fragment
from ..encoders.frontends import FrontendOutput


@dataclass(frozen=True)
class ComparisonSelectionResult:
    method: str
    case_ids: tuple[str, ...]
    gains: tuple[float, ...]
    base_objective: float | None
    final_objective: float | None
    objective_direction: str
    reference_count: int
    representation_version: str
    # Gains always address greedy_case_ids, not the post-swap final list.
    greedy_case_ids: tuple[str, ...] = ()
    swaps: tuple[tuple[str, str, float], ...] = ()


class _Selector:
    def _initialize_pool(self, ids, seed, priority):
        self.case_ids = tuple(sorted(ids))
        self.priority = seeded_case_priority(self.case_ids, seed, priority)

    def select_budgeted(self, labeled_ids: Iterable[str], *, total_budget: int,
                        batch_size: int = 5) -> ComparisonSelectionResult:
        labeled = tuple(labeled_ids)
        if not isinstance(total_budget, int) or total_budget < 0:
            raise ValueError("total budget must be a nonnegative integer")
        partition_cases(self.case_ids, labeled)
        if len(labeled) > total_budget:
            raise ValueError("already queried case count exceeds total budget")
        return self.select(labeled, batch_size=batch_size, remaining_budget=total_budget - len(labeled))


class TimeSpread(_Selector):
    """One-dimensional temporal medoid initialization followed by farthest-first."""
    def __init__(self, incident_starts: Mapping[str, float | None], *, seed: int = 0,
                 priority: Sequence[str] | None = None):
        self._initialize_pool(incident_starts, seed, priority)
        values = {key: (float(value) if value is not None and math.isfinite(value) else None)
                  for key, value in incident_starts.items()}
        valid = [value for value in values.values() if value is not None]
        minimum, maximum = (min(valid), max(valid)) if valid else (0., 0.)
        self.times = {key: (None if value is None else
                           (value - minimum) / (maximum - minimum) if maximum > minimum else 0.)
                      for key, value in values.items()}

    def select(self, labeled_ids=(), unlabeled_ids=None, *, batch_size=5, remaining_budget=30):
        labeled, unlabeled = partition_cases(self.case_ids, labeled_ids, unlabeled_ids)
        count = query_count(batch_size, remaining_budget, len(unlabeled))
        centers = [self.times[key] for key in labeled if self.times[key] is not None]
        selected, gains = [], []
        for step in range(count):
            candidates = [key for key in self.priority if key in unlabeled]
            valid = [key for key in candidates if self.times[key] is not None]
            if not labeled and step == 0 and valid:
                reference = [value for value in self.times.values() if value is not None]
                chosen = min(valid, key=lambda key: math.fsum(abs(self.times[key] - x) for x in reference))
                gain = 0.  # Initialization is not a farthest-first distance gain.
            else:
                distances = {key: (min(abs(self.times[key] - value) for value in centers)
                                   if centers and self.times[key] is not None else 0.) for key in candidates}
                chosen = max(candidates, key=distances.__getitem__)
                gain = distances[chosen]
            selected.append(chosen)
            gains.append(gain)
            unlabeled.remove(chosen)
            if self.times[chosen] is not None:
                centers.append(self.times[chosen])
        return ComparisonSelectionResult("time-spread", tuple(selected), tuple(gains), None, None,
                                         "farthest-first-distance", sum(x is not None for x in self.times.values()),
                                         "frozen-incident-start", tuple(selected))


@dataclass(frozen=True)
class SymptomCaseTokens:
    case_id: str
    tokens: Tensor
    kinds: tuple[str, ...]
    provenance: tuple[str, ...]
    encoder_version: str
    observed: tuple[bool, ...] = ()

    def __post_init__(self):
        if not self.case_id or not self.encoder_version:
            raise ValueError("case ID and encoder version must be nonempty")
        if self.tokens.ndim != 2 or not self.tokens.is_floating_point() or self.tokens.shape[1] < 1:
            raise ValueError("symptom tokens must be floating [tokens,width] vectors")
        if len(self.tokens) != len(self.kinds) or len(self.tokens) != len(self.provenance):
            raise ValueError("token metadata lengths differ")
        if any(kind not in ("node", "edge") for kind in self.kinds):
            raise ValueError("symptom category must be node or edge")
        if any(not item for item in self.provenance):
            raise ValueError("token provenance must be nonempty")
        if self.observed and len(self.observed) != len(self.tokens):
            raise ValueError("observed mask length differs from tokens")


def symptom_case_from_frontend(case_id: str, front: FrontendOutput, *, entity_ids: Sequence[str],
                               encoder_version: str) -> SymptomCaseTokens:
    """Read only ordinary pre-aggregation S and directed-owner observed E.

    Identity strings are provenance metadata, never token coordinates or kernels.
    Recovery, fused h, topology, SCOPE, and root predictions are not accessed.
    """
    rows, kinds, provenance = [], [], []
    for kind, states, mask, owners in (("node", front.local, front.local_mask, entity_ids),
                                       ("edge", front.edge_states, front.edge_mask, front.edge_owner_ids)):
        if states.ndim != 3 or mask.shape != states.shape[:2] or len(owners) != states.shape[0]:
            raise ValueError("frontend observed state/mask/owner shapes disagree")
        if len(set(owners)) != len(owners):
            raise ValueError("frontend owner provenance must be unique")
        for owner, bin_index in mask.nonzero().tolist():
            rows.append(states[owner, bin_index].detach().cpu())
            kinds.append(kind)
            provenance.append(json.dumps((kind, owners[owner], bin_index), separators=(",", ":")))
    tokens = torch.stack(rows) if rows else front.local.detach().cpu().new_empty((0, front.local.shape[-1]))
    return SymptomCaseTokens(case_id, tokens, tuple(kinds), tuple(provenance), encoder_version)


class _CoverageSelector(_Selector):
    @property
    def weights(self):
        return self._weights.clone()

    def _coverage_vector(self, ids):
        result = torch.zeros(self.reference_count, dtype=torch.float64)
        for key in set(ids):
            if key not in self._maxima:
                raise ValueError("unknown case ID in coverage request")
            result = torch.maximum(result, self._maxima[key])
        return result

    def coverage(self, ids):
        return float(self._coverage_vector(ids) @ self._weights)

    def objective(self, selected_ids, *, labeled_ids=()):
        labeled = set(labeled_ids)
        return self.coverage(labeled | set(selected_ids)) - self.coverage(labeled)

    def select(self, labeled_ids=(), unlabeled_ids=None, *, batch_size=5, remaining_budget=30):
        labeled, unlabeled = partition_cases(self.case_ids, labeled_ids, unlabeled_ids)
        count = query_count(batch_size, remaining_budget, len(unlabeled))
        baseline = self._coverage_vector(labeled)
        initial = float(baseline @ self._weights)
        selected, gains = [], []
        for _ in range(count):
            candidates = [key for key in self.priority if key in unlabeled]
            gain = {key: float((torch.maximum(baseline, self._maxima[key]) - baseline) @ self._weights)
                    for key in candidates}
            chosen = max(candidates, key=gain.__getitem__)
            selected.append(chosen)
            gains.append(gain[chosen])
            unlabeled.remove(chosen)
            baseline = torch.maximum(baseline, self._maxima[chosen])
        return ComparisonSelectionResult(self.method, tuple(selected), tuple(gains), initial,
                                         float(baseline @ self._weights), "maximize-coverage",
                                         self.reference_count, self.representation_version, tuple(selected))


class SymptomCover(_CoverageSelector):
    """Type-restricted, category/case/token-balanced local facility location."""
    method = "symptom-cover"

    def __init__(self, cases: Iterable[SymptomCaseTokens], *, seed=0, priority=None):
        cases = tuple(cases)
        self._initialize_pool((case.case_id for case in cases), seed, priority)
        versions = {case.encoder_version for case in cases}
        if len(versions) > 1:
            raise ValueError("cannot mix encoder versions")
        if len({case.tokens.shape[1] for case in cases}) > 1:
            raise ValueError("symptom token widths must agree")
        prepared, kind_cases, counts = {}, defaultdict(set), {}
        for case in cases:
            rows = case.tokens.detach().to(device="cpu", dtype=torch.float64).clone()
            mask = case.observed or (True,) * len(rows)
            unique = {}
            for row, kind, provenance, observed in zip(rows, case.kinds, case.provenance, mask):
                if not observed or not bool(torch.isfinite(row).all()) or float(row.norm()) <= 1e-12:
                    continue
                key = kind, provenance
                if key in unique and not torch.equal(unique[key], row):
                    raise ValueError("one symptom provenance has conflicting values")
                unique[key] = row
            prepared[case.case_id] = {key: row / row.norm() for key, row in sorted(unique.items())}
            counts[case.case_id] = Counter(kind for kind, _ in unique)
            for kind in counts[case.case_id]:
                kind_cases[kind].add(case.case_id)
        keys, vectors, weights = [], [], []
        for key in self.case_ids:
            for (kind, provenance), vector in prepared[key].items():
                keys.append((key, kind, provenance))
                vectors.append(vector)
                weights.append(1. / (len(kind_cases) * len(kind_cases[kind]) * counts[key][kind]))
        self.occurrence_keys = tuple(keys)
        self.reference_count = len(keys)
        self.representation_version = next(iter(versions), "empty-pool")
        self._weights = torch.tensor(weights, dtype=torch.float64)
        self._maxima = {}
        for key in self.case_ids:
            maxima = torch.zeros(len(keys), dtype=torch.float64)
            for kind in kind_cases:
                target_indices = [i for i, (_, category, _) in enumerate(keys) if category == kind]
                candidate = [v for (category, _), v in prepared[key].items() if category == kind]
                if not candidate:
                    continue
                candidate = torch.stack(candidate)
                for offset in range(0, len(target_indices), 2048):
                    indices = target_indices[offset:offset + 2048]
                    reference = torch.stack([vectors[i] for i in indices])
                    maxima[indices] = ((1 + reference @ candidate.T) / 2).clamp(0, 1).amax(1)
            self._maxima[key] = maxima


class PooledMedoid(_Selector):
    """Float64 conditional k-medoids with fixed L and bounded single swaps."""
    def __init__(self, cases: Iterable[CaseTokens], *, seed=0, priority=None):
        cases = tuple(cases)
        self._initialize_pool((case.case_id for case in cases), seed, priority)
        versions = {case.encoder_version for case in cases}
        if len(versions) > 1:
            raise ValueError("cannot mix encoder versions")
        if len({case.tokens.shape[1] for case in cases}) > 1:
            raise ValueError("SCOPE token widths must agree")
        self.representation_version = next(iter(versions), "empty-pool")
        self._pooled = {}
        for case in cases:
            rows = case.tokens.detach().to(device="cpu", dtype=torch.float64).clone()
            if not bool(torch.isfinite(rows).all()):
                raise ValueError("SCOPE tokens must be finite")
            if len(rows) and not torch.allclose(rows.norm(dim=1), torch.ones(len(rows), dtype=torch.float64),
                                              rtol=1e-5, atol=1e-5):
                raise ValueError("Pooled-medoid requires real unit SCOPE tokens")
            unique = {}
            for row, provenance, family in zip(rows, case.provenance, case.families):
                if provenance in unique and (unique[provenance][1] != family
                                             or not torch.equal(unique[provenance][0], row)):
                    raise ValueError("one fragment provenance has conflicting tokens or families")
                unique[provenance] = row, family
            mean = torch.stack([unique[key][0] for key in sorted(unique)]).mean(0) if unique else None
            self._pooled[case.case_id] = mean / mean.norm() if mean is not None and mean.norm() > 1e-12 else None
        self.valid_ids = tuple(key for key in self.case_ids if self._pooled[key] is not None)
        self.reference_count = len(self.valid_ids)
        self._distances = {}
        reference = torch.stack([self._pooled[key] for key in self.valid_ids]) if self.valid_ids else None
        for key in self.case_ids:
            vector = self._pooled[key]
            self._distances[key] = ((1 - reference @ vector) / 2).clamp(0, 1) if reference is not None and vector is not None else torch.ones(self.reference_count, dtype=torch.float64)

    @property
    def pooled(self):
        return {key: value.clone() if value is not None else None for key, value in self._pooled.items()}

    def objective(self, center_ids):
        nearest = torch.ones(self.reference_count, dtype=torch.float64)
        for key in set(center_ids):
            if key not in self._distances:
                raise ValueError("unknown case ID in medoid objective")
            nearest = torch.minimum(nearest, self._distances[key])
        return float(nearest.mean()) if self.reference_count else 0.

    def assignments(self, center_ids):
        """Nearest-medoid clusters for valid target cases; no invented centers."""
        centers = set(center_ids)
        if not centers <= set(self.case_ids):
            raise ValueError("unknown case ID in medoid assignment")
        ordered = [key for key in self.priority if key in centers and self._pooled[key] is not None]
        return {target: (min(ordered, key=lambda key: float(self._distances[key][index])) if ordered else None)
                for index, target in enumerate(self.valid_ids)}

    def select(self, labeled_ids=(), unlabeled_ids=None, *, batch_size=5, remaining_budget=30):
        labeled, unlabeled = partition_cases(self.case_ids, labeled_ids, unlabeled_ids)
        count = query_count(batch_size, remaining_budget, len(unlabeled))
        initial = current = self.objective(labeled)
        selected, gains = [], []
        for _ in range(count):
            candidates = [key for key in self.priority if key in unlabeled]
            objectives = {key: self.objective(labeled | set(selected) | {key}) for key in candidates}
            chosen = min(candidates, key=objectives.__getitem__)
            selected.append(chosen)
            gains.append(current - objectives[chosen])
            current = objectives[chosen]
            unlabeled.remove(chosen)
        greedy_ids = tuple(selected)
        swaps = []
        # Recompute each removal before testing an insertion. Never treat an old
        # removed center's cached nearest distance as still available.
        for _ in range(100):
            best = None
            best_objective = current
            for removed in self.priority:
                if removed not in selected:
                    continue
                retained = labeled | (set(selected) - {removed})
                for inserted in self.priority:
                    if inserted not in unlabeled:
                        continue
                    objective = self.objective(retained | {inserted})
                    if objective < best_objective:
                        best = removed, inserted
                        best_objective = objective
            if best is None:
                break
            removed, inserted = best
            selected[selected.index(removed)] = inserted
            unlabeled.remove(inserted)
            unlabeled.add(removed)
            swaps.append((removed, inserted, current - best_objective))
            current = best_objective
        return ComparisonSelectionResult("pooled-medoid", tuple(selected), tuple(gains), initial, current,
                                         "minimize-mean-distance", self.reference_count,
                                         self.representation_version, greedy_ids, tuple(swaps))


@dataclass(frozen=True)
class MotifCase:
    case_id: str
    codes: frozenset[str]
    preprocessing_identity: str

    def __post_init__(self):
        if not self.case_id or not self.preprocessing_identity or any(not code for code in self.codes):
            raise ValueError("motif case, code and preprocessing identities must be nonempty")


class MotifCover(_CoverageSelector):
    """Greedy exact-code set coverage with no frequency weighting or clustering."""
    method = "motif-cover"

    def __init__(self, cases: Iterable[MotifCase], *, seed=0, priority=None):
        cases = tuple(cases)
        self._initialize_pool((case.case_id for case in cases), seed, priority)
        versions = {case.preprocessing_identity for case in cases}
        if len(versions) > 1:
            raise ValueError("cannot mix frozen preprocessing identities")
        codes = tuple(sorted({code for case in cases for code in case.codes}))
        self.reference_count = len(codes)
        self.representation_version = next(iter(versions), "empty-pool")
        self._weights = torch.ones(len(codes), dtype=torch.float64)
        self._maxima = {case.case_id: torch.tensor([float(code in case.codes) for code in codes], dtype=torch.float64)
                        for case in cases}


def delay_bucket(source: EventRef, target: EventRef, *, delta: float) -> str:
    """Bin only a complete supported physical-delay interval, never its midpoint."""
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    lower, upper = (target.start - source.end) / delta, (target.end - source.start) / delta
    for left, right, name in ((0., 1., "[0,1)"), (1., 2., "[1,2)"),
                              (2., 4., "[2,4)"), (4., 8., "[4,8]")):
        if lower >= left and (upper <= right if right == 8 else upper < right):
            return name
    return "unknown"


def motif_code(case: ObservableCase, fragment: Fragment, preprocessing: Preprocessor, *,
               metric_channels: Sequence[str], trace_channels: Sequence[str]) -> str:
    """Exact anonymous code on one already sampled fragment; no resampling/refit.

    Channel keys are the frozen modality vocabularies, including unsupported keys
    whose values stay unknown. Trace attributes bind to the original owner and
    endpoint event bins before any anonymous vertex/edge canonicalization.
    """
    if not 1 <= len(fragment.edges) <= 3 or not 1 <= len(fragment.events) <= 4:
        raise ValueError("motif requires a real one-to-three-edge fragment")
    if not math.isfinite(preprocessing.delta) or preprocessing.delta <= 0:
        raise ValueError("frozen delta must be finite and positive")
    if any(event.entity >= len(case.entities) for event in fragment.events):
        raise ValueError("fragment entity is outside the case")
    if any(edge.owner >= len(case.relations) for edge in fragment.edges):
        raise ValueError("fragment directed owner is outside the case")
    channels = {"metric": tuple(sorted(set(metric_channels))), "trace": tuple(sorted(set(trace_channels)))}
    categories = {entity.id: entity.category for entity in case.entities}
    samples = defaultdict(list)
    for series in case.numeric:
        for timestamp, value, observed in zip(series.times, series.values, series.observed):
            if observed and math.isfinite(value) and case.reference_start <= timestamp < case.incident_end:
                samples[(series.modality, series.owner, series.channel,
                         case.bin_index(timestamp, preprocessing.delta))].append(value)
    frequencies = defaultdict(int)
    for record in case.logs:
        if case.reference_start <= record.time < case.incident_end:
            frequencies[(record.owner, case.bin_index(record.time, preprocessing.delta))] += record.frequency

    def band(category, channel, values):
        stat = preprocessing.stats.get((category, channel))
        if not values or stat is None or stat.observations <= 0:
            return "unknown"
        z, supported = preprocessing.transform(category, channel, math.fsum(values) / len(values))
        return "unknown" if not supported else "low" if z < -1 else "high" if z > 1 else "reference-band"

    def numeric(modality, owner, category, bin_index):
        return tuple((channel, band(category, channel, samples.get((modality, owner, channel, bin_index), ())))
                     for channel in channels[modality])

    node_attributes = []
    for event in fragment.events:
        entity = case.entities[event.entity]
        frequency = frequencies.get((entity.id, event.bin))
        log_band = band(entity.category, "__log_frequency__", (math.log1p(frequency),) if frequency else ())
        node_attributes.append((entity.category, numeric("metric", entity.id, entity.category, event.bin), log_band))
    edge_attributes = []
    for edge in fragment.edges:
        source, target = fragment.events[edge.source], fragment.events[edge.target]
        owner = case.relations[edge.owner] if edge.owner >= 0 else None
        category = categories[owner.source] + "->" + categories[owner.target] if owner else ""
        owner_id = owner.id if owner else ""
        endpoint_bins = (source.bin,) if source.bin == target.bin else (source.bin, target.bin)
        trace = tuple(numeric("trace", owner_id, category, bin_index) for bin_index in endpoint_bins)
        edge_attributes.append((edge.relation, bool(edge.response), delay_bucket(source, target, delta=preprocessing.delta), trace))
    candidates = []
    for order in permutations(range(len(fragment.events))):
        positions = {old: new for new, old in enumerate(order)}
        equality, temporal = [], []
        for i in order:
            for j in order:
                left, right = fragment.events[i], fragment.events[j]
                equality.append(int(left.entity == right.entity))
                temporal.append("before" if left.end < right.start else "after" if right.end < left.start else "unknown")
        edges = sorted((positions[edge.source], positions[edge.target], *edge_attributes[index])
                       for index, edge in enumerate(fragment.edges))
        candidates.append(json.dumps(("motif-v1", tuple(node_attributes[i] for i in order),
                                      tuple(equality), tuple(temporal), edges), separators=(",", ":")))
    return min(candidates)


def motif_case_from_fragments(case: ObservableCase, fragments: Iterable[Fragment], preprocessing: Preprocessor,
                              *, metric_channels: Sequence[str], trace_channels: Sequence[str]) -> MotifCase:
    """Consume exactly the main pipeline's retained fragment inventory."""
    codes = frozenset(motif_code(case, fragment, preprocessing, metric_channels=metric_channels,
                                trace_channels=trace_channels) for fragment in fragments)
    return MotifCase(case.case_id, codes, preprocessing.identity)
