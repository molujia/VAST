"""Native observations. No root/type/path labels are accepted by model inputs."""
from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass, fields, replace
from math import floor, isfinite
import hashlib
import json
from typing import Mapping


@dataclass(frozen=True)
class Entity:
    id: str
    category: str
    candidate: bool = True


@dataclass(frozen=True)
class NumericSeries:
    modality: str
    owner: str
    channel: str
    times: tuple[float, ...]
    values: tuple[float, ...]
    observed: tuple[bool, ...] = ()

    def __post_init__(self):
        if self.modality not in ('metric', 'trace'):
            raise ValueError('numeric modality must be metric or trace')
        if len(self.times) != len(self.values) or (self.observed and len(self.observed) != len(self.times)):
            raise ValueError('native time/value/mask lengths differ')
        if any(not isfinite(t) for t in self.times) or any(b < a for a,b in zip(self.times,self.times[1:])):
            raise ValueError('native times must be finite and ordered')
        mask = self.observed or (True,) * len(self.times)
        mask = tuple(bool(m and isfinite(v)) for m,v in zip(mask,self.values))
        object.__setattr__(self, 'observed', mask)
        object.__setattr__(self, 'values', tuple(float(v) if m else 0. for v,m in zip(self.values,mask)))


@dataclass(frozen=True)
class LogRecord:
    owner: str
    time: float
    text: str
    frequency: int = 1

    def __post_init__(self):
        if not isfinite(self.time) or self.frequency <= 0:
            raise ValueError('invalid log time/frequency')


@dataclass(frozen=True)
class Relation:
    id: str
    source: str
    target: str
    kind: str
    start: float
    end: float
    # An impact/reverse relation can point to the original directed call owner.
    owner: str | None = None
    response: bool = False
    active_intervals: tuple[tuple[float,float],...] = ()

    def __post_init__(self):
        if not isfinite(self.start) or not isfinite(self.end) or self.start > self.end:
            raise ValueError('invalid relation interval')
        points, continuous = [], []
        indexable = isinstance(self.active_intervals, tuple)
        for interval in self.active_intervals:
            a, b = interval
            if not isfinite(a) or not isfinite(b) or a > b or a < self.start or b > self.end:
                raise ValueError('invalid active relation subinterval')
            indexable = indexable and isinstance(interval, tuple)
            if indexable:
                if a == b:
                    points.append(a)
                else:
                    continuous.append((a, b))
        if not indexable:
            # Historical callers may supply lists despite the tuple annotation.
            # Keep their live mutation semantics; never cache mutable support.
            object.__setattr__(self, '_active_interval_index', None)
            return
        points.sort()
        continuous.sort(key=lambda pair: pair[0])
        starts, prefix_ends = [], []
        latest_end = self.start
        for a, b in continuous:
            starts.append(a)
            latest_end = max(latest_end, b)
            prefix_ends.append(latest_end)
        # Derived, immutable, and deliberately not a dataclass field: the raw
        # observation order, equality/hash/repr, and JSON boundary stay intact.
        object.__setattr__(self, '_active_interval_index',
                           (tuple(points), tuple(starts), tuple(prefix_ends)))

    def __getstate__(self):
        # Do not duplicate derived indexes in pickle payloads either.
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        # Also accepts the public-only state produced before indexing existed.
        self.__post_init__()

    def is_active(self, start: float, end: float) -> bool:
        # Continuous support and bins are half-open; degenerate observed call
        # intervals are points, assigned to the bin beginning at that point.
        if not start < end:
            return False
        if not self.active_intervals:
            return bool(start <= self.start < end if self.start == self.end
                        else start < self.end and end > self.start)
        if self._active_interval_index is None:
            return any(start <= a < end if a == b else start < b and end > a
                       for a, b in self.active_intervals)
        points, starts, prefix_ends = self._active_interval_index
        point_index = bisect_left(points, start)
        if point_index < len(points) and points[point_index] < end:
            return True
        # Only supports beginning before the query end can intersect it. The
        # prefix maximum preserves earlier long/overlapping intervals.
        interval_index = bisect_left(starts, end)
        return bool(interval_index > 0 and prefix_ends[interval_index - 1] > start)


@dataclass(frozen=True)
class MaskedTarget:
    modality: str
    owner: str
    channel: str
    time: float
    value: float | str
    frequency: int = 1


@dataclass(frozen=True)
class ObservableCase:
    case_id: str
    entities: tuple[Entity, ...]
    reference_start: float
    incident_start: float
    incident_end: float
    numeric: tuple[NumericSeries, ...] = ()
    logs: tuple[LogRecord, ...] = ()
    relations: tuple[Relation, ...] = ()

    def __post_init__(self):
        if not self.reference_start <= self.incident_start < self.incident_end:
            raise ValueError('windows must be ordered: reference <= incident start < end')
        ids = {x.id for x in self.entities}
        if len(ids) != len(self.entities) or not self.candidate_ids:
            raise ValueError('catalog must contain unique IDs and at least one candidate')
        owners = {r.id for r in self.relations}
        if len(owners) != len(self.relations):
            raise ValueError('relation owner IDs must be unique')
        for r in self.relations:
            if r.source not in ids or r.target not in ids or (r.owner is not None and r.owner not in owners):
                raise ValueError('relation owner/endpoints absent from catalog')
        for s in self.numeric:
            if s.owner not in (ids if s.modality == 'metric' else owners):
                raise ValueError('numeric owner absent from catalog/edge table')
        if any(log.owner not in ids for log in self.logs):
            raise ValueError('log owner absent from catalog')

    @property
    def candidate_ids(self):
        return tuple(e.id for e in self.entities if e.candidate)

    def bin_index(self, timestamp: float, delta: float) -> int:
        if delta <= 0 or not isfinite(delta):
            raise ValueError('delta must be finite and positive')
        return floor((timestamp-self.reference_start)/delta)

    def shift_time(self, seconds: float) -> ObservableCase:
        return replace(self, reference_start=self.reference_start+seconds,
                       incident_start=self.incident_start+seconds,incident_end=self.incident_end+seconds,
                       numeric=tuple(replace(s,times=tuple(t+seconds for t in s.times)) for s in self.numeric),
                       logs=tuple(replace(l,time=l.time+seconds) for l in self.logs),
                       relations=tuple(replace(r,start=r.start+seconds,end=r.end+seconds,
                                               active_intervals=tuple((a+seconds,b+seconds) for a,b in r.active_intervals))
                                       for r in self.relations))

    def mask_blocks(self, blocks: set[tuple[str,str,int]], *, delta: float):
        """Delete raw values before all learned/derived quantities are recomputed.

        Only telemetry is maskable; structural relation intervals are observed
        call evidence, never alarms/onsets derived from hidden target values.
        The schema deliberately has no field for derived alarms or root metadata.
        """
        targets, series = [], []
        for s in self.numeric:
            unit = 'node' if s.modality == 'metric' else 'edge'
            mask = list(s.observed)
            for j,(t,v,m) in enumerate(zip(s.times,s.values,s.observed)):
                if m and (unit,s.owner,self.bin_index(t,delta)) in blocks:
                    targets.append(MaskedTarget(s.modality,s.owner,s.channel,t,v))
                    mask[j] = False
            series.append(replace(s,observed=tuple(mask)))
        logs = []
        for log in self.logs:
            if ('node',log.owner,self.bin_index(log.time,delta)) in blocks:
                targets.append(MaskedTarget('log',log.owner,'text',log.time,log.text,log.frequency))
            else:
                logs.append(log)
        return replace(self,numeric=tuple(series),logs=tuple(logs)), tuple(targets)


class LabelStore:
    """Curator-owned boundary: query is the only training-facing label operation."""
    def __init__(self, roots: Mapping[str,tuple[str,...]], budget: int, max_roots: int | None = None):
        if budget < 0 or (max_roots is not None and max_roots < 1):
            raise ValueError('invalid budget/root cap')
        self._roots, self.budget, self.max_roots = dict(roots), budget, max_roots
        self._revealed: dict[str,frozenset[str]] = {}
        self._accessed: set[str] = set()
        self._blocked_reason: str | None = None
        self._manifest_hash=hashlib.sha256(json.dumps(self._roots,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @property
    def queried(self):
        return frozenset(self._revealed)

    @property
    def spent(self):
        return len(self._accessed)

    @property
    def pending(self):
        return frozenset(self._accessed)-frozenset(self._revealed)

    def state_dict(self):
        return {'schema':'vast-v3-query-access/1','budget':self.budget,'max_roots':self.max_roots,
                'manifest_hash':self._manifest_hash,'accessed':sorted(self._accessed),
                'revealed':{k:sorted(v) for k,v in self._revealed.items()},'blocked_reason':self._blocked_reason}

    def load_state_dict(self,state):
        if (state.get('schema')!='vast-v3-query-access/1' or state.get('budget')!=self.budget
                or state.get('max_roots')!=self.max_roots or state.get('manifest_hash')!=self._manifest_hash):
            raise ValueError('query-access ledger identity mismatch')
        accessed=set(state['accessed']); revealed={k:frozenset(v) for k,v in state['revealed'].items()}
        if (len(accessed)!=len(state['accessed']) or len(accessed)>self.budget or not accessed<=set(self._roots)
                or not set(revealed)<=accessed or any(v!=frozenset(self._roots[k]) for k,v in revealed.items())):
            raise ValueError('invalid query-access ledger')
        self._accessed=accessed; self._revealed=revealed; self._blocked_reason=state['blocked_reason']

    def query(self, case: ObservableCase, *, persist=None) -> frozenset[str]:
        if self._blocked_reason:
            raise ValueError('label store blocked: '+self._blocked_reason)
        if case.case_id in self._revealed:
            return self._revealed[case.case_id]
        if self.pending and case.case_id not in self.pending:
            raise ValueError('pending reserved root access must be resolved before another query')
        if case.case_id not in self._accessed and self.spent >= self.budget:
            raise ValueError('annotation budget exhausted')
        if case.case_id not in self._roots:
            raise ValueError('root annotation absent from curator store')
        # Write-ahead reservation: an interrupted answer is still a paid case,
        # and recovery may resolve only that same reservation before proceeding.
        self._accessed.add(case.case_id)
        if persist is not None:
            persist(self.state_dict())
        roots = frozenset(self._roots[case.case_id])
        if not roots or not roots <= set(case.candidate_ids):
            self._blocked_reason='complete root set must be nonempty and inside candidate catalog'
            if persist is not None:
                persist(self.state_dict())
            raise ValueError(self._blocked_reason)
        if self.max_roots is not None and len(roots) > self.max_roots:
            self._blocked_reason='queried root count contradicts declared expert prior'
            if persist is not None:
                persist(self.state_dict())
            raise ValueError(self._blocked_reason)
        self._revealed[case.case_id] = roots
        if persist is not None:
            persist(self.state_dict())
        return roots
