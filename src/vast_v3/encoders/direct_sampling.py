"""Exact population indexing with bounded, uniform, family-balanced sampling.

The exhaustive extractor remains a small-graph reference. This version keeps
its eligible population and two-stage distribution, not its realized hash
priorities for a particular seed. Only chosen integer ranks are recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Mapping, Sequence

from .fragments import (EventRef, FragmentEdge, Fragment, ExtractionResult,
                        FAMILIES, _link_key, _seed, structural_family)
from .fragment_index import FragmentIndex
from .fragment_cache import DirectSamplingCache


SCHEMA = 'vast-v3-exact-direct/1'


@dataclass(frozen=True)
class DirectExtractionResult(ExtractionResult):
    sampling_audit: dict


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _quotas(populations, cap, seed):
    """The exact round-robin shuffled quota rule of the exhaustive reference."""
    quotas = dict.fromkeys(FAMILIES, 0)
    rng = random.Random(seed)
    remaining = min(cap, sum(populations.values()))
    while remaining:
        available = [family for family in FAMILIES if quotas[family] < populations[family]]
        rng.shuffle(available)
        for family in available:
            if not remaining:
                break
            quotas[family] += 1
            remaining -= 1
    return quotas


def _uniform_ranks(size, count, seed):
    """Floyd's uniform k-subset algorithm: O(k), no O(population) allocation."""
    rng = random.Random(seed)
    chosen = set()
    for last in range(size - count, size):
        rank = rng.randrange(last + 1)
        chosen.add(last if rank in chosen else rank)
    return sorted(chosen)


def _inventory(events, links, delta, owner_keys):
    by_provenance = {}
    for event in events:
        previous = by_provenance.setdefault(event.provenance, event)
        if previous != event:
            raise ValueError('one observation provenance has conflicting event records')
    unique = tuple(by_provenance[key] for key in sorted(by_provenance))
    positions = {event.provenance: i for i, event in enumerate(unique)}
    eligible = {}
    for edge in links:
        if max(edge.source, edge.target) >= len(events):
            raise ValueError('observed relation endpoint is outside event inventory')
        a, b = events[edge.source], events[edge.target]
        if b.end < a.start or b.start - a.end > 8 * delta:
            continue
        moved = FragmentEdge(positions[a.provenance], positions[b.provenance],
                             edge.relation, edge.owner, edge.response)
        eligible[_link_key(moved, unique, owner_keys)] = moved
    return unique, tuple(eligible[key] for key in sorted(eligible))


def _fragment(events, edges, indices, owner_keys):
    indices = tuple(sorted(indices))
    members = sorted({v for index in indices for v in (edges[index].source, edges[index].target)})
    local = {v: i for i, v in enumerate(members)}
    selected_events = tuple(events[v] for v in members)
    selected_edges = tuple(FragmentEdge(local[edges[i].source], local[edges[i].target],
                                       edges[i].relation, edges[i].owner, edges[i].response)
                           for i in indices)
    evidence = [sorted(e.provenance for e in selected_events),
                sorted(_link_key(e, selected_events, owner_keys) for e in selected_edges)]
    # Match the reference provenance bytes exactly (not the audit JSON format).
    provenance = hashlib.sha256(json.dumps(evidence, separators=(',', ':')).encode()).hexdigest()
    return Fragment(selected_events, selected_edges,
                    structural_family(selected_events, selected_edges), provenance)


def _cached_result(payload, events, edges, owner_keys, anchor_keys, header, sampling_audit):
    """Rebind cached edge-index skeletons to the current view's observations."""
    audit = payload['sampling_audit']
    for key, value in header.items():
        if key not in audit or _json(audit[key]) != _json(value):
            raise ValueError(f'cached direct sampling audit differs at {key}')
    if audit.get('audit_sha256') != _hash({k: v for k, v in audit.items() if k != 'audit_sha256'}):
        raise ValueError('cached direct sampling audit checksum differs')
    if sampling_audit is not None and _json(sampling_audit) != _json(audit):
        raise ValueError('direct sampling audit does not match the cached sample')
    final = tuple(_fragment(events, edges, tuple(indices), owner_keys)
                  for indices in payload['final_edges'])
    if [fragment.provenance for fragment in final] != audit['final_provenance']:
        raise ValueError('cached direct sampling audit does not match rebound fragments')
    current_anchors = {key: anchor for anchor, key in anchor_keys.items()}
    candidates = {}; retained = {}
    for entry in audit['anchors']:
        if entry['anchor_key'] not in current_anchors:
            raise ValueError('cached direct sampling audit has an absent anchor')
        anchor = current_anchors[entry['anchor_key']]
        candidates[anchor] = sum(entry['populations'].values())
        retained[anchor] = sum(entry['quotas'].values())
    families = {family: audit['family_counts'][family] for family in FAMILIES}
    return DirectExtractionResult(final, sum(families.values()), candidates, retained, {},
        families, {f: sum(x.family == f for x in final) for f in FAMILIES}, audit)


def extract_fragments_direct(
    events: Sequence[EventRef], links: Sequence[FragmentEdge], *, delta: float,
    seed: int = 0, per_anchor_cap: int = 8, case_cap: int = 128, max_edges: int = 3,
    owner_keys: Mapping[int, str] | None = None, sampling_audit: dict | None = None,
    sampling_cache: DirectSamplingCache | None = None,
) -> DirectExtractionResult:
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be finite and positive')
    if type(seed) is not int or any(type(n) is not int or n < 0 for n in (per_anchor_cap, case_cap)):
        raise ValueError('seed and fragment caps must be integers; caps must be nonnegative')
    if type(max_edges) is not int or not 1 <= max_edges <= 3:
        raise ValueError('max_edges must lie in 1..3')
    if sampling_audit is not None and (not isinstance(sampling_audit, dict)
                                      or sampling_audit.get('schema') != SCHEMA):
        raise ValueError('unsupported direct sampling audit')
    events, edges = _inventory(events, links, delta, owner_keys)
    anchor_keys = {}
    for event in events:
        anchor_keys.setdefault(event.entity, []).append(event.provenance)
    # Preserve the reference anchor seed identity, independently of numeric IDs.
    anchor_keys = {a: hashlib.sha256(json.dumps(sorted(names)).encode()).hexdigest()
                   for a, names in anchor_keys.items()}
    inventory_hash = _hash({'events': [(e.provenance, anchor_keys[e.entity]) for e in events],
                            'edges': [_link_key(e, events, owner_keys) for e in edges]})
    audit = {'schema': SCHEMA, 'seed': seed, 'inventory_sha256': inventory_hash,
             'caps': {'per_anchor': per_anchor_cap, 'case': case_cap, 'max_edges': max_edges}}
    if sampling_audit is not None:
        for key, value in audit.items():
            if key not in sampling_audit or _json(sampling_audit[key]) != _json(value):
                raise ValueError(f'direct sampling audit differs at {key}')
    # Eligibility, duplicate checks, stable owner identity and entity equality
    # are validated before lookup. Time/category/bin are deliberately rebound
    # from this view, never reused from a cached Fragment or model activation.
    cache_key = _hash(audit) if sampling_cache is not None else None
    if sampling_cache is not None:
        cached = sampling_cache.get(cache_key)
        if cached is not None:
            return _cached_result(cached, events, edges, owner_keys, anchor_keys, audit, sampling_audit)
    index = FragmentIndex(events, edges, max_edges=max_edges)
    anchors = sorted(index.anchor_family_counts, key=anchor_keys.__getitem__)
    requests = {}; anchor_audits = []; retained_counts = {}
    for anchor in anchors:
        populations = {f: index.anchor_family_counts[anchor].get(f, 0) for f in FAMILIES}
        stage = 'anchor/' + anchor_keys[anchor]
        quotas = _quotas(populations, per_anchor_cap, _seed(seed, stage))
        ranks = {f: _uniform_ranks(populations[f], quotas[f], _seed(seed, stage + '/' + f + '/direct-ranks'))
                 for f in FAMILIES}
        requests.update({(anchor, f): values for f, values in ranks.items() if values})
        retained_counts[anchor] = sum(quotas.values())
        anchor_audits.append({'anchor_key': anchor_keys[anchor], 'populations': populations,
                              'quotas': quotas, 'ranks': ranks})
    selected = index.unrank_many(requests)
    retained = {}; retained_edges = {}
    for (anchor, family), groups in selected.items():
        if len(groups) != len(requests[anchor, family]):
            raise RuntimeError('exact index returned the wrong number of sample ranks')
        for group in groups:
            fragment = _fragment(events, edges, group, owner_keys)
            if fragment.family != family or anchor not in {e.entity for e in fragment.events}:
                raise RuntimeError('exact index violated the requested anchor/family stratum')
            retained[fragment.provenance] = fragment
            retained_edges[fragment.provenance] = tuple(sorted(group))
    if set(selected) != set(requests):
        raise RuntimeError('exact index omitted a requested stratum')
    groups = {f: sorted((x for x in retained.values() if x.family == f), key=lambda x: x.provenance)
              for f in FAMILIES}
    populations = {f: len(groups[f]) for f in FAMILIES}
    quotas = _quotas(populations, case_cap, _seed(seed, 'case'))
    ranks = {f: _uniform_ranks(populations[f], quotas[f], _seed(seed, 'case/' + f + '/direct-ranks'))
             for f in FAMILIES}
    final = tuple(sorted((groups[f][r] for f in FAMILIES for r in ranks[f]), key=lambda x: x.provenance))
    audit.update(family_counts={f: index.family_counts.get(f, 0) for f in FAMILIES},
                 anchors=anchor_audits, case={'populations': populations, 'quotas': quotas, 'ranks': ranks},
                 final_provenance=[x.provenance for x in final])
    audit['audit_sha256'] = _hash(audit)
    if sampling_audit is not None and _json(sampling_audit) != _json(audit):
        raise ValueError('direct sampling audit does not match the regenerated sample')
    if sampling_cache is not None:
        sampling_cache.put(cache_key, {'final_edges': [list(retained_edges[x.provenance]) for x in final],
                                      'sampling_audit': audit})
    return DirectExtractionResult(final, index.total_count,
        {a: sum(index.anchor_family_counts[a].values()) for a in anchors}, retained_counts, {},
        {f: index.family_counts.get(f, 0) for f in FAMILIES},
        {f: sum(x.family == f for x in final) for f in FAMILIES}, audit)
