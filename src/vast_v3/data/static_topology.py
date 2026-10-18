"""Explicit, identity-bound AIOps structural prior; never impute trace values.

The user-approved invariant topology comes from one permitted training
reference window. Inherited validity means structural support, not a measured
call at that time and not fault-propagation evidence. No labels are read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from . import adapters
from .manifests import (WindowSpec, AIOPS_MAPPING_VERSION, AIOPS_MAPPING_SOURCE_SHA256, _sha256)
from .repository import _digest, _inside, _stat, file_sha256, manifest_to_dict
from .schema import Entity, Relation
from .serialization import exact_fields, _record


_SCHEMA = 'vast-v3-aiops-static-topology/1'
_SEMANTICS = 'inherited_static_structure_not_contemporaneous_call_evidence'
_FIELDS = ('schema', 'dataset', 'manifest_sha256', 'catalog_sha256', 'reference_window',
           'permitted_case_ids', 'source_file', 'source_sha256', 'entity_mapping_version',
           'entity_mapping_source_sha256', 'entities', 'relations', 'trace_tail_quarantine',
           'identity_sha256')


def _reference(manifest, permitted_case_ids):
    if (not isinstance(permitted_case_ids, (list, tuple, set, frozenset))
            or any(not isinstance(i, str) or not i for i in permitted_case_ids)):
        raise ValueError('static topology permitted cases must be an explicit collection of case IDs')
    allowed = tuple(sorted(permitted_case_ids))
    if (not allowed or len(set(allowed)) != len(allowed)
            or not set(allowed) <= set(manifest.split.train_case_ids)):
        raise ValueError('static topology permitted cases must be a nonempty training subset')
    windows = [w for w in manifest.windows if w.case_id in allowed and not w.reference_missing]
    if not windows:
        raise ValueError('static topology has no permitted nonempty reference window')
    return min(windows, key=lambda w: (w.reference_start, w.reference_end, w.case_id)), allowed


@dataclass(frozen=True)
class StaticTopology:
    """Immutable canonical JSON storage prevents mutable audit/edge identity drift."""
    _payload_json: str

    @property
    def identity_sha256(self):
        return self.to_dict()['identity_sha256']

    @property
    def source_file(self):
        return self.to_dict()['source_file']

    @property
    def source_sha256(self):
        return self.to_dict()['source_sha256']

    @property
    def reference_window(self):
        return WindowSpec.from_dict(self.to_dict()['reference_window'])

    @property
    def relations(self):
        return tuple(_record(r, Relation) for r in self.to_dict()['relations'])

    @property
    def entities(self):
        return tuple(_record(e, Entity) for e in self.to_dict()['entities'])

    def to_dict(self):
        return json.loads(self._payload_json)

    @classmethod
    def from_dict(cls, payload):
        exact_fields(payload, _FIELDS, 'static topology')
        if (payload['schema'] != _SCHEMA or payload['dataset'] != 'aiops2022_pre'
                or payload['entity_mapping_version'] != AIOPS_MAPPING_VERSION
                or payload['entity_mapping_source_sha256'] != AIOPS_MAPPING_SOURCE_SHA256):
            raise ValueError('unsupported static topology schema/dataset/mapping')
        if _digest({k: v for k, v in payload.items() if k != 'identity_sha256'}) != payload['identity_sha256']:
            raise ValueError('static topology identity mismatch')
        for key in ('manifest_sha256', 'catalog_sha256', 'source_sha256'):
            _sha256(payload[key])
        window = WindowSpec.from_dict(payload['reference_window'])
        allowed = payload['permitted_case_ids']
        if (not isinstance(allowed, list) or not allowed or any(not isinstance(i, str) or not i for i in allowed)
                or len(allowed) != len(set(allowed)) or window.case_id not in allowed
                or window.dataset != payload['dataset'] or window.reference_missing):
            raise ValueError('invalid static topology permitted reference')
        source = Path(payload['source_file'])
        expected = Path(window.day) / 'cloudbed/trace/all/trace_jaeger-span.csv'
        if source.is_absolute() or '..' in source.parts or source != expected:
            raise ValueError('static topology source path differs from reference')
        entities = tuple(_record(e, Entity) for e in payload['entities'])
        ids = {e.id for e in entities}
        relations = tuple(_record(r, Relation) for r in payload['relations'])
        if len(ids) != len(entities) or not relations or len({r.id for r in relations}) != len(relations):
            raise ValueError('static topology entities/observed calls empty or duplicate')
        for relation in relations:
            if (relation.kind != 'call' or relation.owner is not None or relation.response
                    or relation.source not in ids or relation.target not in ids
                    or relation.id != adapters._relation_id('call', relation.source, relation.target)
                    or not relation.active_intervals
                    or any(a != b or not window.reference_start <= a < window.reference_end
                           for a, b in relation.active_intervals)):
                raise ValueError('static topology must contain observed reference call owners only')
        quarantine_fields = ('source_file', 'line_number', 'expected_columns', 'observed_columns',
                             'missing_column', 'file_size_bytes', 'row_sha256', 'reason')
        if not isinstance(payload['trace_tail_quarantine'], list) or len(payload['trace_tail_quarantine']) > 1:
            raise ValueError('invalid static topology tail quarantine')
        for record in payload['trace_tail_quarantine']:
            exact_fields(record, quarantine_fields, 'trace quarantine')
            if (record['source_file'] != payload['source_file'] or record['expected_columns'] != 9
                    or record['observed_columns'] != 8 or record['missing_column'] != 'parent_span'
                    or record['reason'] != 'unterminated_final_row_missing_parent_span'
                    or type(record['line_number']) is not int or record['line_number'] < 2
                    or type(record['file_size_bytes']) is not int or record['file_size_bytes'] < 1):
                raise ValueError('invalid static topology tail quarantine')
            _sha256(record['row_sha256'])
        return cls(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False))

    def validate_manifest(self, manifest):
        payload = self.to_dict()
        # Revalidate even an object constructed directly rather than from_dict.
        self.from_dict(payload)
        if (payload['manifest_sha256'] != _digest(manifest_to_dict(manifest))
                or payload['catalog_sha256'] != _digest(asdict(manifest.catalog))):
            raise ValueError('static topology manifest/catalog identity mismatch')
        window, _ = _reference(manifest, payload['permitted_case_ids'])
        if window != self.reference_window:
            raise ValueError('static topology is not the earliest permitted reference')
        known = {e.id: e for e in self.entities}
        if any(known.get(e.id) != e for e in manifest.catalog.entities):
            raise ValueError('static topology catalog entity identity mismatch')
        if any(e.candidate and e.id not in manifest.catalog.candidate_ids for e in self.entities):
            raise ValueError('static topology cannot add candidates')

    def apply(self, loaded, window):
        """Merge structural validity under the same owner; retain all raw numerics."""
        case = loaded.case
        intervals = tuple((a, b) for a, b in ((window.reference_start, window.reference_end),
                                            (window.incident_start, window.incident_end)) if a < b)
        observed = {r.id: [list(pair) for pair in (r.active_intervals or ((r.start, r.end),))]
                    for r in case.relations if r.kind == 'call'}
        edges = {r.id: r for r in case.relations}
        inherited = []
        for source in self.relations:
            if source.id in edges and (edges[source.id].source, edges[source.id].target,
                                       edges[source.id].kind) != (source.source, source.target, 'call'):
                raise ValueError('static topology owner identity conflict')
            edges[source.id] = replace(source, start=intervals[0][0], end=intervals[-1][1],
                                       active_intervals=intervals)
            inherited.append(source.id)
        entities = {e.id: e for e in case.entities}
        for entity in self.entities:
            if entity.id in entities and entities[entity.id] != entity:
                raise ValueError('static topology entity identity conflict')
            entities[entity.id] = entity
        audit = {**loaded.audit, 'static_topology': {'identity_sha256': self.identity_sha256,
            'reference_window': asdict(self.reference_window), 'source_file': self.source_file,
            'source_sha256': self.source_sha256, 'semantics': _SEMANTICS,
            'inherited_relation_ids': inherited, 'observed_call_intervals': observed,
            'observed_call_relation_count': len(observed),
            'observed_trace_value_count': sum(sum(n.observed) for n in case.numeric if n.modality == 'trace')}}
        return adapters.LoadedCase(replace(case, entities=tuple(entities.values()), relations=tuple(edges.values())), audit)


def build_static_topology(manifest, raw_root, *, permitted_case_ids=None):
    """Explicit source-reading operation; normal repository construction is lazy.

    Only the earliest nonempty permitted training reference is used. The source
    file is hashed in full; only timestamps of out-of-reference rows are read
    for filtering, never their owner/parent/value as topology evidence. An empty
    observed call set blocks, rather than searching later/test observations.
    """
    public = manifest_to_dict(manifest)
    if manifest.split.dataset != 'aiops2022_pre':
        raise ValueError('static topology is supported only for AIOps')
    window, allowed = _reference(manifest, manifest.split.train_case_ids if permitted_case_ids is None else permitted_case_ids)
    root = Path(raw_root).resolve()
    path = _inside(root, root / window.day / 'cloudbed/trace/all/trace_jaeger-span.csv')
    before = _stat(path)
    digest = file_sha256(path)
    trace_audit = {}
    rows = (row for row in adapters.aiops_trace_csv_rows(path, audit=trace_audit)
            if window.reference_start <= adapters.native_seconds(row['timestamp'], unit='ms') < window.reference_end)
    loaded = adapters.aiops22_rows_to_case(window, manifest.catalog, trace_rows=rows)
    if _stat(path) != before:
        raise ValueError('static topology source changed while being read')
    if not loaded.case.relations:
        raise ValueError('earliest permitted reference has no observed call edges')
    endpoints = {name for edge in loaded.case.relations for name in (edge.source, edge.target)}
    relative = path.relative_to(root).as_posix()
    payload = {'schema': _SCHEMA, 'dataset': manifest.split.dataset,
        'manifest_sha256': _digest(public), 'catalog_sha256': _digest(asdict(manifest.catalog)),
        'reference_window': asdict(window), 'permitted_case_ids': list(allowed),
        'source_file': relative, 'source_sha256': digest,
        'entity_mapping_version': AIOPS_MAPPING_VERSION, 'entity_mapping_source_sha256': AIOPS_MAPPING_SOURCE_SHA256,
        'entities': [asdict(e) for e in loaded.case.entities if e.candidate or e.id in endpoints],
        'relations': [asdict(r) for r in loaded.case.relations],
        'trace_tail_quarantine': [{**r, 'source_file': relative} for r in trace_audit['trace_tail_quarantine']]}
    return StaticTopology.from_dict(json.loads(json.dumps({**payload, 'identity_sha256': _digest(payload)})))
