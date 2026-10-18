"""Independent curator: raw annotations -> sealed, audited complete root sets.

Never import this module into encoders/selectors. Full audit records include
roots and fault types; only the paid LabelStore query API exposes training
roots. Evaluation roots are restricted to frozen test IDs. Historical model
predictions are not an annotation source.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

from .manifests import (CuratedManifest, canonical_roots, resolve_aiops_owner,
    AIOPS_MAPPING_VERSION, AIOPS_MAPPING_SOURCE, AIOPS_MAPPING_SOURCE_SHA256)
from .repository import _digest, _inside, manifest_to_dict
from .schema import LabelStore
from .semantic_types import SemanticTypeProjection, SemanticTypeRecord
from .serialization import exact_fields, strict_json_loads


@dataclass(frozen=True)
class CuratedLabelRecord:
    case_id: str
    source_id: str
    granularity: str
    raw_roots: tuple[str, ...]
    fault_type: str | int
    evidence_path: str
    evidence_sha256: str
    complete: bool
    completeness_evidence: str
    cosource_confirmation: str = ''


class CurationError(ValueError):
    """All affected cases remain in the report and the entire run is blocked."""
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__('curator mapping/identity audit blocked: ' + '; '.join(issues))


class CuratedLabels:
    def __init__(self, manifest, records, roots, aliases, mapping_version, audit):
        self._manifest, self._records, self._roots = manifest, tuple(records), dict(roots)
        self._aliases, self.mapping_version, self.audit = dict(aliases), mapping_version, dict(audit)
        self.identity_sha256 = _digest(self._payload())

    def _payload(self):
        return {'schema': 'vast-v3-sealed-root-audit/1', 'manifest_sha256': _digest(manifest_to_dict(self._manifest)),
            'entity_mapping_version': self.mapping_version, 'aliases': self._aliases,
            'records': [{**asdict(r), 'canonical_roots': list(self._roots[r.case_id])} for r in self._records],
            'audit': self.audit}

    def to_dict(self):
        """CURATOR ONLY. This payload contains all roots/types, not model input."""
        from copy import deepcopy
        return deepcopy({**self._payload(), 'identity_sha256': self.identity_sha256})

    @classmethod
    def from_dict(cls, payload, manifest):
        exact_fields(payload, ('schema', 'manifest_sha256', 'entity_mapping_version', 'aliases', 'records', 'audit', 'identity_sha256'), 'sealed root audit')
        if payload['schema'] != 'vast-v3-sealed-root-audit/1' or payload['manifest_sha256'] != _digest(manifest_to_dict(manifest)):
            raise ValueError('sealed root audit manifest identity mismatch')
        records = []
        for row in payload['records']:
            exact_fields(row, [f.name for f in fields(CuratedLabelRecord)] + ['canonical_roots'], 'sealed root record')
            data = {f.name: row[f.name] for f in fields(CuratedLabelRecord)}
            data['raw_roots'] = tuple(data['raw_roots'])
            records.append(CuratedLabelRecord(**data))
        result = curate_root_records(manifest, records, aliases=payload['aliases'], mapping_version=payload['entity_mapping_version'])
        if _digest({k: v for k, v in payload.items() if k != 'identity_sha256'}) != payload['identity_sha256'] or result.identity_sha256 != payload['identity_sha256']:
            raise ValueError('sealed root mapping/identity mismatch')
        return result

    def query_store(self, *, budget):
        return LabelStore({i: self._roots[i] for i in self._manifest.split.train_case_ids}, budget,
                          max_roots=2 if self._manifest.split.dataset == 'rcabench' else None)

    def evaluation_store(self):
        """Test-only mapping; never pass it into fitting or model selection."""
        return MappingProxyType({i: self._roots[i] for i in self._manifest.split.test_case_ids})

    def test_type_projection(self) -> SemanticTypeProjection:
        """Export all frozen test types after checking curator-side integrity.

        The complete sealed payload (including roots) is hashed for integrity.
        Roots are not used to choose membership or type values and are never
        exported here. This is not a paid root query or a learner-facing store.
        """
        payload = self._payload()
        if _digest(payload) != self.identity_sha256:
            raise ValueError('sealed root audit identity mismatch before type projection')
        records = {row.case_id: row for row in self._records}
        return SemanticTypeProjection(
            records=tuple(SemanticTypeRecord(records[cid].case_id, records[cid].fault_type,
                                            records[cid].evidence_sha256)
                          for cid in self._manifest.split.test_case_ids),
            manifest_sha256=payload['manifest_sha256'],
            sealed_identity_sha256=self.identity_sha256)


def curate_root_records(manifest: CuratedManifest, records, *, aliases: Mapping[str, str] | None = None,
                        mapping_version: str | None = None) -> CuratedLabels:
    manifest_to_dict(manifest)  # validate complete frozen population first
    aliases = dict(aliases or {})
    if aliases and mapping_version is None:
        raise ValueError('explicit alias mapping requires a declared mapping version')
    mapping_version = mapping_version or (AIOPS_MAPPING_VERSION if manifest.split.dataset == 'aiops2022_pre' else 'canonical-catalog/1')
    if not isinstance(mapping_version, str) or not mapping_version:
        raise ValueError('mapping version must be nonempty')
    if any(not isinstance(k, str) or not k or v not in manifest.catalog.candidate_ids for k, v in aliases.items()):
        raise ValueError('alias outside frozen candidate catalog')
    records = tuple(records)
    ids = [r.case_id for r in records]
    windows = {w.case_id: w for w in manifest.windows}
    if len(ids) != len(set(ids)) or set(ids) != set(windows):
        raise ValueError('curator record population must exactly match frozen windows without duplicates')
    # Aliases are provenance declarations, never a second mapping authority.
    # Validate unused declarations too, so a sealed sidecar cannot carry a
    # latent mapping override that the observation adapters do not share.
    for raw, declared in aliases.items():
        granularities = {r.granularity for r in records if raw in r.raw_roots} or {declared.split(':', 1)[0]}
        for granularity in granularities:
            category = {'service': 'service', 'pod': 'service', 'container': 'service', 'host': 'host', 'node': 'host'}.get(granularity)
            if category is None or not declared.startswith(category + ':'):
                raise ValueError('alias changes entity category')
            name = raw.rsplit('.', 1)[-1] if granularity == 'container' else raw
            frozen = (resolve_aiops_owner(manifest.catalog, name, category)
                      if manifest.split.dataset == 'aiops2022_pre' else manifest.catalog.resolve(name, category))
            if declared != frozen:
                raise ValueError(f'alias changes frozen observation mapping: {raw}')
    roots, issues, evidence_digests = {}, [], {}
    for record in records:
        try:
            if record.source_id != windows[record.case_id].source_id:
                raise ValueError('raw source identity mismatch')
            if record.complete is not True or not isinstance(record.completeness_evidence, str) or not record.completeness_evidence:
                raise ValueError('incomplete root annotation')
            for name in ('evidence_path', 'granularity'):
                if not isinstance(getattr(record, name), str) or not getattr(record, name):
                    raise ValueError(f'missing {name}')
            if type(record.fault_type) not in (str, int) or record.fault_type == '':
                raise ValueError('fault type must retain its raw text or integer enum')
            if not isinstance(record.evidence_sha256, str) or len(record.evidence_sha256) != 64 or any(c not in '0123456789abcdef' for c in record.evidence_sha256):
                raise ValueError('invalid evidence source checksum')
            if record.evidence_path in evidence_digests and evidence_digests[record.evidence_path] != record.evidence_sha256:
                raise ValueError('conflicting evidence source checksums')
            evidence_digests[record.evidence_path] = record.evidence_sha256
            if not isinstance(record.raw_roots, (tuple, list)) or not record.raw_roots or any(not isinstance(r, str) or not r for r in record.raw_roots):
                raise ValueError('raw roots must be a nonempty array of identifiers')
            category = {'service': 'service', 'pod': 'service', 'container': 'service', 'host': 'host', 'node': 'host'}.get(record.granularity)
            if category is None:
                raise ValueError('unknown root granularity')
            mapped = []
            for raw in record.raw_roots:
                if manifest.split.dataset == 'aiops2022_pre':
                    canonical = resolve_aiops_owner(manifest.catalog, raw.rsplit('.', 1)[-1] if record.granularity == 'container' else raw, category)
                else:
                    canonical = canonical_roots(manifest.catalog, record.granularity, [raw])[0]
                mapped.append(canonical)
            if not set(mapped) <= set(manifest.catalog.candidate_ids):
                raise ValueError('complete roots outside frozen catalog')
            value = tuple(i for i in manifest.catalog.candidate_ids if i in mapped)
            if len(value) > 1 and not record.cosource_confirmation:
                raise ValueError('joint roots lack co-source confirmation')
            if manifest.split.dataset == 'rcabench' and len(value) > 2:
                raise ValueError('complete roots contradict expert maximum of two')
            roots[record.case_id] = value
        except (TypeError, ValueError) as exc:
            issues.append(f'{record.case_id}: {exc}')
    if issues:
        raise CurationError(issues)
    ordered = tuple(sorted(records, key=lambda r: r.case_id))
    cardinalities = Counter(str(len(v)) for v in roots.values())
    by_type = {}
    for record in ordered:
        group = by_type.setdefault(str(record.fault_type), Counter())
        group[str(len(roots[record.case_id]))] += 1
    audit = {'case_count': len(roots), 'catalog_version': manifest.catalog.version,
        'catalog_source_sha256': manifest.catalog.source_sha256, 'entity_mapping_version': mapping_version,
        'entity_mapping_source': AIOPS_MAPPING_SOURCE if manifest.split.dataset == 'aiops2022_pre' else 'vast_v3.data.manifests.CandidateCatalog.resolve',
        'entity_mapping_source_sha256': AIOPS_MAPPING_SOURCE_SHA256 if manifest.split.dataset == 'aiops2022_pre' else None,
        'root_cardinality_counts': dict(sorted(cardinalities.items())),
        'root_cardinality_by_fault_type': {k: dict(v) for k, v in sorted(by_type.items())},
        'ambiguous_or_out_of_catalog': [], 'complete': True,
        'context_only_nodes': 'not inspected by label curator; recorded in independent observable case audit',
        'source_sha256': {r.evidence_path: r.evidence_sha256 for r in ordered}}
    return CuratedLabels(manifest, ordered, roots, aliases, mapping_version, audit)


def curate_raw_labels(manifest: CuratedManifest, raw_root: str | Path, *, aliases=None, mapping_version=None) -> CuratedLabels:
    """Read original labels only. Frozen boundaries and splits are not rebuilt."""
    root = Path(raw_root).resolve()
    records, day_tables = [], {}
    for window in manifest.windows:
        if window.dataset == 'rcabench':
            path = _inside(root, root / window.source_id / 'injection.json')
            raw = path.read_bytes(); payload = strict_json_loads(raw)
            try:
                labels = payload['ground_truth']['service']
                fault_type = payload['fault_type']
            except (KeyError, TypeError) as exc:
                raise ValueError(f'RCABench original injection label fields absent: {window.case_id}') from exc
            if not isinstance(labels, list):
                raise ValueError('RCABench complete service roots must be an array')
            records.append(CuratedLabelRecord(window.case_id, window.source_id, 'service', tuple(labels), fault_type,
                str(path), hashlib.sha256(raw).hexdigest(), True, 'original injection.ground_truth.service complete joint set',
                'domain expert 2026-10-15: simultaneous network co-sources, not alternatives'))
        elif window.dataset == 'aiops2022_pre':
            if window.day not in day_tables:
                path = _inside(root, root / 'groundtruth' / f'groundtruth-{window.day}.json')
                raw = path.read_bytes(); payload = strict_json_loads(raw)
                exact_fields(payload, ('timestamp', 'level', 'cmdb_id', 'failure_type'), 'original AIOps groundtruth')
                if any(not isinstance(v, list) for v in payload.values()) or len({len(v) for v in payload.values()}) != 1:
                    raise ValueError('original AIOps parallel label arrays differ')
                table = {}
                for index, timestamp in enumerate(payload['timestamp']):
                    if type(timestamp) is not int or str(timestamp) in table:
                        raise ValueError('original AIOps timestamp must be unique integer seconds')
                    table[str(timestamp)] = {key: values[index] for key, values in payload.items()}
                day_tables[window.day] = path, hashlib.sha256(raw).hexdigest(), table
            path, digest, table = day_tables[window.day]
            parts = window.source_id.split(':')
            if len(parts) < 6 or parts[0] != 'hd1' or parts[1] != window.day or parts[2] not in table:
                raise ValueError(f'frozen AIOps source identity absent from original labels: {window.case_id}')
            row = table[parts[2]]
            if (parts[3], parts[4]) != (row['level'], row['cmdb_id']):
                raise ValueError(f'frozen AIOps original root/level identity mismatch: {window.case_id}')
            records.append(CuratedLabelRecord(window.case_id, window.source_id, row['level'], (row['cmdb_id'],),
                row['failure_type'], str(path), digest, True, 'original AIOps groundtruth timestamp/level/cmdb_id record'))
        else:
            raise ValueError('unknown raw annotation dataset')
    return curate_root_records(manifest, records, aliases=aliases, mapping_version=mapping_version)
