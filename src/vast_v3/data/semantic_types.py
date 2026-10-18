"""Immutable test-type input for semantic curation, without root annotations.

This value object binds already-curated provenance. It does not read source
files, determine test membership, or certify how the source labels were made.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


SCHEMA = 'vast-v3-semantic-types/1'


def _text(value, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError(f'{name} must be UTF-8 encodable text without surrogate code points') from exc


def _sha256(value, name: str) -> None:
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')


@dataclass(frozen=True, slots=True)
class SemanticTypeRecord:
    case_id: str
    fault_type: str | int
    evidence_sha256: str

    def __post_init__(self):
        _text(self.case_id, 'case_id')
        if type(self.fault_type) not in (str, int):
            raise ValueError('fault_type must be an integer or nonempty string, never bool')
        if type(self.fault_type) is str:
            _text(self.fault_type, 'fault_type')
        _sha256(self.evidence_sha256, 'evidence_sha256')


@dataclass(frozen=True, slots=True)
class SemanticTypeProjection:
    """Canonical snapshot; caller-owned records and collections are not kept."""

    records: tuple[SemanticTypeRecord, ...]
    manifest_sha256: str
    sealed_identity_sha256: str

    def __post_init__(self):
        _sha256(self.manifest_sha256, 'manifest_sha256')
        _sha256(self.sealed_identity_sha256, 'sealed_identity_sha256')
        if isinstance(self.records, (str, bytes)):
            raise ValueError('records must be a collection of SemanticTypeRecord values')
        try:
            records = tuple(self.records)
        except TypeError as exc:
            raise ValueError('records must be a collection of SemanticTypeRecord values') from exc
        if any(type(row) is not SemanticTypeRecord for row in records):
            raise ValueError('records must contain only SemanticTypeRecord values')
        # Reconstruct to validate and detach even an existing frozen record.
        records = tuple(SemanticTypeRecord(row.case_id, row.fault_type, row.evidence_sha256)
                        for row in records)
        if len({row.case_id for row in records}) != len(records):
            raise ValueError('records contain duplicate case IDs')
        object.__setattr__(self, 'records', tuple(sorted(records, key=lambda row: row.case_id)))

    def _payload(self) -> dict:
        return {'schema': SCHEMA, 'manifest_sha256': self.manifest_sha256,
                'sealed_identity_sha256': self.sealed_identity_sha256,
                'records': [{'case_id': row.case_id, 'fault_type': row.fault_type,
                             'evidence_sha256': row.evidence_sha256} for row in self.records]}

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(self._payload(), sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict:
        """Return a fresh JSON object with only the approved test-type fields."""
        return {**self._payload(), 'identity_sha256': self.identity_sha256}
