"""Canonical dataset registry with an explicit legacy compatibility shim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


IDENTITY_SCHEMA_VERSION = "canonical-dataset-identity-v1"


@dataclass(frozen=True)
class DatasetSpec:
    canonical_id: str
    legacy_alias: str
    loader_id: str
    source_root: str
    expected_case_count: int
    compatibility_aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetResolution:
    spec: DatasetSpec
    requested_id: str

    @property
    def canonical_id(self) -> str:
        return self.spec.canonical_id

    @property
    def legacy_alias(self) -> str:
        return self.spec.legacy_alias

    @property
    def alias_used(self) -> bool:
        return self.requested_id != self.canonical_id

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "canonical_dataset_id": self.canonical_id,
            "requested_dataset_id": self.requested_id,
            "legacy_alias": self.legacy_alias,
            "alias_used": self.alias_used,
            "loader_id": self.spec.loader_id,
            "source_root": self.spec.source_root,
        }


_REGISTRY = {
    "rcabench": DatasetSpec(
        canonical_id="rcabench",
        legacy_alias="hd4",
        loader_id="rcabench_parquet",
        source_root="${RCABENCH_ROOT}",
        expected_case_count=1422,
    ),
    "aiops25": DatasetSpec(
        canonical_id="aiops25",
        legacy_alias="hd3",
        loader_id="aiops25_native",
        source_root="${AIOPS25_ROOT}",
        expected_case_count=400,
    ),
    "aiops2022_pre": DatasetSpec(
        canonical_id="aiops2022_pre",
        legacy_alias="hd1",
        loader_id="aiops2022_pre",
        source_root="${AIOPS22_ROOT}",
        expected_case_count=482,
        compatibility_aliases=("aiops2022-pre",),
    ),
}
_ALIASES = {
    alias: canonical_id
    for canonical_id, spec in _REGISTRY.items()
    for alias in (spec.legacy_alias, *spec.compatibility_aliases)
}


def canonical_dataset_ids() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_dataset(identifier: str) -> DatasetResolution:
    requested = str(identifier).strip()
    canonical_id = requested if requested in _REGISTRY else _ALIASES.get(requested)
    if canonical_id is None:
        raise ValueError(
            "Unsupported dataset identifier %r; expected one of %s or explicit aliases %s"
            % (requested, sorted(_REGISTRY), sorted(_ALIASES))
        )
    return DatasetResolution(spec=_REGISTRY[canonical_id], requested_id=requested)


def canonical_output_path(root: Path, identifier: str, *parts: str) -> Path:
    resolution = resolve_dataset(identifier)
    output = Path(root) / resolution.canonical_id
    for part in parts:
        component = str(part)
        if not component or Path(component).is_absolute() or ".." in Path(component).parts:
            raise ValueError("Unsafe output-path component: %r" % component)
        output = output / component
    return output


def with_dataset_identity(
    record: Mapping[str, Any], identifier: str
) -> Dict[str, Any]:
    resolution = resolve_dataset(identifier)
    identity = {
        "canonical_dataset_id": resolution.canonical_id,
        "requested_dataset_id": resolution.requested_id,
        "legacy_alias": resolution.legacy_alias,
        "dataset_alias_used": resolution.alias_used,
    }
    output = dict(record)
    for key, expected in identity.items():
        if key in output and output[key] != expected:
            raise ValueError(
                "Conflicting dataset identity field %s: %r != %r"
                % (key, output[key], expected)
            )
        output[key] = expected
    return output


__all__ = [
    "DatasetResolution",
    "DatasetSpec",
    "IDENTITY_SCHEMA_VERSION",
    "canonical_dataset_ids",
    "canonical_output_path",
    "resolve_dataset",
    "with_dataset_identity",
]
