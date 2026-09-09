"""Unified entity indexing for the service-plus-host ranking space."""

import ast
import bz2
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set

import pandas as pd
import pyarrow.parquet as pq

from nexusrcl_rebuild.datasets.common import (
    DatasetManifest,
    ensure_dir,
    load_manifest,
    make_entity_id,
    natural_sort_key,
    normalize_service_name,
    write_json,
)


@dataclass(frozen=True)
class EntityIndex:
    dataset: str
    services: Sequence[str]
    hosts: Sequence[str]
    entity_to_index: Mapping[str, int]
    service_to_hosts: Mapping[str, Sequence[str]]
    host_to_services: Mapping[str, Sequence[str]]
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "services": list(self.services),
            "hosts": list(self.hosts),
            "entity_to_index": dict(self.entity_to_index),
            "service_to_hosts": {key: list(value) for key, value in self.service_to_hosts.items()},
            "host_to_services": {key: list(value) for key, value in self.host_to_services.items()},
            "metadata": dict(self.metadata),
        }


def _load_hd3_pod_mapping(path: Path) -> Mapping[str, Mapping[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = defaultdict(dict)
    for day, interval_map in raw.items():
        for pod_map in interval_map.values():
            for pod_name, host_name in pod_map.items():
                mapping[day][pod_name] = host_name
    return mapping


def _scan_hd1_service_to_hosts(dataset_root: Path, days: Sequence[str]) -> Dict[str, Set[str]]:
    service_to_hosts = defaultdict(set)
    for day in days:
        metric_dir = dataset_root / day / "cloudbed" / "metric" / "container"
        metric_files = sorted(metric_dir.glob("*.csv"))
        if not metric_files:
            continue
        frame = pd.read_csv(metric_files[0], usecols=["cmdb_id"])
        for cmdb_id in frame["cmdb_id"].dropna().unique():
            value = str(cmdb_id)
            if "." not in value:
                continue
            host_name, pod_name = value.split(".", 1)
            service_to_hosts[normalize_service_name(pod_name)].add(host_name)
    return service_to_hosts


def _scan_hd2_service_to_hosts(dataset_root: Path, days: Sequence[str]) -> Dict[str, Set[str]]:
    service_to_hosts = defaultdict(set)
    for day in days:
        trace_dir = dataset_root / day / "trace"
        for trace_path in sorted(trace_dir.glob("*.bz2"), key=lambda path: natural_sort_key(path.name)):
            unique_processes = set()
            with bz2.open(trace_path, "rt", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    raw_process = row.get("process")
                    if raw_process:
                        unique_processes.add(raw_process)
            for raw_process in unique_processes:
                process = ast.literal_eval(raw_process)
                service_name = process.get("serviceName")
                host_name = None
                pod_name = None
                for tag in process.get("tags", []):
                    if tag.get("key") == "node_name":
                        host_name = tag.get("value")
                    elif tag.get("key") == "name":
                        pod_name = tag.get("value")
                if not service_name or not host_name:
                    continue
                canonical = normalize_service_name(pod_name or service_name)
                service_to_hosts[canonical].add(str(host_name))
    return service_to_hosts


def _scan_hd3_service_to_hosts(dataset_root: Path, days: Sequence[str]) -> Dict[str, Set[str]]:
    service_to_hosts = defaultdict(set)
    pod_mapping = _load_hd3_pod_mapping(dataset_root / "pod_instance_mapping.json")
    for day in days:
        for pod_name, host_name in pod_mapping.get(day, {}).items():
            service_to_hosts[normalize_service_name(pod_name)].add(host_name)
    return service_to_hosts


def _scan_hd4_service_to_hosts(dataset_root: Path, days: Sequence[str]) -> Dict[str, Set[str]]:
    service_to_hosts = defaultdict(set)
    parquet_files = [
        "normal_logs.parquet",
        "normal_metrics.parquet",
        "normal_metrics_sum.parquet",
        "normal_traces.parquet",
        "abnormal_logs.parquet",
        "abnormal_metrics.parquet",
        "abnormal_metrics_sum.parquet",
        "abnormal_traces.parquet",
    ]
    for day in days:
        case_dir = dataset_root / day
        if not case_dir.exists():
            continue
        for file_name in parquet_files:
            parquet_path = case_dir / file_name
            if not parquet_path.exists():
                continue
            available_columns = set(pq.ParquetFile(parquet_path).schema_arrow.names)
            columns = [
                column_name
                for column_name in ("service_name", "attr.k8s.service.name")
                if column_name in available_columns
            ]
            if not columns:
                continue
            table = pq.read_table(parquet_path, columns=columns)
            for row in table.to_pylist():
                service_name = row.get("attr.k8s.service.name") or row.get("service_name")
                if service_name:
                    service_to_hosts[normalize_service_name(str(service_name))]
    return service_to_hosts


def build_entity_index(dataset_root: Path, manifest: DatasetManifest) -> EntityIndex:
    dataset = manifest.dataset
    case_days = {case.day for case in manifest.cases}
    days = sorted(case_days or {span.day for span in manifest.day_spans})

    if dataset == "hd1":
        service_to_hosts = _scan_hd1_service_to_hosts(dataset_root, days)
    elif dataset == "hd2":
        service_to_hosts = _scan_hd2_service_to_hosts(dataset_root, days)
    elif dataset == "hd3":
        service_to_hosts = _scan_hd3_service_to_hosts(dataset_root, days)
    elif dataset == "hd4":
        service_to_hosts = _scan_hd4_service_to_hosts(dataset_root, days)
    else:
        raise ValueError("Unsupported dataset: %s" % dataset)

    labeled_services = {
        target.name
        for case in manifest.cases
        for target in case.positives
        if target.entity_type == "service"
    }
    labeled_hosts = {
        target.name
        for case in manifest.cases
        for target in case.positives
        if target.entity_type == "host"
    }

    services = sorted(set(service_to_hosts) | labeled_services)
    hosts = sorted({host for values in service_to_hosts.values() for host in values} | labeled_hosts)

    entity_to_index = {}
    cursor = 0
    for service_name in services:
        entity_to_index[make_entity_id("service", service_name)] = cursor
        cursor += 1
    for host_name in hosts:
        entity_to_index[make_entity_id("host", host_name)] = cursor
        cursor += 1

    host_to_services = defaultdict(set)
    for service_name, service_hosts in service_to_hosts.items():
        for host_name in service_hosts:
            host_to_services[host_name].add(service_name)

    unlabeled_services = sorted(set(services) - labeled_services)
    unlabeled_hosts = sorted(set(hosts) - labeled_hosts)
    return EntityIndex(
        dataset=dataset,
        services=services,
        hosts=hosts,
        entity_to_index=entity_to_index,
        service_to_hosts={key: sorted(value) for key, value in service_to_hosts.items()},
        host_to_services={key: sorted(value) for key, value in host_to_services.items()},
        metadata={
            "service_count": len(services),
            "host_count": len(hosts),
            "unlabeled_service_count": len(unlabeled_services),
            "unlabeled_host_count": len(unlabeled_hosts),
            "days": days,
        },
    )


def write_entity_index(path: Path, entity_index: EntityIndex) -> None:
    ensure_dir(path.parent)
    write_json(path, entity_index.to_dict())


def load_entity_index(path: Path) -> EntityIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EntityIndex(
        dataset=str(payload["dataset"]),
        services=list(payload["services"]),
        hosts=list(payload["hosts"]),
        entity_to_index=dict(payload["entity_to_index"]),
        service_to_hosts={key: list(value) for key, value in payload["service_to_hosts"].items()},
        host_to_services={key: list(value) for key, value in payload["host_to_services"].items()},
        metadata=dict(payload.get("metadata", {})),
    )


def build_entity_index_from_manifest_path(dataset_root: Path, manifest_path: Path) -> EntityIndex:
    return build_entity_index(dataset_root=dataset_root, manifest=load_manifest(manifest_path))
