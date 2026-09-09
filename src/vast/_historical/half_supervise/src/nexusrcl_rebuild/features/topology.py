"""Trace-driven topology extraction for services and hosts."""

import ast
import bz2
import csv
import io
import json
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq

from nexusrcl_rebuild.datasets.common import (
    DatasetManifest,
    ensure_dir,
    natural_sort_key,
    normalize_service_name,
    write_csv,
    write_json,
)

from .entities import EntityIndex


@dataclass(frozen=True)
class TopologyArtifacts:
    dataset: str
    service_service_edges: Mapping[Tuple[str, str], int]
    service_host_edges: Mapping[Tuple[str, str], int]
    host_host_edges: Mapping[Tuple[str, str], int]
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "service_service_edges": [
                {"source": src, "target": dst, "weight": weight}
                for (src, dst), weight in sorted(self.service_service_edges.items())
            ],
            "service_host_edges": [
                {"source": src, "target": dst, "weight": weight}
                for (src, dst), weight in sorted(self.service_host_edges.items())
            ],
            "host_host_edges": [
                {"source": src, "target": dst, "weight": weight}
                for (src, dst), weight in sorted(self.host_host_edges.items())
            ],
            "metadata": dict(self.metadata),
        }


def _parse_literal(payload: Any) -> Any:
    if isinstance(payload, (dict, list)):
        return payload
    if payload is None:
        return None
    text = str(payload)
    if not text or text == "nan":
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _tag_lookup(tags: Any, wanted_keys: Sequence[str]) -> Optional[str]:
    if isinstance(tags, str):
        for key in wanted_keys:
            pattern = re.compile(
                r"['\"]key['\"]\s*:\s*['\"]%s['\"].*?['\"]value['\"]\s*:\s*['\"]([^'\"]*)['\"]"
                % re.escape(key)
            )
            match = pattern.search(tags)
            if match:
                return match.group(1)
    parsed = _parse_literal(tags)
    if not isinstance(parsed, list):
        return None
    for tag in parsed:
        if not isinstance(tag, dict):
            continue
        if tag.get("key") in wanted_keys:
            value = tag.get("value")
            if value is not None:
                return str(value)
    return None


def _process_lookup(process: Any, field: str) -> Optional[str]:
    if isinstance(process, str):
        pattern = re.compile(r"['\"]%s['\"]\s*:\s*['\"]([^'\"]+)['\"]" % re.escape(field))
        match = pattern.search(process)
        if match:
            return match.group(1)
    parsed = _parse_literal(process)
    if isinstance(parsed, dict):
        value = parsed.get(field)
        if value is not None:
            return str(value)
    return None


def _child_of_parent_span_ids(references: Any) -> Sequence[str]:
    if isinstance(references, str):
        pattern = re.compile(
            r"['\"]refType['\"]\s*:\s*['\"]CHILD_OF['\"].*?['\"]spanID['\"]\s*:\s*['\"]([^'\"]+)['\"]"
        )
        return [match.group(1) for match in pattern.finditer(references)]
    parsed_refs = _parse_literal(references)
    if not isinstance(parsed_refs, list):
        return []
    result = []
    for reference in parsed_refs:
        if not isinstance(reference, dict) or reference.get("refType") != "CHILD_OF":
            continue
        result.append(str(reference.get("spanID")))
    return result


def _rpc_service_to_service_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    leaf = value.split(".")[-1].split("/")[-1]
    if not leaf:
        return None
    return leaf.lower()


def _hd1_service_call_edges(dataset_root: Path, days: Sequence[str]) -> Counter:
    edges = Counter()
    for day in days:
        trace_path = dataset_root / day / "cloudbed" / "trace" / "all" / "trace_jaeger-span.csv"
        if not trace_path.exists():
            continue
        frame = pd.read_csv(
            trace_path,
            usecols=["trace_id", "span_id", "parent_span", "cmdb_id"],
        )
        frame["service"] = frame["cmdb_id"].astype(str).map(normalize_service_name)
        span_to_service = {
            (str(row.trace_id), str(row.span_id)): str(row.service)
            for row in frame.itertuples(index=False)
        }
        for row in frame.itertuples(index=False):
            parent_span = str(row.parent_span)
            if not parent_span or parent_span == "nan":
                continue
            parent_service = span_to_service.get((str(row.trace_id), parent_span))
            current_service = str(row.service)
            if parent_service and parent_service != current_service:
                edges[(parent_service, current_service)] += 1
    return edges


def _hd2_trace_rows(trace_dir: Path) -> Iterable[Mapping[str, Any]]:
    for trace_path in sorted(trace_dir.glob("*.bz2"), key=lambda path: natural_sort_key(path.name)):
        with bz2.open(trace_path, "rt", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield {
                    "traceID": row.get("traceID"),
                    "spanID": row.get("spanID"),
                    "references": row.get("references"),
                    "tags": row.get("tags"),
                    "process": row.get("process"),
                }


def _service_call_edges_from_trace_rows(rows: Iterable[Mapping[str, Any]]) -> Counter:
    edges = Counter()
    span_to_service = {}
    pending_rows = []

    for row in rows:
        trace_id = str(row.get("traceID") or row.get("trace_id") or "")
        span_id = str(row.get("spanID") or row.get("span_id") or "")
        service_name = _process_lookup(row.get("process"), "serviceName")
        if not service_name:
            continue
        service_name = normalize_service_name(service_name)
        span_to_service[(trace_id, span_id)] = service_name

        target = _tag_lookup(row.get("tags"), ["net.peer.ip", "peer.service"])
        if target:
            target_service = normalize_service_name(target)
            if target_service and target_service != service_name:
                edges[(service_name, target_service)] += 1
                continue
        target = _rpc_service_to_service_name(_tag_lookup(row.get("tags"), ["rpc.service"]))
        if target:
            target_service = normalize_service_name(target)
            if target_service and target_service != service_name:
                edges[(service_name, target_service)] += 1
                continue
        pending_rows.append((trace_id, service_name, row.get("references")))

    for trace_id, service_name, references in pending_rows:
        for parent_id in _child_of_parent_span_ids(references):
            parent_service = span_to_service.get((trace_id, parent_id))
            if parent_service and parent_service != service_name:
                edges[(parent_service, service_name)] += 1
    return edges


def _hd2_service_call_edges(dataset_root: Path, days: Sequence[str]) -> Counter:
    edges = Counter()
    for day in days:
        trace_dir = dataset_root / day / "trace"
        if trace_dir.exists():
            edges.update(_service_call_edges_from_trace_rows(_hd2_trace_rows(trace_dir)))
    return edges


def _hd3_trace_rows(archive_path: Path) -> Iterable[Mapping[str, Any]]:
    with tarfile.open(archive_path, "r:gz") as tar:
        trace_members = sorted(
            member.name
            for member in tar.getmembers()
            if member.isfile() and "/trace-parquet/" in member.name
        )
        for member_name in trace_members:
            payload = tar.extractfile(member_name)
            if payload is None:
                continue
            table = pq.read_table(
                io.BytesIO(payload.read()),
                columns=["traceID", "spanID", "references", "tags", "process"],
            )
            for row in table.to_pylist():
                yield row


def _hd3_service_call_edges(dataset_root: Path, days: Sequence[str]) -> Counter:
    edges = Counter()
    for day in list(days):
        archive_path = dataset_root / ("%s.tar.gz" % day)
        if archive_path.exists():
            edges.update(_service_call_edges_from_trace_rows(_hd3_trace_rows(archive_path)))
    return edges


def _hd4_trace_rows(case_dir: Path) -> Iterable[Mapping[str, Any]]:
    for file_name in ("normal_traces.parquet", "abnormal_traces.parquet"):
        parquet_path = case_dir / file_name
        if not parquet_path.exists():
            continue
        table = pq.read_table(
            parquet_path,
            columns=[
                "trace_id",
                "span_id",
                "parent_span_id",
                "service_name",
                "attr.k8s.service.name",
            ],
        )
        for row in table.to_pylist():
            yield {
                "traceID": row.get("trace_id"),
                "spanID": row.get("span_id"),
                "references": (
                    [{"refType": "CHILD_OF", "spanID": row.get("parent_span_id")}]
                    if row.get("parent_span_id")
                    else []
                ),
                "process": {
                    "serviceName": row.get("attr.k8s.service.name") or row.get("service_name")
                },
            }


def _hd4_service_call_edges(dataset_root: Path, days: Sequence[str]) -> Counter:
    edges = Counter()
    for day in days:
        case_dir = dataset_root / day
        if case_dir.exists():
            edges.update(_service_call_edges_from_trace_rows(_hd4_trace_rows(case_dir)))
    return edges


def build_topology_artifacts(
    dataset_root: Path,
    manifest: DatasetManifest,
    entity_index: EntityIndex,
) -> TopologyArtifacts:
    days = list(entity_index.metadata.get("days", []))
    if manifest.dataset == "hd1":
        service_service_edges = _hd1_service_call_edges(dataset_root, days)
    elif manifest.dataset == "hd2":
        service_service_edges = _hd2_service_call_edges(dataset_root, days)
    elif manifest.dataset == "hd3":
        service_service_edges = _hd3_service_call_edges(dataset_root, days)
    elif manifest.dataset == "hd4":
        service_service_edges = _hd4_service_call_edges(dataset_root, days)
    else:
        raise ValueError("Unsupported dataset: %s" % manifest.dataset)

    service_host_edges = Counter()
    for service_name, hosts in entity_index.service_to_hosts.items():
        for host_name in hosts:
            service_host_edges[(service_name, host_name)] += 1

    host_host_edges = Counter()
    for (source_service, target_service), weight in service_service_edges.items():
        for source_host in entity_index.service_to_hosts.get(source_service, []):
            for target_host in entity_index.service_to_hosts.get(target_service, []):
                if source_host != target_host:
                    host_host_edges[(source_host, target_host)] += weight

    return TopologyArtifacts(
        dataset=manifest.dataset,
        service_service_edges=dict(service_service_edges),
        service_host_edges=dict(service_host_edges),
        host_host_edges=dict(host_host_edges),
        metadata={
            "service_service_edge_count": len(service_service_edges),
            "service_host_edge_count": len(service_host_edges),
            "host_host_edge_count": len(host_host_edges),
            "days": days,
        },
    )


def write_topology_bundle(output_dir: Path, topology: TopologyArtifacts) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "topology.json", topology.to_dict())
    for file_name, edges in [
        ("service_service_edges.csv", topology.service_service_edges),
        ("service_host_edges.csv", topology.service_host_edges),
        ("host_host_edges.csv", topology.host_host_edges),
    ]:
        write_csv(
            output_dir / file_name,
            [
                {"source": source, "target": target, "weight": weight}
                for (source, target), weight in sorted(edges.items())
            ],
            ["source", "target", "weight"],
        )


def load_topology_bundle(path: Path) -> TopologyArtifacts:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TopologyArtifacts(
        dataset=str(payload["dataset"]),
        service_service_edges={
            (str(item["source"]), str(item["target"])): int(item["weight"])
            for item in payload.get("service_service_edges", [])
        },
        service_host_edges={
            (str(item["source"]), str(item["target"])): int(item["weight"])
            for item in payload.get("service_host_edges", [])
        },
        host_host_edges={
            (str(item["source"]), str(item["target"])): int(item["weight"])
            for item in payload.get("host_host_edges", [])
        },
        metadata=dict(payload.get("metadata", {})),
    )
