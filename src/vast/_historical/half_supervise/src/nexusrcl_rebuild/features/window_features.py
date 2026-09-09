"""Window-level multimodal feature extraction from rebuilt raw telemetry."""

import ast
import bz2
import csv
import io
import math
import re
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

from nexusrcl_rebuild.datasets.common import (
    DatasetManifest,
    LOCAL_TZ,
    local_day_bounds,
    make_entity_id,
    natural_sort_key,
    normalize_service_name,
)

from .artifacts import WindowFeatureBundle, WindowRecord
from .entities import EntityIndex
from .topology import TopologyArtifacts


BUCKET_SECONDS = 60
LOG_SIGNATURE_FEATURE_LIMIT = 24
METRIC_EVENT_KPI_FEATURE_LIMIT = 32
TRACE_OPERATION_FEATURE_LIMIT = 32
TRACE_PEER_FEATURE_LIMIT = 16

TOPOLOGY_FEATURE_COLUMNS = [
    "entity_is_service",
    "entity_is_host",
    "topo_in_degree",
    "topo_out_degree",
    "topo_in_weight",
    "topo_out_weight",
    "topo_cross_degree",
    "topo_cross_weight",
]

LOG_FEATURE_COLUMNS = [
    "log_count",
    "log_error_count",
    "log_warn_count",
    "log_info_count",
    "log_error_ratio",
    "log_warn_ratio",
    "log_unique_message_count",
    "log_message_entropy",
]

METRIC_FEATURE_COLUMNS = [
    "metric_series_count",
    "metric_sample_count",
    "metric_window_mean",
    "metric_abs_z_mean",
    "metric_abs_z_max",
    "metric_last_abs_z_max",
    "metric_delta_abs_z_mean",
    "metric_range_abs_z_mean",
    "metric_anomalous_kpi_count",
]

METRIC_EVENT_FEATURE_COLUMNS = [
    "metric_event_active_kpi_count",
    "metric_event_hit_count",
    "metric_event_active_timestamp_count",
    "metric_event_score_sum",
    "metric_event_score_mean",
    "metric_event_score_max",
    "metric_event_score_p95",
    "metric_event_peak_kpi_count",
]

TRACE_FEATURE_COLUMNS = [
    "trace_span_count",
    "trace_client_span_count",
    "trace_server_span_count",
    "trace_server_span_ratio",
    "trace_error_count",
    "trace_error_ratio",
    "trace_client_error_count",
    "trace_server_error_count",
    "trace_client_error_ratio",
    "trace_server_error_ratio",
    "trace_server_error_ratio_gap",
    "trace_unique_operation_count",
    "trace_unique_peer_count",
    "trace_duration_mean_ms",
    "trace_duration_max_ms",
    "trace_duration_p95_ms",
    "trace_client_duration_mean_ms",
    "trace_server_duration_mean_ms",
    "trace_latency_abs_z_mean",
    "trace_latency_abs_z_max",
    "trace_client_latency_abs_z_mean",
    "trace_server_latency_abs_z_mean",
    "trace_client_latency_abs_z_max",
    "trace_server_latency_abs_z_max",
    "trace_server_latency_abs_z_gap",
    "trace_anomalous_operation_count",
    "trace_client_anomalous_operation_count",
    "trace_server_anomalous_operation_count",
]

TOPOLOGY_CHANGE_FEATURE_COLUMNS = [
    "topology_change_count",
]

SUMMARY_FEATURE_COLUMNS = [
    "has_log_signal",
    "has_metric_signal",
    "has_trace_signal",
    "modalities_present_count",
]

DYNAMIC_FEATURE_COLUMNS = (
    LOG_FEATURE_COLUMNS
    + METRIC_FEATURE_COLUMNS
    + METRIC_EVENT_FEATURE_COLUMNS
    + TRACE_FEATURE_COLUMNS
    + TOPOLOGY_CHANGE_FEATURE_COLUMNS
    + SUMMARY_FEATURE_COLUMNS
)

METRIC_REJECT_KEYWORDS = (
    "bucket",
    "quantile",
    "build_info",
    "os_info",
    "os_version",
    "uname_info",
    "network_info",
    "disk_info",
    "dmi_info",
    "collector",
    "success",
    "last_seen",
    "clocksource",
    "authorizer",
    "protocol_type",
    "address_assign_type",
    "name_assign_type",
    "readonly",
)

METRIC_KEEP_KEYWORDS = (
    "cpu",
    "memory",
    "mem",
    "network",
    "net",
    "disk",
    "fs",
    "io",
    "iowait",
    "read",
    "write",
    "bytes",
    "packet",
    "tcp",
    "udp",
    "process",
    "thread",
    "socket",
    "load",
    "rss",
    "cache",
    "working_set",
    "fault",
    "error",
    "throttle",
    "quota",
    "shares",
    "limit",
    "xfs",
)

LEGACY_INSTANCE_METRIC_NAMES = frozenset(
    {
        "container_blkio_device_usage_total",
        "container_cpu_cfs_periods_total",
        "container_cpu_cfs_throttled_periods_total",
        "container_cpu_cfs_throttled_seconds_total",
        "container_cpu_system_seconds_total",
        "container_cpu_usage_seconds_total",
        "container_cpu_user_seconds_total",
        "container_file_descriptors",
        "container_fs_reads_bytes_total",
        "container_fs_reads_total",
        "container_fs_writes_bytes_total",
        "container_fs_writes_total",
        "container_last_seen",
        "container_memory_cache",
        "container_memory_failures_total",
        "container_memory_mapped_file",
        "container_memory_rss",
        "container_memory_usage_bytes",
        "container_memory_working_set_bytes",
        "container_network_receive_bytes_total",
        "container_network_receive_packets_total",
        "container_network_transmit_bytes_total",
        "container_network_transmit_packets_total",
        "container_processes",
        "container_sockets",
        "container_spec_cpu_period",
        "container_spec_cpu_quota",
        "container_spec_cpu_shares",
        "container_spec_memory_limit_bytes",
        "container_start_time_seconds",
        "container_threads_max",
        "container_threads",
        "container_ulimits_soft",
    }
)

LEGACY_NODE_METRIC_NAMES = frozenset(
    {
        "node_filesystem_device_error",
        "node_filesystem_size_bytes",
        "node_intr_total",
        "node_memory_Active_anon_bytes",
        "node_memory_Active_bytes",
        "node_memory_Active_file_bytes",
        "node_memory_AnonHugePages_bytes",
        "node_memory_AnonPages_bytes",
        "node_memory_Committed_AS_bytes",
        "node_memory_MemAvailable_bytes",
        "node_memory_MemFree_bytes",
        "node_memory_PageTables_bytes",
        "node_vmstat_pgfault",
        "node_xfs_directory_operation_create_total",
        "node_xfs_inode_operation_attribute_changes_total",
        "node_disk_flush_requests_time_seconds_total",
        "node_disk_flush_requests_total",
        "node_disk_reads_merged_total",
        "node_disk_writes_completed_total",
        "node_disk_writes_merged_total",
        "node_disk_write_time_seconds_total",
        "node_disk_written_bytes_total",
        "node_xfs_block_mapping_extent_list_deletions_total",
        "node_xfs_block_mapping_unmaps_total",
        "node_xfs_directory_operation_getdents_total",
        "node_xfs_directory_operation_lookup_total",
        "node_xfs_directory_operation_remove_total",
        "node_xfs_extent_allocation_blocks_allocated_total",
        "node_xfs_extent_allocation_extents_freed_total",
        "node_xfs_inode_operation_attempts_total",
        "node_xfs_inode_operation_found_total",
        "node_xfs_inode_operation_missed_total",
        "node_netstat_IpExt_InOctets",
        "node_netstat_IpExt_OutOctets",
        "node_netstat_Ip_Forwarding",
        "node_netstat_Tcp_CurrEstab",
        "node_netstat_Tcp_InSegs",
        "node_netstat_Tcp_OutSegs",
    }
)

LEGACY_METRIC_NAMES = LEGACY_INSTANCE_METRIC_NAMES | LEGACY_NODE_METRIC_NAMES


def _running_stats_defaultdict() -> DefaultDict[str, "RunningStats"]:
    return defaultdict(RunningStats)


def _trace_running_stats_defaultdict() -> DefaultDict[str, "RunningStats"]:
    return defaultdict(RunningStats)


def _trace_kind_counter_defaultdict() -> DefaultDict[str, Counter]:
    return defaultdict(Counter)


def _trace_kind_list_defaultdict() -> DefaultDict[str, List[float]]:
    return defaultdict(list)


def _trace_kind_stats_defaultdict() -> DefaultDict[str, DefaultDict[str, "RunningStats"]]:
    return defaultdict(_trace_running_stats_defaultdict)


def _entity_window_accumulator_defaultdict() -> DefaultDict[Tuple[str, str], "EntityWindowAccumulator"]:
    return defaultdict(EntityWindowAccumulator)


def _float_defaultdict() -> DefaultDict[int, float]:
    return defaultdict(float)


def _metric_day_series_defaultdict() -> DefaultDict[Tuple[str, str], DefaultDict[int, float]]:
    return defaultdict(_float_defaultdict)


def _pair_running_stats_defaultdict() -> DefaultDict[Tuple[str, str], "RunningStats"]:
    return defaultdict(RunningStats)


def _service_host_bucket_defaultdict() -> DefaultDict[int, set]:
    return defaultdict(set)


def _service_host_presence_defaultdict() -> DefaultDict[str, DefaultDict[int, set]]:
    return defaultdict(_service_host_bucket_defaultdict)


@dataclass
class RunningStats:
    """Streaming statistics for a numeric series."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    first: Optional[float] = None
    last: Optional[float] = None

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if self.first is None:
            self.first = value
        self.last = value

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / float(self.count)

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean_value = self.mean()
        variance = max(self.total_sq / float(self.count) - (mean_value * mean_value), 0.0)
        return math.sqrt(variance)


@dataclass
class LogAccumulator:
    count: int = 0
    error_count: int = 0
    warn_count: int = 0
    info_count: int = 0
    message_counts: Counter = field(default_factory=Counter)

    def update(self, message: str, severity: str) -> None:
        self.count += 1
        if severity == "error":
            self.error_count += 1
        elif severity == "warn":
            self.warn_count += 1
        elif severity == "info":
            self.info_count += 1
        signature = _normalize_log_signature(message)
        if signature:
            self.message_counts[signature] += 1


@dataclass
class MetricAccumulator:
    by_kpi: DefaultDict[str, RunningStats] = field(default_factory=_running_stats_defaultdict)

    def update(self, kpi_name: str, value: float) -> None:
        self.by_kpi[kpi_name].update(value)


@dataclass
class MetricEventAccumulator:
    score_values: List[float] = field(default_factory=list)
    active_timestamps: set = field(default_factory=set)
    kpi_hit_counts: Counter = field(default_factory=Counter)
    kpi_peak_scores: Dict[str, float] = field(default_factory=dict)

    def update(self, timestamp: int, kpi_name: str, score: float) -> None:
        if score <= 0.0:
            return
        self.score_values.append(float(score))
        self.active_timestamps.add(int(timestamp))
        self.kpi_hit_counts[kpi_name] += 1
        self.kpi_peak_scores[kpi_name] = max(
            float(score),
            float(self.kpi_peak_scores.get(kpi_name, 0.0)),
        )


@dataclass
class TraceAccumulator:
    span_count: int = 0
    client_span_count: int = 0
    server_span_count: int = 0
    error_count: int = 0
    durations_ms: List[float] = field(default_factory=list)
    operation_counts: Counter = field(default_factory=Counter)
    peer_counts: Counter = field(default_factory=Counter)
    operation_durations: DefaultDict[str, RunningStats] = field(
        default_factory=_trace_running_stats_defaultdict
    )
    kind_error_counts: Counter = field(default_factory=Counter)
    kind_durations_ms: DefaultDict[str, List[float]] = field(default_factory=_trace_kind_list_defaultdict)
    kind_operation_counts: DefaultDict[str, Counter] = field(default_factory=_trace_kind_counter_defaultdict)
    kind_peer_counts: DefaultDict[str, Counter] = field(default_factory=_trace_kind_counter_defaultdict)
    kind_operation_durations: DefaultDict[str, DefaultDict[str, RunningStats]] = field(
        default_factory=_trace_kind_stats_defaultdict
    )

    def update(
        self,
        operation: str,
        duration_ms: float,
        is_error: bool,
        span_kind: str,
        peer_name: Optional[str],
    ) -> None:
        self.span_count += 1
        if span_kind == "client":
            self.client_span_count += 1
        elif span_kind == "server":
            self.server_span_count += 1
        if is_error:
            self.error_count += 1
        self.durations_ms.append(duration_ms)
        kind_key = span_kind if span_kind in ("client", "server") else "unknown"
        if is_error:
            self.kind_error_counts[kind_key] += 1
        self.kind_durations_ms[kind_key].append(duration_ms)
        if operation:
            self.operation_counts[operation] += 1
            self.operation_durations[operation].update(duration_ms)
            self.kind_operation_counts[kind_key][operation] += 1
            self.kind_operation_durations[kind_key][operation].update(duration_ms)
        if peer_name:
            self.peer_counts[peer_name] += 1
            self.kind_peer_counts[kind_key][peer_name] += 1


@dataclass
class TopologyChangeAccumulator:
    change_values: List[float] = field(default_factory=list)

    def update(self, delta: float) -> None:
        if delta <= 0.0:
            return
        self.change_values.append(float(delta))


@dataclass
class EntityWindowAccumulator:
    log: LogAccumulator = field(default_factory=LogAccumulator)
    metric: MetricAccumulator = field(default_factory=MetricAccumulator)
    metric_event: MetricEventAccumulator = field(default_factory=MetricEventAccumulator)
    trace: TraceAccumulator = field(default_factory=TraceAccumulator)
    topology_change: TopologyChangeAccumulator = field(default_factory=TopologyChangeAccumulator)


@dataclass
class DayBaselines:
    metric_stats: DefaultDict[Tuple[str, str], RunningStats] = field(
        default_factory=_pair_running_stats_defaultdict
    )
    trace_stats: DefaultDict[Tuple[str, str], RunningStats] = field(
        default_factory=_pair_running_stats_defaultdict
    )
    trace_stats_by_kind: DefaultDict[Tuple[str, str, str], RunningStats] = field(
        default_factory=_pair_running_stats_defaultdict
    )


@dataclass(frozen=True)
class WindowSpec:
    dataset: str
    window_id: str
    source_id: str
    window_kind: str
    day: str
    start_ts: int
    end_ts: int
    positive_ids: Sequence[str]
    positive_types: Sequence[str]
    positive_names: Sequence[str]
    metadata: Mapping[str, Any]

    def to_record(self) -> WindowRecord:
        return WindowRecord(
            dataset=self.dataset,
            window_id=self.window_id,
            source_id=self.source_id,
            window_kind=self.window_kind,
            day=self.day,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            positive_ids=list(self.positive_ids),
            positive_types=list(self.positive_types),
            positive_names=list(self.positive_names),
            metadata=dict(self.metadata),
        )


class DayWindowMatcher:
    """Order-independent bucket lookup for sparse window intervals."""

    def __init__(self, windows: Sequence[WindowSpec], bucket_seconds: int = BUCKET_SECONDS):
        self.bucket_seconds = bucket_seconds
        self.bucket_to_windows = defaultdict(list)
        for window in windows:
            start_bucket = int(window.start_ts // bucket_seconds)
            end_bucket = int(window.end_ts // bucket_seconds)
            for bucket in range(start_bucket, end_bucket + 1):
                self.bucket_to_windows[bucket].append(window)

    def match(self, timestamp: int) -> Sequence[WindowSpec]:
        candidates = self.bucket_to_windows.get(int(timestamp // self.bucket_seconds), ())
        return [
            window
            for window in candidates
            if window.start_ts <= timestamp <= window.end_ts
        ]

    def has_bucket(self, timestamp: int) -> bool:
        return int(timestamp // self.bucket_seconds) in self.bucket_to_windows


@dataclass
class DayState:
    matcher: DayWindowMatcher
    windows: Sequence[WindowSpec]
    span_start_ts: int
    span_end_ts: int
    baselines: DayBaselines = field(default_factory=DayBaselines)
    metric_day_series: DefaultDict[Tuple[str, str], DefaultDict[int, float]] = field(
        default_factory=_metric_day_series_defaultdict
    )
    service_host_presence: DefaultDict[str, DefaultDict[int, set]] = field(
        default_factory=_service_host_presence_defaultdict
    )
    entity_windows: DefaultDict[Tuple[str, str], EntityWindowAccumulator] = field(
        default_factory=_entity_window_accumulator_defaultdict
    )
    scan_stats: Counter = field(default_factory=Counter)

    def get_accumulator(self, window_id: str, entity_id: str) -> EntityWindowAccumulator:
        return self.entity_windows[(window_id, entity_id)]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(text)
    except Exception:
        return None


def _canonical_metric_name(metric_name: str) -> str:
    normalized = (metric_name or "").strip()
    normalized = re.sub(r"_ns_[^./]+$", "", normalized)
    normalized = re.sub(r"\.csv$", "", normalized)
    return normalized


def _should_keep_metric_name(metric_name: str, strict_legacy: bool = False) -> bool:
    canonical = _canonical_metric_name(metric_name)
    normalized = canonical.lower()
    if not normalized:
        return False
    if canonical in LEGACY_METRIC_NAMES:
        return True
    if strict_legacy and (
        normalized.startswith("container_") or normalized.startswith("node_")
    ):
        return False
    if any(keyword in normalized for keyword in METRIC_REJECT_KEYWORDS):
        return False
    return any(keyword in normalized for keyword in METRIC_KEEP_KEYWORDS)


def _parse_utc_seconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{10}", text):
        return int(text)
    if re.fullmatch(r"\d{13}", text):
        return int(int(text) / 1000)
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _normalize_log_signature(message: str) -> str:
    text = (message or "").lower()
    text = re.sub(r"[0-9a-f]{8,}", "<hex>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def _infer_log_severity(message: str) -> str:
    text = (message or "").lower()
    severity_match = re.search(r"(?:severity|level)\s*[:=]\s*\"?(error|warn|warning|info)", text)
    if severity_match:
        severity = severity_match.group(1)
        if severity == "warning":
            return "warn"
        return severity
    if "error" in text or "exception" in text or "traceback" in text or " fail" in text:
        return "error"
    if "warn" in text:
        return "warn"
    if "info" in text:
        return "info"
    return "unknown"


def _message_entropy(counter: Counter) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counter.values():
        probability = value / total
        entropy -= probability * math.log(probability + 1e-12)
    return entropy


def _parse_literal(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    text = str(value)
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _tag_lookup(tags: Any, wanted_keys: Sequence[str]) -> Optional[str]:
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


def _process_lookup(process: Any, field_name: str) -> Optional[str]:
    parsed = _parse_literal(process)
    if not isinstance(parsed, dict):
        return None
    value = parsed.get(field_name)
    if value is None:
        return None
    return str(value)


def _process_tag_lookup(process: Any, wanted_keys: Sequence[str]) -> Optional[str]:
    parsed = _parse_literal(process)
    if not isinstance(parsed, dict):
        return None
    return _tag_lookup(parsed.get("tags"), wanted_keys)


def _extract_key_value_fragment(text: str, key: str) -> Optional[str]:
    pattern = r"(?:^|-)%s=(.*?)(?=-[A-Za-z0-9_]+=|$)" % re.escape(key)
    match = re.search(pattern, text or "")
    if not match:
        return None
    return match.group(1)


def _trace_span_kind(tags: Any) -> str:
    value = (_tag_lookup(tags, ["span.kind"]) or "").lower()
    if "client" in value:
        return "client"
    if "server" in value:
        return "server"
    return "unknown"


def _trace_is_error(tags: Any, row: Mapping[str, Any]) -> bool:
    status = (_tag_lookup(tags, ["status.code", "otel.status_code"]) or "").strip().lower()
    if status and status not in ("0", "ok", "unset", "status_code_unset"):
        return True
    status_code = str(
        row.get("status_code")
        or row.get("attr.status_code")
        or row.get("attr.http.response.status_code")
        or ""
    ).strip().lower()
    return bool(
        status_code
        and status_code not in ("0", "0.0", "200", "200.0", "ok", "unset", "status_code_unset")
    )


def _trace_peer_service(tags: Any) -> Optional[str]:
    for key in ("peer.service", "net.peer.ip"):
        value = _tag_lookup(tags, [key])
        if value:
            return normalize_service_name(value)
    rpc_service = _tag_lookup(tags, ["rpc.service"])
    if not rpc_service:
        return None
    leaf = rpc_service.split(".")[-1].split("/")[-1]
    if not leaf:
        return None
    return normalize_service_name(leaf.lower())


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = int(math.ceil((len(sorted_values) - 1) * quantile))
    return float(sorted_values[position])


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    scale = maximum - minimum
    return [float(value - minimum) / float(scale) for value in values]


def _erode_binary_sequence(values: Sequence[int], max_gap: int) -> List[int]:
    result = list(values)
    ones = [index for index, value in enumerate(result) if value == 1]
    for index in range(len(ones) - 1):
        start = ones[index]
        end = ones[index + 1]
        if end - start - 1 <= max_gap:
            for gap_index in range(start + 1, end):
                result[gap_index] = 1
    return result


def _blur_binary_sequence(values: Sequence[int], max_gap: int) -> List[int]:
    result = list(values)
    zeros = [index for index, value in enumerate(result) if value == 0]
    for index in range(len(zeros) - 1):
        start = zeros[index]
        end = zeros[index + 1]
        if end - start - 1 <= max_gap:
            for gap_index in range(start + 1, end):
                result[gap_index] = 0
    return result


def _periodic_pattern_signature(values: Sequence[int], threshold: int = 6) -> Tuple[bool, int, int, int, int]:
    if not values:
        return False, 0, -1, -1, -1

    count_zeros = 0
    while count_zeros < len(values) and values[count_zeros] == 0:
        count_zeros += 1
    if count_zeros >= len(values):
        return False, count_zeros, -1, -1, -1

    periods_of_ones = []
    periods_of_zeros = []
    current = values[count_zeros]
    count = 1
    for index in range(count_zeros + 1, len(values)):
        if values[index] == current:
            count += 1
        else:
            if current == 1:
                periods_of_ones.append(count)
            else:
                periods_of_zeros.append(count)
            current = values[index]
            count = 1
    if current == 1:
        periods_of_ones.append(count)
    else:
        periods_of_zeros.append(count)

    if len(periods_of_ones) < threshold or len(periods_of_zeros) < threshold:
        return False, count_zeros, -1, -1, -1

    one_counts = Counter(periods_of_ones)
    zero_counts = Counter(periods_of_zeros)
    one_period_length, one_period_times = one_counts.most_common(1)[0]
    zero_period_length, zero_period_times = zero_counts.most_common(1)[0]
    if (
        one_period_times <= threshold
        or zero_period_times <= threshold
        or abs(one_period_times - zero_period_times) > 2
    ):
        return False, count_zeros, -1, -1, -1

    offset = count_zeros
    pair_count = min(len(periods_of_ones), len(periods_of_zeros))
    for index in range(pair_count):
        if periods_of_ones[index] == one_period_length and periods_of_zeros[index] == zero_period_length:
            break
        offset += periods_of_ones[index]
        offset += periods_of_zeros[index]
    return True, offset, one_period_length, zero_period_length, max(one_period_times, zero_period_times)


def _synthetic_periodic_mask(
    leading_zeros: int,
    one_period_length: int,
    zero_period_length: int,
    period_times: int,
    full_length: int,
) -> List[int]:
    if full_length <= 0:
        return []
    active_length = int(one_period_length) + 2
    inactive_length = max(0, int(zero_period_length) - 2)
    result = [0] * max(0, int(leading_zeros))
    period_width = max(1, active_length + inactive_length)
    fitted_period_times = max(1, int(full_length / period_width))
    if abs(int(period_times) - fitted_period_times) <= 1:
        period_times = fitted_period_times
    for _ in range(max(0, int(period_times))):
        result.extend([1] * active_length)
        result.extend([0] * inactive_length)
    if len(result) < full_length:
        result.extend([0] * (full_length - len(result)))
    return result[:full_length]


def _sliding_window_sigma_scores(
    values: Sequence[float],
    window_size: int,
    sigma: float = 3.0,
) -> List[float]:
    sample_count = len(values)
    if sample_count == 0:
        return []
    if sample_count <= max(2, window_size + 1):
        return [0.0] * sample_count

    labels = [0.0] * sample_count
    for start_index in range(0, sample_count - window_size - 1):
        window = values[start_index : start_index + window_size]
        local_mean = _mean(window)
        local_std = float(np.std(window))
        if local_std <= 1e-12:
            continue
        lower_limit = local_mean - sigma * local_std
        upper_limit = local_mean + sigma * local_std
        for offset, value in enumerate(window):
            if value < lower_limit or value > upper_limit:
                labels[start_index + offset] += local_std
    return labels


def _eventized_sigma_scores(
    values: Sequence[float],
    window_size: int = 120,
    sigma: float = 3.0,
) -> List[float]:
    if not values:
        return []
    raw_scores = _sliding_window_sigma_scores(values, window_size=window_size, sigma=sigma)
    binary_scores = [1 if score > 0.0 else 0 for score in raw_scores]
    if sum(binary_scores) == 0:
        return [0.0] * len(values)

    eroded = _erode_binary_sequence(binary_scores, 5)
    has_period, leading_zeros, one_length, zero_length, period_times = _periodic_pattern_signature(
        eroded
    )
    if has_period:
        periodic_mask = _synthetic_periodic_mask(
            leading_zeros=leading_zeros,
            one_period_length=one_length,
            zero_period_length=zero_length,
            period_times=period_times,
            full_length=len(values),
        )
        eroded = [max(0, eroded[index] - periodic_mask[index]) for index in range(len(eroded))]

    filtered = _blur_binary_sequence(eroded, 1)
    kept_scores = [raw_scores[index] if filtered[index] == 1 else 0.0 for index in range(len(raw_scores))]
    return _minmax_normalize(kept_scores)


def _zero_feature_map() -> Dict[str, float]:
    return {name: 0.0 for name in DYNAMIC_FEATURE_COLUMNS}


ZERO_DYNAMIC_FEATURES = _zero_feature_map()


def _log_scan_progress(
    dataset_name: str,
    day: str,
    modality: str,
    index: int,
    total: int,
    source_name: str,
) -> None:
    print(
        "[window-features] dataset=%s day=%s modality=%s file=%d/%d source=%s"
        % (dataset_name, day, modality, index, total, source_name),
        flush=True,
    )


def _ordered_entities(entity_index: EntityIndex) -> List[Tuple[str, int, str, str]]:
    reverse = {index: entity_id for entity_id, index in entity_index.entity_to_index.items()}
    ordered = []
    for index in sorted(reverse):
        entity_id = reverse[index]
        entity_type, entity_name = entity_id.split(":", 1)
        ordered.append((entity_id, index, entity_type, entity_name))
    return ordered


def _window_specs_from_manifest(manifest: DatasetManifest) -> List[WindowSpec]:
    windows = []
    for case in manifest.cases:
        windows.append(
            WindowSpec(
                dataset=manifest.dataset,
                window_id=case.case_id,
                source_id=case.case_id,
                window_kind="fault",
                day=case.day,
                start_ts=case.start_ts,
                end_ts=case.end_ts,
                positive_ids=[item.entity_id for item in case.positives],
                positive_types=[item.entity_type for item in case.positives],
                positive_names=[item.name for item in case.positives],
                metadata={
                    "interval_source": case.interval_source,
                    "label_granularity": case.label_granularity,
                    "fault_type": case.fault_type,
                    "raw_target": case.raw_target,
                    "case_metadata": dict(case.metadata),
                },
            )
        )
    for window in manifest.normal_windows:
        windows.append(
            WindowSpec(
                dataset=manifest.dataset,
                window_id=window.window_id,
                source_id=window.window_id,
                window_kind="normal",
                day=window.day,
                start_ts=window.start_ts,
                end_ts=window.end_ts,
                positive_ids=[],
                positive_types=[],
                positive_names=[],
                metadata={
                    "source": window.source,
                    "window_metadata": dict(window.metadata),
                },
            )
        )
    windows.sort(key=lambda item: (item.start_ts, item.window_id))
    return windows


def _build_topology_feature_map(
    entity_index: EntityIndex,
    topology: TopologyArtifacts,
) -> Dict[str, Dict[str, float]]:
    features = {}

    service_in_degree = Counter()
    service_out_degree = Counter()
    service_in_weight = Counter()
    service_out_weight = Counter()
    for (source, target), weight in topology.service_service_edges.items():
        service_out_degree[source] += 1
        service_in_degree[target] += 1
        service_out_weight[source] += weight
        service_in_weight[target] += weight

    host_in_degree = Counter()
    host_out_degree = Counter()
    host_in_weight = Counter()
    host_out_weight = Counter()
    for (source, target), weight in topology.host_host_edges.items():
        host_out_degree[source] += 1
        host_in_degree[target] += 1
        host_out_weight[source] += weight
        host_in_weight[target] += weight

    for service_name in entity_index.services:
        entity_id = make_entity_id("service", service_name)
        features[entity_id] = {
            "entity_is_service": 1.0,
            "entity_is_host": 0.0,
            "topo_in_degree": float(service_in_degree.get(service_name, 0)),
            "topo_out_degree": float(service_out_degree.get(service_name, 0)),
            "topo_in_weight": float(service_in_weight.get(service_name, 0)),
            "topo_out_weight": float(service_out_weight.get(service_name, 0)),
            "topo_cross_degree": float(len(entity_index.service_to_hosts.get(service_name, []))),
            "topo_cross_weight": float(len(entity_index.service_to_hosts.get(service_name, []))),
        }

    for host_name in entity_index.hosts:
        entity_id = make_entity_id("host", host_name)
        features[entity_id] = {
            "entity_is_service": 0.0,
            "entity_is_host": 1.0,
            "topo_in_degree": float(host_in_degree.get(host_name, 0)),
            "topo_out_degree": float(host_out_degree.get(host_name, 0)),
            "topo_in_weight": float(host_in_weight.get(host_name, 0)),
            "topo_out_weight": float(host_out_weight.get(host_name, 0)),
            "topo_cross_degree": float(len(entity_index.host_to_services.get(host_name, []))),
            "topo_cross_weight": float(len(entity_index.host_to_services.get(host_name, []))),
        }

    return features


def _update_log_features(
    day_state: DayState,
    timestamp: int,
    service_entity_id: Optional[str],
    host_entity_id: Optional[str],
    message: str,
) -> None:
    _record_service_host_presence(
        day_state=day_state,
        timestamp=timestamp,
        service_entity_id=service_entity_id,
        host_entity_id=host_entity_id,
    )
    matched_windows = day_state.matcher.match(timestamp)
    if not matched_windows:
        return

    severity = _infer_log_severity(message)
    for window in matched_windows:
        if service_entity_id:
            day_state.get_accumulator(window.window_id, service_entity_id).log.update(message, severity)
        if host_entity_id:
            day_state.get_accumulator(window.window_id, host_entity_id).log.update(message, severity)


def _update_metric_features(
    day_state: DayState,
    timestamp: int,
    entity_id: Optional[str],
    kpi_name: str,
    value: Optional[float],
) -> None:
    if not entity_id or value is None:
        return
    bucket_timestamp = int(timestamp // BUCKET_SECONDS) * BUCKET_SECONDS
    day_state.metric_day_series[(entity_id, kpi_name)][bucket_timestamp] += float(value)
    day_state.baselines.metric_stats[(entity_id, kpi_name)].update(value)
    matched_windows = day_state.matcher.match(timestamp)
    if not matched_windows:
        return
    for window in matched_windows:
        day_state.get_accumulator(window.window_id, entity_id).metric.update(kpi_name, value)


def _record_service_host_presence(
    day_state: DayState,
    timestamp: int,
    service_entity_id: Optional[str],
    host_entity_id: Optional[str],
) -> None:
    if not service_entity_id or not host_entity_id:
        return
    bucket_timestamp = int(timestamp // BUCKET_SECONDS) * BUCKET_SECONDS
    day_state.service_host_presence[service_entity_id][bucket_timestamp].add(host_entity_id)


def _update_trace_features(
    day_state: DayState,
    timestamp: int,
    service_entity_id: Optional[str],
    host_entity_id: Optional[str],
    duration_ms: Optional[float],
    operation_name: str,
    is_error: bool,
    span_kind: str,
    peer_name: Optional[str],
) -> None:
    if duration_ms is None:
        return
    _record_service_host_presence(
        day_state=day_state,
        timestamp=timestamp,
        service_entity_id=service_entity_id,
        host_entity_id=host_entity_id,
    )

    for entity_id in (service_entity_id, host_entity_id):
        if entity_id:
            day_state.baselines.trace_stats[(entity_id, operation_name)].update(duration_ms)
            if span_kind in ("client", "server"):
                day_state.baselines.trace_stats_by_kind[(entity_id, span_kind, operation_name)].update(
                    duration_ms
                )

    matched_windows = day_state.matcher.match(timestamp)
    if not matched_windows:
        return
    for window in matched_windows:
        if service_entity_id:
            day_state.get_accumulator(window.window_id, service_entity_id).trace.update(
                operation=operation_name,
                duration_ms=duration_ms,
                is_error=is_error,
                span_kind=span_kind,
                peer_name=peer_name,
            )
        if host_entity_id:
            day_state.get_accumulator(window.window_id, host_entity_id).trace.update(
                operation=operation_name,
                duration_ms=duration_ms,
                is_error=is_error,
                span_kind=span_kind,
                peer_name=peer_name,
            )


def _summarize_log_features(accumulator: LogAccumulator) -> Dict[str, float]:
    if accumulator.count == 0:
        return {name: 0.0 for name in LOG_FEATURE_COLUMNS}

    count = float(accumulator.count)
    return {
        "log_count": count,
        "log_error_count": float(accumulator.error_count),
        "log_warn_count": float(accumulator.warn_count),
        "log_info_count": float(accumulator.info_count),
        "log_error_ratio": float(accumulator.error_count) / count,
        "log_warn_ratio": float(accumulator.warn_count) / count,
        "log_unique_message_count": float(len(accumulator.message_counts)),
        "log_message_entropy": _message_entropy(accumulator.message_counts),
    }


def _select_log_signature_features(
    day_states: Mapping[str, DayState],
    limit: int = LOG_SIGNATURE_FEATURE_LIMIT,
) -> Tuple[List[str], Mapping[str, str]]:
    window_support = Counter()
    fault_window_count = 0
    for day_state in day_states.values():
        for window in day_state.windows:
            if window.window_kind != "fault":
                continue
            fault_window_count += 1
            seen_signatures = set()
            for (window_id, _entity_id), accumulator in day_state.entity_windows.items():
                if window_id != window.window_id or not accumulator.log.message_counts:
                    continue
                seen_signatures.update(accumulator.log.message_counts.keys())
            for signature in seen_signatures:
                window_support[signature] += 1

    if not window_support or fault_window_count <= 0:
        return [], {}

    selected = []
    max_support = max(2, int(math.floor(0.80 * fault_window_count)))
    for signature, support in sorted(
        window_support.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if support < 2:
            continue
        if support > max_support:
            continue
        selected.append(signature)
        if len(selected) >= max(0, int(limit)):
            break
    columns = ["log_signature_%02d" % index for index in range(len(selected))]
    return columns, {signature: column for signature, column in zip(selected, columns)}


def _summarize_log_signature_features(
    accumulator: LogAccumulator,
    signature_to_column: Mapping[str, str],
) -> Dict[str, float]:
    if not signature_to_column:
        return {}
    feature_map = {column: 0.0 for column in signature_to_column.values()}
    for signature in accumulator.message_counts:
        column = signature_to_column.get(signature)
        if column is not None:
            feature_map[column] = 1.0
    return feature_map


def _make_numbered_feature_map(
    prefix: str,
    tokens: Sequence[str],
) -> Tuple[List[str], Mapping[str, str]]:
    columns = ["%s_%02d" % (prefix, index) for index in range(len(tokens))]
    return columns, {token: column for token, column in zip(tokens, columns)}


def _group_window_entity_accumulators(
    day_state: DayState,
) -> Mapping[str, List[Tuple[str, EntityWindowAccumulator]]]:
    grouped: DefaultDict[str, List[Tuple[str, EntityWindowAccumulator]]] = defaultdict(list)
    for (window_id, entity_id), accumulator in day_state.entity_windows.items():
        grouped[str(window_id)].append((str(entity_id), accumulator))
    return grouped


def _select_sparse_tokens(
    day_states: Mapping[str, DayState],
    token_extractor: Callable[[str, EntityWindowAccumulator, DayBaselines], Mapping[str, float]],
    limit: int,
    min_support: int = 1,
    max_fault_window_fraction: float = 0.90,
) -> List[str]:
    token_support = Counter()
    token_weight = Counter()
    fault_window_count = 0

    for day_state in day_states.values():
        grouped = _group_window_entity_accumulators(day_state)
        for window in day_state.windows:
            if window.window_kind != "fault":
                continue
            fault_window_count += 1
            window_scores: Dict[str, float] = {}
            for entity_id, accumulator in grouped.get(str(window.window_id), ()):
                for token, value in token_extractor(
                    entity_id,
                    accumulator,
                    day_state.baselines,
                ).items():
                    token_name = str(token).strip()
                    token_value = max(0.0, float(value))
                    if not token_name or token_value <= 0.0:
                        continue
                    window_scores[token_name] = max(
                        token_value,
                        float(window_scores.get(token_name, 0.0)),
                    )
            for token_name, token_value in window_scores.items():
                token_support[token_name] += 1
                token_weight[token_name] += token_value

    if not token_support or fault_window_count <= 0:
        return []

    min_support = max(1, int(min_support))
    if fault_window_count >= 5:
        max_support = max(
            min_support,
            int(math.floor(float(max_fault_window_fraction) * float(fault_window_count))),
        )
    else:
        max_support = fault_window_count

    selected = []
    for token_name, _ in sorted(
        token_weight.items(),
        key=lambda item: (-item[1], -token_support[item[0]], item[0]),
    ):
        support = int(token_support[token_name])
        if support < min_support:
            continue
        if fault_window_count >= 5 and support > max_support:
            continue
        selected.append(token_name)
        if len(selected) >= max(0, int(limit)):
            break
    return selected


def _metric_event_kpi_scores(
    _entity_id: str,
    accumulator: EntityWindowAccumulator,
    _baselines: DayBaselines,
) -> Mapping[str, float]:
    return {
        str(kpi_name): float(score)
        for kpi_name, score in accumulator.metric_event.kpi_peak_scores.items()
        if float(score) > 0.0
    }


def _select_metric_event_detail_features(
    day_states: Mapping[str, DayState],
    limit: int = METRIC_EVENT_KPI_FEATURE_LIMIT,
) -> Tuple[List[str], Mapping[str, str], List[str], Mapping[str, str]]:
    tokens = _select_sparse_tokens(
        day_states=day_states,
        token_extractor=_metric_event_kpi_scores,
        limit=limit,
        min_support=1,
    )
    peak_columns, peak_map = _make_numbered_feature_map("metric_kpi_peak", tokens)
    hit_columns, hit_map = _make_numbered_feature_map("metric_kpi_hit", tokens)
    return peak_columns, peak_map, hit_columns, hit_map


def _summarize_metric_event_detail_features(
    accumulator: MetricEventAccumulator,
    peak_map: Mapping[str, str],
    hit_map: Mapping[str, str],
) -> Dict[str, float]:
    if not peak_map and not hit_map:
        return {}
    feature_map = {}
    for kpi_name, column in peak_map.items():
        feature_map[column] = float(accumulator.kpi_peak_scores.get(kpi_name, 0.0))
    for kpi_name, column in hit_map.items():
        hit_count = int(accumulator.kpi_hit_counts.get(kpi_name, 0))
        feature_map[column] = math.log1p(hit_count) if hit_count > 0 else 0.0
    return feature_map


def _trace_operation_abs_z_scores(
    entity_id: str,
    accumulator: EntityWindowAccumulator,
    baselines: DayBaselines,
) -> Dict[str, float]:
    feature_map = {}
    for span_kind in ("client", "server"):
        kind_operation_durations = accumulator.trace.kind_operation_durations.get(span_kind, {})
        for operation_name, stats in kind_operation_durations.items():
            baseline = baselines.trace_stats_by_kind.get((entity_id, span_kind, operation_name))
            baseline_mean = baseline.mean() if baseline is not None else 0.0
            baseline_std = baseline.std() if baseline is not None else 0.0
            scale = baseline_std if baseline_std > 1e-6 else 1.0
            token = "%s::%s" % (span_kind, operation_name)
            feature_map[token] = abs(stats.mean() - baseline_mean) / scale
    return feature_map


def _select_trace_operation_features(
    day_states: Mapping[str, DayState],
    limit: int = TRACE_OPERATION_FEATURE_LIMIT,
) -> Tuple[List[str], Mapping[str, str]]:
    tokens = _select_sparse_tokens(
        day_states=day_states,
        token_extractor=_trace_operation_abs_z_scores,
        limit=limit,
        min_support=1,
    )
    return _make_numbered_feature_map("trace_operation_z", tokens)


def _summarize_trace_operation_detail_features(
    entity_id: str,
    accumulator: TraceAccumulator,
    baselines: DayBaselines,
    token_to_column: Mapping[str, str],
) -> Dict[str, float]:
    if not token_to_column:
        return {}
    wrapper = EntityWindowAccumulator(trace=accumulator)
    scores = _trace_operation_abs_z_scores(
        entity_id=entity_id,
        accumulator=wrapper,
        baselines=baselines,
    )
    return {
        column: float(scores.get(token, 0.0))
        for token, column in token_to_column.items()
    }


def _trace_peer_share_scores(
    _entity_id: str,
    accumulator: EntityWindowAccumulator,
    _baselines: DayBaselines,
) -> Dict[str, float]:
    feature_map = {}
    totals = {
        "client": int(accumulator.trace.client_span_count),
        "server": int(accumulator.trace.server_span_count),
    }
    for span_kind in ("client", "server"):
        span_total = totals[span_kind]
        if span_total <= 0:
            continue
        for peer_name, count in accumulator.trace.kind_peer_counts.get(span_kind, {}).items():
            token = "%s::%s" % (span_kind, peer_name)
            feature_map[token] = max(
                float(feature_map.get(token, 0.0)),
                float(count) / float(span_total),
            )
    return feature_map


def _select_trace_peer_features(
    day_states: Mapping[str, DayState],
    limit: int = TRACE_PEER_FEATURE_LIMIT,
) -> Tuple[List[str], Mapping[str, str]]:
    tokens = _select_sparse_tokens(
        day_states=day_states,
        token_extractor=_trace_peer_share_scores,
        limit=limit,
        min_support=1,
    )
    return _make_numbered_feature_map("trace_peer_share", tokens)


def _summarize_trace_peer_detail_features(
    accumulator: TraceAccumulator,
    token_to_column: Mapping[str, str],
) -> Dict[str, float]:
    if not token_to_column:
        return {}
    wrapper = EntityWindowAccumulator(trace=accumulator)
    scores = _trace_peer_share_scores(
        _entity_id="",
        accumulator=wrapper,
        _baselines=DayBaselines(),
    )
    return {
        column: float(scores.get(token, 0.0))
        for token, column in token_to_column.items()
    }


def _summarize_metric_features(
    entity_id: str,
    accumulator: MetricAccumulator,
    baselines: DayBaselines,
) -> Dict[str, float]:
    if not accumulator.by_kpi:
        return {name: 0.0 for name in METRIC_FEATURE_COLUMNS}

    series_count = len(accumulator.by_kpi)
    sample_count = 0
    window_means = []
    abs_z_scores = []
    last_abs_z_scores = []
    delta_abs_z_scores = []
    range_abs_z_scores = []
    anomalous_count = 0

    for kpi_name, stats in accumulator.by_kpi.items():
        sample_count += stats.count
        window_mean = stats.mean()
        window_means.append(window_mean)
        baseline = baselines.metric_stats.get((entity_id, kpi_name))
        baseline_mean = baseline.mean() if baseline is not None else 0.0
        baseline_std = baseline.std() if baseline is not None else 0.0
        scale = baseline_std if baseline_std > 1e-6 else 1.0

        abs_z = abs(window_mean - baseline_mean) / scale
        abs_z_scores.append(abs_z)
        if abs_z >= 3.0:
            anomalous_count += 1

        if stats.last is not None:
            last_abs_z_scores.append(abs((stats.last or 0.0) - baseline_mean) / scale)
        if stats.first is not None and stats.last is not None:
            delta_abs_z_scores.append(abs((stats.last or 0.0) - (stats.first or 0.0)) / scale)
        if stats.minimum is not None and stats.maximum is not None:
            range_abs_z_scores.append(abs((stats.maximum or 0.0) - (stats.minimum or 0.0)) / scale)

    return {
        "metric_series_count": float(series_count),
        "metric_sample_count": float(sample_count),
        "metric_window_mean": _mean(window_means),
        "metric_abs_z_mean": _mean(abs_z_scores),
        "metric_abs_z_max": max(abs_z_scores) if abs_z_scores else 0.0,
        "metric_last_abs_z_max": max(last_abs_z_scores) if last_abs_z_scores else 0.0,
        "metric_delta_abs_z_mean": _mean(delta_abs_z_scores),
        "metric_range_abs_z_mean": _mean(range_abs_z_scores),
        "metric_anomalous_kpi_count": float(anomalous_count),
    }


def _summarize_metric_event_features(accumulator: MetricEventAccumulator) -> Dict[str, float]:
    if not accumulator.score_values:
        return {name: 0.0 for name in METRIC_EVENT_FEATURE_COLUMNS}

    peak_kpi_count = sum(
        1 for score in accumulator.kpi_peak_scores.values() if float(score) >= 0.8
    )
    return {
        "metric_event_active_kpi_count": float(len(accumulator.kpi_peak_scores)),
        "metric_event_hit_count": float(len(accumulator.score_values)),
        "metric_event_active_timestamp_count": float(len(accumulator.active_timestamps)),
        "metric_event_score_sum": float(sum(accumulator.score_values)),
        "metric_event_score_mean": _mean(accumulator.score_values),
        "metric_event_score_max": max(accumulator.score_values),
        "metric_event_score_p95": _quantile(accumulator.score_values, 0.95),
        "metric_event_peak_kpi_count": float(peak_kpi_count),
    }


def _summarize_topology_change_features(
    accumulator: TopologyChangeAccumulator,
) -> Dict[str, float]:
    if not accumulator.change_values:
        return {name: 0.0 for name in TOPOLOGY_CHANGE_FEATURE_COLUMNS}
    return {
        "topology_change_count": float(sum(accumulator.change_values)),
    }


def _summarize_trace_features(
    entity_id: str,
    accumulator: TraceAccumulator,
    baselines: DayBaselines,
) -> Dict[str, float]:
    if accumulator.span_count == 0:
        return {name: 0.0 for name in TRACE_FEATURE_COLUMNS}

    weighted_z = []
    anomalous_operations = 0
    for operation_name, stats in accumulator.operation_durations.items():
        baseline = baselines.trace_stats.get((entity_id, operation_name))
        baseline_mean = baseline.mean() if baseline is not None else 0.0
        baseline_std = baseline.std() if baseline is not None else 0.0
        scale = baseline_std if baseline_std > 1e-6 else 1.0
        abs_z = abs(stats.mean() - baseline_mean) / scale
        weighted_z.extend([abs_z] * stats.count)
        if abs_z >= 3.0:
            anomalous_operations += 1

    directional = {}
    for span_kind in ("client", "server"):
        kind_span_count = int(
            accumulator.client_span_count if span_kind == "client" else accumulator.server_span_count
        )
        kind_error_count = int(accumulator.kind_error_counts.get(span_kind, 0))
        kind_durations = list(accumulator.kind_durations_ms.get(span_kind, ()))
        kind_operation_durations = accumulator.kind_operation_durations.get(span_kind, {})
        kind_weighted_z = []
        kind_anomalous_operations = 0
        for operation_name, stats in kind_operation_durations.items():
            baseline = baselines.trace_stats_by_kind.get((entity_id, span_kind, operation_name))
            baseline_mean = baseline.mean() if baseline is not None else 0.0
            baseline_std = baseline.std() if baseline is not None else 0.0
            scale = baseline_std if baseline_std > 1e-6 else 1.0
            abs_z = abs(stats.mean() - baseline_mean) / scale
            kind_weighted_z.extend([abs_z] * stats.count)
            if abs_z >= 3.0:
                kind_anomalous_operations += 1
        directional["trace_%s_error_count" % span_kind] = float(kind_error_count)
        directional["trace_%s_error_ratio" % span_kind] = (
            float(kind_error_count) / float(kind_span_count) if kind_span_count > 0 else 0.0
        )
        directional["trace_%s_duration_mean_ms" % span_kind] = _mean(kind_durations)
        directional["trace_%s_latency_abs_z_mean" % span_kind] = _mean(kind_weighted_z)
        directional["trace_%s_latency_abs_z_max" % span_kind] = (
            max(kind_weighted_z) if kind_weighted_z else 0.0
        )
        directional["trace_%s_anomalous_operation_count" % span_kind] = float(
            kind_anomalous_operations
        )

    server_span_ratio = (
        float(accumulator.server_span_count) / float(accumulator.span_count)
        if accumulator.span_count > 0
        else 0.0
    )
    server_error_ratio = float(directional.get("trace_server_error_ratio", 0.0))
    client_error_ratio = float(directional.get("trace_client_error_ratio", 0.0))
    server_latency_z_mean = float(directional.get("trace_server_latency_abs_z_mean", 0.0))
    client_latency_z_mean = float(directional.get("trace_client_latency_abs_z_mean", 0.0))

    return {
        "trace_span_count": float(accumulator.span_count),
        "trace_client_span_count": float(accumulator.client_span_count),
        "trace_server_span_count": float(accumulator.server_span_count),
        "trace_server_span_ratio": server_span_ratio,
        "trace_error_count": float(accumulator.error_count),
        "trace_error_ratio": float(accumulator.error_count) / float(accumulator.span_count),
        "trace_unique_operation_count": float(len(accumulator.operation_counts)),
        "trace_unique_peer_count": float(len(accumulator.peer_counts)),
        "trace_duration_mean_ms": _mean(accumulator.durations_ms),
        "trace_duration_max_ms": max(accumulator.durations_ms) if accumulator.durations_ms else 0.0,
        "trace_duration_p95_ms": _quantile(accumulator.durations_ms, 0.95),
        "trace_latency_abs_z_mean": _mean(weighted_z),
        "trace_latency_abs_z_max": max(weighted_z) if weighted_z else 0.0,
        "trace_anomalous_operation_count": float(anomalous_operations),
        "trace_server_error_ratio_gap": server_error_ratio - client_error_ratio,
        "trace_server_latency_abs_z_gap": server_latency_z_mean - client_latency_z_mean,
        **directional,
    }


def _finalize_metric_event_features(
    day_state: DayState,
    sigma_window_size: int = 120,
    sigma: float = 3.0,
) -> None:
    day_start_ts = int(day_state.span_start_ts)
    day_end_ts = int(day_state.span_end_ts)
    bucket_count = max(1, int((day_end_ts - day_start_ts) / BUCKET_SECONDS))
    for (entity_id, kpi_name), bucket_values in day_state.metric_day_series.items():
        dense_values = [0.0] * bucket_count
        for bucket_timestamp, value in bucket_values.items():
            bucket_index = int((int(bucket_timestamp) - day_start_ts) / BUCKET_SECONDS)
            if 0 <= bucket_index < bucket_count:
                dense_values[bucket_index] = float(value)
        normalized_values = _minmax_normalize(dense_values)
        anomaly_scores = _eventized_sigma_scores(
            normalized_values,
            window_size=min(int(sigma_window_size), max(5, len(normalized_values) - 2)),
            sigma=float(sigma),
        )
        for bucket_index, score in enumerate(anomaly_scores):
            if score <= 0.0:
                continue
            timestamp = day_start_ts + bucket_index * BUCKET_SECONDS
            if not day_state.matcher.has_bucket(timestamp):
                continue
            for window in day_state.matcher.match(timestamp):
                day_state.get_accumulator(window.window_id, entity_id).metric_event.update(
                    timestamp=timestamp,
                    kpi_name=kpi_name,
                    score=float(score),
                )


def replace_metric_events_from_artifact(
    day_states: Mapping[str, DayState],
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Replace native eventized-sigma state with validated artifact events."""

    cleared_accumulator_count = 0
    window_lookup: Dict[str, Tuple[DayState, WindowSpec]] = {}
    for day, day_state in sorted(day_states.items()):
        for accumulator in day_state.entity_windows.values():
            accumulator.metric_event = MetricEventAccumulator()
            cleared_accumulator_count += 1
        for window in day_state.windows:
            for native_id in {str(window.source_id), str(window.window_id)}:
                existing = window_lookup.get(native_id)
                if existing is not None and existing[1].window_id != window.window_id:
                    raise ValueError(
                        "event artifact native case id maps to multiple windows: %s"
                        % native_id
                    )
                window_lookup[native_id] = (day_state, window)

    applied_event_count = 0
    applied_bucket_count = 0
    unmatched_case_ids = set()
    outside_window_event_count = 0
    invalid_event_count = 0
    for event in events:
        native_case_id = str(event.get("native_case_id", "")).strip()
        matched = window_lookup.get(native_case_id)
        if matched is None:
            unmatched_case_ids.add(native_case_id or "<missing>")
            continue
        day_state, window = matched
        service = str(event.get("service", "")).strip()
        if not service:
            invalid_event_count += 1
            continue
        if service.startswith("host::"):
            entity_id = make_entity_id("host", service.split("::", 1)[1])
        elif service.startswith("host:") or service.startswith("service:"):
            entity_id = service
        else:
            entity_id = make_entity_id("service", service)
        kpi_name = str(
            event.get("metric_name") or event.get("metric_id") or "metric_event"
        )
        try:
            start_ts = float(event["start_ts"])
            end_ts = float(event["end_ts"])
            peak_ts = float(event["peak_ts"])
            score = float(event["score"])
        except (KeyError, TypeError, ValueError):
            invalid_event_count += 1
            continue
        if (
            not all(math.isfinite(value) for value in (start_ts, end_ts, peak_ts, score))
            or score <= 0.0
            or end_ts < start_ts
        ):
            invalid_event_count += 1
            continue
        clipped_start = max(int(math.floor(start_ts)), int(window.start_ts))
        clipped_end = min(int(math.ceil(end_ts)), int(window.end_ts))
        if clipped_end < clipped_start:
            outside_window_event_count += 1
            continue
        timestamps = list(
            range(clipped_start, clipped_end + 1, BUCKET_SECONDS)
        )
        if not timestamps:
            timestamps = [
                min(max(int(round(peak_ts)), int(window.start_ts)), int(window.end_ts))
            ]
        accumulator = day_state.get_accumulator(
            window.window_id, entity_id
        ).metric_event
        for timestamp in timestamps:
            accumulator.update(
                timestamp=timestamp,
                kpi_name=kpi_name,
                score=score,
            )
            applied_bucket_count += 1
        applied_event_count += 1
    return {
        "mode": "artifact",
        "input_event_count": len(events),
        "applied_event_count": applied_event_count,
        "applied_bucket_count": applied_bucket_count,
        "unmatched_case_count": len(unmatched_case_ids),
        "unmatched_case_ids": sorted(unmatched_case_ids),
        "outside_window_event_count": outside_window_event_count,
        "invalid_event_count": invalid_event_count,
        "cleared_legacy_accumulator_count": cleared_accumulator_count,
        "legacy_metric_events_retained": False,
    }


def _finalize_topology_change_features(day_state: DayState) -> None:
    for service_entity_id, bucket_map in day_state.service_host_presence.items():
        previous_hosts = None
        for bucket_timestamp in sorted(bucket_map):
            current_hosts = set(bucket_map[bucket_timestamp])
            if not current_hosts:
                continue
            if previous_hosts is None:
                previous_hosts = set(current_hosts)
                continue
            delta = float(len(current_hosts.symmetric_difference(previous_hosts)))
            previous_hosts = set(current_hosts)
            if delta <= 0.0 or not day_state.matcher.has_bucket(bucket_timestamp):
                continue
            for window in day_state.matcher.match(bucket_timestamp):
                day_state.get_accumulator(window.window_id, service_entity_id).topology_change.update(delta)


def _add_summary_features(feature_map: Dict[str, float]) -> None:
    has_log = 1.0 if feature_map.get("log_count", 0.0) > 0 else 0.0
    has_metric = 1.0 if feature_map.get("metric_sample_count", 0.0) > 0 else 0.0
    has_trace = 1.0 if feature_map.get("trace_span_count", 0.0) > 0 else 0.0
    feature_map["has_log_signal"] = has_log
    feature_map["has_metric_signal"] = has_metric
    feature_map["has_trace_signal"] = has_trace
    feature_map["modalities_present_count"] = has_log + has_metric + has_trace


def _hd2_metric_name(csv_path: Path) -> str:
    return _canonical_metric_name(csv_path.stem)


def _iter_hd3_parquet_rows(
    archive_path: Path,
    prefix: str,
    columns: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
    selected_members: Optional[Sequence[str]] = None,
) -> Iterable[Mapping[str, Any]]:
    if not archive_path.exists():
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        if selected_members is not None:
            members = list(selected_members)
        else:
            members = sorted(
                (
                    member.name
                    for member in archive.getmembers()
                    if member.isfile() and ("/%s/" % prefix) in member.name
                ),
                key=natural_sort_key,
            )
        if max_files is not None:
            members = members[:max_files]
        for member_name in members:
            payload = archive.extractfile(member_name)
            if payload is None:
                continue
            table = pq.read_table(io.BytesIO(payload.read()), columns=columns)
            for batch in table.to_batches(max_chunksize=2048):
                for row in batch.to_pylist():
                    yield row


def _iter_hd4_parquet_rows(
    parquet_path: Path,
    columns: Optional[Sequence[str]] = None,
) -> Iterable[Mapping[str, Any]]:
    if not parquet_path.exists():
        return
    selected_columns = list(columns) if columns is not None else None
    if selected_columns is not None:
        available_columns = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        selected_columns = [column_name for column_name in selected_columns if column_name in available_columns]
        if not selected_columns:
            return
    table = pq.read_table(parquet_path, columns=selected_columns)
    for batch in table.to_batches(max_chunksize=2048):
        for row in batch.to_pylist():
            yield row


def _hd4_trace_peer_name(span_name: str) -> Optional[str]:
    match = re.search(r"https?://([^/:]+)", span_name or "")
    if not match:
        return None
    return normalize_service_name(match.group(1))


def _process_hd1_day(
    dataset_root: Path,
    day: str,
    day_state: DayState,
    valid_service_ids: Sequence[str],
    valid_host_ids: Sequence[str],
    scan_limits: Optional[Mapping[str, int]] = None,
) -> None:
    service_ids = set(valid_service_ids)
    host_ids = set(valid_host_ids)
    day_root = dataset_root / day / "cloudbed"
    scan_limits = dict(scan_limits or {})

    log_dir = day_root / "log" / "all"
    log_paths = sorted(log_dir.glob("log_filebeat-*.csv"))
    if scan_limits.get("max_log_files_per_day") is not None:
        log_paths = log_paths[: scan_limits["max_log_files_per_day"]]
    for log_path in log_paths:
        day_state.scan_stats["log_files"] += 1
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = int(float(row["timestamp"]))
                if not day_state.matcher.has_bucket(timestamp):
                    continue
                service_name = normalize_service_name(str(row.get("cmdb_id") or ""))
                service_entity_id = make_entity_id("service", service_name)
                if service_entity_id not in service_ids:
                    service_entity_id = None
                _update_log_features(
                    day_state=day_state,
                    timestamp=timestamp,
                    service_entity_id=service_entity_id,
                    host_entity_id=None,
                    message=str(row.get("value") or ""),
                )
                day_state.scan_stats["log_rows"] += 1

    metric_paths = sorted((day_root / "metric" / "container").glob("*.csv"))
    if scan_limits.get("max_metric_files_per_day") is not None:
        metric_paths = metric_paths[: scan_limits["max_metric_files_per_day"]]
    for metric_path in metric_paths:
        if not _should_keep_metric_name(metric_path.stem, strict_legacy=True):
            continue
        day_state.scan_stats["metric_files"] += 1
        with metric_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = int(float(row["timestamp"]))
                if not day_state.matcher.has_bucket(timestamp):
                    continue
                kpi_name = str(row.get("kpi_name") or metric_path.stem)
                if not _should_keep_metric_name(kpi_name, strict_legacy=True):
                    continue
                cmdb_id = str(row.get("cmdb_id") or "")
                if "." not in cmdb_id:
                    continue
                host_name, pod_name = cmdb_id.split(".", 1)
                service_entity_id = make_entity_id("service", normalize_service_name(pod_name))
                if service_entity_id not in service_ids:
                    continue
                host_entity_id = make_entity_id("host", host_name)
                if host_entity_id in host_ids:
                    _record_service_host_presence(
                        day_state=day_state,
                        timestamp=timestamp,
                        service_entity_id=service_entity_id,
                        host_entity_id=host_entity_id,
                    )
                _update_metric_features(
                    day_state=day_state,
                    timestamp=timestamp,
                    entity_id=service_entity_id,
                    kpi_name=kpi_name,
                    value=_safe_float(row.get("value")),
                )
                day_state.scan_stats["metric_rows"] += 1

    node_metric_paths = sorted((day_root / "metric" / "node").glob("*.csv"))
    if scan_limits.get("max_metric_files_per_day") is not None:
        node_metric_paths = node_metric_paths[: scan_limits["max_metric_files_per_day"]]
    for metric_path in node_metric_paths:
        day_state.scan_stats["metric_files"] += 1
        with metric_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = int(float(row["timestamp"]))
                if not day_state.matcher.has_bucket(timestamp):
                    continue
                kpi_name = str(row.get("kpi_name") or metric_path.stem)
                if not _should_keep_metric_name(kpi_name, strict_legacy=True):
                    continue
                host_entity_id = make_entity_id("host", str(row.get("cmdb_id") or ""))
                if host_entity_id not in host_ids:
                    continue
                _update_metric_features(
                    day_state=day_state,
                    timestamp=timestamp,
                    entity_id=host_entity_id,
                    kpi_name=kpi_name,
                    value=_safe_float(row.get("value")),
                )
                day_state.scan_stats["metric_rows"] += 1

    trace_path = day_root / "trace" / "all" / "trace_jaeger-span.csv"
    max_trace_files = scan_limits.get("max_trace_files_per_day")
    if trace_path.exists() and (max_trace_files is None or int(max_trace_files) > 0):
        day_state.scan_stats["trace_files"] += 1
        with trace_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = int(float(row["timestamp"]))
                if not day_state.matcher.has_bucket(timestamp):
                    continue
                service_entity_id = make_entity_id(
                    "service", normalize_service_name(str(row.get("cmdb_id") or ""))
                )
                if service_entity_id not in service_ids:
                    continue
                span_kind = str(row.get("type") or "").lower()
                if span_kind not in ("client", "server"):
                    span_kind = "unknown"
                _update_trace_features(
                    day_state=day_state,
                    timestamp=timestamp,
                    service_entity_id=service_entity_id,
                    host_entity_id=None,
                    duration_ms=_safe_float(row.get("duration")),
                    operation_name=str(row.get("operation_name") or ""),
                    is_error=_trace_is_error({}, row),
                    span_kind=span_kind,
                    peer_name=None,
                )
                day_state.scan_stats["trace_rows"] += 1


def _process_hd2_day(
    dataset_root: Path,
    day: str,
    day_state: DayState,
    valid_service_ids: Sequence[str],
    valid_host_ids: Sequence[str],
    scan_limits: Optional[Mapping[str, int]] = None,
) -> None:
    service_ids = set(valid_service_ids)
    host_ids = set(valid_host_ids)
    day_root = dataset_root / day
    scan_limits = dict(scan_limits or {})
    log_paths = _select_hd2_hourly_files(day_root / "log", "l_*.csv.bz2", day_state.windows)
    if scan_limits.get("max_log_files_per_day") is not None:
        log_paths = log_paths[: scan_limits["max_log_files_per_day"]]
    for log_index, log_path in enumerate(log_paths, start=1):
        _log_scan_progress("hd2", day, "log", log_index, len(log_paths), log_path.name)
        day_state.scan_stats["log_files"] += 1
        with bz2.open(log_path, "rt", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = _parse_utc_seconds(row.get("@timestamp"))
                if timestamp is None or not day_state.matcher.has_bucket(timestamp):
                    continue
                service_entity_id = None
                host_entity_id = None
                pod_name = str(row.get("k8_pod") or "")
                host_name = str(row.get("k8_node_name") or "")
                if pod_name:
                    candidate = make_entity_id("service", normalize_service_name(pod_name))
                    if candidate in service_ids:
                        service_entity_id = candidate
                if host_name:
                    candidate = make_entity_id("host", host_name)
                    if candidate in host_ids:
                        host_entity_id = candidate
                _update_log_features(
                    day_state=day_state,
                    timestamp=timestamp,
                    service_entity_id=service_entity_id,
                    host_entity_id=host_entity_id,
                    message=str(row.get("message") or ""),
                )
                day_state.scan_stats["log_rows"] += 1

    metric_paths = sorted((day_root / "metric" / "container").glob("*.csv"))
    if scan_limits.get("max_metric_files_per_day") is not None:
        metric_paths = metric_paths[: scan_limits["max_metric_files_per_day"]]
    for metric_index, metric_path in enumerate(metric_paths, start=1):
        metric_name = _hd2_metric_name(metric_path)
        if not _should_keep_metric_name(metric_name, strict_legacy=True):
            continue
        _log_scan_progress(
            "hd2",
            day,
            "container-metric",
            metric_index,
            len(metric_paths),
            metric_path.name,
        )
        day_state.scan_stats["metric_files"] += 1
        with metric_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            column_entities = []
            for idx, column_name in enumerate(header[1:], start=1):
                pod_name = _extract_key_value_fragment(column_name, "pod")
                if not pod_name:
                    continue
                entity_id = make_entity_id("service", normalize_service_name(pod_name))
                if entity_id in service_ids:
                    column_entities.append((idx, entity_id))
            for row in reader:
                if not row:
                    continue
                timestamp = int(float(row[0]))
                if not day_state.matcher.has_bucket(timestamp):
                    continue
                for idx, entity_id in column_entities:
                    if idx >= len(row):
                        continue
                    _update_metric_features(
                        day_state=day_state,
                        timestamp=timestamp,
                        entity_id=entity_id,
                        kpi_name=metric_name,
                        value=_safe_float(row[idx]),
                    )
                day_state.scan_stats["metric_rows"] += 1

    node_metric_paths = sorted((day_root / "metric" / "node").glob("*.csv"))
    if scan_limits.get("max_metric_files_per_day") is not None:
        node_metric_paths = node_metric_paths[: scan_limits["max_metric_files_per_day"]]
    for metric_index, metric_path in enumerate(node_metric_paths, start=1):
        if not _should_keep_metric_name(metric_path.stem, strict_legacy=True):
            continue
        _log_scan_progress(
            "hd2",
            day,
            "node-metric",
            metric_index,
            len(node_metric_paths),
            metric_path.name,
        )
        day_state.scan_stats["metric_files"] += 1
        with metric_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            column_entities = []
            for idx, column_name in enumerate(header[1:], start=1):
                host_name = _extract_key_value_fragment(column_name, "kubernetes_node")
                if not host_name:
                    continue
                entity_id = make_entity_id("host", host_name)
                if entity_id in host_ids:
                    column_entities.append((idx, entity_id))
            for row in reader:
                if not row:
                    continue
                timestamp = int(float(row[0]))
                if not day_state.matcher.has_bucket(timestamp):
                    continue
                for idx, entity_id in column_entities:
                    if idx >= len(row):
                        continue
                    _update_metric_features(
                        day_state=day_state,
                        timestamp=timestamp,
                        entity_id=entity_id,
                        kpi_name=metric_path.stem,
                        value=_safe_float(row[idx]),
                    )
                day_state.scan_stats["metric_rows"] += 1

    trace_paths = _select_hd2_hourly_files(day_root / "trace", "t_*.csv.bz2", day_state.windows)
    if scan_limits.get("max_trace_files_per_day") is not None:
        trace_paths = trace_paths[: scan_limits["max_trace_files_per_day"]]
    for trace_index, trace_path in enumerate(trace_paths, start=1):
        _log_scan_progress("hd2", day, "trace", trace_index, len(trace_paths), trace_path.name)
        day_state.scan_stats["trace_files"] += 1
        with bz2.open(trace_path, "rt", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = _parse_utc_seconds(row.get("startTimeMillis"))
                if timestamp is None or not day_state.matcher.has_bucket(timestamp):
                    continue
                pod_name = _process_tag_lookup(row.get("process"), ["name"])
                host_name = _process_tag_lookup(row.get("process"), ["node_name"])
                service_name = normalize_service_name(
                    pod_name or str(_process_lookup(row.get("process"), "serviceName") or "")
                )
                service_entity_id = make_entity_id("service", service_name) if service_name else None
                if service_entity_id not in service_ids:
                    service_entity_id = None
                host_entity_id = make_entity_id("host", host_name) if host_name else None
                if host_entity_id not in host_ids:
                    host_entity_id = None
                _update_trace_features(
                    day_state=day_state,
                    timestamp=timestamp,
                    service_entity_id=service_entity_id,
                    host_entity_id=host_entity_id,
                    duration_ms=(
                        _safe_float(row.get("duration")) / 1000.0
                        if _safe_float(row.get("duration")) is not None
                        else None
                    ),
                    operation_name=str(row.get("operationName") or ""),
                    is_error=_trace_is_error(row.get("tags"), row),
                    span_kind=_trace_span_kind(row.get("tags")),
                    peer_name=_trace_peer_service(row.get("tags")),
                )
                day_state.scan_stats["trace_rows"] += 1


def _hd3_metric_value(row: Mapping[str, Any]) -> Optional[float]:
    kpi_key = str(row.get("kpi_key") or "")
    if kpi_key and kpi_key in row:
        return _safe_float(row.get(kpi_key))
    ignored = {
        "time",
        "cf",
        "device",
        "instance",
        "kpi_key",
        "kpi_name",
        "kubernetes_node",
        "mountpoint",
        "namespace",
        "object_type",
        "pod",
        "sql_type",
        "type",
    }
    for key, value in row.items():
        if key in ignored:
            continue
        number = _safe_float(value)
        if number is not None:
            return number
    return None


def _process_hd3_day(
    dataset_root: Path,
    day: str,
    day_state: DayState,
    valid_service_ids: Sequence[str],
    valid_host_ids: Sequence[str],
    scan_limits: Optional[Mapping[str, int]] = None,
) -> None:
    service_ids = set(valid_service_ids)
    host_ids = set(valid_host_ids)
    archive_path = dataset_root / ("%s.tar.gz" % day)
    if not archive_path.exists():
        return
    scan_limits = dict(scan_limits or {})
    hour_bounds = _window_hour_bounds(day_state.windows)
    log_members = _select_hd3_hourly_members(archive_path, "log-parquet", hour_bounds)
    trace_members = _select_hd3_hourly_members(archive_path, "trace-parquet", hour_bounds)

    log_columns = ["@timestamp", "message", "k8_pod", "k8_node_name"]
    for row in _iter_hd3_parquet_rows(
        archive_path,
        "log-parquet",
        columns=log_columns,
        max_files=scan_limits.get("max_log_files_per_day"),
        selected_members=log_members,
    ):
        timestamp = _parse_utc_seconds(row.get("@timestamp"))
        if timestamp is None or not day_state.matcher.has_bucket(timestamp):
            continue
        service_entity_id = None
        host_entity_id = None
        pod_name = str(row.get("k8_pod") or "")
        host_name = str(row.get("k8_node_name") or "")
        if pod_name:
            candidate = make_entity_id("service", normalize_service_name(pod_name))
            if candidate in service_ids:
                service_entity_id = candidate
        if host_name:
            candidate = make_entity_id("host", host_name)
            if candidate in host_ids:
                host_entity_id = candidate
        _update_log_features(
            day_state=day_state,
            timestamp=timestamp,
            service_entity_id=service_entity_id,
            host_entity_id=host_entity_id,
            message=str(row.get("message") or ""),
        )
        day_state.scan_stats["log_rows"] += 1
    day_state.scan_stats["log_files"] += 1

    for row in _iter_hd3_parquet_rows(
        archive_path,
        "metric-parquet",
        columns=None,
        max_files=scan_limits.get("max_metric_files_per_day"),
    ):
        timestamp = _parse_utc_seconds(row.get("time"))
        if timestamp is None or not day_state.matcher.has_bucket(timestamp):
            continue
        object_type = str(row.get("object_type") or "").lower()
        service_entity_id = None
        host_entity_id = None
        kpi_name = str(row.get("kpi_key") or "")
        if not _should_keep_metric_name(kpi_name):
            continue
        pod_name = str(row.get("pod") or "")
        host_name = str(row.get("kubernetes_node") or row.get("instance") or "")
        if object_type == "pod" and pod_name:
            candidate = make_entity_id("service", normalize_service_name(pod_name))
            if candidate in service_ids:
                service_entity_id = candidate
        elif object_type == "node" and host_name:
            candidate = make_entity_id("host", host_name)
            if candidate in host_ids:
                host_entity_id = candidate
        value = _hd3_metric_value(row)
        if service_entity_id:
            _update_metric_features(day_state, timestamp, service_entity_id, kpi_name, value)
        if host_entity_id:
            _update_metric_features(day_state, timestamp, host_entity_id, kpi_name, value)
        day_state.scan_stats["metric_rows"] += 1
    day_state.scan_stats["metric_files"] += 1

    trace_columns = ["startTimeMillis", "duration", "operationName", "tags", "process"]
    for row in _iter_hd3_parquet_rows(
        archive_path,
        "trace-parquet",
        columns=trace_columns,
        max_files=scan_limits.get("max_trace_files_per_day"),
        selected_members=trace_members,
    ):
        timestamp = _parse_utc_seconds(row.get("startTimeMillis"))
        if timestamp is None or not day_state.matcher.has_bucket(timestamp):
            continue
        pod_name = _process_tag_lookup(row.get("process"), ["name"])
        host_name = _process_tag_lookup(row.get("process"), ["node_name"])
        service_name = normalize_service_name(
            pod_name or str(_process_lookup(row.get("process"), "serviceName") or "")
        )
        service_entity_id = make_entity_id("service", service_name) if service_name else None
        if service_entity_id not in service_ids:
            service_entity_id = None
        host_entity_id = make_entity_id("host", host_name) if host_name else None
        if host_entity_id not in host_ids:
            host_entity_id = None
        _update_trace_features(
            day_state=day_state,
            timestamp=timestamp,
            service_entity_id=service_entity_id,
            host_entity_id=host_entity_id,
            duration_ms=(
                _safe_float(row.get("duration")) / 1000.0
                if _safe_float(row.get("duration")) is not None
                else None
            ),
            operation_name=str(row.get("operationName") or ""),
            is_error=_trace_is_error(row.get("tags"), row),
            span_kind=_trace_span_kind(row.get("tags")),
            peer_name=_trace_peer_service(row.get("tags")),
        )
    day_state.scan_stats["trace_rows"] += 1
    day_state.scan_stats["trace_files"] += 1


def _process_hd4_day(
    dataset_root: Path,
    day: str,
    day_state: DayState,
    valid_service_ids: Sequence[str],
    valid_host_ids: Sequence[str],
    scan_limits: Optional[Mapping[str, int]] = None,
) -> None:
    del valid_host_ids
    service_ids = set(valid_service_ids)
    case_dir = dataset_root / day
    if not case_dir.exists():
        return
    scan_limits = dict(scan_limits or {})

    log_paths = [case_dir / "normal_logs.parquet", case_dir / "abnormal_logs.parquet"]
    max_log_files = scan_limits.get("max_log_files_per_day")
    if max_log_files is not None:
        log_paths = log_paths[: int(max_log_files)]
    for log_index, log_path in enumerate(log_paths, start=1):
        if not log_path.exists():
            continue
        _log_scan_progress("hd4", day, "log", log_index, len(log_paths), log_path.name)
        day_state.scan_stats["log_files"] += 1
        for row in _iter_hd4_parquet_rows(
            log_path,
            columns=["time", "level", "service_name", "message", "attr.k8s.service.name"],
        ):
            timestamp = _parse_utc_seconds(row.get("time"))
            if timestamp is None or not day_state.matcher.has_bucket(timestamp):
                continue
            service_name = normalize_service_name(
                str(row.get("attr.k8s.service.name") or row.get("service_name") or "")
            )
            service_entity_id = make_entity_id("service", service_name) if service_name else None
            if service_entity_id not in service_ids:
                service_entity_id = None
            message = "%s %s" % (str(row.get("level") or ""), str(row.get("message") or ""))
            _update_log_features(
                day_state=day_state,
                timestamp=timestamp,
                service_entity_id=service_entity_id,
                host_entity_id=None,
                message=message.strip(),
            )
            day_state.scan_stats["log_rows"] += 1

    metric_paths = [
        case_dir / "normal_metrics.parquet",
        case_dir / "normal_metrics_sum.parquet",
        case_dir / "abnormal_metrics.parquet",
        case_dir / "abnormal_metrics_sum.parquet",
    ]
    max_metric_files = scan_limits.get("max_metric_files_per_day")
    if max_metric_files is not None:
        metric_paths = metric_paths[: int(max_metric_files)]
    for metric_index, metric_path in enumerate(metric_paths, start=1):
        if not metric_path.exists():
            continue
        _log_scan_progress("hd4", day, "metric", metric_index, len(metric_paths), metric_path.name)
        day_state.scan_stats["metric_files"] += 1
        for row in _iter_hd4_parquet_rows(
            metric_path,
            columns=["time", "metric", "value", "service_name", "attr.k8s.service.name"],
        ):
            timestamp = _parse_utc_seconds(row.get("time"))
            if timestamp is None or not day_state.matcher.has_bucket(timestamp):
                continue
            kpi_name = str(row.get("metric") or "")
            if not _should_keep_metric_name(kpi_name):
                continue
            service_name = normalize_service_name(
                str(row.get("attr.k8s.service.name") or row.get("service_name") or "")
            )
            service_entity_id = make_entity_id("service", service_name) if service_name else None
            if service_entity_id not in service_ids:
                continue
            _update_metric_features(
                day_state=day_state,
                timestamp=timestamp,
                entity_id=service_entity_id,
                kpi_name=kpi_name,
                value=_safe_float(row.get("value")),
            )
            day_state.scan_stats["metric_rows"] += 1

    trace_paths = [case_dir / "normal_traces.parquet", case_dir / "abnormal_traces.parquet"]
    max_trace_files = scan_limits.get("max_trace_files_per_day")
    if max_trace_files is not None:
        trace_paths = trace_paths[: int(max_trace_files)]
    for trace_index, trace_path in enumerate(trace_paths, start=1):
        if not trace_path.exists():
            continue
        _log_scan_progress("hd4", day, "trace", trace_index, len(trace_paths), trace_path.name)
        day_state.scan_stats["trace_files"] += 1
        for row in _iter_hd4_parquet_rows(
            trace_path,
            columns=[
                "time",
                "span_name",
                "attr.span_kind",
                "service_name",
                "attr.k8s.service.name",
                "duration",
                "attr.status_code",
                "attr.http.response.status_code",
            ],
        ):
            timestamp = _parse_utc_seconds(row.get("time"))
            if timestamp is None or not day_state.matcher.has_bucket(timestamp):
                continue
            service_name = normalize_service_name(
                str(row.get("attr.k8s.service.name") or row.get("service_name") or "")
            )
            service_entity_id = make_entity_id("service", service_name) if service_name else None
            if service_entity_id not in service_ids:
                service_entity_id = None
            span_kind = str(row.get("attr.span_kind") or "").strip().lower()
            if span_kind not in ("client", "server"):
                span_kind = "unknown"
            operation_name = str(row.get("span_name") or "")
            raw_duration = _safe_float(row.get("duration"))
            _update_trace_features(
                day_state=day_state,
                timestamp=timestamp,
                service_entity_id=service_entity_id,
                host_entity_id=None,
                duration_ms=(raw_duration / 1000.0 if raw_duration is not None else None),
                operation_name=operation_name,
                is_error=_trace_is_error({}, row),
                span_kind=span_kind,
                peer_name=_hd4_trace_peer_name(operation_name) if span_kind == "client" else None,
            )
            day_state.scan_stats["trace_rows"] += 1


def _build_day_state(
    dataset_root: Path,
    dataset_name: str,
    day: str,
    day_windows: Sequence[WindowSpec],
    span_start_ts: int,
    span_end_ts: int,
    valid_service_ids: Sequence[str],
    valid_host_ids: Sequence[str],
    scan_limits: Optional[Mapping[str, int]] = None,
) -> DayState:
    day_state = DayState(
        matcher=DayWindowMatcher(day_windows),
        windows=day_windows,
        span_start_ts=int(span_start_ts),
        span_end_ts=int(span_end_ts),
    )
    if dataset_name == "hd1":
        _process_hd1_day(
            dataset_root,
            day,
            day_state,
            valid_service_ids,
            valid_host_ids,
            scan_limits=scan_limits,
        )
    elif dataset_name == "hd2":
        _process_hd2_day(
            dataset_root,
            day,
            day_state,
            valid_service_ids,
            valid_host_ids,
            scan_limits=scan_limits,
        )
    elif dataset_name == "hd3":
        _process_hd3_day(
            dataset_root,
            day,
            day_state,
            valid_service_ids,
            valid_host_ids,
            scan_limits=scan_limits,
        )
    elif dataset_name == "hd4":
        _process_hd4_day(
            dataset_root,
            day,
            day_state,
            valid_service_ids,
            valid_host_ids,
            scan_limits=scan_limits,
        )
    else:
        raise ValueError("Unsupported dataset: %s" % dataset_name)
    _finalize_metric_event_features(day_state=day_state)
    _finalize_topology_change_features(day_state=day_state)
    return day_state


def build_window_feature_bundle(
    dataset_root: Path,
    manifest: DatasetManifest,
    entity_index: EntityIndex,
    topology: TopologyArtifacts,
    scan_limits: Optional[Mapping[str, int]] = None,
    day_workers: int = 1,
    event_selection: Optional[Mapping[str, Any]] = None,
) -> WindowFeatureBundle:
    """Build a reusable per-window, per-entity feature dataset."""

    windows = _window_specs_from_manifest(manifest)
    windows_by_day = defaultdict(list)
    for window in windows:
        windows_by_day[window.day].append(window)

    service_ids = [make_entity_id("service", name) for name in entity_index.services]
    host_ids = [make_entity_id("host", name) for name in entity_index.hosts]
    ordered_days = sorted(windows_by_day)
    day_spans_by_day = {span.day: span for span in manifest.day_spans}
    total_days = len(ordered_days)
    worker_count = max(1, min(int(day_workers), len(ordered_days)))
    print(
        "[window-features] dataset=%s progress=0/%d status=starting workers=%d"
        % (manifest.dataset, total_days, worker_count)
    )
    for day in ordered_days:
        print(
            "[window-features] dataset=%s day=%s windows=%d"
            % (manifest.dataset, day, len(windows_by_day[day]))
        )

    day_states = {}
    completed_days = 0
    if worker_count > 1 and len(ordered_days) > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _build_day_state,
                    dataset_root,
                    manifest.dataset,
                    day,
                    tuple(windows_by_day[day]),
                    int(day_spans_by_day[day].start_ts),
                    int(day_spans_by_day[day].end_ts),
                    tuple(service_ids),
                    tuple(host_ids),
                    dict(scan_limits or {}),
                ): day
                for day in ordered_days
            }
            for future in as_completed(futures):
                day = futures[future]
                day_state = future.result()
                day_states[day] = day_state
                completed_days += 1
                print(
                    "[window-features] dataset=%s progress=%d/%d status=day-complete day=%s scan=%s"
                    % (manifest.dataset, completed_days, total_days, day, dict(day_state.scan_stats))
                )
    else:
        for day in ordered_days:
            day_state = _build_day_state(
                dataset_root,
                manifest.dataset,
                day,
                tuple(windows_by_day[day]),
                int(day_spans_by_day[day].start_ts),
                int(day_spans_by_day[day].end_ts),
                tuple(service_ids),
                tuple(host_ids),
                scan_limits=scan_limits,
            )
            day_states[day] = day_state
            completed_days += 1
            print(
                "[window-features] dataset=%s progress=%d/%d status=day-complete day=%s scan=%s"
                % (manifest.dataset, completed_days, total_days, day, dict(day_state.scan_stats))
            )

    event_replacement_audit: Dict[str, Any] = {
        "mode": "legacy_fallback",
        "legacy_metric_events_retained": True,
    }
    if event_selection is not None:
        selected_mode = str(event_selection.get("mode", "")).strip()
        if selected_mode == "artifact":
            event_replacement_audit = replace_metric_events_from_artifact(
                day_states,
                list(event_selection.get("events", [])),
            )
        elif selected_mode == "legacy_fallback":
            event_replacement_audit = {
                "mode": "legacy_fallback",
                "legacy_metric_events_retained": True,
                "fallback_reason": str(
                    event_selection.get("fallback_reason", "")
                ),
            }
        else:
            raise ValueError(
                "event_selection mode must be artifact or legacy_fallback"
            )

    topology_feature_map = _build_topology_feature_map(entity_index, topology)
    ordered_entities = _ordered_entities(entity_index)
    log_signature_columns, log_signature_map = _select_log_signature_features(day_states)
    metric_kpi_peak_columns, metric_kpi_peak_map, metric_kpi_hit_columns, metric_kpi_hit_map = (
        _select_metric_event_detail_features(day_states)
    )
    trace_operation_columns, trace_operation_map = _select_trace_operation_features(day_states)
    trace_peer_columns, trace_peer_map = _select_trace_peer_features(day_states)
    detail_feature_columns = (
        list(metric_kpi_peak_columns)
        + list(metric_kpi_hit_columns)
        + list(trace_operation_columns)
        + list(trace_peer_columns)
    )

    entity_feature_rows = []
    for window in windows:
        day_state = day_states[window.day]
        positive_ids = set(window.positive_ids)
        for entity_id, entity_index_value, entity_type, entity_name in ordered_entities:
            feature_map = dict(ZERO_DYNAMIC_FEATURES)
            feature_map.update({column: 0.0 for column in log_signature_columns})
            feature_map.update({column: 0.0 for column in detail_feature_columns})
            accumulator = day_state.entity_windows.get((window.window_id, entity_id))
            if accumulator is not None:
                feature_map.update(_summarize_log_features(accumulator.log))
                feature_map.update(_summarize_log_signature_features(accumulator.log, log_signature_map))
                feature_map.update(_summarize_metric_features(entity_id, accumulator.metric, day_state.baselines))
                feature_map.update(_summarize_metric_event_features(accumulator.metric_event))
                feature_map.update(
                    _summarize_metric_event_detail_features(
                        accumulator.metric_event,
                        metric_kpi_peak_map,
                        metric_kpi_hit_map,
                    )
                )
                feature_map.update(_summarize_trace_features(entity_id, accumulator.trace, day_state.baselines))
                feature_map.update(
                    _summarize_trace_operation_detail_features(
                        entity_id,
                        accumulator.trace,
                        day_state.baselines,
                        trace_operation_map,
                    )
                )
                feature_map.update(
                    _summarize_trace_peer_detail_features(
                        accumulator.trace,
                        trace_peer_map,
                    )
                )
                feature_map.update(_summarize_topology_change_features(accumulator.topology_change))
            _add_summary_features(feature_map)

            row = {
                "dataset": manifest.dataset,
                "window_id": window.window_id,
                "source_id": window.source_id,
                "window_kind": window.window_kind,
                "day": window.day,
                "start_ts": window.start_ts,
                "end_ts": window.end_ts,
                "entity_id": entity_id,
                "entity_index": entity_index_value,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "is_positive": 1 if entity_id in positive_ids else 0,
            }
            row.update(topology_feature_map.get(entity_id, {name: 0.0 for name in TOPOLOGY_FEATURE_COLUMNS}))
            row.update(feature_map)
            entity_feature_rows.append(row)

    aggregate_scan_stats = Counter()
    for day, day_state in day_states.items():
        for key, value in day_state.scan_stats.items():
            aggregate_scan_stats[key] += value
        aggregate_scan_stats["days_processed"] += 1
        aggregate_scan_stats["window_count"] += len(day_state.windows)

    metadata = {
        "dataset": manifest.dataset,
        "feature_version": "window_features_v6_graph_events_detail_vectors",
        "window_count": len(windows),
        "fault_window_count": sum(1 for item in windows if item.window_kind == "fault"),
        "normal_window_count": sum(1 for item in windows if item.window_kind == "normal"),
        "entity_count": len(ordered_entities),
        "case_window_seconds": manifest.case_window_seconds,
        "guard_band_seconds": manifest.guard_band_seconds,
        "days": sorted(day_states),
        "topology_feature_columns": TOPOLOGY_FEATURE_COLUMNS,
        "dynamic_feature_columns": list(DYNAMIC_FEATURE_COLUMNS) + list(log_signature_columns) + detail_feature_columns,
        "all_feature_columns": TOPOLOGY_FEATURE_COLUMNS + list(DYNAMIC_FEATURE_COLUMNS) + list(log_signature_columns) + detail_feature_columns,
        "selected_log_signature_columns": log_signature_columns,
        "selected_log_signatures": {column: signature for signature, column in log_signature_map.items()},
        "selected_metric_kpi_peak_columns": {
            column: token for token, column in metric_kpi_peak_map.items()
        },
        "selected_metric_kpi_hit_columns": {
            column: token for token, column in metric_kpi_hit_map.items()
        },
        "selected_trace_operation_columns": {
            column: token for token, column in trace_operation_map.items()
        },
        "selected_trace_peer_columns": {
            column: token for token, column in trace_peer_map.items()
        },
        "scan_stats": dict(aggregate_scan_stats),
        "event_source_selection": {
            "selection": (
                {
                    **{
                        key: value
                        for key, value in event_selection.items()
                        if key != "events"
                    },
                    "event_count": len(event_selection.get("events", [])),
                }
                if event_selection is not None
                else {
                    "mode": "legacy_fallback",
                    "fallback_reason": "backward-compatible native default",
                    "event_count": 0,
                }
            ),
            "consumer_audit": event_replacement_audit,
        },
    }

    print(
        "[window-features] dataset=%s progress=%d/%d status=finished entity_rows=%d"
        % (manifest.dataset, completed_days, total_days, len(entity_feature_rows))
    )

    return WindowFeatureBundle(
        dataset=manifest.dataset,
        windows=[item.to_record() for item in windows],
        entity_feature_rows=entity_feature_rows,
        metadata=metadata,
    )


def _window_hour_bounds(windows: Sequence[WindowSpec]) -> Optional[Tuple[int, int]]:
    if not windows:
        return None
    hours = []
    for window in windows:
        start_hour = datetime.fromtimestamp(window.start_ts, LOCAL_TZ).hour
        end_hour = datetime.fromtimestamp(window.end_ts, LOCAL_TZ).hour
        hours.append(start_hour)
        hours.append(end_hour)
    return min(hours), max(hours)


def _window_slot_bounds(
    windows: Sequence[WindowSpec],
    slots_per_hour: int,
) -> Optional[Tuple[int, int]]:
    if not windows or slots_per_hour <= 0:
        return None
    if 60 % slots_per_hour != 0:
        return None
    minutes_per_slot = max(1, int(60 / slots_per_hour))
    slot_indexes = []
    for window in windows:
        start_dt = datetime.fromtimestamp(window.start_ts, LOCAL_TZ)
        end_ts = max(int(window.start_ts), int(window.end_ts) - 1)
        end_dt = datetime.fromtimestamp(end_ts, LOCAL_TZ)
        start_slot = min(
            slots_per_hour - 1,
            int(start_dt.minute / minutes_per_slot),
        )
        end_slot = min(
            slots_per_hour - 1,
            int(end_dt.minute / minutes_per_slot),
        )
        slot_indexes.append(int(start_dt.hour * slots_per_hour) + start_slot + 1)
        slot_indexes.append(int(end_dt.hour * slots_per_hour) + end_slot + 1)
    return min(slot_indexes), max(slot_indexes)


def _select_hd2_hourly_files(
    directory: Path,
    pattern: str,
    windows: Optional[Sequence[WindowSpec]],
) -> List[Path]:
    paths = sorted(directory.glob(pattern), key=lambda path: natural_sort_key(path.name))
    if not windows or not paths:
        return paths
    file_indexes = []
    for path in paths:
        match = re.search(r"_(\d+)\.csv\.bz2$", path.name)
        if not match:
            return paths
        file_indexes.append(int(match.group(1)))
    max_index = max(file_indexes)
    slots_per_hour: Optional[int] = None
    if max_index <= 24:
        slots_per_hour = 1
    elif max_index % 24 == 0:
        slots_per_hour = int(max_index / 24)
    if slots_per_hour is None:
        return paths
    slot_bounds = _window_slot_bounds(windows, slots_per_hour)
    if slot_bounds is None:
        return paths
    start_slot, end_slot = slot_bounds
    start_slot = max(1, int(start_slot) - 1)
    end_slot = min(int(max_index), int(end_slot) + 1)
    selected = []
    for path, file_index in zip(paths, file_indexes):
        if start_slot <= int(file_index) <= end_slot:
            selected.append(path)
    return selected or paths


def _select_hd3_hourly_members(
    archive_path: Path,
    prefix: str,
    hour_bounds: Optional[Tuple[int, int]],
) -> Optional[List[str]]:
    if hour_bounds is None:
        return None
    start_hour, end_hour = hour_bounds
    with tarfile.open(archive_path, "r:gz") as archive:
        members = sorted(
            (
                member.name
                for member in archive.getmembers()
                if member.isfile() and ("/%s/" % prefix) in member.name
            ),
            key=natural_sort_key,
        )
    selected = []
    for member_name in members:
        match = re.search(r"_(\d{2})-\d{2}-\d{2}\.parquet$", member_name)
        if not match:
            selected.append(member_name)
            continue
        hour = int(match.group(1))
        if start_hour <= hour <= end_hour:
            selected.append(member_name)
    return selected or members
