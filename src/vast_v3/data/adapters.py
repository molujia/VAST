"""Strict raw-observation adapters: no resampling, labels, or learned features.

The only public model input is ``LoadedCase.case``. File paths, hashes and data
quality counters remain in the separate curator-side ``audit`` object.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

from .manifests import (CandidateCatalog, WindowSpec, resolve_aiops_owner,
                        AIOPS_MAPPING_VERSION, AIOPS_MAPPING_SOURCE, AIOPS_MAPPING_SOURCE_SHA256)
from .schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation


@dataclass(frozen=True)
class LoadedCase:
    case: ObservableCase
    audit: dict


def native_seconds(value, *, unit: str) -> float:
    """Convert declared source units only; never guess from timestamp magnitude."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('naive source datetime is forbidden')
        if unit not in ('datetime', 's', 'ms', 'us', 'ns'):
            raise ValueError('unknown timestamp unit')
        result = value.timestamp()
    else:
        factors = {'s': 1., 'ms': 1e3, 'us': 1e6, 'ns': 1e9}
        if unit not in factors or isinstance(value, bool):
            raise ValueError('explicit numeric timestamp unit is required')
        try:
            result = float(value) / factors[unit]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid source timestamp') from exc
    if not math.isfinite(result):
        raise ValueError('source timestamp must be finite')
    return result


def _number(value) -> float:
    if value is None or value == '':
        return float('nan')
    if isinstance(value, bool):
        raise ValueError('boolean is not a numeric observation')
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('malformed numeric observation') from exc


def _required(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'missing/nontext source field {key}')
    return value


def _first_owner_text(row, keys):
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f'nontext resource owner field {key}')
        if value.strip():
            return value.strip()
    return ''


def _resource_service(value):
    """The documented RCABench pod/replicaset suffix rule, not label inference."""
    name = value.rsplit('/', 1)[-1]
    return re.sub(r'-[0-9a-fA-F]{8,10}(?:-[a-z0-9]{5})?$', '', name)


def _rcabench_metric_owner(row, metric):
    # Adapted from HD3_PARQUET_PREPROCESSING.md section 4 and the historical
    # rcl_study.rcabench identity resolver. IDs never become channel features.
    if 'hubble' in metric.lower():
        source = _first_owner_text(row, ('source_workload', 'attr.source_workload', 'attr.source'))
        target = _first_owner_text(row, ('destination_workload', 'attr.destination_workload', 'attr.destination'))
        source, target = _resource_service(source), _resource_service(target)
        if source and target:
            return source, 'hubble'
    direct = _first_owner_text(row, ('service', 'service_name'))
    if direct:
        return direct, 'direct'
    for keys, suffix in ((('attr.k8s.replicaset.name', 'replicaset_name'), True),
                         (('attr.k8s.deployment.name', 'deployment_name'), False),
                         (('attr.k8s.statefulset.name', 'statefulset_name'), False),
                         (('attr.k8s.pod.name', 'pod', 'pod_name'), True)):
        value = _first_owner_text(row, keys)
        if value:
            return (_resource_service(value) if suffix else value), 'resource'
    # This observed Kubernetes namespace-level indicator has no entity owner.
    # Keep an explicit counter; do not invent a service or discard a case.
    if metric == 'k8s.namespace.phase' and _first_owner_text(row, ('attr.k8s.namespace.name',)):
        return None, 'namespace-only'
    raise ValueError(f'unresolved RCABench metric owner at {row.get("_source", "in-memory record")}')


def _check_row(row: Mapping):
    forbidden = {'ground_truth', 'roots', 'root', 'targets', 'target', 'labels', 'label',
                 'fault_type', 'injection', 'positive_ids', 'positive_types', 'is_positive',
                 'metadata_json', 'conclusion'}
    if any(str(k).lower() in forbidden or str(k).lower().startswith(('root_cause', 'fault_', 'positive_')) for k in row):
        raise ValueError('label-bearing fields are forbidden in observation records')


def _physical_identity(row):
    # Full raw label dimensions distinguish physical series; they never become
    # semantic channel tokens or model embeddings.
    return tuple(sorted((k, str(v)) for k, v in row.items()
                        if (k.startswith('attr.') or k in ('source_workload', 'destination_workload',
                            'pod', 'pod_name', 'replicaset_name', 'deployment_name', 'statefulset_name'))
                        and v is not None))


def _relation_id(kind, source, target):
    digest = hashlib.sha256(json.dumps([kind, source, target], separators=(',', ':')).encode()).hexdigest()[:24]
    return f'{kind}:{digest}'


class _Builder:
    def __init__(self, window, catalog):
        self.window, self.catalog = window, catalog
        self.entities = {e.id: e for e in catalog.entities}
        self.numeric = defaultdict(list)
        self.logs = []
        self.edges = defaultdict(set)
        self.audit = Counter({'outside_declared_windows': 0, 'trace_parentless_records': 0,
                              'trace_unresolved_parent_records': 0, 'trace_duplicate_records': 0,
                              'metric_resource_owner_fallback_records': 0,
                              'metric_namespace_only_records': 0, 'metric_hubble_caller_records': 0})

    def owner(self, raw, category='service'):
        canonical = (resolve_aiops_owner(self.catalog, raw, category) if self.window.dataset == 'aiops2022_pre'
                     else self.catalog.resolve(raw, category))
        self.entities.setdefault(canonical, Entity(canonical, category, False))
        return canonical

    def accepts(self, time):
        if self.window.includes(time):
            return True
        self.audit['outside_declared_windows'] += 1
        return False

    def edge(self, kind, source, target, time):
        key = kind, source, target
        self.edges[key].add((time, time))
        return _relation_id(*key)

    def finish(self):
        series = []
        for (modality, owner, channel, identity), records in self.numeric.items():
            records.sort(key=lambda r: r[0])
            series.append(NumericSeries(modality, owner, channel, tuple(t for t, _ in records), tuple(v for _, v in records)))
        relations = []
        for (kind, source, target), intervals in sorted(self.edges.items()):
            intervals = tuple(sorted(intervals))
            relations.append(Relation(_relation_id(kind, source, target), source, target, kind,
                                      intervals[0][0], intervals[-1][1], active_intervals=intervals))
        candidate_ids = set(self.catalog.candidate_ids)
        entities = self.catalog.entities + tuple(self.entities[k] for k in sorted(self.entities) if k not in candidate_ids)
        w = self.window
        case = ObservableCase(w.case_id, entities, w.reference_start, w.incident_start, w.incident_end,
                              tuple(series), tuple(sorted(self.logs, key=lambda log: log.time)), tuple(relations))
        return LoadedCase(case, {**self.audit, 'dataset': w.dataset, 'source_id': w.source_id,
                                 'window_source_sha256': w.source_sha256, 'catalog_source_sha256': self.catalog.source_sha256,
                                 'reference_start': w.reference_start, 'reference_end': w.reference_end,
                                 'reference_evidence': w.reference_evidence,
                                 'reference_empty': w.reference_start == w.reference_end,
                                 'entity_mapping_version': AIOPS_MAPPING_VERSION if w.dataset == 'aiops2022_pre' else 'canonical-catalog/1',
                                 'entity_mapping_source': AIOPS_MAPPING_SOURCE if w.dataset == 'aiops2022_pre' else 'vast_v3.data.manifests.CandidateCatalog.resolve',
                                 'entity_mapping_source_sha256': AIOPS_MAPPING_SOURCE_SHA256 if w.dataset == 'aiops2022_pre' else None,
                                 'trace_duration_unit': 'unresolved_native_unit'})


def _traces(builder, rows, *, source):
    records = {}
    for row in rows:
        _check_row(row)
        time = native_seconds(row.get('time' if source == 'rcabench' else 'timestamp'), unit='ms' if source == 'aiops' else builder.timestamp_unit)
        trace, span = _required(row, 'trace_id'), _required(row, 'span_id')
        raw_owner = _required(row, 'service_name' if source == 'rcabench' else 'cmdb_id')
        if 'duration' not in row:
            raise ValueError('trace duration column missing')
        duration = _number(row.get('duration'))
        parent_key = 'parent_span_id' if source == 'rcabench' else 'parent_span'
        if parent_key not in row:
            raise ValueError(f'trace parent column missing: {parent_key}')
        parent = '' if row[parent_key] is None else row[parent_key]
        if not isinstance(parent, str):
            raise ValueError('nontext parent span ID')
        key = trace, span
        # Parent identity may lie before the child's permitted window. Keep
        # the join metadata, not an extra out-of-window observation/entity.
        record = (time, raw_owner, parent, duration)
        if key in records:
            old = records[key]
            if old[:3] != record[:3] or not (old[3] == duration or math.isnan(old[3]) and math.isnan(duration)):
                raise ValueError('conflicting duplicate span record')
            builder.audit['trace_duplicate_records'] += 1
        else:
            records[key] = record
    for (trace, span), (time, raw_owner, parent, duration) in records.items():
        if not builder.accepts(time):
            continue
        owner = builder.owner(raw_owner)
        if not parent or parent == '0':
            builder.audit['trace_parentless_records'] += 1
            continue
        if parent == span:
            raise ValueError('span cannot be its own parent')
        parent_record = records.get((trace, parent))
        if parent_record is None:
            builder.audit['trace_unresolved_parent_records'] += 1
            continue
        edge = builder.edge('call', builder.owner(parent_record[1]), owner, time)
        builder.numeric[('trace', edge, 'duration_native', ())].append((time, duration))


def rcabench_rows_to_case(window: WindowSpec, catalog: CandidateCatalog, *, metric_rows: Iterable[Mapping] = (),
                           log_rows: Iterable[Mapping] = (), trace_rows: Iterable[Mapping] = (), timestamp_unit: str = 'ns') -> LoadedCase:
    builder = _Builder(window, catalog)
    builder.timestamp_unit = timestamp_unit
    for row in metric_rows:
        _check_row(row)
        time = native_seconds(row.get('time'), unit=timestamp_unit)
        metric = _required(row, 'metric')
        raw_owner, owner_kind = _rcabench_metric_owner(row, metric)
        modality = row.get('_modality', 'metrics')
        if modality not in ('metrics', 'metrics_sum', 'metrics_histogram'):
            raise ValueError('unknown RCABench metric source')
        columns = ('count', 'sum', 'min', 'max') if modality == 'metrics_histogram' else ('value',)
        if any(column not in row for column in columns):
            raise ValueError('metric value column missing')
        values = [(column, _number(row[column])) for column in columns]
        if not builder.accepts(time):
            continue
        if owner_kind == 'namespace-only':
            builder.audit['metric_namespace_only_records'] += 1
            continue
        if owner_kind == 'resource':
            builder.audit['metric_resource_owner_fallback_records'] += 1
        elif owner_kind == 'hubble':
            builder.audit['metric_hubble_caller_records'] += 1
        owner = builder.owner(raw_owner)
        for column, value in values:
            builder.numeric[('metric', owner, f'{modality}/{metric}/{column}', _physical_identity(row))].append((time, value))
    for row in log_rows:
        _check_row(row)
        time = native_seconds(row.get('time'), unit=timestamp_unit)
        raw_owner = _required(row, 'service_name')
        message = row.get('message')
        if not isinstance(message, str):
            raise ValueError('log message must be text')
        if builder.accepts(time):
            builder.logs.append(LogRecord(builder.owner(raw_owner), time, message))
    _traces(builder, trace_rows, source='rcabench')
    return builder.finish()


def aiops22_rows_to_case(window: WindowSpec, catalog: CandidateCatalog, *, metric_rows: Iterable[Mapping] = (),
                        log_rows: Iterable[Mapping] = (), trace_rows: Iterable[Mapping] = ()) -> LoadedCase:
    builder = _Builder(window, catalog)
    for row in metric_rows:
        _check_row(row)
        time = native_seconds(row.get('timestamp'), unit='s')
        raw_owner, metric = _required(row, 'cmdb_id'), _required(row, 'kpi_name')
        if 'value' not in row:
            raise ValueError('metric value column missing')
        value = _number(row.get('value'))
        scope = row.get('_scope')
        if scope not in ('container', 'node'):
            raise ValueError('AIOps metric scope must be explicit')
        if scope == 'container' and '.' not in raw_owner:
            raise ValueError('container owner must encode host.pod')
        if not builder.accepts(time):
            continue
        if scope == 'container':
            host, pod = raw_owner.rsplit('.', 1)
            owner = builder.owner(pod)
            builder.edge('hosted_on', owner, builder.owner(host, 'host'), time)
        else:
            owner = builder.owner(raw_owner, 'host')
        builder.numeric[('metric', owner, f'{scope}/{metric}', (raw_owner,))].append((time, value))
    for row in log_rows:
        _check_row(row)
        time = native_seconds(row.get('timestamp'), unit='s')
        owner = _required(row, 'cmdb_id')
        message = row.get('value')
        if not isinstance(message, str):
            raise ValueError('log message must be text')
        if builder.accepts(time):
            builder.logs.append(LogRecord(builder.owner(owner), time, message))
    _traces(builder, trace_rows, source='aiops')
    return builder.finish()


_AIOPS_TRACE_HEADER = ('timestamp', 'cmdb_id', 'span_id', 'trace_id', 'duration',
                       'type', 'status_code', 'operation_name', 'parent_span')


def strict_csv_rows(path: str | Path, *, required_columns: Iterable[str] = (),
                    _aiops_tail_audit: list | None = None):
    """Strict by default; the private policy is used only by the trace wrapper."""
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        last_line = ''

        def lines():
            nonlocal last_line
            for line in handle:
                last_line = line
                yield line

        reader = csv.reader(lines(), strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f'empty CSV: {path}') from exc
        if len(header) != len(set(header)) or not set(required_columns) <= set(header):
            raise ValueError(f'CSV header duplicate or missing required columns: {path}')
        previous_line = reader.line_num
        for values in reader:
            if len(values) != len(header):
                if (_aiops_tail_audit is not None and tuple(header) == _AIOPS_TRACE_HEADER
                        and len(values) == 8 and reader.line_num == previous_line + 1
                        and not last_line.endswith(('\r', '\n')) and handle.read(1) == ''):
                    # Quarantine only a structurally identifiable final record;
                    # malformed existing fields are not excused by the tail policy.
                    # csv.reader(strict=True) still accepts bare quotes inside
                    # unquoted fields. The narrow exception requires valid CSV
                    # quoting as well as exactly one absent trailing field.
                    field = r'(?:[^,"\r\n]*|"(?:[^"\r\n]|"")*")'
                    if re.fullmatch(field + r'(?:,' + field + r')*', last_line) is None:
                        raise ValueError(f'malformed CSV quoting in truncated tail: {path}:{reader.line_num}')
                    row = dict(zip(header, values))
                    native_seconds(row['timestamp'], unit='ms')
                    for key in ('cmdb_id', 'span_id', 'trace_id'):
                        _required(row, key)
                    _number(row['duration'])
                    _aiops_tail_audit.append({'source_file': str(path), 'line_number': reader.line_num,
                        'expected_columns': 9, 'observed_columns': 8, 'missing_column': 'parent_span',
                        'file_size_bytes': path.stat().st_size,
                        'row_sha256': hashlib.sha256(last_line.encode('utf-8')).hexdigest(),
                        'reason': 'unterminated_final_row_missing_parent_span'})
                    return
                raise ValueError(f'CSV row width mismatch at {path}:{reader.line_num}: expected {len(header)}, got {len(values)}')
            yield {**dict(zip(header, values)), '_source': f'{path}:{reader.line_num}'}
            previous_line = reader.line_num


def aiops_trace_csv_rows(path: str | Path, *, audit: dict):
    """Read AIOps traces with the approved, audited EOF-only quarantine policy."""
    quarantined = audit.setdefault('trace_tail_quarantine', [])
    yield from strict_csv_rows(path,
        required_columns=('timestamp', 'cmdb_id', 'span_id', 'trace_id', 'parent_span', 'duration'),
        _aiops_tail_audit=quarantined)


def _tagged(rows, **tags):
    for row in rows:
        yield {**row, **tags}


def load_aiops22_case(root: str | Path, *, window: WindowSpec, catalog: CandidateCatalog) -> LoadedCase:
    base = Path(root) / window.day / 'cloudbed'
    metric_files = {scope: sorted((base / 'metric' / scope).glob('*.csv')) for scope in ('container', 'node')}
    log_files = sorted((base / 'log' / 'all').glob('log_filebeat-*.csv'))
    trace_file = base / 'trace' / 'all' / 'trace_jaeger-span.csv'
    if not all(metric_files.values()) or not log_files or not trace_file.is_file():
        raise FileNotFoundError(f'required AIOps raw modality files absent: {base}')
    metrics = (row for scope, paths in metric_files.items() for path in paths for row in
               _tagged(strict_csv_rows(path, required_columns=('timestamp', 'cmdb_id', 'kpi_name', 'value')), _scope=scope))
    logs = (row for path in log_files for row in strict_csv_rows(path, required_columns=('timestamp', 'cmdb_id', 'value')))
    trace_audit = {}
    result = aiops22_rows_to_case(window, catalog, metric_rows=metrics, log_rows=logs,
                                  trace_rows=aiops_trace_csv_rows(trace_file, audit=trace_audit))
    result.audit.update(trace_audit)
    result.audit['source_files'] = [str(p) for paths in metric_files.values() for p in paths] + [str(p) for p in log_files] + [str(trace_file)]
    return result


def load_rcabench_case(case_dir: str | Path, *, window: WindowSpec, catalog: CandidateCatalog) -> LoadedCase:
    base = Path(case_dir)
    paths = {(phase, kind): base / f'{phase}_{kind}.parquet' for phase in ('normal', 'abnormal')
             for kind in ('metrics', 'metrics_sum', 'metrics_histogram', 'logs', 'traces')}
    required = [base / 'env.json', *paths.values()]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'required RCABench raw files absent: {missing}')
    # Import only after complete source presence validation. No old feature or
    # injection/conclusion file is opened, even if one exists beside the data.
    import pyarrow.parquet as pq

    required_columns = {
        'metrics': {'time', 'service_name', 'metric', 'value'},
        'metrics_sum': {'time', 'service_name', 'metric', 'value'},
        'metrics_histogram': {'time', 'service_name', 'metric', 'count', 'sum', 'min', 'max'},
        'logs': {'time', 'service_name', 'message'},
        'traces': {'time', 'service_name', 'trace_id', 'span_id', 'parent_span_id', 'duration'},
    }

    def rows(kinds):
        for (phase, kind), path in paths.items():
            if kind not in kinds:
                continue
            reader = pq.ParquetFile(path)
            names = reader.schema_arrow.names
            if len(names) != len(set(names)) or not required_columns[kind] <= set(names):
                raise ValueError(f'Parquet schema duplicate or missing required columns: {path}')
            for batch in reader.iter_batches(batch_size=65536):
                for i, row in enumerate(batch.to_pylist()):
                    yield {**row, '_modality': kind, '_source': f'{path}:{i}'}

    result = rcabench_rows_to_case(window, catalog, metric_rows=rows({'metrics', 'metrics_sum', 'metrics_histogram'}),
                                   log_rows=rows({'logs'}), trace_rows=rows({'traces'}), timestamp_unit='ns')
    result.audit['source_files'] = [str(p) for p in required]
    return result
