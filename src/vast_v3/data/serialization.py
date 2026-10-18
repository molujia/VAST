"""Strict, label-free JSON boundary for native observable cases.

Provenance and curator labels have separate formats; unknown fields are never
ignored, even inside an entity, series, log, or relation record.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
import json
import math

from .schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation


def strict_json_loads(payload: str | bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'nonfinite JSON number: {value}')

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)


def exact_fields(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f'{name} fields differ from the strict schema')


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be nonempty text')


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name} must be boolean')


def _sequence(value, name):
    if not isinstance(value, (tuple, list)):
        raise ValueError(f'{name} must be an array')
    return value


def _record(value, cls):
    exact_fields(value, (f.name for f in fields(cls)), cls.__name__)
    data = dict(value)
    text_fields = {Entity: ('id', 'category'), NumericSeries: ('modality', 'owner', 'channel'),
                   LogRecord: ('owner',), Relation: ('id', 'source', 'target', 'kind')}[cls]
    for name in text_fields:
        _text(data[name], name)
    if cls is Entity:
        _bool(data['candidate'], 'candidate')
    elif cls is NumericSeries:
        for name in ('times', 'values', 'observed'):
            data[name] = tuple(_sequence(data[name], name))
        for value in data['times'] + data['values']:
            _number(value, 'numeric observation')
        for value in data['observed']:
            _bool(value, 'observed')
        if len(data['observed']) != len(data['times']):
            raise ValueError('explicit observed mask length differs')
    elif cls is LogRecord:
        _number(data['time'], 'log time')
        if not isinstance(data['text'], str) or type(data['frequency']) is not int:
            raise ValueError('log text/frequency has invalid type')
    else:
        _number(data['start'], 'relation start'); _number(data['end'], 'relation end')
        _bool(data['response'], 'response')
        if data['owner'] is not None:
            _text(data['owner'], 'relation owner')
        intervals = []
        for interval in _sequence(data['active_intervals'], 'active_intervals'):
            if len(_sequence(interval, 'interval')) != 2:
                raise ValueError('relation interval must have two endpoints')
            for value in interval:
                _number(value, 'relation interval')
            intervals.append(tuple(interval))
        data['active_intervals'] = tuple(intervals)
    return cls(**data)


def observable_from_dict(payload: Mapping) -> ObservableCase:
    exact_fields(payload, (f.name for f in fields(ObservableCase)), 'ObservableCase')
    _text(payload['case_id'], 'case_id')
    for name in ('reference_start', 'incident_start', 'incident_end'):
        _number(payload[name], name)
    data = dict(payload)
    for name, cls in (('entities', Entity), ('numeric', NumericSeries), ('logs', LogRecord), ('relations', Relation)):
        data[name] = tuple(_record(row, cls) for row in _sequence(payload[name], name))
    return ObservableCase(**data)


def observable_to_dict(case: ObservableCase) -> dict:
    if type(case) is not ObservableCase:
        raise ValueError('only ObservableCase can cross the observable boundary')
    payload = json.loads(json.dumps(asdict(case), allow_nan=False))
    observable_from_dict(payload)
    return payload


def serialize_observable(case: ObservableCase) -> str:
    return json.dumps(observable_to_dict(case), sort_keys=True, separators=(',', ':'), allow_nan=False)


def deserialize_observable(payload: str | bytes | Mapping) -> ObservableCase:
    return observable_from_dict(payload if isinstance(payload, Mapping) else strict_json_loads(payload))
