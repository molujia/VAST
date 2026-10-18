"""Exact legacy sanitation with catalog work amortized across native logs."""
import random
import re

import pytest

from vast_v3.data import preprocessing as module


def legacy(text, catalog):
    for entity_id in sorted(catalog, key=len, reverse=True):
        if entity_id:
            text = re.sub(r'(?<![\w])' + re.escape(entity_id) + r'(?![\w])', '<ENTITY>', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)
    text = re.sub(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b', '<UUID>', text)
    return re.sub(r'(?i)\b(request[_-]?id|trace[_-]?id|span[_-]?id)\s*[=:]\s*[\w-]+', r'\1=<REQUEST>', text)


@pytest.mark.parametrize('catalog', [(), ('a', 'a-b', 'b'), ('<ENTITY>', 'ENTITY', 'a'),
    ('svc.a', 'svc[a]', 'svc(a)', '\\node'), ('服务', '节点一', 'é'), ('a', 'a', '')])
def test_matches_sequential_legacy_including_marker_and_boundary_collisions(catalog):
    rng = random.Random(42)
    atoms = list(catalog) + ['a-b', 'svc[a]', '10.1.2.3', 'a10.1.2.3z', 'prefixa', 'a_suffix',
        'trace_id=abcd-ef', 'SPANID : z-8', '6afca142-4107-4981-a3d7-00b35f80b184', '<ENTITY>']
    for _ in range(200):
        text = ''.join(rng.choice(atoms) + rng.choice([' ', '.', '/', ':', '-', '']) for _ in range(8))
        assert module.sanitize_log(text, catalog) == legacy(text, catalog)


def test_catalog_escaping_is_not_repeated_per_record(monkeypatch):
    catalog = tuple(f'unique_cost_service_{i}' for i in range(57))
    calls = []
    original = module.re.escape
    def tracked(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(module.re, 'escape', tracked)
    for index in range(30):
        module.sanitize_log(f'unique_cost_service_17 failed request_id={index}', catalog)
    assert len(calls) <= len(catalog), 'catalog regex construction must be amortized, not per record'


def test_catalog_pattern_cache_is_bounded_and_no_raw_text_is_retained():
    cache = module._sanitization_patterns
    cache.cache_clear()
    for index in range(100):
        catalog = (f'cache_bound_service_{index}',)
        text = f'{catalog[0]} diagnostic private text {index}'
        assert module.sanitize_log(text, catalog) == legacy(text, catalog)
    assert cache.cache_info().maxsize == 64
    assert cache.cache_info().currsize == 64
    patterns = cache(('cache_bound_service_0',))
    assert all(isinstance(entity, str) and isinstance(pattern, re.Pattern) for entity, pattern in patterns)
    assert module.sanitize_log('cache_bound_service_0 bad', ('cache_bound_service_0',)) == '<ENTITY> bad'
