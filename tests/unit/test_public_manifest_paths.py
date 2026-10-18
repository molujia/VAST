"""Portable locations never change the approved standard dataset identities."""

from pathlib import Path

import pytest

from vast_v3.data import manifests


SOURCES = {
    'WINDOWS': 'windows_path',
    'CATALOG': 'entity_catalog_path',
    'TRAIN_POOL': 'train_pool_path',
    'TEST_IDS': 'test_rankings_path',
}
FROZEN_CONFIGS = {
    'rcabench': {
        'expected_windows_sha256': '54de9844a5316e3a17720b7d9ba51695882652b1eb7c2647208d8909c3e374e1',
        'expected_catalog_ids_sha256': 'db0e5629d7dd1be27e96bbfd825c5e454c1717563c1103c77f880e82a9ea082f',
        'expected_train_sha256': '5a1b36dc958f26f8c0661938c8b767b57480c216900be688048225e84a8be913',
        'expected_test_ids_sha256': '184a33a4efbfadcbdfa6da104112c8fe647288007a4e1690f777c38a84b62013',
        'expected_counts': (995, 427),
        'expected_entity_counts': {'service': 54},
        'expected_interval_sources': None,
    },
    'aiops2022_pre': {
        'expected_windows_sha256': '8fd706fbb74db95230e3dd89f4f5a18fb272e5c2d02cfc3b9b35e1969964a84d',
        'expected_catalog_ids_sha256': '626de6e032509f2402a3e55a69766924bf4dc0e8430fb62b57bc4953b9708b59',
        'expected_train_sha256': 'b6af98a6f361685ef6415fac638ff70ab3330f271201a4eb3a4b4044f10d4276',
        'expected_test_ids_sha256': '51107cc4df777e825311f3e89b171b46e6fe16cad82403ef5acd61f7c8bbcb29',
        'expected_counts': (169, 72),
        'expected_entity_counts': {'service': 10, 'host': 6},
        'expected_interval_sources': {
            'run_table_exact': 147,
            'run_table_level_median': 88,
            'run_table_type_median': 6,
        },
    },
}


@pytest.fixture(autouse=True)
def clear_source_environment(monkeypatch):
    for dataset in FROZEN_CONFIGS:
        for source in SOURCES:
            monkeypatch.delenv(f'VAST_{dataset.upper()}_{source}', raising=False)


def configure_sources(monkeypatch, tmp_path, dataset):
    paths = {}
    for source, argument in SOURCES.items():
        # Files need not share a directory or retain their historical names.
        path = tmp_path / f'relocated {source.lower()}' / f'{dataset}.source'
        monkeypatch.setenv(f'VAST_{dataset.upper()}_{source}', str(path))
        paths[argument] = path
    return paths


@pytest.mark.parametrize('dataset', FROZEN_CONFIGS)
def test_standard_manifest_uses_explicit_paths_and_unchanged_frozen_contract(monkeypatch, tmp_path, dataset):
    paths = configure_sources(monkeypatch, tmp_path, dataset)
    captured = {}
    result = object()

    def capture_curator_call(**kwargs):
        captured.update(kwargs)
        return result

    # Isolate source I/O so the real standard entry point's entire contract is
    # checked without distributing the private, frozen source files.
    monkeypatch.setattr(manifests, 'build_curator_manifest', capture_curator_call)
    assert manifests.build_standard_manifest(dataset) is result
    assert captured == {'dataset': dataset, **paths, **FROZEN_CONFIGS[dataset]}
    assert all(isinstance(captured[key], Path) for key in SOURCES.values())


@pytest.mark.parametrize('dataset', FROZEN_CONFIGS)
@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('value', [None, '', ' \t '], ids=['unset', 'empty', 'whitespace'])
def test_standard_manifest_rejects_each_missing_source_before_reading(monkeypatch, tmp_path, dataset, source, value):
    configure_sources(monkeypatch, tmp_path, dataset)
    variable = f'VAST_{dataset.upper()}_{source}'
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr(manifests, 'build_curator_manifest',
                        lambda **kwargs: pytest.fail('incomplete source configuration reached curator'))
    with pytest.raises(ValueError, match=variable):
        manifests.build_standard_manifest(dataset)


@pytest.mark.parametrize('dataset', FROZEN_CONFIGS)
def test_standard_manifest_reports_all_required_sources_when_unconfigured(monkeypatch, dataset):
    monkeypatch.setattr(manifests, 'build_curator_manifest',
                        lambda **kwargs: pytest.fail('missing source configuration reached curator'))
    with pytest.raises(ValueError) as error:
        manifests.build_standard_manifest(dataset)
    for source in SOURCES:
        assert f'VAST_{dataset.upper()}_{source}' in str(error.value)


@pytest.mark.parametrize('dataset', FROZEN_CONFIGS)
def test_standard_manifest_still_rejects_modified_source_bytes(monkeypatch, tmp_path, dataset):
    paths = configure_sources(monkeypatch, tmp_path, dataset)
    windows = paths['windows_path']
    windows.parent.mkdir()
    windows.write_text('not the approved frozen windows\n', encoding='utf-8')
    with pytest.raises(ValueError, match='frozen windows checksum mismatch'):
        manifests.build_standard_manifest(dataset)


def test_unknown_standard_dataset_is_rejected_before_path_configuration():
    with pytest.raises(ValueError, match='unknown standard dataset'):
        manifests.build_standard_manifest('unknown')
