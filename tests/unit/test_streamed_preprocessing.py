"""Exact two-pass reference fitting without retaining the whole case pool."""
from dataclasses import replace
import weakref

import pytest
import torch

from vast_v3.data.preprocessing import Preprocessor, sanitize_log
from vast_v3.data.schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation


def _case(case_id, log_count=0):
    entities = (Entity('service-a', 'service'), Entity('service-b', 'service'),
                Entity('host-c', 'host', False))
    logs = tuple(LogRecord('service-a' if i % 2 else 'host-c', (i % 5) * 1.5,
                           f'{case_id} service-a request_id=req{i % 13} symptom-{i % 11}', 1 + i % 7)
                 for i in range(log_count))
    logs += (LogRecord('service-a', -1., 'before-reference'),
             LogRecord('service-a', 10., 'incident-boundary'),
             LogRecord('service-a', 12., 'after-incident'))
    repeated = NumericSeries('metric', 'service-a', 'cpu', (0., 0., 1., 3., 8., 10., 11.),
                             (0., 10., 20., 30., float('nan'), 1e12, 1e12))
    return ObservableCase(case_id, entities, 0., 10., 12.,
        numeric=(repeated, repeated,
                 NumericSeries('metric', 'host-c', 'capacity', (0., 2., 4., 6., 8., 10.),
                               (10., 10., 10., 10., 20., 1e15)),
                 NumericSeries('trace', 'call-ab', 'duration_native', (0., 0., 2., 4., 8., 10.),
                               (4., 8., 12., 16., 20., 1e15))), logs=logs,
        relations=(Relation('call-ab', 'service-a', 'service-b', 'call', 0., 12.),))


def _cases():
    return [_case('z-case', 513), _case('a-case', 257), _case('m-case')]


class _RecordingTextEncoder:
    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(tuple(texts))
        # Batch-sensitive fixed embeddings expose changed per-case batching.
        base = [sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 1009 for text in texts]
        return (torch.tensor(base, dtype=torch.float32)[:, None] * .125
                + torch.arange(384, dtype=torch.float32)[None, :] * .03125 + len(texts) * .0625)


class _GuardedRepository:
    def __init__(self, cases):
        self._cases = {case.case_id: case for case in cases}
        self.accesses = []

    def __getitem__(self, case_id):
        self.accesses.append(case_id)
        if case_id == 'test-case':
            raise AssertionError('test observations must never be loaded')
        return self._cases[case_id]

    def __iter__(self):
        raise AssertionError('repository population iteration may expose test cases')

    def __len__(self):
        raise AssertionError('fitting needs only the explicit ID list')


def _stream(repository, ids, *, permitted=None, encoder=None):
    method = getattr(Preprocessor, 'fit_repository', None)
    assert callable(method), 'explicit repeatable two-pass repository fit is not implemented'
    return method(repository, ids, permitted_case_ids=set(ids) if permitted is None else permitted,
                  text_encoder=encoder, text_identity='fixed-synthetic-text' if encoder is not None else None)


def test_repository_fit_matches_same_order_list_fit_byte_for_byte_and_exactly_two_passes():
    cases = _cases()
    ids = tuple(case.case_id for case in cases)
    baseline_encoder, streamed_encoder = _RecordingTextEncoder(), _RecordingTextEncoder()
    baseline = Preprocessor.fit(cases, permitted_case_ids=set(ids), text_encoder=baseline_encoder,
                                text_identity='fixed-synthetic-text')
    # Captured from the pre-refactor v2 list fitter, before sharing its kernel.
    assert baseline.identity == 'e8cd87a4a8f4954eac67a168092dd0a135ecff32829099bf988a2371c5b47844'
    repository = _GuardedRepository(cases + [_case('test-case')])
    streamed = _stream(repository, ids, encoder=streamed_encoder)
    assert repository.accesses == list(ids) * 2
    assert streamed.to_dict() == baseline.to_dict()
    assert streamed.identity == baseline.identity
    assert streamed.stats == baseline.stats
    assert streamed.delta == baseline.delta
    assert streamed.gap_series_count == baseline.gap_series_count
    assert streamed.text_reference_mean == baseline.text_reference_mean
    assert streamed.fitted_case_ids == ('a-case', 'm-case', 'z-case')
    assert streamed_encoder.batches == baseline_encoder.batches
    assert [len(batch) for batch in streamed_encoder.batches] == [256, 256, 1, 256, 1]
    assert streamed.stats[('service', 'cpu')].observations == 24
    assert streamed.stats[('service->service', 'duration_native')].observations == 15


def test_repository_fit_keeps_ordered_case_and_log_text_frequency_weighting():
    cases = _cases()
    ids = ('a-case', 'z-case', 'm-case')
    encoder = _RecordingTextEncoder()
    streamed = _stream(_GuardedRepository(cases), ids, encoder=encoder)
    lookup = {case.case_id: case for case in cases}
    expected_texts = [sanitize_log(log.text, tuple(entity.id for entity in lookup[case_id].entities))
                      for case_id in ids for log in lookup[case_id].logs if 0. <= log.time < 10.]
    assert [text for batch in encoder.batches for text in batch] == expected_texts
    assert [len(batch) for batch in encoder.batches] == [256, 1, 256, 256, 1]
    baseline_encoder = _RecordingTextEncoder()
    baseline = Preprocessor.fit([lookup[case_id] for case_id in ids], permitted_case_ids=set(ids),
                                text_encoder=baseline_encoder, text_identity='fixed-synthetic-text')
    assert streamed.identity == baseline.identity


@pytest.mark.parametrize('ids', (('z-case', 'z-case'), ('z-case', 'test-case'),
                                ('z-case', 'a-case', 'test-case')))
def test_duplicate_or_nonpermitted_ids_fail_before_any_case_or_text_access(ids):
    repository = _GuardedRepository(_cases())
    encoder = _RecordingTextEncoder()
    with pytest.raises(ValueError, match='duplicate|permitted'):
        _stream(repository, ids, permitted={'z-case', 'a-case', 'm-case'}, encoder=encoder)
    assert repository.accesses == []
    assert encoder.batches == []


def test_one_shot_id_iterator_is_materialized_once_not_used_as_case_iterator():
    cases = _cases()
    ids = tuple(case.case_id for case in cases)
    repository = _GuardedRepository(cases)
    streamed = _stream(repository, iter(ids), permitted=set(ids))
    baseline = Preprocessor.fit(cases, permitted_case_ids=set(ids))
    assert streamed.identity == baseline.identity
    assert repository.accesses == list(ids) * 2


def test_absent_numeric_reference_support_stops_before_second_pass():
    case = replace(_case('empty'), numeric=())
    repository = _GuardedRepository([case])
    with pytest.raises(ValueError, match='no eligible training reference series'):
        _stream(repository, ('empty',))
    assert repository.accesses == ['empty']


def test_empty_id_population_matches_list_fit_error_without_repository_access():
    repository = _GuardedRepository(_cases())
    with pytest.raises(ValueError, match='no eligible training reference series'):
        _stream(repository, ())
    assert repository.accesses == []


@pytest.mark.parametrize('bad_pass', (1, 2))
def test_repository_key_case_identity_mismatch_is_rejected_on_either_pass(bad_pass):
    class Repository:
        def __init__(self):
            self.accesses = []

        def __getitem__(self, case_id):
            self.accesses.append(case_id)
            return _case('wrong-id' if len(self.accesses) == bad_pass else case_id)

    repository = Repository()
    with pytest.raises(ValueError, match='identity|case.*ID|case.*id'):
        _stream(repository, ('right-id',))
    assert repository.accesses == ['right-id'] * bad_pass


def test_case_objects_are_not_retained_across_the_whole_population():
    class FreshRepository:
        def __init__(self):
            self.refs, self.accesses, self.peak_live = [], [], 0

        def __getitem__(self, case_id):
            self.accesses.append(case_id)
            case = _case(case_id, 3)
            self.refs.append(weakref.ref(case))
            self.peak_live = max(self.peak_live, sum(ref() is not None for ref in self.refs))
            return case

    repository = FreshRepository()
    ids = tuple(f'case-{i:03}' for i in range(40))
    streamed = _stream(repository, ids)
    assert repository.accesses == list(ids) * 2
    assert repository.peak_live <= 2  # Current case plus a transient next load.
    assert not any(ref() is not None for ref in repository.refs)
    assert streamed.stats[('service', 'cpu')].observations == 40 * 8


def test_legacy_v1_roundtrip_and_new_v2_policy_are_unchanged_by_streaming():
    cases = _cases()
    ids = tuple(case.case_id for case in cases)
    streamed = _stream(_GuardedRepository(cases), ids)
    assert streamed.to_dict()['schema'] == 'vast-v3-preprocessing/2'
    assert streamed.scaling_policy == 'median_mad_iqr_population_std_unit_if_constant'
    legacy = streamed.to_dict()
    legacy['schema'] = 'vast-v3-preprocessing/1'
    del legacy['scaling_policy']
    restored = Preprocessor.from_dict(legacy)
    assert restored.to_dict() == legacy
    assert Preprocessor.from_dict(restored.to_dict()).identity == restored.identity
    assert restored.identity != streamed.identity
