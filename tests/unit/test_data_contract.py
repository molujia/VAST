from dataclasses import asdict
import math
import pytest


def api():
    import importlib.util
    assert importlib.util.find_spec('vast_v3.data.schema') is not None, 'observation-only schema is not implemented'
    from vast_v3.data.schema import Entity, NumericSeries, LogRecord, Relation, ObservableCase, LabelStore
    from vast_v3.data.preprocessing import Preprocessor, sanitize_log
    return Entity, NumericSeries, LogRecord, Relation, ObservableCase, LabelStore, Preprocessor, sanitize_log


def fixture():
    Entity, NumericSeries, LogRecord, Relation, Case, Labels, Pre, sanitize = api()
    case = Case('case', (Entity('a','service',True),Entity('b','service',True),Entity('ctx','host',False)),
                0.,4.,8.,(NumericSeries('metric','a','cpu',(0.,1.,2.,4.,5.),(1.,1.,1.,3.,5.)),),
                (LogRecord('a',4.,'request 123 failed',1),),
                (Relation('ab','a','b','call',0.,8.),))
    return case, Labels, Pre, sanitize


def test_catalog_keeps_missing_candidate_and_context_is_not_ranked():
    c, *_ = fixture()
    assert c.candidate_ids == ('a','b')
    assert len(c.entities) == 3
    assert not {'roots','fault_type','ground_truth'} & asdict(c).keys()


def test_relation_intervals_are_half_open_but_observed_points_are_retained():
    _, _, _, Relation, *_ = api()
    continuous = Relation('ab', 'a', 'b', 'call', 1., 3.)
    point = Relation('ab', 'a', 'b', 'call', 3., 3., active_intervals=((3., 3.),))
    assert continuous.is_active(1., 2.) and continuous.is_active(2., 3.)
    assert not continuous.is_active(0., 1.)
    assert not continuous.is_active(3., 4.)
    assert not point.is_active(2., 3.)
    assert point.is_active(3., 4.)
    assert not continuous.is_active(2., 2.)
    assert not point.is_active(3., 3.)


def test_labels_separate_budget_and_joint_roots():
    c, Labels, *_ = fixture()
    labels = Labels({'case':('a','b')},budget=1,max_roots=2)
    assert labels.queried == frozenset()
    assert labels.query(c) == frozenset(('a','b'))
    assert labels.query(c) == frozenset(('a','b'))
    assert labels.spent == 1
    with pytest.raises(ValueError,match='root'):
        Labels({'case':('a','ctx')},1,2).query(c)


def test_reference_fit_does_not_use_fault_or_test_values():
    c, _, Pre, _ = fixture()
    p = Pre.fit([c],permitted_case_ids={'case'})
    assert p.delta == 1.
    assert p.stats[('service','cpu')].center == 1.
    assert p.stats[('service','cpu')].scale == 1.
    assert p.transform('service','cpu',101.) == (100.,True)
    assert p.transform('service','unknown',0.) == (0.,False)
    assert p.transform('service','cpu',math.nan) == (0.,False)
    with pytest.raises(ValueError, match='permitted'):
        Pre.fit([c],permitted_case_ids=set())


def test_time_shift_moves_origin_but_preserves_bins_and_delta():
    c, _, Pre, _ = fixture()
    shifted = c.shift_time(100.)
    p = Pre.fit([c],permitted_case_ids={'case'})
    p2 = Pre.fit([shifted],permitted_case_ids={'case'})
    assert p.delta == p2.delta
    assert c.bin_index(4.2,p.delta) == shifted.bin_index(104.2,p2.delta)


def test_invalid_owner_and_unsorted_native_times_block():
    Entity, Num, Log, Rel, Case, *_ = api()
    with pytest.raises(ValueError,match='time'):
        Num('metric','a','cpu',(2.,1.),(0.,0.))
    with pytest.raises(ValueError,match='owner'):
        Case('x',(Entity('a','service',True),),0.,1.,3.,
             (Num('trace','missing','latency',(0.,1.,2.),(1.,2.,3.)),),(),())


def test_raw_mask_removes_values_logs_and_activity_before_encoding():
    c, *_ = fixture()
    masked, targets = c.mask_blocks({('node','a',4)},delta=1.)
    assert masked.numeric[0].observed[3] is False
    assert masked.numeric[0].values[3] == 0.
    assert len(masked.logs) == 0
    assert len(targets) == 2
    assert c.numeric[0].values[3] == 3.


def test_sanitizer_hides_identity_preserves_symptoms():
    _, _, _, sanitize = fixture()
    s = sanitize('checkout failed at 10.1.2.3 request_id=abc123',('checkout',))
    assert 'checkout' not in s and '10.1.2.3' not in s and 'abc123' not in s
    assert 'failed' in s


def test_log_targets_are_fitted_only_on_reference_and_bound_to_model():
    from dataclasses import replace
    import torch
    c,_,Pre,_=fixture()
    from vast_v3.data.schema import LogRecord
    c=replace(c,logs=(LogRecord('a',1.,'stable',2),LogRecord('a',2.,'stable',2),LogRecord('a',4.,'failed',100)))
    def fixed(texts):
        return torch.tensor([[1. if 'stable' in t else 99.]*384 for t in texts])
    p=Pre.fit([c],permitted_case_ids={'case'},text_encoder=fixed,text_identity='fixed-model-hash')
    assert p.text_reference_mean == (1.,)*384
    assert p.stats[('service','__log_frequency__')].center < 2.
    p2=Pre.from_dict(p.to_dict())
    assert p2.identity==p.identity and p2.text_model_identity=='fixed-model-hash'
