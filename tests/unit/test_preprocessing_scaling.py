"""Approved v2 degenerate-scale fallback and legacy artifact identity."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from vast_v3.data.preprocessing import Preprocessor, Scale
from vast_v3.data.schema import Entity, NumericSeries, ObservableCase


NEW_SCHEMA = 'vast-v3-preprocessing/2'
NEW_POLICY = 'median_mad_iqr_population_std_unit_if_constant'
LEGACY_POLICY = 'median_mad_iqr_unit'


def _case(values, *, incident_values=(), case_id='reference'):
    incident_start = float(max(3, len(values)))
    times = tuple(float(i) for i in range(len(values))) + tuple(
        incident_start + i for i in range(len(incident_values)))
    return ObservableCase(case_id, (Entity('host-a', 'host'),), 0., incident_start,
                          incident_start + len(incident_values) + 1., numeric=(
        NumericSeries('metric', 'host-a', 'capacity', times, tuple(values) + tuple(incident_values)),
        NumericSeries('metric', 'host-a', 'grid', (0., 1., 2.), (0., 0., 0.)),
    ))


def _fit(values, **kwargs):
    case = _case(values, **kwargs)
    return Preprocessor.fit([case], permitted_case_ids={case.case_id})


def _identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _legacy_payload():
    return {'schema': 'vast-v3-preprocessing/1', 'delta': 1., 'fitted_case_ids': ['reference'],
            'gap_series_count': 1, 'text_reference_mean': [], 'text_model_identity': None,
            'stats': [{'category': 'host', 'channel': 'capacity', 'center': 14329019392.,
                       'scale': 1., 'observations': 15}]}


def test_heterogeneous_constant_host_capacities_use_reference_population_std():
    capacity = 14329019392.
    entities = tuple(Entity(f'host-{i}', 'host') for i in range(5))
    numeric = tuple(NumericSeries('metric', entity.id, 'capacity', (0., 1., 2., 3.),
                                  (capacity * (2 if i == 4 else 1),) * 4)
                    for i, entity in enumerate(entities))
    case = ObservableCase('reference', entities, 0., 3., 4., numeric=numeric)
    pre = Preprocessor.fit([case], permitted_case_ids={'reference'})
    stat = pre.stats[('host', 'capacity')]
    reference = np.array([capacity] * 12 + [2 * capacity] * 3)
    assert stat.center == capacity
    assert stat.observations == 15
    assert stat.scale == pytest.approx(np.std(reference, ddof=0))
    assert pre.transform('host', 'capacity', 2 * capacity) == pytest.approx((2.5, True))
    assert all(series.values[0] == series.values[-1] for series in case.numeric)


@pytest.mark.parametrize('values', ((0., 1., 2., 3., 4.), (0., 0., 0., 1., 2.),
                                   (0., 1e-20, 2e-20, 3e-20, 4e-20)))
def test_nonzero_mad_or_iqr_scale_is_bitwise_unchanged(values):
    arr = np.array(values)
    center = float(np.median(arr))
    old_scale = float(1.4826 * np.median(np.abs(arr - center)))
    if old_scale == 0:
        old_scale = float((np.quantile(arr, .75) - np.quantile(arr, .25)) / 1.349)
    assert old_scale > 0
    stat = _fit(values).stats[('host', 'capacity')]
    assert stat.center == center
    assert stat.scale == old_scale


@pytest.mark.parametrize('values', ((0.,), (7.,), (0., 0., 0.), (14329019392.,) * 5))
def test_only_truly_constant_reference_support_uses_unit_fallback(values):
    pre = _fit(values)
    stat = pre.stats[('host', 'capacity')]
    assert stat.center == values[0]
    assert stat.scale == 1.
    assert stat.observations == len(values)
    assert pre.transform('host', 'capacity', values[0]) == (0., True)


@pytest.mark.parametrize('factor', (1e-6, 1., 1000.))
def test_nonconstant_std_fallback_preserves_positive_unit_rescaling(factor):
    values = (10.,) * 9 + (20.,)
    baseline = _fit(values)
    converted = _fit(tuple(value * factor for value in values))
    assert converted.stats[('host', 'capacity')].scale == pytest.approx(
        baseline.stats[('host', 'capacity')].scale * factor)
    for value in values:
        actual, supported = converted.transform('host', 'capacity', value * factor)
        assert supported
        assert actual == pytest.approx(baseline.transform('host', 'capacity', value)[0])


def test_std_fallback_uses_only_the_same_permitted_reference_observations():
    pre = _fit((0.,) * 9 + (10.,), incident_values=(1e40, -1e40))
    stat = pre.stats[('host', 'capacity')]
    assert stat.center == 0.
    assert stat.scale == pytest.approx(3.)
    assert stat.observations == 10
    assert pre.transform('host', 'capacity', 1e40)[0] == pytest.approx(1e40 / 3.)
    with pytest.raises(ValueError, match='outside permitted'):
        Preprocessor.fit([_case((0., 1., 2.), case_id='test')], permitted_case_ids={'reference'})


def test_nonfinite_values_remain_unavailable_and_do_not_enter_std_pool():
    pre = _fit((0.,) * 9 + (10., math.nan, math.inf))
    assert pre.stats[('host', 'capacity')].observations == 10
    assert pre.stats[('host', 'capacity')].scale == pytest.approx(3.)
    assert pre.transform('host', 'capacity', math.nan) == (0., False)
    assert pre.transform('host', 'missing', 12.) == (0., False)


def test_std_fallback_avoids_variance_overflow_for_finite_large_reference_values():
    values = (0.,) * 9 + (1e200,)
    pre = _fit(values)
    assert pre.stats[('host', 'capacity')].scale == pytest.approx(3e199)
    assert pre.transform('host', 'capacity', 1e200)[0] == pytest.approx(10. / 3.)


def test_unrepresentable_reference_statistics_fail_instead_of_fabricating_unit_scale():
    with pytest.raises(ValueError, match='finite.*reference|reference.*finite'):
        _fit((-1e308, 1e308, 1e308))


def test_new_fit_and_manual_fixtures_record_explicit_v2_scaling_policy():
    pre = _fit((0., 1., 2.))
    payload = pre.to_dict()
    assert payload['schema'] == NEW_SCHEMA
    assert payload['scaling_policy'] == NEW_POLICY
    manual = Preprocessor({('host', 'capacity'): Scale(0., 1., 3)}, 1., (), 0)
    assert manual.to_dict()['schema'] == NEW_SCHEMA
    assert manual.to_dict()['scaling_policy'] == NEW_POLICY
    assert Preprocessor.from_dict(payload).to_dict() == payload
    assert Preprocessor.from_dict(payload).identity == pre.identity


def test_legacy_artifact_preserves_exact_stats_serialization_and_identity():
    payload = _legacy_payload()
    pre = Preprocessor.from_dict(deepcopy(payload))
    assert pre.to_dict() == payload
    assert pre.identity == _identity(payload)
    assert pre.scaling_policy == LEGACY_POLICY
    assert pre.transform('host', 'capacity', 2 * 14329019392.) == (14329019392., True)
    assert Preprocessor.from_dict(pre.to_dict()).identity == pre.identity


@pytest.mark.parametrize('schema', (None, 'vast-v3-preprocessing/0', 'vast-v3-preprocessing/999'))
def test_unknown_preprocessing_versions_are_not_silently_relabelled(schema):
    payload = _legacy_payload()
    payload['schema'] = schema
    with pytest.raises(ValueError, match='schema|version'):
        Preprocessor.from_dict(payload)


@pytest.mark.parametrize('policy', (None, LEGACY_POLICY, 'unapproved_policy'))
def test_v2_requires_the_approved_explicit_scaling_policy(policy):
    payload = _legacy_payload()
    payload['schema'] = NEW_SCHEMA
    if policy is not None:
        payload['scaling_policy'] = policy
    with pytest.raises(ValueError, match='policy'):
        Preprocessor.from_dict(payload)


def test_legacy_schema_cannot_claim_the_new_policy():
    payload = _legacy_payload()
    payload['scaling_policy'] = NEW_POLICY
    with pytest.raises(ValueError, match='policy'):
        Preprocessor.from_dict(payload)


def test_v2_identity_blocks_reuse_of_legacy_checkpoint_even_with_identical_stats(tmp_path):
    import torch
    from vast_v3.training.checkpoint import inspect_checkpoint, save_checkpoint

    legacy = Preprocessor.from_dict(_legacy_payload())
    current = Preprocessor(legacy.stats, legacy.delta, legacy.fitted_case_ids,
                           legacy.gap_series_count, legacy.text_reference_mean, legacy.text_model_identity)
    assert current.stats == legacy.stats
    assert current.identity != legacy.identity
    identity = dict(data_hash='fixed-data', config_hash='fixed-config', code_hash='fixed-code',
                    preprocessing_hash=legacy.identity)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = save_checkpoint(tmp_path / 'legacy.pt', model=model, optimizer=optimizer,
                                 identity=identity, progress=dict(stage='smoke', epoch=0, slot=1,
                                 encoder_version=1, stage_complete=True), query_sets={'L': [], 'U': ['reference']})
    assert inspect_checkpoint(checkpoint, expected_identity=identity)['identity'] == identity
    with pytest.raises(ValueError, match='preprocessing_hash'):
        inspect_checkpoint(checkpoint, expected_identity={**identity, 'preprocessing_hash': current.identity})
