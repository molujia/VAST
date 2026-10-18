"""Declared synthetic end-to-end fixtures, never benchmark observations."""
import importlib.util
import json

import pytest
import torch

from vast_v3.data.schema import Entity, NumericSeries, Relation, ObservableCase, LogRecord
from vast_v3.data.preprocessing import Preprocessor
from vast_v3.runtime import RuntimeConfig


def fixture():
    case = ObservableCase('synthetic-smoke',
        (Entity('service:a', 'service'), Entity('service:b', 'service'), Entity('service:missing', 'service')),
        0., 3., 6.,
        (NumericSeries('metric', 'service:a', 'cpu', (0., 1., 2., 3., 4., 5.), (0., 1., 0., 2., 4., 3.)),
         NumericSeries('metric', 'service:b', 'cpu', (0., 1., 2., 3., 4., 5.), (0., 1., 0., 1., 3., 2.))),
        (LogRecord('service:a', 1., 'ready'), LogRecord('service:a', 3., 'failed'),
         LogRecord('service:a', 4., 'timeout')),
        (Relation('ab', 'service:a', 'service:b', 'call', 0., 6.),))
    encoder = lambda texts: torch.tensor([[float(len(t) % 7)] * 384 for t in texts])
    pre = Preprocessor.fit([case], permitted_case_ids={case.case_id},
                           text_encoder=encoder, text_identity='synthetic-text-only')
    return case, pre, encoder


def test_label_free_smoke_runs_all_observation_paths_and_exact_reload(tmp_path):
    assert importlib.util.find_spec('vast_v3.smoke') is not None, 'label-free smoke entry point missing'
    from vast_v3.smoke import run_smoke
    case, pre, encoder = fixture()
    report = run_smoke(case, pre, RuntimeConfig(dataset='rcabench'), text_encoder=encoder,
                       output_dir=tmp_path / 'run', data_hash='declared-synthetic-fixture', device='cpu')
    assert report['complete']
    assert report['root_labels_accessed'] == 0
    assert report['benchmark_result'] is False
    assert report['candidate_count'] == 3
    assert report['checkpoint_reload_max_abs_error'] == 0
    assert report['components']['U_soft']['eligible']
    assert report['components']['local_recovery']['eligible']
    assert report['optimizer_steps'] == 1
    assert report['input_conditioning']['preprocessing_schema'] == pre.to_dict()['schema']
    for key in ('initial_score_max_abs', 'fused_state_max_abs', 'local_state_max_abs', 'edge_state_max_abs'):
        assert report['input_conditioning'][key] >= 0
    assert report['fragment_candidate_count'] >= report['retained_fragments']
    assert 'enumerated_fragment_candidates' not in report
    assert report['U_hypothesis_count'] >= report['candidate_count']
    assert report['U_hypothesis_max_cardinality'] <= 2
    assert set(report['scores']) == set(case.candidate_ids)
    assert json.loads((tmp_path / 'run' / 'smoke-report.json').read_text())['complete']
    with pytest.raises(ValueError, match='exists'):
        run_smoke(case, pre, RuntimeConfig(dataset='rcabench'), text_encoder=encoder,
                  output_dir=tmp_path / 'run', data_hash='declared-synthetic-fixture')


def test_smoke_rejects_label_dependent_variant_before_creating_run(tmp_path):
    assert importlib.util.find_spec('vast_v3.smoke') is not None, 'label-free smoke entry point missing'
    from vast_v3.smoke import run_smoke
    case, pre, encoder = fixture()
    with pytest.raises(ValueError, match='A3'):
        run_smoke(case, pre, RuntimeConfig(dataset='rcabench', variant='A0'), text_encoder=encoder,
                  output_dir=tmp_path / 'run', data_hash='fixture')
    assert not (tmp_path / 'run').exists()


def test_nonfinite_input_diagnostic_does_not_hide_original_smoke_error(tmp_path, monkeypatch):
    from dataclasses import replace
    from vast_v3 import smoke
    original = smoke.build_objectives
    def broken_model(*args, **kwargs):
        objectives = original(*args, **kwargs)
        objectives.model.register_forward_hook(
            lambda module, inputs, output: replace(output, scores=torch.full_like(output.scores, float('nan'))))
        return objectives
    monkeypatch.setattr(smoke, 'build_objectives', broken_model)
    case, pre, encoder = fixture()
    with pytest.raises(ValueError, match='full-catalog finite ranking'):
        smoke.run_smoke(case, pre, RuntimeConfig(dataset='rcabench'), text_encoder=encoder,
                        output_dir=tmp_path / 'run', data_hash='declared-nonfinite-fixture')
    report = json.loads((tmp_path / 'run' / 'smoke-report.json').read_text())
    assert not report['complete']
    assert report['input_conditioning']['initial_score_max_abs'] is None
    assert report['input_conditioning']['initial_score_finite'] is False
    assert report['error'] == 'full-catalog finite ranking contract failed'
