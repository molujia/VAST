"""Runtime migration probes use declared synthetic observable cases only."""
from dataclasses import replace
import json

import pytest
import torch

from test_ranker import setup
from vast_v3.encoders.fragments import EnumerationLimitError
from vast_v3.models.ranker import VASTModel
from vast_v3.runtime import RuntimeConfig, build_objectives, canonical_hash
from vast_v3.training.objectives import TrainingObjectives


def test_default_model_uses_direct_sampling_without_enumeration_guard():
    case, model = setup()
    assert getattr(model, 'fragment_sampler', None) == 'exact-direct-v1'
    model.max_enumerated = 1
    output = model(case, seed=19)
    assert output.fragment_candidates > model.max_enumerated
    assert output.sampling_marks == {}
    assert output.sampling_audit['schema'] == 'vast-v3-exact-direct/1'
    assert output.sampling_audit['seed'] == 19


def test_model_rejects_legacy_marks_on_the_direct_path():
    case, model = setup()
    with pytest.raises(ValueError, match='sampling_marks'):
        model(case, seed=19, sampling_marks={'legacy-candidate': .2})


def test_model_reuses_sampling_metadata_but_recomputes_scores_and_times(monkeypatch):
    import vast_v3.encoders.direct_sampling as direct
    real_index = direct.FragmentIndex
    builds = []
    def counted(*args, **kwargs):
        builds.append(1)
        return real_index(*args, **kwargs)
    monkeypatch.setattr(direct, 'FragmentIndex', counted)
    case, model = setup()
    first = model(case, seed=29)
    with torch.no_grad():
        model.rank[-1].bias.add_(1.)
    shifted = model(case.shift_time(32 * first.delta), seed=29, sampling_audit=first.sampling_audit)
    assert len(builds) == 1, 'identical structural sampling must not rebuild the exact index'
    assert torch.allclose(shifted.scores, first.scores + 1., atol=1e-5)
    assert shifted.sampling_audit == first.sampling_audit
    assert shifted.fragments[0].events[0].start == first.fragments[0].events[0].start + 32 * first.delta
    shifted.scores.sum().backward()
    assert model.scope.token_mlp[0].weight.grad is not None


def test_direct_audit_replays_time_shift_and_rejects_other_seed_or_inventory():
    case, model = setup()
    output = model(case, seed=19)
    assert hasattr(output, 'sampling_audit'), 'direct sampler must expose replay audit'
    moved = model(case.shift_time(32 * output.delta), seed=19,
                  sampling_audit=output.sampling_audit)
    assert output.sampling_audit == moved.sampling_audit
    assert [f.provenance for f in output.fragments] == [f.provenance for f in moved.fragments]
    assert torch.allclose(output.unmarked, moved.unmarked, atol=1e-6)
    with pytest.raises(ValueError, match='audit'):
        model(case, seed=20, sampling_audit=output.sampling_audit)
    with pytest.raises(ValueError, match='audit'):
        model(replace(case, numeric=case.numeric[:1]), seed=19,
              sampling_audit=output.sampling_audit)


def test_enumeration_is_explicit_reference_only_and_rejects_direct_audit():
    case, fixture_model = setup()
    reference = VASTModel(fixture_model.frontend, fragment_sampler='enumerate-v1', max_enumerated=1)
    with pytest.raises(EnumerationLimitError):
        reference(case, seed=19)
    reference.max_enumerated = 100000
    output = reference(case, seed=19)
    assert output.sampling_marks
    assert output.sampling_audit is None
    with pytest.raises(ValueError, match='sampling_audit'):
        reference(case, seed=19, sampling_audit={})
    with pytest.raises(ValueError, match='sampler'):
        VASTModel(fixture_model.frontend, fragment_sampler='auto')


def test_invariance_passes_the_original_direct_audit_to_shifted_view():
    case, fixture_model = setup()

    class RecordingModel(VASTModel):
        def __init__(self, frontend):
            super().__init__(frontend)
            self.calls = []

        def forward(self, case, **kwargs):
            output = super().forward(case, **kwargs)
            self.calls.append((kwargs, output))
            return output

    model = RecordingModel(fixture_model.frontend)
    objective = TrainingObjectives(model)
    losses = objective.case_losses(case, epoch=0, seed=19, components={'invariance'})
    assert losses['invariance'] is not None
    assert len(model.calls) == 2
    first, shifted = model.calls
    assert 'sampling_audit' in shifted[0], 'invariance must use strict direct audit replay'
    assert shifted[0]['sampling_audit'] is first[1].sampling_audit
    assert not shifted[0].get('sampling_marks')
    assert float(losses['invariance'].detach()) < 1e-10


def test_runtime_schema_and_identity_bind_the_sampler_version():
    direct = RuntimeConfig(dataset='rcabench')
    assert direct.schema == 'vast-v3-runtime/3'
    assert direct.fragment_sampler == 'exact-direct-v1'
    reference = replace(direct, fragment_sampler='enumerate-v1')
    assert direct.identity != reference.identity
    assert RuntimeConfig.from_dict(direct.to_dict()) == direct
    with pytest.raises(ValueError, match='schema'):
        replace(direct, schema='vast-v3-runtime/1')
    with pytest.raises(ValueError, match='sampler'):
        replace(direct, fragment_sampler='auto')
    old = direct.to_dict()
    old.pop('fragment_sampler')
    old['schema'] = 'vast-v3-runtime/1'
    with pytest.raises(ValueError):
        RuntimeConfig.from_dict(old)
    _, fixture_model = setup()
    objectives = build_objectives(reference, fixture_model.frontend.preprocessing,
                                 text_encoder=lambda texts: torch.zeros(len(texts), 384))
    assert objectives.model.fragment_sampler == 'enumerate-v1'


def test_sampler_identity_prevents_cross_version_checkpoint_reload(tmp_path):
    from vast_v3.training.checkpoint import save_checkpoint, inspect_checkpoint

    direct = RuntimeConfig(dataset='rcabench')
    assert hasattr(direct, 'fragment_sampler'), 'checkpoint identity must bind sampler version'
    reference = replace(direct, fragment_sampler='enumerate-v1')
    _, model = setup()
    optimizer = torch.optim.Adam(model.parameters())
    identity = dict(data_hash='synthetic', config_hash=reference.identity,
                    preprocessing_hash='synthetic-pre', code_hash='same-code')
    path = tmp_path / 'reference.pt'
    save_checkpoint(path, model=model, optimizer=optimizer, identity=identity,
                    progress=dict(stage='test', epoch=0, slot=0, encoder_version=0,
                                  stage_complete=True), query_sets={'L': [], 'U': []})
    with pytest.raises(ValueError, match='config_hash'):
        inspect_checkpoint(path, expected_identity={**identity, 'config_hash': direct.identity})


def test_smoke_reports_candidate_count_and_persists_compact_audit(tmp_path):
    from vast_v3.smoke import run_smoke

    case, model = setup()
    output = tmp_path / 'direct-smoke'
    config = RuntimeConfig(dataset='rcabench')
    report = run_smoke(case, model.frontend.preprocessing, config,
                       text_encoder=lambda texts: torch.zeros(len(texts), 384),
                       output_dir=output, data_hash='synthetic-runtime-test')
    assert 'enumerated_fragment_candidates' not in report
    assert report['fragment_candidate_count'] >= report['retained_fragments']
    assert report['fragment_sampler'] == 'exact-direct-v1'
    assert report['sampling_mark_count'] == 0
    audit = json.loads((output / 'sampling-audit.json').read_text(encoding='utf-8'))
    assert report['sampling_audit']['schema'] == audit['schema']
    assert report['sampling_audit']['seed'] == config.seed
    assert report['sampling_audit']['sha256'] == canonical_hash(audit)
    assert report['identity']['fragment_sampler'] == config.fragment_sampler
    assert report['checkpoint_sampling_audit_match'] is True
