import importlib.util
import math

import pytest
import torch

from test_ranker import setup


def runtime_api():
    assert importlib.util.find_spec('vast_v3.runtime') is not None, 'runtime factory is missing'
    from vast_v3.runtime import RuntimeConfig, build_objectives, package_code_hash
    return RuntimeConfig, build_objectives, package_code_hash


def test_runtime_roundtrip_is_strict_and_binds_approved_context():
    Config, _, _ = runtime_api()
    config = Config(dataset='rcabench')
    assert config.effective_max_roots == 2
    assert config.log_recovery_radius_steps == 2.0
    assert Config.from_dict(config.to_dict()) == config
    changed = Config(dataset='rcabench', log_recovery_radius_steps=3)
    assert changed.identity != config.identity
    with pytest.raises(ValueError, match='fields'):
        Config.from_dict({**config.to_dict(), 'ground_truth': ['a']})
    for value in (-1, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            Config(dataset='rcabench', log_recovery_radius_steps=value)


def test_runtime_rejects_unapproved_priors_and_invalid_parameters():
    Config, _, _ = runtime_api()
    for kwargs in ({'dataset': 'unknown'}, {'dataset': 'rcabench', 'max_roots': 3},
                   {'dataset': 'rcabench', 'seed': True},
                   {'dataset': 'rcabench', 'max_enumerated': 0},
                   {'dataset': 'rcabench', 'variant': 'unknown'}):
        with pytest.raises(ValueError):
            Config(**kwargs)
    assert Config(dataset='aiops2022_pre').effective_max_roots is None


def test_factory_is_reproducible_and_does_not_change_frozen_text():
    Config, build, code_hash = runtime_api()
    case, fixture_model = setup()
    pre = fixture_model.frontend.preprocessing
    text = torch.nn.Linear(1, 1)
    before = text.weight.detach().clone()
    # Forward is unnecessary: this verifies ownership of a supplied frozen asset.
    config = Config(dataset='rcabench')
    a = build(config, pre, text_encoder=text)
    b = build(config, pre, text_encoder=text)
    assert torch.equal(before, text.weight)
    assert not text.weight.requires_grad
    assert a.model.frontend.log_recovery_radius_steps == 2
    assert a.model.frontend.channels == ('cpu',)
    assert a.max_roots == 2
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name]), name
    assert len(code_hash()) == 64


def test_runtime_preprocessing_rejects_invalid_spacing_and_empty_vocabulary():
    from dataclasses import replace
    Config, build, _ = runtime_api()
    _, model = setup()
    encoder = lambda texts: torch.zeros(len(texts), 384)
    for pre in (replace(model.frontend.preprocessing, delta=0),
                replace(model.frontend.preprocessing, stats={})):
        with pytest.raises(ValueError):
            build(Config(dataset='rcabench'), pre, text_encoder=encoder)
