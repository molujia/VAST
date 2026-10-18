"""Numerical-execution choices are explicit, versioned, and resume-bound."""
from dataclasses import replace

import pytest
import torch

from test_ranker import setup
from vast_v3.runtime import RuntimeConfig, build_objectives


def test_numeric_execution_is_explicit_and_runtime_v3_is_strict():
    config = RuntimeConfig(dataset='rcabench')
    assert config.schema == 'vast-v3-runtime/3'
    # A candidate that failed a real-data gate cannot become the train default.
    assert config.numeric_execution == 'scalar-v1'
    assert config.numeric_batch_size == 32
    assert config.numeric_max_batch_samples == 32768
    assert RuntimeConfig.from_dict(config.to_dict()) == config
    legacy = config.to_dict()
    for field in ('numeric_execution', 'numeric_batch_size', 'numeric_max_batch_samples'):
        legacy.pop(field)
    legacy['schema'] = 'vast-v3-runtime/2'
    with pytest.raises(ValueError):
        RuntimeConfig.from_dict(legacy)
    with pytest.raises(ValueError, match='schema'):
        replace(config, schema='vast-v3-runtime/2')


@pytest.mark.parametrize('field,value', [
    ('numeric_execution', 'auto'), ('numeric_execution', None),
    ('numeric_batch_size', 0), ('numeric_batch_size', -1),
    ('numeric_batch_size', True), ('numeric_batch_size', 3.5),
    ('numeric_max_batch_samples', 0), ('numeric_max_batch_samples', -3),
    ('numeric_max_batch_samples', True), ('numeric_max_batch_samples', 2.0),
])
def test_numeric_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match='numeric'):
        replace(RuntimeConfig(dataset='rcabench'), **{field: value})


def test_numeric_modes_and_resource_boundaries_have_distinct_identities():
    config = RuntimeConfig(dataset='rcabench')
    configs = [config, replace(config, numeric_execution='length-bucket-v1'),
               replace(config, numeric_batch_size=8),
               replace(config, numeric_max_batch_samples=1024)]
    assert len({value.identity for value in configs}) == len(configs)
    assert all(value.schedule() == config.schedule() for value in configs)


def test_factory_binds_numeric_settings_without_changing_architecture_or_initialization():
    _, fixture_model = setup()
    pre = fixture_model.frontend.preprocessing
    text = lambda texts: torch.zeros(len(texts), 384)
    config = RuntimeConfig(dataset='rcabench', numeric_execution='length-bucket-v1', numeric_batch_size=7,
                           numeric_max_batch_samples=111)
    scalar = build_objectives(replace(config, numeric_execution='scalar-v1'), pre,
                              text_encoder=text)
    batched = build_objectives(config, pre, text_encoder=text)
    front = batched.model.frontend
    assert (front.numeric_execution, front.numeric_batch_size,
            front.numeric_max_batch_samples) == ('length-bucket-v1', 7, 111)
    assert scalar.state_dict().keys() == batched.state_dict().keys()
    for name, value in scalar.state_dict().items():
        assert torch.equal(value, batched.state_dict()[name]), name


def test_checkpoint_cannot_silently_cross_numeric_execution_modes(tmp_path):
    from vast_v3.training.checkpoint import save_checkpoint, inspect_checkpoint
    config = RuntimeConfig(dataset='rcabench', numeric_execution='length-bucket-v1')
    scalar = replace(config, numeric_execution='scalar-v1')
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    identity = dict(data_hash='synthetic-data', preprocessing_hash='synthetic-pre',
                    code_hash='same-version', config_hash=scalar.identity)
    path = save_checkpoint(tmp_path / 'scalar.pt', model=model, optimizer=optimizer,
        identity=identity, progress=dict(stage='test', epoch=0, slot=0,
        encoder_version=0, stage_complete=True), query_sets={'L': [], 'U': []})
    with pytest.raises(ValueError, match='config_hash'):
        inspect_checkpoint(path, expected_identity={**identity, 'config_hash': config.identity})
