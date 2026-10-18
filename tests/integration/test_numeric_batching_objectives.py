"""Fixed synthetic PN/source/U paths under scalar and equal-length execution."""
from dataclasses import replace

import pytest
import torch

from vast_v3.data.preprocessing import Preprocessor
from vast_v3.data.schema import Entity, LogRecord, NumericSeries, ObservableCase, Relation
from vast_v3.runtime import RuntimeConfig, build_objectives
from vast_v3.training.checkpoint import save_checkpoint, load_checkpoint

ATOL, RTOL = 2e-5, 2e-4


def fixture():
    times = tuple(float(i) for i in range(8))
    entities = tuple(Entity(name, 'service') for name in ('a', 'b', 'c', 'empty'))
    relations = (Relation('ab', 'a', 'b', 'call', 0., 8.),
                 Relation('cb', 'c', 'b', 'call', 0., 8.))
    metrics = tuple(NumericSeries('metric', name, 'cpu', times,
        (0., 1., 0., 1., 2.+i, 4.-i, 3., 1.)) for i, name in enumerate(('a', 'b', 'c')))
    traces = tuple(NumericSeries('trace', owner, 'latency', times,
        (1., 2., 1., 2., 6.+i, 3., 5.-i, 2.)) for i, owner in enumerate(('ab', 'cb')))
    logs = tuple(LogRecord('b', float(i), 'visible symptom '+str(i % 2), 1+i % 3)
                 for i in range(8))
    case = ObservableCase('numeric-equivalence', entities, 0., 4., 8., metrics+traces, logs, relations)
    text = lambda texts: torch.tensor([[len(t)/100.] * 384 for t in texts])
    pre = Preprocessor.fit([case], permitted_case_ids={case.case_id}, text_encoder=text,
                           text_identity='synthetic-frozen-text')
    return case, pre, text


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def close(a, b, label):
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), label
    torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL, msg=label)
    return float((a.detach()-b.detach()).abs().max()) if a.numel() else 0.


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('roots', [None, frozenset({'a'})], ids=['U', 'PN-source'])
def test_complete_loss_gradient_update_and_same_mode_checkpoint_replay(tmp_path, device, roots):
    case, pre, text = fixture()
    config = RuntimeConfig(dataset='rcabench', numeric_execution='length-bucket-v1')
    scalar = build_objectives(replace(config, numeric_execution='scalar-v1'), pre,
                               text_encoder=text, device=device)
    batched = build_objectives(config, pre, text_encoder=text, device=device)
    maxima = {'scores': 0., 'loss': 0., 'gradient': 0., 'post_update_weight': 0.}
    with torch.no_grad():
        a, b = scalar.model(case, seed=42), batched.model(case, seed=42)
    maxima['scores'] = close(a.scores, b.scores, 'initial scores')
    assert a.sampling_audit == b.sampling_audit
    assert a.candidate_ids == b.candidate_ids
    assert [f.provenance for f in a.fragments] == [f.provenance for f in b.fragments]
    optimizers = [torch.optim.Adam(obj.parameters(), lr=.001) for obj in (scalar, batched)]
    values = []
    for obj in (scalar, batched):
        losses = obj.case_losses(case, epoch=0, roots=roots, seed=42)
        values.append(losses)
        sum(value for value in losses.values() if value is not None).backward()
    for name in scalar.names:
        a, b = values[0][name], values[1][name]
        assert (a is None) == (b is None), name
        if a is not None:
            maxima['loss'] = max(maxima['loss'], close(a, b, name))
    if roots is None:
        assert scalar.library_membership(case.case_id, 0) == batched.library_membership(case.case_id, 0)
        assert values[0]['U_soft'] is not None
    else:
        assert values[0]['pn'] is not None and values[0]['source'] is not None
        assert values[0]['U_soft'] is None
    params = dict(batched.named_parameters())
    for name, parameter in scalar.named_parameters():
        left, right = parameter.grad, params[name].grad
        assert (left is None) == (right is None), name
        if left is not None:
            maxima['gradient'] = max(maxima['gradient'], close(left, right, 'gradient '+name))
    for obj, optimizer in zip((scalar, batched), optimizers):
        torch.nn.utils.clip_grad_norm_(obj.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    for name, left in scalar.state_dict().items():
        right = batched.state_dict()[name]
        maxima['post_update_weight'] = max(maxima['post_update_weight'], close(left, right, 'updated '+name))
    batched.eval()
    with torch.no_grad():
        before = batched.model(case, seed=42)
    identity = dict(data_hash='synthetic', preprocessing_hash=pre.identity,
                    config_hash=config.identity, code_hash='synthetic-execution-test')
    saved = save_checkpoint(tmp_path / 'batch.pt', model=batched, optimizer=optimizers[1],
        identity=identity, progress=dict(stage='synthetic', epoch=0, slot=1,
        encoder_version=1, stage_complete=True), query_sets={'L': [], 'U': [case.case_id]})
    restored = build_objectives(config, pre, text_encoder=text, device=device)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
    load_checkpoint(saved, model=restored, optimizer=restored_optimizer,
                    expected_identity=identity, require_complete=True, restore_rng=False,
                    map_location=device)
    restored.eval()
    with torch.no_grad():
        after = restored.model(case, seed=42, sampling_audit=before.sampling_audit)
    assert torch.equal(before.scores, after.scores)
    assert before.sampling_audit == after.sampling_audit
    print({'device': device, 'branch': 'U' if roots is None else 'PN-source',
           'atol': ATOL, 'rtol': RTOL, 'measured_max_abs': maxima})
