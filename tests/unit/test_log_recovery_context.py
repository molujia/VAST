"""Recovery-only log context uses native seconds, never neighbouring-bin labels."""
from dataclasses import replace
import inspect
from math import log1p

import pytest
import torch

from vast_v3.data.preprocessing import Preprocessor
from vast_v3.data.schema import Entity, LogRecord, ObservableCase, Relation
from vast_v3.encoders.frontends import NativeFrontend
from vast_v3.models.ranker import VASTModel


@pytest.fixture(autouse=True)
def deterministic_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(previous)


class FixedTextFixture(torch.nn.Module):
    """Small local fixture at the frozen external-model boundary, not a substitute."""
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(1, 384, bias=False)
        with torch.no_grad():
            self.project.weight.fill_(.125)

    def forward(self, texts):
        lengths = self.project.weight.new_tensor([[len(text)] for text in texts])
        return self.project(lengths)


def case_with_logs(logs=(), *, end=30., relations=()):
    return ObservableCase('log-context', (Entity('a', 'service'), Entity('b', 'service')),
                          10., 16., end, logs=tuple(logs), relations=relations)


def frontend(**kwargs):
    # Explicit test-only Delta avoids fitting on this logs-only query fixture.
    pre = Preprocessor({}, 2., ('reference',), 0)
    return NativeFrontend(pre, ('service',), ('call',), ('cpu',),
                          text_encoder=FixedTextFixture(), **kwargs)


def masked_case(context=(), *, target_time=19., end=30., relations=()):
    case = case_with_logs((LogRecord('a', target_time, 'withheld symptom', 7), *context),
                          end=end, relations=relations)
    target_bin = case.bin_index(target_time, 2.)
    masked, _ = case.mask_blocks({('node', 'a', target_bin)}, delta=2.)
    return masked, target_bin


def test_log_recovery_radius_has_declared_default():
    parameter = inspect.signature(NativeFrontend).parameters.get('log_recovery_radius_steps')
    assert parameter is not None, 'log recovery radius must be an explicit frontend parameter'
    assert parameter.default == 2.0
    assert frontend().log_recovery_radius_steps == 2.0


@pytest.mark.parametrize('radius', [-.1, float('nan'), float('inf'), -float('inf')])
def test_log_recovery_radius_rejects_nonfinite_or_negative_values(radius):
    with pytest.raises(ValueError, match='radius'):
        frontend(log_recovery_radius_steps=radius)


@pytest.mark.parametrize(('offset_steps', 'eligible'), [
    (-2., True), (2., True), (-2.001, False), (2.001, False),
    (-1.9, True), (1.9, True),
])
def test_native_radius_has_closed_exact_boundaries_around_bin_center(offset_steps, eligible):
    # Bin 4 is [18, 20), with query center 19 and Delta=2. Logs at 23 and
    # 23.002 occupy the SAME bin but must have different eligibility.
    visible = LogRecord('a', 19. + 2. * offset_steps, 'visible symptom')
    case, b = masked_case((visible,))
    output = frontend()(case)
    assert hasattr(output, 'log_context_mask'), 'separate log recovery eligibility is required'
    assert bool(output.log_context_mask[0, b]) == eligible
    assert bool(output.local_context_mask[0, b]) == eligible
    assert bool(output.recovery_local[0, b].abs().sum() > 0) == eligible


def test_fractional_radius_is_in_grid_steps_not_seconds_or_rounded_bins():
    model = frontend(log_recovery_radius_steps=1.25)
    at_boundary, b = masked_case((LogRecord('a', 16.5, 'visible symptom'),))
    outside, _ = masked_case((LogRecord('a', 16.499, 'visible symptom'),))
    assert model.log_recovery_radius_steps == 1.25
    assert model(at_boundary).log_context_mask[0, b]
    assert not model(outside).log_context_mask[0, b]


def test_zero_radius_disables_neighbour_context_for_masked_target():
    case, b = masked_case((LogRecord('a', 17.9, 'visible symptom'),))
    output = frontend(log_recovery_radius_steps=0.)(case)
    assert not output.log_context_mask[0, b]
    assert not output.local_context_mask[0, b]
    assert output.recovery_local[0, b].count_nonzero() == 0


def test_log_context_is_same_entity_only():
    case, b = masked_case((LogRecord('b', 19., 'other entity symptom'),))
    output = frontend()(case)
    assert hasattr(output, 'log_context_mask'), 'separate log recovery eligibility is required'
    assert not output.log_context_mask[0, b]
    assert output.log_context_mask[1, b]
    assert output.recovery_local[0, b].count_nonzero() == 0


@pytest.mark.parametrize(('target_time', 'outside_time', 'end'), [
    (11., 9.9, 30.), (27., 29.75, 29.5), (27., 29.5, 29.5),
])
def test_log_context_excludes_records_outside_permitted_case_window(target_time, outside_time, end):
    case, b = masked_case((LogRecord('a', outside_time, 'outside window'),),
                          target_time=target_time, end=end)
    output = frontend()(case)
    assert hasattr(output, 'log_context_mask'), 'separate log recovery eligibility is required'
    assert not output.log_context_mask[0, b]
    assert output.recovery_local[0, b].count_nonzero() == 0


@pytest.mark.parametrize('outside_time', [9.9, 29.5, 29.75, 30.1])
def test_outside_window_logs_cannot_change_normal_observed_states(outside_time):
    # The final bin is [28, 30), but this case ends at 29.5: a valid bin
    # number does not make a 29.5 or 29.75 record a permitted observation.
    case = case_with_logs((LogRecord('a', 10., 'valid start'),
                           LogRecord('a', 29.25, 'valid last bin')), end=29.5)
    extra = replace(case, logs=case.logs+(LogRecord('a', outside_time, 'outside window', 100),))
    model = frontend()
    baseline, actual = model(case), model(extra)
    for name in ('h', 'local', 'availability', 'coverage', 'telemetry_mask', 'local_mask',
                 'recovery_local', 'log_context_mask'):
        torch.testing.assert_close(getattr(actual, name), getattr(baseline, name), rtol=0, atol=0)


def test_observed_log_window_includes_start_and_excludes_end():
    case = case_with_logs((LogRecord('a', 10., 'at start'), LogRecord('a', 29.5, 'at end')), end=29.5)
    output = frontend()(case)
    assert output.availability[0, 0, 1]
    assert not output.availability[0, 9, 1]
    assert output.local[0, 9].count_nonzero() == 0
    assert output.h[0, 9].count_nonzero() == 0


def test_log_recovery_pools_encoded_visible_records_with_frequency_weights():
    records = (LogRecord('a', 15., 'failure', 1), LogRecord('a', 23., 'retry', 3))
    case, b = masked_case(records)
    model = frontend()
    output = model(case)
    embedded = model.text_encoder([record.text for record in records]).detach()
    extras = torch.tensor([[log1p(record.frequency), record.time-case.reference_start]
                           for record in records])
    encoded = model.log_mlp(torch.cat((embedded, extras), -1))
    pooled = (encoded[0] + 3 * encoded[1]) / 4
    expected = model.local_projection(torch.cat((torch.zeros(64), pooled)))
    torch.testing.assert_close(output.recovery_local[0, b], expected)


def test_masked_target_value_frequency_and_native_time_are_never_recovery_inputs():
    visible = LogRecord('a', 15., 'visible failure', 2)
    first = case_with_logs((LogRecord('a', 18.1, 'hidden first', 1), visible))
    second = replace(first, logs=(LogRecord('a', 19.9, 'hidden completely changed', 999), visible))
    left, _ = first.mask_blocks({('node', 'a', 4)}, delta=2.)
    right, _ = second.mask_blocks({('node', 'a', 4)}, delta=2.)
    model = frontend()
    a, b = model(left), model(right)
    assert hasattr(a, 'log_context_mask'), 'separate log recovery eligibility is required'
    assert a.log_context_mask[0, 4]
    torch.testing.assert_close(a.recovery_local, b.recovery_local, rtol=0, atol=0)
    torch.testing.assert_close(a.h, b.h, rtol=0, atol=0)


def test_recovery_context_is_not_an_observed_event_or_local_state():
    case, b = masked_case((LogRecord('a', 15., 'visible failure'),))
    model = VASTModel(frontend())
    output = model.frontend(case)
    assert output.recovery_local[0, b].abs().sum() > 0
    assert not output.telemetry_mask[0, b]
    assert not output.local_mask[0, b]
    assert not output.availability[0, b].any()
    assert output.local[0, b].count_nonzero() == 0
    assert output.h[0, b].count_nonzero() == 0
    events, _ = model._inventory(case, output)
    assert {(event.entity, event.bin) for event in events} == {(0, 2)}


def test_recovery_radius_cannot_change_normal_encoding_acquisition_or_ranking():
    context = (LogRecord('a', 15., 'visible failure'), LogRecord('a', 23., 'visible retry'),
               LogRecord('b', 19., 'other entity symptom'))
    case, b = masked_case(context, relations=(Relation('ab', 'a', 'b', 'call', 10., 30.),))
    enabled = VASTModel(frontend())
    disabled = VASTModel(frontend(log_recovery_radius_steps=0.))
    disabled.load_state_dict(enabled.state_dict())
    a, c = enabled(case, seed=11), disabled(case, seed=11)
    assert a.front.log_context_mask[0, b] and not c.front.log_context_mask[0, b]
    for name in ('h', 'local', 'availability', 'coverage', 'telemetry_mask', 'local_mask',
                 'edge_states', 'edge_mask'):
        torch.testing.assert_close(getattr(a.front, name), getattr(c.front, name), rtol=0, atol=0)
    assert len(a.fragments) > 0
    assert a.fragments == c.fragments
    assert a.sampling_marks == c.sampling_marks
    torch.testing.assert_close(a.unmarked, c.unmarked, rtol=0, atol=0)
    torch.testing.assert_close(a.scores, c.scores, rtol=0, atol=0)


def test_recovery_backpropagates_through_log_mlp_but_frozen_text_weights_do_not_change():
    case, b = masked_case((LogRecord('a', 15., 'visible failure', 2),))
    text = FixedTextFixture()
    original = text.project.weight.detach().clone()
    model = NativeFrontend(Preprocessor({}, 2., ('reference',), 0), ('service',), ('call',),
                           ('cpu',), text_encoder=text)
    model.train()
    assert not text.training
    assert not text.project.weight.requires_grad
    torch.testing.assert_close(text.project.weight, original, rtol=0, atol=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    model(case).recovery_local[0, b].square().mean().backward()
    gradient = model.log_mlp[0].weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert text.project.weight.grad is None
    optimizer.step()
    torch.testing.assert_close(text.project.weight, original, rtol=0, atol=0)


def test_uniform_time_shift_preserves_log_recovery_context():
    case, b = masked_case((LogRecord('a', 15., 'visible failure'), LogRecord('a', 23., 'retry')))
    model = frontend()
    left, right = model(case), model(case.shift_time(128.))
    assert hasattr(left, 'log_context_mask'), 'separate log recovery eligibility is required'
    assert left.log_context_mask[0, b]
    torch.testing.assert_close(left.log_context_mask, right.log_context_mask)
    torch.testing.assert_close(left.recovery_local, right.recovery_local, rtol=0, atol=0)
