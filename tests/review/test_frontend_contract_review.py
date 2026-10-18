"""Independent R03 review probes: assertions state the required behavior.

These tests intentionally expose implementation gaps; the reviewing agent
does not edit the implementation files under review.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vast_v3.data.schema import Entity, LabelStore, LogRecord, NumericSeries, ObservableCase, Relation
from vast_v3.data.preprocessing import Preprocessor
from vast_v3.encoders.frontends import NativeFrontend
from vast_v3.models.ranker import VASTModel
from vast_v3.training.objectives import PreparedView, TrainingObjectives


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def fixture_case():
    times = tuple(float(i) for i in range(6))
    return ObservableCase(
        "review", (Entity("a", "service"), Entity("b", "service")), 0., 3., 6.,
        (NumericSeries("metric", "a", "cpu", times, (0., 1., 0., 3., 2., 1.)),
         NumericSeries("metric", "b", "cpu", times, (1., 0., 1., 2., 3., 1.)),
         NumericSeries("trace", "ab", "latency", times, (0., 1., 0., 4., 3., 1.))),
        tuple(LogRecord("a", float(t), f"failed retry {t}", t + 1) for t in range(6)),
        (Relation("ab", "a", "b", "call", 0., 6.),))


def front(case, text_encoder=None):
    pre = Preprocessor.fit([case], permitted_case_ids={case.case_id})
    channels = tuple(sorted({s.channel for s in case.numeric}))
    return NativeFrontend(pre, ("service",), ("call",), channels,
                          text_encoder=text_encoder or (lambda ts: torch.ones(len(ts), 384)))


def prepared(case, frontend, blocks):
    masked, targets = case.mask_blocks(blocks, delta=frontend.preprocessing.delta)
    view = PreparedView(masked, targets, frozenset(blocks), frozenset(blocks))
    output = SimpleNamespace(front=frontend(masked), scores=torch.zeros(len(case.candidate_ids)),
                             origin=case.reference_start, delta=frontend.preprocessing.delta)
    return view, output


def test_frozen_text_encoder_parameters_survive_frontend_initialization():
    class FrozenText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(1, 384, bias=False)
            with torch.no_grad():
                self.project.weight.fill_(.125)
            self.requires_grad_(False)

        def forward(self, texts):
            return self.project(torch.ones(len(texts), 1))

    text = FrozenText()
    before = text.project.weight.detach().clone()
    front(fixture_case(), text_encoder=text)
    torch.testing.assert_close(text.project.weight, before)


def test_outside_window_numeric_values_cannot_affect_in_window_encoding():
    case = fixture_case()
    frontend = front(case)
    series = case.numeric[0]
    extended = replace(series, times=series.times + (6., 7.), values=series.values + (0., 0.),
                       observed=series.observed + (True, True))
    changed = replace(extended, values=series.values + (1000., -1000.))
    a = replace(case, numeric=(extended,) + case.numeric[1:])
    b = replace(case, numeric=(changed,) + case.numeric[1:])
    torch.testing.assert_close(frontend(a).h, frontend(b).h)


def test_context_only_channel_is_not_pooled_as_observed_channel():
    case = fixture_case()
    partial = NumericSeries("metric", "a", "memory", tuple(float(i) for i in range(6)),
                            (0., 1., 2., 0., 1., 2.), (True, True, True, False, True, True))
    case = replace(case, numeric=case.numeric + (partial,))
    frontend = front(case)
    changed = replace(partial, values=(0., 1., 999., 0., 1., 2.))
    other = replace(case, numeric=case.numeric[:-1] + (changed,))
    # At bin 3 CPU is observed, memory is not. Its query-context representation
    # is reserved for recovery and must not join the observed channel pool.
    torch.testing.assert_close(frontend(case).h[0, 3], frontend(other).h[0, 3])


def test_log_frequency_targets_exist_for_withheld_log_blocks():
    case = fixture_case()
    frontend = front(case)
    objectives = TrainingObjectives(VASTModel(frontend), text_reference_mean=torch.zeros(384))
    view, output = prepared(case, frontend, {("node", "a", 3)})
    records = objectives._targets(case, view, output)
    assert "log_frequency" in {record["modality"] for record in records}


def test_a1_local_recovery_reaches_log_frontend_when_text_targets_are_enabled():
    case = fixture_case()
    frontend = front(case)
    objectives = TrainingObjectives(VASTModel(frontend), text_reference_mean=torch.zeros(384))
    view, output = prepared(case, frontend, {("node", "a", 3)})
    records = objectives._targets(case, view, output)
    loss = objectives._recovery(output, records, process=False)
    assert loss is not None
    loss.backward()
    gradient = frontend.log_mlp[0].weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_a1_owner_local_trace_recovery_reaches_trace_cnn():
    case = fixture_case()
    frontend = front(case)
    objectives = TrainingObjectives(VASTModel(frontend))
    view, output = prepared(case, frontend, {("edge", "ab", 3)})
    records = objectives._targets(case, view, output)
    loss = objectives._recovery(output, records, process=False)
    assert loss is not None
    loss.backward()
    assert frontend.trace_cnn[0].weight.grad.abs().sum() > 0


def test_u_objective_reaches_ranker_driver_and_frontend_together():
    case = fixture_case()
    frontend = front(case)
    model = VASTModel(frontend)
    objectives = TrainingObjectives(model, max_roots=2)
    loss = objectives.case_losses(case, epoch=0, components=["U_soft"])["U_soft"]
    assert loss is not None
    loss.backward()
    assert model.rank[-1].weight.grad.abs().sum() > 0
    assert objectives.decoder.driver[0].weight.grad.abs().sum() > 0
    assert frontend.metric_cnn[0].weight.grad.abs().sum() > 0


def test_failed_prior_check_still_charges_accessed_root_labels():
    case = fixture_case()
    labels = LabelStore({case.case_id: ("a", "b")}, budget=1, max_roots=1)
    with pytest.raises(ValueError, match="prior"):
        labels.query(case)
    # The complete answer was already accessed before discovering a protocol
    # contradiction. It cannot become a free abandoned label-access trial.
    assert labels.spent == 1
