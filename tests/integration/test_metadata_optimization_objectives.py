"""Same-mode B30 metadata optimization acceptance, never scalar/hybrid parity.

CPU is exact throughout. CUDA retains atol=2e-5/rtol=2e-4 for floating
computation only; metadata, memberships and checkpoint replay remain exact.
The short synthetic Trainer schedule is engineering acceptance, not B30/170
epoch completion or a benchmark result.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Only mode-independent helpers are reused. In particular, NEVER use the old
# scalar _build, checkpoint factory, or its reviewed first-Adam exception.
from test_metric_scalar_trace_bucket_objectives import (
    ATOL, RTOL, HYBRID, _accumulate, _backend_policy, _cases_and_jobs,
    _compare, _compare_outputs, _draw_rng, _gradients, _outputs, _restore_rng,
    _resume_step_exact, _rng_state, _shared_probes,
)
from vast_v3.data.schema import LabelStore
from vast_v3.runtime import RuntimeConfig, build_objectives
from vast_v3.training.checkpoint import load_checkpoint, save_checkpoint
from vast_v3.training.loop import Trainer
from vast_v3.training import objectives as objective_module


ORACLES = {
    "frontends": ("frontends_b30_policy_r1.py", "NativeFrontend",
                  "f6ae94b5f0b7081e0b0ebd04179ecb9519df43162c835e1d43f4dd0e01bf7bd2"),
    "scope": ("scope_b30_policy_r1.py", "ScopeEncoder",
              "266159502d8d43a5ddceca855609aa4f01324fed6e4c98c7520356ab22b960ae"),
}
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable locally; CPU does not establish CUDA equivalence"))]


@pytest.fixture(autouse=True)
def _isolated_execution():
    threads, rng = torch.get_num_threads(), _rng_state()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        _restore_rng(rng)


@contextmanager
def _fp32_policy():
    before = _backend_policy()
    torch.backends.cudnn.allow_tf32 = False
    expected = {**before, "cudnn_tf32": False}
    try:
        assert _backend_policy() == expected
        yield
        assert _backend_policy() == expected
    finally:
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        assert _backend_policy() == before


def _frozen_class(component):
    filename, class_name, expected_sha = ORACLES[component]
    path = Path(__file__).parents[1] / "oracles" / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    name = "vast_v3.encoders._metadata_objective_frozen_" + component
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[name], class_name)


def _build(config, pre, text, device, *, baseline=False):
    assert config.numeric_execution == HYBRID
    result = build_objectives(config, pre, text_encoder=text, device=device)
    assert all(p.dtype == torch.float32 for p in result.parameters())
    if baseline:
        current_front, current_scope = result.model.frontend, result.model.scope
        rng = _rng_state()
        try:
            front = _frozen_class("frontends")(
                pre, current_front.entity_categories, current_front.relation_categories,
                current_front.channels, text_encoder=text,
                log_recovery_radius_steps=config.log_recovery_radius_steps,
                numeric_execution=config.numeric_execution,
                numeric_batch_size=config.numeric_batch_size,
                numeric_max_batch_samples=config.numeric_max_batch_samples,
            ).to(device)
            scope = _frozen_class("scope")(
                width=current_scope.width, entity_types=current_scope.entity_types,
                relation_types=current_scope.relation_types, max_edges=current_scope.max_edges,
                q=current_scope.q, p=current_scope.p,
            ).to(device)
            front.load_state_dict(current_front.state_dict(), strict=True)
            scope.load_state_dict(current_scope.state_dict(), strict=True)
        finally:
            _restore_rng(rng)
        _compare(_rng_state(), rng, "frozen replacement preserves every RNG", exact=True)
        _compare(front.state_dict(), current_front.state_dict(), "frontend initialization", exact=True)
        _compare(scope.state_dict(), current_scope.state_dict(), "scope initialization", exact=True)
        assert type(front) is not type(current_front)
        assert type(scope) is not type(current_scope)
        result.model.frontend, result.model.scope = front, scope
    return result


def _audit_value(value):
    """Preserve missing-target coordinates separately from finite target values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().clone()
        if value.is_floating_point() and torch.isnan(value).any():
            assert not torch.isinf(value).any()
            return {"missing": torch.isnan(value), "values": torch.nan_to_num(value)}
        return value
    if isinstance(value, dict):
        return {key: _audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_audit_value(item) for item in value)
    return copy.deepcopy(value)


@contextmanager
def _objective_audit(objectives):
    """Observe real loss inputs; neither compute nor substitute any objective."""
    recovery, libraries = [], []
    real_recovery = objective_module.hierarchical_recovery_loss
    real_source = objectives._source_errors

    def observe_recovery(records):
        recovery.append(_audit_value(records))
        return real_recovery(records)

    def observe_source(output, records, library):
        libraries.append(copy.deepcopy(library))
        return real_source(output, records, library)

    with patch.object(objective_module, "hierarchical_recovery_loss", observe_recovery), \
            patch.object(objectives, "_source_errors", observe_source):
        yield {"recovery_inputs": recovery, "source_libraries": libraries}


def _compare_audits(actual, expected, label, *, exact):
    assert actual["recovery_inputs"] and actual["source_libraries"]
    _compare(actual["source_libraries"], expected["source_libraries"], label + " libraries", exact=True)
    # Fixed targets and eligibility are metadata, even when targets are floats.
    maximum = 0.
    assert len(actual["recovery_inputs"]) == len(expected["recovery_inputs"])
    for i, (left_records, right_records) in enumerate(zip(
            actual["recovery_inputs"], expected["recovery_inputs"])):
        assert len(left_records) == len(right_records)
        for j, (left, right) in enumerate(zip(left_records, right_records)):
            assert left.keys() == right.keys()
            for key in right:
                maximum = max(maximum, _compare(left[key], right[key], f"{label}[{i}][{j}].{key}",
                                               exact=exact or key != "prediction"))
    return maximum


def _reload_exact(directory, objectives, optimizer, config, pre, text, device, probes, *, baseline):
    objectives.eval()
    identity = dict(data_hash="metadata-two-synthetic-cases", preprocessing_hash=pre.identity,
                    config_hash=config.identity, code_hash="metadata-same-mode-regression")
    saved_rng = _rng_state()
    path = save_checkpoint(
        directory / "same-mode.pt", model=objectives, optimizer=optimizer, identity=identity,
        progress=dict(stage="metadata-regression", epoch=2, slot=0, encoder_version=1,
                      stage_complete=True), query_sets={"L": [], "U": [c.case_id for c in probes[:2]]})
    _compare(_rng_state(), saved_rng, "checkpoint save preserves RNG", exact=True)
    expected_draws = _draw_rng()
    expected_outputs = _outputs(objectives, probes, seed=expected_draws["seed"])
    restored = _build(config, pre, text, device, baseline=baseline)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
    random.seed(config.seed + 1001)
    np.random.seed(config.seed + 1001)
    torch.manual_seed(config.seed + 1001)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 1001)
    _draw_rng()
    disturbed = _rng_state()
    assert disturbed["python"] != saved_rng["python"]
    assert not np.array_equal(disturbed["numpy"][1], saved_rng["numpy"][1])
    assert not torch.equal(disturbed["cpu"], saved_rng["cpu"])
    assert all(not torch.equal(a, b) for a, b in zip(disturbed["cuda"], saved_rng["cuda"]))
    load_checkpoint(path, model=restored, optimizer=restored_optimizer, expected_identity=identity,
                    require_complete=True, restore_rng=True, map_location="cpu")
    _compare(_rng_state(), saved_rng, "checkpoint all RNG streams", exact=True)
    _compare(restored.state_dict(), objectives.state_dict(), "checkpoint full model", exact=True)
    _compare(restored_optimizer.state_dict(), optimizer.state_dict(), "checkpoint full Adam", exact=True)
    actual_draws = _draw_rng()
    _compare(actual_draws, expected_draws, "checkpoint future RNG draws", exact=True)
    restored.eval()
    _compare_outputs(_outputs(restored, probes, seed=actual_draws["seed"]), expected_outputs,
                     "checkpoint exact full forward", exact=True)
    return restored, restored_optimizer


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("branch", ["PN-source", "U", "mixed"])
def test_same_mode_complete_objectives_adam_and_checkpoint(tmp_path, device, branch):
    config = RuntimeConfig(dataset="rcabench", numeric_execution=HYBRID)
    cases, jobs, pre, text = _cases_and_jobs(branch)
    exact = device == "cpu"
    with _fp32_policy():
        baseline = _build(config, pre, text, device, baseline=True)
        baseline_rng = _rng_state()
        candidate = _build(config, pre, text, device)
        _compare(_rng_state(), baseline_rng, "same initialization RNG", exact=True)
        _compare(candidate.state_dict(), baseline.state_dict(), "same initialized state", exact=True)
        probes = _shared_probes(baseline, candidate, cases, jobs)
        maxima = {"initial_full_output": _compare_outputs(
            _outputs(candidate, probes), _outputs(baseline, probes), "initial all output", exact=exact)}
        optimizers = [torch.optim.Adam(obj.parameters(), lr=.001) for obj in (baseline, candidate)]
        values, audits = [], []
        for obj in (baseline, candidate):
            with _objective_audit(obj) as audit:
                values.append(_accumulate(obj, jobs))
            audits.append(audit)
        maxima["losses"] = _compare(values[1][0], values[0][0], "all losses and None modes", exact=exact)
        _compare(values[1][1], values[0][1], "U memberships", exact=True)
        maxima["recovery_inputs"] = _compare_audits(audits[1], audits[0], "recovery", exact=exact)
        maxima["raw_gradients"] = _compare(_gradients(candidate), _gradients(baseline),
                                           "all parameter gradients and None modes", exact=exact)
        norms = [torch.nn.utils.clip_grad_norm_(obj.parameters(), 5., error_if_nonfinite=True)
                 for obj in (baseline, candidate)]
        maxima["clip_norm"] = _compare(norms[1], norms[0], "global norm", exact=exact)
        maxima["clipped_gradients"] = _compare(_gradients(candidate), _gradients(baseline),
                                               "all clipped gradients", exact=exact)
        for opt in optimizers:
            opt.step()
        maxima["post_adam_state"] = _compare(candidate.state_dict(), baseline.state_dict(),
                                             "every post-Adam tensor, no exceptions", exact=exact)
        maxima["adam_state"] = _compare(optimizers[1].state_dict(), optimizers[0].state_dict(),
                                         "every Adam state", exact=exact)
        baseline.eval()
        candidate.eval()
        maxima["post_adam_output"] = _compare_outputs(_outputs(candidate, probes), _outputs(baseline, probes),
                                                       "post-Adam full output", exact=exact)
        loaded = [_reload_exact(tmp_path / name, obj, opt, config, pre, text, device, probes,
                                baseline=is_baseline)
                  for name, obj, opt, is_baseline in (
                      ("baseline", baseline, optimizers[0], True),
                      ("candidate", candidate, optimizers[1], False))]
        maxima["reloaded_output"] = _compare_outputs(_outputs(loaded[1][0], probes), _outputs(loaded[0][0], probes),
                                                      "both reloaded full output", exact=exact)
        # Exact future backward/optimizer replay is a CPU claim only.
        if device == "cpu":
            for original, opt, (restored, restored_opt) in zip((baseline, candidate), optimizers, loaded):
                _resume_step_exact(original, restored, opt, restored_opt, jobs, probes)
            _compare(loaded[1][0].state_dict(), loaded[0][0].state_dict(), "both resumed model", exact=True)
            _compare(loaded[1][1].state_dict(), loaded[0][1].state_dict(), "both resumed Adam", exact=True)
            _compare_outputs(_outputs(loaded[1][0], probes), _outputs(loaded[0][0], probes),
                             "both resumed output", exact=True)
        print({"device": device, "branch": branch, "mode": HYBRID, "cases": len(cases),
               "backwards_per_step": len(jobs), "cpu_exact": exact,
               "cuda_atol": ATOL, "cuda_rtol": RTOL, "measured_max_abs": maxima})


@pytest.mark.parametrize("corruption", ["mask", "structure_code", "gradient"])
def test_comparators_reject_deliberately_corrupted_model_results(corruption):
    config = RuntimeConfig(dataset="rcabench", numeric_execution=HYBRID)
    cases, jobs, pre, text = _cases_and_jobs("PN-source")
    with _fp32_policy():
        obj = _build(config, pre, text, "cpu", baseline=True)
        reference = _outputs(obj, cases[:1])
        _compare_outputs(_outputs(obj, cases[:1]), reference, "same implementation control", exact=True)
        if corruption == "mask":
            mask = reference[0].front.telemetry_mask.clone()
            mask[0, 0] = ~mask[0, 0]
            changed = (replace(reference[0], front=replace(reference[0].front, telemetry_mask=mask)),)
            with pytest.raises(AssertionError, match="telemetry_mask"):
                _compare_outputs(changed, reference, "injected mask", exact=True)
        elif corruption == "structure_code":
            canonical = obj.model.scope._canonical

            def changed_code(*args, **kwargs):
                code, mappings = canonical(*args, **kwargs)
                return (1 - code[0], *code[1:]), mappings

            with patch.object(obj.model.scope, "_canonical", changed_code):
                changed = _outputs(obj, cases[:1])
            with pytest.raises(AssertionError):
                _compare_outputs(changed, reference, "injected canonical code", exact=True)
        else:
            _accumulate(obj, jobs[:1])
            expected = _gradients(obj)
            actual = _gradients(obj)
            key = next(name for name, value in actual.items() if value is not None)
            actual[key].flatten()[0] += .125
            with pytest.raises(AssertionError, match="injected gradient"):
                _compare(actual, expected, "injected gradient", exact=True)


class _RecordingTrainer(Trainer):
    """Observational test subclass; all scheduling, selection and writes are real."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eligibility_audit = []
        self.save_audit = []

    def _eligible(self, epoch, weights):
        populations = super()._eligible(epoch, weights)
        self.eligibility_audit.append({
            "epoch": epoch, "weights": dict(weights), "populations": copy.deepcopy(populations),
            "libraries": copy.deepcopy(self.objectives._libraries),
            "labeled": self.labels.queried,
        })
        return populations

    def _save(self, stages, *, complete=False):
        super()._save(stages, complete=complete)
        self.save_audit.append({
            "loop": copy.deepcopy(self.state), "complete": complete,
            "library_epoch": self.objectives._library_epoch,
            "libraries": copy.deepcopy(self.objectives._libraries),
            "labels": self.labels.state_dict(),
            "query_trace": self.query_trace.state_dict() if self.query_trace else None,
            "rng": _rng_state(),
        })


def _short_trainer(directory, *, baseline):
    cases, _, pre, text = _cases_and_jobs("mixed")
    # Deliberately tiny declared engineering schedule. No production config edit.
    config = RuntimeConfig(dataset="rcabench", numeric_execution=HYBRID, budget=1,
                           query_batch_size=1, warmup_epochs=1, nonfinal_epochs=1,
                           final_epochs=2, u_disabled_epochs=0, u_ramp_epochs=0,
                           case_batch_size=1)
    obj = _build(config, pre, text, "cpu", baseline=baseline)
    labels = LabelStore({case.case_id: ("a",) for case in cases}, budget=1, max_roots=2)
    return _RecordingTrainer(
        obj, {case.case_id: case for case in cases}, labels, config.schedule(),
        checkpoint_dir=directory,
        identity=dict(data_hash="metadata-short-synthetic-pool", config_hash=config.identity,
                      preprocessing_hash=pre.identity, code_hash="metadata-short-trainer-test")), cases


def _assert_epoch_and_stream_coverage(trainer):
    assert len(trainer.eligibility_audit) == 3  # One warmup and two joint epochs.
    all_ids = set(trainer.cases)
    for item in trainer.eligibility_audit:
        populations, labeled = item["populations"], set(item["labeled"])
        for name in ("local_recovery", "process_recovery", "invariance"):
            assert set(populations[name]) == all_ids
        if item["epoch"] == 0:
            assert not labeled and not item["libraries"]
            assert all(not populations[name] for name in ("pn", "source", "U_soft"))
        else:
            assert len(labeled) == 1
            assert set(populations["pn"]) == set(populations["source"]) == labeled
            assert set(populations["U_soft"]) == all_ids - labeled
            assert {cid for cid, epoch in item["libraries"]} == all_ids - labeled
            assert {epoch for cid, epoch in item["libraries"]} == {item["epoch"]}
            frozen = item["libraries"]
            # Memberships computed by the epoch-start eligibility pass must
            # survive both optimizer slots without being rebuilt from new scores.
            relevant = [snap for snap in trainer.save_audit if
                        snap["library_epoch"] == item["epoch"] and snap["loop"]["slot"] > 0]
            assert len(relevant) == 2
            for snap in relevant:
                _compare(snap["libraries"], frozen, "frozen epoch-start U libraries", exact=True)
    update_snapshots = [snap for snap in trainer.save_audit if snap["loop"]["stream"] is not None]
    assert len(update_snapshots) == 6
    for snap in update_snapshots:
        stream = snap["loop"]["stream"]
        assert set(stream["streams"]) == set(trainer.objectives.names)
        # Per-head RNGs are separate and fully checkpointed, including empty heads.
        states = [repr(value["rng_state"]) for value in stream["streams"].values()]
        assert len(set(states)) == len(states)
        for name, value in stream["streams"].items():
            assert sorted(value["deck"]) == sorted(stream["populations"][name])
            assert 0 <= value["position"] <= len(value["deck"])


def _compare_trainers(actual, expected, cases, label):
    _compare(actual.state, expected.state, label + " complete loop", exact=True)
    _compare(actual.objectives.state_dict(), expected.objectives.state_dict(), label + " model", exact=True)
    _compare(actual.optimizer.state_dict(), expected.optimizer.state_dict(), label + " Adam", exact=True)
    _compare(actual.objectives._libraries, expected.objectives._libraries, label + " libraries", exact=True)
    _compare(actual.labels.state_dict(), expected.labels.state_dict(), label + " label ledger", exact=True)
    _compare(actual.query_trace.state_dict(), expected.query_trace.state_dict(), label + " query trace", exact=True)
    for filename in ("query-access.json", "query-trace.json"):
        _compare(json.loads((actual.directory / filename).read_text(encoding="utf-8")),
                 json.loads((expected.directory / filename).read_text(encoding="utf-8")),
                 label + " durable " + filename, exact=True)
    _compare_outputs(_outputs(actual.objectives, cases), _outputs(expected.objectives, cases),
                     label + " final full outputs", exact=True)


def test_short_cpu_trainer_eligibility_streams_query_ledger_and_interrupted_resume(tmp_path):
    with _fp32_policy():
        complete, results = [], []
        for name, baseline in (("baseline", True), ("candidate", False)):
            full, cases = _short_trainer(tmp_path / (name + "-full"), baseline=baseline)
            result = full.run()
            assert result["complete"] and result["updates"] == 6
            assert full.labels.spent == 1 and len(result["query_order"]) == 1
            _assert_epoch_and_stream_coverage(full)
            partial, _ = _short_trainer(tmp_path / (name + "-resume"), baseline=baseline)
            paused = partial.run(max_updates=3)
            assert not paused["complete"] and partial.state["slot"] == 1
            assert partial.objectives._libraries and partial.state["stream"] is not None
            checkpoint = torch.load(partial.directory / "latest.pt", map_location="cpu", weights_only=True)
            assert checkpoint["progress"]["stage_complete"] is False
            assert checkpoint["extra_state"]["libraries"]
            _compare(checkpoint["extra_state"]["loop"]["stream"], partial.state["stream"],
                     "paused independent streams", exact=True)
            resumed, _ = _short_trainer(partial.directory, baseline=baseline)
            # Make every global RNG observably different before Trainer restores it.
            random.seed(1907)
            np.random.seed(1907)
            torch.manual_seed(1907)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1907)
            actual = resumed.run(resume=True)
            _compare(actual, result, name + " resumed reports", exact=True)
            _compare_trainers(resumed, full, cases, name + " resumed")
            _compare(partial.save_audit + resumed.save_audit, full.save_audit,
                     name + " every checkpoint transition", exact=True)
            _compare(partial.eligibility_audit + resumed.eligibility_audit, full.eligibility_audit,
                     name + " eligibility setup is not repeated mid-epoch", exact=True)
            complete.append(full)
            results.append(result)
        _compare(results[1], results[0], "same-mode short Trainer reports", exact=True)
        _compare_trainers(complete[1], complete[0], cases, "same-mode short Trainer")
        _compare(complete[1].eligibility_audit, complete[0].eligibility_audit,
                 "same-mode all eligibility populations", exact=True)
        _compare(complete[1].save_audit, complete[0].save_audit,
                 "same-mode all checkpoint transitions and streams", exact=True)
