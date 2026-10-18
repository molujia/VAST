import importlib
import random

import numpy as np
import pytest
import torch


def api():
    try:
        return importlib.import_module("vast_v3.training.checkpoint")
    except ModuleNotFoundError:
        pytest.fail("Atomic checkpoint implementation is missing")


IDENTITY = {"data_hash": "data-v1", "config_hash": "config-v1",
            "preprocessing_hash": "prep-v1", "code_hash": "code-v1"}


def state():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model(torch.ones(1, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, optimizer


def save(path, model, optimizer, complete=False, extra_state=None):
    return api().save_checkpoint(path, model=model, optimizer=optimizer, identity=IDENTITY,
                                 progress={"stage": "round-2", "epoch": 8, "slot": 3,
                                           "encoder_version": 6, "stage_complete": complete,
                                           "round_index": 2},
                                 query_sets={"L": ["case1"], "U": ["case2", "case3"]},
                                 extra_state=extra_state)


def random_values():
    return random.random(), np.random.random(4), torch.rand(4)


def test_checkpoint_restores_model_optimizer_rng_query_sets_and_progress(tmp_path):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    random.seed(19)
    np.random.seed(20)
    torch.manual_seed(21)
    save(path, model, optimizer, extra_state={"sampler": {"slot": 9}})
    expected = random_values()
    saved_weight = model.weight.detach().clone()
    expected_step = optimizer.state[model.weight]["step"].item()
    with torch.no_grad():
        model.weight.fill_(42)
    optimizer.param_groups[0]["lr"] = 9.0
    optimizer.state[model.weight]["step"].fill_(99)
    restored = api().load_checkpoint(path, model=model, optimizer=optimizer, expected_identity=IDENTITY)
    torch.testing.assert_close(model.weight, saved_weight)
    assert optimizer.param_groups[0]["lr"] == .001
    assert optimizer.state[model.weight]["step"].item() == expected_step
    actual = random_values()
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    torch.testing.assert_close(expected[2], actual[2])
    assert restored["query_sets"] == {"L": ["case1"], "U": ["case2", "case3"]}
    assert restored["progress"]["epoch"] == 8 and restored["progress"]["slot"] == 3
    assert restored["progress"]["encoder_version"] == 6
    assert restored["extra_state"]["sampler"]["slot"] == 9


@pytest.mark.parametrize("key", list(IDENTITY))
def test_identity_mismatch_fails_before_model_or_rng_mutation(tmp_path, key):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer)
    with torch.no_grad():
        model.weight.fill_(10)
    before = model.weight.detach().clone()
    rng_before = torch.get_rng_state().clone()
    wrong = dict(IDENTITY, **{key: "changed"})
    with pytest.raises(ValueError, match="identity|hash"):
        api().load_checkpoint(path, model=model, optimizer=optimizer, expected_identity=wrong)
    torch.testing.assert_close(model.weight, before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)


def test_partial_checkpoint_cannot_be_mistaken_for_stage_completion(tmp_path):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer)
    assert api().inspect_checkpoint(path, expected_identity=IDENTITY)["progress"]["stage_complete"] is False
    with pytest.raises(ValueError, match="complete|partial"):
        api().load_checkpoint(path, model=model, optimizer=optimizer,
                              expected_identity=IDENTITY, require_complete=True)
    save(path, model, optimizer, complete=True)
    assert api().load_checkpoint(path, model=model, optimizer=optimizer,
                                 expected_identity=IDENTITY, require_complete=True)["progress"]["stage_complete"]


def test_atomic_replace_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer)
    before = path.read_bytes()
    def fail_replace(*args):
        raise OSError("simulated replacement failure")
    monkeypatch.setattr(api().os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save(path, model, optimizer, complete=True)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_save_failure_cleans_temporary_file_and_preserves_previous(tmp_path, monkeypatch):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer)
    before = path.read_bytes()
    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")
    monkeypatch.setattr(api().torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        save(path, model, optimizer)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_rejects_overlapping_query_sets_and_missing_identity(tmp_path):
    model, optimizer = state()
    kwargs = dict(model=model, optimizer=optimizer, identity=IDENTITY,
                  progress={"stage": "warmup", "epoch": 0, "slot": 0,
                            "encoder_version": 0, "stage_complete": False})
    with pytest.raises(ValueError, match="disjoint|overlap"):
        api().save_checkpoint(tmp_path / "bad.pt", **kwargs, query_sets={"L": ["a"], "U": ["a"]})
    kwargs["identity"] = {"data_hash": "data-v1"}
    with pytest.raises(ValueError, match="identity|hash"):
        api().save_checkpoint(tmp_path / "bad.pt", **kwargs, query_sets={"L": [], "U": []})
    assert not (tmp_path / "bad.pt").exists()


def test_checkpoint_contains_only_weights_only_loadable_types(tmp_path):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer)
    payload = torch.load(path, weights_only=True)
    assert "python" in payload["rng"] and "numpy" in payload["rng"]
    assert "torch_cpu" in payload["rng"] and "torch_cuda" in payload["rng"]


def test_inspection_does_not_restore_rng(tmp_path):
    model, optimizer = state()
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    api().inspect_checkpoint(path, expected_identity=IDENTITY)
    torch.testing.assert_close(torch.get_rng_state(), before)
