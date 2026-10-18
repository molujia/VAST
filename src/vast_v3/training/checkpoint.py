"""Atomic, identity-bound training checkpoints with explicit completion state.

Only locally produced, trusted checkpoints belong in the training workflow.
Loads additionally use PyTorch's weights-only deserializer: no NumPy objects or
custom Python classes need to be unpickled. All four identity hashes must match
before any model, optimizer or global RNG state is restored.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


SCHEMA = "vast-v3-training-checkpoint/1"
IDENTITY_FIELDS = ("data_hash", "config_hash", "preprocessing_hash", "code_hash")


def _identity(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not set(IDENTITY_FIELDS).issubset(value):
        raise ValueError("Checkpoint identity requires data/config/preprocessing/code hashes")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError("Checkpoint identity hashes cannot be missing or empty")
    return dict(value)


def _progress(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("stage"), str) or not value["stage"]:
        raise ValueError("Progress requires a nonempty stage")
    for key in ("epoch", "slot", "encoder_version"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"Progress {key} must be a nonnegative integer")
    if not isinstance(value.get("stage_complete"), bool):
        raise ValueError("Progress requires an explicit boolean stage_complete marker")
    if "round_index" in value and (isinstance(value["round_index"], bool)
                                  or not isinstance(value["round_index"], int)
                                  or value["round_index"] < 0):
        raise ValueError("round_index must be a nonnegative integer when present")
    return dict(value)


def _queries(value: Mapping[str, Any]) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or set(value) != {"L", "U"}:
        raise ValueError("query_sets must contain L and U populations")
    result = {}
    for name in ("L", "U"):
        result[name] = list(value[name])
        if any(not isinstance(case_id, str) or not case_id for case_id in result[name]):
            raise ValueError("Queried/unqueried case IDs must be nonempty strings")
        if len(result[name]) != len(set(result[name])):
            raise ValueError("Queried/unqueried sets cannot contain duplicate cases")
    if set(result["L"]) & set(result["U"]):
        raise ValueError("L and U query sets must be disjoint, without overlap")
    return result


def _portable(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, float, torch.Tensor)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _portable(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise ValueError("Checkpoint metadata keys must be strings or integers")
            _portable(item)
        return
    raise ValueError("Checkpoint metadata must use tensors and plain scalar/list/tuple/dict types")


def _capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _numpy_state(value):
    return (value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4])


def _validate_rng(state: Mapping, *, check_cuda: bool) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Checkpoint RNG state is incomplete")
    random.Random(0).setstate(state["python"])
    np.random.RandomState(0).set_state(_numpy_state(state["numpy"]))
    torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
    cuda = state["torch_cuda"]
    if not isinstance(cuda, list) or any(not isinstance(s, torch.Tensor) or s.dtype != torch.uint8
                                         or s.ndim != 1 for s in cuda):
        raise ValueError("Invalid CUDA RNG state")
    if check_cuda:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(cuda) != count:
            raise ValueError("CUDA device count differs; exact RNG resume is not possible")


def _restore_rng(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(_numpy_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_checkpoint(path: str | Path, *, model: nn.Module,
                    optimizer: torch.optim.Optimizer, identity: Mapping[str, str],
                    progress: Mapping[str, Any], query_sets: Mapping[str, Any],
                    extra_state: Mapping[str, Any] | None = None) -> Path:
    """Flush a sibling temporary file, then atomically replace the destination.

    ``epoch``/``slot`` describe the next update to perform, chosen by the caller.
    ``stage_complete`` may only be true after all scheduled work for that stage
    succeeded. A saved file alone never marks a stage complete. Extra state can
    hold eligible-stream cursors, masks, proposal membership (plain lists), and
    stage-specific progress; do not save stale trainable activations there.
    """
    payload = {"schema": SCHEMA, "identity": _identity(identity),
               "progress": _progress(progress), "query_sets": _queries(query_sets),
               "extra_state": dict(extra_state or {}), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
               "rng": _capture_rng()}
    _portable(payload["extra_state"])
    _portable(payload["progress"])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                     dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def inspect_checkpoint(path: str | Path, *, expected_identity: Mapping[str, str],
                       require_complete: bool = False, map_location: Any = "cpu") -> dict:
    """Read and validate metadata without restoring model or global RNG state."""
    expected = _identity(expected_identity)
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    actual = _identity(payload.get("identity", {}))
    if actual != expected:
        differences = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise ValueError(f"Checkpoint identity hash mismatch: {differences}")
    _progress(payload.get("progress", {}))
    _queries(payload.get("query_sets", {}))
    if require_complete and not payload["progress"]["stage_complete"]:
        raise ValueError("Checkpoint stage is partial, not complete")
    if not all(key in payload for key in ("model", "optimizer", "optimizer_class", "rng", "extra_state")):
        raise ValueError("Checkpoint payload is incomplete")
    _validate_rng(payload["rng"], check_cuda=False)
    return payload


def load_checkpoint(path: str | Path, *, model: nn.Module,
                    optimizer: torch.optim.Optimizer, expected_identity: Mapping[str, str],
                    require_complete: bool = False, restore_rng: bool = True,
                    map_location: Any = "cpu") -> dict:
    """Restore an identity-matched partial or complete checkpoint for training.

    All identity/completion/RNG and basic model/optimizer compatibility checks
    precede mutation. CUDA topology must match for exact training continuation;
    explicit restore_rng=False permits weight/optimizer inspection elsewhere.
    """
    payload = inspect_checkpoint(path, expected_identity=expected_identity,
                                 require_complete=require_complete, map_location=map_location)
    if restore_rng:
        _validate_rng(payload["rng"], check_cuda=True)
    current = model.state_dict()
    saved = payload["model"]
    if set(current) != set(saved) or any(current[key].shape != saved[key].shape
                                       or current[key].dtype != saved[key].dtype for key in current):
        raise ValueError("Checkpoint model state does not match current model")
    optimizer_class = type(optimizer).__module__ + "." + type(optimizer).__qualname__
    if payload["optimizer_class"] != optimizer_class:
        raise ValueError("Checkpoint optimizer class does not match")
    old_groups, new_groups = optimizer.state_dict()["param_groups"], payload["optimizer"]["param_groups"]
    if len(old_groups) != len(new_groups) or any(len(a["params"]) != len(b["params"])
                                              for a, b in zip(old_groups, new_groups)):
        raise ValueError("Checkpoint optimizer parameter groups do not match")
    model.load_state_dict(saved, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return payload
