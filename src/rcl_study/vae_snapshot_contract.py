from __future__ import annotations

from copy import deepcopy

import hashlib

import json

import math

from pathlib import Path

import re

from typing import Any, Mapping



class SnapshotContractError(ValueError):
    """An input would change the frozen scientific scope or artifact ownership."""

def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, allow_nan=False, ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotContractError(f"cannot read contract input: {path}") from exc
    if not isinstance(value, dict):
        raise SnapshotContractError(f"JSON object required: {path}")
    return value

OUTPUT_NAMESPACE = "outputs/vast"
