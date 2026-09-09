from __future__ import annotations

import argparse

import hashlib

import json

import math

import os

from pathlib import Path

import subprocess

import sys

import time

import traceback

from typing import Any, Mapping, Sequence

def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()

def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"not JSON serializable: {value!r}")

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL object required: {path}")
                rows.append(payload)
    return rows

def _derive_metrics_without_package(rankings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_case = []
    seen = set()
    for row in rankings:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("ranking case ownership drifted")
        seen.add(case_id)
        targets = {str(value) for value in row.get("targets", ())}
        candidates = [
            (
                str(value.get("entity_id", "")).strip()
                if isinstance(value, Mapping)
                else str(value).strip()
            )
            for value in row.get("ranking", ())
        ]
        if not targets or not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("ranking target/candidate coverage drifted")
        ranks = [index for index, candidate in enumerate(candidates, start=1) if candidate in targets]
        if not ranks:
            raise ValueError("ranking has no true root-cause candidate")
        rank = min(ranks)
        per_case.append(
            {
                "case_id": case_id,
                "first_valid_root_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "candidate_count": len(candidates),
            }
        )
    denominator = len(per_case)
    hit_counts = {
        "hit_at_1": sum(row["first_valid_root_rank"] <= 1 for row in per_case),
        "hit_at_3": sum(row["first_valid_root_rank"] <= 3 for row in per_case),
        "hit_at_5": sum(row["first_valid_root_rank"] <= 5 for row in per_case),
    }
    rates = {key: value / denominator for key, value in hit_counts.items()}
    rr_sum = math.fsum(row["reciprocal_rank"] for row in per_case)
    identity = {
        "schema_version": "historical-pairwise-rank-metrics-v1",
        "denominator": denominator,
        "hit_counts": hit_counts,
        **rates,
        "top135": math.fsum(rates.values()) / 3.0,
        "reciprocal_rank_sum": rr_sum,
        "mrr": rr_sum / denominator,
        "per_case": per_case,
    }
    return {**identity, "metrics_sha256": _semantic_sha256(identity)}
