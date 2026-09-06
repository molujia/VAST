from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


class FinalRCLOSERHandshakeError(ValueError):
    """Raised when the JSON boundary around the Torch OSER worker drifts."""


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _ids(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalRCLOSERHandshakeError(f"{context} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if not result or "" in result or len(result) != len(set(result)):
        raise FinalRCLOSERHandshakeError(f"{context} must be unique and nonempty")
    return result


def validate_final_oser_handshake_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalRCLOSERHandshakeError("OSER request must be a mapping")
    request = deepcopy(dict(value))
    supplied = request.pop("request_sha256", None)
    if supplied != _semantic_hash(request):
        raise FinalRCLOSERHandshakeError("OSER request identity drifted")
    if (
        request.get("schema_version") != "final-rcl-oser-handshake-request-v1"
        or request.get("dataset_id") not in {"rcabench", "aiops2022_pre"}
        or request.get("training_mode") not in {"query_only", "oracle_full"}
        or request.get("seed") != 42
        or request.get("outer_query_population") != "real_cases_only"
    ):
        raise FinalRCLOSERHandshakeError("OSER request must retain real outer queries")
    supervised = _ids(request.get("supervised_case_ids"), "supervised cases")
    if request.get("supervised_case_limit") != len(supervised):
        raise FinalRCLOSERHandshakeError("OSER supervised cardinality drifted")
    labels = request.get("real_label_records")
    synthetic = request.get("synthetic_family_rows")
    training = request.get("training_cases")
    inference = request.get("inference_cases")
    base = request.get("base_score_artifact")
    profile = request.get("profile")
    if (
        isinstance(labels, (str, bytes))
        or not isinstance(labels, Sequence)
        or isinstance(synthetic, (str, bytes))
        or not isinstance(synthetic, Sequence)
        or not isinstance(training, Mapping)
        or not isinstance(inference, Mapping)
        or not inference
        or not isinstance(base, Mapping)
        or base.get("schema_version")
        != "conservative-lofo-base-score-artifact-v1"
        or not isinstance(profile, Mapping)
        or profile.get("profile_id") != "oser-p02"
        or len(str(request.get("state_transform_sha256", ""))) != 64
        or not str(request.get("artifact_role", "")).strip()
    ):
        raise FinalRCLOSERHandshakeError("OSER request closure drifted")
    label_ids = tuple(str(dict(row).get("case_id", "")) for row in labels)
    if label_ids != supervised:
        raise FinalRCLOSERHandshakeError("OSER real label order drifted")
    synthetic_keys = tuple(
        "synthetic:" + str(dict(row).get("synthetic_row_sha256", ""))
        for row in synthetic
    )
    expected_training_keys = (*supervised, *synthetic_keys)
    if len(training) != len(expected_training_keys) or set(training) != set(
        expected_training_keys
    ):
        raise FinalRCLOSERHandshakeError("OSER training family coverage drifted")
    request["training_cases"] = {
        key: deepcopy(training[key]) for key in expected_training_keys
    }
    if set(training) & set(inference):
        raise FinalRCLOSERHandshakeError("OSER training/inference overlap is forbidden")
    return {**request, "request_sha256": supplied}


def execute_final_oser_handshake(value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_final_oser_handshake_request(value)
    from .final_rcl_oser import (
        build_final_oser_family_episodes,
        fit_final_oser,
        score_final_oser,
    )

    episodes = build_final_oser_family_episodes(
        real_label_records=request["real_label_records"],
        supervised_case_ids=request["supervised_case_ids"],
        synthetic_rows=request["synthetic_family_rows"],
        training_mode=request["training_mode"],
        supervised_case_limit=request["supervised_case_limit"],
    )
    fitted = fit_final_oser(
        training_cases=request["training_cases"],
        episode_artifact=episodes,
        profile=request["profile"],
        state_transform_sha256=request["state_transform_sha256"],
        seed=42,
    )
    score_artifact = score_final_oser(
        fitted=fitted,
        inference_cases=request["inference_cases"],
        base_score_artifact=request["base_score_artifact"],
        artifact_role=request["artifact_role"],
    ).to_dict()
    oser_status = "active" if episodes.get("status") == "ready" else "inactive_fallback"
    identity = {
        "schema_version": "final-rcl-oser-handshake-result-v1",
        "status": "completed",
        "request_sha256": request["request_sha256"],
        "oser_status": oser_status,
        "episode_artifact": episodes,
        "training_audit": dict(fitted.audit),
        "score_artifact": score_artifact,
    }
    return {**identity, "result_sha256": _semantic_hash(identity)}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinalRCLOSERHandshakeError(f"cannot read OSER request: {path}") from exc
    if not isinstance(value, Mapping):
        raise FinalRCLOSERHandshakeError("OSER request JSON must be an object")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_final_oser_handshake(_read(args.input.resolve()))
    _write(args.output.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FinalRCLOSERHandshakeError",
    "execute_final_oser_handshake",
    "validate_final_oser_handshake_request",
]
