"""Immutable two-teacher pseudo predictions and matched arm projections."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .datasets import resolve_dataset


RAW_PSEUDO_SCHEMA_VERSION = "rcl-raw-pseudo-pool-v2"
RAW_PSEUDO_MANIFEST_VERSION = "rcl-raw-pseudo-pool-manifest-v1"
ALLOWED_SELECTOR_SEEDS = (42, 43, 44)
ALLOWED_LABEL_MODES = ("hard_top1", "partial_top2")
ALLOWED_COVERAGES = (0.1, 0.25, 0.5)
ALLOWED_NEGATIVE_STRATEGIES = (
    "all_non_positive",
    "teacher_bottom_k",
    "margin_abstain",
)
ALLOWED_PSEUDO_LOSS_RATIOS = (0.1, 0.25, 0.5)

_FORBIDDEN_LABEL_TOKENS = (
    "groundtruth",
    "ground_truth",
    "root_cause",
    "oracle_label",
    "oracle_target",
    "authoritative_label",
    "target_service",
    "target_node",
)


@dataclass(frozen=True)
class RawPseudoPool:
    canonical_dataset_id: str
    pseudo_generator_candidate_id: str
    selector_seed: int
    teacher_seeds: Tuple[int, int]
    query_case_ids: Tuple[str, ...]
    query_plan_sha256: str
    eligible_case_ids: Tuple[str, ...]
    outer_test_case_ids: Tuple[str, ...]
    rows: Tuple[Dict[str, Any], ...]
    data_path: str
    manifest_path: str
    sha256: str


@dataclass(frozen=True)
class MatchedPseudoArm:
    arm: str
    canonical_dataset_id: str
    raw_pool_sha256: str
    raw_pool_path: str
    query_plan_sha256: str
    query_case_ids: Tuple[str, ...]
    label_mode: str
    class_conditional_coverage: float
    pseudo_rows: Tuple[Dict[str, Any], ...]
    rejections: Tuple[Dict[str, Any], ...]
    pseudo_count: int
    pseudo_weight_policy: str
    negative_strategy: str = "all_non_positive"
    pseudo_loss_ratio: float = 0.25
    teacher_bottom_fraction: float = 0.25
    minimum_rank_margin: float = 0.50


@dataclass(frozen=True)
class MatchedPseudoArms:
    query_only: MatchedPseudoArm
    all_pseudo: MatchedPseudoArm
    selected_pseudo: MatchedPseudoArm


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_dataset(identifier: str) -> str:
    resolution = resolve_dataset(identifier)
    if resolution.requested_id != resolution.canonical_id:
        raise ValueError(
            "raw pseudo pool requires a canonical dataset identifier"
        )
    return resolution.canonical_id


def _case_ids(values: Iterable[Any], *, role: str) -> Tuple[str, ...]:
    rows = tuple(str(value).strip() for value in values)
    if any(not value for value in rows):
        raise ValueError("%s case IDs must not be empty" % role)
    if len(rows) != len(set(rows)):
        raise ValueError("%s case IDs must be unique" % role)
    return tuple(sorted(rows))


def _find_label_fields(value: Any, path: str = "") -> Tuple[str, ...]:
    findings = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = "%s.%s" % (path, key_text) if path else key_text
            lowered = key_text.lower()
            if any(token in lowered for token in _FORBIDDEN_LABEL_TOKENS):
                findings.append(child_path)
            findings.extend(_find_label_fields(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(
                _find_label_fields(child, "%s[%d]" % (path, index))
            )
    return tuple(findings)


def _normalized_distribution(
    raw: Any,
) -> Tuple[Dict[str, float], str]:
    if not isinstance(raw, Mapping) or not raw:
        return {}, "missing_teacher_distribution"
    values = {}
    for key, value in raw.items():
        candidate = str(key).strip()
        try:
            score = float(value)
        except (TypeError, ValueError):
            return {}, "nonfinite_teacher_distribution"
        if not candidate or not math.isfinite(score) or score < 0.0:
            return {}, "nonfinite_teacher_distribution"
        values[candidate] = score
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        return {}, "nonpositive_teacher_distribution_mass"
    return {
        candidate: values[candidate] / total
        for candidate in sorted(values)
    }, ""


def _rank_distribution(
    distribution: Mapping[str, float],
) -> Tuple[str, ...]:
    return tuple(
        candidate
        for candidate, _ in sorted(
            distribution.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    )


def _confidence(distribution: Mapping[str, float]) -> float:
    ranked = _rank_distribution(distribution)
    if not ranked:
        return 0.0
    first = float(distribution[ranked[0]])
    second = float(distribution[ranked[1]]) if len(ranked) > 1 else 0.0
    margin = max(0.0, min(1.0, (first - second) / max(first, 1e-12)))
    if len(ranked) <= 1:
        normalized_entropy = 0.0
    else:
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in distribution.values()
        )
        normalized_entropy = max(
            0.0, min(1.0, entropy / math.log(len(ranked)))
        )
    return max(
        0.0,
        min(1.0, 0.5 * margin + 0.5 * (1.0 - normalized_entropy)),
    )


def _normalize_raw_pool_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    findings = _find_label_fields(row)
    if findings:
        raise ValueError("raw pseudo pool contains a forbidden label field")
    normalized = dict(row)
    distributions = normalized.get("teacher_distributions", ())
    if not isinstance(distributions, (list, tuple)) or len(distributions) != 2:
        teacher_rankings = ((), ())
    else:
        teacher_rankings = tuple(
            _rank_distribution(distribution)
            if isinstance(distribution, Mapping) and distribution
            else ()
            for distribution in distributions
        )
    mean_distribution = normalized.get("mean_distribution", {})
    mean_ranking = (
        _rank_distribution(mean_distribution)
        if isinstance(mean_distribution, Mapping) and mean_distribution
        else ()
    )
    if str(normalized.get("validity_status", "")) == "valid":
        candidate_sets = tuple(set(ranking) for ranking in teacher_rankings)
        if (
            not mean_ranking
            or any(not ranking for ranking in teacher_rankings)
            or candidate_sets[0] != candidate_sets[1]
            or candidate_sets[0] != set(mean_ranking)
        ):
            raise ValueError(
                "valid teacher prediction candidate coverage is incomplete"
            )
    existing_teacher = normalized.get("teacher_candidate_rankings")
    if existing_teacher is not None and tuple(
        tuple(str(value) for value in ranking)
        for ranking in existing_teacher
    ) != teacher_rankings:
        raise ValueError("teacher candidate ranking drifted from distributions")
    existing_mean = normalized.get("candidate_ranking")
    if existing_mean is not None and tuple(
        str(value) for value in existing_mean
    ) != mean_ranking:
        raise ValueError("candidate ranking drifted from mean distribution")
    normalized["teacher_candidate_rankings"] = [
        list(ranking) for ranking in teacher_rankings
    ]
    normalized["candidate_ranking"] = list(mean_ranking)
    return normalized


def _build_row(
    raw: Mapping[str, Any],
    *,
    teacher_seeds: Tuple[int, int],
) -> Dict[str, Any]:
    findings = _find_label_fields(raw)
    if findings:
        raise ValueError(
            "teacher prediction contains a forbidden label field: %s"
            % list(findings)
        )
    case_id = str(raw.get("case_id", "")).strip()
    distributions = raw.get("teacher_distributions", ())
    if not isinstance(distributions, (list, tuple)) or len(distributions) != 2:
        normalized = ({}, {})
        errors = ("missing_teacher_distribution",) * 2
    else:
        first, first_error = _normalized_distribution(distributions[0])
        second, second_error = _normalized_distribution(distributions[1])
        normalized = (first, second)
        errors = (first_error, second_error)
    if any(errors):
        validity = next(error for error in errors if error)
        mean_distribution: Dict[str, float] = {}
        teacher_top1 = ("", "")
        ranking: Tuple[str, ...] = ()
        agreement = False
    else:
        candidates = sorted(set(normalized[0]) | set(normalized[1]))
        mean_distribution = {
            candidate: 0.5
            * (
                float(normalized[0].get(candidate, 0.0))
                + float(normalized[1].get(candidate, 0.0))
            )
            for candidate in candidates
        }
        teacher_top1 = (
            _rank_distribution(normalized[0])[0],
            _rank_distribution(normalized[1])[0],
        )
        ranking = _rank_distribution(mean_distribution)
        agreement = teacher_top1[0] == teacher_top1[1]
        validity = "valid" if agreement else "teacher_top1_disagreement"
    return _normalize_raw_pool_row({
        "case_id": case_id,
        "teacher_seeds": list(teacher_seeds),
        "teacher_distributions": [
            dict(normalized[0]),
            dict(normalized[1]),
        ],
        "mean_distribution": mean_distribution,
        "teacher_top1": list(teacher_top1),
        "teacher_top1_agreement": agreement,
        "top1": ranking[0] if ranking else "",
        "top2": list(ranking[:2]),
        "candidate_ranking": list(ranking),
        "confidence": (
            _confidence(mean_distribution) if mean_distribution else 0.0
        ),
        "validity_status": validity,
    })


def write_raw_pseudo_pool(
    root: Path,
    *,
    canonical_dataset_id: str,
    pseudo_generator_candidate_id: str,
    selector_seed: int,
    query_case_ids: Sequence[str],
    query_plan_sha256: str,
    eligible_case_ids: Sequence[str],
    outer_test_case_ids: Sequence[str],
    teacher_predictions: Sequence[Mapping[str, Any]],
) -> RawPseudoPool:
    dataset = _canonical_dataset(canonical_dataset_id)
    seed = int(selector_seed)
    if seed not in ALLOWED_SELECTOR_SEEDS:
        raise ValueError("selector seed must be one of 42, 43, and 44")
    query = _case_ids(query_case_ids, role="queried")
    if len(query) != 30:
        raise ValueError("raw pseudo pool requires exactly 30 queried cases")
    query_hash = str(query_plan_sha256).strip().lower()
    if len(query_hash) != 64:
        raise ValueError("query_plan_sha256 must be a SHA256 digest")
    eligible = _case_ids(eligible_case_ids, role="eligible")
    outer_test = _case_ids(outer_test_case_ids, role="outer-test")
    if set(query) & set(eligible):
        raise ValueError("raw pseudo pool contains a queried case")
    if set(outer_test) & set(eligible):
        raise ValueError("raw pseudo pool contains an outer-test case")
    generator = str(pseudo_generator_candidate_id).strip()
    if not generator:
        raise ValueError("pseudo generator candidate ID must not be empty")
    teacher_seeds = (seed, seed + 1000)
    rows = tuple(
        sorted(
            (
                _build_row(raw, teacher_seeds=teacher_seeds)
                for raw in teacher_predictions
            ),
            key=lambda row: row["case_id"],
        )
    )
    row_ids = tuple(row["case_id"] for row in rows)
    if any(not case_id for case_id in row_ids):
        raise ValueError("teacher prediction case ID must not be empty")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("teacher prediction case IDs must be unique")
    if tuple(sorted(row_ids)) != eligible:
        raise ValueError(
            "teacher predictions must cover every eligible case exactly once"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data_path = root / "raw_predictions.jsonl"
    with data_path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row) + b"\n")
    data_sha256 = _sha256_file(data_path)
    reason_counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row["validity_status"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    manifest = {
        "schema_version": RAW_PSEUDO_MANIFEST_VERSION,
        "data_schema_version": RAW_PSEUDO_SCHEMA_VERSION,
        "canonical_dataset_id": dataset,
        "pseudo_generator_candidate_id": generator,
        "selector_seed": seed,
        "teacher_seeds": list(teacher_seeds),
        "query_case_ids": list(query),
        "query_case_count": len(query),
        "query_plan_sha256": query_hash,
        "eligible_case_ids": list(eligible),
        "eligible_case_count": len(eligible),
        "outer_test_case_ids": list(outer_test),
        "data_file": data_path.name,
        "raw_pool_sha256": data_sha256,
        "row_count": len(rows),
        "validity_reason_counts": reason_counts,
        "authoritative_or_oracle_fields_present": False,
        "immutable_after_generation": True,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    return RawPseudoPool(
        canonical_dataset_id=dataset,
        pseudo_generator_candidate_id=generator,
        selector_seed=seed,
        teacher_seeds=teacher_seeds,
        query_case_ids=query,
        query_plan_sha256=query_hash,
        eligible_case_ids=eligible,
        outer_test_case_ids=outer_test,
        rows=rows,
        data_path=str(data_path),
        manifest_path=str(manifest_path),
        sha256=data_sha256,
    )


def load_raw_pseudo_pool(manifest_path: Path) -> RawPseudoPool:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RAW_PSEUDO_MANIFEST_VERSION:
        raise ValueError("unsupported raw pseudo-pool manifest")
    data_path = manifest_path.parent / str(payload.get("data_file", ""))
    actual_hash = _sha256_file(data_path)
    if actual_hash != str(payload.get("raw_pool_sha256", "")):
        raise ValueError("raw pseudo-pool hash mismatch")
    rows = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                findings = _find_label_fields(row)
                if findings:
                    raise ValueError(
                        "raw pseudo pool contains a forbidden label field"
                    )
                rows.append(_normalize_raw_pool_row(row))
    eligible = _case_ids(payload.get("eligible_case_ids", ()), role="eligible")
    if tuple(sorted(row["case_id"] for row in rows)) != eligible:
        raise ValueError("raw pseudo-pool row coverage mismatch")
    return RawPseudoPool(
        canonical_dataset_id=_canonical_dataset(
            payload.get("canonical_dataset_id", "")
        ),
        pseudo_generator_candidate_id=str(
            payload.get("pseudo_generator_candidate_id", "")
        ),
        selector_seed=int(payload.get("selector_seed", -1)),
        teacher_seeds=tuple(
            int(value) for value in payload.get("teacher_seeds", ())
        ),
        query_case_ids=_case_ids(
            payload.get("query_case_ids", ()), role="queried"
        ),
        query_plan_sha256=str(payload.get("query_plan_sha256", "")),
        eligible_case_ids=eligible,
        outer_test_case_ids=_case_ids(
            payload.get("outer_test_case_ids", ()), role="outer-test"
        ),
        rows=tuple(rows),
        data_path=str(data_path),
        manifest_path=str(manifest_path),
        sha256=actual_hash,
    )


def _pseudo_row(
    row: Mapping[str, Any],
    *,
    pool: RawPseudoPool,
    label_mode: str,
    negative_strategy: str,
    pseudo_loss_ratio: float,
) -> Dict[str, Any]:
    targets = (
        [str(row["top1"])]
        if label_mode == "hard_top1"
        else [str(value) for value in row.get("top2", ())]
    )
    if not targets:
        raise ValueError("technically valid pseudo row has no target set")
    return {
        "case_id": str(row["case_id"]),
        "predicted_class": str(row["top1"]),
        "pseudo_target_set": targets,
        "label_mode": label_mode,
        "confidence": float(row["confidence"]),
        "raw_pool_sha256": pool.sha256,
        "teacher_candidate_rankings": [
            [str(value) for value in ranking]
            for ranking in row.get("teacher_candidate_rankings", ())
        ],
        "candidate_ranking": [
            str(value) for value in row.get("candidate_ranking", ())
        ],
        "negative_strategy": str(negative_strategy),
        "pseudo_loss_ratio": float(pseudo_loss_ratio),
        "teacher_bottom_fraction": 0.25,
        "minimum_rank_margin": 0.50,
    }


def derive_matched_pseudo_arms(
    pool: RawPseudoPool,
    *,
    label_mode: str,
    class_conditional_coverage: float,
    negative_strategy: str = "all_non_positive",
    pseudo_loss_ratio: float = 0.25,
) -> MatchedPseudoArms:
    if label_mode not in ALLOWED_LABEL_MODES:
        raise ValueError("unsupported pseudo label mode")
    coverage = float(class_conditional_coverage)
    if coverage not in ALLOWED_COVERAGES:
        raise ValueError("unsupported class-conditional coverage")
    strategy = str(negative_strategy)
    if strategy not in ALLOWED_NEGATIVE_STRATEGIES:
        raise ValueError("unsupported pseudo negative strategy")
    loss_ratio = float(pseudo_loss_ratio)
    if loss_ratio not in ALLOWED_PSEUDO_LOSS_RATIOS:
        raise ValueError("unsupported pseudo loss ratio")
    if _sha256_file(Path(pool.data_path)) != pool.sha256:
        raise ValueError("raw pseudo-pool hash mismatch")
    valid = [
        row for row in pool.rows if row["validity_status"] == "valid"
    ]
    grouped: Dict[str, list] = {}
    for row in valid:
        grouped.setdefault(str(row["top1"]), []).append(row)
    selected_ids = set()
    for rows in grouped.values():
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["confidence"]),
                str(row["case_id"]),
            ),
        )
        count = max(1, int(math.ceil(len(ordered) * coverage)))
        selected_ids.update(
            str(row["case_id"]) for row in ordered[:count]
        )

    all_rows = tuple(
        _pseudo_row(
            row,
            pool=pool,
            label_mode=label_mode,
            negative_strategy=strategy,
            pseudo_loss_ratio=loss_ratio,
        )
        for row in valid
    )
    selected_rows = tuple(
        _pseudo_row(
            row,
            pool=pool,
            label_mode=label_mode,
            negative_strategy=strategy,
            pseudo_loss_ratio=loss_ratio,
        )
        for row in valid
        if str(row["case_id"]) in selected_ids
    )
    invalid_rejections = tuple(
        {
            "case_id": str(row["case_id"]),
            "reason": str(row["validity_status"]),
        }
        for row in pool.rows
        if row["validity_status"] != "valid"
    )
    selected_rejections = tuple(
        {
            "case_id": str(row["case_id"]),
            "reason": (
                str(row["validity_status"])
                if row["validity_status"] != "valid"
                else "below_class_coverage_cutoff"
            ),
        }
        for row in pool.rows
        if row["validity_status"] != "valid"
        or str(row["case_id"]) not in selected_ids
    )

    def arm(
        name: str,
        pseudo_rows: Tuple[Dict[str, Any], ...],
        rejections: Tuple[Dict[str, Any], ...],
    ) -> MatchedPseudoArm:
        return MatchedPseudoArm(
            arm=name,
            canonical_dataset_id=pool.canonical_dataset_id,
            raw_pool_sha256=pool.sha256,
            raw_pool_path=pool.data_path,
            query_plan_sha256=pool.query_plan_sha256,
            query_case_ids=pool.query_case_ids,
            label_mode=label_mode,
            class_conditional_coverage=coverage,
            pseudo_rows=pseudo_rows,
            rejections=rejections,
            pseudo_count=len(pseudo_rows),
            pseudo_weight_policy=(
                "zero"
                if name == "query_only"
                else "case_equal_then_confidence_weighted"
            ),
            negative_strategy=strategy,
            pseudo_loss_ratio=loss_ratio,
            teacher_bottom_fraction=0.25,
            minimum_rank_margin=0.50,
        )

    return MatchedPseudoArms(
        query_only=arm(
            "query_only",
            (),
            tuple(
                {
                    "case_id": str(row["case_id"]),
                    "reason": "query_only_zero_pseudo_weight",
                }
                for row in pool.rows
            ),
        ),
        all_pseudo=arm("all_pseudo", all_rows, invalid_rejections),
        selected_pseudo=arm(
            "selected_pseudo", selected_rows, selected_rejections
        ),
    )


__all__ = [
    "ALLOWED_COVERAGES",
    "ALLOWED_LABEL_MODES",
    "ALLOWED_NEGATIVE_STRATEGIES",
    "ALLOWED_PSEUDO_LOSS_RATIOS",
    "ALLOWED_SELECTOR_SEEDS",
    "MatchedPseudoArm",
    "MatchedPseudoArms",
    "RAW_PSEUDO_MANIFEST_VERSION",
    "RAW_PSEUDO_SCHEMA_VERSION",
    "RawPseudoPool",
    "derive_matched_pseudo_arms",
    "load_raw_pseudo_pool",
    "write_raw_pseudo_pool",
]
