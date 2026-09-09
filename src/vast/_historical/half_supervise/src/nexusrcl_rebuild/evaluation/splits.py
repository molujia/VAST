"""Chronological and leakage-safe window splits for RCA benchmarking."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    dataset: str
    train_window_ids: Sequence[str]
    val_window_ids: Sequence[str]
    test_window_ids: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NestedDatasetSplit:
    dataset: str
    outer_train_window_ids: Sequence[str]
    outer_test_window_ids: Sequence[str]
    inner_train_window_ids: Sequence[str]
    inner_val_window_ids: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


_NESTED_SPLIT_KEYS = (
    "outer_train_window_ids",
    "outer_test_window_ids",
    "inner_train_window_ids",
    "inner_val_window_ids",
)


def _fixed_split_ids(payload: Mapping[str, Any], key: str) -> List[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError("fixed split field must be a list: %s" % key)
    result = [str(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError("fixed split contains duplicate IDs in %s" % key)
    return result


def load_fixed_nested_split(
    path: Path,
    *,
    windows: Optional[pd.DataFrame] = None,
    expected_dataset: Optional[str] = None,
) -> NestedDatasetSplit:
    """Load an immutable nested split and reject stale or leaky membership."""

    split_path = Path(path)
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("invalid fixed split: %s" % split_path) from error
    if not isinstance(payload, Mapping):
        raise ValueError("fixed split must contain a JSON mapping")

    dataset = str(payload.get("dataset") or "")
    if not dataset:
        raise ValueError("fixed split dataset must not be empty")
    if expected_dataset is not None and dataset != str(expected_dataset):
        raise ValueError(
            "fixed split dataset mismatch: expected=%s actual=%s"
            % (expected_dataset, dataset)
        )

    values = {key: _fixed_split_ids(payload, key) for key in _NESTED_SPLIT_KEYS}
    outer_train = set(values["outer_train_window_ids"])
    outer_test = set(values["outer_test_window_ids"])
    inner_train = set(values["inner_train_window_ids"])
    inner_val = set(values["inner_val_window_ids"])
    if outer_train.intersection(outer_test):
        raise ValueError("fixed outer split views overlap")
    if inner_train.intersection(inner_val):
        raise ValueError("fixed inner split views overlap")
    if inner_train.union(inner_val) != outer_train:
        raise ValueError("fixed inner views do not partition outer training")

    if windows is not None:
        required_columns = {"dataset", "window_id"}
        if not required_columns.issubset(windows.columns):
            raise ValueError("feature windows are missing dataset or window_id")
        feature_ids = windows["window_id"].astype(str).tolist()
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("feature windows contain duplicate window IDs")
        feature_id_set = set(feature_ids)
        declared_ids = outer_train.union(outer_test)
        unknown = sorted(declared_ids.difference(feature_id_set))
        if unknown:
            raise ValueError("fixed split contains unknown window IDs: %s" % unknown[:3])
        if declared_ids != feature_id_set:
            raise ValueError("fixed outer views do not cover the feature windows")
        feature_datasets = set(windows["dataset"].astype(str))
        if feature_datasets != {dataset}:
            raise ValueError(
                "feature window dataset mismatch: expected=%s actual=%s"
                % (dataset, sorted(feature_datasets))
            )

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("fixed split metadata must be a mapping")
    return NestedDatasetSplit(
        dataset=dataset,
        outer_train_window_ids=values["outer_train_window_ids"],
        outer_test_window_ids=values["outer_test_window_ids"],
        inner_train_window_ids=values["inner_train_window_ids"],
        inner_val_window_ids=values["inner_val_window_ids"],
        metadata={str(key): value for key, value in metadata.items()},
    )


def _partition_window_ids(
    frame: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
) -> List[List[str]]:
    ordered = frame.sort_values(by=["start_ts", "window_id"]).reset_index(drop=True)
    total = len(ordered)
    if total == 0:
        return [[], [], []]

    if test_ratio > 0.0:
        test_count = int(round(total * test_ratio))
        if total >= 3:
            test_count = max(1, test_count)
    else:
        test_count = 0
    if val_ratio > 0.0:
        val_count = int(round(total * val_ratio))
        if total >= 5:
            val_count = max(1, val_count)
    else:
        val_count = 0
    if total <= 2 and test_ratio > 0.0:
        test_count = 1
    if test_count + val_count >= total:
        val_count = max(0, total - test_count - 1)
    train_count = max(total - test_count - val_count, 1)
    val_count = max(total - train_count - test_count, 0)
    if train_count + val_count + test_count != total:
        test_count = total - train_count - val_count

    train_ids = ordered.iloc[:train_count]["window_id"].astype(str).tolist()
    val_ids = ordered.iloc[train_count : train_count + val_count]["window_id"].astype(str).tolist()
    test_ids = ordered.iloc[train_count + val_count :]["window_id"].astype(str).tolist()
    return [train_ids, val_ids, test_ids]


def build_chronological_split(
    windows: pd.DataFrame,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> DatasetSplit:
    dataset = str(windows["dataset"].iloc[0]) if not windows.empty else "unknown"
    fault_windows = windows[windows["window_kind"] == "fault"].copy()
    normal_windows = windows[windows["window_kind"] == "normal"].copy()

    fault_train, fault_val, fault_test = _partition_window_ids(fault_windows, val_ratio, test_ratio)
    normal_train, normal_val, normal_test = _partition_window_ids(normal_windows, val_ratio, test_ratio)

    return DatasetSplit(
        dataset=dataset,
        train_window_ids=fault_train + normal_train,
        val_window_ids=fault_val + normal_val,
        test_window_ids=fault_test + normal_test,
        metadata={
            "fault_counts": {
                "train": len(fault_train),
                "val": len(fault_val),
                "test": len(fault_test),
            },
            "normal_counts": {
                "train": len(normal_train),
                "val": len(normal_val),
                "test": len(normal_test),
            },
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
    )


def build_nested_chronological_splits(
    windows: pd.DataFrame,
    outer_test_ratio: float = 0.3,
    inner_val_ratio: float = 0.2,
) -> NestedDatasetSplit:
    outer_split = build_chronological_split(
        windows,
        val_ratio=0.0,
        test_ratio=outer_test_ratio,
    )
    outer_train_windows = windows[
        windows["window_id"].astype(str).isin(
            {str(window_id) for window_id in outer_split.train_window_ids}
        )
    ].copy()
    inner_split = build_chronological_split(
        outer_train_windows,
        val_ratio=inner_val_ratio,
        test_ratio=0.0,
    )
    return NestedDatasetSplit(
        dataset=outer_split.dataset,
        outer_train_window_ids=list(outer_split.train_window_ids),
        outer_test_window_ids=list(outer_split.test_window_ids),
        inner_train_window_ids=list(inner_split.train_window_ids),
        inner_val_window_ids=list(inner_split.val_window_ids),
        metadata={
            "outer": dict(outer_split.metadata),
            "inner": dict(inner_split.metadata),
        },
    )
