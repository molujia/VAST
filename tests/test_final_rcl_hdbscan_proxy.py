from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest

from rcl_study.final_rcl_hdbscan_proxy import (
    HDBSCANProxyContractError,
    build_hdbscan_proxy_partition,
    build_hdbscan_proxy_partition_from_geometry,
    validate_hdbscan_proxy_partition,
)


SHA = {
    "representation": "1" * 64,
    "geometry": "2" * 64,
    "active_partition": "3" * 64,
}


def _partition(labels: list[int] | None = None) -> dict[str, object]:
    return build_hdbscan_proxy_partition(
        dataset_id="rcabench",
        fit_case_ids=["a", "b", "c", "n1", "n2"],
        queried_case_ids=["a", "c", "n1"],
        held_out_case_ids=["test"],
        raw_labels=labels or [0, 0, 1, -1, -1],
        membership_strengths=[0.9, 0.8, 0.7, 0.0, 0.0],
        representation_matrix_sha256=SHA["representation"],
        geometry_sha256=SHA["geometry"],
        active_partition_sha256=SHA["active_partition"],
        min_cluster_size=5,
        min_samples=5,
        cluster_selection_method="leaf",
    )


def test_hdbscan_partition_covers_full_fit_pool_and_keeps_noise_independent() -> None:
    partition = _partition()
    audit = validate_hdbscan_proxy_partition(
        partition,
        expected_dataset_id="rcabench",
        expected_fit_case_ids=["a", "b", "c", "n1", "n2"],
        expected_queried_case_ids=["a", "c", "n1"],
        expected_held_out_case_ids=["test"],
        expected_representation_matrix_sha256=SHA["representation"],
        expected_geometry_sha256=SHA["geometry"],
        expected_active_partition_sha256=SHA["active_partition"],
    )

    assert audit == {
        "valid": True,
        "fit_case_count": 5,
        "queried_case_count": 3,
        "noise_case_count": 2,
        "non_noise_mode_count": 2,
        "proxy_partition_sha256": partition["proxy_partition_sha256"],
    }
    assert partition["algorithm"] == "sklearn.cluster.HDBSCAN"
    assert partition["case_mode_ids"]["n1"].startswith("noise:")
    assert partition["case_mode_ids"]["n2"].startswith("noise:")
    assert partition["case_mode_ids"]["n1"] != partition["case_mode_ids"]["n2"]
    assert partition["queried_mode_members"][partition["case_mode_ids"]["n1"]] == ["n1"]
    assert partition["case_records"][3]["is_noise"] is True
    assert partition["case_records"][3]["membership_strength"] == 0.0


def test_non_noise_mode_identity_depends_on_members_not_numeric_label() -> None:
    first = _partition([0, 0, 1, -1, -1])
    relabeled = _partition([9, 9, 4, -1, -1])

    assert first["case_mode_ids"] == relabeled["case_mode_ids"]
    assert first["case_mode_ids"]["a"] == first["case_mode_ids"]["b"]


@dataclass(frozen=True)
class _Geometry:
    dataset_id: str = "rcabench"
    case_ids: tuple[str, ...] = ("a", "b", "c", "n1", "n2")
    labels: tuple[int, ...] = (0, 0, 1, -1, -1)
    membership_strengths: tuple[float, ...] = (0.9, 0.8, 0.7, 0.0, 0.0)
    representation_matrix_sha256: str = "1" * 64
    geometry_sha256: str = "2" * 64
    min_cluster_size: int = 5
    min_samples: int = 5
    cluster_selection_method: str = "leaf"
    clusterer_id: str = "hdbscan"


def test_geometry_adapter_binds_the_authority_partition() -> None:
    partition = build_hdbscan_proxy_partition_from_geometry(
        geometry=_Geometry(),
        queried_case_ids=["a", "c", "n1"],
        held_out_case_ids=["test"],
        active_partition_sha256=SHA["active_partition"],
    )

    assert partition["fit_case_ids"] == ["a", "b", "c", "n1", "n2"]
    assert partition["parameters"] == {
        "min_cluster_size": 5,
        "min_samples": 5,
        "cluster_selection_method": "leaf",
        "allow_single_cluster": False,
    }


def test_proxy_partition_rejects_dbscan_and_test_overlap() -> None:
    partition = _partition()
    attacked = deepcopy(partition)
    attacked["algorithm"] = "DBSCAN"
    with pytest.raises(HDBSCANProxyContractError, match="HDBSCAN"):
        validate_hdbscan_proxy_partition(
            attacked,
            expected_dataset_id="rcabench",
            expected_fit_case_ids=["a", "b", "c", "n1", "n2"],
            expected_queried_case_ids=["a", "c", "n1"],
            expected_held_out_case_ids=["test"],
            expected_representation_matrix_sha256=SHA["representation"],
            expected_geometry_sha256=SHA["geometry"],
            expected_active_partition_sha256=SHA["active_partition"],
        )

    with pytest.raises(HDBSCANProxyContractError, match="held-out"):
        build_hdbscan_proxy_partition(
            dataset_id="rcabench",
            fit_case_ids=["a", "test"],
            queried_case_ids=["a"],
            held_out_case_ids=["test"],
            raw_labels=[0, 0],
            membership_strengths=[1.0, 1.0],
            representation_matrix_sha256=SHA["representation"],
            geometry_sha256=SHA["geometry"],
            active_partition_sha256=SHA["active_partition"],
            min_cluster_size=5,
            min_samples=5,
            cluster_selection_method="leaf",
        )
