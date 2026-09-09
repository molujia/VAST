"""Dataset rebuilding package for HD1, HD2, HD3, and HD4."""

from .hd1 import build_hd1_manifest
from .hd2 import build_hd2_manifest
from .hd3 import build_hd3_manifest
from .hd4 import build_hd4_manifest


def build_all_manifests(*args, **kwargs):
    from .build_manifests import build_all_manifests as _build_all_manifests

    return _build_all_manifests(*args, **kwargs)


__all__ = [
    "build_all_manifests",
    "build_hd1_manifest",
    "build_hd2_manifest",
    "build_hd3_manifest",
    "build_hd4_manifest",
]
