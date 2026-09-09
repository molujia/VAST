"""Public configuration of the owner-selected final B method."""
from __future__ import annotations
import json
from pathlib import Path

REPOSITORY_ROOT=Path(__file__).resolve().parents[2]
HISTORICAL_ROOT=Path(__file__).resolve().parent/"_historical"

def describe_method():
    return {"name":"VAST","status":"final-selected","version":"1.0.0","method":"B",
        "active_learning":"hdbscan_leaf_center","historical_features":"preserved_complete",
        "representation":"historical_plus_48_candidate_latents","posterior_samples":8,
        "descendant_pair_mass":0.25,"inference":"posterior_mean","internal_correction":"oser-p02"}

def load_default_config():
    return json.loads(Path(__file__).with_name("default_config.json").read_text(encoding="utf-8"))

def with_historical_authority(config):
    from rcl_study.vae_snapshot_contract import file_sha256
    inventory=json.loads((REPOSITORY_ROOT/"docs/provenance/source-inventory.json").read_text(encoding="utf-8"))
    prefix="src/vast/_historical/"
    hashes={r["destination_path"][len(prefix):]:r["destination_sha256"] for r in inventory["files"] if r["destination_path"].startswith(prefix)}
    if not hashes or any(file_sha256(HISTORICAL_ROOT/name)!=digest for name,digest in hashes.items()):
        raise ValueError("packaged historical source integrity check failed")
    return {**config,"resolved_historical_source":str(HISTORICAL_ROOT),
        "authority":{"historical_source":{"required_file_sha256s":hashes}}}
