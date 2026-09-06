# VAST

VAST is an active root-cause localization research implementation for multimodal microservice incidents. This repository is an **interim private snapshot** of the currently selected method, not the final paper artifact and not a promise of interface stability.

The snapshot composes:

- HDBSCAN active acquisition with a configurable case budget (the frozen experiment uses 30) and seed 42;
- full outer-training-pool HDBSCAN proxy modes;
- hard-compatible proxy-mode conditional VAE augmentation with the `balanced` profile;
- a weighted pairwise-linear root-cause ranker;
- OSER-Meta observable-state residual correction with the `oser-p02` profile.

Historical DBSCAN acquisition/proxy partitions and the historical `residual_only` outer endpoint router are intentionally excluded.

## Status

The implementation has been collected to establish a clean repository and a stable provenance point while the final seed-42 experiment is still running. No benchmark data, query-plan instances, learned checkpoints, rankings, scores, or experiment outputs are distributed here. Results and full publication documentation will be added only after the method and evidence are finalized.

## Layout

```text
src/vast/                  thin public metadata facade
src/fixed_active_learning/ fixed HDBSCAN acquisition implementation
src/rcl_study/             verified CVAE, OSER, evaluation, and execution code
src/nexusrcl_rebuild/      minimal pairwise-linear backend
configs/final_rcl/         frozen scientific configuration
scripts/                   smoke, formal, and status entry points
tests/                     contract and integration tests
tools/                     snapshot and repository-safety checks
docs/provenance/           copied-source identity and transformation record
```

The internal `rcl_study` names are retained deliberately to minimize divergence from the code exercised by the experiments. A later publication pass may migrate them behind stable `vast.*` interfaces.

## Environment

Python 3.12 and scikit-learn 1.5.2 are the authority environment. The complete method also requires PyTorch with a suitable accelerator for practical CVAE and OSER training.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Runtime inputs

Data is supplied externally at runtime. The runners require fixed 70/30 split inventories, feature-bundle directories, the fixed HDBSCAN authority inputs, an output directory, and a PyTorch-enabled Python executable. Run each command with `--help` for the complete contract:

```bash
python scripts/run_final_rcl_real_smoke.py --help
python scripts/run_final_rcl_formal.py --help
python scripts/status_final_rcl.py --help
```

The three matched experiment arms are:

1. `hdbscan_query_cvae_oser`: 30 HDBSCAN-queried outer-training cases plus CVAE and OSER;
2. `oracle_full_cvae_oser`: all outer-training cases plus the same CVAE and OSER method;
3. `hdbscan_query_pairwise`: the same 30 queried cases with the pairwise-linear downstream control.

Outer-test cases are evaluation-only in every arm.

## Provenance and safety

The snapshot inventory can be checked without benchmark data:

```bash
python tools/snapshot_sources.py --destination-root . --verify-only
python tools/repository_firewall.py .
```

See `docs/provenance/README.md` for the provenance schema. The repository firewall rejects data/result/checkpoint paths, credential material, and authority-machine absolute paths before publication.

