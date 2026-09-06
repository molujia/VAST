# VAST Interim Repository Design

## Purpose

Create a private, revision-friendly repository for the currently selected VAST method while the final seed-42 evaluation is still running. This repository is an interim engineering snapshot, not the final paper artifact or a claim that the method interface is stable.

## Frozen Method Boundary

The snapshot represents exactly the currently selected composition:

- HDBSCAN active acquisition with budget 30 and seed 42;
- HDBSCAN-owned proxy modes over the complete outer-training pool;
- hard-compatible proxy-mode conditional VAE augmentation using the balanced profile;
- weighted pairwise-linear root-cause ranking;
- OSER-Meta observable-state residual correction using the `oser-p02` profile;
- no historical DBSCAN acquisition or proxy partition;
- no historical `residual_only` outer endpoint router.

The repository will retain the algorithmic implementation for both query-only and oracle-full supervision. Dataset-specific results are not part of this snapshot.

## Repository Shape

The interim repository uses a faithful-source layout with a thin public facade:

```text
VAST/
├── src/vast/                  # Small stable facade and version metadata
├── src/fixed_active_learning/ # HDBSCAN acquisition implementation
├── src/rcl_study/             # Verified CVAE, pairwise, OSER, execution, evaluation code
├── src/nexusrcl_rebuild/      # Minimal pairwise backend dependency
├── configs/                   # Method configuration and safe templates
├── scripts/                   # Smoke, formal, and status entry points
├── tests/                     # Focused contract and integration tests
├── docs/provenance/           # Source inventory and snapshot identity
├── README.md                  # Deliberately concise interim documentation
├── pyproject.toml
└── .gitignore
```

The existing `rcl_study` module names are preserved to minimize divergence from the experimentally exercised code. `vast` exposes only a small facade in this revision. A later publication pass may migrate internal modules behind formal `vast.*` namespaces after the final method is settled.

## Source Selection

Only the transitive source closure needed by the selected method and its focused tests is admitted. The HDBSCAN package contributes code and reusable configuration, but its dataset-specific candidate pools, query plans, reproduced results, score files, feature inventories, manifests tied to local data, and cached bytecode are excluded.

The integration snapshot contributes the final RCL contract, HDBSCAN proxy adapter, compatible CVAE stack, pairwise bridge/backend, OSER stack, evaluation, resumable execution, and their direct helpers. Rejected research arms and unrelated historical runners are excluded.

## Data and Secret Firewall

The repository must not contain:

- raw or processed RCABench, AIOps22-pre, or AIOps25 data;
- case inventories, feature matrices, candidate-pool instances, query-plan instances, rankings, scores, checkpoints, or experiment outputs;
- machine-specific absolute paths in public examples;
- GitHub tokens, credentials, SSH material, environment dumps, or local secret files;
- Python caches, test caches, temporary files, or editor state.

Configuration files may retain scientific hyperparameters. Dataset-bound hashes and paths are replaced by documented runtime inputs or safe example values when they are not required to define the algorithm.

## Provenance

`docs/provenance/source-inventory.json` records every copied source path, destination path, source SHA-256, destination SHA-256, component role, and snapshot date. The inventory also records the authority package and OpenSpec change names without embedding data or results. This gives later revisions a precise comparison point.

## Validation

The interim snapshot is accepted when:

1. an automated firewall scan finds no known data/output/checkpoint/cache paths or credential material;
2. every declared source-inventory entry exists and has the recorded hash;
3. all internal Python imports in the selected closure resolve from the repository;
4. the dependency-light contract tests pass locally;
5. the full focused suite passes in the authoritative `rcalab`/PyTorch environments when those dependencies are available;
6. Git reports a clean working tree after commit and the commit is visible on `origin/main`.

The ongoing formal experiment remains authoritative for scores. No partial result is copied into this interim repository.

## Git and Credential Handling

The remote URL remains `https://github.com/molujia/VAST.git` without embedded credentials. Authentication is provided per command from `git_personal_access_token`; the token is never persisted in Git configuration, repository files, commit messages, or generated provenance.

