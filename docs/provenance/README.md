# Final B provenance

The owner selected method B and approved the seed-42 results on 2026-09-09.
Version 1.0.0 supersedes the 2026-09-06 interim discrete snapshot.

`source-inventory.json` records each copied file's source-relative path,
SHA-256 before transformation, destination-relative path, destination SHA-256,
role, and transformation flags. The inventory itself has a canonical semantic
hash. New public runtime, documentation, and release tests are maintained in Git;
they are not falsely presented as byte-identical experimental source files.

## Authorities

- Scientific implementation: `vae-post-al-training-optimization`, completed
  formal run `vae-snapshot-rebase-seed42-20260908-01`.
- Complete historical training/feature implementation:
  `current_label_free_reference_20260821_final`.
- Frozen acquisition: `rcl-active-learning-fixed-v1-20260906`.

The original historical `semi_supervised_bounded.py` SHA-256 is
`c688b1abdc1b5d52aec930190dcbecb97cdf7aabba254ce4a9f27116f78c038b`;
`training/semisupervised.py` is
`076f5aa8b7783216ac82c55220c4612637e12e6ff005238a0cec9619cf24f7d7`.
Where author-machine paths needed replacement, the destination hash differs and
the inventory records both identities. Historical Python modules named
`datasets` are code, not benchmark data; only that exact source package is exempt
from the repository's data-directory exclusion.

## Mechanical changes

`tools/final_sources.py` selects the AST import closure of B and its historical
dependencies. It preserves numerical CVAE, mean-latent, pairwise weighting,
OSER fitting and inference functions. It removes obsolete experiment orchestration
and A generation dispatch, adapts source paths to `src`, routes cache access to
an explicit run directory, and retains the original ranking metric functions
with their hash helpers. Known machine paths become explicit placeholders in
legacy helpers. Those helpers do not discover data for the public runner.

`vast.runtime` replaces author-specific control-run discovery with explicit
feature bundles, independent split membership, frozen query plans and optional
historical supervision plans. Historical preparation still uses the original
private historical package and complete feature semantics. Its source inventory
is checked before loading. Numerical thread settings are not overridden.

The copied historical/latent/stage tests only replace historical source loading
with the packaged authority. Public CLI and release tests exercise new boundaries.
The builder is a maintainer tool requiring the original workspace authorities;
users of this checkout need only `--verify-only`.

## Release validation

Validation uses an isolated Linux checkout and the existing controller/PyTorch
environments. Formal models and artifacts are read only. No formal training is
launched. `tools/validate_reference.py` checks rebuilt full input frames and CVAE
scientific requests, re-encodes test candidates with the saved CVAE, scores the
saved B ranker, applies saved OSER, and compares complete candidate rankings and
metrics. Its output remains outside this repository.

Validated on 2026-09-09:

- Controller regression suite: **45 passed** in 2.33 s. The optional two-interpreter
  test was skipped in that invocation, then run separately below.
- Synthetic full CLI integration: **1 passed** in 156.31 s. Ten synthetic fault
  windows / three candidates exercise a 3-supervision, 3-test, 5-CVAE-step smoke,
  OSER, reporting and status. Resume leaves both checkpoint hashes and mtimes
  unchanged. Modifying an external input causes resume to refuse the changed
  identity. This test is not benchmark evidence.
- Four full reference B units: rebuilt historical input frames and CVAE scientific
  input are exact; re-encoded latent, base scores and final OSER scores all have
  maximum absolute difference **0.0**. Complete candidate rankings and all metric
  objects match the approved reference exactly. **Zero formal models trained.**
- Source inventory: **115 copied files** verified. Local source compilation and
  the repository firewall pass. No data, checkpoints, raw rankings or credentials
  are included in the published checkout.

| Unit | Test cases | Features | Latent max error | Base max error | Final max error | Complete rankings |
|---|---:|---:|---:|---:|---:|---|
| RCABench Query | 427 | 536 | 0.0 | 0.0 | 0.0 | exact |
| RCABench Oracle | 427 | 536 | 0.0 | 0.0 | 0.0 | exact |
| AIOps22-pre Query | 72 | 216 | 0.0 | 0.0 | 0.0 | exact |
| AIOps22-pre Oracle | 72 | 216 | 0.0 | 0.0 | 0.0 | exact |

This is a forward-replay and input-preparation equivalence check with existing
formal weights, supplemented by synthetic training. It is not a second full
formal retraining run or a claim of bitwise portability across all environments.
