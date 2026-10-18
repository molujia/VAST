# VAST

**Propagation-aware active PNU learning for microservice root cause localization.**

VAST encodes metrics, logs and typed topology, selects complete incidents whose
observable propagation patterns add coverage, and learns to rank root-cause
candidates from queried and unqueried incidents.

This is the **3.0.0.dev0 core implementation**, previously developed as VAST-V3.
The method is now called **VAST**. The Python namespace `vast_v3` and versioned
artifact schemas remain unchanged for compatibility. The former CVAE/OSER
implementation is preserved in [Git history](https://github.com/molujia/VAST/tree/b1ae14126100743b03e83d3dd4b83d958a80c389).

The core has passed separate one-case server smoke checks on RCABench and
AIOps22-pre. Full experiments and the two audited TOP1 demonstration packages
are **not yet complete**. This release contains no benchmark scores, data,
trained checkpoints or preselected successful cases. Passing a smoke check
establishes execution, not localization accuracy.

## Method at a glance

1. **Encode before associating.** Separate 1D CNN branches encode native metric
   and directed-edge trace numerics. A frozen Sentence Transformer encodes logs;
   a typed GCN encodes topology. Entity/time association follows these encoders,
   then projection and LayerNorm form fixed-width states. Directed-edge numeric
   states remain available separately, preserving their owners.
2. **Represent propagation processes.** The shared **SCOPE** encoder combines
   anonymous structure, local symptoms, ordered paths and physical time gaps in
   multiple fragment tokens per incident. Candidate marks identify the entity
   being ranked without using its identity as a learned feature.
3. **Acquire complete incidents.** **PropCover** greedily adds cases with the
   greatest conditional fragment-coverage gain relative to already queried
   cases. It uses observable representations, not hidden root/fault-type labels
   or a hard clustering-and-centroid rule. The default budget is six rounds of
   five cases; the shared representation is refreshed between rounds.
4. **Train and rank.** Queried complete root sets provide positive and nonroot
   negative candidates. Observation recovery and training-only **RSC** source
   competition also use unqueried observations. Inference retains the encoder
   and ranker, without source search, labels or test-time fitting.

Nonroot does not mean healthy. Missing telemetry and an absent alert are not
negative root labels. Observable propagation patterns are not asserted to be
complete causal paths. See the [architecture contract](docs/architecture.md).

## Installation

Use Python 3.11 or newer in a fresh virtual environment. Install the appropriate
PyTorch build for your CPU/CUDA environment before running real workloads.

```bash
git clone https://github.com/molujia/VAST.git
cd VAST
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[test,text]'
```

The `text` extra is required for real log encoding. The package never downloads
model weights or datasets automatically: supply the pinned offline
`sentence-transformers/all-MiniLM-L6-v2` files matching
[`configs/text-minilm-lock.json`](configs/text-minilm-lock.json).
Model weights have their own upstream license, separate from this repository.

## Data-free verification

```bash
python -m pytest tests/unit tests/review tests/integration \
  --basetemp=.cache/pytest-public -o cache_dir=.cache/pytest-cache
python -m vast_v3.cli.smoke --help
python -m vast_v3.cli.curate --help
python -m vast_v3.cli.train --help
python -m vast_v3.cli.quickrun --help
```

The tests create synthetic fixtures; they do not reproduce benchmark results.
CUDA-specific tests skip when no CUDA device is available. A narrowly bounded
CPU numerical exception remains explicitly tracked as an expected failure;
it does not relax other output, gradient or checkpoint checks. Use a fresh
`--basetemp` path if another test process is active.

## Running with authorized data

The supported adapters are `rcabench` and `aiops2022_pre`. You must provide
authorized raw telemetry, offline text assets and the original frozen metadata.
No prepared benchmark manifest or labels are distributed in this release.

The entry points have distinct purposes:

| Entry point | Purpose |
| --- | --- |
| `python -m vast_v3.cli.curate` | Independent preparation of a public manifest and separate sealed root labels |
| `python -m vast_v3.cli.smoke` | One permitted training-case execution check; no root reads or accuracy claim |
| `python -m vast_v3.cli.train` | Seed-42, A3, closed-loop B30 reference preparation on the full frozen training pool |
| `python -m vast_v3.cli.quickrun` | Explicit snapshot/run over an existing validated cache subset; 16-epoch engineering check |

The standard-source file locations are configurable; all original checksums,
candidate counts and split checks remain enforced. See
[inputs, commands and resumption](docs/inputs-and-runs.md) and
[`configs/standard-sources.example.env`](configs/standard-sources.example.env).
An arbitrary replacement dataset is **not** accepted as the frozen benchmark.
Other datasets require their own explicitly reviewed manifest/adapter protocol.

The default full schedule is 170 epochs. The cache-subset quickrun uses one
warm-up epoch, five one-epoch query rounds and a final ten-epoch round. Its
completion gate is different and must not be presented as a full-pool result.
Neither entry point performs the still-pending audited TOP1 example search.

## Repository layout

```text
src/vast_v3/
  data/          Native observations, manifests, curator boundary and caches
  encoders/      Modality front ends, SCOPE and exact direct fragment sampling
  acquisition/   PropCover and controlled comparison selectors
  models/        Candidate ranker and training-only RSC
  training/      Objectives, active loop, schedules and identity-bound recovery
  evaluation/    Ranking metrics, semantic partitions and search-ledger primitives
  annotations/   Packaged schema and evidence-consistency validation
  cli/           Curate, smoke, train and cached-subset quickrun
configs/         Offline text lock and source-path configuration example
annotations/     Blank semantic annotation templates, not real labels
tests/           Portable unit, integration and invariant checks
docs/            Method, input/run and release-boundary documentation
```

Internal server launchers, diagnostic scripts and private experiment reports
are intentionally not part of the public core. The search-ledger and semantic
partition library components are present; complete demo search/export commands
and audited example packages remain pending. This is not the full paper
experiment automation suite.

## License and citation

The repository retains its [MIT license](LICENSE). Dataset and model
redistribution rights must be checked independently. Use [CITATION.cff](CITATION.cff)
and record the exact commit; do not attribute historical VAST scores to this
implementation. See [source provenance and migration](docs/provenance.md).
