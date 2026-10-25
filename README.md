# VAST

VAST is a **propagation-aware active semi-supervised learning** framework for
microservice root cause localization, presented in *Which Cases Are Worth
Labeling for Microservice Root Cause Localization?* It selects informative
historical cases for annotation and learns to rank candidate root-cause
services, instances, or hosts from metrics, logs, traces, and typed topology.

**Active learning** determines which complete cases to query. Each query reveals
the case's complete verified root set. **Semi-supervised learning** combines
root supervision on queried cases with observation self-supervision and
source-conditioned learning on permitted labeled and unlabeled cases. Non-root
candidates may be healthy or affected by propagation. Candidates in unqueried
cases remain unlabeled and are not treated as negatives or a third class.

The framework has five stages:

1. **Multimodal encoding and fusion** aligns and fuses telemetry embeddings by
   entity and time while retaining local symptoms and directed edge records.
2. **SCOPE**, the **Symptom-Conditioned Ordered Propagation Encoder**, represents
   local propagation fragments through symptoms, relations, event order, and
   physical delays.
3. **PropCover acquisition** queries complete cases that add propagation
   coverage, without accessing their unknown root or fault-type labels.
4. **Semi-supervised ranker training** combines root supervision, observation
   self-supervision, and **Root-Set-Conditioned Completion (RSC)**. RSC compares
   root-set hypotheses by their reconstruction of masked observations without
   treating inferred hypotheses as verified root labels.
5. **Inference** ranks new-case candidates using the frozen modality encoders,
   SCOPE, and ranker, without querying, reconstruction, or parameter updates.

This repository provides the **3.0.0.dev0 reference core**. The Python namespace
`vast_v3` and versioned artifact schemas remain unchanged for compatibility.
Benchmark data, frozen split assets, trained checkpoints, and the complete
paper-experiment automation suite are not bundled. See the
[architecture contract](docs/architecture.md) for implementation details.

## Repository Layout

```text
VAST/
  src/vast_v3/
    runtime.py     Runtime configuration and model construction
    data/          Telemetry adapters, manifests, curation, and observation caches
    encoders/      Modality encoders, SCOPE, and fragment sampling
    acquisition/   PropCover and comparison selectors
    models/        Candidate ranker and training-only RSC
    training/      Objectives, active-learning loop, schedules, and recovery
    evaluation/    Ranking metrics and evaluation primitives
    annotations/   Annotation schema and consistency validation
    cli/           Curate, smoke, train, and cached-subset quickrun entry points
  configs/         Offline text-model lock and source-path configuration example
  annotations/     Blank annotation templates, not benchmark labels
  tests/           Unit, integration, and invariant checks
  docs/            Architecture, input/run instructions, and source provenance
  pyproject.toml   Package metadata and dependencies
  CITATION.cff     Software citation metadata
  LICENSE
```

Private experiment outputs, server launchers, datasets, and model weights are
not part of the tracked core.

## Requirements

Use **Python 3.11 or newer** in a fresh virtual environment. Install a PyTorch
build suitable for your CPU/CUDA environment. Package dependencies and optional
extras are defined in [pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/molujia/VAST.git
cd VAST
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,text]"
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1` instead. Multiline
commands below use Bash continuation syntax.

Real log encoding requires the `text` extra and a local
`sentence-transformers/all-MiniLM-L6-v2` snapshot matching
[configs/text-minilm-lock.json](configs/text-minilm-lock.json). Obtain the
pinned model files separately under their upstream terms. VAST does not
automatically download weights or datasets, and it does not require an LLM API
key or server.

## Datasets

The paper evaluates **two public benchmarks**, denoted **D1** and **D2**. The
counts below describe VAST's retained/preprocessed versions, not every case in
the upstream releases. Both provide metrics, logs, traces, and incident
annotations.

| Paper ID | CLI dataset ID | Benchmark | Retained cases | Fault types | Source |
| --- | --- | --- | ---: | ---: | --- |
| D1 | `rcabench` | Train-Ticket / RCABench | 1,422 | 25 | [Dataset release](https://zenodo.org/records/17105974), [project page](https://operationspai.github.io/revisiting-rca-evaluation/) |
| D2 | `aiops2022_pre` | Cloud e-commerce / AIOps22-pre | 241 | 15 | [Aiops-Dataset release repository](https://github.com/bbyldebb/Aiops-Dataset) |

- **D1** is the retained collection from the 50-service Train-Ticket benchmark
  released alongside SimpleRCA. Faults are injected under dynamic workloads
  and validated for user-facing service-level indicator degradation.
- **D2** is a preprocessed collection from a cloud-deployed e-commerce system
  with 40 microservice instances and six virtual machines. Scenarios derived
  from real failures were replayed over several days in May 2022, covering
  resource, network, and process faults.

Follow the original dataset release terms. This repository does **not**
redistribute raw telemetry, root labels, VAST-specific frozen split metadata,
or prepared benchmark manifests. An upstream download alone is not a complete
VAST reproduction package. The supported layouts and required frozen assets
are documented in [Inputs, execution and recovery](docs/inputs-and-runs.md).

## Quick Start

### 1. Check the installation without benchmark data

```bash
python -m pytest tests/unit tests/review tests/integration --basetemp=.cache/pytest-public -o cache_dir=.cache/pytest-cache
python -m vast_v3.cli.curate --help
python -m vast_v3.cli.smoke --help
python -m vast_v3.cli.train --help
python -m vast_v3.cli.quickrun --help
```

The default tests use generated fixtures. CUDA and real offline-text checks
require their respective optional resources. Skips and expected failures are
reported by pytest. These checks validate execution and invariants, not the
paper's benchmark scores. Use a fresh `--basetemp` path if another test process
is active.

### 2. Prepare authorized benchmark assets

Real-data commands require the raw telemetry, the pinned offline text model,
and the original frozen windows, candidate catalog, training-pool, and test-ID
files. Configure the four dataset-specific environment variables listed in
[configs/standard-sources.example.env](configs/standard-sources.example.env).
That example file is **not loaded automatically**. Existing checksum, identity,
and population checks remain enforced.

For D1, the variable names are `VAST_RCABENCH_WINDOWS`,
`VAST_RCABENCH_CATALOG`, `VAST_RCABENCH_TRAIN_POOL`, and
`VAST_RCABENCH_TEST_IDS`. For D2, replace `RCABENCH` with `AIOPS2022_PRE`.
See the [input configuration guide](docs/inputs-and-runs.md#required-assets)
for the expected contents and shell-specific examples.

With those assets configured, create the public observation manifest and the
separate sealed root-label artifact:

```bash
python -m vast_v3.cli.curate \
  --dataset rcabench --raw-root /data/rcabench \
  --output-dir /work/curator/rcabench-v001
```

The output contains `public-manifest.json`, `sealed-labels.json`, and
`curator-summary.json`. Keep the sealed labels private. The learner accesses
training labels only through the budgeted query interface, and inference does
not read root labels.

### 3. Run a one-case execution check

```bash
python -m vast_v3.cli.smoke \
  --dataset rcabench --raw-root /data/rcabench \
  --manifest /work/curator/rcabench-v001/public-manifest.json \
  --text-directory /models/minilm-pinned \
  --text-lock configs/text-minilm-lock.json \
  --output /work/smoke/rcabench-v001 --device cpu --threads 4
```

For D2, use `--dataset aiops2022_pre` and the corresponding raw data and
manifest. All paths above are examples. Use separate, non-nested raw and output
directories, and a new output directory for each attempt.

The smoke command uses one permitted training case without root-label access.
It is not a trained B30 model, an accuracy evaluation, or a reproduction of the
paper's results. With an explicit public manifest, this command does not require
the standard-source environment variables.

## Optional Workflows

### Full reference training

The reference CLI runs closed-loop B30 training with seed 42 and the full
objective (`A3`). It queries six batches of five cases and runs 170 epochs:
20 observation-only warm-up epochs, five 20-epoch nonfinal rounds, and a
50-epoch final round.

```bash
python -m vast_v3.cli.train \
  --dataset rcabench --raw-root /data/rcabench \
  --manifest /work/curator/rcabench-v001/public-manifest.json \
  --sealed-labels /work/curator/rcabench-v001/sealed-labels.json \
  --text-directory /models/minilm-pinned \
  --text-lock configs/text-minilm-lock.json \
  --case-cache-root /work/cache/rcabench-v001 \
  --case-cache-max-bytes 107374182400 \
  --run-dir /work/runs/rcabench-b30-v001 --device cuda:0 --threads 4
```

Training still requires the standard-source environment variables, even with
an explicit manifest. Raw, cache, and run directories must be separate and
non-nested. AIOps uses `--dataset aiops2022_pre` and a frozen training-reference
topology, supplied with `--static-topology` or constructed by the reference
procedure. Missing trace measurements remain missing.

Progress is recorded in `status.json` and `events.jsonl`. Resume an ordinary
interruption with the same command plus `--resume`, preserving the original
data, code, configuration, device, and query state. `--max-updates` is a
diagnostic pause, not completion. See the
[full training and recovery guide](docs/inputs-and-runs.md#full-reference-training).

### Cached-subset engineering run

`python -m vast_v3.cli.quickrun --help` describes the `snapshot` and `run`
subcommands. This route requires an existing compatible cache containing
**more than 30 training cases**, the original complete source pins, and a frozen
snapshot of all published cache entries. AIOps also requires its original
static topology. It cannot start from an empty cache.

The 16-epoch schedule is an engineering check, not the 170-epoch full-pool
protocol or an unbiased benchmark estimate. See the
[cached-subset instructions](docs/inputs-and-runs.md#cached-subset-engineering-quickrun)
for required arguments and completion checks.

## Reproducibility Notes

- **Learning terminology.** VAST is an active semi-supervised method. Internal
  `pn` and `U_soft` keys name loss components, not alternative method names.
  Root hypotheses for unlabeled observations are not trusted pseudo-labels.
- **Annotation budget.** B30 counts 30 distinct completely annotated cases,
  including any labels accessed for validation or tuning. Multiple roots in
  one case do not consume multiple case queries.
- **Data splits.** The standard reference CLI preserves D1's 995/427 and D2's
  169/72 training-candidate-pool/test splits. The paper's CFCE/RQ3 splits are
  983/439 and 162/79, respectively. They are different protocols, not
  interchangeable manifests.
- **CFCE and acquisition.** CFCE uses fixed discrete fragment descriptors and
  a frozen pool vocabulary. PropCover instead measures conditional coverage
  through continuous SCOPE-token similarities. Offline CFCE conditions may
  use true types or roots, but deployable acquisition does not access
  unqueried labels.
- **Unseen types.** RQ3 excludes unseen types from preprocessing, vocabulary
  fitting, warm-up, and training, and evaluates their held-out cases without
  PropCover. The internal legacy `lofo` identifier is not a CFCE/RQ3 runner.
- **Evaluation.** Top-k, Avg@5, and MRR use the first correct root in the ranked
  candidate list. They do not measure complete recovery of every root in a
  multi-root case. Compute exact MRR from each case's first correct root rank,
  not only aggregated Top1/3/5 counts.
- **Release scope.** The core provides model, selection, training, and metric
  components, not one-command reproduction of all paper tables. Smoke checks,
  shortened runs, and training-completion flags do not establish localization
  accuracy. Trained demonstrations and full experiment outputs are not bundled.

For input requirements and execution boundaries, see
[Inputs, execution and recovery](docs/inputs-and-runs.md).
Record the exact source commit, input identities, text-model lock, and run
configuration when reporting results.

## License and Citation

VAST retains the [MIT license](LICENSE). Dataset and model-weight licenses and
redistribution terms apply independently. Cite the software using
[CITATION.cff](CITATION.cff) and record the exact commit used. The earlier
CVAE/OSER implementation remains in
[Git history](https://github.com/molujia/VAST/tree/b1ae14126100743b03e83d3dd4b83d958a80c389).
See [source provenance and migration](docs/provenance.md) for the release history
and compatibility boundaries.
