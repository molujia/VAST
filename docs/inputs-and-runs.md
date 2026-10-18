# Inputs, execution and recovery

## Required assets

Real execution requires separately authorized assets; this repository supplies
neither benchmark telemetry nor weights/labels. Paths below are examples.

1. The original frozen windows CSV, candidate catalog CSV, training-pool JSON
   and test-case identity JSONL for the selected dataset. Set the four relevant
   variables from `configs/standard-sources.example.env` in the process
   environment. The program does not load that example file automatically.
2. The raw RCABench or AIOps22-pre directory in the adapter's expected layout.
3. An offline MiniLM snapshot matching `configs/text-minilm-lock.json`. Obtain
   it separately under its upstream terms; never substitute a different model
   while retaining the old lock.
4. For training, the independently generated public manifest and sealed label
   artifact, with optional complete source hashes and frozen AIOps topology.

The four metadata roles are:

| Variable suffix | Input and validation |
| --- | --- |
| `_WINDOWS` | Original `windows.csv`; byte checksum and interval-source counts |
| `_CATALOG` | `entity_features.csv`; only entity IDs/categories, with frozen ID digest and counts |
| `_TRAIN_POOL` | Original label-free candidate-pool JSON; byte checksum, dataset, order and count |
| `_TEST_IDS` | Original historical per-case JSONL; only `case_id` is projected, with ordered ID digest |

The historical test-identity source may contain predictions/labels, but those
fields are not used to construct model inputs or select cases. Keep that source
on the curator side and never publish it as observation data. The standard
function continues to enforce RCABench 995/427 and AIOps22-pre 169/72 train/test
counts. Renaming a different split to those filenames does not bypass checks.

For example, on Linux set `export VAST_RCABENCH_WINDOWS=/path/to/windows.csv`
and the other three RCABench variables; on PowerShell use
`$env:VAST_RCABENCH_WINDOWS = 'C:/path/to/windows.csv'`. Missing variables fail
with an explicit configuration error instead of probing a private server path.

## Curator preparation

The curator reads original annotations, verifies complete root sets and writes
public and sealed artifacts separately. It is not a learner or a model query.
Use a new output directory outside the raw data tree.

```bash
python -m vast_v3.cli.curate \
  --dataset rcabench --raw-root /data/rcabench \
  --output-dir /work/curator/rcabench-v001
```

Keep `sealed-labels.json` private. The training loop accesses it through a
budgeted query store; inference must never read it. The completed
`curator-summary.json` pins the artifacts and counts.

## One-case smoke

```bash
python -m vast_v3.cli.smoke \
  --dataset rcabench --raw-root /data/rcabench \
  --manifest /work/curator/rcabench-v001/public-manifest.json \
  --text-directory /models/minilm-pinned \
  --text-lock configs/text-minilm-lock.json \
  --output /work/smoke/rcabench-v001 --device cpu --threads 4
```

An explicit public manifest makes this smoke independent of the standard-source
environment variables. The default case is selected from training IDs without
looking at outcomes. This checks observation/model execution, not TOP1 accuracy;
its checkpoint is not a trained B30 demonstration model.

## Full reference training

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

The standard-source environment variables are still required: the training CLI
independently checks that the supplied public manifest matches the frozen
population. For AIOps use `--dataset aiops2022_pre` and its assets. An optional
`--static-topology` identifies a previously frozen full-training-reference prior;
otherwise the approved reference procedure builds it once. Missing trace
latencies stay missing.

The CLI fixes seed 42, A3, closed-loop B30 and 170 epochs: 20 warm-up, five
20-epoch nonfinal rounds and 50 final-round epochs. Each epoch has
`ceil(training_pool_size / 4)` update slots. This can be slow on large pools.
It is a reference preparation entry point, not a configurable paper sweep.
Library-level `RuntimeConfig` and `Trainer` expose the implemented regimes.

Raw, observation-cache and run paths must be separate and non-nested. Source
hashing and preprocessing are preparation, not optimizer updates. The cache
quota controls storage only; it does not permit dropping candidates or samples.

## Cached-subset engineering quickrun

`python -m vast_v3.cli.quickrun --help` documents `snapshot` and `run`.
Both require the standard inputs above, the complete original `--source-pins`,
an existing compatible `--case-cache-root`, and a new `--snapshot` path outside
raw/cache/run directories. AIOps also requires the original `--static-topology`.

First run `snapshot`, then `run` with the same arguments and snapshot path.
Only complete compatible training-case caches are included. Missing/corrupt
caches are errors, not permission to reparse an uncached full pool. A newly
created empty cache cannot be used for this shortcut.

The engineering schedule is 16 epochs: warm-up 1, the first five rounds 1 each,
and the final round 10. B30 and six batches of five remain unchanged. The
original U-disable/ramp rules are retained, so U is active only in the final
round. Cache availability can bias the subset; results are not unbiased
benchmark estimates.

## Progress and completion

`status.json` and `events.jsonl` distinguish preparation, eligibility,
acquisition, training and errors. Checkpoint progress and the paid query ledger
record recovery state. A report of five queried cases means the first active
learning batch, not a slow annotation simulator.

Resume an ordinary interruption with the **same command and `--resume`**.
Data, code, preprocessing, configuration, source pins, text lock, device and
query state must match. Never delete identity or policy-invalid markers to
force resumption. This public portability revision has a new package hash and
must not resume a checkpoint from the unchanged original server release.

`--max-updates N` is a diagnostic pause, not completion. A final checkpoint
alone is insufficient: inspect the complete schedule/query trace and frozen
`final-model.json`. Full preparation requires `eligible_for_demo: true`;
the cached-subset route instead requires `eligible_for_quickstart: true` and
does not establish full reference completion. Neither condition certifies
TOP1, an unbiased performance score or an exported demonstration package.

The production profile keeps metric sequences scalar, batches equal-length
trace sequences, uses FP32 tensors, and disables convolution TF32. Execution
policy is versioned and checked independently of nominal configuration.
