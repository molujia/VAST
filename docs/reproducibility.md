# Reproducing the released B method

[Back to README](../README.md) · [Evaluation and interpretation](evaluation.md)

This guide documents the fixed version 1.0.0 runtime. The supplementary multi-seed, ablation, and strict LOFO experiments used a separate research adapter. They are evidence about the method, not additional modes of `scripts/run_vast.py`.

## Input contract

The [runtime example](../configs/runtime.example.json) is a path template. It contains no benchmark inputs. Use the original prepared feature bundles and frozen plans when reproducing the reference results.

### Feature bundle

Each `feature_dir` contains three files:

| File | Contract |
|---|---|
| `windows.csv` | Unique `window_id`; `dataset`; `window_kind` (`fault` or `normal`); `start_ts`; `end_ts`; `positive_ids`; and historical window metadata. Multiple true roots are separated by semicolons. Normal windows may have no roots. |
| `entity_features.csv` | Unique `(window_id, entity_id)` rows, the complete candidate catalog, historical entity metadata, and all original feature columns. |
| `metadata.json` | Ordered `all_feature_columns`. Names and order are part of the feature identity. |

`metadata_json` in the window table provides the authorized fault type through `fault_type` or a supported historical field. The runtime removes labels before historical feature preparation and restores only the labels allowed for supervision. Test roots are retained separately for scoring.

The label boundary does not establish how upstream features were originally fitted. Ordinary RCABench features have a vocabulary path using all fault windows and per-incident event statistics. See [ordinary versus strict preprocessing](evaluation.md#strict-leave-one-fault-type-out-study).

Use dataset ID `rcabench` or `aiops2022_pre`. The latter uses the historical bundle alias `hd1`. Missing-modality masks and defaults for a few state proxies do not authorize removing historical feature columns or test candidates. The released ranking features and candidate inventory must remain complete.

### Independent split manifest

The external split JSON has two arrays, `outer_train_case_ids` and `outer_test_case_ids`, containing the frozen outer fault populations. The runtime reconstructs the historical chronological split and checks exact membership, counts, and disjointness against this independent manifest.

The reference split uses 70% outer training and 30% outer testing, applied separately to fault and normal windows. Inner validation ratio 0.2 is retained in split metadata; this recipe does not subtract another 20% from the declared supervision population.

| Dataset | Outer training faults | Query labels | Oracle labels | Test faults | Query / Oracle posterior variants |
|---|---:|---:|---:|---:|---:|
| RCABench | 995 | 30 | 995 | 427 | 240 / 7,960 |
| AIOps22-pre | 169 | 30 | 169 | 72 | 240 / 1,352 |

Training uses `fault_only`, with root propagation disabled. Normal windows can contribute to historical baselines but do not become additional fault supervision. Query acquisition occurs only in the outer training fault pool. Oracle-full uses every outer training fault label and does not call the budgeted selector.

### Acquisition and supervision are separate plans

The Query-only `query_plan` follows [fixed_active_learning.plan_contract](../src/fixed_active_learning/plan_contract.py). It carries 30 unique ordered case IDs, dataset identity, seed 42, the HDBSCAN/center choices, geometry/partition/representation hashes, selection records, `selected_case_ids`, `plan_sha256`, and `plan_id`. The runtime validates its semantic hash and training-pool membership. An ID list with fabricated hashes cannot replace the frozen plan.

`supervision_plans` is an optional mapping from `query_only` and/or `oracle_full` to historical `QueryPlan` JSON files. These are different from the acquisition plan. Historical fields include:

```text
dataset, normal_cluster_id, window_clusters,
queried_window_ids, queried_roles, queried_labels,
pseudo_labels, pseudo_confidence, metadata
```

The pseudo-label maps must be empty. The runtime obtains roots from authorized windows rather than trusting prefilled root labels. Query supervision must match the selected order; Oracle supervision must cover the complete outer training fault population.

Without an explicit supervision plan, Query uses acquisition order and Oracle uses the outer training order in the feature table. The latter can differ from the reference training order. **Population equality alone is insufficient for exact replay:** ordering determines which posterior random draws are assigned to which incident. Preserve each stage's actual order rather than sorting all IDs into one common order.

### Regenerating an acquisition plan

If the original frozen acquisition inputs are available:

```bash
python scripts/select_queries.py \
  --config /srv/vast-inputs/fixed-al/configs/rcabench.json \
  --feature-root /srv/vast-inputs/features \
  --output /srv/vast-inputs/rcabench-center-seed42.json \
  --verify-reference
```

This requires an external `rcl-fixed-active-learning-config/v1` configuration, unlabeled candidate pool, eligible feature inventory, and representation reference hashes. `--verify-reference` additionally requires the original reference query plan. Consult [config.py](../src/fixed_active_learning/config.py) for field and path resolution. The historical config's `active_learning_seeds` is `[41, 42, 43]`; this release wrapper executes only seed 42. It does not generate missing feature bundles or reference hashes.

## Fixed scientific configuration

[default_config.json](../src/vast/default_config.json) is the authoritative scientific configuration. The runtime JSON configures infrastructure, not model selection.

| Component | Fixed setting |
|---|---|
| Acquisition representation | Multimodal `global_pca_dim32` |
| Clustering / selection | HDBSCAN leaf, Euclidean distance, medoid center, `allow_single_cluster=false`, center selection; no DBSCAN fallback |
| RCABench cluster sizes | `min_cluster_size=5`, `min_samples=5` |
| AIOps22-pre cluster sizes | `min_cluster_size=6`, `min_samples=2` |
| Query budget / seed | 30 / 42 |
| Candidate state widths | Mechanism 11, propagation 6, context 10; missing-modality masks |
| CVAE architecture | Three 16-dimensional factors, hidden width 128, SiLU, LayerNorm |
| CVAE optimization | AdamW; learning rate 0.001; weight decay 0.0001; batch size at most 64; 150 updates; gradient norm clip 5.0 |
| CVAE objectives | Masked reconstruction 1.0; target context 0.75; cycle consistency 0.75; KL ramps over 50 updates to 0.0125 |
| Mechanism supervision | Triplet weight 0.1, margin 0.5; authorized root candidates only |
| Posterior variants | Eight per labeled parent; combined directional pair mass 0.25 of the real parent |
| Base ranker | Historical pairwise `LogisticRegression`, `liblinear` |
| Internal OSER | `oser-p02`; frozen 32-dimensional backbone; trainable residual/gate; Adam at 0.01 for 30 steps |
| OSER episodes | Meta weight 0.5; one inner step at 0.05; real outer-query incidents |
| OSER correction | Residual cap 0.05 score units; gate threshold 0.5; no correction when evidence is missing |

### Posterior variation and scaling

For each candidate, training draws variants using:

```text
z = mu + epsilon * exp(0.5 * clip(logvar, -12, 8))
epsilon ~ Normal(0, I), NumPy seed 42
```

The real incident retains its posterior mean. Every variant keeps the historical features, entity identities, and root labels unchanged. The real directional positive/negative pair weights are preserved. All descendants of a parent together receive 0.25 times that parent's real pair mass, independent of the number of variants.

Historical and latent blocks have separate scalers. Both are fitted on real supervised mean-feature pair differences, without refitting on variants. Inference uses the posterior mean only.

The complete unlabeled outer training fault pool fits the CVAE in formal Query and Oracle runs. Mechanism supervision uses only the permitted labels. OSER episodes isolate a held-out real parent and all its descendants; descendants enter only permitted support sets, and outer queries are real incidents. This internal isolation is a training regularizer, not an outer LOFO evaluation.

### Fields and proxies that need care

The frozen configuration retains some historical fields whose names should not be interpreted as active B behavior:

- `mechanism_consistency_weight=1.5` is not consumed by the current training objective.
- `sampling_radius=0.5` and `mechanism_sampling_policy` do not truncate the Gaussian sampling above.
- `top135_absolute_decline_tolerance` belongs to the original comparison report. The B runner neither loads H automatically nor selects a method using that tolerance.
- The OSER cap bounds a score correction, not the change in Hit@k, TOP135, or MRR.

Some observable states are feature proxies: duration uses active timestamp counts, earliest anomaly uses a within-window rank mapping, and topology depth uses indegree. These are not measured physical time, propagation distance, or causal paths. The mappings are implemented in `materialize_historical_states` in [vae_snapshot_cvae_stage.py](../src/rcl_study/vae_snapshot_cvae_stage.py).

## Execution and recovery

The [README workflow](../README.md#running-vast) is the supported public path. Each requested dataset/regime produces a B unit. `unit` is an internal controller subcommand.

`--prepare-only` creates the run manifest and committed input stage. Continue with the same mode and `--resume`; an existing output root is not a fresh run. Use distinct roots for smoke and formal execution.

Smoke takes three supervised and three test incidents and runs five CVAE updates. Query retains the full unlabeled training fit pool; Oracle's fit pool shrinks with its smoke supervision. OSER keeps its fixed recipe and may record a fallback if tiny groups are insufficient. Formal reference runs used 150 CVAE updates and 30 OSER updates in each of four units, with no fallback.

One runner owns an output directory at a time. The release allows up to two unit controllers and one heavy CVAE/encoding worker concurrently. It is not an eight-GPU experiment scheduler. Ranker and OSER fits are independent across settings.

The runtime records inputs, configuration, code, and environment identity in the manifest. CVAE fitting, latent encoding, base fitting, OSER fitting, prediction, and evaluation have hashed stage receipts. Compatible completed stages are reused; a stage whose outputs were written before a receipt interruption may be recovered after validation. Resume rejects identity changes or corrupt committed inputs. Preserve the failed run for diagnosis instead of deleting successful models to bypass a check.

`report` requires every requested unit to be complete. It validates result receipts, full candidate rankings, true roots, and metric calculations before writing `run.done.json`. It launches no training. Inspect the referenced result artifact for detailed rankings and the `internal_OSER` diagnostics; do not infer completion only from a progress counter or a checkpoint file.

| Symptom | What to inspect |
|---|---|
| Existing output refused | Confirm the original runtime and mode, then use `--resume` if continuing it. |
| Resume identity mismatch | Compare the recorded input, source, environment, and config identities; restore the intended identity or use a separate output for an intentionally different run. |
| Query-plan validation error | Check the full plan schema, semantic hash, ordered selection, dataset, seed, and outer training membership. |
| Source-integrity error | Run the source inventory check; inspect changes to copied scientific files. |
| No new heartbeat | Check whether the runner/worker still exists and whether unit logs advance. A killed process can leave stale JSON. |
| OSER fallback in smoke | Read its diagnostic reason. A tiny-group fallback is not a formal benchmark result. |

Archived experimental outputs have their own stage identities. The public runtime does not silently adopt them as a newly created run's cache. Use reference replay when validating archived final weights.

## Environment and numerical replay

The package controller requirements are in [pyproject.toml](../pyproject.toml). The reference worker uses Python 3.8.18, NumPy 1.21.5, and PyTorch 2.4.0+cu121; it does not need the controller's HDBSCAN dependencies. The source runner passes the checkout through `PYTHONPATH`. Full execution relies on Linux `fcntl`, so Windows is suitable for reading and pure source checks rather than the training runner.

The manifest records interpreter and numerical-library identities, Torch/CUDA, and thread settings. The reference release left `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` unset; worker Torch intra-op and inter-op thread counts were both 32. Unset variables do not guarantee the same BLAS configuration on another installation. Match the recorded libraries and effective thread settings for exact replay. CPU execution and different hardware/library versions need their own numerical assessment; bitwise portability is not promised.

The installation commands provide a working dependency recipe, not a full environment lock. The [README version table](../README.md#pytorch-worker) records the versions checked for the release. Avoid changing an established experiment environment in place just to install this checkout.

## Reference-model replay

With trusted, completed reference B artifacts and all their external dependencies still accessible:

```bash
python tools/validate_reference.py \
  --reference-run /srv/vast-reference/vae-snapshot-rebase-seed42-20260908-01 \
  --output-root /srv/vast-checks/reference-replay
```

The tool rebuilds historical input frames, checks CVAE scientific requests, re-encodes candidates with saved weights, scores the saved base ranker, applies saved OSER, and compares complete rankings and metrics. It requires the referenced inputs, historical supervision plans, shared stages, and models. Only load trusted pickle/checkpoint artifacts.

The September 9 release validation obtained zero latent/base/final score differences in all four formal units, with exact rankings and metrics, without retraining a formal model. The separate synthetic integration test checks training, CLI behavior, compatible resume, and rejection of changed inputs. These checks have different scopes. Detailed records and the 115-file copied-source inventory are in [provenance](provenance/README.md).
