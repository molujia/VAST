# Evaluation and interpretation

> **Archived evidence.** This document preserves evaluation records distributed
> with the v1.0.0 implementation. They are not the final VAST manuscript's
> five-seed (40–44) experiment package. The maintained repository scope is the
> method implementation, not the paper's experiment suite. These tables must
> not be substituted for the final paper's reported means.

[Back to README](../README.md) · [Reproduction guide](reproducibility.md) · [Aggregate results](evaluation-summary.json)

This document separates the frozen release scores from subsequent evidence about the method. All values come from completed, accepted experiment reports. The release remains the continuous-latent B recipe; the supplementary study does not select a new default from its ablations.

## Which results can the public runner reproduce?

| Evidence | Protocol | Public availability |
|---|---|---|
| Released B | Ordinary chronological split; acquisition/training seed 42; Query-only and Oracle-full | Implemented by `scripts/run_vast.py`; requires external prepared features and plans. |
| Ordinary multi-seed comparison | Same frozen ordinary features; acquisition seed 42; training seeds 41, 42, 43 | Aggregate results included; separate experiment adapter and case-level artifacts are not included. |
| Strict outer LOFO | Upstream reconstruction and fitting within each allowed training fold; Query-only | Aggregate results included; this is a separate research pipeline, not a release CLI option. |
| Components, acquisition, budgets, diagnostics, and timing | Matched controls and saved models with their specified populations | Selected findings summarized here; no additional release modes are implied. |

The [JSON summary](evaluation-summary.json) preserves full-precision release metrics, ordinary MRR, strict MRR and primary contrasts, and all 12 latency settings. Source-table SHA-256 hashes identify the accepted inputs. The raw tables and case-level evidence are not part of the public checkout; hashes provide identity, not access to those artifacts.

## Metrics and ordinary populations

For each incident, candidates are ordered by descending final `score`, descending `raw_score`, and ascending `entity_id`. With multiple true roots, use the best true-root position `r`:

```text
Hit@k  = mean(r <= k)
MRR    = mean(1 / r)
TOP135 = (Hit@1 + Hit@3 + Hit@5) / 3
```

Metrics are proportions, not percentages. A difference of 0.01 is one percentage point. TOP135 averages three cutoffs; it is not an average over every cutoff from 1 through 5. The final B scores are post-OSER. `pre_OSER_metrics` describe the base B ranker.

The ordinary split has 995 training / 427 test fault incidents for RCABench and 169 / 72 for AIOps22-pre. Query-only labels 30 training incidents in each dataset. Oracle-full labels all outer training faults; it is a supervision condition, not a guaranteed accuracy upper bound. See [population and input order](reproducibility.md#independent-split-manifest).

The fixed seed-42 metrics are retained in the [aggregate JSON](evaluation-summary.json). They reproduce the archived selected release report, including the adverse AIOps results. Against H, B changes TOP135/MRR as follows:

| Dataset | Supervision | H TOP135 | B − H TOP135 | H MRR | B − H MRR |
|---|---|---:|---:|---:|---:|
| RCABench | Query-only | 0.800937 | +0.032787 | 0.775278 | +0.037642 |
| RCABench | Oracle-full | 0.836066 | +0.012490 | 0.804639 | +0.011529 |
| AIOps22-pre | Query-only | 0.824074 | −0.023148 | 0.782568 | −0.016408 |
| AIOps22-pre | Oracle-full | 0.847222 | −0.004630 | 0.823539 | −0.016461 |

The original comparison tolerated an absolute TOP135 decline of at most 0.05. All four settings met that criterion, but meeting a tolerance is not evidence of improvement. It does not constrain individual Hit metrics: AIOps Query Hit@3 declined by 0.069444. The standalone runner does not use this criterion to select a model.

## Ordinary multi-seed study

The matched study fixes the acquisition plan at seed 42 and trains B with seeds 41, 42, and 43. Means and sample standard deviations summarize those three training runs, not three different test sets or acquisition plans.

| Dataset | Supervision | H MRR | B MRR mean | B sample SD | B − H mean |
|---|---|---:|---:|---:|---:|
| RCABench | Query-only | 0.775278 | 0.809552 | 0.004741 | +0.034274 |
| RCABench | Oracle-full | 0.804639 | 0.814236 | 0.001997 | +0.009597 |
| AIOps22-pre | Query-only | 0.782568 | 0.772312 | 0.005675 | −0.010255 |
| AIOps22-pre | Oracle-full | 0.823539 | 0.808237 | 0.007107 | −0.015302 |

H's three observed outputs are identical. It is represented by one effective control, with SD unavailable rather than three independent copies. B has three effective training runs. Shared models reused by multiple experiments do not become new independent observations.

The direction is consistent with the seed-42 release: higher B mean MRR on RCABench and lower B mean MRR on AIOps22-pre. The complete B–H difference combines latent representation, posterior variation, and OSER; it cannot be attributed entirely to generated variants.

## Strict leave-one-fault-type-out study

### Fitting boundary and mathematical differences

The strict study holds out a fault type from the permitted training population and reconstructs upstream inputs from raw sources. Candidate catalogs, graphs, vocabulary, baselines, event definitions, and normalization are fitted only on the allowed training fold. The held-out type is excluded even from the unlabeled CVAE fit pool.

The ordinary RC event extractor uses per-incident sliding-window means/variances, periodic templates, and input/output min-max statistics. For strict evaluation, those statistics are instead fitted on the allowed training fold and frozen for evaluation. This changes the upstream mathematical definition as well as its fitting boundary. Scores on an ordinary matched subset are descriptive references; they do not isolate the causal effect of leakage removal.

Historical test results informed method selection before this follow-up. Strict fitting isolation does not undo that history, so this is not a newly blinded confirmation experiment.

### Populations and arms

All strict arms use Query-only, budget 30, acquisition seed 42, and training seeds 41, 42, and 43. Arms within a fold share the query population.

| Arm | Meaning |
|---|---|
| H | Complete historical pairwise feature/ranker baseline. |
| Bbase | Historical features plus candidate mean latents and posterior variants, before OSER. |
| H+OSER | H with the internal OSER correction, without CVAE augmentation. |
| Bfull | Complete B, including internal OSER. |

The artificial holdout analysis covers **21 types and 419 test incidents**. Three types already absent from ordinary training form a separate **naturally unseen group of eight incidents**. Their models share one training population per seed; they are not three independent sets of fitted models.

Across both groups, the study contains 22 unique fit populations, 66 population/seed groups, 264 unique evaluations, and 288 logical report views. These quantities describe different units and must not be added or treated as independent sample sizes.

### Artificial holdout results

Macro MRR gives each of the 21 types equal weight. Case-weighted MRR gives each of the 419 incidents equal weight. Each reported value is the mean across the three training seeds; SD describes that training-seed variation.

| Arm | Macro MRR, mean ± SD | Case-weighted MRR, mean ± SD |
|---|---:|---:|
| H | 0.334332 ± 0.000000 | 0.352322 ± 0.000000 |
| Bbase | 0.330352 ± 0.008548 | 0.349488 ± 0.004134 |
| H+OSER | 0.454880 ± 0.033565 | 0.524522 ± 0.025131 |
| Bfull | 0.441693 ± 0.040007 | 0.506420 ± 0.030043 |

H's outputs are identical across seeds; its source-table SD is numerical roundoff, displayed as zero. That is not evidence of independent stochastic replications.

The four primary comparisons use macro MRR, paired type clusters and a shared training-seed axis, 10,000 resamples, and statistical seed 42. Confidence intervals are paired 95% bootstrap intervals; p-values use the registered permutation procedure and Holm correction over these four contrasts.

| Primary contrast | Difference | Paired 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| Bfull − H | +0.107361 | [0.041462, 0.172537] | 0.001600 |
| Bbase − H | −0.003981 | [−0.027092, 0.016421] | 0.720228 |
| H+OSER − H | +0.120547 | [0.059618, 0.185148] | 0.000800 |
| Bfull − Bbase | +0.111342 | [0.051187, 0.173105] | 0.001200 |

Full B improves mean MRR over H in 17 artificial types and declines in four. The primary comparisons support an OSER increment in this setting. **H+OSER nevertheless has higher mean MRR than full B**, and Bbase does not improve over H. Thus this experiment does not support a stable extra benefit from CVAE means/variants. Bfull versus H+OSER was not in the primary contrast family; the mean comparison here adds no post hoc significance claim.

### Naturally unseen types and unavailable AIOps inputs

| Arm | Natural macro MRR, mean ± SD | Natural case-weighted MRR, mean ± SD |
|---|---:|---:|
| H | 0.135081 ± 0.000000 | 0.139378 ± 0.000000 |
| Bbase | 0.175391 ± 0.040070 | 0.159399 ± 0.030260 |
| H+OSER | 0.187005 ± 0.004166 | 0.209792 ± 0.003106 |
| Bfull | 0.364267 ± 0.150984 | 0.343334 ± 0.113707 |

Eight incidents and large B seed variation support only a limited descriptive result. This group is kept separate from artificial holdouts.

The required AIOps upstream v1 source and complete May 1 trace were unavailable. Its 180 expected strict result cells are **NA**, not zero accuracy, failed model runs, or completed strict evidence. They provide no basis for a cross-dataset strict-generalization claim.

## Other supplementary findings

The accepted supplementary analysis also studied the following questions. These summaries retain mixed and infeasible outcomes rather than promoting a new recipe.

| Question | Finding and interpretation |
|---|---|
| Which training components help? | OSER's increment is clearer on RCABench. Extra benefit from candidate means or posterior variants is not consistent. Component experiments share models and cannot be counted as independent replications of the full result. |
| Do triplet supervision, meta-learning, and true type groups help? | Their additional advantages are not uniformly supported by the matched controls and interactions. |
| Which modalities and geometric preprocessing matter? | Acquisition-side geometry and training-side feature availability were varied separately. Effects depend on dataset and control. Nine acquisition-modality settings were natively infeasible and remain NA. |
| Is center acquisition consistently best? | No. On AIOps, center minus boundary MRR area under the budget-learning curve over budgets 8–30 is −0.051660, Holm p = 0.012399; the center-versus-random interval crosses zero. |
| What happens at small budgets? | The budget study has 66 valid settings and six infeasible RCABench budget-8 center/boundary settings. Their initial round requires 16 clusters. The corresponding full 8–30 areas and contrasts are NA; they are not filled with zeros, interpolated, or replaced by 16–30 partial areas. |
| Do mechanism examples support uniform benefit? | Diagnostics include helped, harmed, and tied incidents. They explain observed behavior but do not establish physical causal pathways or universal improvement. |

These findings derive from the accepted supplementary summary identified in the JSON's `source_tables`. Detailed case-level analyses, full ablation tables, and research-adapter execution are outside this source release.

## Runtime measurements

### Design and scope

Timing reuses accepted **training-seed-41 models** for two datasets, two supervision regimes, and H/Bbase/Bfull: 12 settings in total. Each setting has three new-process cold starts and 30 warm repetitions per incident across the trials. There are **36 cold starts, 89,820 warm measurements**, and 8,982 unscored warm-up calls. Timing repetitions are not independent trained models.

Measurements used an NVIDIA RTX A6000, an Intel Xeon Gold 5416S host, 16-CPU process affinity, and the original controller/worker environments. Project resource admission reserved 48 CPUs; BLAS/OpenMP reported 64 threads, with Torch intra-op/inter-op at 32. The selected GPU and project admission were controlled, but the whole shared server was not exclusive. Fixed serial method order can retain time-dependent system effects.

The warm clock starts from an already loaded per-incident feature bundle and includes observable-view preparation, serialization/IPC, GPU posterior encoding, latent assembly, historical ranking, optional OSER, and final ordering. Input-file loading/integrity checks, reference verification, and result writing are outside that clock. Raw telemetry extraction is unmeasured. A synchronized encoder call includes CPU normalization and host/device transfers; it is not pure GPU kernel time.

### Warm latency

Values below are **milliseconds per incident**. Quantiles pool all incidents and their 30 repetitions equally within a setting; they are not quantiles of independent new incidents.

| Dataset | Supervision | Arm | Mean | p50 | p95 |
|---|---|---|---:|---:|---:|
| AIOps22-pre | Query-only | H | 40.092 | 35.370 | 52.713 |
| AIOps22-pre | Query-only | Bbase | 95.954 | 83.695 | 118.234 |
| AIOps22-pre | Query-only | Bfull | 136.428 | 140.610 | 189.437 |
| AIOps22-pre | Oracle-full | H | 40.946 | 35.461 | 52.432 |
| AIOps22-pre | Oracle-full | Bbase | 94.973 | 82.373 | 117.120 |
| AIOps22-pre | Oracle-full | Bfull | 137.356 | 140.419 | 192.727 |
| RCABench | Query-only | H | 216.147 | 212.997 | 290.110 |
| RCABench | Query-only | Bbase | 311.954 | 301.977 | 380.871 |
| RCABench | Query-only | Bfull | 422.894 | 421.323 | 510.419 |
| RCABench | Oracle-full | H | 216.253 | 213.178 | 289.233 |
| RCABench | Oracle-full | Bbase | 313.158 | 302.743 | 383.548 |
| RCABench | Oracle-full | Bfull | 422.829 | 422.636 | 504.115 |

Full B costs more online than H in this measurement. Similar Query/Oracle inference time does not imply similar training cost or label requirements.

### Cold start and numerical checks

Cold time runs from a new process to its first complete ranking, including dependencies, input reading/integrity checks, model loading, and B worker startup. The OS page cache was not flushed. With only three observations per setting, cold quantiles are descriptive.

| Dataset | Supervision | H mean seconds | Bbase mean seconds | Bfull mean seconds |
|---|---|---:|---:|---:|
| AIOps22-pre | Query-only | 1.422 | 36.500 | 36.540 |
| AIOps22-pre | Oracle-full | 1.428 | 36.322 | 35.786 |
| RCABench | Query-only | 1.682 | 36.380 | 37.732 |
| RCABench | Oracle-full | 2.193 | 37.979 | 37.685 |

Natural single-incident inference can change floating-point near-tie ordering relative to the reference batch shape. The timing study requires **every true-root position and every original MRR/Hit metric to remain unchanged**. Only non-root inversions with original score gaps at most `2e-7` are permitted, with score/raw-score error at most `1e-7` (`1e-12` for H) and posterior tolerance `atol=1e-6, rtol=1e-5`.

There were 653 affected setting/incident combinations across the four RC B settings, corresponding to 19,590 warm records. These are not 653 distinct incidents. All root positions and metrics passed the invariance checks. The aggregate JSON retains per-setting counts and maximum numerical errors. Warm records retained complete candidate order and numerical audit results, but not every candidate's full floating-point score vector for every repetition; the audit should not be described as independent recalculation of all warm score vectors.

### Training cost and scale

No extra training was run for timing. Existing training-stage logs have different H/B stage boundaries and were produced under concurrent workloads. They do not establish a controlled pure-training speed comparison. Complete raw-input-to-trained-model cost, acquisition/raw extraction cost, and independent queue/I/O breakdowns are unavailable.

A separate synthetic scale study covers 48 component/size settings with candidate counts 16, 32, 54, 64, 128, and 256, using Gaussian synthetic inputs and saved models. It measures no root cause accuracy. The historical pairwise ranker materializes `N(N−1)` directed non-self comparisons, so the comparison matrix grows quadratically in candidate count and linearly in feature width. Finite measured curves include fixed overhead and numerical-library thresholds. Synthetic component times should neither be added to reconstruct full B latency nor be presented as real deployment measurements.

## Evidence identity

The scientific release revision is [`0df7f8474d3dd0fdbbf72f7069bf835c6851741d`](https://github.com/molujia/VAST/commit/0df7f8474d3dd0fdbbf72f7069bf835c6851741d). Documentation updates do not change its model or defaults.

The original release report hashes are:

- Markdown: `433035f720741866d586a162960bad64f092a15dc4934fa8f39a191d550a50c0`
- JSON: `5cd15b073c1033e953a11b28efa93a45650a26ae59a2164e7a1c76d3178d28e5`

[evaluation-summary.json](evaluation-summary.json) identifies the accepted aggregate tables by filename and byte SHA-256. It contains no case identifiers or machine-specific paths. Source files and transformations for the implementation are recorded separately in [source-inventory.json](provenance/source-inventory.json); release validation is described in [provenance/README.md](provenance/README.md).
