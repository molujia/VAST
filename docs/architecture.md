# VAST architecture contract

This document describes the implemented interfaces, not completed performance
experiments. `src/vast_v3/runtime.py` and `training/schedule.py` define executable
settings; the default model is built by `runtime.build_objectives`.

## Observations and supervision

`ObservableCase` contains a complete candidate catalog, reference/incident
windows, native numeric series, log records and typed relations. Metric series
belong to entities; trace numeric series belong to directed relation owners.
Missing values retain masks. Context-only nodes can support propagation without
becoming ranking candidates. Entity IDs associate records and break score ties;
they are not learned identity features.

`CuratedLabels` is a separate curator-side artifact. The training loop receives
a budgeted query store, not the evaluation store. Each query reveals the complete
joint root set for one incident: positives are roots, negatives are the other
candidates, and unqueried candidates have unknown root status. RCABench uses
the declared maximum of two jointly present roots. AIOps host roots are not
expanded into their hosted services.

## Front ends

Numeric streams use separate metric/trace CNNs, each with two 64-wide layers
(kernel 3, GELU), operating on value, availability and elapsed-seconds channels.
Logs use frozen 384-dimensional MiniLM vectors plus frequency and relative time,
then a trainable 64-wide projection. The two-layer typed GCN uses entity
categories and relation degrees, not hidden fault types.

Modality embeddings are associated by entity and time only after native
encoding. Projection and LayerNorm form a 64-dimensional state. Local symptom
states are retained before cross-entity mixing. A separate owner/time-indexed
trace table preserves individual edge numerics after node summary pooling.
The reference median/MAD scale falls back to IQR, then the same permitted
reference pool's population standard deviation, and finally 1 for constant
channels. Unsupported channels remain unavailable.

Masked recovery removes targets and their derived features before re-encoding.
Log recovery may use visible same-entity logs within the configured radius
(default ±2 grid steps); this does not create observations in a masked bin or
change ordinary acquisition/ranking inputs. Inherited AIOps topology is a
structural prior, not observed latency or proof of current propagation.

## SCOPE and PropCover

Connected fragments contain one to three relation edges. The exact direct
sampler counts the admissible space and samples without enumerating it in full;
it retains structure-family quotas and versioned sampling provenance. The
default cap is 128 fragments per incident.

SCOPE combines anonymous structure, local symptoms, directed edge slots,
ordered-path signatures and physical delays. At a convergence, complete ordered
paths are encoded separately while shared-node structure is retained. Unknown
order is not forced into a causal sequence. Candidate marks are applied before
canonicalization; both marked and unmarked tokens share all encoder parameters.

For acquisition, let each reference fragment occurrence be `o`, and let
`c(o,i)` be its maximum `(1 + cosine)/2` similarity to a token in case `i`.
For already queried cases `L` and a proposed batch `S`, PropCover maximizes
the weighted increase of `max c(o,i)` over `i in L union S`, relative to `L`.
Weights balance family, case and occurrence contributions. The unit selected
and charged is a complete incident, not a fragment. Empty inventories yield
zero gain; seeded label-blind tie priority fills the remaining feasible budget.
This is continuous conditional coverage, not equal allocation to ground-truth
fault categories or hard cluster centroids.

The model re-encodes queried and unqueried cases at each acquisition boundary.
With B30 and batches of five, the loop is query five → train → refresh, repeated
six times. Labels are revealed only after selection. Ground-truth propagation
annotations are not needed by PropCover.

## Learning and inference

Candidate-marked fragment attention and local symptoms feed a shared ranker.
The joint objective includes queried-case PN ranking, masked observation
recovery, labeled source constraints and unqueried soft source competition.
RSC compares source hypotheses through constrained transport and reconstruction;
lower reconstruction error is a learning signal, not proof of a true cause.

Source hypotheses respect the configured maximum root cardinality. No hidden
root set is injected into an unqueried hypothesis library. PN, source and
observation terms use independent eligible case streams with explicit empty-set
handling. The U term is disabled for the first five epochs within a query round,
then ramps for five epochs. LOFO uses no acquisition and no U cases; its full
experiment orchestration is not distributed here.

Inference uses only the frozen observation encoder and candidate ranker. It
does not perform RSC source search, read root labels, fit on test telemetry or
update parameters. Every catalog candidate is ranked, including candidates
with missing observations. Full rankings support exact MRR; cumulative Top1–5
counts support Hit@k, TOP135, Avg@5 and MRR bounds without claiming an exact MRR.
