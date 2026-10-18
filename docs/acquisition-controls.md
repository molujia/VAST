# Acquisition controls

`acquisition/comparison_selectors.py` contains observation-only comparison
selectors. They propose complete case IDs; callers must still charge each
revealed case to the query ledger. Their presence does not mean the corresponding
formal comparison experiments have been run.

- `TimeSpread`: normalize observable incident start times within the permitted
  pool, initialize with a one-dimensional temporal medoid when no labeled
  center exists, then select farthest-first relative to existing centers.
- `SymptomCover`: continuous facility-location coverage of local node and
  directed-owner edge symptom tokens, restricted by token kind and balanced
  across kind, case and occurrence. It does not use topology/SCOPE outputs,
  hidden fault types, root labels or ranker predictions.
- `PooledMedoid`: pool each case's SCOPE tokens into a normalized case vector,
  then use conditional k-medoids with the already labeled cases fixed and
  bounded single swaps. This is a controlled case-level alternative, not the
  main fragment-coverage selector.
- The discrete structural-motif coverage control uses anonymous typed
  observable fragment structure. Motif labels are derived from observations,
  not from ground-truth propagation analysis or hidden root/fault labels.

Each selector fixes a label-independent seeded tie priority, excludes already
queried cases and respects the remaining complete-case budget. Representations
within a selection call must share the same encoder version. The main method
remains `acquisition/propcover.py`; these controls are not silently substituted
into its training CLI.
