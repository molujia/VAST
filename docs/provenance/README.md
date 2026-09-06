# Source provenance

`source-inventory.json` records the mechanically selected implementation closure used by this interim VAST snapshot.

Each `files` entry contains:

- `source_path`: repository-relative path in the authority workspace;
- `destination_path`: repository-relative VAST path;
- `source_sha256`: SHA-256 of the authority source bytes;
- `destination_sha256`: SHA-256 of the committed snapshot bytes;
- `role`: the file's role in the selected method;
- `transforms`, when present: deterministic portability-only transformations such as LF normalization, machine-path placeholder replacement, or `src/` entrypoint adaptation.

The manifest names the fixed active-learning authority, integration change, and pairwise backend. It intentionally contains no source-machine root, data path, case inventory, query-plan instance, result, or checkpoint.

To reproduce the snapshot from an authority workspace located immediately above the VAST checkout:

```bash
python tools/snapshot_sources.py --source-root .. --destination-root .
```

To verify a checkout without access to the authority workspace:

```bash
python tools/snapshot_sources.py --destination-root . --verify-only
```

Verification recomputes the inventory identity and every recorded destination hash. Any missing or modified copied file causes a non-zero exit.

