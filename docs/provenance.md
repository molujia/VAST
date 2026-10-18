# Source provenance and migration

## Core snapshot

The public core was retrieved from the server's frozen
`quickrun-cached16-r1-testfix1` release on 2026-10-18. Every one of its 149
manifest entries was checked before transfer. Only 145 package/test/packaging
files were transferred; the four server packaging/launch/diagnostic scripts
were not downloaded as part of the source archive.

- Upstream source-manifest SHA256:
  `03d139111158df6fdd29c7119707301dc21df54b61fb4ac4b8a6719a5c686298`
- Upstream Python-package identity:
  `ce216763b5bb4a03bdf7a48ced678a7b2e98100cb2299578b5e45dbf1664a5f3`

The public snapshot keeps the algorithms and compatibility namespace. Its
standard-manifest entry point uses explicit environment-configured paths
instead of private server locations, without changing the frozen checksum,
population or label checks. Packaging, public documentation and citation
metadata are updated to VAST. Consequently the public package has its **own**
code identity; the source release hash above must not be used as its hash.
Active server runs continue with their original release and are not migrated
or restarted by this publication.

## Included and excluded material

The release includes all core package modules, annotation schema/templates,
the public offline-model lock and portable unit/integration tests. Seventeen
server-tool test modules are excluded because they require non-distributed
launchers, diagnostic helpers or private report inventories. They are not
silently reported as passing. Frozen numerical test oracles retain their
original bytes and checksums. New path-configuration tests cover portability.

Data, label ledgers, actual split manifests, pretrained/trained weights,
observation caches, runtime logs, private diagnostics and manuscript materials
are not distributed. The two audited TOP1 examples and full paper experiments
remain incomplete. Historical scores do not establish this version's accuracy.

## Earlier implementation and attribution

The previous public implementation remains available at
[`b1ae141`](https://github.com/molujia/VAST/tree/b1ae14126100743b03e83d3dd4b83d958a80c389),
including its documentation and citation metadata. The migration is a normal
descendant commit, not a history rewrite. Local archives of both the former
local and remote revisions and a complete Git bundle were verified before
replacement.

The new package independently implements the approved SCOPE/PropCover/RSC
design. Historical VAST parsers informed field conventions and mapping rules;
the previous CVAE, OSER and HDBSCAN pipeline is not imported into the new model.
[BARO](https://github.com/phamquiluan/baro) informed the clear separation of
installation, reusable modules, tests and examples; its algorithm/source was
not copied. This repository retains the MIT license already selected by its
maintainers. Sentence Transformer weights and benchmark data retain their own
licenses and access conditions.
