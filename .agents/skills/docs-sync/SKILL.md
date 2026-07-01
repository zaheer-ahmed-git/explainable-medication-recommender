---
name: docs-sync
description: Synchronize project documentation after changes to commands, architecture, repository status, schemas, workflows, evaluation, or research claims.
---

# Documentation Synchronization

## Steps

1. Read the changed implementation or configuration before editing docs.
2. Identify the canonical documents affected:
   - `README.md` for current status and entry points;
   - `Documentation/CalculcoSetup.md` for Calculco paths, storage, and OAR;
   - `ARCHITECTURE.md` for boundaries and data flow;
   - `Documentation/DataFoundationRoadmap.md` for milestone status;
   - `TESTING.md` for checks;
   - `WORKFLOWS.md` for procedures;
   - `AGENT-MEMORY.md` for stable facts and pitfalls;
   - `CHANGELOG.md` for notable changes.
3. Update the smallest useful set of documents.
4. Mark capabilities as planned, implemented, evaluated, or deprecated.
5. Verify commands, paths, environment variables, links, dates, dataset versions,
   and terminology. Prefer `$PROJECT_HOME` / `$DATASET_ROOT` over hard-coded
   NFS paths in docs unless illustrating Calculco setup.
6. Remove stale duplication by linking to the canonical source.
7. Inspect the documentation diff for unsupported clinical or performance
   claims.

## Success Criteria

- Documentation matches the current tree.
- Historical files remain clearly historical.
- Research illustrations are not described as validated clinical results.
- Commands and internal links are consistent.
