Last updated: 2026-05-04

# scripts/archive/ — retired apply scripts

Apply scripts whose migrations are baked into the canonical workflow
permanently and whose source/output files are no longer in tree. Kept
as design records of the topology each migration introduced; not for
re-running.

The audit pairs in `scripts/audit_workflows.py` enforce the topology
invariants on the canonical, so accidental regression of a baked-in
migration is caught at audit time.

If you need to resurrect one, move it back to `scripts/` and adapt the
input/output paths to current canonical state.
