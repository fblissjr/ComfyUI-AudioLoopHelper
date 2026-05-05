Last updated: 2026-05-05

# scripts/archive/ — retired apply scripts

Apply scripts kept as design records of the topology each migration introduced. Not for re-running.

**See `./CLAUDE.md` for the per-script inventory** (what each did, why it was originally built, why it's now archived) plus the resurrection procedure if you genuinely need to re-run one.

The audit pairs in `scripts/audit_workflows.py` enforce the topology invariants on the canonical, so accidental regression of a baked-in migration is caught at audit time.
