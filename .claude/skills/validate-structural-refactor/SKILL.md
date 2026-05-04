---
name: validate-structural-refactor
description: Dispatch three parallel Explore agents to validate a structural refactor BEFORE writing changes. Use when the user proposes restructuring multiple CLAUDE.md / docs / agent / skill files (signal: ≥ 3 files, ≥ 2 different concerns), or when planning a curation pass on documentation. Cost ~30s of agent time; turns vague "compress this" into a concrete edit list and prevents premature subtree splits.
---

# Validate before structural refactors

Three-agent validation pass. Run it BEFORE you start writing the refactor, not during. The agents work in parallel; total wall-clock cost is ~30s. Catches real issues every time it's been run; established as convention after the 2026-05-04 CLAUDE.md curation pass that introduced this skill.

## When to use

The trigger is a refactor that touches **≥ 3 files spanning ≥ 2 different concerns** (e.g. CLAUDE.md + docs/ + skills, or root CLAUDE.md + multiple subtree CLAUDE.md). Below that threshold, just edit directly. Above it, the validation pays for itself.

Common triggers:
- "Compress this CLAUDE.md" / "split this CLAUDE.md by topic"
- "Move these rules to a subtree"
- "Reorganize the docs/ structure"
- "Promote internal findings to public docs"

## How to run

Dispatch all three agents in a single message (parallel tool calls). Each is an `Explore` subagent. Briefs below — adapt the specifics to the refactor at hand but keep the questions stable.

### Agent 1: Canonical-home map

> "For each topic in the file(s) being restructured, identify the single canonical home. Where no canonical home exists today and the topic only lives in CLAUDE.md, flag as a TODO gap. Output a topic → canonical-home table."

Validates the pointer-discipline plan. Every restated bullet must either have an existing canonical home (so root can compress to a one-line pointer) or earn a new home written as part of the refactor.

### Agent 2: Rule classifier

> "Read the file(s) fully. Bucket every substantive bullet/section into one of: TURN-1 (must-know on turn 1) / SCRIPTS / TESTS / AUTORESEARCH / DOCS / HARNESS / POSTMORTEM (narrative + rule bundled) / RESTATEMENT (already lives elsewhere — point to where) / LEGACY (dated empirical observation). Output a classification table by section. Final summary: counts per category."

Turns "compress this" into a concrete edit list. The RESTATEMENT and LEGACY categories together usually account for 20–40% of bullets and are the easiest wins.

### Agent 3: Subtree density check

> "For each candidate subtree (e.g. scripts/, tests/, internal/X/), confirm it has ≥ 5 substantive subtree-specific rules — both from the file being restructured and from conventions visible in the subtree code itself that aren't in any CLAUDE.md yet. For each: justified yes/no, rule count, recommended outline."

Prevents premature subtree splits. Below the 5-rule threshold, rules stay in root.

## What to do with the output

1. **Topic gaps** from Agent 1 → write the missing canonical doc as part of the refactor (so root can point at it).
2. **RESTATEMENT bullets** from Agent 2 → compress to one-line pointers.
3. **LEGACY bullets** from Agent 2 → demote to `internal/log/` or `internal/analysis/`.
4. **POSTMORTEM bullets** from Agent 2 → split: rule line stays, narrative moves to `internal/postmortem_*.md`.
5. **Justified subtrees** from Agent 3 → create the subtree CLAUDE.md files; move the matching bullets there.
6. **Borderline subtrees** from Agent 3 → keep in root unless there's a load-bearing reason (e.g. silently-load-bearing constants like widget slot indices justify even a marginal subtree CLAUDE.md).

## Why this beats "just start editing"

A first-pass curation guided only by your own read of the file(s) tends to:
- Restate content from canonical docs (you forget what's in `docs/reference/X.md`)
- Compress the wrong bullets (the 600-word one *feels* compressible but actually has no canonical home)
- Premature-split into subtrees that have only 2 unique rules
- Miss orphaned topics that should have a doc home but don't

The three agents read fresh and report independently. Their outputs are usually 80% convergent (high-confidence findings) and 20% divergent (judgment calls you make). The convergent 80% saves an hour of half-right curation; the 20% is fast to adjudicate.

## References

- `.claude/CLAUDE.md` — CLAUDE.md governance (the policy this skill implements)
- `claude-md-management:claude-md-improver` plugin skill — the broader audit pass; this skill is the validation step that should run *before* its update phase on multi-file refactors
- Past application: see `internal/log/log_2026-05-04.md` for the full play-through (gitignored — won't exist on a public clone)
