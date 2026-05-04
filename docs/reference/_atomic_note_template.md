# Atomic note template

Last updated: 2026-05-04

This is the shape every new `docs/reference/X.md` follows. The template seeds the LLM-wiki direction (`https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f` adapted for this repo). Existing reference docs predate the template and are not retrofit on a schedule — they migrate when an author touches them. The leading underscore exempts this file from being a wiki node itself; it's a meta-doc for authors.

Policy that motivates this template lives in `.claude/CLAUDE.md` "CLAUDE.md governance". The four-layer model puts canonical "what is X / how does X work" content here in `docs/reference/`, with root `CLAUDE.md` carrying only the rules and pointing here for depth.

## The template

```markdown
# <Title>                       ← entity name, concept name, or pattern name

Last updated: YYYY-MM-DD

## Role                         ← one paragraph: the "if you only read this section" answer
## Key facts                    ← bulleted load-bearing facts; quick-scan
## <Variable middle>            ← topic-specific deep section (see variants below)
## Gotchas                      ← non-obvious failure modes; the "what bites people"
## Migration                    ← if there's an apply script or adoption path (optional)
## Audit + tests                ← rules-as-code that catch violations (optional)
## References                   ← incoming + outgoing pointers (always)
```

**Non-negotiable sections**: Last-updated date, Role, Gotchas, References. The first lets readers know if the doc is current. The second is the 30-second answer for "what is this." The third captures the un-Googleable knowledge — what would have been a postmortem narrative elsewhere is now a paragraph here. The fourth makes the wiki a graph, not a forest.

**Optional sections**: Migration (only if there's an apply script or adoption path), Audit + tests (only if there's mechanical enforcement), Key facts (collapse into Role for short notes).

The variable middle section is where most depth lives. Pick a name that matches the topic — see the four variants below.

## Note-type variants

| Type | When to use | Variable middle section name(s) | Example |
|---|---|---|---|
| **Entity** | Documenting a node, class, or component | "Wiring map", "Mechanism", "Snap rules", "Inputs / outputs" | `frame_planner_reference.md` |
| **Concept** | Documenting a pattern, convention, or abstract mechanism | "How it works", "When it applies", "When it doesn't" | TBD seed: `noise_mask_semantics.md` |
| **Source summary** | Documenting findings from reading upstream code (`coderef/`) | "Findings", "Implications for our code", "What changed" | TBD seed |
| **Synthesis** | Cross-cutting reference comparing multiple approaches | "Decision matrix", "Comparison", "When to use which" | `sampler_reference.md` (existing, predates template) |

**Don't force one type into another.** A `coderef/` analysis that doesn't explicitly tie back to "what changed in our code" is OK — it's still a source summary, just one that hasn't yet had implications. Mark it as such.

## Ingest checklist

When a finding stabilizes enough to earn a wiki node, run this five-step checklist. The order matters: each step makes the next cheaper.

### 1. Write the atomic note

Use this template. Pick the variant that fits. Lead with **Role** so a reader knows in 30 seconds what they're looking at. Save **Gotchas** for last — it's the section you'll add to over time as new failure modes surface.

If the topic is currently scattered across N existing docs, the ingest is *consolidation*, not new writing — pull the canonical fragments together, point the originals at the new node, and let restated content shrink to one-liners.

### 2. Update `docs/README.md`

Two places:

- **Task-first nav** ("I want to do X" / "I want to understand X") — pick the section a reader phrasing the question would land on. If none fits, add a new section header.
- **Alphabetical reference table** under `### `reference/` — one row, one-line description matching peer density (~1 dash or comma, not 5 semicolons).

### 3. Cross-link from related notes

Find 2–4 existing reference docs whose topics touch the new one. Add an entry to *their* References sections pointing to the new note. Don't reach — only cross-link where the connection is real.

This is what makes the wiki a graph. A note with one citation is fragile (single-citation orphan risk); a note with 3+ incoming citations is naturally discoverable.

### 4. Append to the session log

One line in `internal/log/log_YYYY-MM-DD.md` under "What changed" or similar:

> Added `docs/reference/<note>.md` (entity/concept). Consolidates [list of source docs]. Cross-linked from [list].

The log is layer 4 in the four-layer model; it's where ingest events get a timestamp.

### 5. Run the hygiene tests

```bash
uv run --group dev python -m pytest tests/test_claude_md_budget.py tests/test_docs_consistency.py -v --rootdir=.
```

These check:
- Pointer-target integrity (every `docs/X.md` reference resolves)
- Orphan check (your new note is cited from at least one CLAUDE.md / `docs/README.md` / other doc — step 2 + 3 satisfy this)
- Budget invariants on root + subtree CLAUDE.md
- Stale phrases per `validate_docs_consistency.py`

A failure here means an ingest step was skipped or done wrong. Don't ship until green.

## What NOT to do

Anti-patterns that have bitten this codebase before:

- **Don't restate root `CLAUDE.md` rules.** Root is the one canonical home for rules; reference docs are the home for *depth* and *mechanism*. If you find yourself writing a rule sentence that could go in CLAUDE.md, push it there and link back.
- **Don't narrate which session uncovered the finding.** Phrases like "after the 2026-05-04 curation pass" or "as we discovered last week" rot fast. Put session details in `internal/log/`; keep the reference doc timeless.
- **Don't reach for cross-links.** Three real connections beat ten weak ones. The orphan check counts citations from any doc; it doesn't reward density.
- **Don't write a stub-then-fill-later note.** A 50-word note nobody reads is worse than no note. Either you have enough material to fill the template, or the finding hasn't stabilized enough to wiki.
- **Don't depend on `internal/` for public references.** Internal paths are gitignored — public-clone readers can't follow them. Reference an internal doc only when the topic legitimately can't go public (case studies, postmortems, dated bench numbers).
- **Don't paste long code blocks.** A wiki note is a *consolidated* explanation, not a code dump. Show the smallest example that conveys the mechanism; link to source for the full thing.

## Maintenance

When a referenced file moves or gets renamed, the pointer-target test fails on push — fix the reference there. When a topic gets superseded (e.g. a node deprecated), update the **Role** to "Historical: superseded by ..." and keep the doc as breadcrumb rather than deleting it; the cross-links from old material still need a destination.

For reference docs >1000 lines: trim public, archive full per the project rule (`internal/archive/`). The atomic-note shape resists this naturally — if you're at 1000 lines, the note is probably trying to be two notes.

## References

- `.claude/CLAUDE.md` — "CLAUDE.md governance" (the policy this template implements)
- `docs/reference/frame_planner_reference.md` — the seed entity-note example
- `docs/README.md` — the index this template tells you to update
- `tests/test_claude_md_budget.py` — the lint pass step 5 invokes
- Karpathy's LLM-wiki gist: `https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`
