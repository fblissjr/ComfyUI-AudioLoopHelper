# Atomic note template

Last updated: 2026-05-04

## Scope

This template shapes every new `docs/reference/*.md` AND informs the same density discipline applied to all CLAUDE.md files (root, `.claude/`, scripts/, tests/, internal/autoresearch/). All these documents are **LLM-consumer documents** — future Claude sessions read them via the Read tool to understand the codebase. Humans rarely read them. The leading underscore exempts this file from being a wiki node itself.

Density target: **≥ 70% unique-fact bullets / ≤ 30% connector glue**. Existing reference docs predate the template and aren't retrofit on a schedule — they migrate when an author touches them. Policy at `.claude/CLAUDE.md` "CLAUDE.md governance".

## The template

```markdown
# <Title>                       ← entity, concept, or pattern name

Last updated: YYYY-MM-DD

## Role                         ← 30s answer; lead with type/location/cardinality; no metaphor
## Disambiguation               ← optional; X ≠ Y ≠ Z when LLMs would conflate
## Key facts                    ← anchor each with file::function or audit ID
## <Variable middle>            ← topic-specific; tables > prose; ASCII DAGs for wiring
## Failure modes                ← symptom → cause table + bulleted edge cases
## Decision table               ← optional; when ≥3 alternatives exist
## Migration                    ← optional; apply script + revert + dry-run
## Audit + tests                ← optional; F-number + check ID + test::method anchors
## References                   ← always; incoming + outgoing; ≥1 from another reference doc
```

**Non-negotiable**: Last-updated, Role, Failure modes, References. **Optional**: Disambiguation (when LLMs would conflate), Decision table (when ≥3 alternatives), Migration, Audit + tests.

## Public vs private placement

The wiki has a public surface (`docs/reference/`, committed) and a private extension (`internal/reference/`, gitignored — created lazily when first private wiki-grade content lands). Both follow this template. Three rules for which `internal/` subdir a private note lands in:

| Content shape | Goes to |
|---|---|
| Atomic-note-shaped private content (durable knowledge that future Claude looks up but can't ship publicly) | `internal/reference/<topic>.md` |
| Dated empirical observation that gets replaced on re-measurement | `internal/analysis/<topic>.md` |
| Investigation narrative — one-time, the *story* matters | `internal/postmortem_<topic>.md` |
| Public-shareable codebase reference | `docs/reference/<topic>.md` |

**Citation discipline across the boundary:**

- **Public → private**: when a public doc cites `internal/X.md`, mark with `(private clone only)` so a public reader knows the link is dead-ended on their clone. Hygiene test `tests/test_claude_md_budget.py::test_internal_citations_marked` enforces.
- **Public → private (preferred form)**: paraphrase + private link. Public doc carries the public-shareable summary inline; cites internal only for the unscrubbed full version. Pattern: `Mechanism: <inline summary>. Full case study: internal/X.md (private clone only).`
- **Private → public**: cite freely, no marker needed.
- **Private → private**: cite freely.

## Note-type variants

| Type | When to use | Variable middle section names | Example |
|---|---|---|---|
| **Entity** | Documenting a node, class, or component | "Wiring map", "Inputs / outputs", "Mechanism", "Snap rules" | `frame_planner_reference.md`, `audio_loop_controller.md` |
| **Concept** | Documenting a pattern, convention, or abstract mechanism | "How it works", "Setters and strippers" | `noise_mask_semantics.md` |
| **Source summary** | Documenting findings from reading upstream code (`coderef/`) | "Findings", "Implications for our code" | TBD |
| **Synthesis** | Cross-cutting reference comparing multiple approaches | "Decision matrix", "Comparison" | `sampler_reference.md` (predates template) |

**Entity-note specifics:**
- Inputs/outputs tables: include **units** on every numeric column (seconds vs pixel-frames vs latent-frames vs samples). LLMs hallucinate units exactly because we don't write them down.
- ASCII wiring diagram when ≥4 inputs from N sources or ≥4 outputs to M consumers — encodes directed edges compactly.
- Bypass-behavior entry: state explicitly what `mode=4` does (passes inputs to outputs of same TYPE only; any input with no matching-type output dead-ends silently).
- Widget-order spec when widgets are positionally indexed: document the canonical `widgets_values[]` array shape — the rename-without-strip class of bug (F4/F6) hinges on this.

## Reference, don't copy

Code-derived content (function signatures, schema bodies, implementation snippets, audit-check IDs, widget-order constants) lives in **one** place — the code. Wiki notes **cite** it; they do not paste it.

- **Cite by stable anchor**: `nodes.py::AudioLoopController.execute`, not `~line 562`. Line numbers rot; class/function names survive refactors.
- **Add the *why* the code can't tell you**: cite the implementation, then state the constraint or rationale that lives only in commit messages, postmortems, and human reasoning.
- **Editorial content is value-add**: failure modes, disambiguations, decision tables, "when to use X vs Y", cross-cutting patterns — these don't exist in code. Author them.
- **Drift detection**: cited audit IDs are verified by `tests/test_claude_md_budget.py::test_cited_audit_ids_exist` against live `record(...)` calls in `audit_workflows.py`. Cited file paths are verified by the pointer-target test. Don't bypass these by paraphrasing instead of citing.

## Ingest checklist

When a finding stabilizes enough to earn a wiki node:

### 1. Write the atomic note

Pick the variant that fits. Lead with **Role** (type/location/cardinality, no metaphor). Save **Failure modes** for last — it's the section you'll add to over time as new symptoms surface.

If the topic is currently scattered across N existing docs, the ingest is *consolidation*, not new writing — pull the canonical fragments together, point the originals at the new node, and let restated content shrink to one-liners.

### 2. Update `docs/README.md`

Two places:
- **Task-first nav** ("I want to do X" / "I want to understand X") — pick the section a reader phrasing the question would land on.
- **Alphabetical reference table** under `### `reference/` — one row, one-line description matching peer density.

### 3. Cross-link from related notes

Find 2–4 existing reference docs whose topics touch the new one. Add an entry to *their* References sections pointing to the new note. **At least one citation must come from another reference doc**, not just `docs/README.md` — single-citation orphan-prone notes lose discoverability when an index entry rots.

### 4. Append to the session log

One line in `internal/log/log_YYYY-MM-DD.md` (private clone only) under "What changed":

> Added `docs/reference/<note>.md` (entity/concept). Consolidates [list]. Cross-linked from [list].

### 5. Run hygiene tests

```bash
uv run --group dev python -m pytest tests/test_claude_md_budget.py tests/test_docs_consistency.py -v --rootdir=.
```

Failures: pointer-target (cited path doesn't exist), orphan check (not cited from anywhere), audit-ID resolution (cited ID not in `audit_workflows.py`), budget overrun, stale phrases. Don't ship until green.

## What NOT to do

- **Don't paste code.** No schema bodies, function signatures, implementation snippets. Cite by `file::ClassName.method` and add the *why*.
- **Don't write motivational paragraphs.** No "this is the failure class that...", "it's worth noting...", "this is also why...". State the fact; the LLM derives the motivation.
- **Don't use approximate line anchors.** `~line 430` rots. Use `nodes.py::ClassName.method` (function/class anchors survive refactors).
- **Don't omit units.** Every numeric output: seconds vs pixel-frames vs latent-frames vs samples. LLMs hallucinate units.
- **Don't bury disambiguations.** `X ≠ Y` callouts go in a top-level Disambiguation section, not buried inside a failure-mode bullet.
- **Don't write "When this matters" sections.** They're human reading-flow scaffolding. Role + Failure modes + Decision table cover the same ground at higher density.
- **Don't ship single-citation notes.** Need ≥1 incoming citation from another reference doc beyond `docs/README.md` index entries.
- **Don't write a stub-then-fill-later note.** A 50-word note nobody reads is worse than no note. Either you have enough material to fill the template, or the finding hasn't stabilized enough to wiki.
- **Don't depend on `internal/` for public references.** Internal paths are gitignored — public-clone readers can't follow them.
- **Don't paste long code blocks.** A wiki note is *consolidated explanation*, not a code dump. Show the smallest example that conveys the mechanism; cite source for the full thing.
- **Don't narrate which session uncovered the finding.** "After the 2026-05-04 curation pass" rots fast. Session details belong in `internal/log/`.

## Length guidance

- Concept notes: 50–100 lines.
- Entity notes: 80–150 lines (table-heavy ones land at the upper end).
- > 200 lines means it's two notes. Split.
- < 40 lines means the finding hasn't stabilized. Don't ship; capture in "Pending review" instead.

## References

- `.claude/CLAUDE.md` — "CLAUDE.md governance" (the policy this template implements)
- `docs/reference/frame_planner_reference.md` — entity-note seed
- `docs/reference/audio_loop_controller.md` — entity-note seed (table-heavy)
- `docs/reference/noise_mask_semantics.md` — concept-note seed
- `docs/README.md` — the index step 2 tells you to update
- `tests/test_claude_md_budget.py` — the lint pass step 5 invokes
- Karpathy's LLM-wiki gist: `https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`
