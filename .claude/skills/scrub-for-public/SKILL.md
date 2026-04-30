---
name: scrub-for-public
description: Scrub a workflow JSON, doc, or example file of private identifiers before open-sourcing. Loads patterns + replacements from .claude/privacy_patterns.local.json (gitignored). Strips embedded base64 image previews from workflow JSON nodes. Prints a diff for user review; writes only when user explicitly confirms.
disable-model-invocation: true
---

<!-- path-privacy: skip-file — this skill's prose documents the path shapes it scrubs -->


Last updated: 2026-04-30

# Scrub for Public

Transform a tracked file or workflow JSON into a publishable version following
the "Privacy" rules in global CLAUDE.md.

## What gets replaced

The literal pattern → replacement table lives in
`.claude/privacy_patterns.local.json` under `scrub_replacements` (gitignored
so the patterns themselves don't get committed). Load that file at start.

Also strip from workflow JSON:

| Pattern | Replacement |
|---------|-------------|
| `"image_url": "data:image/...` (workflow widget previews) | stripped to `""` |
| `"base64": "..."` inside workflow nodes | stripped to `""` |

If the config file is missing, warn the user and fall back to a no-op (don't
guess at patterns — incorrect scrubbing is worse than none).

## Procedure

1. **Load patterns.** Parse `.claude/privacy_patterns.local.json`. The
   `scrub_replacements` array gives ordered (longest-first) literal
   substitutions; apply them in order so `/home/<user>/ComfyUI/models/`
   matches before the more general `/home/<user>/`.

2. **Accept input.** Take a file path as argument. If it's a workflow JSON,
   use orjson to parse; otherwise treat as text.

3. **Preflight.** Run `git status -- <file>` to confirm it's tracked and not
   in `.gitignore` (gitignored files don't need scrubbing for publish purposes).

4. **Apply replacements.** For text files, literal-replace in-memory in the
   declared order. For workflow JSON, parse with orjson and:
   - Walk all nodes' `widgets_values`, looking for string values matching
     the path patterns.
   - Detect base64 previews in widget tuples (typically `"data:image/..."`
     or long strings with only `[A-Za-z0-9+/=]` past length ~200).
   - Preserve structural fields (node IDs, link IDs, types, coordinates).

5. **Cross-check against `internal/`.** For workflow JSON, grep
   `internal/scratch/` and `internal/` for matching creative prompt strings
   that might have been copy-pasted into the public workflow. Flag any hits
   even if they aren't file paths.

6. **Print a diff.** Show user the before/after using `diff -u <orig>
   <scrubbed>` or equivalent in-memory. Do NOT overwrite yet.

7. **Confirm and write.** Only after user says "write" or "yes", produce the
   scrubbed file. For workflow JSON, write to a sibling path with `_public`
   or `_scrubbed` suffix unless user specifies overwrite.

## Rules

- Use `orjson` for JSON (project convention — never stdlib `json`).
- Real-name copyright lines in LICENSE are not leaks when the bare-username
  pattern uses `\b<user>\b` (word boundary handles "First Last" form). Leave
  LICENSE alone.
- Package names in `pyproject.toml` are fine (they're upstream identifiers).
- If a creative prompt contains a subject name/description that's OK to share
  publicly, leave it. If it references real venues, filenames, or
  personally-identifying details, replace with `<subject>` or
  `<creative_prompt>`.
- Preserve JSON formatting: `orjson.dumps(data, option=orjson.OPT_INDENT_2).decode() + "\n"`.

## Example invocation

```
/scrub-for-public example_workflows/audio-loop-music-video_latent.json

→ reports 4 private-path hits, 2 base64 previews stripped, 0 creative-prompt leaks
→ prints diff
→ user confirms "write"
→ writes example_workflows/audio-loop-music-video_latent_public.json
```

## Pair with privacy-scrubber agent

- `privacy-scrubber` (subagent) = AUDIT-only, lists leaks.
- `scrub-for-public` (this skill) = FIX, with user confirmation.

Run the subagent first to see scope; run this skill to apply fixes.
