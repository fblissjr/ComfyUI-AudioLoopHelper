---
name: release-notes
description: Draft the next CHANGELOG.md section from commits since the last release. Reads current version from pyproject.toml and the most recent version header in CHANGELOG.md, then formats the delta in the repo's existing style (semver only, no dates, no major bumps without explicit permission). Prints the draft for user review. Does NOT write.
disable-model-invocation: true
---

Last updated: 2026-04-21

# Release Notes

Draft a CHANGELOG.md entry from recent commits. User-invocable only —
never auto-run (commits should be deliberate).

## Conventions (from global CLAUDE.md)

- Keep `@CHANGELOG.md` updated.
- Semver only. No dates. No major bumps (X.0.0) without explicit user permission.
- Group entries under sections like `### Added`, `### Changed`, `### Fixed`, `### Removed`.
- Top of file has `# Changelog` + format note; below that are version blocks `## 0.4.9`, `## 0.4.8`, ...

## Procedure

1. **Read current state.**

   ```bash
   grep -E '^version = ' pyproject.toml              # current version
   grep -E '^## [0-9]' CHANGELOG.md | head -3        # recent version headers
   ```

2. **Determine target version.**

   - Default: bump PATCH (0.4.9 → 0.4.10).
   - Ask the user before bumping MINOR (0.4.9 → 0.5.0).
   - NEVER propose MAJOR (0.4.9 → 1.0.0) without explicit confirmation.

3. **Get the commit delta.**

   Find the most recent version header in CHANGELOG.md, then:

   ```bash
   git log --oneline --no-decorate <last-release-tag-or-sha>..HEAD
   ```

   If the last release isn't tagged, locate the commit that introduced the
   most recent `## X.Y.Z` header via `git log -p CHANGELOG.md | grep -n '^+## '` and scope from there.

4. **Inspect the diffs** for commits without clear messages:

   ```bash
   git log --stat <range>
   git show <sha>  # for individual commits
   ```

5. **Draft the entry.** Group commits into semver sections. Example:

   ```markdown
   ## 0.4.10

   ### Added
   - `ltx-constraints-auditor` subagent — reviews workflow changes against CLAUDE.md critical constraints.
   - `privacy-scrubber` subagent for pre-commit leak audits.

   ### Changed
   - `AudioLoopController` now derives stride from integer-latent counts rather than `window - overlap` seconds. Eliminates ~0.04s/iter drift at overlap=2.

   ### Fixed
   - STG workflow: bumped `GuiderParameters.cfg` from 1 to 2 to dodge guider bug.
   ```

   Skip commits that are:
   - Pure refactors with no user-visible effect
   - Session-log or doc-only edits in `./internal/`
   - CLAUDE.md revisions (unless they changed a user-facing rule)

6. **Print the draft block** for the user to paste above the current top
   version in CHANGELOG.md. Also print the `pyproject.toml` version bump
   command:

   ```bash
   # After pasting, bump pyproject.toml:
   # version = "0.4.9" → "0.4.10"
   ```

7. **Do NOT write CHANGELOG.md or pyproject.toml directly.** The user
   reviews and commits.

## Tips

- Look for "Fix" / "Fixed" / "fix:" prefixes → `### Fixed`.
- Look for "Add" / "Added" / "feat:" prefixes → `### Added`.
- Look for "Refactor" / "Simplify" / "Changed" → `### Changed`.
- Look for "Remove" / "Delete" → `### Removed`.
- The project uses plain prose over conventional-commit prefixes — read the
  commit body, don't rely on prefix parsing alone.
- Commits touching only `docs/`, `internal/`, or `.claude/` can usually be
  collapsed into one `### Documentation` or dropped entirely.
