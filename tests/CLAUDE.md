# tests/ — pytest suite

Last updated: 2026-05-04

Loads only when Claude is operating inside `tests/`. Root project rules: `../CLAUDE.md`. Governance: `../.claude/CLAUDE.md`.

## Invocation

```bash
# Full suite
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.

# With autoresearch contract tests (skip on clones without duckdb)
uv run --group dev --group analysis --group experiments python -m pytest tests/ -v --rootdir=.

# Single file
uv run --group dev --group analysis python -m pytest tests/test_X.py -v --rootdir=.
```

`--rootdir=.` is required so pytest doesn't walk up into the parent ComfyUI tree. CI runs the full suite plus `scripts/audit_workflows.py` + `scripts/validate_docs_consistency.py` (`.github/workflows/ci.yml`).

## conftest + sys.path

`tests/conftest.py` adds `scripts/` and `tests/` to `sys.path`. Effect: `from <script_module> import X` and `from <test_helper> import Y` both work without package indirection.

A root `./conftest.py` exists with `collect_ignore` that shadows `tests/conftest.py` for `from conftest import X` imports — relevant only if you're consuming the conftest directly (rare).

## Running without ComfyUI loaded

Pytest runs without ComfyUI (CI default). Two affordances make this work:

- **`__init__.py`** guards ComfyUI imports.
- **`nodes.py`** has `_IOStub`/`_Passthrough` fallback under `try: from comfy_api.latest import io / except ImportError:`. New node modules that need `comfy_api` / `comfy.patcher_extension` define their own inline fallbacks (see `nodes_sage.py`, `nodes_easycache.py`). Two consumers is the minimum threshold for extracting to a shared helper; factor out only if a third node needs the same stubs.

## Schema invariant tests need AST parsing, not runtime introspection

When ComfyUI isn't loaded, `define_schema()` returns `_Passthrough` stubs and `schema.inputs` isn't iterable. So invariants on schema shape walk `io.*.Input(...)` calls via `ast` instead. Canonical: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs` (catches the `control_after_generate` widget trap by name) and `tests/test_node_schemas.py::test_keyframe_idxs_cleared_to_none_not_empty_list`.

When adding a new schema invariant, copy the AST-walk pattern. Don't try `from nodes import X; X.define_schema()` — it will work locally with ComfyUI installed and silently no-op on CI.

**Class-scoped invariants** (e.g. "LatentTemporalMask's `edge_taper_seconds` default must be 0.0", where the same input name might appear on a sibling node) use `_scan_io_input_records_in_class(path, class_name)` — locates `class X` in source text, bounds the body by the next top-level `\nclass `, filters records by lineno. Copy the pattern at `tests/test_node_schemas.py::test_latent_temporal_mask_edge_taper_default_is_zero` when adding a per-class default-value guard.

## Shared fakes (`tests/_fakes.py`)

Three-layer hierarchy. Each layer adds the surface a class of tests needs:

```
FakeModelPatcher
    └── FakeModelWithCallbacks   (adds add_callback)
            └── FakeModelWithWrappers   (adds add_wrapper_with_key)
```

**Minimal-interface principle**: a test imports the smallest fake that exposes its required surface. Tests that don't exercise wrappers shouldn't import `FakeModelWithWrappers` — using a thinner fake catches accidental coupling.

Imports go directly: `from _fakes import FakeModelWithCallbacks` (sys.path is set up by `conftest.py`).

**Callable preservation in deepcopy**: closures that participate in callback identity (e.g. for override-replacement tests) need a memo dict so `deepcopy` doesn't break their identity. The fakes handle this; test authors using closures directly should follow the same pattern.

## Memoization fixes need REPEATED-call tests

Single-call tests can't detect framework-cache invalidation bugs — they happen on the *second* call when ComfyUI's TensorLoop reruns a downstream node. Canonical shape: `tests/test_batch_encode.py::TestBatchEncoderCaching` — same input, repeated invocations, assertion on call count or timing.

Any node that uses `id()`-keyed LRU + `IS_CHANGED` for memoization (the `TimestampPromptScheduleBatchEncode` pattern) needs this test shape. Module-level caches (`_BATCH_ENCODE_CACHE`, `_COND_CACHE`) die on ComfyUI restart — they're plain dicts, no persistence — so repeated-call tests don't need cross-process logic.

## `id()`-keyed caches need autouse clear-fixtures

`FakeCLIP` and other fake objects get GC'd rapidly during test runs; Python address recycling produces ghost cache hits. Two protections:

1. **Production cache keys include `type(clip).__name__`** as cheap cross-class insurance.
2. **Autouse fixtures in test files** that own the cache clear it between tests. Don't rely on `pytest`'s test-isolation alone for `id()`-keyed maps.

## Substring contracts on `_LLM_SYSTEM_PROMPT`

8 tests in `tests/test_audio_features.py::TestFormatJsonReport` assert specific load-bearing substrings in the LLM system prompt:

- `is singing` / `are singing together`
- `verbatim` / `identical` / `exactly`
- All 6 tier names
- `montage` + `emotional`
- `dolly out`
- `present progressive`
- `frozen`
- `init image` + `do not re-describe`
- Style-family examples (`comic` / `graphic-novel` / `animated` / `live-action`)

Read these BEFORE rewriting the system prompt. A substring you remove silently breaks the test.

## Apply-script tests need pre-migration state

Have the fixture `shutil.copy2(CANONICAL, dst)` then invoke the apply script's own `--revert` to restore. Keeps fixture state in lockstep with the script's understanding of "before"; avoids a separate fixture-baseline file that drifts when the canonical changes.

When a migration is retired (e.g. baked into the canonical permanently), drop its tests too — the audit pair is the durable invariant; tests against a moving canonical become noise.

## Degenerate-input metric branches need a distinct status

When an extractor handles a degenerate case (e.g. n=1 frame in `subject_consistency` → no comparisons possible → all sims trivially 1.0), returning `status: "ok"` with sentinel numbers pollutes downstream `WHERE status = 'ok'` aggregations — a degenerate render scores identically to a perfect one. Add a distinct `Literal` status (e.g. `single_frame`) so queries can exclude. Same shape as the `trace_empty` / `trace_missing` / `decode_failed` distinctions in `sage_summary.py`.

## CLAUDE.md hygiene tests

`tests/test_claude_md_budget.py` enforces the budget + lint rules from `../.claude/CLAUDE.md` "CLAUDE.md governance" (size budget, pointer-target integrity, orphan check on `docs/reference/`). When a test fails, the failure message names the offending file and the rule violated. Fixes:

- *Budget*: compress / move to subtree CLAUDE.md / move to `docs/`.
- *Pointer*: rename target, update pointer, or remove if no longer relevant.
- *Orphan*: add a citation in `docs/README.md`'s task-first index, or delete the orphan note.

Numeric thresholds (line / byte caps) live in the policy doc — read it before adjusting them.

## References

- `../CLAUDE.md` — root project rules
- `../scripts/CLAUDE.md` — apply-script + audit conventions tested here
- `../.claude/CLAUDE.md` — CLAUDE.md governance policy
- `_fakes.py` — shared fake-model hierarchy
- `conftest.py` — sys.path setup
