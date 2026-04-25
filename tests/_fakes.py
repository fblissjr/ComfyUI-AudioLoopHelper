"""Shared test fakes. `tests/` directory is added to sys.path by
`tests/conftest.py`, so tests import as `from _fakes import ...`."""

from __future__ import annotations

import copy
from typing import Callable


class FakeModelPatcher:
    """Minimal stand-in for `comfy.model_patcher.ModelPatcher`.

    Surface that both `LoopIterationStamp` and `AudioLoopHelperSageAttention`
    touch: `model_options` dict with a `transformer_options` sub-dict, and
    `clone()` returning an independent copy. Subclasses add more surface
    (e.g. `add_callback` for the sage node).

    `clone()` deep-copies `model_options` to match production
    `ModelPatcher.clone()` (which uses `deepcopy_list_dict` at
    `comfy/model_patcher.py:341`). Callables inside are kept as references,
    not deep-copied, so closure identity still works for override tests.
    """

    def __init__(self, transformer_options: dict | None = None):
        self.model_options: dict = {"transformer_options": dict(transformer_options or {})}

    def clone(self):
        clone = type(self)()
        clone.model_options = copy.deepcopy(
            self.model_options,
            memo={id(v): v for v in _walk_callables(self.model_options)},
        )
        return clone


def _walk_callables(obj):
    """Yield callables reachable from obj so deepcopy's memo preserves
    their identity. Assumes acyclic input -- ComfyUI's `model_options` is
    plain dicts + lists + callables with no back-references.
    """
    if callable(obj) and not isinstance(obj, (type, dict, list)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_callables(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_callables(v)


class FakeModelWithCallbacks(FakeModelPatcher):
    """FakeModelPatcher + the `add_callback` surface used by AudioLoopHelperSageAttention."""

    def __init__(self, transformer_options: dict | None = None):
        super().__init__(transformer_options)
        self.callbacks: dict[str, Callable] = {}

    def add_callback(self, call_type: str, fn: Callable) -> None:
        self.callbacks[call_type] = fn


class FakeModelWithWrappers(FakeModelWithCallbacks):
    """FakeModelWithCallbacks + the `add_wrapper_with_key` surface used by
    LTXVideoEasyCache. Mirrors `comfy.model_patcher.ModelPatcher.wrappers`
    layout: `{wrapper_type: {key: [callable, ...]}}`."""

    def __init__(self, transformer_options: dict | None = None):
        super().__init__(transformer_options)
        self.wrappers: dict[str, dict[str | None, list[Callable]]] = {}

    def add_wrapper_with_key(self, wrapper_type: str, key: str | None, fn: Callable) -> None:
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(fn)
