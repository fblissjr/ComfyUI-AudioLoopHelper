"""Pytest-time stand-ins for `comfy_api.latest.io` and friends.

When this plugin is imported outside a ComfyUI runtime (under pytest, in
a CI lint, in a script that just wants to import a node module for its
algorithmic helpers), the `comfy_api` and `comfy.patcher_extension`
imports fail. We don't want those imports to crash module load -- we
want the algorithmic core to stay importable and testable.

Each node module imports `io`, `_ON_CLEANUP`, etc. from here with a
try/except wrapper at top-of-module. In the runtime case, the real
symbols win; in the pytest case, the stubs win and the symbols below
are no-ops sufficient for unit-testing the node's pure-Python logic.
"""

from __future__ import annotations


class _Passthrough:
    def __getattr__(self, _name):
        return _Passthrough()

    def __call__(self, *_args, **_kwargs):
        return _Passthrough()


class _IOStub(_Passthrough):
    """Mirrors the surface of `comfy_api.latest.io` we touch."""

    class ComfyNode:
        pass

    @staticmethod
    def NodeOutput(*args):
        return args


def _override_passthrough(fn):
    """Stand-in for `typing_extensions.override`."""
    return fn


def _stub_constants() -> tuple:
    """Return `(_DIFFUSION_MODEL, _ON_CLEANUP)` literals matching the
    runtime constants `WrappersMP.DIFFUSION_MODEL` and
    `CallbacksMP.ON_CLEANUP` from `comfy.patcher_extension`. Tests rely
    on these literals matching the real ones."""
    return ("diffusion_model", "on_cleanup")


# Importable singletons. Node modules do:
#   try:
#       from comfy_api.latest import io
#       from typing_extensions import override
#   except ImportError:
#       from ._comfy_stubs import io_stub as io, override_stub as override
io_stub = _IOStub()
override_stub = _override_passthrough
