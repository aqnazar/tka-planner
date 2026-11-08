"""Open, reproducible TKA pre-operative planning. Research use only.

This package is both an importable library and a Blender add-on. Blender discovers an
add-on by finding ``bl_info`` and ``register`` in a package's ``__init__``, so they live
here rather than in :mod:`tka_planner.addon` -- an earlier layout put them one level
down, where Blender never looked, which is why the add-on refused to appear in the list
however many script paths were added.

Zipping this directory therefore produces something **Install from Disk** accepts
directly, with the whole package travelling along inside it.

Nothing here imports ``bpy`` at module level, so ``import tka_planner`` still works in
plain Python and the test suite stays Blender-free.
"""

import os as _os

# One BLAS thread per caller, set before numpy is first imported (it is read only then).
# numpy's bundled OpenBLAS (0.3.31 in the 2.4 wheels) kills the process with an access
# violation, or hangs it, when two Python threads enter a multithreaded BLAS routine at
# once, and the planner server runs every request on its own thread: a knee opened while
# the browser fetched meshes died with exit 139 and no traceback (2026-11-08,
# tests/test_concurrent_blas.py). The BLAS work here is (N, 3) @ (3, 3), bound by memory
# rather than arithmetic, so one thread costs nothing measurable. A value already set in
# the environment is respected.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"

bl_info = {
    "name": "TKA Planner",
    "author": "Satbayev University",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > TKA",
    "description": (
        "Open pre-operative planning for total knee arthroplasty. "
        "Research and demonstration only - not a medical device."
    ),
    "category": "Object",
}


def register():
    """Entry point Blender calls when the add-on is enabled."""
    from .addon import register as _register

    _register()


def unregister():
    from .addon import unregister as _unregister

    _unregister()
