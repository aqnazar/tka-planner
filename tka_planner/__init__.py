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
