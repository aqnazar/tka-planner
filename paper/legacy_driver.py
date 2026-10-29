"""Run the frozen legacy script inside Blender and record where it put every object.

This file runs *inside* Blender (``blender -b --factory-startup --python
legacy_driver.py -- <script> <record.json>``), started by :mod:`paper.legacy_runner`.
It is the only way to know the legacy pipeline's poses in the bone's own frame.

The legacy script bakes every move into the mesh (``transform_apply`` after each step,
and it rotates vertices in place for the slope), so at the end of a run every object's
``matrix_world`` is the identity and the moves are gone. Rather than change the script,
this driver loads it as a module -- its ``if __name__ == "__main__"`` block therefore
does not fire -- wraps the few helpers that move geometry, and then calls the script's
own ``run()``, which is exactly what its main block does with ``BATCH_MODE = "F"``.
The wrappers call the originals unchanged and only watch: they keep, per object, the
matrix ``L`` that takes the file's coordinates to the mesh's current local
coordinates, so ``matrix_basis @ L`` maps the STL as read from disk to where the legacy
script left it.

* ``apply_transforms`` bakes ``matrix_basis`` into the mesh: ``L <- basis @ L``.
* ``pca_align_shaft``, ``apply_z_rotation_around_centroid`` and
  ``apply_posterior_slope`` move vertices directly. The rigid motion they made is
  recovered by a least-squares (Kabsch) fit of the vertices before and after, which is
  exact to float precision because the topology does not change.
* ``import_stl`` and ``duplicate_object`` start and copy the record.

Nothing here may change what the script computes. ``paper.legacy_runner --plain`` runs
the same temporary script with plain ``--python`` and no driver, so the two can be
compared output for output.
"""

import json
import sys
import time
import traceback
import types

import bpy
import numpy as np

_ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SCRIPT, RECORD = _ARGS[0], _ARGS[1]

_L = {}        # session_uid -> 4x4 file -> mesh-local
_SOURCE = {}   # session_uid -> file the object was read from
_EVENTS = []   # what the wrappers saw, in order, for the record
_CALLS = {"select_size": [], "select_size_quiet": [], "run_femur": None}


def _uid(obj):
    return obj.session_uid


def _basis(obj):
    # matrix_basis is computed from location/rotation/scale on access, so it is current
    # even straight after the script sets obj.location, before any depsgraph update.
    return np.array(obj.matrix_basis, dtype=float)


def _local_coords(obj):
    n = len(obj.data.vertices)
    flat = np.empty(n * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", flat)
    return flat.reshape(n, 3).astype(float)


def _kabsch(before, after):
    """The rigid 4x4 taking ``before`` onto ``after`` (both (N, 3)), and its residual."""
    cb, ca = before.mean(axis=0), after.mean(axis=0)
    h = (before - cb).T @ (after - ca)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = ca - r @ cb
    residual = float(np.abs(before @ r.T + m[:3, 3] - after).max()) if len(before) else 0.0
    return m, residual


def install(mod):
    original = {name: getattr(mod, name) for name in (
        "import_stl", "apply_transforms", "duplicate_object", "pca_align_shaft",
        "apply_z_rotation_around_centroid", "apply_posterior_slope",
        "select_size", "select_size_quiet", "run_femur")}

    def import_stl(filepath, name, import_scale=0.001):
        obj = original["import_stl"](filepath, name, import_scale)
        _L[_uid(obj)] = np.eye(4)
        _SOURCE[_uid(obj)] = filepath
        return obj

    def apply_transforms(obj):
        key = _uid(obj)
        before = _basis(obj)
        result = original["apply_transforms"](obj)
        _L[key] = before @ _L.get(key, np.eye(4))
        return result

    def duplicate_object(source, new_name):
        dup = original["duplicate_object"](source, new_name)
        _L[_uid(dup)] = _L.get(_uid(source), np.eye(4)).copy()
        _SOURCE[_uid(dup)] = _SOURCE.get(_uid(source))
        return dup

    def vertex_mover(name):
        inner = original[name]

        def wrapped(obj, *args, **kwargs):
            key = _uid(obj)
            l_old = _L.get(key, np.eye(4)).copy()
            basis_before = _basis(obj)
            world_before = _local_coords(obj) @ basis_before[:3, :3].T + basis_before[:3, 3]
            result = inner(obj, *args, **kwargs)
            basis_after = _basis(obj)
            world_after = _local_coords(obj) @ basis_after[:3, :3].T + basis_after[:3, 3]
            motion, residual = _kabsch(world_before, world_after)
            # The inner function may itself have baked the basis (and so updated L via
            # the wrapped apply_transforms); recompute L from first principles instead.
            _L[key] = np.linalg.inv(basis_after) @ motion @ basis_before @ l_old
            _EVENTS.append({"function": name, "object": obj.name,
                            "motion": motion.tolist(), "fit_residual_m": residual})
            return result
        return wrapped

    def select_size(measurement_m, bone, mode, size_table):
        chosen = original["select_size"](measurement_m, bone, mode, size_table)
        _CALLS["select_size"].append({"measurement_mm": measurement_m * 1000, "bone": bone,
                                      "mode": mode, "size": chosen["size"]})
        return chosen

    def select_size_quiet(measurement_m, bone, mode, size_table):
        chosen = original["select_size_quiet"](measurement_m, bone, mode, size_table)
        _CALLS["select_size_quiet"].append({"measurement_mm": measurement_m * 1000,
                                            "bone": bone, "mode": mode,
                                            "size": chosen["size"]})
        return chosen

    def run_femur(*args, **kwargs):
        result = original["run_femur"](*args, **kwargs)
        _CALLS["run_femur"] = {k: v for k, v in result.items() if k != "selected"}
        return result

    mod.import_stl = import_stl
    mod.apply_transforms = apply_transforms
    mod.duplicate_object = duplicate_object
    for name in ("pca_align_shaft", "apply_z_rotation_around_centroid",
                 "apply_posterior_slope"):
        setattr(mod, name, vertex_mover(name))
    mod.select_size = select_size
    mod.select_size_quiet = select_size_quiet
    mod.run_femur = run_femur


def main():
    record = {"blender_version": bpy.app.version_string,
              "blender_build_hash": bpy.app.build_hash.decode()
              if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
              "script": SCRIPT, "ok": False}
    t0 = time.perf_counter()
    try:
        with open(SCRIPT, encoding="utf-8") as handle:
            source = handle.read()
        mod = types.ModuleType("tka_legacy_frozen")
        mod.__file__ = SCRIPT
        exec(compile(source, SCRIPT, "exec"), mod.__dict__)
        install(mod)
        if mod.BATCH_MODE == "T":
            raise RuntimeError("BATCH_MODE is 'T'; the runner only drives single knees.")
        mod.run()
        record["ok"] = True
    except Exception as error:  # the record says what failed; the runner reports it
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    record["seconds"] = round(time.perf_counter() - t0, 2)

    objects = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        key = _uid(obj)
        objects.append({
            "name": obj.name,
            "source": _SOURCE.get(key),
            "tracked": key in _L,
            # file coordinates (as read from the STL) -> legacy world (metres)
            "file_to_world": (_basis(obj) @ _L.get(key, np.eye(4))).tolist(),
            "matrix_world": [list(row) for row in obj.matrix_world],
            "n_vertices": len(obj.data.vertices),
        })
    record["objects"] = objects
    record["events"] = _EVENTS
    record["calls"] = _CALLS
    with open(RECORD, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=float)


main()
