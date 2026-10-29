"""Run a list of booleans through Blender's exact solver, for E8 only.

Runs *inside* Blender, never in the harness process:

    blender -b --factory-startup --python paper/blender_worker.py -- manifest.json

The manifest lists jobs as ``{"name", "operation", "target", "tool", "out"}``, each
mesh an ``.npz`` holding ``vertices`` (float64) and ``faces`` (int64). Each result is
written the same way, so nothing passes through STL's single precision on its way
back, and ``<manifest>.result.json`` records Blender's version and each job's time or
error.

Kept out of the harness process for the reason the agreement test gives: Blender is
GPL, and having ``bpy`` importable where the planner runs makes it far too easy to
depend on it by accident. The harness only ever talks to it through files.
"""

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tka_planner.core.meshio import Mesh  # noqa: E402
from tka_planner.geom.kernels.blender_kernel import BlenderKernel  # noqa: E402


def _load(path) -> Mesh:
    with np.load(path) as data:
        return Mesh(vertices=np.asarray(data["vertices"], dtype=np.float64),
                    faces=np.asarray(data["faces"], dtype=np.int64))


def main() -> None:
    import bpy

    manifest_path = Path(sys.argv[sys.argv.index("--") + 1])
    jobs = json.loads(manifest_path.read_text(encoding="utf-8"))
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    kernel = BlenderKernel()
    results = []
    for job in jobs:
        entry = {"name": job["name"]}
        try:
            target, tool = _load(job["target"]), _load(job["tool"])
            t0 = time.perf_counter()
            out = getattr(kernel, job["operation"])(target, tool).mesh
            entry["seconds"] = time.perf_counter() - t0
            np.savez(job["out"], vertices=out.vertices, faces=out.faces)
            entry["triangles"] = int(out.n_triangles)
        except Exception as error:  # reported per job, so one failure keeps the rest
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["traceback"] = traceback.format_exc(limit=3)
        results.append(entry)
        print(f"{job['name']}: {entry.get('seconds', entry.get('error'))}", flush=True)

    manifest_path.with_suffix(".result.json").write_text(json.dumps({
        "blender": bpy.app.version_string,
        "jobs": results,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
