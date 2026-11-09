"""Export named objects from an archived legacy ``.blend``, read only (runs in Blender).

    blender -b --factory-startup <scene.blend> --python legacy_blend_export.py -- \
        <out_dir> <object>=<file.stl> ...

Where a later run overwrote an archived run's STLs, the scene the legacy script saved
at the end of that run still holds its objects. The script bakes every transform into
the mesh, so exporting an object as it sits reproduces the STL the run wrote. The
scene file is only read; nothing is saved.
"""

import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out_dir, pairs = args[0], args[1:]
os.makedirs(out_dir, exist_ok=True)
for pair in pairs:
    name, filename = pair.split("=", 1)
    obj = bpy.data.objects.get(name)
    if obj is None:
        print(f"[MISSING] {name}")
        continue
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=os.path.join(out_dir, filename),
                          export_selected_objects=True)
    print(f"[EXPORT] {name} -> {filename}")
