# Vendored wheels

The boolean kernel, built for **Blender's** Python rather than the one in your `PATH`.

`tka_planner` needs `manifold3d` to cut geometry. The add-on runs inside Blender, which
ships neither pip nor manifold3d, and installing into Blender's own `site-packages`
needs administrator rights on Windows. So the wheel is unpacked here instead, and
`tka_planner.addon._ensure_package_importable` puts this directory on `sys.path`.

Nothing else belongs here. `numpy` is deliberately absent: Blender bundles its own, and
a second copy on the path is a good way to get two incompatible array types in one
process.

## Rebuilding it

Match the Python version Blender is built against. Blender 5.1 uses CPython 3.13:

```bash
pip install --target vendor --python-version 3.13 --only-binary=:all: "manifold3d>=3.0"
rm -rf vendor/numpy vendor/numpy-*.dist-info vendor/numpy.libs vendor/bin
```

Check it loaded:

```bash
blender --background --factory-startup --python-expr \
  "import sys; sys.path.insert(0, 'vendor'); import manifold3d; print('ok')"
```

Not needed for the test suite or the command line, which use the `manifold3d` from
`pip install -e .` like any other dependency.
