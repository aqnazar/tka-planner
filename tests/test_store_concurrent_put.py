"""Two commits of the same knee store the same mesh at once; neither may fail.

They write the same digest; with one shared staging name the second replace failed on
Windows with PermissionError (WinError 32) and its commit request died (2026-11-08).
"""

import threading

import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.store.meshes import MeshStore


def test_parallel_puts_of_one_mesh_all_succeed(tmp_path):
    rng = np.random.default_rng(0)
    mesh = Mesh(vertices=rng.normal(size=(20000, 3)), faces=rng.integers(0, 20000, size=(40000, 3)))
    store = MeshStore(tmp_path)
    errors, digests = [], []

    def put():
        try:
            digests.append(store.put(mesh))
        except Exception as error:  # noqa: BLE001 - collected for the assertion
            errors.append(repr(error))

    threads = [threading.Thread(target=put) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(set(digests)) == 1
    assert not list(tmp_path.rglob("*.part"))
    loaded = store.get(digests[0])
    assert np.array_equal(loaded.faces, mesh.faces)
