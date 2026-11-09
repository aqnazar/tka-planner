"""The VSD Fischer knees as planner cases, cut to the length of a knee-only scan.

VSDFullBodyBoneModels (Fischer 2023, Sci. Data 10:763; CC BY-NC-SA 4.0) holds whole
lower limbs of 30 cadavers, in the scanner's LPS millimetres like our patients' scans.
The planner is made for knee-only scans, so each femur keeps its distal
``FEMUR_LENGTH_MM`` and each tibia its proximal ``TIBIA_LENGTH_MM`` (about the median of
the study knees). The fibula is cut at the tibia's plane and joined to the tibia, as the
fibular head is part of the tibial segmentation in our scans.

    python -m online.vsd_cases <VSD repo> <out folder>

writes ``<out>/VSD002/FD1Left.stl``, ``TD1Left.stl``, ``FD1Right.stl``, ``TD1Right.stl``.
The output is third-party geometry under a non-commercial licence: never commit it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import manifold3d as m3
import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.geom.mesh import write_stl

FEMUR_LENGTH_MM = 175.0
TIBIA_LENGTH_MM = 175.0
# The VSD surfaces are about 25 times coarser than our scans (edges up to a few mm),
# and the plateau search needs a minimum count of surface points in each dish. The
# triangles are subdivided to this edge length: the same surface, denser points.
REFINE_EDGE_MM = 0.7
# A knee turned further than this about the long axis is turned square (see knee()).
MAX_TURN_KEPT_DEG = 30.0
SIDES = {"L": "Left", "R": "Right"}


def load_subject(path) -> dict:
    """``{bone name: (vertices, faces)}`` from one VSD MAT file (faces zero-based)."""
    import scipy.io

    bones = scipy.io.loadmat(str(path))["B"]
    out = {}
    for index in range(bones.size):
        entry = bones[0, index]
        if not entry[0].size or not entry[2].size:
            continue
        mesh = entry[2][0][0]
        out[str(entry[0][0])] = (np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64) - 1)
    return out


def _manifold(vertices, faces) -> m3.Manifold:
    mesh = m3.Mesh(vert_properties=np.asarray(vertices, np.float32),
                   tri_verts=np.asarray(faces, np.uint32))
    mesh.merge()
    solid = m3.Manifold(mesh)
    if solid.status() != m3.Error.NoError:
        raise ValueError(f"Not a closed surface: {solid.status()}")
    return solid


def _keep(solid: m3.Manifold, *, above: float | None = None,
          below: float | None = None) -> m3.Manifold:
    """The part of ``solid`` with z above ``above`` and/or below ``below`` (capped)."""
    if above is not None:
        solid = solid.trim_by_plane((0.0, 0.0, 1.0), above)
    if below is not None:
        solid = solid.trim_by_plane((0.0, 0.0, -1.0), -below)
    return solid


def _mesh(solids) -> Mesh:
    vertices, faces, offset = [], [], 0
    for solid in solids:
        if solid.is_empty():
            continue
        out = solid.to_mesh()
        v = np.asarray(out.vert_properties, float)[:, :3]
        vertices.append(v)
        faces.append(np.asarray(out.tri_verts, np.int64) + offset)
        offset += len(v)
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def knee(bones: dict, letter: str) -> tuple[Mesh, Mesh, dict]:
    """The femur and tibia (+ fibular head) of one side, knee-only, and how it was turned.

    Some cadavers lie with the leg turned outwards by 45 to 67 degrees, beyond what the
    planner accepts from a scan (``SCAN_ORIENTATION_TURNED``). Such a knee is turned
    about the scanner's long axis, through the femur's distal end, until its posterior
    condylar line runs left-right, the way the knee lies on a clinical knee scan; of the
    two turns that do that, the one that keeps the patella in front. A rigid turn: the
    bones' shapes and the landmarks on them are unchanged, and ``orientation`` records it.
    """
    from tka_planner.core.landmarks_auto import femoral_axes

    femur = _manifold(*bones[f"Femur_{letter}"])
    tibia = _manifold(*bones[f"Tibia_{letter}"])
    fibula = _manifold(*bones[f"Fibula_{letter}"])
    distal = femur.bounding_box()[2]
    proximal = tibia.bounding_box()[5]
    femur = _keep(femur, below=distal + FEMUR_LENGTH_MM)
    cut = proximal - TIBIA_LENGTH_MM
    tibia, fibula = _keep(tibia, above=cut), _keep(fibula, above=cut)

    side = SIDES[letter].lower()
    axes = femoral_axes(_mesh([femur.refine_to_length(REFINE_EDGE_MM)]), side)
    turn = float(axes.diagnostics.get("mediolateral_turn_deg", 0.0))
    orientation = {"turned_deg": 0.0, "measured_turn_deg": round(turn, 1),
                   "about": "scanner +Z through the femur's distal-end centre"}
    if turn > MAX_TURN_KEPT_DEG:
        left = np.asarray(axes.patient_left, float)
        theta = float(np.degrees(np.arctan2(left[1], left[0])))
        bb = femur.bounding_box()
        centre = np.array([(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, bb[2]])
        patella = np.asarray(bones[f"Patella_{letter}"][0], float).mean(axis=0)
        best = None
        for angle in (-theta, 180.0 - theta):
            angle = (angle + 180.0) % 360.0 - 180.0
            r = np.radians(angle)
            ahead = (np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
                     @ (patella - centre)[:2])[1]
            if ahead < 0 and (best is None or abs(angle) < abs(best)):
                best = angle  # LPS: anterior is -Y
        if best is None:
            raise ValueError(f"No turn puts the {side} patella in front.")
        femur, tibia, fibula = (_turn(solid, best, centre) for solid in (femur, tibia, fibula))
        orientation.update(turned_deg=round(best, 3), centre_mm=centre.round(3).tolist())
    return (_mesh([femur.refine_to_length(REFINE_EDGE_MM)]),
            _mesh([tibia.refine_to_length(REFINE_EDGE_MM),
                   fibula.refine_to_length(REFINE_EDGE_MM)]),
            orientation)


def _turn(solid: m3.Manifold, degrees: float, centre) -> m3.Manifold:
    centre = np.asarray(centre, float)
    return (solid.translate(tuple(-centre)).rotate((0.0, 0.0, degrees))
            .translate(tuple(centre)))


def convert(repo, out) -> list:
    written = []
    for path in sorted(p for p in Path(repo, "Bones").glob("*.mat")
                      if not p.stem.startswith("Phantom")):
        bones = load_subject(path)
        folder = Path(out) / f"VSD{path.stem.upper()}"
        turns = {"source": f"VSDFullBodyBoneModels Bones/{path.name}"}
        for letter, side in SIDES.items():
            femur, tibia, orientation = knee(bones, letter)
            write_stl(femur, folder / f"FD1{side}.stl")
            write_stl(tibia, folder / f"TD1{side}.stl")
            turns[side.lower()] = orientation
            written.append(f"{folder.name}:{side.lower()}")
        (folder / "orientation.json").write_text(json.dumps(turns, indent=2), "utf-8")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="VSDFullBodyBoneModels checkout (has Bones/)")
    parser.add_argument("out", help="folder for the case folders")
    args = parser.parse_args(argv)
    written = convert(args.repo, args.out)
    print(f"{len(written)} knees written to {args.out}")


if __name__ == "__main__":
    main()
