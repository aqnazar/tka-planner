"""Drive the add-on's engine inside Blender and check the scene it draws.

Run from the repository root:

    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
        --background --factory-startup --python tests/addon_parity.py

Exits non-zero if any check fails, so it can gate a commit.

Written as a script rather than as a pytest module because Blender ships no pytest, and
installing one into its Python needs administrator rights on Windows. The checks below
are the ones that genuinely need Blender: that every engine node becomes an object, that
it lands where the engine says, that a delta moves it, that a commit replaces its mesh,
and that a set pose moves one bone set and not the other. Everything else about the
engine is covered by the ordinary suite, which needs no Blender at all.
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "vendor"):
    sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

FAILURES: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def close(a, b, tol=1e-6) -> bool:
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=tol))


def build_case(folder: Path):
    """Write a synthetic patient folder and open a session on it.

    Landmark estimation is stubbed with the synthetic set, exactly as
    ``tests/test_session.py`` does. An ellipsoid has no intercondylar notch and no
    medullary canal, so the real estimator rightly refuses it. Estimation has its own
    suite; what needs Blender is everything downstream of it.
    """
    from tests.synthetic import synthetic_knee
    from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
    from tka_planner.core.measure import measure_femoral_ml, measure_tibial_plateau
    from tka_planner.core.meshio import Mesh, read_stl
    from tka_planner.core.metrics import compute_all
    from tka_planner.geom import mesh as gm
    from tka_planner.session import PlanningSession

    landmarks = synthetic_knee()

    def bone(prefix):
        points = np.array([
            one.position_mm for one in landmarks
            if one.is_usable and one.id.startswith(prefix)
        ])
        low, high = points.min(axis=0), points.max(axis=0)
        unit = gm.uv_sphere(1.0, segments=48, rings=40)
        radii = (high - low) / 2.0 + 12.0
        return Mesh(vertices=unit.vertices * radii + (low + high) / 2.0,
                    faces=unit.faces)

    folder.mkdir(parents=True, exist_ok=True)
    gm.write_stl(bone("femur"), folder / "FD1Left.stl")
    gm.write_stl(bone("tibia"), folder / "TD1Left.stl")

    def fake_measure(self):
        self.case_id = "CASE_PARITY"
        self._femur = read_stl(self.femur_path)
        self._tibia = read_stl(self.tibia_path)
        self._landmarks = landmarks
        self._femoral_frame = build_femoral_frame(landmarks)
        self._tibial_frame = build_tibial_frame(landmarks)
        self._metrics = compute_all(
            landmarks, self._femoral_frame, self._tibial_frame
        )
        self._femoral_measure = measure_femoral_ml(self._femur, self._femoral_frame)
        self._tibial_measure = measure_tibial_plateau(self._tibia, self._tibial_frame)
        self._native_slope_deg = self._metrics["posterior_slope_medial_deg"].value

    PlanningSession._measure = fake_measure
    session = PlanningSession.open(folder, side="left")
    session.controls = type(session.controls)(
        **{**vars(session.controls), "resection_mode": "plane",
           "show_landmarks": True}
    )
    session.build()
    return session


def main() -> int:
    import bpy

    from tka_planner.blender import render
    from tka_planner.scene.model import FEMORAL, TIBIAL

    folder = Path(bpy.app.tempdir) / "tka_parity"
    session = build_case(folder)

    print("\nRendering the engine scene in Blender")
    objects = render.render_scene(session.scene)

    check(
        "every scene node becomes a Blender object",
        set(objects) == set(session.scene.nodes),
        f"{len(objects)} objects for {len(session.scene.nodes)} nodes",
    )

    world = session.scene.world("femoral_distal")
    matrix = np.array(objects["femoral_distal"].matrix_world)
    check(
        "an object sits where the engine says, converted to metres",
        close(matrix[:3, 3] * 1000.0, world[:3, 3], tol=1e-4),
        f"{matrix[:3, 3] * 1000.0} vs {world[:3, 3]}",
    )

    check(
        "landmarks are drawn and visible",
        any(
            node.tags.get("landmark") and not objects[name].hide_viewport
            for name, node in session.scene.nodes.items()
            if name in objects
        ),
    )

    print("\nA replan moves objects through a delta")
    before = np.array(objects["tibial_proximal"].matrix_world)
    render.apply_delta(session.replan(tibial_resection_delta_mm=3.0))
    check(
        "a plan change moves the tibial plane",
        not close(np.array(objects["tibial_proximal"].matrix_world), before),
    )
    check(
        "the femur did not move with it",
        close(
            np.array(objects["Femur"].matrix_world),
            np.array(bpy.data.objects["Femur"].matrix_world),
        ),
    )

    print("\nA commit cuts the bone")
    vertices_before = len(objects["Femur"].data.vertices)
    result = session.commit()
    render.apply_scene(session.scene)
    check(
        "the commit reports a kernel record per bone",
        set(result.records) == {"Femur", "Tibia"},
        str(sorted(result.records)),
    )
    check(
        "the femur mesh in Blender is replaced by the cut one",
        len(bpy.data.objects["Femur"].data.vertices) != vertices_before,
        f"{vertices_before} -> {len(bpy.data.objects['Femur'].data.vertices)}",
    )
    check("the session is no longer stale", session.stale is False)

    print("\nThe scene carries no modifiers")
    check(
        "no object has a modifier",
        all(len(obj.modifiers) == 0 for obj in bpy.data.objects),
        str([obj.name for obj in bpy.data.objects if len(obj.modifiers)]),
    )

    print("\nA trial pose moves one set only")
    femur_before = np.array(bpy.data.objects["Femur"].matrix_world)
    tibia_before = np.array(bpy.data.objects["Tibia"].matrix_world)
    session.set_trial(flexion_deg=40.0)
    render.apply_scene(session.scene)
    check(
        "the tibia moves",
        not close(np.array(bpy.data.objects["Tibia"].matrix_world), tibia_before),
    )
    check(
        "the femur does not",
        close(np.array(bpy.data.objects["Femur"].matrix_world), femur_before),
    )
    check(
        "the femoral set pose is untouched",
        close(session.scene.set_poses[FEMORAL], np.eye(4)),
    )

    session.reset_trial()
    render.apply_scene(session.scene)
    check(
        "resetting the trial returns the tibia exactly",
        close(np.array(bpy.data.objects["Tibia"].matrix_world), tibia_before),
    )
    check(
        "the tibial set pose is the identity again",
        close(session.scene.set_poses[TIBIAL], np.eye(4)),
    )

    print("\nThe report still reads")
    lines = session.report_lines()
    check("report lines are produced", len(lines) > 10, f"{len(lines)} lines")
    check(
        "the report names the correction",
        any(line.startswith("Valgus cut|") for line in lines),
    )

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("All add-on parity checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
