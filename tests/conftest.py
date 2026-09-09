"""Fixtures shared by the session, store and server suites.

The synthetic knee lives here rather than in one suite because three suites now need
the same thing: a patient folder, an implant library and an opened session whose
measurement is stubbed. Duplicating that would mean three chances for the stub to drift
away from what ``PlanningSession._measure`` actually produces.
"""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.geom import mesh as gm
from tka_planner.session import PlanningSession


def _bone_like(landmarks, prefix: str):
    """A dense ellipsoid enclosing one bone's landmarks.

    Measurement takes a cross-section through a thin slab and needs vertices spread
    through it, so a bounding box with eight corners is not enough. An ellipsoid over
    the landmark cloud gives a mesh the real measurement functions can work on, which
    is worth more here than a stub: it means these tests drive the whole planning path
    rather than a mock of it.
    """
    from tka_planner.core.meshio import Mesh

    points = np.array(
        [one.position_mm for one in landmarks
         if one.is_usable and one.id.startswith(prefix)]
    )
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    radii = (high - low) / 2.0 + 12.0

    unit = gm.uv_sphere(1.0, segments=48, rings=40)
    return Mesh(vertices=unit.vertices * radii + centre, faces=unit.faces)


@pytest.fixture
def folder(tmp_path):
    """A patient folder in the naming convention the add-on expects."""
    landmarks = synthetic_knee()
    gm.write_stl(_bone_like(landmarks, "femur"), tmp_path / "FD1Left.stl")
    gm.write_stl(_bone_like(landmarks, "tibia"), tmp_path / "TD1Left.stl")
    return tmp_path


@pytest.fixture
def library(tmp_path):
    """A minimal implant library in the shape resolve_component_meshes expects.

    One folder per chart size, holding the seven parts under the library's own naming.
    Only one size is populated, which also exercises the outward search that finds a
    part from a neighbouring size when a folder is incomplete.
    """
    from tka_planner.core.sizing import load_size_chart
    from tka_planner.session import find_size_chart

    root = tmp_path / "library"
    for label in load_size_chart(find_size_chart()).labels:
        (root / label).mkdir(parents=True)

    size = root / "M2"
    for filename in (
        "Implant(Femoral)Left_M2.stl",
        "CuttingBlock(Femoral)Left_M2.stl",
        "CuttingBlock(Femoral)Shell_Left_M2.stl",
        "Implant(Tibial)Plate_M2.stl",
        "Tibial_Insert_M2.stl",
        "CuttingBlock(Tibia)_M2.stl",
        "CuttingBlock(Tibia)Shell_M2.stl",
    ):
        gm.write_stl(gm.box(20.0), size / filename)
    return root


@pytest.fixture
def raw_session(folder, library, monkeypatch):
    """A session on synthetic anatomy.

    Measurement is stubbed so these tests exercise planning, posing and committing
    rather than landmark estimation, which has its own suite. Everything the planning
    path reads must be set here, so this stub is also a statement of what ``_measure``
    is responsible for producing.
    """
    from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
    from tka_planner.core.measure import measure_femoral_ml, measure_tibial_plateau
    from tka_planner.core.metrics import compute_all
    from tka_planner.core.meshio import read_stl

    landmarks = synthetic_knee()

    def fake_measure(self):
        self.case_id = "CASE_TEST"
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

    monkeypatch.setattr(PlanningSession, "_measure", fake_measure)
    return PlanningSession.open(folder, side="left", library=library)


@pytest.fixture
def session(raw_session):
    """An opened session with its first plan built, which is where most tests start."""
    raw_session.build()
    return raw_session


