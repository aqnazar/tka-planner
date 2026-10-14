"""The orientation gate: a knee lying too far from LPS is refused, cleanly.

Before the gate, metamorphic testing turned every knee upside down and the planner
accepted all ten, with rotational errors up to 112 degrees; at 90 degrees seven of ten
crashed inside the planner. Within the gate the landmark searches run in the bones' own
axes and do not care how the knee lies (see test_landmarks_auto_invariance); beyond it
the signs they take from the scanner cannot be trusted, so the scan is refused with a
finding that says why.
"""

import numpy as np
import pytest

from tka_planner.core.landmarks_auto import femoral_axes, tibial_axes
from tka_planner.core.meshio import Mesh
from tka_planner.core.qc import (
    ORIENTATION_MAX_TILT_DEG,
    Severity,
    check_scan_orientation,
)
from tka_planner.core.sides import Side

from .test_landmarks_auto import proximal_tibia
from .test_landmarks_auto_femur import distal_femur
from .test_landmarks_auto_invariance import _rotation


def _knee():
    """A left knee: the constructed femur standing 6 mm above the constructed tibia.

    The femur's shaft is carried 12 mm forwards of its condylar block, as a real femur's
    is, so that the posterior condyles stand well behind the line of the shaft.
    """
    femur = distal_femur(150.0)
    vertices = femur.vertices.copy()
    vertices[:, 1] -= 12.0 * np.clip((vertices[:, 2] - 30.0) / 25.0, 0.0, 1.0)
    vertices[:, 2] += 6.0
    return Mesh(vertices=vertices, faces=femur.faces), proximal_tibia("left")


def _codes(femur, tibia, rotation=np.eye(3)):
    moved = [Mesh(vertices=m.vertices @ rotation.T, faces=m.faces) for m in (femur, tibia)]
    report = check_scan_orientation(femoral_axes(moved[0], Side.LEFT),
                                    tibial_axes(moved[1]))
    return {f.code: f.severity for f in report.findings}


def test_a_knee_lying_square_is_accepted():
    assert _codes(*_knee()) == {"SCAN_ORIENTATION_OK": Severity.INFO}


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
@pytest.mark.parametrize("angle", [-30.0, 30.0])
def test_a_knee_turned_within_the_gate_is_accepted(axis, angle):
    assert _codes(*_knee(), _rotation(axis, angle)) == {
        "SCAN_ORIENTATION_OK": Severity.INFO}


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0]])
def test_a_knee_tilted_beyond_the_gate_is_refused(axis):
    codes = _codes(*_knee(), _rotation(axis, ORIENTATION_MAX_TILT_DEG + 15.0))
    assert codes.get("SCAN_ORIENTATION_TILTED") is Severity.ERROR


def test_a_knee_turned_about_its_long_axis_beyond_the_gate_is_refused():
    codes = _codes(*_knee(), _rotation([0, 0, 1], 60.0))
    assert codes.get("SCAN_ORIENTATION_TURNED") is Severity.ERROR


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0]])
def test_an_upside_down_knee_is_refused(axis):
    codes = _codes(*_knee(), _rotation(axis, 180.0))
    assert codes.get("SCAN_UPSIDE_DOWN") is Severity.ERROR


def test_a_knee_facing_backwards_is_refused():
    """Half a turn about the long axis: tilt and turn both look fine, as a line."""
    codes = _codes(*_knee(), _rotation([0, 0, 1], 180.0))
    assert "SCAN_ORIENTATION_TILTED" not in codes
    assert "SCAN_ORIENTATION_TURNED" not in codes
    assert codes.get("SCAN_FACING_BACKWARDS") is Severity.ERROR


def test_swapped_bones_are_refused():
    femur, tibia = _knee()
    report = check_scan_orientation(femoral_axes(tibia, Side.LEFT),
                                    tibial_axes(femur))
    assert "SCAN_UPSIDE_DOWN" in [f.code for f in report.errors]
