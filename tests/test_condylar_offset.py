"""Posterior condylar offset on a femur whose offsets are known by construction.

The femur is a point cloud: a straight shaft of radius 12 mm and two spherical condyles.
Its posterior cortex tangent is the shaft's posterior line, so each offset is the
condyle's posterior extent minus 12 mm, which is set in advance and then measured.
"""

import numpy as np
import pytest

from tka_planner.core.condylar_offset import (
    planned_condylar_offset,
    posterior_condylar_offset_metrics,
    posterior_cortex_line,
)
from tka_planner.core.frames import AnatomicalFrame
from tka_planner.core.landmarks import Landmark, LandmarkSet, LandmarkStatus
from tka_planner.core.provenance import Quality
from tka_planner.core.sides import Side

SHAFT_RADIUS = 12.0


def _sphere(centre, radius, n=3000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    unit = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    return np.asarray(centre) + radius * unit


def _femur(side="left", medial_radius=20.0, lateral_radius=18.0):
    """LPS: +X patient-left, +Y posterior, +Z proximal. Each condyle's centre sits 8 mm
    behind the shaft axis, so its offset is ``8 + radius - 12``."""
    side = Side.parse(side)
    medial = side.medial_direction(np.array([1.0, 0.0, 0.0]))
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    rings = [np.c_[SHAFT_RADIUS * np.cos(angles), SHAFT_RADIUS * np.sin(angles),
                   np.full_like(angles, z)] for z in np.arange(30.0, 200.0, 1.0)]
    medial_condyle = _sphere(20.0 * medial + [0.0, 8.0, 0.0], medial_radius)
    lateral_condyle = _sphere(-20.0 * medial + [0.0, 8.0, 0.0], lateral_radius)
    return np.vstack(rings + [medial_condyle, lateral_condyle]), medial_condyle


def _landmarks(side="left", status=LandmarkStatus.PRESENT, canal=True):
    side = Side.parse(side)
    medial = side.medial_direction(np.array([1.0, 0.0, 0.0]))
    points = {
        "femur.notch_centre": [0.0, 0.0, 0.0],
        "femur.epicondyle_lateral": list(-40.0 * medial + [0.0, 0.0, 15.0]),
        "femur.epicondyle_medial_prominence": list(40.0 * medial + [0.0, 0.0, 15.0]),
    }
    if canal:
        points["femur.canal_centre_distal"] = [0.0, 0.0, 70.0]
        points["femur.canal_centre_proximal"] = [0.0, 0.0, 190.0]
    return LandmarkSet("CASE_PCO", side, "LPS", [
        Landmark(id=k, position_mm=np.asarray(v, dtype=float), status=status,
                 origin="synthetic") for k, v in points.items()])


def _frame(side="left", rotation=np.eye(3), offset=np.zeros(3)):
    return AnatomicalFrame(
        bone="femur", side=Side.parse(side), origin=offset,
        x_anterior=rotation @ np.array([0.0, -1.0, 0.0]),
        y_patient_left=rotation @ np.array([1.0, 0.0, 0.0]),
        z_proximal=rotation @ np.array([0.0, 0.0, 1.0]),
        method="synthetic", quality=Quality.MEASURED,
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_each_compartment_offset_is_the_condyle_behind_the_shaft_tangent(side):
    points, _ = _femur(side)
    metrics = posterior_condylar_offset_metrics(points, _landmarks(side), _frame(side))
    assert metrics["pco_medial_mm"].value == pytest.approx(8.0 + 20.0 - 12.0, abs=0.05)
    assert metrics["pco_lateral_mm"].value == pytest.approx(8.0 + 18.0 - 12.0, abs=0.05)


def test_a_straight_shaft_gives_a_tangent_on_its_posterior_line():
    points, _ = _femur()
    line = posterior_cortex_line(points, _landmarks(), _frame())
    assert line.diagnostics["tangent_angle_to_canal_axis_deg"] == pytest.approx(0.0,
                                                                                abs=0.05)
    assert line.distance(np.array([[0.0, SHAFT_RADIUS, 120.0]]))[0] == pytest.approx(
        0.0, abs=0.05)


def test_the_offset_does_not_depend_on_where_the_scanner_put_the_knee():
    points, _ = _femur()
    landmarks = _landmarks()
    reference = posterior_condylar_offset_metrics(points, landmarks, _frame())

    angle = np.radians(27.0)
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    shift = np.array([80.0, -20.0, -550.0])
    moved = LandmarkSet("CASE_PCO", "left", "LPS", [
        Landmark(id=one.id, position_mm=rotation @ one.position_mm + shift,
                 status=one.status, origin="synthetic") for one in landmarks])
    metrics = posterior_condylar_offset_metrics(
        points @ rotation.T + shift, moved, _frame(rotation=rotation, offset=shift))
    for name in ("pco_medial_mm", "pco_lateral_mm"):
        assert metrics[name].value == pytest.approx(reference[name].value, abs=1e-6)


def test_a_component_two_millimetres_further_back_gains_two_millimetres():
    points, medial_condyle = _femur()
    component = np.vstack([medial_condyle + [0.0, 2.0, 0.0],
                           _sphere([20.0, 8.0, 0.0], 18.0)])  # lateral as it was
    record = planned_condylar_offset(points, component, _landmarks(), _frame())
    assert record["medial"]["change_mm"] == pytest.approx(2.0, abs=0.05)
    assert record["lateral"]["change_mm"] == pytest.approx(0.0, abs=0.05)
    assert record["radiographic"]["planned_mm"] == pytest.approx(18.0, abs=0.05)


def test_picked_landmarks_give_a_measured_offset_and_estimated_ones_an_estimate():
    points, _ = _femur()
    picked = posterior_condylar_offset_metrics(points, _landmarks(), _frame())
    guessed = posterior_condylar_offset_metrics(
        points, _landmarks(status=LandmarkStatus.ESTIMATED), _frame())
    assert picked["pco_medial_mm"].quality is Quality.MEASURED
    assert guessed["pco_medial_mm"].quality is Quality.ESTIMATED
    assert guessed["pco_medial_mm"].assumptions[0].id == "automatic_landmark_estimate"


def test_without_the_canal_the_offset_is_not_computable_and_says_why():
    points, _ = _femur()
    metrics = posterior_condylar_offset_metrics(points, _landmarks(canal=False),
                                                _frame())
    metric = metrics["pco_medial_mm"]
    assert metric.value is None
    assert metric.quality is Quality.NOT_COMPUTABLE
    assert "femur.canal_centre_distal" in metric.missing


def test_too_little_shaft_is_refused_rather_than_fitted():
    points, _ = _femur()
    short = points[(points[:, 2] < 76.0)]   # 6 mm of shaft above the distal canal level
    metrics = posterior_condylar_offset_metrics(short, _landmarks(), _frame())
    assert metrics["pco_lateral_mm"].quality is Quality.NOT_COMPUTABLE
