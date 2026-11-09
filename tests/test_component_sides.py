"""A right knee gets the right-hand part of its own size.

The library exports the femoral component left-handed in every size, and right-handed
only in L4. Searching every size for a right-handed part therefore always found L4's and
rescaled it -- and L4 is not quite the family's shape (80 mm wide where the chart says 84),
so every right knee got a component about 5% too narrow. The right part is the left part
mirrored across the patient's sagittal plane (checked vertex for vertex on L4), so a right
knee now takes its own size's left part, mirrored.
"""

from __future__ import annotations

import numpy as np
import pytest

from tka_planner.core.sizing import load_size_chart
from tka_planner.geom import mesh as gm
from tka_planner.scene.build import read_component, resolve_component_meshes
from tka_planner.session import find_size_chart


class _Sizing:
    def __init__(self, label, ml):
        self.nearest_discrete_size = label
        self.implant_ml_mm = ml


@pytest.fixture
def chart():
    return load_size_chart(find_size_chart())


@pytest.fixture
def sided_library(tmp_path, chart):
    """M2 exports a left part only; L4 exports both hands, as the real library does."""
    root = tmp_path / "library"
    for label in chart.labels:
        (root / label).mkdir(parents=True)
    # An asymmetric block, so a mirror is visible: it reaches further to patient-left.
    left = gm.box((20.0, 30.0, 10.0))
    left = gm.transformed(left, np.array([[1, 0, 0, 0], [0, 1, 0, 5.0],
                                          [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float))
    gm.write_stl(left, root / "M2" / "Implant(Femoral)Left_M2.stl")
    gm.write_stl(left, root / "L4" / "Implant_Femoral_Left.stl")
    gm.write_stl(left, root / "L4" / "Implant_Femoral_Right.stl")
    return root


def test_a_right_knee_takes_its_own_size_mirrored(sided_library, chart):
    ml = chart.value_at("femur_ML", chart.label_parameter("M2"))
    spec = resolve_component_meshes(sided_library, chart=chart,
                                    sizing=_Sizing("M2", ml), side="right")
    femoral = spec["femoral_component"]
    assert femoral["source_size"] == "M2"
    assert "left" in femoral["path"].lower()
    assert femoral["mirror"] is True
    assert femoral["scale"] == pytest.approx(1.0)


def test_a_left_knee_is_not_mirrored(sided_library, chart):
    ml = chart.value_at("femur_ML", chart.label_parameter("M2"))
    spec = resolve_component_meshes(sided_library, chart=chart,
                                    sizing=_Sizing("M2", ml), side="left")
    assert spec["femoral_component"]["source_size"] == "M2"
    assert not spec["femoral_component"].get("mirror")


def test_an_exported_right_part_is_used_as_it_is(sided_library, chart):
    ml = chart.value_at("femur_ML", chart.label_parameter("L4"))
    spec = resolve_component_meshes(sided_library, chart=chart,
                                    sizing=_Sizing("L4", ml), side="right")
    assert "right" in spec["femoral_component"]["path"].lower()
    assert not spec["femoral_component"].get("mirror")


def test_a_mirrored_part_is_reflected_across_the_patients_sagittal_plane(sided_library,
                                                                          chart):
    ml = chart.value_at("femur_ML", chart.label_parameter("M2"))
    spec = resolve_component_meshes(sided_library, chart=chart,
                                    sizing=_Sizing("M2", ml), side="right")
    left = read_component(dict(spec["femoral_component"], mirror=False))
    right = read_component(spec["femoral_component"])
    # Femoral CAD +Y is patient-left: the mirror negates it and nothing else.
    np.testing.assert_allclose(right.vertices, left.vertices * [1.0, -1.0, 1.0])
    # A reflection turns the triangles inside out unless their winding is reversed.
    assert _signed_volume(right) == pytest.approx(_signed_volume(left))
    assert _signed_volume(left) > 0


def _signed_volume(mesh) -> float:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
