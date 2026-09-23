"""Component fit on the cut surface, from constructed shapes with known answers."""

import numpy as np
import pytest

from tka_planner.core.fit import fill_section, section_fit, slice_mesh
from tka_planner.geom import mesh as gm

# LPS: +X patient-left, -Y anterior, +Z proximal. On a left knee +X is lateral.
LATERAL = (1.0, 0.0, 0.0)
ANTERIOR = (0.0, -1.0, 0.0)
UP = (0.0, 0.0, 1.0)


def tibia_block(ml=70.0, ap=50.0):
    """A bone whose section at z = 0 is an ML x AP rectangle.

    It runs on above the cut, as a real tibia does before the saw goes through it."""
    return gm.box((ml, ap, 50.0), gm.translation((0.0, 0.0, -15.0)))


def tray(ml=60.0, ap=40.0, shift=(0.0, 0.0)):
    """A 4 mm plate seated on z = 0, optionally slid in the plane."""
    return gm.box((ml, ap, 4.0), gm.translation((shift[0], shift[1], 2.0)))


def fit(bone, component, *, side=1.0, normal=UP):
    return section_fit(cut="tibial_proximal", bone=bone, component=component,
                       point=(0.0, 0.0, 0.0), normal=normal, component_side=side,
                       lateral=LATERAL, anterior=ANTERIOR)


def test_a_centred_smaller_tray_is_all_underhang():
    result = fit(tibia_block(), tray())

    assert result.bone_area_mm2 == pytest.approx(3500.0, rel=0.01)
    assert result.component_area_mm2 == pytest.approx(2400.0, rel=0.01)
    assert result.coverage_fraction == pytest.approx(2400.0 / 3500.0, abs=0.01)
    assert result.max_overhang_mm == 0.0
    assert result.flags == ()
    for zone in result.zones.values():
        # 5 mm at the sides and ends, the corners diagonal: sqrt(5^2 + 5^2).
        assert zone["max_underhang_mm"] == pytest.approx(np.hypot(5.0, 5.0), abs=0.3)


def test_a_tray_slid_anteriorly_overhangs_anteriorly_only():
    result = fit(tibia_block(), tray(shift=(0.0, -8.0)))  # -Y is anterior

    zones = result.zones
    assert zones["anteromedial"]["max_overhang_mm"] == pytest.approx(3.0, abs=0.3)
    assert zones["anterolateral"]["max_overhang_mm"] == pytest.approx(3.0, abs=0.3)
    assert zones["posteromedial"]["max_overhang_mm"] == 0.0
    assert zones["posterolateral"]["max_overhang_mm"] == 0.0
    assert set(result.flags) == {"OVERHANG_ANTEROMEDIAL", "OVERHANG_ANTEROLATERAL"}


def test_medial_and_lateral_follow_the_knee_not_the_scanner():
    """A tray slid towards patient-left overhangs laterally on a left knee."""
    result = fit(tibia_block(), tray(ml=66.0, shift=(6.0, 0.0)))

    assert result.zones["anterolateral"]["max_overhang_mm"] == pytest.approx(4.0, abs=0.3)
    assert result.zones["anteromedial"]["max_overhang_mm"] == 0.0


def test_the_component_side_selects_which_face_is_the_footprint():
    """The femoral component lies below its cut; its footprint is taken below."""
    bone = gm.box((70.0, 50.0, 40.0), gm.translation((0.0, 0.0, 20.0)))
    component = gm.box((60.0, 40.0, 4.0), gm.translation((0.0, 0.0, -2.0)))

    result = fit(bone, component, side=-1.0)
    assert result.component_area_mm2 == pytest.approx(2400.0, rel=0.01)

    with pytest.raises(ValueError, match="does not pass through"):
        fit(bone, component, side=1.0)


def test_a_hollow_section_is_filled_hollow():
    """Even-odd filling leaves a canal empty, as the notch between condyles must be."""
    outer = gm.box((40.0, 40.0, 10.0))
    inner = gm.box((20.0, 20.0, 10.0))
    segments = np.concatenate([slice_mesh(outer, (0, 0, 0), UP),
                               slice_mesh(inner, (0, 0, 0), UP)])[:, :, :2]
    xs = ys = np.arange(-25.0, 25.0, 0.5) + 0.2501

    mask = fill_section(segments, xs, ys)

    assert mask.sum() * 0.25 == pytest.approx(1600.0 - 400.0, rel=0.02)
    assert not mask[len(ys) // 2, len(xs) // 2]


def test_slicing_a_triangle_soup_gives_closed_segments():
    segments = slice_mesh(tibia_block(), (0.0, 0.0, -10.0), UP)

    assert len(segments) == 8  # two triangles per side face
    assert np.allclose(segments[:, :, 2], -10.0)


def test_the_fit_serialises():
    record = fit(tibia_block(), tray()).to_dict()

    assert record["method"] == "fit.section.v1"
    assert set(record["zones"]) == {"anteromedial", "anterolateral",
                                    "posteromedial", "posterolateral"}


def test_a_planned_case_reports_the_fit_of_both_components(session):
    from tka_planner.pipeline import fit_case

    result = fit_case(session.measurement, session.plan, session.sizing,
                      session.library)

    assert result["missing"] == []
    for name in ("tibial_component", "femoral_component"):
        assert 0.0 < result[name]["coverage_fraction"] <= 1.0
        assert result["library_sizes"][name]["scale"] > 0


def test_no_library_means_no_fit(session):
    from tka_planner.pipeline import fit_case

    assert fit_case(session.measurement, session.plan, session.sizing, None) is None
