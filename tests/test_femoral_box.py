"""The femoral cut box, placed on constructed femurs whose anatomy is known."""

import numpy as np
import pytest

from tka_planner.core.femoral_box import (
    BoxDesign,
    _trochlea_top,
    femoral_cut_box,
    split_box,
)
from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm

IDENTITY = np.eye(4)
# A made-up design, deliberately unlike any real one. The real design is private and
# lives with the implant library.
DESIGN = BoxDesign(anterior_flare_deg=7.0, chamfer_deg=40.0,
                   posterior_condyle_thickness_mm=8.0, flange_thickness_mm=3.5,
                   distal_face_fraction=0.45, posterior_chamfer_fraction=0.25)
FLARE = np.tan(np.radians(7.0))


def combined(*meshes):
    vertices, faces, offset = [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def femur(*, ridge_x=34.0, ridge_top=40.0, cortex_x=30.0, medial_back=-33.0,
          lateral_back=-30.0, length=150.0):
    """A distal femur in the component frame, cut flat at z = 0.

    +X anterior, +Y patient-left (medial on a right knee), +Z proximal. The shaft's
    anterior cortex runs straight up at X = ``cortex_x``; the trochlea stands
    ``ridge_x - cortex_x`` proud of it up to ``ridge_top``; the two posterior condyles
    reach back to ``medial_back`` (+Y) and ``lateral_back`` (-Y).
    """
    shaft = gm.box((cortex_x + 15.0, 40.0, length),
                   gm.translation(((cortex_x - 15.0) / 2, 0.0, length / 2)))
    trochlea = gm.box((ridge_x + 15.0, 76.0, ridge_top),
                      gm.translation(((ridge_x - 15.0) / 2, 0.0, ridge_top / 2)))
    medial = gm.box((-medial_back, 33.0, 30.0),
                    gm.translation((medial_back / 2, 21.5, 15.0)))
    lateral = gm.box((-lateral_back, 33.0, 30.0),
                     gm.translation((lateral_back / 2, -21.5, 15.0)))
    return combined(shaft, trochlea, medial, lateral)


def test_the_posterior_cut_takes_the_thickness_off_the_more_prominent_condyle():
    box = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)

    assert box.posterior_cut_x_mm == pytest.approx(-33.0 + 8.0, abs=0.3)
    assert box.posterior_medial_mm == pytest.approx(8.0, abs=0.3)
    assert box.posterior_lateral_mm == pytest.approx(5.0, abs=0.3)


def test_the_more_prominent_condyle_is_found_on_either_side():
    lateral_prominent = femur(medial_back=-30.0, lateral_back=-33.0)
    right = femoral_cut_box(lateral_prominent, IDENTITY, medial_sign=1.0, design=DESIGN)
    left = femoral_cut_box(lateral_prominent, IDENTITY, medial_sign=-1.0, design=DESIGN)

    assert right.posterior_lateral_mm == pytest.approx(8.0, abs=0.3)
    assert right.posterior_medial_mm == pytest.approx(5.0, abs=0.3)
    # On a left knee patient-left (+Y) is lateral, so the labels swap.
    assert left.posterior_medial_mm == pytest.approx(8.0, abs=0.3)
    assert left.posterior_lateral_mm == pytest.approx(5.0, abs=0.3)


def test_the_anterior_cut_leaves_the_bone_where_the_trochlea_meets_the_shaft():
    box = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)

    assert box.flange_height_mm == pytest.approx(40.0, abs=1.5)
    assert box.anterior_cut_x_at(box.flange_height_mm) == pytest.approx(30.0, abs=0.3)


def test_the_anterior_cut_keeps_the_design_flare():
    box = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)

    assert box.anterior_flare_deg == pytest.approx(7.0)
    # The cut runs through the cortex at the flange height and 40 mm lower sits
    # 40 * tan(flare) further back: the design's angle, not one fitted to the bone.
    assert (box.anterior_cut_x_at(box.flange_height_mm)
            - box.anterior_cut_x_at(box.flange_height_mm - 40.0)) == pytest.approx(
        40.0 * FLARE, abs=1e-6)


def test_the_anterior_resection_and_the_patellofemoral_lowering():
    box = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)

    # Deepest at the distal end, where the flared plane is furthest back.
    assert box.anterior_mm == pytest.approx(34.0 - box.anterior_cut_x_at(0.0), abs=0.3)
    assert box.anterior_mm == pytest.approx(4.0 + 40.0 * FLARE, abs=0.5)
    assert box.patellofemoral_lowering_mm == pytest.approx(box.anterior_mm - 3.5)


def test_the_anterior_cut_does_not_notch_a_straight_shaft():
    box = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)

    assert box.anterior_notch_mm <= 0.3


def test_a_cortex_standing_forwards_of_the_cut_is_notched():
    """Above the flange the cortex comes forwards faster than the cut flares, so the
    cut runs into it: that is a notch, and it is measured."""
    stepped = combined(
        femur(),
        gm.box((10.0, 40.0, 80.0), gm.translation((30.0, 0.0, 110.0))),
    )
    box = femoral_cut_box(stepped, IDENTITY, medial_sign=1.0, design=DESIGN)

    assert box.anterior_notch_mm > 0.5


def test_the_box_is_split_in_the_design_proportions():
    split = split_box(-20.0, 30.0, DESIGN)

    assert split["box_ap_mm"] == pytest.approx(50.0)
    assert split["distal_face_mm"] == pytest.approx(22.5)
    assert split["posterior_chamfer_mm"] == pytest.approx(12.5)
    # The anterior chamfer rises from the front of the distal face (x = 15) at the
    # chamfer angle until it meets the flared anterior cut.
    run = 1.0 / np.tan(np.radians(40.0))
    assert split["anterior_chamfer_mm"] == pytest.approx(15.0 / (run - FLARE))
    assert split["cad_origin_x_mm"] == pytest.approx(-20.0 + 12.5 + 11.25)


def test_the_box_moves_with_the_bone():
    shifted = gm.transformed(femur(), gm.translation((5.0, 0.0, 0.0)))
    base = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)
    moved = femoral_cut_box(shifted, IDENTITY, medial_sign=1.0, design=DESIGN)

    assert moved.posterior_cut_x_mm == pytest.approx(base.posterior_cut_x_mm + 5.0,
                                                     abs=0.3)
    assert moved.cad_origin_x_mm == pytest.approx(base.cad_origin_x_mm + 5.0, abs=0.3)


def test_the_box_is_measured_in_the_component_s_own_frame():
    """A femur placed anywhere in the scan, with the component posed on it, gives the
    box it gives in the component frame."""
    turn = np.radians(35.0)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(turn), -np.sin(turn), 0.0],
                    [np.sin(turn), np.cos(turn), 0.0],
                    [0.0, 0.0, 1.0]]
    pose[:3, :3] = pose[:3, :3] @ [[1.0, 0.0, 0.0],
                                   [0.0, np.cos(0.3), -np.sin(0.3)],
                                   [0.0, np.sin(0.3), np.cos(0.3)]]
    pose[:3, 3] = [40.0, -120.0, 900.0]
    local = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN)
    placed = femoral_cut_box(gm.transformed(femur(), pose), pose, medial_sign=1.0,
                             design=DESIGN)

    for name in ("posterior_cut_x_mm", "anterior_cut_x_at_distal_mm",
                 "flange_height_mm", "posterior_medial_mm", "posterior_lateral_mm",
                 "anterior_mm", "cad_origin_x_mm"):
        assert getattr(placed, name) == pytest.approx(getattr(local, name), abs=0.3), name


def test_a_scan_too_short_for_the_shaft_is_refused():
    with pytest.raises(ValueError, match="anterior cortex"):
        femoral_cut_box(femur(length=70.0), IDENTITY, medial_sign=1.0, design=DESIGN)


def test_a_femur_with_no_trochlea_is_refused():
    flat = femur(ridge_x=30.0)
    with pytest.raises(ValueError, match="trochlea"):
        femoral_cut_box(flat, IDENTITY, medial_sign=1.0, design=DESIGN)


def smooth_trochlea(noise_mm=0.0, seed=0):
    """An anterior profile whose trochlea fades smoothly into a straight shaft: a peak
    of 8 mm at 25 mm, decaying so that it is 10 % of its peak at 45 mm."""
    heights = np.arange(0.0, 101.0)
    decay = np.log(10.0) / 20.0
    excess = np.where(heights < 25.0, 8.0 * heights / 25.0,
                      8.0 * np.exp(-decay * (heights - 25.0)))
    profile = 30.0 + 0.05 * heights + excess
    profile += np.random.default_rng(seed).normal(0.0, noise_mm, len(heights))
    return heights, profile


def test_a_smooth_trochlea_ends_where_it_has_faded_to_a_tenth_of_its_height():
    heights, profile = smooth_trochlea()

    index, _, _ = _trochlea_top(heights, profile)

    assert heights[index] == pytest.approx(45.0, abs=1.0)


def test_the_trochlea_top_is_not_thrown_by_noise():
    for seed in range(10):
        heights, profile = smooth_trochlea(noise_mm=0.15, seed=seed)
        index, _, _ = _trochlea_top(heights, profile)
        assert heights[index] == pytest.approx(45.0, abs=3.0), seed


def test_an_empty_band_is_stepped_over():
    heights, profile = smooth_trochlea()
    profile[50] = np.nan

    index, _, _ = _trochlea_top(heights, profile)

    assert heights[index] == pytest.approx(45.0, abs=1.0)


def test_the_record_names_the_cad_parameters_it_drives():
    record = femoral_cut_box(femur(), IDENTITY, medial_sign=1.0, design=DESIGN).to_dict()

    assert set(record["cad_parameters"]) == {
        "C_Distal_cut_surface", "B_Posterior_chamfer_projection",
        "D_Anterior_chamfer_projection"}
    assert record["cad_parameters"]["C_Distal_cut_surface"] == record["distal_face_mm"]
    assert set(record["resections_mm"]) == {"posterior_medial", "posterior_lateral",
                                            "anterior"}
    assert len(record["trochlea_excess_mm"]) == len(record["trochlea_heights_mm"])


def test_a_design_is_read_from_a_dictionary():
    data = {"anterior_flare_deg": 7, "chamfer_deg": 40,
            "posterior_condyle_thickness_mm": 8, "flange_thickness_mm": 3.5,
            "distal_face_fraction": 0.45, "posterior_chamfer_fraction": 0.25,
            "comment": "extra keys are ignored"}

    assert BoxDesign.from_dict(data) == DESIGN


@pytest.mark.parametrize("change", [
    {"distal_face_fraction": 0.8},          # the fractions leave no room for a chamfer
    {"posterior_chamfer_fraction": -0.1},
    {"chamfer_deg": 85.0},                  # steeper than the flare can meet
    {"posterior_condyle_thickness_mm": 0.0},
])
def test_an_impossible_design_is_refused(change):
    values = {"anterior_flare_deg": 7.0, "chamfer_deg": 40.0,
              "posterior_condyle_thickness_mm": 8.0, "flange_thickness_mm": 3.5,
              "distal_face_fraction": 0.45, "posterior_chamfer_fraction": 0.25}
    with pytest.raises(ValueError):
        BoxDesign(**{**values, **change})
