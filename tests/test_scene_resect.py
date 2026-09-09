"""Committing the resection: the one place a boolean runs."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.scene import resect as sr
from tka_planner.scene.model import FEMORAL, TIBIAL, Node, Scene


def _plan_at(*, z: float = 0.0):
    """A stand-in plan with one femoral and one tibial resection plane."""
    from tka_planner.core.planning import ResectionPlane, SurgicalPlan
    from tka_planner.core.provenance import Quality

    def plane(name):
        return ResectionPlane(
            name=name, point=np.array([0.0, 0.0, z]),
            normal=np.array([0.0, 0.0, 1.0]),
            medial_depth_mm=9.0, lateral_depth_mm=9.0, reference="test",
        )

    return SurgicalPlan(
        philosophy="mechanical", distal_femoral_valgus_cut_deg=5.0,
        valgus_source="test", valgus_quality=Quality.MEASURED,
        tibial_slope_deg=3.0,
        resections={
            "femoral_distal": plane("femoral_distal"),
            "tibial_proximal": plane("tibial_proximal"),
        },
        components={"femoral_component": np.eye(4), "tibial_component": np.eye(4)},
    )


@pytest.fixture
def scene():
    built = Scene()
    built.add(Node(name="Femur", set_id=FEMORAL, tags={"bone": True}), gm.box(100.0))
    built.add(Node(name="Tibia", set_id=TIBIAL, tags={"bone": True}), gm.box(100.0))
    return built


@pytest.fixture
def blocked(scene):
    """A scene with a cutting block and its shell seated on the femur."""
    block = gm.box(40.0, gm.translation((0.0, 0.0, -10.0)))
    scene.add(
        Node(name="femoral_cutting_block", set_id=FEMORAL,
             tags={"component": True, "group": "femoral"}),
        block,
    )
    scene.add(
        Node(name="femoral_cutting_block_shell", set_id=FEMORAL,
             tags={"component": True, "group": "femoral", "shell": True}),
        block,
    )
    return scene


# ----------------------------------------------------------------------
# The discard box
# ----------------------------------------------------------------------


def test_a_cutter_keeping_proximal_sits_below_the_plane():
    cutter = sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="proximal")
    assert cutter.vertices[:, 2].max() == pytest.approx(0.0, abs=1e-9)


def test_a_cutter_keeping_distal_sits_above_the_plane():
    cutter = sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="distal")
    assert cutter.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)


def test_a_cutter_is_large_enough_to_swallow_a_bone():
    cutter = sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="proximal")
    assert cutter.extent[0] == pytest.approx(sr.CUTTER_SIZE_MM)


def test_a_cutter_follows_an_oblique_normal():
    normal = np.array([0.0, 0.6, 0.8])
    cutter = sr.cutter_for((0.0, 0.0, 0.0), normal, keep="proximal")
    centre = cutter.vertices.mean(axis=0)

    assert centre == pytest.approx(-normal * sr.CUTTER_SIZE_MM / 2.0)


def test_an_unknown_keep_side_is_refused():
    with pytest.raises(ValueError, match="proximal"):
        sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="sideways")


# ----------------------------------------------------------------------
# Cut plane mode
# ----------------------------------------------------------------------


def test_cutting_a_box_in_half_halves_its_volume(scene):
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Femur",), build_shells=False
    )
    assert gm.volume(result.scene.mesh_for("Femur")) == pytest.approx(
        500_000.0, rel=1e-4
    )


def test_the_femur_keeps_the_bone_above_its_plane(scene):
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Femur",), build_shells=False
    )
    assert result.scene.mesh_for("Femur").vertices[:, 2].min() == pytest.approx(
        0.0, abs=1e-6
    )


def test_the_tibia_keeps_the_bone_below_its_plane(scene):
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Tibia",), build_shells=False
    )
    assert result.scene.mesh_for("Tibia").vertices[:, 2].max() == pytest.approx(
        0.0, abs=1e-6
    )


def test_the_commit_records_the_kernel_for_every_bone_it_cut(scene):
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Femur",), build_shells=False
    )
    assert result.records["Femur"].backend == "manifold3d"
    assert result.records["Femur"].operation == "difference"


def test_the_commit_delta_replaces_the_bone_mesh(scene):
    before = scene.nodes["Femur"].mesh_id
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Femur",), build_shells=False
    )
    assert "Femur" in result.delta.meshes
    assert result.scene.nodes["Femur"].mesh_id != before


def test_only_the_named_bones_are_cut(scene):
    before = scene.nodes["Tibia"].mesh_id
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Femur",), build_shells=False
    )
    assert result.scene.nodes["Tibia"].mesh_id == before


def test_both_bones_are_cut_by_default(scene):
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", build_shells=False
    )
    assert set(result.records) == {"Femur", "Tibia"}


def test_the_commit_serialises_its_geometry_provenance_for_the_plan_file(scene):
    result = sr.commit_resection(
        scene, _plan_at(), mode="plane", bones=("Femur",), build_shells=False
    )
    payload = result.to_dict()

    assert payload["resections"]["Femur"]["backend"] == "manifold3d"
    assert payload["resections"]["Femur"]["fallback"] is None


def test_mode_none_cuts_nothing(scene):
    before = scene.nodes["Femur"].mesh_id
    result = sr.commit_resection(scene, _plan_at(), mode="none")

    assert result.scene.nodes["Femur"].mesh_id == before
    assert result.records == {}


def test_an_unknown_mode_is_refused(scene):
    with pytest.raises(ValueError, match="block or plane"):
        sr.commit_resection(scene, _plan_at(), mode="lasers")


# ----------------------------------------------------------------------
# Cutting block mode
# ----------------------------------------------------------------------


def test_block_mode_subtracts_the_block(blocked):
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=False
    )
    # The block is a 40 mm cube fully inside the 100 mm bone.
    assert gm.volume(result.scene.mesh_for("Femur")) == pytest.approx(
        1_000_000.0 - 64_000.0, rel=1e-6
    )


def test_block_mode_builds_the_patient_specific_shell(blocked):
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=True
    )
    assert "Femur.Shell" in result.scene.nodes
    assert gm.volume(result.scene.mesh_for("Femur.Shell")) == pytest.approx(
        64_000.0, rel=1e-6
    )


def test_the_shell_is_taken_before_the_bone_is_resected(blocked):
    """A shell cut from an already-resected femur would miss the surface it mates with.

    If the shell were taken after the difference, intersecting the block with a bone
    that no longer contains the block's volume would give nothing at all.
    """
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=True
    )
    shell = result.scene.mesh_for("Femur.Shell")

    assert shell is not None
    assert gm.volume(shell) > 0.0


def test_the_shell_belongs_to_its_bone_set(blocked):
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=True
    )
    assert result.scene.nodes["Femur.Shell"].set_id == FEMORAL


def test_shells_can_be_turned_off(blocked):
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=False
    )
    assert "Femur.Shell" not in result.scene.nodes


def test_the_shell_records_its_own_kernel_call(blocked):
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=True
    )
    assert result.records["Femur.Shell"].operation == "intersect"


def test_block_mode_without_a_block_leaves_the_bone_whole_and_says_so(scene):
    before = scene.nodes["Femur"].mesh_id
    result = sr.commit_resection(
        scene, _plan_at(), mode="block", bones=("Femur",), build_shells=False
    )

    assert result.scene.nodes["Femur"].mesh_id == before
    assert any("no cutting block" in note for note in result.notes)


def test_a_block_is_cut_at_its_posed_position(scene):
    """The tool is taken in world space, so moving a block moves what it removes."""
    scene.add(
        Node(name="femoral_cutting_block", set_id=FEMORAL,
             rest=gm.translation((0.0, 0.0, -10.0)),
             tags={"component": True, "group": "femoral"}),
        gm.box(40.0),
    )
    result = sr.commit_resection(
        scene, _plan_at(), mode="block", bones=("Femur",), build_shells=False
    )
    cut = result.scene.mesh_for("Femur")

    assert gm.volume(cut) == pytest.approx(1_000_000.0 - 64_000.0, rel=1e-6)


def test_the_two_modes_are_never_stacked(blocked):
    """Block mode must not also apply the plane, which would swallow the block's work."""
    result = sr.commit_resection(
        blocked, _plan_at(), mode="block", bones=("Femur",), build_shells=False
    )
    cut = result.scene.mesh_for("Femur")

    # A plane cut as well would have removed everything below z = 0.
    assert cut.vertices[:, 2].min() == pytest.approx(-50.0, abs=1e-6)
