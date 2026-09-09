"""The planning session, driven with no Blender and no browser."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.geom import mesh as gm
from tka_planner.scene.model import FEMORAL, TIBIAL
from tka_planner.session import ADJUSTMENT_FIELDS, PlanningSession, find_bone_files


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
def session(folder, library, monkeypatch):
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
    opened = PlanningSession.open(folder, side="left", library=library)
    opened.build()
    return opened


# ----------------------------------------------------------------------
# Opening
# ----------------------------------------------------------------------


def test_the_project_naming_convention_is_found(folder):
    femur, tibia = find_bone_files(folder, "left")
    assert femur.name == "FD1Left.stl"
    assert tibia.name == "TD1Left.stl"


def test_files_named_for_the_bone_are_found_too(tmp_path):
    gm.write_stl(gm.box(10.0), tmp_path / "Femur_segmentation.stl")
    gm.write_stl(gm.box(10.0), tmp_path / "Tibia_segmentation.stl")

    femur, tibia = find_bone_files(tmp_path, "left")
    assert "femur" in femur.name.lower()
    assert "tibia" in tibia.name.lower()


def test_a_folder_without_bones_says_what_it_expected(tmp_path):
    with pytest.raises(FileNotFoundError, match="FD1Left.stl"):
        find_bone_files(tmp_path, "left")


# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------


def test_all_thirteen_adjustments_are_exposed():
    assert len(ADJUSTMENT_FIELDS) == 13
    assert "coronal_correction_deg" in ADJUSTMENT_FIELDS
    assert "tibial_shift_ml_mm" in ADJUSTMENT_FIELDS


def test_a_fresh_session_reports_the_plan_as_computed(session):
    assert session.plan.adjustments.is_identity


def test_every_adjustment_can_be_set_and_reaches_the_plan(session):
    for name in ADJUSTMENT_FIELDS:
        session.replan(**{name: 1.5})
        assert getattr(session.plan.adjustments, name) == pytest.approx(1.5), name
        session.reset_adjustments()


def test_every_adjustment_moves_something_in_the_scene(session):
    for name in ADJUSTMENT_FIELDS:
        delta = session.replan(**{name: 2.0})
        assert delta.poses, name
        session.reset_adjustments()


def test_replanning_marks_the_plan_adjusted(session):
    session.replan(tibial_resection_delta_mm=2.0)
    assert not session.plan.adjustments.is_identity


def test_reset_adjustments_returns_to_the_computed_plan(session):
    before = session.scene.world("tibial_proximal").copy()
    session.replan(tibial_resection_delta_mm=2.0)
    session.reset_adjustments()

    assert session.scene.world("tibial_proximal") == pytest.approx(before)
    assert session.plan.adjustments.is_identity


def test_an_unknown_control_is_refused(session):
    with pytest.raises(ValueError, match="Unknown control"):
        session.replan(nonsense_mm=1.0)


def test_the_alignment_philosophy_can_be_switched(session):
    session.replan(philosophy="kinematic")
    assert session.plan.philosophy == "kinematic"


def test_the_tibial_resection_depth_is_a_control(session):
    before = session.scene.world("tibial_proximal").copy()
    session.replan(tibial_resection_mm=12.0)

    assert session.scene.world("tibial_proximal")[:3, 3] != pytest.approx(
        before[:3, 3]
    )


# ----------------------------------------------------------------------
# Which bones a change reaches
# ----------------------------------------------------------------------


def test_a_femoral_change_affects_only_the_femur(session):
    assert session.bones_affected_by(("femoral_resection_delta_mm",)) == ("Femur",)


def test_a_tibial_change_affects_only_the_tibia(session):
    assert session.bones_affected_by(("tibial_slope_delta_deg",)) == ("Tibia",)


def test_a_coronal_correction_affects_both_bones(session):
    assert set(session.bones_affected_by(("coronal_correction_deg",))) == {
        "Femur", "Tibia"
    }


def test_an_insert_change_affects_no_bone(session):
    assert session.bones_affected_by(("insert_thickness_mm",)) == ()


def test_a_shift_is_free_in_plane_mode(session):
    """In plane mode a component's own pose moves nothing a cutter depends on."""
    session.replan(resection_mode="plane")
    assert session.bones_affected_by(("femoral_shift_ap_mm",)) == ()


def test_a_shift_is_not_free_in_block_mode(session):
    """The block shares its implant's pose exactly, so moving one moves the cutter."""
    session.replan(resection_mode="block")
    assert session.bones_affected_by(("femoral_shift_ap_mm",)) == ("Femur",)


# ----------------------------------------------------------------------
# Reduce
# ----------------------------------------------------------------------


def test_the_trial_controls_move_the_tibial_set_only(session):
    session.set_trial(flexion_deg=30.0)

    assert not np.allclose(session.scene.set_poses[TIBIAL], np.eye(4))
    assert session.scene.set_poses[FEMORAL] == pytest.approx(np.eye(4))


def test_resetting_the_trial_returns_the_tibial_set_to_identity(session):
    session.set_trial(flexion_deg=45.0, varus_valgus_deg=5.0, drawer_ap_mm=3.0)
    session.reset_trial()

    assert session.scene.set_poses[TIBIAL] == pytest.approx(np.eye(4))


def test_the_trial_does_not_change_the_plan(session):
    session.set_trial(flexion_deg=45.0)
    assert session.plan.adjustments.is_identity


def test_the_trial_does_not_move_a_rest_transform(session):
    before = session.scene.nodes["Tibia"].rest.copy()
    session.set_trial(flexion_deg=45.0)

    assert session.scene.nodes["Tibia"].rest == pytest.approx(before)


def test_an_unknown_trial_control_is_refused(session):
    with pytest.raises(ValueError, match="Unknown trial control"):
        session.set_trial(twist_deg=5.0)


def test_the_trial_delta_names_the_tibial_set(session):
    delta = session.set_trial(flexion_deg=10.0)
    assert "TibialSet" in delta.poses


# ----------------------------------------------------------------------
# Commit
# ----------------------------------------------------------------------


def test_a_freshly_built_session_is_stale(session):
    assert session.stale is True


def test_committing_clears_the_stale_flag(session):
    session.commit()
    assert session.stale is False


def test_changing_the_plan_after_a_commit_makes_it_stale(session):
    session.commit()
    session.replan(tibial_resection_delta_mm=1.0)

    assert session.stale is True


def test_an_insert_change_after_a_commit_does_not_make_it_stale(session):
    session.commit()
    session.replan(insert_thickness_mm=11.0)

    assert session.stale is False


def test_committing_again_clears_the_stale_flag(session):
    session.commit()
    session.replan(tibial_resection_delta_mm=1.0)
    session.commit()

    assert session.stale is False


def test_a_commit_records_the_kernel_it_used(session):
    session.replan(resection_mode="plane")
    result = session.commit()

    assert result.records
    assert all(
        record.backend == "manifold3d" for record in result.records.values()
    )


def test_a_commit_actually_cuts_the_bone(session):
    session.replan(resection_mode="plane")
    before = gm.volume(session.scene.mesh_for("Femur"))
    session.commit()

    assert gm.volume(session.scene.mesh_for("Femur")) < before


def test_a_commit_can_be_limited_to_one_bone(session):
    session.replan(resection_mode="plane")
    before = session.scene.nodes["Tibia"].mesh_id
    session.commit(bones=("Femur",))

    assert session.scene.nodes["Tibia"].mesh_id == before


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def test_report_lines_carry_the_correction_and_the_resections(session):
    text = "\n".join(session.report_lines())

    assert "Valgus cut|" in text
    assert "Femoral medial|" in text
    assert "Tibial lateral|" in text


def test_report_lines_carry_the_case_and_the_sizing(session):
    text = "\n".join(session.report_lines())

    assert "CASE|CASE_TEST (left)" in text
    assert "Nearest size|" in text


def test_report_lines_warn_when_hka_is_not_computable(session):
    text = "\n".join(session.report_lines())
    assert "WARN|HKA needs hip and ankle" in text


def test_report_lines_follow_the_plan(session):
    session.replan(coronal_correction_deg=3.0)
    text = "\n".join(session.report_lines())

    assert f"Valgus cut|{session.plan.distal_femoral_valgus_cut_deg:.1f} deg" in text


# ----------------------------------------------------------------------
# The whole point
# ----------------------------------------------------------------------


def test_the_session_imports_no_bpy(session):
    import sys

    session.replan(femoral_varus_delta_deg=1.0)
    session.set_trial(flexion_deg=20.0)
    session.commit()

    assert "bpy" not in sys.modules
