"""Alignment planning and sizing measurement.

The resection-depth sign convention gets the most attention here. The distal femur is
cut from below and the proximal tibia from above, so "how much bone is removed" has
opposite geometry on the two sides. Applying one convention to both produces negative
tibial depths -- which is what happened on the first real run, and is why the depths are
asserted positive on both bones from constructed geometry.
"""

import numpy as np
import pytest

from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.planning import (
    KINEMATIC,
    MECHANICAL,
    plan_alignment,
)
from tka_planner.core.provenance import Quality
from tests.synthetic import mirror_landmarks, synthetic_knee
from tests.test_frames import as_estimated


def frames_for(landmarks):
    return build_femoral_frame(landmarks), build_tibial_frame(landmarks)


def leaned_tibia(landmarks, lean_deg):
    """Rotate every tibial and fibular landmark about the ML axis through the joint.

    Positive flexes the tibia: its distal end swings posteriorly.
    """
    from tka_planner.core.landmarks import Landmark, LandmarkSet

    pivot = np.mean([landmarks.position("tibia.spine_medial"),
                     landmarks.position("tibia.spine_lateral")], axis=0)
    theta = np.radians(lean_deg)
    # About LPS +X; with +Y posterior and +Z superior, a positive angle carries the
    # distal (-Z) end towards +Y.
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(theta), -np.sin(theta)],
                         [0.0, np.sin(theta), np.cos(theta)]])
    rebuilt = []
    for lm in landmarks:
        if lm.position_mm is not None and lm.id.startswith(("tibia.", "fibula.")):
            lm = Landmark(id=lm.id, position_mm=pivot + rotation @ (lm.position_mm - pivot),
                          status=lm.status, origin=lm.origin, reason=lm.reason,
                          metadata=dict(lm.metadata))
        rebuilt.append(lm)
    return LandmarkSet(case_id=landmarks.case_id, side=landmarks.side,
                       coordinate_system=landmarks.coordinate_system, landmarks=rebuilt)


def plan_for(landmarks, **kwargs):
    femoral, tibial = frames_for(landmarks)
    # The commercial default: 9 mm distal femur, 9 mm below the less affected plateau.
    kwargs.setdefault("femoral_thickness_mm", 9.0)
    kwargs.setdefault("tibial_resection_mm", 9.0)
    return plan_alignment(landmarks, femoral, tibial, **kwargs)


class TestResectionDepths:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_both_bones_report_positive_depths(self, side):
        """The bug the first real run exposed.

        Removed bone lies distal to the femoral cut and proximal to the tibial one, so a
        single sign convention gets one of them backwards.
        """
        plan = plan_for(synthetic_knee(side), tibial_resection_mm=10.0)

        for name, resection in plan.resections.items():
            assert resection.medial_depth_mm > 0, f"{name} medial depth is negative"
            assert resection.lateral_depth_mm > 0, f"{name} lateral depth is negative"

    @pytest.mark.parametrize("thickness", [8.0, 9.0, 11.0])
    def test_femoral_depth_equals_component_thickness_at_the_lower_condyle(
        self, thickness
    ):
        """The cut is seated so the deeper compartment loses exactly the implant's
        thickness; the other loses less, by the joint line's obliquity."""
        plan = plan_for(
            synthetic_knee("left", mldfa_deg=87.0),
            femoral_thickness_mm=thickness,
        )
        femoral = plan.resections["femoral_distal"]

        assert max(femoral.medial_depth_mm,
                   femoral.lateral_depth_mm) == pytest.approx(thickness, abs=0.01)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("resection", [8.0, 9.0, 10.0])
    def test_the_default_is_measured_from_the_less_affected_plateau(self, side,
                                                                    resection):
        """Where a commercial stylus rests: the lowest point of the higher plateau."""
        plan = plan_for(synthetic_knee(side, mpta_deg=85.0),
                        tibial_resection_mm=resection)
        tibial = plan.resections["tibial_proximal"]

        assert max(tibial.medial_depth_mm,
                   tibial.lateral_depth_mm) == pytest.approx(resection, abs=0.01)
        assert plan.diagnostics["tibial_reference"] == "less_affected_plateau"

    def test_the_more_affected_plateau_can_be_the_reference(self):
        """The other stylus setting: a small depth below the worn side."""
        plan = plan_for(synthetic_knee("left", mpta_deg=85.0),
                        tibial_reference="more_affected_plateau",
                        tibial_resection_mm=2.0)
        tibial = plan.resections["tibial_proximal"]

        assert min(tibial.medial_depth_mm,
                   tibial.lateral_depth_mm) == pytest.approx(2.0, abs=0.01)

    def test_an_unknown_reference_is_refused(self):
        with pytest.raises(ValueError, match="tibial_reference"):
            plan_for(synthetic_knee("left"), tibial_reference="plateau")

    @pytest.mark.parametrize("resection", [16.0, 21.0, 22.0])
    def test_the_legacy_datum_is_the_top_of_the_tibia(self, resection):
        """The size chart's tibial depth runs from the top of the tibia, the eminence,
        not from the plateau -- the datum the legacy pipeline used."""
        landmarks = synthetic_knee("left", mpta_deg=85.0)
        plan = plan_for(landmarks, tibial_resection_mm=resection,
                        tibial_reference="top_of_tibia")
        tibial = plan.resections["tibial_proximal"]

        spines = np.array([landmarks.position("tibia.spine_medial"),
                           landmarks.position("tibia.spine_lateral")])
        top = spines[np.argmax(spines @ tibial.normal)]
        assert np.dot(top - tibial.point, tibial.normal) == pytest.approx(
            resection, abs=0.01)
        # Measured from the plateau the same cut reads shallower, by the eminence height.
        assert max(tibial.medial_depth_mm, tibial.lateral_depth_mm) < resection

    def test_a_tibia_mesh_sets_the_datum_at_its_highest_point(self):
        """With the bone present, the datum is the mesh's top along the cut normal, which
        can sit above the picked spines."""
        from tka_planner.core.meshio import Mesh

        landmarks = synthetic_knee("left")
        plan = plan_for(landmarks)
        normal = plan.resections["tibial_proximal"].normal
        spine = np.array(landmarks.position("tibia.spine_medial"))
        peak = spine + 3.0 * normal
        mesh = Mesh(vertices=np.array([spine, peak, spine - 40.0 * normal]),
                    faces=np.array([[0, 1, 2]]))

        with_mesh = plan_for(landmarks, tibia_mesh=mesh, tibial_resection_mm=21.0,
                             tibial_reference="top_of_tibia")
        tibial = with_mesh.resections["tibial_proximal"]

        assert with_mesh.diagnostics["tibial_resection_datum_source"] == "mesh"
        assert np.dot(peak - tibial.point, tibial.normal) == pytest.approx(21.0, abs=0.01)

    def test_without_spines_or_mesh_the_plateau_stands_in(self):
        """The tibial frame itself needs the spines, so this fallback is only reachable
        through the datum helper; it must still pick the higher plateau."""
        from tka_planner.core.landmarks import LandmarkSet
        from tka_planner.core.planning import _tibial_proximal_point

        full = synthetic_knee("left", mpta_deg=85.0)
        plateaus = ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest")
        landmarks = LandmarkSet(
            case_id=full.case_id, side=full.side,
            coordinate_system=full.coordinate_system,
            landmarks=[lm for lm in full if lm.id in plateaus],
        )
        normal = plan_for(full).resections["tibial_proximal"].normal

        point, source = _tibial_proximal_point(landmarks, normal)

        assert source == "plateau landmarks"
        heights = [np.dot(full.position(i), normal) for i in plateaus]
        assert np.dot(point, normal) == pytest.approx(max(heights))

    def test_the_plan_states_both_datums(self):
        diagnostics = plan_for(synthetic_knee("left", mpta_deg=85.0)).diagnostics

        assert diagnostics["tibial_resection_datum"].startswith(
            "the lowest point of the less affected")
        assert diagnostics["tibial_resection_from_datum_mm"] == 9.0
        assert diagnostics["femoral_resection_from_datum_mm"] == 9.0

    def test_a_valgus_femur_resects_more_medially(self):
        """mLDFA below 90 means the lateral condyle sits proximal, so it loses less."""
        plan = plan_for(synthetic_knee("left", mldfa_deg=84.0))
        femoral = plan.resections["femoral_distal"]

        assert femoral.medial_depth_mm > femoral.lateral_depth_mm

    def test_a_symmetric_joint_line_resects_evenly(self):
        plan = plan_for(synthetic_knee("left", mldfa_deg=90.0, mpta_deg=90.0))

        for resection in plan.resections.values():
            assert resection.medial_depth_mm == pytest.approx(
                resection.lateral_depth_mm, abs=0.01
            )


class TestValgusCutAngle:
    @pytest.mark.parametrize("ama_deg", [4.0, 6.0, 8.0])
    def test_measured_from_the_patient_when_the_head_is_present(self, ama_deg):
        """The angle a distal femoral jig would be set to, for this patient."""
        plan = plan_for(
            synthetic_knee("left", ama_deg=ama_deg, include_head=True)
        )

        assert plan.distal_femoral_valgus_cut_deg == pytest.approx(ama_deg, abs=0.01)
        assert plan.valgus_quality is Quality.MEASURED
        assert "measured" in plan.valgus_source

    def test_falls_back_to_the_declared_assumption_on_a_knee_only_scan(self):
        """The legacy pipeline's fixed 6 degrees, now attributed rather than hidden."""
        plan = plan_for(synthetic_knee("left", include_head=False))

        assert plan.distal_femoral_valgus_cut_deg == pytest.approx(6.0)
        assert plan.valgus_quality is Quality.ESTIMATED
        assert "assumed" in plan.valgus_source

    @pytest.mark.parametrize("estimated", [
        ("femur.head_centre",),
        ("femur.canal_centre_distal", "femur.canal_centre_proximal"),
    ])
    def test_estimated_inputs_keep_the_patient_angle_but_not_the_tier(self, estimated):
        """Still this patient's angle -- not the population 6 degrees -- but estimated.

        Falling back to the assumption here would discard a real head centre merely
        because it was located by machine; keeping the measured tier would overstate it.
        """
        landmarks = as_estimated(
            synthetic_knee("left", ama_deg=8.0, include_head=True), *estimated
        )
        plan = plan_for(landmarks)

        assert plan.distal_femoral_valgus_cut_deg == pytest.approx(8.0, abs=0.01)
        assert plan.valgus_quality is Quality.ESTIMATED


class TestPhilosophies:
    def test_kinematic_resects_evenly_regardless_of_obliquity(self):
        """Kinematic alignment replaces what it removes, so both compartments lose the
        component's thickness whatever the native joint line does."""
        plan = plan_for(
            synthetic_knee("left", mldfa_deg=83.0),
            target=KINEMATIC, femoral_thickness_mm=9.0,
        )
        femoral = plan.resections["femoral_distal"]

        assert femoral.medial_depth_mm == pytest.approx(9.0, abs=0.01)
        assert femoral.lateral_depth_mm == pytest.approx(9.0, abs=0.01)

    def test_mechanical_does_not(self):
        plan = plan_for(synthetic_knee("left", mldfa_deg=83.0), target=MECHANICAL)
        femoral = plan.resections["femoral_distal"]

        assert abs(femoral.medial_depth_mm - femoral.lateral_depth_mm) > 1.0

    def test_kinematic_reproduces_the_native_slope(self):
        plan = plan_for(
            synthetic_knee("left", posterior_slope_deg=9.0),
            target=KINEMATIC, native_slope_deg=9.0,
        )
        assert plan.tibial_slope_deg == pytest.approx(9.0)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("lean_deg", [-6.0, 0.0, 6.0])
    def test_the_cut_slope_is_measured_against_the_tibial_axis(self, side, lean_deg):
        """A tibia lying flexed in the scanner must not change the slope of its cut.

        The whole tibia is rotated rigidly about the mediolateral axis, as it is when
        the knee is scanned slightly bent. The native slope metric, measured in the
        tibial frame, does not change; the planned cut, which reproduces it, must not
        either. Measured against the scanner instead, it would be off by the lean.
        """
        from tka_planner.core.metrics import posterior_slope_medial

        landmarks = leaned_tibia(synthetic_knee(side, posterior_slope_deg=9.0),
                                 lean_deg)
        femoral, tibial_frame = frames_for(landmarks)
        native = posterior_slope_medial(landmarks, tibial_frame).value
        assert native == pytest.approx(9.0, abs=0.01)

        plan = plan_alignment(landmarks, femoral, tibial_frame, target=KINEMATIC,
                              femoral_thickness_mm=9.0, tibial_resection_mm=21.0,
                              native_slope_deg=native)
        normal = plan.resections["tibial_proximal"].normal
        in_sagittal = normal - np.dot(normal, tibial_frame.sagittal_normal) \
            * tibial_frame.sagittal_normal
        slope = np.degrees(np.arccos(np.clip(
            np.dot(in_sagittal, tibial_frame.z_proximal)
            / np.linalg.norm(in_sagittal), -1.0, 1.0)))

        assert slope == pytest.approx(9.0, abs=0.05)
        assert abs(plan.diagnostics["tibial_axis_sagittal_lean_deg"]) == pytest.approx(
            abs(lean_deg), abs=0.05)
        assert plan.diagnostics["cut_ml_slope_shared"]

    def test_mechanical_uses_its_own_slope_target(self):
        plan = plan_for(
            synthetic_knee("left", posterior_slope_deg=9.0),
            target=MECHANICAL, native_slope_deg=9.0,
        )
        assert plan.tibial_slope_deg == pytest.approx(3.0)


class TestComponentPoses:
    def test_poses_are_orthonormal_and_right_handed(self):
        plan = plan_for(synthetic_knee("left"))

        for name, matrix in plan.components.items():
            rotation = matrix[:3, :3]
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9), name
            assert np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-9), name

    def test_components_sit_on_their_cut_planes(self):
        plan = plan_for(synthetic_knee("left"))

        for component, resection in (
            ("femoral_component", "femoral_distal"),
            ("tibial_component", "tibial_proximal"),
        ):
            assert np.allclose(
                plan.components[component][:3, 3],
                plan.resections[resection].point,
                atol=1e-9,
            )


class TestMirrorInvariance:
    @pytest.mark.parametrize("philosophy", [MECHANICAL, KINEMATIC])
    def test_resection_depths_survive_mirroring(self, philosophy):
        """A knee does not need a deeper cut for being a right knee."""
        original = synthetic_knee("left", mldfa_deg=84.0, mpta_deg=86.0)
        mirrored = mirror_landmarks(original)

        left = plan_for(original, target=philosophy)
        right = plan_for(mirrored, target=philosophy)

        for name in left.resections:
            assert left.resections[name].medial_depth_mm == pytest.approx(
                right.resections[name].medial_depth_mm, abs=1e-6
            ), f"{name} medial depth changed when mirrored"
            assert left.resections[name].lateral_depth_mm == pytest.approx(
                right.resections[name].lateral_depth_mm, abs=1e-6
            ), f"{name} lateral depth changed when mirrored"

    def test_valgus_angle_survives_mirroring(self):
        original = synthetic_knee("left", ama_deg=7.0, include_head=True)
        mirrored = mirror_landmarks(original)

        assert plan_for(original).distal_femoral_valgus_cut_deg == pytest.approx(
            plan_for(mirrored).distal_femoral_valgus_cut_deg, abs=1e-6
        )

    def test_posterior_slope_survives_mirroring(self):
        original = synthetic_knee("left", posterior_slope_deg=8.0)
        mirrored = mirror_landmarks(original)

        assert plan_for(original, target=KINEMATIC,
                        native_slope_deg=8.0).tibial_slope_deg == pytest.approx(
            plan_for(mirrored, target=KINEMATIC,
                     native_slope_deg=8.0).tibial_slope_deg, abs=1e-6
        )


class TestWarnings:
    def test_a_cut_that_misses_a_compartment_is_flagged(self):
        """A cut too shallow below the less worn plateau may never reach the worn one.

        The negative depth is the correct answer rather than an error -- it says the
        saw does not touch that compartment -- but it is a surgically meaningless plan,
        so it must be surfaced rather than reported as a resection.
        """
        plan = plan_for(synthetic_knee("left", mpta_deg=84.0),
                        tibial_resection_mm=1.0)
        tibial = plan.resections["tibial_proximal"]

        assert min(tibial.medial_depth_mm, tibial.lateral_depth_mm) < 0
        assert any("outside the bone" in warning for warning in plan.warnings)

    def test_a_normal_plan_raises_no_warnings(self):
        plan = plan_for(synthetic_knee("left"), tibial_resection_mm=10.0)
        assert plan.warnings == ()

    def test_serialises_for_the_plan_file(self):
        record = plan_for(synthetic_knee("left")).to_dict()

        assert record["philosophy"] == "mechanical"
        assert "distal_femoral_valgus_cut_deg" in record
        assert record["valgus_quality"] in ("measured", "estimated")
        assert len(record["components"]["femoral_component"]) == 4
