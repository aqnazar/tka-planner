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


def frames_for(landmarks):
    return build_femoral_frame(landmarks), build_tibial_frame(landmarks)


def plan_for(landmarks, **kwargs):
    femoral, tibial = frames_for(landmarks)
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

    @pytest.mark.parametrize("resection", [8.0, 10.0, 12.0])
    def test_tibial_depth_is_referenced_to_the_higher_plateau(self, resection):
        """The less worn side is the reference, because the other has lost bone."""
        plan = plan_for(synthetic_knee("left", mpta_deg=85.0),
                        tibial_resection_mm=resection)
        tibial = plan.resections["tibial_proximal"]

        assert max(tibial.medial_depth_mm,
                   tibial.lateral_depth_mm) == pytest.approx(resection, abs=0.01)

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
        """A shallow cut referenced to the higher plateau may never reach the worn one.

        The negative depth is the correct answer rather than an error -- it says the
        saw does not touch that compartment -- but it is a surgically meaningless plan,
        so it must be surfaced rather than reported as a resection.
        """
        plan = plan_for(synthetic_knee("left", mpta_deg=84.0),
                        tibial_resection_mm=0.0)
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
