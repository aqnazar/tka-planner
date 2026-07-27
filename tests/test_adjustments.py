"""Manual adjustment of a computed plan.

Every control a surgeon can drag is a value in :class:`Adjustments`, and the plan stays a
pure function of the landmarks, the frames, the target and those adjustments. So each
control is tested by asserting the exact quantity it claims to change -- a ten millimetre
resection delta must move both compartment depths by ten millimetres, not by "more" --
and by asserting that everything it does *not* claim to change is untouched.

The invariant under most pressure here is the shared mediolateral slope. The whole
planning module exists to guarantee the two cuts agree in the coronal plane, so the
varus/valgus control must rotate both cuts together and leave them still agreeing.
"""

import numpy as np
import pytest

from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.geometry import angle_between, unit
from tka_planner.core.planning import (
    KINEMATIC,
    MECHANICAL,
    Adjustments,
    plan_alignment,
)
from tka_planner.core.sides import LPS_ANTERIOR, LPS_PATIENT_LEFT, LPS_SUPERIOR
from tests.synthetic import mirror_landmarks, synthetic_knee


def plan_for(landmarks, **kwargs):
    femoral = build_femoral_frame(landmarks)
    tibial = build_tibial_frame(landmarks)
    return plan_alignment(landmarks, femoral, tibial, **kwargs)


def coronal_angle(normal):
    """Signed tilt of a cut normal in the coronal plane, in degrees.

    Measured about the patient frame's anteroposterior axis, which is the axis the
    varus/valgus control turns, so the assertion and the control speak the same language.
    """
    in_coronal = unit(np.asarray(normal) - np.dot(normal, LPS_ANTERIOR) * LPS_ANTERIOR)
    return float(np.degrees(np.arctan2(
        np.dot(in_coronal, LPS_PATIENT_LEFT), np.dot(in_coronal, LPS_SUPERIOR)
    )))


def sagittal_angle(normal):
    """How far a cut normal is lifted out of the coronal plane, in degrees.

    This is the flexion or posterior slope, and it is measured as an elevation rather
    than as an angle in a projection because only the elevation is invariant under a
    coronal rotation. Projecting into the sagittal plane and taking the angle there
    changes when the cut is rotated about the anteroposterior axis, purely because the
    projection's proximal component shrinks -- so a correct varus/valgus control would
    appear to alter the slope.
    """
    return float(np.degrees(np.arcsin(
        np.clip(np.dot(unit(np.asarray(normal)), LPS_ANTERIOR), -1.0, 1.0)
    )))


def ml_slope(normal):
    """The mediolateral component of a cut, which both cuts must share exactly."""
    return coronal_angle(normal)


class TestNoAdjustment:
    def test_default_adjustments_reproduce_the_plain_plan(self):
        """The zero adjustment has to be exactly the identity, or every readout drifts
        the moment the panel is opened."""
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(landmarks, adjustments=Adjustments())

        for name in plain.resections:
            a, b = plain.resections[name], adjusted.resections[name]
            assert np.allclose(a.point, b.point, atol=1e-9)
            assert np.allclose(a.normal, b.normal, atol=1e-9)
            assert a.medial_depth_mm == pytest.approx(b.medial_depth_mm, abs=1e-9)
            assert a.lateral_depth_mm == pytest.approx(b.lateral_depth_mm, abs=1e-9)

        for name in plain.components:
            assert np.allclose(
                plain.components[name], adjusted.components[name], atol=1e-9
            )

    def test_omitting_adjustments_entirely_matches_passing_the_default(self):
        landmarks = synthetic_knee("right")
        assert np.allclose(
            plan_for(landmarks).resections["femoral_distal"].normal,
            plan_for(landmarks, adjustments=Adjustments())
            .resections["femoral_distal"].normal,
        )


class TestCoronalCorrection:
    """The varus/valgus control, which must move both cuts as one."""

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("correction", [-4.0, -1.5, 2.0, 5.0])
    def test_both_cuts_rotate_by_the_same_angle(self, side, correction):
        landmarks = synthetic_knee(side)
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(coronal_correction_deg=correction)
        )

        femoral_shift = coronal_angle(
            adjusted.resections["femoral_distal"].normal
        ) - coronal_angle(plain.resections["femoral_distal"].normal)
        tibial_shift = coronal_angle(
            adjusted.resections["tibial_proximal"].normal
        ) - coronal_angle(plain.resections["tibial_proximal"].normal)

        assert femoral_shift == pytest.approx(tibial_shift, abs=1e-6), (
            "the two cuts moved by different amounts, so the construct has to absorb "
            "the difference"
        )
        assert abs(femoral_shift) == pytest.approx(abs(correction), abs=1e-6)

    @pytest.mark.parametrize("correction", [-3.0, 3.0, 6.0])
    def test_the_shared_mediolateral_slope_survives(self, correction):
        """The invariant the planning module is built to protect."""
        adjusted = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(coronal_correction_deg=correction),
        )
        femoral = adjusted.resections["femoral_distal"].normal
        tibial = adjusted.resections["tibial_proximal"].normal

        assert ml_slope(femoral) == pytest.approx(ml_slope(tibial), abs=1e-6)

    def test_the_sagittal_angles_are_left_alone(self):
        """Varus/valgus is a coronal control; it must not tilt the slope."""
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(coronal_correction_deg=5.0)
        )

        for name in ("femoral_distal", "tibial_proximal"):
            assert sagittal_angle(adjusted.resections[name].normal) == pytest.approx(
                sagittal_angle(plain.resections[name].normal), abs=1e-6
            )

    PAIRS = (
        ("femoral_component", "femoral_distal"),
        ("tibial_component", "tibial_proximal"),
        ("femoral_cutting_block", "femoral_distal"),
        ("tibial_cutting_block", "tibial_proximal"),
    )

    @pytest.mark.parametrize("correction", [0.0, 4.0, -2.5])
    def test_every_component_sits_on_its_own_cut(self, correction):
        """The invariant worth asserting: a component's proximal axis *is* its cut
        normal, before and after adjustment. Checking instead that the component's
        normal swept the same angle as the correction would be wrong -- a cut carrying
        posterior slope lies out of the coronal plane, so a 4 degree coronal rotation
        sweeps its normal through slightly less than 4 degrees in three dimensions. The
        coronal angle is exactly the correction, which is what the surgeon set and what
        the test above already asserts.
        """
        plan = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(coronal_correction_deg=correction),
        )
        for component, cut in self.PAIRS:
            assert np.allclose(
                plan.components[component][:3, 2],
                plan.resections[cut].normal, atol=1e-9,
            ), f"{component} is not seated on {cut}"

    def test_the_components_actually_move(self):
        """Both implants and both cutting blocks move, or the scene would disagree with
        the plan it was built from."""
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(coronal_correction_deg=4.0)
        )

        for name, _ in self.PAIRS:
            moved = float(np.degrees(angle_between(
                plain.components[name][:3, 2], adjusted.components[name][:3, 2]
            )))
            assert moved == pytest.approx(4.0, abs=0.01)


class TestResectionDeltas:
    @pytest.mark.parametrize("delta", [-2.0, 1.0, 3.5])
    def test_a_femoral_delta_deepens_both_compartments_equally(self, delta):
        landmarks = synthetic_knee("left", mldfa_deg=87.0)
        plain = plan_for(landmarks).resections["femoral_distal"]
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(femoral_resection_delta_mm=delta)
        ).resections["femoral_distal"]

        assert adjusted.medial_depth_mm == pytest.approx(
            plain.medial_depth_mm + delta, abs=1e-6)
        assert adjusted.lateral_depth_mm == pytest.approx(
            plain.lateral_depth_mm + delta, abs=1e-6)

    @pytest.mark.parametrize("delta", [-2.0, 1.0, 3.5])
    def test_a_tibial_delta_deepens_both_compartments_equally(self, delta):
        landmarks = synthetic_knee("left", mpta_deg=86.0)
        plain = plan_for(landmarks).resections["tibial_proximal"]
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(tibial_resection_delta_mm=delta)
        ).resections["tibial_proximal"]

        assert adjusted.medial_depth_mm == pytest.approx(
            plain.medial_depth_mm + delta, abs=1e-6)
        assert adjusted.lateral_depth_mm == pytest.approx(
            plain.lateral_depth_mm + delta, abs=1e-6)

    def test_a_resection_delta_does_not_rotate_the_cut(self):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(femoral_resection_delta_mm=3.0)
        )

        assert np.allclose(
            plain.resections["femoral_distal"].normal,
            adjusted.resections["femoral_distal"].normal, atol=1e-9,
        )

    def test_the_femoral_delta_leaves_the_tibial_cut_alone(self):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks).resections["tibial_proximal"]
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(femoral_resection_delta_mm=4.0)
        ).resections["tibial_proximal"]

        assert np.allclose(plain.point, adjusted.point, atol=1e-9)
        assert adjusted.medial_depth_mm == pytest.approx(
            plain.medial_depth_mm, abs=1e-9)


class TestSlopeAndFlexion:
    @pytest.mark.parametrize("delta", [-2.0, 3.0, 5.0])
    def test_a_slope_delta_tilts_only_the_tibial_cut(self, delta):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(tibial_slope_delta_deg=delta)
        )

        moved = float(np.degrees(angle_between(
            plain.resections["tibial_proximal"].normal,
            adjusted.resections["tibial_proximal"].normal,
        )))
        assert moved == pytest.approx(abs(delta), abs=1e-6)
        assert np.allclose(
            plain.resections["femoral_distal"].normal,
            adjusted.resections["femoral_distal"].normal, atol=1e-9,
        )

    @pytest.mark.parametrize("delta", [1.0, 3.0])
    def test_a_flexion_delta_tilts_only_the_femoral_cut(self, delta):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(femoral_flexion_delta_deg=delta)
        )

        moved = float(np.degrees(angle_between(
            plain.resections["femoral_distal"].normal,
            adjusted.resections["femoral_distal"].normal,
        )))
        assert moved == pytest.approx(abs(delta), abs=1e-6)
        assert np.allclose(
            plain.resections["tibial_proximal"].normal,
            adjusted.resections["tibial_proximal"].normal, atol=1e-9,
        )

    def test_the_reported_slope_includes_the_delta(self):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks, target=MECHANICAL)
        adjusted = plan_for(
            landmarks, target=MECHANICAL,
            adjustments=Adjustments(tibial_slope_delta_deg=2.0),
        )
        assert adjusted.tibial_slope_deg == pytest.approx(
            plain.tibial_slope_deg + 2.0, abs=1e-9)


class TestPerCutVarus:
    def test_a_femoral_varus_delta_breaks_the_shared_slope_and_says_so(self):
        """Per-cut varus is an override, so the disagreement it creates is reported
        rather than hidden -- the point of the shared reference is that a difference is
        an error unless someone chose it deliberately."""
        adjusted = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(femoral_varus_delta_deg=3.0),
        )
        femoral = adjusted.resections["femoral_distal"].normal
        tibial = adjusted.resections["tibial_proximal"].normal

        assert abs(ml_slope(femoral) - ml_slope(tibial)) == pytest.approx(3.0, abs=1e-6)
        assert any("mediolateral" in w.lower() for w in adjusted.warnings)

    def test_no_warning_when_the_cuts_still_agree(self):
        adjusted = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(coronal_correction_deg=4.0),
        )
        assert not any(
            "cuts disagree" in w.lower() for w in adjusted.warnings
        )


class TestComponentShifts:
    @pytest.mark.parametrize("axis,amount", [("ap", 4.0), ("ml", -3.0)])
    def test_a_shift_moves_the_component_but_not_its_cut(self, axis, amount):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks,
            adjustments=Adjustments(**{f"femoral_shift_{axis}_mm": amount}),
        )

        travelled = float(np.linalg.norm(
            adjusted.components["femoral_component"][:3, 3]
            - plain.components["femoral_component"][:3, 3]
        ))
        assert travelled == pytest.approx(abs(amount), abs=1e-6)
        assert np.allclose(
            plain.resections["femoral_distal"].point,
            adjusted.resections["femoral_distal"].point, atol=1e-9,
        )

    def test_a_shift_stays_in_the_plane_of_the_cut(self):
        """Moving a component anteriorly must not lift it off its cut."""
        landmarks = synthetic_knee("left")
        plan = plan_for(
            landmarks, adjustments=Adjustments(femoral_shift_ap_mm=5.0)
        )
        normal = plan.resections["femoral_distal"].normal
        offset = (plan.components["femoral_component"][:3, 3]
                  - plan.resections["femoral_distal"].point)

        assert float(np.dot(offset, normal)) == pytest.approx(0.0, abs=1e-6)

    def test_the_cutting_block_follows_its_implant(self):
        """They share one CAD origin, so a shift that separated them would put the block
        somewhere the implant will not go."""
        plan = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(femoral_shift_ml_mm=3.0),
        )
        assert np.allclose(
            plan.components["femoral_component"],
            plan.components["femoral_cutting_block"], atol=1e-12,
        )


class TestComponentRotation:
    @pytest.mark.parametrize("delta", [-3.0, 2.0])
    def test_a_rotation_delta_turns_the_component_about_its_cut_normal(self, delta):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(femoral_rotation_delta_deg=delta)
        )

        normal = plain.resections["femoral_distal"].normal
        turned = float(np.degrees(angle_between(
            plain.components["femoral_component"][:3, 0],
            adjusted.components["femoral_component"][:3, 0],
        )))
        assert turned == pytest.approx(abs(delta), abs=1e-6)
        # The cut itself does not move, only the component on it.
        assert np.allclose(
            normal, adjusted.resections["femoral_distal"].normal, atol=1e-9)

    def test_tibial_rotation_turns_the_tray(self):
        landmarks = synthetic_knee("left")
        plain = plan_for(landmarks)
        adjusted = plan_for(
            landmarks, adjustments=Adjustments(tibial_rotation_delta_deg=5.0)
        )
        turned = float(np.degrees(angle_between(
            plain.components["tibial_component"][:3, 0],
            adjusted.components["tibial_component"][:3, 0],
        )))
        assert turned == pytest.approx(5.0, abs=1e-6)


class TestGaps:
    """Insert thickness only means something beside the gap it fills."""

    def test_a_gap_is_reported_for_each_compartment(self):
        plan = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(insert_thickness_mm=9.0),
        )
        assert "extension_gap_medial_mm" in plan.diagnostics
        assert "extension_gap_lateral_mm" in plan.diagnostics

    @pytest.mark.parametrize("increase", [1.0, 3.0])
    def test_a_thicker_insert_closes_the_gap_by_exactly_its_increase(self, increase):
        landmarks = synthetic_knee("left")
        thin = plan_for(
            landmarks, adjustments=Adjustments(insert_thickness_mm=8.0)
        ).diagnostics
        thick = plan_for(
            landmarks, adjustments=Adjustments(insert_thickness_mm=8.0 + increase)
        ).diagnostics

        for side in ("medial", "lateral"):
            key = f"extension_gap_{side}_mm"
            assert thick[key] == pytest.approx(thin[key] - increase, abs=1e-6)

    def test_a_deeper_tibial_cut_opens_the_gap(self):
        landmarks = synthetic_knee("left")
        shallow = plan_for(
            landmarks,
            adjustments=Adjustments(insert_thickness_mm=9.0),
        ).diagnostics
        deep = plan_for(
            landmarks,
            adjustments=Adjustments(
                insert_thickness_mm=9.0, tibial_resection_delta_mm=2.0
            ),
        ).diagnostics

        assert deep["extension_gap_medial_mm"] == pytest.approx(
            shallow["extension_gap_medial_mm"] + 2.0, abs=1e-6)

    def test_no_insert_thickness_means_no_gap_rather_than_a_guess(self):
        """The pipeline's standing rule: absent an input, report nothing."""
        plan = plan_for(synthetic_knee("left"))
        assert plan.diagnostics.get("extension_gap_medial_mm") is None


class TestMirrorInvariance:
    """A knee is not planned differently for being a right knee."""

    @pytest.mark.parametrize("adjustments", [
        Adjustments(coronal_correction_deg=3.0),
        Adjustments(tibial_slope_delta_deg=2.0, femoral_resection_delta_mm=1.5),
        Adjustments(insert_thickness_mm=10.0, tibial_resection_delta_mm=2.0),
    ])
    def test_scalars_survive_reflection(self, adjustments):
        left = plan_for(synthetic_knee("left"), adjustments=adjustments)
        right = plan_for(
            mirror_landmarks(synthetic_knee("left")), adjustments=adjustments
        )

        for name in left.resections:
            assert right.resections[name].medial_depth_mm == pytest.approx(
                left.resections[name].medial_depth_mm, abs=1e-6)
            assert right.resections[name].lateral_depth_mm == pytest.approx(
                left.resections[name].lateral_depth_mm, abs=1e-6)

        assert right.tibial_slope_deg == pytest.approx(left.tibial_slope_deg, abs=1e-6)
        for key in ("extension_gap_medial_mm", "extension_gap_lateral_mm"):
            if left.diagnostics.get(key) is not None:
                assert right.diagnostics[key] == pytest.approx(
                    left.diagnostics[key], abs=1e-6)


class TestSerialisation:
    def test_the_adjustments_travel_in_the_plan(self):
        """A manually adjusted plan has to be re-derivable, which means the adjustments
        are part of the record rather than something that happened in a viewport."""
        plan = plan_for(
            synthetic_knee("left"),
            adjustments=Adjustments(
                coronal_correction_deg=2.0, insert_thickness_mm=9.0
            ),
        )
        recorded = plan.to_dict()["adjustments"]

        assert recorded["coronal_correction_deg"] == pytest.approx(2.0)
        assert recorded["insert_thickness_mm"] == pytest.approx(9.0)

    def test_an_unadjusted_plan_records_them_as_zero_rather_than_absent(self):
        plan = plan_for(synthetic_knee("left"))
        assert plan.to_dict()["adjustments"]["coronal_correction_deg"] == 0.0
