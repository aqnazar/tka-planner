"""Anatomical frames.

The decisive test in this file is
:meth:`TestFemoralAxisLadder.test_assumed_ama_recovers_the_true_mechanical_axis`. When
the femoral head is absent, the frame derives a mechanical axis by rotating the measured
diaphyseal axis by an assumed mechanical-anatomical angle -- and the direction of that
rotation is a pure sign decision with no local evidence to check it against. Rotating
laterally instead of medially produces a perfectly well-formed frame that is wrong by
twice the angle, and every metric built on it would be quietly wrong by the same amount.

The synthetic knee is constructed with its diaphyseal axis tilted laterally from a known
mechanical axis, so the frame can only recover that axis by rotating the correct way.
"""

import numpy as np
import pytest

from tka_planner.core.frames import (
    FrameConstructionError,
    build_femoral_frame,
    build_tibial_frame,
)
from tka_planner.core.landmarks import Landmark, LandmarkSet, LandmarkStatus
from tka_planner.core.provenance import Quality
from tka_planner.core.sides import LPS_SUPERIOR, Side
from tests.synthetic import mirror_landmarks, synthetic_knee


def drop(landmarks: LandmarkSet, *ids: str) -> LandmarkSet:
    """Rebuild a set without the named landmarks, to exercise degradation."""
    kept = [
        Landmark(id=lm.id, position_mm=lm.position_mm, status=lm.status,
                 origin=lm.origin, reason=lm.reason, metadata=dict(lm.metadata))
        for lm in landmarks if lm.id not in ids
    ]
    return LandmarkSet(
        case_id=landmarks.case_id, side=landmarks.side,
        coordinate_system=landmarks.coordinate_system, landmarks=kept,
    )


# ----------------------------------------------------------------------
# Frame validity
# ----------------------------------------------------------------------


class TestFrameValidity:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("builder", [build_femoral_frame, build_tibial_frame])
    def test_axes_are_orthonormal_and_right_handed(self, side, builder):
        """A left-handed frame would mirror every measurement built on it."""
        frame = builder(synthetic_knee(side))
        axes = np.array([frame.x_anterior, frame.y_patient_left, frame.z_proximal])

        assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-9)
        assert np.isclose(float(np.linalg.det(axes)), 1.0, atol=1e-9)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_medial_points_toward_the_midline(self, side):
        """Medial is a different world direction on each knee; the frame must know."""
        frame = build_femoral_frame(synthetic_knee(side))
        # The knee sits away from the midline, so medial points back toward X = 0.
        assert np.sign(frame.medial[0]) == -np.sign(frame.origin[0])

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_anterior_axis_points_anteriorly(self, side):
        """In LPS, anterior is -Y."""
        for frame in (build_femoral_frame(synthetic_knee(side)),
                      build_tibial_frame(synthetic_knee(side))):
            assert frame.x_anterior[1] < 0

    def test_local_and_world_coordinates_round_trip(self):
        frame = build_femoral_frame(synthetic_knee("left"))
        rng = np.random.default_rng(0)
        points = rng.normal(scale=50.0, size=(20, 3)) + frame.origin

        assert np.allclose(frame.to_world(frame.to_local(points)), points, atol=1e-9)

    def test_origin_maps_to_the_local_origin(self):
        frame = build_tibial_frame(synthetic_knee("right"))
        assert np.allclose(frame.to_local(frame.origin), np.zeros(3), atol=1e-9)

    def test_matrix_matches_the_axes(self):
        frame = build_femoral_frame(synthetic_knee("left"))
        matrix = frame.matrix

        assert np.allclose(matrix[:3, 0], frame.x_anterior)
        assert np.allclose(matrix[:3, 2], frame.z_proximal)
        assert np.allclose(matrix[:3, 3], frame.origin)


# ----------------------------------------------------------------------
# The femoral axis ladder
# ----------------------------------------------------------------------


class TestFemoralAxisLadder:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_uses_the_true_mechanical_axis_when_the_head_is_present(self, side):
        frame = build_femoral_frame(synthetic_knee(side, include_head=True))

        assert frame.method == "frames.femur.mechanical.v1"
        assert frame.quality is Quality.MEASURED
        assert frame.assumptions == ()
        # The synthetic mechanical axis is exactly vertical.
        assert np.allclose(frame.z_proximal, LPS_SUPERIOR, atol=1e-9)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_falls_back_to_an_assumed_ama_when_the_head_is_absent(self, side):
        """The situation for every case in this cohort."""
        frame = build_femoral_frame(synthetic_knee(side, include_head=False))

        assert frame.method == "frames.femur.ama_assumed.v1"
        assert frame.quality is Quality.ESTIMATED
        assert [a.id for a in frame.assumptions] == ["population_femoral_ama"]

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("ama_deg", [4.0, 6.0, 8.0])
    def test_assumed_ama_recovers_the_true_mechanical_axis(self, side, ama_deg):
        """The test that pins the rotation direction.

        The synthetic knee's diaphyseal axis is tilted laterally from a vertical
        mechanical axis by exactly ``ama_deg``. Rotating it medially by the same angle
        must return the mechanical axis. Rotating the wrong way still yields a valid,
        orthonormal, entirely plausible frame -- wrong by ``2 * ama_deg``.
        """
        from tka_planner.core.provenance import Assumption, POPULATION_FEMORAL_AMA

        assumption = Assumption(
            id=POPULATION_FEMORAL_AMA.id,
            description=POPULATION_FEMORAL_AMA.description,
            value=ama_deg,
            unit="deg",
            source=POPULATION_FEMORAL_AMA.source,
        )
        landmarks = synthetic_knee(side, ama_deg=ama_deg, include_head=False)
        frame = build_femoral_frame(landmarks, ama_assumption=assumption)

        assert np.allclose(frame.z_proximal, LPS_SUPERIOR, atol=1e-9), (
            "The assumed AMA did not recover the mechanical axis; the coronal "
            "rotation is going the wrong way."
        )

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_the_two_methods_agree_when_the_assumption_is_correct(self, side):
        """Measured and estimated paths must coincide when the assumption holds.

        This is what justifies the fallback at all: it is the same axis, arrived at
        with less evidence.
        """
        measured = build_femoral_frame(synthetic_knee(side, include_head=True))
        estimated = build_femoral_frame(synthetic_knee(side, include_head=False))

        assert np.allclose(measured.z_proximal, estimated.z_proximal, atol=1e-9)
        assert measured.quality is not estimated.quality

    def test_records_the_diaphyseal_lever_arm(self):
        """Angular uncertainty scales inversely with it, so it must be visible."""
        frame = build_femoral_frame(synthetic_knee("left", include_head=False))
        assert frame.diagnostics["anatomical_axis_baseline_mm"] == pytest.approx(140.0)

    def test_records_the_mechanical_axis_length_when_measured(self):
        frame = build_femoral_frame(synthetic_knee("left", include_head=True))
        assert frame.diagnostics["mechanical_axis_length_mm"] == pytest.approx(400.0)

    def test_reports_stea_out_of_plane_tilt(self):
        """The component discarded by projection is a pick-quality signal.

        The synthetic sTEA is exactly perpendicular to the mechanical axis, so the tilt
        must read zero; on real picks it will not, and a large value flags a bad point.
        """
        frame = build_femoral_frame(synthetic_knee("left", include_head=True))
        assert frame.diagnostics["stea_out_of_plane_deg"] == pytest.approx(0.0, abs=1e-6)


class TestTibialAxisLadder:
    def test_uses_the_mechanical_axis_when_the_ankle_is_present(self):
        frame = build_tibial_frame(synthetic_knee("left", include_ankle=True))

        assert frame.method == "frames.tibia.mechanical.v1"
        assert frame.quality is Quality.MEASURED

    def test_falls_back_to_the_anatomical_proxy(self):
        frame = build_tibial_frame(synthetic_knee("left", include_ankle=False))

        assert frame.method == "frames.tibia.anatomical_proxy.v1"
        assert frame.quality is Quality.ESTIMATED
        assert [a.id for a in frame.assumptions] == ["tibial_ama_negligible"]

    def test_the_tibial_proxy_is_a_good_one(self):
        """Why the tibia degrades gracefully and the femur does not.

        The tibial mechanical and anatomical axes are near-collinear, so the fallback
        reproduces the measured axis. The femoral fallback, by contrast, has to apply a
        5-to-7 degree correction that varies between patients.
        """
        measured = build_tibial_frame(synthetic_knee("left", include_ankle=True))
        estimated = build_tibial_frame(synthetic_knee("left", include_ankle=False))

        assert np.allclose(measured.z_proximal, estimated.z_proximal, atol=1e-9)

    def test_records_the_rotational_reference_used(self):
        """Akagi and the medial-third reference differ by 2-4 degrees of rotation,
        so a plan is ambiguous unless it says which was used."""
        akagi = build_tibial_frame(synthetic_knee("left"),
                                   rotational_reference="akagi_2004")
        third = build_tibial_frame(synthetic_knee("left"),
                                   rotational_reference="medial_third")

        assert akagi.diagnostics["rotational_reference"] == "akagi_2004"
        assert third.diagnostics["rotational_reference"] == "medial_third"

    def test_the_two_rotational_references_actually_differ(self):
        akagi = build_tibial_frame(synthetic_knee("left"),
                                   rotational_reference="akagi_2004")
        third = build_tibial_frame(synthetic_knee("left"),
                                   rotational_reference="medial_third")

        assert not np.allclose(akagi.x_anterior, third.x_anterior, atol=1e-4)


# ----------------------------------------------------------------------
# Mirror invariance
# ----------------------------------------------------------------------


class TestMirrorInvariance:
    """Reflecting a knee and flipping its side must not change its anatomy.

    This is the flagship structural test. The legacy pipeline scattered ``if side ==
    'left'`` sign flips through its measurement code, and any one of them being wrong
    produced a pipeline correct on one knee and silently wrong on the other.
    """

    @pytest.mark.parametrize("builder", [build_femoral_frame, build_tibial_frame])
    def test_frame_axes_mirror_correctly(self, builder):
        original = builder(synthetic_knee("left"))
        mirrored = builder(mirror_landmarks(synthetic_knee("left")))

        reflect = np.array([-1.0, 1.0, 1.0])
        # Proximal and anterior are side-independent, so they reflect unchanged.
        assert np.allclose(mirrored.z_proximal, original.z_proximal * reflect, atol=1e-9)
        assert np.allclose(mirrored.x_anterior, original.x_anterior * reflect, atol=1e-9)
        assert np.allclose(mirrored.origin, original.origin * reflect, atol=1e-9)

    @pytest.mark.parametrize("builder", [build_femoral_frame, build_tibial_frame])
    def test_medial_direction_mirrors(self, builder):
        """Medial must remain medial through the reflection."""
        original = builder(synthetic_knee("left"))
        mirrored = builder(mirror_landmarks(synthetic_knee("left")))

        reflect = np.array([-1.0, 1.0, 1.0])
        assert np.allclose(mirrored.medial, original.medial * reflect, atol=1e-9)

    def test_a_mirrored_left_knee_is_a_right_knee(self):
        mirrored = mirror_landmarks(synthetic_knee("left"))
        assert mirrored.side is Side.RIGHT
        assert build_femoral_frame(mirrored).side is Side.RIGHT

    @pytest.mark.parametrize("builder", [build_femoral_frame, build_tibial_frame])
    def test_quality_and_method_are_side_independent(self, builder):
        original = builder(synthetic_knee("left"))
        mirrored = builder(mirror_landmarks(synthetic_knee("left")))

        assert mirrored.method == original.method
        assert mirrored.quality is original.quality


# ----------------------------------------------------------------------
# Degradation and failure
# ----------------------------------------------------------------------


class TestDegradation:
    def test_missing_knee_centre_names_the_landmark(self):
        landmarks = drop(synthetic_knee("left"), "femur.notch_centre")
        with pytest.raises(FrameConstructionError, match="femur.notch_centre"):
            build_femoral_frame(landmarks)

    def test_missing_sulcus_falls_back_to_the_anatomical_axis(self):
        """The sulcus cannot be found automatically, so the frame must survive without it.

        It is a depression rather than a surface extreme. Rather than fail, the frame
        uses the anatomical transepicondylar axis and records the substitution, so
        component rotation is known to be off by the 1-2 degrees between the two axes.
        """
        landmarks = drop(synthetic_knee("left"), "femur.epicondyle_medial_sulcus")
        frame = build_femoral_frame(landmarks)

        assert frame.diagnostics["rotational_reference"] == "atea"
        assert frame.quality is Quality.ESTIMATED
        assert "atea_substituted_for_stea" in [a.id for a in frame.assumptions]

    def test_the_surgical_axis_is_preferred_when_available(self):
        frame = build_femoral_frame(synthetic_knee("left"))
        assert frame.diagnostics["rotational_reference"] == "stea"

    def test_losing_both_medial_epicondylar_landmarks_fails(self):
        landmarks = drop(
            synthetic_knee("left"),
            "femur.epicondyle_medial_sulcus", "femur.epicondyle_medial_prominence",
        )
        with pytest.raises(FrameConstructionError, match="rotational reference"):
            build_femoral_frame(landmarks)

    def test_no_axis_at_all_lists_every_method_tried(self):
        """The error must explain the whole ladder, not just the last rung."""
        landmarks = drop(
            synthetic_knee("left"),
            "femur.canal_centre_distal", "femur.canal_centre_proximal",
        )
        with pytest.raises(FrameConstructionError) as excinfo:
            build_femoral_frame(landmarks)

        message = str(excinfo.value)
        assert "frames.femur.mechanical.v1" in message
        assert "frames.femur.ama_assumed.v1" in message
        assert "out_of_scan" in message  # says *why* the head was unavailable

    def test_missing_tibial_spines_are_reported(self):
        landmarks = drop(synthetic_knee("left"), "tibia.spine_lateral")
        with pytest.raises(FrameConstructionError, match="spine"):
            build_tibial_frame(landmarks)

    def test_swapped_akagi_landmarks_are_caught(self):
        """A reversed anteroposterior reference would invert the frame silently."""
        landmarks = synthetic_knee("left")
        pcl = landmarks.position("tibia.pcl_insertion_midpoint").copy()
        tubercle = landmarks.position(
            "tibia.tubercle_patellar_tendon_medial_border"
        ).copy()

        swapped = drop(landmarks, "tibia.pcl_insertion_midpoint",
                       "tibia.tubercle_patellar_tendon_medial_border")
        swapped.add(Landmark(id="tibia.pcl_insertion_midpoint", position_mm=tubercle,
                             status=LandmarkStatus.PRESENT, origin="test"))
        swapped.add(Landmark(id="tibia.tubercle_patellar_tendon_medial_border",
                             position_mm=pcl, status=LandmarkStatus.PRESENT,
                             origin="test"))

        with pytest.raises(FrameConstructionError, match="swapped|backwards"):
            build_tibial_frame(swapped)


class TestSerialisation:
    def test_frame_serialises_with_its_provenance(self):
        record = build_femoral_frame(synthetic_knee("left")).to_dict()

        assert record["bone"] == "femur"
        assert record["side"] == "left"
        assert record["handedness"] == "right"
        assert record["quality"] == "estimated"
        assert record["assumptions"] == ["population_femoral_ama"]
        assert "anatomical_axis_baseline_mm" in record["diagnostics"]

    def test_measured_frame_records_no_assumptions(self):
        record = build_femoral_frame(
            synthetic_knee("left", include_head=True)
        ).to_dict()

        assert record["quality"] == "measured"
        assert record["assumptions"] == []
