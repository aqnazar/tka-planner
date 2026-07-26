"""Quality control.

The femoral-head detector is the load-bearing test here. The entire capability model
depends on it answering correctly in both directions: it must find a head that is
present, and -- far more importantly for this cohort -- it must not hallucinate one on a
knee-only scan. A false positive would let the pipeline compute a mechanical axis from
nothing and report it as measured.
"""

import numpy as np
import pytest

from tka_planner.core.landmarks import (
    CoordinateSystem,
    Landmark,
    LandmarkSet,
    LandmarkStatus,
)
from tka_planner.core.meshio import Mesh
from tka_planner.core.qc import (
    QCReport,
    QualityControlError,
    Severity,
    assess_femur_coverage,
    assess_mesh_quality,
    assess_tibia_coverage,
    check_landmarks_on_mesh,
    check_laterality,
    check_plausible_bone_scale,
)
from tka_planner.core.sides import Side


# ----------------------------------------------------------------------
# Synthetic bones
# ----------------------------------------------------------------------


def cylinder_points(length_mm, radius_mm=14.0, n=6000, centre=(0, 0, 0), seed=0):
    """A diaphysis: the shape a femur segment has when the head is not in the scan."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-length_mm / 2, length_mm / 2, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([
        centre[0] + radius_mm * np.cos(theta),
        centre[1] + radius_mm * np.sin(theta),
        centre[2] + z,
    ])


def sphere_cap_points(centre, radius_mm, n=3000, cap_deg=200.0, seed=1):
    rng = np.random.default_rng(seed)
    max_polar = np.radians(min(cap_deg, 360.0)) / 2.0
    polar = np.arccos(rng.uniform(np.cos(max_polar), 1.0, n))
    azimuth = rng.uniform(0, 2 * np.pi, n)
    return np.asarray(centre) + radius_mm * np.column_stack([
        np.sin(polar) * np.cos(azimuth),
        np.sin(polar) * np.sin(azimuth),
        np.cos(polar),
    ])


def as_mesh(points):
    """Wrap a point cloud as a Mesh. Coverage checks only read vertices."""
    points = np.asarray(points, dtype=float)
    n_faces = len(points) // 3
    faces = np.arange(n_faces * 3).reshape(n_faces, 3)
    return Mesh(vertices=points[: n_faces * 3], faces=faces)


def knee_only_femur(length_mm=175.0, centre_x=80.0):
    """A knee-region femur: shaft only, no head. Every case in this cohort."""
    return as_mesh(cylinder_points(length_mm, centre=(centre_x, 0.0, -500.0)))


def full_femur(shaft_length_mm=400.0, centre_x=80.0, head_radius_mm=23.0):
    """A femur reaching the hip, with a genuine head at the proximal end."""
    shaft = cylinder_points(shaft_length_mm, centre=(centre_x, 0.0, -400.0))
    head_centre = (centre_x, 0.0, -400.0 + shaft_length_mm / 2 + head_radius_mm * 0.6)
    head = sphere_cap_points(head_centre, head_radius_mm, n=4000)
    return as_mesh(np.vstack([shaft, head]))


# ----------------------------------------------------------------------
# Scan coverage -- the heart of the capability model
# ----------------------------------------------------------------------


class TestFemurCoverage:
    def test_reports_no_head_on_a_knee_only_scan(self):
        """The situation in every case in the cohort."""
        coverage, report = assess_femur_coverage(knee_only_femur())

        assert coverage["femoral_head_present"] is False
        assert "below the" in coverage["detection"]
        assert any(f.code == "SCAN_KNEE_ONLY_FEMUR" for f in report.findings)

    def test_finds_a_head_when_one_is_present(self):
        coverage, report = assess_femur_coverage(full_femur())

        assert coverage["femoral_head_present"] is True
        assert 18.0 <= coverage["head_fit_radius_mm"] <= 30.0
        assert coverage["head_fit_rms_mm"] <= 1.5
        assert any(f.code == "SCAN_FEMORAL_HEAD_PRESENT" for f in report.findings)

    def test_a_long_shaft_alone_is_not_mistaken_for_a_head(self):
        """The false positive that would matter most.

        A femur long enough to pass the length test but with the head cropped away must
        still be reported as headless -- a cylinder fits a sphere badly, and the RMS
        gate is what catches it.
        """
        coverage, _ = assess_femur_coverage(
            as_mesh(cylinder_points(420.0, centre=(80.0, 0.0, -400.0)))
        )
        assert coverage["femoral_head_present"] is False

    def test_records_the_reasoning_either_way(self):
        """The report must be able to explain the conclusion, not just state it."""
        for mesh in (knee_only_femur(), full_femur()):
            coverage, _ = assess_femur_coverage(mesh)
            assert coverage["detection"]
            assert "length_mm" in coverage

    def test_a_head_sized_sphere_at_the_wrong_radius_is_rejected(self):
        """Radius bounds exclude structures that are spherical but not a femoral head."""
        shaft = cylinder_points(400.0, centre=(80.0, 0.0, -400.0))
        blob = sphere_cap_points((80.0, 0.0, -140.0), radius_mm=60.0, n=4000)
        coverage, _ = assess_femur_coverage(as_mesh(np.vstack([shaft, blob])))

        assert coverage["femoral_head_present"] is False

    @pytest.mark.parametrize("shaft_radius_mm", [10.0, 14.0, 18.0, 22.0, 25.0])
    @pytest.mark.parametrize("length_mm", [360.0, 420.0, 500.0])
    def test_no_headless_shaft_is_ever_read_as_a_head(
        self, shaft_radius_mm, length_mm
    ):
        """The false-positive sweep, and the reason the RMS bound is 1.0 rather than 1.5.

        A headless femur long enough to clear the length gate must still be reported as
        having no head. An earlier threshold of 1.5 mm failed this for thick shafts
        (18-25 mm radius at 360-420 mm length): a short axial band of a wide cylinder
        fits a sphere at a plausible radius with a convincing residual.

        The sweep is parameterised rather than written as one case because the hole was
        found by sweeping, and only a sweep will notice if it reopens.
        """
        shaft = cylinder_points(
            length_mm, radius_mm=shaft_radius_mm, n=25000,
            centre=(80.0, 0.0, -400.0), seed=int(shaft_radius_mm),
        )
        coverage, _ = assess_femur_coverage(as_mesh(shaft))

        assert coverage["femoral_head_present"] is False, (
            f"A headless {length_mm:.0f} mm shaft of radius {shaft_radius_mm} mm was "
            f"mistaken for a femoral head: {coverage['detection']}"
        )

    def test_residual_is_what_separates_a_head_from_a_shaft(self):
        """Pins the empirical basis for the chosen criterion.

        Neither wrap-around nor direction isotropy discriminates -- both overlap
        between shafts and head caps, and wrap-around is actually higher for shafts.
        The radial residual separates them by more than an order of magnitude, so that
        is what the gate uses.
        """
        from tka_planner.core.geometry import fit_sphere_robust, regional_mask
        from tka_planner.core.sides import LPS_SUPERIOR

        shaft = cylinder_points(420.0, radius_mm=20.0, n=25000,
                                centre=(80.0, 0.0, -400.0))
        shaft_fit = fit_sphere_robust(
            shaft[regional_mask(shaft, LPS_SUPERIOR, fraction=0.15, end="high")]
        )
        head_fit = fit_sphere_robust(
            sphere_cap_points((0.0, 0.0, 0.0), 23.0, n=6000, cap_deg=200.0)
        )

        assert head_fit.rms_mm < 0.5 < shaft_fit.rms_mm

        # And the metrics that look plausible but do not work, recorded so the mistake
        # is not repeated: a shaft wraps around more completely than a head cap does.
        assert shaft_fit.surface_spread > head_fit.surface_spread

    def test_the_longest_real_femur_stays_well_below_the_length_gate(self):
        """Guards the margin between real data and the head-detection threshold.

        The longest femur in the cohort is 248 mm. The gate sits at 350 mm because the
        head-to-condyle distance is the femur's whole length, so a scan holding 350 mm
        of bone without a head is close to anatomically impossible. A threshold near
        248 mm would leave no room at all.
        """
        from tka_planner.core.qc import MIN_FEMUR_LENGTH_FOR_HEAD_MM

        longest_real_femur_mm = 248.0
        assert MIN_FEMUR_LENGTH_FOR_HEAD_MM > longest_real_femur_mm + 100.0


class TestTibiaCoverage:
    def test_reports_no_malleoli_on_a_knee_only_scan(self):
        coverage, report = assess_tibia_coverage(
            as_mesh(cylinder_points(180.0, centre=(80.0, 0.0, -650.0)))
        )
        assert coverage["malleoli_present"] is False
        assert any(f.code == "SCAN_KNEE_ONLY_TIBIA" for f in report.findings)

    def test_accepts_a_full_length_tibia(self):
        coverage, _ = assess_tibia_coverage(
            as_mesh(cylinder_points(340.0, centre=(80.0, 0.0, -650.0)))
        )
        assert coverage["malleoli_present"] is True


# ----------------------------------------------------------------------
# Laterality -- the empirical LPS confirmation
# ----------------------------------------------------------------------


class TestLaterality:
    def test_accepts_a_left_knee_at_positive_x(self):
        """Confirmed across the cohort: left knees occupy +61 to +96 mm in X."""
        report = check_laterality(knee_only_femur(centre_x=80.0), Side.LEFT)

        assert report.ok
        assert any(f.code == "LATERALITY_CONSISTENT" for f in report.findings)

    def test_accepts_a_right_knee_at_negative_x(self):
        report = check_laterality(knee_only_femur(centre_x=-80.0), Side.RIGHT)
        assert any(f.code == "LATERALITY_CONSISTENT" for f in report.findings)

    def test_flags_a_knee_on_the_wrong_side(self):
        """Catches both a mislabelled side and an LPS/RAS mix-up, which mirrors X."""
        report = check_laterality(knee_only_femur(centre_x=+80.0), Side.RIGHT)

        assert report.warnings
        finding = report.warnings[0]
        assert finding.code == "LATERALITY_MISMATCH"
        assert "mirrors the anatomy" in finding.message

    def test_is_a_warning_not_an_error(self):
        """A scan cropped about a shifted origin could legitimately trip this."""
        report = check_laterality(knee_only_femur(centre_x=+80.0), Side.RIGHT)
        assert report.ok  # no errors, so the pipeline may proceed

    def test_skips_the_check_outside_lps(self):
        report = check_laterality(
            knee_only_femur(), Side.LEFT, coordinate_system=CoordinateSystem.RAS
        )
        assert any(f.code == "LATERALITY_NOT_CHECKED" for f in report.findings)


# ----------------------------------------------------------------------
# Scale
# ----------------------------------------------------------------------


class TestScaleGuard:
    def test_accepts_a_realistic_bone(self):
        assert check_plausible_bone_scale(knee_only_femur(), "femur").ok

    def test_rejects_a_mesh_supplied_in_metres(self):
        """The failure that would otherwise produce plausible-looking nonsense."""
        metres = as_mesh(knee_only_femur().vertices / 1000.0)
        report = check_plausible_bone_scale(metres, "femur")

        assert not report.ok
        assert report.errors[0].code == "SCALE_TOO_SMALL"
        assert "factor of 1000" in report.errors[0].message

    def test_rejects_an_implausibly_large_mesh(self):
        huge = as_mesh(knee_only_femur().vertices * 10.0)
        report = check_plausible_bone_scale(huge, "femur")

        assert not report.ok
        assert report.errors[0].code == "SCALE_TOO_LARGE"


# ----------------------------------------------------------------------
# Landmarks against the mesh -- the coordinate-system gate
# ----------------------------------------------------------------------


def landmark_set_at(positions, side="left"):
    return LandmarkSet(
        case_id="CASE_TEST", side=side, coordinate_system="LPS",
        landmarks=[
            Landmark(id=landmark_id, position_mm=position,
                     status=LandmarkStatus.PRESENT, origin="test")
            for landmark_id, position in positions.items()
        ],
    )


class TestLandmarksOnMesh:
    def test_accepts_landmarks_inside_the_bone(self):
        mesh = knee_only_femur(centre_x=80.0)
        centre = mesh.vertices.mean(axis=0)
        landmarks = landmark_set_at({
            "femur.notch_centre": centre,
            "femur.epicondyle_lateral": centre + [10.0, 0.0, 0.0],
        })

        report = check_landmarks_on_mesh(landmarks, {"femur": mesh})
        assert report.ok
        assert any(f.code == "LANDMARKS_ON_MESH" for f in report.findings)

    def test_catches_a_coordinate_system_mismatch(self):
        """The decisive test.

        Reading RAS landmarks as LPS negates X and Y, moving a knee at +80 mm to
        -80 mm -- 160 mm from the bone. The check must fail loudly and name the likely
        cause, because the resulting plan would otherwise look entirely reasonable.
        """
        mesh = knee_only_femur(centre_x=80.0)
        centre = mesh.vertices.mean(axis=0)
        mirrored = centre * np.array([-1.0, -1.0, 1.0])
        landmarks = landmark_set_at({"femur.notch_centre": mirrored})

        report = check_landmarks_on_mesh(landmarks, {"femur": mesh})

        assert not report.ok
        finding = report.errors[0]
        assert finding.code == "LANDMARKS_OFF_MESH"
        assert "coordinate system mismatch" in finding.message
        assert finding.context["worst_excursion_mm"] > 100

    def test_ignores_landmarks_for_bones_not_supplied(self):
        mesh = knee_only_femur()
        centre = mesh.vertices.mean(axis=0)
        landmarks = landmark_set_at({
            "femur.notch_centre": centre,
            "tibia.spine_medial": [9999.0, 9999.0, 9999.0],
        })

        # Only the femur is offered for checking, so the wild tibial point is skipped.
        assert check_landmarks_on_mesh(landmarks, {"femur": mesh}).ok

    def test_ignores_unusable_landmarks(self):
        mesh = knee_only_femur()
        landmarks = LandmarkSet(
            case_id="CASE_TEST", side="left", coordinate_system="LPS",
            landmarks=[Landmark(id="femur.head_centre",
                                status=LandmarkStatus.OUT_OF_SCAN,
                                reason="knee-only field of view")],
        )
        report = check_landmarks_on_mesh(landmarks, {"femur": mesh})
        assert report.ok

    def test_margin_allows_landmarks_just_off_the_surface(self):
        """Canal centres and the head centre are legitimately off the surface."""
        mesh = knee_only_femur()
        high = mesh.vertices.max(axis=0)
        landmarks = landmark_set_at({"femur.notch_centre": high + [2.0, 0.0, 0.0]})

        assert check_landmarks_on_mesh(landmarks, {"femur": mesh},
                                       margin_mm=5.0).ok
        assert not check_landmarks_on_mesh(landmarks, {"femur": mesh},
                                           margin_mm=1.0).ok


# ----------------------------------------------------------------------
# Mesh topology
# ----------------------------------------------------------------------


TETRA_VERTICES = np.array([
    [0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 100.0],
])
TETRA_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


class TestMeshQuality:
    def test_recognises_a_closed_surface(self):
        mesh = Mesh(vertices=TETRA_VERTICES, faces=TETRA_FACES)
        quality, report = assess_mesh_quality(mesh, already_welded=True)

        assert quality.watertight
        assert quality.n_boundary_edges == 0
        assert quality.n_nonmanifold_edges == 0
        assert any(f.code == "MESH_WATERTIGHT" for f in report.findings)

    def test_detects_an_open_surface(self):
        """A tetrahedron missing a face has three boundary edges."""
        mesh = Mesh(vertices=TETRA_VERTICES, faces=TETRA_FACES[:3])
        quality, report = assess_mesh_quality(mesh, already_welded=True)

        assert not quality.watertight
        assert quality.n_boundary_edges == 3
        assert any(f.code == "MESH_NOT_CLOSED" for f in report.findings)

    def test_detects_non_manifold_edges(self):
        """A third face on one edge makes it non-manifold.

        This is not hypothetical: Patient_001's right femur segmentation has nine such
        edges, which is exactly the condition that pushes Blender's exact boolean
        solver into its fast fallback and silently changes the cut geometry.
        """
        vertices = np.vstack([TETRA_VERTICES, [[-40.0, -40.0, 20.0]]])
        faces = np.vstack([TETRA_FACES, [[0, 1, 4]]])
        quality, report = assess_mesh_quality(
            Mesh(vertices=vertices, faces=faces), already_welded=True
        )

        assert quality.n_nonmanifold_edges >= 1
        assert not quality.watertight
        finding = next(f for f in report.findings if f.code == "MESH_NONMANIFOLD")
        assert "fast solver" in finding.message

    def test_detects_degenerate_faces(self):
        faces = np.vstack([TETRA_FACES, [[1, 1, 2]]])
        quality, report = assess_mesh_quality(
            Mesh(vertices=TETRA_VERTICES, faces=faces), already_welded=True
        )

        assert quality.n_degenerate_faces == 1
        assert any(f.code == "MESH_DEGENERATE_FACES" for f in report.findings)

    def test_detects_duplicate_faces(self):
        faces = np.vstack([TETRA_FACES, TETRA_FACES[0:1]])
        quality, _ = assess_mesh_quality(
            Mesh(vertices=TETRA_VERTICES, faces=faces), already_welded=True
        )
        assert quality.n_duplicate_faces == 1

    def test_reports_an_empty_mesh_as_an_error(self):
        mesh = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        quality, report = assess_mesh_quality(mesh, already_welded=True)

        assert not report.ok
        assert report.errors[0].code == "MESH_EMPTY"
        assert not quality.watertight

    def test_quality_serialises_for_the_plan(self):
        quality, _ = assess_mesh_quality(
            Mesh(vertices=TETRA_VERTICES, faces=TETRA_FACES), already_welded=True
        )
        record = quality.to_dict()

        assert record["watertight"] is True
        assert record["n_triangles"] == 4


# ----------------------------------------------------------------------
# Report plumbing
# ----------------------------------------------------------------------


class TestQCReport:
    def test_separates_severities(self):
        report = QCReport()
        report.add("A", Severity.ERROR, "bad")
        report.add("B", Severity.WARNING, "hmm")
        report.add("C", Severity.INFO, "fine")

        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert not report.ok

    def test_raises_with_every_error_named(self):
        report = QCReport()
        report.add("A", Severity.ERROR, "first problem")
        report.add("B", Severity.ERROR, "second problem")

        with pytest.raises(QualityControlError) as excinfo:
            report.raise_if_errors()

        assert "first problem" in str(excinfo.value)
        assert "second problem" in str(excinfo.value)

    def test_silent_when_clean(self):
        report = QCReport()
        report.add("C", Severity.INFO, "fine")
        report.raise_if_errors()  # must not raise

    def test_serialises_for_the_plan(self):
        report = QCReport()
        report.add("A", Severity.WARNING, "careful", length_mm=151.0)
        record = report.to_dict()

        assert record[0]["code"] == "A"
        assert record[0]["severity"] == "warning"
        assert record[0]["context"]["length_mm"] == 151.0

    def test_merges_reports(self):
        first, second = QCReport(), QCReport()
        first.add("A", Severity.INFO, "one")
        second.add("B", Severity.ERROR, "two")
        first.extend(second)

        assert len(first.findings) == 2
        assert not first.ok
