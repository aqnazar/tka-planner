"""Landmark registry, schema and Slicer interchange.

The tests that matter most here are the refusals. A landmark reader that guesses when
information is absent -- a missing coordinate system, an unrecognised label, a status
that disagrees with the data it carries -- produces a plan that looks fine and is wrong.
Each of those cases is asserted to raise.
"""

import json

import numpy as np
import pytest

from tka_planner.core.landmarks import (
    REGISTRY,
    CoordinateSystem,
    Landmark,
    LandmarkSet,
    LandmarkStatus,
    MissingLandmarks,
    definitions_for_bone,
    load_landmarks,
    read_landmark_set,
    read_slicer_fcsv,
    read_slicer_markups,
    resolve_landmark_id,
    write_landmark_set,
)
from tka_planner.core.sides import Side


# ----------------------------------------------------------------------
# Coordinate systems
# ----------------------------------------------------------------------


class TestCoordinateSystem:
    @pytest.mark.parametrize("value,expected", [
        ("LPS", CoordinateSystem.LPS),
        ("lps", CoordinateSystem.LPS),
        ("RAS", CoordinateSystem.RAS),
        (1, CoordinateSystem.LPS),   # legacy .fcsv numeric codes
        (0, CoordinateSystem.RAS),
        ("1", CoordinateSystem.LPS),
    ])
    def test_parses_slicer_spellings(self, value, expected):
        assert CoordinateSystem.parse(value) is expected

    @pytest.mark.parametrize("value", ["", "XYZ", "world", None, 7])
    def test_refuses_unknown_systems(self, value):
        """Anything not clearly LPS or RAS must raise, never default."""
        with pytest.raises(ValueError, match="(?i)unrecognised|unknown"):
            CoordinateSystem.parse(value)

    def test_lps_to_ras_negates_first_two_axes(self):
        points = np.array([[10.0, 20.0, 30.0]])
        converted = CoordinateSystem.LPS.convert_to(points, CoordinateSystem.RAS)
        assert np.allclose(converted, [[-10.0, -20.0, 30.0]])

    def test_conversion_is_its_own_inverse(self):
        """Applying the transform when unnecessary is as damaging as omitting it."""
        rng = np.random.default_rng(0)
        points = rng.normal(scale=100.0, size=(50, 3))

        there = CoordinateSystem.LPS.convert_to(points, CoordinateSystem.RAS)
        back = CoordinateSystem.RAS.convert_to(there, CoordinateSystem.LPS)
        assert np.allclose(points, back)

    def test_same_system_is_a_no_op_copy(self):
        points = np.array([[1.0, 2.0, 3.0]])
        result = CoordinateSystem.LPS.convert_to(points, CoordinateSystem.LPS)

        assert np.allclose(result, points)
        assert result is not points  # a copy, so callers cannot alias the input


# ----------------------------------------------------------------------
# Status semantics
# ----------------------------------------------------------------------


class TestLandmarkStatus:
    def test_only_positioned_statuses_carry_coordinates(self):
        assert LandmarkStatus.PRESENT.has_position
        assert LandmarkStatus.DERIVED.has_position
        assert LandmarkStatus.ESTIMATED.has_position
        assert not LandmarkStatus.OUT_OF_SCAN.has_position
        assert not LandmarkStatus.NOT_PICKED.has_position
        assert not LandmarkStatus.REJECTED.has_position

    def test_out_of_scan_is_distinct_from_not_picked(self):
        """The distinction the whole capability model rests on.

        Out-of-scan is a permanent property of the imaging; not-picked is an unfinished
        workflow. Only the first justifies substituting an assumption.
        """
        assert LandmarkStatus.OUT_OF_SCAN.is_permanently_unavailable
        assert not LandmarkStatus.NOT_PICKED.is_permanently_unavailable


# ----------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------


class TestRegistry:
    def test_ids_are_namespaced_by_bone(self):
        for landmark_id, definition in REGISTRY.items():
            assert landmark_id.startswith(f"{definition.bone}.")

    def test_every_landmark_has_a_prose_definition(self):
        """The definition is reproduced in the report, so it must be readable."""
        for definition in REGISTRY.values():
            assert len(definition.definition) > 40
            assert definition.display_name

    def test_covers_the_bones_this_increment_needs(self):
        assert {d.bone for d in REGISTRY.values()} >= {
            "femur", "tibia", "fibula", "patella"
        }

    def test_femoral_head_and_ankle_are_flagged_out_of_scan(self):
        """Both are absent from every case in this cohort."""
        assert REGISTRY["femur.head_centre"].typically_out_of_scan
        assert REGISTRY["tibia.ankle_centre"].typically_out_of_scan

    def test_stea_and_atea_share_the_lateral_landmark(self):
        """Both axes use the lateral prominence; only the medial end differs."""
        lateral = REGISTRY["femur.epicondyle_lateral"]
        assert "stea" in lateral.feeds and "atea" in lateral.feeds
        assert REGISTRY["femur.epicondyle_medial_sulcus"].feeds == (
            "stea", "condylar_twist_angle"
        )
        assert REGISTRY["femur.epicondyle_medial_prominence"].feeds == ("atea",)

    def test_knee_centre_does_not_come_from_the_epicondyles(self):
        """Deliberate: sharing a landmark between the coronal and rotational
        constructions correlates their errors and flatters the sensitivity analysis."""
        notch = REGISTRY["femur.notch_centre"]
        assert "femoral_knee_centre" in notch.feeds
        for epicondyle in ("femur.epicondyle_lateral",
                           "femur.epicondyle_medial_sulcus"):
            assert "femoral_knee_centre" not in REGISTRY[epicondyle].feeds

    def test_patella_is_registered_but_not_picked_yet(self):
        """Registered now so the schema need not be re-versioned later."""
        for definition in definitions_for_bone("patella"):
            assert not definition.picked_in_increment_1

    def test_picked_only_filter_excludes_derived_landmarks(self):
        picked = definitions_for_bone("femur", picked_only=True)
        assert all(d.picked_in_increment_1 for d in picked)
        assert not any("canal_centre" in d.id for d in picked)


class TestLabelResolution:
    @pytest.mark.parametrize("label", [
        "femur.epicondyle_lateral",
        "Lateral epicondyle",
        "lateral_epicondyle",
        "LE",
        "  LATERAL EPICONDYLE  ",
        "epicondyle_lateral",
    ])
    def test_resolves_the_many_ways_people_label_a_point(self, label):
        assert resolve_landmark_id(label) == "femur.epicondyle_lateral"

    @pytest.mark.parametrize("label", ["P-1", "", "some_other_thing", "F-3"])
    def test_returns_none_for_unrecognised_labels(self, label):
        """Slicer's default labels must not silently become anatomy."""
        assert resolve_landmark_id(label) is None


# ----------------------------------------------------------------------
# Landmark objects
# ----------------------------------------------------------------------


class TestLandmark:
    def test_rejects_unregistered_ids(self):
        with pytest.raises(KeyError, match="Unknown landmark id"):
            Landmark(id="femur.invented_point", position_mm=[0, 0, 0],
                     status=LandmarkStatus.PRESENT)

    def test_status_and_data_must_agree(self):
        """A status promising a position must have one, and vice versa.

        Without this, a REJECTED landmark could keep its coordinates and be picked up
        by code that only checks for None.
        """
        with pytest.raises(ValueError, match="promises a position"):
            Landmark(id="femur.notch_centre", status=LandmarkStatus.PRESENT)

        with pytest.raises(ValueError, match="would let a discarded point be used"):
            Landmark(id="femur.notch_centre", position_mm=[1, 2, 3],
                     status=LandmarkStatus.REJECTED)

    def test_rejects_non_finite_positions(self):
        with pytest.raises(ValueError, match="NaN or infinity"):
            Landmark(id="femur.notch_centre", position_mm=[1, np.nan, 3],
                     status=LandmarkStatus.PRESENT)

    def test_out_of_scan_needs_no_position(self):
        landmark = Landmark(
            id="femur.head_centre",
            status=LandmarkStatus.OUT_OF_SCAN,
            reason="knee-only field of view",
        )
        assert not landmark.is_usable
        assert landmark.reason == "knee-only field of view"


# ----------------------------------------------------------------------
# LandmarkSet
# ----------------------------------------------------------------------


@pytest.fixture
def knee_set():
    return LandmarkSet(
        case_id="CASE_001",
        side="left",
        coordinate_system="LPS",
        landmarks=[
            Landmark(id="femur.epicondyle_lateral", position_mm=[110.0, -5.0, -545.0],
                     status=LandmarkStatus.PRESENT, origin="picked"),
            Landmark(id="femur.epicondyle_medial_sulcus",
                     position_mm=[45.0, -5.0, -545.0],
                     status=LandmarkStatus.PRESENT, origin="picked"),
            Landmark(id="femur.notch_centre", position_mm=[78.0, 5.0, -550.0],
                     status=LandmarkStatus.PRESENT, origin="picked"),
            Landmark(id="femur.head_centre", status=LandmarkStatus.OUT_OF_SCAN,
                     reason="knee-only field of view"),
        ],
    )


class TestLandmarkSet:
    def test_reports_identity_and_frame(self, knee_set):
        assert knee_set.case_id == "CASE_001"
        assert knee_set.side is Side.LEFT
        assert knee_set.coordinate_system is CoordinateSystem.LPS

    def test_get_returns_a_placeholder_rather_than_raising(self, knee_set):
        """Absence must be reportable, not fatal -- the report needs to say *why*."""
        landmark = knee_set.get("tibia.spine_medial")
        assert landmark.status is LandmarkStatus.NOT_PICKED
        assert not landmark.is_usable

    def test_get_still_rejects_ids_that_do_not_exist(self, knee_set):
        with pytest.raises(KeyError):
            knee_set.get("femur.not_a_real_landmark")

    def test_position_is_none_for_unusable_landmarks(self, knee_set):
        assert knee_set.position("femur.head_centre") is None
        assert knee_set.position("femur.notch_centre") is not None

    def test_require_names_what_is_missing_and_why(self, knee_set):
        with pytest.raises(MissingLandmarks) as excinfo:
            knee_set.require("femur.notch_centre", "femur.head_centre")

        message = str(excinfo.value)
        assert "femur.head_centre" in message
        assert "out_of_scan" in message
        assert "femur.notch_centre" not in message  # that one was fine

    def test_require_returns_positions_when_satisfied(self, knee_set):
        positions = knee_set.require("femur.epicondyle_lateral", "femur.notch_centre")
        assert len(positions) == 2
        assert np.allclose(positions[0], [110.0, -5.0, -545.0])

    def test_availability_helpers(self, knee_set):
        assert knee_set.available("femur.notch_centre")
        assert not knee_set.available("femur.notch_centre", "femur.head_centre")
        assert knee_set.missing_of(
            "femur.notch_centre", "femur.head_centre"
        ) == ["femur.head_centre"]

    def test_rejects_duplicate_landmarks(self, knee_set):
        with pytest.raises(ValueError, match="already in the set"):
            knee_set.add(Landmark(id="femur.notch_centre", position_mm=[0, 0, 0],
                                  status=LandmarkStatus.PRESENT))

    def test_converts_coordinate_system_and_preserves_status(self, knee_set):
        converted = knee_set.to_coordinate_system("RAS")

        assert converted.coordinate_system is CoordinateSystem.RAS
        assert np.allclose(
            converted.position("femur.epicondyle_lateral"), [-110.0, 5.0, -545.0]
        )
        # An out-of-scan landmark stays out of scan; it has nothing to convert.
        assert converted.get("femur.head_centre").status is LandmarkStatus.OUT_OF_SCAN

    def test_conversion_leaves_the_original_untouched(self, knee_set):
        original = knee_set.position("femur.epicondyle_lateral").copy()
        knee_set.to_coordinate_system("RAS")
        assert np.allclose(knee_set.position("femur.epicondyle_lateral"), original)

    def test_positions_array_gathers_only_usable_points(self, knee_set):
        assert knee_set.positions_array().shape == (3, 3)


# ----------------------------------------------------------------------
# Slicer interchange
# ----------------------------------------------------------------------


def write_markups_json(path, points, coordinate_system="LPS", units="mm"):
    document = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/main/Modules/"
                   "Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": coordinate_system,
            "coordinateUnits": units,
            "controlPoints": [
                {"id": str(i + 1), "label": label, "position": list(position)}
                for i, (label, position) in enumerate(points)
            ],
        }],
    }
    path.write_text(json.dumps(document, indent=2))
    return path


def write_fcsv(path, points, header_system="LPS"):
    lines = [
        "# Markups fiducial file version = 5.6",
        f"# CoordinateSystem = {header_system}",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    for i, (label, (x, y, z)) in enumerate(points):
        lines.append(f"vtkMRMLMarkupsFiducialNode_{i},{x},{y},{z},"
                     f"0,0,0,1,1,1,0,{label},,")
    path.write_text("\n".join(lines) + "\n")
    return path


SAMPLE_POINTS = [
    ("Lateral epicondyle", (110.0, -5.0, -545.0)),
    ("Medial epicondylar sulcus", (45.0, -5.0, -545.0)),
    ("Intercondylar notch centre", (78.0, 5.0, -550.0)),
]


class TestSlicerMarkupsJson:
    def test_reads_labels_positions_and_frame(self, tmp_path):
        path = write_markups_json(tmp_path / "femur.mrk.json", SAMPLE_POINTS)
        landmark_set = read_slicer_markups(path, case_id="CASE_001", side="left")

        assert landmark_set.coordinate_system is CoordinateSystem.LPS
        assert len(landmark_set) == 3
        assert np.allclose(
            landmark_set.position("femur.epicondyle_lateral"), [110.0, -5.0, -545.0]
        )

    def test_records_unmatched_labels_instead_of_inventing_anatomy(self, tmp_path):
        """Slicer's default ``P-1`` labels must not become landmarks."""
        path = write_markups_json(
            tmp_path / "mixed.mrk.json",
            SAMPLE_POINTS + [("P-1", (0.0, 0.0, 0.0))],
        )
        landmark_set = read_slicer_markups(path, case_id="CASE_001", side="left")

        assert len(landmark_set) == 3
        assert landmark_set.source["unmatched_labels"] == ["P-1"]

    def test_rejects_a_missing_coordinate_system(self, tmp_path):
        path = tmp_path / "no_frame.mrk.json"
        path.write_text(json.dumps({
            "markups": [{"type": "Fiducial", "controlPoints": []}]
        }))
        with pytest.raises(ValueError, match="omits 'coordinateSystem'"):
            read_slicer_markups(path, case_id="CASE_001", side="left")

    def test_rejects_mixed_coordinate_systems(self, tmp_path):
        path = tmp_path / "mixed_frames.mrk.json"
        path.write_text(json.dumps({"markups": [
            {"coordinateSystem": "LPS", "controlPoints": []},
            {"coordinateSystem": "RAS", "controlPoints": []},
        ]}))
        with pytest.raises(ValueError, match="mixes coordinate systems"):
            read_slicer_markups(path, case_id="CASE_001", side="left")

    def test_rejects_non_millimetre_units(self, tmp_path):
        path = write_markups_json(
            tmp_path / "metres.mrk.json", SAMPLE_POINTS, units="m"
        )
        with pytest.raises(ValueError, match="exclusively in millimetres"):
            read_slicer_markups(path, case_id="CASE_001", side="left")

    def test_rejects_a_file_with_no_markups(self, tmp_path):
        path = tmp_path / "empty.mrk.json"
        path.write_text(json.dumps({"@schema": "x", "markups": []}))
        with pytest.raises(ValueError, match="no 'markups' array"):
            read_slicer_markups(path, case_id="CASE_001", side="left")

    def test_ras_input_converts_to_lps(self, tmp_path):
        """The check that a mis-declared frame would visibly fail."""
        path = write_markups_json(
            tmp_path / "ras.mrk.json", SAMPLE_POINTS, coordinate_system="RAS"
        )
        landmark_set = read_slicer_markups(
            path, case_id="CASE_001", side="left"
        ).to_coordinate_system("LPS")

        assert np.allclose(
            landmark_set.position("femur.epicondyle_lateral"), [-110.0, 5.0, -545.0]
        )


class TestSlicerFcsv:
    def test_reads_the_legacy_format(self, tmp_path):
        path = write_fcsv(tmp_path / "femur.fcsv", SAMPLE_POINTS)
        landmark_set = read_slicer_fcsv(path, case_id="CASE_001", side="left")

        assert landmark_set.coordinate_system is CoordinateSystem.LPS
        assert len(landmark_set) == 3

    @pytest.mark.parametrize("code,expected", [
        ("0", CoordinateSystem.RAS),
        ("1", CoordinateSystem.LPS),
    ])
    def test_understands_legacy_numeric_frame_codes(self, tmp_path, code, expected):
        path = write_fcsv(tmp_path / f"legacy_{code}.fcsv", SAMPLE_POINTS,
                          header_system=code)
        assert read_slicer_fcsv(
            path, case_id="CASE_001", side="left"
        ).coordinate_system is expected

    def test_rejects_a_file_without_the_frame_header(self, tmp_path):
        path = tmp_path / "headerless.fcsv"
        path.write_text(
            "# Markups fiducial file version = 5.6\n"
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,"
            "associatedNodeID\n"
            "n1,1,2,3,0,0,0,1,1,1,0,Lateral epicondyle,,\n"
        )
        with pytest.raises(ValueError, match="Refusing to guess"):
            read_slicer_fcsv(path, case_id="CASE_001", side="left")

    def test_agrees_with_the_json_format(self, tmp_path):
        from_json = read_slicer_markups(
            write_markups_json(tmp_path / "a.mrk.json", SAMPLE_POINTS),
            case_id="CASE_001", side="left")
        from_fcsv = read_slicer_fcsv(
            write_fcsv(tmp_path / "a.fcsv", SAMPLE_POINTS),
            case_id="CASE_001", side="left")

        for landmark in from_json:
            assert np.allclose(
                landmark.position_mm, from_fcsv.position(landmark.id)
            )


class TestRealSlicerFile:
    """Against an actual Slicer export, not a fixture written by these tests.

    Fixtures written by the same test file that reads them only prove the reader is
    self-consistent. This one is a genuine 3D Slicer export (a plane markup from the
    calibration stand, containing no patient data), vendored into the repository so the
    check travels with it.
    """

    @property
    def real_file(self):
        from pathlib import Path

        return Path(__file__).parent / "fixtures" / "slicer_real_export.mrk.json"

    def test_parses_a_genuine_slicer_export(self):
        landmark_set = read_slicer_markups(
            self.real_file, case_id="CASE_STAND", side="left"
        )

        # It is a Plane markup carrying Slicer's default 'P-1' label, so no anatomy is
        # matched -- but the header must parse, and the label must be reported as
        # unmatched rather than silently dropped or invented into a landmark.
        assert landmark_set.coordinate_system is CoordinateSystem.LPS
        assert landmark_set.source["unmatched_labels"] == ["P-1"]
        assert len(landmark_set) == 0

    def test_records_the_slicer_schema_version(self):
        landmark_set = read_slicer_markups(
            self.real_file, case_id="CASE_STAND", side="left"
        )
        assert "markups-schema" in landmark_set.source["schema"]


# ----------------------------------------------------------------------
# Native format
# ----------------------------------------------------------------------


class TestNativeFormat:
    def test_round_trips(self, tmp_path, knee_set):
        path = write_landmark_set(knee_set, tmp_path / "landmarks.json")
        restored = read_landmark_set(path)

        assert restored.case_id == knee_set.case_id
        assert restored.side is knee_set.side
        assert restored.coordinate_system is knee_set.coordinate_system
        assert len(restored) == len(knee_set)

        for landmark in knee_set:
            other = restored.get(landmark.id)
            assert other.status is landmark.status
            if landmark.is_usable:
                assert np.allclose(other.position_mm, landmark.position_mm)

    def test_preserves_out_of_scan_reasons(self, tmp_path, knee_set):
        """The reason text reaches the report, so it must survive serialisation."""
        path = write_landmark_set(knee_set, tmp_path / "landmarks.json")
        restored = read_landmark_set(path)

        assert restored.get("femur.head_centre").reason == "knee-only field of view"

    def test_writes_human_readable_json(self, tmp_path, knee_set):
        path = write_landmark_set(knee_set, tmp_path / "landmarks.json")
        document = json.loads(path.read_text())

        assert document["units"] == "mm"
        assert document["coordinate_system"] == "LPS"
        assert document["side"] == "left"

    def test_rejects_a_foreign_schema_version(self, tmp_path):
        path = tmp_path / "future.json"
        path.write_text(json.dumps({
            "schema_version": "tka-landmarks/9.9.9",
            "case_id": "CASE_001", "side": "left",
            "coordinate_system": "LPS", "units": "mm", "points": [],
        }))
        with pytest.raises(ValueError, match="schema version"):
            read_landmark_set(path)

    def test_rejects_non_millimetre_units(self, tmp_path):
        path = tmp_path / "metres.json"
        path.write_text(json.dumps({
            "schema_version": "tka-landmarks/1.0.0",
            "case_id": "CASE_001", "side": "left",
            "coordinate_system": "LPS", "units": "m", "points": [],
        }))
        with pytest.raises(ValueError, match="exclusively in millimetres"):
            read_landmark_set(path)


class TestLoadDispatch:
    def test_dispatches_on_extension(self, tmp_path, knee_set):
        native = write_landmark_set(knee_set, tmp_path / "native.json")
        markups = write_markups_json(tmp_path / "x.mrk.json", SAMPLE_POINTS)
        fcsv = write_fcsv(tmp_path / "x.fcsv", SAMPLE_POINTS)

        assert load_landmarks(native).case_id == "CASE_001"
        assert len(load_landmarks(markups, case_id="C", side="left")) == 3
        assert len(load_landmarks(fcsv, case_id="C", side="left")) == 3

    def test_slicer_files_require_explicit_case_identity(self, tmp_path):
        """Slicer exports carry no case id or side; they must be supplied."""
        path = write_markups_json(tmp_path / "x.mrk.json", SAMPLE_POINTS)
        with pytest.raises(ValueError, match="carries no case identity"):
            load_landmarks(path)

    def test_rejects_unknown_file_types(self, tmp_path):
        path = tmp_path / "landmarks.txt"
        path.write_text("nope")
        with pytest.raises(ValueError, match="Unrecognised landmark file type"):
            load_landmarks(path)
