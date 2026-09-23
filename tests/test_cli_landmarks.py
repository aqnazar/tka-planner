"""The rating study's command-line path: template out, rater's file back in.

The conversion is where a coordinate-system mistake would otherwise enter silently, so
the test that matters most is that a file whose points miss the bone is refused rather
than written.
"""

import json

import numpy as np

from tka_planner.cli import main
from tka_planner.core.landmarks import LandmarkStatus, read_landmark_set
from tka_planner.geom import mesh as gm


def place(template_path, positions):
    """Fill named points in a template, as a rater would in Slicer."""
    document = json.loads(template_path.read_text())
    for point in document["markups"][0]["controlPoints"]:
        if point["label"] in positions:
            point["position"] = list(positions[point["label"]])
            point["positionStatus"] = "defined"
    template_path.write_text(json.dumps(document))
    return template_path


def femur_box(tmp_path, centre):
    """A stand-in femur: a box around where the picks are, in LPS millimetres."""
    box = gm.box((80.0, 60.0, 150.0), gm.translation(centre))
    return gm.write_stl(box, tmp_path / "femur.stl")


PICKS = {
    "femur.epicondyle_lateral": (110.0, 0.0, -540.0),
    "femur.notch_centre": (80.0, 5.0, -550.0),
}


def test_template_then_convert_writes_a_native_file_with_the_rater(tmp_path):
    template = tmp_path / "t.mrk.json"
    assert main(["landmarks", "template", "--out", str(template)]) == 0
    place(template, PICKS)
    femur = femur_box(tmp_path, (80.0, 0.0, -545.0))

    out = tmp_path / "r1_s1.json"
    code = main(["landmarks", "convert", str(template), "--case-id", "C",
                 "--side", "left", "--rater", "R1", "--session", "1",
                 "--femur", str(femur), "--out", str(out)])

    assert code == 0
    landmarks = read_landmark_set(out)
    assert landmarks.source["rater"] == "R1"
    assert landmarks.coordinate_system.value == "LPS"
    assert landmarks.get("femur.notch_centre").status is LandmarkStatus.PRESENT
    assert np.allclose(landmarks.position("femur.notch_centre"), (80.0, 5.0, -550.0))
    # The unplaced rest of the template did not become landmarks.
    assert len(landmarks) == len(PICKS)


def test_points_off_the_bone_are_refused_and_nothing_is_written(tmp_path):
    """What a file declared LPS but picked in RAS looks like: mirrored off the bone."""
    template = tmp_path / "t.mrk.json"
    main(["landmarks", "template", "--out", str(template)])
    place(template, PICKS)
    femur = femur_box(tmp_path, (-80.0, 0.0, -545.0))

    out = tmp_path / "r1_s1.json"
    code = main(["landmarks", "convert", str(template), "--case-id", "C",
                 "--side", "left", "--rater", "R1", "--session", "1",
                 "--femur", str(femur), "--out", str(out)])

    assert code == 2
    assert not out.exists()
