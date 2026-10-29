"""The E12 legacy-baseline machinery, on synthetic data.

The Blender run itself needs Blender and the cohort, so it is not tested here; what is
tested is everything around it that decides whether its numbers are right: that the
frozen script is changed only where intended, that the log is read correctly, that a
pose recovered through the recorded transforms is the pose, and that the distance
helpers agree with the brute-force reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from paper.legacy_runner import (
    FROZEN,
    FROZEN_SHA256,
    EXPECTED_CONFIG,
    configure_script,
    cut_face_offset,
    frozen_config,
    parse_log,
    poses_in_stl_frame,
    rigid_check,
    sha256_bytes,
    verify_configured,
)
from paper.meshdist import (
    SurfaceIndex,
    closest_point_pairwise,
    nearest_vertex_distance,
    surface_distances,
)
from paper.pose_compare import plane_difference, plane_offset_mm, pose_difference
from tests.surface_distance import closest_point_on_triangles, point_to_surface


SOURCE = '''"""doc
PATIENT_DIR/
"""
import os
PATIENT_DIR    = r"C:\\old\\patient"
SCALED_STLS_DIR = r"C:\\old\\lib"
CSV_PATH        = os.path.join(SCALED_STLS_DIR, "SizeChart.csv")

SIDE = "L"   # "L" (left)
USE_PCA = "F"
'''


# ---------------------------------------------------------------------------------------
# Configuring the frozen script
# ---------------------------------------------------------------------------------------

def test_frozen_script_is_unchanged():
    assert sha256_bytes(FROZEN.read_bytes()) == FROZEN_SHA256


def test_frozen_configuration_is_the_paper_1_one():
    assert frozen_config(FROZEN.read_text(encoding="utf-8")) == EXPECTED_CONFIG


def test_configure_replaces_only_the_named_globals():
    new, replaced = configure_script(SOURCE, {"PATIENT_DIR": r"D:\knee",
                                              "SIDE": "R"})
    assert set(replaced.values()) == {"PATIENT_DIR", "SIDE"}
    record = verify_configured(SOURCE, new, replaced)
    assert record["frozen_sha256"] != record["configured_sha256"]
    namespace = {}
    exec(compile(new, "configured", "exec"), namespace)
    assert namespace["PATIENT_DIR"] == r"D:\knee"
    assert namespace["SIDE"] == "R"
    assert namespace["CSV_PATH"].startswith(r"C:\old\lib")
    # the docstring line mentioning PATIENT_DIR/ is not an assignment
    assert "PATIENT_DIR/" in new


def test_configure_refuses_other_globals():
    with pytest.raises(ValueError):
        configure_script(SOURCE, {"USE_PCA": "T"})


def test_verify_catches_a_change_outside_the_substituted_lines():
    new, replaced = configure_script(SOURCE, {"SIDE": "R"})
    tampered = new.replace('USE_PCA = "F"', 'USE_PCA = "T"')
    with pytest.raises(ValueError):
        verify_configured(SOURCE, tampered, replaced)


def test_configure_on_the_real_script_keeps_every_other_line():
    frozen = FROZEN.read_text(encoding="utf-8")
    new, replaced = configure_script(frozen, {
        "PATIENT_DIR": r"C:\scratch\P000L", "SCALED_STLS_DIR": r"C:\lib",
        "CSV_PATH": r"C:\lib\SizeChart.csv", "SIDE": "L"})
    record = verify_configured(frozen, new, replaced)
    assert sorted(record["replaced_lines"].values()) == sorted(
        ["PATIENT_DIR", "SCALED_STLS_DIR", "CSV_PATH", "SIDE"])
    assert len(record["replaced_lines"]) == 4


# ---------------------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------------------

FROZEN_LOG = """TKA AUTO PIPELINE v3 — 2026-04-29 18:37:57
Patient: C:\\x\\Patient_002\\Left
Side: left
  PCA disabled — applying manual Z rotation: 90.0°
  Cutting bone: keeping distal 150mm
  TEA disabled — using manual Z rotation: 0.0°
  Femur distal region — ML: 90.2mm  AP: 72.2mm
  Femur full bbox     — Height: 150.0mm
  FEMUR ML = 90.2mm → Size L4 (84.0mm, -6.2mm) [round-down]
  [WARNING] Femur AP cross-check → L3
  Post-valgus re-anchor: -0.672mm
  Component anchor XY: (-1.30, 3.17) mm
[FEMUR DONE] Size: L4  Implant bottom Z: 0.45mm
  Tibia proximal region — ML: 82.6mm  AP: 65.0mm
  [SIZE MISMATCH] Tibia → L1 (using femur-driven L4)
  Tibial anchor XY: (0.00, 3.17) mm  (Y from femoral implant)
  Bone above cut: 22.05mm  (target proximal_cut: 22.05mm)
[TIBIA DONE] Size: L4
PIPELINE COMPLETE — Log: x
"""

LATER_VARIANT_LOG = """TKA AUTO PIPELINE v3 — 2026-07-27 15:05:12
Side: left
  [Retained mode] Pre-detecting TEA from femur...
  PCA shaft correction: 15.9° from Z
  TEA detected: -102.4° from Y axis
    Lateral: (33.4, -6.5, -526.9) mm
  FEMUR ML = 81.5mm → Size L2 (80.0mm, -1.5mm) [round-down]
"""


def test_parse_log_reads_the_numbers():
    log = parse_log(FROZEN_LOG)
    assert log["femur_ml_mm"] == 90.2 and log["femur_ap_mm"] == 72.2
    assert log["size"] == "L4" and log["size_mode"] == "ML"
    assert log["cross_check_size"] == "L3" and log["tibia_size"] == "L1"
    assert log["tibia_ml_mm"] == 82.6 and log["tibia_ap_mm"] == 65.0
    assert log["femoral_anchor_mm"] == [-1.3, 3.17]
    assert log["bone_above_cut_mm"] == 22.05
    assert log["complete"] and log["frozen_v3_signature"]


def test_parse_log_tells_the_later_variant_apart():
    log = parse_log(LATER_VARIANT_LOG)
    assert log["pca"] == "on" and log["tea"] == "on"
    assert log["tea_report_style"] == "lateral_medial"
    assert not log["frozen_v3_signature"]
    assert not log["complete"]


# ---------------------------------------------------------------------------------------
# From the legacy world to the STL frame
# ---------------------------------------------------------------------------------------

def _rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _h(r=np.eye(3), t=(0, 0, 0), scale=1.0):
    m = np.eye(4)
    m[:3, :3] = np.asarray(r) * scale
    m[:3, 3] = t
    return m


def test_poses_in_stl_frame_undo_the_legacy_moves():
    # A bone read in mm, scaled to metres, turned 90 degrees about Z, tilted 6 degrees
    # of valgus about X and shifted -- the legacy femur's history.
    bone = _h(_rot("x", -6.0) @ _rot("z", 90.0), (0.003, -0.002, 0.81), 0.001)
    # The component the planner "really" means, in the STL frame (mm).
    truth = _h(_rot("z", -90.0) @ _rot("y", 4.0), (12.0, -30.0, -790.0))
    component = bone @ truth        # where it ends up in the legacy world
    record = {"objects": [
        {"name": "Femur", "file_to_world": bone.tolist()},
        {"name": "Implant_Femoral_Left_L2", "file_to_world": component.tolist()},
    ]}
    poses = poses_in_stl_frame(record)
    np.testing.assert_allclose(poses["components"]["femoral_component"], truth, atol=1e-9)
    assert poses["rigid"]["femoral_component"]
    np.testing.assert_allclose(np.array(poses["world_mm"]["femoral_component"])[:3, 3],
                               component[:3, 3] * 1000)


def test_cut_face_offset_finds_the_flat_face():
    # A 10 x 10 face 0.4 above the pose origin, and a slanted triangle to ignore.
    v = np.array([[0, 0, 0.4], [10, 0, 0.4], [10, 10, 0.4], [0, 10, 0.4],
                  [0, 0, -5], [10, 0, -4], [0, 10, -5]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    got = cut_face_offset(v, f, np.eye(4))
    assert got["cut_offset_mm"] == pytest.approx(0.4)
    assert got["cut_faces"] == 2
    assert got["cut_area_mm2"] == pytest.approx(100.0)
    assert cut_face_offset(v, f, _h(t=(0, 0, 50)))["cut_faces"] == 0


def test_rigid_check_rejects_scale_and_mirror():
    assert rigid_check(_h(_rot("z", 30)))
    assert not rigid_check(_h(_rot("z", 30), scale=0.001))
    assert not rigid_check(_h(np.diag([1, 1, -1])))


# ---------------------------------------------------------------------------------------
# Comparing poses and planes
# ---------------------------------------------------------------------------------------

AXES = dict(anterior=np.array([0, -1.0, 0]), lateral=np.array([1.0, 0, 0]),
            proximal=np.array([0, 0, 1.0]))


def test_pose_difference_splits_translation_and_rotation():
    new = _h(np.eye(3), (10, 20, 30))
    legacy = _h(_rot("y", 5.0), (10, 17, 31))   # 3 mm anterior, 1 mm proximal
    d = pose_difference(legacy, new, **AXES)
    assert d["anterior_mm"] == pytest.approx(3.0)
    assert d["proximal_mm"] == pytest.approx(1.0)
    assert d["lateral_mm"] == pytest.approx(0.0)
    assert d["position_mm"] == pytest.approx(np.sqrt(10))
    assert d["rotation_deg"] == pytest.approx(5.0)
    assert d["normal_angle_deg"] == pytest.approx(5.0)
    # rotation about +Y tips the normal toward +X, which AXES calls lateral
    assert d["coronal_tilt_deg"] == pytest.approx(5.0)
    assert d["sagittal_tilt_deg"] == pytest.approx(0.0, abs=1e-9)
    assert d["axial_turn_deg"] == pytest.approx(0.0, abs=1e-9)


def test_pose_difference_axial_turn():
    d = pose_difference(_h(_rot("z", 7.0)), _h(np.eye(3)), **AXES)
    assert d["axial_turn_deg"] == pytest.approx(7.0)
    assert d["normal_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_plane_offset_parallel_and_tilted():
    z = np.array([0, 0, 1.0])
    assert plane_offset_mm((0, 0, 5), z, (0, 0, 2), z, (0, 0, 0), z) == pytest.approx(3.0)
    tilted = np.array([np.sin(0.1), 0, np.cos(0.1)])
    # Both planes pass through the centre's axis at the same height: no offset there.
    assert plane_offset_mm((50, 0, 2 - 50 * np.tan(0.1)), tilted, (0, 0, 2), z,
                           (0, 0, 0), z) == pytest.approx(0.0, abs=1e-9)
    d = plane_difference((0, 0, 4), tilted, (0, 0, 2), z, centre=(0, 0, 0), proximal=z)
    assert d["angle_deg"] == pytest.approx(np.degrees(0.1))
    assert d["offset_at_centre_mm"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------------------

def _sphere(n=24, r=10.0, centre=(0, 0, 0)):
    from types import SimpleNamespace

    u, v = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
    pts = np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)], -1)
    pts = pts.reshape(-1, 3) * r + centre
    rows, cols = 2 * n, n
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b = i * cols + j, i * cols + j + 1
            c, d = (i + 1) * cols + j, (i + 1) * cols + j + 1
            faces += [(a, b, d), (a, d, c)]
    return SimpleNamespace(vertices=pts, faces=np.array(faces))


def test_pairwise_closest_point_matches_the_reference():
    rng = np.random.default_rng(1)
    tri = rng.normal(size=(300, 3, 3))
    p = rng.normal(size=(300, 3)) * 2
    got = closest_point_pairwise(p, tri[:, 0], tri[:, 1], tri[:, 2])
    want = np.array([closest_point_on_triangles(p[i:i + 1], tri[i:i + 1])[0, 0]
                     for i in range(len(p))])
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_surface_index_matches_brute_force():
    mesh = _sphere()
    rng = np.random.default_rng(2)
    points = rng.normal(size=(400, 3)) * 6
    index = SurfaceIndex(mesh.vertices, mesh.faces)
    d, exact = index.distance(points)
    brute = point_to_surface(points, mesh)
    np.testing.assert_allclose(d[exact], brute[exact], atol=1e-9)
    assert np.all(d[~exact] >= brute[~exact] - 1e-9)


def test_surface_distances_of_offset_spheres():
    a, b = _sphere(), _sphere(centre=(0.5, 0, 0))
    same = surface_distances(a, a)
    assert same["hausdorff"] == pytest.approx(0.0, abs=1e-9)
    moved = surface_distances(a, b, sample=5000)
    assert 0.4 < moved["hausdorff"] <= 0.5 + 1e-6
    assert 0 < moved["mean"] < moved["hausdorff"]


def test_nearest_vertex_distance_exact_within_a_cell():
    rng = np.random.default_rng(3)
    targets = rng.uniform(0, 10, size=(2000, 3))
    points = targets[:500] + rng.normal(scale=1e-4, size=(500, 3))
    d = nearest_vertex_distance(points, targets, cell=0.05)
    brute = np.linalg.norm(points[:, None] - targets[None], axis=2).min(axis=1)
    np.testing.assert_allclose(d, brute, atol=1e-12)
    far = nearest_vertex_distance(np.array([[100.0, 100, 100]]), targets, cell=0.05)
    assert np.isinf(far[0])
