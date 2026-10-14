"""Monte Carlo propagation of landmark picking error.

The machinery must be trustworthy before any interval it produces is quoted, and the
properties that make it so can all be checked on synthetic knees: no dispersion gives no
spread, the spread grows in proportion to the dispersion, a seed fixes the result, a
surface landmark stays on its surface, and a draw that cannot be planned is counted
rather than crashing the run.
"""

import numpy as np
import pytest

from tka_planner.core.landmarks import REGISTRY
from tka_planner.core.meshio import Mesh
from tka_planner.pipeline import measure_from_landmarks
from tka_planner.validation.montecarlo import (
    PLACEHOLDER_TABLES,
    Dispersion,
    SurfaceModel,
    perturb,
    placeholder_dispersion,
    propagate,
    read_dispersion,
    write_dispersion,
)
from tka_planner.validation.synthetic import bone_like, synthetic_knee


def measured(side="left", **anatomy):
    landmarks = synthetic_knee(side, **anatomy)
    return measure_from_landmarks(
        landmarks, bone_like(landmarks, "femur"), bone_like(landmarks, "tibia"),
        femur_path="femur.stl", tibia_path="tibia.stl", case_id="CASE_MC", side=side,
    )


def uniform(sd_mm):
    return Dispersion.isotropic(
        {k: sd_mm for k, d in REGISTRY.items() if d.picked_in_increment_1},
        label=f"test {sd_mm} mm",
    )


@pytest.fixture(scope="module")
def case():
    return measured()


class TestSpread:
    def test_zero_dispersion_gives_zero_spread(self, case):
        result = propagate(case, uniform(0.0), n=5, seed=1, surface="free")
        assert result.n_failed == 0
        for row in result.summary():
            assert row["sd"] == pytest.approx(0.0, abs=1e-9), row["output"]
            assert row["mean"] == pytest.approx(row["nominal"], abs=1e-9), row["output"]

    def test_the_nominal_plan_sits_exactly_on_itself(self, case):
        result = propagate(case, uniform(0.0), n=1, seed=1, surface="free")
        for name, value in result.nominal.items():
            if name.endswith(("_deg",)) and name.startswith(("bone.", "frame.")):
                assert value == pytest.approx(0.0, abs=1e-9), name

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_spread_scales_with_the_dispersion(self, side):
        # The same seed draws the same standard normals, so doubling every SD doubles
        # every displacement; for small displacements every output is near linear.
        knee = measured(side)
        small = {r["output"]: r["sd"] for r in
                 propagate(knee, uniform(0.1), n=40, seed=3, surface="free").summary()}
        large = {r["output"]: r["sd"] for r in
                 propagate(knee, uniform(0.2), n=40, seed=3, surface="free").summary()}
        for name in ("metric.mldfa_deg", "metric.mpta_deg", "metric.condylar_twist_deg",
                     "bone.femoral_depth_medial_mm", "bone.tibial_depth_lateral_mm",
                     "bone.femoral_component_rotation_deg"):
            assert small[name] > 0, name
            assert large[name] / small[name] == pytest.approx(2.0, rel=0.03), name

    def test_landmarks_without_a_row_are_held_fixed(self, case):
        only = Dispersion.isotropic({"tibia.plateau_medial_lowest": 1.0})
        result = propagate(case, only, n=20, seed=2, surface="free")
        assert result.perturbed == ("tibia.plateau_medial_lowest",)
        assert "femur.condyle_distal_medial" in result.held_fixed
        sd = {r["output"]: r["sd"] for r in result.summary()}
        # Nothing femoral reads the tibial plateau.
        assert sd["metric.mldfa_deg"] == pytest.approx(0.0, abs=1e-9)
        assert sd["metric.mpta_deg"] > 0


class TestReproducibility:
    def test_a_seed_fixes_every_draw(self, case):
        a = propagate(case, uniform(1.5), n=15, seed=11, surface="free")
        b = propagate(case, uniform(1.5), n=15, seed=11, surface="free")
        assert a.samples.keys() == b.samples.keys()
        for name in a.samples:
            np.testing.assert_array_equal(a.samples[name], b.samples[name])
        assert a.sizes == b.sizes

    def test_a_different_seed_draws_differently(self, case):
        a = propagate(case, uniform(1.5), n=15, seed=11, surface="free")
        b = propagate(case, uniform(1.5), n=15, seed=12, surface="free")
        assert not np.array_equal(a.samples["metric.mldfa_deg"],
                                  b.samples["metric.mldfa_deg"])


class TestFailures:
    def test_unbuildable_frames_are_counted_not_raised(self, case):
        # Re-picks this wild swap the Akagi line's ends often enough to turn the tibial
        # frame backwards, which the frame builder refuses.
        wild = Dispersion.isotropic({"tibia.pcl_insertion_midpoint": 40.0,
                                     "tibia.tubercle_patellar_tendon_medial_border": 40.0})
        result = propagate(case, wild, n=60, seed=5, surface="free")
        assert result.failures.get("FrameConstructionError", 0) > 0
        assert result.n_failed + len(result.sizes) == 60
        assert all(len(v) == len(result.sizes) for v in result.samples.values())
        assert "FrameConstructionError" in result.failure_examples


def _plane_patch(centre, size_mm=30.0, steps=15):
    """A flat square mesh in the transverse plane through ``centre``."""
    xs = np.linspace(-size_mm / 2, size_mm / 2, steps + 1)
    grid = np.array([[x, y, 0.0] for y in xs for x in xs]) + centre
    faces = []
    for j in range(steps):
        for i in range(steps):
            a = j * (steps + 1) + i
            faces += [[a, a + 1, a + steps + 2], [a, a + steps + 2, a + steps + 1]]
    return Mesh(vertices=grid, faces=np.array(faces))


class TestSurface:
    @pytest.fixture
    def setup(self, case):
        target = "femur.condyle_distal_medial"
        point = case.landmarks.position(target)
        meshes = {"femur": _plane_patch(point), "tibia": _plane_patch(point - 500.0)}
        frames = {"femur": case.femoral_frame, "tibia": case.tibial_frame}
        return case.landmarks, target, point, meshes, frames, Dispersion.isotropic({target: 3.0})

    def _heights(self, landmarks, target, meshes, frames, dispersion, mode):
        model = SurfaceModel.build(landmarks, meshes, mode)
        rng = np.random.default_rng(0)
        return model, np.array([
            perturb(landmarks, dispersion, rng, frames=frames, surface=model)
            .position(target) for _ in range(50)])

    def test_tangent_moves_a_surface_landmark_within_its_surface(self, setup):
        landmarks, target, point, meshes, frames, dispersion = setup
        model, moved = self._heights(landmarks, target, meshes, frames, dispersion, "tangent")
        assert model.on_surface(target)
        assert np.allclose(moved[:, 2], point[2], atol=1e-9)
        assert moved[:, 0].std() > 1.0 and moved[:, 1].std() > 1.0

    def test_project_snaps_onto_the_mesh(self, setup):
        landmarks, target, point, meshes, frames, dispersion = setup
        _, moved = self._heights(landmarks, target, meshes, frames, dispersion, "project")
        assert np.allclose(moved[:, 2], point[2], atol=1e-9)
        assert np.all(np.abs(moved[:, :2] - point[:2]) <= 15.0 + 1e-9)

    def test_free_leaves_the_surface(self, setup):
        landmarks, target, point, meshes, frames, dispersion = setup
        _, moved = self._heights(landmarks, target, meshes, frames, dispersion, "free")
        assert moved[:, 2].std() > 1.0

    def test_off_surface_landmarks_move_freely(self, setup):
        landmarks, _, _, meshes, _, _ = setup
        model = SurfaceModel.build(landmarks, meshes, "tangent")
        assert not model.on_surface("femur.canal_centre_distal")
        assert not model.on_surface("femur.condyle_distal_lateral")  # 75 mm away

    def test_a_surface_model_needs_meshes(self, case):
        with pytest.raises(ValueError, match="meshes"):
            SurfaceModel.build(case.landmarks, None, "tangent")


class TestCovariance:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_a_covariance_is_read_in_the_bone_frame(self, side):
        knee = measured(side)
        target = "femur.notch_centre"
        lateral_only = Dispersion({target: np.diag([0.0, 4.0, 0.0])})
        rng = np.random.default_rng(4)
        frames = {"femur": knee.femoral_frame, "tibia": knee.tibial_frame}
        nominal = knee.landmarks.position(target)
        for _ in range(10):
            moved = perturb(knee.landmarks, lateral_only, rng, frames=frames,
                            surface=SurfaceModel("free")).position(target) - nominal
            along = float(moved @ knee.femoral_frame.lateral)
            assert np.linalg.norm(moved - along * knee.femoral_frame.lateral) < 1e-9

    def test_a_covariance_must_be_positive_semidefinite(self):
        with pytest.raises(ValueError, match="semidefinite"):
            Dispersion({"femur.notch_centre": np.diag([1.0, -1.0, 1.0])})

    def test_an_unknown_landmark_is_refused(self):
        with pytest.raises(KeyError):
            Dispersion.isotropic({"femur.nonsense": 1.0})


class TestTables:
    def test_round_trip(self, tmp_path):
        table = Dispersion(
            {"femur.notch_centre": 2.25 * np.eye(3),
             "tibia.spine_medial": np.array([[1.0, 0.2, 0.0],
                                             [0.2, 2.0, 0.1],
                                             [0.0, 0.1, 0.5]])},
            label="PLACEHOLDER for a test",
        )
        path = write_dispersion(table, tmp_path / "d.csv")
        back = read_dispersion(path)
        assert back.placeholder
        assert back.sd_mm("femur.notch_centre") == pytest.approx(1.5)
        for key, cov in table.covariances.items():
            np.testing.assert_allclose(back.covariances[key], cov)

    def test_e3_style_table_with_only_sd_column(self, tmp_path):
        path = tmp_path / "e3.csv"
        path.write_text("landmark_id,sd_mm\nfemur.notch_centre,1.2\n"
                        "tibia.spine_lateral,0\n", encoding="utf-8")
        table = read_dispersion(path)
        assert not table.placeholder
        assert set(table.covariances) == {"femur.notch_centre"}

    @pytest.mark.parametrize("name", sorted(PLACEHOLDER_TABLES))
    def test_shipped_tables_are_labelled_placeholders(self, name):
        table = placeholder_dispersion(name)
        assert table.placeholder
        assert "femur.condyle_distal_medial" in table.covariances
        # The canal centres are derived, not picked, so no observer table moves them.
        assert "femur.canal_centre_distal" not in table.covariances
