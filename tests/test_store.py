"""The case store: meshes on disk, records in SQLite, and the commit cache."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.scene.model import mesh_id
from tka_planner.store import MeshStore, Store, plan_key


@pytest.fixture
def meshes(tmp_path):
    return MeshStore(tmp_path / "meshes")


def test_round_trip_is_exact(meshes):
    """Geometry must come back bit for bit, or the id stops meaning anything."""
    original = gm.uv_sphere(7.3, segments=24, rings=16)

    digest = meshes.put(original)
    restored = meshes.get(digest)

    assert np.array_equal(restored.vertices, original.vertices)
    assert np.array_equal(restored.faces, original.faces)
    assert mesh_id(restored) == digest


def test_storing_the_same_mesh_twice_writes_one_file(meshes):
    mesh = gm.box(12.0)

    first = meshes.put(mesh)
    written = meshes.path_for(first).stat().st_mtime_ns
    second = meshes.put(mesh)

    assert first == second
    assert meshes.path_for(first).stat().st_mtime_ns == written
    assert len(list(meshes.root.rglob("*.npz"))) == 1


def test_no_part_files_are_left_behind(meshes):
    meshes.put(gm.box(4.0))
    assert not list(meshes.root.rglob("*.part"))


def test_a_missing_mesh_raises(meshes):
    with pytest.raises(KeyError):
        meshes.get("0" * 16)


def test_cases_round_trip(tmp_path):
    store = Store(tmp_path / "store")
    store.save_case(
        "CASE_007", side="left", femur_path="F.stl", tibia_path="T.stl", library="lib"
    )

    assert [case["case_id"] for case in store.cases()] == ["CASE_007"]
    assert store.case("CASE_007")["side"] == "left"
    assert store.case("CASE_404") is None


def test_saving_a_case_twice_updates_rather_than_duplicates(tmp_path):
    store = Store(tmp_path / "store")
    store.save_case("CASE_001", side="left", femur_path="F.stl", tibia_path="T.stl")
    store.save_case("CASE_001", side="right", femur_path="F2.stl", tibia_path="T.stl")

    assert len(store.cases()) == 1
    assert store.case("CASE_001")["side"] == "right"


def test_the_commit_cache_survives_a_new_process(tmp_path):
    """The point of the cache: a cold Store finds what a previous one committed."""
    root = tmp_path / "store"
    mesh = gm.box(20.0)

    first = Store(root)
    digest = first.put_mesh(mesh)
    first.save_commit("plankey", {"Femur": {"mesh_id": digest, "record": {"b": "m"}}})

    cached = Store(root).cached_commit("plankey")
    assert cached["Femur"]["mesh_id"] == digest
    assert cached["Femur"]["record"] == {"b": "m"}


def test_an_uncommitted_plan_misses(tmp_path):
    assert Store(tmp_path / "store").cached_commit("never-committed") is None


def test_a_cache_entry_whose_mesh_is_gone_is_a_miss(tmp_path):
    """Deleting the mesh directory must degrade to recomputation, not to a broken hit."""
    root = tmp_path / "store"
    store = Store(root)
    digest = store.put_mesh(gm.box(6.0))
    store.save_commit("plankey", {"Femur": {"mesh_id": digest, "record": {}}})

    store.meshes.path_for(digest).unlink()

    assert store.cached_commit("plankey") is None


def test_plans_are_listed_newest_first(tmp_path):
    store = Store(tmp_path / "store")
    store.save_plan("k1", case_id="CASE_001", controls={"a": 1}, scalars={})
    store.save_plan("k2", case_id="CASE_001", controls={"a": 2}, scalars={})

    keys = {plan["plan_key"] for plan in store.plans_for("CASE_001")}
    assert keys == {"k1", "k2"}
    assert store.plans_for("CASE_002") == []


def test_owners_do_not_see_each_others_cases(tmp_path):
    """Multi-user readiness, checked rather than asserted in a docstring."""
    root = tmp_path / "store"
    Store(root, owner="ann").save_case(
        "CASE_001", side="left", femur_path="F.stl", tibia_path="T.stl"
    )

    assert Store(root, owner="bob").cases() == []
    assert len(Store(root, owner="ann").cases()) == 1


# ----------------------------------------------------------------------
# The plan key
# ----------------------------------------------------------------------


def test_the_plan_key_changes_when_a_cut_moves(session):
    before = plan_key(session)

    session.replan(tibial_resection_delta_mm=3.0)

    assert plan_key(session) != before


def test_the_plan_key_ignores_display_only_controls(session):
    """A landmark toggle must not throw away a commit that took seconds to compute."""
    before = plan_key(session)

    session.replan(show_landmarks=True, isolate_landmarks=True, show_axes=False)

    assert plan_key(session) == before


def test_the_plan_key_returns_when_a_control_returns(session):
    before = plan_key(session)

    session.replan(femoral_resection_delta_mm=2.5)
    session.replan(femoral_resection_delta_mm=0.0)

    assert plan_key(session) == before


def test_the_plan_key_notices_the_resection_mode(session):
    before = plan_key(session)

    session.replan(resection_mode="plane")

    assert plan_key(session) != before


def test_the_plan_key_needs_a_plan(raw_session):
    with pytest.raises(ValueError, match="no plan"):
        plan_key(raw_session)


def test_a_bone_key_ignores_the_other_bone(session):
    """The whole point of a per-bone key: a tibial change must not recut the femur."""
    femur_before = plan_key(session, ("Femur",))
    tibia_before = plan_key(session, ("Tibia",))

    session.replan(tibial_slope_delta_deg=2.0)

    assert plan_key(session, ("Femur",)) == femur_before
    assert plan_key(session, ("Tibia",)) != tibia_before
