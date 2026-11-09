"""The online landmark study: what the build packs, and how saves come home."""

import datetime as dt
import json
import shutil
import subprocess

import pytest

from online import build_site
from online.pull import write_saves
from tka_planner.core.landmarks import read_landmark_set
from tka_planner.server import SessionManager
from tka_planner.store import Store
from tka_planner.study import pickable_landmarks


@pytest.fixture
def manager(raw_session, tmp_path, monkeypatch):
    from tka_planner.server import sessions as sessions_module

    def fake_open(cls, folder, *, side, library=None, landmarks=None, **options):
        raw_session.build()
        return raw_session

    monkeypatch.setattr(sessions_module.PlanningSession, "open", classmethod(fake_open))
    return SessionManager(store=Store(tmp_path / "store"), study_dir=tmp_path / "study")


def _record(key, folder, position, *, complete=True, received="2026-11-09T10:11:12.345Z"):
    landmark = pickable_landmarks()[0].id
    return {"key": key, "value": {
        "received_utc": received, "case_id": folder.name, "side": "left", "source": "online",
        "body": {"observer": "Dr  Test", "round": 1, "mode": "blank", "complete": complete,
                 "started_utc": "2026-11-09T10:00:00Z", "total_seconds": 30.0,
                 "picks": [{"id": landmark, "position_mm": position, "seconds": 3.0}]},
    }}


def _on_femur(manager, folder):
    entry = manager.open(folder, side="left")
    mesh = entry.session.scene.meshes[entry.pristine["Femur"]]
    return [float(v) for v in mesh.vertices[0]]


def test_a_pulled_save_is_written_by_save_picks_under_the_surgeons_time(manager, folder, tmp_path):
    out = tmp_path / "study"
    record = _record("picks/abc/x_left_blank_r1/1", folder, _on_femur(manager, folder))

    written = write_saves([record], output=out, roots=[folder.parent], manager=manager,
                          log=lambda *_: None)

    assert written == [record["key"]]
    files = list((out / "picks").rglob("*.landmarks.json"))
    assert len(files) == 1 and "_blank_r1_20261109T101112Z" in files[0].name
    landmarks = read_landmark_set(files[0])
    assert landmarks.source["via"] == "online"
    assert landmarks.source["observer_key"] == "dr test"
    study = json.loads(next((out / "picks").rglob("*.study.json")).read_text("utf-8"))
    assert study["complete"] is True and "surface_gap_mm" in study["picks"][0]


def test_a_save_is_pulled_once(manager, folder, tmp_path):
    out = tmp_path / "study"
    record = _record("picks/abc/x_left_blank_r1/1", folder, _on_femur(manager, folder))
    write_saves([record], output=out, roots=[folder.parent], manager=manager, log=lambda *_: None)
    again = write_saves([record], output=out, roots=[folder.parent], manager=manager,
                        log=lambda *_: None)
    assert again == []
    assert len(list((out / "picks").rglob("*.landmarks.json"))) == 1


def test_a_pick_off_the_bone_is_refused_at_pull(manager, folder, tmp_path):
    record = _record("picks/abc/x_left_blank_r1/2", folder, [9999.0, 9999.0, 9999.0])
    with pytest.raises(ValueError, match="surface"):
        write_saves([record], output=tmp_path / "study", roots=[folder.parent],
                    manager=manager, log=lambda *_: None)


def test_the_dataset_knees_split_into_three_fixed_groups_without_overlap(tmp_path):
    for i in range(30):
        (tmp_path / f"VSD{i:03d}").mkdir()
    first = build_site.assign_groups(tmp_path)
    assert first == build_site.assign_groups(tmp_path)
    assert sorted(first) == ["A", "B", "C"]
    members = [c for group in first.values() for c in group]
    assert all(len(group) == 10 for group in first.values())
    assert sorted(members) == sorted(p.name for p in tmp_path.iterdir())


def test_a_passphrase_is_three_groups_of_four_unambiguous_characters():
    phrase = build_site.make_passphrase()
    parts = phrase.split("-")
    assert len(parts) == 3 and all(len(p) == 4 for p in parts)
    assert not set(phrase) & set("01OI58")


NODE_DECRYPT = r"""
const fs = require("fs");
const [salt, iterations, pass, file] = process.argv.slice(1);
(async () => {
  const s = Uint8Array.from(salt.match(/../g), (h) => parseInt(h, 16));
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveBits"]);
  const bits = new Uint8Array(await crypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt: s, iterations: Number(iterations) }, base, 512));
  const aes = await crypto.subtle.importKey("raw", bits.slice(0, 32), "AES-GCM", false, ["decrypt"]);
  const sealed = new Uint8Array(fs.readFileSync(file));
  const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: sealed.slice(0, 12) }, aes, sealed.slice(12));
  const text = await new Response(new Blob([plain]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
  const token = [...bits.slice(32)].map((b) => b.toString(16).padStart(2, "0")).join("");
  process.stdout.write(JSON.stringify({ payload: JSON.parse(text), token }));
})().catch((e) => { process.stderr.write(String(e)); process.exit(1); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_what_python_seals_the_browser_code_opens(tmp_path):
    salt = "00112233445566778899aabbccddeeff"
    key, token = build_site.derive("ABCD-EFGH-JKLM", bytes.fromhex(salt))
    payload = {"knees": [{"case_id": "P009L"}], "n": 3}
    (tmp_path / "x.bin").write_bytes(build_site.encrypt(key, payload))
    done = subprocess.run(
        ["node", "-e", NODE_DECRYPT, salt, str(build_site.PBKDF2_ITERATIONS),
         "ABCD-EFGH-JKLM", str(tmp_path / "x.bin")],
        capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    opened = json.loads(done.stdout)
    assert opened == {"payload": payload, "token": token}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_wrong_passphrase_opens_nothing(tmp_path):
    salt = "00112233445566778899aabbccddeeff"
    key, _ = build_site.derive("ABCD-EFGH-JKLM", bytes.fromhex(salt))
    (tmp_path / "x.bin").write_bytes(build_site.encrypt(key, {"a": 1}))
    done = subprocess.run(
        ["node", "-e", NODE_DECRYPT, salt, str(build_site.PBKDF2_ITERATIONS),
         "ABCD-EFGH-JKLX", str(tmp_path / "x.bin")],
        capture_output=True, text=True, timeout=60)
    assert done.returncode != 0
