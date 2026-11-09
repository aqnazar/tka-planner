"""Build the online landmark study: a static site with every knee encrypted.

The planner runs here, not online. For each knee this opens the one app's own session,
asks it for the Pick targets exactly as the local Pick mode does (both modes: ``blank``
and ``adjust``, the latter with the automatic positions), takes the two uncut bones,
and packs them into one file encrypted under the study passphrase (AES-256-GCM, key by
PBKDF2-SHA256). The browser decrypts it; Netlify only ever holds ciphertext.

    python -m online.build_site            # defaults: Custom Implant/Surgeon Study/...

The built folder, the passphrase and the admin key live outside the repository
(``<Surgeon Study>/online_site/``). Research use only; not a medical device.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import random
import secrets
import shutil
import time
from pathlib import Path

import manifold3d as m3
import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.geom.gltf import write_glb
from tka_planner.server.sessions import SessionManager
from tka_planner.study import pick_targets, pickable_landmarks, study_schema

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO.parents[1] / "Surgeon Study"
SITE_SOURCE = Path(__file__).resolve().parent / "site"

PBKDF2_ITERATIONS = 310_000
# The web copy of each bone is simplified within this tolerance (0.05 mm moves the
# surface by under 0.2 mm on a 1-million-triangle femur: below picking precision, and
# a knee loads as about 3 MB instead of 35). Picks are written back against the full
# meshes, whose surface-gap check would catch anything worse.
WEB_TOLERANCE_MM = 0.05
GROUP_SEED = 20261109
GROUPS = ("A", "B", "C")
VSD_CREDIT = ("VSD knees: Fischer M.C.M., Sci. Data 10, 763 (2023), "
              "VSDFullBodyBoneModels, CC BY-NC-SA 4.0; cut to knee-only length.")


def derive(passphrase: str, salt: bytes) -> tuple[bytes, str]:
    """The AES key and the save token, as the browser derives them (same PBKDF2)."""
    bits = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt,
                               PBKDF2_ITERATIONS, dklen=64)
    return bits[:32], bits[32:].hex()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encrypt(key: bytes, payload: dict) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    data = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), 6)
    return nonce + AESGCM(key).encrypt(nonce, data, None)


def make_passphrase() -> str:
    # 3 x 4 characters from 30 unambiguous ones: about 59 bits, behind 310 000 PBKDF2
    # rounds. Typed once per browser.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ234679"
    return "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3))


def load_secrets(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text("utf-8"))
    found = {"passphrase": make_passphrase(), "admin_key": secrets.token_hex(24),
             "salt": secrets.token_hex(16)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=2), "utf-8")
    return found


def assign_groups(vsd_root: Path) -> dict:
    """Cadavers into groups A, B, C at random (fixed seed), both knees together."""
    cadavers = sorted(p.name for p in vsd_root.iterdir() if p.is_dir())
    random.Random(GROUP_SEED).shuffle(cadavers)
    size = -(-len(cadavers) // len(GROUPS))
    return {group: sorted(cadavers[i * size:(i + 1) * size])
            for i, group in enumerate(GROUPS)}


def web_mesh(mesh: Mesh) -> bytes:
    solid = m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                    tri_verts=np.asarray(mesh.faces, np.uint32))
    solid.merge()
    manifold = m3.Manifold(solid)
    if manifold.status() == m3.Error.NoError:
        out = manifold.simplify(WEB_TOLERANCE_MM).to_mesh()
        mesh = Mesh(vertices=np.asarray(out.vert_properties, float)[:, :3],
                    faces=np.asarray(out.tri_verts, np.int64))
    return write_glb(mesh)


def knee_bundle(manager: SessionManager, folder: Path, side: str) -> dict:
    entry = manager.open(folder, side=side)
    try:
        session = entry.session
        targets = {mode: pick_targets(entry, mode) for mode in ("blank", "adjust")}
        bones, meshes = {}, {}
        for name, bone in targets["blank"]["bones"].items():
            mesh_id = f"{session.case_id}-{side}-{name.lower()}"
            meshes[mesh_id] = base64.b64encode(
                web_mesh(session.scene.meshes[bone["mesh"]])).decode("ascii")
            bones[name] = {**bone, "mesh": mesh_id}
        for given in targets.values():
            given["bones"] = bones
        return {"case_id": session.case_id, "side": side, "targets": targets,
                "meshes": meshes}
    finally:
        manager.close(entry.session_id)


def build(*, cases: Path, vsd: Path, out: Path, store: Path, only=None,
          log=print) -> dict:
    site = out / "dist"
    secrets_path = out / "secrets.json"
    found = load_secrets(secrets_path)
    salt = bytes.fromhex(found["salt"])
    key, token = derive(found["passphrase"], salt)

    groups = assign_groups(vsd)
    (out / "groups.json").write_text(json.dumps(groups, indent=2), "utf-8")
    knees = [("study", folder, side) for folder in sorted(cases.iterdir()) if folder.is_dir()
             for side in _sides(folder)]
    group_of = {cadaver: group for group, members in groups.items() for cadaver in members}
    knees += [(group_of[folder.name], folder, side)
              for folder in sorted(vsd.iterdir()) if folder.is_dir()
              for side in ("left", "right")]
    if only:
        knees = [k for k in knees if f"{k[1].name}:{k[2]}" in only]

    # Start clean, but keep the installed function dependencies (npm install).
    if site.exists():
        for child in site.iterdir():
            if child.name in ("node_modules", "package-lock.json", ".netlify"):
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(SITE_SOURCE, site, dirs_exist_ok=True)
    public = site / "public"
    for name in ("viewer.js", "glb.js", "study.js", "style.css"):
        shutil.copy2(REPO / "web" / name, public / name)
    shutil.copytree(REPO / "web" / "vendor", public / "vendor")
    (public / "data").mkdir()

    manager = SessionManager(library=REPO / "data" / "implants", store=store)
    listed = []
    for group, folder, side in knees:
        started = time.perf_counter()
        bundle = knee_bundle(manager, folder, side)
        name = hashlib.sha256(salt + f"{bundle['case_id']}:{side}".encode()).hexdigest()[:20]
        (public / "data" / f"{name}.bin").write_bytes(encrypt(key, bundle))
        listed.append({"case_id": bundle["case_id"], "side": side, "group": group,
                       "file": f"data/{name}.bin"})
        log(f"{bundle['case_id']} {side} ({group}) in {time.perf_counter() - started:.1f} s, "
            f"{(public / 'data' / f'{name}.bin').stat().st_size / 1e6:.1f} MB")

    schema = study_schema()
    manifest = {"knees": listed, "groups": list(GROUPS), "schema": schema,
                "credit": VSD_CREDIT}
    (public / "data" / "manifest.bin").write_bytes(encrypt(key, manifest))
    (public / "config.json").write_text(json.dumps({
        "salt": found["salt"], "iterations": PBKDF2_ITERATIONS,
    }), "utf-8")
    allowed = sorted(f"{k['case_id']}:{k['side']}" for k in listed)
    landmarks = sorted(d.id for d in pickable_landmarks())
    (site / "netlify" / "lib" / "config.mjs").write_text(
        "// Written by online/build_site.py. Hashes only; no secret is in this file.\n"
        f"export const AUTH_HASH = {json.dumps(sha256_hex(token))};\n"
        f"export const ADMIN_HASH = {json.dumps(sha256_hex(found['admin_key']))};\n"
        f"export const KNEES = new Set({json.dumps(allowed)});\n"
        f"export const LANDMARKS = new Set({json.dumps(landmarks)});\n"
        f"export const MODES = new Set({json.dumps(schema['pick_modes'])});\n",
        "utf-8")
    return {"site": site, "knees": len(listed), "passphrase": found["passphrase"]}


def _sides(folder: Path) -> list:
    named = [side for side in ("left", "right")
             if (folder / f"FD1{side.capitalize()}.stl").is_file()]
    return named or ["left", "right"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=STUDY / "cases")
    parser.add_argument("--vsd", type=Path, default=STUDY / "vsd_cases")
    parser.add_argument("--out", type=Path, default=STUDY / "online_site")
    parser.add_argument("--store", type=Path, default=STUDY / "store")
    parser.add_argument("--only", nargs="*", help="case:side keys, for a trial build")
    args = parser.parse_args(argv)
    result = build(cases=args.cases, vsd=args.vsd, out=args.out, store=args.store,
                   only=set(args.only) if args.only else None)
    print(f"{result['knees']} knees built into {result['site']}")


if __name__ == "__main__":
    main()
