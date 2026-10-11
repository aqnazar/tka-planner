"""Figure: each demo knee's tibia cut by the library block and by the knee's own tibial block
(tibial_block.py), coronal section through the plate's keel, with the built plate, in the
tray frame (Z = 0 is the plate's base). Private (patient-derived).

  python tblock_figure.py   -> block_audit/tblock_cut_sections.svg (then shoot.py)
"""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
from block_check import section  # noqa: E402
from demo_server import BUILDS, DemoApi  # noqa: E402
from eval_stage_b import LIBRARY  # noqa: E402
from fit_check import to_frame  # noqa: E402
from tka_planner.core.meshio import Mesh, read_stl  # noqa: E402
from tka_planner.server.sessions import SessionManager  # noqa: E402

VIEW = (-50.0, 50.0, -50.0, 25.0)        # x0, x1, z0, z1 (mm)
S, W, H = 4.0, 400, 300
FONT = 'font-family="IBM Plex Mono, monospace"'


def path(segs, colour, width):
    x0, x1, z0, z1 = VIEW
    keep = segs[(np.minimum(segs[:, 0], segs[:, 2]) > x0) & (np.maximum(segs[:, 0], segs[:, 2]) < x1)
                & (np.minimum(segs[:, 1], segs[:, 3]) > z0) & (np.maximum(segs[:, 1], segs[:, 3]) < z1)]
    d = " ".join(f"M{(a - x0) * S:.1f},{(z1 - b) * S:.1f}L{(c - x0) * S:.1f},{(z1 - e) * S:.1f}" for a, b, c, e in keep)
    return f'<path d="{d}" stroke="{colour}" stroke-width="{width}" fill="none"/>'


def main():
    man = SessionManager(store=None, library=str(LIBRARY)); api = DemoApi(man)
    stats = json.loads((HERE / "block_audit/tibial_commit_compare.json").read_text())
    rows = []
    for job in json.loads((BUILDS / "jobs_demo.json").read_text()):
        name = job["name"]
        plate = read_stl(BUILDS / name / "plate.stl")
        keel = plate.vertices[plate.vertices[:, 2] < -5]
        y = float(np.median(keel[:, 1]))
        entry = man.open(pathlib.Path(job["bones"][0]).parent, side=job["side"])
        T = np.asarray(api.poses(entry)["tibial"], float)
        cuts = {}
        for tag in ("library", "built"):
            if tag == "built":
                api.use_built_block(entry)
            man.commit(entry)
            m = entry.session.scene.mesh_for("Tibia")
            cuts[tag] = section(Mesh(vertices=to_frame(m, T).vertices, faces=m.faces), y)
        man.close(entry.session_id)
        ps = section(plate, y)
        panels = []
        for tag, title in (("library", "library block (scaled)"), ("built", "own block (its plate)")):
            st = stats[name][tag]
            note = f"keel hole to Z {st.get('hole_z_min')} mm; plate keel to Z {stats[name]['plate_keel_z_min']} mm"
            panels.append([f'<rect width="{W}" height="{H}" fill="#fbfaf7" stroke="#ccc"/>',
                           f'<line x1="0" x2="{W}" y1="{VIEW[3] * S}" y2="{VIEW[3] * S}" stroke="#bbb" stroke-dasharray="4 4"/>',
                           path(cuts[tag], "#6b625a", 1.6), path(ps, "#2f61d8", 1.6),
                           f'<text x="10" y="20" font-size="14" {FONT} font-weight="600">{name}  {title}</text>',
                           f'<text x="10" y="{H - 10}" font-size="10" {FONT} fill="#a33">{note}</text>'])
        rows.append(panels)
        print(name, "Y", round(y, 2), flush=True)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * W + 30}" height="{len(rows) * (H + 20) + 64}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="10" y="24" font-size="15" {FONT}>Tibia cut by the cutting block, coronal section through the keel (tray frame)</text>',
           f'<text x="10" y="40" font-size="13" {FONT}>grey: tibia after Commit, blue: the knee\'s built plate, dashed: the cut (Z = 0)</text>']
    for i, panels in enumerate(rows):
        for j, p in enumerate(panels):
            out.append(f'<g transform="translate({10 + j * (W + 10)},{54 + i * (H + 20)})">' + "".join(p) + "</g>")
    out.append("</svg>")
    dst = HERE / "block_audit/tblock_cut_sections.svg"
    dst.write_text("\n".join(out), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
