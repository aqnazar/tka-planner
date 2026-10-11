"""Figure: each demo knee's femur cut by the library block (scaled by ML) and by the knee's
own block (femoral_block.py), sagittal section through a condyle, with the built femoral
component. Private (patient-derived).

  python block_figure.py   -> block_audit/block_cut_sections.svg (then shoot.py)
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

Y = 20.0                                 # sagittal plane through a condyle (mm, CAD Y)
VIEW = (-45.0, 55.0, -15.0, 80.0)        # x0, x1, z0, z1
S, W, H = 4.0, 400, 380                  # px per mm, panel size


def path(segs, x0, z1, colour, width, dash=""):
    d = " ".join(f"M{(a - x0) * S:.1f},{(z1 - b) * S:.1f}L{(c - x0) * S:.1f},{(z1 - e) * S:.1f}"
                 for a, b, c, e in segs)
    return f'<path d="{d}" stroke="{colour}" stroke-width="{width}" fill="none" {dash}/>'


def panel(title, femur_segs, comp_segs, note):
    x0, x1, z0, z1 = VIEW
    keep = lambda s: s[(np.minimum(s[:, 0], s[:, 2]) > x0) & (np.maximum(s[:, 0], s[:, 2]) < x1)
                       & (np.minimum(s[:, 1], s[:, 3]) > z0) & (np.maximum(s[:, 1], s[:, 3]) < z1)]
    body = [f'<rect width="{W}" height="{H}" fill="#fbfaf7" stroke="#ccc"/>',
            path(keep(femur_segs), x0, z1, "#6b625a", 1.6),
            path(keep(comp_segs), x0, z1, "#2f61d8", 1.6),
            f'<text x="10" y="20" font-size="14" font-family="IBM Plex Mono, monospace" font-weight="600">{title}</text>',
            f'<text x="10" y="{H - 12}" font-size="10" font-family="IBM Plex Mono, monospace" fill="#a33">{note}</text>']
    return body


def main():
    man = SessionManager(store=None, library=str(LIBRARY)); api = DemoApi(man)
    offsets = json.loads((HERE / "block_audit/commit_compare.json").read_text())
    rows = []
    for job in json.loads((BUILDS / "jobs_demo.json").read_text()):
        name = job["name"]
        entry = man.open(pathlib.Path(job["bones"][0]).parent, side=job["side"])
        F = np.asarray(api.poses(entry)["femoral"], float)
        comp = section(read_stl(BUILDS / name / "femoral.stl"), Y)
        cuts = {}
        for tag in ("library", "built"):
            if tag == "built":
                api.use_built_block(entry)
            man.commit(entry)
            m = entry.session.scene.mesh_for("Femur")
            cuts[tag] = section(Mesh(vertices=to_frame(m, F).vertices, faces=m.faces), Y)
        man.close(entry.session_id)
        off = offsets[name]
        def note(tag):
            f = off[tag]["face_median_offset_mm"]
            return "median cut-to-J-curve: ant %+.1f  ant-ch %+.1f  dist %+.1f  post %+.1f mm" % (
                f["anterior"], f["anterior chamfer"], f["distal"], f["posterior"])
        rows.append([panel(f"{name}  library block (x ML)", cuts["library"], comp, note("library")),
                     panel(f"{name}  own block (J-curve)", cuts["built"], comp, note("built"))])
        print(name, flush=True)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * W + 30}" height="{len(rows) * (H + 20) + 64}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="10" y="24" font-size="15" font-family="IBM Plex Mono, monospace">Femur cut by the cutting block, sagittal section at Y = {Y:+.0f} mm (femoral CAD frame)</text>',
           '<text x="10" y="40" font-size="13" font-family="IBM Plex Mono, monospace">grey: femur after Commit, blue: the knee\'s built femoral component</text>']
    for i, (a, b) in enumerate(rows):
        for j, p in enumerate((a, b)):
            out.append(f'<g transform="translate({10 + j * (W + 10)},{54 + i * (H + 20)})">' + "".join(p) + "</g>")
    out.append("</svg>")
    dst = HERE / "block_audit/block_cut_sections.svg"
    dst.write_text("\n".join(out), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
