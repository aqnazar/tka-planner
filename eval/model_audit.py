"""Every model file's references, against the latest version of what they reference
(read only, no documents opened): Driver Tests, Author Sandbox, and each knee in Patient
Builds. A reference that is not the latest version of its file is flagged.

  python model_audit.py   -> block_audit/model_audit.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import call_fusion  # noqa: E402
from implant_pipeline import SOURCES  # noqa: E402

S = r'''
import adsk.core, adsk.fusion, json
SRC = __SRC__
def run(_c):
    app = adsk.core.Application.get()
    dt = app.data.findFileById(SRC["femoral"]).parentFolder
    work = dt.parentFolder
    folders = [("Driver Tests", dt)]
    for f in work.dataFolders:
        if f.name == "Author Sandbox": folders.append(("Author Sandbox", f))
        if f.name == "Patient Builds":
            for k in f.dataFolders: folders.append(("Patient Builds/" + k.name, k))
    out = {}
    for label, folder in folders:
        rows = []
        for df in folder.dataFiles:
            refs = []
            try:
                for r in df.childReferences:
                    latest = app.data.findFileById(r.id).latestVersionNumber
                    refs.append([r.name, r.versionNumber, latest, r.parentFolder.name])
            except Exception as ex:
                refs = [["?", str(ex)[:80]]]
            rows.append([df.name, df.latestVersionNumber, refs])
        out[label] = rows
    print(json.dumps(out))
'''


def main():
    res = json.loads(call_fusion(S.replace("__SRC__", repr(SOURCES)), read_only=True).strip().splitlines()[-1])
    (HERE / "block_audit" / "model_audit.json").write_text(json.dumps(res, indent=1))
    stale = []
    for label, rows in res.items():
        for name, v, refs in rows:
            for r in refs:
                if len(r) == 4 and r[1] != r[2]:
                    stale.append(f"{label}/{name} v{v} -> {r[0]} v{r[1]} (latest v{r[2]}, {r[3]})")
            print(f"{label:28s} {name:32s} v{v:<3d} " + "; ".join(f"{r[0]} v{r[1]}" + ("" if r[1] == r[2] else f" (latest v{r[2]})")
                                                        for r in refs if len(r) == 4))
    print("\nstale references:", len(stale))
    for s in stale:
        print("  ", s)
    return res


if __name__ == "__main__":
    main()
