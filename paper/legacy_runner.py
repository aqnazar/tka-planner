"""Run the Paper 1 pipeline, ``legacy/TKA_Script_v3.py``, on one knee, unmodified.

    python -m paper.legacy_runner P009L                    # Blender 5.1, catalogue library
    python -m paper.legacy_runner P009L --blender 4.4      # another Blender
    python -m paper.legacy_runner P009L --plain            # no driver, outputs only

The frozen script is configured by module-level globals in its Section 1. The runner
writes a temporary copy in which only the path and side globals (``PATIENT_DIR``,
``SCALED_STLS_DIR``, ``SIDE`` and, when a size chart is given separately,
``CSV_PATH``) are replaced, checks line by line and by hash that every other line is
the frozen file's, and records both hashes. Everything else -- ``USE_PCA = "F"``,
``SIZE_MODE = "ML"``, the 90 degree import rotation, ``STEP_BY_STEP_MODE = "F"`` -- is
the frozen configuration that produced Paper 1 (``legacy/PROVENANCE.md``).

The script writes an ``Output/`` folder next to its inputs, so the knee's two bones are
copied into a fresh scratch folder first; nothing is ever written beside the cohort.

Paper 1 was run on Blender 5.1 (the lead author, 2026-10-14; the archived ``.blend``
files carry format version 501), which is the default here.

Its results come back in two frames. The **legacy world** is the script's own: metres,
X anterior, Y patient-left, Z proximal after the fixed 90 degree turn, the femoral
condyles on Z = 0. The **STL frame** is the input mesh's own (LPS, millimetres), which
is the frame the new planner works in. ``legacy_driver.py`` records, for every object,
the matrix from its file to the legacy world; a pose in the bone's STL frame is then
``inv(F_bone) @ F_component``. The transform is checked by mapping the legacy's
resected bone back and finding it on the input mesh vertex for vertex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .harness import REPO, Case, cohort_cases, cohort_root, library_root, out_root

FROZEN = REPO / "legacy" / "TKA_Script_v3.py"
FROZEN_SHA256 = "8e0bb5feb62e22657b4743f4b4f208b7e6f7d74e66b20ccca76d5a03ec624fee"
DRIVER = Path(__file__).resolve().parent / "legacy_driver.py"

BLENDER_DEFAULT = "5.1"
BLENDERS = {
    "4.4": r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    "5.1": r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
}

# The only globals the runner may set. Everything else in the script is frozen.
PATH_GLOBALS = ("PATIENT_DIR", "SCALED_STLS_DIR", "CSV_PATH", "SIDE")

# The frozen run configuration the runner relies on and refuses to run without.
EXPECTED_CONFIG = {
    "STEP_BY_STEP_MODE": '"F"', "BATCH_MODE": '"F"', "COMPARE_SIZE": '"F"',
    "USE_PCA": '"F"', "USE_TEA": '"F"', "SIZE_MODE": '"ML"', "SIZE_ROUND": '"D"',
    "MANUAL_SIZE": '""', "BONE_IMPORT_MODE": '"R"', "BONE_MANUAL_Z_ROT_DEG": "90.0",
}

# What the script reads from a size folder (its _stl_names), side-specific implant last.
LIBRARY_PARTS = (
    "CuttingBlockFemoral(L&R)_{size}.stl",
    "CuttingBlockFemoralShell(L&R)_{size}.stl",
    "CuttingBlockTibia(L&R)_{size}.stl",
    "CuttingBlockTibiaShell(L&R)_{size}.stl",
    "Tibial_Insert(L&R)_{size}.stl",
    "Tibial_Plate(L&R)_{size}.stl",
    "Implant(Femoral){Side}_{size}.stl",
)
CHART_COLUMNS = ("SIZE", "femur_ML_mm", "femur_AP_mm", "tibia_ML_mm", "tibia_AP_mm",
                 "patella_d_mm", "insert_thin_mm", "insert_thick_mm",
                 "femur_distal_cut_m", "femur_origin_z_m", "tibia_proximal_cut_m",
                 "tibia_origin_z_m")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------
# Configuring the frozen script
# ---------------------------------------------------------------------------------------

def _global_line(name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(name)}\s*=")


def configure_script(source: str, values: dict[str, str]) -> tuple[str, dict[int, str]]:
    """The script with only the named path/side globals replaced.

    Returns the new text and ``{line_number: global}`` for the lines replaced. Each
    global must be assigned on exactly one top-level line; anything else is refused
    rather than guessed at.
    """
    unknown = set(values) - set(PATH_GLOBALS)
    if unknown:
        raise ValueError(f"Only {', '.join(PATH_GLOBALS)} may be set, not "
                         f"{', '.join(sorted(unknown))}.")
    lines = source.split("\n")
    replaced: dict[int, str] = {}
    for name, value in values.items():
        hits = [i for i, line in enumerate(lines) if _global_line(name).match(line)]
        if len(hits) != 1:
            raise ValueError(f"{name} is assigned on {len(hits)} top-level lines, "
                             f"expected exactly one.")
        lines[hits[0]] = f"{name} = {value!r}  # set by paper.legacy_runner"
        replaced[hits[0] + 1] = name
    return "\n".join(lines), replaced


def verify_configured(frozen: str, configured: str, replaced: dict[int, str]) -> dict:
    """Prove the configured copy differs from the frozen script only where it should.

    Every line but the replaced ones must be byte-identical, so the hash of "every other
    line" is the same for both files; that hash, and both whole-file hashes, go into the
    run record.
    """
    a, b = frozen.split("\n"), configured.split("\n")
    if len(a) != len(b):
        raise ValueError("The configured script has a different number of lines.")
    differing = {i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y}
    if not differing <= set(replaced):
        raise ValueError(f"Lines {sorted(differing - set(replaced))} differ from the "
                         f"frozen script but were not meant to change.")
    for number, name in replaced.items():
        if not _global_line(name).match(b[number - 1]):
            raise ValueError(f"Line {number} should assign {name}.")
    keep = [i for i in range(len(a)) if i + 1 not in replaced]
    rest_a = sha256_bytes("\n".join(a[i] for i in keep).encode("utf-8"))
    rest_b = sha256_bytes("\n".join(b[i] for i in keep).encode("utf-8"))
    if rest_a != rest_b:
        raise ValueError("The unchanged lines do not hash identically.")
    return {
        "frozen_sha256": sha256_bytes(frozen.encode("utf-8")),
        "configured_sha256": sha256_bytes(configured.encode("utf-8")),
        "unchanged_lines_sha256": rest_a,
        "replaced_lines": {str(k): v for k, v in sorted(replaced.items())},
        "n_lines": len(a),
    }


def frozen_config(source: str) -> dict[str, str]:
    """The frozen values of the run-configuration globals, as written in the source."""
    found = {}
    for name in EXPECTED_CONFIG:
        for line in source.split("\n"):
            if _global_line(name).match(line):
                found[name] = line.split("=", 1)[1].split("#", 1)[0].strip()
                break
    return found


# ---------------------------------------------------------------------------------------
# The implant library the script reads
# ---------------------------------------------------------------------------------------

def check_library(stl_dir: Path, chart_path: Path, side: str) -> dict:
    """What the script would fail to find, for every size in the chart.

    The legacy script reads fixed file names (``_stl_names``) and twelve columns from
    the chart. A missing file is only fatal for the size it picks, so the problems are
    listed per size rather than raised.
    """
    import csv

    side_cap = "Left" if side.lower().startswith("l") else "Right"
    report = {"stl_dir": str(stl_dir), "size_chart": str(chart_path),
              "missing_columns": [], "missing_files": {}}
    if not chart_path.is_file():
        report["missing_columns"] = ["<no size chart>"]
        return report
    with open(chart_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        report["missing_columns"] = [c for c in CHART_COLUMNS
                                     if c not in (reader.fieldnames or [])]
        sizes = [row["SIZE"].strip() for row in reader] if "SIZE" in (
            reader.fieldnames or []) else []
    report["sizes"] = sizes
    for size in sizes:
        missing = [p.format(size=size, Side=side_cap) for p in LIBRARY_PARTS
                   if not (stl_dir / size / p.format(size=size, Side=side_cap)).is_file()]
        if missing:
            report["missing_files"][size] = missing
    report["chart_sha256"] = sha256_bytes(chart_path.read_bytes())
    return report


# ---------------------------------------------------------------------------------------
# Reading the script's log
# ---------------------------------------------------------------------------------------

_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_log(text: str) -> dict:
    """Everything the legacy log says about a run, as numbers and flags.

    Also tells the frozen v3 script apart from the later variant whose logs are in the
    same archive folders: the frozen script's TEA report says "Endpoint A/B", the later
    one "Lateral/Medial", and only the frozen configuration prints "PCA disabled".
    """
    def first(pattern, cast=float, group=1):
        m = re.search(pattern, text)
        return cast(m.group(group)) if m else None

    out = {
        "started": first(r"TKA AUTO PIPELINE v3 — ([\d\- :]+)", str),
        "side": first(r"\nSide: (\w+)", str),
        "pca": ("off" if "PCA disabled" in text
                else "on" if "via PCA" in text or "PCA shaft correction" in text else None),
        "tea": ("on" if ("TEA detected" in text or "Pre-detecting TEA" in text)
                else "off" if "TEA disabled" in text else None),
        "tea_report_style": ("endpoint" if "Endpoint A:" in text
                             else "lateral_medial" if "Lateral:" in text else None),
        "femur_ml_mm": first(rf"Femur distal region — ML: {_NUM}mm"),
        "femur_ap_mm": first(rf"Femur distal region — ML: {_NUM}mm  AP: {_NUM}mm", group=2),
        "femur_height_mm": first(rf"Femur full bbox\s+— Height: {_NUM}mm"),
        "size_mode": first(r"FEMUR (ML|AP) = ", str),
        "size": first(r"FEMUR (?:ML|AP) = [\d.]+mm → Size (\w+)", str),
        "size_override": first(r"\[(?:MANUAL|OVERRIDE)\] Size(?: override)?: (\w+)", str),
        "cross_check_size": first(r"\[WARNING\] Femur (?:AP|ML) cross-check → (\w+)", str),
        "out_of_range": "out of range. Using" in text,
        "post_valgus_reanchor_mm": first(rf"Post-valgus re-anchor: {_NUM}mm"),
        "femoral_anchor_mm": None,
        "tibia_ml_mm": first(rf"Tibia proximal region — ML: {_NUM}mm"),
        "tibia_ap_mm": first(rf"Tibia proximal region — ML: {_NUM}mm  AP: {_NUM}mm", group=2),
        "tibia_size": first(r"\[SIZE MISMATCH\] Tibia → (\w+)", str),
        "tibial_anchor_mm": None,
        "bone_above_cut_mm": first(rf"Bone above cut: {_NUM}mm"),
        "boolean_exact_failed": text.count("EXACT boolean failed"),
        "boolean_failed": text.count("FAST boolean also failed") + text.count(
            "[ERROR] Boolean failed"),
        "upside_down": text.count("UPSIDE-DOWN"),
        "no_cut_needed": text.count("No cut needed."),
        "femur_done": "[FEMUR DONE]" in text,
        "tibia_done": "[TIBIA DONE]" in text,
        "complete": "PIPELINE COMPLETE" in text,
    }
    if out["size"] is None and out["size_override"]:
        out["size"] = out["size_override"]
    if out["tibia_size"] is None and "Tibia size matches femur" in text:
        out["tibia_size"] = out["size"]
    m = re.search(rf"Component anchor XY: \({_NUM}, {_NUM}\) mm", text)
    if m:
        out["femoral_anchor_mm"] = [float(m.group(1)), float(m.group(2))]
    m = re.search(rf"Tibial anchor XY: \({_NUM}, {_NUM}\) mm", text)
    if m:
        out["tibial_anchor_mm"] = [float(m.group(1)), float(m.group(2))]
    out["frozen_v3_signature"] = (out["pca"] == "off" and out["tea"] == "off"
                                  and out["tea_report_style"] is None)
    return out


# ---------------------------------------------------------------------------------------
# From the legacy world back to the STL frame
# ---------------------------------------------------------------------------------------

BONE_OBJECTS = {"femur": "Femur", "tibia": "Tibia"}
COMPONENT_OBJECTS = {
    "femoral_component": ("femur", "Implant_Femoral_"),
    "femoral_cutting_block": ("femur", "CuttingBlock_Femoral_"),
    "tibial_component": ("tibia", "Tibial_Plate_"),
    "tibial_insert": ("tibia", "Tibial_Insert_"),
    "tibial_cutting_block": ("tibia", "CuttingBlock_Tibia_"),
}


def _object(record: dict, *, exact: str | None = None, prefix: str | None = None):
    for obj in record.get("objects", []):
        if (exact is not None and obj["name"] == exact) or (
                prefix is not None and obj["name"].startswith(prefix)):
            return obj
    return None


def rigid_check(matrix: np.ndarray, scale: float = 1.0, tol: float = 1e-5) -> bool:
    """Whether the 3x3 part of ``matrix`` is a rotation times ``scale``."""
    r = np.asarray(matrix)[:3, :3] / scale
    return bool(np.allclose(r.T @ r, np.eye(3), atol=tol)
                and np.isclose(np.linalg.det(r), 1.0, atol=tol))


def poses_in_stl_frame(record: dict) -> dict:
    """Every legacy component's pose in its bone's STL frame (mm), and in the world.

    ``file_to_world`` maps a file's millimetres to legacy metres, so the 0.001 import
    scale cancels in ``inv(F_bone) @ F_component`` and the pose is a rigid motion from
    the component's CAD frame to the bone's STL frame. Its third column is the cut
    normal and its translation the CAD origin, which the library puts on the cut.
    """
    bones = {}
    for bone, name in BONE_OBJECTS.items():
        obj = _object(record, exact=name)
        if obj is not None:
            bones[bone] = np.array(obj["file_to_world"])
    out = {"bones": {k: v.tolist() for k, v in bones.items()}, "components": {},
           "world_mm": {}}
    for key, (bone, prefix) in COMPONENT_OBJECTS.items():
        obj = _object(record, prefix=prefix)
        if obj is None or bone not in bones:
            continue
        f = np.array(obj["file_to_world"])
        pose = np.linalg.inv(bones[bone]) @ f
        out["components"][key] = pose.tolist()
        world = f.copy()
        world[:3, :3] /= 0.001
        world[:3, 3] *= 1000.0
        out["world_mm"][key] = world.tolist()
        out.setdefault("rigid", {})[key] = rigid_check(pose)
    return out


def cut_face_offset(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray, *,
                    window_mm: float = 3.0, parallel_tol: float = 1e-4) -> dict:
    """Where the resected bone's flat cut face sits relative to a component's XY plane.

    The library's rule is that a part's CAD origin lies on its cut, so the face the
    cutting block leaves should sit at height zero. It is measured rather than assumed:
    the faces parallel to the plane and within ``window_mm`` of it are taken, and their
    area-weighted median height is the offset (positive: proximal of the origin).
    """
    pose = np.asarray(pose, dtype=float)
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    keep = area > 1e-12
    unit = cross[keep] / area[keep, None]
    height = (tri[keep].mean(axis=1) - pose[:3, 3]) @ pose[:3, 2]
    flat = (np.abs(np.abs(unit @ pose[:3, 2]) - 1.0) < parallel_tol) & (
        np.abs(height) <= window_mm)
    if not flat.any():
        return {"cut_offset_mm": None, "cut_faces": 0, "cut_area_mm2": 0.0}
    h, w = height[flat], area[keep][flat] / 2
    order = np.argsort(h)
    cumulative = np.cumsum(w[order])
    median = float(h[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    return {"cut_offset_mm": median, "cut_faces": int(flat.sum()),
            "cut_area_mm2": float(w.sum())}


def back_map_check(output_stl: Path, input_stl: Path, file_to_world: np.ndarray, *,
                   tol_mm: float = 0.01, plane: np.ndarray | None = None) -> dict:
    """Map the legacy's resected bone back to the STL frame and find it on the input.

    The vertices the cut did not touch must land on input vertices, to the precision of
    float32 coordinates in metres (a few micrometres); the ones the booleans made lie
    on the cut faces, inside the bone. The fraction within ``tol_mm`` and the spread of
    those distances say whether the transform is right.

    ``plane`` (the component pose in the STL frame) also measures where the flat cut
    face lies relative to it (:func:`cut_face_offset`).
    """
    from tka_planner.core.meshio import read_stl

    from .meshdist import nearest_vertex_distance

    out_mesh = read_stl(output_stl, compute_hash=False)
    in_mesh = read_stl(input_stl, compute_hash=False)
    inverse = np.linalg.inv(file_to_world)
    back = out_mesh.vertices @ inverse[:3, :3].T + inverse[:3, 3]
    d = nearest_vertex_distance(back, in_mesh.vertices, cell=0.05)
    matched = d <= tol_mm
    extra = cut_face_offset(back, out_mesh.faces, plane) if plane is not None else {}
    return {
        **extra,
        "n_vertices": int(len(back)),
        "fraction_on_input_vertex": float(matched.mean()),
        "median_matched_um": float(np.median(d[matched]) * 1000) if matched.any() else None,
        "max_matched_um": float(d[matched].max() * 1000) if matched.any() else None,
    }


# ---------------------------------------------------------------------------------------
# Running one knee
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Library:
    """The implant parts the script reads, and the chart it sizes from."""

    stl_dir: Path
    size_chart: Path | None = None   # None: the script's own SCALED_STLS_DIR/SizeChart.csv

    @property
    def chart(self) -> Path:
        return self.size_chart or self.stl_dir / "SizeChart.csv"


def default_library() -> Library:
    """``TKA_LEGACY_LIBRARY`` if set (with ``TKA_LEGACY_SIZE_CHART``), else the catalogue."""
    stl = os.environ.get("TKA_LEGACY_LIBRARY")
    if stl:
        chart = os.environ.get("TKA_LEGACY_SIZE_CHART")
        return Library(Path(stl), Path(chart) if chart else None)
    root = library_root() or REPO / "data" / "implants"
    return Library(root)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run_knee(case: Case, *, out_dir: Path, blender: str = BLENDER_DEFAULT,
             blender_exe: str | None = None, library: Library | None = None,
             timeout: float = 1800.0, plain: bool = False,
             keep_scene: bool = False) -> dict:
    """Run the frozen script on one knee in headless Blender and collect its results.

    Writes ``<out_dir>/<case>/legacy_result.json`` and returns the same dict. A failed
    or timed-out run is returned with ``ok`` false and the reason, not raised.
    """
    library = library or default_library()
    exe = blender_exe or BLENDERS[blender]
    work = Path(out_dir) / case.case_id
    if _inside(work, cohort_root()):
        raise ValueError("The legacy run folder must not be inside the cohort folder.")
    if work.exists():
        shutil.rmtree(work)
    side_cap = "Left" if case.side == "left" else "Right"
    knee_dir = work / "knee"
    input_dir = knee_dir / side_cap
    input_dir.mkdir(parents=True)
    shutil.copyfile(case.femur, input_dir / f"FD1{side_cap}.stl")
    shutil.copyfile(case.tibia, input_dir / f"TD1{side_cap}.stl")

    frozen = FROZEN.read_text(encoding="utf-8")
    if sha256_bytes(FROZEN.read_bytes()) != FROZEN_SHA256:
        raise RuntimeError("legacy/TKA_Script_v3.py no longer matches its recorded hash.")
    config = frozen_config(frozen)
    if config != EXPECTED_CONFIG:
        raise RuntimeError(f"Unexpected frozen configuration: {config}")
    values = {"PATIENT_DIR": str(knee_dir), "SCALED_STLS_DIR": str(library.stl_dir),
              "SIDE": "L" if case.side == "left" else "R"}
    if library.size_chart is not None:
        values["CSV_PATH"] = str(library.size_chart)
    configured, replaced = configure_script(frozen, values)
    script_record = verify_configured(frozen, configured, replaced)
    script_path = work / "TKA_Script_v3_configured.py"
    script_path.write_text(configured, encoding="utf-8", newline="\n")

    record_path = work / "driver_record.json"
    command = [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python"]
    command += [str(script_path)] if plain else [str(DRIVER), "--", str(script_path),
                                                 str(record_path)]
    result = {
        "case": case.case_id, "side": case.side, "ok": False, "plain": plain,
        "blender_requested": blender, "blender_exe": exe,
        "library": check_library(library.stl_dir, library.chart, case.side),
        "script": script_record, "frozen_config": config,
        "inputs": {"femur_sha256": sha256_bytes(case.femur.read_bytes()),
                   "tibia_sha256": sha256_bytes(case.tibia.read_bytes())},
    }
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
        result["returncode"] = proc.returncode
        (work / "blender_stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n"
                                                 + proc.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout after {timeout:.0f} s"
    result["seconds"] = round(time.perf_counter() - t0, 1)

    output = input_dir / "Output"
    logs = sorted(output.glob("tka_log_*.txt")) if output.is_dir() else []
    if logs:
        result["log_file"] = logs[-1].name
        result["log"] = parse_log(logs[-1].read_text(encoding="utf-8", errors="replace"))
    result["outputs"] = sorted(p.name for p in output.glob("*.stl")) if output.is_dir() else []

    if not plain and record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        result["blender_version"] = record.get("blender_version")
        result["driver_ok"] = record.get("ok")
        if record.get("error"):
            result["error"] = record["error"]
        result["calls"] = record.get("calls")
        result["slope_fit_residual_m"] = max(
            (e["fit_residual_m"] for e in record.get("events", [])), default=0.0)
        poses = poses_in_stl_frame(record)
        result["poses"] = poses
        result["back_map"] = back_map(result, case, output)
    elif plain:
        m = re.search(r"Blender (\d+\.\d+\.\d+)",
                      (work / "blender_stdout.txt").read_text(encoding="utf-8")
                      if (work / "blender_stdout.txt").is_file() else "")
        result["blender_version"] = m.group(1) if m else None

    log = result.get("log") or {}
    result["ok"] = bool(log.get("complete")) and not result.get("error")
    result["output_sha256"] = {p.name: sha256_bytes(p.read_bytes())
                               for p in sorted(output.glob("*.stl"))} if output.is_dir() else {}

    # The copied bones and the saved scene are large and add nothing once collected.
    for name in (f"FD1{side_cap}.stl", f"TD1{side_cap}.stl"):
        (input_dir / name).unlink(missing_ok=True)
    if not keep_scene and output.is_dir():
        for blend in output.glob("*.blend*"):
            blend.unlink()
    (work / "legacy_result.json").write_text(json.dumps(result, indent=1, default=str),
                                             encoding="utf-8")
    return result


def back_map(result: dict, case: Case, output: Path) -> dict:
    """:func:`back_map_check` for both resected bones of a run."""
    poses = result.get("poses") or {}
    size = (result.get("log") or {}).get("size")
    side_cap = "Left" if case.side == "left" else "Right"
    checks = {}
    for bone, stl, key in (("femur", case.femur, "femoral_component"),
                           ("tibia", case.tibia, "tibial_component")):
        produced = output / f"{bone.capitalize()}_{side_cap}_{size}_result.stl"
        if bone in poses.get("bones", {}) and produced.is_file():
            checks[bone] = back_map_check(produced, stl, np.array(poses["bones"][bone]),
                                          plane=poses["components"].get(key))
    return checks


def output_dir(result: dict, out_dir: Path) -> Path:
    """Where a run's ``Output/`` folder is."""
    side_cap = "Left" if result["side"] == "left" else "Right"
    return Path(out_dir) / result["case"] / "knee" / side_cap / "Output"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper.legacy_runner",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="+", help="case ids, e.g. P009L, or 'all'")
    parser.add_argument("--blender", default=BLENDER_DEFAULT, choices=sorted(BLENDERS))
    parser.add_argument("--blender-exe", help="a Blender executable, overriding --blender")
    parser.add_argument("--library", help="folder of size folders (default: "
                        "TKA_LEGACY_LIBRARY or data/implants)")
    parser.add_argument("--size-chart", help="SizeChart.csv, if not in --library")
    parser.add_argument("--out", help="output folder (default: <paper out>/legacy_runs)")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--plain", action="store_true",
                        help="run the configured script directly, without the driver")
    parser.add_argument("--keep-scene", action="store_true",
                        help="keep the .blend the script saves")
    args = parser.parse_args(argv)
    library = (Library(Path(args.library), Path(args.size_chart) if args.size_chart else None)
               if args.library else default_library())
    cases = cohort_cases(include_reference=True)
    wanted = [c for c in cases if args.cases == ["all"] or c.case_id in args.cases]
    out = Path(args.out) if args.out else out_root() / "legacy_runs"
    for case in wanted:
        r = run_knee(case, out_dir=out, blender=args.blender, blender_exe=args.blender_exe,
                     library=library, timeout=args.timeout, plain=args.plain,
                     keep_scene=args.keep_scene)
        log = r.get("log") or {}
        print(f"{case.case_id}: ok={r['ok']} size={log.get('size')} "
              f"blender={r.get('blender_version')} {r['seconds']} s "
              f"{r.get('error', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
