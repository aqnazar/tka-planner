"""The viewer's own files, checked without a browser.

A browser is the only thing that can say whether the viewer *looks* right, and nothing
here claims otherwise. What these tests catch is the class of failure that wastes the
most time when it does happen: a file the page asks for that is not there, an import
that resolves to nothing, or a control the panel cannot build because the schema and
the session disagree. All of those are silent in a browser console and obvious here.
"""

import json
import re
import shutil
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

from tka_planner.panel import CONTROL_SCHEMA, TRIAL_SCHEMA, control_names, with_sizes
from tka_planner.server.httpd import WEB_ROOT
from tka_planner.session import ADJUSTMENT_FIELDS, Controls, TrialControls


@pytest.fixture(scope="module")
def index():
    return (WEB_ROOT / "index.html").read_text(encoding="utf-8")


def test_the_web_root_holds_the_viewer():
    for name in ("index.html", "app.js", "viewer.js", "glb.js", "style.css"):
        assert (WEB_ROOT / name).is_file(), name


def test_three_is_vendored_rather_than_fetched():
    """The planner must draw a bone on a machine with no internet connection."""
    vendor = WEB_ROOT / "vendor"
    assert (vendor / "three.module.js").is_file()
    assert (vendor / "OrbitControls.js").is_file()
    assert (vendor / "LICENSE-three.txt").is_file()
    assert "MIT" in (vendor / "LICENSE-three.txt").read_text(encoding="utf-8")


def test_nothing_the_page_loads_comes_from_the_network(index):
    for source in re.findall(r'(?:src|href)="([^"]+)"', index):
        assert not source.startswith(("http://", "https://", "//")), source


def test_every_file_the_page_asks_for_exists(index):
    for source in re.findall(r'(?:src|href)="\./([^"]+)"', index):
        assert (WEB_ROOT / source).is_file(), source


def test_the_import_map_resolves_three(index):
    body = re.search(
        r'<script type="importmap">(.*?)</script>', index, re.DOTALL
    ).group(1)
    mapping = json.loads(body)["imports"]

    assert (WEB_ROOT / mapping["three"].lstrip("./")).is_file()


def test_every_module_import_resolves():
    """A relative import that misses fails silently in the console and nowhere else."""
    for path in WEB_ROOT.glob("*.js"):
        source = path.read_text(encoding="utf-8")
        for target in re.findall(r'from "(\./[^"]+)"', source):
            assert (path.parent / target).resolve().is_file(), f"{path.name} -> {target}"


def test_orbit_controls_imports_only_the_mapped_name():
    """OrbitControls pulls from 'three'; anything else would need another vendored file."""
    source = (WEB_ROOT / "vendor" / "OrbitControls.js").read_text(encoding="utf-8")

    assert set(re.findall(r'from ["\']([^"\']+)["\']', source)) == {"three"}


def test_every_element_the_controller_reaches_for_is_in_the_page(index):
    script = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bel\("([^"]+)"\)', script))

    present = set(re.findall(r'id="([^"]+)"', index))
    assert ids <= present, sorted(ids - present)


# ----------------------------------------------------------------------
# The panel schema against the session
# ----------------------------------------------------------------------


def test_the_schema_covers_every_control_exactly():
    """One control added to the session and forgotten in the panel is invisible."""
    assert control_names() == {field.name for field in fields(Controls)}


def test_the_schema_covers_the_three_trial_controls():
    named = {control["name"] for control in TRIAL_SCHEMA}
    assert named == {field.name for field in fields(TrialControls)}


def test_all_thirteen_adjustments_appear_in_the_panel():
    assert set(ADJUSTMENT_FIELDS) <= control_names()


@pytest.mark.parametrize(
    "control",
    [c for group in CONTROL_SCHEMA for c in group["controls"]] + list(TRIAL_SCHEMA),
    ids=lambda control: control["name"],
)
def test_every_control_is_described_well_enough_to_build(control):
    assert control["kind"] in ("float", "bool", "choice")
    if control["kind"] == "float":
        assert control["min"] < control["max"]
        assert control["step"] > 0
        assert control.get("unit") in ("mm", "deg")
    if control["kind"] == "choice":
        assert control["choices"]


def test_a_float_control_defaults_inside_its_own_range():
    defaults = Controls()
    for group in CONTROL_SCHEMA:
        for control in group["controls"]:
            if control["kind"] != "float":
                continue
            value = getattr(defaults, control["name"])
            assert control["min"] <= value <= control["max"], control["name"]


def test_the_size_menu_is_filled_from_the_chart():
    filled = with_sizes(["S1", "M2", "L4"])
    sizes = next(
        control for group in filled for control in group["controls"]
        if control["name"] == "size_override"
    )

    assert sizes["choices"][0] == ["", "Solve from anatomy"]
    assert ["M2", "M2"] in sizes["choices"]
    # The unfilled schema must not have been mutated on the way through.
    assert len(CONTROL_SCHEMA[0]["controls"][1]["choices"]) == 1


def test_the_viewer_converts_millimetres_where_the_encoder_does():
    """Both halves of the unit boundary must agree, or every pose is a metre out."""
    from tka_planner.geom.gltf import MM_PER_M

    source = (WEB_ROOT / "viewer.js").read_text(encoding="utf-8")
    declared = int(re.search(r"const MM_PER_M = (\d+)", source).group(1))

    assert declared == MM_PER_M


def test_the_viewer_never_computes_a_clinical_number():
    """Every number on the screen comes from the server. This is that, as a test.

    The check is deliberately shallow -- it looks for the names of the planning
    functions, not for arithmetic -- because its job is to catch the moment somebody
    reimplements a measurement in the viewer to save a round trip.
    """
    source = "".join(
        path.read_text(encoding="utf-8") for path in WEB_ROOT.glob("*.js")
    )
    for forbidden in ("valgus_cut", "resection_depth", "hka", "mldfa", "solve_size"):
        assert forbidden not in source.lower()


# ----------------------------------------------------------------------
# The wire format, read by the viewer's own code
# ----------------------------------------------------------------------

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is not installed")

CHECKER = Path(__file__).parent / "viewer_glb_check.mjs"


def _check_glb(mesh, tmp_path):
    from tka_planner.geom.gltf import write_glb_file

    path = write_glb_file(mesh, tmp_path / "mesh.glb")
    finished = subprocess.run(
        [NODE, str(CHECKER), str(path),
         str(mesh.vertices.shape[0]), str(mesh.faces.size)],
        capture_output=True, text=True, timeout=60,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
    return json.loads(finished.stdout)


@needs_node
@pytest.mark.parametrize("build", ["box", "sphere", "odd"])
def test_the_viewer_reads_what_the_encoder_writes(tmp_path, build):
    """The two halves of the wire format, checked against each other.

    A padding rule or an accessor offset that the two disagree about shows up in a
    browser as a bone that silently fails to appear, and nowhere else.
    """
    from tka_planner.geom import mesh as gm

    mesh = {
        "box": lambda: gm.box(40.0),
        "sphere": lambda: gm.uv_sphere(18.0, segments=32, rings=24),
        # An odd vertex count is what forces the encoder to pad between the two chunks.
        "odd": lambda: gm.uv_sphere(3.0, segments=7, rings=5),
    }[build]()

    report = _check_glb(mesh, tmp_path)

    assert report["problems"] == []
    assert report["extent"] == pytest.approx(
        (mesh.extent / 1000.0).tolist(), abs=1e-6
    )


@needs_node
def test_a_committed_mesh_survives_the_round_trip(tmp_path, session):
    """The bone that matters: a real boolean result, encoded and read back."""
    session.commit()

    report = _check_glb(session.scene.mesh_for("Femur"), tmp_path)

    assert report["problems"] == []
    assert report["vertices"] > 0
