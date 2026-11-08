"""The implant rules in the package: the Fusion parameters of the three parts from a plan.

The rules lived in ``eval/`` (``demo_cases.derive`` and the scripts it called) until the
one app needed them in the planner. They moved without changing a number: the fixtures
in ``tests/fixtures/implant_rules/`` were written by the eval derivation before the move,
on the two public demo knees, and every parameter here must equal them to 1e-6.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "implant_rules"
BONES = REPO / "eval" / "demo_cases"
LIBRARY = REPO / "data" / "implants"
KNEES = ("DEMO_SMALL", "DEMO_LARGE")

pytestmark = pytest.mark.skipif(
    not (LIBRARY / "femoral_box.json").exists() or not (BONES / "DEMO_LARGE" / "FD.stl").exists(),
    reason="needs the public demo bones and the catalogue in data/implants")


@pytest.fixture(scope="module")
def planned():
    """Each demo knee measured and planned once, as the eval derivation planned it."""
    from tka_planner.pipeline import measure_case, plan_case

    out = {}
    for name in KNEES:
        m = measure_case(BONES / name / "FD.stl", BONES / name / "TD.stl", "right", case_id=name)
        p, sizing = plan_case(m, library=LIBRARY)
        out[name] = (m, p, sizing)
    return out


def _numbers_equal(a, b, path=""):
    """Every leaf equal: numbers to 1e-6, strings and the rest exactly. A string holding
    a length ("81.7500 mm") compares as text: the Fusion expression is the parameter."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} vs {sorted(b)}"
        for key in a:
            _numbers_equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            _numbers_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        assert a == b, f"{path}: {a!r} vs {b!r}"
    else:
        assert abs(float(a) - float(b)) <= 1e-6, f"{path}: {a} vs {b}"


@pytest.mark.parametrize("name", KNEES)
def test_every_parameter_matches_the_eval_derivation(planned, name):
    from tka_planner.implant import derive_parameters

    m, p, sizing = planned[name]
    job = derive_parameters(m, p, sizing, LIBRARY)
    fixture = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["job"]
    # Round-trip through JSON, as the fixture was written, so tuples and numpy floats
    # compare as the plain values a job file holds.
    _numbers_equal(fixture, json.loads(json.dumps(job)))


def test_the_job_carries_the_parts_and_the_rules_behind_them(planned):
    from tka_planner.implant import derive_parameters

    m, p, sizing = planned["DEMO_LARGE"]
    job = derive_parameters(m, p, sizing, LIBRARY)
    for key in ("femoral", "plate", "insert", "femoral_fit", "tray_rule", "anatomy", "plan"):
        assert key in job
    assert job["name"] == "DEMO_LARGE" and job["side"] == "right"


def test_eval_derive_delegates_to_the_package(planned):
    sys.path.insert(0, str(REPO / "eval"))
    try:
        import demo_cases
        from tka_planner.implant import derive_parameters

        m, p, sizing = planned["DEMO_SMALL"]
        a = demo_cases.derive("DEMO_SMALL", "right", m, p, sizing, LIBRARY)
        b = derive_parameters(m, p, sizing, LIBRARY, name="DEMO_SMALL")
        _numbers_equal(json.loads(json.dumps(a)), json.loads(json.dumps(b)))
        source = Path(demo_cases.__file__).read_text(encoding="utf-8")
        assert "derive_parameters" in source
    finally:
        sys.path.remove(str(REPO / "eval"))


def test_the_tray_rule_keeps_the_outline_inside_the_cut():
    """The two-circle tray on a round cut: no overhang beyond the 1 mm inset."""
    from tka_planner.implant.tray import fit

    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    outline = np.c_[38 * np.cos(t), 25 * np.sin(t)]
    row = {"name": "x", "side": "right", "femur_ml_mm": 70.0, "outline_mm": outline.tolist(),
           "tibial": {"pcl_notch_width": 10.0, "pcl_notch_depth": 4.0}}
    res = fit(row)
    assert 0.5 < res["coverage"] < 1.0
    assert (res["_mask"] & ~res["_allowed"]).sum() * 0.25 ** 2 <= 0.5
    assert res["params"]["Tibial_Plateau_ML"] < 76.0

