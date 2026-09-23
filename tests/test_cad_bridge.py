"""The specification-to-CAD mapping, which the Fusion script and the preview share."""

import json
from pathlib import Path

import pytest

from tka_planner.cad_bridge import load_mapping, lookup, resolve_parameters
from tka_planner.cli import main

EXAMPLE_MAP = Path(__file__).parent.parent / "fusion" / "parameter_map.example.json"

SPEC = {
    "schema": "tka-planner/implant-spec",
    "femoral_component": {"dimensions_mm": {"ml": 76.5, "notch_width": 18.25}},
    "tibial_component": {"dimensions_mm": {"ml": 81.25, "tray_thickness": None}},
    "insert": {"thickness_mm": 5.679},
}


def mapping(*entries):
    return {"units": "mm", "parameters": list(entries)}


def test_a_value_is_found_by_its_dotted_path():
    assert lookup(SPEC, "femoral_component.dimensions_mm.ml") == 76.5
    with pytest.raises(KeyError):
        lookup(SPEC, "femoral_component.dimensions_mm.nope")


def test_scale_and_offset_are_applied():
    [item] = resolve_parameters(SPEC, mapping(
        {"fusion": "Tray_ML", "from": "tibial_component.dimensions_mm.ml",
         "offset_mm": -1.0, "scale": 1.0}))

    assert item.value_mm == pytest.approx(80.25)
    assert item.expression == "80.250 mm"


def test_missing_and_non_numeric_values_are_reported_not_raised():
    items = resolve_parameters(SPEC, mapping(
        {"fusion": "A", "from": "femoral_component.dimensions_mm.condyle_x"},
        {"fusion": "B", "from": "tibial_component.dimensions_mm.tray_thickness"},
        {"fusion": "C", "from": "insert.thickness_mm"},
    ))

    assert [i.expression is None for i in items] == [True, True, False]
    assert items[0].problem == "not in this specification"
    assert "not a number" in items[1].problem


def test_a_mapping_without_parameters_is_refused(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"units": "mm", "parameters": []}))
    with pytest.raises(ValueError, match="non-empty"):
        load_mapping(path)


def test_the_example_mapping_names_only_real_specification_paths(session):
    """Every path in the shipped example exists in a real specification."""
    from tka_planner.pipeline import implant_spec_case

    spec = implant_spec_case(session.measurement, session.plan, session.sizing,
                             session.library)
    # The test femur is an ellipsoid, with no notch between condyles to measure; those
    # values are then absent by design, and only those may be.
    no_condyles = not spec["femoral_component"]["diagnostics"]["condyles_found"]
    for entry in load_mapping(EXAMPLE_MAP)["parameters"]:
        try:
            lookup(spec, entry["from"])
        except KeyError:
            assert no_condyles and ("condyle" in entry["from"]
                                    or "notch_width" in entry["from"]), entry["from"]


def test_the_preview_command_lists_what_would_be_set(tmp_path, capsys):
    spec_path = tmp_path / "implant_spec.json"
    spec_path.write_text(json.dumps(SPEC))
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(mapping(
        {"fusion": "Femoral_ML", "from": "femoral_component.dimensions_mm.ml"})))

    assert main(["cad-params", "--spec", str(spec_path), "--map", str(map_path)]) == 0
    assert "Femoral_ML" in capsys.readouterr().out
