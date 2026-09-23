"""From a patient's implant specification to the CAD model's parameters.

The planner measures the implant the patient needs (:func:`tka_planner.pipeline.
implant_spec_case`); the parametric CAD model builds it. Between them is a mapping from
the model's own parameter names to values in the specification, kept in a small JSON
file beside the model so that renaming a parameter in the CAD never needs a code
change::

    {
      "units": "mm",
      "parameters": [
        {"fusion": "Femoral_ML", "from": "femoral_component.dimensions_mm.ml"},
        {"fusion": "Insert_Thickness", "from": "insert.thickness_mm",
         "scale": 1.0, "offset_mm": 0.0}
      ]
    }

``from`` is a dotted path into the specification. ``scale`` and ``offset_mm`` are
optional and applied as ``value * scale + offset_mm``, for a CAD dimension defined from
a different datum than the planner's -- a margin inside the bone outline, say.

The Fusion script in ``fusion/`` imports this module, so what is previewed here with
``tka cad-params`` is exactly what is set in the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ParameterValue", "load_mapping", "resolve_parameters", "lookup"]


@dataclass(frozen=True)
class ParameterValue:
    """One CAD parameter, its value, and where the value came from."""

    fusion: str
    source: str
    value_mm: float | None
    problem: str | None = None

    @property
    def expression(self) -> str | None:
        """What to type into the CAD parameter, with its unit."""
        return None if self.value_mm is None else f"{self.value_mm:.3f} mm"


def load_mapping(path) -> dict:
    """Read and check a parameter mapping."""
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if mapping.get("units", "mm") != "mm":
        raise ValueError("Only millimetre mappings are supported.")
    entries = mapping.get("parameters")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A mapping needs a non-empty 'parameters' list.")
    for entry in entries:
        if not entry.get("fusion") or not entry.get("from"):
            raise ValueError(f"Every parameter needs 'fusion' and 'from': {entry}")
    return mapping


def lookup(spec: dict, dotted: str):
    """The value at a dotted path in the specification, or ``KeyError``."""
    value = spec
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def resolve_parameters(spec: dict, mapping: dict) -> list[ParameterValue]:
    """Every mapped parameter's value for this patient.

    A path the specification lacks -- a notch width the femur did not show, say -- is
    returned with its problem rather than raised, so the CAD script can report every
    gap at once and leave those parameters as they were.
    """
    if spec.get("schema") != "tka-planner/implant-spec":
        raise ValueError("Not an implant specification.")
    resolved = []
    for entry in mapping["parameters"]:
        source = entry["from"]
        try:
            raw = lookup(spec, source)
        except KeyError:
            resolved.append(ParameterValue(entry["fusion"], source, None,
                                           "not in this specification"))
            continue
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            resolved.append(ParameterValue(entry["fusion"], source, None,
                                           f"not a number: {raw!r}"))
            continue
        value = float(raw) * float(entry.get("scale", 1.0)) \
            + float(entry.get("offset_mm", 0.0))
        resolved.append(ParameterValue(entry["fusion"], source, value))
    return resolved
