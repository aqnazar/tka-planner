"""Implant sizing, continuous rather than discrete.

The implant system this pipeline plans for is parametric: its geometry is a single
master shape driven by continuous parameters. The twelve sizes in ``SizeChart.csv`` are
samples of that family, published in discrete form because that is how traditional
implant systems are supplied -- not because the design is limited to twelve shapes.

Measuring the exported library confirms it. Every femoral implant STL has the same
aspect ratio to four decimal places (1 : 1.1788 : 0.8350 at S1 and identically at L3),
and the chart's femoral columns step exactly linearly. The parametric model is already
there, flattened into a lookup table.

So this module solves for a **continuous size parameter** and treats the discrete chart
as a legacy comparison mode. The practical consequence shows up immediately: case 005
has a femoral mediolateral width of 86.4 mm against a largest chart size of 84.0 mm, and
the legacy pipeline clamped it to L4 with only a printed warning. A continuous parameter
has no largest size to clamp against.

What the chart is and is not
----------------------------
Not every column is parametric, and pretending otherwise would be its own kind of
dishonesty:

* ``femur_ML``, ``femur_AP``, ``tibia_AP`` and both resection columns step **exactly**
  linearly. These describe a genuine one-parameter family.
* ``tibia_ML`` steps irregularly (2 or 4 mm), which looks hand-adjusted rather than
  generated. It is interpolated, not extrapolated from a fitted law.
* ``insert_thin`` and ``insert_thick`` are quantised to whole millimetres, and correctly
  so: polyethylene inserts are supplied in fixed thicknesses. These stay discrete.

The model is therefore piecewise-linear interpolation across the chart, with linear
extrapolation beyond either end. It reproduces the chart exactly at chart points, so
legacy parity is provable, while remaining continuous in between and unbounded outside.

Resection depth is not sizing
-----------------------------
``femur_distal_cut`` and ``tibia_proximal_cut`` are exactly proportional to implant
width (``0.1125`` and ``0.2625`` times ``femur_ML`` respectively, across all twelve
rows). They therefore describe the **thickness of the component**, a property of the
implant, not a surgical decision about the patient. Conflating the two is why the legacy
pipeline could not plan alignment at all: choosing a size fixed the resection depth,
leaving no free variable through which to express an alignment target. Here they are
reported as component properties, and the resection plane is set by the alignment
philosophy.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SizeChart",
    "SizingDecision",
    "load_size_chart",
    "solve_parametric_size",
    "select_discrete_size",
]

# Columns that are pure functions of the size parameter, verified exactly linear.
PARAMETRIC_COLUMNS = (
    "femur_ML", "femur_AP", "tibia_AP", "femur_distal_cut", "tibia_proximal_cut",
)
# Columns that are interpolated but not assumed linear.
INTERPOLATED_COLUMNS = ("tibia_ML", "patella_d")
# Columns that are genuinely discrete and must not be interpolated.
QUANTISED_COLUMNS = ("insert_thin", "insert_thick")

_CSV_COLUMNS = {
    "femur_ML": "femur_ML_mm",
    "femur_AP": "femur_AP_mm",
    "tibia_ML": "tibia_ML_mm",
    "tibia_AP": "tibia_AP_mm",
    "patella_d": "patella_d_mm",
    "insert_thin": "insert_thin_mm",
    "insert_thick": "insert_thick_mm",
    "femur_distal_cut": "femur_distal_cut_m",
    "tibia_proximal_cut": "tibia_proximal_cut_m",
}
# Columns the chart stores in metres, converted on load. The pipeline is millimetres
# throughout; a mixed-unit table is exactly the sort of thing that survives unnoticed.
_METRE_COLUMNS = ("femur_distal_cut", "tibia_proximal_cut")


@dataclass(frozen=True)
class SizeChart:
    """The implant family, read from ``SizeChart.csv`` and expressed continuously.

    ``parameter`` is the continuous size coordinate: 0 at the smallest chart row and
    increasing by 1 per row, so the chart's own sizes sit at integers. It has no units
    and no anatomical meaning -- it is simply the family's free variable.
    """

    labels: tuple[str, ...]
    columns: dict[str, np.ndarray]
    source_path: str | None = None
    sha256: str | None = None

    @property
    def parameter_grid(self) -> np.ndarray:
        return np.arange(len(self.labels), dtype=float)

    @property
    def n_sizes(self) -> int:
        return len(self.labels)

    def value_at(self, column: str, parameter: float) -> float:
        """Evaluate a column at a continuous size parameter.

        Interpolates within the chart and extrapolates linearly beyond it, using the
        slope of the outermost interval. Quantised columns are rounded to whole
        millimetres, because an insert 11.4 mm thick does not exist.
        """
        if column not in self.columns:
            raise KeyError(
                f"Unknown size chart column {column!r}. Available: "
                f"{sorted(self.columns)}"
            )
        values = self.columns[column]
        grid = self.parameter_grid

        if parameter < grid[0]:
            slope = values[1] - values[0]
            result = values[0] + slope * (parameter - grid[0])
        elif parameter > grid[-1]:
            slope = values[-1] - values[-2]
            result = values[-1] + slope * (parameter - grid[-1])
        else:
            result = float(np.interp(parameter, grid, values))

        if column in QUANTISED_COLUMNS:
            return float(round(result))
        return float(result)

    def parameter_for(self, column: str, target: float) -> float:
        """Invert a column: find the size parameter giving ``target``.

        The parametric columns are strictly increasing, so the inverse is unique and
        extrapolates cleanly past either end -- which is the whole point, since that is
        where the discrete chart used to clamp.
        """
        values = self.columns[column]
        grid = self.parameter_grid
        if not np.all(np.diff(values) > 0):
            raise ValueError(
                f"Column {column!r} is not strictly increasing, so it cannot be "
                f"inverted to a size parameter. Drive sizing from a parametric column "
                f"such as femur_ML."
            )

        if target <= values[0]:
            slope = values[1] - values[0]
            return float(grid[0] + (target - values[0]) / slope)
        if target >= values[-1]:
            slope = values[-1] - values[-2]
            return float(grid[-1] + (target - values[-1]) / slope)
        return float(np.interp(target, values, grid))

    def is_exactly_linear(self, column: str, tolerance: float = 1e-9) -> bool:
        """Whether a column is a true linear function of the size parameter.

        Used by the report to state which parts of the family are genuinely parametric
        and which were hand-adjusted.
        """
        steps = np.diff(self.columns[column])
        return bool(np.ptp(steps) < tolerance)

    def nearest_label(self, parameter: float) -> str:
        index = int(np.clip(round(parameter), 0, self.n_sizes - 1))
        return self.labels[index]

    def label_parameter(self, label: str) -> float:
        try:
            return float(self.labels.index(label))
        except ValueError:
            raise KeyError(
                f"Unknown size {label!r}. Available: {', '.join(self.labels)}"
            ) from None


def load_size_chart(path: "str | Path") -> SizeChart:
    """Read ``SizeChart.csv``, converting the metre columns to millimetres."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Size chart not found: {path}")

    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if not rows:
        raise ValueError(f"{path.name} contains no size rows.")

    missing = [
        csv_name for csv_name in _CSV_COLUMNS.values() if csv_name not in rows[0]
    ]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

    columns: dict[str, np.ndarray] = {}
    for name, csv_name in _CSV_COLUMNS.items():
        values = np.array([float(row[csv_name]) for row in rows])
        if name in _METRE_COLUMNS:
            values = values * 1000.0
        columns[name] = values

    return SizeChart(
        labels=tuple(row["SIZE"].strip() for row in rows),
        columns=columns,
        source_path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class SizingDecision:
    """A sizing result and the reasoning behind it."""

    method: str
    size_parameter: float
    nearest_discrete_size: str
    scale_factor: float
    driving_dimension: str
    measured_ml_mm: float
    measured_ap_mm: float | None
    implant_ml_mm: float
    implant_ap_mm: float
    femoral_thickness_mm: float
    tibial_construct_thickness_mm: float
    insert_thin_mm: float
    insert_thick_mm: float
    ap_mismatch_mm: float | None = None
    beyond_discrete_range: bool = False
    flags: tuple[str, ...] = ()
    rationale: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "size_parameter": round(self.size_parameter, 4),
            "nearest_discrete_size": self.nearest_discrete_size,
            "scale_factor": round(self.scale_factor, 6),
            "driving_dimension": self.driving_dimension,
            "measured_ml_mm": round(self.measured_ml_mm, 2),
            "measured_ap_mm": (round(self.measured_ap_mm, 2)
                               if self.measured_ap_mm is not None else None),
            "implant_ml_mm": round(self.implant_ml_mm, 2),
            "implant_ap_mm": round(self.implant_ap_mm, 2),
            "femoral_thickness_mm": round(self.femoral_thickness_mm, 3),
            "tibial_construct_thickness_mm": round(
                self.tibial_construct_thickness_mm, 3),
            "insert_thin_mm": self.insert_thin_mm,
            "insert_thick_mm": self.insert_thick_mm,
            "ap_mismatch_mm": (round(self.ap_mismatch_mm, 2)
                               if self.ap_mismatch_mm is not None else None),
            "beyond_discrete_range": self.beyond_discrete_range,
            "flags": list(self.flags),
            "rationale": self.rationale,
            "diagnostics": self.diagnostics,
        }


def solve_parametric_size(
    chart: SizeChart,
    *,
    measured_ml_mm: float,
    measured_ap_mm: float | None = None,
    driving_dimension: str = "femur_ML",
    reference_size: str | None = None,
    parameter_override: float | None = None,
) -> SizingDecision:
    """Solve for the continuous size parameter matching this patient's anatomy.

    The driving dimension is matched exactly; every other implant parameter follows from
    the same size parameter. ``scale_factor`` is relative to ``reference_size`` (the
    middle of the chart by default) and is what a mesh would be multiplied by.

    Where the patient's anteroposterior width is supplied, the residual the single
    parameter cannot absorb is reported as ``ap_mismatch_mm``. That number is the honest
    limit of a one-parameter family: matching mediolateral and anteroposterior width
    independently needs two parameters, and this reports how much that would be worth
    per patient rather than asserting it in the abstract.

    ``parameter_override`` is a size chosen by hand instead of solved. The measurement is
    still carried and still compared, so the decision records what the anatomy asked for
    alongside what was actually chosen -- which is the point of overriding it at all.
    """
    solved = chart.parameter_for(driving_dimension, measured_ml_mm)
    parameter = solved if parameter_override is None else float(parameter_override)

    reference_size = reference_size or chart.labels[chart.n_sizes // 2]
    reference_parameter = chart.label_parameter(reference_size)
    reference_ml = chart.value_at("femur_ML", reference_parameter)

    implant_ml = chart.value_at("femur_ML", parameter)
    implant_ap = chart.value_at("femur_AP", parameter)

    ap_mismatch = None
    flags: list[str] = []
    if measured_ap_mm is not None:
        ap_mismatch = implant_ap - measured_ap_mm
        if abs(ap_mismatch) > 3.0:
            flags.append("AP_ASPECT_MISMATCH")

    beyond = not (0.0 <= parameter <= chart.n_sizes - 1)
    if beyond:
        flags.append("BEYOND_DISCRETE_CHART_RANGE")

    nearest = chart.nearest_label(parameter)
    if beyond:
        placement = (
            f"This falls outside the discrete chart, where the legacy pipeline would "
            f"have clamped to {nearest}. "
        )
    else:
        placement = f"Nearest discrete size is {nearest}. "

    if parameter_override is not None:
        flags.append("SIZE_OVERRIDDEN")
        rationale = (
            f"Size set by hand to parameter {parameter:.3f}, implant width "
            f"{implant_ml:.1f} mm. The measurement itself was {driving_dimension} "
            f"{measured_ml_mm:.1f} mm, which solves to parameter {solved:.3f} "
            f"({chart.value_at('femur_ML', solved):.1f} mm). "
            + placement
            + f"Scale {implant_ml / reference_ml:.4f} relative to {reference_size}."
        )
    else:
        rationale = (
            f"Measured {driving_dimension} of {measured_ml_mm:.1f} mm maps to size "
            f"parameter {parameter:.3f} (chart sizes sit at integers). "
            + placement
            + f"Implant width {implant_ml:.1f} mm, scale "
              f"{implant_ml / reference_ml:.4f} relative to {reference_size}."
        )

    return SizingDecision(
        method="sizing.parametric.v1",
        size_parameter=parameter,
        nearest_discrete_size=nearest,
        scale_factor=implant_ml / reference_ml,
        driving_dimension=driving_dimension,
        measured_ml_mm=measured_ml_mm,
        measured_ap_mm=measured_ap_mm,
        implant_ml_mm=implant_ml,
        implant_ap_mm=implant_ap,
        femoral_thickness_mm=chart.value_at("femur_distal_cut", parameter),
        tibial_construct_thickness_mm=chart.value_at("tibia_proximal_cut", parameter),
        insert_thin_mm=chart.value_at("insert_thin", parameter),
        insert_thick_mm=chart.value_at("insert_thick", parameter),
        ap_mismatch_mm=ap_mismatch,
        beyond_discrete_range=beyond,
        flags=tuple(flags),
        rationale=rationale,
        diagnostics={
            "reference_size": reference_size,
            "tibia_ML_mm": round(chart.value_at("tibia_ML", parameter), 2),
            "tibia_AP_mm": round(chart.value_at("tibia_AP", parameter), 2),
            "patella_d_mm": round(chart.value_at("patella_d", parameter), 2),
        },
    )


def select_discrete_size(
    chart: SizeChart,
    *,
    measured_ml_mm: float,
    measured_ap_mm: float | None = None,
    driving_dimension: str = "femur_ML",
    rounding: str = "D",
) -> SizingDecision:
    """Choose one of the twelve chart sizes, reproducing the legacy behaviour.

    Retained for comparison rather than for use: it is what the traditional approach
    does, and quantifying the difference against the parametric solver across the cohort
    is a result the paper wants.

    ``rounding`` is ``"U"`` for the smallest size covering the measurement or ``"D"``
    for the largest not exceeding it, matching the legacy ``SIZE_ROUND`` switch. A bone
    outside the chart is clamped, and the clamp is flagged -- the legacy code merely
    printed a warning, which is why case 005 was silently undersized.
    """
    if rounding not in ("U", "D"):
        raise ValueError("rounding must be 'U' (round up) or 'D' (round down).")

    values = chart.columns[driving_dimension]
    flags: list[str] = []

    if rounding == "U":
        candidates = np.flatnonzero(values >= measured_ml_mm)
        index = int(candidates[0]) if len(candidates) else chart.n_sizes - 1
    else:
        candidates = np.flatnonzero(values <= measured_ml_mm)
        index = int(candidates[-1]) if len(candidates) else 0

    clamped = measured_ml_mm > values[-1] or measured_ml_mm < values[0]
    if clamped:
        flags.append("CLAMPED_TO_CHART_LIMIT")
        if measured_ml_mm > values[-1]:
            flags.append("EXCEEDS_LARGEST_SIZE")

    parameter = float(index)
    implant_ml = float(values[index])
    implant_ap = chart.value_at("femur_AP", parameter)
    difference = implant_ml - measured_ml_mm

    rationale = (
        f"Measured {driving_dimension} of {measured_ml_mm:.1f} mm selects "
        f"{chart.labels[index]} ({implant_ml:.1f} mm, {difference:+.1f} mm) by "
        f"round-{'up' if rounding == 'U' else 'down'}."
    )
    if clamped:
        rationale += (
            f" The measurement lies outside the chart, so the size was clamped to its "
            f"limit and the implant is undersized by {abs(difference):.1f} mm."
        )

    return SizingDecision(
        method=f"sizing.discrete.v1.round_{'up' if rounding == 'U' else 'down'}",
        size_parameter=parameter,
        nearest_discrete_size=chart.labels[index],
        scale_factor=1.0,
        driving_dimension=driving_dimension,
        measured_ml_mm=measured_ml_mm,
        measured_ap_mm=measured_ap_mm,
        implant_ml_mm=implant_ml,
        implant_ap_mm=implant_ap,
        femoral_thickness_mm=chart.value_at("femur_distal_cut", parameter),
        tibial_construct_thickness_mm=chart.value_at("tibia_proximal_cut", parameter),
        insert_thin_mm=chart.value_at("insert_thin", parameter),
        insert_thick_mm=chart.value_at("insert_thick", parameter),
        ap_mismatch_mm=(implant_ap - measured_ap_mm
                        if measured_ap_mm is not None else None),
        beyond_discrete_range=clamped,
        flags=tuple(flags),
        rationale=rationale,
        diagnostics={"size_difference_mm": round(difference, 2)},
    )
