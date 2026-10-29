"""The fit, as a colour map a viewer can draw without computing anything.

:func:`tka_planner.core.fit.section_fit` measures the fit on a grid in each cut plane and
can keep the signed distance at every cell. This turns that grid into a mesh lying on
the cut, with a colour and a value at every vertex, so the viewer only has to draw it:
the colour scale is decided here, once, and the legend the viewer shows is served from
the same table the colours were read from.

The scale is diverging and fixed, so two knees can be compared by eye: red where the
component overhangs the bone, blue where it leaves bone uncovered, near-white where it
is seated on bone. The ends are clamped at :data:`SCALE_MM`; the legend says so. The
colours are ColorBrewer's RdBu, which stays readable to the commonest colour-vision
deficiencies.

The grid is drawn at half its resolution (0.5 mm cells rather than 0.25 mm): the map is
for seeing where, and the numbers in the table are still read from the full grid.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.fit import FitField
from tka_planner.core.meshio import Mesh

__all__ = ["SCALE_MM", "STOPS", "legend", "colours_for", "fit_map_mesh",
           "FIT_MAP_NODES", "DRAW_STRIDE", "LIFT_MM"]

# The scale runs from this much overhang to this much uncovered bone.
SCALE_MM = 6.0
# (value in mm, RGB) -- ColorBrewer RdBu, five classes, red for overhang.
STOPS = (
    (-SCALE_MM, (0.698, 0.094, 0.169)),
    (-2.0, (0.937, 0.541, 0.384)),
    (0.0, (0.969, 0.969, 0.969)),
    (2.0, (0.404, 0.663, 0.812)),
    (SCALE_MM, (0.129, 0.400, 0.675)),
)
# Every second cell of the fit grid is drawn.
DRAW_STRIDE = 2
# The map is lifted this far off the cut towards the component, so it is not buried in
# the cut face of the bone it describes.
LIFT_MM = 0.3

# Which scene node shows each component's map, and which set it moves with.
FIT_MAP_NODES = {
    "tibial_component": ("FitMap.tibial", "tibial"),
    "femoral_component": ("FitMap.femoral", "femoral"),
}


def legend(overhang_flag_mm: float | None = None) -> dict:
    """What the viewer needs to draw the legend: the stops, the ends and the words."""
    record = {
        "title": "Fit on the cut",
        "unit": "mm",
        "min_mm": -SCALE_MM,
        "max_mm": SCALE_MM,
        "stops": [[value, list(colour)] for value, colour in STOPS],
        "negative": "component overhangs bone",
        "positive": "bone left uncovered",
        "zero": "component seated on bone",
        "clamped": f"values beyond {SCALE_MM:g} mm are drawn at the end colour",
        "resolution_mm": None,
    }
    if overhang_flag_mm is not None:
        record["flag_mm"] = -float(overhang_flag_mm)
    return record


def colours_for(values_mm: np.ndarray) -> np.ndarray:
    """RGB in 0-1 for each value, interpolated linearly between the stops."""
    values = np.clip(np.asarray(values_mm, dtype=float), -SCALE_MM, SCALE_MM)
    xs = np.array([value for value, _ in STOPS])
    rgb = np.array([colour for _, colour in STOPS])
    return np.stack([np.interp(values, xs, rgb[:, k]) for k in range(3)], axis=-1)


def fit_map_mesh(grid: FitField, *, stride: int = DRAW_STRIDE,
                 lift_mm: float = LIFT_MM) -> Mesh | None:
    """A mesh on the cut, one vertex per drawn cell, carrying its value and colour.

    Cells with neither bone nor component are left out, so the map has the outline of
    the two together. Returns ``None`` when nothing is left to draw.
    """
    values = grid.signed_mm[::stride, ::stride]
    rows, columns = values.shape
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return None
    points = grid.points(rows=slice(None, None, stride), columns=slice(None, None, stride))
    points = points + grid.component_side * lift_mm * grid.normal

    index = np.full(values.shape, -1, dtype=np.int64)
    index[valid] = np.arange(int(valid.sum()))
    a, b = index[:-1, :-1], index[:-1, 1:]
    c, d = index[1:, 1:], index[1:, :-1]
    triangles = []
    # Two triangles per quad where all four corners are drawn, and the one triangle
    # three corners make where only three are, so the edge is not ragged.
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    triangles += [np.stack([a[full], b[full], c[full]], axis=1),
                  np.stack([a[full], c[full], d[full]], axis=1)]
    for p, q, r, missing in ((a, b, c, d), (a, c, d, b), (a, b, d, c), (b, c, d, a)):
        only = (p >= 0) & (q >= 0) & (r >= 0) & (missing < 0)
        triangles.append(np.stack([p[only], q[only], r[only]], axis=1))
    faces = np.concatenate(triangles).astype(np.int64)
    if not len(faces):
        return None
    flat = values[valid]
    return Mesh(
        vertices=points[valid].astype(float),
        faces=faces,
        metadata={"vertex_colours": colours_for(flat).astype(np.float32),
                  "vertex_values_mm": flat.astype(np.float32),
                  "cut": grid.cut,
                  "resolution_mm": float(abs(grid.xs[1] - grid.xs[0]) * stride)
                  if len(grid.xs) > 1 else None},
    )
