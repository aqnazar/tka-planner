"""The patient-specific implant: its dimensions, measured on the cuts.

The implant is parametric in every dimension, not only in size. So the planner does not
pick an implant; it measures the one this patient needs and hands the measurements to
the CAD model. Everything here is measured on the **resection surfaces**, because that is
where each component meets the bone:

* the **femoral component** at the distal femoral cut: its overall mediolateral width,
  each condyle's width and anteroposterior length, the intercondylar notch between them,
  the anteroposterior depth from the posterior condyles to the anterior cortex, and the
  outline of the cut;
* the **tibial tray** at the tibial cut: its mediolateral width, its anteroposterior depth
  overall and in each compartment (the tibia is asymmetric), the notch the PCL leaves at
  the back, and the outline;
* the **insert**: the thickness that closes the joint, from the plan.

Every dimension is expressed in the component's own CAD axes, with its origin where the
plan puts the component's CAD origin, so the numbers mean the same thing in the CAD
model as here::

    femoral component   +X anterior, +Y patient-left, +Z proximal
    tibial tray         +X patient-left, +Y posterior, +Z proximal

The outlines are the traced boundary of each section, resampled to 96 points evenly
spaced along it, in the component's X and Y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fit import fill_section, slice_mesh
from .meshio import Mesh

__all__ = ["ImplantSpec", "measure_femoral_section", "measure_tibial_section",
           "measure_femoral_depth", "section_mask", "traced_outline"]

GRID_MM = 0.25
OUTLINE_POINTS = 96


@dataclass(frozen=True)
class ImplantSpec:
    """One component's measured dimensions, in its own CAD frame."""

    component: str
    dimensions: dict
    outline_mm: list
    frame: str
    method: str = "implant_spec.section.v1"
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "method": self.method,
            "frame": self.frame,
            "dimensions_mm": {k: round(float(v), 2) for k, v in self.dimensions.items()},
            "outline_mm": [[round(float(x), 2), round(float(y), 2)]
                           for x, y in self.outline_mm],
            "diagnostics": self.diagnostics,
        }


def section_mask(mesh: Mesh, pose: np.ndarray, *, offset_mm: float = 0.0,
                 grid_mm: float = GRID_MM):
    """The section of a mesh by a component's own XY plane, filled on a grid.

    ``pose`` is the component's 4x4 pose; the section is taken at local z =
    ``offset_mm`` and returned in local x, y. Returns ``(mask, xs, ys)``.
    """
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3] / np.linalg.norm(pose[:3, 0])
    origin = pose[:3, 3]
    normal = rotation[:, 2]
    segments = slice_mesh(mesh, origin + offset_mm * normal, normal)
    if not len(segments):
        raise ValueError("The component's plane does not cut the bone.")
    local = (segments - origin) @ rotation
    flat = local[:, :, :2]
    low = flat.reshape(-1, 2).min(axis=0) - 2 * grid_mm
    high = flat.reshape(-1, 2).max(axis=0) + 2 * grid_mm
    xs = np.arange(low[0], high[0], grid_mm) + grid_mm * 0.5013
    ys = np.arange(low[1], high[1], grid_mm) + grid_mm * 0.4987
    return fill_section(flat, xs, ys), xs, ys


def traced_outline(mask, xs, ys, points: int = OUTLINE_POINTS) -> list:
    """The outer boundary of a section, traced and resampled evenly by arc length.

    Boundary cells are chained to their nearest unvisited neighbour, starting from the
    most posterior-left cell, which walks the outer edge of a simply connected section.
    Casting rays from the centroid instead fails on the distal femur, whose centroid
    falls in the intercondylar notch, off the bone.
    """
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    rows, cols = np.nonzero(mask & ~interior)
    if len(rows) < 3:
        raise ValueError("The section is too small to outline.")
    cells = {(int(r), int(c)) for r, c in zip(rows, cols)}
    current = min(cells, key=lambda rc: (rc[1], rc[0]))
    chain = [current]
    cells.discard(current)
    neighbours = [(dr, dc) for reach in (1, 2, 3)
                  for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)
                  if max(abs(dr), abs(dc)) == reach]
    while cells:
        step = next(((current[0] + dr, current[1] + dc) for dr, dc in neighbours
                     if (current[0] + dr, current[1] + dc) in cells), None)
        if step is None:
            break
        chain.append(step)
        cells.discard(step)
        current = step

    trace = np.array([[xs[c], ys[r]] for r, c in chain])
    trace = np.vstack([trace, trace[:1]])
    lengths = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(trace, axis=0), axis=1))])
    targets = np.linspace(0.0, lengths[-1], points, endpoint=False)
    return [(float(np.interp(t, lengths, trace[:, 0])),
             float(np.interp(t, lengths, trace[:, 1]))) for t in targets]


def _runs(occupied: np.ndarray) -> list:
    """(start, stop) index pairs of the True runs in a boolean row."""
    padded = np.concatenate([[False], occupied, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2]))


def measure_femoral_section(femur: Mesh, pose: np.ndarray, *,
                            medial_sign: float) -> ImplantSpec:
    """The distal femoral cut, measured in the femoral component's frame.

    ``medial_sign`` is +1 when the medial side is local +Y (patient-left: a right knee)
    and -1 when it is local -Y (a left knee).
    """
    mask, xs, ys = section_mask(femur, pose)
    gx, gy = np.meshgrid(xs, ys)
    ap = gx[mask]
    ml = gy[mask]

    # The condyles are two lobes across the back of the section, with the notch between
    # them. Look along ML through the posterior part of the cut, where the notch is.
    posterior_band = (gx >= ap.min()) & (gx <= ap.min() + 0.35 * np.ptp(ap))
    dims = {
        "ml": float(np.ptp(ml)),
        "ap_distal": float(np.ptp(ap)),
    }
    # Rows of the mask run along local Y, the mediolateral axis the lobes lie along.
    across_y = (mask & posterior_band).any(axis=1)
    runs_y = sorted(sorted(_runs(across_y), key=lambda r: r[1] - r[0],
                           reverse=True)[:2])
    if len(runs_y) == 2:
        (a0, a1), (b0, b1) = runs_y
        low_width = float(ys[a1 - 1] - ys[a0])
        high_width = float(ys[b1 - 1] - ys[b0])
        notch = float(ys[b0] - ys[a1 - 1])
        low_ap = float(np.ptp(gx[mask & (gy >= ys[a0]) & (gy <= ys[a1 - 1])]))
        high_ap = float(np.ptp(gx[mask & (gy >= ys[b0]) & (gy <= ys[b1 - 1])]))
        # Low Y is patient-right. On a left knee patient-right is medial.
        if medial_sign < 0:
            medial = (low_width, low_ap)
            lateral = (high_width, high_ap)
        else:
            medial = (high_width, high_ap)
            lateral = (low_width, low_ap)
        dims.update({
            "medial_condyle_width": medial[0],
            "lateral_condyle_width": lateral[0],
            "medial_condyle_ap": medial[1],
            "lateral_condyle_ap": lateral[1],
            "notch_width": notch,
        })
    return ImplantSpec(
        component="femoral_component",
        dimensions=dims,
        outline_mm=traced_outline(mask, xs, ys),
        frame="+X anterior, +Y patient-left, +Z proximal; origin on the distal cut",
        diagnostics={"grid_mm": GRID_MM,
                     "section_centre_mm": [float(ap.min() + ap.max()) / 2.0,
                                           float(ml.min() + ml.max()) / 2.0],
                     "condyles_found": len(runs_y) == 2},
    )


def measure_femoral_depth(femur: Mesh, pose: np.ndarray, *, height_mm: float) -> dict:
    """The femur's anteroposterior depth over the height the component covers.

    From the posterior condyles to the anterior cortex, taken over the bone between the
    distal cut and ``height_mm`` above it -- the height of the component's anterior
    flange -- in the component's frame. This is the dimension the component's box must
    span.
    """
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3] / np.linalg.norm(pose[:3, 0])
    # Sampled over the surface rather than taken at the vertices, so a coarse mesh with
    # long triangles spanning the band is measured as well as a dense segmentation.
    from .insert import surface_samples

    local = (surface_samples(femur, per_mm2=1.0) - pose[:3, 3]) @ rotation
    band = (local[:, 2] >= 0.0) & (local[:, 2] <= height_mm)
    if not band.any():
        raise ValueError("No femur above the distal cut within the component's height.")
    x = local[band, 0]
    return {"ap_overall": float(np.ptp(x)), "anterior_mm": float(x.max()),
            "posterior_mm": float(x.min()), "height_mm": float(height_mm)}


def measure_tibial_section(tibia: Mesh, pose: np.ndarray, *,
                           medial_sign: float) -> ImplantSpec:
    """The tibial cut, measured in the tray's frame.

    ``medial_sign`` is +1 when the medial side is local +X (patient-left: a right knee)
    and -1 when it is local -X (a left knee). Local +Y is posterior.
    """
    mask, xs, ys = section_mask(tibia, pose)
    gx, gy = np.meshgrid(xs, ys)
    ml = gx[mask]
    ap = gy[mask]
    width = float(np.ptp(ml))
    left, right = ml.min(), ml.max()

    def compartment_ap(centre):
        """AP depth through the middle of a compartment, a band a fifth of the width."""
        cols = mask & (np.abs(gx - centre) <= width / 10.0)
        return float(np.ptp(gy[cols])) if cols.any() else 0.0

    low_ap = compartment_ap(left + 0.25 * width)
    high_ap = compartment_ap(right - 0.25 * width)

    # The PCL notch: the back edge of the section (largest +Y, since +Y is posterior)
    # dips forwards between the compartments, wherever it happens to be. Its depth at a
    # column is how far the back edge sits in front of the lower of the two highest
    # points either side of it -- water trapped between two walls -- and the notch is
    # the deepest such dip away from the section's ends.
    has = mask.any(axis=0)
    columns = np.flatnonzero(has)
    back = np.array([ys[mask[:, j]].max() for j in columns])
    left_wall = np.maximum.accumulate(back)
    right_wall = np.maximum.accumulate(back[::-1])[::-1]
    trapped = np.minimum(left_wall, right_wall) - back
    inner = (xs[columns] > left + 0.1 * width) & (xs[columns] < right - 0.1 * width)
    trapped[~inner] = 0.0
    notch_depth = float(trapped.max()) if len(trapped) else 0.0
    notch_width = 0.0
    if notch_depth > 0.5:
        deepest = int(np.argmax(trapped))
        above_half = trapped > 0.5 * notch_depth
        lo = hi = deepest
        while lo > 0 and above_half[lo - 1]:
            lo -= 1
        while hi < len(above_half) - 1 and above_half[hi + 1]:
            hi += 1
        notch_width = float(xs[columns[hi]] - xs[columns[lo]])

    medial_ap, lateral_ap = ((high_ap, low_ap) if medial_sign > 0 else (low_ap, high_ap))
    return ImplantSpec(
        component="tibial_component",
        dimensions={
            "ml": width,
            "ap": float(np.ptp(ap)),
            "medial_ap": medial_ap,
            "lateral_ap": lateral_ap,
            "pcl_notch_depth": notch_depth,
            "pcl_notch_width": notch_width,
        },
        outline_mm=traced_outline(mask, xs, ys),
        frame="+X patient-left, +Y posterior, +Z proximal; origin on the tibial cut",
        diagnostics={"grid_mm": GRID_MM,
                     "section_centre_mm": [float(left + right) / 2.0,
                                           float(ap.min() + ap.max()) / 2.0]},
    )
