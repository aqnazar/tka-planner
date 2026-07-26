"""Sizing measurements taken the way the implant chart defines them.

Sizing is only as good as the dimension driving it, and the two bones want different
treatment.

**Femur: the bounding box along the mediolateral axis.** The distal femur is widest at
the condyles and epicondyles, and that width is what the femoral component spans, so the
extent of the distal block along the frame's mediolateral axis is the measurement.
Measuring in the *frame's* axis rather than the world's matters, because a femur lies at
whatever rotation the scanner found it in and a world-aligned box inflates with that
rotation.

**Tibia: the resection cross-section, not a bounding box.** A bounding box of the
proximal tibia is measured to the wrong features. The intercondylar eminence rises above
the articular surface and the tubercle juts forward below it, so a box drawn round the
whole proximal end reports the extent of the spines and the tuberosity rather than of
the plateau the component sits on.

The fix is to measure the surface the implant actually meets. Cutting a few millimetres
into the tibia from the articular surface produces the resection cross-section, and the
anteroposterior extent of *that* outline is the tibial anteroposterior dimension. This
is the method the implant system's own sizing was defined against.

Finding the articular surface needs one precaution: the highest point of a proximal
tibia is the eminence, not the plateau. The reference level is therefore taken from the
weight-bearing surface with the central intercondylar region excluded, so the cut lands
on the plateau rather than partway down the spines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import AnatomicalFrame
from .geometry import regional_mask
from .meshio import Mesh

__all__ = [
    "SizingMeasurement",
    "measure_femoral_ml",
    "measure_tibial_plateau",
]

# Fraction of the plateau width, either side of the midline, treated as intercondylar
# and excluded when locating the articular surface. The eminence occupies roughly the
# central third; 0.20 either way is comfortably clear of the weight-bearing surface.
EMINENCE_HALF_WIDTH_FRACTION = 0.20


@dataclass(frozen=True)
class SizingMeasurement:
    """A dimension together with how it was obtained."""

    ml_mm: float
    ap_mm: float
    method: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ml_mm": round(self.ml_mm, 2),
            "ap_mm": round(self.ap_mm, 2),
            "method": self.method,
            "diagnostics": self.diagnostics,
        }


def measure_femoral_ml(
    mesh: Mesh,
    frame: AnatomicalFrame,
    *,
    distal_fraction: float = 0.25,
) -> SizingMeasurement:
    """Mediolateral and anteroposterior extent of the distal femur.

    A bounding box of the distal block, taken in the frame's own axes. Both dimensions
    come from the same region, so they describe one anatomical structure rather than two
    unrelated extremes of the segmentation.
    """
    distal = mesh.vertices[
        regional_mask(mesh.vertices, frame.z_proximal,
                      fraction=distal_fraction, end="low")
    ]
    local = frame.to_local(distal)

    return SizingMeasurement(
        ml_mm=float(np.ptp(local[:, 1])),
        ap_mm=float(np.ptp(local[:, 0])),
        method="measure.femur.frame_aligned_distal_bbox.v1",
        diagnostics={
            "region": f"distal {distal_fraction:.0%} of the segment",
            "n_vertices": int(len(distal)),
            "world_aligned_ml_mm": round(float(mesh.extent[0]), 2),
        },
    )


def measure_tibial_plateau(
    mesh: Mesh,
    frame: AnatomicalFrame,
    *,
    resection_depth_mm: float = 5.0,
    slab_thickness_mm: float = 1.5,
) -> SizingMeasurement:
    """Plateau dimensions from the resection cross-section.

    Cuts ``resection_depth_mm`` into the tibia below the articular surface and measures
    the outline exposed. This is the surface the tibial tray sits on, so its extent is
    the dimension the component must match.

    The articular reference deliberately excludes the central intercondylar region: the
    highest point of a proximal tibia belongs to the eminence, and referencing it would
    put the cut several millimetres too high, still inside the spines, reporting their
    extent instead of the plateau's.

    Both extents are reported from the same cross-section. The anteroposterior one is
    the point of the exercise -- a bounding box of the whole proximal tibia is inflated
    by the tubercle anteriorly and the eminence above -- but the mediolateral extent of
    the same outline is the honest partner to it.
    """
    local = frame.to_local(mesh.vertices)
    anterior, patient_left, proximal = local[:, 0], local[:, 1], local[:, 2]

    # Restrict to the proximal end before looking for the articular surface.
    proximal_region = proximal >= proximal.max() - 60.0
    if proximal_region.sum() < 100:
        proximal_region = proximal >= np.percentile(proximal, 80)

    # Exclude the intercondylar region, so the reference is weight-bearing surface.
    width = float(np.ptp(patient_left[proximal_region]))
    midline = float(np.median(patient_left[proximal_region]))
    off_midline = np.abs(patient_left - midline) > EMINENCE_HALF_WIDTH_FRACTION * width

    articular = proximal_region & off_midline
    if articular.sum() < 50:
        articular = proximal_region

    # The articular surface height, taken robustly rather than from a single vertex.
    surface_level = float(np.percentile(proximal[articular], 99.0))
    cut_level = surface_level - resection_depth_mm

    half = slab_thickness_mm / 2.0
    on_cut = np.abs(proximal - cut_level) <= half
    if on_cut.sum() < 20:
        # Widen once rather than fail; a coarse mesh may have few vertices in a
        # 1.5 mm slab.
        on_cut = np.abs(proximal - cut_level) <= slab_thickness_mm * 2.0
    if on_cut.sum() < 3:
        raise ValueError(
            f"No tibial cross-section found {resection_depth_mm} mm below the "
            f"articular surface. The mesh may be truncated above the plateau."
        )

    return SizingMeasurement(
        ml_mm=float(np.ptp(patient_left[on_cut])),
        ap_mm=float(np.ptp(anterior[on_cut])),
        method="measure.tibia.resection_cross_section.v1",
        diagnostics={
            "resection_depth_mm": resection_depth_mm,
            "articular_surface_level_local_mm": round(surface_level, 2),
            "cut_level_local_mm": round(cut_level, 2),
            "n_vertices_on_cut": int(on_cut.sum()),
            "eminence_excluded": True,
            # For comparison: what a naive bounding box of the proximal end would say.
            "naive_proximal_bbox_ap_mm": round(
                float(np.ptp(anterior[proximal_region])), 2
            ),
        },
    )
