"""The femoral cut box: where the five femoral cuts go, measured on the patient's bone.

The femoral component sits on five cuts: distal, posterior, anterior, and two chamfers
joining the distal face to the other two. The distal cut is planned with the alignment.
This places the other four, in the component's own frame (+X anterior, +Y patient-left,
+Z proximal, origin where the plan seats the component on the distal cut).

The design's angles are fixed, by the author's ruling; only lengths change from patient
to patient. So the anterior cut keeps the design's flare from the component axis, the
chamfers keep their angle, and the posterior cut stays parallel to the axis. What the
bone decides is where each sits.

* **Posterior cut** -- posterior referencing: the component's posterior-condyle
  thickness comes off the more prominent condyle, so that condyle's offset is restored
  and the other loses less, according to the planned rotation.
* **Anterior cut** -- flush at the top of the trochlea. Above the trochlea the anterior
  cortex runs up the shaft nearly parallel to the axis, and a flared plane can cross it
  only once. The cut crosses it where the trochlear ridges have blended into the shaft
  line, and the flange ends there: no notch in the cortex above it, and no gap under its
  tip. How far this lowers the patellofemoral joint against the native trochlea is
  reported, for review across the cohort.
* **Chamfers** -- the box between the two is divided between the distal face and the
  chamfers in the design's proportions.

The design's numbers are private design data. They are passed in as a
:class:`BoxDesign`, read from the implant library, and are not written here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .meshio import Mesh

__all__ = ["BoxDesign", "FemoralCutBox", "femoral_cut_box", "femur_samples",
           "split_box"]

# The shaft line is fitted to the anterior cortex over this height above the distal cut.
# Every scan in the cohort reaches at least 148 mm.
SHAFT_WINDOW_MM = (60.0, 100.0)
# The trochlea stands proud of the shaft line and fades into it smoothly, with no edge to
# find. It is taken to end where it has fallen to this fraction of its own height, and
# stays there for TROCHLEA_RUN_MM: relative to the trochlea rather than an absolute
# tolerance, so a millimetre-scale ridge is not lost in a tenth of a millimetre of noise,
# and sustained, so one noisy band does not end it early.
TROCHLEA_FRACTION = 0.10
TROCHLEA_FLOOR_MM = 0.25
TROCHLEA_RUN_MM = 5.0
# Below this the femur has no trochlea to speak of.
MIN_TROCHLEA_MM = 0.5
# A trochlea still fading this close to the shaft window may have biased the shaft line.
WINDOW_MARGIN_MM = 5.0
# A trochlea shorter than this is not a trochlea but noise at the cut.
MIN_FLANGE_HEIGHT_MM = 10.0
# The posterior condyles are looked for up to this height above the distal cut.
POSTERIOR_CONDYLE_HEIGHT_MM = 45.0
PROFILE_STEP_MM = 1.0
SAMPLES_PER_MM2 = 1.0


@dataclass(frozen=True)
class BoxDesign:
    """The femoral component's fixed angles and thicknesses, from its CAD design."""

    anterior_flare_deg: float
    chamfer_deg: float
    posterior_condyle_thickness_mm: float
    flange_thickness_mm: float
    distal_face_fraction: float
    posterior_chamfer_fraction: float

    def __post_init__(self):
        if self.posterior_condyle_thickness_mm <= 0 or self.flange_thickness_mm <= 0:
            raise ValueError("The component's thicknesses must be positive.")
        if not (0.0 < self.distal_face_fraction and 0.0 < self.posterior_chamfer_fraction
                and self.distal_face_fraction + self.posterior_chamfer_fraction < 1.0):
            raise ValueError("The distal face and posterior chamfer must each take a "
                             "positive share of the box and leave room for the anterior "
                             "chamfer.")
        if not 0.0 < self.chamfer_deg < 90.0 or not 0.0 <= self.anterior_flare_deg < 45.0:
            raise ValueError("The chamfer and flare angles are out of range.")
        run = 1.0 / np.tan(np.radians(self.chamfer_deg))
        if run <= np.tan(np.radians(self.anterior_flare_deg)):
            raise ValueError("The anterior chamfer is too steep ever to meet the flared "
                             "anterior cut.")

    @classmethod
    def from_dict(cls, data: dict) -> "BoxDesign":
        return cls(**{name: float(data[name]) for name in cls.__dataclass_fields__})


def split_box(posterior_cut_x_mm: float, anterior_cut_x_at_distal_mm: float,
              design: BoxDesign) -> dict:
    """Divide the box between the distal face and the chamfers, in the design's
    proportions of its depth at the distal cut.

    The posterior chamfer runs back from the distal face to the posterior cut; the
    anterior chamfer rises from the front of the distal face until it meets the flared
    anterior cut. The component's CAD origin is the centre of the distal face.
    """
    depth = anterior_cut_x_at_distal_mm - posterior_cut_x_mm
    if depth <= 0.0:
        raise ValueError("The anterior cut lies behind the posterior cut.")
    distal_face = design.distal_face_fraction * depth
    posterior_chamfer = design.posterior_chamfer_fraction * depth
    front_of_face = posterior_cut_x_mm + posterior_chamfer + distal_face
    run = 1.0 / np.tan(np.radians(design.chamfer_deg))
    flare = np.tan(np.radians(design.anterior_flare_deg))
    anterior_chamfer = (anterior_cut_x_at_distal_mm - front_of_face) / (run - flare)
    return {
        "box_ap_mm": float(depth),
        "distal_face_mm": float(distal_face),
        "posterior_chamfer_mm": float(posterior_chamfer),
        "anterior_chamfer_mm": float(anterior_chamfer),
        "cad_origin_x_mm": float(posterior_cut_x_mm + posterior_chamfer
                                 + distal_face / 2.0),
    }


@dataclass(frozen=True)
class FemoralCutBox:
    """Where the femoral cuts go, and what each takes off, in the component frame."""

    posterior_cut_x_mm: float
    anterior_cut_x_at_distal_mm: float
    anterior_flare_deg: float
    flange_height_mm: float
    shaft_slope_deg: float
    posterior_medial_mm: float
    posterior_lateral_mm: float
    anterior_mm: float
    patellofemoral_lowering_mm: float
    anterior_notch_mm: float
    split: dict = field(default_factory=dict)
    trochlea_excess_mm: tuple = ()
    warnings: tuple = ()

    @property
    def cad_origin_x_mm(self) -> float:
        return self.split["cad_origin_x_mm"]

    def anterior_cut_x_at(self, height_mm: float) -> float:
        """The anterior cut's AP position at a height above the distal cut."""
        return float(self.anterior_cut_x_at_distal_mm
                     + height_mm * np.tan(np.radians(self.anterior_flare_deg)))

    def to_dict(self) -> dict:
        split = {k: round(float(v), 2) for k, v in self.split.items()}
        return {
            "method": "femoral_box.v1",
            "frame": "femoral component: +X anterior, +Y patient-left, +Z proximal; "
                     "origin where the plan seats it on the distal cut",
            "posterior_cut_x_mm": round(self.posterior_cut_x_mm, 2),
            "anterior_cut_x_at_distal_mm": round(self.anterior_cut_x_at_distal_mm, 2),
            "anterior_flare_deg": round(self.anterior_flare_deg, 3),
            "flange_height_mm": round(self.flange_height_mm, 2),
            "shaft_slope_deg": round(self.shaft_slope_deg, 2),
            **split,
            "resections_mm": {
                "posterior_medial": round(self.posterior_medial_mm, 2),
                "posterior_lateral": round(self.posterior_lateral_mm, 2),
                "anterior": round(self.anterior_mm, 2),
            },
            "patellofemoral_lowering_mm": round(self.patellofemoral_lowering_mm, 2),
            "anterior_notch_mm": round(self.anterior_notch_mm, 2),
            # The anterior profile's height over the shaft line at each millimetre above
            # the distal cut, from which the flange height was read: for review.
            "trochlea_heights_mm": [round(i * PROFILE_STEP_MM, 1)
                                    for i in range(len(self.trochlea_excess_mm))],
            "trochlea_excess_mm": [None if not np.isfinite(v) else round(float(v), 2)
                                   for v in self.trochlea_excess_mm],
            "warnings": list(self.warnings),
            "cad_parameters": {
                "C_Distal_cut_surface": split["distal_face_mm"],
                "B_Posterior_chamfer_projection": split["posterior_chamfer_mm"],
                "D_Anterior_chamfer_projection": split["anterior_chamfer_mm"],
            },
        }


def femoral_cut_box(femur: Mesh, pose: np.ndarray, *, medial_sign: float,
                    design: BoxDesign, samples: np.ndarray | None = None) -> FemoralCutBox:
    """Place the femoral cut box on the bone, in the frame of the planned component.

    ``medial_sign`` is +1 when the medial side is local +Y (a right knee) and -1 when it
    is local -Y (a left knee). ``samples`` are points over the femur's surface in scan
    coordinates (:func:`femur_samples`), for a caller that re-plans often and keeps them;
    without them the femur is sampled here. Raises ``ValueError`` with the reason when
    the bone does not allow the box to be placed.
    """
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3] / np.linalg.norm(pose[:3, 0])
    if samples is None:
        samples = femur_samples(femur)
    local = (samples - pose[:3, 3]) @ rotation
    x, y, z = local.T

    distal = (z >= 0.0) & (z <= 2.0)
    if not distal.any():
        raise ValueError("The distal cut does not meet the femur.")
    centre_y = float(y[distal].min() + y[distal].max()) / 2.0
    within = np.abs(y - centre_y) <= float(np.ptp(y[distal])) / 2.0

    low, high = SHAFT_WINDOW_MM
    top = float(z[within].max())
    if top < high:
        raise ValueError(
            f"The scan ends {top:.0f} mm above the distal cut; placing the anterior cut "
            f"on the anterior cortex needs the shaft up to {high:.0f} mm.")
    heights = np.arange(0.0, high + PROFILE_STEP_MM / 2.0, PROFILE_STEP_MM)
    profile = _anterior_profile(x[within], z[within], heights)
    index, shaft_slope_deg, excess = _trochlea_top(heights, profile)

    flare = float(np.tan(np.radians(design.anterior_flare_deg)))
    flange_height = float(heights[index])
    anterior_at_distal = float(profile[index]) - flange_height * flare
    plane = anterior_at_distal + heights * flare
    warnings = []
    if flange_height > low - WINDOW_MARGIN_MM:
        warnings.append(
            f"The trochlea is still fading {flange_height:.0f} mm above the distal cut, "
            f"close to where the shaft line is fitted from ({low:.0f} mm); the flange "
            f"height may be underestimated.")
    finite = np.isfinite(profile)
    below = finite & (heights <= flange_height)
    above = finite & (heights > flange_height)
    anterior = float(np.max(profile[below] - plane[below]))
    notch = float(max(0.0, np.max(profile[above] - plane[above]))) if above.any() else 0.0

    condyles = within & (z >= 0.0) & (z <= POSTERIOR_CONDYLE_HEIGHT_MM)
    side = np.sign(y - centre_y)
    medial = condyles & (side == np.sign(medial_sign))
    lateral = condyles & (side == -np.sign(medial_sign))
    if not medial.any() or not lateral.any():
        raise ValueError("Both posterior condyles are needed to place the posterior cut.")
    back_medial, back_lateral = float(x[medial].min()), float(x[lateral].min())
    posterior_cut = min(back_medial, back_lateral) + design.posterior_condyle_thickness_mm

    return FemoralCutBox(
        posterior_cut_x_mm=posterior_cut,
        anterior_cut_x_at_distal_mm=anterior_at_distal,
        anterior_flare_deg=float(design.anterior_flare_deg),
        flange_height_mm=flange_height,
        shaft_slope_deg=shaft_slope_deg,
        posterior_medial_mm=posterior_cut - back_medial,
        posterior_lateral_mm=posterior_cut - back_lateral,
        anterior_mm=anterior,
        patellofemoral_lowering_mm=anterior - design.flange_thickness_mm,
        anterior_notch_mm=notch,
        split=split_box(posterior_cut, anterior_at_distal, design),
        trochlea_excess_mm=tuple(float(v) for v in excess),
        warnings=tuple(warnings),
    )


def femur_samples(femur: Mesh) -> np.ndarray:
    """Points over the femur's surface, in scan coordinates.

    Sampled over the surface rather than taken at the vertices, so a coarse mesh with
    long triangles is measured as well as a dense segmentation. They do not depend on
    the plan, so a caller that re-plans often can keep them.
    """
    from .insert import surface_samples

    return surface_samples(femur, per_mm2=SAMPLES_PER_MM2)


def _anterior_profile(x: np.ndarray, z: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """The most anterior bone in each band of height, NaN where a band holds none."""
    bands = np.floor((z - heights[0]) / PROFILE_STEP_MM + 0.5).astype(int)
    keep = (bands >= 0) & (bands < len(heights))
    profile = np.full(len(heights), -np.inf)
    np.maximum.at(profile, bands[keep], x[keep])
    profile[np.isinf(profile)] = np.nan
    return profile


def _trochlea_top(heights: np.ndarray,
                  profile: np.ndarray) -> tuple[int, float, np.ndarray]:
    """The index of the height where the trochlea has faded into the shaft.

    A line is fitted to the anterior cortex over the shaft window, and the trochlea is
    what stands proud of it below. From its highest point the profile is followed up to
    the first height where it has fallen to a tenth of that (or a quarter of a
    millimetre, whichever is more) and stays there for five millimetres: that is the top
    of the trochlea. Bands with no bone are stepped over. Returns the index, the shaft
    line's slope from the component axis in degrees, and the excess over the line.
    """
    low, high = SHAFT_WINDOW_MM
    window = (heights >= low) & (heights <= high) & np.isfinite(profile)
    if int(window.sum()) < 10:
        raise ValueError("Too little anterior cortex in the shaft to fit its line.")
    slope, intercept = np.polyfit(heights[window], profile[window], 1)
    excess = profile - (intercept + slope * heights)

    below = np.isfinite(excess) & (heights < low)
    if not below.any() or float(np.max(excess[below])) < MIN_TROCHLEA_MM:
        raise ValueError("No trochlea stands proud of the anterior cortex, so the "
                         "anterior cut has nothing to be flush with.")
    peak = int(np.flatnonzero(below)[np.argmax(excess[below])])
    threshold = max(TROCHLEA_FLOOR_MM, TROCHLEA_FRACTION * float(excess[peak]))
    run = int(round(TROCHLEA_RUN_MM / PROFILE_STEP_MM))
    candidates = np.flatnonzero(np.isfinite(excess) & (heights > heights[peak]))
    for k, index in enumerate(candidates):
        ahead = candidates[k:k + run]
        if len(ahead) == run and bool(np.all(excess[ahead] <= threshold)):
            break
    else:
        raise ValueError("The trochlea does not fade into the shaft below the top of "
                         "the scan.")
    if heights[index] < MIN_FLANGE_HEIGHT_MM:
        raise ValueError("No trochlea stands proud of the anterior cortex, so the "
                         "anterior cut has nothing to be flush with.")
    return int(index), float(np.degrees(np.arctan(slope))), excess
