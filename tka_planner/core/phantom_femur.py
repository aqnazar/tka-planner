"""A phantom femur: the rest of the bone, drawn from the entered femur length.

A knee-only scan holds the distal femur and no more, so the hip is never measured. The
planner asks for the femur's length (hip centre to knee centre) and, with it, can draw
what the scan would have shown had it been of the whole femur: a shaft, a neck and a
head sphere, sized from the length by published relations and placed on the femoral
frame's own mechanical axis.

It is a schematic, not an anatomy. Everything beyond the scan is a population prior, so
the phantom says so in every document it appears in: the quality tier is ``assumed``,
each constant carries its source and whether the source was read, and the hip-centre
error that follows from the priors is stated (about 11 mm, one standard deviation, for a
410 mm femur). Nothing downstream may treat it as a measurement, and it changes no
metric: the frame's mechanical axis is the planner's own, and the phantom only places a
head on it at the entered length.

What it is built from (literature basis in ``docs/Prep for paper``, all constants below
are named there with their status):

* the head centre sits at the entered length ``L`` along the frame's mechanical axis;
* the anatomical (shaft) axis leans laterally from the mechanical one by the
  anatomical-mechanical angle, 6.0 deg (5.7 +- 1.2 Jabalameli 2015; 6.46 +- 1.26
  Jingjit 2014; 6.0-6.2 Oguz 1996);
* the neck joins the shaft at the neck-shaft angle, 123.2 deg (3D CT, 247 patients,
  Yang and Qian 2025), so its length follows from the offset and the angle and not
  from a second source;
* the head diameter is the sex-specific mean when the sex is known (male 49.65, female
  44.41 mm, Yang and Qian 2025; 49 and 44 in 170 dry femora, Mall 2000), and otherwise
  0.100 of the femur's maximum length (49/464 male, 44/434 female, Mall 2000), the
  maximum length being the entered one plus 40 mm. That 40 mm is this project's own
  estimate (head radius plus the knee centre's height above the condyles): no
  published conversion between hip-to-knee length and maximum length was found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import AnatomicalFrame
from .geometry import unit

__all__ = [
    "PHANTOM_METHOD", "PhantomFemur", "build_phantom_femur", "head_diameter_mm",
    "hip_centre_error_budget_mm",
]

PHANTOM_METHOD = "phantom_femur.v1"

#: Anatomical-mechanical angle of the femur, degrees. Measured series: 5.7 +- 1.2
#: (Jabalameli, Acta Med Iran 2015; PMID 26024704), 6.46 +- 1.26 (Jingjit, J Med Assoc
#: Thai 2014; PMID 25764640), 6.24 and 5.96 (Oguz, Surg Radiol Anat 1996; PMID 8685808).
AMA_DEG = 6.0
AMA_SD_DEG = 1.26
#: Neck-shaft angle, 3D CT, 247 patients (Yang and Qian, Ther Clin Risk Manag 2025;
#: PMID 40547822). Radiographic and dry-bone "inclination" is about 129 deg (Toogood,
#: Clin Orthop 2009; Boese, Skeletal Radiol 2016): a definition, not a disagreement.
NECK_SHAFT_ANGLE_DEG = 123.24
#: Anteversion, same source; it is not drawn (the neck lies in the coronal plane) and
#: is carried only in the error budget.
ANTEVERSION_SD_DEG = 7.47
#: Head diameter by sex, mm (Yang and Qian 2025; Mall, Forensic Sci Int 2000 agrees).
HEAD_DIAMETER_BY_SEX_MM = {"male": 49.65, "female": 44.41}
HEAD_DIAMETER_REFERENCE_MM = 46.17
#: Head diameter over maximum femur length (Mall 2000: 49/464, 44/434).
HEAD_TO_MAXIMUM_LENGTH = 0.100
HEAD_DIAMETER_PAIRED_SD_MM = 3.9
#: This project's estimate, not a published value: maximum length minus hip-to-knee
#: length. To be calibrated on full-femur scans before the paper prints it.
MAXIMUM_MINUS_MECHANICAL_MM = 40.0
MAXIMUM_MINUS_MECHANICAL_SD_MM = 6.0
#: How far the tip of the greater trochanter stands above the head centre (Yi, Surg
#: Radiol Anat 2019; PMID 30488299).
TROCHANTER_ABOVE_HEAD_MM = 7.0
#: Drawing proportions (not data): shaft and neck radius as a fraction of head diameter.
SHAFT_RADIUS_PER_HEAD = 0.30
NECK_RADIUS_PER_HEAD = 0.32
#: The shaft is drawn from this far above the knee centre, beyond a distal scan.
SHAFT_STARTS_ABOVE_KNEE_MM = 80.0
#: Plausibility check of the entered length against the width: FL = 187.8 + 3.39 EB,
#: SEE 16.4 mm (Abledu, Anat Cell Biol 2016; PMID 27722014, 50 dry femora, maximum
#: length). Informational: the planner's ML is not the dry-bone epicondylar breadth.
LENGTH_FROM_WIDTH = (187.8, 3.39, 16.4)
LENGTH_CHECK_SEE_MULTIPLE = 2.0
#: A length mis-read by this much is assumed, for the budget (the entered length is a
#: reading off a film or scanogram).
ENTERED_LENGTH_SD_MM = 5.0


@dataclass(frozen=True)
class PhantomFemur:
    """The schematic's parts, in the scan's patient (LPS) frame, millimetres."""

    side: str
    length_mm: float
    sex: str | None
    head_centre: np.ndarray
    head_diameter_mm: float
    neck_foot: np.ndarray
    neck_radius_mm: float
    shaft_start: np.ndarray
    shaft_end: np.ndarray
    shaft_radius_mm: float
    mechanical_axis: np.ndarray
    anatomical_axis: np.ndarray
    offset_mm: float
    neck_length_mm: float
    budget: dict
    length_check: dict
    assumptions: tuple[str, ...]
    method: str = PHANTOM_METHOD
    quality: str = "assumed"
    not_modelled: tuple[str, ...] = field(default=(
        "anteversion (the neck is drawn in the coronal plane)",
        "the greater and lesser trochanters, the shaft's bow and its flare",
        "any patient-specific varus or valgus: the axis is a population prior",
    ))

    def to_dict(self) -> dict:
        def point(p):
            return [round(float(c), 2) for c in p]

        return {
            "method": self.method,
            "quality": self.quality,
            "side": self.side,
            "schematic": True,
            "note": ("A schematic femur drawn from the entered length and population "
                     "priors. It is not measured anatomy and cannot support alignment "
                     "decisions."),
            "entered_length_mm": round(self.length_mm, 1),
            "sex": self.sex,
            "head": {"centre_mm": point(self.head_centre),
                     "diameter_mm": round(self.head_diameter_mm, 1)},
            "neck": {"from_head_centre_mm": point(self.head_centre),
                     "to_shaft_mm": point(self.neck_foot),
                     "radius_mm": round(self.neck_radius_mm, 1),
                     "length_mm": round(self.neck_length_mm, 1),
                     "neck_shaft_angle_deg": NECK_SHAFT_ANGLE_DEG},
            "shaft": {"from_mm": point(self.shaft_start), "to_mm": point(self.shaft_end),
                      "radius_mm": round(self.shaft_radius_mm, 1),
                      "offset_mm": round(self.offset_mm, 1)},
            "axes": {"mechanical": [round(float(c), 5) for c in self.mechanical_axis],
                     "anatomical": [round(float(c), 5) for c in self.anatomical_axis],
                     "ama_deg": AMA_DEG},
            "error_budget_mm": self.budget,
            "length_check": self.length_check,
            "assumptions": list(self.assumptions),
            "not_modelled": list(self.not_modelled),
        }


def head_diameter_mm(length_mm: float, sex: str | None = None) -> tuple[float, float]:
    """The head's diameter and its one-sigma error, from the entered length or the sex."""
    if sex is not None:
        if sex not in HEAD_DIAMETER_BY_SEX_MM:
            raise ValueError(f"sex must be 'male', 'female' or None, not {sex!r}.")
        return HEAD_DIAMETER_BY_SEX_MM[sex], 2.4
    maximum = length_mm + MAXIMUM_MINUS_MECHANICAL_MM
    return HEAD_TO_MAXIMUM_LENGTH * maximum, HEAD_DIAMETER_PAIRED_SD_MM


def hip_centre_error_budget_mm(length_mm: float) -> dict:
    """One-sigma error of the head centre that the priors leave, by direction.

    Medial-lateral: the anatomical-mechanical angle's spread over the length. Anterior-
    posterior: the offset's lever through the anteversion's spread. Along the axis: the
    entered length's own reading error. The error in carrying the shaft axis up from a
    distal scan is not published and is not in these.
    """
    offset = length_mm * np.sin(np.radians(AMA_DEG))
    medial_lateral = length_mm * np.radians(AMA_SD_DEG)
    anterior_posterior = float(offset * np.radians(ANTEVERSION_SD_DEG))
    along = ENTERED_LENGTH_SD_MM
    return {
        "medial_lateral": round(float(medial_lateral), 1),
        "anterior_posterior": round(anterior_posterior, 1),
        "along_axis": along,
        "combined": round(float(np.sqrt(medial_lateral ** 2 + anterior_posterior ** 2
                                        + along ** 2)), 1),
        "basis": "derived from published spreads of the angles; excludes the error of "
                 "carrying the shaft axis up from a distal-only scan",
    }


def build_phantom_femur(frame: AnatomicalFrame, length_mm: float, *,
                        width_mm: float | None = None,
                        sex: str | None = None) -> PhantomFemur:
    """Draw the femur's missing part on a femoral frame, at the entered length.

    ``frame`` is the femoral frame (its origin the knee centre, ``z_proximal`` the
    mechanical axis the planner uses, measured or assumed). ``width_mm`` is the femoral
    mediolateral width, used only to check the entered length against it.
    """
    if frame.bone != "femur":
        raise ValueError(f"A phantom femur needs the femoral frame, not {frame.bone}.")
    if not (np.isfinite(length_mm) and length_mm > 0):
        raise ValueError("The femur length must be a positive number of millimetres.")

    origin = np.asarray(frame.origin, dtype=float)
    z = unit(np.asarray(frame.z_proximal, dtype=float))
    lateral = unit(np.asarray(frame.lateral, dtype=float))
    ama = np.radians(AMA_DEG)
    # The shaft leans laterally as it rises: the head sits medial to the shaft's axis.
    anatomical = unit(np.cos(ama) * z + np.sin(ama) * lateral)

    head_centre = origin + length_mm * z
    diameter, head_sd = head_diameter_mm(length_mm, sex)

    # Foot of the head's perpendicular on the shaft axis, then back down the shaft by
    # what the neck-shaft angle asks, so the neck meets the shaft at that angle.
    along = float(np.dot(head_centre - origin, anatomical))
    foot_level = origin + along * anatomical
    offset = float(np.linalg.norm(head_centre - foot_level))
    drop = offset / np.tan(np.radians(180.0 - NECK_SHAFT_ANGLE_DEG))
    neck_foot = foot_level - drop * anatomical
    neck_length = float(np.linalg.norm(head_centre - neck_foot))

    shaft_start = origin + SHAFT_STARTS_ABOVE_KNEE_MM * anatomical
    shaft_end = foot_level + TROCHANTER_ABOVE_HEAD_MM * anatomical

    check: dict = {"available": False}
    if width_mm is not None:
        intercept, slope, see = LENGTH_FROM_WIDTH
        predicted_maximum = intercept + slope * float(width_mm)
        maximum = length_mm + MAXIMUM_MINUS_MECHANICAL_MM
        gap = maximum - predicted_maximum
        check = {
            "available": True,
            "entered_maximum_length_mm": round(maximum, 1),
            "predicted_from_width_mm": round(predicted_maximum, 1),
            "difference_mm": round(gap, 1),
            "limit_mm": round(LENGTH_CHECK_SEE_MULTIPLE * see, 1),
            "implausible": bool(abs(gap) > LENGTH_CHECK_SEE_MULTIPLE * see),
            "note": "plausibility only: width predicts length to about 16 mm (1 SEE)",
        }

    budget = hip_centre_error_budget_mm(length_mm)
    budget["head_diameter_sd"] = head_sd
    assumptions = (
        f"the head centre lies {length_mm:g} mm from the knee centre along the frame's "
        f"mechanical axis (entered, not measured)",
        f"anatomical-mechanical angle {AMA_DEG:g} deg, sd {AMA_SD_DEG:g} (measured "
        f"radiographic series, PMID 26024704, 25764640, 8685808)",
        f"neck-shaft angle {NECK_SHAFT_ANGLE_DEG:g} deg, 3D CT (PMID 40547822)",
        ("head diameter from the stated sex (PMID 40547822, 10978643)" if sex else
         f"head diameter {HEAD_TO_MAXIMUM_LENGTH:g} of the maximum femur length, which "
         f"is the entered length plus {MAXIMUM_MINUS_MECHANICAL_MM:g} mm (this "
         f"project's own estimate, to be calibrated; ratio from PMID 10978643)"),
        f"greater trochanter {TROCHANTER_ABOVE_HEAD_MM:g} mm above the head centre "
        f"(PMID 30488299)",
        "shaft and neck radii are drawing proportions of the head diameter, not data",
    )
    return PhantomFemur(
        side=str(frame.side), length_mm=float(length_mm), sex=sex,
        head_centre=head_centre, head_diameter_mm=float(diameter),
        neck_foot=neck_foot, neck_radius_mm=float(NECK_RADIUS_PER_HEAD * diameter),
        shaft_start=shaft_start, shaft_end=shaft_end,
        shaft_radius_mm=float(SHAFT_RADIUS_PER_HEAD * diameter),
        mechanical_axis=z, anatomical_axis=anatomical, offset_mm=offset,
        neck_length_mm=neck_length, budget=budget, length_check=check,
        assumptions=assumptions,
    )
