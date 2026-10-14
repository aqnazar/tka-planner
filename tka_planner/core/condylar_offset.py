"""Posterior condylar offset: how far each condyle stands behind the femoral shaft.

Bellemans et al. (2002) found it the strongest geometric predictor of the flexion a
cruciate-retaining knee reaches: each millimetre of offset lost cost about 6 degrees.
Their measure is radiographic -- the maximal thickness of the posterior condyle,
projected posteriorly from the tangent to the posterior cortex of the femoral shaft, on
a true lateral film. This module measures the same thing on the surface mesh, before
surgery from the bone and after it from the placed femoral component, and per
compartment, which a lateral film superimposes.

Definition
----------
All of it happens in one sagittal plane of the femur.

* **Shaft axis** ``a``: from ``femur.canal_centre_distal`` to
  ``femur.canal_centre_proximal`` -- the anatomical (diaphyseal) axis, not the
  mechanical one, so the femoral head is not needed.
* **Sagittal plane**: normal to the femoral frame's mediolateral axis (the
  transepicondylar axis) made perpendicular to ``a``. Projecting every point into it is
  what a true lateral radiograph does.
* **Posterior cortex line**: the posterior silhouette of the shaft between the two canal
  levels -- in 2 mm bands along ``a``, the most posterior surface point of each band,
  which is the outline a lateral film shows -- fitted with a straight line, and then
  moved posteriorly until it touches the silhouette without crossing it. That is the
  tangent. Its fit residual is reported, because a femur bowed over the fitted length
  makes "the" tangent ambiguous.
* **Offset** of a compartment: the largest distance, perpendicular to the tangent in the
  sagittal plane, from the tangent to any point of that condyle, positive posteriorly.
  For a mesh the maximum of a distance along a direction is reached at a vertex, so
  vertices are exact here. Compartments are split by the plane through the knee centre
  (``femur.notch_centre``) normal to the mediolateral axis; the side tells which half is
  medial. Only points distal of the fitted shaft segment count, so the cortex itself
  never does.
* **Planned offset**: the same distance, from the same tangent of the same femur, to the
  placed femoral component's posterior condyles -- the component as the plan poses and
  shapes it. The change is planned minus preoperative: negative means offset lost.

The radiographic single value is the larger of the two compartments, since the film
shows the outermost outline of both.

Provenance
----------
The metric uses no mechanical axis, so a knee-only scan does not demote it: the distal
shaft is in view on every scan in this cohort (147 to 248 mm of femur), as it is on a
knee radiograph. It is ``measured`` when the canal centres, the epicondyles and the
notch centre were picked by a person, and ``estimated`` otherwise, carrying the
automatic-landmark assumption like every other metric. It is ``not_computable`` without
the canal centres, or when the scan holds too little shaft to fit a line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import AnatomicalFrame
from .geometry import unit
from .landmarks import LandmarkSet, LandmarkStatus
from .provenance import AUTOMATIC_LANDMARK_ESTIMATE, Metric, Quality

__all__ = [
    "PCO_DEFINITION",
    "CortexLine",
    "posterior_cortex_line",
    "condylar_offsets",
    "posterior_condylar_offset_metrics",
    "planned_condylar_offset",
]

METHOD = "metrics.posterior_condylar_offset.v1"
BAND_MM = 2.0
MIN_BANDS = 8

PCO_DEFINITION = (
    "Posterior condylar offset (Bellemans 2002, per compartment): in the sagittal "
    "plane of the femur, the largest distance from the tangent to the posterior cortex "
    "of the femoral shaft to the posterior surface of the condyle, measured "
    "perpendicular to the tangent."
)
PCO_SIGN = (
    "Positive is posterior of the shaft tangent. A change below zero means offset lost, "
    "which Bellemans associated with about 6 degrees less flexion per millimetre."
)
INPUTS = ("femur.canal_centre_distal", "femur.canal_centre_proximal",
          "femur.epicondyle_lateral", "femur.notch_centre")


@dataclass(frozen=True)
class CortexLine:
    """The posterior cortex tangent, in the femur's sagittal plane.

    ``axial`` and ``posterior`` span the plane; the tangent is the set of points whose
    posterior coordinate is ``intercept + slope * axial`` (coordinates relative to
    ``origin``). ``ml`` is the plane's normal, ``medial_sign`` which way along it the
    medial compartment lies.
    """

    origin: np.ndarray
    axial: np.ndarray
    posterior: np.ndarray
    ml: np.ndarray
    medial_sign: float
    intercept: float
    slope: float
    diagnostics: dict = field(default_factory=dict)

    def distance(self, points) -> np.ndarray:
        """Signed distance of points behind the tangent, perpendicular to it, in the
        sagittal plane (mm, + posterior)."""
        rel = np.atleast_2d(np.asarray(points, dtype=float)) - self.origin
        s, r = rel @ self.axial, rel @ self.posterior
        return (r - self.intercept - self.slope * s) / np.hypot(1.0, self.slope)

    def is_medial(self, points) -> np.ndarray:
        rel = np.atleast_2d(np.asarray(points, dtype=float)) - self.origin
        return self.medial_sign * (rel @ self.ml) > 0.0


def posterior_cortex_line(
    vertices: np.ndarray,
    landmarks: LandmarkSet,
    frame: AnatomicalFrame,
    *,
    band_mm: float = BAND_MM,
) -> CortexLine:
    """Fit the posterior cortex tangent of the shaft. Raises ``ValueError`` when the
    landmarks or the shaft are not there to fit it."""
    missing = landmarks.missing_of(*INPUTS)
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    distal, proximal, knee = landmarks.require(
        "femur.canal_centre_distal", "femur.canal_centre_proximal", "femur.notch_centre")
    axial = unit(np.asarray(proximal, dtype=float) - np.asarray(distal, dtype=float))
    ml = unit(frame.y_patient_left - np.dot(frame.y_patient_left, axial) * axial)
    posterior = unit(np.cross(ml, axial))
    if float(np.dot(posterior, frame.posterior)) < 0.0:
        posterior = -posterior
    medial_sign = 1.0 if float(np.dot(frame.medial, ml)) > 0.0 else -1.0
    origin = np.asarray(knee, dtype=float)

    vertices = np.asarray(vertices, dtype=float)
    rel = vertices - origin
    s, r = rel @ axial, rel @ posterior
    low = float((np.asarray(distal) - origin) @ axial)
    high = float((np.asarray(proximal) - origin) @ axial)
    inside = (s >= low) & (s <= high)
    if not inside.any():
        raise ValueError("no shaft surface between the canal centres")
    bands = np.floor((s[inside] - low) / band_mm).astype(np.int64)
    order = np.lexsort((r[inside], bands))
    bands, ss, rr = bands[order], s[inside][order], r[inside][order]
    last = np.ones(len(bands), dtype=bool)
    last[:-1] = bands[1:] != bands[:-1]
    ss, rr = ss[last], rr[last]
    if len(ss) < MIN_BANDS:
        raise ValueError(f"only {len(ss)} bands of shaft to fit the cortex line")

    slope, intercept = np.polyfit(ss, rr, 1)
    residual = rr - (intercept + slope * ss)
    tangent = intercept + float(residual.max())
    return CortexLine(
        origin=origin, axial=axial, posterior=posterior, ml=ml,
        medial_sign=medial_sign, intercept=float(tangent), slope=float(slope),
        diagnostics={
            "shaft_start_mm": round(low, 3),
            "shaft_length_mm": round(high - low, 2),
            "bands": int(len(ss)),
            "fit_rms_mm": round(float(np.sqrt(np.mean(residual ** 2))), 3),
            "tangent_offset_from_fit_mm": round(float(residual.max()), 3),
            "tangent_angle_to_canal_axis_deg": round(
                float(np.degrees(np.arctan(slope))), 2),
        },
    )


def condylar_offsets(line: CortexLine, vertices: np.ndarray) -> dict:
    """The offset of each compartment, from the most posterior vertex of that half.

    Only points distal of the shaft segment the tangent was fitted on are considered,
    so the cortex itself never counts as a condyle.
    """
    vertices = np.asarray(vertices, dtype=float)
    distance = line.distance(vertices)
    distal = (vertices - line.origin) @ line.axial < line.diagnostics["shaft_start_mm"]
    medial = line.is_medial(vertices)
    out = {}
    for name, mask in (("medial", distal & medial), ("lateral", distal & ~medial)):
        if not mask.any():
            out[name] = None
            continue
        index = np.flatnonzero(mask)[int(np.argmax(distance[mask]))]
        out[name] = {"offset_mm": float(distance[index]),
                     "point_mm": vertices[index].copy()}
    return out


def _quality(landmarks: LandmarkSet):
    """Measured only if every point the tangent and the split read was picked. The
    medial epicondyle is whichever one the frame's rotational reference used."""
    used = INPUTS + tuple(
        i for i in ("femur.epicondyle_medial_sulcus",
                    "femur.epicondyle_medial_prominence")
        if landmarks.available(i))[:1]
    if any(landmarks.get(i).status is LandmarkStatus.ESTIMATED for i in used):
        return Quality.ESTIMATED, (AUTOMATIC_LANDMARK_ESTIMATE,)
    return Quality.MEASURED, ()


def posterior_condylar_offset_metrics(
    femur_vertices: np.ndarray,
    landmarks: LandmarkSet,
    frame: AnatomicalFrame,
) -> dict[str, Metric]:
    """The preoperative offset of each compartment, as metrics."""
    names = ("pco_medial_mm", "pco_lateral_mm")
    try:
        line = posterior_cortex_line(femur_vertices, landmarks, frame)
    except ValueError as error:
        return {name: Metric.unavailable(
            name, "mm", PCO_DEFINITION, missing=tuple(landmarks.missing_of(*INPUTS)),
            reason=f"the posterior cortex tangent cannot be fitted: {error}",
            would_require="the femoral canal centres and at least 16 mm of shaft in "
                          "the scan",
            method=METHOD) for name in names}
    offsets = condylar_offsets(line, femur_vertices)
    quality, assumptions = _quality(landmarks)
    metrics = {}
    for name, compartment in zip(names, ("medial", "lateral")):
        found = offsets[compartment]
        if found is None:
            metrics[name] = Metric.unavailable(
                name, "mm", PCO_DEFINITION, missing=(),
                reason=f"no {compartment} condyle surface distal of the knee centre",
                method=METHOD)
            continue
        metrics[name] = Metric(
            name=name, value=round(found["offset_mm"], 3), unit="mm",
            quality=quality, definition=PCO_DEFINITION, sign_convention=PCO_SIGN,
            method=METHOD, inputs=INPUTS, assumptions=assumptions,
            diagnostics={**line.diagnostics,
                         "condyle_point_mm": [round(float(v), 3)
                                              for v in found["point_mm"]]},
        )
    return metrics


def planned_condylar_offset(
    femur_vertices: np.ndarray,
    component_vertices: np.ndarray,
    landmarks: LandmarkSet,
    frame: AnatomicalFrame,
) -> dict:
    """Preoperative, planned and change per compartment, for a placed component.

    ``component_vertices`` are the femoral component's vertices in world coordinates, as
    the plan poses and shapes it. The tangent is the patient's, so the change is the
    component's posterior condyle against the bone's, along one line.
    """
    try:
        line = posterior_cortex_line(femur_vertices, landmarks, frame)
    except ValueError as error:
        return {"available": False, "reason": str(error), "method": METHOD}
    quality, assumptions = _quality(landmarks)
    before = condylar_offsets(line, femur_vertices)
    after = condylar_offsets(line, component_vertices)
    record = {"available": True, "method": METHOD, "definition": PCO_DEFINITION,
              "status": quality.value, "assumptions": [a.id for a in assumptions],
              "cortex_line": line.diagnostics}
    for compartment in ("medial", "lateral"):
        pre, post = before[compartment], after[compartment]
        record[compartment] = {
            "preoperative_mm": None if pre is None else round(pre["offset_mm"], 3),
            "planned_mm": None if post is None else round(post["offset_mm"], 3),
            "change_mm": (None if pre is None or post is None
                          else round(post["offset_mm"] - pre["offset_mm"], 3)),
        }
    values = [(record[c]["preoperative_mm"], record[c]["planned_mm"])
              for c in ("medial", "lateral")]
    if all(v[0] is not None and v[1] is not None for v in values):
        pre = max(v[0] for v in values)
        post = max(v[1] for v in values)
        record["radiographic"] = {"preoperative_mm": pre, "planned_mm": post,
                                  "change_mm": round(post - pre, 3)}
    return record
