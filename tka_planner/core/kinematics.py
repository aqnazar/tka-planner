"""The flexion trajectory (WP-D level D1) and the search for the flexion it allows.

Reduce mode has always flexed the knee about one fixed hinge, the transepicondylar axis
(level D0). A real knee does not: the femur also slides and turns on the tibia as it
flexes. How much, and which way, is the question this module refuses to settle with one
curve, because the two fluoroscopic studies of cruciate-retaining knees in deep flexion
show that the motion an implanted knee makes is not the native one, and that it varies
between knees by more than its mean (Bellemans et al. 2002: medial contact 1.97 mm
anterior, SD 4.46). So the trajectory is a parameterised family and the planner reports
the spread over named scenarios that bracket it, rather than one prediction.

The motion, relative to the femur
---------------------------------
At a flexion angle ``theta`` the tibial set is first moved in its own rest frame, then
flexed about the hinge::

    pose(theta) = hinge(theta) @ tibial_motion(theta)

``tibial_motion`` is a rigid motion of the tibia in the transverse plane of the tibial
frame, made of

* an **axial rotation** of the tibia about a line parallel to its long axis, through a
  pivot on the flexion axis whose mediolateral position is a parameter (``pivot_ml_mm``
  from the axis midpoint, positive medial). Positive is *internal* tibial rotation --
  the tibia's anterior turning towards medial -- which is the same thing as external
  rotation of the femur on the tibia, the sense of lateral rollback;
* an **anteroposterior translation** of the femur on the tibia, measured at the pivot,
  positive when the femur moves *anteriorly* (the "paradoxical anterior slide"). It may
  be either sign.

Both grow linearly with flexion, at the rate set by their value at
``reference_flexion_deg``, and continue at that rate beyond it. Applying the motion at
rest and then flexing makes the rotation about the tibia's own long axis wherever
flexion has carried it, which is the joint-coordinate-system order (Grood and Suntay
1983): flexion about a femoral axis, rotation about a tibial one.

A pivot and two motions describe a rigid transverse motion completely; the AP translation
of each compartment follows from them. A point ``s`` mm medial of the pivot moves
anteriorly on the tibia by ``t cos(phi) + s sin(phi)``, so a medial pivot (``t = 0``,
pivot at the medial compartment) leaves the medial condyle in place and rolls the
lateral one back. :meth:`Trajectory.compartment_translation_mm` reports it.

The neutral trajectory has no rotation and no translation, and composes to exactly the
hinge Reduce mode has always used: the test suite checks that the matrices are equal,
not merely close.

What the published numbers are, and are not
--------------------------------------------
Both papers estimate each condyle's contact as its lowest point relative to the tibial
baseplate, so their millimetres are contact-point migration, not rigid-body translation.
Here they are applied as rigid translation, which is the D1 approximation: a prescribed
motion, not one solved from contact (that would be D2). Both studies measured a
weight-bearing activity; Reduce mode is a passive preview with no soft tissue.

The maximum-flexion search
--------------------------
:func:`search_max_flexion` flexes the knee in coarse steps until some pair of parts
first runs into each other, then bisects between the last clear angle and the first
blocked one down to the requested resolution. It knows nothing of meshes: it is handed a
function returning the contacts at an angle, so it is tested here on a toy geometry with
a known answer and driven by the real collision check in
:func:`tka_planner.pipeline.flexion_case`. A coarse sweep can step over a contact that
comes and goes within one step; the step is part of the result so a reader can see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .geometry import unit

__all__ = [
    "Trajectory",
    "SCENARIOS",
    "NEUTRAL",
    "scenario",
    "tibial_motion",
    "Contact",
    "FlexionLimit",
    "search_max_flexion",
    "envelope",
    "with_parameters",
]


@dataclass(frozen=True)
class Trajectory:
    """One prescribed flexion trajectory: the parameters a plan records.

    ``status`` says how far to trust the numbers: ``"model"`` for the neutral hinge,
    ``"literature"`` for values taken from a published mean, and ``"placeholder"`` for
    values that stand in until they are fitted to data. A placeholder is never to be
    reported as a finding.
    """

    name: str
    label: str
    axial_rotation_deg: float = 0.0    # internal tibial rotation at the reference angle
    pivot_ml_mm: float = 0.0           # pivot from the flexion-axis midpoint, + medial
    ap_translation_mm: float = 0.0     # femur on tibia at the pivot, + anterior
    reference_flexion_deg: float = 90.0
    status: str = "model"
    source: str = ""

    def __post_init__(self):
        if self.reference_flexion_deg <= 0.0:
            raise ValueError("The reference flexion angle must be positive.")
        if self.status not in ("model", "literature", "placeholder", "custom"):
            raise ValueError(f"Unknown trajectory status {self.status!r}.")

    @property
    def is_neutral(self) -> bool:
        return self.axial_rotation_deg == 0.0 and self.ap_translation_mm == 0.0

    def at(self, flexion_deg: float) -> tuple[float, float]:
        """``(internal rotation deg, femoral AP translation mm)`` at an angle."""
        fraction = float(flexion_deg) / self.reference_flexion_deg
        return self.axial_rotation_deg * fraction, self.ap_translation_mm * fraction

    def compartment_translation_mm(self, flexion_deg: float, ml_mm: float) -> float:
        """How far the femur has moved anteriorly on the tibia at a point ``ml_mm``
        medial of the flexion-axis midpoint (negative: lateral), from extension.

        This is what the fluoroscopy papers report per compartment, and what this
        parameterisation derives rather than takes as input.
        """
        rotation, translation = self.at(flexion_deg)
        phi = np.radians(rotation)
        offset = float(ml_mm) - self.pivot_ml_mm
        return float(translation * np.cos(phi) + offset * np.sin(phi))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "axial_rotation_deg": self.axial_rotation_deg,
            "pivot_ml_mm": self.pivot_ml_mm,
            "ap_translation_mm": self.ap_translation_mm,
            "reference_flexion_deg": self.reference_flexion_deg,
            "status": self.status,
            "source": self.source,
            "sign_convention": (
                "axial rotation + internal tibial rotation (femoral external rotation "
                "on the tibia); pivot + medial of the flexion-axis midpoint; AP "
                "translation + femur anterior on the tibia, at the pivot; both linear "
                "in flexion at the rate set by the reference angle"
            ),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "Trajectory":
        keys = ("name", "label", "axial_rotation_deg", "pivot_ml_mm",
                "ap_translation_mm", "reference_flexion_deg", "status", "source")
        return cls(**{k: record[k] for k in keys if k in record})


NEUTRAL = Trajectory(
    name="S3",
    label="Neutral: pure hinge",
    status="model",
    source="Level D0: rigid flexion about the transepicondylar axis, as Reduce mode "
           "has always done. No translation, no axial rotation.",
)

# Bellemans et al. 2002 (J Bone Joint Surg Br 84-B:50-53), 29 Profix CR knees, deep
# squat: medial contact 1.97 mm anterior (SD 4.46), lateral 0.27 mm anterior (SD 5.46),
# tibial internal rotation 2.7 deg (SD 7.9), extension to maximum flexion. The pivot is
# put at the joint centre and the translation there is the mean of the two
# compartments; with the reported rotation that gives, on a knee whose condyles sit
# 22 mm either side of the centre, about 2.2 mm medially and 0.1 mm laterally -- within
# 0.2 mm of the published means, which were measured independently of the rotation.
OBSERVED_CR = Trajectory(
    name="S2",
    label="Observed CR: paradoxical anterior slide",
    axial_rotation_deg=2.7,
    pivot_ml_mm=0.0,
    ap_translation_mm=(1.97 + 0.27) / 2.0,
    reference_flexion_deg=109.0,
    status="literature",
    source="Bellemans et al. 2002, JBJS-Br 84-B:50-53 (means; contact-point migration "
           "applied as rigid translation). Reference angle 109 deg: the mean "
           "weight-bearing maximum flexion of CR fixed-bearing knees in Banks et al. "
           "2003 (Clin Orthop 410:131-138), standing in for Bellemans' own mean "
           "maximum flexion, which is to be checked against the paper.",
)

# PLACEHOLDER. A medial pivot with lateral rollback, the native-knee pattern Bellemans
# cites (Iwaki, Pinskerova and Freeman 2000). The numbers are not fitted to anything:
# they give a medial condyle that stays put and a lateral one that rolls back about
# 15 mm by 120 degrees, the order of magnitude that literature describes. They are to
# be replaced by a least-squares fit to a chosen dataset (developer question X-11).
NATIVE_LIKE = Trajectory(
    name="S1",
    label="Native-like: medial pivot, lateral rollback (PLACEHOLDER)",
    axial_rotation_deg=20.0,
    pivot_ml_mm=20.0,
    ap_translation_mm=0.0,
    reference_flexion_deg=120.0,
    status="placeholder",
    source="PLACEHOLDER, to be fitted: medial pivot about 20 mm medial of the centre, "
           "20 deg internal tibial rotation by 120 deg (lateral rollback about 15 mm). "
           "Candidate source: Iwaki, Pinskerova and Freeman 2000.",
)

SCENARIOS: dict[str, Trajectory] = {t.name: t for t in (NATIVE_LIKE, OBSERVED_CR, NEUTRAL)}


def scenario(name_or_trajectory) -> Trajectory:
    """A named scenario, or a trajectory passed through unchanged."""
    if isinstance(name_or_trajectory, Trajectory):
        return name_or_trajectory
    if isinstance(name_or_trajectory, dict):
        return Trajectory.from_dict(name_or_trajectory)
    try:
        return SCENARIOS[str(name_or_trajectory)]
    except KeyError:
        raise ValueError(f"Unknown trajectory {name_or_trajectory!r}; expected one of "
                         f"{', '.join(SCENARIOS)}.") from None


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = unit(axis)
    angle = np.radians(float(degrees))
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def tibial_motion(
    trajectory: Trajectory,
    flexion_deg: float,
    *,
    centre,
    anterior,
    medial,
) -> np.ndarray:
    """The tibia's rigid motion in its rest frame at a flexion angle, as a 4x4.

    ``centre`` is the flexion-axis midpoint; ``anterior`` and ``medial`` are the
    tibial frame's directions (the frame's own, so no side logic is needed here: the
    rotation axis ``anterior x medial`` points proximally on a right knee and distally
    on a left one, which is exactly what makes "internal" mean the same on both). The
    rotation is applied first, about the line through the pivot, then the tibia slides
    posteriorly by the femur's anterior translation.

    The neutral trajectory returns the identity exactly.
    """
    if trajectory.is_neutral:
        return np.eye(4)
    anterior = unit(np.asarray(anterior, dtype=float))
    medial = unit(np.asarray(medial, dtype=float))
    rotation_deg, translation_mm = trajectory.at(flexion_deg)
    pivot = np.asarray(centre, dtype=float) + trajectory.pivot_ml_mm * medial

    turn = np.eye(4)
    turn[:3, :3] = _rotation(np.cross(anterior, medial), rotation_deg)
    turn[:3, 3] = pivot - turn[:3, :3] @ pivot
    slide = np.eye(4)
    slide[:3, 3] = -translation_mm * anterior
    return slide @ turn


# ----------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Contact:
    """Two parts overlapping at one flexion angle.

    ``kind`` is ``"impingement"`` when the overlap stops flexion and ``"articulation"``
    when it is the intended bearing contact, which the collision check decides and
    states (see :func:`tka_planner.pipeline.flexion_case`). ``location`` is the
    overlap's centre in the femoral frame: (anterior, patient-left, proximal) mm from
    the knee centre.
    """

    pair: str
    flexion_deg: float
    volume_mm3: float
    depth_mm: float
    location: np.ndarray
    kind: str = "impingement"
    region: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.kind == "impingement"

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "flexion_deg": round(float(self.flexion_deg), 3),
            "kind": self.kind,
            "region": self.region,
            "volume_mm3": round(float(self.volume_mm3), 3),
            "depth_mm": round(float(self.depth_mm), 3),
            "location_femoral_mm": [round(float(v), 2) for v in self.location],
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class FlexionLimit:
    """The flexion a plan reaches before two parts first run into each other.

    ``max_flexion_deg`` is the last angle found clear; the contact that stops it was
    found at most ``resolution_deg`` later. ``censored`` means nothing was hit by
    ``stop_deg``, so the limit is at least that.

    Pairs already blocked at the starting angle cannot limit flexion they never
    allowed, so a plan problem at extension is reported (``blocked_at_start``) rather
    than read as a flexion limit of zero. Such a pair is set aside only until the sweep
    first finds it clear; from then on a new contact of that pair counts like any
    other. ``never_cleared`` names the pairs that stayed in contact throughout, which
    therefore never took part in the search.
    """

    trajectory: str
    max_flexion_deg: float
    censored: bool
    contacts: tuple[Contact, ...]
    start_deg: float
    stop_deg: float
    step_deg: float
    resolution_deg: float
    blocked_at_start: tuple[Contact, ...] = ()
    never_cleared: tuple[str, ...] = ()
    evaluations: int = 0

    @property
    def first_pair(self) -> str | None:
        """The pair that stops flexion; the deepest overlap if several do at once."""
        if not self.contacts:
            return None
        return max(self.contacts, key=lambda c: c.volume_mm3).pair

    def to_dict(self) -> dict:
        return {
            "trajectory": self.trajectory,
            "max_flexion_deg": round(float(self.max_flexion_deg), 3),
            "censored": self.censored,
            "first_pair": self.first_pair,
            "contacts": [c.to_dict() for c in self.contacts],
            "blocked_at_start": [c.to_dict() for c in self.blocked_at_start],
            "never_cleared": list(self.never_cleared),
            "search": {"start_deg": self.start_deg, "stop_deg": self.stop_deg,
                       "step_deg": self.step_deg, "resolution_deg": self.resolution_deg,
                       "evaluations": self.evaluations},
        }


def search_max_flexion(
    contacts_at: Callable[[float], list],
    *,
    trajectory: str = "",
    start_deg: float = 0.0,
    stop_deg: float = 170.0,
    step_deg: float = 5.0,
    resolution_deg: float = 0.5,
) -> FlexionLimit:
    """Flex in ``step_deg`` steps until a blocking contact, then bisect to
    ``resolution_deg``.

    ``contacts_at(angle)`` returns the :class:`Contact` list at an angle; only those
    whose :attr:`Contact.blocks` is true stop flexion.
    """
    if step_deg <= 0.0 or resolution_deg <= 0.0:
        raise ValueError("The step and the resolution must be positive.")
    evaluations = 0

    def blocking(angle):
        nonlocal evaluations
        evaluations += 1
        return [c for c in contacts_at(float(angle)) if c.blocks]

    at_start = tuple(blocking(start_deg))
    ignored = {c.pair for c in at_start}

    clear = float(start_deg)
    angles = np.arange(start_deg + step_deg, stop_deg + 1e-9, step_deg)
    for angle in angles:
        found_here = blocking(angle)
        # A pair set aside at extension is taken back once it has been seen clear.
        ignored &= {c.pair for c in found_here}
        hits = [c for c in found_here if c.pair not in ignored]
        if not hits:
            clear = float(angle)
            continue
        armed_ignore = frozenset(ignored)
        low, high, found = clear, float(angle), hits
        while high - low > resolution_deg:
            middle = (low + high) / 2.0
            middle_hits = [c for c in blocking(middle) if c.pair not in armed_ignore]
            if middle_hits:
                high, found = middle, middle_hits
            else:
                low = middle
        return FlexionLimit(
            trajectory=trajectory, max_flexion_deg=low, censored=False,
            contacts=tuple(found), start_deg=start_deg, stop_deg=stop_deg,
            step_deg=step_deg, resolution_deg=resolution_deg,
            blocked_at_start=at_start, never_cleared=tuple(sorted(ignored)),
            evaluations=evaluations,
        )
    return FlexionLimit(
        trajectory=trajectory, max_flexion_deg=clear, censored=True, contacts=(),
        start_deg=start_deg, stop_deg=stop_deg, step_deg=step_deg,
        resolution_deg=resolution_deg, blocked_at_start=at_start,
        never_cleared=tuple(sorted(ignored)), evaluations=evaluations,
    )


def envelope(limits: dict) -> dict:
    """The spread of maximum flexion over scenarios: the honest output of D1.

    A single scenario is not a prediction for one knee (the published dispersion exceeds
    the mean), so the planner reports the range the scenarios bracket.
    """
    values = {name: limit.max_flexion_deg for name, limit in limits.items()}
    if not values:
        return {"min_deg": None, "max_deg": None, "spread_deg": None}
    low = min(values, key=values.get)
    high = max(values, key=values.get)
    return {
        "min_deg": round(float(values[low]), 3),
        "min_trajectory": low,
        "max_deg": round(float(values[high]), 3),
        "max_trajectory": high,
        "spread_deg": round(float(values[high] - values[low]), 3),
        "any_censored": any(limit.censored for limit in limits.values()),
    }


def with_parameters(trajectory: Trajectory, **changes) -> Trajectory:
    """A copy of a trajectory with some parameters changed, marked ``custom``."""
    return replace(trajectory, status="custom", **changes)
