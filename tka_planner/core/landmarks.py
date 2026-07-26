"""Anatomical landmarks: the registry, the schema, and Slicer interchange.

Landmarks are the input from which every frame and every metric is derived, so this
module carries more of the system's meaning than its size suggests.

Three ideas shape it.

**The registry is the specification.** Each landmark has one authoritative definition
that serves simultaneously as the picking instruction shown in the add-on, the glossary
entry in the report, and the documentation of what a published number actually measured.
A landmark defined in three places drifts in three directions; here there is one.

**Absent is not one thing.** A landmark can be missing because the anatomy lies outside
the scan (``OUT_OF_SCAN``) or because nobody has picked it yet (``NOT_PICKED``). The
first is a permanent property of the data that the report must state; the second is an
incomplete workflow. Collapsing them into a null is how planning software ends up
silently substituting a population average for a measurement. Every knee-only CT in this
cohort has the femoral head marked ``OUT_OF_SCAN``, and that fact propagates all the way
into the report.

**Coordinate systems are declared, never inferred.** STL records no coordinate system,
and Slicer will happily write markups in either LPS or RAS. Mixing them mirrors the
anatomy left-for-right, which produces a plan that is wrong in a way that looks entirely
plausible. So the frame is mandatory on load and cross-checked against the mesh by
:mod:`tka_planner.core.qc`.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .sides import Side

__all__ = [
    "CoordinateSystem",
    "LandmarkStatus",
    "LandmarkDefinition",
    "Landmark",
    "LandmarkSet",
    "REGISTRY",
    "definitions_for_bone",
    "read_slicer_fcsv",
    "read_slicer_markups",
    "read_landmark_set",
    "write_landmark_set",
    "load_landmarks",
]

SCHEMA_VERSION = "tka-landmarks/1.0.0"


# ----------------------------------------------------------------------
# Coordinate systems
# ----------------------------------------------------------------------


class CoordinateSystem(Enum):
    """Medical image coordinate conventions.

    DICOM is natively LPS, and 3D Slicer exports both models and markups in LPS by
    default while displaying RAS internally. The two differ by a 180-degree rotation
    about the superior axis -- equivalently, negating the first two coordinates.
    """

    LPS = "LPS"  # +X patient-left,  +Y posterior, +Z superior  (DICOM native)
    RAS = "RAS"  # +X patient-right, +Y anterior,  +Z superior

    @classmethod
    def parse(cls, value: "str | int | CoordinateSystem") -> "CoordinateSystem":
        """Accept the spellings Slicer uses, including its legacy numeric codes."""
        if isinstance(value, cls):
            return value
        # Older .fcsv files write 0 for RAS and 1 for LPS.
        if isinstance(value, int) or (isinstance(value, str) and value.strip() in "01"
                                      and value.strip() != ""):
            code = int(value)
            if code == 0:
                return cls.RAS
            if code == 1:
                return cls.LPS
            raise ValueError(f"Unknown numeric coordinate system code: {code}")
        if isinstance(value, str):
            token = value.strip().upper()
            if token in ("LPS", "RAS"):
                return cls[token]
        raise ValueError(
            f"Unrecognised coordinate system {value!r}. Expected 'LPS' or 'RAS'."
        )

    def convert_to(
        self, points: np.ndarray, target: "CoordinateSystem"
    ) -> np.ndarray:
        """Convert coordinates between LPS and RAS.

        The transform is its own inverse: negate X and Y, leave Z. Applying it when it
        was not needed is exactly as damaging as omitting it when it was, which is why
        the source frame must be declared rather than guessed.
        """
        points = np.asarray(points, dtype=float)
        if self is target:
            return points.copy()
        return points * np.array([-1.0, -1.0, 1.0])


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------


class LandmarkStatus(Enum):
    """Why a landmark does or does not have a position.

    The distinction between :attr:`OUT_OF_SCAN` and :attr:`NOT_PICKED` drives the whole
    capability model: the first says a metric can never be computed from this data, the
    second says the workflow is unfinished. A report that cannot tell them apart cannot
    be honest about its own limitations.
    """

    PRESENT = "present"          # picked by a human on visible anatomy
    DERIVED = "derived"          # computed from other landmarks or mesh geometry
    ESTIMATED = "estimated"      # produced by an automatic estimator, needs review
    OUT_OF_SCAN = "out_of_scan"  # the anatomy is outside the imaged volume
    NOT_PICKED = "not_picked"    # simply not done yet
    REJECTED = "rejected"        # picked, then failed a quality check

    @property
    def has_position(self) -> bool:
        return self in (
            LandmarkStatus.PRESENT,
            LandmarkStatus.DERIVED,
            LandmarkStatus.ESTIMATED,
        )

    @property
    def is_permanently_unavailable(self) -> bool:
        """True when no amount of further work on *this* scan will supply the point."""
        return self is LandmarkStatus.OUT_OF_SCAN


# ----------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LandmarkDefinition:
    """The authoritative definition of one landmark.

    ``definition`` is deliberately prose: it is shown verbatim to whoever is picking and
    reproduced in the report glossary, so a reader of the paper can see precisely what
    was measured rather than inferring it from a variable name.
    """

    id: str
    bone: str
    display_name: str
    definition: str
    feeds: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    observer_sd_mm: float | None = None
    typically_out_of_scan: bool = False
    picked_in_increment_1: bool = True


def _define(*definitions: LandmarkDefinition) -> dict[str, LandmarkDefinition]:
    registry: dict[str, LandmarkDefinition] = {}
    for definition in definitions:
        if definition.id in registry:
            raise ValueError(f"Duplicate landmark id in registry: {definition.id}")
        registry[definition.id] = definition
    return registry


REGISTRY: dict[str, LandmarkDefinition] = _define(
    # ---------------- Femur ----------------
    LandmarkDefinition(
        id="femur.epicondyle_lateral",
        bone="femur",
        display_name="Lateral epicondyle",
        definition=(
            "The most prominent point of the lateral epicondyle. Shared by the "
            "surgical and anatomical transepicondylar axes."
        ),
        feeds=("stea", "atea", "condylar_twist_angle"),
        aliases=("lateral_epicondyle", "LE", "epicondyle_lat"),
        observer_sd_mm=1.7,
    ),
    LandmarkDefinition(
        id="femur.epicondyle_medial_sulcus",
        bone="femur",
        display_name="Medial epicondylar sulcus",
        definition=(
            "The deepest point of the sulcus on the medial epicondyle -- the groove, "
            "not the surrounding peak. Medial end of the SURGICAL transepicondylar "
            "axis (sTEA)."
        ),
        feeds=("stea", "condylar_twist_angle"),
        aliases=("medial_sulcus", "MS", "sulcus"),
        observer_sd_mm=2.4,  # the hardest reliable pick on the distal femur
    ),
    LandmarkDefinition(
        id="femur.epicondyle_medial_prominence",
        bone="femur",
        display_name="Medial epicondylar prominence",
        definition=(
            "The most prominent point of the medial epicondyle. Medial end of the "
            "ANATOMICAL transepicondylar axis (aTEA)."
        ),
        feeds=("atea",),
        aliases=("medial_epicondyle", "ME", "epicondyle_med"),
        observer_sd_mm=1.6,
    ),
    LandmarkDefinition(
        id="femur.condyle_distal_medial",
        bone="femur",
        display_name="Distal medial condyle",
        definition=(
            "The most distal point of the medial condyle, measured along the frame's "
            "proximal axis."
        ),
        feeds=("distal_condylar_axis", "mldfa", "distal_resection_medial",
               "joint_line_obliquity"),
        aliases=("distal_medial", "DMC"),
        observer_sd_mm=0.9,
    ),
    LandmarkDefinition(
        id="femur.condyle_distal_lateral",
        bone="femur",
        display_name="Distal lateral condyle",
        definition="The most distal point of the lateral condyle.",
        feeds=("distal_condylar_axis", "mldfa", "distal_resection_lateral",
               "joint_line_obliquity"),
        aliases=("distal_lateral", "DLC"),
        observer_sd_mm=0.9,
    ),
    LandmarkDefinition(
        id="femur.condyle_posterior_medial",
        bone="femur",
        display_name="Posterior medial condyle",
        definition="The most posterior point of the medial condyle.",
        feeds=("posterior_condylar_axis", "condylar_twist_angle",
               "posterior_resection_medial"),
        aliases=("posterior_medial", "PMC"),
        observer_sd_mm=1.0,
    ),
    LandmarkDefinition(
        id="femur.condyle_posterior_lateral",
        bone="femur",
        display_name="Posterior lateral condyle",
        definition="The most posterior point of the lateral condyle.",
        feeds=("posterior_condylar_axis", "condylar_twist_angle",
               "posterior_resection_lateral"),
        aliases=("posterior_lateral", "PLC"),
        observer_sd_mm=1.0,
    ),
    LandmarkDefinition(
        id="femur.notch_centre",
        bone="femur",
        display_name="Intercondylar notch centre",
        definition=(
            "The centre of the intercondylar notch at its roof. Used as the femoral "
            "knee centre, deliberately in preference to the transepicondylar midpoint "
            "so that the coronal and rotational constructions do not share a landmark."
        ),
        feeds=("femoral_knee_centre", "whiteside_line"),
        aliases=("notch", "intercondylar_notch"),
        observer_sd_mm=1.5,
    ),
    LandmarkDefinition(
        id="femur.trochlear_groove_anterior",
        bone="femur",
        display_name="Anterior trochlear groove",
        definition=(
            "The deepest point of the trochlear groove at its anterior extent. "
            "Anterior end of Whiteside's line."
        ),
        feeds=("whiteside_line",),
        aliases=("trochlear_groove", "whiteside_anterior"),
        observer_sd_mm=1.8,
    ),
    LandmarkDefinition(
        id="femur.canal_centre_distal",
        bone="femur",
        display_name="Femoral canal centre (distal level)",
        definition=(
            "Centroid of the medullary canal cross-section at the distal reference "
            "level. Normally derived automatically; the level used is recorded in the "
            "plan."
        ),
        feeds=("femoral_anatomical_axis",),
        picked_in_increment_1=False,
    ),
    LandmarkDefinition(
        id="femur.canal_centre_proximal",
        bone="femur",
        display_name="Femoral canal centre (proximal level)",
        definition=(
            "Centroid of the medullary canal cross-section at the proximal reference "
            "level. Together with the distal centre this defines the anatomical "
            "(diaphyseal) axis; their separation is the fit's lever arm and is "
            "reported as a confidence indicator."
        ),
        feeds=("femoral_anatomical_axis",),
        picked_in_increment_1=False,
    ),
    LandmarkDefinition(
        id="femur.head_centre",
        bone="femur",
        display_name="Femoral head centre",
        definition=(
            "Centre of a sphere fitted to the femoral head articular surface. The "
            "proximal end of the femoral mechanical axis, and therefore a prerequisite "
            "for HKA and mLDFA."
        ),
        feeds=("femoral_mechanical_axis", "hka", "mldfa"),
        aliases=("hip_centre", "femoral_head", "FHC"),
        observer_sd_mm=1.2,
        typically_out_of_scan=True,
    ),
    # ---------------- Tibia ----------------
    LandmarkDefinition(
        id="tibia.spine_medial",
        bone="tibia",
        display_name="Medial intercondylar tubercle",
        definition="Apex of the medial intercondylar tubercle (medial tibial spine).",
        feeds=("tibial_knee_centre",),
        aliases=("medial_spine",),
        observer_sd_mm=1.3,
    ),
    LandmarkDefinition(
        id="tibia.spine_lateral",
        bone="tibia",
        display_name="Lateral intercondylar tubercle",
        definition="Apex of the lateral intercondylar tubercle (lateral tibial spine).",
        feeds=("tibial_knee_centre",),
        aliases=("lateral_spine",),
        observer_sd_mm=1.3,
    ),
    LandmarkDefinition(
        id="tibia.pcl_insertion_midpoint",
        bone="tibia",
        display_name="PCL insertion midpoint",
        definition=(
            "Midpoint of the tibial attachment footprint of the posterior cruciate "
            "ligament. Posterior end of the Akagi line."
        ),
        feeds=("akagi_line", "tibial_rotation"),
        aliases=("pcl", "pcl_insertion"),
        observer_sd_mm=2.0,
    ),
    LandmarkDefinition(
        id="tibia.tubercle_patellar_tendon_medial_border",
        bone="tibia",
        display_name="Patellar tendon medial border",
        definition=(
            "The medial border of the patellar tendon attachment on the tibial "
            "tubercle. Anterior end of the Akagi line as originally defined "
            "(Akagi et al., 2004)."
        ),
        feeds=("akagi_line", "tibial_rotation"),
        aliases=("akagi_anterior", "patellar_tendon_medial"),
        observer_sd_mm=2.2,
    ),
    LandmarkDefinition(
        id="tibia.tubercle_medial_third",
        bone="tibia",
        display_name="Tibial tubercle, medial third",
        definition=(
            "The junction of the medial and middle thirds of the tibial tubercle. An "
            "alternative rotational reference, differing from the Akagi line by "
            "roughly 2-4 degrees; recorded so the two methods can be compared."
        ),
        feeds=("tibial_rotation",),
        aliases=("medial_third",),
        observer_sd_mm=2.0,
    ),
    LandmarkDefinition(
        id="tibia.plateau_medial_lowest",
        bone="tibia",
        display_name="Medial plateau, deepest point",
        definition=(
            "The most distal point of the medial articular surface, including any "
            "wear. The reference from which medial tibial resection is measured."
        ),
        feeds=("mpta", "tibial_resection_medial", "coronal_joint_line"),
        aliases=("medial_plateau",),
        observer_sd_mm=1.1,
    ),
    LandmarkDefinition(
        id="tibia.plateau_lateral_lowest",
        bone="tibia",
        display_name="Lateral plateau, deepest point",
        definition="The most distal point of the lateral articular surface.",
        feeds=("mpta", "tibial_resection_lateral", "coronal_joint_line"),
        aliases=("lateral_plateau",),
        observer_sd_mm=1.1,
    ),
    LandmarkDefinition(
        id="tibia.plateau_medial_anterior",
        bone="tibia",
        display_name="Medial plateau, anterior rim",
        definition=(
            "Anterior rim of the medial plateau in its mid-compartment sagittal plane. "
            "With the posterior rim this defines the native posterior slope."
        ),
        feeds=("posterior_slope_medial",),
        observer_sd_mm=1.4,
    ),
    LandmarkDefinition(
        id="tibia.plateau_medial_posterior",
        bone="tibia",
        display_name="Medial plateau, posterior rim",
        definition=(
            "Posterior rim of the medial plateau in its mid-compartment sagittal plane."
        ),
        feeds=("posterior_slope_medial",),
        observer_sd_mm=1.4,
    ),
    LandmarkDefinition(
        id="tibia.canal_centre_proximal",
        bone="tibia",
        display_name="Tibial canal centre (proximal level)",
        definition=(
            "Centroid of the tibial canal cross-section at the proximal reference "
            "level."
        ),
        feeds=("tibial_anatomical_axis",),
        picked_in_increment_1=False,
    ),
    LandmarkDefinition(
        id="tibia.canal_centre_distal",
        bone="tibia",
        display_name="Tibial canal centre (distal level)",
        definition=(
            "Centroid of the tibial canal cross-section at the distal reference level."
        ),
        feeds=("tibial_anatomical_axis",),
        picked_in_increment_1=False,
    ),
    LandmarkDefinition(
        id="tibia.ankle_centre",
        bone="tibia",
        display_name="Ankle centre",
        definition=(
            "Midpoint of the medial and lateral malleolar apices, or the centre of the "
            "tibial plafond. The distal end of the tibial mechanical axis."
        ),
        feeds=("tibial_mechanical_axis", "hka", "mpta"),
        aliases=("ankle", "malleolar_midpoint"),
        observer_sd_mm=1.5,
        typically_out_of_scan=True,
    ),
    # ---------------- Fibula ----------------
    LandmarkDefinition(
        id="fibula.head_apex",
        bone="fibula",
        display_name="Fibular head apex",
        definition=(
            "The apex of the fibular head. A secondary coronal reference, available in "
            "every case in this cohort because the fibula is segmented alongside the "
            "tibia."
        ),
        feeds=("coronal_reference_secondary",),
        aliases=("fibular_head",),
        observer_sd_mm=1.6,
    ),
    # ---------------- Patella (registered now, used later) ----------------
    LandmarkDefinition(
        id="patella.ridge_apex",
        bone="patella",
        display_name="Patellar ridge apex",
        definition=(
            "The apex of the median ridge of the patella. Registered so the schema "
            "need not be re-versioned when patellar resurfacing is added; not picked "
            "or used in the current increment."
        ),
        feeds=(),
        picked_in_increment_1=False,
    ),
    LandmarkDefinition(
        id="patella.centre",
        bone="patella",
        display_name="Patellar centre",
        definition=(
            "Geometric centre of the patellar articular surface. Reserved for future "
            "patellar planning."
        ),
        feeds=(),
        picked_in_increment_1=False,
    ),
)


def definitions_for_bone(bone: str, *, picked_only: bool = False):
    """Registry entries for one bone, in registry order.

    Drives both the picking checklist and the report glossary, so the order landmarks
    appear in the UI is the order they are declared above.
    """
    return [
        definition
        for definition in REGISTRY.values()
        if definition.bone == bone
        and (definition.picked_in_increment_1 or not picked_only)
    ]


# Alias lookup, built once. Slicer control-point labels rarely match our ids.
def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for definition in REGISTRY.values():
        for key in (definition.id, definition.display_name, *definition.aliases):
            index[_normalise_label(key)] = definition.id
        # Also accept the id without its bone prefix, e.g. "notch_centre".
        index.setdefault(_normalise_label(definition.id.split(".", 1)[1]),
                         definition.id)
    return index


def _normalise_label(label: str) -> str:
    """Fold a human-written label to a comparable key."""
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


_ALIAS_INDEX = _build_alias_index()


def resolve_landmark_id(label: str) -> str | None:
    """Map a free-form label onto a registry id, or ``None`` if unrecognised."""
    return _ALIAS_INDEX.get(_normalise_label(label))


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------


@dataclass
class Landmark:
    """One landmark observation on one case."""

    id: str
    position_mm: np.ndarray | None = None
    status: LandmarkStatus = LandmarkStatus.NOT_PICKED
    origin: str = "unknown"
    reason: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.id not in REGISTRY:
            raise KeyError(
                f"Unknown landmark id {self.id!r}. Landmarks must be declared in the "
                f"registry before use."
            )
        if self.position_mm is not None:
            self.position_mm = np.asarray(self.position_mm, dtype=float).reshape(3)
            if not np.all(np.isfinite(self.position_mm)):
                raise ValueError(f"{self.id}: position contains NaN or infinity.")

        if self.status.has_position and self.position_mm is None:
            raise ValueError(
                f"{self.id}: status {self.status.value!r} promises a position but "
                f"none was supplied."
            )
        if not self.status.has_position and self.position_mm is not None:
            raise ValueError(
                f"{self.id}: status {self.status.value!r} carries a position, which "
                f"would let a discarded point be used by mistake."
            )

    @property
    def definition(self) -> LandmarkDefinition:
        return REGISTRY[self.id]

    @property
    def is_usable(self) -> bool:
        return self.status.has_position


class LandmarkSet:
    """All landmark observations for one case-side, in one coordinate system.

    The set is deliberately not a bare dict: it owns the case identity, the coordinate
    system and the provenance, because a bag of coordinates without those is
    unreproducible and, in the coordinate system's case, actively dangerous.
    """

    def __init__(
        self,
        case_id: str,
        side: "str | Side",
        coordinate_system: "str | CoordinateSystem",
        landmarks: "dict[str, Landmark] | list[Landmark] | None" = None,
        source: dict | None = None,
    ):
        self.case_id = case_id
        self.side = Side.parse(side)
        self.coordinate_system = CoordinateSystem.parse(coordinate_system)
        self.source = source or {}

        self._landmarks: dict[str, Landmark] = {}
        if isinstance(landmarks, dict):
            landmarks = list(landmarks.values())
        for landmark in landmarks or []:
            self.add(landmark)

    # -- container behaviour ------------------------------------------

    def add(self, landmark: Landmark) -> None:
        if landmark.id in self._landmarks:
            raise ValueError(f"Landmark {landmark.id} is already in the set.")
        self._landmarks[landmark.id] = landmark

    def __contains__(self, landmark_id: str) -> bool:
        return landmark_id in self._landmarks

    def __len__(self) -> int:
        return len(self._landmarks)

    def __iter__(self):
        return iter(self._landmarks.values())

    def get(self, landmark_id: str) -> Landmark:
        """Return a landmark, or a ``NOT_PICKED`` placeholder if it was never recorded.

        Absence is represented explicitly rather than raising, so downstream code can
        report *why* a metric is unavailable instead of crashing on a missing key.
        """
        if landmark_id not in REGISTRY:
            raise KeyError(f"Unknown landmark id {landmark_id!r}.")
        return self._landmarks.get(
            landmark_id, Landmark(id=landmark_id, status=LandmarkStatus.NOT_PICKED)
        )

    def position(self, landmark_id: str) -> np.ndarray | None:
        landmark = self.get(landmark_id)
        return landmark.position_mm if landmark.is_usable else None

    def require(self, *landmark_ids: str) -> list[np.ndarray]:
        """Fetch positions, raising a message naming exactly what is missing and why."""
        missing = []
        positions = []
        for landmark_id in landmark_ids:
            landmark = self.get(landmark_id)
            if not landmark.is_usable:
                missing.append(f"{landmark_id} ({landmark.status.value})")
            else:
                positions.append(landmark.position_mm)
        if missing:
            raise MissingLandmarks(missing)
        return positions

    def available(self, *landmark_ids: str) -> bool:
        return all(self.get(i).is_usable for i in landmark_ids)

    def missing_of(self, *landmark_ids: str) -> list[str]:
        return [i for i in landmark_ids if not self.get(i).is_usable]

    # -- transformation ------------------------------------------------

    def to_coordinate_system(self, target: "str | CoordinateSystem") -> "LandmarkSet":
        """Return an equivalent set expressed in another coordinate system."""
        target = CoordinateSystem.parse(target)
        if target is self.coordinate_system:
            return self

        converted = []
        for landmark in self:
            position = landmark.position_mm
            if position is not None:
                position = self.coordinate_system.convert_to(
                    position.reshape(1, 3), target
                ).reshape(3)
            converted.append(
                Landmark(
                    id=landmark.id,
                    position_mm=position,
                    status=landmark.status,
                    origin=landmark.origin,
                    reason=landmark.reason,
                    metadata=dict(landmark.metadata),
                )
            )
        return LandmarkSet(
            case_id=self.case_id,
            side=self.side,
            coordinate_system=target,
            landmarks=converted,
            source={**self.source, "converted_from": self.coordinate_system.value},
        )

    def positions_array(self) -> np.ndarray:
        """All usable positions as ``(N, 3)`` -- used for the mesh bounds check."""
        usable = [lm.position_mm for lm in self if lm.is_usable]
        return np.array(usable) if usable else np.zeros((0, 3))

    def __repr__(self) -> str:
        usable = sum(1 for lm in self if lm.is_usable)
        return (
            f"LandmarkSet({self.case_id}, {self.side}, "
            f"{self.coordinate_system.value}, {usable}/{len(self)} usable)"
        )


class MissingLandmarks(LookupError):
    """Raised when required landmarks are unavailable; carries the list."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("Missing required landmarks: " + ", ".join(missing))


# ----------------------------------------------------------------------
# Slicer interchange
# ----------------------------------------------------------------------


def read_slicer_fcsv(
    path: "str | Path",
    *,
    case_id: str,
    side: "str | Side",
) -> LandmarkSet:
    """Read a legacy Slicer ``.fcsv`` markups file.

    The format carries its coordinate system in a ``# CoordinateSystem =`` header,
    which older versions write as the numeric codes 0 (RAS) and 1 (LPS). A file without
    that header is rejected rather than assumed, because guessing wrong mirrors the
    anatomy.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")

    coordinate_system = None
    columns = None
    data_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            match = re.match(r"#\s*CoordinateSystem\s*=\s*(\S+)", stripped, re.I)
            if match:
                coordinate_system = CoordinateSystem.parse(match.group(1))
            match = re.match(r"#\s*columns\s*=\s*(.+)", stripped, re.I)
            if match:
                columns = [c.strip() for c in match.group(1).split(",")]
            continue
        data_lines.append(stripped)

    if coordinate_system is None:
        raise ValueError(
            f"{path.name} has no '# CoordinateSystem =' header. Refusing to guess "
            f"between LPS and RAS: choosing wrong mirrors the anatomy left for right."
        )
    if columns is None:
        columns = ["id", "x", "y", "z", "ow", "ox", "oy", "oz",
                   "vis", "sel", "lock", "label", "desc", "associatedNodeID"]

    landmarks, unmatched = [], []
    for row in csv.reader(data_lines):
        record = dict(zip(columns, row))
        label = record.get("label", "").strip()
        landmark_id = resolve_landmark_id(label)
        if landmark_id is None:
            unmatched.append(label)
            continue
        landmarks.append(
            Landmark(
                id=landmark_id,
                position_mm=[float(record["x"]), float(record["y"]),
                             float(record["z"])],
                status=LandmarkStatus.PRESENT,
                origin=f"slicer_fcsv:{path.name}",
                metadata={"source_label": label},
            )
        )

    return LandmarkSet(
        case_id=case_id,
        side=side,
        coordinate_system=coordinate_system,
        landmarks=landmarks,
        source={
            "kind": "slicer_fcsv",
            "file": path.name,
            "unmatched_labels": unmatched,
            "imported_utc": _utc_now(),
        },
    )


def read_slicer_markups(
    path: "str | Path",
    *,
    case_id: str,
    side: "str | Side",
) -> LandmarkSet:
    """Read a current Slicer ``.mrk.json`` markups file.

    Each markup node declares its own ``coordinateSystem``. Nodes disagreeing with one
    another are rejected rather than silently reconciled.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8-sig"))

    markups = document.get("markups")
    if not markups:
        raise ValueError(f"{path.name} contains no 'markups' array.")

    systems = set()
    landmarks, unmatched = [], []
    for markup in markups:
        declared = markup.get("coordinateSystem")
        if declared is None:
            raise ValueError(
                f"{path.name}: a markup node omits 'coordinateSystem'. Refusing to "
                f"guess between LPS and RAS."
            )
        systems.add(CoordinateSystem.parse(declared))

        # Slicer records units per node. The pipeline is millimetres throughout, and a
        # silent unit mismatch would scale the anatomy by a thousand.
        units = markup.get("coordinateUnits", "mm")
        if isinstance(units, str) and units.strip().lower() not in ("mm", "millimeter",
                                                                   "millimetre"):
            raise ValueError(
                f"{path.name}: markup declares units {units!r}; this pipeline works "
                f"exclusively in millimetres."
            )

        for point in markup.get("controlPoints", []):
            label = str(point.get("label", "")).strip()
            landmark_id = resolve_landmark_id(label)
            if landmark_id is None:
                unmatched.append(label)
                continue
            landmarks.append(
                Landmark(
                    id=landmark_id,
                    position_mm=point["position"],
                    status=LandmarkStatus.PRESENT,
                    origin=f"slicer_markups:{path.name}",
                    metadata={"source_label": label},
                )
            )

    if len(systems) > 1:
        raise ValueError(
            f"{path.name} mixes coordinate systems "
            f"({', '.join(sorted(s.value for s in systems))}). Split the file or fix "
            f"it in Slicer; combining them would mirror some landmarks and not others."
        )

    return LandmarkSet(
        case_id=case_id,
        side=side,
        coordinate_system=systems.pop(),
        landmarks=landmarks,
        source={
            "kind": "slicer_markups",
            "file": path.name,
            "schema": document.get("@schema"),
            "unmatched_labels": unmatched,
            "imported_utc": _utc_now(),
        },
    )


# ----------------------------------------------------------------------
# Native format
# ----------------------------------------------------------------------


def write_landmark_set(landmark_set: LandmarkSet, path: "str | Path") -> Path:
    """Write the native landmark file.

    Kept separate from the plan so that landmarks -- the human input -- can be swapped
    or perturbed without touching anything else. The sensitivity study depends on this:
    it rewrites landmarks thousands of times and re-derives plans from them.
    """
    path = Path(path)
    document = {
        "schema_version": SCHEMA_VERSION,
        "case_id": landmark_set.case_id,
        "side": landmark_set.side.value,
        "coordinate_system": landmark_set.coordinate_system.value,
        "units": "mm",
        "source": landmark_set.source,
        "points": [
            {
                "id": landmark.id,
                "position_mm": (
                    [round(float(v), 6) for v in landmark.position_mm]
                    if landmark.position_mm is not None
                    else None
                ),
                "status": landmark.status.value,
                "origin": landmark.origin,
                **({"reason": landmark.reason} if landmark.reason else {}),
                **({"metadata": landmark.metadata} if landmark.metadata else {}),
            }
            for landmark in landmark_set
        ],
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return path


def read_landmark_set(path: "str | Path") -> LandmarkSet:
    """Read the native landmark file."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8-sig"))

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: schema version {version!r} is not the expected "
            f"{SCHEMA_VERSION!r}."
        )
    if document.get("units") != "mm":
        raise ValueError(
            f"{path.name}: units are {document.get('units')!r}; this pipeline works "
            f"exclusively in millimetres."
        )

    landmarks = [
        Landmark(
            id=entry["id"],
            position_mm=entry.get("position_mm"),
            status=LandmarkStatus(entry["status"]),
            origin=entry.get("origin", "unknown"),
            reason=entry.get("reason"),
            metadata=entry.get("metadata", {}),
        )
        for entry in document.get("points", [])
    ]

    return LandmarkSet(
        case_id=document["case_id"],
        side=document["side"],
        coordinate_system=document["coordinate_system"],
        landmarks=landmarks,
        source=document.get("source", {}),
    )


def load_landmarks(
    path: "str | Path",
    *,
    case_id: str | None = None,
    side: "str | Side | None" = None,
) -> LandmarkSet:
    """Read landmarks from whichever supported format the file happens to be.

    Slicer files carry no case identity, so ``case_id`` and ``side`` are required for
    them; the native format records its own.
    """
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".fcsv"):
        _require_identity(case_id, side, path)
        return read_slicer_fcsv(path, case_id=case_id, side=side)
    if name.endswith(".mrk.json"):
        _require_identity(case_id, side, path)
        return read_slicer_markups(path, case_id=case_id, side=side)
    if name.endswith(".json"):
        return read_landmark_set(path)

    raise ValueError(
        f"Unrecognised landmark file type: {path.name}. Expected .fcsv, .mrk.json, "
        f"or a native .json landmark file."
    )


def _require_identity(case_id, side, path: Path) -> None:
    if case_id is None or side is None:
        raise ValueError(
            f"{path.name} is a Slicer export and carries no case identity. Supply "
            f"case_id and side explicitly."
        )


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
