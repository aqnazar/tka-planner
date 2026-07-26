"""Laterality — the single place side-dependent logic is allowed to live.

Almost every silent, systematic error in knee planning software is a side error.
The pattern that causes it is a conditional like ``if side == "left": angle = -angle``
scattered through measurement code: each site looks locally reasonable, no single one
is obviously wrong, and the result is a pipeline that is correct on left knees and
quietly wrong on right ones.

The legacy pipeline had exactly this. ``find_condyle_contact`` split the femur at the
bounding-box midline and called the lower half "medial" unconditionally, which is true
for one side and false for the other. It went unnoticed because the function discarded
the labels immediately -- but any per-compartment measurement built on it would have
inherited a systematic side-dependent error.

The fix is structural rather than vigilant. Anatomical directions are *asked for*, never
derived at the point of use::

    medial = side.medial_direction(frame.y_patient_left)
    depth  = float(np.dot(point - reference, medial))

No call site needs to know which side it is looking at, so no call site can get it
wrong. A CI check enforces that ``if side ==`` and bare ``"left"``/``"right"``
comparisons appear nowhere outside this module, and a mirror-invariance test asserts
that reflecting a case and flipping its side leaves every scalar metric unchanged.

Coordinate convention
---------------------
Everything downstream works in the DICOM/LPS patient frame that CT segmentation already
provides:

======  =====================
``+X``  patient **L**\\ eft
``+Y``  **P**\\ osterior
``+Z``  **S**\\ uperior
======  =====================

Anatomical frames built on top of it are right-handed and named semantically
(``x_anterior``, ``y_patient_left``, ``z_proximal``) rather than by axis letter, because
"the Y axis" carries no anatomical meaning while "patient-left" does.

Note that *patient-left* is a fixed spatial direction whereas *medial* and *lateral* are
not: medial points toward the body midline, so it is patient-right on a left knee and
patient-left on a right one. That distinction is the entire content of this module.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = ["Side", "LPS_PATIENT_LEFT", "LPS_POSTERIOR", "LPS_SUPERIOR", "LPS_ANTERIOR"]


# World axis directions in the LPS patient frame.
LPS_PATIENT_LEFT = np.array([1.0, 0.0, 0.0])
LPS_POSTERIOR = np.array([0.0, 1.0, 0.0])
LPS_SUPERIOR = np.array([0.0, 0.0, 1.0])
LPS_ANTERIOR = -LPS_POSTERIOR


class Side(Enum):
    """Which knee. Owns every medial/lateral decision in the codebase.

    ``Side`` deliberately exposes no way to ask "is this the left one?". It answers
    anatomical questions instead -- which direction is medial, which of two points is
    more lateral -- so that callers express intent rather than branching on identity.
    """

    LEFT = "left"
    RIGHT = "right"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, value: "str | Side") -> "Side":
        """Accept the several spellings that appear across Slicer, CSVs and filenames.

        ``"L"``, ``"l"``, ``"left"``, ``"Left"`` and ``Side.LEFT`` all mean the same
        thing. Anything else raises rather than defaulting, because a silently
        defaulted side is precisely the failure this module exists to prevent.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"Cannot interpret {value!r} as a side.")

        normalised = value.strip().lower()
        if normalised in ("l", "left"):
            return cls.LEFT
        if normalised in ("r", "right"):
            return cls.RIGHT
        raise ValueError(
            f"Unrecognised side {value!r}. Expected one of: L, left, R, right."
        )

    # ------------------------------------------------------------------
    # Anatomical directions
    # ------------------------------------------------------------------

    @property
    def medial_sign(self) -> int:
        """Coefficient turning a patient-left vector into a medial-pointing one.

        Medial points toward the body midline. A left knee sits at positive X in LPS,
        so its midline direction is patient-*right* (``-1``); a right knee's is
        patient-left (``+1``).
        """
        return -1 if self is Side.LEFT else +1

    @property
    def lateral_sign(self) -> int:
        """Coefficient turning a patient-left vector into a lateral-pointing one."""
        return -self.medial_sign

    def medial_direction(self, y_patient_left: np.ndarray) -> np.ndarray:
        """Unit vector pointing medially, given a frame's patient-left axis."""
        return self.medial_sign * _unit(y_patient_left)

    def lateral_direction(self, y_patient_left: np.ndarray) -> np.ndarray:
        """Unit vector pointing laterally, given a frame's patient-left axis."""
        return self.lateral_sign * _unit(y_patient_left)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def split_compartments(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        y_patient_left: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition points into ``(medial, lateral)`` boolean masks about a midline.

        This replaces the legacy bounding-box split, which labelled compartments by
        world position and was therefore correct on only one side.

        Points exactly on the dividing plane are assigned to the lateral compartment, so
        the two masks are always disjoint and exhaustive.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected an (N, 3) point array, got {points.shape}.")

        medial = self.medial_direction(y_patient_left)
        offsets = (points - np.asarray(origin, dtype=float)) @ medial
        medial_mask = offsets > 0.0
        return medial_mask, ~medial_mask

    def more_lateral(
        self,
        a: np.ndarray,
        b: np.ndarray,
        y_patient_left: np.ndarray,
    ) -> np.ndarray:
        """Return whichever of two points lies further laterally."""
        lateral = self.lateral_direction(y_patient_left)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return a if np.dot(a, lateral) >= np.dot(b, lateral) else b

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    @property
    def opposite(self) -> "Side":
        """The contralateral knee.

        Also the side a case takes on when mirrored through the sagittal plane, which
        is what the mirror-invariance test relies on.
        """
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    def __str__(self) -> str:
        return self.value


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalise, refusing degenerate input rather than emitting NaNs."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalise a zero-length direction vector.")
    return vector / norm
