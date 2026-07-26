"""How a number was arrived at, carried alongside the number itself.

This is the module that makes the openness claim concrete. A commercial planner shows
you an angle; it does not show you whether that angle was measured from your anatomy or
substituted from a population average because the necessary landmark was outside the
scan. Here the two are never confused, because a value cannot be constructed without
declaring which it is.

The distinction is not academic for this cohort. Every scan is knee-only, so the
femoral head is absent and the femoral mechanical axis cannot be measured. A pipeline
can respond to that in three ways: refuse to run, silently substitute an assumption, or
substitute the assumption and say so. Only the third is both useful and honest, and it
requires the assumption to be a first-class object -- named, valued, sourced, attached
to the metrics it affects, and overridable per patient.

That is what turns the legacy pipeline's ``VALGUS_ANGLE_DEG = 6.0`` from a hidden
constant into a stated modelling decision a reader can challenge.

Three quality tiers are used throughout:

``MEASURED``
    Every input came from the patient's anatomy.
``ESTIMATED``
    An assumption stood in for something unavailable. The assumption travels with the
    value and appears in the report next to it.
``NOT_COMPUTABLE``
    Required inputs are missing and no accepted substitute exists. The value is
    ``None`` -- never a plausible-looking guess -- and carries what is missing, why,
    and what it would take to obtain it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Quality",
    "Assumption",
    "Metric",
    "POPULATION_FEMORAL_AMA",
    "TIBIAL_AMA_NEGLIGIBLE",
    "AUTOMATIC_LANDMARK_ESTIMATE",
    "ATEA_SUBSTITUTED_FOR_STEA",
]


class Quality(Enum):
    """How well supported a derived quantity is by the patient's own anatomy."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    NOT_COMPUTABLE = "not_computable"

    @property
    def is_available(self) -> bool:
        return self is not Quality.NOT_COMPUTABLE

    def combine(self, other: "Quality") -> "Quality":
        """The quality of a result derived from two inputs: the weaker of the two.

        Quality never improves through composition. An angle between a measured axis
        and an estimated one is estimated, and anything built on an unavailable input
        is unavailable.
        """
        order = [Quality.NOT_COMPUTABLE, Quality.ESTIMATED, Quality.MEASURED]
        return min(self, other, key=order.index)


@dataclass(frozen=True)
class Assumption:
    """A value substituted for something the data could not supply.

    Recorded rather than applied silently. ``reason_required`` states what was missing,
    which matters because the same assumption is defensible when the anatomy is outside
    the scan and indefensible when someone simply has not picked the landmark yet.
    """

    id: str
    description: str
    value: float
    unit: str
    source: str
    applies_to: tuple[str, ...] = ()
    reason_required: str = ""
    patient_specific: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "applies_to": list(self.applies_to),
            "reason_required": self.reason_required,
            "patient_specific": self.patient_specific,
        }


# ----------------------------------------------------------------------
# The assumptions this pipeline is currently willing to make
# ----------------------------------------------------------------------

POPULATION_FEMORAL_AMA = Assumption(
    id="population_femoral_ama",
    description=(
        "Femoral mechanical-anatomical angle: the angle between the femoral mechanical "
        "axis (femoral head centre to knee centre) and the anatomical axis of the "
        "diaphysis. Used to derive a mechanical axis when the femoral head is outside "
        "the scan, by rotating the measured diaphyseal axis medially in the coronal "
        "plane."
    ),
    value=6.0,
    unit="deg",
    source=(
        "Population mean; commonly cited as 5-7 degrees with real inter-patient "
        "variance. This is the same constant the legacy pipeline applied as a fixed "
        "6 degree valgus correction, now declared rather than hidden."
    ),
    applies_to=("femoral_mechanical_axis", "mldfa_deg", "hka_deviation_deg"),
    reason_required="femoral head outside the scan field of view",
    patient_specific=False,
)

AUTOMATIC_LANDMARK_ESTIMATE = Assumption(
    id="automatic_landmark_estimate",
    description=(
        "One or more landmarks feeding this metric were positioned by the automatic "
        "estimator rather than picked by a human on the anatomy. The value is a "
        "machine estimate awaiting review."
    ),
    value=0.0,
    unit="",
    source=(
        "tka_planner.core.landmarks_auto. Confidence is recorded per landmark; the "
        "tibial rotational references and the medial epicondylar sulcus are the least "
        "reliable and should be corrected first."
    ),
    applies_to=(),
    reason_required="landmarks have not been reviewed by a human",
    patient_specific=False,
)

ATEA_SUBSTITUTED_FOR_STEA = Assumption(
    id="atea_substituted_for_stea",
    description=(
        "The anatomical transepicondylar axis (lateral prominence to medial "
        "prominence) was used as the frame's rotational reference in place of the "
        "surgical axis (lateral prominence to medial sulcus), because the sulcus was "
        "unavailable."
    ),
    value=1.5,
    unit="deg",
    source=(
        "The two axes differ by roughly 1-2 degrees. The medial sulcus is a depression "
        "rather than a surface extreme, so no automatic estimator can locate it -- it "
        "requires a human pick. Component rotation planned on the anatomical axis is "
        "therefore internally rotated by about this much relative to a surgical-axis "
        "plan."
    ),
    applies_to=("femoral_frame_rotation", "condylar_twist_deg"),
    reason_required="medial epicondylar sulcus not available",
    patient_specific=False,
)

TIBIAL_AMA_NEGLIGIBLE = Assumption(
    id="tibial_ama_negligible",
    description=(
        "The tibial mechanical and anatomical axes are treated as collinear, so the "
        "proximal diaphyseal axis stands in for the mechanical axis when the malleoli "
        "are outside the scan."
    ),
    value=0.0,
    unit="deg",
    source=(
        "The tibial mechanical-anatomical angle is roughly 0-2 degrees, an order of "
        "magnitude smaller than the femoral one. This asymmetry is why MPTA degrades "
        "gracefully on a knee-only scan while mLDFA does not."
    ),
    applies_to=("tibial_mechanical_axis", "mpta_deg", "hka_deviation_deg"),
    reason_required="malleoli outside the scan field of view",
    patient_specific=False,
)


# ----------------------------------------------------------------------
# The metric envelope
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """One reported quantity together with everything needed to judge it.

    Every metric in the system has this same shape, and that uniformity is load-bearing:
    the report renderer, the cohort CSV writer and the sensitivity analyser all iterate
    over metrics generically. A metric with a bespoke structure would have to be special-
    cased in three places and would eventually be forgotten in one of them.

    ``definition`` and ``sign_convention`` are prose because they are reproduced in the
    report. A reader of the paper should not have to read the source to learn whether a
    smaller mLDFA means more valgus or less.
    """

    name: str
    value: float | None
    unit: str
    quality: Quality
    definition: str
    sign_convention: str = ""
    method: str = ""
    inputs: tuple[str, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    reference_range: tuple[float, float] | None = None
    missing: tuple[str, ...] = ()
    reason: str = ""
    would_require: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.quality is Quality.NOT_COMPUTABLE and self.value is not None:
            raise ValueError(
                f"{self.name}: marked not computable but carries a value. An "
                f"unavailable metric must be None so it cannot be read as a result."
            )
        if self.quality.is_available and self.value is None:
            raise ValueError(
                f"{self.name}: quality {self.quality.value!r} promises a value but "
                f"none was supplied."
            )
        if self.quality is Quality.ESTIMATED and not self.assumptions:
            raise ValueError(
                f"{self.name}: marked estimated but names no assumption. An estimate "
                f"whose assumption is not recorded is indistinguishable from a "
                f"measurement, which defeats the point of the tier."
            )

    @property
    def is_outside_reference_range(self) -> bool:
        """True when a value sits outside its normal range -- flagged, never corrected."""
        if self.value is None or self.reference_range is None:
            return False
        low, high = self.reference_range
        return not (low <= self.value <= high)

    @classmethod
    def unavailable(
        cls,
        name: str,
        unit: str,
        definition: str,
        *,
        missing: tuple[str, ...],
        reason: str,
        would_require: str = "",
        method: str = "",
    ) -> "Metric":
        """Construct a metric that cannot be computed, recording why.

        The ``would_require`` text is what lets the report tell a reader how to obtain
        the missing measurement rather than merely that it is absent.
        """
        return cls(
            name=name,
            value=None,
            unit=unit,
            quality=Quality.NOT_COMPUTABLE,
            definition=definition,
            method=method,
            missing=missing,
            reason=reason,
            would_require=would_require,
        )

    def to_dict(self) -> dict:
        record = {
            "value": self.value,
            "unit": self.unit,
            "status": self.quality.value,
            "definition": self.definition,
        }
        if self.sign_convention:
            record["sign_convention"] = self.sign_convention
        if self.method:
            record["method"] = self.method
        if self.inputs:
            record["inputs"] = list(self.inputs)
        if self.assumptions:
            record["assumptions"] = [a.id for a in self.assumptions]
        if self.reference_range:
            record["reference_range"] = list(self.reference_range)
            record["outside_reference_range"] = self.is_outside_reference_range
        if self.missing:
            record["missing"] = list(self.missing)
        if self.reason:
            record["reason"] = self.reason
        if self.would_require:
            record["would_require"] = self.would_require
        if self.diagnostics:
            record["diagnostics"] = self.diagnostics
        return record
