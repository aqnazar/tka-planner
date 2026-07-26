"""Quality control: what the data can and cannot support.

This module answers questions the operator should not be trusted to answer by
assertion. Is the femoral head in this scan? Is the mesh closed enough to boolean? Are
these landmarks even on this bone? Are we sure this is millimetres?

The design rule throughout is that **scan coverage is detected, not declared**. A
pipeline that lets someone tick "full-limb scan" in a config file will eventually
compute a hip-knee-ankle angle from a knee-only CT and report it as measured. Detecting
it from the geometry removes the opportunity.

The detection that matters most for this cohort is the femoral head. Every case here is
a knee-region scan, so the head is absent and the femoral mechanical axis cannot be
measured. Rather than trusting a filename or a flag, :func:`assess_femur_coverage` tries
to fit a sphere to the proximal end and requires two independent conditions to agree:
the bone must be long enough to plausibly reach the hip, *and* the sphere fit must
converge tightly at an anatomically credible radius. A diaphysis will not pass the
second test -- a cylinder fits a sphere badly -- so a short segment cannot be mistaken
for a head.

Findings are data, not exceptions. They travel into the plan and the report so that a
reader can see which checks ran and what they concluded, which is the difference between
software that is trustworthy and software that merely says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .geometry import fit_sphere_robust, regional_mask
from .landmarks import CoordinateSystem, LandmarkSet
from .meshio import Mesh, weld_vertices
from .sides import LPS_PATIENT_LEFT, LPS_SUPERIOR, Side

__all__ = [
    "Severity",
    "Finding",
    "QCReport",
    "MeshQuality",
    "assess_mesh_quality",
    "assess_femur_coverage",
    "assess_tibia_coverage",
    "check_plausible_bone_scale",
    "check_landmarks_on_mesh",
    "check_laterality",
]


# Anatomical plausibility bounds. Deliberately generous -- these catch unit errors and
# gross mistakes, not subtle ones, and a bound that trips on real anatomy is worse than
# no bound at all.
FEMUR_HEAD_RADIUS_MM = (18.0, 30.0)
PLAUSIBLE_BONE_LENGTH_MM = (60.0, 600.0)
PLAUSIBLE_EPICONDYLAR_WIDTH_MM = (35.0, 130.0)
LANDMARK_MESH_MARGIN_MM = 5.0

# Length below which the femoral head cannot be present. The head-to-condyle distance
# *is* the femur's length, 400-500 mm in adults, so any scan containing both must be at
# least that long. 350 mm is generous and still far above the 248 mm longest femur in
# this cohort, which matters -- see the note on gate strength below.
MIN_FEMUR_LENGTH_FOR_HEAD_MM = 350.0
MIN_TIBIA_LENGTH_FOR_ANKLE_MM = 300.0

# Sphere-fit acceptance for the femoral head.
#
# These thresholds were set by measurement, and the measurements were surprising enough
# to be worth recording, because the obvious choices are wrong.
#
# Forcing the sphere fit to run on the real knee-only femurs here (bypassing the length
# gate) fits the proximal shaft at radius 15-17 mm with RMS 0.6-1.5 mm. A short axial
# band of diaphysis is simply not obviously non-spherical. Two apparently sensible
# criteria turn out not to discriminate at all:
#
#   * Surface spread (wrap-around) is HIGHER for a shaft (0.87-0.99) than for a genuine
#     head cap (0.59-0.75), because a cylinder wraps completely around its axis.
#   * Direction isotropy, which was meant to separate a tube from a ball, overlaps
#     badly: synthetic shafts score 0.44-0.59 against head caps at 0.32-0.71.
#
# What does separate them is the residual, but only with a tighter bound than first
# chosen: genuine heads fit at RMS well under 0.5 mm, whereas every synthetic headless
# shaft that survived the radius bound sat at 1.12-1.50 mm. Hence 1.0 mm rather than
# 1.5 mm -- at 1.5 mm a thick-shafted headless femur passed every check.
#
# Even so, the length gate remains the primary protection, and deliberately so. It is
# anatomically airtight: the head-to-condyle distance IS the femur's length, so 350 mm
# of bone without a head is close to impossible in a scan that starts at the joint.
# Spread and isotropy are still computed and reported, because they are informative to
# a human reviewing a borderline case, but they are not gates. Anyone tempted to relax
# the length threshold should re-run the shaft sweep first.
HEAD_SPHERE_MAX_RMS_MM = 1.0


class Severity(Enum):
    """How much a finding should stop you.

    ``ERROR`` means the result would be meaningless, so the pipeline refuses.
    ``WARNING`` means it is computable but the reader must know. ``INFO`` records a
    check that ran and passed, so the report can show what was verified rather than
    only what failed.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One quality-control observation, destined for the plan and the report."""

    code: str
    severity: Severity
    message: str
    context: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


@dataclass
class QCReport:
    """A collection of findings with convenience accessors."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, code, severity, message, **context) -> None:
        self.findings.append(Finding(code, severity, message, context))

    def extend(self, other: "QCReport") -> None:
        self.findings.extend(other.findings)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise QualityControlError(self.errors)

    def to_dict(self) -> list[dict]:
        return [
            {"code": f.code, "severity": f.severity.value,
             "message": f.message, "context": f.context}
            for f in self.findings
        ]


class QualityControlError(ValueError):
    """Raised when a check finds something that makes the result meaningless."""

    def __init__(self, errors: list[Finding]):
        self.errors = errors
        super().__init__(
            "Quality control failed:\n" + "\n".join(f"  {f}" for f in errors)
        )


# ----------------------------------------------------------------------
# Mesh quality
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MeshQuality:
    """Topological state of a surface mesh.

    ``watertight`` is the property Blender's exact boolean solver wants. A mesh that is
    not closed can still be cut, but the result may be malformed in ways that are hard
    to see and easy to export, so the state is recorded in the plan either way.
    """

    n_triangles: int
    n_vertices: int
    n_boundary_edges: int
    n_nonmanifold_edges: int
    n_degenerate_faces: int
    n_duplicate_faces: int
    watertight: bool

    def to_dict(self) -> dict:
        return {
            "n_triangles": self.n_triangles,
            "n_vertices": self.n_vertices,
            "n_boundary_edges": self.n_boundary_edges,
            "n_nonmanifold_edges": self.n_nonmanifold_edges,
            "n_degenerate_faces": self.n_degenerate_faces,
            "n_duplicate_faces": self.n_duplicate_faces,
            "watertight": self.watertight,
        }


def assess_mesh_quality(
    mesh: Mesh, *, already_welded: bool = False
) -> tuple[MeshQuality, QCReport]:
    """Measure a mesh's topology and report anything that threatens a boolean cut.

    STL stores unwelded triangle soup, so vertices are merged first -- edge counts are
    meaningless otherwise, and every edge would look like a boundary.
    """
    report = QCReport()
    welded = mesh if already_welded else weld_vertices(mesh)

    faces = welded.faces
    if len(faces) == 0:
        report.add("MESH_EMPTY", Severity.ERROR, "Mesh contains no triangles.")
        return (
            MeshQuality(0, welded.n_vertices, 0, 0, 0, 0, watertight=False),
            report,
        )

    # A face is degenerate if it references the same vertex more than once.
    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    n_degenerate = int(degenerate.sum())
    valid_faces = faces[~degenerate]

    # Edges as undirected vertex pairs; how many faces share each one determines
    # whether the surface is closed there.
    edges = np.vstack([
        valid_faces[:, [0, 1]],
        valid_faces[:, [1, 2]],
        valid_faces[:, [2, 0]],
    ])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)

    n_boundary = int(np.sum(counts == 1))
    n_nonmanifold = int(np.sum(counts > 2))

    sorted_faces = np.sort(valid_faces, axis=1)
    n_unique_faces = len(np.unique(sorted_faces, axis=0))
    n_duplicate = int(len(sorted_faces) - n_unique_faces)

    watertight = n_boundary == 0 and n_nonmanifold == 0
    quality = MeshQuality(
        n_triangles=len(faces),
        n_vertices=welded.n_vertices,
        n_boundary_edges=n_boundary,
        n_nonmanifold_edges=n_nonmanifold,
        n_degenerate_faces=n_degenerate,
        n_duplicate_faces=n_duplicate,
        watertight=watertight,
    )

    if n_nonmanifold:
        report.add(
            "MESH_NONMANIFOLD", Severity.WARNING,
            f"{n_nonmanifold} non-manifold edges. Exact boolean cutting may fail and "
            f"fall back to the fast solver, which changes the resulting geometry.",
            n_nonmanifold_edges=n_nonmanifold,
        )
    if n_boundary:
        report.add(
            "MESH_NOT_CLOSED", Severity.WARNING,
            f"{n_boundary} boundary edges: the surface is not closed. Cut volumes are "
            f"ill-defined against an open surface.",
            n_boundary_edges=n_boundary,
        )
    if n_degenerate:
        report.add(
            "MESH_DEGENERATE_FACES", Severity.WARNING,
            f"{n_degenerate} degenerate triangles were ignored.",
            n_degenerate_faces=n_degenerate,
        )
    if watertight:
        report.add(
            "MESH_WATERTIGHT", Severity.INFO,
            f"Mesh is closed and manifold ({len(faces)} triangles).",
            n_triangles=len(faces),
        )
    return quality, report


# ----------------------------------------------------------------------
# Scan coverage
# ----------------------------------------------------------------------


def assess_femur_coverage(
    mesh: Mesh,
    *,
    superior: np.ndarray = LPS_SUPERIOR,
    proximal_fraction: float = 0.15,
) -> tuple[dict, QCReport]:
    """Decide whether the femoral head is in the scan, from the geometry alone.

    Two independent conditions must both hold:

    1. the bone is at least :data:`MIN_FEMUR_LENGTH_FOR_HEAD_MM` long, since a head
       cannot be present in a segment far too short to reach the hip; and
    2. a sphere fitted to the proximal end converges with RMS below
       :data:`HEAD_SPHERE_MAX_RMS_MM` at a radius within
       :data:`FEMUR_HEAD_RADIUS_MM`.

    Requiring both matters. Length alone would accept a long diaphysis; the fit alone
    could be fooled by a condylar surface, which is locally spherical at roughly the
    right radius. Together they are hard to satisfy by accident.

    When the head is absent the femoral mechanical axis is not measurable, and every
    metric depending on it -- HKA, mLDFA -- degrades or becomes unavailable.
    """
    report = QCReport()
    vertices = mesh.vertices
    projections = vertices @ superior
    length = float(projections.max() - projections.min())

    coverage = {
        "length_mm": round(length, 1),
        "femoral_head_present": False,
        "detection": "",
    }

    if length < MIN_FEMUR_LENGTH_FOR_HEAD_MM:
        coverage["detection"] = (
            f"bone length {length:.0f} mm is below the "
            f"{MIN_FEMUR_LENGTH_FOR_HEAD_MM:.0f} mm threshold; proximal sphere fit "
            f"not attempted"
        )
        report.add(
            "SCAN_KNEE_ONLY_FEMUR", Severity.INFO,
            f"Femoral head is outside the scan (segment is {length:.0f} mm; a whole "
            f"femur is 400-500 mm). The femoral mechanical axis cannot be measured "
            f"from this data.",
            length_mm=round(length, 1),
        )
        return coverage, report

    proximal = vertices[regional_mask(vertices, superior,
                                      fraction=proximal_fraction, end="high")]
    try:
        # Robust rather than plain least squares: the proximal window inevitably also
        # contains neck, trochanter and some shaft, and their pull on an unweighted fit
        # is enough to turn a real head into a failed detection.
        fit = fit_sphere_robust(proximal)
    except ValueError as error:
        coverage["detection"] = f"proximal sphere fit failed: {error}"
        report.add(
            "SCAN_HEAD_FIT_FAILED", Severity.INFO,
            "The proximal end does not admit a sphere fit; treating the femoral head "
            "as absent.",
        )
        return coverage, report

    radius_ok = FEMUR_HEAD_RADIUS_MM[0] <= fit.radius_mm <= FEMUR_HEAD_RADIUS_MM[1]
    rms_ok = fit.rms_mm <= HEAD_SPHERE_MAX_RMS_MM
    present = bool(radius_ok and rms_ok and fit.converged)

    coverage.update({
        "femoral_head_present": present,
        "head_fit_radius_mm": round(fit.radius_mm, 2),
        "head_fit_rms_mm": round(fit.rms_mm, 3),
        # Reported for a human reviewing a borderline case, not used as gates -- see
        # the threshold note above for why neither discriminates.
        "head_fit_surface_spread": round(fit.surface_spread, 3),
        "head_fit_direction_isotropy": round(fit.direction_isotropy, 3),
        "detection": (
            f"robust proximal sphere fit: radius {fit.radius_mm:.1f} mm "
            f"(accept {FEMUR_HEAD_RADIUS_MM[0]:.0f}-{FEMUR_HEAD_RADIUS_MM[1]:.0f}), "
            f"RMS {fit.rms_mm:.2f} mm (accept <= {HEAD_SPHERE_MAX_RMS_MM})"
        ),
    })

    if present:
        report.add(
            "SCAN_FEMORAL_HEAD_PRESENT", Severity.INFO,
            f"Femoral head detected: radius {fit.radius_mm:.1f} mm, RMS "
            f"{fit.rms_mm:.2f} mm. The femoral mechanical axis is measurable.",
            **{k: coverage[k] for k in ("head_fit_radius_mm", "head_fit_rms_mm")},
        )
    else:
        report.add(
            "SCAN_NO_FEMORAL_HEAD", Severity.INFO,
            f"No femoral head found despite sufficient length: {coverage['detection']}.",
        )
    return coverage, report


def assess_tibia_coverage(
    mesh: Mesh, *, superior: np.ndarray = LPS_SUPERIOR
) -> tuple[dict, QCReport]:
    """Decide whether the malleoli are in the scan.

    Length is the only test here. Unlike the femoral head there is no compact shape to
    fit -- the malleoli are a pair of flares, not a ball -- so a geometric confirmation
    would be far less decisive than the sphere fit and is not attempted. Since every
    case in this cohort falls well short of the threshold, a more elaborate test would
    add risk without changing any answer.
    """
    report = QCReport()
    projections = mesh.vertices @ superior
    length = float(projections.max() - projections.min())

    present = length >= MIN_TIBIA_LENGTH_FOR_ANKLE_MM
    coverage = {
        "length_mm": round(length, 1),
        "malleoli_present": bool(present),
        "detection": (
            f"bone length {length:.0f} mm against a "
            f"{MIN_TIBIA_LENGTH_FOR_ANKLE_MM:.0f} mm threshold"
        ),
    }

    if not present:
        report.add(
            "SCAN_KNEE_ONLY_TIBIA", Severity.INFO,
            f"Malleoli are outside the scan (segment is {length:.0f} mm). The tibial "
            f"mechanical axis cannot be measured directly, though the proximal "
            f"anatomical axis is a close proxy for it.",
            length_mm=round(length, 1),
        )
    return coverage, report


# ----------------------------------------------------------------------
# Sanity gates
# ----------------------------------------------------------------------


def check_plausible_bone_scale(mesh: Mesh, bone: str) -> QCReport:
    """Catch unit errors before they become plausible-looking anatomy.

    A mesh accidentally supplied in metres has an extent around 0.2 rather than 200.
    Everything downstream would still run and produce numbers, so the failure has to be
    caught here or not at all.
    """
    report = QCReport()
    extent = mesh.extent
    longest = float(extent.max())
    low, high = PLAUSIBLE_BONE_LENGTH_MM

    if longest < low:
        report.add(
            "SCALE_TOO_SMALL", Severity.ERROR,
            f"{bone} spans only {longest:.3f} mm. This is almost certainly metres "
            f"rather than millimetres (a factor of 1000). This pipeline works "
            f"exclusively in millimetres.",
            longest_extent=longest, bone=bone,
        )
    elif longest > high:
        report.add(
            "SCALE_TOO_LARGE", Severity.ERROR,
            f"{bone} spans {longest:.0f} mm, beyond any human long bone. Check the "
            f"units and that the file contains a single bone.",
            longest_extent=longest, bone=bone,
        )
    return report


def check_laterality(
    mesh: Mesh,
    side: Side,
    *,
    coordinate_system: CoordinateSystem = CoordinateSystem.LPS,
) -> QCReport:
    """Verify the declared side agrees with where the bone sits in the patient frame.

    In LPS, +X is patient-left, so a left knee lies at positive X about the midline and
    a right knee at negative X. This was confirmed across all twenty meshes in the
    cohort: left knees occupy X from +21 to +137 mm and right knees -135 to -25 mm,
    with no overlap.

    The check catches a mislabelled side and, just as importantly, an LPS/RAS mix-up --
    reading RAS landmarks as LPS mirrors the anatomy, which shows up here as a bone
    apparently on the wrong side of the body.

    It is a warning rather than an error because a scan cropped entirely to one side of
    the midline, or reconstructed about a shifted origin, could legitimately trip it.
    """
    report = QCReport()
    if coordinate_system is not CoordinateSystem.LPS:
        report.add(
            "LATERALITY_NOT_CHECKED", Severity.INFO,
            f"Laterality check applies to LPS only; mesh is declared "
            f"{coordinate_system.value}.",
        )
        return report

    centroid_x = float(mesh.vertices.mean(axis=0) @ LPS_PATIENT_LEFT)
    expected_positive = side is Side.LEFT
    observed_positive = centroid_x > 0

    if expected_positive != observed_positive:
        report.add(
            "LATERALITY_MISMATCH", Severity.WARNING,
            f"Declared side is {side}, but the bone's centroid sits at "
            f"X = {centroid_x:+.1f} mm in LPS, where a {side} knee is expected at "
            f"X {'> 0' if expected_positive else '< 0'}. Either the side is wrong or "
            f"the coordinate system is: reading RAS data as LPS mirrors the anatomy.",
            centroid_x_mm=round(centroid_x, 1), declared_side=str(side),
        )
    else:
        report.add(
            "LATERALITY_CONSISTENT", Severity.INFO,
            f"Declared side {side} agrees with the bone's position "
            f"(X = {centroid_x:+.1f} mm in LPS).",
            centroid_x_mm=round(centroid_x, 1),
        )
    return report


def check_landmarks_on_mesh(
    landmarks: LandmarkSet,
    meshes: dict[str, Mesh],
    *,
    margin_mm: float = LANDMARK_MESH_MARGIN_MM,
) -> QCReport:
    """Verify landmarks actually fall on the bones they claim to describe.

    This is the decisive coordinate-system gate. An LPS/RAS mismatch displaces landmarks
    by twice their distance from the midline -- typically 100 to 300 mm -- so they land
    nowhere near the bone. The check costs nothing and catches the single most damaging
    silent failure in the pipeline.

    A bounding-box test rather than a surface-distance one is deliberate: some landmarks
    (canal centres, the femoral head centre) are legitimately *inside* the bone or off
    its surface, so proximity to the surface is the wrong question. Being nowhere near
    the bone at all is the right one.

    ``meshes`` maps bone name to mesh; landmarks for bones not supplied are skipped.
    """
    report = QCReport()

    bounds = {
        bone: (mesh.vertices.min(axis=0) - margin_mm,
               mesh.vertices.max(axis=0) + margin_mm)
        for bone, mesh in meshes.items()
    }

    outside = []
    checked = 0
    for landmark in landmarks:
        if not landmark.is_usable:
            continue
        bone = landmark.definition.bone
        if bone not in bounds:
            continue

        checked += 1
        low, high = bounds[bone]
        position = landmark.position_mm
        if np.any(position < low) or np.any(position > high):
            excursion = float(
                np.max(np.maximum(low - position, position - high))
            )
            outside.append((landmark.id, excursion))

    if outside:
        worst = max(excursion for _, excursion in outside)
        report.add(
            "LANDMARKS_OFF_MESH", Severity.ERROR,
            f"{len(outside)} of {checked} landmarks lie outside their bone's bounding "
            f"box, the furthest by {worst:.0f} mm. The usual cause is a coordinate "
            f"system mismatch: landmarks stored as RAS but read as LPS (or the "
            f"reverse) are mirrored about the midline. Check the 'coordinateSystem' "
            f"field of the markup file against the mesh.",
            offenders=[landmark_id for landmark_id, _ in outside],
            worst_excursion_mm=round(worst, 1),
        )
    elif checked:
        report.add(
            "LANDMARKS_ON_MESH", Severity.INFO,
            f"All {checked} landmarks fall within their bone's bounds.",
            n_checked=checked,
        )
    return report
