"""Flexion and trial reduction, as matrices.

Both are rigid transforms of the tibial set about the flexion axis. The Blender version
achieved them with a parented empty, an animation action and a stack of baked meshes;
none of that was about the motion, all of it was about making Blender's modifier stack
stop re-evaluating during playback. With committed geometry there is nothing to
re-evaluate, so the motion is what it always was: a 4x4.

The composition rule is ``pose = pivot @ R @ pivot^-1``, so a pose is a function of the
control values alone. Setting the same values twice gives the same matrix, and zeroing
them gives the identity. The trial rig's drift bug is unreachable from here.
"""

from __future__ import annotations

import numpy as np

from tka_planner.geom.mesh import rotation_to

__all__ = [
    "flexion_axis", "pivot_frame", "pose_about", "flexion_pose", "trial_pose",
    "rotation_about",
]


def flexion_axis(plan, femoral_frame, landmarks) -> tuple:
    """Where the knee hinges, and about what.

    The transepicondylar axis, through the midpoint of the epicondyles. The femoral
    condyles are close to circular in the sagittal plane and the epicondyles sit near
    the centres of those circles, so the tibia rides around them at a nearly constant
    radius and stays in contact through the arc.

    Ported unchanged from the Blender builder, including its fallback to the femoral
    component origin when neither epicondyle pair is available. Using the posterior
    condyles instead put the axis about two centimetres off the centre of curvature,
    and the joint swung apart as it flexed.
    """
    direction = np.asarray(femoral_frame.y_patient_left, dtype=float)

    for pair in (
        ("femur.epicondyle_lateral", "femur.epicondyle_medial_sulcus"),
        ("femur.epicondyle_lateral", "femur.epicondyle_medial_prominence"),
    ):
        if landmarks is not None and landmarks.available(*pair):
            lateral, medial = landmarks.require(*pair)
            midpoint = (np.asarray(lateral) + np.asarray(medial)) / 2.0
            return midpoint, direction

    return (
        np.asarray(plan.components["femoral_component"][:3, 3], dtype=float),
        direction,
    )


def rotation_about(axis, degrees: float) -> np.ndarray:
    """A 3x3 rotation of ``degrees`` about ``axis``, by Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(float(degrees))
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def pivot_frame(origin_mm, direction) -> np.ndarray:
    """A 4x4 at the flexion axis whose +X runs along it.

    Flexion turns about local X, a varus or valgus stress about local Y, and the drawer
    slides along local Y. Naming the axes once here is what lets the pose functions
    below stay three lines each.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("The flexion axis direction must not be the zero vector.")

    # `rotation_to` takes +Z onto a direction. Pre-rotating by -90 degrees about Y
    # takes +X onto +Z first, so the composition takes +X onto the direction, which is
    # the convention the trial controls are written in.
    basis = rotation_to(direction / norm) @ rotation_about(
        np.array([0.0, 1.0, 0.0]), -90.0
    )
    frame = np.eye(4)
    frame[:3, :3] = basis
    frame[:3, 3] = np.asarray(origin_mm, dtype=float)
    return frame


def pose_about(pivot: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """A world pose applying ``rotation`` in the pivot's own frame, about its origin."""
    local = np.eye(4)
    local[:3, :3] = rotation
    return pivot @ local @ np.linalg.inv(pivot)


def flexion_pose(pivot: np.ndarray, *, flexion_deg: float) -> np.ndarray:
    """The tibial set pose at a point in the flexion arc."""
    return pose_about(pivot, rotation_about(np.array([1.0, 0.0, 0.0]), flexion_deg))


def trial_pose(
    pivot: np.ndarray,
    *,
    flexion_deg: float = 0.0,
    varus_valgus_deg: float = 0.0,
    drawer_ap_mm: float = 0.0,
) -> np.ndarray:
    """The tibial set pose under the three trial controls.

    Flexion turns about the pivot's local X. A varus or valgus stress then turns about
    the tibia's *already flexed* local Y, which is what a real stress exam is relative
    to: the tibia's current position, not the femur's fixed frame. Composing the stress
    on the right of the flexion is what puts it in the flexed frame.

    The drawer instead slides along the pivot's **rest** Y, because "anterior" for that
    test means the joint's own anterior rather than wherever flexion left the tibia
    pointing. That is why the translation is applied outside the rotation rather than
    within the pivot's local frame.
    """
    rotation = (
        rotation_about(np.array([1.0, 0.0, 0.0]), flexion_deg)
        @ rotation_about(np.array([0.0, 1.0, 0.0]), varus_valgus_deg)
    )
    turned = pose_about(pivot, rotation)

    slide = np.eye(4)
    slide[:3, 3] = pivot[:3, 1] * float(drawer_ap_mm)
    return slide @ turned
