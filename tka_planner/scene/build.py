"""Turn a computed plan into a scene, without a renderer.

This is the Blender builder's job with Blender taken out. Everything visible was
decided in :mod:`tka_planner.core` before this module ran; the builder only places it.
That inversion is what kept the planning logic testable, and it is why removing Blender
costs nothing here.

The bones are placed uncut. Cutting is :mod:`tka_planner.scene.resect`, and it happens
once, on commit, rather than on every change of a control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tka_planner.core.meshio import read_stl
from tka_planner.geom import mesh as gm

from . import motion
from .model import FEMORAL, TIBIAL, Node, Scene

__all__ = ["build_scene", "resolve_component_meshes", "INSERT_SPACER"]

INSERT_SPACER = "TibialInsertSpacer"

# One colour per category, carried over from the Blender scene so the two look alike
# while both exist: bone grey, implant blue, cutting-block green, shell red, plane cyan,
# axis orange, landmark yellow. Cut bone keeps its own warmer grey, so a resected
# surface still reads as bone rather than as implant.
BONE_COLOUR = (0.62, 0.62, 0.62)
RESECTED_COLOUR = (0.55, 0.52, 0.50)
IMPLANT_COLOUR = (0.18, 0.38, 0.85)
PLANE_COLOUR = (0.25, 0.80, 0.85)
AXIS_COLOUR = (0.95, 0.55, 0.10)
BLOCK_COLOUR = (0.30, 0.70, 0.35)
SHELL_COLOUR = (0.85, 0.18, 0.18)
# Landmark colour is yellow throughout, per-status shade rather than a different hue,
# so provenance still reads at a glance without leaving the yellow family the category
# is known by.
LANDMARK_COLOURS = {
    "present": (1.00, 0.85, 0.05),
    "estimated": (0.85, 0.60, 0.05),
    "derived": (1.00, 0.95, 0.45),
}

PLANE_RADIUS_MM = 55.0
AXIS_LENGTH_MM = 150.0
AXIS_RADIUS_MM = 1.2
LANDMARK_RADIUS_MM = 2.5


def build_scene(
    *,
    femur_path,
    tibia_path,
    plan,
    femoral_frame,
    tibial_frame,
    landmarks=None,
    components: dict | None = None,
    show_planes: bool = True,
    show_axes: bool = True,
    show_landmarks: bool = False,
    insert_thickness_mm: float | None = None,
    insert_footprint_mm: tuple | None = None,
) -> Scene:
    """Build the full scene from an already-computed plan.

    ``components`` maps a component name to ``{"path": ..., "scale": ..., "group": ...}``
    as :func:`resolve_component_meshes` returns, so the implants appear seated on their
    cuts. Omit it to show the anatomy and the planned cuts alone.
    """
    scene = Scene()

    scene.add(
        Node(name="Femur", set_id=FEMORAL, colour=BONE_COLOUR, tags={"bone": True}),
        read_stl(femur_path),
    )
    scene.add(
        Node(name="Tibia", set_id=TIBIAL, colour=BONE_COLOUR, tags={"bone": True}),
        read_stl(tibia_path),
    )

    if show_planes:
        plane_mesh = gm.disc(PLANE_RADIUS_MM)
        for name, resection in plan.resections.items():
            scene.add(
                Node(
                    name=name,
                    set_id=_set_for(name),
                    rest=gm.plane_matrix(resection.point, resection.normal),
                    colour=PLANE_COLOUR,
                    alpha=0.35,
                    tags={"plane": True, "edit_only": True},
                ),
                plane_mesh,
            )

    if show_axes:
        for label, frame, set_id, length in (
            ("FemoralMechanicalAxis", femoral_frame, FEMORAL, AXIS_LENGTH_MM),
            ("TibialMechanicalAxis", tibial_frame, TIBIAL, -AXIS_LENGTH_MM),
        ):
            direction = np.asarray(frame.z_proximal, dtype=float)
            direction = direction / np.linalg.norm(direction)
            midpoint = np.asarray(frame.origin, dtype=float) + direction * (length / 2.0)
            scene.add(
                Node(
                    name=label,
                    set_id=set_id,
                    rest=gm.plane_matrix(midpoint, direction),
                    colour=AXIS_COLOUR,
                    tags={"axis": True, "edit_only": True},
                ),
                gm.cylinder(AXIS_RADIUS_MM, abs(length)),
            )

    if landmarks is not None:
        # One shared sphere, instanced by rest transform. The Blender builder made a
        # separate mesh per landmark so each could be clicked in the outliner; a viewer
        # that can pick by node name does not need that, and one mesh draws in one call.
        marker = gm.uv_sphere(LANDMARK_RADIUS_MM)
        for landmark in landmarks:
            if not landmark.is_usable:
                continue
            scene.add(
                Node(
                    name=landmark.id,
                    set_id=_set_for(landmark.id),
                    rest=gm.translation(landmark.position_mm),
                    colour=LANDMARK_COLOURS.get(
                        landmark.status.value, (0.9, 0.8, 0.2)
                    ),
                    visible=show_landmarks,
                    tags={"landmark": True, "status": landmark.status.value},
                ),
                marker,
            )

    # Every part belonging to a bone shares that bone's CAD origin, so they all take the
    # same pose. That is the library's own guarantee and it is what makes a cutting
    # block land exactly on its implant, without any per-part offsets.
    for name, spec in (components or {}).items():
        group = spec.get("group", "femoral")
        pose = plan.components.get(f"{group}_component")
        if pose is None or not Path(spec["path"]).is_file():
            scene.notes.append(f"{name}: mesh not found, skipped")
            continue
        scene.add(
            Node(
                name=name,
                set_id=FEMORAL if group == "femoral" else TIBIAL,
                rest=seat(pose, spec.get("scale", 1.0)),
                colour=_component_colour(name),
                tags={
                    "component": True,
                    "group": group,
                    "scale": float(spec.get("scale", 1.0)),
                    "source_ml": spec.get("source_ml"),
                    "shell": "shell" in name,
                },
            ),
            read_stl(spec["path"]),
        )

    if insert_thickness_mm is not None:
        ml_mm, ap_mm = insert_footprint_mm or (70.0, 48.0)
        scene.add(
            Node(
                name=INSERT_SPACER,
                set_id=TIBIAL,
                rest=seat(plan.components["tibial_component"], 1.0),
                colour=IMPLANT_COLOUR,
                alpha=0.6,
                tags={"insert": True, "footprint_mm": (ml_mm, ap_mm)},
            ),
            gm.slab(ml_mm, ap_mm, insert_thickness_mm),
        )

    origin, direction = motion.flexion_axis(plan, femoral_frame, landmarks)
    scene.pivot = motion.pivot_frame(origin, direction)
    return scene


def seat(pose, scale_factor: float = 1.0) -> np.ndarray:
    """Put a component's CAD origin on a cut, at a parametric scale.

    ``scale_factor`` is what makes the family parametric: the exported library is a
    single master geometry under a uniform scale, so any size between or beyond the
    twelve published ones is that master at the appropriate factor.

    Scaling is about the CAD origin, so the parametric factor never shifts the seating.
    Placement uses the component's native CAD origin, untouched. An earlier version
    re-originned each mesh to its bounding box instead, which put the femoral component
    42 mm below its cut and 32 mm to the side, because a femoral component's highest
    points are the anterior flange and the posterior condyles while its mating surface
    sits between them.
    """
    pose = np.asarray(pose, dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = pose[:3, :3] * float(scale_factor)
    matrix[:3, 3] = pose[:3, 3]
    return matrix


def _set_for(name: str) -> str:
    """Which bone set a plane or a landmark belongs to, from its id."""
    text = str(name).lower()
    if text.startswith(("tibia", "fibula")) or "tibial" in text:
        return TIBIAL
    return FEMORAL


def _component_colour(name: str):
    if "shell" in name:
        return SHELL_COLOUR
    if "cutting_block" in name:
        return BLOCK_COLOUR
    return IMPLANT_COLOUR


def resolve_component_meshes(
    library: "str | Path",
    *,
    chart,
    sizing,
    side: str,
) -> dict:
    """Find the implant meshes for a plan and the scale that realises its exact size.

    The library holds twelve discrete exports, and the plan asks for a continuous size.
    Because those exports are one master geometry under a uniform scale, the nearest
    discrete mesh multiplied by the ratio of the two widths *is* the requested size --
    not an approximation to it. When the plan lands on a chart size the ratio is one and
    the mesh is used untouched.
    """
    library = Path(library)
    label = sizing.nearest_discrete_size
    if not (library / label).is_dir():
        return {}
    is_left = str(side).lower().startswith("l")

    # The exported folders are not uniformly complete -- L4, for instance, holds the
    # implants but no cutting blocks. Rather than drop those parts, search outwards from
    # the nearest size for a folder that does export them, and scale from whichever size
    # was found. Because the library is one master geometry under a uniform scale, a part
    # taken from a neighbouring size and rescaled is the same part.
    order = sorted(
        (lbl for lbl in chart.labels if (library / lbl).is_dir()),
        key=lambda lbl: abs(chart.label_parameter(lbl)
                            - chart.label_parameter(label)),
    )

    # Filenames are matched on content rather than exact spelling, because the exported
    # library is not internally consistent: the M2 folder holds
    # "Implant(Femoral)Left_M2.stl" while L4 holds "Implant_Femoral_Left.stl" -- same
    # component, different separators and no size suffix. Normalising to alphanumerics
    # makes both resolve.
    def normalise(name: str) -> str:
        return "".join(c for c in name.lower() if c.isalnum())

    def find_in(folder_label, *, must, side_token, exclude=()):
        files = [
            (normalise(p.name), p)
            for p in (library / folder_label).glob("*.stl")
        ]
        matches = [
            p for key, p in files
            if all(term in key for term in must)
            and not any(term in key for term in exclude)
        ]
        if side_token:
            sided = [p for p in matches if side_token in normalise(p.name)]
            other = "right" if side_token == "left" else "left"
            sided = [p for p in sided if other not in normalise(p.name)]
            if sided:
                return sided[0]
            # Fall back to a laterality-neutral export, e.g. "(L&R)".
            neutral = [
                p for p in matches
                if "left" not in normalise(p.name) and "right" not in normalise(p.name)
            ]
            return neutral[0] if neutral else None
        return matches[0] if matches else None

    def find(*, must, side_token, exclude=()):
        """Search outwards from the requested size until the part turns up."""
        for folder_label in order:
            found = find_in(
                folder_label, must=must, side_token=side_token, exclude=exclude
            )
            if found is not None:
                return found, folder_label
        return None, None

    resolved = {}

    # Every part belonging to a bone shares that bone's CAD origin, so they all take the
    # same pose. That is the library's own guarantee and it is what makes a cutting block
    # land exactly on its implant.
    wanted = (
        ("femoral_component", ("implant", "femoral"), "femoral", True),
        ("femoral_cutting_block", ("cuttingblock", "femoral"), "femoral", False),
        ("femoral_cutting_block_shell", ("cuttingblock", "femoral", "shell"),
         "femoral", False),
        ("tibial_component", ("tibial", "plate"), "tibial", False),
        ("tibial_insert", ("tibial", "insert"), "tibial", False),
        ("tibial_cutting_block", ("cuttingblock", "tibia"), "tibial", False),
        ("tibial_cutting_block_shell", ("cuttingblock", "tibia", "shell"),
         "tibial", False),
    )
    for name, must, group, sided in wanted:
        path, found_label = find(
            must=must,
            side_token=("left" if is_left else "right") if sided else None,
            exclude=() if "shell" in name else ("shell",),
        )
        if path is not None:
            source_ml = chart.value_at(
                "femur_ML", chart.label_parameter(found_label)
            )
            resolved[name] = {
                "path": str(path),
                "scale": sizing.implant_ml_mm / source_ml,
                "group": group,
                "source_size": found_label,
                "source_ml": source_ml,
            }

    # The plastic insert is sometimes exported without "tibial" in its name.
    if "tibial_insert" not in resolved:
        path, found_label = find(
            must=("insert",), side_token=None, exclude=("shell",)
        )
        if path is not None:
            source_ml = chart.value_at(
                "femur_ML", chart.label_parameter(found_label)
            )
            resolved["tibial_insert"] = {
                "path": str(path),
                "scale": sizing.implant_ml_mm / source_ml,
                "group": "tibial",
                "source_size": found_label,
                "source_ml": source_ml,
            }
    return resolved


