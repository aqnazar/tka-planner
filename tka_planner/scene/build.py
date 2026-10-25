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

from tka_planner.core.meshio import Mesh, read_stl
from tka_planner.geom import mesh as gm

from . import motion
from .model import FEMORAL, TIBIAL, Node, Scene

__all__ = ["build_scene", "resolve_component_meshes", "INSERT_SPACER"]

INSERT_SPACER = "TibialInsertSpacer"

# One colour per category, carried over from the Blender scene so the two look alike
# while both exist: bone grey, implant blue, cutting-block green, shell deep orange,
# plane cyan, axis orange, landmark yellow. Cut bone keeps its own warmer grey, so a
# resected surface still reads as bone rather than as implant. The shell was red until
# 2026-10-17, which the fit map also uses for overhang (author's ruling: orange).
BONE_COLOUR = (0.62, 0.62, 0.62)
RESECTED_COLOUR = (0.55, 0.52, 0.50)
IMPLANT_COLOUR = (0.18, 0.38, 0.85)
PLANE_COLOUR = (0.25, 0.80, 0.85)
AXIS_COLOUR = (0.95, 0.55, 0.10)
BLOCK_COLOUR = (0.30, 0.70, 0.35)
# The phantom femur is not anatomy, so it takes no bone or implant colour: a pale violet,
# drawn translucent, that nothing else in the scene uses.
PHANTOM_COLOUR = (0.62, 0.55, 0.85)
SHELL_COLOUR = (0.93, 0.42, 0.05)
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
    show_cutting_blocks: bool = True,
    phantom=None,
    show_phantom: bool = True,
    insert_thickness_mm: float | None = None,
    insert_footprint_mm: tuple | None = None,
    insert_stretch: dict | None = None,
    component_scales: dict | None = None,
    femur_mesh: Mesh | None = None,
    tibia_mesh: Mesh | None = None,
) -> Scene:
    """Build the full scene from an already-computed plan.

    ``femur_mesh`` and ``tibia_mesh`` are the bones as measured -- with the detached
    pieces quality control removed -- so the scene shows and cuts what was measured.
    Without them the files are read as they are.

    ``components`` maps a component name to ``{"path": ..., "scale": ..., "group": ...}``
    as :func:`resolve_component_meshes` returns, so the implants appear seated on their
    cuts. Omit it to show the anatomy and the planned cuts alone.
    """
    scene = Scene()

    scene.add(
        Node(name="Femur", set_id=FEMORAL, colour=BONE_COLOUR, tags={"bone": True}),
        femur_mesh if femur_mesh is not None else read_stl(femur_path),
    )
    scene.add(
        Node(name="Tibia", set_id=TIBIAL, colour=BONE_COLOUR, tags={"bone": True}),
        tibia_mesh if tibia_mesh is not None else read_stl(tibia_path),
    )

    # Planes and axes are always built and their toggles set visibility, rather than
    # the toggles deciding whether they exist. A node that is not there cannot be shown
    # again without a rebuild, and rebuilding re-reads the segmentation.
    plane_mesh = gm.disc(PLANE_RADIUS_MM)
    for name, resection in plan.resections.items():
        scene.add(
            Node(
                name=name,
                set_id=_set_for(name),
                rest=gm.plane_matrix(resection.point, resection.normal),
                colour=PLANE_COLOUR,
                alpha=0.35,
                visible=show_planes,
                tags={"plane": True, "edit_only": True},
            ),
            plane_mesh,
        )

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
                visible=show_axes,
                tags={"axis": True, "edit_only": True},
            ),
            gm.cylinder(AXIS_RADIUS_MM, abs(length)),
        )

    if phantom is not None:
        # Already in the scan's frame, so the node sits at the origin with its mesh.
        from tka_planner.geom.phantom import phantom_mesh

        scene.add(
            Node(
                name="PhantomFemur",
                set_id=FEMORAL,
                rest=np.eye(4),
                colour=PHANTOM_COLOUR,
                alpha=0.45,
                visible=show_phantom,
                tags={"phantom": True, "schematic": True},
            ),
            phantom_mesh(phantom),
        )
        scene.notes.append(
            "Phantom femur: a schematic drawn from the entered femur length and "
            "population priors, not measured anatomy (hip-centre error about "
            f"{phantom.budget['combined']:g} mm, one SD).")

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
        transforms = component_scales or {}
        scale = transforms.get("scales", {}).get(group, spec.get("scale", 1.0))
        rest = place(pose, scale, transforms.get("shifts", {}).get(group))
        if name == "tibial_insert":
            rest = rest @ insert_stretch_matrix(insert_stretch)
        scene.add(
            Node(
                name=name,
                set_id=FEMORAL if group == "femoral" else TIBIAL,
                rest=rest,
                colour=_component_colour(name),
                visible=show_cutting_blocks or "cutting_block" not in name,
                tags={
                    "component": True,
                    "group": group,
                    "scale": float(spec.get("scale", 1.0)),
                    "source_ml": spec.get("source_ml"),
                    "shell": "shell" in name,
                    "cutting_block": "cutting_block" in name,
                },
            ),
            read_component(spec),
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

    ``scale_factor`` may also be three factors, one per CAD axis: the patient-specific
    implant is the library part stretched to the patient's own width and depth, which a
    single factor cannot express.

    Scaling is about the CAD origin, so the parametric factor never shifts the seating.
    Placement uses the component's native CAD origin, untouched. An earlier version
    re-originned each mesh to its bounding box instead, which put the femoral component
    42 mm below its cut and 32 mm to the side, because a femoral component's highest
    points are the anterior flange and the posterior condyles while its mating surface
    sits between them.
    """
    pose = np.asarray(pose, dtype=float)
    factors = np.broadcast_to(np.asarray(scale_factor, dtype=float), (3,))
    matrix = np.eye(4)
    matrix[:3, :3] = pose[:3, :3] * factors[None, :]
    matrix[:3, 3] = pose[:3, 3]
    return matrix


def place(pose, scale_factor=1.0, shift=None) -> np.ndarray:
    """:func:`seat`, then slide the part by ``shift`` along its own CAD axes, in mm.

    The patient-specific implant is the library part stretched to the patient and slid
    onto the centre of the cut, and the slide is along the component's own axes so it
    stays in the plane of the cut.
    """
    matrix = seat(pose, scale_factor)
    if shift is not None:
        rotation = np.asarray(pose, dtype=float)[:3, :3]
        rotation = rotation / np.linalg.norm(rotation, axis=0)[None, :]
        matrix[:3, 3] = matrix[:3, 3] + rotation @ np.asarray(shift, dtype=float)
    return matrix


def insert_stretch_matrix(stretch: dict | None) -> np.ndarray:
    """The insert's height change as a matrix in its own CAD frame; identity if none."""
    if not stretch:
        return np.eye(4)
    from tka_planner.core.insert import stretch_matrix

    return stretch_matrix(stretch["floor_cad_mm"], stretch["stretch"])


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
            other = "right" if side_token == "left" else "left"

            def handed(token, against):
                return [p for p in matches if token in normalise(p.name)
                        and against not in normalise(p.name)]

            sided = handed(side_token, other)
            if sided:
                return sided[0], False
            # Fall back to a laterality-neutral export, e.g. "(L&R)".
            neutral = [
                p for p in matches
                if "left" not in normalise(p.name) and "right" not in normalise(p.name)
            ]
            if neutral:
                return neutral[0], False
            # Then to this size's other hand, mirrored. The library exports most sizes
            # left-handed only, and the right part is the left one reflected across the
            # sagittal plane (L4, which exports both, matches vertex for vertex). Taking
            # the other hand of the right size beats taking the right hand of the wrong
            # size: searching every size for a right part found only L4's, which is
            # narrower than the chart says, so every right knee came out 5% narrow.
            opposite = handed(other, side_token)
            return (opposite[0], True) if opposite else (None, False)
        return (matches[0] if matches else None), False

    def find(*, must, side_token, exclude=()):
        """Search outwards from the requested size until the part turns up."""
        for folder_label in order:
            found, mirror = find_in(
                folder_label, must=must, side_token=side_token, exclude=exclude
            )
            if found is not None:
                return found, folder_label, mirror
        return None, None, False

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
        path, found_label, mirror = find(
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
                "mirror": mirror,
            }

    # The plastic insert is sometimes exported without "tibial" in its name.
    if "tibial_insert" not in resolved:
        path, found_label, _ = find(
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
                "mirror": False,
            }
    return resolved


# The CAD axis that points to patient-left, per bone (see the component conventions):
# femoral parts are +X anterior, +Y patient-left; tibial parts +X patient-left.
_CAD_LATERAL_AXIS = {"femoral": 1, "tibial": 0}


def read_component(spec: dict, reader=None) -> Mesh:
    """Read a resolved library part, mirrored when the resolver asked for it.

    A mirrored part is the other hand's export reflected across the patient's sagittal
    plane: the CAD axis to patient-left is negated, and each triangle's winding reversed
    so its faces still point outwards.
    """
    mesh = (reader or read_stl)(spec["path"])
    if not spec.get("mirror"):
        return mesh
    axis = _CAD_LATERAL_AXIS[spec.get("group", "femoral")]
    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    vertices[:, axis] *= -1.0
    return Mesh(vertices=vertices, faces=np.ascontiguousarray(mesh.faces[:, ::-1]),
                source_path=mesh.source_path, sha256=mesh.sha256,
                metadata=dict(mesh.metadata, mirrored=True))


