"""Fusion 360 script: build a patient's implant from the planner's specification.

It asks for the patient's ``implant_spec.json`` (written by ``tka measure`` or by the
application's export) and for the parameter mapping of the open design, sets every mapped
parameter, and reports what it set and what it could not. Optionally it then exports the
design as STL.

The mapping is resolved by ``tka_planner.cad_bridge`` from this repository, the same code
``tka cad-params`` previews with, so the values set here are the values previewed there.
"""

import os
import sys

import adsk.core
import adsk.fusion

# This folder is fusion/TKA_ApplyImplantSpec inside the repository; the package is two
# levels up.
REPOSITORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPOSITORY not in sys.path:
    sys.path.insert(0, REPOSITORY)


def _pick(ui, title):
    dialog = ui.createFileDialog()
    dialog.title = title
    dialog.filter = "JSON (*.json)"
    if dialog.showOpen() != adsk.core.DialogResults.DialogOK:
        return None
    return dialog.filename


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        import json

        from tka_planner.cad_bridge import load_mapping, resolve_parameters

        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("Open the implant design first.")
            return

        spec_path = _pick(ui, "The patient's implant_spec.json")
        if not spec_path:
            return
        map_path = _pick(ui, "The parameter mapping for this design")
        if not map_path:
            return

        with open(spec_path, encoding="utf-8") as handle:
            spec = json.load(handle)
        mapping = load_mapping(map_path)

        set_, skipped = [], []
        for item in resolve_parameters(spec, mapping):
            parameter = design.allParameters.itemByName(item.fusion)
            if parameter is None:
                skipped.append(f"{item.fusion}: no such parameter in this design")
            elif item.expression is None:
                skipped.append(f"{item.fusion}: {item.problem} ({item.source})")
            else:
                parameter.expression = item.expression
                set_.append(f"{item.fusion} = {item.expression}")

        summary = (f"Case {spec.get('case_id')} ({spec.get('side')})\n\n"
                   f"Set {len(set_)}:\n" + "\n".join(set_))
        if skipped:
            summary += f"\n\nNot set {len(skipped)}:\n" + "\n".join(skipped)
        ui.messageBox(summary)

        folder = mapping.get("export_stl_to")
        if folder:
            folder = os.path.join(os.path.dirname(spec_path), folder)
            os.makedirs(folder, exist_ok=True)
            exporter = design.exportManager
            for body in design.rootComponent.bRepBodies:
                target = os.path.join(folder, f"{body.name}.stl")
                options = exporter.createSTLExportOptions(body, target)
                options.meshRefinement = \
                    adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
                exporter.execute(options)
            ui.messageBox(f"Exported the bodies as STL to\n{folder}")
    except Exception as error:  # Fusion shows nothing for an uncaught error
        ui.messageBox(f"Failed: {error}")
