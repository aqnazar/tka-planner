"""Fusion 360 script: write the open design's parameters to a JSON file.

Run it once on the implant model (Utilities > Scripts and Add-Ins > Scripts > +, pick
this folder, then Run). It asks where to save and writes every user parameter -- name,
expression, value, unit and comment -- plus the model parameters that carry a name
someone gave them. That list is what the parameter mapping is written from.
"""

import json

import adsk.core
import adsk.fusion


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("Open the implant design first.")
            return

        units = design.unitsManager

        def record(parameter, kind):
            return {
                "kind": kind,
                "name": parameter.name,
                "expression": parameter.expression,
                "value": units.formatInternalValue(parameter.value, parameter.unit, False)
                if parameter.unit else parameter.value,
                "unit": parameter.unit,
                "comment": parameter.comment,
            }

        parameters = [record(p, "user") for p in design.userParameters]
        # Model parameters are named d1, d2 ... unless someone renamed them, and a
        # renamed one is a dimension the designer meant to be driven.
        for p in design.allParameters:
            if p.name and not (p.name.startswith("d") and p.name[1:].isdigit()) \
                    and adsk.fusion.ModelParameter.cast(p) is not None:
                parameters.append(record(p, "model"))

        dialog = ui.createFileDialog()
        dialog.title = "Save the design's parameters"
        dialog.filter = "JSON (*.json)"
        dialog.initialFilename = f"{design.rootComponent.name}_parameters.json"
        if dialog.showSave() != adsk.core.DialogResults.DialogOK:
            return
        with open(dialog.filename, "w", encoding="utf-8") as handle:
            json.dump({"design": design.rootComponent.name,
                       "parameters": parameters}, handle, indent=2)
        ui.messageBox(f"{len(parameters)} parameters written to\n{dialog.filename}")
    except Exception as error:  # Fusion shows nothing for an uncaught error
        ui.messageBox(f"Failed: {error}")
