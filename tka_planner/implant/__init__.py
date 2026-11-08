"""The patient-specific implant: the Fusion sizing rules, and the parts built from them.

Pure numpy. ``derive_parameters`` turns a measured, planned knee into the parameters of
the three parametric Fusion designs (femoral component, tibial plate, insert);
:mod:`.builds` finds the parts Fusion built for a knee and says whether they still match
the plan. The Fusion transport and build scripts are :mod:`tka_planner.fusion`.
"""

from .rules import derive_parameters, femoral_fit, ml_slide

__all__ = ["derive_parameters", "femoral_fit", "ml_slide"]
