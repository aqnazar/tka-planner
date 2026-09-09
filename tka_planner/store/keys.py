"""The key that decides whether a commit has already been computed.

Two plans that would cut the bones identically must produce the same key, and two that
would not must differ. So the key is taken over what a boolean actually reads: the
source geometry, the resection mode, and the planes and poses the plan resolved to. It
is deliberately *not* taken over the control values.

That distinction matters more than it looks. Controls are an input to planning and
several of them are display-only, so a key over controls would miss the cache every time
somebody toggled the landmarks, and would also miss when two different routes through
the controls arrived at the same cut. Keying the output instead makes the cache exact.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

__all__ = ["plan_key"]

# Twelve decimal places on a millimetre is a picometre. Rounding there costs nothing
# clinically and stops the last bit of floating point noise from splitting a cache
# entry in two.
PLACES = 12


def plan_key(session) -> str:
    """A hex digest over everything the committed geometry depends on."""
    plan = session.plan
    if plan is None:
        raise ValueError("The session has no plan yet; call build() first.")

    payload = {
        "femur": _source_digest(session.femur_path),
        "tibia": _source_digest(session.tibia_path),
        "side": str(session.side),
        "mode": str(session.controls.resection_mode),
        "shells": bool(session.controls.build_bone_shells),
        "resections": {
            name: {
                "point": _round(resection.point),
                "normal": _round(_unit(resection.normal)),
            }
            for name, resection in sorted(plan.resections.items())
        },
        "components": {
            name: _round(pose) for name, pose in sorted(plan.components.items())
        },
    }
    # In block mode the cut is made by the implant's cutting block, so the implant size
    # is part of what the boolean reads even though it moves no plane.
    if session.sizing is not None:
        payload["implant_ml_mm"] = round(float(session.sizing.implant_ml_mm), PLACES)

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _source_digest(path) -> str:
    """Identify a source mesh by its file hash, falling back to its path.

    A case whose segmentation is re-exported must not reuse a commit computed from the
    previous export, and a file hash is the only thing that notices that.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return f"path:{path}"
    return digest.hexdigest()[:32]


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _round(value):
    return np.round(np.asarray(value, dtype=float), PLACES).tolist()
