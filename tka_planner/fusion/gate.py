"""The save gate: a rebuilt part that changed without a reason is not saved.

After a part is rebuilt and before it is saved, its body's volume and bounding box are
compared with the part's previous build. When the parameters that drive it are the same
as last time, the part should come out the same; a change above 2 % in volume or 1 mm in
any side of its bounding box then means a feature rebuilt differently (a lost reference,
a flipped arc), and the save is refused with the feature health in the report (author
2026-11-08). Changed parameters explain any change.

Plain Python with no imports: the build sends this module's functions into Fusion as
source, so the gate that runs there is the one tested here.
"""

GATE_VOLUME_FRACTION = 0.02
GATE_BBOX_MM = 1.0
# What drives each part: its own parameters, and for the insert and the blocks the parts
# they are built on.
PART_INPUTS = {"femoral": ("femoral",), "plate": ("plate",),
               "insert": ("insert", "femoral", "plate"),
               "block": ("femoral",), "tblock": ("plate",)}


def gate_body(bodies, name):
    """The body a part exports: by name, or the largest for "*"."""
    if not bodies:
        return None
    if name == "*":
        return max(bodies.values(), key=lambda b: b["vol"])
    return bodies.get(name)


def save_gate(part, prev_build, prev_job, job, bodies, body_name):
    """Whether ``part`` may be saved: ``{"ok", "checked", "reason"}``.

    ``prev_build`` and ``prev_job`` are the knee's last build report and job,
    ``bodies`` the rebuilt part's bodies ``{name: {"vol": mm3, "bb": [x0, y0, z0, x1,
    y1, z1]}}``, ``body_name`` the one it exports.
    """
    prev = (prev_build or {}).get(part) or {}
    if not prev.get("saved") or not prev.get("bodies"):
        return {"ok": True, "checked": False, "reason": "no previous build to compare with"}
    changed = [k for k in PART_INPUTS.get(part, (part,))
               if (prev_job or {}).get(k) != (job or {}).get(k)]
    if changed:
        return {"ok": True, "checked": False,
                "reason": "parameters changed: " + ", ".join(changed)}
    old, new = gate_body(prev["bodies"], body_name), gate_body(bodies, body_name)
    if new is None:
        return {"ok": False, "checked": True, "reason": f"no body {body_name!r} after the rebuild"}
    if old is None:
        return {"ok": True, "checked": False, "reason": f"no body {body_name!r} last time"}
    volume = abs(new["vol"] - old["vol"]) / max(abs(old["vol"]), 1e-9)
    sides_old = [old["bb"][i + 3] - old["bb"][i] for i in range(3)]
    sides_new = [new["bb"][i + 3] - new["bb"][i] for i in range(3)]
    side = max(abs(a - b) for a, b in zip(sides_old, sides_new))
    problems = []
    if volume > GATE_VOLUME_FRACTION:
        problems.append(f"volume {100 * (new['vol'] - old['vol']) / old['vol']:+.1f}% "
                        f"({old['vol']:.0f} -> {new['vol']:.0f} mm3)")
    if side > GATE_BBOX_MM:
        problems.append(f"bounding box side changed by {side:.1f} mm")
    if problems:
        return {"ok": False, "checked": True,
                "reason": "parameters unchanged but " + "; ".join(problems)
                          + " (limits 2% and 1 mm): not saved"}
    return {"ok": True, "checked": True,
            "reason": f"unchanged within limits (volume {100 * volume:.2f}%, side {side:.2f} mm)"}
