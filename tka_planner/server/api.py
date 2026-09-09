"""The routes, as plain functions over a session manager.

Nothing in this module knows what carries it. A route takes a method, a path and a
decoded body, and returns a status and a payload; the HTTP adapter in ``httpd`` turns
that into a socket, and a test calls it directly. That is what keeps the standard-
library server a detail rather than a commitment: putting a different transport in front
is an adapter, not a rewrite.

Every route that changes something answers with the same envelope. The viewer therefore
has one function for applying a response no matter which control produced it, and there
is no route whose answer it has to special-case. The envelope carries the delta, the
clipping half-spaces, the report lines, both sets of control values and the stale flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from tka_planner.geom.gltf import write_glb
from tka_planner.report.bundle import export_session
from tka_planner.scene.model import SceneDelta

from .encode import clip_planes, scene_snapshot

__all__ = ["Api", "Binary", "ApiError"]


@dataclass
class Binary:
    """A response that is bytes rather than JSON."""

    content_type: str
    data: bytes
    immutable: bool = False


class ApiError(Exception):
    """A request that cannot be served, with the status it deserves."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Api:
    """Routes over a :class:`~tka_planner.server.sessions.SessionManager`."""

    def __init__(self, manager):
        self.manager = manager

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, method: str, path: str, body=None):
        """Route a request. Returns ``(status, payload)``; never raises for a caller."""
        try:
            return 200, self._route(method.upper(), _segments(path), body or {})
        except ApiError as error:
            return error.status, {"error": error.message}
        except KeyError as error:
            return 404, {"error": str(error)}
        except (ValueError, TypeError) as error:
            return 400, {"error": str(error)}
        except FileNotFoundError as error:
            return 404, {"error": str(error)}

    def _route(self, method, parts, body):
        if not parts or parts[0] != "api":
            raise ApiError(404, f"No route for /{'/'.join(parts)}")
        rest = parts[1:]

        if rest == ["health"] and method == "GET":
            return {"status": "ok", "sessions": self.manager.open_sessions}
        if rest == ["cases"] and method == "GET":
            return {"cases": self.manager.available_cases()}
        if rest == ["sessions"] and method == "POST":
            return self.open_session(body)

        if len(rest) >= 2 and rest[0] == "sessions":
            return self._session_route(method, rest[1], rest[2:], body)

        raise ApiError(404, f"No route for {method} /{'/'.join(parts)}")

    def _session_route(self, method, session_id, tail, body):
        entry = self.manager.get(session_id)

        if not tail:
            if method == "GET":
                return self.snapshot(entry)
            if method == "DELETE":
                self.manager.close(session_id)
                return {"closed": session_id}
        elif tail == ["controls"] and method == "POST":
            return self.set_controls(entry, body)
        elif tail == ["controls", "reset"] and method == "POST":
            return self.reset_controls(entry)
        elif tail == ["commit"] and method == "POST":
            return self.commit(entry)
        elif tail == ["trial"] and method == "POST":
            return self.set_trial(entry, body)
        elif tail == ["trial", "reset"] and method == "POST":
            return self.reset_trial(entry)
        elif tail == ["arc"] and method == "POST":
            return self.arc(entry, body)
        elif tail == ["export"] and method == "POST":
            return self.export(entry, body)
        elif len(tail) == 2 and tail[0] == "meshes" and method == "GET":
            return self.mesh(entry, tail[1])

        raise ApiError(404, f"No route for {method} on session {session_id}")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def open_session(self, body) -> dict:
        folder = body.get("folder")
        if not folder:
            raise ApiError(400, "A folder is required to open a session.")
        entry = self.manager.open(
            folder, side=body.get("side", "left"), library=body.get("library")
        )
        return self.snapshot(entry)

    def snapshot(self, entry) -> dict:
        """Everything a viewer needs to draw from cold."""
        return _envelope(entry, scene=scene_snapshot(entry.session.scene))

    def set_controls(self, entry, body) -> dict:
        changes = body.get("changes") or body
        if not isinstance(changes, dict) or not changes:
            raise ApiError(400, "No control changes were given.")
        delta = entry.session.replan(**changes)
        return _envelope(entry, delta)

    def reset_controls(self, entry) -> dict:
        return _envelope(entry, entry.session.reset_adjustments())

    def commit(self, entry) -> dict:
        outcome = self.manager.commit(entry)
        return _envelope(
            entry,
            outcome.delta,
            commit={
                "computed": outcome.computed,
                "reused": outcome.reused,
                "unchanged": outcome.unchanged,
                "records": {
                    name: (record if isinstance(record, dict) else record.to_dict())
                    for name, record in outcome.records.items()
                },
                "notes": list(outcome.notes),
            },
        )

    def set_trial(self, entry, body) -> dict:
        changes = body.get("changes") or body
        if not isinstance(changes, dict) or not changes:
            raise ApiError(400, "No trial changes were given.")
        return _envelope(entry, entry.session.set_trial(**changes))

    def reset_trial(self, entry) -> dict:
        return _envelope(entry, entry.session.reset_trial())

    def arc(self, entry, body) -> dict:
        frames = self.manager.flexion_arc(
            entry,
            max_deg=float(body.get("max_deg", 120.0)),
            steps=int(body.get("steps", 24)),
        )
        return {"frames": frames}

    def export(self, entry, body) -> dict:
        out = body.get("out")
        if not out:
            raise ApiError(400, "An output folder is required.")
        written = export_session(
            entry.session, Path(out), write_meshes=bool(body.get("meshes", True))
        )
        return {"written": written, "out": str(Path(out).resolve())}

    def mesh(self, entry, filename: str) -> Binary:
        """One mesh as binary glTF, addressed by content hash.

        Marked immutable, which is honest rather than optimistic: the name *is* the
        hash of the content, so a browser that caches it for ever can never be wrong.
        """
        digest = filename[:-4] if filename.endswith(".glb") else filename
        mesh = entry.session.scene.meshes.get(digest)
        if mesh is None and self.manager.store is not None:
            try:
                mesh = self.manager.store.get_mesh(digest)
            except KeyError:
                mesh = None
        if mesh is None:
            raise ApiError(404, f"No mesh {digest!r} in this session.")
        return Binary("model/gltf-binary", write_glb(mesh, name=digest), immutable=True)


def _envelope(entry, delta=None, **extra) -> dict:
    session = entry.session
    return {
        "session_id": entry.session_id,
        "case_id": session.case_id,
        "side": session.side,
        "delta": (delta or SceneDelta()).to_dict(),
        "clip": clip_planes(session.plan),
        "report": session.report_lines(),
        "controls": asdict(session.controls),
        "trial": asdict(session.trial),
        "stale": bool(session.stale),
        **extra,
    }


def _segments(path: str) -> list:
    return [part for part in str(path).split("?")[0].split("/") if part]
