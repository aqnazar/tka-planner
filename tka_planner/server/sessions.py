"""Live sessions, and the store that outlives them.

A :class:`~tka_planner.session.PlanningSession` knows how to plan and how to cut. What
it deliberately does not know is that it might have cut this exact plan before. That
belongs here, because caching is a property of the installation rather than of the
planning, and because the session stays usable in a test with no store at all.

The commit path is the only interesting code in this module. It works one bone at a
time, because that is the granularity at which a change matters: adjusting the tibial
slope must leave a committed femur alone. For each bone it asks the store whether this
exact cut has been computed before, and only reaches for the kernel when the answer is
no.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tka_planner.scene.build import SHELL_COLOUR
from tka_planner.scene.model import FEMORAL, TIBIAL, Node, SceneDelta
from tka_planner.scene.motion import trial_pose
from tka_planner.session import PlanningSession
from tka_planner.store import Store, plan_key
from tka_planner.store.keys import BONES

__all__ = ["SessionManager", "SessionEntry", "CommitOutcome"]

SET_FOR_BONE = {"Femur": FEMORAL, "Tibia": TIBIAL}
GROUP_FOR_BONE = {"Femur": "femoral", "Tibia": "tibial"}


@dataclass
class CommitOutcome:
    """What a commit did, and how much of it was work."""

    delta: SceneDelta
    records: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    computed: list = field(default_factory=list)
    reused: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)


@dataclass
class SessionEntry:
    """One open session, with the bookkeeping the manager needs around it."""

    session_id: str
    session: PlanningSession

    # The bone meshes as they were read from disk. A second commit must cut the whole
    # bone again rather than cut what the first commit left, so the uncut geometry has
    # to survive the first cut. It costs nothing to keep: the scene's mesh table is
    # content-addressed and already holds it.
    pristine: dict = field(default_factory=dict)
    # The plan key each bone's committed geometry was computed from, so a commit that
    # changes nothing does nothing.
    committed: dict = field(default_factory=dict)

    def remember_pristine(self) -> None:
        scene = self.session.scene
        self.pristine = {
            bone: scene.nodes[bone].mesh_id
            for bone in BONES
            if bone in scene.nodes
        }
        self.committed = {}


class SessionManager:
    """Opens sessions, keeps them, and puts the store behind their commits."""

    def __init__(self, *, store=None, cases_root=None, library=None, kernel=None):
        self.store = store if isinstance(store, Store) else (
            Store(store) if store is not None else None
        )
        self.cases_root = Path(cases_root) if cases_root else None
        self.library = Path(library) if library else None
        self.kernel = kernel
        self._entries: dict = {}

    # ------------------------------------------------------------------
    # Opening and finding
    # ------------------------------------------------------------------

    def available_cases(self) -> list:
        """Case folders under the configured root, plus anything the store remembers.

        A folder counts as a case when it holds a femur and a tibia the session's own
        filename rules can find, so what this lists is exactly what will open.
        """
        found = {}
        if self.cases_root and self.cases_root.is_dir():
            for folder in sorted(p for p in self.cases_root.iterdir() if p.is_dir()):
                for side in ("left", "right"):
                    try:
                        femur, tibia = _find(folder, side)
                    except FileNotFoundError:
                        continue
                    found[f"{folder.name}:{side}"] = {
                        "case_id": folder.name,
                        "side": side,
                        "folder": str(folder),
                        "femur": str(femur),
                        "tibia": str(tibia),
                        "known": False,
                    }
        for case in (self.store.cases() if self.store else []):
            key = f"{case['case_id']}:{case['side']}"
            entry = found.setdefault(key, {
                "case_id": case["case_id"],
                "side": case["side"],
                "folder": str(Path(case["femur_path"]).parent),
                "femur": case["femur_path"],
                "tibia": case["tibia_path"],
                "known": True,
            })
            entry["known"] = True
        return sorted(found.values(), key=lambda case: (case["case_id"], case["side"]))

    def open(self, folder, *, side: str, library=None) -> SessionEntry:
        """Open a patient folder, measure it, and build the first plan."""
        session = PlanningSession.open(
            folder, side=side, library=library or self.library
        )
        session.build()

        entry = SessionEntry(session_id=uuid.uuid4().hex[:16], session=session)
        entry.remember_pristine()
        self._entries[entry.session_id] = entry

        if self.store is not None:
            self.store.save_case(
                session.case_id,
                side=side,
                femur_path=session.femur_path,
                tibia_path=session.tibia_path,
                library=library or self.library,
            )
        return entry

    def get(self, session_id: str) -> SessionEntry:
        entry = self._entries.get(session_id)
        if entry is None:
            raise KeyError(f"No open session {session_id!r}.")
        return entry

    def close(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    @property
    def open_sessions(self) -> list:
        return sorted(self._entries)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self, entry: SessionEntry) -> CommitOutcome:
        """Cut every bone whose cut has changed, reusing the store where it can."""
        session = entry.session
        outcome = CommitOutcome(delta=SceneDelta())

        for bone in BONES:
            if bone not in session.scene.nodes:
                continue
            key = plan_key(session, (bone,))

            if entry.committed.get(bone) == key:
                outcome.unchanged.append(bone)
                continue

            cached = self.store.cached_commit(key) if self.store else None
            if cached is not None:
                self._apply_cached(entry, bone, cached, outcome)
            else:
                self._cut(entry, bone, key, outcome)
            entry.committed[bone] = key

        # Every bone is now either freshly cut, restored from the store or already
        # correct, so the scene matches the plan whatever route each bone took.
        session.stale = False
        self._record_plan(session)
        outcome.delta.notes.extend(outcome.notes)
        return outcome

    def _cut(self, entry, bone, key, outcome) -> None:
        session = entry.session
        scene = session.scene

        # Cut the bone as it was read, not as the last commit left it. Without this a
        # second commit would subtract the new block from the already-resected bone,
        # which removes more bone and can never put any back.
        pristine = entry.pristine.get(bone)
        if pristine is not None and scene.nodes[bone].mesh_id != pristine:
            scene.nodes[bone].mesh_id = pristine
            outcome.delta.meshes[bone] = pristine

        result = session.commit(kernel=self.kernel, bones=(bone,))

        outcome.delta = outcome.delta.merge(result.delta)
        outcome.records.update(result.records)
        outcome.notes.extend(result.notes)
        outcome.computed.append(bone)

        if self.store is None:
            return
        entries = {}
        for name in _committed_names(bone):
            node = scene.nodes.get(name)
            if node is None or node.mesh_id is None or name not in result.records:
                continue
            digest = self.store.put_mesh(scene.meshes[node.mesh_id])
            entries[name] = {
                "mesh_id": digest,
                "record": result.records[name].to_dict(),
            }
        if entries:
            self.store.save_commit(key, entries)

    def _apply_cached(self, entry, bone, cached, outcome) -> None:
        """Put geometry the store already holds back into the scene."""
        scene = entry.session.scene
        for name, record in cached.items():
            mesh = self.store.get_mesh(record["mesh_id"])
            node = scene.nodes.get(name)
            if node is None:
                node = _shell_node(name, bone)
                scene.add(node, mesh)
                outcome.delta.meshes[name] = node.mesh_id
            else:
                outcome.delta.meshes[name] = scene.replace_mesh(name, mesh)
            outcome.records[name] = record["record"]
        outcome.reused.append(bone)

    def _record_plan(self, session) -> None:
        if self.store is None:
            return
        from dataclasses import asdict

        self.store.save_plan(
            plan_key(session),
            case_id=session.case_id,
            controls=asdict(session.controls),
            scalars=session._scalars(),
        )

    # ------------------------------------------------------------------
    # Reduce mode
    # ------------------------------------------------------------------

    def flexion_arc(self, entry, *, max_deg: float = 120.0, steps: int = 24) -> list:
        """The scripted arc, as poses rather than as an animation.

        Computed here and sent whole, so playback in the viewer is a list index rather
        than a request per frame, and so the pose maths stays in one place.
        """
        steps = max(2, int(steps))
        session = entry.session
        frames = []
        for index in range(steps + 1):
            flexion = max_deg * index / steps
            frames.append({
                "flexion_deg": float(flexion),
                "pose": trial_pose(
                    session.scene.pivot,
                    flexion_deg=flexion,
                    varus_valgus_deg=session.trial.varus_valgus_deg,
                    drawer_ap_mm=session.trial.drawer_ap_mm,
                ).tolist(),
            })
        return frames


def _committed_names(bone: str) -> tuple:
    return (bone, f"{bone}.Shell")


def _shell_node(name: str, bone: str) -> Node:
    """Rebuild a shell node the resection would have created.

    A shell's identity is entirely a function of its name, so a cached commit can
    recreate it without the store having to remember colours and tags.
    """
    return Node(
        name=name,
        set_id=SET_FOR_BONE[bone],
        colour=SHELL_COLOUR,
        tags={"shell": True, "group": GROUP_FOR_BONE[bone]},
    )


def _find(folder, side):
    from tka_planner.session import find_bone_files

    return find_bone_files(folder, side)
