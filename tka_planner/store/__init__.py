"""Case persistence: a SQLite database and a content-addressed mesh directory.

Nothing here knows about planning. It stores what a session produced and hands it back
when the same plan comes round again, which is what makes returning to a committed plan
instant rather than a second run of the booleans.
"""

from .db import CaseRecord, Store
from .keys import plan_key
from .meshes import MeshStore

__all__ = ["Store", "CaseRecord", "MeshStore", "plan_key"]
