"""Case persistence: a SQLite database and a content-addressed mesh directory.

Nothing here knows about planning. It stores what a session produced and hands it back
when the same plan comes round again, which is what makes returning to a committed plan
instant rather than a second run of the booleans.

What it holds is patient-derived geometry, so a store is patient data and belongs
wherever the segmentations it was built from belong. Nothing writes one inside the
repository, and the case identifier is whatever the folder was called, which is why the
project's rule that a case is named ``CASE_00N`` matters at the point a folder is named
rather than here.
"""

from .db import CaseRecord, Store
from .keys import plan_key
from .meshes import MeshStore

__all__ = ["Store", "CaseRecord", "MeshStore", "plan_key"]
