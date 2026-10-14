"""Every paper experiment. Importing this package registers them all.

One module per experiment, named ``eNN_<topic>.py``, each with a ``run(ctx)`` decorated
by :func:`paper.harness.experiment`. Add a module here and to the import list below.
"""

from . import e00_cohort  # noqa: F401
from . import e02_invariance  # noqa: F401
from . import e06c_hip_centre  # noqa: F401
from . import e_regression  # noqa: F401
from . import e07a_sizing  # noqa: F401
from . import e07b_fit  # noqa: F401
from . import e01_synthetic_accuracy  # noqa: F401
from . import e05_montecarlo  # noqa: F401
from . import e10_flexion  # noqa: F401
from . import e12_legacy_baseline  # noqa: F401
