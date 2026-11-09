"""Synthetic knees for the test suite.

The generator lives in :mod:`tka_planner.validation.synthetic`, where the paper's
evaluation harness can reach it too; this module keeps the tests' imports unchanged.
"""

from tka_planner.validation.synthetic import *  # noqa: F401,F403
from tka_planner.validation.synthetic import (  # noqa: F401
    DEFAULT_AMA_DEG,
    DEFAULT_FEMORAL_WIDTH_MM,
    DEFAULT_MLDFA_DEG,
    DEFAULT_MPTA_DEG,
    DEFAULT_SLOPE_DEG,
    DEFAULT_TIBIAL_WIDTH_MM,
    DEFAULT_TWIST_DEG,
    mirror_landmarks,
    synthetic_knee,
)
