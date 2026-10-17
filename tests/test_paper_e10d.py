"""E10d's two pieces of model geometry, on the toy bones of test_flexion_stop.

The experiment itself runs on the cohort; what it rests on is checked here: the thicker
posterior condyle pushes the tibia back by the thickening (ramped in over 90 deg), and
the lever-arm row reads a contact's distance from the axis and the slope 57.3/r.
"""

import numpy as np
import pytest

from paper.experiments.e10_flexion import (_lever_row, _limit,
                                           _thicker_posterior_condyle)
from tests.test_flexion_stop import _box, _model

POSTERIOR = np.array([0.0, 1.0, 0.0])     # LPS +Y


def _corner_model():
    return _model(femur=_box(y=(-10.0, 8.5), z=(3.0, 103.0)),
                  tibia=_box(y=(-30.0, -8.0), z=(-105.0, 0.0)))


def test_a_thicker_posterior_condyle_pushes_the_tibia_back_once_it_bears():
    model = _corner_model()
    pushed = _thicker_posterior_condyle(model, 2.0, POSTERIOR)
    for angle, ramp in ((0.0, 0.0), (45.0, 0.5), (120.0, 1.0)):
        shift = pushed.pose("S3", angle)[:3, 3] - model.pose("S3", angle)[:3, 3]
        assert shift == pytest.approx(2.0 * ramp * POSTERIOR, abs=1e-9)


def test_pushing_the_tibia_back_delays_the_contact_by_about_the_lever_rule():
    """The stopping corner is about 9 mm from the axis, so 1 mm of push should move the
    limit by roughly 57.3/9 deg -- give or take the geometry factor, which the row
    states."""
    model = _corner_model()
    base = _limit(model, "S3")
    row = _lever_row(model, base, -POSTERIOR)
    assert row["pair"] == "tibial_bone_vs_femoral_bone"
    assert row["r_mm"] == pytest.approx(np.hypot(8.5, 3.0), abs=1.0)
    assert row["slope_57_3_over_r"] == pytest.approx(57.2958 / row["r_mm"], abs=1e-3)
    later = _limit(_thicker_posterior_condyle(model, 0.5, POSTERIOR), "S3")
    assert later.max_flexion_deg > base.max_flexion_deg
