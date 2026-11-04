"""PopTK model: surviving fraction, dose memory across surgery, recurrence, and the tool."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from ovca import Forecaster, PatientRecord, ProfileError, check_profile, default_profile
from ovca.models import get_model
from ovca.models.base import BaseModel
from ovca.patient import Schedule

ROOT = Path(__file__).resolve().parents[1]
SCHED = Schedule([0, 1, 2], 3.5, [4.5, 5.5, 6.5])


def th(**kw):
    return {k: np.array([float(v)]) for k, v in kw.items()}


def test_surviving_fraction_matches_numerical_integral():
    m = get_model("poptk")
    kill, het = 0.3, 0.5
    for n in (1, 3, 6):
        f = lambda z: np.exp(-n * kill * np.log(10) * 10 ** (het * z)) * np.exp(-z * z / 2)  # noqa
        ref = quad(f, -12, 12, limit=200)[0] / np.sqrt(2 * np.pi)
        assert m.log_surviving([n], [kill], [het])[0, 0] == pytest.approx(np.log(ref), rel=1e-6)


def test_no_heterogeneity_gives_constant_log_kill_per_dose():
    m = get_model("poptk")
    ls = m.log_surviving(np.arange(7), [0.3], [1e-6])[0] / np.log(10)
    assert np.allclose(np.diff(ls), -0.3, atol=1e-6)


def test_heterogeneity_makes_each_dose_kill_less():
    m = get_model("poptk")
    k = -np.diff(m.log_surviving(np.arange(7), [0.3], [0.4])[0])
    assert np.all(np.diff(k) < 0)


def test_trajectory_growth_surgery_and_reading_before_dose():
    m = get_model("poptk")
    p = th(growth=0.2, kill=0.3, het=1e-6)
    t = np.array([0.0, 0.5, 1.0, 3.4, 3.6, 7.0])
    log_y = np.log(m.simulate_patient_batch(t, SCHED, p, p, np.array([100.0]),
                                            np.array([0.1]))[0])
    ln10 = np.log(10)
    # t = 0 is a dose day: the reading comes before the dose
    assert log_y[0] == pytest.approx(np.log(100))
    assert log_y[1] == pytest.approx(np.log(100) + 0.2 * 0.5 - 0.3 * ln10)
    assert log_y[2] == pytest.approx(np.log(100) + 0.2 * 1.0 - 0.3 * ln10)
    # across surgery: x q, no dose in between
    assert log_y[4] - log_y[3] == pytest.approx(0.2 * 0.2 + np.log(0.1))
    # after all 6 doses
    assert log_y[5] == pytest.approx(np.log(100) + 0.2 * 7 - 6 * 0.3 * ln10 + np.log(0.1))


def test_act_kill_remembers_nact_doses():
    """With heterogeneity, the first ACT dose kills less after more NACT doses."""
    m = get_model("poptk")
    p = th(growth=0.0, kill=0.3, het=0.5)
    drop = []
    for nact in ([0], [0, 1, 2]):
        s = Schedule(nact, 3.5, [4.5, 5.5])
        y = m.simulate_patient_batch([4.4, 4.6], s, p, p, np.array([100.0]), np.array([1.0]))[0]
        drop.append(np.log(y[0] / y[1]))
    assert drop[1] < drop[0]


def test_segment_growth_matches_full_simulation():
    m = get_model("poptk")
    p = th(growth=0.15, kill=0.4, het=0.3)
    t = np.array([4.0, 5.0, 6.2, 9.0])
    full = np.log(m.simulate_patient_batch(t, SCHED, p, p, np.array([50.0]), np.array([0.2])))
    seg = m.segment_log_growth(t, SCHED.t_surgery, p, SCHED, "post")
    assert np.allclose(full - full[:, :1], seg - seg[:, :1])


def test_recurrence_exact_matches_grid():
    m = get_model("poptk")
    p = {k: np.array(v) for k, v in dict(growth=[0.1, 0.25, 0.02], kill=[0.3, 0.2, 0.6],
                                         het=[0.1, 0.4, 0.2]).items()}
    c0, q = np.array([300.0, 800.0, 50.0]), np.array([0.2, 0.3, 0.05])
    exact = m.recurrence_time_batch(SCHED, p, p, c0, q, 35.0, 36.0)
    grid = BaseModel.recurrence_time_batch(m, SCHED, p, p, c0, q, 35.0, 36.0, step=0.001)
    fin = np.isfinite(grid)
    assert np.array_equal(fin, np.isfinite(exact))
    assert np.allclose(exact[fin], grid[fin], atol=2e-3)


def test_profile_defaults_and_mcmc_refused():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = check_profile({"model": "poptk"}, verbose=False)
    assert p["surgery"]["shift_params"] == []
    assert {"growth_pre", "kill_pre", "het_pre", "level_pre", "level_post", "sigma"} == set(
        p["priors"])
    with pytest.raises(ProfileError, match="growth_death"):
        check_profile({"model": "poptk", "backend": "mcmc"}, verbose=False)
    p = check_profile({"model": "poptk", "surgery": {"shift_params": ["growth"]}},
                      verbose=False)
    assert "shift_growth" in p["priors"]


def test_growth_death_default_shifts_unchanged():
    assert default_profile("growth_death")["surgery"]["shift_params"] == ["gamma", "delta"]


def test_forecaster_recovers_poptk_patient():
    base = PatientRecord.from_synthetic(pd.read_csv(ROOT / "data/synthetic_data.csv"), "SYN38")
    s, t = base.schedule, base.t
    m = get_model("poptk")
    truth = dict(growth=0.12, kill=0.35, het=0.4)
    p = th(**truth)
    y = m.simulate_patient_batch(t, s, p, p, np.array([600.0]), np.array([0.1]))[0]
    y = y * np.exp(0.15 * np.random.default_rng(1).standard_normal(t.size))
    rec = PatientRecord("PTK", t, y, s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = Forecaster(default_profile("poptk")).fit(rec)
    S = snap.summary((0.05, 0.5, 0.95)).set_index("parameter")
    for k, v in truth.items():
        assert S.loc[f"{k}_pre", "q5"] <= v <= S.loc[f"{k}_pre", "q95"], k
    true_rec = m.recurrence_time_batch(s, p, p, np.array([600.0]), np.array([0.1]), 35.0,
                                       36.0)[0] - s.t_end
    lo, _, hi = snap.recurrence_interval(0.9)
    assert lo <= true_rec <= hi
