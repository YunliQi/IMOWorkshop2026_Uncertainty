"""Sensitive/resistant model: exact solution, drug inputs, background b0, and the tool."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from ovca import Forecaster, PatientRecord, ProfileError, check_profile, default_profile
from ovca.models import get_model
from ovca.patient import Schedule

ROOT = Path(__file__).resolve().parents[1]
SCHED = Schedule([0, 0.7, 1.4], 2.5, [3.5, 4.2, 4.9, 5.6], bev_starts=[4.2, 4.9, 5.6])


def th(**kw):
    return {k: np.array([float(v)]) for k, v in kw.items()}


def reference(m, t, theta, s, c0, q):
    """Numerical ODE through every pulse / BEV edge, surgery multiplies S and R by q."""
    edges = np.concatenate([m.pulse_intervals(s).ravel(), s.bev_intervals().ravel(),
                            [s.t_surgery]])
    pts = np.unique(np.concatenate([[0.0], edges[(edges > 0) & (edges < t.max())], t]))
    x, out = m.initial_state(c0, theta), {0.0: m.initial_state(c0, theta).sum()}
    for a, b in zip(pts[:-1], pts[1:]):
        if a == s.t_surgery:
            x = q * x
        x = solve_ivp(m.rhs_full, (a, b), x, args=(theta, s), method="LSODA", rtol=1e-11,
                      atol=1e-14).y[:, -1]
        out[b] = x.sum() if b != s.t_surgery else q * x.sum()
    return np.array([out[v] for v in t])


def test_exact_solution_matches_ode():
    m = get_model("resistance")
    theta = dict(g=0.3, k=15.0, rho=0.02)
    t = np.linspace(0, 9, 37)
    ref = reference(m, t, theta, SCHED, 100.0, 0.05)
    ex = m.simulate_patient_batch(t, SCHED, th(**theta), th(**theta), np.array([100.0]),
                                  np.array([0.05]))[0]
    assert np.allclose(np.log(ex), np.log(ref), atol=1e-7)


def test_without_drugs_total_grows_at_g_whatever_the_mix():
    m = get_model("resistance")
    s = Schedule([], 50.0, [])
    for rho in (1e-4, 0.3):
        ln = m.segment_log_growth([1.0, 4.0], 0.0, th(g=0.2, k=10, rho=rho), s, "pre")[0]
        assert np.allclose(ln, [0.2, 0.8])


def test_chemo_selects_resistant_cells():
    m = get_model("resistance")
    f = m.resistant_fraction([0.0, 2.4, 6.5], th(g=0.2, k=12, rho=0.01), SCHED)[0]
    assert f[0] == pytest.approx(0.01 / 1.01)
    assert f[0] < f[1] < f[2]


def test_one_pulse_kills_s_by_exp_minus_k_pulse():
    from ovca.models.resistance import SensitiveResistant
    m = SensitiveResistant(m=0.0)                      # no switching: S alone
    s = Schedule([1.0], 50.0, [])
    p = th(g=0.0, k=10.0, rho=1e-12)
    ln = m.segment_log_growth([1.0 + m.pulse], 1.0, p, s, "pre")[0, 0]
    assert ln == pytest.approx(-10.0 * m.pulse, rel=1e-6)


def test_bev_slows_growth():
    m = get_model("resistance")
    s = Schedule([], 50.0, [], bev_starts=[1.0])           # BEV on over [1, 2)
    ln = m.segment_log_growth([2.0], 1.0, th(g=0.4, k=10, rho=0.01), s, "pre")[0, 0]
    assert ln == pytest.approx(0.4 * (1 - m.d_bev))


def test_segment_growth_matches_full_simulation():
    m = get_model("resistance")
    p = th(g=0.25, k=12, rho=0.03)
    t = np.array([3.0, 4.0, 5.3, 8.0])
    full = np.log(m.simulate_patient_batch(t, SCHED, p, p, np.array([50.0]), np.array([0.2])))
    seg = m.segment_log_growth(t, SCHED.t_surgery, p, SCHED, "post")
    assert np.allclose(full - full[:, :1], seg - seg[:, :1])


def test_recurrence_exact_matches_grid_with_per_draw_cutoff():
    m = get_model("resistance")
    p = {k: np.array(v) for k, v in dict(g=[0.1, 0.3, 0.02], k=[12, 6, 20],
                                         rho=[0.01, 0.2, 1e-4]).items()}
    c0, q = np.array([300.0, 800.0, 50.0]), np.array([0.2, 0.3, 0.05])
    cut = np.array([27.0, 20.0, 30.0])
    exact = m.recurrence_time_batch(SCHED, p, p, c0, q, cut, 36.0)
    grid = np.arange(SCHED.t_end, SCHED.t_end + 36.0 + 1e-9, 0.001)
    above = m.simulate_patient_batch(grid, SCHED, p, p, c0, q) >= cut[:, None]
    g_rec = np.where(above.any(1), grid[np.argmax(above, 1)], np.inf)
    assert np.array_equal(np.isfinite(exact), np.isfinite(g_rec))
    fin = np.isfinite(g_rec)
    assert np.allclose(exact[fin], g_rec[fin], atol=2e-3)


def test_profile_has_background_and_others_do_not():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = check_profile({"model": "resistance"}, verbose=False)
    assert p["surgery"]["shift_params"] == []
    assert {"g_pre", "k_pre", "rho_pre", "level_pre", "level_post", "sigma", "b0"} == set(
        p["priors"])
    for other in ("growth_death", "poptk"):
        assert "b0" not in default_profile(other)["priors"]
    with pytest.raises(ProfileError, match="growth_death"):
        check_profile({"model": "resistance", "backend": "mcmc"}, verbose=False)
    with pytest.raises(ProfileError, match="b0"):
        check_profile({"model": "resistance", "priors": {"b0": -1.0}}, verbose=False)


def synthetic(truth, b0=9.0, sd=0.12, seed=1):
    base = PatientRecord.from_synthetic(pd.read_csv(ROOT / "data/synthetic_data.csv"), "SYN38")
    s = base.schedule
    m = get_model("resistance")
    p = th(**truth)
    y = m.simulate_patient_batch(base.t, s, p, p, np.array([600.0]), np.array([0.1]))[0] + b0
    y = y * np.exp(sd * np.random.default_rng(seed).standard_normal(base.t.size))
    return PatientRecord("SR", base.t, y, s), m, p


def test_problem_adds_background_but_levels_are_tumour_part():
    rec, m, p = synthetic(dict(g=0.2, k=12, rho=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = Forecaster(default_profile("resistance"))
    pb = fc.problem(rec)
    u = pb.prior.sample(np.random.default_rng(0), 5)
    b0 = pb.background(u)
    assert np.allclose(pb.simulate(u, rec.t) - pb.simulate(u, rec.t, background=False),
                       b0[:, None])
    nat = pb.natural(u)
    assert np.allclose(nat["level_pre"], pb.prior.natural(u)["level_pre"])
    assert np.allclose(nat["b0"], b0)


def test_forecaster_recovers_patient():
    truth = dict(g=0.2, k=12.0, rho=0.01)
    rec, m, p = synthetic(truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = Forecaster(default_profile("resistance")).fit(rec)
    S = snap.summary((0.05, 0.5, 0.95)).set_index("parameter")
    for k, v in {**{f"{k}_pre": v for k, v in truth.items()}, "b0": 9.0}.items():
        assert S.loc[k, "q5"] <= v <= S.loc[k, "q95"], k
    true_rec = m.recurrence_time_batch(rec.schedule, p, p, np.array([600.0]), np.array([0.1]),
                                       35.0 - 9.0, 36.0)[0] - rec.schedule.t_end
    lo, _, hi = snap.recurrence_interval(0.9)
    assert lo <= true_rec <= hi
