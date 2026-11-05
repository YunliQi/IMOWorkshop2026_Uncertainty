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
    """Numerical ODE through every pulse / BEV / PARP edge, surgery multiplies S and R by q."""
    edges = np.concatenate([m.pulse_intervals(s).ravel(), s.bev_intervals().ravel(),
                            s.parp_intervals().ravel(), [s.t_surgery]])
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


def test_bev_kills_at_k_bev_and_brake_option_slows_growth():
    s = Schedule([], 50.0, [], bev_starts=[1.0])           # BEV on over [1, 2)
    m = get_model("resistance")                             # default: BEV kill, eps_bev = 1
    ln = m.segment_log_growth([2.0], 1.0, th(g=0.4, k=10, rho=0.01), s, "pre")[0, 0]
    assert m.k_bev == pytest.approx(0.00235 * 30)
    assert ln == pytest.approx(0.4 - m.k_bev, abs=1e-4)
    m = get_model("resistance", bev="brake")
    ln = m.segment_log_growth([2.0], 1.0, th(g=0.4, k=10, rho=0.01), s, "pre")[0, 0]
    assert ln == pytest.approx(0.4 * (1 - m.d_bev))
    with pytest.raises(ValueError, match="bev"):
        get_model("resistance", bev="nope")


MAINT = Schedule([0, 0.7, 1.4], 2.5, [3.5, 4.2, 4.9, 5.6], bev_starts=[4.2, 4.9],
                 maint_bev=[[6.6, 9.0]], parp=[[6.6, 14.0]], hrd=True)


def test_parp_kill_depends_on_hr_status_and_spares_resistant_cells():
    from ovca.models.resistance import SensitiveResistant
    p = th(g=0.2, k=10, rho=1e-9)
    for hrd, rate in ((True, 0.03081), (False, 0.01236)):
        m = SensitiveResistant(m=0.0)
        s = Schedule([], 50.0, [], parp=[[1.0, 3.0]], hrd=hrd)
        ln = m.segment_log_growth([3.0], 1.0, p, s, "pre")[0, 0]
        assert ln == pytest.approx(2 * (0.2 - rate * 30), rel=1e-6)
    m = SensitiveResistant(m=0.0)                       # all resistant: 20% of the kill
    s = Schedule([], 50.0, [], parp=[[1.0, 3.0]], hrd=True)
    ln = m.segment_log_growth([3.0], 1.0, th(g=0.2, k=10, rho=1e9), s, "pre")[0, 0]
    assert ln == pytest.approx(2 * (0.2 - 0.2 * 0.03081 * 30), rel=1e-6)
    s = Schedule([], 50.0, [], parp=[[1.0, 3.0]])       # unknown: HRP rate by default
    assert get_model("resistance").k_parp(s) == pytest.approx(0.01236 * 30)
    assert get_model("resistance", hrd_unknown="mean").k_parp(s) == pytest.approx(
        0.5 * (0.03081 + 0.01236) * 30)


def test_exact_solution_matches_ode_with_maintenance():
    m = get_model("resistance")
    theta = dict(g=0.3, k=15.0, rho=0.02)
    t = np.linspace(0, 16, 65)
    ref = reference(m, t, theta, MAINT, 100.0, 0.05)
    ex = m.simulate_patient_batch(t, MAINT, th(**theta), th(**theta), np.array([100.0]),
                                  np.array([0.05]))[0]
    assert np.allclose(np.log(ex), np.log(ref), atol=1e-7)


@pytest.mark.parametrize("end", [14.0, np.inf])
def test_recurrence_during_and_after_maintenance_matches_grid(end):
    m = get_model("resistance")
    s = Schedule(MAINT.nact_starts, MAINT.t_surgery, MAINT.act_starts,
                 bev_starts=MAINT.bev_starts, maint_bev=MAINT.maint_bev,
                 parp=[[6.6, end]], hrd=False)
    p = {k: np.array(v) for k, v in dict(g=[0.1, 0.5, 0.9, 0.02], k=[12, 6, 12, 20],
                                         rho=[0.01, 0.2, 0.5, 1e-4]).items()}
    c0, q = np.array([300.0, 800.0, 500.0, 50.0]), np.array([0.2, 0.3, 0.2, 0.05])
    cut = np.array([27.0, 20.0, 25.0, 30.0])
    rec = m.recurrence_time_batch(s, p, p, c0, q, cut, 36.0)
    grid = np.arange(s.t_end, s.t_end + 36.0 + 1e-9, 0.001)
    above = m.simulate_patient_batch(grid, s, p, p, c0, q) >= cut[:, None]
    g_rec = np.where(above.any(1), grid[np.argmax(above, 1)], np.inf)
    assert np.array_equal(np.isfinite(rec), np.isfinite(g_rec))
    fin = np.isfinite(g_rec)
    assert fin.sum() >= 2 and (g_rec[fin] < 14.0).any()     # some cross during maintenance
    assert np.allclose(rec[fin], g_rec[fin], atol=5e-3)


def test_schedule_record_round_trip_keeps_maintenance():
    s = Schedule([0.0], 2.0, [3.0, 3.7], bev_starts=[3.0], maint_bev=[[4.5, 9.0]],
                 parp=[[4.5, np.inf]], hrd=False)
    r = Schedule.from_record(s.to_record())
    assert np.array_equal(r.maint_bev, s.maint_bev) and np.array_equal(r.parp, s.parp)
    assert r.hrd is False
    assert Schedule.from_record(Schedule([0.0], 2.0, [3.0]).to_record()).hrd is None


def test_maintenance_blocks_from_sheet_columns():
    from ovca.real_data import hr_status, maintenance_blocks
    d = pd.DataFrame({
        "DAY Maintenance": [300, 330, 360, 390, 420, 1000],
        "Maintenance": ["Olaparib/BEV", np.nan, "held due to anemia", np.nan, np.nan, "Doxil"],
        "Maintenance cycle": ["daily", "BEV - IV, every 3 weeks", np.nan, np.nan, "Ongoing",
                              "1"],
        "HR status": ["HRD", "GIS: 50", np.nan, np.nan, np.nan, np.nan]})
    blocks, stop = maintenance_blocks(d)
    assert stop == 1000
    assert blocks["parp"].tolist() == [[300, 360], [390, np.inf]]
    assert blocks["bev"].tolist() == [[300, 351], [390, np.inf]]
    assert hr_status(d) is True
    assert hr_status(d.assign(**{"HR status": ["unknown"] + [np.nan] * 5})) is None


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
    assert {"g_pre", "k_pre", "rho_pre", "level_pre", "level_post", "sigma", "b0",
            "q_surgery"} == set(p["priors"])
    for other in ("growth_death", "poptk"):
        assert "b0" not in default_profile(other)["priors"]
        assert default_profile(other)["data"]["rule"] == "cutoff"
    assert p["priors"]["b0"]["dist"] == "truncated_lognormal"
    assert p["priors"]["rho_pre"]["median"] == pytest.approx(0.0055)
    assert p["data"]["rule"] == "cutoff" and p["data"]["cutoff"] == 50.0
    assert p["forecast"]["horizon"] == pytest.approx(4000 / 30)
    with pytest.raises(ProfileError, match="model_options"):
        check_profile({"model": "resistance", "model_options": {"bev": "x"}}, verbose=False)
    with pytest.raises(ProfileError, match="rule"):
        check_profile({"model": "resistance", "data": {"rule": "x"}}, verbose=False)
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


def test_truncated_lognormal_prior():
    from scipy import integrate
    from ovca.priors import Dist
    d = Dist("b0", "truncated_lognormal", dict(median=12.0, sd_log=0.6, low=2.0, high=35.0))
    assert integrate.quad(lambda u: np.exp(d.logpdf_u(u)), -40, 40)[0] == pytest.approx(1.0)
    x = d.to_natural(d.sample_u(np.random.default_rng(0), 20000))
    assert 2.0 < x.min() and x.max() < 35.0 and 10.5 < np.median(x) < 12.5


def test_rise_rule_matches_brute_force():
    rec, m, p = synthetic(dict(g=0.2, k=12, rho=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = default_profile("resistance")
        prof["data"]["rule"] = "rise"                # the rule stays available as an option
        fc = Forecaster(prof)
    pb = fc.problem(rec)
    u = pb.prior.sample(np.random.default_rng(3), 40)
    tau = pb.recurrence(u, 60.0)
    s = rec.schedule
    grid = np.arange(s.t_surgery, s.t_end + 60.0, 0.001)
    y = pb.simulate(u, grid)
    nadir = np.minimum.accumulate(y, axis=1)
    above = (y >= 2 * nadir) & (grid[None] >= s.t_end)
    ref = np.where(above.any(1), grid[np.argmax(above, 1)], np.inf)
    assert np.array_equal(np.isfinite(tau), np.isfinite(ref))
    fin = np.isfinite(ref)
    assert fin.sum() > 10 and np.allclose(tau[fin], ref[fin], atol=0.01)


def test_forecaster_recovers_patient():
    truth = dict(g=0.2, k=12.0, rho=0.01)
    rec, m, p = synthetic(truth)
    prof = default_profile("resistance")
    prof["data"]["cutoff"] = 35.0                  # compare with the exact crossing of 35
    prof["forecast"]["horizon"] = 36.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = Forecaster(prof).fit(rec)
    S = snap.summary((0.05, 0.5, 0.95)).set_index("parameter")
    for k, v in {**{f"{k}_pre": v for k, v in truth.items()}, "b0": 9.0}.items():
        assert S.loc[k, "q5"] <= v <= S.loc[k, "q95"], k
    true_rec = m.recurrence_time_batch(rec.schedule, p, p, np.array([600.0]), np.array([0.1]),
                                       35.0 - 9.0, 36.0)[0] - rec.schedule.t_end
    lo, _, hi = snap.recurrence_interval(0.9)
    assert lo <= true_rec <= hi


def test_continuous_surgery_recovers_q_and_forecasts_before_surgery():
    truth = dict(g=0.2, k=12.0, rho=0.01)
    rec, m, p = synthetic(truth)                  # surgery leaves q = 0.1
    prof = default_profile("resistance")
    prof["surgery"]["mode"] = "continuous"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = Forecaster(prof)
        pb = fc.problem(rec)
        assert "q_surgery" in pb.names and "level_post" not in pb.names
        u = pb.prior.sample(np.random.default_rng(0), 4)
        th_pre, th_post, c0, q, _ = pb.components(u)
        ref = m.simulate_patient_batch(rec.t, rec.schedule, th_pre, th_post, c0, q)
        assert np.allclose(pb.simulate(u, rec.t, background=False), ref)
        snap = fc.fit(rec)
        S = snap.summary((0.05, 0.5, 0.95)).set_index("parameter")
        assert S.loc["surgery_jump", "q5"] <= 0.1 <= S.loc["surgery_jump", "q95"]
        before = fc.fit(rec, t_now=rec.schedule.t_surgery - 1e-6)
    assert before.posterior.problem.post_mode == "continuous"
    assert np.isfinite(before.forecast.recurrence).mean() > 0.5
