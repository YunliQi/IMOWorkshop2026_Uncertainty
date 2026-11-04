"""Forecasting contract tests: run with pytest or directly with Python."""
import copy
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import halfnorm, lognorm, norm, uniform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ovca import Forecaster, History, PatientRecord, ProfileError, Snapshot
from ovca import check_profile, default_profile
from ovca.backends.base import Posterior
from ovca.priors import Dist, JointPrior
from ovca.problem import InferenceProblem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def record():
    return PatientRecord.from_synthetic(pd.read_csv(ROOT / "data/synthetic_data.csv"), "SYN38")


@pytest.fixture
def profile():
    p = default_profile()
    p["abc"].update(n_particles=300, n_moves=6)
    p["forecast"].update(n_draws=300, step=0.2)
    return p


@pytest.fixture(scope="module")
def history(record):
    p = default_profile()
    p["abc"].update(n_particles=300, n_moves=6)
    p["forecast"].update(n_draws=300, step=0.2)
    return Forecaster(p).replay(record, verbose=False)


def test_profile_defaults_and_sources():
    defaults = default_profile()
    for given in ({}, defaults):
        checked = check_profile(given, verbose=False)
        assert checked.pop("_sources") == {}
        assert checked == defaults
    given = copy.deepcopy(defaults)
    given["priors"]["gamma_pre"]["median"] = 0.08
    given["abc"]["n_particles"] += 100
    before = copy.deepcopy(given)
    checked = check_profile(given, verbose=False)
    assert given == before
    assert checked["_sources"] == {"priors.gamma_pre": "you", "abc.n_particles": "you"}
    assert check_profile(checked, verbose=False)["_sources"] == checked["_sources"]


INVALID_PROFILES = [
    ({"forcast": {}}, ("forcast", "did you mean", "forecast")),
    ({"priors": {"gama_pre": 0.1}}, ("gama_pre", "not a parameter")),
    ({"priors": {"gamma_pre": {"dist": "lognormal", "median": 0.1, "sd_lg": 0.5}}},
     ("gamma_pre", "sd_lg", "did you mean", "sd_log")),
    ({"priors": {"sigma": {"dist": "banana"}}}, ("sigma", "not supported")),
    ({"priors": {"sigma": {"dist": "normal", "mean": 0.1}}}, ("sigma", "needs", "sd")),
    *[({"priors": {"shift_gamma": {"dist": "normal", "mean": 0, "sd": sd}}},
       ("shift_gamma", "sd", "> 0")) for sd in (0, -1)],
    ({"priors": {"sigma": {"dist": "lognormal", "median": 0.1, "sd_log": 0}}},
     ("sigma", "sd_log", "> 0")),
    *[({"priors": {"sigma": {"dist": "uniform", "low": lo, "high": 1}}},
       ("sigma", "low < high")) for lo in (1, 2)],
    *[({"priors": {name: {"dist": "normal", "mean": 0.1, "sd": 1}}},
       (name, "must be > 0")) for name in ("gamma_pre", "sigma")],
    ({"priors": {"shift_gamma": {"dist": "lognormal", "median": 1, "sd_log": 1}}},
     ("shift_gamma", "shift is a log ratio")),
    ({"priors": {"gamma_pre": {"dist": "lognormal", "median": "first_reading", "sd_log": 1}}},
     ("gamma_pre", "first_reading", "only works")),
    ({"surgery": {"shift_params": ["banana"]}}, ("shift_params", "banana", "not a parameter")),
    ({"surgery": {"shift_params": ["gamma"]}, "priors": {"shift_delta": 0}},
     ("shift_delta", "not in surgery.shift_params")),
    ({"abc": {"acceptance": "threshold"}}, ("sigma", "cannot estimate sigma")),
    ({"abc": {"n_particles": 99}}, ("n_particles", ">= 100")),
    *[({"abc": {"alpha": a}}, ("alpha", "(0, 1)")) for a in (-0.1, 0, 1, 1.1)],
    *[({"next_visit": {"min_gap": lo, "max_gap": hi}}, ("next_visit", "min_gap < max_gap"))
      for lo, hi in ((-1, 6), (2, 2), (3, 2))],
    ({"backend": "abcc"}, ("backend", "not supported")),
    ({"model": "missing_model"}, ("model", "unknown model")),
]


@pytest.mark.parametrize("given, fragments", INVALID_PROFILES)
def test_invalid_profiles(given, fragments):
    with pytest.raises(ProfileError) as exc:
        check_profile(given, verbose=False)
    for fragment in fragments:
        assert fragment in str(exc.value)


def test_profile_lists_multiple_errors():
    with pytest.raises(ProfileError) as exc:
        check_profile({"forcast": {}, "abc": {"n_particles": 2, "alpha": 2},
                       "next_visit": {"min_gap": 9, "max_gap": 2}}, verbose=False)
    message = str(exc.value)
    assert "4 problem(s)" in message
    for key in ("forcast", "abc.n_particles", "abc.alpha", "next_visit"):
        assert key in message


def test_profile_accepts_valid_correlation():
    p = check_profile({"correlations": {"gamma_pre:delta_pre": 0.4}}, verbose=False)
    assert p["correlations"] == {"gamma_pre:delta_pre": 0.4}


@pytest.mark.parametrize("given, fragment", [
    ({"priors": {"gamma_pre": {"dist": "uniform", "low": 0, "high": 1}},
      "correlations": {"gamma_pre:delta_pre": 0.3}}, "correlations need lognormal or normal"),
    *[({"correlations": {"gamma_pre:delta_pre": r}}, "must be a number in (-1, 1)")
      for r in (-1, 1, -1.2, 1.2)],
    ({"correlations": {"missing:delta_pre": 0.3}}, "'missing' is not a parameter"),
])
def test_profile_correlation_errors(given, fragment):
    with pytest.raises(ProfileError) as exc:
        check_profile(given, verbose=False)
    assert fragment in str(exc.value)


@pytest.mark.parametrize("kind,args,reference", [
    ("lognormal", {"median": 0.2, "sd_log": 0.7}, lognorm(s=0.7, scale=0.2)),
    ("normal", {"mean": -0.3, "sd": 0.6}, norm(loc=-0.3, scale=0.6)),
    ("halfnormal", {"sd": 0.8}, halfnorm(scale=0.8)),
    ("uniform", {"low": -2, "high": 3}, uniform(loc=-2, scale=5)),
])
def test_dist_density_and_samples(kind, args, reference):
    dist = Dist("x", kind, args)
    # Finite tails avoid exponential overflow; omitted mass is below 1e-12.
    integral, error = quad(lambda u: np.exp(dist.logpdf_u(u)), -40, 40, epsabs=1e-9)
    assert integral == pytest.approx(1, abs=1e-8)
    assert error < 1e-7
    u = dist.sample_u(np.random.default_rng(12), 20_000)
    x = dist.to_natural(u)
    np.testing.assert_allclose(dist.to_u(x), u, atol=1e-10)
    np.testing.assert_allclose(dist.to_natural(dist.to_u(x)), x, atol=1e-10)
    np.testing.assert_allclose(reference.cdf(np.quantile(x, [0.1, 0.5, 0.9])),
                               [0.1, 0.5, 0.9], atol=0.015)


def test_fixed_dist_is_a_point_mass():
    # Fixed values have no Lebesgue density on u and are excluded from sampling.
    d = Dist("sigma", "fixed", {"value": 0.15})
    assert d.fixed and d.support() == (0.15, 0.15)
    prior = JointPrior([], {}, {"sigma": 0.15})
    u = prior.sample(np.random.default_rng(3), 10)
    assert u.shape == (10, 0)
    np.testing.assert_array_equal(prior.natural(u)["sigma"], np.full(10, 0.15))


def test_joint_prior_correlation():
    dists = {"x": Dist("x", "lognormal", {"median": 0.2, "sd_log": 0.7}),
             "y": Dist("y", "normal", {"mean": -0.1, "sd": 0.6})}
    prior = JointPrior(["x", "y"], dists, {}, {("x", "y"): 0.65})
    u = prior.sample(np.random.default_rng(4), 20_000)
    assert np.corrcoef(u.T)[0, 1] == pytest.approx(0.65, abs=0.05)
    assert np.isfinite(prior.logpdf(u)).all()


def test_number_prior_is_fixed(record, profile):
    profile["priors"]["sigma"] = 0.15
    snap = Forecaster(profile).fit(record.upto(record.schedule.t_end))
    assert "sigma" not in snap.posterior.problem.names
    np.testing.assert_array_equal(snap.posterior.natural()["sigma"], np.full(300, 0.15))


@pytest.fixture
def small_record():
    return PatientRecord.from_arrays("SYN_TEST", [0.5, 2, 4.5, 6], [120, 70, 20, 12],
                                     [0, 2], 4, [5])


@pytest.mark.parametrize("t_now,mode", [(2, "pre_only"), (6, "level")])
def test_problem_anchors_and_levels(small_record, t_now, mode):
    pb = InferenceProblem(small_record, t_now, check_profile({}, verbose=False))
    assert pb.a_pre == 0.5
    assert pb.post_mode == mode
    assert pb.a_post == (4 if mode == "pre_only" else 4.5)
    assert "jump" not in pb.names                     # no surgery-fraction parameter at all
    assert ("level_post" in pb.names) == (mode == "level")
    assert any(n.startswith("shift_") for n in pb.names) == (mode == "level")
    u = pb.prior.sample(np.random.default_rng(3), 7)
    nat = pb.natural(u)
    curve = pb.simulate(u, [pb.a_pre, pb.a_post_report])
    np.testing.assert_allclose(nat["level_pre"], curve[:, 0])
    np.testing.assert_allclose(nat["level_pre"], pb.prior.natural(u)["level_pre"])
    if mode == "level":
        np.testing.assert_allclose(nat["level_post"], curve[:, 1])
        np.testing.assert_allclose(curve[:, 1], pb.prior.natural(u)["level_post"])
    else:                                             # nothing forecast after surgery
        assert np.isnan(curve[:, 1]).all() and np.isnan(nat["gamma_post"]).all()
        assert np.isnan(pb.recurrence(u, 36)).all()


def test_pre_only_snapshot_has_no_post_forecast(small_record):
    p = check_profile({"abc": {"n_particles": 300}}, verbose=False)
    snap = Forecaster(p).fit(small_record, 2)
    r = snap.report()
    assert np.isnan(r["recurrence_median"]) and np.isnan(r["next_visit"])
    b = snap.band()
    assert b.loc[b.t >= small_record.schedule.t_surgery, "q50"].isna().all()
    assert b.loc[b.t < small_record.schedule.t_surgery, "q50"].notna().all()
    nxt = Forecaster(p).step(snap, 4.5, 20).shrinkage()
    assert np.isnan(nxt["surprise"])                  # the old fit had no forecast there
    with pytest.raises(ProfileError, match="jump"):
        check_profile({"priors": {"jump": 0.2}}, verbose=False)


@pytest.mark.parametrize("censor", [None, 35])
def test_loglik_matches_hand_computation(small_record, censor):
    p = check_profile({"data": {"censor_below": censor}, "priors": {"sigma": 0.15}}, verbose=False)
    pb = InferenceProblem(small_record, None, p)
    values = dict(gamma_pre=0.1, delta_pre=0.8, shift_gamma=0.2, shift_delta=-0.1,
                  level_pre=120, level_post=20)
    u = np.array([[pb.prior.dists[n].to_u(values[n]) for n in pb.names]])
    log_t = np.log(pb.simulate(u, pb.t)[0])
    z = (np.log(pb.y) - log_t) / 0.15
    terms = -0.5 * z**2 - np.log(0.15) - 0.5 * np.log(2 * np.pi)
    if censor:
        mask = pb.y < censor
        assert mask.sum() == 2
        terms[mask] = norm.logcdf((np.log(censor) - log_t[mask]) / 0.15)
    assert pb.loglik(u)[0] == pytest.approx(terms.sum(), abs=1e-10)


def test_decay_reduces_old_reading_influence(small_record):
    # Fixed anchors keep the predicted curve unchanged when an old reading changes.
    base = {"priors": {"level_pre": 120, "level_post": 20, "sigma": 0.15}}
    plain = InferenceProblem(small_record, None, check_profile(base, verbose=False))
    u = plain.prior.sample(np.random.default_rng(1), 1)
    y = plain.simulate(u, plain.t)[0]
    exact = replace(small_record, y=y)
    disturbed = replace(exact, y=y * np.array([2, 1, 1, 1]))
    losses = []
    for tau in (None, 2.0):
        p = check_profile({**base, "data": {"decay_tau": tau}}, verbose=False)
        a, b = [InferenceProblem(r, None, p) for r in (exact, disturbed)]
        losses.append(a.loglik(u)[0] - b.loglik(u)[0])
        if tau:
            np.testing.assert_allclose(a.w, np.exp(-(a.t_now - a.t) / tau))
            assert a.w[0] < a.w[-1] == 1
    assert 0 < losses[1] < losses[0]
    assert losses[1] / losses[0] == pytest.approx(np.exp(-(6 - 0.5) / 2))


def assert_posterior(post, expected_n=None):
    assert isinstance(post, Posterior)
    assert post.u.shape == (len(post.weights), len(post.names))
    if expected_n is not None:
        assert len(post.weights) == expected_n
    assert post.names == post.problem.names == post.problem.prior.names
    assert np.isfinite(post.u).all()
    assert np.isfinite(post.weights).all() and (post.weights >= 0).all()
    assert post.weights.sum() == pytest.approx(1)
    assert 1 <= post.ess() <= len(post.weights) + 1e-8


def test_abc_likelihood_covers_clean_truth(record, profile):
    clean = replace(record, y=np.interp(record.t, record.truth_t, record.truth_curve))
    profile["abc"].update(n_particles=1500, n_moves=12)
    profile["priors"]["sigma"] = 0.05
    snap = Forecaster(profile).fit(clean)
    assert_posterior(snap.posterior, 1500)
    assert snap.posterior.diagnostics["converged"]
    assert snap.posterior.diagnostics["beta"] == 1
    summary = snap.summary().set_index("parameter")
    for name in ("gamma_post", "delta_post"):
        assert summary.loc[name, "q10"] <= record.truth[name] <= summary.loc[name, "q90"]
        assert summary.loc[name, "q90"] / summary.loc[name, "q10"] < 1.5


def test_abc_threshold_returns_posterior(record, profile):
    profile["priors"]["sigma"] = 0.15
    profile["abc"].update(acceptance="threshold", max_generations=3,
                          max_sims_per_gen=6000, batch=600)
    profile["abc"]["n_moves"] = default_profile()["abc"]["n_moves"]
    snap = Forecaster(profile).fit(record.upto(record.schedule.t_end))
    assert_posterior(snap.posterior, 300)
    assert snap.posterior.diagnostics["acceptance"] == "threshold"
    assert snap.posterior.diagnostics["generations"]
    assert np.isfinite(snap.posterior.diagnostics["distances"]).all()
    lo, mid, hi = snap.recurrence_interval()
    assert lo <= mid <= hi


def test_fit_step_and_reports(record, profile):
    fc = Forecaster(profile)
    observed = record.upto(record.schedule.t_end)
    snap = fc.fit(observed, t_now=record.schedule.t_end)
    assert isinstance(snap, Snapshot)
    report = snap.report()
    assert set(report) == {"patient", "t_now", "n_readings", "backend", "recurrence_low",
                           "recurrence_median", "recurrence_high", "recurrence_level",
                           "recurrence_capped_at", "p_recur_6mo", "p_recur_12mo", "next_visit",
                           "band_at_next_visit", "seconds"}
    assert report["patient"] == "SYN38" and report["n_readings"] == len(observed.t)
    assert report["recurrence_low"] <= report["recurrence_median"] <= report["recurrence_high"]
    assert 0 <= report["p_recur_6mo"] <= report["p_recur_12mo"] <= 1
    i = len(observed.t)
    updated = fc.step(snap, record.t[i], record.y[i])
    assert updated.previous is snap
    assert updated.seed_used == snap.seed_used + 1
    assert len(updated.posterior.problem.t) == i + 1
    assert len(snap.record.t) == i
    shrink = updated.shrinkage()
    assert set(shrink) == {"t_before", "t_after", "at", "band_ratio_before", "band_ratio_after",
                           "recurrence_width_before", "recurrence_width_after",
                           "new_reading_month", "new_reading", "pit", "surprise"}
    assert shrink["t_before"] == snap.t_now and shrink["t_after"] == record.t[i]
    assert 0 <= shrink["pit"] <= 1 and 0 <= shrink["surprise"] <= 1
    assert updated.report()["shrinkage"] == shrink


def test_replay_tables(record, history):
    assert isinstance(history, History) and len(history) == len(record.t) - 1
    np.testing.assert_array_equal(history.times, record.t[1:])
    table, params = history.table(), history.parameter_table()
    assert len(table) == len(history)
    np.testing.assert_array_equal(table.t_now, history.times)
    assert not params.duplicated(["t_now", "parameter"]).any()
    assert len(params) == sum(len(s.summary()) for s in history.snapshots)
    for i, snap in enumerate(history.snapshots):
        assert len(snap.posterior.problem.t) == i + 2
        assert snap.previous is (history[i - 1] if i else None)
        rows = params[params.t_now == snap.t_now]
        assert set(rows.parameter) == set(snap.summary().parameter)
        rows = rows.dropna(subset=["q50"])   # post-surgery rows are empty before a post reading
        assert (rows.q10 <= rows.q50).all() and (rows.q50 <= rows.q90).all()


@pytest.mark.parametrize("method", ["plot_tube", "plot_panels", "plot_parameters", "plot_parameter_space",
                                    "plot_recurrence", "plot_recurrence_cdf", "recurrence_cdf_overlay",
                                    "snapshot", "snapshot_recurrence", "snapshot_recurrence_cdf"])
def test_plots_save(history, tmp_path, method):
    assert matplotlib.get_backend().lower() == "agg"
    path = tmp_path / f"{method}.png"
    snapshot_plots = {"snapshot": history[-1].plot, "snapshot_recurrence": history[-1].plot_recurrence,
                      "snapshot_recurrence_cdf": history[-1].plot_recurrence_cdf,
                      "recurrence_cdf_overlay": lambda path: history.plot_recurrence_cdf(path=path, facet=False)}
    plot = snapshot_plots.get(method) or getattr(history, method)
    fig = plot(path=path)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_recurrence_times(history, tmp_path):
    """No-horizon recurrence times: exact and finite for growth_death, consistent with the
    forecast inside its horizon, NaN (and an empty figure) before a post-surgery reading."""
    for snap in history.snapshots:
        rec = snap.recurrence_times(n=2000)
        if not snap.forecast.post_known:
            assert np.isnan(rec).all()
            plt.close(snap.plot_recurrence(path=tmp_path / "empty.png"))
            plt.close(snap.plot_recurrence_cdf(path=tmp_path / "empty_cdf.png"))
            continue
        assert np.isfinite(rec).all() and (rec >= 0).all()
        end = snap.record.schedule.t_end
        horizon = max(snap.t_now, end) + snap.followup - end      # the forecast's window
        share_forecast = np.mean(np.isfinite(snap.forecast.recurrence))
        assert abs(np.mean(rec <= horizon) - share_forecast) < 0.1
    assert any(not s.forecast.post_known for s in history.snapshots)   # both cases covered


@pytest.mark.skipif(__import__("importlib").util.find_spec("pymc") is None, reason="optional PyMC dependency")
def test_mcmc_contract(record):
    p = default_profile(backend="mcmc")
    p["mcmc"].update(draws=1000, tune=1000, chains=2, cores=1)
    post = Forecaster(p).fit(record).posterior
    assert_posterior(post, 2000)
    assert post.backend == "mcmc"
    assert post.names == ["gamma_pre", "delta_pre", "shift_gamma", "shift_delta",
                          "level_pre", "level_post", "sigma"]
    np.testing.assert_allclose(post.weights, np.full(2000, 1 / 2000))
    nat = post.natural()
    for i, name in enumerate(post.names):
        np.testing.assert_allclose(nat[name], post.problem.prior.dists[name].to_natural(post.u[:, i]))


@pytest.mark.skipif(__import__("importlib").util.find_spec("pymc") is None, reason="optional PyMC dependency")
def test_mcmc_matches_abc(record):
    p = default_profile()
    p["abc"].update(n_particles=3000, n_moves=20)
    p["mcmc"].update(draws=2000, tune=2000, chains=4, cores=1)
    abc = Forecaster(p).fit(record)
    other = copy.deepcopy(p)
    other["backend"] = "mcmc"
    mcmc = Forecaster(other).fit(record)
    assert abc.posterior.diagnostics["converged"]
    sa, sm = [s.summary().set_index("parameter") for s in (abc, mcmc)]
    for name in ("gamma_post", "delta_post", "sigma"):
        assert sm.loc[name, "q50"] == pytest.approx(sa.loc[name, "q50"], rel=0.15)
    a, m = abc.recurrence_interval(), mcmc.recurrence_interval()
    assert np.isfinite([a[0], m[0]]).all()
    assert max(a[0], m[0]) <= min(a[2], m[2])


@pytest.mark.skipif(__import__("importlib").util.find_spec("pymc") is None, reason="optional PyMC dependency")
@pytest.mark.parametrize("pre_only,anchor", [(False, "first_reading"), (False, "segment_start"),
                                            (True, "first_reading")])
def test_mcmc_likelihood_matches_shared_problem(record, pre_only, anchor):
    from ovca.backends.mcmc import _build_model

    p = default_profile(backend="mcmc")
    p["initial_level"]["anchor"] = anchor
    p["data"].update(censor_below=35, decay_tau=12)
    p["correlations"] = {"gamma_pre:delta_pre": 0.4}
    p["priors"]["sigma"] = {"dist": "halfnormal", "sd": 0.3}
    p["priors"]["level_pre"] = {"dist": "uniform", "low": 1, "high": 3000}
    p["priors"]["shift_delta"] = 0.0
    fc = Forecaster(p)
    t_now = record.t[record.t < record.schedule.t_surgery][-1] if pre_only else record.t[-1]
    problem = fc.problem(record, t_now=t_now)
    model = _build_model(problem)
    potential = model.replace_rvs_by_values([model["loglik"]])[0]
    fn = model.compile_fn(potential)
    rng = np.random.default_rng(831)
    for u in problem.prior.sample(rng, 5):
        point = model.initial_point()
        gauss = u[problem.prior._gauss]
        point["prior_z"] = np.linalg.solve(problem.prior._chol, gauss - problem.prior._mean)
        for i in problem.prior._other:
            name = problem.names[i]
            suffix = "_log__" if problem.prior.dists[name].kind == "halfnormal" else "_interval__"
            point[name + suffix] = u[i]
        assert float(fn(point)) == pytest.approx(problem.loglik(u)[0], abs=1e-7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", *sys.argv[1:]]))
