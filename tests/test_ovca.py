"""Run:  .venv/bin/python -m pytest tests -q     (or: .venv/bin/python tests/test_ovca.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ovca import calibration, generator  # noqa: E402
from ovca.models import BaseModel, get_model  # noqa: E402
from ovca.patient import Schedule, merge_intervals  # noqa: E402


def test_exact_solution_matches_numerical_ode():
    """GrowthDeath's closed form agrees with BaseModel's generic ODE solver."""
    m = get_model("growth_death")
    rng = np.random.default_rng(0)
    for _ in range(20):
        iv = merge_intervals(np.sort(rng.uniform(0, 4, rng.integers(0, 5))))
        t = np.sort(np.concatenate([[0.0], rng.uniform(0, 8, 15)]))
        theta = dict(gamma=rng.uniform(0.01, 0.8), delta=rng.uniform(0.2, 3.0))
        x0 = [rng.uniform(30, 3000)]
        exact = m.solve(t, 0.0, x0, theta, iv)
        numeric = BaseModel.solve(m, t, 0.0, x0, theta, iv)
        assert np.allclose(exact, numeric, rtol=1e-6), np.max(np.abs(exact / numeric - 1))


def test_surgery_jump():
    m = get_model("growth_death")
    p = generator.sample_parameters(np.random.default_rng(1), m, "X")
    ts = p.schedule.t_surgery
    before, after = m.simulate_patient([ts - 1e-9, ts], p)
    assert np.isclose(after / before, p.q_surgery, rtol=1e-6)


def test_recurrence_time_without_treatment():
    """T = 10 exp(0.2 t) after surgery with no ACT left: crosses 35 at ln(3.5)/0.2."""
    m = get_model("growth_death")
    p = generator.sample_parameters(np.random.default_rng(2), m, "X")
    p.schedule = Schedule([], 1.0, [])               # t_end = t_surgery = 1
    p.theta_post = dict(gamma=0.2, delta=1.0)
    p.ca125_0, p.theta_pre, p.q_surgery = 10.0, dict(gamma=0.0, delta=0.0), 1.0
    tau = generator.recurrence_time(m, p)
    assert abs(tau - (1.0 + np.log(3.5) / 0.2)) < 0.011


def test_rmse_leaves_out_low_readings_and_t0():
    y_obs = np.array([500.0, 100.0, 20.0, 50.0])
    y_pred = np.array([0.0, 110.0, 1e6, 40.0])        # T0 and the < 35 point are ignored
    assert np.isclose(calibration.rmse(y_obs, y_pred), np.sqrt((10**2 + 10**2) / 2))


def test_calibration_recovers_parameters_on_clean_data():
    """With treatment gaps (so gamma and delta are separable) and no noise, the fit is exact."""
    m = get_model("growth_death")
    iv = merge_intervals([0.5, 3.0])
    t = np.linspace(0, 5, 12)
    theta = dict(gamma=0.3, delta=0.9)
    y = m.simulate_segment(t, 0.0, 2000.0, theta, iv)
    f = calibration.fit_segment(m, t, y, iv)
    assert np.allclose([f["gamma"], f["delta"]], [0.3, 0.9], rtol=1e-4)


class _Logistic(BaseModel):
    """A model with no closed form, to check that the generic numerical path works end to end."""
    name = "logistic_test"
    param_names = ("r", "delta")
    bounds = {"r": (1e-3, 5.0), "delta": (1e-3, 10.0)}
    starts = ((0.3, 1.0), (1.0, 2.0))
    K = 1e5

    def rhs(self, t, x, theta, drug):
        return theta["r"] * x * (1 - x / self.K) - theta["delta"] * drug * x


def test_new_model_through_generator_and_calibration():
    m = _Logistic()
    p = generator.sample_parameters(np.random.default_rng(3), m, "L", population=dict(
        names=["r_pre", "delta_pre", "r_post", "delta_post"], median=[0.3, 1.2, 0.3, 1.2],
        sd_log=[0.01] * 4, corr=np.eye(4)))
    obs, truth = generator.generate_longitudinal(m, p, np.random.default_rng(4), noise_sd=0.0)
    assert np.isfinite(obs.ca125).all() and truth["model"] == "logistic_test"
    t, y, iv = calibration.segment_data(obs, p.schedule, "pre")
    f = calibration.fit_segment(m, t, y, iv)
    assert f["fitted"] and f["rmse"] < 1e-3 * y.max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
