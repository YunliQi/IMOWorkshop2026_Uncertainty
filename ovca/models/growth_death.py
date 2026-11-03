"""The poster model: exponential growth, killed while treatment is on.

    dT/dt = (gamma - delta * D(t)) * T

Because D(t) is 0 or 1, the exact solution is
    log T(t) = log T(t0) + gamma * (t - t0) - delta * (time on treatment between t0 and t),
so `solve` skips the ODE solver (tests check it against BaseModel's numerical solve).
"""
import numpy as np

from ..patient import time_on
from .base import BaseModel

LOG_CAP = np.log(1e12)   # keeps extreme parameter guesses from overflowing during fitting


class GrowthDeath(BaseModel):
    name = "growth_death"
    param_names = ("gamma", "delta")
    bounds = {"gamma": (1e-3, 10.0), "delta": (1e-3, 10.0)}   # per month, as in the poster code
    starts = ((0.1, 0.5), (0.5, 0.5), (0.3, 2.0), (1.0, 3.0), (0.05, 1.0))

    def rhs(self, t, x, theta, drug):
        return (theta["gamma"] - theta["delta"] * drug) * x

    def solve(self, t, t0, x0, theta, intervals):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        log_x0 = np.log(max(float(np.asarray(x0).ravel()[0]), 1e-300))
        log_x = log_x0 + theta["gamma"] * (t - t0) - theta["delta"] * time_on(t, t0, intervals)
        return np.exp(np.minimum(log_x, LOG_CAP))[:, None]

    # ---- vectorised versions of the base-class batch methods (exact solution) ----------
    def _log_t_surgery(self, s, th_pre, ca125_0):
        on = time_on([s.t_surgery], 0.0, s.intervals("pre"))[0]
        return np.log(ca125_0) + th_pre["gamma"] * s.t_surgery - th_pre["delta"] * on

    def simulate_patient_batch(self, t, schedule, theta_pre, theta_post, ca125_0, q):
        s = schedule
        t = np.atleast_1d(np.asarray(t, dtype=float))
        ts = s.t_surgery
        g1, d1 = np.asarray(theta_pre["gamma"])[:, None], np.asarray(theta_pre["delta"])[:, None]
        g2, d2 = np.asarray(theta_post["gamma"])[:, None], np.asarray(theta_post["delta"])[:, None]
        t_pre = np.minimum(t, ts)
        log_pre = (np.log(np.asarray(ca125_0))[:, None] + g1 * t_pre
                   - d1 * time_on(t_pre, 0.0, s.intervals("pre"))[None, :])
        log_ts = self._log_t_surgery(s, theta_pre, np.asarray(ca125_0))[:, None]
        t_post = np.maximum(t, ts)
        log_post = (log_ts + np.log(np.asarray(q))[:, None] + g2 * (t_post - ts)
                    - d2 * time_on(t_post, ts, s.intervals("post"))[None, :])
        return np.exp(np.minimum(np.where(t[None, :] < ts, log_pre, log_post), LOG_CAP))

    def recurrence_time_batch(self, schedule, theta_pre, theta_post, ca125_0, q, cutoff,
                              followup, step=None):
        """Exact: after the end of ACT, D = 0 and log T grows linearly at gamma_post."""
        s = schedule
        log_end = np.log(self.simulate_patient_batch([s.t_end], s, theta_pre, theta_post,
                                                     ca125_0, q)[:, 0])
        g = np.asarray(theta_post["gamma"], dtype=float)
        gap = np.log(cutoff) - log_end
        with np.errstate(divide="ignore", invalid="ignore"):
            dt = np.where(gap <= 0, 0.0, np.where(g > 0, gap / g, np.inf))
        return np.where(dt <= followup, s.t_end + dt, np.inf)
