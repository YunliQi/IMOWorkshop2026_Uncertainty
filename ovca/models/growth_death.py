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
