"""Base class for tumour models.

A model describes one treatment segment (before or after surgery):

    dx/dt = rhs(t, x, theta, D(t)),     CA-125 = observe(x)

with D(t) = 1 while a treatment is active. A patient is simulated as
pre-surgery segment -> surgery (x -> surgery(x, q)) -> post-surgery segment, each segment with
its own parameters theta_pre / theta_post (the two halves of the poster model).

To add a model, make a new file in ovca/models/ with a subclass that sets `name`,
`param_names`, `bounds`, `starts` and implements `rhs`; then register it in models/__init__.py.
Override `initial_state`, `observe` or `surgery` if the model has more than one state
(e.g. sensitive and resistant cells), and `solve` if there is an exact solution.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp

from ..patient import PatientParameters, drug_on


class BaseModel(ABC):
    name = "base"
    param_names: tuple = ()          # per-segment parameters, e.g. ("gamma", "delta")
    bounds: dict = {}                # name -> (low, high), used by calibration
    starts: tuple = ()               # starting guesses for calibration, in param_names order
    state_names: tuple = ("T",)

    # ---- model definition --------------------------------------------------------------
    @abstractmethod
    def rhs(self, t, x, theta, drug):
        """Time derivative of the state x, with drug = D(t) (0 or 1)."""

    def initial_state(self, ca125_0, theta):
        """State that gives CA-125 = ca125_0. Default: a single state equal to CA-125."""
        return np.array([float(ca125_0)])

    def observe(self, x):
        """CA-125 from states; x has shape (n_times, n_states). Default: sum of states."""
        return np.asarray(x).sum(axis=-1)

    def surgery(self, x, q):
        """State right after surgery. Default: every state is multiplied by q."""
        return q * np.asarray(x, dtype=float)

    # ---- simulation --------------------------------------------------------------------
    def solve(self, t, t0, x0, theta, intervals):
        """States at sorted times t >= t0, starting from x0 at t0. Returns (len(t), n_states).

        Generic numerical solution: integrates piece by piece between the times D(t)
        switches, so the solver never steps over a jump.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x0 = np.asarray(x0, dtype=float)
        out = np.empty((t.size, x0.size))
        if t.size == 0:
            return out
        if np.any(np.diff(t) < 0) or t[0] < t0 - 1e-12:
            raise ValueError("t must be sorted and start at or after t0")
        out[t <= t0] = x0
        edges = intervals.ravel()
        cuts = np.unique(np.concatenate([[t0], edges[(edges > t0) & (edges < t[-1])], [t[-1]]]))
        x = x0
        for a, b in zip(cuts[:-1], cuts[1:]):
            drug = float(drug_on(0.5 * (a + b), intervals)[0])
            sel = (t > a) & (t <= b)
            t_eval = t[sel] if sel.any() and t[sel][-1] == b else np.append(t[sel], b)
            sol = solve_ivp(self.rhs, (a, b), x, t_eval=t_eval, args=(theta, drug),
                            method="LSODA", rtol=1e-10, atol=1e-12)
            if not sol.success:
                raise RuntimeError(sol.message)
            out[sel] = sol.y[:, :sel.sum()].T
            x = sol.y[:, -1]
        return out

    def simulate_segment(self, t, t0, ca125_0, theta, intervals):
        """CA-125 over one segment that starts at CA-125 = ca125_0 at time t0."""
        x0 = self.initial_state(ca125_0, theta)
        return self.observe(self.solve(t, t0, x0, theta, intervals))

    def simulate_patient(self, t, p: PatientParameters):
        """Noise-free CA-125 of a patient at sorted times t (months from t = 0)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        s = p.schedule
        t_pre, t_post = t[t < s.t_surgery], t[t >= s.t_surgery]
        x_pre = self.solve(np.append(t_pre, s.t_surgery), 0.0,
                           self.initial_state(p.ca125_0, p.theta_pre), p.theta_pre,
                           s.intervals("pre"))
        x_after = self.surgery(x_pre[-1], p.q_surgery)
        x_post = self.solve(t_post, s.t_surgery, x_after, p.theta_post, s.intervals("post"))
        return np.concatenate([self.observe(x_pre[:-1]), self.observe(x_post)])

    # ---- helpers -----------------------------------------------------------------------
    def theta(self, vector):
        return dict(zip(self.param_names, map(float, vector)))

    def vector(self, theta):
        return np.array([theta[k] for k in self.param_names], dtype=float)

    def bounds_arrays(self):
        lo = np.array([self.bounds[k][0] for k in self.param_names], dtype=float)
        hi = np.array([self.bounds[k][1] for k in self.param_names], dtype=float)
        return lo, hi
