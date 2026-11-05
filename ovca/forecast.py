"""Forecast from a Posterior: tumour / CA-125 bands, recurrence, next visit, surprise."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backends.base import Posterior
from .patient import Schedule


@dataclass
class Forecast:
    t: np.ndarray                 # months
    tumour: np.ndarray            # (n, len(t)) noise-free model value
    ca125: np.ndarray             # (n, len(t)) with observation error (each draw's sigma)
    recurrence: np.ndarray        # (n,) absolute month the curve reaches the cutoff (inf = no)
    t_now: float
    schedule: Schedule
    cutoff: float
    censor_below: float = None

    @property
    def post_known(self):
        """False before the first post-surgery reading: nothing is forecast after surgery."""
        return not np.all(np.isnan(self.recurrence))

    def band(self, levels=(0.1, 0.5, 0.9), noisy=False):
        """Quantiles over draws at each time: tumour (parameter uncertainty) or CA-125
        readings (parameter uncertainty + observation error). Columns q10, q50, ..."""
        v = self.ca125 if noisy else self.tumour
        return pd.DataFrame({"t": self.t, **{f"q{_pct(l)}": np.quantile(v, l, axis=0)
                                             for l in levels}})

    def band_at(self, t, level=0.8, noisy=False):
        """(low, median, high) at one time."""
        i = min(np.searchsorted(self.t, t), len(self.t) - 1)
        v = (self.ca125 if noisy else self.tumour)[:, i]
        a = (1 - level) / 2
        return tuple(float(np.quantile(v, q)) for q in (a, 0.5, 1 - a))

    def recurrence_cdf(self, t=None):
        t = self.t if t is None else np.atleast_1d(t)
        return np.mean(self.recurrence[:, None] <= np.asarray(t)[None, :], axis=0)

    def recurrence_interval(self, level=0.8, relative=True, cap=None):
        """(low, median, high) recurrence time, months after the end of ACT by default.
        inf = not within the horizon; cap replaces inf (and larger values) with a number."""
        if not self.post_known:
            return (np.nan, np.nan, np.nan)
        r = self.recurrence - (self.schedule.t_end if relative else 0.0)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        out = [float(np.quantile(r, q, method="inverted_cdf")) for q in (lo, 0.5, hi)]
        return tuple(min(v, cap) for v in out) if cap is not None else tuple(out)

    def p_recur_within(self, months_after_end):
        if not self.post_known:
            return np.nan
        return float(np.mean(self.recurrence - self.schedule.t_end <= months_after_end))

    def next_visit(self, rule="risk", p=0.10, width=1.0, min_gap=1.0, max_gap=6.0):
        """Suggested month of the next CA-125 visit after t_now.
        risk:  first time the chance of having crossed the cutoff since t_now reaches p;
        width: first time the 80% band of log CA-125 is wider than `width`.
        Kept within [t_now + min_gap, t_now + max_gap]. NaN before the first post-surgery
        reading (nothing is forecast after surgery yet)."""
        if not self.post_known:
            return np.nan
        lo, hi = self.t_now + min_gap, self.t_now + max_gap
        cand = self.t[(self.t >= lo) & (self.t <= hi)]
        if rule == "risk":
            hit = self.recurrence_cdf(cand) - np.mean(self.recurrence <= self.t_now) >= p
        elif rule == "width":
            lq = np.log(self.ca125)[:, np.searchsorted(self.t, cand)]
            hit = np.quantile(lq, 0.9, axis=0) - np.quantile(lq, 0.1, axis=0) >= width
        else:
            raise ValueError("rule must be 'risk' or 'width'")
        return float(cand[np.argmax(hit)]) if hit.any() else float(hi)

    def pit(self, t_new, y_new):
        """P(predicted reading <= y_new); near 0 or 1 = surprising."""
        i = min(np.searchsorted(self.t, t_new), len(self.t) - 1)
        v = self.ca125[:, i]
        if not np.isfinite(v).any():          # not forecast (after surgery, pre-only fit)
            return np.nan
        if self.censor_below and y_new < self.censor_below:
            return float(np.mean(v < self.censor_below))
        return float(np.mean(v <= y_new))

    def surprise(self, t_new, y_new):
        """Two-sided tail probability of the new reading (small = surprising)."""
        f = self.pit(t_new, y_new)
        return float(2 * min(f, 1 - f)) if np.isfinite(f) else np.nan


def _pct(l):
    v = l * 100
    return f"{v:g}".replace(".", "_") if v != int(v) else str(int(v))


def make_forecast(post: Posterior, horizon, step, n_draws, condition_no_recurrence=True,
                  seed=0):
    pb = post.problem
    s = pb.schedule
    rng = np.random.default_rng(seed)
    u = post.resample(n_draws, rng)
    t_stop = max(pb.t_now, s.t_end) + horizon
    grid = np.arange(0.0, t_stop + step / 2, step)
    tumour = pb.simulate(u, grid)
    tau = pb.recurrence(u, t_stop - s.t_end)
    if condition_no_recurrence:
        surv = pb.t > s.t_end + 1e-9
        if surv.any() and not pb.observed_recurrence():
            keep = tau > pb.t[surv].max()
            if keep.sum() >= 50:
                u, tumour, tau = u[keep], tumour[keep], tau[keep]
    sigma = pb.components(u)[4]
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (len(u),))[:, None]
    noisy = tumour * np.exp(sigma * rng.standard_normal(tumour.shape))
    return Forecast(grid, tumour, noisy, tau, pb.t_now, s, pb.cutoff, pb.censor)
