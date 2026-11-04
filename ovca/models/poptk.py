"""PopTK: tumour kinetics with heterogeneous drug sensitivity (Team Purple's popTK_ovarian.py).

Between doses the tumour grows exponentially. Each dose kills instantly, and cells differ in
how sensitive they are: a cell's log10 sensitivity is x ~ N(mu, het^2) within the tumour, and
after n doses the surviving fraction is

    S(n) = E_x[ exp(-n * 10^x) ]           (Gauss-Hermite quadrature over x)

so the n-th dose multiplies the tumour by S(n) / S(n - 1). Sensitive cells die first, so each
dose kills less than the one before (selection of pre-existing resistance). Surgery multiplies
the tumour by q and leaves the sensitivity mix unchanged, so the dose count carries over: ACT
kill depends on how many NACT doses came before.

Mapping to the original code (popTK_ovarian.py, times in days, tumour in cells):
    growth   growth_rate, per month here               (= ln 2 / doubling time)
    kill     log10 kill per dose of the median cell     (= 10^sensitivity / ln 10, dose = 1)
    het      within-tumour sd of log10 sensitivity      (the original `sd`, default 0.083)
Simplifications (agreed with the team):
    - one combined chemotherapy drug, one dose at each cycle start (Schedule.nact_starts and
      act_starts); no VEGFi (the frontline window ends before maintenance), so `correlation`,
      `interaction` and `dose` drop out;
    - CA-125 = c x cells: c and initial_size only enter as a product, which is the tool's
      level parameter; for the same reason the cure threshold (absolute cells) is not used;
    - exponential growth only (Gompertz would break scale invariance).
A reading on a dose day is taken before that dose, as in the original code.
"""
import numpy as np
from scipy.special import logsumexp, roots_hermite

from ..patient import PatientParameters
from .base import BaseModel

LN10 = np.log(10.0)
LOG_CAP = np.log(1e12)
N_NODES = 40


def dose_times(schedule):
    """Every chemotherapy dose, in months: one per NACT and ACT cycle start."""
    return np.sort(np.concatenate([np.asarray(schedule.nact_starts, float),
                                   np.asarray(schedule.act_starts, float)]))


def doses_before(t, schedule):
    """Number of doses given strictly before each time t (a reading on a dose day comes first)."""
    return np.searchsorted(dose_times(schedule), np.atleast_1d(np.asarray(t, float)), side="left")


class PopTK(BaseModel):
    name = "poptk"
    param_names = ("growth", "kill", "het")
    bounds = {"growth": (1e-3, 5.0), "kill": (1e-3, 5.0), "het": (1e-3, 2.0)}
    starts = ((0.15, 0.3, 0.15), (0.3, 0.5, 0.3), (0.05, 0.2, 0.08))
    scale_invariant = True
    default_shift_params = ()        # surgery removes cells but changes no parameter

    def __init__(self, n_nodes=N_NODES):
        y, w = roots_hermite(n_nodes)
        self._z = np.sqrt(2.0) * y                       # standard-normal nodes
        self._log_w = np.log(w) - 0.5 * np.log(np.pi)    # weights sum to 1

    def rhs(self, t, x, theta, drug):
        """Growth only: the kill happens at dose times, so the generic ODE solver is not used."""
        return theta["growth"] * x

    # ---- the surviving fraction -------------------------------------------------------------
    def log_surviving(self, n, kill, het):
        """log S(n) for N parameter sets and dose counts n: kill, het arrays of N; n an int array.
        Returns (N, len(n))."""
        kill = np.atleast_1d(np.asarray(kill, float))[:, None, None]
        het = np.atleast_1d(np.asarray(het, float))[:, None, None]
        n = np.atleast_1d(np.asarray(n, float))[None, :, None]
        # exponent 10^x per dose, x = mu + het * z, 10^mu = kill * ln 10
        per_dose = kill * LN10 * np.exp(LN10 * het * self._z[None, None, :])
        return logsumexp(self._log_w[None, None, :] - n * per_dose, axis=2)

    def _log_t(self, t, t0, theta, schedule):
        """log T(t) - log T(t0) for t, t0 in the same segment (no surgery in between)."""
        t = np.atleast_1d(np.asarray(t, float))
        g = np.atleast_1d(np.asarray(theta["growth"], float))[:, None]
        n_t, n_0 = doses_before(t, schedule), doses_before([t0], schedule)
        counts, inv = np.unique(np.concatenate([n_0, n_t]), return_inverse=True)
        ls = self.log_surviving(counts, theta["kill"], theta["het"])
        return g * (t - t0)[None, :] + ls[:, inv[1:]] - ls[:, inv[:1]]

    # ---- the tool's interface -----------------------------------------------------------------
    def segment_log_growth(self, t, t0, theta, schedule, phase):
        return self._log_t(t, t0, theta, schedule)

    def log_growth_batch(self, t, t0, theta, intervals):
        raise NotImplementedError("PopTK needs the whole dose history; use segment_log_growth")

    def simulate_patient_batch(self, t, schedule, theta_pre, theta_post, ca125_0, q):
        s = schedule
        t = np.atleast_1d(np.asarray(t, float))
        ts = s.t_surgery
        log_c0 = np.log(np.asarray(ca125_0, float))[:, None]
        log_pre = log_c0 + self._log_t(np.minimum(t, ts), 0.0, theta_pre, s)
        log_ts = log_c0 + self._log_t([ts], 0.0, theta_pre, s)
        log_post = (log_ts + np.log(np.asarray(q, float))[:, None]
                    + self._log_t(np.maximum(t, ts), ts, theta_post, s))
        return np.exp(np.minimum(np.where(t[None, :] < ts, log_pre, log_post), LOG_CAP))

    def simulate_patient(self, t, p: PatientParameters):
        one = lambda th: {k: np.array([float(v)]) for k, v in th.items()}   # noqa: E731
        return self.simulate_patient_batch(t, p.schedule, one(p.theta_pre), one(p.theta_post),
                                           np.array([p.ca125_0]), np.array([p.q_surgery]))[0]

    def simulate_segment(self, t, t0, ca125_0, theta, intervals):
        raise NotImplementedError("PopTK segments depend on the dose history; "
                                  "use simulate_patient / simulate_patient_batch")

    def recurrence_time_batch(self, schedule, theta_pre, theta_post, ca125_0, q, cutoff,
                              followup, step=None):
        """Exact: after the last dose there is no more kill, so log T grows linearly at growth."""
        s = schedule
        t0 = max(s.t_end, float(dose_times(s)[-1]) if dose_times(s).size else s.t_end)
        log_end = np.log(self.simulate_patient_batch([t0], s, theta_pre, theta_post,
                                                     ca125_0, q)[:, 0])
        g = np.asarray(theta_post["growth"], float)
        gap = np.log(cutoff) - log_end
        with np.errstate(divide="ignore", invalid="ignore"):
            dt = np.where(gap <= 0, 0.0, np.where(g > 0, gap / g, np.inf))
        return np.where(t0 + dt <= s.t_end + followup, t0 + dt, np.inf)
