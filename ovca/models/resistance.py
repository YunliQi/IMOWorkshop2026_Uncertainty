"""Sensitive and resistant cells with chemotherapy pulses (Team Purple's S/R model).

    dS/dt = g (1 - D_S(t)) S - m S - k c(t) S
    dR/dt = g (1 - D_R(t)) R + m S - eps k c(t) R
    D_S = min(d_bev u_bev(t), 1),   D_R = min(eps_bev d_bev u_bev(t), 1)
    c(t) = 1 for `pulse` after each infusion (every NACT and ACT cycle start), else 0
    surgery: S, R -> q S, q R;      CA-125 = a (S + R) + b0

Estimated per patient (per month): g growth, k chemotherapy kill on S while a pulse is on,
rho = R0 / S0 at the first reading. The tool also estimates b0 (background CA-125) and the
levels: a, S0 and R0 only enter through a (S0 + R0), which is the level parameter, and their
ratio rho; the surgery fraction (beta = 1 - q) comes from the post-surgery readings.

Fixed (team table, converted to months with 30 days per month):
    m       S -> R switching    6.7e-6 per day  (Kozlowska: 1e-5 per division x 0.667 / day)
    eps     kill on R / kill on S   0.02        (Kozlowska alpha1)
    d_bev   bevacizumab growth brake 0.25       (1 - HR, GOG-218 / ICON7)
    eps_bev BEV effect on R / on S   1.0
    pulse   3 days per infusion
PARP inhibitors are maintenance only, after the frontline window the tool fits, so they are
left out. Within a stretch where c and u_bev are constant the system is linear and lower
triangular, so it is solved exactly (no ODE solver).
"""
import numpy as np

from ..patient import PatientParameters, drug_on
from .base import BaseModel

DPM = 30.0
LOG_CAP = np.log(1e12)


def _phi(x):
    """expm1(x) / x, = 1 at x = 0."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-8
    return np.where(small, 1.0 + 0.5 * x, np.expm1(np.where(small, 1.0, x))
                    / np.where(small, 1.0, x))


class SensitiveResistant(BaseModel):
    name = "resistance"
    param_names = ("g", "k", "rho")
    bounds = {"g": (1e-3, 5.0), "k": (1e-2, 300.0), "rho": (1e-6, 10.0)}
    starts = ((0.24, 12.0, 0.01), (0.15, 6.0, 0.05), (0.4, 20.0, 0.002))
    state_names = ("S", "R")
    scale_invariant = True
    default_shift_params = ()        # surgery removes cells but changes no parameter
    background = True                # CA-125 = a N + b0: the tool samples b0

    def __init__(self, m=6.7e-6, eps=0.02, d_bev=0.25, eps_bev=1.0, pulse_days=3.0):
        self.m = m * DPM                     # per month
        self.eps, self.d_bev, self.eps_bev = eps, d_bev, eps_bev
        self.pulse = pulse_days / DPM        # months

    # ---- treatment inputs --------------------------------------------------------------
    def infusions(self, schedule):
        return np.sort(np.concatenate([np.asarray(schedule.nact_starts, float),
                                       np.asarray(schedule.act_starts, float)]))

    def pulse_intervals(self, schedule):
        d = self.infusions(schedule)
        return np.column_stack([d, d + self.pulse]) if d.size else np.zeros((0, 2))

    def drug_free_from(self, schedule):
        """After this time no pulse and no BEV is on."""
        ends = [schedule.t_end]
        for iv in (self.pulse_intervals(schedule), schedule.bev_intervals()):
            if iv.size:
                ends.append(float(iv[:, 1].max()))
        return max(ends)

    # ---- generic interface (reference ODE, used by tests) -------------------------------
    def initial_state(self, ca125_0, theta):
        rho = float(theta["rho"])
        return float(ca125_0) * np.array([1.0, rho]) / (1.0 + rho)

    def rhs_full(self, t, x, theta, schedule):
        S, R = x
        c = drug_on([t], self.pulse_intervals(schedule))[0]
        u = drug_on([t], schedule.bev_intervals())[0]
        ds, dr = min(self.d_bev * u, 1.0), min(self.eps_bev * self.d_bev * u, 1.0)
        g, k = theta["g"], theta["k"]
        return [g * (1 - ds) * S - self.m * S - k * c * S,
                g * (1 - dr) * R + self.m * S - self.eps * k * c * R]

    def rhs(self, t, x, theta, drug):
        raise NotImplementedError("the S/R model needs the schedule; use simulate_patient")

    # ---- exact solution ----------------------------------------------------------------
    def _log_n(self, t, theta, schedule):
        """log N(t) - log N(0) without surgery (surgery scales S and R alike, so it only adds
        log q). N(0) = 1 split as S : R = 1 : rho. t >= 0. Returns (N, len(t))."""
        return self._solve(t, theta, schedule)[0]

    def resistant_fraction(self, t, theta, schedule):
        """R / (S + R) at times t, (N, len(t)); surgery keeps the mix."""
        return self._solve(t, theta, schedule)[1]

    def _solve(self, t, theta, schedule):
        t = np.atleast_1d(np.asarray(t, float))
        g = np.atleast_1d(np.asarray(theta["g"], float))
        k = np.atleast_1d(np.asarray(theta["k"], float))
        rho = np.atleast_1d(np.asarray(theta["rho"], float))
        n = max(len(g), len(k), len(rho))
        g, k, rho = (np.broadcast_to(v, (n,)) for v in (g, k, rho))
        pulses, bev = self.pulse_intervals(schedule), schedule.bev_intervals()
        tq = np.unique(t)
        edges = np.concatenate([pulses.ravel(), bev.ravel()])
        edges = edges[(edges > 0) & (edges < tq[-1])] if tq.size else edges[:0]
        pts = np.unique(np.concatenate([[0.0], edges, tq]))
        out = np.empty((n, pts.size))
        frac = np.empty((n, pts.size))
        s, r, logn = 1.0 / (1.0 + rho), rho / (1.0 + rho), np.zeros(n)
        out[:, 0], frac[:, 0] = 0.0, r
        for i in range(1, pts.size):
            a, b = pts[i - 1], pts[i]
            h, mid = b - a, 0.5 * (a + b)
            c = drug_on([mid], pulses)[0]
            u = drug_on([mid], bev)[0]
            ds, dr = min(self.d_bev * u, 1.0), min(self.eps_bev * self.d_bev * u, 1.0)
            a1 = g * (1 - ds) - self.m - k * c
            a2 = g * (1 - dr) - self.eps * k * c
            e2 = np.exp(a2 * h)
            s_new = s * np.exp(a1 * h)
            r_new = r * e2 + self.m * s * h * e2 * _phi((a1 - a2) * h)
            tot = s_new + r_new
            logn = logn + np.log(tot)
            s, r = s_new / tot, r_new / tot
            out[:, i], frac[:, i] = logn, r
        idx = np.searchsorted(pts, t)
        return out[:, idx], frac[:, idx]

    # ---- the tool's interface -----------------------------------------------------------
    def segment_log_growth(self, t, t0, theta, schedule, phase):
        t = np.atleast_1d(np.asarray(t, float))
        ln = self._log_n(np.concatenate([[t0], t]), theta, schedule)
        return ln[:, 1:] - ln[:, :1]

    def log_growth_batch(self, t, t0, theta, intervals):
        raise NotImplementedError("the S/R model needs the whole schedule; use "
                                  "segment_log_growth")

    def simulate_patient_batch(self, t, schedule, theta_pre, theta_post, ca125_0, q):
        """Tumour CA-125 a N(t) (without the background b0), (N, len(t))."""
        s = schedule
        t = np.atleast_1d(np.asarray(t, float))
        ts = s.t_surgery
        log_c0 = np.log(np.asarray(ca125_0, float))[:, None]
        pre = self._log_n(np.append(np.minimum(t, ts), ts), theta_pre, s)
        log_pre, log_ts = log_c0 + pre[:, :-1], log_c0 + pre[:, -1:]
        post = self._log_n(np.append(np.maximum(t, ts), ts), theta_post, s)
        log_post = (log_ts + np.log(np.asarray(q, float))[:, None]
                    + post[:, :-1] - post[:, -1:])
        return np.exp(np.minimum(np.where(t[None, :] < ts, log_pre, log_post), LOG_CAP))

    def simulate_patient(self, t, p: PatientParameters):
        one = lambda th: {k: np.array([float(v)]) for k, v in th.items()}   # noqa: E731
        return self.simulate_patient_batch(t, p.schedule, one(p.theta_pre), one(p.theta_post),
                                           np.array([p.ca125_0]), np.array([p.q_surgery]))[0]

    def simulate_segment(self, t, t0, ca125_0, theta, intervals):
        raise NotImplementedError("the S/R model needs the whole schedule; use "
                                  "simulate_patient / simulate_patient_batch")

    def recurrence_time_batch(self, schedule, theta_pre, theta_post, ca125_0, q, cutoff,
                              followup, step=None):
        """Exact. cutoff is the tumour part a N to reach (array of N when b0 varies). Once no
        drug is on, S and R both grow at g (switching keeps S + R), so log N is linear."""
        s = schedule
        t0 = self.drug_free_from(s)
        log_t0 = np.log(self.simulate_patient_batch([t0], s, theta_pre, theta_post,
                                                    ca125_0, q)[:, 0])
        cut = np.broadcast_to(np.asarray(cutoff, float), log_t0.shape)
        g = np.broadcast_to(np.asarray(theta_post["g"], float), log_t0.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(cut > 0, np.log(np.where(cut > 0, cut, 1.0)) - log_t0, -np.inf)
            dt = np.where(gap <= 0, 0.0, np.where(g > 0, gap / g, np.inf))
        return np.where(t0 + dt <= s.t_end + followup, t0 + dt, np.inf)
