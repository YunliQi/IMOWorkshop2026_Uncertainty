"""Sensitive and resistant cells with chemotherapy pulses, bevacizumab and PARP inhibitors
(Team Purple's S/R model).

    dS/dt = g S - m S - k c(t) S - k_bev b(t) S - k_parp p(t) S
    dR/dt = g R + m S - eps k c(t) R - eps_bev k_bev b(t) R - eps_parp k_parp p(t) R
    c(t) = 1 for `pulse` after each infusion (every NACT and ACT cycle start), else 0
    b(t) = 1 while on bevacizumab (frontline cycles given with BEV, and BEV maintenance)
    p(t) = 1 while on PARP maintenance; k_parp = k_parp_hrd if HRD else k_parp_hrp
    surgery: S, R -> q S, q R;      CA-125 = a (S + R) + b0

Estimated per patient (per month): g growth, k chemotherapy kill on S while a pulse is on,
rho = R0 / S0 at the first reading. The tool also estimates b0 (background CA-125) and the
levels: a, S0 and R0 only enter through a (S0 + R0), which is the level parameter, and their
ratio rho; the surgery fraction (beta = 1 - q) comes from the post-surgery readings.

Fixed (team table, converted to months with 30 days per month):
    m           S -> R switching          6.7e-6 per day  (Kozlowska: 1e-5 / division x 0.667)
    eps         kill on R / kill on S     0.02            (Kozlowska alpha1)
    k_bev       BEV kill                  0.00235 per day (GOG-0218)
    eps_bev     BEV on R / on S           1.0
    k_parp_hrd  PARP kill, HRD            0.03081 per day (PAOLA-1 HRD+)
    k_parp_hrp  PARP kill, HRP            0.01236 per day (PRIMA HRp)
    eps_parp    PARP on R / on S          0.2
    pulse       3 days per infusion
HRD status unknown: `hrd_unknown` = "hrp" (default), "hrd" or "mean" (average of the two rates).
bev="brake" gives the earlier BEV form instead, a growth brake g (1 - d_bev b(t)) with
d_bev = 0.25 (1 - HR, GOG-218 / ICON7), capped so BEV alone never shrinks the tumour.
Within a stretch where c, b and p are constant the system is linear and lower triangular, so it
is solved exactly (no ODE solver).
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

    def __init__(self, m=6.7e-6, eps=0.02, bev="kill", k_bev=0.00235, d_bev=0.25,
                 eps_bev=1.0, k_parp_hrd=0.03081, k_parp_hrp=0.01236, eps_parp=0.2,
                 hrd_unknown="hrp", pulse_days=3.0):
        if bev not in ("kill", "brake"):
            raise ValueError(f"bev must be 'kill' or 'brake', not {bev!r}")
        if hrd_unknown not in ("hrp", "hrd", "mean"):
            raise ValueError(f"hrd_unknown must be 'hrp', 'hrd' or 'mean', not {hrd_unknown!r}")
        self.m = m * DPM                     # rates per month
        self.eps, self.bev, self.d_bev, self.eps_bev = eps, bev, d_bev, eps_bev
        self.k_bev = k_bev * DPM
        self.k_parp_hrd, self.k_parp_hrp = k_parp_hrd * DPM, k_parp_hrp * DPM
        self.eps_parp, self.hrd_unknown = eps_parp, hrd_unknown
        self.pulse = pulse_days / DPM        # months

    def k_parp(self, schedule):
        """PARP kill rate (per month) for this patient's HR status."""
        hrd = getattr(schedule, "hrd", None)
        if hrd is None:
            return {"hrp": self.k_parp_hrp, "hrd": self.k_parp_hrd,
                    "mean": 0.5 * (self.k_parp_hrd + self.k_parp_hrp)}[self.hrd_unknown]
        return self.k_parp_hrd if hrd else self.k_parp_hrp

    def rates(self, g, k, c, u, p, schedule):
        """Net growth rates (a_S, a_R) per month with chemo c, BEV u and PARP p (0 or 1)."""
        kp = self.k_parp(schedule) * p
        if self.bev == "kill":
            a1 = g - self.m - k * c - self.k_bev * u - kp
            a2 = g - self.eps * k * c - self.eps_bev * self.k_bev * u - self.eps_parp * kp
        else:
            ds, dr = min(self.d_bev * u, 1.0), min(self.eps_bev * self.d_bev * u, 1.0)
            a1 = g * (1 - ds) - self.m - k * c - kp
            a2 = g * (1 - dr) - self.eps * k * c - self.eps_parp * kp
        return a1, a2

    # ---- treatment inputs --------------------------------------------------------------
    def infusions(self, schedule):
        return np.sort(np.concatenate([np.asarray(schedule.nact_starts, float),
                                       np.asarray(schedule.act_starts, float)]))

    def pulse_intervals(self, schedule):
        d = self.infusions(schedule)
        return np.column_stack([d, d + self.pulse]) if d.size else np.zeros((0, 2))

    def drug_free_from(self, schedule):
        """After this time no pulse, no BEV and no PARP is on (inf if a drug is ongoing)."""
        ends = [schedule.t_end]
        for iv in (self.pulse_intervals(schedule), schedule.bev_intervals(),
                   schedule.parp_intervals()):
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
        p = drug_on([t], schedule.parp_intervals())[0]
        a1, a2 = self.rates(theta["g"], theta["k"], c, u, p, schedule)
        return [a1 * S, a2 * R + self.m * S]

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
        parp = schedule.parp_intervals()
        tq = np.unique(t)
        edges = np.concatenate([pulses.ravel(), bev.ravel(), parp.ravel()])
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
            p = drug_on([mid], parp)[0]
            a1, a2 = self.rates(g, k, c, u, p, schedule)
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
                              followup, step=0.02):
        """First time >= end of ACT at which the tumour part a N reaches cutoff (an array of N
        when b0 varies), inf if not within followup months. While a drug is still on after ACT
        (BEV, PARP) the curve can fall and rise, so that stretch is searched on a grid of
        `step` months and the crossing refined in between; once no drug is on, S and R both
        grow at g (switching keeps S + R), so log N is linear and the crossing is exact."""
        s = schedule
        t_end, t_stop = s.t_end, s.t_end + followup
        n = len(np.atleast_1d(ca125_0))
        cut = np.broadcast_to(np.asarray(cutoff, float), (n,))
        log_cut = np.log(np.where(cut > 0, cut, 1.0))
        out = np.full(n, np.inf)
        out[cut <= 0] = t_end
        t0 = self.drug_free_from(s)
        todo = cut > 0
        if t0 > t_end + 1e-12:                   # drugs on after ACT: grid search
            hi = min(t0, t_stop)
            grid = np.unique(np.append(np.arange(t_end, hi, step), hi))
            sub = lambda d: {k: np.asarray(v)[todo] for k, v in d.items()}   # noqa: E731
            ln = np.log(np.maximum(self.simulate_patient_batch(
                grid, s, sub(theta_pre), sub(theta_post), np.asarray(ca125_0)[todo],
                np.asarray(q)[todo]), 1e-300)) - log_cut[todo][:, None]
            above = ln >= 0
            hit = above.any(1)
            j = np.argmax(above, 1)
            jj = np.maximum(j - 1, 0)
            r = np.arange(len(j))
            a, b = ln[r, jj], ln[r, j]
            frac = np.where((j > 0) & (b > a), -a / np.where(b > a, b - a, 1.0), 0.0)
            tc = grid[jj] + frac * (grid[j] - grid[jj])
            idx = np.flatnonzero(todo)
            out[idx[hit]] = tc[hit]
            todo = todo.copy()
            todo[idx[hit]] = False
        if t0 >= t_stop or not todo.any():
            return np.where(out <= t_stop, out, np.inf)
        start = max(t0, t_end)
        log_t0 = np.log(self.simulate_patient_batch([start], s, theta_pre, theta_post,
                                                    ca125_0, q)[:, 0])
        g = np.broadcast_to(np.asarray(theta_post["g"], float), log_t0.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = log_cut - log_t0
            dt = np.where(gap <= 0, 0.0, np.where(g > 0, gap / g, np.inf))
        out = np.where(todo, start + dt, out)
        return np.where(out <= t_stop, out, np.inf)
