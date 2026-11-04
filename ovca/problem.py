"""The probabilistic model of one patient at one time point, shared by every backend.

Parameters (sampled on the unconstrained scale u, see priors.py):
    {p}_pre      model parameters before surgery
    shift_{p}    surgery effect: log {p}_post = log {p}_pre + shift_{p}
    level_pre    tumour burden at the anchor of the pre-surgery segment (its first reading)
    level_post   tumour burden at the anchor of the post-surgery segment (its first reading)
    sigma        observation error
There is no surgery-fraction parameter: the post-surgery level comes from the post-surgery
readings. Before the first post-surgery reading (post_mode "pre_only") only the pre-surgery
segment is fitted; shifts and level_post are not sampled and every post-surgery quantity (curve
after surgery, recurrence) is NaN.
Observation model: log y_i = log T(t_i) + sigma * e_i, e_i ~ N(0, 1); readings below
data.censor_below (if set) only say "below"; with data.decay_tau, reading i enters the
log-likelihood with weight exp(-(t_now - t_i) / tau).

A backend needs only: names, prior (sample / logpdf on u), loglik(u), and for forecasting
simulate(u, t) and recurrence(u, followup).
"""
import numpy as np
from scipy.stats import norm

from .models import BaseModel, get_model
from .priors import Dist, JointPrior, normalise
from .profile import check_prior
from .record import PatientRecord

LOG_2PI = np.log(2 * np.pi)


class InferenceProblem:
    def __init__(self, record: PatientRecord, t_now, profile, model: BaseModel = None):
        self.profile = profile
        self.model = model or get_model(profile["model"])
        self.record = record
        self.schedule = s = record.schedule
        t_now = float(record.t.max() if t_now is None else t_now)
        rec = record.upto(t_now)
        if len(rec.t) == 0:
            raise ValueError(f"no readings at or before t_now = {t_now}")
        self.t_now, self.t, self.y = t_now, rec.t, rec.y
        d = profile["data"]
        self.cutoff = d["cutoff"]
        self.censor = d["censor_below"]
        tau = d["decay_tau"]
        self.w = np.ones_like(self.t) if not tau else np.exp(-(t_now - self.t) / tau)
        self.log_y = np.log(self.y)
        self.cens = (self.y < self.censor) if self.censor else np.zeros(len(self.y), bool)

        # anchors of the two segments
        pre, post = self.t < s.t_surgery, self.t >= s.t_surgery
        first_reading = profile["initial_level"]["anchor"] == "first_reading"
        self.a_pre = float(self.t[pre][0]) if (first_reading and pre.any()) else 0.0
        self.post_mode = "level" if post.any() else "pre_only"
        self.a_post = (float(self.t[post][0]) if (first_reading and post.any())
                       else s.t_surgery)
        # where level_post is reported (same time for every t_now, for trajectories)
        post_all = record.t >= s.t_surgery
        self.a_post_report = (float(record.t[post_all][0]) if (first_reading and post_all.any())
                              else self.a_post)
        y_pre = self.y[pre][0] if pre.any() else self.y[0]
        y_post = self.y[post][0] if post.any() else None

        # prior over the parameters this problem uses
        sp = profile["surgery"]["shift_params"]
        if self.post_mode == "level":
            use = ([f"{p}_pre" for p in self.model.param_names] + [f"shift_{p}" for p in sp]
                   + ["level_pre", "level_post", "sigma"])
        else:
            use = [f"{p}_pre" for p in self.model.param_names] + ["level_pre", "sigma"]
        dists, fixed = {}, {}
        for n in use:
            spec = normalise(profile["priors"][n])
            if spec.get("median") == "first_reading":
                spec["median"] = float(y_pre if n == "level_pre" else y_post)
            dd = check_prior(n, spec, [], False, True)
            if dd.fixed:
                fixed[n] = dd.args["value"]
            else:
                dists[n] = dd
        names = [n for n in use if n in dists]
        corr = {}
        for key, r in (profile.get("correlations") or {}).items():
            a, b = tuple(key) if isinstance(key, (tuple, list)) else tuple(key.split(":"))
            if a in dists and b in dists:
                corr[(a, b)] = float(r)
        self.prior = JointPrior(names, dists, fixed, corr)
        self.names = names
        self.sp = sp

    # ---- parameters --------------------------------------------------------------------
    def components(self, u):
        """u (N, k) -> theta_pre, theta_post, ca125_0, q, sigma (arrays of N)."""
        nat = self.prior.natural(u)
        m, s = self.model, self.schedule
        th_pre = {p: nat[f"{p}_pre"] for p in m.param_names}
        g_pre = m.log_growth_batch([self.a_pre, s.t_surgery], 0.0, th_pre, s.intervals("pre"))
        log_c0 = np.log(nat["level_pre"]) - g_pre[:, 0]
        if self.post_mode == "pre_only":          # nothing known after surgery yet
            nan = np.full(len(log_c0), np.nan)
            return th_pre, {p: nan for p in m.param_names}, np.exp(log_c0), nan, nat["sigma"]
        th_post = {p: th_pre[p] * np.exp(nat[f"shift_{p}"]) if p in self.sp else th_pre[p]
                   for p in m.param_names}
        log_ts_minus = log_c0 + g_pre[:, 1]
        g_post = m.log_growth_batch([self.a_post], s.t_surgery, th_post, s.intervals("post"))
        log_ts_plus = np.log(nat["level_post"]) - g_post[:, 0]
        q = np.exp(np.clip(log_ts_plus - log_ts_minus, -700, 700))   # derived, not a parameter
        return th_pre, th_post, np.exp(log_c0), q, nat["sigma"]

    def natural(self, u):
        """Every parameter in natural units, as a dict of arrays (also the derived ones)."""
        th_pre, th_post, c0, q, sigma = self.components(u)
        nat = self.prior.natural(u)
        out = {f"{p}_pre": v for p, v in th_pre.items()}
        out.update({f"{p}_post": v for p, v in th_post.items()})
        out.update({f"shift_{p}": np.log(th_post[p] / th_pre[p]) for p in self.model.param_names})
        lvl = self.simulate(u, [self.a_pre, self.a_post_report])
        out.update(level_pre=lvl[:, 0], level_post=lvl[:, 1], ca125_0=c0, surgery_jump=q,
                   sigma=np.broadcast_to(sigma, c0.shape).copy())
        return out

    # ---- simulation and likelihood -----------------------------------------------------
    def simulate(self, u, t):
        """Noise-free tumour burden (N, len(t))."""
        th_pre, th_post, c0, q, _ = self.components(u)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.post_mode == "pre_only":
            s = self.schedule
            out = np.full((len(c0), t.size), np.nan)
            pre = t < s.t_surgery
            if pre.any():
                g = self.model.log_growth_batch(t[pre], 0.0, th_pre, s.intervals("pre"))
                out[:, pre] = np.exp(np.log(c0)[:, None] + g)
            return out
        return self.model.simulate_patient_batch(t, self.schedule, th_pre, th_post, c0, q)

    def recurrence(self, u, followup):
        """Absolute month when the model curve reaches the cutoff after ACT (inf = never)."""
        th_pre, th_post, c0, q, _ = self.components(u)
        if self.post_mode == "pre_only":
            return np.full(len(c0), np.nan)
        return self.model.recurrence_time_batch(self.schedule, th_pre, th_post, c0, q,
                                                self.cutoff, followup)

    def loglik(self, u):
        """Weighted log-likelihood of the readings, (N,)."""
        _, _, _, _, sigma = self.components(u)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (len(np.atleast_2d(u)),))
        log_T = np.log(np.maximum(self.simulate(u, self.t), 1e-300))
        sig = sigma[:, None]
        z = (self.log_y[None, :] - log_T) / sig
        ll = -0.5 * z ** 2 - np.log(sig) - 0.5 * LOG_2PI
        if self.cens.any():
            zc = (np.log(self.censor) - log_T[:, self.cens]) / sig
            ll[:, self.cens] = norm.logcdf(zc)
        return (ll * self.w[None, :]).sum(axis=1)

    def distance(self, u, rng=None):
        """Weighted RMS of log differences (threshold acceptance). With rng, the simulated
        readings get observation error sigma (needed: as epsilon -> 0 this gives the exact
        posterior; noise-free simulations would collapse onto the best-fitting curve)."""
        T = self.simulate(u, self.t)
        if rng is not None:
            sigma = np.broadcast_to(np.asarray(self.components(u)[4], dtype=float), (len(T),))
            T = T * np.exp(sigma[:, None] * rng.standard_normal(T.shape))
        lo = self.censor if self.censor else 1e-6
        r = np.log(np.maximum(T, lo)) - np.log(np.maximum(self.y, lo))[None, :]
        return np.sqrt((r ** 2 * self.w[None, :]).sum(axis=1) / self.w.sum())

    def describe(self):
        return (f"{self.record.patient_id}: t_now = {self.t_now:.2f}, {len(self.t)} readings, "
                f"anchors pre {self.a_pre:.2f} / post {self.a_post:.2f} ({self.post_mode}), "
                f"sampled {self.names}, fixed {list(self.prior.fixed)}")
