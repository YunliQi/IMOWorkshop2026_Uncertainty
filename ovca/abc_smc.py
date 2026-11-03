"""ABC-SMC for one patient: posterior of the model parameters given the CA-125 readings seen so
far, then a forecast of CA-125 and of recurrence, and a suggested time for the next visit.

Parameterisation (one model for both segments):
    log theta_pre                                  model parameters before surgery
    shift_p:   log p_post = log p_pre + shift_p    surgery moves the parameters (for p in
                                                   shift_params; others stay the same)
    log_q:     T(after surgery) = q * T(before)    surgery removes tumour
    log_ca125_0                                    CA-125 at t = 0

Simulator: model.simulate_patient_batch + multiplicative assay noise (lognormal, sd noise_sd).
Distance between simulated and observed readings, on the log scale:
    r_i = log max(sim_i, c) - log max(obs_i, c)    c = censor_below (35): a reading below 35
                                                   only says "below 35"
    d   = sqrt( sum_i w_i r_i^2 / sum_i w_i )      w_i = exp(-(t_now - t_i) / decay), or 1
ABC-SMC (Beaumont et al. 2009, adaptive epsilon): particles from the prior, then generations
with epsilon = ABC_ALPHA quantile of the previous distances, Gaussian perturbation with twice
the weighted particle covariance, importance weights prior / proposal.

Typical use:
    post = fit(model, t, y, schedule, t_now=..., prior=prior_for(spec, y[0]))
    fc = post.forecast()
    fc.recurrence_interval(0.8); fc.next_visit(rule="risk", p=0.1)
    post2 = post.update(t_new, y_new)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from . import config as C
from .models import BaseModel
from .patient import Schedule


# ---------------------------------------------------------------- parameter space and prior
def parameter_names(model: BaseModel, shift_params=C.ABC_SHIFT_PARAMS):
    return ([f"log_{p}_pre" for p in model.param_names]
            + [f"shift_{p}" for p in model.param_names if p in shift_params]
            + ["log_q", "log_ca125_0"])


@dataclass
class Prior:
    """Multivariate normal on the sampling scale (see parameter_names)."""
    names: list
    mean: np.ndarray
    cov: np.ndarray

    def sample(self, rng, n):
        return rng.multivariate_normal(self.mean, self.cov, n)

    def logpdf(self, x):
        return multivariate_normal(self.mean, self.cov, allow_singular=True).logpdf(x)

    def table(self):
        sd = np.sqrt(np.diag(self.cov))
        return pd.DataFrame(dict(parameter=self.names, mean=self.mean, sd=sd,
                                 q10=self.mean - 1.2816 * sd, q90=self.mean + 1.2816 * sd))


@dataclass
class PriorSpec:
    """Prior for everything except log_ca125_0, which depends on the patient's first reading."""
    names: list
    mean: np.ndarray
    cov: np.ndarray


def default_prior_spec(model: BaseModel, shift_params=C.ABC_SHIFT_PARAMS, table=None):
    table = table or C.ABC_PRIOR
    names = parameter_names(model, shift_params)[:-1]
    mean = np.array([table[n][0] for n in names], dtype=float)
    sd = np.array([table[n][1] for n in names], dtype=float)
    return PriorSpec(names, mean, np.diag(sd ** 2))


def empirical_prior_spec(model: BaseModel, fits, shift_params=C.ABC_SHIFT_PARAMS,
                         inflate=1.25, min_patients=5):
    """Prior estimated from RMSE calibration of training patients (calibration.calibrate_cohort):
    log pre-surgery parameters and log(post / pre) shifts, from fits that are not stuck at a
    bound; sd widened by `inflate`. log_q keeps the default (the fits do not estimate it)."""
    spec = default_prior_spec(model, shift_params)
    ok_pre = fits.fitted_pre.astype(bool) & ~fits.at_bound_pre.astype(bool)
    ok_both = ok_pre & fits.fitted_post.astype(bool) & ~fits.at_bound_post.astype(bool)
    cols = {}
    for p in model.param_names:
        cols[f"log_{p}_pre"] = np.log(fits.loc[ok_pre, f"{p}_pre"])
        if p in shift_params:
            cols[f"shift_{p}"] = np.log(fits.loc[ok_both, f"{p}_post"] / fits.loc[ok_both, f"{p}_pre"])
    est = pd.DataFrame(cols)
    mean, cov = spec.mean.copy(), spec.cov.copy()
    idx = {n: i for i, n in enumerate(spec.names)}
    groups = [[f"log_{p}_pre" for p in model.param_names],
              [f"shift_{p}" for p in model.param_names if p in shift_params]]
    for g in groups:
        sub = est[g].dropna()
        if len(sub) < min_patients:
            continue
        ii = [idx[n] for n in g]
        mean[ii] = sub.mean().values
        c = np.atleast_2d(sub.cov().values) * inflate ** 2
        cov[np.ix_(ii, ii)] = c
    return PriorSpec(spec.names, mean, cov)


def prior_for(spec: PriorSpec, y0, noise_sd=C.NOISE_SD_LOG):
    """Full prior for one patient: spec + log_ca125_0 ~ Normal(log first reading, noise_sd^2)."""
    k = len(spec.names)
    cov = np.zeros((k + 1, k + 1))
    cov[:k, :k] = spec.cov
    cov[k, k] = noise_sd ** 2
    return Prior(spec.names + ["log_ca125_0"], np.append(spec.mean, np.log(y0)), cov)


def unpack(model: BaseModel, names, x):
    """Sampling-scale particles (N, k) -> theta_pre, theta_post (dicts of arrays), ca125_0, q."""
    col = {n: x[:, i] for i, n in enumerate(names)}
    theta_pre = {p: np.exp(col[f"log_{p}_pre"]) for p in model.param_names}
    theta_post = {p: np.exp(col[f"log_{p}_pre"] + col.get(f"shift_{p}", 0.0))
                  for p in model.param_names}
    q = np.minimum(np.exp(col["log_q"]), 1.0)
    return theta_pre, theta_post, np.exp(col["log_ca125_0"]), q


# ---------------------------------------------------------------- distance
def _clog(y, cutoff):
    y = np.asarray(y, dtype=float)
    return np.log(np.maximum(y, cutoff if cutoff else 1e-6))


def decay_weights(t_obs, t_now, decay):
    t_obs = np.asarray(t_obs, dtype=float)
    return np.ones_like(t_obs) if not decay else np.exp(-(t_now - t_obs) / decay)


def distance(sim, obs, weights, cutoff=C.CA125_CUTOFF):
    """Weighted RMS of log differences; readings below cutoff are censored at cutoff."""
    r = _clog(sim, cutoff) - _clog(obs, cutoff)[None, :]
    return np.sqrt((r ** 2 * weights).sum(axis=1) / weights.sum())


# ---------------------------------------------------------------- ABC-SMC
@dataclass
class Generation:
    epsilon: float
    n_sims: int
    accept_rate: float


@dataclass
class ABCPosterior:
    model: BaseModel
    prior: Prior
    particles: np.ndarray          # (N, k) on the sampling scale
    weights: np.ndarray            # (N,), sum to 1
    distances: np.ndarray
    schedule: Schedule
    t_obs: np.ndarray              # months
    y_obs: np.ndarray
    t_now: float
    settings: dict
    history: list = field(default_factory=list)

    @property
    def names(self):
        return self.prior.names

    def resample(self, n, rng):
        return self.particles[rng.choice(len(self.weights), n, p=self.weights)]

    def natural(self, x=None):
        """Particles in natural units: model parameters before/after surgery, q, ca125_0."""
        x = self.particles if x is None else x
        th_pre, th_post, c0, q = unpack(self.model, self.names, x)
        out = {f"{p}_pre": v for p, v in th_pre.items()}
        out.update({f"{p}_post": v for p, v in th_post.items()})
        out.update(q_surgery=q, ca125_0=c0)
        out.update({n: x[:, i] for i, n in enumerate(self.names) if n.startswith("shift_")})
        return pd.DataFrame(out)

    def summary(self, levels=(0.1, 0.5, 0.9)):
        nat = self.natural()
        rows = []
        for c in nat.columns:
            rows.append(dict(parameter=c, **{f"q{int(l * 100)}": _wquantile(nat[c].values,
                                                                            self.weights, l)
                                             for l in levels}))
        return pd.DataFrame(rows)

    def forecast(self, horizon=C.FOLLOWUP_MONTHS, step=0.05, n=2000, seed=0,
                 condition_no_recurrence=True):
        """Forecast from t_now: CA-125 bands and recurrence times for n posterior draws."""
        rng = np.random.default_rng(seed)
        x = self.resample(n, rng)
        th_pre, th_post, c0, q = unpack(self.model, self.names, x)
        s = self.schedule
        t_stop = max(self.t_now, s.t_end) + horizon
        grid = np.arange(0.0, t_stop + step / 2, step)
        tumour = self.model.simulate_patient_batch(grid, s, th_pre, th_post, c0, q)
        tau = self.model.recurrence_time_batch(s, th_pre, th_post, c0, q, C.CA125_CUTOFF,
                                               t_stop - s.t_end)
        if condition_no_recurrence:
            # still recurrence-free at t_now: no surveillance reading so far was >= cutoff
            surv = self.t_obs > s.t_end + 1e-9
            if surv.any() and not (self.y_obs[surv] >= C.CA125_CUTOFF).any():
                keep = tau > self.t_obs[surv].max()
                if keep.sum() >= 50:
                    x, tumour, tau = x[keep], tumour[keep], tau[keep]
        noisy = tumour * np.exp(rng.normal(0, self.settings["noise_sd"], tumour.shape))
        return Forecast(grid, tumour, noisy, tau, self.t_now, s, x)

    def update(self, t_new, y_new, **kwargs):
        """Add readings and refit from the same prior (all data used again)."""
        t = np.append(self.t_obs, np.atleast_1d(t_new))
        y = np.append(self.y_obs, np.atleast_1d(y_new))
        order = np.argsort(t)
        settings = {**self.settings, **kwargs}
        t_now = max(float(np.max(t_new)), self.t_now)
        return fit(self.model, t[order], y[order], self.schedule, t_now=t_now,
                   prior=self.prior, **settings)


def _wquantile(v, w, q):
    v = np.asarray(v, dtype=float)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw / cw[-1], q)])


def _weighted_cov(x, w):
    m = (w[:, None] * x).sum(0)
    d = x - m
    return (w[:, None] * d).T @ d


def fit(model: BaseModel, t_obs, y_obs, schedule: Schedule, t_now=None, prior: Prior = None,
        decay=None, noise_sd=C.NOISE_SD_LOG, censor_below=C.CA125_CUTOFF,
        n_particles=C.ABC_N_PARTICLES, alpha=C.ABC_ALPHA,
        max_generations=C.ABC_MAX_GENERATIONS, min_accept_rate=C.ABC_MIN_ACCEPT_RATE,
        max_sims_per_gen=C.ABC_MAX_SIMS_PER_GEN, batch=C.ABC_BATCH, seed=0, verbose=False):
    """ABC-SMC posterior for one patient, using readings with t_obs <= t_now (months)."""
    t_obs, y_obs = np.asarray(t_obs, dtype=float), np.asarray(y_obs, dtype=float)
    t_now = float(t_obs.max()) if t_now is None else float(t_now)
    m = t_obs <= t_now + 1e-9
    t_obs, y_obs = t_obs[m], y_obs[m]
    if prior is None:
        prior = prior_for(default_prior_spec(model), y_obs[0], noise_sd)
    settings = dict(decay=decay, noise_sd=noise_sd, censor_below=censor_below,
                    n_particles=n_particles, alpha=alpha, max_generations=max_generations,
                    min_accept_rate=min_accept_rate, max_sims_per_gen=max_sims_per_gen,
                    batch=batch, seed=seed)
    rng = np.random.default_rng(seed)
    w_obs = decay_weights(t_obs, t_now, decay)

    def dist(x):
        th_pre, th_post, c0, q = unpack(model, prior.names, x)
        sim = model.simulate_patient_batch(t_obs, schedule, th_pre, th_post, c0, q)
        sim = sim * np.exp(rng.normal(0, noise_sd, sim.shape))
        return distance(sim, y_obs, w_obs, censor_below)

    n = n_particles
    x = prior.sample(rng, n)
    d = dist(x)
    w = np.full(n, 1.0 / n)
    history = [Generation(np.inf, n, 1.0)]
    for g in range(1, max_generations + 1):
        eps = float(np.quantile(d, alpha))
        cov = 2.0 * _weighted_cov(x, w) + 1e-10 * np.eye(x.shape[1])
        chol = np.linalg.cholesky(cov)
        acc_x, acc_d, n_sims = [], [], 0
        while sum(len(a) for a in acc_x) < n and n_sims < max_sims_per_gen:
            parents = x[rng.choice(n, batch, p=w)]
            prop = parents + rng.standard_normal(parents.shape) @ chol.T
            dp = dist(prop)
            keep = dp <= eps
            acc_x.append(prop[keep])
            acc_d.append(dp[keep])
            n_sims += batch
        new_x = np.concatenate(acc_x)[:n]
        new_d = np.concatenate(acc_d)[:n]
        rate = len(np.concatenate(acc_d)) / n_sims
        if len(new_x) < max(50, n // 5):            # budget ran out: keep the last generation
            history.append(Generation(eps, n_sims, rate))
            break
        # importance weights: prior / mixture proposal
        inv = np.linalg.inv(cov)
        diff = new_x[:, None, :] - x[None, :, :]
        log_k = -0.5 * np.einsum("ijk,kl,ijl->ij", diff, inv, diff)
        log_prop = logsumexp(log_k + np.log(w)[None, :], axis=1)
        log_w = prior.logpdf(new_x) - log_prop
        new_w = np.exp(log_w - log_w.max())
        x, d, w = new_x, new_d, new_w / new_w.sum()
        if len(x) < n:                               # partial generation: pad by resampling
            pad = rng.choice(len(x), n - len(x), p=w)
            x, d = np.concatenate([x, x[pad]]), np.concatenate([d, d[pad]])
            w = np.full(n, 1.0 / n)
        history.append(Generation(eps, n_sims, rate))
        if verbose:
            print(f"gen {g}: eps={eps:.3f} accept={rate:.3f} sims={n_sims}")
        if rate < min_accept_rate:
            break
    return ABCPosterior(model, prior, x, w, d, schedule, t_obs, y_obs, t_now, settings, history)


# ---------------------------------------------------------------- forecast
@dataclass
class Forecast:
    t: np.ndarray                 # months
    tumour: np.ndarray            # (n, len(t)) noise-free model value
    ca125: np.ndarray             # (n, len(t)) with assay noise
    recurrence: np.ndarray        # (n,) absolute month when the curve reaches 35 (inf = never)
    t_now: float
    schedule: Schedule
    particles: np.ndarray

    def band(self, levels=(0.1, 0.5, 0.9), noisy=False):
        """Quantiles over draws at each time: tumour (parameter uncertainty) or noisy CA-125
        (parameter + assay noise)."""
        v = self.ca125 if noisy else self.tumour
        return pd.DataFrame({"t": self.t, **{f"q{int(l * 100)}": np.quantile(v, l, axis=0)
                                             for l in levels}})

    def recurrence_cdf(self, t=None):
        """P(CA-125 has reached 35 by t | data), for t in months (default: forecast grid)."""
        t = self.t if t is None else np.atleast_1d(t)
        return np.mean(self.recurrence[:, None] <= np.asarray(t)[None, :], axis=0)

    def recurrence_interval(self, level=0.8, relative=True, cap=None):
        """(low, median, high) recurrence time; relative to the end of ACT by default.
        inf means 'not within the forecast horizon'; cap replaces inf (and larger values)
        with a number, e.g. FOLLOWUP_MONTHS + 1, for tables and widths."""
        r = self.recurrence - (self.schedule.t_end if relative else 0.0)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        out = [float(np.quantile(r, q, method="inverted_cdf")) for q in (lo, 0.5, hi)]
        return tuple(min(v, cap) for v in out) if cap is not None else tuple(out)

    def p_recur_within(self, months_after_end):
        return float(np.mean(self.recurrence - self.schedule.t_end <= months_after_end))

    def next_visit(self, rule="risk", p=0.10, width=1.0, min_gap=1.0, max_gap=6.0):
        """Suggested month of the next CA-125 visit after t_now.
        rule="risk":  first time the chance that CA-125 has crossed 35 since t_now reaches p;
        rule="width": first time the 80% band of log CA-125 is wider than `width`.
        Kept within [t_now + min_gap, t_now + max_gap]."""
        lo, hi = self.t_now + min_gap, self.t_now + max_gap
        cand = self.t[(self.t >= lo) & (self.t <= hi)]
        if rule == "risk":
            base = np.mean(self.recurrence <= self.t_now)
            score = self.recurrence_cdf(cand) - base
            hit = score >= p
        elif rule == "width":
            lq = np.log(self.ca125)
            i = np.searchsorted(self.t, cand)
            wdt = np.quantile(lq[:, i], 0.9, axis=0) - np.quantile(lq[:, i], 0.1, axis=0)
            hit = wdt >= width
        else:
            raise ValueError("rule must be 'risk' or 'width'")
        return float(cand[np.argmax(hit)]) if hit.any() else float(hi)

    def surprise(self, t_new, y_new, cutoff=C.CA125_CUTOFF):
        """Probability of an outcome at least as unusual as the new reading (small = surprising).
        Reading below cutoff: P(forecast below cutoff). Reading at or above cutoff:
        P(forecast >= cutoff) x two-sided tail of y_new among those forecasts."""
        i = min(np.searchsorted(self.t, t_new), len(self.t) - 1)
        v = self.ca125[:, i]
        if cutoff and y_new < cutoff:
            return float(np.mean(v < cutoff))
        above = v[v >= cutoff] if cutoff else v
        if above.size == 0:
            return 0.0
        f = np.mean(above <= y_new)
        return float(above.size / v.size * 2 * min(f, 1 - f))

    def pit(self, t_new, y_new, cutoff=C.CA125_CUTOFF):
        """Where a new reading falls in the forecast: P(predicted CA-125 <= y_new). Near 0 or 1
        = surprising. A reading below cutoff is censored: returns P(predicted < cutoff)."""
        i = np.searchsorted(self.t, t_new)
        i = min(i, len(self.t) - 1)
        v = self.ca125[:, i]
        if cutoff and y_new < cutoff:
            return float(np.mean(v < cutoff))
        return float(np.mean(v <= y_new))


# ---------------------------------------------------------------- data helpers
def patient_inputs(data, patient):
    """From data/synthetic_data.csv rows: clinic readings (months, CA-125), schedule, daily rows."""
    d = data[data.patient == patient].sort_values("time")
    v = d[d.visit == 1]
    sched = Schedule.from_record(d.iloc[0])
    return v.time.values / C.DAYS_PER_MONTH, v.CA125.values, sched, d
