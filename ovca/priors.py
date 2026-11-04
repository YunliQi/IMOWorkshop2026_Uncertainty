"""Prior distributions for one patient's parameters.

Each sampled parameter lives on an unconstrained scale u (log for positive parameters, logit
for bounded ones, identity otherwise); backends sample u and the prior density is on u
(Jacobian included). A prior written as a plain number means "fixed at this value".

Supported distributions (natural scale):
    {"dist": "lognormal", "median": m, "sd_log": s}      positive;  u = log x ~ N(log m, s^2)
    {"dist": "normal", "mean": m, "sd": s}               real line; u = x
    {"dist": "halfnormal", "sd": s}                      positive;  u = log x
    {"dist": "uniform", "low": a, "high": b}             (a, b);    u = logit((x - a) / (b - a))
    {"dist": "fixed", "value": v}  (or just the number v)
`median` of a level prior may be the string "first_reading" (resolved per patient).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

DIST_ARGS = {
    "lognormal": ("median", "sd_log"),
    "normal": ("mean", "sd"),
    "halfnormal": ("sd",),
    "uniform": ("low", "high"),
    "fixed": ("value",),
}
POSITIVE = {"lognormal", "halfnormal"}
GAUSSIAN_U = {"lognormal", "normal"}      # u is exactly normal: can take correlations


def normalise(spec):
    """A number -> fixed; a dict is returned as a (shallow) copy."""
    if isinstance(spec, (int, float, np.floating, np.integer)) and not isinstance(spec, bool):
        return {"dist": "fixed", "value": float(spec)}
    return dict(spec)


@dataclass
class Dist:
    """One resolved (numbers only) prior."""
    name: str
    kind: str
    args: dict

    @property
    def fixed(self):
        return self.kind == "fixed"

    def support(self):
        if self.kind in POSITIVE:
            return 0.0, np.inf
        if self.kind == "uniform":
            return self.args["low"], self.args["high"]
        if self.kind == "fixed":
            return self.args["value"], self.args["value"]
        return -np.inf, np.inf

    # u <-> natural
    def to_natural(self, u):
        a = self.args
        if self.kind in POSITIVE:
            return np.exp(u)
        if self.kind == "uniform":
            return a["low"] + (a["high"] - a["low"]) * expit(u)
        return u

    def to_u(self, x):
        a = self.args
        x = np.asarray(x, dtype=float)
        if self.kind in POSITIVE:
            return np.log(x)
        if self.kind == "uniform":
            return logit((x - a["low"]) / (a["high"] - a["low"]))
        return x

    def u_mean_sd(self):
        """Mean and sd of u for the Gaussian-u kinds."""
        a = self.args
        if self.kind == "lognormal":
            return np.log(a["median"]), a["sd_log"]
        if self.kind == "normal":
            return a["mean"], a["sd"]
        raise ValueError(self.kind)

    def sample_u(self, rng, n):
        a = self.args
        if self.kind in GAUSSIAN_U:
            m, s = self.u_mean_sd()
            return rng.normal(m, s, n)
        if self.kind == "halfnormal":
            return np.log(np.abs(rng.normal(0, a["sd"], n)) + 1e-300)
        if self.kind == "uniform":
            return logit(rng.uniform(1e-12, 1 - 1e-12, n))
        raise ValueError(self.kind)

    def logpdf_u(self, u):
        a = self.args
        if self.kind in GAUSSIAN_U:
            m, s = self.u_mean_sd()
            return norm.logpdf(u, m, s)
        if self.kind == "halfnormal":       # x = e^u, p(u) = p_x(e^u) e^u
            x = np.exp(u)
            return np.log(2.0) + norm.logpdf(x, 0, a["sd"]) + u
        if self.kind == "uniform":          # p(u) = sigmoid(u)(1 - sigmoid(u))
            return -np.logaddexp(0, u) - np.logaddexp(0, -u)
        raise ValueError(self.kind)

    def describe(self):
        a = self.args
        if self.kind == "fixed":
            return f"fixed = {a['value']:g}"
        if self.kind == "lognormal":
            lo, hi = a["median"] * np.exp(-1.2816 * a["sd_log"]), a["median"] * np.exp(1.2816 * a["sd_log"])
            return f"lognormal(median {a['median']:.3g}, sd_log {a['sd_log']:g}); 80%: {lo:.3g} – {hi:.3g}"
        if self.kind == "normal":
            return (f"normal(mean {a['mean']:g}, sd {a['sd']:g}); 80%: "
                    f"{a['mean'] - 1.2816 * a['sd']:.3g} – {a['mean'] + 1.2816 * a['sd']:.3g}")
        if self.kind == "halfnormal":
            return f"halfnormal(sd {a['sd']:g}); 80% below {1.645 * a['sd']:.3g}"
        return f"uniform({a['low']:g}, {a['high']:g})"


@dataclass
class JointPrior:
    """Prior over the sampled parameters (in `names` order) on the u scale.

    Parameters with lognormal / normal priors may be correlated (Gaussian on u); all others are
    independent. `fixed` holds the fixed parameters (natural values)."""
    names: list
    dists: dict
    fixed: dict
    corr: dict = field(default_factory=dict)     # (name_a, name_b) -> correlation

    def __post_init__(self):
        self.k = len(self.names)
        self._gauss = [i for i, n in enumerate(self.names) if self.dists[n].kind in GAUSSIAN_U]
        self._other = [i for i in range(self.k) if i not in self._gauss]
        if self._gauss:
            m = np.array([self.dists[self.names[i]].u_mean_sd()[0] for i in self._gauss])
            s = np.array([self.dists[self.names[i]].u_mean_sd()[1] for i in self._gauss])
            R = np.eye(len(self._gauss))
            pos = {self.names[i]: j for j, i in enumerate(self._gauss)}
            for (a, b), r in self.corr.items():
                R[pos[a], pos[b]] = R[pos[b], pos[a]] = r
            self._mean, self._cov = m, np.outer(s, s) * R
            self._chol = np.linalg.cholesky(self._cov)
            self._icov = np.linalg.inv(self._cov)
            self._logdet = 2 * np.log(np.diag(self._chol)).sum()

    def sample(self, rng, n):
        u = np.empty((n, self.k))
        if self._gauss:
            u[:, self._gauss] = self._mean + rng.standard_normal((n, len(self._gauss))) @ self._chol.T
        for i in self._other:
            u[:, i] = self.dists[self.names[i]].sample_u(rng, n)
        return u

    def logpdf(self, u):
        u = np.atleast_2d(u)
        lp = np.zeros(len(u))
        if self._gauss:
            d = u[:, self._gauss] - self._mean
            lp += -0.5 * (np.einsum("ij,jk,ik->i", d, self._icov, d) + self._logdet
                          + len(self._gauss) * np.log(2 * np.pi))
        for i in self._other:
            lp += self.dists[self.names[i]].logpdf_u(u[:, i])
        return lp

    def natural(self, u):
        """u (N, k) -> dict name -> natural values (N,), fixed ones broadcast."""
        u = np.atleast_2d(u)
        out = {n: self.dists[n].to_natural(u[:, i]) for i, n in enumerate(self.names)}
        out.update({n: np.full(len(u), v) for n, v in self.fixed.items()})
        return out

    def table(self):
        import pandas as pd
        rows = [dict(parameter=n, prior=self.dists[n].describe(), sampled=True)
                for n in self.names]
        rows += [dict(parameter=n, prior=f"fixed = {v:g}", sampled=False)
                 for n, v in self.fixed.items()]
        return pd.DataFrame(rows)
