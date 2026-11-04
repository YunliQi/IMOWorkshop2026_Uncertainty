"""The contract between a backend and the rest of the tool.

A backend gets an InferenceProblem (data, schedule, prior, likelihood, simulator) and the
backend's section of the profile, and returns a Posterior: draws of the sampled parameters on
the unconstrained scale u with weights. Forecasts, next-visit rules, updates and plots only use
the Posterior, so they work the same for every backend.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..problem import InferenceProblem


@dataclass
class Posterior:
    problem: InferenceProblem
    u: np.ndarray                 # (N, k) draws, columns = problem.names, unconstrained scale
    weights: np.ndarray           # (N,), sum to 1 (equal for MCMC)
    backend: str
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.u = np.atleast_2d(np.asarray(self.u, dtype=float))
        w = np.asarray(self.weights, dtype=float)
        if self.u.shape != (len(w), len(self.problem.names)):
            raise ValueError(f"u must be (N, {len(self.problem.names)}) matching weights")
        if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("weights must be finite, >= 0 and not all 0")
        self.weights = w / w.sum()

    @property
    def names(self):
        return self.problem.names

    def resample(self, n, rng):
        return self.u[rng.choice(len(self.weights), n, p=self.weights)]

    def natural(self):
        """Every parameter (sampled, fixed and derived) in natural units, one row per draw."""
        return pd.DataFrame(self.problem.natural(self.u))

    def summary(self, levels=(0.1, 0.5, 0.9)):
        nat = self.natural()
        rows = [dict(parameter=c, **{f"q{int(round(l * 100))}": wquantile(nat[c].values,
                                                                        self.weights, l)
                                     for l in levels}) for c in nat.columns]
        return pd.DataFrame(rows)

    def ess(self):
        return float(1.0 / np.sum(self.weights ** 2))


def wquantile(v, w, q):
    v, w = np.asarray(v, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(v) | np.isinf(v)
    if not ok.any():                     # e.g. post-surgery parameters before any such reading
        return float("nan")
    v, w = v[ok], w[ok]
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(cw / cw[-1], q), len(v) - 1)])


class Backend(ABC):
    name = "base"

    @abstractmethod
    def fit(self, problem: InferenceProblem, settings: dict, seed: int = 0,
            verbose: bool = False) -> Posterior:
        """Posterior of problem.names given problem's data. settings = profile[self.name]."""
