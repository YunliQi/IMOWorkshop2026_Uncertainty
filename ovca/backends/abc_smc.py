"""ABC-SMC backend.

Two acceptance kernels (profile["abc"]["acceptance"]):

"likelihood" (default). The acceptance kernel is the observation-error density
    K = prod_i N(log y_i | log T(t_i), sigma^2)^{w_i}
of a noise-free simulation T (Wilkinson 2013: ABC with this kernel is exact inference under
"reading = model x lognormal error"). The tolerance shrinks over generations by tempering,
K^beta with beta: 0 -> 1, i.e. an error sd of sigma / sqrt(beta) that ends at sigma. The next
beta keeps an effective sample size of alpha * N; particles are then resampled and moved with
n_moves Metropolis steps under a Gaussian perturbation kernel (covariance = scale x particle
covariance). Because K contains the 1/sigma normalising factor, sigma can be estimated.

"threshold". Classic ABC-SMC (Beaumont et al. 2009): simulate readings with observation error
sigma (fixed) and accept when their weighted RMS log distance to the data is below epsilon (the
alpha-quantile of the previous generation's distances); importance weights prior / mixture of
perturbation kernels; stop when acceptance falls below min_accept_rate. The noise must be in the
simulation: as epsilon -> 0 this gives the exact posterior, whereas noise-free simulations would
collapse onto the single best-fitting curve. A finite epsilon makes the posterior a bit wider.
"""
import numpy as np
from scipy.special import logsumexp

from .base import Backend, Posterior


def _wcov(x, w):
    m = (w[:, None] * x).sum(0)
    d = x - m
    return (w[:, None] * d).T @ d


def _ess(logw):
    w = np.exp(logw - logw.max())
    return w.sum() ** 2 / (w ** 2).sum()


def _systematic(rng, w, n):
    pos = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), pos), len(w) - 1)


class ABCBackend(Backend):
    name = "abc"

    def fit(self, problem, settings, seed=0, verbose=False):
        if settings["acceptance"] == "likelihood":
            return self._fit_likelihood(problem, settings, seed, verbose)
        return self._fit_threshold(problem, settings, seed, verbose)

    # ------------------------------------------------------------------ kernel = likelihood
    def _fit_likelihood(self, problem, st, seed, verbose):
        rng = np.random.default_rng(seed)
        prior = problem.prior
        n, k = st["n_particles"], len(problem.names)
        pert = st["perturbation"]
        scale = 2.38 ** 2 / max(k, 1) if pert["scale"] == "auto" else float(pert["scale"])
        u = prior.sample(rng, n)
        lp, ll = prior.logpdf(u), problem.loglik(u)
        ll = np.where(np.isfinite(ll), ll, -np.inf)
        beta, history = 0.0, []
        for g in range(st["max_generations"]):
            # next beta: ESS of the incremental weights = alpha * n
            target = st["alpha"] * n
            if _ess((1.0 - beta) * ll) >= target:
                dbeta = 1.0 - beta
            else:
                lo, hi = 0.0, 1.0 - beta
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if _ess(mid * ll) >= target else (lo, mid)
                dbeta = max(lo, 1e-12)
            logw = dbeta * ll
            w = np.exp(logw - logw.max())
            w /= w.sum()
            beta = min(1.0, beta + dbeta)
            idx = _systematic(rng, w, n)
            u, lp, ll = u[idx], lp[idx], ll[idx]
            # move: Metropolis with a Gaussian perturbation kernel
            acc_total = 0.0
            for _ in range(st["n_moves"]):
                cov = scale * np.cov(u, rowvar=False).reshape(k, k) + 1e-10 * np.eye(k)
                chol = np.linalg.cholesky(cov)
                prop = u + rng.standard_normal((n, k)) @ chol.T
                lp_p = prior.logpdf(prop)
                ll_p = problem.loglik(prop)
                ll_p = np.where(np.isfinite(ll_p), ll_p, -np.inf)
                log_a = (lp_p + beta * ll_p) - (lp + beta * ll)
                acc = np.log(rng.random(n)) < log_a
                u[acc], lp[acc], ll[acc] = prop[acc], lp_p[acc], ll_p[acc]
                rate = acc.mean()
                acc_total += rate
                if pert.get("adapt", True):
                    scale *= 1.3 if rate > 0.45 else 0.7 if rate < 0.15 else 1.0
            history.append(dict(generation=g + 1, beta=beta, tolerance=1 / np.sqrt(beta),
                                accept_rate=acc_total / st["n_moves"], scale=scale))
            if verbose:
                print(f"gen {g + 1}: beta={beta:.4f} accept={acc_total / st['n_moves']:.2f} "
                      f"scale={scale:.3f}")
            if beta >= 1.0:
                break
        diag = dict(acceptance="likelihood", generations=history, beta=beta,
                    converged=beta >= 1.0, n_loglik=n * len(history) * (st["n_moves"] + 0) + n)
        if beta < 1.0:
            import warnings
            warnings.warn(f"ABC stopped at beta = {beta:.3f} < 1 (max_generations reached): "
                          f"the posterior is wider than it should be", stacklevel=3)
        return Posterior(problem, u, np.full(n, 1.0 / n), "abc", diag)

    # ------------------------------------------------------------------ threshold
    def _fit_threshold(self, problem, st, seed, verbose):
        rng = np.random.default_rng(seed)
        prior = problem.prior
        n, k = st["n_particles"], len(problem.names)
        pert = st["perturbation"]
        scale = 2.0 if pert["scale"] == "auto" else float(pert["scale"])
        x = prior.sample(rng, n)
        d = problem.distance(x, rng)
        w = np.full(n, 1.0 / n)
        history = []
        for g in range(1, st["max_generations"] + 1):
            eps = float(np.quantile(d, st["alpha"]))
            cov = scale * _wcov(x, w) + 1e-10 * np.eye(k)
            chol = np.linalg.cholesky(cov)
            acc_x, acc_d, sims = [], [], 0
            while sum(len(a) for a in acc_x) < n and sims < st["max_sims_per_gen"]:
                prop = x[rng.choice(n, st["batch"], p=w)]
                prop = prop + rng.standard_normal(prop.shape) @ chol.T
                ok = np.isfinite(prior.logpdf(prop))
                dp = np.where(ok, problem.distance(prop, rng), np.inf)
                keep = dp <= eps
                acc_x.append(prop[keep])
                acc_d.append(dp[keep])
                sims += st["batch"]
            all_x, all_d = np.concatenate(acc_x), np.concatenate(acc_d)
            rate = len(all_d) / sims
            history.append(dict(generation=g, epsilon=eps, sims=sims, accept_rate=rate))
            if verbose:
                print(f"gen {g}: eps={eps:.3f} accept={rate:.3f}")
            if len(all_x) < max(50, n // 5):
                break
            new_x, new_d = all_x[:n], all_d[:n]
            inv = np.linalg.inv(cov)
            diff = new_x[:, None, :] - x[None, :, :]
            log_k = -0.5 * np.einsum("ijk,kl,ijl->ij", diff, inv, diff)
            log_w = prior.logpdf(new_x) - logsumexp(log_k + np.log(w)[None, :], axis=1)
            nw = np.exp(log_w - log_w.max())
            x, d, w = new_x, new_d, nw / nw.sum()
            if len(x) < n:
                pad = rng.choice(len(x), n - len(x), p=w)
                x, d = np.concatenate([x, x[pad]]), np.concatenate([d, d[pad]])
                w = np.full(n, 1.0 / n)
            if rate < st["min_accept_rate"]:
                break
        return Posterior(problem, x, w, "abc", dict(acceptance="threshold",
                                                    generations=history, distances=d))
