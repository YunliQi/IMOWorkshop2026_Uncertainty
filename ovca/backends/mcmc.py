"""Single-patient NUTS inference for the shared growth-death model."""
import numpy as np

from .base import Backend, Posterior
from ..models.growth_death import LOG_CAP
from ..patient import time_on


def _build_model(problem):
    """Construct the PyMC target corresponding to InferenceProblem.loglik."""
    import pymc as pm
    import pytensor.tensor as pt

    prior = problem.prior
    with pm.Model() as model:
        values = dict(prior.fixed)
        if prior._gauss:
            # Non-centering also handles arbitrary correlations on the shared u scale.
            z = pm.Normal("prior_z", 0, 1, shape=len(prior._gauss))
            u = prior._mean + pt.dot(prior._chol, z)
            for j, i in enumerate(prior._gauss):
                name = prior.names[i]
                values[name] = pt.exp(u[j]) if prior.dists[name].kind == "lognormal" else u[j]
        for i in prior._other:
            name = prior.names[i]
            dist = prior.dists[name]
            if dist.kind == "halfnormal":
                values[name] = pm.HalfNormal(name, sigma=dist.args["sd"])
            else:
                values[name] = pm.Uniform(name, lower=dist.args["low"], upper=dist.args["high"])
        # Save natural values for extraction in the exact shared parameter order.
        for name in prior.names:
            if name not in model.named_vars:
                pm.Deterministic(name, values[name])

        s, t = problem.schedule, problem.t
        pre = t < s.t_surgery
        log_pre = (pt.log(values["level_pre"]) + values["gamma_pre"] * (t - problem.a_pre)
                   - values["delta_pre"] * time_on(t, problem.a_pre, s.intervals("pre")))
        if problem.post_mode == "level":
            g = values["gamma_pre"] * pt.exp(values.get("shift_gamma", 0.0))
            d = values["delta_pre"] * pt.exp(values.get("shift_delta", 0.0))
            log_post = (pt.log(values["level_post"]) + g * (t - problem.a_post)
                        - d * time_on(t, problem.a_post, s.intervals("post")))
            log_t = pt.where(pre, log_pre, log_post)
        else:
            log_t = log_pre
        log_t = pt.clip(log_t, np.log(1e-300), LOG_CAP)
        sigma = values["sigma"]
        logp = pm.logp(pm.Normal.dist(mu=log_t, sigma=sigma), problem.log_y)
        if problem.cens.any():
            censored = pm.logcdf(pm.Normal.dist(mu=log_t, sigma=sigma), np.log(problem.censor))
            logp = pt.where(problem.cens, censored, logp)
        pm.Potential("loglik", pt.sum(problem.w * logp))
    return model


class MCMCBackend(Backend):
    name = "mcmc"

    def fit(self, problem, settings, seed=0, verbose=False):
        if problem.model.name != "growth_death":
            raise ValueError("MCMC currently supports only the growth_death model")
        n = settings["draws"] * settings["chains"]
        if not problem.names:
            return Posterior(problem, np.empty((n, 0)), np.ones(n), self.name,
                             {"r_hat": {}, "ess": {}, "divergences": 0})
        try:
            import pymc as pm
            import arviz as az
        except ImportError as exc:
            raise ImportError("MCMC requires the optional dependencies: install ovca[mcmc]") from exc
        model = _build_model(problem)
        with model:
            idata = pm.sample(draws=settings["draws"], tune=settings["tune"],
                              chains=settings["chains"], cores=settings["cores"],
                              target_accept=settings["target_accept"], random_seed=seed,
                              progressbar=verbose, compute_convergence_checks=False)
        draws = np.column_stack([problem.prior.dists[name].to_u(
            idata.posterior[name].values.reshape(-1)) for name in problem.names])
        summary = az.summary(idata, var_names=problem.names, kind="diagnostics")
        diagnostics = {"r_hat": summary["r_hat"].to_dict(),
                       "ess": summary["ess_bulk"].to_dict(),
                       "divergences": int(idata.sample_stats.diverging.sum()), "idata": idata}
        return Posterior(problem, draws, np.ones(len(draws)), self.name, diagnostics)
