"""MCMC backend: NOT IMPLEMENTED YET (left for the team; see docs/BACKEND_GUIDE.md).

What to implement: MCMCBackend.fit(problem, settings, seed) -> Posterior, sampling the same
posterior as the ABC backend:

    prior     problem.prior (one Dist per name in problem.names, on the u scale; correlations
              in problem.prior.corr) -- or rebuild it in PyMC from problem.prior.dists
    log T(t)  for growth_death, closed form (see InferenceProblem.components): with
              level_pre at problem.a_pre and level_post at problem.a_post,
                log T(t) = log level_pre + g_pre * (t - a_pre) - d_pre * on(a_pre, t)   t < ts
                log T(t) = log level_post + g_post * (t - a_post) - d_post * on(a_post, t)  t >= ts
              where on(a, t) is the treatment-on time between a and t (signed: negative if
              t < a), g_post = g_pre * exp(shift_gamma), d_post = d_pre * exp(shift_delta).
              Before the first post-surgery reading (problem.post_mode == "pre_only") only the
              pre-surgery readings and parameters exist (no shifts, no level_post).
    likelihood  log y_i ~ Normal(log T(t_i), sigma), weight problem.w[i] (use pm.Potential with
              w * logp when decay is on); censored readings (problem.cens) use
              log Phi((log censor_below - log T) / sigma).
    output    Posterior(problem, u=draws on the u scale (N, k) in problem.names order,
              weights=np.ones(N), backend="mcmc", diagnostics={"r_hat": ..., "ess": ...,
              "divergences": ...})

settings = profile["mcmc"]: sampler, draws, tune, chains, target_accept, cores. The checks in
tests/test_tool.py::test_mcmc_* (skipped until this exists) must pass.
"""
from .base import Backend


class MCMCBackend(Backend):
    name = "mcmc"

    def fit(self, problem, settings, seed=0, verbose=False):
        raise NotImplementedError(
            "The MCMC backend is not implemented yet. See docs/BACKEND_GUIDE.md for the "
            "interface it must follow; use backend='abc' meanwhile.")
