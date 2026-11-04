# Writing a backend (MCMC)

The forecasting tool (`ovca.Forecaster`) separates **what** is inferred from **how**.
The probabilistic model of one patient at one time point is `ovca.problem.InferenceProblem`.
A backend turns that problem into a `Posterior`. Forecasts, the next-visit rule, `step`,
`replay` and every figure read only the `Posterior`, so a new backend needs no changes anywhere
else.

`ovca/backends/mcmc.py` implements single-patient PyMC NUTS for growth-death. Install
the optional dependencies with `pip install -e ".[mcmc]"` and select
`profile["backend"] = "mcmc"`. It targets the likelihood-mode ABC posterior.
Sampler diagnostics and the ArviZ inference data are in `posterior.diagnostics`.
The notebook’s second model and hierarchical fitting are not included.

## 0. Input and output at a glance

```
Forecaster.fit(record, t_now)
   │  builds
   ▼
InferenceProblem  ──►  YourBackend.fit(problem, settings, seed)  ──►  Posterior(problem, u, weights)
(data + prior +                                                       │
 likelihood + simulator)                                              ▼
                                                    forecast, report, step, replay, every figure
```

| | What | Shape / type | Example (SYN38 at the end of ACT, 9 readings) |
|---|---|---|---|
| **in** | `problem.names` | list of k names | `['gamma_pre', 'delta_pre', 'shift_gamma', 'shift_delta', 'level_pre', 'level_post', 'sigma']` (k = 7) |
| **in** | `problem.t`, `problem.y` | (n,) months, U/mL | the 9 readings with `t <= t_now`, sorted |
| **in** | `problem.prior.logpdf(u)` | (N, k) → (N,) | `[-4.95, -8.09, ...]` |
| **in** | `problem.loglik(u)` | (N, k) → (N,) | `[-2.19, -716.11, ...]`: the target your sampler must reproduce |
| **in** | `settings` | dict | `{"sampler": "nuts", "draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9, "cores": None}` |
| **out** | `Posterior.u` | (N, k) float, unconstrained scale, columns = `problem.names` | 4 chains × 1000 draws → (4000, 7) |
| **out** | `Posterior.weights` | (N,) | `np.ones(4000)` |
| **out** | `Posterior.diagnostics` | dict | `{"r_hat": {...}, "ess": {...}, "divergences": 0, "idata": idata}` |

Before the first post-surgery reading (`problem.post_mode == "pre_only"`), `names` is
`['gamma_pre', 'delta_pre', 'shift_gamma', 'shift_delta', 'level_pre', 'surgery_jump', 'sigma']`
(no `level_post`). The likelihood does not depend on the shifts or `surgery_jump`, so their
posterior is their prior; sample them anyway, they make the forecast after surgery.

Quick look from Python:

```python
fc = Forecaster(default_profile())
problem = fc.problem(record, t_now=record.schedule.t_end)
print(problem.describe())                      # names, anchors, mode
u = problem.prior.sample(np.random.default_rng(0), 5)
problem.prior.logpdf(u), problem.loglik(u)     # what your model must reproduce
problem.natural(u)                             # natural units, incl. gamma_post, delta_post
```

## 1. The contract

```python
class MCMCBackend(Backend):
    name = "mcmc"
    def fit(self, problem: InferenceProblem, settings: dict, seed=0, verbose=False) -> Posterior:
        ...
```

| Argument | What it is |
|---|---|
| `problem` | Data, schedule, prior, likelihood and simulator of one patient at `problem.t_now` (§2) |
| `settings` | `profile["mcmc"]`: `sampler`, `draws`, `tune`, `chains`, `target_accept`, `cores` (add keys to `default_profile` and `check_profile` in `ovca/profile.py` if you need more) |
| `seed` | Random seed. `replay` passes `profile["seed"] + k` for the k-th refit |

Return `Posterior(problem, u, weights, backend="mcmc", diagnostics={...})`:

- `u`: array `(N, k)` of draws on the **unconstrained scale**, with columns in `problem.names` order (§3). Flatten chains and draws together.
- `weights`: `np.ones(N)`. They are normalised for you.
- `diagnostics`: anything useful, e.g. `r_hat`, `ess`, `divergences` and the `InferenceData`.

## 2. What `InferenceProblem` gives you

| Attribute / method | Meaning |
|---|---|
| `names` | Sampled parameters, in order. For example `['gamma_pre', 'delta_pre', 'shift_gamma', 'shift_delta', 'level_pre', 'level_post', 'sigma']`. A parameter whose prior is a plain number is **fixed** and not in `names` (`prior.fixed`) |
| `prior.dists[name]` | `Dist`: `kind` ∈ lognormal / normal / halfnormal / uniform, `args` (numbers; `"first_reading"` is already replaced by the reading) |
| `prior.corr` | `{(a, b): r}` correlations on the u scale (lognormal / normal parameters only) |
| `prior.logpdf(u)`, `prior.sample(rng, n)` | Prior on the u scale, Jacobian included |
| `t`, `y`, `log_y` | Readings used (months, U/mL), all with `t <= t_now` |
| `w` | Likelihood weight of each reading (time decay; all 1 by default) |
| `cens`, `censor` | Boolean mask of censored readings and the censoring level (None = no censoring) |
| `schedule` | `Schedule`: `nact_starts`, `t_surgery`, `act_starts`, `intervals("pre"/"post")` |
| `a_pre`, `a_post` | Anchor times of the two level parameters (each segment's first reading) |
| `post_mode` | `"level"` (there is a post-surgery reading) or `"pre_only"` (not yet: `surgery_jump` is sampled instead of `level_post`, and only the pre-surgery readings enter the likelihood) |
| `sp` | Parameters that surgery shifts (`surgery.shift_params`) |
| `loglik(u)` | Numpy reference log-likelihood, shape (N,): **your PyMC model must match it** (§5) |
| `simulate(u, t)`, `natural(u)` | Numpy simulator and natural-unit parameters, for checks |

## 3. Unconstrained scale u

| Prior kind | u |
|---|---|
| lognormal, halfnormal | `log x` |
| normal | `x` |
| uniform(a, b) | `logit((x - a) / (b - a))` |

In PyMC it is easiest to define the natural parameters with their own distributions:
`pm.LogNormal(mu=log(median), sigma=sd_log)`, `pm.Normal`, `pm.HalfNormal` and
`pm.Uniform`. Then convert the draws with `problem.prior.dists[n].to_u(x)` before building
`Posterior`. Correlations: put the correlated Gaussian-u parameters in one `pm.MvNormal`
on the u scale and exponentiate the lognormal ones.

## 4. The model (growth_death)

With `g_post = g_pre · exp(shift_gamma)` and `d_post = d_pre · exp(shift_delta)`. A parameter
missing from `sp` has shift 0. `on(a, t)` is the treatment-on time between a and t
(`ovca.patient.time_on(t, a, intervals)`). Readings are never before their segment's anchor,
because the anchor is the segment's first reading:

```
t <  t_surgery:  log T(t) = log level_pre  + g_pre  (t - a_pre)  - d_pre  on_pre(a_pre, t)
t >= t_surgery:  log T(t) = log level_post + g_post (t - a_post) - d_post on_post(a_post, t)
```

In `"pre_only"` mode there are no post-surgery readings, so the likelihood has only the first
line; the shifts and `surgery_jump` are sampled from their priors. The time-on terms depend only on the
data, so compute them in numpy and pass them in as constants. Then the model is a few PyTensor lines, like `mcmc.ipynb`, and NUTS gets
gradients for free.

Two changes from the current `mcmc.ipynb`:

- **`sigma` is a parameter** (prior `profile["priors"]["sigma"]`), not a fixed `noise_sd`.
  Fixed only when `"sigma"` is in `prior.fixed`.
- **The levels are anchored at each segment's first reading.** That replaces `T0` and `T_post0`
  (anchoring at t = 0 and at surgery). The notebook's `T_post0` is the same idea; only the
  anchor time moves to `a_post`.

For other models, implement `BaseModel.log_tumour_symbolic` and set `symbolic = True`.

## 5. Likelihood

```
uncensored i:  w_i · log Normal(log y_i | log T(t_i), sigma)
censored i:    w_i · log Phi((log censor - log T(t_i)) / sigma)
```

With weights that are not all 1, or with censoring, add the terms with
`pm.Potential("loglik", (w * logp).sum())` instead of an `observed=` variable. Check it against
the reference: for a handful of u vectors, your log-likelihood must equal `problem.loglik(u)`
to about 1e-8.

## 6. Validation

- `test_mcmc_contract` and `test_mcmc_matches_abc` in `tests/test_tool.py` check the posterior contract and agreement with ABC when PyMC is installed.
- `.venv/bin/python -m pytest tests -q` passes.
- `Forecaster({"backend": "mcmc"}).replay(record)` gives the same figures as ABC, up to Monte Carlo noise. For the growth_death model the two must agree, because both target the same posterior.
- Optional: install with `uv pip install -e ".[mcmc]"` (pymc, arviz).

## 7. Connecting to the plotting API

The figures never talk to a backend directly. They read the `Posterior` you return, through
two routes:

```
Posterior ── resample by weights ──► problem.simulate / recurrence ──► Forecast ──► tube, panels, snap.plot
    ├────── resample by weights ──► problem.recurrence (no horizon) ──► plot_recurrence, plot_recurrence_cdf
    └────── problem.natural(u) ──► parameter table ──► plot_parameters, plot_parameter_distributions,
                                                         plot_parameter_space
```

So if `fit` returns a correct `Posterior`, every figure works without changes:

```python
fc = Forecaster(profile)                     # profile["backend"] = "mcmc"
snap = fc.fit(record, t_now)                 # Snapshot
snap.plot("snap.png")
hist = fc.replay(record)                     # History: one fit per visit
hist.plot_tube("tube.png"); hist.plot_panels("panels.png")
hist.plot_parameters("params.png"); hist.plot_parameter_distributions("post.png")
hist.plot_parameter_space("space.png")
snap.plot_recurrence("rec.png"); hist.plot_recurrence("rec_history.png")
snap.plot_recurrence_cdf("cdf.png"); hist.plot_recurrence_cdf("cdf_history.png")
hist.table(); hist.parameter_table()         # the numbers behind the figures
```

### What each figure reads

| Method | Reads | From your `Posterior` |
|---|---|---|
| `snap.plot`, `hist.plot_tube`, `hist.plot_panels` | `snap.forecast`: tumour / CA-125 bands, recurrence times, next visit | draws resampled with `weights`, pushed through `problem.simulate` and `problem.recurrence`; each draw's `sigma` adds the observation error |
| `hist.plot_parameters` | `hist.parameter_table()`: weighted quantiles per refit | `natural()` columns + `weights` |
| `hist.plot_parameter_distributions` | `natural()` resampled with `weights` (1500 draws per refit) | same |
| `hist.plot_parameter_space` | weighted mean and covariance of pairs of `natural()` columns | same |
| `snap.plot_recurrence`, `hist.plot_recurrence`, `*.plot_recurrence_cdf` | `snap.recurrence_times()`: each draw's recurrence time with no horizon (exact for growth_death) | draws resampled with `weights`, pushed through `problem.recurrence` |

`natural()` returns every sampled parameter plus the derived ones (`gamma_post`, `delta_post`,
`ca125_0`, `surgery_jump`, and `level_post` reported at one fixed time for every refit). Those
are the column names the figures look for.

### Checklist for the figures to come out right

1. **Columns of `u` in `problem.names` order, on the unconstrained scale** (§3). A column in the wrong order or on the natural scale gives wrong parameter panels and wrong bands, with no error.
2. **Weights:** finite and ≥ 0. Equal weights (`np.ones(N)`) are the normal case for MCMC.
3. **Enough draws:** a few hundred at least. Forecasts resample `profile["forecast"]["n_draws"]` draws and the violins resample 1500, so more draws than that only cost time.
4. **Before the first post-surgery reading** (`post_mode == "pre_only"`), `problem.names` has `surgery_jump` instead of `level_post`. Return every column, including the shifts and `surgery_jump` that the readings don't inform: the forecast after surgery comes from them.
5. **Diagnostics are not plotted.** Put `r_hat`, `ess` and divergences in `diagnostics`; they are kept on `snap.posterior.diagnostics` for your own checks.

### Options you can pass to the plotting methods

| Option | Methods | Effect |
|---|---|---|
| `path` | all | save to this file (the figure is returned either way) |
| `reference` | `plot_tube`, `plot_panels` | black dashed line: `"auto"` (truth for synthetic, else `"forecast"`), `"forecast"`, `"hindsight"`, `"fit"`, `"none"` |
| `params` | `plot_parameters`, `plot_parameter_distributions` | which parameters (column names of `natural()`) |
| `pairs` | `plot_parameter_space` | list of `(x, y, title)` |
| `ncol` | `plot_panels`, `hist.plot_recurrence_cdf` | panels per row |
| `facet` | `hist.plot_recurrence_cdf` | one panel per refit (default); `False`: every refit on one axis, light = early |
| `noisy` | `snap.plot` | band for readings (with σ) instead of the noise-free tumour |
| `until` | `plot_recurrence`, `plot_recurrence_cdf` | time axis runs until this share of draws has recurred (default 0.999 for a snapshot; 0.9 for a history, 0.99 with `log_x`) |
| `levels` | `snap.plot_recurrence`, `hist.plot_recurrence` | central intervals to shade (default 95%, 80%, 50%) |
| `log_x`, `xmax` | `hist.plot_recurrence` (`xmax` also `hist.plot_recurrence_cdf`) | log time axis (default linear), so early wide and late narrow forecasts both show; fixed axis end |

The functions themselves are in `ovca/viz.py` (`viz.tube(hist, path)` etc., the same as the
methods). A new model with other parameter names needs entries in `viz.LABELS` (axis titles),
`viz.LINEAR` (parameters drawn on a linear axis; the rest are log), and the default lists
`BAND_PARAMS`, `ALL_PARAMS`, `SPACE_PAIRS`. You can also pass `params=` / `pairs=` instead.

### Comparing MCMC with ABC in one figure

Replay the same record with both backends, then draw the MCMC medians on top of the ABC figure.
The axes of `plot_parameters` follow the order of `params`:

```python
p_abc, p_mc = default_profile(backend="abc"), default_profile(backend="mcmc")
h_abc, h_mc = Forecaster(p_abc).replay(record), Forecaster(p_mc).replay(record)

params = ["gamma_post", "delta_post", "shift_delta", "sigma"]
fig = h_abc.plot_parameters(params=params)          # ABC: orange bands
S = h_mc.parameter_table()
for ax, p in zip(fig.axes, params):
    d = S[S.parameter == p]
    ax.step(d.t_now, d.q50, where="post", color="black", lw=1.2, label="MCMC median")
    ax.step(d.t_now, d.q10, where="post", color="black", lw=0.6, ls=":")
    ax.step(d.t_now, d.q90, where="post", color="black", lw=0.6, ls=":")
fig.axes[0].legend(fontsize=8, frameon=False)
fig.savefig("abc_vs_mcmc.png", dpi=140, bbox_inches="tight")

# the same for the forecasts
T = h_abc.table()[["t_now", "rec_low", "rec_median", "rec_high"]].merge(
    h_mc.table()[["t_now", "rec_low", "rec_median", "rec_high"]],
    on="t_now", suffixes=("_abc", "_mcmc"))
```

`test_mcmc_matches_abc` in `tests/test_tool.py` checks the same agreement in numbers. The
existing plot test also draws every figure from a replay, so it catches a `Posterior` that the
figures cannot use.
