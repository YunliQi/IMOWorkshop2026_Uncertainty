# IMOWorkshop2026_Uncertainty

Synthetic CA-125 data for HGSOC patients (NACT → surgery → ACT → surveillance), tumour models
built on one base class, and RMSE calibration of a model to the synthetic data.
All data here is synthetic; no real patient data is used.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run_pipeline.py        # ~5 s: generate -> calibrate -> figures
.venv/bin/python tests/test_ovca.py     # or: .venv/bin/python -m pytest tests -q
```

## Usage

Run everything (generate 60 patients → RMSE calibration → figures):

```bash
.venv/bin/python run_pipeline.py                      # defaults from ovca/config.py
.venv/bin/python run_pipeline.py --n 100 --seed 1     # other cohort size / seed
.venv/bin/python run_pipeline.py --gen-model growth_death --fit-model growth_death
```

From Python (run from the repository root):

```python
import numpy as np
from ovca import generator, calibration
from ovca.models import get_model
from ovca.patient import PatientParameters, Schedule

model = get_model("growth_death")        # any model registered in ovca/models/__init__.py

# --- generate synthetic data ---
# a cohort: parameters drawn from config.POPULATION, gates applied, one seed per patient
data, readings, truth = generator.generate_cohort(model, n=60, seed=2026)
generator.save(data)                     # -> data/synthetic_data.csv (format below)
readings, truth = generator.load()       # read it back later

# one patient again from its seed (the patient_seed column)
p, (daily, obs, tr) = generator.generate_patient(model, "SYN01", patient_seed=123)

# one patient from parameters you choose (times in months)
p = PatientParameters(
    patient_id="P1", model="growth_death",
    theta_pre={"gamma": 0.10, "delta": 0.90},     # before surgery
    theta_post={"gamma": 0.10, "delta": 0.50},    # after surgery
    ca125_0=800.0, q_surgery=0.2,
    schedule=Schedule(nact_starts=[0, 1, 3], t_surgery=5, act_starts=[6, 7, 9]))
daily, obs, tr = generator.generate_longitudinal(model, p, np.random.default_rng(0))

# --- calibrate a model to the data (RMSE, each segment separately) ---
fits = calibration.calibrate_cohort(model, readings, truth)   # truth supplies each schedule
r = truth.iloc[0]
one = calibration.calibrate_patient(model, readings[readings.patient == r.patient],
                                    Schedule.from_record(r))
calibration.recovery_table(fits, truth, model.param_names)    # fitted vs true

# --- simulate a model directly ---
t = np.linspace(0, 20, 200)                                    # months
model.simulate_patient(t, p)                                   # both segments + surgery jump
model.simulate_segment(t, 0.0, 500.0, {"gamma": 0.1, "delta": 0.9},
                       Schedule([0, 1], 3, []).intervals("pre"))   # one segment
```

What each returns:

| Call | Returns |
|---|---|
| `generate_cohort` | `data` (one row per patient-day), `readings` (clinic visits only: patient, day, month, phase, ca125, ca125_true), `truth` (one row per patient) |
| `generate_patient` | `(PatientParameters, (daily, readings, truth_dict))` |
| `calibrate_cohort` | one row per patient: `<param>_pre/_post`, `rmse_*`, `n_used_*` (readings ≥ 35 used), `fitted_*`, `at_bound_*` |

To change cohort size, noise, schedule rules, gates or the population, edit `ovca/config.py`.

## Forecasting tool (`ovca.Forecaster`, `tool_demo.ipynb`)

Inputs: one patient's readings, a model, any time point, and optionally a time decay. Outputs:
- the posterior of the parameters;
- tumour-burden / CA-125 bands over time;
- a recurrence interval;
- a suggested month for the next visit.

After a new reading, `step` refits and reports how much the uncertainty shrank. `replay` refits
after every visit. Install once with `uv pip install --python .venv/bin/python -e .`; after that
`import ovca` works from any folder.

```python
import pandas as pd
from ovca import Forecaster, PatientRecord, default_profile, check_profile

rec = PatientRecord.from_synthetic(pd.read_csv("data/synthetic_data.csv"), "SYN38")
# or PatientRecord.from_arrays(pid, t_months, ca125, nact_starts, t_surgery, act_starts)

profile = default_profile("growth_death", backend="abc")    # every setting, with defaults
profile["priors"]["gamma_pre"] = {"dist": "lognormal", "median": 0.05, "sd_log": 0.5}
profile["priors"]["sigma"] = 0.15                         # a number = fixed (default: estimated)
profile["data"]["decay_tau"] = 12                         # optional time decay (months)
profile = check_profile(profile)                          # prints the priors; errors name the problem

fc = Forecaster(profile)
snap = fc.fit(rec, t_now=rec.schedule.t_end)              # at the end of ACT
snap.report()            # recurrence interval, P(recur in 6 / 12 mo), next visit, band there
snap.summary()           # posterior quantiles of every parameter
snap2 = fc.step(snap, t_new=14.0, y_new=3.1)              # a new reading
snap2.shrinkage()        # band and recurrence-interval width before / after, surprise
hist = fc.replay(rec)    # refit after every visit (~0.1 s per fit)
hist.table(); hist.parameter_table()
hist.plot_tube("tube.png"); hist.plot_panels("panels.png")
hist.plot_parameters("params.png"); hist.plot_parameter_space("space.png")
snap.recurrence_times()  # each draw's recurrence time with no horizon (months after ACT)
snap.plot_recurrence("rec.png"); hist.plot_recurrence("rec_history.png")   # recurrence-time densities
snap.plot_recurrence_cdf("cdf.png"); hist.plot_recurrence_cdf("cdf_history.png")   # P(recurred by t)
```

**Model** (`ovca/problem.py`), shared by every backend:
- **Parameters:** the model's parameters before surgery (`gamma_pre`, `delta_pre`) and the surgery shifts `shift_gamma`, `shift_delta` (log post = log pre + shift).
- **Levels:** `level_pre` and `level_post`, the tumour burden at each segment's first reading. There is no surgery fraction q: the post-surgery level is a free parameter.
- **Observation error:** `sigma`, estimated by default. Readings follow log y = log T(t) + σ·ε.
- **Before the first post-surgery reading:** only the pre-surgery part is fitted. Nothing is forecast after surgery (bands, recurrence interval and next visit are empty) until that reading is in.

**Profile** (`ovca/profile.py`): one dict holds everything:
- `backend`, `model`;
- `priors` (one entry per parameter: lognormal / normal / halfnormal / uniform / a number = fixed) and `correlations`;
- `surgery`, `initial_level`, `data` (`censor_below`, `decay_tau`, `cutoff`);
- `abc`, `mcmc`, `forecast`, `next_visit`, `seed`.

`check_profile` catches unknown or misspelt keys and suggests the right one. It also catches:
- priors that don't match the model's parameters;
- invalid distributions or arguments;
- positive parameters given a prior that allows negative values;
- shifts that are not model parameters;
- settings that the chosen backend doesn't use.

**Backends** (`ovca/backends/`):
- **`abc`:** ABC-SMC. By default the acceptance kernel is the observation-error density (Wilkinson 2013), tempered from wide to exact, with Metropolis moves under a Gaussian perturbation kernel. This lets σ be estimated. `abc.acceptance = "threshold"` is classic distance < ε ABC-SMC: simulations include noise with a fixed σ. It gives slightly wider intervals and is about 15× slower.
- **`mcmc`:** PyMC NUTS for the single-patient growth-death model, using the same priors and likelihood as likelihood-mode ABC. Install with `pip install -e ".[mcmc]"`, then select `backend="mcmc"`. Sampling diagnostics are available in `snap.posterior.diagnostics`. See [docs/BACKEND_GUIDE.md](docs/BACKEND_GUIDE.md).

On the 30 test patients (SYN31–60), the 80% intervals contain the true value 70–93% of the
time across parameters. The recurrence interval contains it 93% of the time at the end of ACT.
The median σ estimate is 0.15 (true 0.15).

### Using MCMC

Install the optional PyMC and ArviZ dependencies in the environment running the tool:

```bash
pip install -e ".[mcmc]"
```

In `tool_demo.ipynb`, an optional installation cell appears near the start; uncomment and run
it if needed, then restart the kernel. Section 7 fits the demo patient with MCMC, displays
sampling diagnostics, and plots the ABC and MCMC forecasts side by side using `Snapshot.plot`.

```python
from ovca import Forecaster, default_profile

profile = default_profile("growth_death", backend="mcmc")
profile["mcmc"].update(draws=1000, tune=1000, chains=4, cores=1,
                       target_accept=0.9)
profile["seed"] = 2026
fc = Forecaster(profile)
snap = fc.fit(rec, t_now=rec.schedule.t_end)
print(snap.summary())
print(snap.recurrence_interval())
snap.plot()

diagnostics = snap.posterior.diagnostics
print("divergences:", diagnostics["divergences"])
print("R-hat:", diagnostics["r_hat"])
print("bulk effective sample size:", diagnostics["ess"])
# diagnostics["idata"] contains the ArviZ inference data.
```

`draws` and `tune` are per chain; `cores` controls parallel sampling and may be `None`.
Only the `nuts` sampler is currently supported. Check divergences, R-hat, and effective
sample sizes before interpreting a fit; more tuning or draws may be needed. The same `step`,
`replay`, reports, and plotting methods work with MCMC, although replay runs a new sampler
at each visit and can take substantially longer than a single fit.

MCMC and likelihood-mode ABC use the same single-patient growth–death model and profile
priors, including correlations, fixed parameters, observation noise, censoring, and time
decay. Before the first post-surgery reading both fit only the pre-surgery segment. Their
posterior estimates can differ because of Monte Carlo error. Threshold-mode ABC is an
approximation with fixed observation noise, so it need not agree exactly with MCMC. The
notebook's model 2 and hierarchical fitting are not supported by the MCMC backend.

## ABC-SMC forecasts, first version (`abc_smc.ipynb`)

The first version, kept for reference: `ovca/abc_smc.py`, with a surgery fraction `q` and a
fixed noise. New work should use the forecasting tool above.

`ovca/abc_smc.py` gives, for one patient at any time point `t_now`: the posterior of the model
parameters (ABC-SMC), a forecast of CA-125 and of recurrence with intervals, a suggested time for
the next CA-125 visit, and an update once that reading is in. Surgery is a jump (`q`) plus a
shift of the model parameters on the log scale (`shift_*`). Readings below 35 are used as they are
(`censor_below=35` treats them as "below 35" only); optional time decay down-weights old readings.
[`abc_smc.ipynb`](abc_smc.ipynb) is the pipeline, figures and usage guide (runs in ~7 min):

```bash
uv pip install --python .venv/bin/python nbformat nbconvert ipykernel   # to run the notebook
.venv/bin/jupyter nbconvert --to notebook --execute --inplace abc_smc.ipynb
```

```python
from ovca import abc_smc as A
post = A.fit(model, t_obs, y_obs, schedule, t_now=schedule.t_end,
             prior=A.prior_for(A.default_prior_spec(model), y_obs[0]))
fc = post.forecast()
fc.recurrence_interval(0.8); fc.next_visit(rule="risk", p=0.10)
post = post.update(t_new, y_new)
```

## Layout

| File | What it does |
|---|---|
| `ovca/config.py` | every number: cutoff 35, D = 1 for one month per treatment, noise, schedule, population of parameters |
| `ovca/patient.py` | `Schedule` (NACT, surgery, ACT, D(t)) and `PatientParameters` (one patient's full parameter set) |
| `ovca/models/base.py` | `BaseModel`: segment dynamics `dx/dt = rhs(t, x, θ, D(t))`, generic ODE solver, two-phase patient simulation with the surgery jump |
| `ovca/models/growth_death.py` | the poster model `dT/dt = (γ − δ·D(t))·T`, with its exact solution |
| `ovca/models/__init__.py` | model registry (`get_model("growth_death")`) |
| `ovca/generator.py` | parameter set → longitudinal CA-125 data (noise, surveillance, recurrence detection, true recurrence time) |
| `ovca/calibration.py` | RMSE fit of a model to the data, each segment separately |
| `ovca/abc_smc.py` | first-version ABC-SMC (with q, fixed noise), used by `abc_smc.ipynb` |
| `ovca/forecaster.py` | the tool: `Forecaster` (`fit`, `step`, `replay`), `Snapshot`, `History` |
| `ovca/profile.py`, `ovca/priors.py` | profiles (`default_profile`, `check_profile`) and prior distributions |
| `ovca/record.py`, `ovca/problem.py` | `PatientRecord` (input) and the shared probabilistic model |
| `ovca/backends/` | `abc_smc.py` (ABC-SMC), `mcmc.py` (PyMC NUTS) |
| `ovca/forecast.py`, `ovca/viz.py` | forecasts (bands, recurrence, next visit, surprise) and figures |
| `ovca/plots.py` | figures in `figures/` |
| `run_pipeline.py` | runs everything; `--gen-model` and `--fit-model` can differ |

## Pipeline

1. **Parameter set** (`generator.sample_parameters`): γ, δ before and after surgery are log-normal
   around a typical patient (pre and post correlated 0.6), plus baseline CA-125, surgery fraction q
   and a random schedule. A `PatientParameters` can also be written by hand.
   - **Schedule** (whole months): 3–4 NACT cycles from month 0, each on for 1 month then a
     0–1 month gap; surgery 1–2 months after the last NACT cycle ends; first ACT 1–2 months
     after surgery; 3 ACT cycles with the same 0–1 month gaps.
   - **Gates** (a failing draw is redrawn): NACT brings CA-125 only partly down (true CA-125
     stays ≥ 30 U/mL before surgery; just before surgery it is ~40 / 80 / 190 at the 10th / 50th /
     90th percentile); surgery is the big drop
     (CA-125 after/before in [0.01, 0.3]). After surgery CA-125 may go as low as the model takes it.
2. **Longitudinal data** (`generator.generate_longitudinal`): the model is simulated day by day
   before surgery, the state is multiplied by q at surgery, then simulated after surgery with the
   post-surgery parameters, up to 36 months after the end of ACT. CA-125 = model value with 15%
   multiplicative noise. Clinic visits (flagged `visit = 1`) are at each cycle start, around
   surgery, at the end of ACT and every ~3 months after; the first surveillance reading ≥ 35
   counts as detected recurrence (one confirm visit, then visits stop).
3. **Calibration** (`calibration.calibrate_cohort`): least squares on RMSE, as in the poster code:
   T0 of each segment = its first reading; readings below 35 count as 0 and are left out of the RMSE.

## Data format

`data/synthetic_data.csv`, one row per patient per day:

| Column | Meaning |
|---|---|
| `tumor_burden` | value of the generating model (noise-free) |
| `CA125` | `tumor_burden` with assay noise |
| `time` | day since the first NACT cycle |
| `treatment_applied` | D(t): 1 while a treatment cycle is on, else 0 |
| `patient`, `model`, `cohort_seed`, `patient_seed` | who and how it was generated (`generator.generate_patient(model, patient, patient_seed)` regenerates one patient) |
| `phase`, `visit` | `pre`/`post` surgery; 1 on clinic-visit days (the readings calibration uses) |
| `noise_sd_log`, `gamma_pre`, `delta_pre`, `gamma_post`, `delta_post`, `ca125_0`, `q_surgery` | parameters used |
| `nact_starts`, `t_surgery`, `act_starts`, `t_end` | schedule, in months |
| `true_recurrence_abs`, `true_recurrence_rel`, `t_detected` | true and detected recurrence, in months (`inf` = none) |

Other outputs: `results/calibration.csv`, `figures/0–2_*.png`.

## Adding a model

Create `ovca/models/<name>.py` with a `BaseModel` subclass that sets `name`, `param_names`,
`bounds`, `starts` and implements `rhs(t, x, theta, drug)`. For more than one state (e.g.
sensitive and resistant cells, Lotka–Volterra style), also override `initial_state`, `observe`
and `surgery`. Register it in `ovca/models/__init__.py` and add its population to
`config.POPULATION`. The generic solver integrates between treatment switches, so no exact
solution is needed.
