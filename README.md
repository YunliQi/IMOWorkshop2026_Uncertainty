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

## ABC-SMC forecasts (`abc_smc.ipynb`)

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
| `ovca/abc_smc.py` | ABC-SMC posterior, forecast, next-visit rule and update for one patient |
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
