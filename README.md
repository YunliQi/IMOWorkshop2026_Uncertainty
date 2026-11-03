# IMOWorkshop2026_Uncertainty

Synthetic CA-125 data for HGSOC patients (NACT → surgery → ACT → surveillance), tumour models
built on one base class, and RMSE calibration of a model to the synthetic data.
All data here is synthetic; no real patient data is used.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run_pipeline.py        # ~5 s: generate -> calibrate -> figures
.venv/bin/python tests/test_ovca.py     # or: .venv/bin/python -m pytest tests -q
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
