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
   - **Gates** (a failing draw is redrawn): surgery must be a clear jump, CA-125 after/before in
     [0.05, 0.8]; true CA-125 must stay ≥ 5 U/mL until the end of ACT.
2. **Longitudinal data** (`generator.generate_longitudinal`): the model is simulated before
   surgery, the state is multiplied by q at surgery, then simulated after surgery with the
   post-surgery parameters. CA-125 is read at each visit with 15% multiplicative noise; the first
   surveillance reading ≥ 35 counts as detected recurrence (one confirm visit, then follow-up stops).
3. **Calibration** (`calibration.calibrate_cohort`): least squares on RMSE, as in the poster code:
   T0 of each segment = its first reading; readings below 35 count as 0 and are left out of the RMSE.

Outputs: `data/synthetic_ca125.csv` (readings), `data/synthetic_treatments.csv` (NACT/ACT
cycles with start and end month, and surgery), `data/synthetic_truth.csv` (true parameters and
recurrence times), `results/calibration.csv`, `figures/0–2_*.png`.

## Adding a model

Create `ovca/models/<name>.py` with a `BaseModel` subclass that sets `name`, `param_names`,
`bounds`, `starts` and implements `rhs(t, x, theta, drug)`. For more than one state (e.g.
sensitive and resistant cells, Lotka–Volterra style), also override `initial_state`, `observe`
and `surgery`. Register it in `ovca/models/__init__.py` and add its population to
`config.POPULATION`. The generic solver integrates between treatment switches, so no exact
solution is needed.
