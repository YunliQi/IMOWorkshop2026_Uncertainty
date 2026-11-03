"""Synthetic data generator.

Two steps, usable on their own:
  1. sample_parameters(): typical-patient values + population spread -> one PatientParameters
     (model parameters before/after surgery, baseline CA-125, surgery fraction q, schedule).
     You can also build a PatientParameters by hand and skip this step.
  2. generate_longitudinal(): PatientParameters + model -> CA-125 readings at clinic visits,
     with assay noise, surveillance and recurrence detection, plus the true recurrence time.

generate_cohort() repeats both for N patients.  Run:  .venv/bin/python -m ovca.generator
Outputs (data/): synthetic_ca125.csv (what a clinician would see) and synthetic_truth.csv
(true parameters and recurrence times, for checking only).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .models import BaseModel, get_model
from .patient import PatientParameters, Schedule

DATA = Path(__file__).resolve().parents[1] / "data"


def sample_schedule(rng):
    """3-4 NACT cycles ~3 weeks apart -> surgery -> 3 ACT cycles (months)."""
    n_nact = int(rng.choice([3, 4]))
    nact = 0.3 + 0.7 * np.arange(n_nact) + rng.uniform(-0.05, 0.05, n_nact)
    t_surg = nact[-1] + 0.8 + rng.uniform(0.0, 0.5)
    act = t_surg + 1.0 + rng.uniform(0, 0.3) + 0.7 * np.arange(3)
    return Schedule(nact, t_surg, act)


def sample_parameters(rng, model: BaseModel, patient_id, population=None):
    """Draw one patient's parameter set from the population (log-normal around the median)."""
    pop = population or C.POPULATION[model.name]
    sd = np.asarray(pop["sd_log"])
    v = np.exp(rng.multivariate_normal(np.log(pop["median"]), np.outer(sd, sd) * pop["corr"]))
    values = dict(zip(pop["names"], v))
    theta_pre = {k: float(values[f"{k}_pre"]) for k in model.param_names}
    theta_post = {k: float(values[f"{k}_post"]) for k in model.param_names}
    ca125_0 = float(np.exp(rng.normal(np.log(C.CA125_0_MEDIAN), C.CA125_0_SD_LOG)))
    q = float(np.clip(np.exp(rng.normal(np.log(C.Q_MEDIAN), C.Q_SD_LOG)), *C.Q_RANGE))
    return PatientParameters(patient_id, model.name, theta_pre, theta_post, ca125_0, q,
                             sample_schedule(rng))


def recurrence_time(model: BaseModel, p: PatientParameters, cutoff=C.CA125_CUTOFF,
                    followup=C.FOLLOWUP_MONTHS, step=C.GRID_STEP):
    """First time after the end of ACT that the true curve reaches the cutoff (inf if never)."""
    t_end = p.schedule.t_end
    grid = np.arange(t_end, t_end + followup + step / 2, step)
    above = model.simulate_patient(grid, p) >= cutoff
    return float(grid[np.argmax(above)]) if above.any() else np.inf


def visit_times(p: PatientParameters, rng):
    """Clinic visits: baseline, each NACT cycle, just before surgery, ~2 weeks after surgery,
    each ACT cycle, end of ACT, then surveillance every ~3 months."""
    s = p.schedule
    n_surv = int(C.FOLLOWUP_MONTHS // C.SURVEILLANCE_EVERY)
    surv = s.t_end + C.SURVEILLANCE_EVERY * np.arange(1, n_surv + 1) \
        + rng.uniform(-0.3, 0.3, n_surv)
    return np.concatenate([[0.0], s.nact_starts + 0.01, [s.t_surgery - 0.1],
                           [s.t_surgery + 0.5], s.act_starts + 0.01, [s.t_end], surv])


def generate_longitudinal(model: BaseModel, p: PatientParameters, rng,
                          noise_sd=C.NOISE_SD_LOG, cutoff=C.CA125_CUTOFF):
    """CA-125 readings for one patient, and a truth record.

    Recurrence is detected at the first surveillance reading >= cutoff; one confirming visit
    follows a month later, then follow-up stops.
    """
    s = p.schedule
    t = visit_times(p, rng)
    true = model.simulate_patient(t, p)
    noisy = true * np.exp(rng.normal(0, noise_sd, t.size))
    obs = pd.DataFrame(dict(patient=p.patient_id, month=np.round(t, 4),
                            phase=np.where(t < s.t_surgery, "pre", "post"),
                            ca125=noisy, ca125_true=true))
    is_surv = obs.month > s.t_end + 1e-6
    hit = obs[is_surv & (obs.ca125 >= cutoff)]
    t_det = np.inf
    if len(hit):
        t_det = float(hit.month.iloc[0])
        obs = obs[~is_surv | (obs.month <= t_det + 1e-6)]
        tc = t_det + 1.0
        Tc = float(model.simulate_patient([tc], p)[0])
        obs = pd.concat([obs, pd.DataFrame([dict(
            patient=p.patient_id, month=round(tc, 4), phase="post",
            ca125=Tc * float(np.exp(rng.normal(0, noise_sd))), ca125_true=Tc)])],
            ignore_index=True)
    tau = recurrence_time(model, p, cutoff)
    truth = dict(**p.to_record(), true_recurrence_abs=tau,
                 true_recurrence_rel=tau - s.t_end, t_detected=t_det)
    return obs.reset_index(drop=True), truth


def generate_cohort(model: BaseModel, n=C.N_PATIENTS, seed=C.SEED):
    rng = np.random.default_rng(seed)
    all_obs, truths = [], []
    for i in range(n):
        p = sample_parameters(rng, model, f"SYN{i + 1:02d}")
        obs, truth = generate_longitudinal(model, p, rng)
        all_obs.append(obs)
        truths.append(truth)
    return pd.concat(all_obs, ignore_index=True), pd.DataFrame(truths)


def save(obs, truth, folder=DATA):
    folder.mkdir(exist_ok=True)
    obs[["patient", "month", "phase", "ca125"]].to_csv(folder / "synthetic_ca125.csv", index=False)
    obs.to_csv(folder / "synthetic_ca125_with_truth.csv", index=False)
    truth.to_csv(folder / "synthetic_truth.csv", index=False)


def load(folder=DATA):
    obs = pd.read_csv(folder / "synthetic_ca125_with_truth.csv")
    truth = pd.read_csv(folder / "synthetic_truth.csv")
    return obs, truth


def summary(obs, truth):
    rel = truth.true_recurrence_rel
    return (f"{len(truth)} patients, {len(obs)} CA-125 readings; true recurrence "
            f"<=6 mo {(rel <= 6).mean():.0%}, <=12 mo {(rel <= 12).mean():.0%}, "
            f"none within {C.FOLLOWUP_MONTHS} mo {np.isinf(rel).mean():.0%}")


def main(model_name="growth_death", n=C.N_PATIENTS, seed=C.SEED):
    obs, truth = generate_cohort(get_model(model_name), n, seed)
    save(obs, truth)
    print(summary(obs, truth))
    return obs, truth


if __name__ == "__main__":
    main()
