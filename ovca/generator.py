"""Synthetic data generator.

Two steps, usable on their own:
  1. sample_parameters(): typical-patient values + population spread -> one PatientParameters
     (model parameters before/after surgery, baseline CA-125, surgery fraction q, schedule).
     You can also build a PatientParameters by hand and skip this step.
  2. generate_longitudinal(): PatientParameters + model -> daily tumour burden and CA-125,
     with assay noise, surveillance and recurrence detection, plus the true recurrence time.

generate_cohort() repeats both for N patients.  Run:  .venv/bin/python -m ovca.generator
Output: data/synthetic_data.csv, one row per patient-day. Columns: tumor_burden (model value),
CA125 (with noise), time (day), treatment_applied (D(t)), then auxiliary columns: patient,
model, seeds, phase, visit (1 = clinic reading that day), noise_sd_log, model parameters,
ca125_0, q_surgery, schedule and true recurrence (these are in months).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .models import BaseModel, get_model
from .patient import PatientParameters, Schedule, drug_on

DATA = Path(__file__).resolve().parents[1] / "data"


def _gap(rng, rng_months):
    """Whole number of months, uniform on [low, high]."""
    return float(rng.integers(rng_months[0], rng_months[1] + 1))


def sample_schedule(rng):
    """NACT cycles -> surgery -> ACT cycles, all on whole months (first NACT at month 0).

    Each cycle is on for TREATMENT_DURATION, then a 0-1 month gap (GAP_MONTHS) before the next.
    Surgery comes SURGERY_GAP months after the last NACT cycle ends; the first ACT cycle starts
    ACT_START_GAP months after surgery.
    """
    d = C.TREATMENT_DURATION
    nact = [0.0]
    for _ in range(int(rng.choice(C.N_NACT_CHOICES)) - 1):
        nact.append(nact[-1] + d + _gap(rng, C.GAP_MONTHS))
    t_surg = nact[-1] + d + _gap(rng, C.SURGERY_GAP)
    act = [t_surg + _gap(rng, C.ACT_START_GAP)]
    for _ in range(C.N_ACT - 1):
        act.append(act[-1] + d + _gap(rng, C.GAP_MONTHS))
    return Schedule(nact, t_surg, act)


def pre_surgery_min(model: BaseModel, p: PatientParameters, step=0.05):
    """Lowest true CA-125 before surgery (during NACT)."""
    t = np.append(np.arange(0.0, p.schedule.t_surgery, step), p.schedule.t_surgery - 1e-9)
    return float(model.simulate_patient(t, p).min())


def surgery_jump(model: BaseModel, p: PatientParameters):
    """CA-125 just after surgery divided by CA-125 just before (the size of the jump)."""
    ts = p.schedule.t_surgery
    before, after = model.simulate_patient([ts - 1e-9, ts], p)
    return float(after / before)


def sample_parameters(rng, model: BaseModel, patient_id, population=None,
                      pre_min=C.PRE_SURGERY_MIN, jump_range=C.SURGERY_JUMP_RANGE):
    """Draw one patient's parameter set from the population (log-normal around the median).

    Two gates; a draw that fails either is redrawn:
      - NACT does not bring CA-125 down too far: true CA-125 stays >= pre_min before surgery;
      - surgery is the big drop: after/before CA-125 within jump_range.
    After surgery CA-125 may fall as low as the model takes it. Pass None to switch a gate off.
    """
    for _ in range(C.MAX_REDRAWS):
        p = _draw_parameters(rng, model, patient_id, population)
        if jump_range is not None and not (jump_range[0] <= surgery_jump(model, p)
                                           <= jump_range[1]):
            continue
        if pre_min is not None and pre_surgery_min(model, p) < pre_min:
            continue
        return p
    raise RuntimeError(f"no draw passed the gates in {C.MAX_REDRAWS} tries")


def _draw_parameters(rng, model, patient_id, population):
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
    """Clinic visits: at the start of each NACT cycle (the first one is the baseline), just
    before surgery, ~2 weeks after surgery, at the start of each ACT cycle, at the end of ACT,
    then surveillance every ~3 months."""
    s = p.schedule
    n_surv = int(C.FOLLOWUP_MONTHS // C.SURVEILLANCE_EVERY)
    surv = s.t_end + C.SURVEILLANCE_EVERY * np.arange(1, n_surv + 1) \
        + rng.uniform(-0.3, 0.3, n_surv)
    return np.concatenate([s.nact_starts, [s.t_surgery - 0.1], [s.t_surgery + 0.5],
                           s.act_starts, [s.t_end], surv])


DAILY_COLUMNS = ["tumor_burden", "CA125", "time", "treatment_applied"]
DATA_FILE = "synthetic_data.csv"


def daily_table(model: BaseModel, p: PatientParameters, rng, noise_sd=C.NOISE_SD_LOG,
                seeds=None):
    """One row per day from day 0 to the end of follow-up (end of ACT + FOLLOWUP_MONTHS).

    tumor_burden: the generating model's noise-free value; CA125: tumor_burden with
    multiplicative noise; time: day; treatment_applied: D(t), 1 while a cycle is on, else 0.
    """
    s = p.schedule
    days = np.arange(0, int(round((s.t_end + C.FOLLOWUP_MONTHS) * C.DAYS_PER_MONTH)) + 1)
    t = days / C.DAYS_PER_MONTH
    burden = model.simulate_patient(t, p)
    iv = np.vstack([s.intervals("pre"), s.intervals("post")])
    return pd.DataFrame({"tumor_burden": burden,
                         "CA125": burden * np.exp(rng.normal(0, noise_sd, days.size)),
                         "time": days,
                         "treatment_applied": drug_on(t, iv).astype(int),
                         "patient": p.patient_id, "model": p.model, **(seeds or {}),
                         "phase": np.where(t < s.t_surgery, "pre", "post"),
                         "noise_sd_log": noise_sd})


def generate_longitudinal(model: BaseModel, p: PatientParameters, rng,
                          noise_sd=C.NOISE_SD_LOG, cutoff=C.CA125_CUTOFF, seeds=None):
    """The daily data of one patient, plus its clinic readings and truth record.

    Clinic readings are the rows on visit days (flagged visit = 1), so a visit shows exactly the
    CA125 of that day. Recurrence is detected at the first surveillance reading >= cutoff; one
    confirming visit follows a month later, then clinic visits stop. The daily rows always run
    to the end of follow-up. Patient parameters, schedule and true recurrence are added as
    constant columns.
    """
    s = p.schedule
    daily = daily_table(model, p, rng, noise_sd, seeds)
    vdays = np.unique(np.clip(np.round(visit_times(p, rng) * C.DAYS_PER_MONTH).astype(int),
                              0, daily.time.iloc[-1]))
    d = daily.set_index("time")
    is_surv = vdays / C.DAYS_PER_MONTH > s.t_end + 1e-6
    hit = vdays[is_surv & (d.loc[vdays, "CA125"].values >= cutoff)]
    t_det = np.inf
    if hit.size:
        t_det = hit[0] / C.DAYS_PER_MONTH
        vdays = vdays[~is_surv | (vdays <= hit[0])]
        if hit[0] + C.DAYS_PER_MONTH in d.index:
            vdays = np.append(vdays, hit[0] + C.DAYS_PER_MONTH)
    daily["visit"] = daily.time.isin(vdays).astype(int)
    tau = recurrence_time(model, p, cutoff)
    truth = dict(**p.to_record(), **(seeds or {}), true_recurrence_abs=tau,
                 true_recurrence_rel=tau - s.t_end, t_detected=t_det)
    for k, v in truth.items():
        if k not in daily.columns:
            daily[k] = v
    return daily, readings(daily), truth


def readings(data):
    """Clinic readings (visit = 1 rows) in the shape calibration uses."""
    v = data[data.visit == 1]
    return pd.DataFrame(dict(patient=v.patient.values, day=v.time.values,
                             month=np.round(v.time.values / C.DAYS_PER_MONTH, 4),
                             phase=v.phase.values, ca125=v.CA125.values,
                             ca125_true=v.tumor_burden.values))


def truth_table(data):
    """One row per patient: parameters, schedule, seeds and true recurrence."""
    per_day = set(DAILY_COLUMNS) | {"phase", "visit", "noise_sd_log"}
    cols = [c for c in data.columns if c not in per_day]
    return data.groupby("patient", sort=False)[cols[1:]].first().reset_index()


def generate_cohort(model: BaseModel, n=C.N_PATIENTS, seed=C.SEED):
    """Returns (data, readings, truth): data has one row per patient-day.

    Each patient gets its own seed (drawn from the cohort seed), so one patient can be
    regenerated on its own with generate_patient(model, pid, patient_seed).
    """
    patient_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, n)
    days = []
    for i in range(n):
        _, (daily, _, _) = generate_patient(model, f"SYN{i + 1:02d}", int(patient_seeds[i]), seed)
        days.append(daily)
    data = pd.concat(days, ignore_index=True)
    return data, readings(data), truth_table(data)


def generate_patient(model: BaseModel, patient_id, patient_seed, cohort_seed=None):
    """Parameters and data of one patient from its own seed."""
    rng = np.random.default_rng(patient_seed)
    p = sample_parameters(rng, model, patient_id)
    seeds = dict(cohort_seed=cohort_seed, patient_seed=patient_seed)
    return p, generate_longitudinal(model, p, rng, seeds=seeds)


def save(data, folder=DATA):
    """data/synthetic_data.csv: the four core columns first, then the auxiliary ones."""
    folder.mkdir(exist_ok=True)
    cols = DAILY_COLUMNS + [c for c in data.columns if c not in DAILY_COLUMNS]
    data[cols].to_csv(folder / DATA_FILE, index=False, float_format="%.6g")


def load(folder=DATA):
    """Returns (readings, truth) rebuilt from data/synthetic_data.csv."""
    data = pd.read_csv(folder / DATA_FILE)
    return readings(data), truth_table(data)


def summary(obs, truth):
    rel = truth.true_recurrence_rel
    return (f"{len(truth)} patients, {len(obs)} CA-125 readings; true recurrence "
            f"<=6 mo {(rel <= 6).mean():.0%}, <=12 mo {(rel <= 12).mean():.0%}, "
            f"none within {C.FOLLOWUP_MONTHS} mo {np.isinf(rel).mean():.0%}")


def main(model_name="growth_death", n=C.N_PATIENTS, seed=C.SEED):
    data, obs, truth = generate_cohort(get_model(model_name), n, seed)
    save(data)
    print(summary(obs, truth))
    return obs, truth


if __name__ == "__main__":
    main()
