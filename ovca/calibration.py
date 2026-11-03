"""Calibrate a model to (synthetic) CA-125 data by minimising RMSE.

Each segment (pre-surgery, post-surgery) is fitted on its own, as in the poster code:
  - T0 of the segment = its first reading;
  - readings below the cutoff (35) count as 0 and their error is left out of the RMSE;
  - parameters are bounded by model.bounds; several starting guesses, best one kept.

Run:  .venv/bin/python -m ovca.calibration     (needs data/ from the generator)
Output: results/calibration.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from .generator import load
from .models import BaseModel, get_model
from .patient import Schedule

RESULTS = Path(__file__).resolve().parents[1] / "results"
MIN_POINTS = 2      # usable readings (besides T0) needed to fit a segment


def usable_mask(y, cutoff=C.CA125_CUTOFF):
    """Readings that enter the RMSE: >= cutoff, and not the first one (it sets T0)."""
    m = np.asarray(y, dtype=float) >= cutoff
    if m.size:
        m[0] = False
    return m


def rmse(y_obs, y_pred, cutoff=C.CA125_CUTOFF):
    """RMSE over usable readings only; NaN if there are none."""
    m = usable_mask(y_obs, cutoff)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean((np.asarray(y_pred)[m] - np.asarray(y_obs)[m]) ** 2)))


def segment_data(obs_p, schedule: Schedule, phase):
    """(t, y, treatment intervals) of one patient's 'pre' or 'post' segment."""
    o = obs_p[obs_p.phase == phase].sort_values("month")
    return o.month.values, o.ca125.values, schedule.intervals(phase)


def fit_segment(model: BaseModel, t, y, intervals, cutoff=C.CA125_CUTOFF):
    """Best-RMSE parameters for one segment. Returns a dict with theta, rmse, n_used, flags."""
    m = usable_mask(y, cutoff)
    out = dict(n_used=int(m.sum()), rmse=np.nan, at_bound=False, fitted=False,
               **{k: np.nan for k in model.param_names})
    if len(t) == 0 or m.sum() < MIN_POINTS:
        return out
    t0, T0 = t[0], y[0]
    lo, hi = model.bounds_arrays()

    def resid(x):
        return model.simulate_segment(t[m], t0, T0, model.theta(x), intervals) - y[m]

    best = None
    for x0 in model.starts:
        r = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    theta = model.theta(best.x)
    out.update(theta, fitted=True,
               rmse=rmse(y, model.simulate_segment(t, t0, T0, theta, intervals), cutoff),
               at_bound=bool(np.any(np.isclose(best.x, lo, rtol=1e-3) |
                                    np.isclose(best.x, hi, rtol=1e-3))))
    return out


def calibrate_patient(model: BaseModel, obs_p, schedule: Schedule):
    row = {}
    for phase in ("pre", "post"):
        f = fit_segment(model, *segment_data(obs_p, schedule, phase))
        row.update({f"{k}_{phase}": v for k, v in f.items()})
    return row


def calibrate_cohort(model: BaseModel, obs, truth):
    rows = []
    for _, r in truth.iterrows():
        rows.append(dict(patient=r.patient, model=model.name,
                         **calibrate_patient(model, obs[obs.patient == r.patient],
                                             Schedule.from_record(r))))
    return pd.DataFrame(rows)


def recovery_table(fits, truth, param_names):
    """Fitted vs true parameters (only meaningful when generator and fitted model match)."""
    m = fits.merge(truth, on="patient", suffixes=("", "_true"))
    rows = []
    for ph in ("pre", "post"):
        for k in [*param_names, "net"]:
            if k == "net":   # delta - gamma: the part the data can identify while D is on
                if not {"gamma", "delta"} <= set(param_names):
                    continue
                fit = m[f"delta_{ph}"] - m[f"gamma_{ph}"]
                tru = m[f"delta_{ph}_true"] - m[f"gamma_{ph}_true"]
                err = np.abs(fit - tru)
            else:
                fit, tru = m[f"{k}_{ph}"], m[f"{k}_{ph}_true"]
                err = np.abs(np.log(fit / tru))
            ok = fit.notna()
            rows.append(dict(parameter=f"{k}_{ph}", n_fitted=int(ok.sum()), n=len(m),
                             corr=float(np.corrcoef(fit[ok], tru[ok])[0, 1]) if ok.sum() > 2
                             else np.nan,
                             median_abs_error=float(err[ok].median())))
    return pd.DataFrame(rows)


def main(model_name="growth_death"):
    obs, truth = load()
    model = get_model(model_name)
    fits = calibrate_cohort(model, obs, truth)
    RESULTS.mkdir(exist_ok=True)
    fits.to_csv(RESULTS / "calibration.csv", index=False)
    if set(truth.model) == {model.name}:
        print(recovery_table(fits, truth, model.param_names).round(2).to_string(index=False))
    return fits


if __name__ == "__main__":
    main()
