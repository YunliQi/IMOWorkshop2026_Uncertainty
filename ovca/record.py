"""PatientRecord: one patient's readings and treatment schedule, the input of the tool."""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import config as C
from .patient import Schedule


@dataclass
class PatientRecord:
    """Readings (months from t = 0, CA-125 in U/mL) and the frontline schedule.

    truth (synthetic data only): true parameter values (natural scale, same names as
    Posterior.natural()) and the true noise-free curve (truth_t, truth_curve).
    t_stop: last month the frontline model applies (e.g. second-line treatment starts).
    t_recurrence: observed recurrence, absolute month (clinical for real data, the true
    crossing time for synthetic data); nan = unknown, inf = none."""
    patient_id: str
    t: np.ndarray
    y: np.ndarray
    schedule: Schedule
    truth: dict = None
    truth_t: np.ndarray = None
    truth_curve: np.ndarray = None
    t_stop: float = np.inf
    t_recurrence: float = np.nan
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        t = np.asarray(self.t, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if t.shape != y.shape or t.ndim != 1:
            raise ValueError("t and y must be 1-d arrays of the same length")
        if not len(t):
            raise ValueError("a record needs at least one reading")
        if np.any(~np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("CA-125 readings must be positive numbers")
        o = np.argsort(t, kind="stable")
        self.t, self.y = t[o], y[o]

    @classmethod
    def from_arrays(cls, patient_id, t_months, ca125, nact_starts, t_surgery, act_starts,
                    **kw):
        return cls(patient_id, t_months, ca125, Schedule(nact_starts, t_surgery, act_starts),
                   **kw)

    @classmethod
    def from_synthetic(cls, data, patient):
        """From data/synthetic_data.csv (generator.save): visit rows are the readings."""
        d = data[data.patient == patient].sort_values("time")
        if d.empty:
            raise ValueError(f"no rows for patient {patient!r}")
        v = d[d.visit == 1]
        r = d.iloc[0]
        s = Schedule.from_record(r)
        t, y = v.time.values / C.DAYS_PER_MONTH, v.CA125.values
        tt = d.time.values / C.DAYS_PER_MONTH
        curve = d.tumor_burden.values
        truth = {"gamma_pre": r.gamma_pre, "delta_pre": r.delta_pre,
                 "gamma_post": r.gamma_post, "delta_post": r.delta_post,
                 "shift_gamma": np.log(r.gamma_post / r.gamma_pre),
                 "shift_delta": np.log(r.delta_post / r.delta_pre),
                 "ca125_0": r.ca125_0, "surgery_jump": r.q_surgery, "sigma": r.noise_sd_log,
                 "recurrence_rel": r.true_recurrence_rel}
        pre, post = t[t < s.t_surgery], t[t >= s.t_surgery]
        if len(pre):
            truth["level_pre"] = float(np.interp(pre[0], tt, curve))
        if len(post):
            truth["level_post"] = float(np.interp(post[0], tt, curve))
        return cls(str(patient), t, y, s, truth=truth, truth_t=tt, truth_curve=curve,
                   t_recurrence=float(r.true_recurrence_abs))

    # ---- views -------------------------------------------------------------------------
    def upto(self, t_now):
        """Readings at or before t_now."""
        keep = self.t <= t_now + 1e-9
        return replace(self, t=self.t[keep], y=self.y[keep])

    def with_reading(self, t_new, y_new):
        t_new, y_new = np.atleast_1d(t_new).astype(float), np.atleast_1d(y_new).astype(float)
        return replace(self, t=np.append(self.t, t_new), y=np.append(self.y, y_new))

    def table(self):
        s = self.schedule
        phase = np.where(self.t < s.t_surgery, "pre", "post")
        return pd.DataFrame({"month": self.t, "ca125": self.y, "phase": phase})
