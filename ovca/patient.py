"""One patient's treatment schedule and model parameters."""
from dataclasses import dataclass

import numpy as np

from . import config as C


def merge_intervals(starts, duration=C.TREATMENT_DURATION):
    """Treatment-on intervals [s, s + duration] as an (n, 2) array, overlapping ones merged."""
    out = []
    for s in np.sort(np.asarray(starts, dtype=float)):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s + duration)
        else:
            out.append([s, s + duration])
    return np.array(out, dtype=float).reshape(-1, 2)


def drug_on(t, intervals):
    """D(t): 1.0 while any treatment interval is active, else 0.0 (vectorised over t)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if intervals.size == 0:
        return np.zeros_like(t)
    return np.any((t[:, None] >= intervals[:, 0]) & (t[:, None] < intervals[:, 1]), axis=1)\
        .astype(float)


def time_on(t, t0, intervals):
    """Total treatment-on time between t0 and each t (vectorised over t)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if intervals.size == 0:
        return np.zeros_like(t)
    lo = np.maximum(intervals[:, 0], t0)[None, :]
    hi = np.minimum(intervals[:, 1][None, :], t[:, None])
    return np.clip(hi - lo, 0.0, None).sum(axis=1)


def _join(x):
    return ";".join(f"{v:.4f}" for v in x)


def _split(s):
    return np.array([float(v) for v in str(s).split(";") if v != ""])


@dataclass
class Schedule:
    """NACT cycles -> surgery -> ACT cycles (start times in months from the first NACT visit)."""
    nact_starts: np.ndarray
    t_surgery: float
    act_starts: np.ndarray
    duration: float = C.TREATMENT_DURATION

    def __post_init__(self):
        self.nact_starts = np.asarray(self.nact_starts, dtype=float)
        self.act_starts = np.asarray(self.act_starts, dtype=float)
        self.t_surgery = float(self.t_surgery)

    @property
    def t_end(self):
        """End of frontline treatment: last ACT start + one treatment duration."""
        return float(self.act_starts[-1] + self.duration) if self.act_starts.size else self.t_surgery

    def intervals(self, phase):
        """Treatment-on intervals of the 'pre' (NACT) or 'post' (ACT) segment."""
        return merge_intervals(self.nact_starts if phase == "pre" else self.act_starts,
                               self.duration)

    def to_record(self):
        return dict(nact_starts=_join(self.nact_starts), t_surgery=self.t_surgery,
                    act_starts=_join(self.act_starts), t_end=self.t_end)

    @classmethod
    def from_record(cls, r):
        return cls(_split(r["nact_starts"]), r["t_surgery"], _split(r["act_starts"]))


@dataclass
class PatientParameters:
    """Everything needed to simulate one patient with a given model.

    theta_pre / theta_post: the model's parameters before and after surgery, e.g.
    {"gamma": 0.2, "delta": 1.4}. ca125_0: CA-125 at t = 0. q_surgery: surgery (a jump) multiplies the
    tumour state by q (fraction left).
    """
    patient_id: str
    model: str
    theta_pre: dict
    theta_post: dict
    ca125_0: float
    q_surgery: float
    schedule: Schedule

    def to_record(self):
        return dict(patient=self.patient_id, model=self.model,
                    **{f"{k}_pre": v for k, v in self.theta_pre.items()},
                    **{f"{k}_post": v for k, v in self.theta_post.items()},
                    ca125_0=self.ca125_0, q_surgery=self.q_surgery, **self.schedule.to_record())

    @classmethod
    def from_record(cls, r, param_names):
        return cls(str(r["patient"]), str(r["model"]),
                   {k: float(r[f"{k}_pre"]) for k in param_names},
                   {k: float(r[f"{k}_post"]) for k in param_names},
                   float(r["ca125_0"]), float(r["q_surgery"]), Schedule.from_record(r))
