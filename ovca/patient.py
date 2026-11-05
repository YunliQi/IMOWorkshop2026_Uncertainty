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


def union_intervals(iv, close_gaps=0.0):
    """Union of (n, 2) intervals (ends may be inf); holes shorter than close_gaps are filled."""
    iv = np.asarray(iv, dtype=float).reshape(-1, 2)
    out = []
    for a, b in iv[np.argsort(iv[:, 0])]:
        if out and a <= out[-1][1] + close_gaps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
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
    """NACT cycles -> surgery -> ACT cycles (start times in months from the first NACT visit).
    bev_starts: cycle starts that include bevacizumab (optional; only the resistance model
    uses them).
    maint_bev, parp: maintenance intervals (n, 2) in months, end inf = ongoing (optional; only
    the resistance model uses them). hrd: True (HRD), False (HRP) or None (unknown)."""
    nact_starts: np.ndarray
    t_surgery: float
    act_starts: np.ndarray
    duration: float = C.TREATMENT_DURATION
    bev_starts: np.ndarray = None
    maint_bev: np.ndarray = None
    parp: np.ndarray = None
    hrd: object = None

    def __post_init__(self):
        self.nact_starts = np.asarray(self.nact_starts, dtype=float)
        self.act_starts = np.asarray(self.act_starts, dtype=float)
        self.t_surgery = float(self.t_surgery)
        self.bev_starts = np.asarray([] if self.bev_starts is None else self.bev_starts,
                                     dtype=float)
        self.maint_bev = np.asarray([] if self.maint_bev is None else self.maint_bev,
                                    dtype=float).reshape(-1, 2)
        self.parp = np.asarray([] if self.parp is None else self.parp, dtype=float).reshape(-1, 2)

    @property
    def t_end(self):
        """End of frontline treatment: last ACT start + one treatment duration."""
        return float(self.act_starts[-1] + self.duration) if self.act_starts.size else self.t_surgery

    def intervals(self, phase):
        """Treatment-on intervals of the 'pre' (NACT) or 'post' (ACT) segment."""
        return merge_intervals(self.nact_starts if phase == "pre" else self.act_starts,
                               self.duration)

    def bev_intervals(self):
        """Bevacizumab-on intervals: one treatment duration after each BEV cycle start, and
        the maintenance BEV intervals."""
        cyc = merge_intervals(self.bev_starts, self.duration)
        return union_intervals(np.vstack([cyc, self.maint_bev])) if self.maint_bev.size else cyc

    def parp_intervals(self):
        """PARP-inhibitor-on intervals (maintenance)."""
        return union_intervals(self.parp) if self.parp.size else self.parp

    def to_record(self):
        out = dict(nact_starts=_join(self.nact_starts), t_surgery=self.t_surgery,
                   act_starts=_join(self.act_starts), t_end=self.t_end)
        if self.bev_starts.size:
            out["bev_starts"] = _join(self.bev_starts)
        if self.maint_bev.size:
            out["maint_bev"] = _join(self.maint_bev.ravel())
        if self.parp.size:
            out["parp"] = _join(self.parp.ravel())
        if self.hrd is not None:
            out["hrd"] = bool(self.hrd)
        return out

    @classmethod
    def from_record(cls, r):
        def get(k):
            v = r.get(k, "")
            return _split(v) if isinstance(v, str) else np.array([])
        hrd = r.get("hrd", None)
        hrd = None if hrd is None or (isinstance(hrd, float) and np.isnan(hrd)) else \
            str(hrd).lower() in ("true", "1", "1.0")
        return cls(_split(r["nact_starts"]), r["t_surgery"], _split(r["act_starts"]),
                   bev_starts=get("bev_starts"), maint_bev=get("maint_bev"),
                   parp=get("parp"), hrd=hrd)


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
