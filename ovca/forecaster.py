"""The forecasting tool.

    from ovca import Forecaster, PatientRecord, default_profile
    fc = Forecaster(default_profile("growth_death", backend="abc"))
    snap = fc.fit(record, t_now=12)              # posterior + forecast at month 12
    snap.recurrence_interval(); snap.next_visit(); snap.report()
    snap2 = fc.step(snap, t_new=15, y_new=42)    # a new reading: refit, how much it shrank
    hist = fc.replay(record)                     # refit after every visit
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backends import get_backend
from .backends.base import Posterior
from .forecast import Forecast, make_forecast
from .models import get_model
from .problem import InferenceProblem
from .profile import check_profile, describe
from .record import PatientRecord


class Forecaster:
    def __init__(self, profile=None, verbose=False):
        self.profile = check_profile(profile, verbose=verbose)
        self.model = get_model(self.profile["model"], **self.profile["model_options"])
        self.backend = get_backend(self.profile["backend"])

    def __repr__(self):
        return describe(self.profile)

    def problem(self, record: PatientRecord, t_now=None):
        return InferenceProblem(record, t_now, self.profile, self.model)

    def fit(self, record: PatientRecord, t_now=None, seed=None, verbose=False):
        """Posterior and forecast using the readings at or before t_now (default: all)."""
        p = self.profile
        seed = p["seed"] if seed is None else seed
        t0 = time.perf_counter()
        pb = self.problem(record, t_now)
        post = self.backend.fit(pb, p[p["backend"]], seed=seed, verbose=verbose)
        f = p["forecast"]
        fc = make_forecast(post, f["horizon"], f["step"], f["n_draws"],
                           f["condition_no_recurrence"], seed=seed)
        return Snapshot(record, pb.t_now, post, fc, p, seconds=time.perf_counter() - t0,
                        seed_used=seed)

    def step(self, snap: "Snapshot", t_new, y_new, seed=None):
        """Add a reading (or several), refit from the prior with all readings, and compare."""
        rec = snap.record.with_reading(t_new, y_new)
        t_now = max(float(np.max(t_new)), snap.t_now)
        new = self.fit(rec, t_now, seed=snap.seed_used + 1 if seed is None else seed)
        new.previous = snap
        new.new_reading = (float(np.max(t_new)), float(np.atleast_1d(y_new)[-1]))
        return new

    def replay(self, record: PatientRecord, start=2, t_stop=None, verbose=True):
        """Refit after every visit, from the visit number `start` on, using the visits as
        they happened. Returns a History."""
        t_stop = record.t_stop if t_stop is None else t_stop
        times = [t for t in record.t if t <= t_stop + 1e-9][start - 1:]
        snaps, prev = [], None
        for k, tk in enumerate(times):
            s = self.fit(record, tk, seed=self.profile["seed"] + k)
            if prev is not None:
                s.previous = prev
                s.new_reading = (tk, float(record.y[record.t <= tk + 1e-9][-1]))
            snaps.append(s)
            prev = s
            if verbose:
                lo, md, hi = s.recurrence_interval(cap=s.followup + 1)
                print(f"  {record.patient_id} visit {k + start:2d} (month {tk:5.2f}): "
                      f"recurrence 80% [{_fmt(lo, s.followup)}, {_fmt(hi, s.followup)}] mo, "
                      f"{s.seconds:.1f} s")
        return History(record, snaps, self.profile)


def _fmt(v, cap):
    if not np.isfinite(v):
        return "n/a"
    return f"> {cap:g}" if v > cap else f"{v:.1f}"


@dataclass
class Snapshot:
    """Everything known about one patient at t_now."""
    record: PatientRecord
    t_now: float
    posterior: Posterior
    forecast: Forecast
    profile: dict
    seconds: float = 0.0
    previous: "Snapshot" = None
    new_reading: tuple = None
    seed_used: int = 0

    @property
    def followup(self):
        return self.profile["forecast"]["horizon"]

    # ---- forwarding --------------------------------------------------------------------
    def band(self, levels=(0.1, 0.5, 0.9), noisy=False):
        return self.forecast.band(levels, noisy)

    def recurrence_interval(self, level=0.8, relative=True, cap=None):
        return self.forecast.recurrence_interval(level, relative, cap)

    def p_recur_within(self, months_after_end):
        return self.forecast.p_recur_within(months_after_end)

    def next_visit(self, **kw):
        return self.forecast.next_visit(**{**self.profile["next_visit"], **kw})

    def summary(self, levels=(0.1, 0.5, 0.9)):
        return self.posterior.summary(levels)

    # ---- comparisons -------------------------------------------------------------------
    def shrinkage(self, previous: "Snapshot" = None, at=None, level=0.8):
        """How the uncertainty changed since `previous` (default: the snapshot this one was
        updated from): band width (high / low, tumour) at time `at` (default: the next
        suggested visit of the previous snapshot), recurrence-interval width, and how
        surprising the new reading was under the previous forecast."""
        prev = previous or self.previous
        if prev is None:
            raise ValueError("no previous snapshot to compare with")
        at = prev.next_visit() if at is None else at
        if not np.isfinite(at):                     # previous fit had no post-surgery forecast
            at = self.t_now
        cap = self.followup + 1
        b0, b1 = prev.forecast.band_at(at, level), self.forecast.band_at(at, level)
        r0 = prev.recurrence_interval(level, cap=cap)
        r1 = self.recurrence_interval(level, cap=cap)
        out = dict(t_before=prev.t_now, t_after=self.t_now, at=at,
                   band_ratio_before=b0[2] / b0[0], band_ratio_after=b1[2] / b1[0],
                   recurrence_width_before=r0[2] - r0[0],
                   recurrence_width_after=r1[2] - r1[0])
        if self.new_reading is not None:
            tn, yn = self.new_reading
            out.update(new_reading_month=tn, new_reading=yn,
                       pit=prev.forecast.pit(tn, yn), surprise=prev.forecast.surprise(tn, yn))
        return out

    def report(self, level=0.8):
        """The tool's answer at t_now, as a dict."""
        cap = self.followup + 1
        lo, md, hi = self.recurrence_interval(level, cap=cap)
        nv = self.next_visit()
        out = dict(patient=self.record.patient_id, t_now=self.t_now,
                   n_readings=len(self.posterior.problem.t), backend=self.profile["backend"],
                   recurrence_low=lo, recurrence_median=md, recurrence_high=hi,
                   recurrence_level=level, recurrence_capped_at=cap,
                   p_recur_6mo=self.p_recur_within(6), p_recur_12mo=self.p_recur_within(12),
                   next_visit=nv, band_at_next_visit=self.forecast.band_at(nv, level),
                   seconds=self.seconds)
        if self.previous is not None:
            out["shrinkage"] = self.shrinkage()
        return out

    def plot(self, ax=None, **kw):
        from . import viz
        return viz.snapshot(self, ax=ax, **kw)


@dataclass
class History:
    """Snapshots after each visit (Forecaster.replay)."""
    record: PatientRecord
    snapshots: list
    profile: dict
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, i):
        return self.snapshots[i]

    @property
    def times(self):
        return np.array([s.t_now for s in self.snapshots])

    def table(self, level=0.8):
        """One row per refit: recurrence interval, next visit, and the change since the
        previous refit (band widths at the next reading, surprise of the new reading)."""
        rows = []
        for i, s in enumerate(self.snapshots):
            cap = s.followup + 1
            lo, md, hi = s.recurrence_interval(level, cap=cap)
            r = dict(t_now=s.t_now, rec_low=lo, rec_median=md, rec_high=hi,
                     rec_width=hi - lo, next_visit=s.next_visit(), seconds=s.seconds)
            tr = self.record.t_recurrence
            if not np.isnan(tr):        # observed recurrence (months after ACT) inside?
                obs = min(tr - self.record.schedule.t_end, cap)
                r["observed_recurrence"] = obs
                r["observed_in_interval"] = (bool(lo <= obs <= hi) if np.isfinite(lo)
                                             else None)
            nxt = self.snapshots[i + 1].t_now if i + 1 < len(self.snapshots) else None
            if nxt is not None:
                b = s.forecast.band_at(nxt, level)
                r["band_ratio_next"] = b[2] / b[0]
            if s.previous is not None and s.new_reading is not None:
                r["surprise_new_reading"] = s.previous.forecast.surprise(*s.new_reading)
            rows.append(r)
        return pd.DataFrame(rows)

    def parameter_table(self, levels=(0.1, 0.25, 0.5, 0.75, 0.9)):
        """Posterior quantiles of every parameter after each refit (long format)."""
        out = []
        for s in self.snapshots:
            t = s.summary(levels)
            t.insert(0, "t_now", s.t_now)
            out.append(t)
        return pd.concat(out, ignore_index=True)

    # ---- figures (see ovca.viz) ----------------------------------------------------------
    def plot_tube(self, path=None, **kw):
        from . import viz
        return viz.tube(self, path, **kw)

    def plot_panels(self, path=None, **kw):
        from . import viz
        return viz.panels(self, path, **kw)

    def plot_parameters(self, path=None, **kw):
        from . import viz
        return viz.parameter_bands(self, path, **kw)

    def plot_parameter_space(self, path=None, **kw):
        from . import viz
        return viz.parameter_space(self, path, **kw)

    def plot_parameter_distributions(self, path=None, **kw):
        from . import viz
        return viz.parameter_distributions(self, path, **kw)
