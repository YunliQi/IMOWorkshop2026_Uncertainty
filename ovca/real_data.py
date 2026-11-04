"""Read real patients from the anonymised workbook (one sheet per patient).

The workbook is never part of the repository: pass the path of a local, read-only copy.

    from ovca.real_data import patient_ids, read_patient, frontline_record
    path = "engineering/real_data/anonymized_data_protocol1_corrected_recurrenceday.xlsx"
    for pid in patient_ids(path):
        rec = frontline_record(read_patient(path, pid))     # PatientRecord for the tool

Columns used (days from an arbitrary origin): DAY, CA-125(U/ml), DAY NACT, DAY surgery,
DAY ACT, DAY Maintenance, DAY recurrent, recurrent, and the regimen columns NACT / ACT (with
NACT cycle / ACT cycle) to find cycles given with bevacizumab.
"""
import numpy as np
import pandas as pd

from . import config as C
from .patient import Schedule
from .record import PatientRecord

DPM = float(C.DAYS_PER_MONTH)


def patient_ids(path):
    """Sheets that hold a patient (they have a DAY column)."""
    x = pd.ExcelFile(path)
    return [s for s in x.sheet_names if "DAY" in pd.read_excel(x, sheet_name=s, nrows=1).columns]


def bev_days(d):
    """Cycle days given with bevacizumab. A row naming a regimen starts a block of cycles;
    later rows may carry notes: "plus BEV" / "BEV added" turns it on for the rest of the block,
    "BEV held" turns it off for that cycle, other notes ("Neulasta added") change nothing."""
    out = []
    for drug, days in (("NACT", "DAY NACT"), ("ACT", "DAY ACT")):
        if drug not in d or days not in d:
            continue
        on = False
        for txt, day in zip(d[drug], pd.to_numeric(d[days], errors="coerce")):
            up = "" if pd.isna(txt) else str(txt).upper()
            note = any(w in up for w in ("PLUS", "ADD", "HELD"))
            if up and not note:                       # a new regimen
                on = "BEV" in up
            elif "BEV" in up and ("PLUS" in up or "ADD" in up):
                on = True
            held = "HELD" in up and "BEV" in up
            if on and not held and np.isfinite(day):
                out.append(float(day))
    return np.sort(np.array(out, dtype=float))


def read_patient(path, sheet):
    """One sheet as a dict of readings (day, CA-125) and treatment days (all in days)."""
    d = pd.read_excel(path, sheet_name=sheet)
    num = lambda c: (pd.to_numeric(d[c], errors="coerce") if c in d          # noqa: E731
                     else pd.Series(np.nan, index=d.index))
    day, ca = num("DAY"), num("CA-125(U/ml)")
    ok = (day.notna() & ca.notna()).values
    r = pd.DataFrame({"day": day.values[ok], "ca125": ca.values[ok]}).sort_values("day")
    surg = num("DAY surgery").dropna().values
    status = d["recurrent"].dropna() if "recurrent" in d else pd.Series([], dtype=object)
    rec_day = num("DAY recurrent").dropna()
    return dict(sheet=sheet, readings=r.reset_index(drop=True),
                nact=np.sort(num("DAY NACT").dropna().values),
                act=np.sort(num("DAY ACT").dropna().values),
                maintenance=np.sort(num("DAY Maintenance").dropna().values),
                surgery=float(surg[0]) if len(surg) else np.nan,
                bev_days=bev_days(d),
                recurrent=str(status.iloc[0]) if len(status) else "",
                recurrence_day=float(rec_day.iloc[0]) if len(rec_day) else np.nan)


def frontline(p, gap_days=90):
    """Frontline schedule (months from the first reading) and the readings before second-line
    treatment or maintenance, whichever comes first. Frontline ACT = ACT starts up to the
    first gap > gap_days. Returns t, y, schedule, t_stop (months)."""
    r = p["readings"]
    if r.empty:
        raise ValueError(f"{p['sheet']}: no CA-125 readings")
    if not np.isfinite(p["surgery"]):
        raise ValueError(f"{p['sheet']}: no surgery day")
    d0 = r.day.min()
    act = p["act"]
    cut = len(act)
    for i in range(1, len(act)):
        if act[i] - act[i - 1] > gap_days:
            cut = i
            break
    act1, second = act[:cut], (act[cut] if cut < len(act) else np.inf)
    maint = p["maintenance"].min() if len(p["maintenance"]) else np.inf
    stop = min(second, maint)
    keep = (r.day <= stop).values   # a reading on the day 2nd-line starts comes before dosing
    t = (r.day.values[keep] - d0) / DPM
    y = r.ca125.values[keep]
    nact = (p["nact"][p["nact"] < p["surgery"]] - d0) / DPM
    bev = p["bev_days"]
    bev = bev[bev <= (act1[-1] if len(act1) else p["surgery"])]       # frontline cycles only
    sched = Schedule(nact, (p["surgery"] - d0) / DPM, (act1 - d0) / DPM,
                     bev_starts=(bev - d0) / DPM)
    return t, y, sched, (stop - d0) / DPM


def frontline_record(p, gap_days=90):
    """PatientRecord of the frontline window. t_recurrence: clinical recurrence (months from
    the first reading) if recurrent == "yes", inf if "no", nan otherwise (unknown)."""
    t, y, sched, t_stop = frontline(p, gap_days)
    status = p["recurrent"].strip().lower()
    if status == "yes":
        t_rec = (p["recurrence_day"] - p["readings"].day.min()) / DPM
    else:
        t_rec = np.inf if status == "no" else np.nan
    return PatientRecord(p["sheet"], t, y, sched, t_stop=t_stop, t_recurrence=t_rec,
                         meta={"recurrent": p["recurrent"]})
