"""Read real patients from the anonymised workbook (one sheet per patient).

The workbook is never part of the repository: pass the path of a local, read-only copy.

    from ovca.real_data import patient_ids, read_patient, frontline_record
    path = "engineering/real_data/anonymized_data_protocol1_corrected_recurrenceday.xlsx"
    for pid in patient_ids(path):
        rec = frontline_record(read_patient(path, pid))     # PatientRecord for the tool

Columns used (days from an arbitrary origin): DAY, CA-125(U/ml), DAY NACT, DAY surgery,
DAY ACT, DAY Maintenance, DAY recurrent, recurrent, HR status, and the regimen columns NACT /
ACT / Maintenance (with NACT cycle / ACT cycle / Maintenance cycle) to find cycles given with
bevacizumab and the maintenance drugs.
"""
import numpy as np
import pandas as pd

from . import config as C
from .patient import Schedule, union_intervals
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


PARP_NAMES = ("OLAPARIB", "NIRAPARIB", "RUCAPARIB")
OTHER_NAMES = ("DOXIL", "PACLITAXEL", "CARBOPLATIN", "PEMBRO", "MIRVETUXIMAB", "ELAHERE",
               "GEMCITABINE", "TOPOTECAN")
PERIOD_DAYS = {"parp": 30.0, "bev": 21.0}      # one row = one month of tablets / one BEV cycle


def maintenance_blocks(d, gap_days=90):
    """Frontline maintenance from the Maintenance columns: {"parp": (n, 2), "bev": (n, 2)}
    on-intervals in days (end inf = ongoing), and the day it stopped being frontline
    maintenance (inf if never): a non-maintenance drug (e.g. Doxil) or a gap > gap_days.

    A Maintenance entry naming PARP inhibitors and / or BEV starts a block (drug names in the
    Maintenance cycle notes do not); each dated row of the block is one dose period (30 days
    of tablets, or a 21-day BEV cycle). A row with "held" holds the named drug (or every drug)
    for that period. "ongoing" anywhere in the block keeps the last drug on (end inf). Rows
    before the first named drug are ignored."""
    empty = {"parp": np.zeros((0, 2)), "bev": np.zeros((0, 2))}
    if "DAY Maintenance" not in d:
        return empty, np.inf
    days = pd.to_numeric(d["DAY Maintenance"], errors="coerce").values
    col = lambda c: ["" if pd.isna(v) else str(v).upper()                 # noqa: E731
                     for v in (d[c] if c in d else [np.nan] * len(d))]
    drugs = lambda t: ({"parp"} if any(w in t for w in PARP_NAMES) else set()) | \
        ({"bev"} if ("BEV" in t or "BEVACIZUMAB" in t) else set())         # noqa: E731
    on, ongoing, last, stop = set(), False, -np.inf, np.inf
    starts = {"parp": [], "bev": []}
    for day, reg, note in zip(days, col("Maintenance"), col("Maintenance cycle")):
        t = reg + " " + note
        names = drugs(t)
        held = "HELD" in t and "RESTART" not in t
        if any(w in reg for w in OTHER_NAMES) and not held:
            stop = day if np.isfinite(day) else last
            break
        if drugs(reg) and not held and "DECLINED" not in reg:
            on = drugs(reg)
        if "ONGOING" in t:
            ongoing = True
        if not np.isfinite(day) or not on:
            continue
        if np.isfinite(last) and day - last > gap_days:
            stop = day
            break
        hold = (names or on) if held else set()
        for k in on - hold:
            starts[k].append(day)
        last = day
    out = {}
    for k, st in starts.items():
        st = np.asarray(st, float)
        iv = np.column_stack([st, st + PERIOD_DAYS[k]]) if st.size else np.zeros((0, 2))
        iv = union_intervals(iv, close_gaps=14.0)
        if ongoing and iv.size and k in on:
            iv[-1, 1] = np.inf
        out[k] = iv
    return out, stop


def hr_status(d):
    """True (HRD), False (HRP) or None (unknown) from the first HR status entry."""
    v = d["HR status"].dropna() if "HR status" in d else pd.Series([], dtype=object)
    v = str(v.iloc[0]).strip().upper() if len(v) else ""
    return True if v == "HRD" else False if v == "HRP" else None


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
                maint=maintenance_blocks(d),
                hrd=hr_status(d),
                recurrent=str(status.iloc[0]) if len(status) else "",
                recurrence_day=float(rec_day.iloc[0]) if len(rec_day) else np.nan)


def frontline(p, gap_days=90, cut="maintenance"):
    """Frontline schedule (months from the first reading) and the readings up to the cut.
    cut="maintenance": before second-line treatment or maintenance, whichever comes first;
    cut="second_line": through frontline maintenance, before second-line treatment (the next
    ACT block, or a non-maintenance drug in the maintenance columns). Frontline ACT = ACT
    starts up to the first gap > gap_days. The schedule carries the frontline maintenance
    (PARP, BEV) and HR status either way. Returns t, y, schedule, t_stop (months)."""
    if cut not in ("maintenance", "second_line"):
        raise ValueError(f"cut must be 'maintenance' or 'second_line', not {cut!r}")
    r = p["readings"]
    if r.empty:
        raise ValueError(f"{p['sheet']}: no CA-125 readings")
    if not np.isfinite(p["surgery"]):
        raise ValueError(f"{p['sheet']}: no surgery day")
    d0 = r.day.min()
    act = p["act"]
    n1 = len(act)
    for i in range(1, len(act)):
        if act[i] - act[i - 1] > gap_days:
            n1 = i
            break
    act1, second = act[:n1], (act[n1] if n1 < len(act) else np.inf)
    maint = p["maintenance"].min() if len(p["maintenance"]) else np.inf
    blocks, maint_stop = p.get("maint", ({"parp": np.zeros((0, 2)), "bev": np.zeros((0, 2))},
                                         np.inf))
    stop = min(second, maint) if cut == "maintenance" else min(second, maint_stop)
    keep = (r.day <= stop).values   # a reading on the day 2nd-line starts comes before dosing
    t = (r.day.values[keep] - d0) / DPM
    y = r.ca125.values[keep]
    nact = (p["nact"][p["nact"] < p["surgery"]] - d0) / DPM
    bev = p["bev_days"]
    bev = bev[bev <= (act1[-1] if len(act1) else p["surgery"])]       # frontline cycles only
    # maintenance that starts after the last frontline cycle starts and before second-line
    # treatment (and ends there at the latest)
    ms, m0 = min(second, maint_stop), (act1[-1] if len(act1) else p["surgery"])
    keep_iv = lambda iv: (np.column_stack([iv[:, 0], np.minimum(iv[:, 1], ms)])  # noqa: E731
                          [(iv[:, 0] < ms) & (iv[:, 0] >= m0)] if iv.size else iv)
    sched = Schedule(nact, (p["surgery"] - d0) / DPM, (act1 - d0) / DPM,
                     bev_starts=(bev - d0) / DPM,
                     maint_bev=(keep_iv(blocks["bev"]) - d0) / DPM,
                     parp=(keep_iv(blocks["parp"]) - d0) / DPM, hrd=p.get("hrd"))
    return t, y, sched, (stop - d0) / DPM


def frontline_record(p, gap_days=90, cut="maintenance"):
    """PatientRecord of the frontline window. t_recurrence: clinical recurrence (months from
    the first reading) if recurrent == "yes", inf if "no", nan otherwise (unknown)."""
    t, y, sched, t_stop = frontline(p, gap_days, cut)
    status = p["recurrent"].strip().lower()
    if status == "yes":
        t_rec = (p["recurrence_day"] - p["readings"].day.min()) / DPM
    else:
        t_rec = np.inf if status == "no" else np.nan
    return PatientRecord(p["sheet"], t, y, sched, t_stop=t_stop, t_recurrence=t_rec,
                         meta={"recurrent": p["recurrent"]})
