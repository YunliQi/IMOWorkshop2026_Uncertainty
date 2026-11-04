"""ovca.real_data on a small made-up workbook with the real layout (no real data in the repo)."""
import numpy as np
import pandas as pd
import pytest

from ovca.real_data import frontline, frontline_record, patient_ids, read_patient


def sheet(rows):
    cols = ["DAY", "CA-125(U/ml)", "NACT", "NACT cycle", "DAY NACT", "DAY surgery", "ACT",
            "ACT cycle", "DAY ACT", "Maintenance", "DAY Maintenance", "recurrent",
            "DAY recurrent"]
    return pd.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows], columns=cols)


@pytest.fixture
def book(tmp_path):
    # P1: NACT with BEV (3 cycles), ACT without BEV, then a note-only row, maintenance, recurrence
    p1 = sheet([
        dict(DAY=0, **{"CA-125(U/ml)": 800, "NACT": "3C Carbo/Taxol/BEV", "NACT cycle": 1,
                       "DAY NACT": 0, "DAY surgery": 70, "ACT": "3C Carbo/Taxol",
                       "ACT cycle": 4, "DAY ACT": 100, "Maintenance": "Olaparib",
                       "DAY Maintenance": 400, "recurrent": "yes", "DAY recurrent": 520}),
        dict(DAY=21, **{"CA-125(U/ml)": 300, "NACT cycle": 2, "DAY NACT": 21,
                        "ACT cycle": 5, "DAY ACT": 121}),
        dict(DAY=42, **{"CA-125(U/ml)": 120, "NACT cycle": 3, "DAY NACT": 42,
                        "ACT": "Neulasta added", "ACT cycle": 6, "DAY ACT": 142}),
        dict(DAY=100, **{"CA-125(U/ml)": 40}),
        dict(DAY=200, **{"CA-125(U/ml)": 12}),
        dict(DAY=400, **{"CA-125(U/ml)": 10}),       # maintenance day: kept (before dosing)
        dict(DAY=450, **{"CA-125(U/ml)": 15}),       # during maintenance: dropped
    ])
    # P2: no NACT; ACT block with BEV added at cycle 2 and held at cycle 3; second line later
    p2 = sheet([
        dict(DAY=0, **{"CA-125(U/ml)": 90, "NACT": "no", "DAY surgery": -20,
                       "ACT": "6C Carbo/Taxol", "ACT cycle": 1, "DAY ACT": 10,
                       "recurrent": "no"}),
        dict(DAY=30, **{"CA-125(U/ml)": 30, "ACT": "plus BEV", "ACT cycle": 2,
                        "DAY ACT": 31}),
        dict(DAY=60, **{"CA-125(U/ml)": 20, "ACT": "BEV held", "ACT cycle": 3,
                        "DAY ACT": 52}),
        dict(DAY=90, **{"CA-125(U/ml)": 18, "ACT cycle": 4, "DAY ACT": 73}),
        dict(DAY=300, **{"CA-125(U/ml)": 60, "ACT": "Doxil/BEV", "ACT cycle": 1,
                         "DAY ACT": 300}),       # second line (gap > 90 days)
    ])
    path = tmp_path / "fake.xlsx"
    with pd.ExcelWriter(path) as w:
        p1.to_excel(w, sheet_name="P1", index=False)
        p2.to_excel(w, sheet_name="P2", index=False)
        pd.DataFrame({"note": ["not a patient"]}).to_excel(w, sheet_name="README", index=False)
    return path


def test_patient_ids_skip_sheets_without_day(book):
    assert patient_ids(book) == ["P1", "P2"]


def test_read_patient_days_and_bev(book):
    p = read_patient(book, "P1")
    assert list(p["readings"].day) == [0, 21, 42, 100, 200, 400, 450]
    assert p["surgery"] == 70 and p["recurrent"] == "yes" and p["recurrence_day"] == 520
    # NACT block has BEV; the ACT block does not; "Neulasta added" is only a note
    assert list(p["bev_days"]) == [0, 21, 42]
    q = read_patient(book, "P2")
    # off at cycle 1, "plus BEV" from cycle 2, held at cycle 3, on again at 4; Doxil/BEV block
    assert list(q["bev_days"]) == [31, 73, 300]


def test_frontline_stops_at_maintenance_or_second_line(book):
    t, y, s, t_stop = frontline(read_patient(book, "P1"))
    assert t_stop == pytest.approx(400 / 30)
    assert t[-1] == pytest.approx(400 / 30) and len(t) == 6
    assert np.allclose(s.nact_starts, np.array([0, 21, 42]) / 30)
    assert s.t_surgery == pytest.approx(70 / 30)
    assert np.allclose(s.bev_starts, np.array([0, 21, 42]) / 30)
    t, y, s, t_stop = frontline(read_patient(book, "P2"))
    assert t_stop == pytest.approx(300 / 30)                       # second line starts
    assert np.allclose(s.act_starts, np.array([10, 31, 52, 73]) / 30)
    assert np.allclose(s.bev_starts, np.array([31, 73]) / 30)      # frontline BEV only
    assert s.nact_starts.size == 0


def test_frontline_record_recurrence_status(book):
    r1 = frontline_record(read_patient(book, "P1"))
    assert r1.t_recurrence == pytest.approx(520 / 30) and r1.patient_id == "P1"
    assert frontline_record(read_patient(book, "P2")).t_recurrence == np.inf
    p = read_patient(book, "P2")
    p["recurrent"] = "suggestive - based on imaging"
    assert np.isnan(frontline_record(p).t_recurrence)


def test_no_surgery_is_an_error(book):
    p = read_patient(book, "P1")
    p["surgery"] = np.nan
    with pytest.raises(ValueError, match="surgery"):
        frontline(p)
