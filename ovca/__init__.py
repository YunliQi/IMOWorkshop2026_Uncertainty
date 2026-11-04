"""Synthetic CA-125 data, tumour models, RMSE calibration, and the forecasting tool.

Tool: Forecaster(profile).fit / step / replay  (see README, "Forecasting tool").
"""
from .forecaster import Forecaster, History, Snapshot
from .profile import ProfileError, check_profile, default_profile
from .record import PatientRecord

__all__ = ["Forecaster", "History", "Snapshot", "PatientRecord", "default_profile",
           "check_profile", "ProfileError"]
