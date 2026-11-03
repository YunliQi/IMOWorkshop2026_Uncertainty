"""All tunable numbers in one place. Time unit: months (same as the poster code)."""
import numpy as np

SEED = 2026

# --- Data and fitting conventions (workshop spec) ---
CA125_CUTOFF = 35.0          # readings below this count as 0 and are left out of the RMSE
TREATMENT_DURATION = 1.0     # D(t) = 1 for one month after each treatment start
DAYS_PER_MONTH = 30

# --- Treatment schedule (months) ---
# Each cycle is on for TREATMENT_DURATION; after it ends there is a gap of a whole number of
# months drawn uniformly from GAP_MONTHS before the next event.
N_NACT_CHOICES = (3, 4)      # number of NACT cycles (equally likely)
N_ACT = 3                    # number of ACT cycles
GAP_MONTHS = (0, 1)          # gap between the end of one cycle and the start of the next (0 = back to back)
SURGERY_GAP = (1, 2)         # end of the last NACT cycle -> surgery
ACT_START_GAP = (1, 2)       # surgery -> first ACT cycle

# --- Synthetic cohort ---
N_PATIENTS = 60
NOISE_SD_LOG = 0.15          # multiplicative assay noise: y = T * exp(N(0, 0.15^2)), about 15%
FOLLOWUP_MONTHS = 36         # surveillance after the end of ACT
SURVEILLANCE_EVERY = 3.0     # a CA-125 reading every ~3 months during surveillance
GRID_STEP = 0.01             # months, for finding when the true curve crosses the cutoff

# Population of patient parameters, per model. Each value is log-normal around the median
# ("typical patient"); `corr` is on the log scale. Tuned with engineering/tune_population.py
# (both gates on): NACT brings CA-125 only partly down (true CA-125 just before surgery about
# 40 / 80 / 190 at the 10th / 50th / 90th percentile), surgery is the big drop, ACT pushes it
# lower; about 16% / 32% recur within 6 / 12 months after ACT, ~17% not within 36 months,
# ~3% persistent. About 40% of raw draws are redrawn by the gates.
POPULATION = {
    "growth_death": dict(
        names=["gamma_pre", "delta_pre", "gamma_post", "delta_post"],
        median=[0.10, 0.90, 0.10, 0.50],
        sd_log=[0.40, 0.20, 0.45, 0.25],
        corr=np.array([
            [1.0, 0.0, 0.6, 0.0],
            [0.0, 1.0, 0.0, 0.6],
            [0.6, 0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ]),
    ),
}
CA125_0_MEDIAN, CA125_0_SD_LOG = 800.0, 0.45  # baseline CA-125 (U/mL)
Q_MEDIAN, Q_SD_LOG = 0.20, 0.6                # fraction of tumour left by surgery
Q_RANGE = (0.001, 1.0)
SURGERY_JUMP_RANGE = (0.01, 0.3)   # gate: CA-125 just after / just before surgery (surgery is
                                   # the big drop, but not to nothing), else redraw
PRE_SURGERY_MIN = 30.0       # gate: NACT alone does not push true CA-125 below this (U/mL);
                             # after surgery it may go as low as the model takes it
MAX_REDRAWS = 1000
