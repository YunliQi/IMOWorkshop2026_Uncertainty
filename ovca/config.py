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
# ("typical patient"); `corr` is on the log scale. Tuned with engineering/tune_population.py so
# that, after the NADIR_MIN redraw, the nadir median is ~12 U/mL and about 20% / 33% recur
# within 6 / 12 months of the end of ACT. About 60% of raw draws are redrawn by NADIR_MIN, so
# the final cohort is shifted to slower kill than these medians.
POPULATION = {
    "growth_death": dict(
        names=["gamma_pre", "delta_pre", "gamma_post", "delta_post"],
        median=[0.10, 0.95, 0.04, 0.70],
        sd_log=[0.50, 0.25, 0.55, 0.25],
        corr=np.array([
            [1.0, 0.0, 0.6, 0.0],
            [0.0, 1.0, 0.0, 0.6],
            [0.6, 0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ]),
    ),
}
CA125_0_MEDIAN, CA125_0_SD_LOG = 800.0, 0.6   # baseline CA-125 (U/mL)
Q_MEDIAN, Q_SD_LOG = 0.35, 0.6                # fraction of tumour left by surgery
Q_RANGE = (0.02, 1.0)
SURGERY_JUMP_RANGE = (0.05, 0.8)   # gate: CA-125 just after / just before surgery must be in
                                   # this range (a clear drop, but not near-total), else redraw
NADIR_MIN = 5.0              # redraw a patient whose true CA-125 falls below this (U/mL) by end of ACT
MAX_REDRAWS = 1000
