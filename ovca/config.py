"""All tunable numbers in one place. Time unit: months (same as the poster code)."""
import numpy as np

SEED = 2026

# --- Data and fitting conventions (workshop spec) ---
CA125_CUTOFF = 35.0          # readings below this count as 0 and are left out of the RMSE
TREATMENT_DURATION = 1.0     # D(t) = 1 for one month after each treatment start
DAYS_PER_MONTH = 30

# --- Synthetic cohort ---
N_PATIENTS = 60
NOISE_SD_LOG = 0.15          # multiplicative assay noise: y = T * exp(N(0, 0.15^2)), about 15%
FOLLOWUP_MONTHS = 36         # surveillance after the end of ACT
SURVEILLANCE_EVERY = 3.0     # a CA-125 reading every ~3 months during surveillance
GRID_STEP = 0.01             # months, for finding when the true curve crosses the cutoff

# Population of patient parameters, per model. Each value is log-normal around the median
# ("typical patient", chosen to look like the poster fits); `corr` is on the log scale.
POPULATION = {
    "growth_death": dict(
        names=["gamma_pre", "delta_pre", "gamma_post", "delta_post"],
        median=[0.20, 1.40, 0.18, 1.10],
        sd_log=[0.70, 0.40, 0.50, 0.40],
        corr=np.array([
            [1.0, 0.0, 0.6, 0.0],
            [0.0, 1.0, 0.0, 0.6],
            [0.6, 0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ]),
    ),
}
CA125_0_MEDIAN, CA125_0_SD_LOG = 800.0, 0.9   # baseline CA-125 (U/mL)
Q_MEDIAN, Q_SD_LOG = 0.35, 0.6                # fraction of tumour left by surgery
Q_RANGE = (0.02, 1.0)
