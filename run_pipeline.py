"""Synthetic data -> RMSE calibration -> figures.

    .venv/bin/python run_pipeline.py
    .venv/bin/python run_pipeline.py --n 100 --seed 1 --gen-model growth_death --fit-model growth_death

The generator model and the fitted model can differ (e.g. generate with a richer model and
check how well the poster model fits it).
"""
import argparse

from ovca import calibration, config as C, generator, plots
from ovca.models import MODELS, get_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_PATIENTS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--gen-model", default="growth_death", choices=sorted(MODELS))
    ap.add_argument("--fit-model", default="growth_death", choices=sorted(MODELS))
    a = ap.parse_args()
    gen, fit = get_model(a.gen_model), get_model(a.fit_model)

    print(f"== 1. generate synthetic cohort ({gen.name})")
    data, obs, truth = generator.generate_cohort(gen, a.n, a.seed)
    generator.save(data)
    print(generator.summary(obs, truth))

    print(f"\n== 2. RMSE calibration ({fit.name})")
    fits = calibration.calibrate_cohort(fit, obs, truth)
    calibration.RESULTS.mkdir(exist_ok=True)
    fits.to_csv(calibration.RESULTS / "calibration.csv", index=False)
    if gen.name == fit.name:
        print(calibration.recovery_table(fits, truth, fit.param_names).round(2)
              .to_string(index=False))

    print("\n== 3. figures")
    plots.main(gen, fit, obs, truth, fits)
