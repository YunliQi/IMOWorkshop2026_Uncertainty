"""Profiles: every setting of a forecast in one dict, with defaults and a checker.

    profile = default_profile("growth_death", backend="abc")
    profile["priors"]["gamma_pre"] = {"dist": "lognormal", "median": 0.05, "sd_log": 0.5}
    profile["priors"]["sigma"] = 0.15              # a number = fixed
    profile = check_profile(profile)               # prints the resolved priors, raises on errors

A profile may be partial: anything left out takes its default. See default_profile() for every
key and PARAMETERS below for the sampled quantities.
"""
import copy
import difflib
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .models import MODELS, get_model
from .priors import DIST_ARGS, GAUSSIAN_U, POSITIVE, Dist, normalise

BACKENDS = ("abc", "mcmc")

# What each prior entry means (for the check table and the docs). {p} = a model parameter.
PARAMETERS = {
    "{p}_pre": "model parameter {p} before surgery",
    "shift_{p}": "surgery effect on {p}: log {p}_post = log {p}_pre + shift_{p}",
    "level_pre": "tumour burden at the first pre-surgery reading (U/mL)",
    "level_post": "tumour burden at the first post-surgery reading (U/mL)",
    "sigma": "observation error: sd of log CA-125 around the model (a number = fixed)",
    "b0": "background CA-125 not made by the tumour (U/mL; models with background only)",
}

# Settings (other than priors) that differ by model; anything not named takes the default.
MODEL_SETTINGS = {
    "resistance": {
        "data": {"rule": "cutoff", "cutoff": 50.0},   # recurrence: model CA-125 reaches 50
        "forecast": {"horizon": 4000.0 / C.DAYS_PER_MONTH},   # look-ahead 4000 days
    },
}

# Default priors of the model parameters, per model (natural scale, before surgery).
MODEL_PRIORS = {
    "growth_death": {
        "gamma_pre": {"dist": "lognormal", "median": 0.10, "sd_log": 0.7},
        "delta_pre": {"dist": "lognormal", "median": 0.80, "sd_log": 0.5},
    },
    "poptk": {   # growth per month; kill = log10 kill per dose (median cell); het = sd of log10
        "growth_pre": {"dist": "lognormal", "median": 0.15, "sd_log": 0.7},   # sensitivity
        "kill_pre": {"dist": "lognormal", "median": 0.30, "sd_log": 0.7},
        "het_pre": {"dist": "lognormal", "median": 0.15, "sd_log": 0.8},
    },
    "resistance": {   # per month (team table): g growth 0.008/day (doubling 2.9 months),
        "g_pre": {"dist": "lognormal", "median": 0.24, "sd_log": 0.7},    # k kill on S during a
        "k_pre": {"dist": "lognormal", "median": 12.0, "sd_log": 0.7},    # pulse 0.4/day, rho =
        "rho_pre": {"dist": "lognormal", "median": 0.0055, "sd_log": 1.5},  # R0/S0 (R0 share
        "b0": {"dist": "truncated_lognormal", "median": 12.0, "sd_log": 0.6,  # e^-5.211); b0
               "low": 2.0, "high": 35.0},                                 # median 12 in 2-35
    },
}
# background CA-125 (normal range 10-35; many nadirs in our data are below 15)
B0_PRIOR = {"dist": "lognormal", "median": 8.0, "sd_log": 0.6}
COMMON_PRIORS = {
    "level_pre": {"dist": "lognormal", "median": "first_reading", "sd_log": 1.0},
    "level_post": {"dist": "lognormal", "median": "first_reading", "sd_log": 1.0},
    "sigma": {"dist": "lognormal", "median": 0.15, "sd_log": 0.5},
}
SHIFT_PRIOR = {"dist": "normal", "mean": 0.0, "sd": 0.6}


def _model_priors(model):
    if model.name in MODEL_PRIORS:
        return copy.deepcopy(MODEL_PRIORS[model.name])
    start = model.starts[0] if model.starts else [1.0] * len(model.param_names)
    return {f"{p}_pre": {"dist": "lognormal", "median": float(v), "sd_log": 1.0}
            for p, v in zip(model.param_names, start)}


def default_profile(model="growth_death", backend="abc"):
    """A complete profile with every key at its default value."""
    m = get_model(model) if isinstance(model, str) else model
    if m.default_shift_params is not None:
        shift_params = list(m.default_shift_params)
    else:
        shift_params = ([p for p in m.param_names if p in C.ABC_SHIFT_PARAMS]
                        or list(m.param_names))
    priors = _model_priors(m)
    priors.update({f"shift_{p}": dict(SHIFT_PRIOR) for p in shift_params})
    priors.update(copy.deepcopy(COMMON_PRIORS))
    if getattr(m, "background", False):
        priors.setdefault("b0", dict(B0_PRIOR))
    out = {
        "backend": backend,
        "model": m.name,
        "model_options": {},             # passed to the model, e.g. {"bev": "brake"}
        "priors": priors,
        "correlations": {},              # {"gamma_pre:delta_pre": 0.3}, lognormal/normal only
        "surgery": {
            "shift_params": shift_params,            # which parameters surgery may change
        },                                           # (no post-surgery forecast before the
                                                     # first post-surgery reading)
        "initial_level": {
            "anchor": "first_reading",   # level parameters at each segment's first reading;
        },                               # "segment_start": at t = 0 and at surgery
        "data": {
            "censor_below": C.ABC_CENSOR_BELOW,      # None: readings used as they are
            "decay_tau": None,                       # months; weight exp(-(t_now - t_i)/tau)
            "rule": "cutoff",            # recurrence: "cutoff" = model CA-125 reaches cutoff;
                                         # "rise" = reaches rise_factor x its nadir since surgery
            "cutoff": C.CA125_CUTOFF,
            "rise_factor": 2.0,
            "rule_step": 0.05,           # months; grid for the "rise" rule
        },
        "abc": {
            "acceptance": "likelihood",  # kernel = observation-error density (sigma estimable);
                                         # "threshold": accept if distance < epsilon
            "n_particles": C.ABC_N_PARTICLES,
            "alpha": C.ABC_ALPHA,        # each generation keeps this fraction (ESS / epsilon)
            "max_generations": 60,
            "n_moves": 10,               # likelihood mode: MH moves per generation
            "perturbation": {"type": "gaussian", "scale": "auto", "adapt": True},
            "min_accept_rate": C.ABC_MIN_ACCEPT_RATE,   # threshold mode: stop below this
            "max_sims_per_gen": C.ABC_MAX_SIMS_PER_GEN,  # threshold mode
            "batch": C.ABC_BATCH,                        # threshold mode
        },
        "mcmc": {
            "sampler": "nuts", "draws": 1000, "tune": 1000, "chains": 4,
            "target_accept": 0.9, "cores": None,
        },
        "forecast": {
            "horizon": float(C.FOLLOWUP_MONTHS),     # months after max(t_now, end of ACT)
            "step": 0.05,
            "n_draws": 2000,
            "condition_no_recurrence": True,
        },
        "next_visit": {"rule": "risk", "p": 0.10, "width": 1.0, "min_gap": 1.0, "max_gap": 6.0},
        "seed": 0,
    }
    for sec, vals in MODEL_SETTINGS.get(m.name, {}).items():
        out[sec].update(copy.deepcopy(vals))
    return out


class ProfileError(ValueError):
    """Raised by check_profile with every problem found."""


def _suggest(key, options):
    m = difflib.get_close_matches(str(key), list(options), n=1, cutoff=0.6)
    return f" (did you mean {m[0]!r}?)" if m else ""


def _merge(defaults, given, path, errors, sources, allow_new=False):
    """defaults updated by given; unknown keys are errors (with a suggestion)."""
    out = copy.deepcopy(defaults)
    for k, v in given.items():
        if k not in defaults and not allow_new:
            errors.append(f"{path}{k}: unknown key{_suggest(k, defaults)}")
            continue
        if isinstance(defaults.get(k), dict) and defaults[k] and isinstance(v, dict):
            out[k] = _merge(defaults[k], v, f"{path}{k}.", errors, sources)
        else:
            out[k] = copy.deepcopy(v)
            if not _same(v, defaults.get(k)):
                sources[f"{path}{k}"] = "you"
    return out


def _same(a, b):
    try:
        return bool(a == b) if not isinstance(a, np.ndarray) else np.array_equal(a, b)
    except Exception:
        return False


def parameter_names(model, shift_params):
    return ([f"{p}_pre" for p in model.param_names] + [f"shift_{p}" for p in shift_params]
            + ["level_pre", "level_post", "sigma"])


def _num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) \
        and np.isfinite(x)


def check_prior(name, spec, errors, positive, allow_first_reading):
    """Validate one prior entry; returns a Dist (with 'first_reading' kept) or None."""
    try:
        spec = normalise(spec)
    except (TypeError, ValueError):
        errors.append(f"priors.{name}: must be a number (fixed) or a dict like "
                      f"{{'dist': 'lognormal', 'median': 0.1, 'sd_log': 0.5}}")
        return None
    kind = spec.pop("dist", None)
    if kind not in DIST_ARGS:
        errors.append(f"priors.{name}: dist {kind!r} not supported{_suggest(kind, DIST_ARGS)}; "
                      f"use one of {sorted(DIST_ARGS)}")
        return None
    need = DIST_ARGS[kind]
    for k in spec:
        if k not in need:
            errors.append(f"priors.{name}: {kind} takes {list(need)}, not {k!r}"
                          f"{_suggest(k, need)}")
    missing = [k for k in need if k not in spec]
    if missing:
        errors.append(f"priors.{name}: {kind} needs {missing}")
        return None
    bad = False
    for k in need:
        v = spec[k]
        if k == "median" and v == "first_reading":
            if not allow_first_reading:
                errors.append(f"priors.{name}: median 'first_reading' only works for "
                              f"level_pre / level_post")
                bad = True
            continue
        if not _num(v):
            errors.append(f"priors.{name}.{k}: must be a finite number, got {v!r}")
            bad = True
    if bad:
        return None
    a = spec
    if kind in ("lognormal", "truncated_lognormal") and a["median"] != "first_reading" \
            and a["median"] <= 0:
        errors.append(f"priors.{name}.median: must be > 0")
    for k in ("sd_log", "sd"):
        if k in a and a[k] <= 0:
            errors.append(f"priors.{name}.{k}: must be > 0")
    if kind in ("uniform", "truncated_lognormal") and not a["low"] < a["high"]:
        errors.append(f"priors.{name}: {kind} needs low < high")
    if kind == "truncated_lognormal" and a["low"] <= 0:
        errors.append(f"priors.{name}.low: must be > 0")
    if positive:
        lo = {"uniform": a.get("low"), "fixed": a.get("value"), "normal": -np.inf}.get(kind, 0.0)
        if kind == "normal" or (lo is not None and lo < 0) or (kind == "fixed" and a["value"] <= 0):
            errors.append(f"priors.{name}: this parameter must be > 0, but a {kind} prior "
                          f"{'can be' if kind != 'fixed' else 'is'} <= 0; use lognormal, "
                          f"halfnormal, uniform with low >= 0, or a positive number")
    return Dist(name, kind, {k: (v if v == "first_reading" else float(v)) for k, v in a.items()})


def check_profile(profile=None, verbose=True, strict_unused=False):
    """Fill in defaults, check everything, print the resolved priors.

    Raises ProfileError listing every problem. Returns the complete profile (a new dict);
    its "_sources" entry says which values you set."""
    profile = dict(profile or {})
    profile.pop("_sources", None)
    errors, notes, sources = [], [], {}

    backend = profile.get("backend", "abc")
    if backend not in BACKENDS:
        errors.append(f"backend: {backend!r} not supported{_suggest(backend, BACKENDS)}; "
                      f"use one of {list(BACKENDS)}")
        backend = "abc"
    mname = profile.get("model", "growth_death")
    if mname not in MODELS:
        raise ProfileError(f"model: unknown model {mname!r}{_suggest(mname, MODELS)}; "
                           f"registered: {sorted(MODELS)}")
    opts = profile.get("model_options") or {}
    try:
        model = get_model(mname, **opts)
    except (TypeError, ValueError) as e:
        raise ProfileError(f"model_options: {e}") from None
    base = default_profile(model, backend)

    # shift_params first: they decide which shift priors exist
    given_surgery = profile.get("surgery", {}) or {}
    sp = given_surgery.get("shift_params", base["surgery"]["shift_params"])
    if not isinstance(sp, (list, tuple)):
        errors.append("surgery.shift_params: must be a list of model parameter names")
        sp = base["surgery"]["shift_params"]
    for p in sp:
        if p not in model.param_names:
            errors.append(f"surgery.shift_params: {p!r} is not a parameter of {mname} "
                          f"{list(model.param_names)}{_suggest(p, model.param_names)}")
    sp = [p for p in sp if p in model.param_names]
    for p in sp:
        base["priors"].setdefault(f"shift_{p}", dict(SHIFT_PRIOR))
    for p in model.param_names:
        if p not in sp:
            base["priors"].pop(f"shift_{p}", None)

    known_prior_names = set(parameter_names(model, model.param_names))
    given_priors = profile.get("priors", {}) or {}
    for k in given_priors:
        if k not in base["priors"]:
            if k in known_prior_names:         # a shift of a parameter surgery does not change
                errors.append(f"priors.{k}: {k[6:]!r} is not in surgery.shift_params {sp}; "
                              f"add it there or remove this prior")
            else:
                errors.append(f"priors.{k}: not a parameter of this profile"
                              f"{_suggest(k, base['priors'])}; parameters are "
                              f"{list(base['priors'])}")
    rest = {k: v for k, v in profile.items() if k != "priors"}
    out = _merge({k: v for k, v in base.items() if k != "priors"}, rest, "", errors, sources)
    out["priors"] = copy.deepcopy(base["priors"])
    for k, v in given_priors.items():
        if k in out["priors"]:
            out["priors"][k] = copy.deepcopy(v)
            if not _same(normalise(v) if not isinstance(v, dict) else v, base["priors"][k]):
                sources[f"priors.{k}"] = "you"
    out["surgery"]["shift_params"] = list(sp)

    # priors
    dists = {}
    positive = ({f"{p}_pre" for p in model.param_names}
                | {"level_pre", "level_post", "sigma", "b0"})
    for name, spec in out["priors"].items():
        d = check_prior(name, spec, errors, name in positive, name.startswith("level_"))
        if d is not None:
            dists[name] = d
            if name.startswith("shift_") and d.kind in POSITIVE:
                errors.append(f"priors.{name}: a shift is a log ratio and can be negative; "
                              f"a {d.kind} prior only allows positive shifts (use normal)")
    for p in model.param_names:          # prior vs the model's own bounds
        d = dists.get(f"{p}_pre")
        if d is not None and d.kind == "lognormal" and p in model.bounds:
            lo, hi = model.bounds[p]
            med = d.args["median"]
            if not lo <= med <= hi:
                notes.append(f"priors.{p}_pre: median {med:g} is outside the model bounds "
                             f"[{lo:g}, {hi:g}] used by RMSE calibration")

    # correlations
    corr = out.get("correlations") or {}
    parsed = {}
    if not isinstance(corr, dict):
        errors.append("correlations: must be a dict like {'gamma_pre:delta_pre': 0.3}")
        corr = {}
    for key, r in corr.items():
        pair = tuple(key) if isinstance(key, (tuple, list)) else tuple(str(key).split(":"))
        if len(pair) != 2:
            errors.append(f"correlations.{key}: write the pair as 'a:b'")
            continue
        a, b = pair
        for n in (a, b):
            if n not in dists:
                errors.append(f"correlations.{key}: {n!r} is not a parameter"
                              f"{_suggest(n, out['priors'])}")
            elif dists[n].kind not in GAUSSIAN_U:
                errors.append(f"correlations.{key}: {n!r} has a {dists[n].kind} prior; "
                              f"correlations need lognormal or normal priors")
        if not _num(r) or not -1 < r < 1:
            errors.append(f"correlations.{key}: must be a number in (-1, 1)")
        else:
            parsed[(a, b)] = float(r)
    if parsed and not errors:
        names = sorted({n for ab in parsed for n in ab})
        R = np.eye(len(names))
        for (a, b), r in parsed.items():
            R[names.index(a), names.index(b)] = R[names.index(b), names.index(a)] = r
        if np.linalg.eigvalsh(R).min() <= 0:
            errors.append("correlations: the correlation matrix is not positive definite")

    # surgery / levels / data
    anchor = out["initial_level"]["anchor"]
    if anchor not in ("first_reading", "segment_start"):
        errors.append(f"initial_level.anchor: {anchor!r}; use 'first_reading' or "
                      f"'segment_start'")
    elif anchor == "first_reading" and not getattr(model, "scale_invariant", False):
        errors.append(f"initial_level.anchor: 'first_reading' needs a model whose solution "
                      f"scales with its start value; {mname} is not marked scale_invariant, "
                      f"use 'segment_start'")
    d = out["data"]
    for k in ("censor_below", "decay_tau"):
        if d[k] is not None and (not _num(d[k]) or d[k] <= 0):
            errors.append(f"data.{k}: must be None or a number > 0")
    if not _num(d["cutoff"]) or d["cutoff"] <= 0:
        errors.append("data.cutoff: must be a number > 0")
    if d["rule"] not in ("cutoff", "rise"):
        errors.append(f"data.rule: {d['rule']!r}; use 'cutoff' or 'rise'")
    if not _num(d["rise_factor"]) or d["rise_factor"] <= 1:
        errors.append("data.rise_factor: must be a number > 1")
    if not _num(d["rule_step"]) or d["rule_step"] <= 0:
        errors.append("data.rule_step: must be a number > 0 (months)")

    # backend settings
    a = out["abc"]
    if a["acceptance"] not in ("likelihood", "threshold"):
        errors.append(f"abc.acceptance: {a['acceptance']!r}; use 'likelihood' or 'threshold'"
                      f"{_suggest(a['acceptance'], ('likelihood', 'threshold'))}")
    for k, lo in (("n_particles", 100), ("max_generations", 1), ("n_moves", 1),
                  ("max_sims_per_gen", 1), ("batch", 1)):
        if not (isinstance(a[k], (int, np.integer)) and not isinstance(a[k], bool)) or a[k] < lo:
            errors.append(f"abc.{k}: must be an integer >= {lo}")
    for k in ("alpha", "min_accept_rate"):
        if not _num(a[k]) or not 0 < a[k] < 1:
            errors.append(f"abc.{k}: must be a number in (0, 1)")
    pert = a["perturbation"]
    if pert.get("type") != "gaussian":
        errors.append(f"abc.perturbation.type: only 'gaussian' is available")
    if not (pert.get("scale") == "auto" or (_num(pert.get("scale")) and pert["scale"] > 0)):
        errors.append("abc.perturbation.scale: 'auto' or a number > 0 (multiplies the "
                      "particle covariance)")
    if backend == "abc" and a["acceptance"] == "threshold" and "sigma" in dists and not dists["sigma"].fixed:
        errors.append("priors.sigma: threshold acceptance cannot estimate sigma (a smaller "
                      "sigma always looks better); fix it with a number, e.g. 0.15, or use "
                      "abc.acceptance = 'likelihood'")
    mc = out["mcmc"]
    for k in ("draws", "tune", "chains"):
        if not isinstance(mc[k], (int, np.integer)) or isinstance(mc[k], bool) or mc[k] < 1:
            errors.append(f"mcmc.{k}: must be an integer >= 1")
    if not _num(mc["target_accept"]) or not 0 < mc["target_accept"] < 1:
        errors.append("mcmc.target_accept: must be in (0, 1)")
    if mc["sampler"] != "nuts":
        errors.append("mcmc.sampler: only 'nuts' is available")
    if mc["cores"] is not None and (not isinstance(mc["cores"], (int, np.integer))
                                     or isinstance(mc["cores"], bool) or mc["cores"] < 1):
        errors.append("mcmc.cores: must be None or an integer >= 1")
    if backend == "mcmc" and mname != "growth_death":
        errors.append("backend 'mcmc' currently supports only growth_death")
    other = "mcmc" if backend == "abc" else "abc"
    if any(k.startswith(other + ".") for k in sources):
        msg = f"{other}: settings given but backend is {backend!r}; they are not used"
        (errors if strict_unused else notes).append(msg)
    if a["acceptance"] == "likelihood":
        used = [k for k in ("min_accept_rate", "max_sims_per_gen", "batch")
                if f"abc.{k}" in sources]
        if used:
            notes.append(f"abc.{', abc.'.join(used)}: only used with acceptance = 'threshold'")
    else:
        if "abc.n_moves" in sources:
            notes.append("abc.n_moves: only used with acceptance = 'likelihood'")

    # forecast / next visit / seed
    f = out["forecast"]
    if not _num(f["horizon"]) or f["horizon"] <= 0:
        errors.append("forecast.horizon: must be > 0 (months)")
    if not _num(f["step"]) or f["step"] <= 0 or (_num(f["horizon"]) and f["step"] >= f["horizon"]):
        errors.append("forecast.step: must be > 0 and smaller than the horizon")
    if not isinstance(f["n_draws"], (int, np.integer)) or f["n_draws"] < 100:
        errors.append("forecast.n_draws: must be an integer >= 100")
    nv = out["next_visit"]
    if nv["rule"] not in ("risk", "width"):
        errors.append(f"next_visit.rule: {nv['rule']!r}; use 'risk' or 'width'")
    if not _num(nv["p"]) or not 0 < nv["p"] < 1:
        errors.append("next_visit.p: must be in (0, 1)")
    if not (_num(nv["min_gap"]) and _num(nv["max_gap"]) and 0 <= nv["min_gap"] < nv["max_gap"]):
        errors.append("next_visit: need 0 <= min_gap < max_gap")
    if not isinstance(out["seed"], (int, np.integer)):
        errors.append("seed: must be an integer")

    if errors:
        raise ProfileError("profile has %d problem(s):\n  - " % len(errors)
                           + "\n  - ".join(errors))
    out["_sources"] = sources
    if verbose:
        print(describe(out, notes))
    else:
        for n in notes:
            warnings.warn(n, stacklevel=2)
    return out


def prior_table(profile):
    model = get_model(profile["model"], **(profile.get("model_options") or {}))
    src = profile.get("_sources", {})
    rows = []
    for name, spec in profile["priors"].items():
        d = check_prior(name, spec, [], False, True)
        p = name.replace("_pre", "") if name.endswith("_pre") else name.replace("shift_", "")
        key = (name if name in PARAMETERS else
               "{p}_pre" if name.endswith("_pre") and name != "level_pre" else
               "shift_{p}" if name.startswith("shift_") else name)
        desc = PARAMETERS.get(key, "").replace("{p}", p)
        if d.kind == "lognormal" and d.args["median"] == "first_reading":
            text = f"lognormal(median = that reading, sd_log {d.args['sd_log']:g})"
        else:
            text = d.describe()
        rows.append(dict(parameter=name, prior=text,
                         set_by=src.get(f"priors.{name}", "default"), meaning=desc))
    return pd.DataFrame(rows)


def describe(profile, notes=()):
    t = prior_table(profile)
    lines = [f"profile: backend = {profile['backend']}, model = {profile['model']}, "
             f"surgery shifts {profile['surgery']['shift_params']}, "
             f"abc acceptance = {profile['abc']['acceptance']}"]
    lines.append(t.to_string(index=False))
    you = [k for k in profile.get("_sources", {}) if not k.startswith("priors.")]
    if you:
        lines.append("other settings you set: " + ", ".join(you))
    lines += [f"note: {n}" for n in notes]
    return "\n".join(lines)
