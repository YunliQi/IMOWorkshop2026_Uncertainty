"""Figures for snapshots and histories (Forecaster.fit / replay).

    hist.plot_tube("tube.png")              # tumour band, refit after each visit (one axis);
                                            # black dashed = truth (synthetic) or the
                                            # fit to the readings so far (real data)
    hist.plot_panels("panels.png")          # one panel per refit
    hist.plot_parameters("params.png")      # each parameter's posterior after every visit
    hist.plot_parameter_space("space.png")  # posterior path in parameter space (80% ellipses)
    hist.plot_parameter_distributions("d.png")  # full posterior of every parameter, per visit
    snap.plot()                             # one snapshot

True values are drawn when the record has them (synthetic data). Every function returns the
figure and saves it when a path is given.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from .backends.base import wquantile

DATA_C, MODEL_C, INK, MUTED, GRID, BG = ("#2a78d6", "#eb6834", "#1f1f1e", "#8a897f",
                                         "#e6e5e0", "#fcfcfb")
RAMP = LinearSegmentedColormap.from_list("time", ["#f8d3bf", "#eb6834", "#7a2a0b"])
STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
         "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG}
LABELS = {"gamma_pre": "γ before surgery (/mo)", "delta_pre": "δ before surgery (/mo)",
          "gamma_post": "γ after surgery (/mo)", "delta_post": "δ after surgery (/mo)",
          "shift_gamma": "surgery shift of log γ", "shift_delta": "surgery shift of log δ",
          "level_pre": "level at first pre-surgery reading", "sigma": "observation error σ",
          "level_post": "level at first post-surgery reading", "ca125_0": "CA-125 at t = 0",
          "surgery_jump": "after / before surgery",
          # poptk
          "growth_pre": "growth rate (/mo)", "growth_post": "growth after surgery (/mo)",
          "kill_pre": "log10 kill per dose (median cell)",
          "kill_post": "log10 kill per dose after surgery",
          "het_pre": "within-tumour sd of log10 sensitivity",
          "het_post": "sensitivity sd after surgery",
          "shift_growth": "surgery shift of log growth", "shift_kill": "surgery shift of log kill",
          "shift_het": "surgery shift of log het",
          "g_pre": "growth g (/mo)", "g_post": "growth g after surgery (/mo)",
          "k_pre": "chemo kill k on S (/mo, during a pulse)",
          "k_post": "chemo kill k after surgery (/mo)",
          "rho_pre": "R0 / S0 at the first reading", "rho_post": "R0 / S0 (after surgery)",
          "shift_g": "surgery shift of log g", "shift_k": "surgery shift of log k",
          "shift_rho": "surgery shift of log rho", "b0": "background CA-125 b0 (U/mL)"}
LINEAR = {"shift_gamma", "shift_delta", "shift_growth", "shift_kill", "shift_het",
          "shift_g", "shift_k", "shift_rho"}


def _clean_log(ax, which="both"):
    for a in {"x": [ax.xaxis], "y": [ax.yaxis], "both": [ax.xaxis, ax.yaxis]}[which]:
        lo, hi = (ax.get_xlim() if a is ax.xaxis else ax.get_ylim())
        narrow = lo > 0 and hi / lo < 6          # fewer than two 1-2-5 ticks: use 1..9
        a.set_major_locator(LogLocator(subs=np.arange(1, 10) if narrow else (1, 2, 5)))
        a.set_major_formatter(FormatStrFormatter("%g"))
        a.set_minor_formatter(NullFormatter())


def _marks(ax, s, cutoff):
    for x in np.concatenate([s.nact_starts, s.act_starts]):
        ax.axvline(x, color=MUTED, ls=":", lw=0.6, zorder=0)
    ax.axvline(s.t_surgery, color=INK, lw=0.9, zorder=0)
    ax.axhline(cutoff, color=INK, ls="-.", lw=0.7, zorder=0)


def _ylim(rec):
    hi = max(3e3, 2 * rec.y.max())
    lo = min(0.3, 0.5 * rec.y.min())
    return lo, hi


def _xmax(hist):
    r = hist.record
    end = r.t[-1] if not np.isfinite(r.t_stop) else max(r.t[-1], r.t_stop)
    return end + 4


def _save(fig, path, dpi=140):
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def _cap_text(v, cap):
    if not np.isfinite(v):
        return "n/a"
    return f"> {cap:g}" if v > cap else f"{v:.0f}"


def _truth_curve(ax, rec):
    if rec.truth_curve is not None:
        ax.plot(rec.truth_t, rec.truth_curve, color=INK, ls="--", lw=0.9,
                label="true tumour burden (hidden)")


def _reference(ax, hist, reference="auto", only=None):
    """Black dashed reference curve.
    "auto": the true curve for synthetic data, else "forecast".
    "forecast": between visits k and k + 1, the median of the fit to readings 1..k (the left
                reading and the ones before it); drawn by the callers in place of the orange
                median line.
    "hindsight": between visits k and k + 1, the median of the refit that already includes
                 reading k + 1 (compare with the orange forecast made at visit k).
    "fit": the model fitted to every reading (median of the last refit).
    "none": no reference curve."""
    rec = hist.record
    if reference == "auto":
        if rec.truth_curve is not None:
            return _truth_curve(ax, rec)
        reference = "forecast"
    if reference == "forecast":
        return
    if reference == "hindsight":
        snaps = hist.snapshots
        for k, (s, nxt) in enumerate(zip(snaps[:-1], snaps[1:])):
            if only is not None and k != only:
                continue
            b = nxt.band((0.5,))
            seg = (b.t >= s.t_now) & (b.t <= nxt.t_now)
            ax.plot(b.t[seg], b.q50[seg], color=INK, ls="--", lw=1.1, zorder=3,
                    label="refit after the next reading (hindsight)"
                    if k == (only or 0) else None)
    elif reference == "fit":
        last = hist.snapshots[-1]
        b = last.band((0.5,))
        ax.plot(b.t, b.q50, color=INK, ls="--", lw=1.1, zorder=3,
                label=f"model fit to all readings (up to month {last.t_now:.1f})")


def _median_style(hist, reference):
    """Style of each refit's median line: black dashed for reference="forecast"."""
    fc = reference == "forecast" or (reference == "auto" and hist.record.truth_curve is None)
    return (dict(color=INK, ls="--", lw=1.2, zorder=3) if fc else dict(color=MODEL_C, lw=1.6)), fc


def _stop(ax, rec, label=False):
    if np.isfinite(rec.t_stop):
        ax.axvline(rec.t_stop, color=INK, lw=2, alpha=0.4,
                   label="model stops (2nd line / maintenance)" if label else None)
    if np.isfinite(rec.t_recurrence):
        ax.axvline(rec.t_recurrence, color="#c0392b", lw=1.6, ls=(0, (4, 2)),
                   label="observed recurrence" if label else None)


# ---------------------------------------------------------------- one snapshot
def snapshot(snap, ax=None, path=None, noisy=False):
    with plt.rc_context(STYLE):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 4.5))
        rec, pb = snap.record, snap.posterior.problem
        b = snap.band((0.025, 0.1, 0.5, 0.9, 0.975), noisy=noisy)
        fut = b.t >= snap.t_now
        ax.fill_between(b.t[fut], b.q2_5[fut], b.q97_5[fut], color=MODEL_C, alpha=0.15, lw=0,
                        label="95% band")
        ax.fill_between(b.t[fut], b.q10[fut], b.q90[fut], color=MODEL_C, alpha=0.35, lw=0,
                        label="80% band")
        ax.plot(b.t, b.q50, color=MODEL_C, lw=1.4)
        _truth_curve(ax, rec)
        ax.scatter(pb.t, pb.y, s=18, color=DATA_C, zorder=4, label="readings used")
        later = rec.t > snap.t_now + 1e-9
        if later.any():
            ax.scatter(rec.t[later], rec.y[later], s=18, facecolor="none", edgecolor=DATA_C,
                       zorder=4, label="later readings")
        _marks(ax, rec.schedule, pb.cutoff)
        _stop(ax, rec, label=True)
        ax.axvline(snap.t_now, color=INK, lw=1.6, alpha=0.4)
        nv = snap.next_visit()
        if np.isfinite(nv):
            ax.axvline(nv, color=DATA_C, lw=1.2, ls="--")
        cap = snap.followup
        lo, md, hi = snap.recurrence_interval(cap=cap + 1)
        ax.set_title(f"{rec.patient_id} at month {snap.t_now:.1f}: 80% recurrence "
                     f"[{_cap_text(lo, cap)}, {_cap_text(hi, cap)}] months after ACT; next visit "
                     f"suggested at month {nv:.1f} (blue dashed)", loc="left", fontsize=9.5)
        ax.set_yscale("log")
        ax.set_ylim(*_ylim(rec))
        _clean_log(ax, "y")
        ax.set_xlabel("months")
        ax.set_ylabel("tumour burden / CA-125 (U/mL)")
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        return _save(fig, path)


# ---------------------------------------------------------------- history: tumour
def tube(hist, path=None, reference="auto"):
    """One axis: between visit k and k + 1, the band from the fit after visit k; below, how
    wide the uncertainty is after each refit."""
    rec, snaps = hist.record, hist.snapshots
    xmax = _xmax(hist)
    cap = snaps[0].followup
    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
        ends = [s.t_now for s in snaps[1:]] + [xmax]
        for s, t_next in zip(snaps, ends):
            b = s.band((0.025, 0.1, 0.5, 0.9, 0.975))
            seg = (b.t >= s.t_now) & (b.t <= t_next)
            ax.fill_between(b.t[seg], b.q2_5[seg], b.q97_5[seg], color=MODEL_C, alpha=0.15, lw=0)
            ax.fill_between(b.t[seg], b.q10[seg], b.q90[seg], color=MODEL_C, alpha=0.35, lw=0)
            style, as_ref = _median_style(hist, reference)
            ax.plot(b.t[seg], b.q50[seg], **style,
                    label=("model fit to the readings so far (median)" if as_ref and
                           s is snaps[0] else None))
        _reference(ax, hist, reference)
        ax.scatter(rec.t, rec.y, s=30, color=DATA_C, zorder=4, label="CA-125 readings (visits)")
        for s in snaps:
            ax.axvline(s.t_now, color=DATA_C, lw=0.5, alpha=0.5, zorder=0)
            ax2.axvline(s.t_now, color=DATA_C, lw=0.5, alpha=0.5, zorder=0)
        _marks(ax, rec.schedule, snaps[0].posterior.problem.cutoff)
        _stop(ax, rec, label=True)
        ax.fill_between([], [], color=MODEL_C, alpha=0.35, label="80% band (refit at each visit)")
        ax.fill_between([], [], color=MODEL_C, alpha=0.15, label="95% band")
        ax.set_yscale("log")
        ax.set_ylim(*_ylim(rec))
        _clean_log(ax, "y")
        ax.set_ylabel("tumour burden / CA-125 (U/mL)")
        ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=2)
        rel = rec.t_recurrence - rec.schedule.t_end
        tr = (f" (observed recurrence {rel:.1f} mo after ACT)" if np.isfinite(rel) else "")
        ax.set_title(f"{rec.patient_id}: each visit adds a reading, the model is refitted and a "
                     f"new uncertainty zone starts{tr}. Dotted = treatment cycles, solid = "
                     f"surgery", loc="left", fontsize=10)
        T = hist.table()
        ax2.step(T.t_now, T.band_ratio_next, where="post", color=MODEL_C, lw=2, label="tumour band q90 / q10 at next visit")
        ax2.set_yscale("log")
        _clean_log(ax2, "y")
        ax2.set_ylabel("80% band width\n(q90 / q10) at next visit")
        ax2b = ax2.twinx()
        ax2b.step(T.t_now, T.rec_width, where="post", color=DATA_C, lw=1.5, ls="--")
        ax2b.set_ylabel(f"80% recurrence interval\nwidth (months, capped at {cap + 1:g})",
                        color=DATA_C)
        ax2b.spines["right"].set_visible(True)
        ax2b.grid(False)
        ax2.set_xlabel("months")
        ax2.set_xlim(0, xmax)
        ax2.set_title("How wide the uncertainty is after each refit (orange: tumour band; blue "
                      "dashed: recurrence interval)", loc="left", fontsize=10)
        fig.tight_layout()
        return _save(fig, path)


def panels(hist, path=None, ncol=4, reference="auto"):
    rec, snaps = hist.record, hist.snapshots
    xmax, cap = _xmax(hist), snaps[0].followup
    n = len(snaps)
    nrow = int(np.ceil(n / ncol))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.9 * nrow), sharex=True, sharey=True,
                                 squeeze=False)
        for i, (ax, s) in enumerate(zip(axes.flat, snaps)):
            b = s.band((0.025, 0.1, 0.5, 0.9, 0.975))
            fut = b.t >= s.t_now
            ax.fill_between(b.t[fut], b.q2_5[fut], b.q97_5[fut], color=MODEL_C, alpha=0.15, lw=0)
            ax.fill_between(b.t[fut], b.q10[fut], b.q90[fut], color=MODEL_C, alpha=0.35, lw=0)
            style, _ = _median_style(hist, reference)
            ax.plot(b.t[fut], b.q50[fut], **style)
            _reference(ax, hist, reference, only=i)
            used = rec.t <= s.t_now + 1e-9
            ax.scatter(rec.t[used], rec.y[used], s=14, color=DATA_C, zorder=4)
            k = used.sum() - 1
            ax.scatter([rec.t[k]], [rec.y[k]], s=60, facecolor="none", edgecolor=DATA_C, lw=1.6,
                       zorder=5)
            if k + 1 < len(rec.t):
                ax.scatter([rec.t[k + 1]], [rec.y[k + 1]], s=40, marker="*", color=INK, zorder=5)
            _marks(ax, rec.schedule, s.posterior.problem.cutoff)
            _stop(ax, rec)
            ax.axvline(s.t_now, color=INK, lw=1.6, alpha=0.4)
            lo, md, hi = s.recurrence_interval(cap=cap + 1)
            ax.set_title(f"after visit {k + 1} (month {s.t_now:.1f}): recurrence "
                         f"[{_cap_text(lo, cap)}, {_cap_text(hi, cap)}] mo", fontsize=8.5,
                         loc="left")
            ax.set_yscale("log")
            ax.set_ylim(*_ylim(rec))
            ax.set_xlim(0, xmax)
        for ax in axes.flat[n:]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("tumour burden (log)")
        for ax in axes[-1]:
            ax.set_xlabel("months")
        fig.suptitle(f"{rec.patient_id}: tumour-burden uncertainty refitted after every visit. "
                     f"Orange = 80% and 95% band from the current fit,\nfilled dots = readings "
                     f"used, ring = newest reading, star = next reading"
                     + (", dashed = truth (hidden)"
                        if rec.truth_curve is not None and reference == "auto" else
                        ", dashed = model fit to all readings" if reference == "fit" else
                        ", dashed = refit once the next reading is in (hindsight)"
                        if reference == "hindsight" else
                        ", dashed = model fit to the readings so far (median)"),
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, path, dpi=130)


# ---------------------------------------------------------------- history: parameters
BAND_PARAMS = ("gamma_pre", "delta_pre", "shift_gamma", "shift_delta",
               "gamma_post", "delta_post", "level_post", "sigma")


def _prior_quantiles(snap, n=20000, seed=0):
    """Prior 10% / 90% of each sampled parameter whose prior does not depend on the data
    (levels are centred on the first reading, so they are left out); {_post} = {_pre} when
    surgery does not change the parameter."""
    pb = snap.posterior.problem
    u = pb.prior.sample(np.random.default_rng(seed), n)
    nat = pb.prior.natural(u)
    out = {}
    for name in pb.prior.names:
        if name.startswith("level_") or name.startswith("shift_"):
            continue
        out[name] = tuple(np.quantile(nat[name], [0.1, 0.9]))
        if name.endswith("_pre") and name[:-4] not in pb.sp:
            out[name[:-4] + "_post"] = out[name]
    return out


def parameter_bands(hist, path=None, params=BAND_PARAMS):
    """Each parameter's posterior (median, 50%, 80%) after every refit; dashed = truth."""
    rec, snaps = hist.record, hist.snapshots
    S = hist.parameter_table()
    params = [p for p in params if p in set(S.parameter)]
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.7 * nrow), sharex=True,
                                 squeeze=False)
        prior = _prior_quantiles(snaps[-1])
        for ax, p in zip(axes.flat, params):
            d = S[S.parameter == p]
            if p in prior:
                for v, lab in zip(prior[p], ("prior 80% range", None)):
                    ax.axhline(v, color=MUTED, ls="--", lw=1.0, label=lab, zorder=1)
            ax.fill_between(d.t_now, d.q10, d.q90, step="post", color=MODEL_C, alpha=0.18, lw=0,
                            label="80% interval")
            ax.fill_between(d.t_now, d.q25, d.q75, step="post", color=MODEL_C, alpha=0.35, lw=0,
                            label="50% interval")
            ax.step(d.t_now, d.q50, where="post", color=MODEL_C, lw=1.8, label="median")
            if rec.truth and p in rec.truth:
                ax.axhline(rec.truth[p], color=INK, ls="--", lw=1.2, label="true value")
            for s in snaps:
                ax.axvline(s.t_now, color=DATA_C, lw=0.4, alpha=0.4, zorder=0)
            ax.axvline(rec.schedule.t_surgery, color=INK, lw=0.9, zorder=0)
            ax.axvline(rec.schedule.t_end, color=MUTED, lw=0.9, ls="--", zorder=0)
            if p not in LINEAR:
                ax.set_yscale("log")
                _clean_log(ax, "y")
            ax.set_title(LABELS.get(p, p), loc="left", fontsize=10)
        for i in range(len(params)):                 # last panel in its column: label x
            if i + ncol >= len(params):
                axes.flat[i].tick_params(labelbottom=True)
        for ax in axes.flat[len(params):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("month of the update (blue lines = readings)")
        axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
        fig.suptitle(f"{rec.patient_id}: posterior of each parameter after every visit (solid "
                     f"black = surgery, dashed grey = end of ACT)"
                     + ("" if rec.truth else ". No true values exist (real patient)."),
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, path)


ALL_PARAMS = ("gamma_pre", "delta_pre", "level_pre", "sigma", "shift_gamma", "shift_delta",
              "gamma_post", "delta_post", "level_post", "surgery_jump")
SAMPLED = {"gamma_pre", "delta_pre", "level_pre", "sigma", "shift_gamma", "shift_delta",
           "level_post"}


def parameter_distributions(hist, path=None, params=ALL_PARAMS, n=1500, seed=0):
    """Full posterior of every parameter after each refit, one violin per visit (x = visit,
    labelled with its month). Parameters not estimable yet (post-surgery ones before the first
    post-surgery reading) are left empty; dashed = truth (synthetic data)."""
    rec, snaps = hist.record, hist.snapshots
    rng = np.random.default_rng(seed)
    draws = []
    for s in snaps:
        nat = s.posterior.natural()
        idx = rng.choice(len(nat), n, p=s.posterior.weights)
        draws.append(nat.iloc[idx].reset_index(drop=True))
    params = [p for p in params if p in draws[-1].columns]
    ncol = 2
    nrow = int(np.ceil(len(params) / ncol))
    x = np.arange(len(snaps))
    surg = np.searchsorted(hist.times, rec.schedule.t_surgery)
    end = np.searchsorted(hist.times, rec.schedule.t_end)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(17, 2.9 * nrow), sharex=True,
                                 squeeze=False)
        for ax, p in zip(axes.flat, params):
            log = p not in LINEAR
            vals, pos = [], []
            for i, d in enumerate(draws):
                v = d[p].values
                v = v[np.isfinite(v) & ((v > 0) if log else True)]
                if len(v) > 10:
                    vals.append(np.log10(v) if log else v)
                    pos.append(i)
            if vals:
                vp = ax.violinplot(vals, positions=pos, widths=0.8, showextrema=False,
                                   showmedians=False)
                for b in vp["bodies"]:
                    b.set_facecolor(MODEL_C)
                    b.set_edgecolor(MODEL_C)
                    b.set_alpha(0.45)
                med = [np.median(v) for v in vals]
                q10 = [np.quantile(v, 0.1) for v in vals]
                q90 = [np.quantile(v, 0.9) for v in vals]
                ax.vlines(pos, q10, q90, color=MODEL_C, lw=1.2)
                ax.plot(pos, med, "o-", color=MODEL_C, ms=3, lw=0.8)
            if rec.truth and p in rec.truth and np.isfinite(rec.truth[p]):
                tv = rec.truth[p]
                ax.axhline(np.log10(tv) if log else tv, color=INK, ls="--", lw=1.1)
            ax.axvline(surg - 0.5, color=INK, lw=0.9, zorder=0)
            ax.axvline(end - 0.5, color=MUTED, lw=0.9, ls="--", zorder=0)
            if log:
                lo, hi = ax.get_ylim()
                ticks = [t for t in np.log10(np.outer([1, 2, 5], 10.0 ** np.arange(-3, 5)).ravel())
                         if lo <= t <= hi]
                if len(ticks) > 7:
                    ticks = [t for t in ticks if abs(t - round(t)) < 1e-9 or
                             abs(t - round(t) - np.log10(5)) < 1e-9 or
                             abs(t - round(t) + 1 - np.log10(5)) < 1e-9]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{10 ** t:g}" for t in ticks])
            kind = "" if p in SAMPLED else " (derived)"
            ax.set_title(LABELS.get(p, p) + kind, loc="left", fontsize=10)
        for ax in axes.flat[len(params):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:.1f}" for t in hist.times], rotation=90, fontsize=7.5)
            ax.set_xlabel("month of the update (one violin per visit)")
        fig.suptitle(f"{rec.patient_id}: posterior of every parameter after each visit (violin = "
                     f"full distribution, bar = 80% interval, dot = median; solid = surgery, "
                     f"dashed grey = end of ACT"
                     + ("; black dashed = true value)" if rec.truth else ")"),
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return _save(fig, path)


def _ellipse(x, y, w, logx, logy, level=0.8, n=100):
    fx = np.log if logx else (lambda v: v)
    fy = np.log if logy else (lambda v: v)
    gx = np.exp if logx else (lambda v: v)
    gy = np.exp if logy else (lambda v: v)
    z = np.column_stack([fx(x), fy(y)])
    m = (w[:, None] * z).sum(0)
    d = z - m
    vals, vecs = np.linalg.eigh((w[:, None] * d).T @ d)
    a = np.linspace(0, 2 * np.pi, n)
    r = np.sqrt(-2 * np.log(1 - level))
    pts = m + (np.column_stack([np.cos(a), np.sin(a)]) * np.sqrt(np.maximum(vals, 0)) * r) @ vecs.T
    return np.array([gx(m[0]), gy(m[1])]), np.column_stack([gx(pts[:, 0]), gy(pts[:, 1])])


SPACE_PAIRS = (("gamma_pre", "delta_pre", "Before surgery: growth vs kill"),
               ("gamma_post", "delta_post", "After surgery: growth vs kill"),
               ("shift_gamma", "shift_delta", "Surgery as a shift in parameter space"))


def parameter_space(hist, path=None, pairs=SPACE_PAIRS):
    """Posterior path in parameter space: 80% ellipse after each refit (light = early, dark =
    late), centres joined; × = truth."""
    rec, snaps = hist.record, hist.snapshots
    times = hist.times
    norm = Normalize(times.min(), times.max())
    nats = [s.posterior.natural() for s in snaps]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.7 * len(pairs), 5.6), squeeze=False)
        for ax, (px, py, title) in zip(axes[0], pairs):
            lx, ly = px not in LINEAR, py not in LINEAR
            cs, ct = [], []
            for s, nat in zip(snaps, nats):
                if not (np.isfinite(nat[px]).all() and np.isfinite(nat[py]).all()):
                    continue                   # not estimable yet (before a post reading)
                c, pts = _ellipse(nat[px].values, nat[py].values, s.posterior.weights, lx, ly)
                ax.plot(pts[:, 0], pts[:, 1], color=RAMP(norm(s.t_now)), lw=1.0, alpha=0.85)
                cs.append(c)
                ct.append(s.t_now)
            if not cs:
                ax.set_title(title + " (no fits yet)", loc="left", fontsize=10)
                continue
            cs = np.array(cs)
            ax.plot(cs[:, 0], cs[:, 1], color=MUTED, lw=0.8, zorder=3)
            sc = ax.scatter(cs[:, 0], cs[:, 1], c=ct, cmap=RAMP, norm=norm, s=28,
                            edgecolor=BG, lw=0.6, zorder=4)
            ax.scatter(*cs[0], s=90, facecolor="none", edgecolor=INK, lw=1, zorder=5)
            ax.annotate("first fit", cs[0], xytext=(6, 6), textcoords="offset points",
                        fontsize=8)
            if rec.truth and px in rec.truth and py in rec.truth:
                ax.scatter([rec.truth[px]], [rec.truth[py]], marker="x", s=110, color=INK,
                           lw=2, zorder=6, label="true value")
                ax.legend(fontsize=8, frameon=False, loc="best")
            if px in LINEAR and py in LINEAR:
                ax.axhline(0, color=MUTED, lw=0.7)
                ax.axvline(0, color=MUTED, lw=0.7)
                ax.text(0.02, 0.02, "(0, 0) = surgery changes nothing", transform=ax.transAxes,
                        fontsize=8, color=MUTED)
            if lx:
                ax.set_xscale("log")
                _clean_log(ax, "x")
            if ly:
                ax.set_yscale("log")
                _clean_log(ax, "y")
            ax.set_xlabel(LABELS.get(px, px))
            ax.set_ylabel(LABELS.get(py, py))
            ax.set_title(title, loc="left", fontsize=10)
        cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.85, pad=0.01)
        cb.set_label("month of the update")
        fig.suptitle(f"{rec.patient_id}: posterior path in parameter space. Each ring = 80% "
                     f"region after one visit (light = early, dark = late); dots = centres",
                     x=0.01, ha="left", fontsize=11)
        return _save(fig, path)
