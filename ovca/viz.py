"""Figures for snapshots and histories (Forecaster.fit / replay).

    hist.plot_tube("tube.png")              # tumour band, refit after each visit (one axis);
                                            # black dashed = truth (synthetic) or the
                                            # fit to the readings so far (real data)
    hist.plot_panels("panels.png")          # one panel per refit
    hist.plot_parameters("params.png")      # each parameter's posterior after every visit
    hist.plot_parameter_space("space.png")  # posterior path in parameter space (80% ellipses)
    hist.plot_parameter_distributions("d.png")  # full posterior of every parameter, per visit
    hist.plot_recurrence("rec.png")         # recurrence-time density after every refit (ridgeline)
    hist.plot_recurrence_cdf("cdf.png")     # P(recurred by t) after every refit, one axis
    snap.plot()                             # one snapshot
    snap.plot_recurrence()                  # recurrence-time density at one snapshot
    snap.plot_recurrence_cdf()              # P(recurred by t) at one snapshot

True values are drawn when the record has them (synthetic data). Every function returns the
figure and saves it when a path is given.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from scipy.stats import gaussian_kde

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
          "surgery_jump": "after / before surgery"}
LINEAR = {"shift_gamma", "shift_delta"}


def _clean_log(ax, which="both"):
    for a in {"x": [ax.xaxis], "y": [ax.yaxis], "both": [ax.xaxis, ax.yaxis]}[which]:
        a.set_major_locator(LogLocator(subs=(1, 2, 5)))
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
        for ax, p in zip(axes.flat, params):
            d = S[S.parameter == p]
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
    labelled with its month). Before the first post-surgery reading the post-surgery parameters
    come from their priors; dashed = truth (synthetic data)."""
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


# ---------------------------------------------------------------- recurrence time
REC_LEVELS = (0.95, 0.8, 0.5)


def _recurrence_density(rec, grid):
    """KDE of recurrence times (months after ACT) on grid, reflected at 0 (no recurrence before
    the end of ACT). Bandwidth: Silverman with min(sd, IQR / 1.34), so the long right tail
    doesn't widen it. Area = share of draws that recur (inf draws can't be in a density)."""
    finite = rec[np.isfinite(rec)]
    sd = finite.std() if finite.size > 1 else 0.0
    if not sd > 0:
        return np.zeros_like(grid)
    iqr = np.subtract(*np.quantile(finite, [0.75, 0.25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    kde = gaussian_kde(finite, bw_method=0.9 * spread * finite.size ** -0.2 / sd)
    return (kde(grid) + kde(-grid)) * finite.size / rec.size


def _recurrence_xmax(rec, until):
    """Time by which `until` of the draws have recurred (the largest finite time if never)."""
    x = np.quantile(rec, until, method="inverted_cdf")
    return float(x) if np.isfinite(x) else float(np.max(rec[np.isfinite(rec)]))


def _fill_intervals(ax, grid, density, rec, levels, base=0.0, label=True):
    """Shade under the density by central intervals: one hue, darker = more central."""
    levels = sorted(levels, reverse=True)                  # widest first, narrower on top
    for level, shade in zip(levels, RAMP(np.linspace(0.15, 0.7, len(levels)))):
        lo, hi = np.quantile(rec, [(1 - level) / 2, (1 + level) / 2], method="inverted_cdf")
        inside = (grid >= lo) & (grid <= hi)
        ax.fill_between(grid[inside], base, base + density[inside], color=shade, lw=0,
                        label=f"{level:.0%} interval" if label else None)


def recurrence_density(snap, ax=None, path=None, until=0.999, levels=REC_LEVELS):
    """Density of the recurrence time at one snapshot, in months since diagnosis (t = 0), shaded
    by central intervals, up to the month by which `until` of the draws have recurred. The grey
    line is the end of ACT; nothing recurs before it."""
    with plt.rc_context(STYLE):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 3.8))
        ax.set_axisbelow(True)                       # grid behind the shading
        rec, record = snap.recurrence_times(), snap.record
        t_end = record.schedule.t_end
        ax.axvline(t_end, color=MUTED, lw=1.4, zorder=0)    # end of ACT
        ax.set_xlabel("months since diagnosis")
        ax.set_ylabel("density")
        if np.isnan(rec).all():                      # no relapse forecast: nothing to show
            ax.set_xlim(0, max(t_end, snap.t_now) + 12)
            return _save(fig, path)
        xmax = t_end + _recurrence_xmax(rec, until)
        grid = np.linspace(0, xmax, 1500)
        density = np.where(grid >= t_end, _recurrence_density(rec, grid - t_end), 0.0)
        _fill_intervals(ax, grid, density, rec + t_end, levels)
        ax.plot(grid, density, color=RAMP(1.0), lw=1.5)
        median = t_end + np.quantile(rec, 0.5, method="inverted_cdf")
        if np.isfinite(median):
            ax.axvline(median, color=RAMP(1.0), lw=1.2, ls=":", label=f"median {median:.1f}")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, None)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        return _save(fig, path)


def _recurrence_cdf(rec, grid):
    """Share of draws that have recurred by each time on grid (inf = not yet)."""
    return np.searchsorted(np.sort(rec), grid, side="right") / rec.size


def _end_of_act(ax, s):
    """Unlabelled grey line at the end of ACT."""
    ax.axvline(s.t_end, color=MUTED, lw=1.4, zorder=0)


def recurrence_cdf(snap, ax=None, path=None, until=0.999):
    """P(recurred by t), t in months since diagnosis, at one snapshot; grey line = end of ACT."""
    with plt.rc_context(STYLE):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 3.8))
        ax.set_axisbelow(True)                       # grid behind the lines
        rec, record, pb = snap.recurrence_times(), snap.record, snap.posterior.problem
        s = record.schedule
        ax.set_title(f"{record.patient_id}: {len(pb.t)} readings (month {snap.t_now:.1f})",
                     loc="left", fontsize=9.5)
        _end_of_act(ax, s)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("months since diagnosis")
        ax.set_ylabel("P(recurred by then)")
        if np.isnan(rec).all():
            ax.text(0.5, 0.5, "no relapse forecast",
                    transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_xlim(0, max(s.t_end, snap.t_now) + 12)
            return _save(fig, path)
        xmax = s.t_end + _recurrence_xmax(rec, until)
        grid = np.linspace(0, xmax, 2000)
        ax.plot(grid, _recurrence_cdf(rec + s.t_end, grid), color=RAMP(1.0), lw=1.5)
        ax.set_xlim(0, xmax)
        return _save(fig, path)


def recurrence_cdfs(hist, path=None, until=0.9, xmax=None, facet=True, ncol=4):
    """P(recurred by t) after every refit, t in months since diagnosis; grey line = end of ACT.
    facet=True (default): one panel per refit, labelled by the number of readings and the month.
    facet=False: every refit on
    one axis, light (early) to dark (late); steeper curves = a sharper forecast. The axis runs
    until `until` of the draws have recurred in every refit. Refits before the first
    post-surgery reading have no curve."""
    rec_all = [s.recurrence_times() for s in hist.snapshots]
    known = [not np.isnan(r).all() for r in rec_all]
    record = hist.record
    t_end = record.schedule.t_end
    if xmax is None:
        xmax = t_end + max([_recurrence_xmax(r, until) for r, k in zip(rec_all, known) if k]
                           + [12.0])
    grid = np.linspace(0, xmax, 2000)
    if facet:
        return _recurrence_cdf_panels(hist, rec_all, known, grid, path, ncol)
    norm = Normalize(hist.times.min(), hist.times.max())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _end_of_act(ax, record.schedule)
        for s, rec, k in zip(hist.snapshots, rec_all, known):
            if k:
                ax.plot(grid, _recurrence_cdf(rec + t_end, grid), color=RAMP(norm(s.t_now)),
                        lw=1.3)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=RAMP), ax=ax, pad=0.01)
        cb.set_label("month of the update")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("months since diagnosis")
        ax.set_ylabel("P(recurred by then)")
        ax.set_title(f"{record.patient_id}: P(recurred)", loc="left", fontsize=10)
        return _save(fig, path)


def _recurrence_cdf_panels(hist, rec_all, known, grid, path, ncol):
    """recurrence_cdfs(facet=True): one panel per refit, shared axes, as in panels()."""
    record, snaps = hist.record, hist.snapshots
    t_end = record.schedule.t_end
    n = len(snaps)
    nrow = int(np.ceil(n / ncol))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.6 * nrow), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, s, rec, k in zip(axes.flat, snaps, rec_all, known):
            ax.set_axisbelow(True)                   # grid behind the lines
            _end_of_act(ax, record.schedule)
            if k:
                ax.plot(grid, _recurrence_cdf(rec + t_end, grid), color=RAMP(1.0), lw=1.2)
            else:
                ax.text(0.5, 0.5, "", transform=ax.transAxes,
                        ha="center", va="center", color=MUTED, fontsize=8)
            ax.set_title(f"{len(s.posterior.problem.t)} readings (month {s.t_now:.1f})",
                         loc="left", fontsize=8.5)
            ax.set_xlim(0, grid[-1])
            ax.set_ylim(0, 1.02)
        for ax in axes.flat[n:]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("P(recurred)")
        for ax in axes[-1]:
            ax.set_xlabel("months since diagnosis")
        fig.suptitle(f"{record.patient_id}: P(recurred by t) after every refit",
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, path, dpi=130)


def recurrence_densities(hist, path=None, until=None, levels=REC_LEVELS, xmax=None, log_x=False):
    """Recurrence-time density after every refit, stacked (ridgeline): earliest at the top, in
    months since diagnosis (t = 0); the grey line is the end of ACT.
    Each curve is scaled to its own peak, so the shapes compare, not the heights.
    The time axis runs until `until` of the draws have recurred in every refit (default 0.9 on a
    linear axis: the long tails of early fits are cut at the right edge; 0.99 with log_x).
    log_x=True shows early wide and late narrow forecasts equally well; the curves are then
    densities of log time (density x t), which keeps the shaded areas right on that axis."""
    until = (0.99 if log_x else 0.9) if until is None else until
    rec_all = [s.recurrence_times() for s in hist.snapshots]
    known = [not np.isnan(r).all() for r in rec_all]
    record = hist.record
    t_end = record.schedule.t_end
    if xmax is None:
        xmax = max([t_end + _recurrence_xmax(r, until) for r, k in zip(rec_all, known) if k]
                   + [t_end + 1.0])
    if log_x:
        starts = [t_end + np.quantile(r, 0.01, method="inverted_cdf")
                  for r, k in zip(rec_all, known) if k]
        xmin = max(0.1, 0.7 * min(starts + [t_end, xmax]))
        grid = np.geomspace(xmin, xmax, 1500)
    else:
        xmin, grid = 0.0, np.linspace(0, xmax, 1500)
    n = len(hist.snapshots)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 1.2 + 0.55 * n))
        ax.set_axisbelow(True)                       # grid behind the shading
        ax.axvline(t_end, color=MUTED, lw=1.4, zorder=0)    # end of ACT
        for i, (s, rec, k) in enumerate(zip(hist.snapshots, rec_all, known)):
            base = n - 1 - i
            if not k:                                # no relapse forecast: empty row
                continue
            density = np.where(grid >= t_end, _recurrence_density(rec, grid - t_end), 0.0)
            density = density * (grid if log_x else 1.0)
            density = 0.9 * density / density.max() if density.max() > 0 else density
            _fill_intervals(ax, grid, density, rec + t_end, levels, base=base, label=i == n - 1)
            ax.plot(grid, base + density, color=RAMP(1.0), lw=1)
            median = t_end + np.quantile(rec, 0.5, method="inverted_cdf")
            if np.isfinite(median) and median <= xmax:
                ax.plot([median, median], [base, base + 0.9], color="white", lw=1.2)
        ax.set_yticks(np.arange(n)[::-1] + 0.3)
        ax.set_yticklabels([f"{len(s.posterior.problem.t)} readings (month {s.t_now:.1f})"
                            for s in hist.snapshots], fontsize=8)
        ax.grid(axis="y", visible=False)
        if log_x:
            ax.set_xscale("log")
            _clean_log(ax, "x")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.1, n)
        ax.set_xlabel("Probability of recurrence at months since diagnosis" + (" (log scale)" if log_x else ""))
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=4)
        fig.tight_layout()
        return _save(fig, path)
