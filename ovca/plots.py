"""Figures -> figures/

  0_how_synthetic_data_is_made.png  one synthetic patient, step by step
  1_calibration_examples.png        six patients: readings, true curve, RMSE fit
  2_parameter_recovery.png          fitted vs true parameters
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .calibration import segment_data  # noqa: E402
from .models import BaseModel  # noqa: E402
from .patient import PatientParameters, drug_on  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
DATA_C, MODEL_C, INK, MUTED, GRID, DROP_C, BG = (
    "#2a78d6", "#eb6834", "#1f1f1e", "#8a897f", "#e6e5e0", "#b9b8b0", "#fcfcfb")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG})


def _patient(model, truth, pid):
    r = truth[truth.patient == pid].iloc[0]
    return PatientParameters.from_record(r, model.param_names), r


def _marks(ax, s):
    for x in np.concatenate([s.nact_starts, s.act_starts]):
        ax.axvline(x, color=MUTED, ls="--", lw=0.7, zorder=0)
    ax.axvline(s.t_surgery, color=INK, lw=1.2, zorder=0)


def fig_explainer(model: BaseModel, obs, truth, pid="SYN12"):
    if pid not in set(truth.patient):
        pid = truth[np.isfinite(truth.t_detected)].patient.iloc[0]
    p, r = _patient(model, truth, pid)
    s = p.schedule
    op = obs[obs.patient == pid]
    tmax = op.month.max() + 1.5
    tt = np.linspace(0, tmax, 3000)
    T = model.simulate_patient(tt, p)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.8], height_ratios=[0.55, 1, 1],
                          hspace=0.45, wspace=0.3)
    axD = fig.add_subplot(gs[0, :2])
    axT = fig.add_subplot(gs[1, :2], sharex=axD)
    axY = fig.add_subplot(gs[2, :2], sharex=axD)
    axQ, axG, axTau = (fig.add_subplot(gs[i, 2]) for i in range(3))

    # 1. schedule and D(t)
    iv = np.vstack([s.intervals("pre"), s.intervals("post")])
    D = drug_on(tt, iv)
    axD.fill_between(tt, D, step="post", color=DATA_C, alpha=0.25, lw=0)
    axD.step(tt, D, where="post", color=DATA_C, lw=1.5)
    _marks(axD, s)
    axD.set_ylim(-0.1, 2.3)
    axD.set_yticks([0, 1])
    axD.set_ylabel("D(t)")
    axD.text(s.nact_starts[0], 1.35, f"NACT × {len(s.nact_starts)}", va="center", fontsize=9)
    axD.text(s.t_surgery, 1.95, "surgery", ha="center", va="center", fontsize=9, weight="bold")
    axD.text(s.act_starts[0], 1.35, f"ACT × {len(s.act_starts)}", va="center", fontsize=9)
    axD.text(s.t_end + 0.3, 0.5, "end of ACT → surveillance every 3 mo,\nD(t) = 0 from here on",
             fontsize=9, va="center", color=MUTED)
    axD.set_title("① Random schedule: D(t) = 1 for one month after each cycle (dashed lines)",
                  loc="left", fontsize=11)

    # 2. hidden tumour with the surgery jump
    axT.semilogy(tt, T, color=INK, lw=1.8)
    _marks(axT, s)
    axT.axhline(C.CA125_CUTOFF, color=MODEL_C, lw=1, ls="-.")
    axT.text(8, C.CA125_CUTOFF * 1.15, "35 U/mL", color=MODEL_C, fontsize=9)
    T_before = model.simulate_patient([s.t_surgery - 1e-6], p)[0]
    T_after = model.simulate_patient([s.t_surgery], p)[0]
    axT.annotate("", xy=(s.t_surgery + 0.15, T_after), xytext=(s.t_surgery + 0.15, T_before),
                 arrowprops=dict(arrowstyle="-|>", color=MODEL_C, lw=1.8))
    axT.text(s.t_surgery - 0.3, 4, f"surgery: T × q\nq = {p.q_surgery:.2f}\n"
             f"({T_before:.0f} → {T_after:.0f})", color=MODEL_C, fontsize=9, va="center",
             ha="right")
    pre_txt = ", ".join(f"{k} = {v:.2f}" for k, v in p.theta_pre.items())
    post_txt = ", ".join(f"{k} = {v:.2f}" for k, v in p.theta_post.items())
    axT.annotate(f"pre-surgery: {pre_txt} /mo", xy=(0.4, T[np.searchsorted(tt, 0.4)]),
                 xytext=(0.6, 4e3), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED,
                                                               lw=0.8))
    axT.text(s.t_end + 1.5, 0.6, f"post-surgery: {post_txt} /mo", fontsize=9, color=INK)
    if np.isfinite(r.true_recurrence_abs):
        axT.plot(r.true_recurrence_abs, C.CA125_CUTOFF, "o", color=MODEL_C, ms=8, zorder=5)
        axT.annotate(f"true recurrence = {r.true_recurrence_rel:.1f} mo after end of ACT",
                     xy=(r.true_recurrence_abs, C.CA125_CUTOFF),
                     xytext=(r.true_recurrence_abs - 9, 400), fontsize=9,
                     arrowprops=dict(arrowstyle="-|>", color=MODEL_C, lw=1))
    axT.set_ylim(0.2, 2e4)
    axT.set_ylabel("hidden tumour T(t)\n(CA-125 units, log)")
    axT.set_title(f"② Hidden tumour from model '{model.name}'; "
                  "surgery = instant drop to a random fraction q", loc="left", fontsize=11)

    # 3. what is observed
    axY.semilogy(tt, T, color=INK, lw=1, alpha=0.35)
    _marks(axY, s)
    axY.axhline(C.CA125_CUTOFF, color=MODEL_C, lw=1, ls="-.")
    use = op.ca125 >= C.CA125_CUTOFF
    axY.vlines(op.month, op.ca125_true, op.ca125, color=MUTED, lw=0.8)
    axY.plot(op.month[use], op.ca125[use], "o", color=DATA_C, ms=7,
             label="reading ≥ 35: used in RMSE")
    axY.plot(op.month[~use], op.ca125[~use], "o", mfc="none", mec=DROP_C, mew=1.5, ms=7,
             label="reading < 35: counted as 0, left out of RMSE")
    if np.isfinite(r.t_detected):
        axY.axvspan(r.t_detected, r.t_detected + 1, color=MODEL_C, alpha=0.12, lw=0)
        axY.text(r.t_detected - 0.3, 300, "first surveillance\nreading ≥ 35 → detected;\n"
                 "one confirm visit +1 mo,\nthen follow-up stops", ha="right", va="center",
                 fontsize=9, color=MODEL_C)
    axY.set_ylim(0.2, 2e4)
    axY.set_xlabel("months from first NACT")
    axY.set_ylabel("observed CA-125 (U/mL, log)")
    axY.legend(loc="upper center", fontsize=9, frameon=False, ncol=2)
    axY.set_title(f"③ Observed data: each visit with {C.NOISE_SD_LOG:.0%} multiplicative noise "
                  "(grey sticks = noise)", loc="left", fontsize=11)

    # right column: cohort distributions
    axQ.hist(truth.q_surgery, bins=np.linspace(0, 1, 21), color=DATA_C, alpha=0.8, ec=BG)
    axQ.axvline(p.q_surgery, color=MODEL_C, lw=1.5)
    axQ.text(p.q_surgery + 0.02, axQ.get_ylim()[1] * 0.9, pid, color=MODEL_C, fontsize=9)
    axQ.set_xlabel("q = fraction of CA-125 left after surgery")
    axQ.set_ylabel("patients")
    axQ.set_title(f"Surgery fraction q, all {len(truth)} patients\n"
                  f"log q ~ Normal(log {C.Q_MEDIAN}, {C.Q_SD_LOG}²), clipped to {list(C.Q_RANGE)}",
                  loc="left", fontsize=10)

    k = model.param_names[0]
    axG.scatter(truth[f"{k}_pre"], truth[f"{k}_post"], s=22, color=DATA_C, alpha=0.8, ec=BG,
                lw=0.5)
    axG.scatter([p.theta_pre[k]], [p.theta_post[k]], s=60, color=MODEL_C, zorder=5)
    axG.set_xscale("log")
    axG.set_yscale("log")
    axG.set_xlabel(f"{k} before surgery (/month)")
    axG.set_ylabel(f"{k} after surgery (/month)")
    axG.set_title("Parameters are drawn jointly\n(log-normal, pre and post correlated)",
                  loc="left", fontsize=10)

    rel = truth.true_recurrence_rel
    axTau.hist(np.minimum(rel, C.FOLLOWUP_MONTHS + 3),
               bins=np.arange(0, C.FOLLOWUP_MONTHS + 6, 3), color=DATA_C, alpha=0.8, ec=BG)
    if np.isfinite(r.true_recurrence_rel):
        axTau.axvline(r.true_recurrence_rel, color=MODEL_C, lw=1.5)
    axTau.set_xticks([0, 12, 24, 36, 40.5])
    axTau.set_xticklabels(["0", "12", "24", "36", "none"])
    axTau.set_xlabel("true recurrence, months after end of ACT")
    axTau.set_ylabel("patients")
    axTau.set_title(f"Resulting recurrence times\n≤6 mo {(rel <= 6).mean():.0%}, "
                    f"≤12 mo {(rel <= 12).mean():.0%}, none in {C.FOLLOWUP_MONTHS} mo "
                    f"{np.isinf(rel).mean():.0%}  (n = {len(truth)})", loc="left", fontsize=10)

    fig.suptitle(f"How a synthetic patient is made (example {pid}); all data synthetic",
                 x=0.06, ha="left", fontsize=13, weight="bold")
    fig.savefig(FIG / "0_how_synthetic_data_is_made.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(model: BaseModel, fit_model: BaseModel, obs, truth, fits, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pid in zip(axes.flat, truth.patient.head(n)):
        p, r = _patient(model, truth, pid)
        op = obs[obs.patient == pid]
        f = fits[fits.patient == pid].iloc[0]
        tt = np.linspace(0, op.month.max() + 1, 800)
        ax.plot(tt, model.simulate_patient(tt, p), color=INK, lw=1.2, ls="--",
                label="true (hidden) tumour")
        for ph in ("pre", "post"):
            t, y, iv = segment_data(op, p.schedule, ph)
            if f[f"fitted_{ph}"]:
                theta = {k: f[f"{k}_{ph}"] for k in fit_model.param_names}
                td = np.linspace(t[0], t[-1], 300)
                ax.plot(td, fit_model.simulate_segment(td, t[0], y[0], theta, iv), color=MODEL_C,
                        lw=2, label="RMSE fit" if ph == "pre" else None)
        lo = op.ca125 < C.CA125_CUTOFF
        ax.scatter(op.month[~lo], op.ca125[~lo], s=28, color=DATA_C, zorder=3,
                   label="CA-125 (used)")
        ax.scatter(op.month[lo], op.ca125[lo], s=28, facecolor="none", edgecolor=DATA_C,
                   zorder=3, label="CA-125 < 35 (left out)")
        _marks(ax, p.schedule)
        ax.axhline(C.CA125_CUTOFF, color=INK, lw=0.8, ls="-.", zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(5000, op.ca125.max() * 2))
        rel = r.true_recurrence_rel
        ax.set_title(f"{pid}: " + ("no recurrence in 36 mo" if np.isinf(rel) else
                                  f"true recurrence {rel:.1f} mo after ACT"), fontsize=10)
        ax.set_xlabel("Months")
    for a in axes[:, 0]:
        a.set_ylabel("CA-125 (U/mL, log)")
    axes[0, 0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle(f"RMSE calibration of '{fit_model.name}' on data from '{model.name}'. "
                 "Dashed = treatment starts, solid = surgery, dash-dot = 35 U/mL", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "1_calibration_examples.png", dpi=160)
    plt.close(fig)


def fig_recovery(model: BaseModel, truth, fits):
    m = fits.merge(truth, on="patient", suffixes=("", "_true"))
    panels = [(f"{k}_{ph}", f"{k} {ph}") for ph in ("pre", "post") for k in model.param_names]
    fig, axes = plt.subplots(2, len(model.param_names), figsize=(4.2 * len(model.param_names), 7.5),
                             squeeze=False)
    for ax, (k, lab) in zip(axes.flat, panels):
        ok = m[k].notna()
        x, y = m.loc[ok, k + "_true"], m.loc[ok, k]
        ax.scatter(x, y, s=24, color=DATA_C, edgecolor=BG, lw=0.5)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        rr = np.corrcoef(x, y)[0, 1] if ok.sum() > 2 else np.nan
        ax.set_title(f"{lab}: n={ok.sum()}, r={rr:.2f}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("RMSE fit")
    fig.suptitle("Parameter recovery from noisy CA-125 (diagonal = perfect)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "2_parameter_recovery.png", dpi=160)
    plt.close(fig)


def main(model, fit_model, obs, truth, fits):
    FIG.mkdir(exist_ok=True)
    fig_explainer(model, obs, truth)
    fig_calibration(model, fit_model, obs, truth, fits)
    if model.name == fit_model.name:
        fig_recovery(model, truth, fits)
    print("figures written to", FIG)
