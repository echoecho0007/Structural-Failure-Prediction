"""Plot style, EDA figures and analytical structural visualisations.

Every figure is saved at 300 dpi (PNG) together with a caption in results/plots/captions.json.
Structural field plots are ANALYTICAL Euler-Bernoulli fields, not finite-element results.

Usage: python -m src.visualization   (EDA + structural figures)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from src import analytical_solver as an  # noqa: E402
from src import config  # noqa: E402

CAPTIONS_PATH = config.PLOT_DIR / "captions.json"
PALETTE = {"Random Forest": "#2a6f97", "HistGradBoost": "#e07a1f", "XGBoost": "#e07a1f",
           "MLP (sklearn)": "#3a9d5d", "MLP (PyTorch)": "#3a9d5d", "Power-law (log-OLS)": "#7a7a7a",
           "MLP (raw inputs)": "#a05195", "Analytical": "#111111"}


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def savefig(fig, name: str, caption: str):
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PLOT_DIR / f"{name}.png")
    plt.close(fig)
    caps = json.loads(CAPTIONS_PATH.read_text()) if CAPTIONS_PATH.exists() else {}
    caps[name] = caption
    CAPTIONS_PATH.write_text(json.dumps(caps, indent=2))


def _disp(df, col):
    return df[col] * config.DISPLAY_SCALE.get(col, 1.0)


# ============================================================================ EDA
def eda(df: pd.DataFrame) -> dict:
    set_style()
    # 1. feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, c in zip(axes.flat, config.FEATURES):
        ax.hist(_disp(df, c), bins=50, color="#2a6f97", alpha=0.85, edgecolor="white", linewidth=0.3)
        lo, hi = config.PARAM_RANGES[c]
        s = config.DISPLAY_SCALE.get(c, 1.0)
        ax.axvline(lo * s, color="k", ls=":", lw=0.8)
        ax.axvline(hi * s, color="k", ls=":", lw=0.8)
        ax.set_xlabel(config.FEATURE_LABELS[c])
        ax.set_ylabel("Count")
    fig.tight_layout()
    savefig(fig, "01_feature_distributions",
            "Distributions of the six input parameters after rejection sampling (N = %d). Dotted lines mark "
            "the uniform sampling bounds. Non-uniform shapes (e.g. fewer thin beams, more thick/wide beams) are "
            "caused by the physical-validity constraints (slenderness, small deflection, stress <= 3 sigma_y)." % len(df))

    # 2. target distributions
    tg = [("max_stress_Pa", r"$\sigma_{max}$ [MPa]", 1e-6), ("tip_deflection_m", r"$\delta_{max}$ [mm]", 1e3),
          ("max_strain", r"$\varepsilon_{max}$ [$\mu\varepsilon$]", 1e6), ("factor_of_safety", "FoS [-]", 1)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (c, lab, s) in zip(axes, tg):
        v = df[c] * s
        ax.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()), 50), color="#e07a1f",
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("Count")
    axes[3].axvline(1, color="crimson", lw=1.2, label="FoS = 1")
    axes[3].legend()
    fig.tight_layout()
    savefig(fig, "02_target_distributions",
            "Distributions of the analytical response quantities on logarithmic axes. Stress and deflection span "
            "two to three orders of magnitude, motivating the log10 target transform used for training. "
            "%.1f %% of samples exceed the yield criterion (FoS < 1)." % (100 * df["yield_exceeded"].mean()))

    # 3. correlation heatmaps (Pearson on raw values, Spearman rank)
    cols = config.FEATURES + ["max_stress_Pa", "tip_deflection_m", "max_strain", "factor_of_safety"]
    short = ["L", "b", "h", "F", "E", r"$\sigma_y$", r"$\sigma_{max}$", r"$\delta_{max}$", r"$\varepsilon_{max}$", "FoS"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, meth in zip(axes, ["pearson", "spearman"]):
        cm = df[cols].corr(method=meth)
        sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                    xticklabels=short, yticklabels=short, cbar_kws={"shrink": 0.75}, annot_kws={"size": 7})
        ax.set_title(f"{meth.capitalize()} correlation")
    fig.tight_layout()
    savefig(fig, "03_correlation_heatmap",
            "Pearson (linear) and Spearman (rank) correlation matrices of inputs and responses. Inputs are weakly "
            "correlated with each other only through the rejection-sampling constraints. Force and length correlate "
            "positively with stress; thickness negatively, consistent with sigma = 6FL/(bh^2).")

    # 4. feature-response relationships
    sub = df.sample(3000, random_state=config.SEED)
    pairs = [("force_N", "max_stress_Pa", 1e-6, r"$\sigma_{max}$ [MPa]"), ("thickness_m", "max_stress_Pa", 1e-6, r"$\sigma_{max}$ [MPa]"),
             ("length_m", "tip_deflection_m", 1e3, r"$\delta_{max}$ [mm]"), ("youngs_modulus_Pa", "tip_deflection_m", 1e3, r"$\delta_{max}$ [mm]")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (x, y, s, lab) in zip(axes, pairs):
        sc = ax.scatter(_disp(sub, x), sub[y] * s, c=sub["yield_exceeded"], cmap="coolwarm", s=3, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel(config.FEATURE_LABELS[x])
        ax.set_ylabel(lab)
    fig.legend(sc.legend_elements()[0], ["Safe", "Yield exceeded"], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(fig, "04_feature_relationships",
            "Marginal relationships between selected inputs and responses (3,000 random samples, colour = yield class). "
            "The large scatter at fixed input value shows that each response depends jointly on several parameters.")

    # 5. outlier analysis (IQR rule on log10 targets)
    logs = np.log10(df[["max_stress_Pa", "tip_deflection_m", "max_strain", "factor_of_safety"]])
    q1, q3 = logs.quantile(0.25), logs.quantile(0.75)
    iqr = q3 - q1
    out = ((logs < q1 - 1.5 * iqr) | (logs > q3 + 1.5 * iqr)).sum()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.boxplot(data=logs.rename(columns=dict(zip(logs.columns, [r"$\log_{10}\sigma_{max}$", r"$\log_{10}\delta_{max}$",
                                                                  r"$\log_{10}\varepsilon_{max}$", r"$\log_{10}$FoS"]))),
                ax=ax, color="#9ecae1", fliersize=1.5)
    ax.set_ylabel("log10 value (SI units)")
    fig.tight_layout()
    savefig(fig, "05_outlier_boxplots",
            "Box plots of log10-transformed responses. IQR-rule outlier counts: " +
            ", ".join(f"{k.split('_')[0] if k != 'factor_of_safety' else 'FoS'} = {int(v)}" for k, v in out.items()) +
            ". These are genuine extreme-but-valid beams, not data errors, and were retained.")

    summary = df.describe().T
    summary.to_csv(config.RESULTS_DIR / "dataset_summary_statistics.csv")
    return {"iqr_outliers_log10": {k: int(v) for k, v in out.items()}}


# ============================================================================ structural figures
def beam_diagram():
    set_style()
    p = config.BASELINE
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.add_patch(Rectangle((-0.08, -0.35), 0.08, 0.7, hatch="////", fill=False, lw=1))
    ax.add_patch(Rectangle((0, -0.06), 1.0, 0.12, color="#9ecae1", ec="k", lw=1))
    ax.add_patch(FancyArrowPatch((1.0, 0.45), (1.0, 0.07), arrowstyle="-|>", mutation_scale=18, lw=1.8, color="crimson"))
    ax.text(1.02, 0.35, r"$F$", color="crimson", fontsize=13)
    x = np.linspace(0, 1, 200)
    w = x ** 2 * (3 - x) / 2
    ax.plot(x, -0.06 - 0.22 * w, "k--", lw=1)
    ax.text(0.42, -0.42, "deflected shape\n(exaggerated)", fontsize=8, ha="center", style="italic")
    ax.annotate("", (0, -0.55), (1, -0.55), arrowprops=dict(arrowstyle="<->"))
    ax.text(0.5, -0.63, r"$L$", ha="center", fontsize=12)
    ax.text(-0.04, 0.42, "Fixed support\n(clamped)", ha="center", fontsize=8)
    ax.annotate(r"$\delta_{max}$", (1.0, -0.28), (1.12, -0.18), arrowprops=dict(arrowstyle="->"), fontsize=11)
    # cross-section inset
    ins = ax.inset_axes([0.42, 0.55, 0.16, 0.42])
    ins.add_patch(Rectangle((-0.5, -0.4), 1.0, 0.8, color="#9ecae1", ec="k"))
    ins.plot([-0.6, 0.6], [0, 0], color="k", ls="-.", lw=0.7)
    ins.text(0, -0.58, r"$b$", ha="center")
    ins.text(0.58, 0, r"$h$", va="center")
    ins.text(-0.62, 0.0, "N.A.", fontsize=7, ha="right", va="center")
    ins.set_xlim(-1.2, 0.8); ins.set_ylim(-0.75, 0.6); ins.axis("off")
    ins.set_title("Section", fontsize=8)
    ax.set_xlim(-0.2, 1.45); ax.set_ylim(-0.75, 0.75); ax.set_aspect("equal"); ax.axis("off")
    savefig(fig, "06_beam_diagram",
            "Problem definition: prismatic cantilever of length L with rectangular b x h cross-section, clamped at "
            "x = 0 and loaded by a transverse point force F at the free end. Maximum bending moment M = FL occurs "
            "at the support; maximum deflection at the tip.")


def stress_field_plot(params=None, name="07_bending_stress_field"):
    set_style()
    p = params or config.BASELINE
    L, b, h, F, E, sy = (p[k] for k in config.FEATURES)
    x = np.linspace(0, L, 300)
    y = np.linspace(-h / 2, h / 2, 120)
    Xg, Yg = np.meshgrid(x, y)
    S = an.bending_stress_field(Xg, Yg, F, L, b, h) / 1e6
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], width_ratios=[1.4, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    lim = np.abs(S).max()
    im = ax.pcolormesh(Xg, Yg * 1e3, S, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax.contour(Xg, Yg * 1e3, np.abs(S), levels=[sy / 1e6], colors="k", linewidths=1.2, linestyles="--")
    ax.set_xlabel(r"Position along beam $x$ [m] (fixed end at $x=0$)")
    ax.set_ylabel(r"Distance from neutral axis $y$ [mm]")
    ax.set_title(r"Analytical bending stress $\sigma_{xx}(x,y)=M(x)\,y/I$ (Euler–Bernoulli; not FEA)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(r"$\sigma_{xx}$ [MPa]  (+ tension)")
    ax.grid(False)
    # through-thickness profile at support
    ax2 = fig.add_subplot(gs[1, 0])
    for xf in [0, 0.5, 0.9]:
        ax2.plot(an.bending_stress_field(xf * L, y, F, L, b, h) / 1e6, y * 1e3, label=f"x = {xf:.1f} L")
    ax2.axvline(sy / 1e6, color="crimson", ls=":", lw=1)
    ax2.axvline(-sy / 1e6, color="crimson", ls=":", lw=1, label=r"$\pm\sigma_y$")
    ax2.set_xlabel(r"$\sigma_{xx}$ [MPa]"); ax2.set_ylabel(r"$y$ [mm]")
    ax2.set_title("Through-thickness profile"); ax2.legend()
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x, an.bending_stress_field(x, h / 2, F, L, b, h) / 1e6, color="#2a6f97")
    ax3.axhline(sy / 1e6, color="crimson", ls=":", lw=1, label=r"$\sigma_y$")
    ax3.set_xlabel(r"$x$ [m]"); ax3.set_ylabel(r"Outer-fibre $\sigma_{xx}$ [MPa]")
    ax3.set_title("Along the length"); ax3.legend()
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(x, -an.deflection_profile(x, F, L, b, h, E) * 1e3, color="#3a9d5d")
    ax4.set_xlabel(r"$x$ [m]"); ax4.set_ylabel(r"Deflection $w(x)$ [mm]")
    ax4.set_title("Elastic curve")
    fig.tight_layout()
    sig = an.max_bending_stress(F, L, b, h)
    savefig(fig, name,
            f"Analytical stress and displacement fields for the baseline beam (L = {L} m, b = {b*1e3:.0f} mm, "
            f"h = {h*1e3:.0f} mm, F = {F:.0f} N, E = {E/1e9:.0f} GPa, sigma_y = {sy/1e6:.0f} MPa). "
            f"Peak |sigma| = {sig/1e6:.0f} MPa at the outer fibres of the support (FoS = {sy/sig:.2f}); the dashed "
            "contour, where present, would mark |sigma| = sigma_y. Linear variation through the thickness and along the "
            "length follows directly from M(x) = F(L - x). These are closed-form beam-theory fields, not finite-element results.")


def fos_map(predictor=None, name="08_fos_safety_map"):
    """Factor-of-safety map over (F, h) with the analytical FoS = 1 boundary and optional ML boundaries."""
    set_style()
    p = dict(config.BASELINE)
    F = np.linspace(*config.PARAM_RANGES["force_N"], 220)
    h = np.linspace(*config.PARAM_RANGES["thickness_m"], 220)
    Fg, Hg = np.meshgrid(F, h)
    fos = p["yield_strength_Pa"] / an.max_bending_stress(Fg, p["length_m"], p["width_m"], Hg)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.contourf(Fg, Hg * 1e3, np.log10(fos), levels=40, cmap="RdYlGn")
    cb = fig.colorbar(im, ax=ax)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    cb.set_ticks(np.log10(ticks)); cb.set_ticklabels([str(t) for t in ticks]); cb.set_label("Factor of safety (log scale)")
    ax.contour(Fg, Hg * 1e3, fos, levels=[1], colors="k", linewidths=2)
    handles = [plt.Line2D([], [], color="k", lw=2, label="Analytical FoS = 1")]
    if predictor is not None:
        grid = pd.DataFrame({k: np.full(Fg.size, v) for k, v in p.items()})
        grid["force_N"], grid["thickness_m"] = Fg.ravel(), Hg.ravel()
        styles = {"random_forest": ":", "boosting": "--", "neural_network": "-."}
        for m, ls in styles.items():
            f_ml = predictor.predict(grid, m)["factor_of_safety"].reshape(Fg.shape)
            ax.contour(Fg, Hg * 1e3, f_ml, levels=[1], colors=[PALETTE[predictor.model_labels[m]]], linewidths=1.6, linestyles=ls)
            handles.append(plt.Line2D([], [], color=PALETTE[predictor.model_labels[m]], ls=ls, lw=1.6,
                                      label=f"{predictor.model_labels[m]} FoS = 1"))
    # validity domain mask (small deflection / stress cap)
    d = an.tip_deflection(Fg, p["length_m"], p["width_m"], Hg, p["youngs_modulus_Pa"]) / p["length_m"]
    invalid = (d > config.CONSTRAINTS["max_deflection_ratio"]) | (1 / fos > config.CONSTRAINTS["max_stress_over_yield"])
    with plt.rc_context({"hatch.color": "#555555", "hatch.linewidth": 0.5}):
        ax.contourf(Fg, Hg * 1e3, invalid.astype(float), levels=[0.5, 1.5], colors="none", hatches=["xx"])
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="none", hatch="xx", ec="#555555", label="Outside training domain"))
    ax.text(700, 44, "SAFE", fontsize=13, weight="bold", color="#1b5e20")
    ax.text(6200, 45, "YIELD\nEXCEEDED", fontsize=11, weight="bold", color="#7f0000", ha="center")
    ax.set_xlabel(r"Applied force $F$ [N]"); ax.set_ylabel(r"Thickness $h$ [mm]")
    ax.set_title(f"FoS map (L = {p['length_m']} m, b = {p['width_m']*1e3:.0f} mm, "
                 f"$\\sigma_y$ = {p['yield_strength_Pa']/1e6:.0f} MPa)")
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.grid(False)
    savefig(fig, name,
            "Factor-of-safety map over applied force and thickness with the other parameters fixed at baseline. The "
            "solid black curve is the analytical yield boundary FoS = 1 (h = sqrt(6FL/(b sigma_y))); coloured curves "
            "are the boundaries implied by each ML stress model. Hatched area: combinations excluded from the training "
            "data by the validity constraints, where ML predictions are extrapolations.")


def main():
    from src.preprocessing import load_dataset
    df = load_dataset()
    info = eda(df)
    beam_diagram()
    stress_field_plot()
    try:
        from src.predictor import Predictor
        fos_map(Predictor())
    except FileNotFoundError:
        fos_map(None)
    print("EDA + structural figures saved.", info)


if __name__ == "__main__":
    main()
