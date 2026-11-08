"""Engineering sensitivity analysis.

1. One-at-a-time (OAT) sweeps around the baseline beam for the five required effects
   (F->stress, h->stress, L->deflection, E->deflection, sigma_y->FoS) plus b->stress.
   Analytical curve vs each ML model; the part of each sweep outside the training domain
   (validity constraints) is shaded.
2. Local log-log slopes (elasticities) d log y / d log x estimated from each model and
   compared with the exact exponents of the beam formulas.
3. Monotonicity check: does each model reproduce the sign of the physical trend?
4. Global variance shares in log space from the exact power-law structure.

Usage: python -m src.sensitivity_analysis
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.analytical_solver import solve_dataframe  # noqa: E402
from src.data_generation import validity_mask  # noqa: E402
from src.predictor import Predictor  # noqa: E402
from src.preprocessing import load_dataset  # noqa: E402
from src.visualization import PALETTE, savefig, set_style  # noqa: E402

ML = ["random_forest", "boosting", "neural_network"]
# (swept feature, response, exact exponent, display info)
SWEEPS = [
    ("force_N", "max_stress_Pa", 1.0, "Applied force $F$ [N]", r"$\sigma_{max}$ [MPa]", 1, 1e-6),
    ("thickness_m", "max_stress_Pa", -2.0, "Thickness $h$ [mm]", r"$\sigma_{max}$ [MPa]", 1e3, 1e-6),
    ("width_m", "max_stress_Pa", -1.0, "Width $b$ [mm]", r"$\sigma_{max}$ [MPa]", 1e3, 1e-6),
    ("length_m", "tip_deflection_m", 3.0, "Length $L$ [m]", r"$\delta_{max}$ [mm]", 1, 1e3),
    ("youngs_modulus_Pa", "tip_deflection_m", -1.0, "Young's modulus $E$ [GPa]", r"$\delta_{max}$ [mm]", 1e-9, 1e3),
    ("yield_strength_Pa", "factor_of_safety", 1.0, r"Yield strength $\sigma_y$ [MPa]", "Factor of safety [-]", 1e-6, 1),
]
EXPONENTS = {"max_stress_Pa": {"length_m": 1, "width_m": -1, "thickness_m": -2, "force_N": 1, "youngs_modulus_Pa": 0, "yield_strength_Pa": 0},
             "tip_deflection_m": {"length_m": 3, "width_m": -1, "thickness_m": -3, "force_N": 1, "youngs_modulus_Pa": -1, "yield_strength_Pa": 0}}


def sweep_frame(feature, n=120, base=None):
    base = base or config.BASELINE
    lo, hi = config.PARAM_RANGES[feature]
    df = pd.DataFrame({k: np.full(n, v) for k, v in base.items()})[config.FEATURES]
    df[feature] = np.linspace(lo, hi, n)
    return df


def main():
    set_style()
    P = Predictor()
    labels = P.model_labels
    results = {"baseline": config.BASELINE, "sweeps": {}}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for ax, (feat, resp, expo, xl, yl, xs, ys) in zip(axes.flat, SWEEPS):
        df = sweep_frame(feat)
        ref = solve_dataframe(df)
        valid = validity_mask(df, ref)
        x = df[feat].values
        ax.plot(x * xs, ref[resp] * ys, color="k", lw=2.4, label="Analytical", zorder=5)
        entry = {"exact_exponent": expo, "valid_fraction_of_sweep": float(valid.mean()), "models": {}}
        for m, ls in zip(ML, [":", "--", "-."]):
            pr = P.predict(df, m)[resp]
            ax.plot(x * xs, pr * ys, ls=ls, lw=1.8, color=PALETTE[labels[m]], label=labels[m], zorder=6)
            v = valid & np.isfinite(pr)
            slope = float(np.polyfit(np.log(x[v]), np.log(pr[v]), 1)[0]) if v.sum() > 5 else float("nan")
            d = np.diff(pr[v])
            mono = float(np.mean(np.sign(d) == np.sign(expo))) if len(d) else float("nan")
            rel = np.abs(pr[v] - ref[resp][v]) / ref[resp][v]
            entry["models"][m] = {"loglog_slope": slope, "slope_error": slope - expo,
                                  "fraction_steps_correct_sign": mono, "max_rel_error_%": float(100 * rel.max()),
                                  "mean_rel_error_%": float(100 * rel.mean())}
        # shade invalid parts
        if (~valid).any():
            xv = x * xs
            ylim = ax.get_ylim()
            ax.fill_between(xv, *ylim, where=~valid, color="0.85", alpha=0.6, step="mid", label="Outside training domain", zorder=0)
            ax.set_ylim(ylim)
        if resp == "factor_of_safety":
            ax.axhline(1, color="crimson", ls=":", lw=1)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"exact exponent = {expo:+g}", fontsize=9.5)
        results["sweeps"][f"{feat}->{resp}"] = entry
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "16_sensitivity_sweeps",
            "One-at-a-time sensitivity sweeps about the baseline beam (L = 1 m, b = 50 mm, h = 40 mm, F = 2 kN, "
            "E = 200 GPa, sigma_y = 250 MPa). Black: analytical solution; coloured: ML predictions. Grey bands mark "
            "parts of the sweep excluded from the training data (validity constraints), where the models extrapolate.")

    # ---- elasticity comparison plot
    fig, ax = plt.subplots(figsize=(8, 3.8))
    keys = list(results["sweeps"].keys())
    xs_ = np.arange(len(keys))
    for i, m in enumerate(ML):
        vals = [results["sweeps"][k]["models"][m]["loglog_slope"] for k in keys]
        ax.bar(xs_ + (i - 1) * 0.25, vals, 0.25, color=PALETTE[labels[m]], label=labels[m])
    ax.scatter(xs_, [results["sweeps"][k]["exact_exponent"] for k in keys], marker="_", s=900, color="k", lw=2.5,
               label="Exact exponent", zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs_)
    short = {"force_N": "F", "thickness_m": "h", "width_m": "b", "length_m": "L", "youngs_modulus_Pa": "E",
             "yield_strength_Pa": r"$\sigma_y$", "max_stress_Pa": r"$\sigma_{max}$", "tip_deflection_m": r"$\delta_{max}$",
             "factor_of_safety": "FoS"}
    ax.set_xticklabels([" → ".join(short[p] for p in k.split("->")) for k in keys], fontsize=9)
    ax.set_ylabel(r"Log-log slope $\partial\ln y/\partial\ln x$")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    savefig(fig, "17_elasticity_comparison",
            "Elasticities (log-log slopes) recovered from each model along the OAT sweeps (valid domain only) compared "
            "with the exact exponents of sigma = 6FL/(bh^2), delta = 4FL^3/(Ebh^3) and FoS = sigma_y/sigma.")

    # ---- global variance shares in log space (exact power-law => additive in log space)
    df = load_dataset()
    lx = np.log10(df[config.FEATURES])
    shares = {}
    for t, ex in EXPONENTS.items():
        ly = np.log10(df[t])
        tot = ly.var()
        # covariance decomposition: Var(ly) = sum_i a_i Cov(log x_i, ly); exact and sums to 1
        # even when inputs are correlated (here correlation comes from the rejection sampling).
        shares[t] = {f: float(ex[f] * np.cov(lx[f], ly)[0, 1] / tot) for f in config.FEATURES}
        shares[t]["_sum"] = float(sum(shares[t].values()))
    results["log_variance_shares"] = shares
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.38
    xx = np.arange(len(config.FEATURES))
    for i, t in enumerate(EXPONENTS):
        ax.bar(xx + (i - 0.5) * w, [shares[t][f] for f in config.FEATURES], w, label=t.replace("_Pa", "").replace("_m", ""),
               color=["#2a6f97", "#e07a1f"][i])
    ax.set_xticks(xx); ax.set_xticklabels(["L", "b", "h", "F", "E", r"$\sigma_y$"])
    ax.set_ylabel("Share of Var(log10 y)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "18_global_variance_shares",
            "Global sensitivity over the dataset: share of the variance of log10(response) attributable to each input, "
            "computed with the covariance decomposition S_i = a_i Cov(log10 x_i, log10 y) / Var(log10 y), where a_i are the "
            "exact exponents. The shares sum to 1 and remain valid although rejection sampling correlates the inputs.")

    out = config.RESULTS_DIR / "sensitivity_results.json"
    out.write_text(json.dumps(results, indent=2))
    for k, e in results["sweeps"].items():
        print(k, {m: round(v["loglog_slope"], 3) for m, v in e["models"].items()}, "exact", e["exact_exponent"])
    print("variance shares:", {t: {f: round(v, 3) for f, v in s.items()} for t, s in shares.items()})


if __name__ == "__main__":
    main()
