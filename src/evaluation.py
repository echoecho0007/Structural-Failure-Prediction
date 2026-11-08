"""Evaluate reloaded models on the held-out test split (touched only here).

Outputs: results/metrics.csv, results/experiment_results.json, comparison/diagnostic plots.
Also runs an out-of-distribution (extrapolation) experiment on applied force.

Usage: python -m src.evaluation [--skip-ood]
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,  # noqa: E402
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.analytical_solver import solve_dataframe  # noqa: E402
from src.predictor import Predictor  # noqa: E402
from src.preprocessing import FeatureScaler, load_dataset, log_targets, ood_split, split  # noqa: E402
from src.visualization import PALETTE, savefig, set_style  # noqa: E402

REG_MODELS = ["random_forest", "boosting", "neural_network", "power_law_baseline", "mlp_raw_features"]
ML_MODELS = ["random_forest", "boosting", "neural_network"]
QTY = {"max_stress_Pa": ("Max. bending stress", "MPa", 1e-6), "tip_deflection_m": ("Tip deflection", "mm", 1e3),
       "max_strain": ("Max. strain", "microstrain", 1e6), "factor_of_safety": ("Factor of safety", "-", 1.0)}


def regression_metrics(y_true, y_pred) -> dict:
    rel = np.abs(y_pred - y_true) / np.abs(y_true)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
            "R2": float(r2_score(y_true, y_pred)),
            "R2_log10": float(r2_score(np.log10(y_true), np.log10(np.clip(y_pred, 1e-300, None)))),
            "MAPE_%": float(100 * rel.mean()), "median_APE_%": float(100 * np.median(rel)),
            "P95_APE_%": float(100 * np.percentile(rel, 95)), "max_APE_%": float(100 * rel.max())}


def classification_metrics(y_true, y_pred, proba=None) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision_score(y_true, y_pred)),
           "recall": float(recall_score(y_true, y_pred)), "F1": float(f1_score(y_true, y_pred)),
           "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}
    if proba is not None:
        out["ROC_AUC"] = float(roc_auc_score(y_true, proba))
    return out


# --------------------------------------------------------------------------- plots
def plot_pred_vs_actual(test, preds, labels, target, name, caption):
    set_style()
    lab, unit, s = QTY[target]
    fig, axes = plt.subplots(1, len(ML_MODELS), figsize=(12, 3.9), sharex=True, sharey=True)
    yt = test[target].values * s
    for ax, m in zip(axes, ML_MODELS):
        yp = preds[m][target] * s
        ax.scatter(yt, yp, s=3, alpha=0.35, color=PALETTE[labels[m]], rasterized=True)
        lo, hi = yt.min() * 0.8, yt.max() * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (analytical)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r2 = r2_score(yt, yp)
        med = 100 * np.median(np.abs(yp - yt) / yt)
        ax.set_title(labels[m])
        ax.text(0.04, 0.95, f"$R^2$ = {r2:.5f}\nmedian |err| = {med:.2f} %", transform=ax.transAxes, va="top",
                fontsize=8.5, bbox=dict(fc="white", ec="0.8", alpha=0.9))
        ax.set_xlabel(f"Analytical {lab.lower()} [{unit}]")
    axes[0].set_ylabel(f"ML-predicted {lab.lower()} [{unit}]")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, name, caption)


def plot_residuals(test, preds, labels):
    set_style()
    fig, axes = plt.subplots(2, len(ML_MODELS), figsize=(12, 6.2), sharey="row")
    for r, target in enumerate(["max_stress_Pa", "tip_deflection_m"]):
        lab, unit, s = QTY[target]
        yt = test[target].values
        for c, m in enumerate(ML_MODELS):
            ax = axes[r, c]
            rel = 100 * (preds[m][target] - yt) / yt
            ax.scatter(yt * s, rel, s=3, alpha=0.3, color=PALETTE[labels[m]], rasterized=True)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(f"Analytical {lab.lower()} [{unit}]")
            if c == 0:
                ax.set_ylabel("Relative residual [%]\n(pred - true)/true")
            if r == 0:
                ax.set_title(labels[m])
    lim = max(abs(np.array(ax.get_ylim())).max() for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(-min(lim, 40), min(lim, 40))
    fig.tight_layout()
    savefig(fig, "11_residuals",
            "Relative residuals versus the analytical value for stress (top) and tip deflection (bottom) on the test "
            "set (axis clipped at +/-40 %). Tree ensembles show larger, heteroscedastic errors at the extremes of the "
            "target range, where training data are sparse; the MLP residuals are small and nearly unbiased.")


def plot_model_comparison(metrics_df, labels):
    set_style()
    models = [m for m in REG_MODELS if m != "power_law_baseline"]  # exact to machine precision; listed in tables
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    width = 0.38
    xs = np.arange(len(models))
    for i, target in enumerate(["max_stress_Pa", "tip_deflection_m"]):
        sub = metrics_df[(metrics_df.target == target)].set_index("model").loc[models]
        off = (i - 0.5) * width
        axes[0].bar(xs + off, 1 - sub["R2"], width, label=QTY[target][0], color=["#2a6f97", "#e07a1f"][i])
        axes[1].bar(xs + off, sub["median_APE_%"], width, color=["#2a6f97", "#e07a1f"][i])
        axes[2].bar(xs + off, sub["P95_APE_%"], width, color=["#2a6f97", "#e07a1f"][i])
    for ax, t in zip(axes, [r"$1 - R^2$ (lower is better)", "Median abs. % error", "95th-percentile abs. % error"]):
        ax.set_yscale("log"); ax.set_title(t)
        ax.set_xticks(xs); ax.set_xticklabels([labels[m] for m in models], rotation=30, ha="right")
    axes[0].legend()
    fig.tight_layout()
    savefig(fig, "12_model_comparison",
            "Test-set comparison of all regression models (log scale). 1 - R^2 is shown so that differences between "
            "near-perfect models remain visible. 'MLP (raw inputs)' is an ablation without the log10 input transform. "
            "The power-law (log-OLS) baseline is omitted because it is exact to machine precision (errors ~1e-12 %), "
            "the analytical model being a product of powers of the inputs.")


def plot_confusion(test, cls_results, labels):
    set_style()
    keys = ML_MODELS + ["derived_from_best_regressor"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, k in zip(axes, keys):
        cm = np.array(cls_results[k]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Safe", "Yield\nexceeded"]); ax.set_yticklabels(["Safe", "Yield\nexceeded"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Analytical (true)")
        title = labels.get(k, "FoS < 1 from best\nstress regressor")
        ax.set_title(f"{title}\nF1 = {cls_results[k]['F1']:.4f}", fontsize=9.5)
        ax.grid(False)
    fig.tight_layout()
    savefig(fig, "13_confusion_matrices",
            "Confusion matrices for yield classification on the test set (N = %d). The first three panels are direct "
            "classifiers trained on the six inputs; the last panel thresholds the FoS implied by the best stress "
            "regressor (FoS = sigma_y / sigma_pred < 1)." % len(test))


def plot_fos_error(test, preds, labels):
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fos = test["factor_of_safety"].values
    bins = np.logspace(np.log10(fos.min()), np.log10(fos.max()), 25)
    for m in ML_MODELS:
        wrong = (preds[m]["factor_of_safety"] < 1) != (fos < 1)
        idx = np.digitize(fos, bins)
        rate = [wrong[idx == i].mean() if (idx == i).sum() > 10 else np.nan for i in range(1, len(bins))]
        ax.plot(np.sqrt(bins[1:] * bins[:-1]), 100 * np.array(rate), "o-", ms=3, label=labels[m], color=PALETTE[labels[m]])
    ax.axvline(1, color="crimson", ls=":", label="FoS = 1")
    ax.set_xscale("log"); ax.set_xlabel("Analytical factor of safety [-]")
    ax.set_ylabel("Misclassification rate [%]"); ax.legend()
    fig.tight_layout()
    savefig(fig, "14_fos_misclassification",
            "Rate at which the regression-derived yield classification is wrong, binned by the true factor of safety. "
            "Errors concentrate in a narrow band around FoS = 1, where a small relative stress error flips the class; "
            "beams far from the yield boundary are always classified correctly.")


# --------------------------------------------------------------------------- OOD experiment
def ood_experiment(df, predictor_backends, mlp_hidden):
    """Train on F <= 85th percentile, test on larger forces (extrapolation)."""
    from src.models import TorchMLP, fit_boosting, make_boosting, make_random_forest, make_sklearn_mlp
    tr, te, thr = ood_split(df, "force_N", 0.85)
    Xtr, Xte = tr[config.FEATURES].values, te[config.FEATURES].values
    ytr = log_targets(tr)
    out = {"split_feature": "force_N", "threshold_N": thr, "n_train": len(tr), "n_test": len(te), "models": {}}
    preds = {}
    preds["random_forest"] = make_random_forest("reg", n_estimators=100, min_samples_leaf=2).fit(Xtr, ytr).set_params(n_jobs=1).predict(Xte)
    be = predictor_backends["boosting"]
    rng = np.random.default_rng(config.SEED)
    vmask = rng.random(len(Xtr)) < 0.1
    gbs = [fit_boosting(make_boosting(be, "reg"), be, Xtr[~vmask], ytr[~vmask, j], Xtr[vmask], ytr[vmask, j]) for j in range(2)]
    preds["boosting"] = np.column_stack([m.predict(Xte) for m in gbs])
    fs, ys = FeatureScaler(True).fit(Xtr), StandardScaler().fit(ytr)
    if predictor_backends["mlp"] == "torch":
        mlp = TorchMLP(hidden=tuple(mlp_hidden)).fit(fs.transform(Xtr[~vmask]), ys.transform(ytr[~vmask]),
                                                     fs.transform(Xtr[vmask]), ys.transform(ytr[vmask]))
    else:
        mlp = make_sklearn_mlp("reg", hidden=tuple(mlp_hidden)).fit(fs.transform(Xtr), ys.transform(ytr))
    preds["neural_network"] = ys.inverse_transform(np.asarray(mlp.predict(fs.transform(Xte))).reshape(len(Xte), -1))
    preds["power_law_baseline"] = LinearRegression().fit(np.log10(Xtr), ytr).predict(np.log10(Xte))
    for m, p in preds.items():
        out["models"][m] = {t: regression_metrics(te[t].values, 10 ** p[:, j]) for j, t in enumerate(config.REG_TARGETS)}
    return out, te, {m: 10 ** p for m, p in preds.items()}


def plot_ood(te, preds, thr, labels):
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    F = te["force_N"].values
    order = np.argsort(F)
    for m in ["random_forest", "boosting", "neural_network"]:
        rel = 100 * (preds[m][:, 0] - te["max_stress_Pa"].values) / te["max_stress_Pa"].values
        bins = np.linspace(F.min(), F.max(), 12)
        idx = np.digitize(F, bins)
        mid = [F[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 5]
        val = [np.median(rel[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() > 5]
        ax.plot(mid, val, "o-", label=labels[m], color=PALETTE[labels[m]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Applied force F [N] (training used F <= {thr:.0f} N)")
    ax.set_ylabel("Median relative stress error [%]")
    ax.legend()
    del order
    fig.tight_layout()
    savefig(fig, "15_extrapolation_force",
            "Out-of-distribution test: models retrained only on beams with F <= %.0f N and evaluated on larger forces. "
            "Tree ensembles cannot predict outside the range of training targets (piecewise-constant extrapolation), "
            "so their stress predictions saturate and the error grows with F; the MLP on log inputs extrapolates far "
            "better because the log-space mapping is nearly linear." % thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ood", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    df = load_dataset()
    s = split(df)
    test = s.df_test
    # sanity: test targets must equal the analytical solver output
    ref = solve_dataframe(test)
    assert np.allclose(ref["max_stress_Pa"], test["max_stress_Pa"]), "test targets inconsistent with solver"

    P = Predictor()  # reload all saved models from disk
    labels = P.model_labels
    preds = {m: P.predict(test, m) for m in REG_MODELS}

    rows = []
    for m in REG_MODELS:
        for t in QTY:
            rows.append({"model": m, "model_label": labels[m], "target": t,
                         **regression_metrics(test[t].values, preds[m][t])})
    mdf = pd.DataFrame(rows)

    # classification: direct classifiers + derived from each regressor
    y = test["yield_exceeded"].values
    cls = {}
    for m in ML_MODELS:
        lab, prob = P.classify(test, m)
        cls[m] = classification_metrics(y, lab, prob)
    for m in REG_MODELS:
        cls[f"derived_{m}"] = classification_metrics(y, preds[m]["yield_exceeded"], -np.log(preds[m]["factor_of_safety"]))
    best_reg = min(ML_MODELS, key=lambda m: mdf[(mdf.model == m) & (mdf.target == "max_stress_Pa")]["RMSE"].item())
    cls["derived_from_best_regressor"] = dict(cls[f"derived_{best_reg}"], source_model=best_reg)
    # misclassifications near boundary
    fos = test["factor_of_safety"].values
    wrong = preds[best_reg]["yield_exceeded"] != y
    cls["derived_from_best_regressor"]["true_FoS_range_of_errors"] = (
        [float(fos[wrong].min()), float(fos[wrong].max())] if wrong.any() else None)

    # inference timing (analytical vs ML) for 10k samples - informational only
    timing = {}
    Xb = test[config.FEATURES]
    t1 = time.perf_counter(); solve_dataframe(Xb); timing["analytical_solver_ms"] = 1e3 * (time.perf_counter() - t1)
    for m in ML_MODELS:
        t1 = time.perf_counter(); P.predict(Xb, m); timing[f"{m}_ms"] = 1e3 * (time.perf_counter() - t1)

    best = {}
    for t in QTY:
        sub = mdf[(mdf.target == t) & mdf.model.isin(ML_MODELS)]
        best[t] = sub.loc[sub["RMSE"].idxmin(), "model"]
    best["classification_direct"] = max(ML_MODELS, key=lambda m: cls[m]["F1"])

    results = {"n_test": len(test), "n_train": len(s.df_train), "n_val": len(s.df_val),
               "backends": P.backends, "model_labels": labels,
               "regression": {m: {t: mdf[(mdf.model == m) & (mdf.target == t)].drop(columns=["model", "model_label", "target"]).iloc[0].to_dict()
                                  for t in QTY} for m in REG_MODELS},
               "classification": cls, "best_model_per_task": best, "inference_time_ms_test_set": timing,
               "registry": P.registry}

    # plots
    plot_pred_vs_actual(test, preds, labels, "max_stress_Pa", "09_pred_vs_actual_stress",
                        "Predicted versus analytical maximum bending stress on the held-out test set (N = %d, log-log axes). "
                        "Points on the dashed line are perfect predictions." % len(test))
    plot_pred_vs_actual(test, preds, labels, "tip_deflection_m", "10_pred_vs_actual_deflection",
                        "Predicted versus analytical maximum tip deflection on the held-out test set (N = %d, log-log axes)." % len(test))
    plot_residuals(test, preds, labels)
    plot_model_comparison(mdf, labels)
    plot_confusion(test, cls, labels)
    plot_fos_error(test, preds, labels)

    if not args.skip_ood:
        mlp_hidden = json.loads((config.MODEL_DIR / "neural_network/config.json").read_text())["best_params"]["hidden"]
        ood, te, opreds = ood_experiment(df, P.backends, mlp_hidden)
        results["extrapolation_force"] = ood
        plot_ood(te, opreds, ood["threshold_N"], labels)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    mdf.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    (config.RESULTS_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(mdf[mdf.target.isin(config.REG_TARGETS)][["model_label", "target", "MAE", "RMSE", "R2", "median_APE_%", "P95_APE_%"]]
          .to_string(index=False))
    print("Classification F1:", {k: round(v["F1"], 4) for k, v in cls.items()})
    if "extrapolation_force" in results:
        print("OOD stress median APE %:", {m: round(v["max_stress_Pa"]["median_APE_%"], 2)
                                            for m, v in results["extrapolation_force"]["models"].items()})
    print(f"Evaluation finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
