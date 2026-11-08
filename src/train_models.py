"""Train, tune (on the validation split) and save all models.

Regression (targets = log10 stress, log10 tip deflection):
    random_forest   - multi-output RandomForestRegressor
    boosting        - XGBoost (or sklearn HistGradientBoosting fallback), one model per target
    neural_network  - MLP (PyTorch or sklearn fallback), multi-output, log-standardised inputs
    power_law_baseline - ordinary least squares on log10 features (physics-informed reference)
    mlp_raw_features   - ablation: same MLP on linearly standardised (non-log) inputs

Classification (yield_exceeded): RF, boosting and MLP classifiers trained directly on inputs.

Usage: python -m src.train_models [--backend auto|sklearn|native] [--fast]
"""
from __future__ import annotations

import argparse
import itertools
import time

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.models import (TorchMLP, fit_boosting, make_boosting, make_random_forest, make_sklearn_mlp,
                        resolve_backend, save_json)
from src.preprocessing import FeatureScaler, load_dataset, log_targets, split

MD = config.MODEL_DIR


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def tune_random_forest(Xtr, ytr, Xv, yv, fast):
    grid = [dict(max_features=mf, min_samples_leaf=ml) for mf, ml in itertools.product([1.0, 0.6], [2, 4])]
    if fast:
        grid = grid[:1]
    log = []
    for p in grid:
        m = make_random_forest("reg", n_estimators=40 if fast else 100, **p).fit(Xtr, ytr)
        log.append({**p, "val_rmse_log10": _rmse(m.predict(Xv), yv)})
    best = min(log, key=lambda r: r["val_rmse_log10"])
    params = {k: best[k] for k in ("max_features", "min_samples_leaf")}
    model = make_random_forest("reg", n_estimators=40 if fast else 100, **params).fit(Xtr, ytr)
    return model, params, log


def tune_boosting(backend, Xtr, ytr, Xv, yv, fast):
    """One model per target; the hyper-parameters are shared and chosen on mean validation RMSE."""
    if backend == "xgboost":
        grid = [dict(learning_rate=lr, max_depth=d) for lr, d in itertools.product([0.05, 0.1], [6, 8])]
    else:
        grid = [dict(learning_rate=lr, max_leaf_nodes=n) for lr, n in itertools.product([0.05, 0.1], [31, 63])]
    if fast:
        grid = grid[:1]
    log, fitted = [], {}
    for i, p in enumerate(grid):
        ms = [fit_boosting(make_boosting(backend, "reg", **p), backend, Xtr, ytr[:, j], Xv, yv[:, j])
              for j in range(ytr.shape[1])]
        pred = np.column_stack([m.predict(Xv) for m in ms])
        log.append({**p, "val_rmse_log10": _rmse(pred, yv)})
        fitted[i] = ms
    i_best = int(np.argmin([r["val_rmse_log10"] for r in log]))
    return fitted[i_best], grid[i_best], log


def tune_mlp(backend, Xtr_s, ytr_s, Xv_s, yv_s, y_scaler, yv, fast):
    grid = [dict(hidden=h, lr=1e-3) for h in [(128, 128, 64), (256, 256, 128)]]
    if fast:
        grid = grid[:1]
    log, fitted = [], []
    for p in grid:
        t0 = time.time()
        if backend == "torch":
            m = TorchMLP(hidden=p["hidden"], lr=p["lr"], max_epochs=60 if fast else 400).fit(Xtr_s, ytr_s, Xv_s, yv_s)
        else:
            m = make_sklearn_mlp("reg", hidden=p["hidden"], lr=p["lr"])
            if fast:
                m.set_params(max_iter=60)
            m.fit(Xtr_s, ytr_s)
        pred = y_scaler.inverse_transform(np.asarray(m.predict(Xv_s)).reshape(len(Xv_s), -1))
        log.append({"hidden": list(p["hidden"]), "lr": p["lr"], "val_rmse_log10": _rmse(pred, yv),
                    "train_seconds": round(time.time() - t0, 1)})
        fitted.append(m)
    i = int(np.argmin([r["val_rmse_log10"] for r in log]))
    return fitted[i], grid[i], log


def save_mlp(model, backend, path_stem):
    if backend == "torch":
        model.save(path_stem.with_suffix(".pt"))
        return str(path_stem.with_suffix(".pt").relative_to(config.ROOT))
    joblib.dump(model, path_stem.with_suffix(".joblib"), compress=3)
    return str(path_stem.with_suffix(".joblib").relative_to(config.ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "sklearn", "native"])
    ap.add_argument("--fast", action="store_true", help="tiny grids / few epochs (smoke test only)")
    args = ap.parse_args()
    be = resolve_backend(args.backend)
    print("Backends:", be)
    np.random.seed(config.SEED)

    df = load_dataset()
    s = split(df)
    Xtr, Xv = s.X_train.values, s.X_val.values
    ytr, yv = log_targets(s.df_train), log_targets(s.df_val)
    ctr, cv = s.df_train["yield_exceeded"].values, s.df_val["yield_exceeded"].values
    timings, registry = {}, {"backends": be, "features": config.FEATURES, "reg_targets": config.REG_TARGETS,
                             "target_transform": "log10", "files": {}}
    for d in ("random_forest", "xgboost", "neural_network", "baselines"):
        (MD / d).mkdir(parents=True, exist_ok=True)

    # ---------------- Random forest
    t0 = time.time()
    rf, rf_p, rf_log = tune_random_forest(Xtr, ytr, Xv, yv, args.fast)
    timings["random_forest"] = time.time() - t0
    joblib.dump(rf, MD / "random_forest/rf_regressor.joblib", compress=3)
    save_json({"best_params": rf_p, "tuning": rf_log}, MD / "random_forest/config.json")
    registry["files"]["random_forest"] = "models/random_forest/rf_regressor.joblib"
    print(f"RF done {timings['random_forest']:.1f}s  best={rf_p}")

    # ---------------- Gradient boosting
    t0 = time.time()
    gbs, gb_p, gb_log = tune_boosting(be["boosting"], Xtr, ytr, Xv, yv, args.fast)
    timings["boosting"] = time.time() - t0
    gb_files = []
    for tgt, m in zip(config.REG_TARGETS, gbs):
        f = MD / f"xgboost/{be['boosting']}_{tgt}.joblib"
        joblib.dump(m, f, compress=3)
        gb_files.append(str(f.relative_to(config.ROOT)))
    save_json({"backend": be["boosting"], "best_params": gb_p, "tuning": gb_log}, MD / "xgboost/config.json")
    registry["files"]["boosting"] = gb_files
    print(f"Boosting ({be['boosting']}) done {timings['boosting']:.1f}s  best={gb_p}")

    # ---------------- MLP (log-standardised inputs, standardised log targets)
    fs = FeatureScaler(log_features=True).fit(Xtr)
    ys = StandardScaler().fit(ytr)
    t0 = time.time()
    mlp, mlp_p, mlp_log = tune_mlp(be["mlp"], fs.transform(Xtr), ys.transform(ytr), fs.transform(Xv),
                                   ys.transform(yv), ys, yv, args.fast)
    timings["neural_network"] = time.time() - t0
    joblib.dump({"x": fs, "y": ys}, MD / "neural_network/scalers.joblib")
    registry["files"]["neural_network"] = save_mlp(mlp, be["mlp"], MD / "neural_network/mlp_regressor")
    registry["files"]["nn_scalers"] = "models/neural_network/scalers.joblib"
    hist = mlp.history if be["mlp"] == "torch" else {"train_loss": list(map(float, mlp.loss_curve_)),
                                                     "val_score_r2": list(map(float, mlp.validation_scores_))}
    save_json({"backend": be["mlp"], "best_params": {k: list(v) if isinstance(v, tuple) else v for k, v in mlp_p.items()},
               "tuning": mlp_log, "history": hist, "activation": "SiLU" if be["mlp"] == "torch" else "ReLU",
               "optimizer": "Adam", "input_transform": "log10 + standardise", "target_transform": "log10 + standardise"},
              MD / "neural_network/config.json")
    print(f"MLP ({be['mlp']}) done {timings['neural_network']:.1f}s  best={mlp_p}")

    # ---------------- Ablation: MLP on raw standardised inputs
    raw = FeatureScaler(log_features=False).fit(Xtr)
    t0 = time.time()
    if be["mlp"] == "torch":
        mlp_raw = TorchMLP(hidden=mlp_p["hidden"], max_epochs=60 if args.fast else 400).fit(
            raw.transform(Xtr), ys.transform(ytr), raw.transform(Xv), ys.transform(yv))
    else:
        mlp_raw = make_sklearn_mlp("reg", hidden=mlp_p["hidden"])
        if args.fast:
            mlp_raw.set_params(max_iter=60)
        mlp_raw.fit(raw.transform(Xtr), ys.transform(ytr))
    timings["mlp_raw_features"] = time.time() - t0
    joblib.dump({"x": raw, "y": ys}, MD / "baselines/mlp_raw_scalers.joblib")
    registry["files"]["mlp_raw_features"] = save_mlp(mlp_raw, be["mlp"], MD / "baselines/mlp_raw_features")
    registry["files"]["mlp_raw_scalers"] = "models/baselines/mlp_raw_scalers.joblib"

    # ---------------- Physics-informed power-law baseline: log y = c + sum_i a_i log x_i
    pl = LinearRegression().fit(np.log10(Xtr), ytr)
    joblib.dump(pl, MD / "baselines/power_law.joblib")
    registry["files"]["power_law_baseline"] = "models/baselines/power_law.joblib"
    save_json({"exponents": {t: dict(zip(config.FEATURES, map(float, pl.coef_[i])))
                             for i, t in enumerate(config.REG_TARGETS)},
               "intercepts_log10": dict(zip(config.REG_TARGETS, map(float, pl.intercept_)))},
              MD / "baselines/power_law_coefficients.json")

    # ---------------- Direct yield classifiers
    clf_log = {}
    rfc = make_random_forest("clf", n_estimators=40 if args.fast else 100, min_samples_leaf=2).fit(Xtr, ctr)
    clf_log["random_forest"] = float(f1_score(cv, rfc.predict(Xv)))
    joblib.dump(rfc, MD / "random_forest/rf_classifier.joblib", compress=3)
    gbc = fit_boosting(make_boosting(be["boosting"], "clf", **gb_p), be["boosting"], Xtr, ctr, Xv, cv)
    clf_log["boosting"] = float(f1_score(cv, gbc.predict(Xv)))
    joblib.dump(gbc, MD / f"xgboost/{be['boosting']}_classifier.joblib", compress=3)
    if be["mlp"] == "torch":
        mlc = TorchMLP(hidden=mlp_p["hidden"], task="clf", max_epochs=60 if args.fast else 400).fit(
            fs.transform(Xtr), ctr, fs.transform(Xv), cv)
    else:
        mlc = make_sklearn_mlp("clf", hidden=mlp_p["hidden"])
        if args.fast:
            mlc.set_params(max_iter=60)
        mlc.fit(fs.transform(Xtr), ctr)
    clf_log["neural_network"] = float(f1_score(cv, mlc.predict(fs.transform(Xv))))
    registry["files"]["classifiers"] = {
        "random_forest": "models/random_forest/rf_classifier.joblib",
        "boosting": f"models/xgboost/{be['boosting']}_classifier.joblib",
        "neural_network": save_mlp(mlc, be["mlp"], MD / "neural_network/mlp_classifier"),
    }
    registry["classifier_val_f1"] = clf_log
    registry["train_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    registry["fast_mode"] = args.fast
    save_json(registry, MD / "registry.json")
    print("Classifier validation F1:", clf_log)
    print("Saved registry -> models/registry.json")


if __name__ == "__main__":
    main()
