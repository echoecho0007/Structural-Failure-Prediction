"""Build report.pdf from report_template.tex, filling every number from the saved result files.

Usage (from the repository root):  python report/build_report.py
Requires: pdflatex (TeX Live / MiKTeX).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.analytical_solver import solve  # noqa: E402

REP = ROOT / "report"
R = json.loads((ROOT / "results/experiment_results.json").read_text())
SENS = json.loads((ROOT / "results/sensitivity_results.json").read_text())
GEN = json.loads((ROOT / "data/generation_stats.json").read_text())
REG = json.loads((ROOT / "models/registry.json").read_text())
CFG = {d: json.loads((ROOT / f"models/{d}/config.json").read_text()) for d in ("random_forest", "xgboost", "neural_network")}
df = pd.read_csv(config.DATASET_PATH)
reg, cls, lab = R["regression"], R["classification"], R["model_labels"]
be = R["backends"]


def f(x, d=4):
    return f"{x:.{d}f}"


def r2(x):
    return f"{x:.6f}" if x > 0.9999 else f"{x:.4f}"


V = {}
V["DATE"] = dt.date.today().isoformat()
V["N"] = f"{len(df):,}".replace(",", "\\,")
V["NTEST"], V["NTRAIN"], V["NVAL"] = (f"{R[k]:,}".replace(",", "\\,") for k in ("n_test", "n_train", "n_val"))
V["NDRAWN"] = f"{GEN['n_drawn']:,}".replace(",", "\\,")
V["NACC"] = f"{GEN['n_accepted_total']:,}".replace(",", "\\,")
V["ACC_RATE"] = f(100 * GEN["acceptance_rate"], 1)
V["YIELD_PCT"] = f(100 * df["yield_exceeded"].mean(), 1)
V["NPOS"] = str(sum(cls["neural_network"]["confusion_matrix"][1]))

gb_name = "XGBoost~\\cite{chen2016}" if be["boosting"] == "xgboost" else "scikit-learn \\texttt{HistGradientBoostingRegressor}"
mlp_name = "PyTorch~\\cite{paszke2019}" if be["mlp"] == "torch" else "scikit-learn \\texttt{MLPRegressor}~\\cite{pedregosa2011}"
V["BACKEND_SENTENCE"] = (f"gradient boosting --- {gb_name.split('~')[0]}; neural network --- {mlp_name.split('~')[0]}.")
if be["boosting"] != "xgboost" or be["mlp"] != "torch":
    V["BACKEND_SENTENCE"] += (" XGBoost and PyTorch implementations are included in the code and are selected automatically when "
                              "installed, but could not be installed in the build environment (no package-index access), so they "
                              "were not executed for this report.")
    V["BACKEND_LIMIT"] = ("The project specification called for XGBoost and PyTorch. These packages could not be installed in the "
                          "build environment, so the reported results use scikit-learn's histogram gradient boosting (same algorithm "
                          "family as XGBoost) and \\texttt{MLPRegressor}. The XGBoost/PyTorch code paths exist but are untested; "
                          "their numbers would differ slightly from those reported here.")
else:
    V["BACKEND_LIMIT"] = "Results were produced with XGBoost and PyTorch as specified."
V["GB_DESC"] = (f"{gb_name}: an additive ensemble of shallow trees fitted sequentially to the residuals, with histogram-binned "
                "features (the approach popularised by LightGBM~\\cite{ke2017}) and early stopping on held-out data. One model per target.")
hp = CFG["neural_network"]
V["MLP_DESC"] = (f"{mlp_name}: fully connected network with hidden layers {tuple(hp['best_params']['hidden'])}, "
                 f"{hp['activation']} activations, Adam optimiser~\\cite{{kingma2015}} (learning rate {hp['best_params']['lr']}), "
                 "mini-batches of 256, $L_2$ penalty $10^{-5}$ and early stopping on a validation fraction.")

S, D = "max_stress_Pa", "tip_deflection_m"
for key, m in (("RF", "random_forest"), ("GB", "boosting"), ("MLP", "neural_network")):
    V[f"{key}_S_R2"], V[f"{key}_D_R2"] = r2(reg[m][S]["R2"]), r2(reg[m][D]["R2"])
    V[f"{key}_S_MED"], V[f"{key}_D_MED"] = f(reg[m][S]["median_APE_%"], 2), f(reg[m][D]["median_APE_%"], 2)
V["RF_S_MAX"] = f(reg["random_forest"][S]["max_APE_%"], 0)
V["RAW_S_MED"] = f(reg["mlp_raw_features"][S]["median_APE_%"], 2)
V["RATIO_GB"] = f"{reg['boosting'][S]['median_APE_%'] / reg['neural_network'][S]['median_APE_%']:.0f}"
V["RATIO_RF"] = f"{reg['random_forest'][S]['median_APE_%'] / reg['neural_network'][S]['median_APE_%']:.0f}"
V["RATIO_RAW"] = f"{reg['mlp_raw_features'][S]['median_APE_%'] / reg['neural_network'][S]['median_APE_%']:.0f}"
V["MLP_C_F1"] = f(cls["neural_network"]["F1"])
cm = cls["neural_network"]["confusion_matrix"]
V["MLP_C_ERR"] = str(cm[0][1] + cm[1][0])
V["DER_F1"] = f(cls["derived_from_best_regressor"]["F1"])
er = cls["derived_from_best_regressor"].get("true_FoS_range_of_errors")
V["DER_ERR_FOS"] = f(er[0], 5) if er else "n/a"

o = R["extrapolation_force"]
V["OOD_THR"] = f"{o['threshold_N']:.0f}"
V["OOD_NTR"], V["OOD_NTE"] = (f"{o[k]:,}".replace(",", "\\,") for k in ("n_train", "n_test"))
V["OOD_RF"] = f(o["models"]["random_forest"][S]["median_APE_%"], 1)
V["OOD_GB"] = f(o["models"]["boosting"][S]["median_APE_%"], 1)
V["OOD_MLP"] = f(o["models"]["neural_network"][S]["median_APE_%"], 2)
V["OOD_RF_R2"] = f(o["models"]["random_forest"][S]["R2"], 3)

t = R["inference_time_ms_test_set"]
V["T_AN"], V["T_MLP"], V["T_RF"], V["T_GB"] = (f(t[k], 1) for k in ("analytical_solver_ms", "neural_network_ms", "random_forest_ms", "boosting_ms"))
V["TRAIN_TIME"] = f"{sum(REG['train_seconds'].values()):.0f}"

b = solve(*(config.BASELINE[k] for k in config.FEATURES))
V["BASE_S"], V["BASE_D"], V["BASE_FOS"] = f(float(b[S]) / 1e6, 0), f(float(b[D]) * 1e3, 1), f(float(b["factor_of_safety"]), 2)

corr = df[config.FEATURES].corr()
V["R_HF"], V["R_LF"] = f(corr.loc["thickness_m", "force_N"], 2), f(corr.loc["length_m", "force_N"], 2)
logs = __import__("numpy").log10(df[[S, D]])
q1, q3 = logs.quantile(0.25), logs.quantile(0.75)
iqr = q3 - q1
out = ((logs < q1 - 1.5 * iqr) | (logs > q3 + 1.5 * iqr)).sum()
V["IQR_S"], V["IQR_D"] = str(int(out[S])), str(int(out[D]))

sw = SENS["sweeps"]
V["MLP_H_SLOPE"] = f(sw["thickness_m->max_stress_Pa"]["models"]["neural_network"]["loglog_slope"], 3)
V["RF_H_SLOPE"] = f(sw["thickness_m->max_stress_Pa"]["models"]["random_forest"]["loglog_slope"], 2)
V["RF_E_SLOPE"] = f(sw["youngs_modulus_Pa->tip_deflection_m"]["models"]["random_forest"]["loglog_slope"], 2)
vs = SENS["log_variance_shares"]
V["VS_F_S"], V["VS_H_S"] = f(100 * vs[S]["force_N"], 0), f(100 * vs[S]["thickness_m"], 1)
V["VS_L_D"], V["VS_F_D"] = f(100 * vs[D]["length_m"], 0), f(100 * vs[D]["force_N"], 0)

# ---------------- tables
sym = {"length_m": ("Length", "$L$", "m", 1), "width_m": ("Width", "$b$", "mm", 1e3), "thickness_m": ("Thickness", "$h$", "mm", 1e3),
       "force_N": ("Force", "$F$", "N", 1), "youngs_modulus_Pa": ("Young's modulus", "$E$", "GPa", 1e-9),
       "yield_strength_Pa": ("Yield strength", "$\\sigma_y$", "MPa", 1e-6)}
rows = []
for k, (n, s_, u, sc) in sym.items():
    lo, hi = config.PARAM_RANGES[k]
    rows.append(f"{n} & {s_} & {u} & {lo*sc:g}--{hi*sc:g} & {df[k].mean()*sc:.3g} & {df[k].median()*sc:.3g}\\\\")
V["TABLE_INPUTS"] = "\n".join(rows)

hp_rows = [
    ("Random forest", "max\\_features $\\in\\{1.0,0.6\\}$, min\\_samples\\_leaf $\\in\\{2,4\\}$; 100 trees",
     CFG["random_forest"]["best_params"], CFG["random_forest"]["tuning"]),
    (lab["boosting"], ("learning\\_rate $\\in\\{0.05,0.1\\}$, max\\_depth $\\in\\{6,8\\}$; early stopping" if be["boosting"] == "xgboost"
                       else "learning\\_rate $\\in\\{0.05,0.1\\}$, max\\_leaf\\_nodes $\\in\\{31,63\\}$; up to 2000 iterations, early stopping (50)"),
     CFG["xgboost"]["best_params"], CFG["xgboost"]["tuning"]),
    (lab["neural_network"], "hidden layers $\\in\\{(128,128,64),(256,256,128)\\}$; Adam, lr $10^{-3}$",
     CFG["neural_network"]["best_params"], CFG["neural_network"]["tuning"]),
]
hrows = []
for name, grid, best, tun in hp_rows:
    bv = min(x["val_rmse_log10"] for x in tun)
    bs = ", ".join(f"{k.replace('_', chr(92) + '_')}={tuple(v) if isinstance(v, list) else v}" for k, v in best.items())
    hrows.append(f"{name} & {grid} & {bs} ({bv:.4f})\\\\")
V["TABLE_HP"] = "\n".join(hrows)

order = ["random_forest", "boosting", "neural_network", "mlp_raw_features", "power_law_baseline"]
trows = []
for tgt, tname, sc in ((S, "Stress", 1e-6), (D, "Deflection", 1e3)):
    ml = ["random_forest", "boosting", "neural_network"]
    best = {c: (min if c != "R2" else max)(ml, key=lambda m: reg[m][tgt][c]) for c in ["MAE", "RMSE", "R2", "median_APE_%", "P95_APE_%", "max_APE_%"]}
    for i, m in enumerate(order):
        x = reg[m][tgt]
        def cell(c, txt):
            return f"\\textbf{{{txt}}}" if best.get(c) == m else txt
        def num(v):
            return f"{v:.2e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.4g}"
        trows.append(f"{tname if i == 0 else ''} & {lab[m]} & {cell('MAE', num(x['MAE']*sc))} & {cell('RMSE', num(x['RMSE']*sc))} & "
                     f"{cell('R2', r2(x['R2']))} & {cell('median_APE_%', num(x['median_APE_%']))} & "
                     f"{cell('P95_APE_%', num(x['P95_APE_%']))} & {cell('max_APE_%', num(x['max_APE_%']))}\\\\")
    trows.append("\\midrule" if tgt == S else "")
V["TABLE_REG"] = "\n".join(trows)

crow = []
names = {"random_forest": f"{lab['random_forest']} (direct)", "boosting": f"{lab['boosting']} (direct)",
         "neural_network": f"{lab['neural_network']} (direct)", "derived_random_forest": f"FoS from {lab['random_forest']} stress",
         "derived_boosting": f"FoS from {lab['boosting']} stress", "derived_neural_network": f"FoS from {lab['neural_network']} stress",
         "derived_mlp_raw_features": "FoS from MLP (raw inputs) stress", "derived_power_law_baseline": "FoS from power-law reference"}
for k, n in names.items():
    c = cls[k]
    crow.append(f"{n} & {f(c['accuracy'])} & {f(c['precision'])} & {f(c['recall'])} & {f(c['F1'])} & {f(c['ROC_AUC'])}\\\\")
    if k == "neural_network":
        crow.append("\\midrule")
V["TABLE_CLS"] = "\n".join(crow)

orow = []
for m in ["random_forest", "boosting", "neural_network", "power_law_baseline"]:
    x = o["models"][m]
    orow.append(f"{lab[m]} & {f(x[S]['R2'])} & {f(x[S]['median_APE_%'], 2)} & {f(x[D]['R2'])} & {f(x[D]['median_APE_%'], 2)}\\\\")
V["TABLE_OOD"] = "\n".join(orow)

pretty = {"force_N->max_stress_Pa": "$F\\to\\sigma_{\\max}$", "thickness_m->max_stress_Pa": "$h\\to\\sigma_{\\max}$",
          "width_m->max_stress_Pa": "$b\\to\\sigma_{\\max}$", "length_m->tip_deflection_m": "$L\\to\\delta_{\\max}$",
          "youngs_modulus_Pa->tip_deflection_m": "$E\\to\\delta_{\\max}$", "yield_strength_Pa->factor_of_safety": "$\\sigma_y\\to$ FoS"}
srow = []
for k, e in sw.items():
    mm = e["models"]
    srow.append(f"{pretty[k]} & {e['exact_exponent']:+g} & {mm['random_forest']['loglog_slope']:+.3f} & {mm['boosting']['loglog_slope']:+.3f} & "
                f"{mm['neural_network']['loglog_slope']:+.3f} & "
                + " / ".join(f"{100*mm[m]['fraction_steps_correct_sign']:.0f}\\%" for m in ["random_forest", "boosting", "neural_network"]) + "\\\\")
V["TABLE_SLOPES"] = "\n".join(srow)

tex = (REP / "report_template.tex").read_text()
for k, v in V.items():
    tex = tex.replace(f"<<{k}>>", str(v))
left = re.findall(r"<<[A-Z0-9_]+>>", tex)
if left:
    sys.exit(f"Unfilled placeholders: {sorted(set(left))}")
(REP / "report.tex").write_text(tex)

if shutil.which("pdflatex") is None:
    sys.exit("pdflatex not found - report.tex written, PDF not built.")
for _ in range(3):  # TOC + references
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=REP,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
shutil.copy(REP / "report.pdf", ROOT / "report.pdf")
print("report.pdf written")
