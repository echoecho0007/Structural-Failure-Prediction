# Machine Learning-Based Structural Failure Prediction and Stress Analysis Under Variable Mechanical Loading

Machine-learning surrogates (random forest, gradient-boosted trees, multi-layer perceptron) that predict the
maximum bending stress, tip deflection, strain, factor of safety and yield status of an end-loaded rectangular
cantilever, benchmarked against exact Euler–Bernoulli beam theory — including physical-trend checks,
extrapolation tests and an interactive Streamlit dashboard.

> **Data provenance:** the dataset is **synthetic**, generated from closed-form beam theory. It is not
> experimental data and not FEA output. Stress fields shown are analytical beam-theory fields, not FEA.

> **Backend note (read this):** the specification calls for **XGBoost** and **PyTorch**. Both are implemented
> (`src/models.py`) and selected automatically when installed. The build environment had no package-index
> access, so the results below were produced with the scikit-learn fallbacks —
> `HistGradientBoostingRegressor` (same algorithm family as XGBoost) and `MLPRegressor`. The XGBoost/PyTorch
> paths and the Streamlit UI could not be executed there; see [Verification status](#verification-status).

---

## Motivation

Surrogate models replace expensive solvers (e.g. FEA) with fast learned approximations. Before trusting them on
problems without closed-form answers, it is useful to test them where the exact answer is known everywhere. The
end-loaded cantilever lets every prediction be checked exactly, both inside and outside the training data.

## Mathematical foundation

Rectangular cantilever, clamped at x = 0, point load F at x = L. Linear-elastic, static, small deflection.

| Quantity | Formula |
|---|---|
| Second moment of area | I = b h³ / 12 |
| Max. bending stress (support, outer fibre) | σ_max = 6 F L / (b h²) |
| Tip deflection | δ_max = F L³ / (3 E I) = 4 F L³ / (E b h³) |
| Max. strain | ε_max = σ_max / E |
| Factor of safety | FoS = σ_y / σ_max |
| Yield label | 1 if FoS < 1 (yield criterion exceeded), else 0 |

Assumptions: Euler–Bernoulli kinematics (no shear deformation), homogeneous isotropic linear-elastic material,
no stress concentrations, residual stress or self-weight. "Failure" here means only that the nominal elastic stress
reaches yield — **not** fatigue, buckling, fracture or plastic collapse.

Key observation: both responses are products of powers of the inputs, so they are *linear in log space*. Models
therefore learn log10 targets, and a log-space least-squares fit is included as an exact, physics-informed reference.

## System architecture

```
 inputs (L, b, h, F, E, σy)
        │
        ▼
 src/analytical_solver.py  ──►  src/data_generation.py  (uniform sampling + validity filter, 30 000 beams)
                                         │
                                         ▼
                               src/preprocessing.py  (70/15/15 stratified split, log10 targets,
                                         │              scalers fitted on train only)
                                         ▼
                               src/train_models.py   (grid search on validation split, save models)
                                         │
             ┌───────────────────────────┼─────────────────────────────┐
             ▼                           ▼                             ▼
   src/evaluation.py          src/sensitivity_analysis.py      app.py (Streamlit)
   (test metrics, OOD test)   (OAT sweeps, elasticities)       via src/predictor.py + src/app_logic.py
             └──────────── results/ + results/plots/ ─────────┘  ──►  report/build_report.py ──► report.pdf
```

## Dataset

30 000 beams; inputs drawn uniformly from L 0.5–2.0 m, b 20–100 mm, h 10–50 mm, F 100–10 000 N, E 60–210 GPa,
σ_y 150–500 MPa, then **kept only if** L/h ≥ 10 (slender), δ/L ≤ 0.05 (small deflection) and σ_max/σ_y ≤ 3
(nominal stress meaningful as a yield screen). Acceptance rate 28.4 %; 19.0 % of beams exceed yield.
Details: [`data/dataset_description.md`](data/dataset_description.md).

## Models

| Model | Inputs | Targets | Notes |
|---|---|---|---|
| Random forest | raw SI | log10 σ, log10 δ (multi-output) | 100 trees, tuned max_features / min_samples_leaf |
| Gradient boosting | raw SI | one model per target | XGBoost if installed, else sklearn HistGradientBoosting; early stopping |
| MLP | log10 + standardised | standardised log targets | 256-256-128, Adam; PyTorch if installed, else sklearn |
| MLP (raw inputs) | standardised, no log | same | ablation |
| Power-law (log-OLS) | log10 | log10 targets | exact physics-informed reference, not a competitor |
| Classifiers (RF / boosting / MLP) | as above | yield label | plus labels derived from predicted FoS |

Strain and FoS are derived from the predicted stress (ε = σ̂/E, FoS = σ_y/σ̂), keeping all outputs consistent.

## Results (held-out test set, N = 4 500 — produced by this code, seed 42)

Backends: scikit-learn HistGradientBoosting and scikit-learn MLP (see backend note).

| Model | Stress R² | Stress median / P95 / max abs. error | Deflection R² | Deflection median error |
|---|---|---|---|---|
| Random forest | 0.9687 | 6.08 % / 22.0 % / 135 % | 0.9361 | 10.6 % |
| HistGradBoost | 0.9964 | 2.18 % / 7.71 % / 38.6 % | 0.9906 | 4.00 % |
| **MLP** | **0.999997** | **0.080 % / 0.266 % / 0.62 %** | **0.999994** | **0.095 %** |
| MLP, raw inputs (ablation) | 0.9995 | 0.88 % / 4.90 % / 32.2 % | 0.9987 | 1.20 % |
| Power-law log-OLS (reference) | 1.000000 | ~1e-12 % | 1.000000 | ~1e-12 % |

Yield classification (positive = FoS < 1, 854 of 4 500):

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random forest (direct) | 0.9522 | 0.9323 | 0.8068 | 0.8650 |
| HistGradBoost (direct) | 0.9722 | 0.9451 | 0.9063 | 0.9253 |
| MLP (direct) | 0.9953 | 0.9906 | 0.9848 | 0.9877 |
| FoS < 1 from MLP stress | 0.9998 | 0.9988 | 1.0000 | 0.9994 |

The single error of the derived classifier is a beam with true FoS = 1.00024.

**Extrapolation** (train on F ≤ 6 574 N, test on larger F): median stress error RF 26.0 %, boosting 19.5 %,
MLP 0.26 %. Tree ensembles cannot predict beyond the range of their training targets.

**Physical trends** (log-log slope along sweeps vs exact exponent): h → σ: RF −1.66, boosting −1.91, MLP −1.999
(exact −2); E → δ: RF −0.74, boosting −0.997, MLP −0.997 (exact −1).

**Honest caveat:** the closed-form solver is roughly 85–1400× faster than the ML models here (0.3 ms vs 25–410 ms for 4 500
beams). The value of the project is methodological — model choice, representation, trend fidelity and extrapolation —
not speed.

All numbers: `results/metrics.csv`, `results/experiment_results.json`, `results/sensitivity_results.json`.
Figures (300 dpi) with captions: `results/plots/` and `results/plots/captions.json`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python ≥ 3.10 should work; the build used Python 3.13. If `xgboost` or `torch` fail to install, remove those lines —
the code falls back to scikit-learn automatically.

## Running

```bash
python run_pipeline.py                 # data → training → evaluation → sensitivity → figures → tests (~6–7 min, 2 cores)
python run_pipeline.py --backend sklearn   # reproduce the reported numbers exactly (forces sklearn fallbacks)
python run_pipeline.py --report        # also rebuild report.pdf (needs pdflatex)
python run_pipeline.py --fast          # quick smoke test; metrics NOT representative
streamlit run app.py                   # interactive dashboard (after the pipeline has run)
```
Individual stages: `python -m src.data_generation`, `python -m src.train_models`, `python -m src.evaluation`,
`python -m src.sensitivity_analysis`, `python -m src.visualization`, `python -m pytest tests`.

With XGBoost and PyTorch installed, the default `--backend auto` uses them, so your metrics will differ slightly
from the table above (use `--backend sklearn` to reproduce it).

## Directory structure

```
Structural-Failure-Prediction/
├── app.py                     Streamlit dashboard
├── run_pipeline.py            one-command reproduction
├── requirements.txt
├── report.pdf                 technical report (18 pages)
├── data/
│   ├── generated_dataset.csv
│   ├── dataset_description.md
│   └── generation_stats.json
├── src/
│   ├── config.py              ranges, constraints, seeds, paths
│   ├── analytical_solver.py   beam-theory reference
│   ├── data_generation.py     sampling, validity filter, validation
│   ├── preprocessing.py       splits, scalers, log transforms
│   ├── models.py              RF / boosting / MLP factories (XGBoost, PyTorch, sklearn)
│   ├── train_models.py        tuning + training + saving
│   ├── predictor.py           reloads saved models; single prediction API
│   ├── evaluation.py          test metrics, OOD experiment, diagnostic plots
│   ├── sensitivity_analysis.py
│   ├── visualization.py       EDA, beam diagram, stress field, FoS map
│   └── app_logic.py           input validation & comparison tables (testable without Streamlit)
├── models/                    random_forest/, xgboost/ (boosting models), neural_network/, baselines/, registry.json
├── results/                   metrics.csv, experiment_results.json, sensitivity_results.json, plots/
├── report/                    report_template.tex, build_report.py, report.tex
└── tests/                     solver hand-calculations, pipeline/reload/metric-consistency tests, app smoke stub
```
Note: `models/xgboost/` holds the gradient-boosting models of whichever backend ran (file names start with
`sklearn-hgb_` or `xgboost_`). The random-forest files are ~38 MB; GitHub accepts them, or uncomment the model lines in
`.gitignore` and retrain.

## Verification status

| Item | Status in build environment |
|---|---|
| Dataset generation + validation | ✅ executed |
| Analytical solver vs hand calculations | ✅ 6 tests pass |
| RF / HistGradBoost / sklearn-MLP training, saving, reloading | ✅ executed; reload test passes |
| Saved metrics = recomputed metrics from reloaded models | ✅ test passes |
| Full pipeline run twice from empty `data/`, `models/`, `results/` | ✅ same data and models; metrics agree to ~1e-13 relative (one RF-classifier tie at p = 0.5 flipped → fixed by single-threaded forest inference; repeated evaluations now bit-identical) |
| Sensitivity analysis, 18 figures, PDF report | ✅ generated and visually checked |
| XGBoost / PyTorch code paths | ⚠️ implemented, **not executed** (packages unavailable) |
| Streamlit UI | ⚠️ not launched (package unavailable). `app.py` was executed end-to-end against a minimal Streamlit stand-in (`tests/app_smoke/`) to check that all logic, model calls and plots run; the real UI rendering is untested |

To smoke-test the app logic without Streamlit: `PYTHONPATH=tests/app_smoke:. python app.py`.

## Limitations

- Synthetic, noise-free data from a simplified linear-elastic model; real FEA/experimental data would be noisier and scarcer.
- The target is exactly a power law, so a log-linear fit is perfect; the benchmark tests how well generic learners recover known physics, not discovery of new physics.
- Rejection sampling makes inputs non-uniform and correlated; independent (E, σ_y) sampling does not match real alloy pairings.
- Yield screening only — no shear, stress concentration, buckling, fatigue, fracture or plasticity.
- Small hyper-parameter grids.

## Future improvements

- Replace the analytical solver with FEA (plane-stress/3-D) to capture clamp stress concentrations; study accuracy vs number of simulations.
- Geometries/loads without closed-form solutions (distributed loads, I-sections, plates with holes).
- Field prediction (full stress maps) with CNNs/GNNs; physics-informed losses; monotonicity constraints.
- Uncertainty quantification to set safety margins for yield classification near FoS = 1.

## References

Gere & Goodno, *Mechanics of Materials* (2013); Budynas & Nisbett, *Shigley's Mechanical Engineering Design* (2015);
Breiman (2001) *Machine Learning* 45:5–32; Friedman (2001) *Ann. Statist.* 29:1189–1232; Chen & Guestrin (2016) KDD;
Pedregosa et al. (2011) *JMLR* 12:2825–2830; Kingma & Ba (2015) ICLR; Li et al. (2010) *J. Phys. Chem. A* 114:6022–6032.
Full list in `report.pdf`.
