# Dataset description — `generated_dataset.csv`

**Nature of the data: SYNTHETIC, ANALYTICAL.** Every target value is computed with closed-form
Euler–Bernoulli beam theory (`src/analytical_solver.py`). The data are **not** experimental
measurements and **not** finite-element (FEA) results.

Regenerate exactly with `python -m src.data_generation` (seed 42, NumPy `default_rng`).

## System

Prismatic cantilever, rectangular cross-section b × h, clamped at x = 0, transverse point load F at the
free end x = L. Linear-elastic, isotropic, homogeneous material; static load; small deflection;
no shear deformation, stress concentrations, residual stress, self-weight, buckling, fatigue or fracture.

## Columns (SI units throughout)

| Column | Symbol | Unit | Role | Definition |
|---|---|---|---|---|
| `length_m` | L | m | input | U(0.5, 2.0) before filtering |
| `width_m` | b | m | input | U(0.02, 0.10) |
| `thickness_m` | h | m | input | U(0.01, 0.05) (bending depth) |
| `force_N` | F | N | input | U(100, 10 000) |
| `youngs_modulus_Pa` | E | Pa | input | U(60 GPa, 210 GPa) |
| `yield_strength_Pa` | σ_y | Pa | input | U(150 MPa, 500 MPa) |
| `max_stress_Pa` | σ_max | Pa | target | 6FL / (b h²) — outer fibre at the support |
| `tip_deflection_m` | δ_max | m | target | F L³ / (3 E I) = 4 F L³ / (E b h³), I = b h³/12 |
| `max_strain` | ε_max | – | target | σ_max / E |
| `factor_of_safety` | FoS | – | target | σ_y / σ_max |
| `yield_exceeded` | – | {0,1} | class label | 1 if FoS < 1 else 0 |

## Physical-validity filtering (rejection sampling)

Inputs are drawn independently and uniformly, then a sample is **kept only if**

1. L/h ≥ 10 — slender beam, so Euler–Bernoulli theory (shear deformation neglected) is appropriate;
2. δ_max / L ≤ 0.05 — small-deflection (geometrically linear) assumption;
3. σ_max / σ_y ≤ 3 — the nominal elastic stress is used only as a *yield screening* quantity; cases
   far into the plastic range, where an elastic stress is physically meaningless, are discarded.

120 000 candidates were drawn; 34 085 (28.4 %) passed; the first 30 000 were kept.

**Consequence:** the filter changes the input distributions (fewer thin, long, heavily loaded or
low-modulus beams) and induces weak correlations between inputs (e.g. Pearson r(h, F) ≈ 0.31,
r(L, F) ≈ −0.34). Input distributions are therefore *not* uniform — see `results/plots/01_feature_distributions.png`.
E and σ_y are sampled independently, so not every (E, σ_y) pair corresponds to a specific
commercial alloy; the ranges span typical aluminium, titanium and steel values.

## Summary statistics (N = 30 000)

| Quantity | mean | median | min | max |
|---|---|---|---|---|
| σ_max [MPa] | 184.6 | 162.3 | 2.23 | 879.8 |
| δ_max [mm] | 23.7 | 18.7 | 0.053 | 99.1 |
| ε_max [µε] | 1323 | 1181 | 11.6 | 6657 |
| FoS [–] | 4.05 | 1.96 | 0.333 | 170.5 |
| Yield exceeded | 18.98 % of samples | | | |

Full statistics: `results/dataset_summary_statistics.csv`. Generation metadata: `data/generation_stats.json`.

## Validation performed automatically (`src/data_generation.validate`)

No NaN/inf; all inputs inside sampling bounds; all responses positive; every row satisfies the
validity constraints; every target column reproduces the analytical solver to rtol 1e-12; no duplicate
feature rows. Generation aborts if any check fails.

## Splits used for modelling (`src/preprocessing.py`)

70 / 15 / 15 % train / validation / test (21 000 / 4 500 / 4 500), stratified on `yield_exceeded`,
seed 42. The test split is used only once, in `src/evaluation.py`.
