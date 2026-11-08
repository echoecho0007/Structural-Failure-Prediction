"""Streamlit dashboard: ML vs analytical cantilever beam analysis.

Run:  streamlit run app.py
(Requires trained models: python run_pipeline.py)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import config
from src.analytical_solver import bending_stress_field, deflection_profile
from src.app_logic import MATERIALS, classifier_votes, compare, validate_inputs

st.set_page_config(page_title="Cantilever ML Stress Analysis", layout="wide")


@st.cache_resource
def load_predictor():
    from src.predictor import Predictor
    return Predictor()


st.title("ML-based structural response prediction — end-loaded cantilever")
st.caption("Machine-learning surrogates benchmarked against Euler–Bernoulli beam theory. "
           "Training data are synthetic (analytical), not experimental or FEA. "
           "Linear-elastic, static, small-deflection yield screening only — not fatigue, buckling or fracture.")

try:
    P = load_predictor()
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun `python run_pipeline.py` to generate data and train the models.")
    st.stop()

with st.sidebar:
    st.header("Beam geometry")
    L = st.number_input("Length L [m]", min_value=0.01, value=1.0, step=0.1, format="%.3f")
    b = st.number_input("Width b [mm]", min_value=0.1, value=50.0, step=5.0) / 1e3
    h = st.number_input("Thickness h [mm]", min_value=0.1, value=40.0, step=2.0) / 1e3
    st.header("Loading")
    F = st.number_input("Tip force F [N]", min_value=0.0, value=2000.0, step=250.0)
    st.header("Material")
    mat = st.selectbox("Preset", list(MATERIALS.keys()), index=0)
    E0, sy0 = MATERIALS[mat] or (200e9, 250e6)
    E = st.number_input("Young's modulus E [GPa]", min_value=0.1, value=E0 / 1e9, step=5.0,
                        disabled=mat != "Custom") * 1e9
    sy = st.number_input("Yield strength σy [MPa]", min_value=0.1, value=sy0 / 1e6, step=10.0,
                         disabled=mat != "Custom") * 1e6
    st.caption(f"Training ranges: L {config.PARAM_RANGES['length_m']} m, b 20–100 mm, h 10–50 mm, "
               "F 100–10 000 N, E 60–210 GPa, σy 150–500 MPa.")

p = {"length_m": L, "width_m": b, "thickness_m": h, "force_N": F, "youngs_modulus_Pa": E, "yield_strength_Pa": sy}
v = validate_inputs(p)
for msg in v["errors"]:
    st.error(msg)
if v["errors"]:
    st.stop()
for msg in v["warnings"]:
    st.warning(msg)

table = compare(P, p)
ref = table.iloc[0]
best = table.iloc[3]  # MLP: best model on the test set
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Max stress (MLP)", f"{best['Stress [MPa]']:.2f} MPa", f"{best['Stress error [%]']:+.3f} % vs analytical", delta_color="off")
c2.metric("Tip deflection (MLP)", f"{best['Tip deflection [mm]']:.3f} mm", f"{best['Deflection error [%]']:+.3f} %", delta_color="off")
c3.metric("Max strain (MLP)", f"{best['Strain [µε]']:.1f} µε")
c4.metric("Factor of safety (MLP)", f"{best['FoS [-]']:.3f}", f"analytical {ref['FoS [-]']:.3f}", delta_color="off")
if ref["Yield exceeded"]:
    c5.error("Analytical: YIELD CRITERION EXCEEDED")
else:
    c5.success("Analytical: SAFE (FoS ≥ 1)")

st.subheader("Analytical reference vs ML models")
st.dataframe(table.style.format({c: "{:.4g}" for c in table.columns if table[c].dtype.kind == "f"}),
             use_container_width=True, hide_index=True)
st.subheader("Direct yield classifiers")
st.dataframe(classifier_votes(P, p).style.format({"P(yield exceeded)": "{:.4f}"}), use_container_width=True, hide_index=True)

st.subheader("Analytical fields for this beam (beam theory, not FEA)")
x = np.linspace(0, L, 200)
y = np.linspace(-h / 2, h / 2, 80)
Xg, Yg = np.meshgrid(x, y)
S = bending_stress_field(Xg, Yg, F, L, b, h) / 1e6
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
lim = max(np.abs(S).max(), 1e-9)
im = axes[0].pcolormesh(Xg, Yg * 1e3, S, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
fig.colorbar(im, ax=axes[0], label="σxx [MPa]")
axes[0].set_xlabel("x [m] (fixed end at 0)"); axes[0].set_ylabel("y [mm]"); axes[0].set_title("Bending stress σxx = M(x)y/I")
axes[1].plot(x, -deflection_profile(x, F, L, b, h, E) * 1e3)
axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("w(x) [mm]"); axes[1].set_title("Elastic curve")
fig.tight_layout()
st.pyplot(fig)

with st.expander("Model test-set performance and project figures"):
    rp = config.RESULTS_DIR / "experiment_results.json"
    if rp.exists():
        res = json.loads(rp.read_text())
        st.json({m: {t: {k: round(v[k], 6) for k in ("R2", "median_APE_%", "P95_APE_%")}
                     for t, v in d.items() if t in config.REG_TARGETS} for m, d in res["regression"].items()})
    for name in ["09_pred_vs_actual_stress", "12_model_comparison", "16_sensitivity_sweeps", "08_fos_safety_map"]:
        f = config.PLOT_DIR / f"{name}.png"
        if f.exists():
            st.image(str(f), use_container_width=True)
