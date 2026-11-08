"""Framework-independent logic for the Streamlit app (unit-testable without Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.analytical_solver import solve, solve_dataframe
from src.data_generation import validity_mask

MATERIALS = {  # representative nominal (minimum-spec-type) values for presets only; verify for real design
    "Structural steel (S275-type)": (200e9, 275e6),
    "Stainless steel 304 (annealed)": (193e9, 215e6),
    "High-strength steel": (210e9, 460e6),
    "Aluminium alloy 6061-T6": (69e9, 276e6),
    "Aluminium alloy 7075-T6": (72e9, 480e6),
    "Commercially pure titanium (grade 2)": (105e9, 275e6),
    "Custom": None,
}


def validate_inputs(p: dict) -> dict:
    """Errors block the calculation; warnings flag extrapolation / theory-validity issues."""
    errors, warnings = [], []
    for k in config.FEATURES:
        v = p.get(k)
        if v is None or not np.isfinite(v) or v <= 0:
            errors.append(f"{k}: must be a positive, finite number.")
    if errors:
        return {"errors": errors, "warnings": warnings}
    for k, (lo, hi) in config.PARAM_RANGES.items():
        if not lo <= p[k] <= hi:
            s = config.DISPLAY_SCALE.get(k, 1.0)
            warnings.append(f"{k} = {p[k]*s:g} is outside the training range [{lo*s:g}, {hi*s:g}] - ML output is an extrapolation.")
    df = pd.DataFrame([p])[config.FEATURES]
    ref = solve_dataframe(df)
    if not validity_mask(df, ref)[0]:
        c = config.CONSTRAINTS
        if p["length_m"] / p["thickness_m"] < c["min_slenderness_L_over_h"]:
            warnings.append("L/h < 10: beam is not slender; Euler-Bernoulli theory (no shear deformation) is inaccurate.")
        if ref["tip_deflection_m"][0] / p["length_m"] > c["max_deflection_ratio"]:
            warnings.append("Tip deflection > 5 % of span: small-deflection theory is no longer reliable.")
        if ref["max_stress_Pa"][0] / p["yield_strength_Pa"] > c["max_stress_over_yield"]:
            warnings.append("Nominal elastic stress > 3x yield: far outside the elastic regime and the training domain.")
    return {"errors": errors, "warnings": warnings}


def compare(predictor, p: dict) -> pd.DataFrame:
    """Analytical vs every ML model for one beam; display units."""
    ref = solve(*(p[k] for k in ["length_m", "width_m", "thickness_m", "force_N", "youngs_modulus_Pa", "yield_strength_Pa"]))
    X = pd.DataFrame([p])[config.FEATURES]
    rows = [{"Model": "Analytical (reference)", "Stress [MPa]": float(ref["max_stress_Pa"]) / 1e6,
             "Tip deflection [mm]": float(ref["tip_deflection_m"]) * 1e3, "Strain [µε]": float(ref["max_strain"]) * 1e6,
             "FoS [-]": float(ref["factor_of_safety"]), "Yield exceeded": bool(ref["yield_exceeded"]),
             "Stress error [%]": 0.0, "Deflection error [%]": 0.0}]
    for m in ["random_forest", "boosting", "neural_network"]:
        o = predictor.predict(X, m)
        rows.append({"Model": predictor.model_labels[m], "Stress [MPa]": o["max_stress_Pa"][0] / 1e6,
                     "Tip deflection [mm]": o["tip_deflection_m"][0] * 1e3, "Strain [µε]": o["max_strain"][0] * 1e6,
                     "FoS [-]": o["factor_of_safety"][0], "Yield exceeded": bool(o["yield_exceeded"][0]),
                     "Stress error [%]": 100 * (o["max_stress_Pa"][0] / float(ref["max_stress_Pa"]) - 1),
                     "Deflection error [%]": 100 * (o["tip_deflection_m"][0] / float(ref["tip_deflection_m"]) - 1)})
    return pd.DataFrame(rows)


def classifier_votes(predictor, p: dict) -> pd.DataFrame:
    X = pd.DataFrame([p])[config.FEATURES]
    rows = []
    for m in ["random_forest", "boosting", "neural_network"]:
        lab, prob = predictor.classify(X, m)
        rows.append({"Classifier": predictor.model_labels[m], "P(yield exceeded)": float(prob[0]),
                     "Prediction": "Yield exceeded" if lab[0] else "Safe"})
    return pd.DataFrame(rows)
