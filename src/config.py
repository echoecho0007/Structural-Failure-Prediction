"""Central configuration: paths, parameter ranges, physical-validity constraints, seeds.

All quantities are in SI units internally (m, N, Pa).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOT_DIR = RESULTS_DIR / "plots"
DATASET_PATH = DATA_DIR / "generated_dataset.csv"

SEED = 42
N_SAMPLES = 30_000

FEATURES = ["length_m", "width_m", "thickness_m", "force_N", "youngs_modulus_Pa", "yield_strength_Pa"]
FEATURE_LABELS = {
    "length_m": r"Length $L$ [m]",
    "width_m": r"Width $b$ [m]",
    "thickness_m": r"Thickness $h$ [m]",
    "force_N": r"Force $F$ [N]",
    "youngs_modulus_Pa": r"Young's modulus $E$ [GPa]",
    "yield_strength_Pa": r"Yield strength $\sigma_y$ [MPa]",
}
# Display scale factors (SI value * factor = display value)
DISPLAY_SCALE = {"youngs_modulus_Pa": 1e-9, "yield_strength_Pa": 1e-6}

# Primary regression targets learned by the ML models
REG_TARGETS = ["max_stress_Pa", "tip_deflection_m"]
# Derived targets (computed from predicted stress + known inputs)
ALL_TARGETS = ["max_stress_Pa", "tip_deflection_m", "max_strain", "factor_of_safety", "yield_exceeded"]

# Uniform sampling ranges [low, high] (SI units)
PARAM_RANGES = {
    "length_m": (0.5, 2.0),
    "width_m": (0.02, 0.10),
    "thickness_m": (0.01, 0.05),
    "force_N": (100.0, 10_000.0),
    "youngs_modulus_Pa": (60e9, 210e9),
    "yield_strength_Pa": (150e6, 500e6),
}

# Physical-validity constraints applied by rejection sampling.
CONSTRAINTS = {
    # Euler-Bernoulli (shear deformation negligible) requires a slender beam.
    "min_slenderness_L_over_h": 10.0,
    # Small-deflection (linear geometry) assumption: tip deflection <= 5 % of span.
    "max_deflection_ratio": 0.05,
    # Nominal elastic stress is only a meaningful *screening* quantity near yield;
    # cases where it exceeds the yield strength by more than 3x are discarded.
    "max_stress_over_yield": 3.0,
}

# Baseline beam for one-at-a-time sensitivity sweeps (inside the valid domain).
BASELINE = {
    "length_m": 1.0,
    "width_m": 0.05,
    "thickness_m": 0.04,
    "force_N": 2000.0,
    "youngs_modulus_Pa": 200e9,
    "yield_strength_Pa": 250e6,
}
