"""Reproducible synthetic dataset generation from the analytical beam solver.

The dataset is SYNTHETIC: every target is computed with closed-form Euler-Bernoulli
beam theory (src/analytical_solver.py). It is not experimental or FEA data.

Usage:  python -m src.data_generation [--n 30000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.analytical_solver import solve_dataframe


def _sample_uniform(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame({k: rng.uniform(lo, hi, n) for k, (lo, hi) in config.PARAM_RANGES.items()})


def validity_mask(df: pd.DataFrame, targets: dict) -> np.ndarray:
    c = config.CONSTRAINTS
    slender = df["length_m"] / df["thickness_m"] >= c["min_slenderness_L_over_h"]
    small_def = targets["tip_deflection_m"] / df["length_m"] <= c["max_deflection_ratio"]
    stress_ok = targets["max_stress_Pa"] / df["yield_strength_Pa"] <= c["max_stress_over_yield"]
    return (slender & small_def & stress_ok).values


def generate(n: int = config.N_SAMPLES, seed: int = config.SEED) -> tuple[pd.DataFrame, dict]:
    """Rejection-sample until `n` physically valid beams are collected."""
    rng = np.random.default_rng(seed)
    accepted, drawn = [], 0
    while sum(len(a) for a in accepted) < n:
        batch = _sample_uniform(rng, 4 * n)
        drawn += len(batch)
        t = solve_dataframe(batch)
        keep = validity_mask(batch, t)
        accepted.append(batch[keep])
    df = pd.concat(accepted, ignore_index=True).iloc[:n].copy()
    targets = solve_dataframe(df)
    for k, v in targets.items():
        df[k] = v
    stats = {"n_drawn": int(drawn), "n_accepted_total": int(sum(len(a) for a in accepted)),
             "acceptance_rate": float(sum(len(a) for a in accepted) / drawn), "n_kept": int(n), "seed": seed}
    return df, stats


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of failed checks (empty list == dataset valid)."""
    errors = []
    if df.isna().any().any():
        errors.append("NaN values present")
    if not np.isfinite(df.select_dtypes("number").values).all():
        errors.append("non-finite values present")
    for k, (lo, hi) in config.PARAM_RANGES.items():
        if (df[k] < lo).any() or (df[k] > hi).any():
            errors.append(f"{k} outside sampling range")
    if (df[["max_stress_Pa", "tip_deflection_m", "max_strain", "factor_of_safety"]] <= 0).any().any():
        errors.append("non-positive response quantity")
    if not validity_mask(df, df).all():
        errors.append("physical-validity constraint violated")
    # internal consistency with the solver
    t = solve_dataframe(df)
    for k in config.ALL_TARGETS:
        if not np.allclose(df[k].values, t[k], rtol=1e-12):
            errors.append(f"{k} inconsistent with analytical solver")
    if df.duplicated(subset=config.FEATURES).any():
        errors.append("duplicate feature rows")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_SAMPLES)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    df, stats = generate(args.n, args.seed)
    errors = validate(df)
    if errors:
        raise SystemExit("Dataset validation FAILED: " + "; ".join(errors))
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PATH, index=False)
    stats["class_balance_yield_exceeded"] = float(df["yield_exceeded"].mean())
    stats["validation"] = "passed"
    (config.DATA_DIR / "generation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved {len(df)} samples -> {config.DATASET_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
