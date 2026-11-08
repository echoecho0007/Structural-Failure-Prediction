"""Analytical reference solver: rectangular cantilever, transverse point load at the free end.

Euler-Bernoulli beam theory, linear-elastic, small deflection, static load.
All inputs/outputs in SI units. Functions accept scalars or NumPy arrays.
"""
from __future__ import annotations

import numpy as np


def second_moment_of_area(b, h):
    """I = b h^3 / 12  [m^4] (bending about the neutral axis parallel to the width)."""
    return np.asarray(b) * np.asarray(h) ** 3 / 12.0


def max_bending_stress(F, L, b, h):
    """sigma_max = M_max c / I = (F L)(h/2)/(b h^3/12) = 6 F L / (b h^2)  [Pa], at the fixed support."""
    return 6.0 * np.asarray(F) * np.asarray(L) / (np.asarray(b) * np.asarray(h) ** 2)


def tip_deflection(F, L, b, h, E):
    """delta_max = F L^3 / (3 E I) = 4 F L^3 / (E b h^3)  [m]."""
    return np.asarray(F) * np.asarray(L) ** 3 / (3.0 * np.asarray(E) * second_moment_of_area(b, h))


def max_strain(sigma, E):
    """epsilon_max = sigma_max / E  [-] (uniaxial Hooke's law)."""
    return np.asarray(sigma) / np.asarray(E)


def factor_of_safety(sigma_y, sigma):
    """FoS = sigma_y / sigma_max  [-]."""
    return np.asarray(sigma_y) / np.asarray(sigma)


def yield_exceeded(fos):
    """1 if FoS < 1 (yield criterion exceeded), else 0 (safe under the adopted criterion)."""
    return (np.asarray(fos) < 1.0).astype(int)


# ---- Field quantities used for visualisation -------------------------------------------------

def bending_moment(x, F, L):
    """M(x) = -F (L - x) for 0 <= x <= L (x measured from the fixed end). Magnitude returned."""
    return np.asarray(F) * (np.asarray(L) - np.asarray(x))


def bending_stress_field(x, y, F, L, b, h):
    """sigma_xx(x, y) = M(x) y / I, y measured from the neutral axis (+ top fibre in tension
    for a downward tip load)."""
    return bending_moment(x, F, L) * np.asarray(y) / second_moment_of_area(b, h)


def deflection_profile(x, F, L, b, h, E):
    """w(x) = F x^2 (3L - x) / (6 E I), the elastic curve of an end-loaded cantilever."""
    x = np.asarray(x)
    return F * x ** 2 * (3.0 * L - x) / (6.0 * E * second_moment_of_area(b, h))


def solve(L, b, h, F, E, sigma_y) -> dict:
    """Compute every reference output for one or many beams."""
    sigma = max_bending_stress(F, L, b, h)
    delta = tip_deflection(F, L, b, h, E)
    eps = max_strain(sigma, E)
    fos = factor_of_safety(sigma_y, sigma)
    return {
        "max_stress_Pa": sigma,
        "tip_deflection_m": delta,
        "max_strain": eps,
        "factor_of_safety": fos,
        "yield_exceeded": yield_exceeded(fos),
    }


def solve_dataframe(df):
    """Vectorised solve on a DataFrame that has the canonical feature columns."""
    return solve(df["length_m"].values, df["width_m"].values, df["thickness_m"].values,
                 df["force_N"].values, df["youngs_modulus_Pa"].values, df["yield_strength_Pa"].values)


if __name__ == "__main__":
    # Worked example (hand calculation reproduced in tests/test_analytical_solver.py)
    out = solve(L=1.0, b=0.05, h=0.04, F=2000.0, E=200e9, sigma_y=250e6)
    for k, v in out.items():
        print(f"{k:>18s}: {float(v):.6g}")
