"""Hand-calculation checks for the analytical solver.

Case A (baseline): L = 1 m, b = 50 mm, h = 40 mm, F = 2 kN, E = 200 GPa, sigma_y = 250 MPa
    I      = 0.05 * 0.04^3 / 12           = 2.6667e-7 m^4
    sigma  = 6 * 2000 * 1 / (0.05 * 0.04^2) = 12000 / 8e-5 = 150 MPa
    delta  = 2000 * 1^3 / (3 * 200e9 * 2.6667e-7) = 2000 / 160000 = 12.5 mm
    eps    = 150e6 / 200e9 = 7.5e-4
    FoS    = 250 / 150 = 1.6667 -> safe
Case B: L = 2 m, b = 20 mm, h = 50 mm, F = 10 kN, E = 70 GPa, sigma_y = 200 MPa
    sigma  = 6 * 1e4 * 2 / (0.02 * 0.0025) = 120000 / 5e-5 = 2400 MPa
    delta  = 4 * 1e4 * 8 / (70e9 * 0.02 * 1.25e-4) = 320000 / 175000 = 1.828571 m
    FoS    = 200 / 2400 = 0.083333 -> yield exceeded
"""
import numpy as np
import pytest

from src import analytical_solver as an


def test_case_a_baseline():
    out = an.solve(L=1.0, b=0.05, h=0.04, F=2000.0, E=200e9, sigma_y=250e6)
    assert an.second_moment_of_area(0.05, 0.04) == pytest.approx(2.666667e-7, rel=1e-6)
    assert out["max_stress_Pa"] == pytest.approx(150e6, rel=1e-12)
    assert out["tip_deflection_m"] == pytest.approx(0.0125, rel=1e-12)
    assert out["max_strain"] == pytest.approx(7.5e-4, rel=1e-12)
    assert out["factor_of_safety"] == pytest.approx(250 / 150, rel=1e-12)
    assert int(out["yield_exceeded"]) == 0


def test_case_b_yield_exceeded():
    out = an.solve(L=2.0, b=0.02, h=0.05, F=1e4, E=70e9, sigma_y=200e6)
    assert out["max_stress_Pa"] == pytest.approx(2.4e9, rel=1e-12)
    assert out["tip_deflection_m"] == pytest.approx(1.828571428571, rel=1e-10)
    assert out["factor_of_safety"] == pytest.approx(1 / 12, rel=1e-12)
    assert int(out["yield_exceeded"]) == 1


def test_two_deflection_forms_agree():
    rng = np.random.default_rng(0)
    F, L, b, h, E = rng.uniform(100, 1e4, 50), rng.uniform(0.5, 2, 50), rng.uniform(.02, .1, 50), rng.uniform(.01, .05, 50), rng.uniform(60e9, 210e9, 50)
    assert np.allclose(an.tip_deflection(F, L, b, h, E), 4 * F * L ** 3 / (E * b * h ** 3), rtol=1e-12)


def test_fields_consistent_with_maxima():
    F, L, b, h, E = 2000.0, 1.0, 0.05, 0.04, 200e9
    assert an.bending_stress_field(0.0, h / 2, F, L, b, h) == pytest.approx(an.max_bending_stress(F, L, b, h))
    assert an.bending_stress_field(L, h / 2, F, L, b, h) == pytest.approx(0.0)
    assert an.deflection_profile(L, F, L, b, h, E) == pytest.approx(an.tip_deflection(F, L, b, h, E))
    assert an.deflection_profile(0.0, F, L, b, h, E) == 0.0


def test_scaling_laws():
    base = dict(L=1.0, b=0.05, h=0.04, F=2000.0, E=200e9, sigma_y=250e6)
    s0, d0 = an.solve(**base)["max_stress_Pa"], an.solve(**base)["tip_deflection_m"]
    assert an.solve(**{**base, "h": 0.08})["max_stress_Pa"] == pytest.approx(s0 / 4)
    assert an.solve(**{**base, "L": 2.0})["tip_deflection_m"] == pytest.approx(8 * d0)
    assert an.solve(**{**base, "E": 100e9})["tip_deflection_m"] == pytest.approx(2 * d0)
    assert an.solve(**{**base, "F": 4000.0})["max_stress_Pa"] == pytest.approx(2 * s0)


def test_fos_boundary():
    assert an.yield_exceeded(1.0) == 0
    assert an.yield_exceeded(0.999999) == 1
