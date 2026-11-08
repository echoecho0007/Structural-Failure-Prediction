"""Integration checks: dataset validity, model reload, saved metrics consistency, app logic."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_generation import generate, validate

needs_data = pytest.mark.skipif(not config.DATASET_PATH.exists(), reason="dataset not generated")
needs_models = pytest.mark.skipif(not (config.MODEL_DIR / "registry.json").exists(), reason="models not trained")
needs_results = pytest.mark.skipif(not (config.RESULTS_DIR / "experiment_results.json").exists(), reason="no results")


def test_generation_is_reproducible():
    a, _ = generate(500, seed=7)
    b, _ = generate(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert validate(a) == []


@needs_data
def test_saved_dataset_valid():
    df = pd.read_csv(config.DATASET_PATH)
    assert len(df) == config.N_SAMPLES
    assert validate(df) == []


@needs_models
def test_models_reload_and_predict():
    from src.predictor import Predictor
    P = Predictor()
    x = pd.DataFrame([config.BASELINE])
    for m in ["random_forest", "boosting", "neural_network", "power_law_baseline", "mlp_raw_features"]:
        out = P.predict(x, m)
        assert np.isfinite(out["max_stress_Pa"]).all()
        assert out["max_stress_Pa"][0] == pytest.approx(150e6, rel=0.25)
    assert P.predict(x, "power_law_baseline")["max_stress_Pa"][0] == pytest.approx(150e6, rel=1e-8)
    for m in ["random_forest", "boosting", "neural_network"]:
        lab, p = P.classify(x, m)
        assert lab[0] in (0, 1) and 0 <= p[0] <= 1


@needs_results
@needs_models
@needs_data
def test_saved_metrics_match_recomputation():
    """Re-score the reloaded MLP on the test split and compare to results/metrics.csv."""
    from src.evaluation import regression_metrics
    from src.predictor import Predictor
    from src.preprocessing import load_dataset, split
    test = split(load_dataset()).df_test
    P = Predictor()
    saved = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    for m in ["random_forest", "neural_network"]:
        pred = P.predict(test, m)["max_stress_Pa"]
        r = regression_metrics(test["max_stress_Pa"].values, pred)
        row = saved[(saved.model == m) & (saved.target == "max_stress_Pa")].iloc[0]
        assert r["RMSE"] == pytest.approx(row["RMSE"], rel=1e-9)
        assert r["R2"] == pytest.approx(row["R2"], rel=1e-9)
    res = json.loads((config.RESULTS_DIR / "experiment_results.json").read_text())
    assert res["regression"]["neural_network"]["max_stress_Pa"]["R2"] == pytest.approx(
        saved[(saved.model == "neural_network") & (saved.target == "max_stress_Pa")]["R2"].item(), rel=1e-12)


def test_input_validation():
    from src.app_logic import validate_inputs
    ok = dict(config.BASELINE)
    assert validate_inputs(ok)["errors"] == []
    bad = dict(ok, thickness_m=-0.01)
    assert validate_inputs(bad)["errors"]
    out_of_range = dict(ok, force_N=50_000)
    v = validate_inputs(out_of_range)
    assert v["errors"] == [] and v["warnings"]  # allowed, but flagged as extrapolation
