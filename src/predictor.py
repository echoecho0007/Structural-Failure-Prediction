"""Load the saved models and produce predictions in SI units.

Used by evaluation, sensitivity analysis and the Streamlit app, so every consumer goes
through the same reload path (this also verifies that saved models can be loaded).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config

MODEL_NAMES = {"random_forest": "Random Forest", "boosting": "Gradient Boosting", "neural_network": "MLP",
               "power_law_baseline": "Power-law (log-OLS)", "mlp_raw_features": "MLP (raw inputs)"}


def _load_mlp(path):
    if path.endswith(".pt"):
        from src.models import TorchMLP
        return TorchMLP.load(config.ROOT / path)
    return joblib.load(config.ROOT / path)


class Predictor:
    def __init__(self):
        reg_path = config.MODEL_DIR / "registry.json"
        if not reg_path.exists():
            raise FileNotFoundError("models/registry.json missing - run `python -m src.train_models` first.")
        self.registry = json.loads(reg_path.read_text())
        f = self.registry["files"]
        self.backends = self.registry["backends"]
        self.rf = joblib.load(config.ROOT / f["random_forest"])
        self.gb = [joblib.load(config.ROOT / p) for p in f["boosting"]]
        self.mlp = _load_mlp(f["neural_network"])
        sc = joblib.load(config.ROOT / f["nn_scalers"])
        self.xs, self.ys = sc["x"], sc["y"]
        self.mlp_raw = _load_mlp(f["mlp_raw_features"])
        self.raw_sc = joblib.load(config.ROOT / f["mlp_raw_scalers"])
        self.pl = joblib.load(config.ROOT / f["power_law_baseline"])
        c = f["classifiers"]
        self.clf = {"random_forest": joblib.load(config.ROOT / c["random_forest"]),
                    "boosting": joblib.load(config.ROOT / c["boosting"]),
                    "neural_network": _load_mlp(c["neural_network"])}
        # Single-threaded inference makes forest predictions bit-reproducible (parallel tree
        # averaging changes floating-point summation order and can flip exact 0.5 ties).
        self.rf.n_jobs = 1
        self.clf["random_forest"].n_jobs = 1

    @property
    def model_labels(self):
        lab = dict(MODEL_NAMES)
        lab["boosting"] = "XGBoost" if self.backends["boosting"] == "xgboost" else "HistGradBoost"
        lab["neural_network"] = "MLP (PyTorch)" if self.backends["mlp"] == "torch" else "MLP (sklearn)"
        return lab

    @staticmethod
    def _X(X):
        if isinstance(X, pd.DataFrame):
            return X[config.FEATURES].values.astype(float)
        return np.atleast_2d(np.asarray(X, dtype=float))

    def predict_log(self, X, model: str) -> np.ndarray:
        """Return log10 predictions, shape (n, 2): [stress, deflection]."""
        X = self._X(X)
        if model == "random_forest":
            return self.rf.predict(X)
        if model == "boosting":
            return np.column_stack([m.predict(X) for m in self.gb])
        if model == "neural_network":
            out = np.asarray(self.mlp.predict(self.xs.transform(X))).reshape(len(X), -1)
            return self.ys.inverse_transform(out)
        if model == "mlp_raw_features":
            out = np.asarray(self.mlp_raw.predict(self.raw_sc["x"].transform(X))).reshape(len(X), -1)
            return self.raw_sc["y"].inverse_transform(out)
        if model == "power_law_baseline":
            return self.pl.predict(np.log10(X))
        raise KeyError(model)

    def predict(self, X, model: str) -> dict:
        """All outputs in SI units. Strain and FoS are derived from the predicted stress
        and the (known) inputs E and sigma_y."""
        Xa = self._X(X)
        y = 10.0 ** self.predict_log(Xa, model)
        E, sy = Xa[:, 4], Xa[:, 5]
        stress, defl = y[:, 0], y[:, 1]
        fos = sy / stress
        return {"max_stress_Pa": stress, "tip_deflection_m": defl, "max_strain": stress / E,
                "factor_of_safety": fos, "yield_exceeded": (fos < 1).astype(int)}

    def classify(self, X, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Direct classifier: returns (label, P(yield exceeded))."""
        Xa = self._X(X)
        m = self.clf[model]
        Xin = self.xs.transform(Xa) if model == "neural_network" else Xa
        p = m.predict_proba(Xin)[:, 1]
        return (p >= 0.5).astype(int), p
