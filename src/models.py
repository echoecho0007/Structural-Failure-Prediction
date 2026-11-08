"""Model factories with backend selection.

Backends
--------
* Gradient boosting: XGBoost (`xgboost.XGBRegressor`) when installed, otherwise
  scikit-learn's `HistGradientBoostingRegressor` (a histogram-based gradient-boosted
  tree ensemble of the same algorithmic family).
* Neural network: a PyTorch MLP when `torch` is installed, otherwise scikit-learn's
  `MLPRegressor` with the same architecture and optimiser (Adam).

The backend actually used is recorded in results/experiment_results.json so reported
numbers can always be traced to the code path that produced them.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from src import config

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def resolve_backend(requested: str = "auto") -> dict:
    if requested == "sklearn":
        return {"boosting": "sklearn-hgb", "mlp": "sklearn-mlp"}
    if requested == "native":
        missing = [p for p, ok in (("xgboost", HAS_XGB), ("torch", HAS_TORCH)) if not ok]
        if missing:
            raise ImportError(f"--backend native requires: {missing}")
    return {"boosting": "xgboost" if HAS_XGB else "sklearn-hgb", "mlp": "torch" if HAS_TORCH else "sklearn-mlp"}


# ----------------------------------------------------------------------------- tree models
def make_random_forest(task="reg", **kw):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = RandomForestRegressor if task == "reg" else RandomForestClassifier
    params = dict(n_estimators=200, n_jobs=-1, random_state=config.SEED)
    params.update(kw)
    return cls(**params)


def make_boosting(backend: str, task="reg", **kw):
    if backend == "xgboost":
        import xgboost as xgb
        params = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.9,
                      colsample_bytree=1.0, tree_method="hist", random_state=config.SEED,
                      early_stopping_rounds=50, n_jobs=-1)
        params.update(kw)
        return xgb.XGBRegressor(**params) if task == "reg" else xgb.XGBClassifier(eval_metric="logloss", **params)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    params = dict(max_iter=2000, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=50, random_state=config.SEED)
    params.update(kw)
    return (HistGradientBoostingRegressor if task == "reg" else HistGradientBoostingClassifier)(**params)


def fit_boosting(model, backend, X, y, X_val=None, y_val=None):
    if backend == "xgboost":
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:  # HistGB uses an internal split of the training data for early stopping
        model.fit(X, y)
    return model


# ----------------------------------------------------------------------------- MLP (PyTorch)
class TorchMLP:
    """Fully connected network trained with Adam + early stopping on a validation set.

    task='reg': MSE loss on (already scaled) targets, multi-output.
    task='clf': BCE-with-logits loss, single binary output; predict_proba returns P(class 1).
    """

    def __init__(self, hidden=(128, 128, 64), lr=1e-3, weight_decay=1e-6, batch_size=256,
                 max_epochs=400, patience=30, task="reg", seed=config.SEED):
        self.hidden, self.lr, self.weight_decay = tuple(hidden), lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.task, self.seed = task, seed
        self.net, self.history = None, {"train_loss": [], "val_loss": []}

    def _build(self, n_in, n_out):
        import torch.nn as nn
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val, y_val):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        Xv = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y, dtype=float).reshape(len(X), -1), dtype=torch.float32)
        yv = torch.as_tensor(np.asarray(y_val, dtype=float).reshape(len(Xv), -1), dtype=torch.float32)
        self.n_in, self.n_out = X.shape[1], y.shape[1]
        self.net = self._build(self.n_in, self.n_out)
        loss_fn = torch.nn.MSELoss() if self.task == "reg" else torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(X), generator=gen)
            tot = 0.0
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), y[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                vl = loss_fn(self.net(Xv), yv).item()
            sched.step(vl)
            self.history["train_loss"].append(tot / len(X))
            self.history["val_loss"].append(vl)
            if vl < best - 1e-9:
                best, wait = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_loss_ = best
        return self

    def _forward(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()

    def predict(self, X):
        out = self._forward(X)
        if self.task == "clf":
            return (out.ravel() > 0).astype(int)
        return out if self.n_out > 1 else out.ravel()

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self._forward(X).ravel()))
        return np.column_stack([1 - p, p])

    def save(self, path: Path):
        import torch
        torch.save({"state_dict": self.net.state_dict(), "n_in": self.n_in, "n_out": self.n_out,
                    "hidden": self.hidden, "task": self.task, "history": self.history}, path)

    @classmethod
    def load(cls, path: Path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(hidden=ck["hidden"], task=ck["task"])
        m.n_in, m.n_out, m.history = ck["n_in"], ck["n_out"], ck["history"]
        m.net = m._build(m.n_in, m.n_out)
        m.net.load_state_dict(ck["state_dict"])
        return m


def make_sklearn_mlp(task="reg", hidden=(128, 128, 64), alpha=1e-5, lr=1e-3):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    cls = MLPRegressor if task == "reg" else MLPClassifier
    return cls(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
               learning_rate_init=lr, batch_size=256, max_iter=600, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=30, tol=1e-7, random_state=config.SEED)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
