"""Splitting, feature scaling and target transforms (fit on TRAINING data only).

Design decisions
----------------
* Split 70/15/15 (train/validation/test), stratified on the yield class so all
  models see identical, class-balanced partitions. The test set is touched only
  once, for final evaluation.
* Regression targets (stress, deflection) span ~3 orders of magnitude, so models
  learn log10(target). Predictions are mapped back to SI units before scoring.
* Feature standardisation (for the MLP) is fitted on the training split only.
  Tree ensembles are scale-invariant and receive raw SI features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame


def load_dataset(path=config.DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data_generation` first.")
    return pd.read_csv(path)


def split(df: pd.DataFrame, seed: int = config.SEED) -> Splits:
    df_train, df_tmp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["yield_exceeded"])
    df_val, df_test = train_test_split(df_tmp, test_size=0.50, random_state=seed, stratify=df_tmp["yield_exceeded"])
    f = config.FEATURES
    return Splits(df_train[f], df_val[f], df_test[f],
                  df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True))


def ood_split(df: pd.DataFrame, feature: str = "force_N", quantile: float = 0.85):
    """Extrapolation split: train on the lower part of `feature`'s range, test above it."""
    thr = df[feature].quantile(quantile)
    return df[df[feature] <= thr].reset_index(drop=True), df[df[feature] > thr].reset_index(drop=True), float(thr)


def log_targets(df: pd.DataFrame) -> np.ndarray:
    return np.log10(df[config.REG_TARGETS].values)


def inverse_log(y_log: np.ndarray) -> np.ndarray:
    return 10.0 ** y_log


class FeatureScaler:
    """Standardises log-transformed features. log10 makes the power-law physics additive,
    which helps the MLP; the transform uses no target information and is fit on train only."""

    def __init__(self, log_features: bool = True):
        self.log_features = log_features
        self.scaler = StandardScaler()

    def _pre(self, X):
        X = np.asarray(X, dtype=float)
        return np.log10(X) if self.log_features else X

    def fit(self, X):
        self.scaler.fit(self._pre(X))
        return self

    def transform(self, X):
        return self.scaler.transform(self._pre(X))

    def fit_transform(self, X):
        return self.fit(X).transform(X)
