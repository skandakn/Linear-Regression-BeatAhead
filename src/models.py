"""
src/models.py - Linear models and classifiers for BeatAhead ischemia detection.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from .feature_pipeline import TabularPreprocessor


class BeatAheadLinearClassifier:
    """End-to-end Linear Classifier pipeline for intraoperative myocardial ischemia.

    Combines:
    1. TabularPreprocessor (Median Imputation + Standardization)
    2. Regularized Logistic Regression (Balanced class weighting for imbalanced clinical events)
    3. Calibrated decision threshold
    """

    def __init__(
        self,
        feature_cols: List[str],
        penalty: str = "l2",
        C: float = 0.1,
        class_weight: str = "balanced",
        solver: str = "liblinear",
        random_state: int = 42,
        threshold: float = 0.5,
    ):
        self.feature_cols = list(feature_cols)
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=1000,
            random_state=random_state,
        )
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BeatAheadLinearClassifier":
        """Fits preprocessing pipeline and linear model on training data."""
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts positive class probability (P(ischemia=1))."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary classification based on decision threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def get_feature_coefficients(self) -> pd.DataFrame:
        """Returns sorted feature coefficients and odds ratios for clinical interpretability."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        coefs = self.model.coef_[0]
        df_coef = pd.DataFrame({
            "feature": self.feature_cols,
            "coefficient": coefs,
            "abs_coef": np.abs(coefs),
            "odds_ratio": np.exp(coefs),
        }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
        return df_coef


class BeatAheadRidgeProbabilityModel:
    """Linear Probability Model (LPM) via Ridge Regression.

    Directly estimates the linear event probability and continuous trajectory.
    """

    def __init__(
        self,
        feature_cols: List[str],
        alpha: float = 1.0,
        random_state: int = 42,
        threshold: float = 0.5,
    ):
        self.feature_cols = list(feature_cols)
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BeatAheadRidgeProbabilityModel":
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.preprocessor.transform(X)
        raw_preds = self.model.predict(X_scaled)
        return np.clip(raw_preds, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)
