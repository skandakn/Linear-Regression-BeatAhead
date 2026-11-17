"""
src/feature_pipeline.py - Feature imputation and standardization pipeline.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    """Preprocesses tabular physiological feature matrices.

    Enforces strict featurization ordering:
    - Fits imputation and scaling exclusively on training partition.
    - Applies transformation out-of-sample without leakage.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """Fits imputer and standard scaler on training data."""
        X_mat = X[self.feature_cols].copy()
        imputed = self.imputer.fit_transform(X_mat)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transforming data.")
        X_mat = X[self.feature_cols].copy()
        imputed = self.imputer.transform(X_mat)
        scaled = self.scaler.transform(imputed)
        return scaled

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits on training data and returns transformed array."""
        return self.fit(X).transform(X)
