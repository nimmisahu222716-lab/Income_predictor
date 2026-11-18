"""
Linear Support Vector Machine (Linear SVM) Implementation Module.

Configured exclusively with kernel='linear', C=1.0, probability=True, random_state=42.
Features ColumnTransformer for numerical standardization and one-hot encoding.
Extracts support vector distributions and hyperplane coefficient weights.
"""

import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class LinearSVMWrapper:
    """
    Wrapper for Scikit-learn's SVC with linear kernel, C=1.0, probability=True.
    Integrates ColumnTransformer preprocessing to prevent data leakage.
    """
    def __init__(self, C=1.0, probability=True, random_state=42):
        self.C = C
        self.probability = probability
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names_out_ = []
        self.model = SVC(
            kernel='linear',
            C=self.C,
            probability=self.probability,
            random_state=self.random_state
        )

    def fit(self, X, y, num_cols, cat_cols):
        """
        Fits ColumnTransformer strictly on training records, then trains Linear SVM.
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )

        X_transformed = self.preprocessor.fit_transform(X)

        # Extract output feature names
        num_names = list(num_cols)
        cat_encoder = self.preprocessor.named_transformers_['cat']
        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        self.feature_names_out_ = num_names + cat_names

        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.model.predict(X_proc)

    def predict_proba(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.model.predict_proba(X_proc)

    def get_svm_metrics(self, saved_model_path=None):
        """
        Extracts Linear SVM specific complexity metrics:
        - Total support vectors
        - Support vectors per class
        - Transformed input feature dimension
        - Serialized model size
        """
        n_sv = int(self.model.n_support_.sum())
        n_sv_per_class = {
            f"Class {cls} ({'<=50K' if cls == 0 else '>50K'})": int(count)
            for cls, count in zip(self.model.classes_, self.model.n_support_)
        }
        n_features = int(self.model.n_features_in_)

        model_size_kb = 'N/A'
        if saved_model_path and os.path.exists(saved_model_path):
            model_size_kb = f"{os.path.getsize(saved_model_path) / 1024:.2f} KB"

        return {
            'kernel': 'linear',
            'C': self.C,
            'total_support_vectors': n_sv,
            'support_vectors_per_class': n_sv_per_class,
            'number_of_features': n_features,
            'model_size': model_size_kb,
            'tree_depth': 'N/A',
            'leaf_nodes': 'N/A',
            'total_nodes': 'N/A',
            'decision_rules': 'N/A'
        }

    def get_top_coefficients(self, top_k=10):
        """
        Extracts top positive and negative linear hyperplane coefficient weights.
        """
        coefs = self.model.coef_.ravel()
        feat_coef = pd.DataFrame({
            'feature': self.feature_names_out_,
            'coefficient': coefs,
            'abs_coef': np.abs(coefs)
        }).sort_values(by='abs_coef', ascending=False)

        top_positive = feat_coef[feat_coef['coefficient'] > 0].head(top_k)
        top_negative = feat_coef[feat_coef['coefficient'] < 0].head(top_k)

        return {
            'top_overall': feat_coef.head(top_k * 2),
            'top_positive': top_positive,
            'top_negative': top_negative
        }
