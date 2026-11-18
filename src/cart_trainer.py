"""
CART (Classification and Regression Tree) Implementation Module.

Utilizes Scikit-learn's DecisionTreeClassifier configured with the Gini impurity
splitting criterion, extracting full structural complexity and decision rules.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt


class CARTClassifierWrapper:
    """
    Wrapper for Scikit-learn's DecisionTreeClassifier (CART) with
    isolated categorical encoding to eliminate data leakage.
    """
    def __init__(self, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.encoder = None
        self.cat_cols = []
        self.num_cols = []
        self.feature_names_ = []
        self.model = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

    def fit(self, X, y, cat_cols=None, num_cols=None):
        """
        Fits OrdinalEncoder on training categorical columns and trains CART.
        """
        X_df = X.copy()
        if cat_cols is None or num_cols is None:
            self.cat_cols = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
            self.num_cols = [c for c in X_df.columns if c not in self.cat_cols]
        else:
            self.cat_cols = list(cat_cols)
            self.num_cols = list(num_cols)

        self.feature_names_ = self.num_cols + self.cat_cols

        if self.cat_cols:
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_df[self.cat_cols] = self.encoder.fit_transform(X_df[self.cat_cols])

        # Arrange features consistently
        X_mat = X_df[self.feature_names_].to_numpy()
        self.model.fit(X_mat, y)
        return self

    def _transform(self, X):
        X_df = X.copy()
        if self.cat_cols and self.encoder is not None:
            X_df[self.cat_cols] = self.encoder.transform(X_df[self.cat_cols])
        return X_df[self.feature_names_].to_numpy()

    def predict(self, X):
        X_mat = self._transform(X)
        return self.model.predict(X_mat)

    def predict_proba(self, X):
        X_mat = self._transform(X)
        return self.model.predict_proba(X_mat)

    def get_tree_metrics(self):
        """
        Extracts CART structural complexity metrics.
        """
        tree = self.model.tree_
        features_used = [
            self.feature_names_[i] for i, imp in enumerate(self.model.feature_importances_) if imp > 0
        ]
        return {
            'max_depth': int(self.model.get_depth()),
            'leaf_nodes': int(self.model.get_n_leaves()),
            'total_nodes': int(tree.node_count),
            'decision_rules': int(self.model.get_n_leaves()),
            'features_used': features_used,
            'feature_importances': dict(zip(self.feature_names_, self.model.feature_importances_))
        }

    def export_text_rules(self, max_depth=4):
        """Returns string representation of CART decision rules."""
        return export_text(self.model, feature_names=self.feature_names_, max_depth=max_depth)

    def plot_tree(self, max_vis_depth=3, figsize=(22, 10), save_path=None):
        """
        Visualizes the CART tree up to max_vis_depth using Scikit-learn plot_tree.
        """
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            self.model,
            max_depth=max_vis_depth,
            feature_names=self.feature_names_,
            class_names=['<=50K', '>50K'],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        plt.title(f"CART Decision Tree Structure (Gini Impurity, Depth 0 to {max_vis_depth})",
                  fontsize=14, weight='bold', pad=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[CART] Tree visualization saved to {save_path}")
        return fig
