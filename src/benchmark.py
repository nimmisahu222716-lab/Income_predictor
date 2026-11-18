"""
Streamlined Centralized Benchmarking and Evaluation Module.

Focuses strictly on core classification metrics, execution timing,
confusion matrices, 5-fold cross-validation, and the master comparison table.
All redundant and duplicated metrics have been removed.
"""

import os
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix
)

from src.id3 import ID3Classifier, AdaptiveQuantileDiscretizer
from src.cart_trainer import CARTClassifierWrapper
from src.svm_trainer import LinearSVMWrapper


def compute_core_metrics(y_true, y_pred, y_proba_pos, train_time, pred_time):
    """
    Computes essential non-redundant classification metrics.
    Positive class is defined as Class 1 (>50K).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_score = roc_auc_score(y_true, y_proba_pos)
    except Exception:
        auc_score = 0.5

    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1 Score': float(f1),
        'Balanced Accuracy': float(bal_acc),
        'Specificity': float(specificity),
        'ROC-AUC': float(auc_score),
        'Training Time': float(train_time),
        'Prediction Time': float(pred_time),
        'Confusion Matrix': cm,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }


class BenchmarkRunner:
    """
    Streamlined execution harness for CART, ID3, and Linear SVM.
    """
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
        self.complexity = {}
        self.cv_results = {}

    def run_benchmark(self, X_train, X_test, y_train, y_test, num_cols, cat_cols):
        """
        Executes end-to-end training, timing, inference, and core metric calculation.
        """
        print("\n" + "="*70)
        print("STARTING BENCHMARKING (CORE ESSENTIAL METRICS)")
        print("="*70)

        # 1. CART
        print("\n[Benchmark] (1/3) Training CART Algorithm (Gini impurity)...")
        cart_wrapper = CARTClassifierWrapper(
            max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42
        )
        t0 = time.perf_counter()
        cart_wrapper.fit(X_train, y_train, cat_cols=cat_cols, num_cols=num_cols)
        t_train_cart = time.perf_counter() - t0

        t0 = time.perf_counter()
        cart_preds = cart_wrapper.predict(X_test)
        t_pred_cart = time.perf_counter() - t0
        cart_probs = cart_wrapper.predict_proba(X_test)[:, 1]

        self.models['CART'] = cart_wrapper
        self.predictions['CART'] = cart_preds
        self.probabilities['CART'] = cart_probs
        self.metrics['CART'] = compute_core_metrics(y_test, cart_preds, cart_probs, t_train_cart, t_pred_cart)
        self.complexity['CART'] = cart_wrapper.get_tree_metrics()
        print(f"  CART: Train {t_train_cart:.4f}s | Acc: {self.metrics['CART']['Accuracy']:.4f} | F1: {self.metrics['CART']['F1 Score']:.4f}")

        # 2. ID3
        print("\n[Benchmark] (2/3) Training ID3 Algorithm (Genuine Scratch Implementation)...")
        discretizer = AdaptiveQuantileDiscretizer(n_bins=4).fit(X_train[num_cols])
        X_train_disc = pd.concat([X_train[cat_cols], discretizer.transform(X_train[num_cols])], axis=1)
        X_test_disc = pd.concat([X_test[cat_cols], discretizer.transform(X_test[num_cols])], axis=1)

        id3 = ID3Classifier(max_depth=6, min_samples_split=20, min_info_gain=1e-4)
        t0 = time.perf_counter()
        id3.fit(X_train_disc, y_train)
        t_train_id3 = time.perf_counter() - t0

        t0 = time.perf_counter()
        id3_preds = id3.predict(X_test_disc)
        t_pred_id3 = time.perf_counter() - t0
        id3_probs = id3.predict_proba(X_test_disc)[:, 1]

        self.models['ID3'] = id3
        self.discretizer = discretizer
        self.predictions['ID3'] = id3_preds
        self.probabilities['ID3'] = id3_probs
        self.metrics['ID3'] = compute_core_metrics(y_test, id3_preds, id3_probs, t_train_id3, t_pred_id3)
        self.complexity['ID3'] = id3.get_tree_metrics()
        print(f"  ID3:  Train {t_train_id3:.4f}s | Acc: {self.metrics['ID3']['Accuracy']:.4f} | F1: {self.metrics['ID3']['F1 Score']:.4f}")

        # 3. Linear SVM
        print("\n[Benchmark] (3/3) Training Linear SVM (Standardized & One-Hot Encoded)...")
        svm_wrapper = LinearSVMWrapper(C=1.0, probability=True, random_state=42)
        t0 = time.perf_counter()
        svm_wrapper.fit(X_train, y_train, num_cols=num_cols, cat_cols=cat_cols)
        t_train_svm = time.perf_counter() - t0

        t0 = time.perf_counter()
        svm_preds = svm_wrapper.predict(X_test)
        t_pred_svm = time.perf_counter() - t0
        svm_probs = svm_wrapper.predict_proba(X_test)[:, 1]

        self.models['Linear SVM'] = svm_wrapper
        self.predictions['Linear SVM'] = svm_preds
        self.probabilities['Linear SVM'] = svm_probs
        self.metrics['Linear SVM'] = compute_core_metrics(y_test, svm_preds, svm_probs, t_train_svm, t_pred_svm)
        self.complexity['Linear SVM'] = svm_wrapper.get_svm_metrics()
        print(f"  SVM:  Train {t_train_svm:.4f}s | Acc: {self.metrics['Linear SVM']['Accuracy']:.4f} | F1: {self.metrics['Linear SVM']['F1 Score']:.4f}")

        return self

    def generate_comparison_table(self):
        """
        Creates clean, streamlined comparison table with essential metrics only.
        """
        rows = []
        for algo in ['CART', 'ID3', 'Linear SVM']:
            m = self.metrics[algo]
            c = self.complexity[algo]

            row = {
                'Algorithm': algo,
                'Accuracy': f"{m['Accuracy']:.4f}",
                'Precision': f"{m['Precision']:.4f}",
                'Recall': f"{m['Recall']:.4f}",
                'F1 Score': f"{m['F1 Score']:.4f}",
                'Balanced Accuracy': f"{m['Balanced Accuracy']:.4f}",
                'Specificity': f"{m['Specificity']:.4f}",
                'ROC-AUC': f"{m['ROC-AUC']:.4f}",
                'Training Time (s)': f"{m['Training Time']:.4f}",
                'Prediction Time (s)': f"{m['Prediction Time']:.4f}",
                'Tree Depth': str(c.get('max_depth', 'N/A')),
                'Leaf Nodes': str(c.get('leaf_nodes', 'N/A')),
                'Total Nodes': str(c.get('total_nodes', 'N/A')),
                'Support Vectors': str(c.get('total_support_vectors', 'N/A'))
            }
            rows.append(row)

        df_comparison = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, 'model_comparison_table.csv')
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n[Benchmark] Streamlined Comparison Table saved to {csv_path}")

        best_summary = {
            'Highest Accuracy': max(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['Accuracy']),
            'Highest Precision': max(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['Precision']),
            'Highest Recall': max(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['Recall']),
            'Highest F1 Score': max(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['F1 Score']),
            'Highest Balanced Accuracy': max(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['Balanced Accuracy']),
            'Fastest Training Algorithm': min(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['Training Time']),
            'Fastest Prediction Algorithm': min(['CART', 'ID3', 'Linear SVM'], key=lambda a: self.metrics[a]['Prediction Time'])
        }

        return df_comparison, best_summary
