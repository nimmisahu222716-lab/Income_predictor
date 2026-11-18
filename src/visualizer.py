"""
Streamlined Visualization Suite for Classification Benchmarks.

Generates the core comparison plots:
1. Confusion Matrices (heatmap analysis for all 3 algorithms)
2. Performance Metrics Comparison (grouped bar chart of Accuracy, Precision, Recall, F1, Balanced Acc)
3. ROC Curves (overlaid curves with AUC scores)
4. Execution Time (Training vs Prediction latency)
5. CART and ID3 Decision Tree Diagrams
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set_theme(style='whitegrid', font='sans-serif')
PALETTE = {'CART': '#3498DB', 'ID3': '#2ECC71', 'Linear SVM': '#E74C3C'}


def save_plot(fig, filename, output_dir='outputs', dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualizer] Saved: {filename}")
    return filepath


def plot_confusion_matrices(metrics_dict, output_dir='outputs'):
    """
    Generates side-by-side Confusion Matrix heatmaps for CART, ID3, and Linear SVM.
    """
    class_labels = ['<=50K', '>50K']
    algos = ['CART', 'ID3', 'Linear SVM']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, algo in enumerate(algos):
        ax = axes[idx]
        cm = metrics_dict[algo]['Confusion Matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={'fontsize': 12, 'weight': 'bold'}, ax=ax
        )
        ax.set_title(
            f"{algo} Confusion Matrix\nAcc: {metrics_dict[algo]['Accuracy']:.4f} | F1: {metrics_dict[algo]['F1 Score']:.4f}",
            fontsize=12, weight='bold', pad=10
        )
        ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
        ax.set_ylabel('True Label', fontsize=11, labelpad=8)

    plt.tight_layout()
    return save_plot(fig, 'confusion_matrices.png', output_dir)


def plot_performance_comparison(metrics_dict, output_dir='outputs'):
    """
    Consolidated performance bar chart comparing all primary classification metrics.
    """
    algos = ['CART', 'ID3', 'Linear SVM']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy']

    data = []
    for m in metrics_names:
        for a in algos:
            data.append({'Metric': m, 'Algorithm': a, 'Score': metrics_dict[a][m]})

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_plot, x='Metric', y='Score', hue='Algorithm',
        palette=PALETTE, ax=ax, edgecolor='#2C3E50', lw=1.2
    )
    ax.set_title("Comparative Classification Performance Across Key Metrics", fontsize=14, weight='bold', pad=12)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, labelpad=8)
    ax.set_xlabel("Evaluation Metric", fontsize=11, labelpad=8)
    ax.set_ylim(0, 1.08)
    ax.legend(title="Algorithm", fontsize=10, title_fontsize=11, loc='lower right')

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    return save_plot(fig, 'performance_metrics_comparison.png', output_dir)


def plot_roc_curves(y_true, probabilities_dict, output_dir='outputs'):
    """
    Overlaid ROC Curves for all three algorithms with AUC scores.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for algo, probs in probabilities_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{algo} (AUC = {roc_auc:.4f})", color=PALETTE[algo], lw=2.5)

    ax.plot([0, 1], [0, 1], color='#7F8C8D', linestyle='--', lw=1.5, label='Chance Baseline (AUC = 0.5000)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, labelpad=8)
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, weight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return save_plot(fig, 'roc_curve_comparison.png', output_dir)


def plot_execution_time(metrics_dict, output_dir='outputs'):
    """
    Side-by-side comparison of Training Time and Prediction Latency.
    """
    algos = ['CART', 'ID3', 'Linear SVM']
    colors = [PALETTE[a] for a in algos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Training Time
    t_train = [metrics_dict[a]['Training Time'] for a in algos]
    bars1 = ax1.bar(algos, t_train, color=colors, width=0.5, edgecolor='#2C3E50', lw=1.2)
    ax1.set_title("Training Time Comparison (seconds)", fontsize=13, weight='bold')
    ax1.set_ylabel("Execution Time (s)", fontsize=11)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.4f}s", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')

    # Prediction Time
    t_pred = [metrics_dict[a]['Prediction Time'] for a in algos]
    bars2 = ax2.bar(algos, t_pred, color=colors, width=0.5, edgecolor='#2C3E50', lw=1.2)
    ax2.set_title("Prediction Latency Comparison (seconds)", fontsize=13, weight='bold')
    ax2.set_ylabel("Inference Time (s)", fontsize=11)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.4f}s", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    return save_plot(fig, 'execution_time_comparison.png', output_dir)


def plot_cart_tree(cart_wrapper, output_dir='outputs', max_vis_depth=3):
    """
    Renders and saves the CART Decision Tree diagram.
    """
    return cart_wrapper.plot_tree(max_vis_depth=max_vis_depth, save_path=os.path.join(output_dir, 'cart_decision_tree.png'))


def plot_id3_tree(id3_classifier, output_dir='outputs', max_vis_depth=3):
    """
    Renders and saves the ID3 Decision Tree diagram.
    """
    return id3_classifier.plot_tree(max_vis_depth=max_vis_depth, save_path=os.path.join(output_dir, 'id3_decision_tree.png'))
