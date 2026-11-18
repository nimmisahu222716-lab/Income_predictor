"""
Streamlined Master Execution Script for Comparative Performance Analysis.

Executes the streamlined workflow:
1. Data acquisition and cleaning
2. Stratified train-test split (80:20)
3. Model training and benchmarking (Core metrics only)
4. 5-Fold Stratified Cross-Validation with paired t-tests
5. High-Resolution Visualizations (Confusion matrices, ROC curves, Metrics bar chart, Timing, Decision Trees)
6. Master comparison table CSV export
7. Interactive custom prediction demonstration (Inference)
"""

import os
import sys
import tempfile
import pandas as pd
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib')

from src.data_loader import download_and_load_dataset, clean_dataset, prepare_data_split
from src.id3 import AdaptiveQuantileDiscretizer
from src.benchmark import BenchmarkRunner
from src.visualizer import (
    plot_confusion_matrices, plot_performance_comparison,
    plot_roc_curves, plot_execution_time, plot_cart_tree, plot_id3_tree
)


def run_full_pipeline(sample_size=10000):
    print("=" * 80)
    print("STREAMLINED BENCHMARK: CART vs. ID3 vs. Linear SVM")
    print("=" * 80)

    # 1. Load Data
    raw_df = download_and_load_dataset(data_dir='data')

    # 2. Clean Data
    clean_df = clean_dataset(raw_df)

    # 3. Stratified Split (80:20)
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data_split(
        clean_df, sample_size=sample_size, test_size=0.20, random_state=42
    )

    # 4. Centralized Benchmarking (Core Metrics)
    runner = BenchmarkRunner(output_dir='outputs')
    runner.run_benchmark(X_train, X_test, y_train, y_test, num_cols=num_cols, cat_cols=cat_cols)

    # 5. Generate Clean Comparison Table
    df_comparison, best_summary = runner.generate_comparison_table()

    # 7. Generate Essential Visualizations
    print("\n" + "=" * 70)
    print("GENERATING ESSENTIAL HIGH-RESOLUTION VISUALIZATIONS")
    print("=" * 70)

    # 1. Confusion Matrices (heatmap analysis for all 3 algorithms)
    plot_confusion_matrices(runner.metrics, output_dir='outputs')

    # 2. Consolidated Performance Comparison Bar Chart
    plot_performance_comparison(runner.metrics, output_dir='outputs')

    # 3. Overlaid ROC Curves
    plot_roc_curves(y_test, runner.probabilities, output_dir='outputs')

    # 4. Execution Time (Training vs Prediction latency)
    plot_execution_time(runner.metrics, output_dir='outputs')

    # 5. CART Decision Tree Diagram
    plot_cart_tree(runner.models['CART'], output_dir='outputs', max_vis_depth=3)

    # 6. ID3 Decision Tree Diagram
    plot_id3_tree(runner.models['ID3'], output_dir='outputs', max_vis_depth=3)

    # 8. Interactive Prediction Interface (CLI)
    discretizer = AdaptiveQuantileDiscretizer(n_bins=4).fit(X_train[num_cols])

    if '--interactive' in sys.argv:
        from src.inference import interactive_terminal_prompt
        interactive_terminal_prompt(
            runner.models, discretizer, num_cols, cat_cols,
            metrics_results=runner.metrics, master_table=df_comparison
        )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results saved in: {os.path.abspath('outputs')}")
    print("=" * 80)
    return runner, df_comparison, best_summary


if __name__ == '__main__':
    run_full_pipeline(sample_size=10000)
