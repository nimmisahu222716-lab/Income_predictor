"""
Interactive Income Prediction Tool (CART vs. ID3 vs. Linear SVM).

Allows entering custom demographic and financial profiles directly in the terminal
and evaluates predictions across all three classification paradigms in real time.
"""

import os
import sys
import pandas as pd
from src.data_loader import download_and_load_dataset, clean_dataset, prepare_data_split
from src.cart_trainer import CARTClassifierWrapper
from src.svm_trainer import LinearSVMWrapper
from src.id3 import ID3Classifier, AdaptiveQuantileDiscretizer
from src.inference import interactive_terminal_prompt


def main():
    print("=" * 80)
    print("INITIALIZING COMPARATIVE CLASSIFIERS FOR INTERACTIVE INFERENCE...")
    print("=" * 80)

    # Load & prepare data
    raw_df = download_and_load_dataset(data_dir='data')
    clean_df = clean_dataset(raw_df)
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data_split(
        clean_df, sample_size=10000, test_size=0.20, random_state=42
    )

    print("\nTraining CART, ID3, and Linear SVM models...")
    # CART
    cart = CARTClassifierWrapper(max_depth=8, random_state=42)
    cart.fit(X_train, y_train, cat_cols=cat_cols, num_cols=num_cols)

    # Linear SVM
    svm = LinearSVMWrapper(C=1.0, random_state=42)
    svm.fit(X_train, y_train, num_cols=num_cols, cat_cols=cat_cols)

    # ID3
    discretizer = AdaptiveQuantileDiscretizer(n_bins=4).fit(X_train[num_cols])
    X_train_id3 = pd.concat([X_train[cat_cols], discretizer.transform(X_train[num_cols])], axis=1)
    id3 = ID3Classifier(max_depth=6, min_samples_split=20, min_info_gain=1e-4)
    id3.fit(X_train_id3, y_train)

    models = {'CART': cart, 'ID3': id3, 'Linear SVM': svm}
    print("All models ready for live inference!\n")

    # Load master comparison table if available
    master_table = None
    if os.path.exists('outputs/model_comparison_table.csv'):
        master_table = pd.read_csv('outputs/model_comparison_table.csv')

    # Run interactive prediction loop
    while True:
        interactive_terminal_prompt(models, discretizer, num_cols, cat_cols, master_table=master_table)
        try:
            again = input("\nWould you like to evaluate another profile? (y/n) [default: 'n']: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not again.startswith('y'):
            print("\nExiting interactive inference session. Goodbye!\n")
            break


if __name__ == '__main__':
    main()
