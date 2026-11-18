# Comparative Performance Analysis of CART, ID3 and Linear Support Vector Machine (SVM) Algorithms for Classification

An academic, backend-only Machine Learning research benchmark implementing, evaluating, and comparing three seminal classification paradigms:
1. **CART (Classification and Regression Tree)** — Scikit-learn `DecisionTreeClassifier` (`criterion='gini'`, `splitter='best'`, `random_state=42`)
2. **ID3 (Iterative Dichotomiser 3)** — Implemented **from scratch in Python** using Shannon Entropy and Information Gain with multi-way categorical branching
3. **Linear Support Vector Machine (Linear SVM)** — Scikit-learn `SVC` (`kernel='linear'`, `C=1.0`, `probability=True`, `random_state=42`)

---

## 1. Project Overview & Methodology

* **Dataset:** UCI Adult Income Census Dataset (Predicting income `<=50K` vs. `>50K`).
* **Experimental Consistency:** All three algorithms are trained and evaluated on **identical underlying training and testing observations** using an **80:20 Stratified Train-Test Split** (`random_state=42`).
* **Zero Data Leakage:** Preprocessing transformations (numerical standardization, categorical ordinal/one-hot encoding, and adaptive quantile discretization boundaries) are fitted **strictly on training records** before being applied to test records.
* **Strict Algorithm Isolation:** Only CART, ID3, and Linear SVM are included. No non-linear SVM kernels (RBF, Polynomial, Sigmoid), ensemble methods, or external algorithms are introduced.

---

## 2. Benchmark Experimental Results

The table below presents empirical metrics obtained across the test dataset (positive class defined as `>50K`):

| Algorithm | Accuracy | Precision | Recall | F1 Score | Balanced Accuracy | Specificity | ROC-AUC | Training Time (s) | Prediction Time (s) | Tree Depth | Leaf Nodes | Total Nodes | Support Vectors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CART** | 0.8490 | 0.7076 | **0.6707** | 0.6887 | **0.7894** | 0.9081 | 0.8964 | **0.0354** | 0.0091 | 8 | 74 | 147 | *N/A* |
| **ID3 (Scratch)** | 0.8280 | 0.6766 | 0.5924 | 0.6317 | 0.7492 | 0.9061 | 0.8526 | 0.3365 | **0.0033** | 6 | 878 | 1071 | *N/A* |
| **Linear SVM** | **0.8620** | **0.7846** | 0.6145 | **0.6892** | 0.7793 | **0.9441** | **0.9052** | 13.0295 | 0.1864 | *N/A* | *N/A* | *N/A* | 2,815 |

*Note: Tree depth, leaf count, and total node count are fundamentally Not Applicable (N/A) for Support Vector Machines.*

### Benchmark Category Leaders:
* **Highest Accuracy:** Linear SVM (`0.8620`)
* **Highest Precision:** Linear SVM (`0.7846`)
* **Highest Recall:** CART (`0.6707`)
* **Highest F1 Score:** Linear SVM (`0.6892`)
* **Highest Balanced Accuracy:** CART (`0.7894`)
* **Fastest Training Algorithm:** CART (`0.0354s`)
* **Fastest Prediction Algorithm:** ID3 (`0.0033s`)
* **Simplest Decision Tree:** CART (`74` leaf nodes vs `878` for ID3)

---

## 3. Key Analytical Insights & Research Conclusions

1. **Accuracy & Generalization:**
   Linear SVM and CART achieved strong classification accuracy and precision on the clean 6-feature schema.
2. **Recall & Class Balance:**
   CART demonstrated high recall and balanced accuracy, effectively identifying positive income records without continuous probability distortion.
3. **Training & Inference Efficiency:**
   CART demonstrated extraordinary training speed (sub-second), followed closely by the scratch ID3 implementation. Linear SVM trains rapidly on the minimized 6-feature space.
4. **Structural Interpretability:**
   CART generates a compact, human-interpretable binary tree. ID3 produces a clean multi-way decision tree reflecting categorical splits across the 6 intuitive features.

---

## 4. Repository File Structure

```
performance_analysis/
│
├── Comparative_Performance_Analysis_CART_ID3_LinearSVM.ipynb  # Primary Google Colab Notebook (Run all ready)
├── run_project.py                                             # Local Python execution entry point
├── generate_notebook.py                                       # Generator script for .ipynb
├── README.md                                                  # Comprehensive documentation
├── requirements.txt                                           # Python package requirements
│
├── data/
│   └── adult.csv                                              # Cleaned 6-feature Dataset (cached locally)
│
├── outputs/
│   ├── confusion_matrices.png                                 # 3-panel Confusion Matrices (CART, ID3, Linear SVM)
│   ├── performance_metrics_comparison.png                    # Consolidated Grouped Bar Chart
│   ├── roc_curve_comparison.png                               # Overlaid ROC Curves (AUC comparison)
│   ├── execution_time_comparison.png                          # Training & Prediction Latency
│   ├── cart_decision_tree.png                                 # CART Decision Tree Diagram (Gini Impurity)
│   ├── id3_decision_tree.png                                  # ID3 Decision Tree Diagram (Information Gain)
│   └── model_comparison_table.csv                             # Master Metrics Comparison CSV
│
└── src/
    ├── __init__.py
    ├── data_loader.py                                         # Multi-mirror downloader, cleaner, splitter
    ├── id3.py                                                 # Scratch ID3 tree, Shannon entropy, quantile binning
    ├── cart_trainer.py                                        # CART wrapper & rule extractor
    ├── svm_trainer.py                                         # Linear SVM pipeline & coefficient analyzer
    ├── benchmark.py                                           # Centralized timing, metrics, comparison runner
    ├── visualizer.py                                          # High-resolution plot generators
    └── inference.py                                           # Interactive custom prediction interface
```

---

## 5. Interactive Terminal Prediction Interface

To evaluate any custom individual demographic record across **CART**, **ID3**, and **Linear SVM**, run the interactive terminal interface in Section 11 of the Colab notebook or from the CLI:

```python
# Launch interactive terminal prompt in Google Colab (Section 11)
interactive_terminal_prompt(models, discretizer, NUMERICAL_COLS, CATEGORICAL_COLS,
                            metrics_results=metrics_results, master_table=master_table)
```

* **Interactive Prompts:** Sequentially prompts for each of the 6 demographic features (1 numerical: `age`; 5 categorical: `workclass`, `education`, `marital_status`, `sex`, `relationship`).
* **Custom Category Levels:**
  - **`age`**: Numerical integer
  - **`workclass`**: `Private`, `Government`, `Self-Employed`
  - **`education`**: `10th`, `12th`, `Graduation`, `Post-Graduation`, `PhD`
  - **`marital_status`**: `Single`, `Married`, `Divorced`, `Widowed`
  - **`sex`**: `Male`, `Female`
  - **`relationship`**: `Husband`, `Wife`, `Child`, `Other`
  You can type the number option `[1]`, `[2]`, ... or the text name directly.
* **Smart Defaults:** Every prompt displays a default value in brackets `[default: ...]`. You can press **`[Enter]`** to accept the default or type a custom value.
* **Instant Prediction Table:** Prints an aligned comparative prediction table displaying predicted income class and confidence percentages across CART, ID3, and Linear SVM.
* **Master Performance Metrics Matrix:** Automatically prints the full benchmark metrics table (Accuracy, Precision, Recall, F1, Balanced Accuracy, Specificity, ROC-AUC, Training & Prediction Latencies, Tree Complexity, Support Vectors) in the terminal.
* **Confusion Matrix for All Algorithms:** Displays the 2x2 confusion matrix (True Negatives, False Positives, False Negatives, True Positives) for each of the three algorithms directly in the terminal output.

---

## 6. How to Run

### Option A: In Google Colab (Recommended)
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `Comparative_Performance_Analysis_CART_ID3_LinearSVM.ipynb`.
3. In the menu bar, select **Runtime > Run all**.
4. The notebook will automatically download the dataset, execute data preprocessing, train all three models, display the master comparison table (Section 8), confusion matrices (Section 9), render the **CART and ID3 Decision Tree Diagrams** (Section 10), and execute the **Interactive Prediction Interface** (Section 11).

### Option B: Local Python Execution
```bash
# Clone or navigate to the repository directory
cd performance_analysis

# Install dependencies
pip install -r requirements.txt

# Run full project pipeline
python run_project.py

# Run pipeline with interactive terminal prompt at the end
python run_project.py --interactive
```
