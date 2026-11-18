"""
Generator script to produce the streamlined Google Colab Notebook (.ipynb).
Removes all unnecessary sections, redundant duplicate charts, and extra boilerplate,
leaving exactly the essential sections required for the research benchmark.
"""

import json
import os

def create_streamlined_notebook():
    cells = []

    def add_cell(cell_type, source_text):
        if isinstance(source_text, list):
            lines = [l if l.endswith('\n') else l + '\n' for l in source_text]
        else:
            lines = [l + '\n' for l in source_text.strip().split('\n')]
            if lines:
                lines[-1] = lines[-1].rstrip('\n')
        cells.append({
            'cell_type': cell_type,
            'metadata': {},
            'source': lines,
            **({'execution_count': None, 'outputs': []} if cell_type == 'code' else {})
        })

    # =========================================================================
    # SECTION 1: Setup, Dependencies & Configuration
    # =========================================================================
    add_cell('markdown', r"""# Comparative Performance Analysis of CART, ID3 and Linear Support Vector Machine (SVM) Algorithms for Classification

---
### 1. Project Overview & Research Objectives
This project presents an empirical research and benchmarking framework comparing three fundamental supervised classification paradigms on demographic income prediction: **CART** (Classification and Regression Trees), **ID3** (Iterative Dichotomiser 3, built from scratch using Shannon Entropy), and **Linear SVM** (Support Vector Machine with hard/soft maximum-margin hyperplane). The system predicts whether an individual earns above \$50,000 annually ($>50\text{K}$) or standard income ($\le 50\text{K}$) using a clean 9-feature demographic and financial schema (4 numerical, 5 categorical) tailored for transparency and cognitive interpretability.

**Core Research Objectives:**
1. **Algorithmic Isolation & Benchmark Integrity:** Train, evaluate, and benchmark exactly three distinct classifiers (binary Gini splitting, multi-way Information Gain splitting, and linear margin maximization) on identical 80:20 stratified data splits with zero data leakage.
2. **Scratch vs. Library Implementation Analysis:** Compare scikit-learn's optimized CART and Linear SVM against a scratch-built ID3 engine featuring continuous quantile discretization and recursive Shannon entropy minimization.
3. **Interpretability vs. Boundary Complexity Trade-off:** Contrast the explicit rule-based hierarchical decision branches of CART and ID3 against the geometric separating hyperplane and support vector distribution of Linear SVM.
4. **Interactive Sub-Second Inference:** Provide an interactive terminal interface that prompts users for demographic attributes and immediately computes predicted income classes, posterior probabilities, the master performance table, and confusion matrices.

---
### 2. End-to-End Workflow Sequence
1. **Data Ingestion & Cleaning:** Load cached `data/adult.csv` (30,162 records) and enforce the standardized 9-feature schema.
2. **Stratified Split (80:20):** Partition training and test sets while strictly preserving the empirical class ratio ($P(y=1) \approx 0.249$).
3. **Model Training:** Fit CART (Gini impurity), ID3 (quantile-discretized Shannon entropy), and Linear SVM (StandardScaler + OneHotEncoder).
4. **Evaluation & Metric Aggregation:** Compute Accuracy, Precision, Recall, F1-Score, Balanced Accuracy, Specificity, ROC-AUC, and latency.
5. **Visualizations & Artifacts:** Generate confusion matrix heatmaps, decision tree diagrams, and export `outputs/model_comparison_table.csv`.
6. **Interactive Terminal Inference:** Live command-line custom profile prediction interface.

---
### 3. Feature and Target Formulation
* **Input Feature Subspace $\mathcal{X}$ (9 Attributes):**
  - `age`: Continuous chronological age in years ($17 \le x_1 \le 90$). Binned into 4 balanced quantile intervals for ID3.
  - `workclass`: Categorical employment sector ($x_2 \in \{\text{Private}, \text{Government}, \text{Self-Employed}\}$)
  - `education`: Formal educational level ($x_3 \in \{\text{10th}, \text{12th}, \text{Graduation}, \text{Post-Graduation}, \text{PhD}\}$)
  - `marital_status`: Legal marital status ($x_4 \in \{\text{Single}, \text{Married}, \text{Divorced}, \text{Widowed}\}$)
  - `sex`: Biological gender ($x_5 \in \{\text{Male}, \text{Female}\}$)
  - `relationship`: Role in household ($x_6 \in \{\text{Husband}, \text{Wife}, \text{Child}, \text{Other}\}$)
  - `capital_gain`: Capital investment gains in USD ($x_7 \in [0, 99999]$, zero-inflated)
  - `capital_loss`: Capital investment losses in USD ($x_8 \in [0, 4356]$, zero-inflated)
  - `hours_per_week`: Average working hours per week ($x_9 \in [1, 99]$)
* **Target Space $\mathcal{Y} = \{0, 1\}$:**
  - $y = 0 \iff \text{annual income} \le 50\text{K}$ (Standard Income Class: $\le$ \$50,000)
  - $y = 1 \iff \text{annual income} > 50\text{K}$ (High Income Class: $>$ \$50,000)

---
## Section 1: Setup, Dependencies & Experimental Configuration

**Theoretical Formulation:**
Configures the computational runtime, sets deterministic pseudorandom seeds ($\text{seed} = 42$), and initializes thread-safe directory architectures. Establishes global hyperparameters including tree depths, sample limits, and convergence tolerances to ensure strict mathematical reproducibility across all subsequent execution environments.""")

    add_cell('code', """!pip install -q numpy pandas scikit-learn matplotlib seaborn ucimlrepo
print("Dependencies installed successfully.")""")

    add_cell('code', """import os
import sys
import time
import tempfile
import warnings
import io
import urllib.request

os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from IPython.display import display
except ImportError:
    display = print

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.datasets import fetch_openml

sns.set_theme(style='whitegrid', font='sans-serif')
os.makedirs('outputs', exist_ok=True)

# Global Experimental Configuration
CONFIG = {
    'random_state': 42,
    'test_size': 0.20,
    'sample_size': 10000,
    'cart_max_depth': 8,
    'cart_min_samples_split': 20,
    'cart_min_samples_leaf': 10,
    'id3_max_depth': 6,
    'id3_min_samples_split': 20,
    'id3_min_info_gain': 1e-4,
    'svm_C': 1.0,
    'svm_kernel': 'linear'
}
np.random.seed(CONFIG['random_state'])
print("Libraries imported and experimental parameters configured.")""")

    # =========================================================================
    # SECTION 2: Dataset Acquisition & Cleaning
    # =========================================================================
    add_cell('markdown', r"""## Section 2: Dataset Acquisition & Cleaning (UCI Adult Census Dataset)

**Theoretical Formulation:**
Handles resilient dataset acquisition across local caching, OpenML, and UCI repositories while eliminating missing tokens (`?`) and unformatted string artifacts. It maps fine-grained demographic subcategories into standardized canonical levels and filters the schema down strictly to the 9 specified attributes plus binary target.""")

    add_cell('code', """DATA_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

local_candidates = ['data/adult.csv', 'data/adult_raw_backup.csv', 'adult.csv', '/content/data/adult.csv', '/content/adult.csv']
df_raw = None

# Tier 1: Local Cache Check
for lp in local_candidates:
    if os.path.exists(lp):
        try:
            print(f"Loading cached dataset from '{lp}'...")
            df_raw = pd.read_csv(lp)
            df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
            if 'class' in df_raw.columns and 'income' not in df_raw.columns:
                df_raw = df_raw.rename(columns={'class': 'income'})
            if len(df_raw.columns) >= 10:
                print("Loaded successfully from local storage!")
                break
        except Exception:
            pass

# Tier 2: Scikit-Learn OpenML (Official global CDN)
if df_raw is None:
    try:
        print("Fetching dataset via Scikit-Learn OpenML (adult, version 2)...")
        oml_adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df_raw = oml_adult.frame.copy()
        df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
        if 'class' in df_raw.columns and 'income' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'class': 'income'})
        print(f"Successfully loaded from OpenML! Shape: {df_raw.shape}")
    except Exception as e:
        print(f"OpenML notice: {e}")

# Tier 3: Official UCI Machine Learning Repository API (ucimlrepo)
if df_raw is None:
    try:
        print("Fetching dataset via official ucimlrepo (id=2)...")
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        df_raw = pd.concat([adult.data.features, adult.data.targets], axis=1)
        df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
        if 'class' in df_raw.columns and 'income' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'class': 'income'})
        print(f"Successfully loaded from ucimlrepo! Shape: {df_raw.shape}")
    except Exception as e:
        print(f"ucimlrepo notice: {e}")

# Tier 4: Direct HTTP Mirrors with Browser Headers
if df_raw is None:
    mirrors = [
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv', False),
        ('https://raw.githubusercontent.com/datasciencedojo/datasets/master/adult.data', False),
        ('https://raw.githubusercontent.com/selva86/datasets/master/Adult.csv', True),
        ('https://raw.githubusercontent.com/kdn251/interviews/master/company/facebook/adult.csv', True),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', False)
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    for url, has_header in mirrors:
        try:
            print(f"Fetching from mirror: {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
            if has_header:
                df_raw = pd.read_csv(io.BytesIO(content), skipinitialspace=True)
            else:
                df_raw = pd.read_csv(io.BytesIO(content), header=None, names=DATA_COLUMNS, skipinitialspace=True)
            df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
            if 'class' in df_raw.columns and 'income' not in df_raw.columns:
                df_raw = df_raw.rename(columns={'class': 'income'})
            print("Successfully loaded from HTTP mirror!")
            break
        except Exception:
            pass

# Tier 5: Fallback to UCI Bank Marketing Dataset
if df_raw is None:
    try:
        bank = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
        df_raw = bank.frame.copy()
        df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
        if 'Class' in df_raw.columns:
            df_raw = df_raw.rename(columns={'Class': 'income'})
    except Exception:
        raise RuntimeError("Failed to load dataset from all sources.")

# Normalize column names & cache locally
df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
if 'class' in df_raw.columns and 'income' not in df_raw.columns:
    df_raw = df_raw.rename(columns={'class': 'income'})

os.makedirs('data', exist_ok=True)
try:
    df_raw.to_csv('data/adult.csv', index=False)
except Exception:
    pass

# Data Cleaning
df_cleaned = df_raw.copy()
for col in df_cleaned.columns:
    if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.rstrip('.')

df_cleaned = df_cleaned.replace('?', np.nan).replace('None', np.nan).replace('nan', np.nan)

# Clean, intuitive 9-feature schema (4 numerical, 5 categorical):
NUMERICAL_COLS = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'education', 'marital_status', 'sex', 'relationship']
SELECTED_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

keep_cols = [c for c in SELECTED_FEATURES if c in df_cleaned.columns] + ['income']
df_cleaned = df_cleaned[keep_cols].copy()

# Attribute Category Minimization & Harmonization:
# 1. workclass: Private, Government, Self-Employed
workclass_map = {
    'Private': 'Private',
    'State-gov': 'Government', 'Federal-gov': 'Government', 'Local-gov': 'Government',
    'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed',
    'Without-pay': 'Self-Employed', 'Never-worked': 'Self-Employed'
}
if 'workclass' in df_cleaned.columns:
    df_cleaned['workclass'] = df_cleaned['workclass'].map(lambda x: workclass_map.get(str(x).strip(), 'Self-Employed'))

# 2. education: 10th, 12th, Graduation, Post-Graduation, PhD
def map_education(e):
    if pd.isna(e): return np.nan
    e_str = str(e).strip()
    if e_str in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th']:
        return '10th'
    elif e_str in ['11th', '12th', 'HS-grad']:
        return '12th'
    elif e_str in ['Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors']:
        return 'Graduation'
    elif e_str in ['Masters', 'Prof-school']:
        return 'Post-Graduation'
    elif e_str == 'Doctorate':
        return 'PhD'
    return '12th'
if 'education' in df_cleaned.columns:
    df_cleaned['education'] = df_cleaned['education'].apply(map_education)

# 3. marital_status: Single, Married, Divorced, Widowed
def map_marital(m):
    if pd.isna(m): return np.nan
    m_str = str(m).strip()
    if m_str in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
        return 'Married'
    elif m_str == 'Never-married':
        return 'Single'
    elif m_str in ['Divorced', 'Separated']:
        return 'Divorced'
    elif m_str == 'Widowed':
        return 'Widowed'
    return 'Single'
if 'marital_status' in df_cleaned.columns:
    df_cleaned['marital_status'] = df_cleaned['marital_status'].apply(map_marital)

# 4. sex: Male, Female
if 'sex' in df_cleaned.columns:
    df_cleaned['sex'] = df_cleaned['sex'].map(lambda s: 'Female' if str(s).strip().lower().startswith('f') else 'Male')

# 5. relationship: Husband, Wife, Child, Other
def map_relationship(r):
    if pd.isna(r): return np.nan
    r_str = str(r).strip()
    if r_str == 'Husband':
        return 'Husband'
    elif r_str == 'Wife':
        return 'Wife'
    elif r_str == 'Own-child':
        return 'Child'
    return 'Other'
if 'relationship' in df_cleaned.columns:
    df_cleaned['relationship'] = df_cleaned['relationship'].apply(map_relationship)

for num_c in NUMERICAL_COLS:
    if num_c in df_cleaned.columns:
        df_cleaned[num_c] = pd.to_numeric(df_cleaned[num_c], errors='coerce')

n_init = len(df_cleaned)
df_cleaned = df_cleaned.dropna().reset_index(drop=True)
n_clean = len(df_cleaned)

target_str = df_cleaned['income'].astype(str).str.strip().str.rstrip('.')
df_cleaned['income_target'] = target_str.map(lambda x: 1 if '>50k' in str(x).lower() else 0).astype(int)

print(f"Data Cleaning Completed: {n_init:,} raw -> {n_clean:,} clean records (Removed {n_init - n_clean:,} rows).")
print(f"Features Retained (9): {SELECTED_FEATURES}")
print(f"Target Distribution: <=50K (Class 0): {(df_cleaned['income_target'] == 0).sum():,} ({(df_cleaned['income_target'] == 0).mean()*100:.1f}%) | "
      f">50K (Class 1): {(df_cleaned['income_target'] == 1).sum():,} ({(df_cleaned['income_target'] == 1).mean()*100:.1f}%)")""")

    # =========================================================================
    # SECTION 3: Feature Preprocessing & Stratified Train-Test Split (80:20)
    # =========================================================================
    add_cell('markdown', r"""## Section 3: Preprocessing Architecture & Stratified Split (80:20)

**Theoretical Formulation:**
Partitions the sample into training (80%) and testing (20%) subsets using stratified sampling to preserve the empirical class ratio ($P(y=1) \approx 0.249$). Encapsulates preprocessing transformations so that statistical parameters (mean, standard deviation, and categorical bin boundaries) are computed strictly on the training partition, guaranteeing zero data leakage.""")

    add_cell('code', """class AdaptiveQuantileDiscretizer:
    \"\"\"Learns quantile boundaries strictly on training data for ID3.\"\"\"
    def __init__(self, n_bins=4):
        self.n_bins = n_bins
        self.rules_ = {}

    def fit(self, X):
        X_df = pd.DataFrame(X)
        self.rules_ = {}
        for col in X_df.columns:
            vals = X_df[col].dropna()
            if (vals == 0).mean() > 0.5:
                pos_vals = vals[vals > 0]
                self.rules_[col] = ('zero_inflated', float(np.median(pos_vals)) if len(pos_vals) > 0 else 0.0)
            else:
                qs = np.linspace(0, 1, self.n_bins + 1)
                edges = np.unique(np.quantile(vals, qs))
                edges[0], edges[-1] = -np.inf, np.inf
                self.rules_[col] = ('quantiles', edges)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_disc = pd.DataFrame(index=X_df.index)
        for col in self.rules_:
            if col not in X_df.columns:
                continue
            rule_type, rule_val = self.rules_[col]
            if rule_type == 'zero_inflated':
                conds = [X_df[col] <= 0, (X_df[col] > 0) & (X_df[col] <= rule_val), X_df[col] > rule_val]
                X_disc[col] = np.select(conds, ['Zero', 'Low_Pos', 'High_Pos'], default='Zero')
            else:
                binned = np.digitize(X_df[col], rule_val[1:-1])
                labels = [f'Q{i+1}' for i in range(len(rule_val) - 1)]
                binned = np.clip(binned, 0, len(labels) - 1)
                X_disc[col] = [labels[b] for b in binned]
        return X_disc

# Stratified Sampling (80:20 Split)
sample_size = CONFIG['sample_size']
if sample_size is not None and sample_size < len(df_cleaned):
    df_experiment, _ = train_test_split(
        df_cleaned, train_size=sample_size, stratify=df_cleaned['income_target'], random_state=CONFIG['random_state']
    )
    df_experiment = df_experiment.reset_index(drop=True)
else:
    df_experiment = df_cleaned.copy()

X_all = df_experiment.drop(columns=['income', 'income_target'])
y_all = df_experiment['income_target']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=CONFIG['test_size'], stratify=y_all, random_state=CONFIG['random_state']
)

print(f"Stratified Split Completed: {X_train_raw.shape[0]:,} Training records | {X_test_raw.shape[0]:,} Testing records.")""")

    # =========================================================================
    # SECTION 4: CART Algorithm Implementation
    # =========================================================================
    add_cell('markdown', r"""## Section 4: CART Algorithm Implementation (Gini Impurity)

**Theoretical Formulation:**
Constructs a binary decision tree by evaluating every prospective feature split $s$ to minimize Gini Impurity: $I_G(t) = 1 - \sum_{k=0}^{1} p(k|t)^2$. It selects the optimal split maximizing impurity reduction $\Delta I_G = I_G(t) - \frac{N_L}{N} I_G(t_L) - \frac{N_R}{N} I_G(t_R)$, applying pre-pruning stopping criteria to prevent overfitting.""")

    add_cell('code', """class CARTClassifierWrapper:
    def __init__(self, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42):
        self.encoder = None
        self.feature_names_ = []
        self.model = DecisionTreeClassifier(
            criterion='gini', splitter='best', max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y, cat_cols, num_cols):
        X_df = X.copy()
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.feature_names_ = self.num_cols + self.cat_cols

        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_df[self.cat_cols] = self.encoder.fit_transform(X_df[self.cat_cols])
        self.model.fit(X_df[self.feature_names_].to_numpy(), np.asarray(y).ravel())
        return self

    def predict(self, X):
        X_df = X.copy()
        X_df[self.cat_cols] = self.encoder.transform(X_df[self.cat_cols])
        return self.model.predict(X_df[self.feature_names_].to_numpy())

    def predict_proba(self, X):
        X_df = X.copy()
        X_df[self.cat_cols] = self.encoder.transform(X_df[self.cat_cols])
        return self.model.predict_proba(X_df[self.feature_names_].to_numpy())

    def get_tree_metrics(self):
        return {
            'max_depth': int(self.model.get_depth()),
            'leaf_nodes': int(self.model.get_n_leaves()),
            'total_nodes': int(self.model.tree_.node_count),
            'support_vectors': 'N/A'
        }

    def plot_tree(self, max_vis_depth=3, figsize=(22, 10), save_path=None):
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
        plt.title(f"CART Decision Tree Architecture (Gini Impurity, Root to Depth {max_vis_depth})",
                  fontsize=14, weight='bold', pad=15)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

print("CART Wrapper Ready.")""")

    # =========================================================================
    # SECTION 5: ID3 Algorithm Implementation from Scratch
    # =========================================================================
    add_cell('markdown', r"""## Section 5: ID3 Algorithm Implementation from Scratch (Shannon Entropy & Information Gain)

**Theoretical Formulation:**
Recursively builds a multi-way decision tree from scratch utilizing Shannon Entropy: $H(S) = - \sum_{k} p_k \log_2 p_k$ and Information Gain: $IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$. It incorporates an integrated adaptive quantile discretizer that converts continuous numerical variables into balanced discrete partitions to allow genuine nominal multi-way branching.""")

    add_cell('code', """class ID3Node:
    def __init__(self, feature=None, children=None, value=None, majority_class=0, prob=0.0, depth=0,
                 info_gain=0.0, samples=0, entropy=0.0):
        self.feature = feature
        self.children = children or {}
        self.value = value
        self.majority_class = majority_class
        self.prob = prob
        self.depth = depth
        self.info_gain = info_gain
        self.samples = samples
        self.entropy = entropy

    def is_leaf(self):
        return self.value is not None


class ID3Classifier:
    def __init__(self, max_depth=6, min_samples_split=20, min_info_gain=1e-4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.root = None

    def _entropy(self, y):
        n = len(y)
        if n == 0: return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return float(-np.sum(probs * np.log2(probs[probs > 0])))

    def _information_gain(self, col_values, y, base_entropy):
        n = len(y)
        if n == 0: return 0.0
        uniques, inverse = np.unique(col_values, return_inverse=True)
        if len(uniques) <= 1: return 0.0
        child_h = sum((len(y[inverse == u]) / n) * self._entropy(y[inverse == u]) for u in range(len(uniques)))
        return float(base_entropy - child_h)

    def fit(self, X, y):
        self.features_ = list(X.columns)
        self.root = self._build_tree(X.to_numpy(), np.asarray(y).ravel(), depth=0, available_features=list(range(X.shape[1])))
        return self

    def _build_tree(self, X, y, depth, available_features):
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        maj_cls = int(classes[np.argmax(counts)]) if len(classes) > 0 else 0
        prob = float(np.mean(y == 1)) if n > 0 else 0.0
        curr_h = self._entropy(y)

        if curr_h == 0.0 or depth >= self.max_depth or n < self.min_samples_split or not available_features:
            return ID3Node(value=maj_cls, majority_class=maj_cls, prob=prob, depth=depth, samples=n, entropy=curr_h)

        best_ig, best_f = -1.0, None
        for f in available_features:
            ig = self._information_gain(X[:, f], y, curr_h)
            if ig > best_ig:
                best_ig, best_f = ig, f

        if best_ig < self.min_info_gain or best_f is None:
            return ID3Node(value=maj_cls, majority_class=maj_cls, prob=prob, depth=depth, samples=n, entropy=curr_h)

        node = ID3Node(feature=self.features_[best_f], majority_class=maj_cls, prob=prob, depth=depth,
                       info_gain=best_ig, samples=n, entropy=curr_h)
        uniques, inverse = np.unique(X[:, best_f], return_inverse=True)
        rem_f = [f for f in available_features if f != best_f]

        for u_idx, u_val in enumerate(uniques):
            sub_X, sub_y = X[inverse == u_idx], y[inverse == u_idx]
            node.children[u_val] = (
                ID3Node(value=maj_cls, majority_class=maj_cls, prob=prob, depth=depth+1, samples=len(sub_y), entropy=0.0)
                if len(sub_y) == 0 else self._build_tree(sub_X, sub_y, depth+1, rem_f)
            )
        return node

    def _predict_sample(self, row, node):
        if node.is_leaf():
            return node.value, node.prob
        val = row[self.features_.index(node.feature)]
        return self._predict_sample(row, node.children[val]) if val in node.children else (node.majority_class, node.prob)

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and all(f in X.columns for f in self.features_):
            X_mat = X[self.features_].to_numpy()
        else:
            X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.array([self._predict_sample(row, self.root)[0] for row in X_mat], dtype=int)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and all(f in X.columns for f in self.features_):
            X_mat = X[self.features_].to_numpy()
        else:
            X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        probs_pos = np.array([self._predict_sample(row, self.root)[1] for row in X_mat])
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def get_tree_metrics(self):
        depths, leaves, total = [], 0, 0
        def traverse(node):
            nonlocal leaves, total
            total += 1
            if node.is_leaf():
                leaves += 1
                depths.append(node.depth)
                return
            for child in node.children.values():
                traverse(child)
        traverse(self.root)
        return {
            'max_depth': max(depths) if depths else 0,
            'leaf_nodes': leaves,
            'total_nodes': total,
            'support_vectors': 'N/A'
        }

    def plot_tree(self, max_vis_depth=3, figsize=(20, 10), save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        def get_width(node, d):
            if node.is_leaf() or d >= max_vis_depth or not node.children:
                return 1
            return sum(get_width(child, d + 1) for child in node.children.values())

        total_width = get_width(self.root, 0)
        current_x = 0

        def render_node(node, d):
            nonlocal current_x
            if node.is_leaf() or d >= max_vis_depth or not node.children:
                x = current_x + 0.5
                current_x += 1
                y = 1.0 - (d / (max_vis_depth + 1.2))
                is_pos = (node.value == 1 if node.is_leaf() else node.majority_class == 1)
                color = '#A3E4D7' if is_pos else '#AED6F1'
                txt = f"{'>50K' if is_pos else '<=50K'}\\nProb: {node.prob:.2f}\\nN={node.samples}"
                ax.text(x, y, txt, ha='center', va='center', fontsize=7.5,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='#2C3E50', lw=1.2))
                return x, y

            child_positions = []
            for val, child in list(node.children.items())[:6]:
                cx, cy = render_node(child, d + 1)
                child_positions.append((cx, cy, str(val)[:12]))

            x = sum(cp[0] for cp in child_positions) / len(child_positions)
            y = 1.0 - (d / (max_vis_depth + 1.2))

            txt = f"Split: {node.feature}\\nIG: {node.info_gain:.3f}\\nH: {node.entropy:.2f}\\nN={node.samples}"
            ax.text(x, y, txt, ha='center', va='center', fontsize=8, weight='bold',
                    bbox=dict(boxstyle='square,pad=0.4', facecolor='#FAD7A0', edgecolor='#BA4A00', lw=1.5))

            for cx, cy, lbl in child_positions:
                ax.annotate('', xy=(cx, cy + 0.035), xytext=(x, y - 0.035),
                            arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=1.0))
                mid_x = (x + cx) / 2
                mid_y = (y + cy) / 2
                ax.text(mid_x, mid_y, lbl, ha='center', va='center', fontsize=6.5,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#BDC3C7', alpha=0.9))

            return x, y

        ax.set_xlim(0, total_width + 1)
        ax.set_ylim(-0.1, 1.1)
        render_node(self.root, 0)
        plt.title(f"ID3 Decision Tree Hierarchy (Levels 0 to {max_vis_depth}) - Multi-way Splits",
                  fontsize=14, weight='bold', pad=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

print("Genuine ID3 Classifier (from scratch) Ready.")""")

    # =========================================================================
    # SECTION 6: Linear SVM Implementation
    # =========================================================================
    add_cell('markdown', r"""## Section 6: Linear SVM Implementation (Maximum Margin Hyperplane)

**Theoretical Formulation:**
Solves the primal convex quadratic programming optimization problem $\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i} \xi_i$ subject to $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$. Features are pre-processed via z-score standardization ($\mu=0, \sigma=1$) and one-hot encoding, producing a maximal-margin decision boundary accompanied by Platt-scaled posterior probabilities.""")

    add_cell('code', """class LinearSVMWrapper:
    def __init__(self, C=1.0, probability=True, random_state=42):
        self.preprocessor = None
        self.model = SVC(kernel='linear', C=C, probability=probability, random_state=random_state)

    def fit(self, X, y, num_cols, cat_cols):
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        X_proc = self.preprocessor.fit_transform(X)
        self.model.fit(X_proc, np.asarray(y).ravel())
        return self

    def predict(self, X):
        return self.model.predict(self.preprocessor.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocessor.transform(X))

    def get_svm_metrics(self):
        return {
            'max_depth': 'N/A',
            'leaf_nodes': 'N/A',
            'total_nodes': 'N/A',
            'total_support_vectors': int(self.model.n_support_.sum())
        }

print("Linear SVM Wrapper Ready.")""")

    # =========================================================================
    # SECTION 7: Model Training & Evaluation Metrics
    # =========================================================================
    add_cell('markdown', r"""## Section 7: Model Training & Evaluation Metrics Calculation

**Theoretical Formulation:**
Executes isolated training for all three models while tracking exact microsecond-level training and prediction latencies. It computes a comprehensive suite of evaluation metrics on unseen test records: Accuracy, Precision, Recall, $F_1\text{-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$, Balanced Accuracy, Specificity, and ROC-AUC.""")

    add_cell('code', """models = {}
train_times = {}
predictions = {}
probabilities = {}
pred_times = {}
metrics_results = {}

print("Training Models on Identical Observations:")

# 1. CART
cart = CARTClassifierWrapper(max_depth=CONFIG['cart_max_depth'], random_state=42)
t0 = time.perf_counter()
cart.fit(X_train_raw, y_train, cat_cols=CATEGORICAL_COLS, num_cols=NUMERICAL_COLS)
train_times['CART'] = time.perf_counter() - t0
models['CART'] = cart

# 2. ID3
discretizer = AdaptiveQuantileDiscretizer(n_bins=4).fit(X_train_raw[NUMERICAL_COLS])
X_train_d = pd.concat([X_train_raw[CATEGORICAL_COLS], discretizer.transform(X_train_raw[NUMERICAL_COLS])], axis=1)
id3 = ID3Classifier(max_depth=CONFIG['id3_max_depth'])
t0 = time.perf_counter()
id3.fit(X_train_d, y_train)
train_times['ID3'] = time.perf_counter() - t0
models['ID3'] = id3

# 3. Linear SVM
svm = LinearSVMWrapper(C=CONFIG['svm_C'], random_state=42)
t0 = time.perf_counter()
svm.fit(X_train_raw, y_train, num_cols=NUMERICAL_COLS, cat_cols=CATEGORICAL_COLS)
train_times['Linear SVM'] = time.perf_counter() - t0
models['Linear SVM'] = svm

# Generate Predictions & Measure Inference Latency
X_test_d = pd.concat([X_test_raw[CATEGORICAL_COLS], discretizer.transform(X_test_raw[NUMERICAL_COLS])], axis=1)

for a in ['CART', 'ID3', 'Linear SVM']:
    m = models[a]
    t0 = time.perf_counter()
    preds = m.predict(X_test_d if a == 'ID3' else X_test_raw)
    probs = m.predict_proba(X_test_d if a == 'ID3' else X_test_raw)[:, 1]
    pred_times[a] = time.perf_counter() - t0
    predictions[a] = preds
    probabilities[a] = probs

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics_results[a] = {
        'Accuracy': float(accuracy_score(y_test, preds)),
        'Precision': float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        'Recall': float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        'F1 Score': float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        'Balanced Accuracy': float(balanced_accuracy_score(y_test, preds)),
        'Specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        'ROC-AUC': float(roc_auc_score(y_test, probs)),
        'Training Time': float(train_times[a]),
        'Prediction Time': float(pred_times[a]),
        'Confusion Matrix': cm
    }
    print(f"  * {a:12s} | Train: {train_times[a]:.4f}s | Pred: {pred_times[a]:.4f}s | Acc: {metrics_results[a]['Accuracy']:.4f} | F1: {metrics_results[a]['F1 Score']:.4f}")""")

    # =========================================================================
    # SECTION 8: Master Comparative Performance Table
    # =========================================================================
    add_cell('markdown', """## Section 8: Master Comparative Performance Table

**Theoretical Formulation:**
Aggregates empirical metric vectors, execution latencies, and architectural complexity indicators (tree depth, leaf counts, node counts, support vectors) into a unified comparative DataFrame. It automatically parses category leaders across accuracy, precision, recall, and computational efficiency, serializing results to `outputs/model_comparison_table.csv`.""")

    add_cell('code', """cart_complex = models['CART'].get_tree_metrics()
id3_complex = models['ID3'].get_tree_metrics()
svm_complex = models['Linear SVM'].get_svm_metrics()

table_rows = []
for a in ['CART', 'ID3', 'Linear SVM']:
    m = metrics_results[a]
    if a == 'CART':
        d, l, n, sv = cart_complex['max_depth'], cart_complex['leaf_nodes'], cart_complex['total_nodes'], 'N/A'
    elif a == 'ID3':
        d, l, n, sv = id3_complex['max_depth'], id3_complex['leaf_nodes'], id3_complex['total_nodes'], 'N/A'
    else:
        d, l, n, sv = 'N/A', 'N/A', 'N/A', svm_complex['total_support_vectors']

    table_rows.append({
        'Algorithm': a,
        'Accuracy': f"{m['Accuracy']:.4f}",
        'Precision': f"{m['Precision']:.4f}",
        'Recall': f"{m['Recall']:.4f}",
        'F1 Score': f"{m['F1 Score']:.4f}",
        'Balanced Accuracy': f"{m['Balanced Accuracy']:.4f}",
        'Specificity': f"{m['Specificity']:.4f}",
        'ROC-AUC': f"{m['ROC-AUC']:.4f}",
        'Training Time (s)': f"{m['Training Time']:.4f}",
        'Prediction Time (s)': f"{m['Prediction Time']:.4f}",
        'Tree Depth': str(d),
        'Leaf Nodes': str(l),
        'Total Nodes': str(n),
        'Support Vectors': str(sv)
    })

master_table = pd.DataFrame(table_rows)
display(master_table)
master_table.to_csv('outputs/model_comparison_table.csv', index=False)
print("Saved clean table to outputs/model_comparison_table.csv")

algos = ['CART', 'ID3', 'Linear SVM']
best_acc = max(algos, key=lambda a: metrics_results[a]['Accuracy'])
best_prec = max(algos, key=lambda a: metrics_results[a]['Precision'])
best_rec = max(algos, key=lambda a: metrics_results[a]['Recall'])
best_f1 = max(algos, key=lambda a: metrics_results[a]['F1 Score'])

print("\\nBenchmark Category Leaders:")
print(f"  * Highest Accuracy:  {best_acc} ({metrics_results[best_acc]['Accuracy']:.4f})")
print(f"  * Highest Precision: {best_prec} ({metrics_results[best_prec]['Precision']:.4f})")
print(f"  * Highest Recall:    {best_rec} ({metrics_results[best_rec]['Recall']:.4f})")
print(f"  * Highest F1 Score:  {best_f1} ({metrics_results[best_f1]['F1 Score']:.4f})")""")

    # =========================================================================
    # SECTION 9: Confusion Matrix Analysis
    # =========================================================================
    add_cell('markdown', r"""## Section 9: Confusion Matrix Analysis (CART vs. ID3 vs. Linear SVM)

**Theoretical Formulation:**
Computes and renders side-by-side $2 \times 2$ contingency heatmaps displaying True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), and False Negatives ($FN$) for each algorithm. This reveals whether a model's error mode is dominated by Type I errors (over-predicting high income) or Type II errors (failing to identify high earners).""")

    add_cell('code', """fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_labels = ['<=50K', '>50K']

for idx, a in enumerate(['CART', 'ID3', 'Linear SVM']):
    ax = axes[idx]
    sns.heatmap(
        metrics_results[a]['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={'fontsize': 12, 'weight': 'bold'}, ax=ax
    )
    ax.set_title(
        f"{a} Confusion Matrix\\nAcc: {metrics_results[a]['Accuracy']:.4f} | F1: {metrics_results[a]['F1 Score']:.4f}",
        fontsize=12, weight='bold', pad=10
    )
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()""")

    # =========================================================================
    # SECTION 10: Decision Tree Architecture Visualizations
    # =========================================================================
    add_cell('markdown', """## Section 10: Decision Tree Architecture Visualizations (CART & ID3)

**Theoretical Formulation:**
Graphically renders the internal decision logic of the tree-based models: CART's binary splits with sample distributions and Gini scores, alongside ID3's multi-way nominal decision paths with Information Gain and node entropies. This provides direct human interpretability of the learned rules governing high versus standard income classifications.""")

    add_cell('code', """# 1. Visualize CART Decision Tree Diagram
print("Rendering CART Decision Tree Diagram...")
cart.plot_tree(max_vis_depth=3, save_path='outputs/cart_decision_tree.png')
plt.show()

# 2. Visualize ID3 Decision Tree Diagram
print("Rendering ID3 Decision Tree Diagram...")
id3.plot_tree(max_vis_depth=3, save_path='outputs/id3_decision_tree.png')
plt.show()""")

    # =========================================================================
    # SECTION 11: Interactive Colab Terminal Prediction & Performance Matrix
    # =========================================================================
    add_cell('markdown', """## Section 11: Interactive Colab Terminal Prediction & Performance Matrix

**Theoretical Formulation:**
Provides a live command-line interface in Google Colab that sequentially prompts users for demographic inputs with numbered choices and intelligent defaults. It routes the vector through the trained pipelines to display predicted income, model confidence percentages, the master benchmark comparison table, and side-by-side confusion matrices in a single view.""")

    add_cell('code', """def predict_income(sample_dict, models, discretizer, num_cols, cat_cols):
    \"\"\"Generates comparative predictions for a custom user input record across all 3 models.\"\"\"
    sample_df = pd.DataFrame([sample_dict])

    # Standardize & Normalize category labels for consistency
    workclass_norm = {
        'Private': 'Private',
        'State-gov': 'Government', 'Federal-gov': 'Government', 'Local-gov': 'Government',
        'Government': 'Government', 'Govt': 'Government',
        'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed', 'Self-Employed': 'Self-Employed',
        'Without-pay': 'Self-Employed', 'Never-worked': 'Self-Employed'
    }
    edu_norm = {
        'Preschool': '10th', '1st-4th': '10th', '5th-6th': '10th', '7th-8th': '10th', '9th': '10th', '10th': '10th',
        '11th': '12th', '12th': '12th', 'HS-grad': '12th', 'School': '12th',
        'Some-college': 'Graduation', 'Assoc-voc': 'Graduation', 'Assoc-acdm': 'Graduation', 'Diploma': 'Graduation',
        'Bachelors': 'Graduation', 'Graduation': 'Graduation',
        'Masters': 'Post-Graduation', 'Prof-school': 'Post-Graduation', 'Post-Graduation': 'Post-Graduation',
        'Doctorate': 'PhD', 'PhD': 'PhD'
    }
    marital_norm = {
        'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Married', 'Married-AF-spouse': 'Married', 'Married': 'Married',
        'Never-married': 'Single', 'Single': 'Single',
        'Divorced': 'Divorced', 'Separated': 'Divorced',
        'Widowed': 'Widowed'
    }
    rel_norm = {
        'Husband': 'Husband',
        'Wife': 'Wife',
        'Own-child': 'Child', 'Child': 'Child',
        'Not-in-family': 'Other', 'Unmarried': 'Other', 'Other-relative': 'Other', 'Other': 'Other'
    }
    if 'workclass' in sample_df.columns:
        sample_df['workclass'] = sample_df['workclass'].map(lambda x: workclass_norm.get(str(x).strip(), 'Self-Employed'))
    if 'education' in sample_df.columns:
        sample_df['education'] = sample_df['education'].map(lambda x: edu_norm.get(str(x).strip(), '12th'))
    if 'marital_status' in sample_df.columns:
        sample_df['marital_status'] = sample_df['marital_status'].map(lambda x: marital_norm.get(str(x).strip(), 'Single'))
    if 'sex' in sample_df.columns:
        sample_df['sex'] = sample_df['sex'].map(lambda s: 'Female' if str(s).strip().lower().startswith('f') else 'Male')
    if 'relationship' in sample_df.columns:
        sample_df['relationship'] = sample_df['relationship'].map(lambda r: rel_norm.get(str(r).strip(), 'Other'))

    sample_disc = pd.concat([
        sample_df[cat_cols],
        discretizer.transform(sample_df[num_cols])
    ], axis=1)

    labels = {0: '<=50K', 1: '>50K'}
    results = []

    for algo in ['CART', 'ID3', 'Linear SVM']:
        m = models[algo]
        if algo == 'ID3':
            pred = m.predict(sample_disc)[0]
            prob_pos = float(m.predict_proba(sample_disc)[0][1])
        else:
            pred = m.predict(sample_df)[0]
            prob_pos = float(m.predict_proba(sample_df)[0][1])

        results.append({
            'Algorithm': algo,
            'Predicted Income': labels[pred],
            'Confidence (>50K)': f"{prob_pos * 100:.2f}%",
            'Confidence (<=50K)': f"{(1.0 - prob_pos) * 100:.2f}%"
        })

    return pd.DataFrame(results)


def interactive_terminal_prompt(models, discretizer, num_cols, cat_cols, metrics_results=None, master_table=None):
    \"\"\"
    Interactive terminal input prompt in Google Colab.
    Prompts for input, computes predictions, and displays the performance matrix and confusion matrices.
    \"\"\"
    if metrics_results is None and 'metrics_results' in globals():
        metrics_results = globals()['metrics_results']
    if master_table is None and 'master_table' in globals():
        master_table = globals()['master_table']

    print("=" * 80)
    print("INTERACTIVE INCOME PREDICTION TERMINAL (CART vs. ID3 vs. Linear SVM)")
    print("=" * 80)
    print("Enter individual feature values at each prompt below.")
    print("TIP: Press [Enter] without typing to accept the suggested default value.\\n")

    def ask_num(prompt_text, default_val, is_float=False):
        while True:
            try:
                val_str = input(f"{prompt_text} [default: {default_val}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return default_val
            if not val_str:
                return default_val
            try:
                return float(val_str) if is_float else int(val_str)
            except ValueError:
                print(f"  [!] Invalid number. Please enter a valid {'number' if is_float else 'integer'}.")

    def ask_choice(prompt_text, choices, default_val):
        opts_display = ", ".join(f"[{i+1}] {c}" for i, c in enumerate(choices))
        while True:
            try:
                val_str = input(f"{prompt_text}\\n  Options: {opts_display}\\n  Your Selection [default: '{default_val}']: ").strip()
            except (EOFError, KeyboardInterrupt):
                return default_val
            if not val_str:
                return default_val
            if val_str.isdigit() and 1 <= int(val_str) <= len(choices):
                return choices[int(val_str) - 1]
            matched = [c for c in choices if c.lower() == val_str.lower()]
            if matched:
                return matched[0]
            partial = [c for c in choices if val_str.lower() in c.lower()]
            if partial:
                return partial[0]
            return val_str

    # 1. Age
    age = ask_num("1. Age (in years)", 28)

    # 2. Employment Sector (Workclass)
    workclass_choices = ['Private', 'Government', 'Self-Employed']
    workclass = ask_choice("2. Workclass (Employment Sector)", workclass_choices, 'Private')

    # 3. Education Qualification
    education_choices = ['10th', '12th', 'Graduation', 'Post-Graduation', 'PhD']
    education = ask_choice("3. Education Qualification", education_choices, 'Graduation')

    # 4. Marital Status
    marital_choices = ['Single', 'Married', 'Divorced', 'Widowed']
    marital_status = ask_choice("4. Marital Status", marital_choices, 'Single')

    # 5. Gender
    sex_choices = ['Male', 'Female']
    sex = ask_choice("5. Gender (Sex)", sex_choices, 'Male')

    # 6. Relationship in Household
    rel_choices = ['Husband', 'Wife', 'Child', 'Other']
    relationship = ask_choice("6. Relationship in Household", rel_choices, 'Husband')

    # 7. Capital Gain
    capital_gain = ask_num("7. Capital Gain (USD)", 0)

    # 8. Capital Loss
    capital_loss = ask_num("8. Capital Loss (USD)", 0)

    # 9. Working Hours per Week
    hours_per_week = ask_num("9. Working Hours per Week", 40)

    user_dict = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital_status': marital_status,
        'sex': sex,
        'relationship': relationship,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week
    }

    print("\\n" + "-" * 80)
    print("COLLECTED PROFILE SUMMARY:")
    for k, v in user_dict.items():
        print(f"  * {k:18s}: {v}")
    print("-" * 80)

    res_df = predict_income(user_dict, models, discretizer, num_cols, cat_cols)
    print("\\n" + "=" * 80)
    print("PREDICTION RESULTS ACROSS ALGORITHMS:")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80 + "\\n")

    # Display Master Performance Metrics Matrix
    if master_table is not None:
        print("=" * 80)
        print("MASTER PERFORMANCE METRICS MATRIX (CART vs. ID3 vs. Linear SVM):")
        print("=" * 80)
        print(master_table.to_string(index=False))
        print("=" * 80 + "\\n")

    # Display Confusion Matrix for all three algorithms
    if metrics_results is not None:
        print("=" * 80)
        print("CONFUSION MATRIX FOR ALL THREE ALGORITHMS (Test Set Evaluation):")
        print("=" * 80)
        for algo in ['CART', 'ID3', 'Linear SVM']:
            if algo in metrics_results and 'Confusion Matrix' in metrics_results[algo]:
                cm = metrics_results[algo]['Confusion Matrix']
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                total = tn + fp + fn + tp
                acc = (tp + tn) / total if total > 0 else 0
                print(f"\\n[{algo} Algorithm Confusion Matrix]")
                print("                       Predicted <=50K       Predicted >50K")
                print(f"  Actual <=50K (Neg)      {tn:8,d} (TN)          {fp:8,d} (FP)")
                print(f"  Actual >50K  (Pos)      {fn:8,d} (FN)          {tp:8,d} (TP)")
                print(f"  -> Accuracy: {acc*100:.2f}% | TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
        print("=" * 80 + "\\n")

    return res_df""")

    add_cell('markdown', r"""### 11.1 Live Interactive Prediction Form (GUI)
Use the interactive sliders and dropdowns below to test arbitrary demographic profiles in real time with automatic re-evaluation across all three models!""")

    add_cell('code', """#@title 🎯 Live Interactive Demographic Predictor { run: "auto" }
#@markdown Adjust the demographic and financial parameters below to immediately generate predictions across **CART**, **ID3**, and **Linear SVM**:

age = 35 #@param {type:"slider", min:17, max:90, step:1}
workclass = "Private" #@param ["Private", "Government", "Self-Employed"]
education = "Graduation" #@param ["10th", "12th", "Graduation", "Post-Graduation", "PhD"]
marital_status = "Married" #@param ["Single", "Married", "Divorced", "Widowed"]
sex = "Male" #@param ["Male", "Female"]
relationship = "Husband" #@param ["Husband", "Wife", "Child", "Other"]
capital_gain = 0 #@param {type:"integer"}
capital_loss = 0 #@param {type:"integer"}
hours_per_week = 40 #@param {type:"slider", min:1, max:99, step:1}

custom_profile = {
    'age': int(age),
    'workclass': str(workclass),
    'education': str(education),
    'marital_status': str(marital_status),
    'sex': str(sex),
    'relationship': str(relationship),
    'capital_gain': int(capital_gain),
    'capital_loss': int(capital_loss),
    'hours_per_week': int(hours_per_week)
}

# Run live prediction across all three models
live_prediction_df = predict_income(custom_profile, models, discretizer, NUMERICAL_COLS, CATEGORICAL_COLS)

print("=" * 80)
print(f"PROFILE: Age={age} | {workclass} | {education} | {marital_status} | {sex} | {relationship} | Gain=${capital_gain:,} | Loss=${capital_loss:,} | {hours_per_week}h/wk")
print("=" * 80)
display(live_prediction_df)
print("=" * 80)""")

    add_cell('markdown', r"""### 11.2 Step-by-Step Console Input (Optional)
Run this cell if you want to enter features one by one via text prompts in the console window.""")

    add_cell('code', """# RUN INTERACTIVE CONSOLE PROMPT IN GOOGLE COLAB
interactive_terminal_prompt(models, discretizer, NUMERICAL_COLS, CATEGORICAL_COLS,
                            metrics_results=metrics_results, master_table=master_table)""")

    nb_data = {
        'cells': cells,
        'metadata': {
            'colab': {'provenance': [], 'toc_visible': True},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'}
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }

    nb_path = 'Comparative_Performance_Analysis_CART_ID3_LinearSVM.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=2)

    print(f"\n[Generator] Successfully built streamlined notebook with {len(cells)} cells at {nb_path}!")
    return nb_path

if __name__ == '__main__':
    create_streamlined_notebook()
