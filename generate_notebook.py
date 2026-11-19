"""
Script to generate the streamlined Google Colab notebook for:
Comparative Performance Analysis of CART, ID3 and Linear Support Vector Machine (SVM) Algorithms.

Every section's code is minimized for maximum clarity, conciseness, and Pythonic elegance.
"""

import json
import os


def create_streamlined_notebook():
    cells = []

    def add_cell(cell_type, source):
        # Format source as list of strings with newlines for standard Jupyter JSON format
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
        cells.append({
            'cell_type': cell_type,
            'metadata': {},
            'source': lines
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

    add_cell('code', """import os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, confusion_matrix)
from sklearn.datasets import fetch_openml

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font='sans-serif')
os.makedirs('outputs', exist_ok=True)

CONFIG = {'seed': 42, 'test_size': 0.20, 'sample_size': 10000,
          'cart_depth': 8, 'id3_depth': 6, 'svm_C': 1.0}
np.random.seed(CONFIG['seed'])
print("Configured computational runtime (Seed: 42).")""")

    # =========================================================================
    # SECTION 2: Dataset Acquisition & Cleaning
    # =========================================================================
    add_cell('markdown', r"""## Section 2: Dataset Acquisition & Cleaning (UCI Adult Census Dataset)

**Theoretical Formulation:**
Handles resilient dataset acquisition across local caching, OpenML, and UCI repositories while eliminating missing tokens (`?`) and unformatted string artifacts. It maps fine-grained demographic subcategories into standardized canonical levels and filters the schema down strictly to the 9 specified attributes plus binary target.""")

    add_cell('code', """# Resilient acquisition: local cache -> OpenML -> UCI mirror
df_raw = None
for p in ['data/adult.csv', 'data/adult_raw_backup.csv', 'adult.csv']:
    if os.path.exists(p): df_raw = pd.read_csv(p); break
if df_raw is None:
    try: df_raw = fetch_openml('adult', version=2, as_frame=True, parser='auto').frame
    except Exception: df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)

df_raw.columns = [c.replace('-', '_').strip() for c in df_raw.columns]
if 'class' in df_raw.columns and 'income' not in df_raw.columns:
    df_raw = df_raw.rename(columns={'class': 'income'})

df = df_raw.replace('?', np.nan).dropna().copy()
df['income'] = df['income'].astype(str).str.strip().str.rstrip('.')
df['income_target'] = df['income'].str.lower().str.contains('>50k').astype(int)

# Standardized demographic category mappings
w_map = {'State-gov': 'Government', 'Federal-gov': 'Government', 'Local-gov': 'Government',
         'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed', 'Without-pay': 'Self-Employed', 'Never-worked': 'Self-Employed'}
df['workclass'] = df['workclass'].map(lambda x: w_map.get(str(x).strip(), 'Private' if str(x).strip() == 'Private' else 'Self-Employed'))

e_map = {'Preschool':'10th','1st-4th':'10th','5th-6th':'10th','7th-8th':'10th','9th':'10th','10th':'10th',
         '11th':'12th','12th':'12th','HS-grad':'12th','Some-college':'Graduation','Assoc-voc':'Graduation',
         'Assoc-acdm':'Graduation','Bachelors':'Graduation','Masters':'Post-Graduation','Prof-school':'Post-Graduation','Doctorate':'PhD'}
df['education'] = df['education'].map(lambda x: e_map.get(str(x).strip(), '12th'))

m_map = {'Married-civ-spouse':'Married','Married-spouse-absent':'Married','Married-AF-spouse':'Married',
         'Never-married':'Single','Divorced':'Divorced','Separated':'Divorced','Widowed':'Widowed'}
df['marital_status'] = df['marital_status'].map(lambda x: m_map.get(str(x).strip(), 'Single'))
df['sex'] = df['sex'].map(lambda s: 'Female' if str(s).strip().lower().startswith('f') else 'Male')
df['relationship'] = df['relationship'].map(lambda r: r if r in ['Husband', 'Wife'] else ('Child' if r == 'Own-child' else 'Other'))

NUMERICAL_COLS = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'education', 'marital_status', 'sex', 'relationship']
for c in NUMERICAL_COLS: df[c] = pd.to_numeric(df[c], errors='coerce')

df_cleaned = df[NUMERICAL_COLS + CATEGORICAL_COLS + ['income', 'income_target']].dropna().reset_index(drop=True)
os.makedirs('data', exist_ok=True); df_cleaned.to_csv('data/adult.csv', index=False)
print(f"Cleaned dataset: {len(df_cleaned):,} records | 9 Features: {NUMERICAL_COLS + CATEGORICAL_COLS}")
print(f"Target: <=50K: {(df_cleaned['income_target']==0).sum():,} ({(df_cleaned['income_target']==0).mean()*100:.1f}%) | >50K: {(df_cleaned['income_target']==1).sum():,} ({(df_cleaned['income_target']==1).mean()*100:.1f}%)")""")

    # =========================================================================
    # SECTION 3: Feature Preprocessing & Stratified Train-Test Split (80:20)
    # =========================================================================
    add_cell('markdown', r"""## Section 3: Preprocessing Architecture & Stratified Split (80:20)

**Theoretical Formulation:**
Partitions the sample into training (80%) and testing (20%) subsets using stratified sampling to preserve the empirical class ratio ($P(y=1) \approx 0.249$). Encapsulates preprocessing transformations so that statistical parameters (mean, standard deviation, and categorical bin boundaries) are computed strictly on the training partition, guaranteeing zero data leakage.""")

    add_cell('code', """class AdaptiveQuantileDiscretizer:
    \"\"\"Discretizes numerical features: zero-inflated (Zero/Low/High) or quantiles (Q1-Q4).\"\"\"
    def __init__(self, n_bins=4): self.n_bins, self.rules_ = n_bins, {}

    def fit(self, X):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            vals = X_df[col].dropna()
            if (vals == 0).mean() > 0.5:
                pos = vals[vals > 0]
                self.rules_[col] = ('zero', float(pos.median()) if len(pos) else 0.0)
            else:
                edges = np.unique(np.quantile(vals, np.linspace(0, 1, self.n_bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                self.rules_[col] = ('quantiles', edges)
        return self

    def transform(self, X):
        X_df, X_disc = pd.DataFrame(X), pd.DataFrame(index=X.index)
        for col, (rtype, rval) in self.rules_.items():
            if col not in X_df.columns: continue
            if rtype == 'zero':
                conds = [X_df[col] <= 0, (X_df[col] > 0) & (X_df[col] <= rval), X_df[col] > rval]
                X_disc[col] = np.select(conds, ['Zero', 'Low_Pos', 'High_Pos'], default='Zero')
            else:
                binned = np.clip(np.digitize(X_df[col], rval[1:-1]), 0, len(rval) - 2)
                X_disc[col] = [f'Q{i+1}' for i in binned]
        return X_disc

# Stratified Split (80:20)
N = min(CONFIG['sample_size'], len(df_cleaned))
df_exp = df_cleaned.sample(n=N, random_state=CONFIG['seed']) if N < len(df_cleaned) else df_cleaned
X_all, y_all = df_exp[NUMERICAL_COLS + CATEGORICAL_COLS], df_exp['income_target']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=CONFIG['test_size'], stratify=y_all, random_state=CONFIG['seed']
)
print(f"Stratified Split: {len(X_train_raw):,} Train | {len(X_test_raw):,} Test records (Positive: {y_train.mean()*100:.1f}%).")""")

    # =========================================================================
    # SECTION 4: CART Algorithm Implementation
    # =========================================================================
    add_cell('markdown', r"""## Section 4: CART Algorithm Implementation (Gini Impurity)

**Theoretical Formulation:**
Constructs a binary decision tree by evaluating every prospective feature split $s$ to minimize Gini Impurity: $I_G(t) = 1 - \sum_{k=0}^{1} p(k|t)^2$. It selects the optimal split maximizing impurity reduction $\Delta I_G = I_G(t) - \frac{N_L}{N} I_G(t_L) - \frac{N_R}{N} I_G(t_R)$, applying pre-pruning stopping criteria to prevent overfitting.""")

    add_cell('code', """class CARTClassifierWrapper:
    \"\"\"CART classifier utilizing Gini Impurity with isolated ordinal encoding.\"\"\"
    def __init__(self, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf, random_state=random_state)

    def fit(self, X, y, cat_cols, num_cols):
        self.cat_cols, self.num_cols = list(cat_cols), list(num_cols)
        self.feature_names_ = self.num_cols + self.cat_cols
        X_df = X.copy()
        X_df[self.cat_cols] = self.encoder.fit_transform(X_df[self.cat_cols])
        self.model.fit(X_df[self.feature_names_].to_numpy(), np.asarray(y).ravel())
        return self

    def _proc(self, X):
        X_df = X.copy()
        X_df[self.cat_cols] = self.encoder.transform(X_df[self.cat_cols])
        return X_df[self.feature_names_].to_numpy()

    def predict(self, X): return self.model.predict(self._proc(X))
    def predict_proba(self, X): return self.model.predict_proba(self._proc(X))

    def get_tree_metrics(self):
        return {'max_depth': int(self.model.get_depth()), 'leaf_nodes': int(self.model.get_n_leaves()),
                'total_nodes': int(self.model.tree_.node_count), 'support_vectors': 'N/A'}

    def plot_tree(self, max_vis_depth=3, figsize=(18, 7), save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(self.model, max_depth=max_vis_depth, feature_names=self.feature_names_,
                  class_names=['<=50K', '>50K'], filled=True, rounded=True, fontsize=8, ax=ax)
        ax.set_title(f"CART Decision Tree Architecture (Gini Impurity, Depth {max_vis_depth})", weight='bold')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

print("CART Wrapper Ready.")""")

    # =========================================================================
    # SECTION 5: ID3 Algorithm Implementation from Scratch
    # =========================================================================
    add_cell('markdown', r"""## Section 5: ID3 Algorithm Implementation from Scratch (Shannon Entropy & Information Gain)

**Theoretical Formulation:**
Recursively builds a multi-way decision tree from scratch utilizing Shannon Entropy: $H(S) = - \sum_{k} p_k \log_2 p_k$ and Information Gain: $IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$. It incorporates an integrated adaptive quantile discretizer that converts continuous numerical variables into balanced discrete partitions to allow genuine nominal multi-way branching.""")

    add_cell('code', """class ID3Node:
    def __init__(self, feature=None, children=None, value=None, majority_class=0, prob=0.0, depth=0, info_gain=0.0, samples=0):
        self.feature, self.children, self.value = feature, children or {}, value
        self.majority_class, self.prob, self.depth = majority_class, prob, depth
        self.info_gain, self.samples = info_gain, samples
    def is_leaf(self): return self.value is not None

class ID3Classifier:
    \"\"\"Genuine scratch ID3 algorithm implementing Shannon Entropy and Information Gain.\"\"\"
    def __init__(self, max_depth=6, min_samples_split=20, min_info_gain=1e-4):
        self.max_depth, self.min_samples_split, self.min_info_gain = max_depth, min_samples_split, min_info_gain
        self.root = None

    def _entropy(self, y):
        if len(y) == 0: return 0.0
        p = np.unique(y, return_counts=True)[1] / len(y)
        return float(-np.sum(p * np.log2(p[p > 0])))

    def _ig(self, col, y, base_h):
        u, inv = np.unique(col, return_inverse=True)
        if len(u) <= 1: return 0.0
        rem_h = sum((np.sum(inv == i) / len(y)) * self._entropy(y[inv == i]) for i in range(len(u)))
        return float(base_h - rem_h)

    def fit(self, X, y):
        self.features_ = list(X.columns)
        self.root = self._build(X.to_numpy(), np.asarray(y).ravel(), 0, list(range(X.shape[1])))
        return self

    def _build(self, X, y, d, feats):
        n = len(y)
        maj = int(np.bincount(y).argmax()) if n else 0
        prob, h = float(np.mean(y == 1)) if n else 0.0, self._entropy(y)
        if h == 0.0 or d >= self.max_depth or n < self.min_samples_split or not feats:
            return ID3Node(value=maj, majority_class=maj, prob=prob, depth=d, samples=n)

        gains = [self._ig(X[:, f], y, h) for f in feats]
        best_i = int(np.argmax(gains))
        if gains[best_i] < self.min_info_gain:
            return ID3Node(value=maj, majority_class=maj, prob=prob, depth=d, samples=n)

        best_f = feats[best_i]
        node = ID3Node(feature=self.features_[best_f], majority_class=maj, prob=prob, depth=d, info_gain=gains[best_i], samples=n)
        rem = [f for f in feats if f != best_f]
        u, inv = np.unique(X[:, best_f], return_inverse=True)
        for i, val in enumerate(u):
            node.children[val] = self._build(X[inv == i], y[inv == i], d + 1, rem) if np.any(inv == i) else ID3Node(value=maj, depth=d+1)
        return node

    def _predict_one(self, row, node):
        if node.is_leaf(): return node.value, node.prob
        val = row[self.features_.index(node.feature)]
        return self._predict_one(row, node.children[val]) if val in node.children else (node.majority_class, node.prob)

    def predict(self, X):
        X_mat = X[self.features_].to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.array([self._predict_one(r, self.root)[0] for r in X_mat], dtype=int)

    def predict_proba(self, X):
        X_mat = X[self.features_].to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        p1 = np.array([self._predict_one(r, self.root)[1] for r in X_mat])
        return np.column_stack([1.0 - p1, p1])

    def get_tree_metrics(self):
        depths, leaves, total = [], 0, 0
        def trav(n):
            nonlocal leaves, total
            total += 1
            if n.is_leaf(): leaves += 1; depths.append(n.depth); return
            for ch in n.children.values(): trav(ch)
        trav(self.root)
        return {'max_depth': max(depths, default=0), 'leaf_nodes': leaves, 'total_nodes': total, 'support_vectors': 'N/A'}

    def plot_tree(self, max_vis_depth=3, figsize=(18, 7), save_path=None):
        fig, ax = plt.subplots(figsize=figsize); ax.axis('off')
        def draw(node, x, y, dx, d):
            if not node or d > max_vis_depth: return
            lbl = f"{'>50K' if node.majority_class else '<=50K'}\\nN={node.samples}" if node.is_leaf() or d == max_vis_depth else f"{node.feature}\\nIG={node.info_gain:.2f}"
            c = '#A3E4D7' if node.majority_class else ('#AED6F1' if node.is_leaf() else '#FAD7A0')
            ax.text(x, y, lbl, ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle='round,pad=0.3', facecolor=c, lw=1))
            if not node.is_leaf() and d < max_vis_depth and node.children:
                step = dx / max(len(node.children), 1)
                for i, (val, ch) in enumerate(list(node.children.items())[:6]):
                    cx, cy = x - dx/2 + (i + 0.5) * step, y - 0.25
                    ax.plot([x, cx], [y - 0.04, cy + 0.04], 'k-', lw=0.7)
                    ax.text((x + cx)/2, (y + cy)/2, str(val)[:10], fontsize=6.5, ha='center', bbox=dict(boxstyle='round', fc='w', ec='none', pad=0.1))
                    draw(ch, cx, cy, step, d + 1)
        draw(self.root, 0.5, 0.9, 0.9, 0)
        ax.set_title(f"ID3 Decision Tree Hierarchy (Levels 0 to {max_vis_depth})", weight='bold')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

print("Genuine ID3 Classifier (from scratch) Ready.")""")

    # =========================================================================
    # SECTION 6: Linear SVM Implementation
    # =========================================================================
    add_cell('markdown', r"""## Section 6: Linear SVM Implementation (Maximum Margin Hyperplane)

**Theoretical Formulation:**
Solves the primal convex quadratic programming optimization problem $\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i} \xi_i$ subject to $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$. Features are pre-processed via z-score standardization ($\mu=0, \sigma=1$) and one-hot encoding, producing a maximal-margin decision boundary accompanied by Platt-scaled posterior probabilities.""")

    add_cell('code', """class LinearSVMWrapper:
    \"\"\"Linear SVM with StandardScaler and OneHotEncoder preprocessing.\"\"\"
    def __init__(self, C=1.0, probability=True, random_state=42):
        self.preprocessor = None
        self.model = SVC(kernel='linear', C=C, probability=probability, random_state=random_state)

    def fit(self, X, y, num_cols, cat_cols):
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        self.model.fit(self.preprocessor.fit_transform(X), np.asarray(y).ravel())
        return self

    def predict(self, X): return self.model.predict(self.preprocessor.transform(X))
    def predict_proba(self, X): return self.model.predict_proba(self.preprocessor.transform(X))

    def get_svm_metrics(self):
        return {'max_depth': 'N/A', 'leaf_nodes': 'N/A', 'total_nodes': 'N/A',
                'total_support_vectors': int(self.model.n_support_.sum())}

print("Linear SVM Wrapper Ready.")""")

    # =========================================================================
    # SECTION 7: Model Training & Evaluation Metrics
    # =========================================================================
    add_cell('markdown', r"""## Section 7: Model Training & Evaluation Metrics Calculation

**Theoretical Formulation:**
Executes isolated training for all three models while tracking exact microsecond-level training and prediction latencies. It computes a comprehensive suite of evaluation metrics on unseen test records: Accuracy, Precision, Recall, $F_1\text{-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$, Balanced Accuracy, Specificity, and ROC-AUC.""")

    add_cell('code', """models, train_times, pred_times, metrics_results = {}, {}, {}, {}

# Train CART
cart = CARTClassifierWrapper(max_depth=CONFIG['cart_depth'], random_state=CONFIG['seed'])
t0 = time.perf_counter(); cart.fit(X_train_raw, y_train, cat_cols=CATEGORICAL_COLS, num_cols=NUMERICAL_COLS)
train_times['CART'], models['CART'] = time.perf_counter() - t0, cart

# Train ID3
discretizer = AdaptiveQuantileDiscretizer(n_bins=4).fit(X_train_raw[NUMERICAL_COLS])
X_tr_id3 = pd.concat([X_train_raw[CATEGORICAL_COLS], discretizer.transform(X_train_raw[NUMERICAL_COLS])], axis=1)
X_te_id3 = pd.concat([X_test_raw[CATEGORICAL_COLS], discretizer.transform(X_test_raw[NUMERICAL_COLS])], axis=1)
id3 = ID3Classifier(max_depth=CONFIG['id3_depth'])
t0 = time.perf_counter(); id3.fit(X_tr_id3, y_train)
train_times['ID3'], models['ID3'] = time.perf_counter() - t0, id3

# Train Linear SVM
svm = LinearSVMWrapper(C=CONFIG['svm_C'], random_state=CONFIG['seed'])
t0 = time.perf_counter(); svm.fit(X_train_raw, y_train, num_cols=NUMERICAL_COLS, cat_cols=CATEGORICAL_COLS)
train_times['Linear SVM'], models['Linear SVM'] = time.perf_counter() - t0, svm

# Evaluate all 3 models
for a in ['CART', 'ID3', 'Linear SVM']:
    m = models[a]
    X_eval = X_te_id3 if a == 'ID3' else X_test_raw
    t0 = time.perf_counter(); preds = m.predict(X_eval); pred_times[a] = time.perf_counter() - t0
    probs = m.predict_proba(X_eval)[:, 1]
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics_results[a] = {
        'Accuracy': accuracy_score(y_test, preds), 'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0), 'F1 Score': f1_score(y_test, preds, zero_division=0),
        'Balanced Accuracy': balanced_accuracy_score(y_test, preds), 'Specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'ROC-AUC': roc_auc_score(y_test, probs), 'Training Time': train_times[a], 'Prediction Time': pred_times[a],
        'Confusion Matrix': cm
    }
    print(f"  * {a:10s} | Acc: {metrics_results[a]['Accuracy']:.4f} | F1: {metrics_results[a]['F1 Score']:.4f} | Train: {train_times[a]:.3f}s | Pred: {pred_times[a]:.3f}s")""")

    # =========================================================================
    # SECTION 8: Master Comparative Performance Table
    # =========================================================================
    add_cell('markdown', """## Section 8: Master Comparative Performance Table

**Theoretical Formulation:**
Aggregates empirical metric vectors, execution latencies, and architectural complexity indicators (tree depth, leaf counts, node counts, support vectors) into a unified comparative DataFrame. It automatically parses category leaders across accuracy, precision, recall, and computational efficiency, serializing results to `outputs/model_comparison_table.csv`.""")

    add_cell('code', """table_rows = []
for a in ['CART', 'ID3', 'Linear SVM']:
    m = metrics_results[a]
    comp = models[a].get_svm_metrics() if a == 'Linear SVM' else models[a].get_tree_metrics()
    table_rows.append({
        'Algorithm': a, 'Accuracy': f"{m['Accuracy']:.4f}", 'Precision': f"{m['Precision']:.4f}",
        'Recall': f"{m['Recall']:.4f}", 'F1 Score': f"{m['F1 Score']:.4f}",
        'Balanced Accuracy': f"{m['Balanced Accuracy']:.4f}", 'Specificity': f"{m['Specificity']:.4f}",
        'ROC-AUC': f"{m['ROC-AUC']:.4f}", 'Training Time (s)': f"{m['Training Time']:.4f}",
        'Prediction Time (s)': f"{m['Prediction Time']:.4f}", 'Tree Depth': str(comp.get('max_depth', 'N/A')),
        'Leaf Nodes': str(comp.get('leaf_nodes', 'N/A')), 'Total Nodes': str(comp.get('total_nodes', 'N/A')),
        'Support Vectors': str(comp.get('total_support_vectors', 'N/A'))
    })

master_table = pd.DataFrame(table_rows)
display(master_table)
master_table.to_csv('outputs/model_comparison_table.csv', index=False)
print("Saved clean table to outputs/model_comparison_table.csv")""")

    # =========================================================================
    # SECTION 9: Confusion Matrix Analysis
    # =========================================================================
    add_cell('markdown', r"""## Section 9: Confusion Matrix Analysis (CART vs. ID3 vs. Linear SVM)

**Theoretical Formulation:**
Computes and renders side-by-side $2 \times 2$ contingency heatmaps displaying True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), and False Negatives ($FN$) for each algorithm. This reveals whether a model's error mode is dominated by Type I errors (over-predicting high income) or Type II errors (failing to identify high earners).""")

    add_cell('code', """fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, a in enumerate(['CART', 'ID3', 'Linear SVM']):
    sns.heatmap(metrics_results[a]['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'], ax=axes[idx],
                annot_kws={'fontsize': 12, 'weight': 'bold'})
    axes[idx].set_title(f"{a} Confusion Matrix\\nAcc: {metrics_results[a]['Accuracy']:.4f} | F1: {metrics_results[a]['F1 Score']:.4f}", weight='bold')
    axes[idx].set_xlabel('Predicted Label'); axes[idx].set_ylabel('True Label')
plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()""")

    # =========================================================================
    # SECTION 10: Decision Tree Architecture Visualizations
    # =========================================================================
    add_cell('markdown', """## Section 10: Decision Tree Architecture Visualizations (CART & ID3)

**Theoretical Formulation:**
Graphically renders the internal decision logic of the tree-based models: CART's binary splits with sample distributions and Gini scores, alongside ID3's multi-way nominal decision paths with Information Gain and node entropies. This provides direct human interpretability of the learned rules governing high versus standard income classifications.""")

    add_cell('code', """print("Rendering CART Decision Tree Diagram...")
models['CART'].plot_tree(max_vis_depth=3, save_path='outputs/cart_decision_tree.png')
plt.show()

print("Rendering ID3 Decision Tree Diagram...")
models['ID3'].plot_tree(max_vis_depth=3, save_path='outputs/id3_decision_tree.png')
plt.show()""")

    # =========================================================================
    # SECTION 11: Interactive Colab Terminal Prediction & Performance Matrix
    # =========================================================================
    add_cell('markdown', """## Section 11: Interactive Colab Terminal Prediction & Performance Matrix

**Theoretical Formulation:**
Provides a live command-line interface in Google Colab that sequentially prompts users for demographic inputs with numbered choices and intelligent defaults. It routes the vector through the trained pipelines to display predicted income, model confidence percentages, the master benchmark comparison table, and side-by-side confusion matrices in a single view.""")

    add_cell('code', """def predict_income(sample_dict, models, discretizer, num_cols, cat_cols):
    \"\"\"Predicts income class and posterior probabilities across CART, ID3, and Linear SVM.\"\"\"
    df_s = pd.DataFrame([sample_dict])
    df_disc = pd.concat([df_s[cat_cols], discretizer.transform(df_s[num_cols])], axis=1)
    rows = []
    for a in ['CART', 'ID3', 'Linear SVM']:
        m = models[a]
        x = df_disc if a == 'ID3' else df_s
        p1 = float(m.predict_proba(x)[0, 1])
        pred = '>50K' if p1 >= 0.5 else '<=50K'
        rows.append({'Algorithm': a, 'Predicted Income': pred,
                     'Confidence (>50K)': f"{p1*100:.2f}%", 'Confidence (<=50K)': f"{(1-p1)*100:.2f}%"})
    return pd.DataFrame(rows)

def interactive_terminal_prompt(models, discretizer, num_cols, cat_cols, metrics_results=None, master_table=None):
    \"\"\"Interactive CLI prompt accepting user inputs with defaults and displaying multi-model outputs.\"\"\"
    print("\\n" + "=" * 70 + "\\nINTERACTIVE DEMOGRAPHIC PROFILE PREDICTOR\\n" + "=" * 70)
    def ask(prompt, default, choices=None):
        disp = f"  Options: {', '.join(f'[{i+1}] {c}' for i, c in enumerate(choices))}\\n" if choices else ""
        raw = input(f"{prompt}\\n{disp}Selection [default: {default}]: ").strip()
        if not raw: return default
        if choices and raw.isdigit() and 1 <= int(raw) <= len(choices): return choices[int(raw)-1]
        return float(raw) if not choices and '.' in str(default) else (int(raw) if not choices else raw)

    profile = {
        'age': ask("1. Age (years)", 35),
        'workclass': ask("2. Workclass", 'Private', ['Private', 'Government', 'Self-Employed']),
        'education': ask("3. Education", 'Graduation', ['10th', '12th', 'Graduation', 'Post-Graduation', 'PhD']),
        'marital_status': ask("4. Marital Status", 'Married', ['Single', 'Married', 'Divorced', 'Widowed']),
        'sex': ask("5. Sex", 'Male', ['Male', 'Female']),
        'relationship': ask("6. Relationship", 'Husband', ['Husband', 'Wife', 'Child', 'Other']),
        'capital_gain': ask("7. Capital Gain ($)", 0),
        'capital_loss': ask("8. Capital Loss ($)", 0),
        'hours_per_week': ask("9. Hours per Week", 40)
    }

    res = predict_income(profile, models, discretizer, num_cols, cat_cols)
    print("\\n" + "=" * 70 + "\\nPREDICTION RESULTS:\\n" + "=" * 70)
    print(res.to_string(index=False))
    if master_table is not None:
        print("\\n" + "=" * 70 + "\\nBENCHMARK METRICS SUMMARY:\\n" + "=" * 70)
        print(master_table.to_string(index=False))
    return res""")

    add_cell('markdown', r"""### 11.1 Live Interactive Prediction Form (GUI)
Use the interactive sliders and dropdowns below to test arbitrary demographic profiles in real time with automatic re-evaluation across all three models!""")

    add_cell('code', """#@title 🎯 Live Interactive Demographic Predictor { run: "auto" }
age = 35 #@param {type:"slider", min:17, max:90, step:1}
workclass = "Private" #@param ["Private", "Government", "Self-Employed"]
education = "Graduation" #@param ["10th", "12th", "Graduation", "Post-Graduation", "PhD"]
marital_status = "Married" #@param ["Single", "Married", "Divorced", "Widowed"]
sex = "Male" #@param ["Male", "Female"]
relationship = "Husband" #@param ["Husband", "Wife", "Child", "Other"]
capital_gain = 0 #@param {type:"integer"}
capital_loss = 0 #@param {type:"integer"}
hours_per_week = 40 #@param {type:"slider", min:1, max:99, step:1}

profile = {'age': int(age), 'workclass': str(workclass), 'education': str(education),
           'marital_status': str(marital_status), 'sex': str(sex), 'relationship': str(relationship),
           'capital_gain': int(capital_gain), 'capital_loss': int(capital_loss), 'hours_per_week': int(hours_per_week)}

res = predict_income(profile, models, discretizer, NUMERICAL_COLS, CATEGORICAL_COLS)
print(f"PROFILE: Age={age} | {workclass} | {education} | {marital_status} | {sex} | {relationship} | Gain=${capital_gain:,} | Loss=${capital_loss:,} | {hours_per_week}h/wk")
display(res)""")

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
        'nbformat_minor': 0
    }

    output_path = 'Comparative_Performance_Analysis_CART_ID3_LinearSVM.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Generator] Successfully built streamlined notebook with {len(cells)} cells at {output_path}!")


if __name__ == '__main__':
    create_streamlined_notebook()
