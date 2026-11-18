"""
Genuine ID3 Decision Tree Algorithm Implementation from Scratch.

Implements Ross Quinlan's classic ID3 algorithm using Shannon Entropy
and Information Gain for multi-way categorical branching. Includes
adaptive quantile-based numerical discretization, unseen category handling,
rule extraction, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AdaptiveQuantileDiscretizer:
    """
    Documented Quantile-Based Discretization for Numerical Attributes.

    Prevents data leakage by computing bin boundaries strictly on training data
    and applying them consistently to testing data.
    - Zero-inflated attributes (capital_gain, capital_loss) are split into:
      'Zero', 'Low_Pos', 'High_Pos' based on the median of non-zero values.
    - Continuous attributes (age, education_num, hours_per_week) are partitioned
      into 4 quartiles ('Q1', 'Q2', 'Q3', 'Q4').
    """
    def __init__(self, n_bins=4):
        self.n_bins = n_bins
        self.rules_ = {}

    def fit(self, X):
        X_df = pd.DataFrame(X)
        self.rules_ = {}
        for col in X_df.columns:
            vals = X_df[col].dropna()
            zero_ratio = (vals == 0).mean()
            if zero_ratio > 0.5:
                pos_vals = vals[vals > 0]
                if len(pos_vals) > 0:
                    med_pos = np.median(pos_vals)
                    self.rules_[col] = ('zero_inflated', float(med_pos))
                else:
                    self.rules_[col] = ('all_zero', 0.0)
            else:
                qs = np.linspace(0, 1, self.n_bins + 1)
                edges = np.unique(np.quantile(vals, qs))
                edges[0] = -np.inf
                edges[-1] = np.inf
                self.rules_[col] = ('quantiles', edges)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_disc = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            rule_type, rule_val = self.rules_[col]
            if rule_type == 'zero_inflated':
                med_pos = rule_val
                conds = [
                    X_df[col] <= 0,
                    (X_df[col] > 0) & (X_df[col] <= med_pos),
                    X_df[col] > med_pos
                ]
                choices = ['Zero', 'Low_Pos', 'High_Pos']
                X_disc[col] = np.select(conds, choices, default='Zero')
            elif rule_type == 'all_zero':
                X_disc[col] = 'Zero'
            else:
                edges = rule_val
                binned = np.digitize(X_df[col], edges[1:-1])
                labels = [f'Q{i+1}' for i in range(len(edges) - 1)]
                binned = np.clip(binned, 0, len(labels) - 1)
                X_disc[col] = [labels[b] for b in binned]
        return X_disc


class ID3Node:
    """
    Represents an individual node in the ID3 decision tree.
    """
    def __init__(self, feature=None, children=None, value=None, info_gain=0.0,
                 samples=0, entropy=0.0, depth=0, class_counts=None, majority_class=None, prob=0.0):
        self.feature = feature                  # Splitting attribute name (None if leaf)
        self.children = children or {}          # Dict mapping attribute branch value -> child ID3Node
        self.value = value                      # Leaf predicted class (0 or 1, None if internal node)
        self.info_gain = info_gain              # Information Gain obtained at this split
        self.samples = samples                  # Count of training samples reaching this node
        self.entropy = entropy                  # Shannon entropy of this node's sample subset
        self.depth = depth                      # Depth of node in tree (root = 0)
        self.class_counts = class_counts or {}  # Distribution {0: count0, 1: count1}
        self.majority_class = majority_class    # Fallback majority class
        self.prob = prob                        # Estimated probability of class 1

    def is_leaf(self):
        return self.value is not None


class ID3Classifier:
    """
    Ross Quinlan's ID3 (Iterative Dichotomiser 3) Classifier.

    Constructs a decision tree by greedily selecting the attribute that
    maximizes Information Gain (minimizes entropy) at each recursive step.
    Supports multi-way categorical branching, configurable depth,
    unseen category fallback, and comprehensive complexity auditing.
    """
    def __init__(self, max_depth=6, min_samples_split=20, min_info_gain=1e-4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.root = None
        self.features_ = []
        self.classes_ = np.array([0, 1])
        self.split_history_ = []  # Detailed log of every split for audit

    def _entropy(self, y):
        """Calculates Shannon Entropy: H(S) = -sum(p_i * log2(p_i))"""
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def _information_gain(self, col_values, y, base_entropy):
        """
        Calculates Information Gain:
        IG(S, A) = H(S) - sum( (|S_v| / |S|) * H(S_v) )
        """
        n = len(y)
        if n == 0:
            return 0.0
        uniques, inverse = np.unique(col_values, return_inverse=True)
        if len(uniques) <= 1:
            return 0.0

        child_entropy_sum = 0.0
        for u_idx in range(len(uniques)):
            sub_y = y[inverse == u_idx]
            child_entropy_sum += (len(sub_y) / n) * self._entropy(sub_y)

        return float(base_entropy - child_entropy_sum)

    def fit(self, X, y):
        """
        Fits ID3 Decision Tree on categorical/discretized training data.
        """
        if isinstance(X, pd.DataFrame):
            self.features_ = list(X.columns)
            X_mat = X.to_numpy()
        else:
            self.features_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_mat = np.asarray(X)
        y_vec = np.asarray(y)

        self.split_history_ = []
        self.root = self._build_tree(
            X_mat, y_vec, depth=0, available_features=list(range(X_mat.shape[1]))
        )
        return self

    def _build_tree(self, X, y, depth, available_features):
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
        majority_class = int(classes[np.argmax(counts)]) if len(classes) > 0 else 0
        prob_pos = float(np.mean(y == 1)) if n_samples > 0 else 0.0
        curr_entropy = self._entropy(y)

        # Stopping Criteria:
        # 1. Pure node (entropy == 0)
        # 2. Maximum tree depth reached
        # 3. Sample count below min_samples_split
        # 4. No candidate features remaining
        if (curr_entropy == 0.0 or
            depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(available_features) == 0):
            return ID3Node(
                value=majority_class, samples=n_samples, entropy=curr_entropy,
                depth=depth, class_counts=class_counts, majority_class=majority_class, prob=prob_pos
            )

        # Select candidate feature maximizing Information Gain
        best_ig = -1.0
        best_feat_idx = None
        for f_idx in available_features:
            ig = self._information_gain(X[:, f_idx], y, curr_entropy)
            if ig > best_ig:
                best_ig = ig
                best_feat_idx = f_idx

        # If Information Gain is non-positive or negligible, stop and create leaf
        if best_ig < self.min_info_gain or best_feat_idx is None:
            return ID3Node(
                value=majority_class, samples=n_samples, entropy=curr_entropy,
                depth=depth, class_counts=class_counts, majority_class=majority_class, prob=prob_pos
            )

        feat_name = self.features_[best_feat_idx]
        self.split_history_.append({
            'depth': depth,
            'feature': feat_name,
            'information_gain': float(best_ig),
            'samples': int(n_samples),
            'entropy': float(curr_entropy)
        })

        node = ID3Node(
            feature=feat_name, info_gain=best_ig, samples=n_samples,
            entropy=curr_entropy, depth=depth, class_counts=class_counts,
            majority_class=majority_class, prob=prob_pos
        )

        uniques, inverse = np.unique(X[:, best_feat_idx], return_inverse=True)
        rem_features = [f for f in available_features if f != best_feat_idx]

        for u_idx, u_val in enumerate(uniques):
            mask = inverse == u_idx
            sub_X = X[mask]
            sub_y = y[mask]
            if len(sub_y) == 0:
                child = ID3Node(
                    value=majority_class, samples=0, entropy=0.0,
                    depth=depth + 1, class_counts={}, majority_class=majority_class, prob=prob_pos
                )
            else:
                child = self._build_tree(sub_X, sub_y, depth + 1, rem_features)
            node.children[u_val] = child

        return node

    def _predict_sample(self, row, node):
        """Recursively traverses the tree for a single test sample with unseen value fallback."""
        if node.is_leaf():
            return node.value, node.prob

        val = row[self.features_.index(node.feature)]
        if val in node.children:
            return self._predict_sample(row, node.children[val])
        else:
            # Unseen categorical branch encountered: gracefully fallback to node's majority class
            return node.majority_class, node.prob

    def predict(self, X):
        """Generates class predictions (0 or 1)."""
        if isinstance(X, pd.DataFrame) and all(f in X.columns for f in self.features_):
            X_mat = X[self.features_].to_numpy()
        else:
            X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        preds = [self._predict_sample(row, self.root)[0] for row in X_mat]
        return np.array(preds, dtype=int)

    def predict_proba(self, X):
        """Generates class posterior probabilities [[P(0), P(1)]]."""
        if isinstance(X, pd.DataFrame) and all(f in X.columns for f in self.features_):
            X_mat = X[self.features_].to_numpy()
        else:
            X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        probs_pos = np.array([self._predict_sample(row, self.root)[1] for row in X_mat])
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def get_tree_metrics(self):
        """Extracts structural complexity metrics from the constructed tree."""
        depths = []
        leaves = 0
        total_nodes = 0
        features_used = set()
        rules = []

        def traverse(node, current_rule):
            nonlocal leaves, total_nodes
            total_nodes += 1
            if node.is_leaf():
                leaves += 1
                depths.append(node.depth)
                rule_str = ' AND '.join(current_rule) if current_rule else 'True'
                class_lbl = '>50K' if node.value == 1 else '<=50K'
                rules.append(f"IF {rule_str} THEN Income = {class_lbl} (prob={node.prob:.2f}, n={node.samples})")
                return

            features_used.add(node.feature)
            for val, child in node.children.items():
                traverse(child, current_rule + [f"({node.feature} == '{val}')"])

        traverse(self.root, [])
        return {
            'max_depth': max(depths) if depths else 0,
            'leaf_nodes': leaves,
            'total_nodes': total_nodes,
            'decision_rules': len(rules),
            'features_used': sorted(list(features_used)),
            'rules': rules
        }

    def plot_tree(self, max_vis_depth=3, figsize=(20, 10), save_path=None):
        """
        Visualizes the top levels of the ID3 decision tree using Matplotlib.
        """
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
                txt = f"{'>50K' if is_pos else '<=50K'}\nProb: {node.prob:.2f}\nN={node.samples}"
                ax.text(x, y, txt, ha='center', va='center', fontsize=7.5,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='#2C3E50', lw=1.2))
                return x, y

            child_positions = []
            for val, child in list(node.children.items())[:6]:  # Limit max branches for clean visual
                cx, cy = render_node(child, d + 1)
                child_positions.append((cx, cy, str(val)[:12]))

            x = sum(cp[0] for cp in child_positions) / len(child_positions)
            y = 1.0 - (d / (max_vis_depth + 1.2))

            txt = f"Split: {node.feature}\nIG: {node.info_gain:.3f}\nH: {node.entropy:.2f}\nN={node.samples}"
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
            print(f"[ID3] Tree visualization saved to {save_path}")
        return fig
