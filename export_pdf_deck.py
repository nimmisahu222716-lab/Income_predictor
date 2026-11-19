import os
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    pdf_path = "Comparative_Performance_Analysis_Presentation.pdf"
    # 16:9 widescreen in points (13.333 inches x 7.5 inches = 960 x 540 pt)
    PAGE_WIDTH = 960
    PAGE_HEIGHT = 540

    c = canvas.Canvas(pdf_path, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    # Color palette
    NAVY = colors.HexColor("#1E3A52")
    SLATE = colors.HexColor("#4A6572")
    MUTED_BLUE = colors.HexColor("#8C9EA8")
    CARD_BG = colors.HexColor("#F5F7FA")
    CARD_BORDER = colors.HexColor("#DAE1E7")
    TEXT_DARK = colors.HexColor("#212B36")
    TEXT_MUTED = colors.HexColor("#64748B")
    WHITE = colors.HexColor("#FFFFFF")
    GREEN = colors.HexColor("#16A085")
    RED = colors.HexColor("#C0392B")

    def draw_decorations(c, title, category=""):
        # Top-left triangles
        c.setFillColor(SLATE)
        p = c.beginPath()
        p.moveTo(0, PAGE_HEIGHT)
        p.lineTo(80, PAGE_HEIGHT)
        p.lineTo(0, PAGE_HEIGHT - 80)
        p.close()
        c.drawPath(p, fill=1, stroke=0)

        c.setFillColor(MUTED_BLUE)
        p2 = c.beginPath()
        p2.moveTo(25, PAGE_HEIGHT)
        p2.lineTo(75, PAGE_HEIGHT)
        p2.lineTo(25, PAGE_HEIGHT - 50)
        p2.close()
        c.drawPath(p2, fill=1, stroke=0)

        # Bottom-right triangles
        c.setFillColor(SLATE)
        p3 = c.beginPath()
        p3.moveTo(PAGE_WIDTH, 0)
        p3.lineTo(PAGE_WIDTH - 80, 0)
        p3.lineTo(PAGE_WIDTH, 80)
        p3.close()
        c.drawPath(p3, fill=1, stroke=0)

        c.setFillColor(MUTED_BLUE)
        p4 = c.beginPath()
        p4.moveTo(PAGE_WIDTH - 25, 0)
        p4.lineTo(PAGE_WIDTH - 75, 0)
        p4.lineTo(PAGE_WIDTH - 25, 50)
        p4.close()
        c.drawPath(p4, fill=1, stroke=0)

        # Title & Category
        if category:
            c.setFont("Helvetica-Bold", 8.5)
            c.setFillColor(SLATE)
            c.drawString(100, PAGE_HEIGHT - 38, category.upper())

        c.setFont("Helvetica-Bold", 17)
        c.setFillColor(NAVY)
        c.drawString(100, PAGE_HEIGHT - (58 if category else 45), title)

        # Footer
        c.setFont("Helvetica", 7.5)
        c.setFillColor(TEXT_MUTED)
        c.drawCentredString(PAGE_WIDTH / 2, 18, "Comparative Benchmark: CART vs ID3 vs Linear SVM | Adult Income Census")

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    draw_decorations(c, "", "")
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(SLATE)
    c.drawCentredString(PAGE_WIDTH / 2, 390, "COMPARATIVE ANALYSIS OF")

    c.setFont("Helvetica-Bold", 34)
    c.setFillColor(NAVY)
    c.drawCentredString(PAGE_WIDTH / 2, 340, "CART, ID3 & LINEAR SVM")

    c.setFont("Helvetica", 13.5)
    c.setFillColor(TEXT_DARK)
    c.drawCentredString(PAGE_WIDTH / 2, 302, "Benchmarking Algorithms for Demographic Income Classification")

    c.setFont("Helvetica", 10.5)
    c.setFillColor(SLATE)
    c.drawCentredString(PAGE_WIDTH / 2, 276, "Gini Impurity vs. Shannon Entropy vs. Maximum Margin Hyperplane")

    # Metadata Card (faint transparent grey, NO outline, left-aligned bulleted list)
    card_w = 480
    card_h = 120
    card_x = PAGE_WIDTH/2 - card_w/2
    card_y = 115

    c.setFillColor(colors.HexColor("#F1F5F9"))
    c.roundRect(card_x, card_y, card_w, card_h, 6, fill=1, stroke=0) # NO outline!

    bottom_bullets = [
        ("Dataset:", " UCI Adult Census (30,162 clean records)"),
        ("Split:", " 80:20 Stratified"),
        ("Target:", " Annual Income (> 50K vs <= 50K)"),
        ("Metrics:", " Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, Latency")
    ]
    by = card_y + card_h - 26
    for lbl, val in bottom_bullets:
        c.setFont("Helvetica-Bold", 9.5)
        c.setFillColor(SLATE)
        c.drawString(card_x + 25, by, "•")
        c.setFillColor(NAVY)
        c.drawString(card_x + 40, by, lbl)
        c.setFont("Helvetica", 9.5)
        c.setFillColor(TEXT_DARK)
        c.drawString(card_x + 40 + c.stringWidth(lbl, "Helvetica-Bold", 9.5), by, val)
        by -= 24
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 2: Objectives
    # -------------------------------------------------------------
    draw_decorations(c, "Project Overview & Research Objectives", "FOUNDATIONAL BENCHMARK")
    objs = [
        ("1. Algorithmic Isolation", [
            "Evaluated three distinct supervised classification paradigms under identical experimental constraints.",
            "Applied an 80:20 stratified partition across 30,162 records to eliminate data leakage.",
            "Analyzed empirical metrics on standardized 9-feature demographic input vectors."
        ]),
        ("2. Scratch vs Library Analysis", [
            "Implemented a native ID3 decision tree from scratch using recursive Shannon entropy minimization.",
            "Developed an adaptive quantile discretization module for continuous demographic attributes.",
            "Compared scratch algorithmic fidelity against production-optimized scikit-learn classifiers."
        ]),
        ("3. Boundary vs Interpretability", [
            "Analyzed structural trade-offs between hierarchical rule paths and geometric hyperplanes.",
            "Evaluated decision transparency across axis-aligned and multi-way branching topologies.",
            "Compared Platt-scaled margin confidence scores with terminal leaf posterior probabilities."
        ]),
        ("4. Sub-Second Interactive Inference", [
            "Developed a dual-mode interactive interface supporting live Colab GUI forms and CLI prompts.",
            "Applied automated preprocessing pipelines for instantaneous demographic profile inference.",
            "Evaluated microsecond-level execution latencies alongside multi-model confidence outputs."
        ])
    ]
    card_positions = [(75, 275), (490, 275), (75, 95), (490, 95)]
    card_w2 = 395
    card_h2 = 165
    for (otitle, obullets), (cx, cy) in zip(objs, card_positions):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(cx, cy, card_w2, card_h2, 6, fill=1, stroke=1)
        # Header banner
        c.setFillColor(NAVY)
        c.roundRect(cx, cy + card_h2 - 30, card_w2, 30, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 10.5)
        c.setFillColor(WHITE)
        c.drawString(cx + 15, cy + card_h2 - 20, otitle)

        # Bullets
        ty = cy + card_h2 - 45
        for bullet in obullets:
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(SLATE)
            c.drawString(cx + 15, ty, "•")

            c.setFont("Helvetica", 8.5)
            c.setFillColor(TEXT_DARK)
            
            # Wrap bullet text
            words = bullet.split(' ')
            cur = ""
            for w in words:
                if len(cur + " " + w) > 65:
                    c.drawString(cx + 28, ty, cur)
                    ty -= 12
                    cur = w
                else:
                    cur = cur + " " + w if cur else w
            if cur:
                c.drawString(cx + 28, ty, cur)
                ty -= 15
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 3: Model Introduction
    # -------------------------------------------------------------
    draw_decorations(c, "Model Introduction & Algorithmic Foundations", "ALGORITHM TAXONOMY")
    models = [
        ("CART (Binary Tree)", NAVY, [
            "Paradigm: Non-parametric binary recursive tree.",
            "Splitting Metric: Gini Impurity reduction:",
            "  I_G(t) = 1 - sum(p_i^2)",
            "Boundary: Axis-aligned orthogonal step partitions.",
            "Implementation: Scikit-Learn DecisionTreeClassifier",
            "  (max_depth=8, criterion='gini').",
            "Output: Leaf probability distribution & binary class."
        ]),
        ("ID3 (Scratch Shannon Tree)", SLATE, [
            "Paradigm: Multi-way nominal categorical tree.",
            "Splitting Metric: Information Gain maximization:",
            "  IG(D, A) = H(D) - sum(|D_v|/|D| * H(D_v))",
            "Entropy: H(D) = - sum(p_i * log2(p_i))",
            "Continuous Handling: 4-bin Quantile Discretizer.",
            "Implementation: 100% scratch Python engine",
            "  (max_depth=6, min_samples_split=10)."
        ]),
        ("Linear SVM (Max Margin)", colors.HexColor("#28465F"), [
            "Paradigm: Maximum-margin linear hyperplane.",
            "Optimization Objective:",
            "  min 0.5*||w||^2 + C * sum(xi_i)",
            "Decision Rule: y_pred = sign(w^T * x + b)",
            "Preprocessing: StandardScaler + OneHotEncoder.",
            "Implementation: SVC(kernel='linear', C=1.0,",
            "  probability=True, random_state=42)."
        ])
    ]
    card_w3 = 265
    lefts3 = [75, 350, 625]
    for (mtitle, mcol, bullets), mx in zip(models, lefts3):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(mx, 90, card_w3, 370, 6, fill=1, stroke=1)
        c.setFillColor(mcol)
        c.roundRect(mx, 425, card_w3, 35, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(WHITE)
        c.drawCentredString(mx + card_w3/2, 437, mtitle)

        by = 395
        for b in bullets:
            c.setFont("Helvetica", 9)
            c.setFillColor(TEXT_DARK)
            c.drawString(mx + 15, by, b)
            by -= 20
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 4: Why This Model Is Used
    # -------------------------------------------------------------
    draw_decorations(c, "Model Selection Rationale: Why These Models?", "COMPARATIVE DESIGN")
    whys = [
        ("Why CART?", NAVY, [
            "• Universal Robustness: Handles mixed continuous & categorical attributes with zero scaling.",
            "• Cognitive Interpretability: Transparent, binary if-then rules easily audited by policy makers.",
            "• Monotonic Shift Invariance: Unaffected by outlier spikes in capital gains.",
            "• Computational Speed: Ultra-fast O(N log N) tree building with sub-millisecond evaluation."
        ]),
        ("Why ID3 from Scratch?", SLATE, [
            "• Information Theory: Explores pure Shannon entropy reduction and Information Gain dynamics.",
            "• Multi-Way Branching: Captures full nominal categorical interactions without binary fragmentation.",
            "• Full Transparency: Scratch engine exposes all internal node creation and impurity metrics.",
            "• Quantile Discretization: Reconciles continuous numerical inputs with nominal tree logic."
        ]),
        ("Why Linear SVM?", colors.HexColor("#28465F"), [
            "• Geometric Contrast: Serves as continuous mathematical counterpoint to piecewise step functions.",
            "• Maximum-Margin Principle: Guarantees maximum boundary separation, lowering generalization error.",
            "• Convex Global Optimality: Quadratic programming guarantees a single global minimum.",
            "• High-Dimensional Resilience: Highly effective in sparse spaces created by One-Hot Encoding."
        ])
    ]
    for (wtitle, wcol, bullets), wx in zip(whys, lefts3):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(wx, 90, card_w3, 370, 6, fill=1, stroke=1)
        c.setFillColor(wcol)
        c.roundRect(wx, 425, card_w3, 35, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(WHITE)
        c.drawCentredString(wx + card_w3/2, 437, wtitle)

        wy = 390
        for b in bullets:
            words = b.split(' ')
            cur = ""
            for w in words:
                if len(cur + " " + w) > 42:
                    c.setFont("Helvetica", 8.5)
                    c.setFillColor(TEXT_DARK)
                    c.drawString(wx + 15, wy, cur)
                    wy -= 13
                    cur = "   " + w
                else:
                    cur = cur + " " + w if cur else w
            if cur:
                c.setFont("Helvetica", 8.5)
                c.setFillColor(TEXT_DARK)
                c.drawString(wx + 15, wy, cur)
                wy -= 22
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 5: Process Flow Sequence
    # -------------------------------------------------------------
    draw_decorations(c, "End-to-End Workflow & Pipeline Architecture", "EXECUTION SEQUENCE")
    steps = [
        ("Stage 1: Ingestion & Clean", "Acquires 30,162 records from UCI Adult Census; removes '?' missing tokens; cleans headers."),
        ("Stage 2: 9-Feature Schema", "Isolates 4 numerical (age, hours, capital gain/loss) and 5 categorical (workclass, education, etc.)."),
        ("Stage 3: Stratified 80:20 Split", "Partitions dataset into 8,000 Train & 2,000 Test strictly preserving 75.1% : 24.9% income ratio."),
        ("Stage 4: Transformations", "CART: Direct label/ordinal encoding.\nID3: 4-interval quantile discretization.\nSVM: StandardScaler + OneHotEncoder."),
        ("Stage 5: Parallel Training", "Trains CART (depth=8), ID3 (depth=6 scratch), and Linear SVM (C=1.0) with latency profiling."),
        ("Stage 6: Master Evaluation", "Generates performance matrices, confusion heatmaps, ROC curves, decision tree graphs, and UI.")
    ]
    coords = [(75, 275), (350, 275), (625, 275), (75, 95), (350, 95), (625, 95)]
    for (stitle, sdesc), (sx, sy) in zip(steps, coords):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(sx, sy, 265, 160, 6, fill=1, stroke=1)
        c.setFillColor(NAVY if sy > 200 else SLATE)
        c.roundRect(sx, sy + 125, 265, 35, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 10.5)
        c.setFillColor(WHITE)
        c.drawString(sx + 15, sy + 137, stitle)

        sy_text = sy + 105
        for l in sdesc.split('\n'):
            words = l.split(' ')
            cur = ""
            for w in words:
                if len(cur + " " + w) > 42:
                    c.setFont("Helvetica", 8.5)
                    c.setFillColor(TEXT_DARK)
                    c.drawString(sx + 15, sy_text, cur)
                    sy_text -= 13
                    cur = w
                else:
                    cur = cur + " " + w if cur else w
            if cur:
                c.setFont("Helvetica", 8.5)
                c.setFillColor(TEXT_DARK)
                c.drawString(sx + 15, sy_text, cur)
                sy_text -= 15
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 6: Benchmark Table
    # -------------------------------------------------------------
    draw_decorations(c, "Empirical Benchmark Across Adult Census Dataset", "MASTER PERFORMANCE TABLE")
    # Draw table
    table_x = 75
    table_y = 330
    col_w = [100, 75, 75, 75, 75, 75, 80, 75, 75, 105]
    row_h = 24

    headers = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Bal. Acc.", "Specificity", "ROC-AUC", "Train (s)", "Complexity"]
    c.setFillColor(NAVY)
    c.rect(table_x, table_y + row_h * 3, sum(col_w), row_h, fill=1, stroke=0)
    cur_x = table_x
    for h, w in zip(headers, col_w):
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(WHITE)
        c.drawCentredString(cur_x + w/2, table_y + row_h * 3 + 7, h)
        cur_x += w

    data_rows = [
        ["CART", "82.70%", "69.10%", "55.22%", "61.38%", "73.52%", "91.81%", "0.8875", "0.032 s", "Depth 8 | 141 Nodes"],
        ["ID3 (Scratch)", "82.40%", "66.59%", "58.84%", "62.47%", "74.52%", "90.21%", "0.8741", "0.188 s", "Depth 6 | 262 Nodes"],
        ["Linear SVM", "79.05%", "71.58%", "26.31%", "38.47%", "61.42%", "96.54%", "0.8596", "8.725 s", "3,390 Support Vecs"]
    ]
    for r_idx, row in enumerate(data_rows):
        cur_x = table_x
        row_y = table_y + row_h * (2 - r_idx)
        c.setFillColor(WHITE if r_idx % 2 == 0 else CARD_BG)
        c.rect(table_x, row_y, sum(col_w), row_h, fill=1, stroke=0)
        c.setStrokeColor(CARD_BORDER)
        c.line(table_x, row_y, table_x + sum(col_w), row_y)
        for val, w in zip(row, col_w):
            c.setFont("Helvetica-Bold" if val == row[0] else "Helvetica", 8.5)
            c.setFillColor(TEXT_DARK)
            c.drawCentredString(cur_x + w/2, row_y + 7, val)
            cur_x += w

    # Takeaway cards
    takeaways = [
        ("Accuracy & Discrimination Leader: CART", NAVY,
         "CART achieves the highest overall accuracy (82.70%) and ROC-AUC (0.8875), with rapid sub-second convergence (0.032s) and optimal binary splits."),
        ("Recall & Minority Leader: ID3 Scratch", SLATE,
         "ID3 captures the highest proportion of high earners (>50K) at 58.84% recall and 62.47% F1-score via multi-way entropy minimization."),
        ("Precision Leader: Linear SVM", colors.HexColor("#28465F"),
         "Linear SVM exhibits minimal false positives (96.54% Specificity, 71.58% Precision) but suffers lower positive recall due to class imbalance.")
    ]
    for (ttitle, tcol, tbody), tx_pos in zip(takeaways, lefts3):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(tx_pos, 85, card_w3, 215, 6, fill=1, stroke=1)
        c.setFillColor(tcol)
        c.roundRect(tx_pos, 265, card_w3, 35, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(WHITE)
        c.drawCentredString(tx_pos + card_w3/2, 277, ttitle)

        words = tbody.split(' ')
        cur = ""
        ty = 240
        for w in words:
            if len(cur + " " + w) > 42:
                c.setFont("Helvetica", 8.5)
                c.setFillColor(TEXT_DARK)
                c.drawString(tx_pos + 15, ty, cur)
                ty -= 14
                cur = w
            else:
                cur = cur + " " + w if cur else w
        if cur:
            c.setFont("Helvetica", 8.5)
            c.setFillColor(TEXT_DARK)
            c.drawString(tx_pos + 15, ty, cur)
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 7: Graphical Benchmark Visualizations
    # -------------------------------------------------------------
    draw_decorations(c, "Comparative Performance & ROC Curve Visualizations", "BENCHMARK CHARTS")
    if os.path.exists("outputs/performance_metrics_comparison.png"):
        c.drawImage("outputs/performance_metrics_comparison.png", 75, 170, width=500, height=270, preserveAspectRatio=True)
    if os.path.exists("outputs/roc_curve_comparison.png"):
        c.drawImage("outputs/roc_curve_comparison.png", 595, 170, width=290, height=270, preserveAspectRatio=True)

    # Caption box
    c.setFillColor(CARD_BG)
    c.setStrokeColor(CARD_BORDER)
    c.roundRect(75, 75, 810, 80, 6, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(NAVY)
    c.drawString(95, 135, "Key Graphical Observations:")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(TEXT_DARK)
    c.drawString(95, 115, "• Metrics Profile: CART & ID3 exhibit superior F1 balance across both classes; Linear SVM excels in specificity (96.5%) but dips in recall.")
    c.drawString(95, 95, "• ROC Analysis: CART (AUC=0.8875) and ID3 (AUC=0.8741) maintain convex dominance across all operational classification thresholds.")
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 8: Step-by-Step Decision Example
    # -------------------------------------------------------------
    draw_decorations(c, "Step-by-Step Decision Example with Simple Explanation", "INTERPRETABLE INFERENCE")
    # Banner
    c.setFillColor(CARD_BG)
    c.setStrokeColor(CARD_BORDER)
    c.roundRect(75, 410, 810, 50, 6, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(SLATE)
    c.drawString(90, 442, "TARGET DEMOGRAPHIC PROFILE WALKTHROUGH:")
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(NAVY)
    c.drawString(90, 422, "Age: 38 | Workclass: Private | Education: Graduation | Marital Status: Married | Sex: Male | Gain: $5,000 | Hours: 45 h/wk")

    if os.path.exists("outputs/step_by_step_decision_flow.png"):
        c.drawImage("outputs/step_by_step_decision_flow.png", 75, 160, width=810, height=240, preserveAspectRatio=True)

    c.setFillColor(CARD_BG)
    c.setStrokeColor(CARD_BORDER)
    c.roundRect(75, 75, 810, 75, 6, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(NAVY)
    c.drawString(95, 130, "Step-by-Step Decision Progression Explanation:")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(TEXT_DARK)
    c.drawString(95, 112, "1. Marital Status (Married) routes profile to dual-income branch -> 2. Capital Gain ($5,000 > 0) confirms significant investment returns.")
    c.drawString(95, 94, "3. Higher Education (Graduation) verifies corporate career trajectory -> 4. Work Engagement (45 hrs >= 40) lands in final leaf (P(>50K) = 91.4%).")
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 9: Proper Decision Tree Diagram
    # -------------------------------------------------------------
    draw_decorations(c, "Hierarchical Decision Tree Architecture (CART & ID3)", "TREE STRUCTURE")
    if os.path.exists("outputs/cart_decision_tree.png"):
        c.drawImage("outputs/cart_decision_tree.png", 75, 175, width=810, height=285, preserveAspectRatio=True)

    c.setFillColor(CARD_BG)
    c.setStrokeColor(CARD_BORDER)
    c.roundRect(75, 75, 810, 90, 6, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(NAVY)
    c.drawString(95, 145, "Decision Tree Structural Mechanics:")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(TEXT_DARK)
    c.drawString(95, 127, "• Binary Splitting Logic: CART evaluates candidate thresholds choosing the split that minimizes weighted child Gini impurity:")
    c.drawString(95, 111, "   I_G(D_L, D_R) = (|D_L|/|D|)*G(D_L) + (|D_R|/|D|)*G(D_R). Each node displays criteria, sample count, and class distribution.")
    c.drawString(95, 93, "• Depth Regularization: Constraining tree depth to 8 levels controls model variance, producing 71 pure leaves without overfitting.")
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 10: Algorithm Comparison Matrix
    # -------------------------------------------------------------
    draw_decorations(c, "Comprehensive Algorithm Comparison Matrix", "HEAD-TO-HEAD CONTRAST")
    c_x = 75
    c_y = 75
    c_w = [130, 220, 230, 230]
    c_headers = ["Evaluation Dimension", "CART (Gini Trees)", "ID3 (Shannon Trees)", "Linear SVM (Max Margin)"]
    c_rows = [
        ["Mathematical Metric", "Gini Impurity: I_G = 1 - sum(p_i^2)", "Shannon Entropy: H = -sum(p_i * log2(p_i))", "Hinge Loss + L2 Penalty: 0.5*||w||^2 + C*sum(xi)"],
        ["Decision Boundary", "Orthogonal, axis-aligned step partitions", "Multi-way nominal hierarchical partitions", "Continuous planar hyperplane (w^T*x + b = 0)"],
        ["Interpretability", "High: Explicit binary flowchart rules", "High: Intuitive multi-branch decision tree", "Low: Dense weight vector (black box)"],
        ["Categorical Data", "Ordinal/binary numeric thresholding", "Native multi-way nominal splitting", "Requires One-Hot Encoding (high sparsity)"],
        ["Continuous Data", "Dynamic candidate threshold scanning", "Quantile discretization (binned)", "Continuous scaling (StandardScaler)"],
        ["Training Complexity", "Fast: O(N * D * log N) ~0.032s", "Moderate: O(N * D) ~0.188s", "Heavy: O(N^2 ~ N^3) ~8.72s"]
    ]
    rh = 55
    c.setFillColor(NAVY)
    c.rect(c_x, c_y + rh * 6, sum(c_w), 30, fill=1, stroke=0)
    cur_x = c_x
    for h, w in zip(c_headers, c_w):
        c.setFont("Helvetica-Bold", 9.5)
        c.setFillColor(WHITE)
        c.drawCentredString(cur_x + w/2, c_y + rh * 6 + 10, h)
        cur_x += w

    for r_idx, row in enumerate(c_rows):
        cur_x = c_x
        row_y = c_y + rh * (5 - r_idx)
        c.setFillColor(WHITE if r_idx % 2 == 0 else CARD_BG)
        c.rect(c_x, row_y, sum(c_w), rh, fill=1, stroke=0)
        c.setStrokeColor(CARD_BORDER)
        c.line(c_x, row_y, c_x + sum(c_w), row_y)
        for c_idx, (val, w) in enumerate(zip(row, c_w)):
            c.setFont("Helvetica-Bold" if c_idx == 0 else "Helvetica", 8)
            c.setFillColor(TEXT_DARK)
            # wrap
            words = val.split(' ')
            cur = ""
            ty = row_y + 35
            for word in words:
                if len(cur + " " + word) > 38:
                    c.drawString(cur_x + 10, ty, cur)
                    ty -= 12
                    cur = word
                else:
                    cur = cur + " " + word if cur else word
            if cur:
                c.drawString(cur_x + 10, ty, cur)
            cur_x += w
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 11: Confusion Matrix Diagnostics
    # -------------------------------------------------------------
    draw_decorations(c, "Confusion Matrix Analysis & Error Diagnostics", "ERROR DECOMPOSITION")
    if os.path.exists("outputs/confusion_matrices.png"):
        c.drawImage("outputs/confusion_matrices.png", 75, 230, width=810, height=220, preserveAspectRatio=True)

    cm_cards = [
        ("CART Error Profile", NAVY,
         "• TP: 275 | FN: 223 | TN: 1,378 | FP: 124\n• Balance: Evenly distributed Type I & Type II errors; solid 69.1% precision and 55.2% recall across testing observations."),
        ("ID3 Error Profile", SLATE,
         "• TP: 293 (Highest) | FN: 205 | TN: 1,354 | FP: 148\n• Balance: Strongest sensitivity for high earners (>50K), capturing 18 more true high-earners than CART."),
        ("Linear SVM Error Profile", colors.HexColor("#28465F"),
         "• TP: 131 | FN: 367 | TN: 1,449 (Highest) | FP: 53\n• Balance: Heavily biased towards majority class; ultra-conservative with minimal false alarms (96.5% Specificity).")
    ]
    for (ctitle, ccol, cbody), cx in zip(cm_cards, lefts3):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(cx, 75, card_w3, 140, 6, fill=1, stroke=1)
        c.setFillColor(ccol)
        c.roundRect(cx, 185, card_w3, 30, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(WHITE)
        c.drawCentredString(cx + card_w3/2, 195, ctitle)

        words = cbody.split('\n')
        cy = 165
        for l in words:
            c.setFont("Helvetica", 8)
            c.setFillColor(TEXT_DARK)
            c.drawString(cx + 10, cy, l)
            cy -= 15
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 12: Horizontal Bar Chart of Top Features
    # -------------------------------------------------------------
    draw_decorations(c, "Top Feature Importance Ranking & One-Line Descriptions", "FEATURE ATTRIBUTION")
    if os.path.exists("outputs/feature_importance_horizontal.png"):
        c.drawImage("outputs/feature_importance_horizontal.png", 75, 75, width=810, height=385, preserveAspectRatio=True)
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 13: Limitations & Future Enhancements
    # -------------------------------------------------------------
    draw_decorations(c, "Model Limitations & Strategic Future Enhancements", "RESEARCH CRITIQUE")
    lims = [
        ("Key Architectural Limitations", RED, [
            "• Class Imbalance Skew: 75:25 negative-to-positive ratio heavily biases Linear SVM towards <=50K, depressing recall without cost-sensitive weights.",
            "• Discretization Information Loss: Binning continuous features (age, hours) into 4 quantile intervals for ID3 strips subtle gradient cues.",
            "• Linear Hyperplane Constraints: Standard Linear SVM cannot capture non-linear feature interactions without kernel transformation.",
            "• Static Demographics: 1994 census does not reflect modern economic inflation, remote work, or technology-sector salary dynamics."
        ]),
        ("Future Research Enhancements", GREEN, [
            "• Ensemble Tree Architectures: Deploy Random Forest & Gradient Boosted Trees (XGBoost/LightGBM) to reduce variance and boost accuracy.",
            "• Non-Linear Kernel SVM: Implement Radial Basis Function (RBF) and Polynomial kernels to map complex non-linear decision boundaries.",
            "• Imbalance Mitigation: Integrate SMOTE (Synthetic Minority Over-sampling) or asymmetric class weights (C_+ > C_-) to elevate recall.",
            "• Advanced Feature Engineering: Formulate net investment cashflow (capital_gain - capital_loss) and education-experience interactions."
        ])
    ]
    lim_w = 390
    lim_lefts = [80, 490]
    for (ltitle, lcol, lbullets), lx in zip(lims, lim_lefts):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(lx, 85, lim_w, 375, 6, fill=1, stroke=1)
        c.setFillColor(lcol)
        c.roundRect(lx, 425, lim_w, 35, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(WHITE)
        c.drawCentredString(lx + lim_w/2, 437, ltitle)

        ly = 400
        for b in lbullets:
            words = b.split(' ')
            cur = ""
            for w in words:
                if len(cur + " " + w) > 58:
                    c.setFont("Helvetica", 8.5)
                    c.setFillColor(TEXT_DARK)
                    c.drawString(lx + 15, ly, cur)
                    ly -= 13
                    cur = "   " + w
                else:
                    cur = cur + " " + w if cur else w
            if cur:
                c.setFont("Helvetica", 8.5)
                c.setFillColor(TEXT_DARK)
                c.drawString(lx + 15, ly, cur)
                ly -= 22
    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 14: Conclusion & Deployment Recommendations
    # -------------------------------------------------------------
    draw_decorations(c, "Executive Summary & Deployment Recommendations", "FINAL SYNTHESIS")
    recs = [
        ("When to Deploy CART", NAVY, [
            "• Recommended For: General production environments requiring balanced accuracy (82.70%), ultra-fast training (0.032s), and fully auditable decision rules.",
            "• Operational Advantage: Native handling of mixed numerical/categorical inputs with zero scaling requirements."
        ]),
        ("When to Deploy ID3", SLATE, [
            "• Recommended For: Mission-critical scenarios where detecting high earners is paramount (highest positive Recall of 58.84% and top F1-Score of 62.47%).",
            "• Operational Advantage: Holistic multi-way branching preserves nominal categorical contexts without binary fragmentation."
        ]),
        ("When to Deploy Linear SVM", colors.HexColor("#28465F"), [
            "• Recommended For: Risk-averse financial environments where False Positives carry severe penalties (highest Precision of 71.58% and Specificity of 96.54%).",
            "• Operational Advantage: Global convex optimality and robust mathematical margin guarantees in high-dimensional feature spaces."
        ])
    ]
    for (rtitle, rcol, rbullets), rx in zip(recs, lefts3):
        c.setFillColor(CARD_BG)
        c.setStrokeColor(CARD_BORDER)
        c.roundRect(rx, 85, card_w3, 375, 6, fill=1, stroke=1)
        c.setFillColor(rcol)
        c.roundRect(rx, 425, card_w3, 35, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(WHITE)
        c.drawCentredString(rx + card_w3/2, 437, rtitle)

        ry = 390
        for b in rbullets:
            words = b.split(' ')
            cur = ""
            for w in words:
                if len(cur + " " + w) > 42:
                    c.setFont("Helvetica", 8.5)
                    c.setFillColor(TEXT_DARK)
                    c.drawString(rx + 15, ry, cur)
                    ry -= 14
                    cur = "   " + w
                else:
                    cur = cur + " " + w if cur else w
            if cur:
                c.setFont("Helvetica", 8.5)
                c.setFillColor(TEXT_DARK)
                c.drawString(rx + 15, ry, cur)
                ry -= 25
    c.showPage()

    c.save()
    print(f"PDF Presentation generated successfully at {pdf_path}")

if __name__ == '__main__':
    generate_pdf()
