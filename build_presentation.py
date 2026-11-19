import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    # Theme Colors matching user's template
    NAVY = RGBColor(30, 58, 82)        # #1E3A52 - Main Headers
    SLATE = RGBColor(74, 101, 114)     # #4A6572 - Subtitles / Accents
    MUTED_BLUE = RGBColor(140, 158, 168) # #8C9EA8 - Secondary Triangles
    CARD_BG = RGBColor(245, 247, 250)  # #F5F7FA - Card fill
    CARD_BORDER = RGBColor(218, 225, 231) # #DAE1E7
    TEXT_DARK = RGBColor(33, 43, 54)   # #212B36 - Body text
    TEXT_MUTED = RGBColor(100, 116, 139) # #64748B - Muted labels
    WHITE = RGBColor(255, 255, 255)
    GREEN = RGBColor(22, 160, 133)
    RED = RGBColor(192, 57, 43)

    def add_decorations(slide, title_text, category_text=""):
        # Top-left accent triangles
        t1 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(0), Inches(0), Inches(1.1), Inches(1.1))
        t1.fill.solid()
        t1.fill.fore_color.rgb = SLATE
        t1.line.fill.background()
        t1.rotation = 180

        t2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(0.4), Inches(0), Inches(0.8), Inches(0.8))
        t2.fill.solid()
        t2.fill.fore_color.rgb = MUTED_BLUE
        t2.line.fill.background()
        t2.rotation = 180

        # Bottom-right accent triangles
        b1 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(12.2), Inches(6.4), Inches(1.133), Inches(1.1))
        b1.fill.solid()
        b1.fill.fore_color.rgb = SLATE
        b1.line.fill.background()

        b2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(12.1), Inches(6.7), Inches(0.8), Inches(0.8))
        b2.fill.solid()
        b2.fill.fore_color.rgb = MUTED_BLUE
        b2.line.fill.background()

        # Title & Category Header Box
        header_box = slide.shapes.add_textbox(Inches(1.5), Inches(0.35), Inches(8.2), Inches(1.0))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        if category_text:
            p_cat = tf.paragraphs[0]
            p_cat.text = category_text.upper()
            p_cat.font.size = Pt(10)
            p_cat.font.bold = True
            p_cat.font.color.rgb = SLATE
            p_cat.space_after = Pt(2)
            p_title = tf.add_paragraph()
        else:
            p_title = tf.paragraphs[0]

        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

        # Footer
        footer_box = slide.shapes.add_textbox(Inches(3.8), Inches(7.15), Inches(6.0), Inches(0.3))
        ftf = footer_box.text_frame
        p_foot = ftf.paragraphs[0]
        p_foot.text = "Comparative Benchmark: CART vs ID3 vs Linear SVM | Adult Income Census"
        p_foot.font.size = Pt(8.5)
        p_foot.font.color.rgb = TEXT_MUTED
        p_foot.alignment = PP_ALIGN.CENTER

    def create_card(slide, left, top, width, height, title="", title_color=NAVY):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1)

        if title:
            # Title banner
            banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, Inches(0.45))
            banner.fill.solid()
            banner.fill.fore_color.rgb = title_color
            banner.line.fill.background()

            tx = slide.shapes.add_textbox(left, top, width, Inches(0.45))
            tf = tx.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.text = title
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
        return card

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    
    # Sleek corner accents (without disconnected thin frame lines)
    t1 = slide1.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(0), Inches(0), Inches(1.15), Inches(1.15))
    t1.fill.solid()
    t1.fill.fore_color.rgb = SLATE
    t1.line.fill.background()
    t1.rotation = 180

    t2 = slide1.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(0.42), Inches(0), Inches(0.8), Inches(0.8))
    t2.fill.solid()
    t2.fill.fore_color.rgb = MUTED_BLUE
    t2.line.fill.background()
    t2.rotation = 180

    b1 = slide1.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(12.18), Inches(6.35), Inches(1.15), Inches(1.15))
    b1.fill.solid()
    b1.fill.fore_color.rgb = SLATE
    b1.line.fill.background()

    b2 = slide1.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(12.1), Inches(6.7), Inches(0.8), Inches(0.8))
    b2.fill.solid()
    b2.fill.fore_color.rgb = MUTED_BLUE
    b2.line.fill.background()

    # Footer
    footer_box1 = slide1.shapes.add_textbox(Inches(3.8), Inches(7.15), Inches(6.0), Inches(0.3))
    ftf1 = footer_box1.text_frame
    p_foot1 = ftf1.paragraphs[0]
    p_foot1.text = "Comparative Benchmark: CART vs ID3 vs Linear SVM | Adult Income Census"
    p_foot1.font.size = Pt(8.5)
    p_foot1.font.color.rgb = TEXT_MUTED
    p_foot1.alignment = PP_ALIGN.CENTER
    
    # Center-aligned middle text block
    title_box = slide1.shapes.add_textbox(Inches(1.416), Inches(1.3), Inches(10.5), Inches(3.2))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

    # Top line - smaller, slate accent color
    p = tf1.paragraphs[0]
    p.text = "COMPARATIVE ANALYSIS OF"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SLATE
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(6)

    # Main Title - biggest and boldest!
    p2 = tf1.add_paragraph()
    p2.text = "CART, ID3 & LINEAR SVM"
    p2.font.size = Pt(38)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.alignment = PP_ALIGN.CENTER
    p2.space_after = Pt(14)

    # Subtitle
    p3 = tf1.add_paragraph()
    p3.text = "Benchmarking Algorithms for Demographic Income Classification"
    p3.font.size = Pt(15.5)
    p3.font.color.rgb = TEXT_DARK
    p3.alignment = PP_ALIGN.CENTER
    p3.space_after = Pt(12)

    # Tagline
    p4 = tf1.add_paragraph()
    p4.text = "Gini Impurity vs. Shannon Entropy vs. Maximum Margin Hyperplane"
    p4.font.size = Pt(12)
    p4.font.color.rgb = SLATE
    p4.alignment = PP_ALIGN.CENTER

    # Bottom Box (faint transparent grey fill, NO outline, left-aligned bulleted list)
    card_meta = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.366), Inches(4.75), Inches(6.6), Inches(1.85))
    card_meta.fill.solid()
    card_meta.fill.fore_color.rgb = RGBColor(243, 246, 249) # faint transparent grey
    card_meta.line.fill.background() # remove outline completely

    meta_tx = slide1.shapes.add_textbox(Inches(3.766), Inches(4.9), Inches(5.8), Inches(1.55))
    mtf = meta_tx.text_frame
    mtf.word_wrap = True
    mtf.margin_left = mtf.margin_top = mtf.margin_right = mtf.margin_bottom = 0

    bottom_items = [
        ("Dataset:", " UCI Adult Census (30,162 clean records)"),
        ("Split:", " 80:20 Stratified"),
        ("Target:", " Annual Income (> 50K vs <= 50K)"),
        ("Metrics:", " Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, Latency")
    ]

    for idx, (label, val) in enumerate(bottom_items):
        p_item = mtf.paragraphs[0] if idx == 0 else mtf.add_paragraph()
        p_item.alignment = PP_ALIGN.LEFT
        p_item.space_after = Pt(4)

        run_bullet = p_item.add_run()
        run_bullet.text = "•  "
        run_bullet.font.bold = True
        run_bullet.font.size = Pt(11)
        run_bullet.font.color.rgb = SLATE

        run_label = p_item.add_run()
        run_label.text = label
        run_label.font.bold = True
        run_label.font.size = Pt(11)
        run_label.font.color.rgb = NAVY

        run_val = p_item.add_run()
        run_val.text = val
        run_val.font.bold = False
        run_val.font.size = Pt(11)
        run_val.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 2: Project Objectives
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_decorations(slide2, "Project Overview & Research Objectives", "FOUNDATIONAL BENCHMARK")

    # 4 Cards for Objectives with exactly 3 bullets each
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

    card_w = Inches(5.35)
    card_h = Inches(2.55)
    positions = [
        (Inches(1.15), Inches(1.55)),
        (Inches(6.8), Inches(1.55)),
        (Inches(1.15), Inches(4.35)),
        (Inches(6.8), Inches(4.35))
    ]

    for (title, bullets), (left, top) in zip(objs, positions):
        create_card(slide2, left, top, card_w, card_h, title, NAVY)
        tx = slide2.shapes.add_textbox(left + Inches(0.2), top + Inches(0.52), card_w - Inches(0.4), card_h - Inches(0.58))
        tf = tx.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        for b_idx, bullet in enumerate(bullets):
            p = tf.paragraphs[0] if b_idx == 0 else tf.add_paragraph()
            p.space_after = Pt(6)
            
            run_bullet = p.add_run()
            run_bullet.text = "•  "
            run_bullet.font.bold = True
            run_bullet.font.size = Pt(10.5)
            run_bullet.font.color.rgb = SLATE

            run_text = p.add_run()
            run_text.text = bullet
            run_text.font.bold = False
            run_text.font.size = Pt(10.5)
            run_text.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 3: Model Introduction
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_decorations(slide3, "Model Introduction & Algorithmic Foundations", "ALGORITHM TAXONOMY")

    models_info = [
        ("CART (Binary Decision Tree)", NAVY, [
            "Paradigm: Non-parametric binary recursive tree.",
            "Splitting Metric: Gini Impurity reduction:",
            "  I_G(t) = 1 - sum(p_i^2)",
            "Boundary: Axis-aligned orthogonal step partitions.",
            "Implementation: Scikit-Learn DecisionTreeClassifier (max_depth=8, criterion='gini').",
            "Output: Leaf probability distribution & binary class."
        ]),
        ("ID3 (Scratch Shannon Tree)", SLATE, [
            "Paradigm: Multi-way nominal categorical tree.",
            "Splitting Metric: Information Gain maximization:",
            "  IG(D, A) = H(D) - sum(|D_v|/|D| * H(D_v))",
            "Entropy: H(D) = - sum(p_i * log2(p_i))",
            "Continuous Handling: 4-bin Quantile Discretizer.",
            "Implementation: 100% scratch Python engine (max_depth=6, min_samples_split=10)."
        ]),
        ("Linear SVM (Maximum Margin)", RGBColor(40, 70, 95), [
            "Paradigm: Maximum-margin linear hyperplane.",
            "Optimization Objective:",
            "  min 0.5*||w||^2 + C * sum(xi_i)",
            "Decision Rule: y_pred = sign(w^T * x + b)",
            "Preprocessing: StandardScaler + OneHotEncoder.",
            "Implementation: SVC(kernel='linear', C=1.0, probability=True, random_state=42)."
        ])
    ]

    card_w3 = Inches(3.55)
    card_h3 = Inches(5.1)
    lefts3 = [Inches(1.0), Inches(4.88), Inches(8.76)]

    for (title, col, bullets), left in zip(models_info, lefts3):
        create_card(slide3, left, Inches(1.6), card_w3, card_h3, title, col)
        tx = slide3.shapes.add_textbox(left + Inches(0.2), Inches(2.2), card_w3 - Inches(0.4), card_h3 - Inches(0.7))
        tf = tx.text_frame
        tf.word_wrap = True
        for b in bullets:
            p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
            p.text = b
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 4: Why This Model Is Used
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_decorations(slide4, "Model Selection Rationale: Why These Models?", "COMPARATIVE DESIGN")

    rationales = [
        ("Why CART?", NAVY, [
            "Universal Robustness: Naturally handles mixed continuous & categorical attributes without feature scale distortions.",
            "Cognitive Interpretability: Produces transparent, binary if-then rules easily audited by financial policy makers.",
            "Invariance to Monotonic Shifts: Splitting decisions are unaffected by monotonic scaling or outliers in capital gains.",
            "Computational Speed: Ultra-fast O(N log N) tree building with sub-millisecond evaluation latency."
        ]),
        ("Why ID3 from Scratch?", SLATE, [
            "Information-Theoretic Foundation: Explores pure Shannon entropy reduction and Information Gain dynamics.",
            "Multi-Way Branching: Captures full nominal categorical interactions (e.g., all 5 education tiers simultaneously) without binary fragmentation.",
            "Full Algorithmic Transparency: Scratch implementation allows complete inspection of node creation, impurity metrics, and tree traversal.",
            "Continuous Discretization: Demonstrates adaptive quantile binning to reconcile continuous inputs with nominal tree logic."
        ]),
        ("Why Linear SVM?", RGBColor(40, 70, 95), [
            "Continuous Geometric Contrast: Serves as the mathematical counterpoint to recursive piecewise-constant step functions.",
            "Maximum-Margin Principle: Guarantees maximum boundary separation between income classes, reducing generalization error.",
            "Convex Global Optimality: Quadratic programming guarantees a single global minimum—no local minima traps.",
            "High-Dimensional Efficacy: Highly effective in the sparse, high-dimensional space created by one-hot encoded demographic features."
        ])
    ]

    for (title, col, bullets), left in zip(rationales, lefts3):
        create_card(slide4, left, Inches(1.6), card_w3, card_h3, title, col)
        tx = slide4.shapes.add_textbox(left + Inches(0.2), Inches(2.2), card_w3 - Inches(0.4), card_h3 - Inches(0.7))
        tf = tx.text_frame
        tf.word_wrap = True
        for b in bullets:
            p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
            p.text = "• " + b
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 5: Process Flow Sequence
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_decorations(slide5, "End-to-End Workflow & Pipeline Architecture", "EXECUTION SEQUENCE")

    steps = [
        ("Stage 1: Data Ingestion", "Acquires 30,162 records from UCI Adult Census; removes '?' missing tokens; cleans text headers."),
        ("Stage 2: 9-Feature Schema", "Isolates 4 numerical (age, hours, capital gain/loss) and 5 categorical (workclass, education, marital, sex, relation)."),
        ("Stage 3: Stratified Split", "Partitions dataset into 80% Train (8,000) & 20% Test (2,000) strictly preserving 75.1% : 24.9% income ratio."),
        ("Stage 4: Transformations", "CART: Direct label/ordinal encoding.\nID3: 4-interval quantile discretization.\nSVM: StandardScaler + OneHotEncoder."),
        ("Stage 5: Parallel Model Fit", "Trains CART (depth=8), ID3 (depth=6 from scratch), and Linear SVM (C=1.0) with microsecond latency logging."),
        ("Stage 6: Master Evaluation", "Generates performance matrices, confusion heatmaps, ROC curves, decision tree graphs, and interactive UI.")
    ]

    card_w5 = Inches(3.55)
    card_h5 = Inches(2.3)
    coords5 = [
        (Inches(1.0), Inches(1.6)),
        (Inches(4.88), Inches(1.6)),
        (Inches(8.76), Inches(1.6)),
        (Inches(1.0), Inches(4.3)),
        (Inches(4.88), Inches(4.3)),
        (Inches(8.76), Inches(4.3))
    ]

    for i, ((title, desc), (left, top)) in enumerate(zip(steps, coords5)):
        create_card(slide5, left, top, card_w5, card_h5, f"{title}", NAVY if i < 3 else SLATE)
        tx = slide5.shapes.add_textbox(left + Inches(0.2), top + Inches(0.55), card_w5 - Inches(0.4), card_h5 - Inches(0.65))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 6: Benchmark Table
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_decorations(slide6, "Empirical Benchmark Across Adult Census Dataset", "MASTER PERFORMANCE TABLE")

    # Table dimensions
    t_left = Inches(1.0)
    t_top = Inches(1.6)
    t_width = Inches(11.33)
    t_height = Inches(2.2)

    rows, cols = 4, 10
    table_shape = slide6.shapes.add_table(rows, cols, t_left, t_top, t_width, t_height)
    tbl = table_shape.table

    headers = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Bal. Acc.", "Specificity", "ROC-AUC", "Train (s)", "Complexity"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    data_rows = [
        ["CART", "82.70%", "69.10%", "55.22%", "61.38%", "73.52%", "91.81%", "0.8875", "0.032 s", "Depth 8 | 141 Nodes"],
        ["ID3 (Scratch)", "82.40%", "66.59%", "58.84%", "62.47%", "74.52%", "90.21%", "0.8741", "0.188 s", "Depth 6 | 262 Nodes"],
        ["Linear SVM", "79.05%", "71.58%", "26.31%", "38.47%", "61.42%", "96.54%", "0.8596", "8.725 s", "3,390 Support Vectors"]
    ]

    for i, row in enumerate(data_rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_DARK
            p.alignment = PP_ALIGN.CENTER
            if j == 0:
                p.font.bold = True

    # Takeaway cards below table
    takeaways = [
        ("Accuracy & Discrimination Leader: CART (82.70% / 0.8875 ROC-AUC)", NAVY,
         "CART achieves the highest overall accuracy and discrimination capacity, with rapid sub-second convergence (0.032s) and optimal binary splits."),
        ("Recall & Minority Detection Leader: ID3 Scratch (58.84% Recall / 62.47% F1)", SLATE,
         "ID3 captures the highest proportion of high earners (>50K), yielding the strongest F1-score balance via multi-way entropy minimization."),
        ("Precision & Specificity Leader: Linear SVM (71.58% Precision / 96.54% Specificity)", RGBColor(40, 70, 95),
         "Linear SVM exhibits minimal false positives but suffers lower positive recall due to high-dimensional class imbalance.")
    ]

    t_card_w = Inches(3.55)
    t_card_h = Inches(2.6)
    t_lefts = [Inches(1.0), Inches(4.88), Inches(8.76)]

    for (title, col, body), left in zip(takeaways, t_lefts):
        create_card(slide6, left, Inches(4.2), t_card_w, t_card_h, title, col)
        tx = slide6.shapes.add_textbox(left + Inches(0.2), Inches(4.8), t_card_w - Inches(0.4), t_card_h - Inches(0.7))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = body
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 7: Graphical Benchmark Visualizations
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_decorations(slide7, "Comparative Performance & ROC Curve Visualizations", "BENCHMARK CHARTS")

    # Embed performance_metrics_comparison.png and roc_curve_comparison.png
    if os.path.exists("outputs/performance_metrics_comparison.png"):
        slide7.shapes.add_picture("outputs/performance_metrics_comparison.png", Inches(1.0), Inches(1.6), width=Inches(6.6))
    if os.path.exists("outputs/roc_curve_comparison.png"):
        slide7.shapes.add_picture("outputs/roc_curve_comparison.png", Inches(7.9), Inches(1.6), width=Inches(4.4))

    # Bottom caption card
    c_card = create_card(slide7, Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.2), "Key Graphical Observations", NAVY)
    tx = slide7.shapes.add_textbox(Inches(1.2), Inches(6.05), Inches(10.9), Inches(0.7))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Metrics Profile: CART & ID3 exhibit superior F1 balance across both classes; Linear SVM excels in specificity (96.5%) but dips in recall.\n• ROC Analysis: CART (AUC=0.8875) and ID3 (AUC=0.8741) maintain convex dominance across all operational thresholds."
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 8: Step-by-Step Decision Example
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_decorations(slide8, "Step-by-Step Decision Example with Simple Explanation", "INTERPRETABLE INFERENCE")

    # Demographic profile banner
    prof_box = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(11.33), Inches(0.85))
    prof_box.fill.solid()
    prof_box.fill.fore_color.rgb = CARD_BG
    prof_box.line.color.rgb = CARD_BORDER
    prof_tx = slide8.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.1), Inches(0.75))
    ptf = prof_tx.text_frame
    ptf.word_wrap = True
    pp = ptf.paragraphs[0]
    pp.text = "Target Subject Demographic Profile:"
    pp.font.bold = True
    pp.font.size = Pt(10)
    pp.font.color.rgb = SLATE
    pp2 = ptf.add_paragraph()
    pp2.text = "Age: 38 | Workclass: Private | Education: Graduation | Marital Status: Married | Sex: Male | Rel: Husband | Gain: $5,000 | Loss: $0 | Hours: 45 h/wk"
    pp2.font.bold = True
    pp2.font.size = Pt(11)
    pp2.font.color.rgb = NAVY

    # Flowchart image
    if os.path.exists("outputs/step_by_step_decision_flow.png"):
        slide8.shapes.add_picture("outputs/step_by_step_decision_flow.png", Inches(1.0), Inches(2.45), width=Inches(11.33))

    # Explanatory card below flowchart
    exp_card = create_card(slide8, Inches(1.0), Inches(5.8), Inches(11.33), Inches(1.2), "Traversing the Decision Tree Step-by-Step", SLATE)
    etx = slide8.shapes.add_textbox(Inches(1.2), Inches(6.25), Inches(10.9), Inches(0.65))
    etf = etx.text_frame
    etf.word_wrap = True
    ep = etf.paragraphs[0]
    ep.text = "1. Marital Status (Married) routes profile to dual-income branch -> 2. Capital Gain ($5,000 > 0) confirms significant investment returns -> 3. Higher Education (Graduation) verifies corporate career trajectory -> 4. Work Engagement (45 hrs >= 40) lands in final high-earner leaf (P(>50K) = 91.4%)."
    ep.font.size = Pt(10.5)
    ep.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 9: Proper Decision Tree Diagram
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_decorations(slide9, "Hierarchical Decision Tree Architecture (CART & ID3)", "TREE STRUCTURE")

    # Tree image
    if os.path.exists("outputs/cart_decision_tree.png"):
        slide9.shapes.add_picture("outputs/cart_decision_tree.png", Inches(1.0), Inches(1.5), width=Inches(11.33))

    tree_card = create_card(slide9, Inches(1.0), Inches(5.5), Inches(11.33), Inches(1.5), "Decision Tree Structural Mechanics", NAVY)
    ttx = slide9.shapes.add_textbox(Inches(1.2), Inches(5.95), Inches(10.9), Inches(0.95))
    ttf = ttx.text_frame
    ttf.word_wrap = True
    tp = ttf.paragraphs[0]
    tp.text = "• Binary Splitting Logic: CART evaluates every candidate threshold (e.g. marital_status <= 0.5, capital_gain <= 5095.5) choosing the split that minimizes weighted child Gini impurity: I_G(D_L, D_R) = (|D_L|/|D|)*G(D_L) + (|D_R|/|D|)*G(D_R).\n• Depth Regularization: Constraining tree depth to 8 levels controls model variance, yielding 71 pure leaves while completely eliminating runaway overfitting."
    tp.font.size = Pt(10.5)
    tp.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 10: Multi-Dimensional Algorithm Comparison
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_decorations(slide10, "Comprehensive Algorithm Comparison Matrix", "HEAD-TO-HEAD CONTRAST")

    comp_left = Inches(1.0)
    comp_top = Inches(1.6)
    comp_w = Inches(11.33)
    comp_h = Inches(4.9)

    tbl_comp = slide10.shapes.add_table(7, 4, comp_left, comp_top, comp_w, comp_h).table

    headers_comp = ["Evaluation Dimension", "CART (Gini Trees)", "ID3 (Shannon Trees)", "Linear SVM (Max Margin)"]
    for j, h in enumerate(headers_comp):
        cell = tbl_comp.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    comp_matrix_data = [
        ["Mathematical Metric", "Gini Impurity:\nI_G = 1 - sum(p_i^2)", "Shannon Entropy:\nH = -sum(p_i * log2(p_i))", "Hinge Loss + L2 Penalty:\n0.5*||w||^2 + C*sum(xi_i)"],
        ["Decision Boundary", "Orthogonal, axis-aligned step partitions", "Multi-way nominal hierarchical partitions", "Continuous planar separating hyperplane (w^T*x + b = 0)"],
        ["Interpretability", "High: Explicit binary flowchart rules", "High: Intuitive multi-branch decision tree", "Low: Dense weight vector (black box)"],
        ["Categorical Handling", "Ordinal/binary numeric thresholding", "Native multi-way nominal splitting", "Requires One-Hot Encoding (high sparsity)"],
        ["Continuous Features", "Dynamic candidate threshold scanning", "Quantile discretization (binned)", "Continuous scaling (StandardScaler)"],
        ["Training Complexity", "Fast: O(N * D * log N) ~0.032s", "Moderate: O(N * D) ~0.188s", "Computationally Heavy: O(N^2 ~ N^3) ~8.72s"]
    ]

    for i, row in enumerate(comp_matrix_data):
        for j, val in enumerate(row):
            cell = tbl_comp.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_DARK
            if j == 0:
                p.font.bold = True
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 11: Confusion Matrix Diagnostics
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_decorations(slide11, "Confusion Matrix Analysis & Error Diagnostics", "ERROR DECOMPOSITION")

    # Embed confusion_matrices.png
    if os.path.exists("outputs/confusion_matrices.png"):
        slide11.shapes.add_picture("outputs/confusion_matrices.png", Inches(1.0), Inches(1.6), width=Inches(11.33))

    # Error analysis cards
    cm_cards = [
        ("CART Error Profile", NAVY,
         "• True Positives: 275 | False Negatives: 223\n• True Negatives: 1,378 | False Positives: 124\n• Balance: Evenly distributed Type I and Type II errors; solid 69.1% precision and 55.2% recall."),
        ("ID3 Error Profile", SLATE,
         "• True Positives: 293 (Highest) | FN: 205\n• True Negatives: 1,354 | False Positives: 148\n• Balance: Strongest sensitivity for high earners (>50K), capturing 18 more true high-earners than CART."),
        ("Linear SVM Error Profile", RGBColor(40, 70, 95),
         "• True Positives: 131 | False Negatives: 367\n• True Negatives: 1,449 (Highest) | FP: 53\n• Balance: Heavily biased towards majority class; ultra-conservative with minimal false alarms (96.5% Specificity).")
    ]

    for (title, col, body), left in zip(cm_cards, t_lefts):
        create_card(slide11, left, Inches(4.5), t_card_w, Inches(2.2), title, col)
        tx = slide11.shapes.add_textbox(left + Inches(0.2), Inches(5.05), t_card_w - Inches(0.4), Inches(1.5))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = body
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 12: Horizontal Bar Chart of Top Features
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_decorations(slide12, "Top Feature Importance Ranking & One-Line Descriptions", "FEATURE ATTRIBUTION")

    # Embed feature_importance_horizontal.png
    if os.path.exists("outputs/feature_importance_horizontal.png"):
        slide12.shapes.add_picture("outputs/feature_importance_horizontal.png", Inches(1.0), Inches(1.5), width=Inches(11.33))

    # =========================================================================
    # SLIDE 13: Limitations & Future Enhancements
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_decorations(slide13, "Model Limitations & Strategic Future Enhancements", "RESEARCH CRITIQUE")

    lims = [
        ("Key Architectural Limitations", RED, [
            "Class Imbalance Skew: The 75:25 negative-to-positive class distribution heavily biases Linear SVM towards predicting the <=50K majority, depressing recall without cost-sensitive weight tuning.",
            "Information Loss via Discretization: Binning continuous features (age, hours) into 4 quantile intervals for ID3 strips fine-grained gradient signals along numerical boundaries.",
            "Linear Hyperplane Constraints: Standard Linear SVM cannot capture complex non-linear feature interactions (e.g. age * education * capital gains) without non-linear kernel projection.",
            "Static Census Demographics: 1994 census data does not reflect contemporary economic inflation, gig-economy dynamics, or modern technology-sector salary structures."
        ]),
        ("Future Research Enhancements", GREEN, [
            "Ensemble Tree Architectures: Deploy Random Forest and Gradient Boosted Decision Trees (XGBoost / LightGBM) to dramatically reduce variance and capture higher-order feature interactions.",
            "Non-Linear Kernel SVM: Implement Radial Basis Function (RBF) and Polynomial kernels to project non-linear demographic boundaries into higher-dimensional Hilbert spaces.",
            "Class Imbalance Mitigation: Integrate SMOTE (Synthetic Minority Over-sampling Technique) or asymmetric class weights (C_+ > C_-) to elevate minority class recall.",
            "Advanced Feature Engineering: Construct net financial cashflow metrics (capital_gain - capital_loss) and demographic interaction indices to enrich feature representation."
        ])
    ]

    lim_w = Inches(5.45)
    lim_h = Inches(5.0)
    lim_lefts = [Inches(1.0), Inches(6.85)]

    for (title, col, bullets), left in zip(lims, lim_lefts):
        create_card(slide13, left, Inches(1.6), lim_w, lim_h, title, col)
        tx = slide13.shapes.add_textbox(left + Inches(0.2), Inches(2.2), lim_w - Inches(0.4), lim_h - Inches(0.7))
        tf = tx.text_frame
        tf.word_wrap = True
        for b in bullets:
            p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
            p.text = "• " + b
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 14: Conclusion & Deployment Recommendations
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_decorations(slide14, "Executive Summary & Deployment Recommendations", "FINAL SYNTHESIS")

    recs = [
        ("When to Deploy CART", NAVY, [
            "Best For: General production systems requiring balanced accuracy (82.70%), ultra-fast training (0.032s), and fully auditable decision rules.",
            "Operational Advantage: Native handling of mixed numerical/categorical inputs with zero scaling requirements."
        ]),
        ("When to Deploy ID3", SLATE, [
            "Best For: Applications where detecting high earners is paramount (highest positive Recall of 58.84% and top F1-Score of 62.47%).",
            "Operational Advantage: Holistic multi-way branching preserves nominal categorical contexts without binary binarization."
        ]),
        ("When to Deploy Linear SVM", RGBColor(40, 70, 95), [
            "Best For: Risk-averse financial environments where False Positives carry severe penalties (highest Precision of 71.58% and Specificity of 96.54%).",
            "Operational Advantage: Global convex optimality and robust mathematical margin guarantees in high-dimensional feature spaces."
        ])
    ]

    for (title, col, bullets), left in zip(recs, t_lefts):
        create_card(slide14, left, Inches(1.6), t_card_w, Inches(4.9), title, col)
        tx = slide14.shapes.add_textbox(left + Inches(0.2), Inches(2.2), t_card_w - Inches(0.4), Inches(4.1))
        tf = tx.text_frame
        tf.word_wrap = True
        for b in bullets:
            p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
            p.text = b
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(14)

    # Save presentation cleanly handling any open file locks
    out_pptx = "Comparative_Performance_Analysis_Presentation.pptx"
    updated_pptx = "Comparative_Performance_Analysis_Presentation_Updated.pptx"
    
    saved_any = False
    try:
        prs.save(out_pptx)
        print(f"Presentation saved successfully to {out_pptx}")
        saved_any = True
    except PermissionError:
        print(f"Note: {out_pptx} is currently open in PowerPoint/viewer.")

    try:
        prs.save(updated_pptx)
        print(f"Presentation saved successfully to {updated_pptx}")
        saved_any = True
    except PermissionError:
        print(f"Note: {updated_pptx} is currently open in PowerPoint/viewer.")

    if not saved_any:
        fallback_pptx = "Comparative_Performance_Analysis_Presentation_v2.pptx"
        prs.save(fallback_pptx)
        print(f"Saved to fallback file: {fallback_pptx}")

if __name__ == '__main__':
    create_deck()
