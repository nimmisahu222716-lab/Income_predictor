"""
Inference Module for Comparative Classification Models.

Enables interactive single-sample, batch, and terminal-based predictions
across CART, ID3, and Linear SVM models.
"""

import pandas as pd


def predict_sample(sample_input, models, discretizer, num_cols, cat_cols):
    """
    Generates comparative predictions for custom user inputs across CART, ID3, and Linear SVM.

    Parameters:
    -----------
    sample_input : dict or pd.DataFrame
        Dictionary or DataFrame containing all 9 required features.
    models : dict
        Trained models dictionary {'CART': cart_wrapper, 'ID3': id3_clf, 'Linear SVM': svm_wrapper}.
    discretizer : AdaptiveQuantileDiscretizer
        Fitted quantile discretizer for ID3 numerical features.
    num_cols : list
        List of numerical feature names.
    cat_cols : list
        List of categorical feature names.

    Returns:
    --------
    pd.DataFrame
        Comparative table with predicted class and confidence percentages.
    """
    if isinstance(sample_input, dict):
        sample_df = pd.DataFrame([sample_input])
    elif isinstance(sample_input, pd.DataFrame):
        sample_df = sample_input.copy()
    else:
        raise ValueError("sample_input must be a Python dictionary or pandas DataFrame.")

    # Validate required columns
    all_cols = set(num_cols + cat_cols)
    missing = all_cols - set(sample_df.columns)
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

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

    # Preprocess for ID3 (Quantile Discretization)
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
    """
    Collects user input interactively via Python terminal prompts (input()) in Google Colab or terminal,
    runs inference across CART, ID3, and Linear SVM, and prints comparative prediction results along
    with the master performance metrics comparison matrix and confusion matrices.
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE INCOME PREDICTION TERMINAL (CART vs. ID3 vs. Linear SVM)")
    print("=" * 80)
    print("Enter individual feature values at each prompt below.")
    print("TIP: Press [Enter] without typing to accept the suggested default value.\n")

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
                val_str = input(f"{prompt_text}\n  Options: {opts_display}\n  Your Selection [default: '{default_val}']: ").strip()
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

    print("\n" + "-" * 80)
    print("COLLECTED PROFILE SUMMARY:")
    for k, v in user_dict.items():
        print(f"  * {k:18s}: {v}")
    print("-" * 80)

    # Run predictions
    results_df = predict_sample(user_dict, models, discretizer, num_cols, cat_cols)
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS ACROSS ALGORITHMS:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Display Master Performance Metrics Matrix
    if master_table is not None:
        print("=" * 80)
        print("MASTER PERFORMANCE METRICS MATRIX (CART vs. ID3 vs. Linear SVM):")
        print("=" * 80)
        print(master_table.to_string(index=False))
        print("=" * 80 + "\n")

    # Display Confusion Matrix for each algorithm
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
                print(f"\n[{algo} Algorithm Confusion Matrix]")
                print("                       Predicted <=50K       Predicted >50K")
                print(f"  Actual <=50K (Neg)      {tn:8,d} (TN)          {fp:8,d} (FP)")
                print(f"  Actual >50K  (Pos)      {fn:8,d} (FN)          {tp:8,d} (TP)")
                print(f"  -> Accuracy: {acc*100:.2f}% | TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
        print("=" * 80 + "\n")

    return results_df
