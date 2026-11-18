"""
Data Loading and Preprocessing Module for Adult Census Income Dataset.

Handles downloading, inspecting, cleaning, encoding, and stratified splitting
of the UCI Adult Income Dataset with a simplified, intuitive 7-feature schema
optimized for universal clarity and relevance (eliminating redundant and US-specific attributes).
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Clean, intuitive, universal feature set (9 attributes: 4 numerical, 5 categorical)
NUMERICAL_COLS = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'education', 'marital_status', 'sex', 'relationship']
SELECTED_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS


def download_and_load_dataset(data_dir='data', filename='adult.csv'):
    """
    Acquires the UCI Adult Census Income Dataset using a robust, multi-tier strategy:
    Tier 1: Local cache detection (data/adult.csv, adult.csv, /content/adult.csv)
    Tier 2: Scikit-learn OpenML repository (fetch_openml('adult', version=2))
    Tier 3: Official UCI Machine Learning Repository API (ucimlrepo.fetch_ucirepo(id=2))
    Tier 4: Direct HTTP mirrors with browser User-Agent headers
    Tier 5: Fallback to UCI Bank Marketing Dataset (as specified in requirements)
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)

    # -------------------------------------------------------------
    # Tier 1: Local File Inspection
    # -------------------------------------------------------------
    local_candidates = [
        filepath,
        filename,
        os.path.join('/content', filename),
        os.path.join('/content', data_dir, filename)
    ]
    for lp in local_candidates:
        if os.path.exists(lp):
            try:
                print(f"[DataLoader] Found cached dataset at '{lp}'")
                df = pd.read_csv(lp)
                df.columns = [col.replace('-', '_').strip() for col in df.columns]
                if 'class' in df.columns and 'income' not in df.columns:
                    df = df.rename(columns={'class': 'income'})
                if len(df.columns) >= 7:
                    return df
            except Exception as e:
                print(f"[DataLoader] Notice: Failed reading local '{lp}': {e}")

    df_raw = None

    # -------------------------------------------------------------
    # Tier 2: Scikit-Learn OpenML (Built-in, Highly Reliable in Colab)
    # -------------------------------------------------------------
    try:
        print("[DataLoader] Attempting to load via Scikit-Learn OpenML (adult, v2)...")
        from sklearn.datasets import fetch_openml
        oml_data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df_raw = oml_data.frame.copy()
        df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
        if 'class' in df_raw.columns and 'income' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'class': 'income'})
        print(f"[DataLoader] Successfully loaded from OpenML! Shape: {df_raw.shape}")
    except Exception as e:
        print(f"[DataLoader] OpenML attempt failed: {e}")

    # -------------------------------------------------------------
    # Tier 3: Official UCI Machine Learning Repository API (ucimlrepo)
    # -------------------------------------------------------------
    if df_raw is None:
        try:
            print("[DataLoader] Attempting to load via official ucimlrepo (id=2)...")
            from ucimlrepo import fetch_ucirepo
            adult = fetch_ucirepo(id=2)
            df_features = adult.data.features
            df_targets = adult.data.targets
            df_raw = pd.concat([df_features, df_targets], axis=1)
            df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
            if 'class' in df_raw.columns and 'income' not in df_raw.columns:
                df_raw = df_raw.rename(columns={'class': 'income'})
            print(f"[DataLoader] Successfully loaded from ucimlrepo! Shape: {df_raw.shape}")
        except Exception as e:
            print(f"[DataLoader] ucimlrepo attempt failed: {e}")

    # -------------------------------------------------------------
    # Tier 4: Direct HTTP Mirrors with Custom Browser Headers
    # -------------------------------------------------------------
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
                print(f"[DataLoader] Fetching from mirror: {url}...")
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
                print(f"[DataLoader] Successfully downloaded from mirror! Shape: {df_raw.shape}")
                break
            except Exception as e:
                print(f"[DataLoader] Mirror failed ({url}): {e}")

    # -------------------------------------------------------------
    # Tier 5: Fallback to UCI Bank Marketing Dataset (as per requirements)
    # -------------------------------------------------------------
    if df_raw is None:
        print("[DataLoader] WARNING: Primary Adult dataset unavailable across all mirrors. Activating fallback: UCI Bank Marketing dataset...")
        try:
            from sklearn.datasets import fetch_openml
            bank = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
            df_raw = bank.frame.copy()
            df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
            if 'Class' in df_raw.columns:
                df_raw = df_raw.rename(columns={'Class': 'income'})
            print(f"[DataLoader] Fallback Bank Marketing dataset loaded! Shape: {df_raw.shape}")
        except Exception as e:
            print(f"[DataLoader] Fallback dataset attempt failed: {e}")

    if df_raw is None:
        raise RuntimeError(
            "Could not load Adult dataset from local storage, OpenML, ucimlrepo, or remote mirrors.\n"
            "Please check network connection or manually upload 'adult.csv' into your environment."
        )

    # Standardize column naming
    df_raw.columns = [col.replace('-', '_').strip() for col in df_raw.columns]
    if 'class' in df_raw.columns and 'income' not in df_raw.columns:
        df_raw = df_raw.rename(columns={'class': 'income'})

    # Cache locally for instant reloading
    try:
        df_raw.to_csv(filepath, index=False)
        print(f"[DataLoader] Cached dataset to '{filepath}'")
    except Exception:
        pass

    return df_raw


def clean_dataset(df):
    """
    Cleans raw dataset and retains the 9 intuitive attributes:
    - Numerical (4): age, capital_gain, capital_loss, hours_per_week
    - Categorical (5): workclass, education, marital_status, sex, relationship
    - Target: income (<=50K -> 0, >50K -> 1)
    """
    df_clean = df.copy()

    # Column name normalization
    df_clean.columns = [c.replace('-', '_').strip() for c in df_clean.columns]
    if 'class' in df_clean.columns and 'income' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'class': 'income'})

    # Select only the clean, intuitive features + income
    target_col = 'income'
    keep_cols = [c for c in SELECTED_FEATURES if c in df_clean.columns] + [target_col]
    df_clean = df_clean[keep_cols].copy()

    # Normalize string/text columns
    str_cols = [col for col in df_clean.columns if not pd.api.types.is_numeric_dtype(df_clean[col])]
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.rstrip('.')

    # Handle missing values ('?' -> NaN)
    df_clean = df_clean.replace('?', np.nan).replace('None', np.nan).replace('nan', np.nan)

    # -------------------------------------------------------------------------
    # Attribute Category Minimization & Harmonization as Requested:
    # - age: numerical
    # - workclass: Private, Government, Self-Employed
    # - education: 10th, 12th, Graduation, Post-Graduation, PhD
    # - marital_status: Single, Married, Divorced, Widowed
    # - sex: Male, Female
    # - relationship: Husband, Wife, Child, Other
    # -------------------------------------------------------------------------
    # 1. workclass: Private, Government, Self-Employed
    workclass_map = {
        'Private': 'Private',
        'State-gov': 'Government', 'Federal-gov': 'Government', 'Local-gov': 'Government',
        'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed',
        'Without-pay': 'Self-Employed', 'Never-worked': 'Self-Employed'
    }
    if 'workclass' in df_clean.columns:
        df_clean['workclass'] = df_clean['workclass'].map(lambda x: workclass_map.get(str(x).strip(), 'Self-Employed'))

    # 2. education: 10th, 12th, Graduation, Post-Graduation, PhD
    def map_education(e):
        if pd.isna(e):
            return np.nan
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
    if 'education' in df_clean.columns:
        df_clean['education'] = df_clean['education'].apply(map_education)

    # 3. marital_status: Single, Married, Divorced, Widowed
    def map_marital(m):
        if pd.isna(m):
            return np.nan
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
    if 'marital_status' in df_clean.columns:
        df_clean['marital_status'] = df_clean['marital_status'].apply(map_marital)

    # 4. sex: Male, Female
    if 'sex' in df_clean.columns:
        df_clean['sex'] = df_clean['sex'].map(lambda s: 'Female' if str(s).strip().lower().startswith('f') else 'Male')

    # 5. relationship: Husband, Wife, Child, Other
    def map_relationship(r):
        if pd.isna(r):
            return np.nan
        r_str = str(r).strip()
        if r_str == 'Husband':
            return 'Husband'
        elif r_str == 'Wife':
            return 'Wife'
        elif r_str == 'Own-child':
            return 'Child'
        return 'Other'
    if 'relationship' in df_clean.columns:
        df_clean['relationship'] = df_clean['relationship'].apply(map_relationship)

    # Ensure numeric columns are properly typed
    for col in NUMERICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    n_before = len(df_clean)
    df_clean = df_clean.dropna().reset_index(drop=True)
    n_after_na = len(df_clean)

    # Retain all valid survey respondents (identical profiles reflect demographic density)
    n_final = len(df_clean)

    print(f"[DataLoader] Cleaned dataset (9-Feature Schema):")
    print(f"  - Total clean survey records: {n_final:,}")
    print(f"  - Retained features (9): {SELECTED_FEATURES}")

    # Encode binary target: 0 = <=50K, 1 = >50K
    target_str = df_clean['income'].astype(str).str.strip().str.rstrip('.')
    df_clean['income_target'] = target_str.map(lambda x: 1 if '>50k' in str(x).lower() else 0).astype(int)

    pos_count = (df_clean['income_target'] == 1).sum()
    neg_count = (df_clean['income_target'] == 0).sum()
    print(f"  - Target distribution: <=50K (Class 0): {neg_count:,} ({neg_count/len(df_clean)*100:.1f}%), "
          f">50K (Class 1): {pos_count:,} ({pos_count/len(df_clean)*100:.1f}%)")

    return df_clean


def prepare_data_split(df_clean, sample_size=10000, test_size=0.20, random_state=42):
    """
    Subsamples (if requested) and splits data into train and test sets using
    stratified sampling on the target class.
    """
    if sample_size is not None and sample_size < len(df_clean):
        print(f"[DataLoader] Extracting stratified subsample of {sample_size:,} records (from {len(df_clean):,} total) for rapid execution...")
        df_sampled, _ = train_test_split(
            df_clean,
            train_size=sample_size,
            stratify=df_clean['income_target'],
            random_state=random_state
        )
        df_sampled = df_sampled.reset_index(drop=True)
    else:
        print(f"[DataLoader] Using full dataset of {len(df_clean):,} records.")
        df_sampled = df_clean.copy()

    X = df_sampled.drop(columns=['income', 'income_target'])
    y = df_sampled['income_target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"[DataLoader] Stratified Split (80:20):")
    print(f"  - Training samples: {X_train.shape[0]:,} ({y_train.mean()*100:.1f}% positive)")
    print(f"  - Testing samples:  {X_test.shape[0]:,} ({y_test.mean()*100:.1f}% positive)")

    return X_train, X_test, y_train, y_test, NUMERICAL_COLS, CATEGORICAL_COLS
