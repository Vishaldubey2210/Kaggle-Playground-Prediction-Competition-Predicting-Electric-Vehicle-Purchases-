import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Optional


# Standard mapping dictionary for ordinal or binary string features
ORDINAL_MAPPINGS = {
    "Range_Anxiety_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Home_Charging_Possible": {"No": 0, "Yes": 1},
    "Subsidy_Available": {"No": 0, "Yes": 1},
    "Gender": {"Female": 0, "Male": 1, "Other": 2},
}

TARGET_MAPPING = {"No": 0, "Yes": 1}
REVERSE_TARGET_MAPPING = {0: "No", 1: "Yes"}


def prepare_target(df: pd.DataFrame, target_col: str = "Will_Buy_EV") -> pd.Series:
    """
    Convert target column to numeric binary labels (0 and 1).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    if df[target_col].dtype == object or isinstance(df[target_col].iloc[0], str):
        return df[target_col].map(TARGET_MAPPING).fillna(0).astype(int)
    return df[target_col].astype(int)


def handle_missing_values(df: pd.DataFrame, fill_num: str = "median") -> pd.DataFrame:
    """
    Handle missing values across numerical and categorical features.
    
    Args:
        df: Input dataframe.
        fill_num: Imputation strategy for numeric columns ('median' or 'mean').
        
    Returns:
        pd.DataFrame: Imputed copy of dataframe.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            val = df[col].median() if fill_num == "median" else df[col].mean()
            df[col] = df[col].fillna(val)

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Missing")

    return df


def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    cat_cols: Optional[List[str]] = None,
    method: str = "frequency"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Encode categorical features for machine learning models.
    
    Args:
        train_df: Training set.
        test_df: Optional test set.
        cat_cols: Specific columns to encode. If None, auto-detected.
        method: 'frequency' or 'category' (pandas category type for LightGBM/CatBoost).
        
    Returns:
        Tuple of (train_encoded, test_encoded)
    """
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None

    if cat_cols is None:
        cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    if method == "category":
        for col in cat_cols:
            train[col] = train[col].astype("category")
            if test is not None and col in test.columns:
                test[col] = test[col].astype("category")

    elif method == "frequency":
        for col in cat_cols:
            freq = train[col].value_counts(normalize=True).to_dict()
            train[f"{col}_freq"] = train[col].map(freq).fillna(0)
            if test is not None and col in test.columns:
                test[f"{col}_freq"] = test[col].map(freq).fillna(0)

    return train, test


def create_stratified_folds(
    df: pd.DataFrame,
    target_col: str = "Will_Buy_EV",
    n_splits: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Add a 'fold' column to the dataframe using StratifiedKFold.
    """
    df = df.copy()
    df["fold"] = -1
    y = prepare_target(df, target_col=target_col) if target_col in df.columns else df[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold

    return df
