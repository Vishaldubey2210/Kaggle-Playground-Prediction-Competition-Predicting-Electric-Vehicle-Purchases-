import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from .utils import calculate_metrics


DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}


def train_lgbm_cv(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    target_col: str = "Will_Buy_EV",
    fold_col: str = "fold",
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 50,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float], pd.DataFrame]:
    """
    Train a LightGBM model across cross-validation folds.
    
    Returns:
        oof_preds: Out-of-fold probability predictions.
        test_preds: Average test set probability predictions.
        overall_metrics: Dictionary of CV scores (ROC-AUC, Log Loss, etc.).
        feature_importance_df: DataFrame tracking average feature importances.
    """
    if params is None:
        params = DEFAULT_LGBM_PARAMS.copy()

    n_folds = train_df[fold_col].nunique()
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df)) if test_df is not None else None
    
    feature_importances = np.zeros(len(feature_cols))
    fold_scores = []

    cat_features = [
        col for col in feature_cols 
        if train_df[col].dtype.name in ["category", "object"]
    ]

    # Convert object columns to category for LightGBM
    train_data = train_df.copy()
    test_data = test_df.copy() if test_df is not None else None
    for col in cat_features:
        train_data[col] = train_data[col].astype("category")
        if test_data is not None:
            test_data[col] = test_data[col].astype("category")

    for fold in range(n_folds):
        trn_idx = train_data[fold_col] != fold
        val_idx = train_data[fold_col] == fold

        X_train, y_train = train_data.loc[trn_idx, feature_cols], train_data.loc[trn_idx, target_col]
        X_val, y_val = train_data.loc[val_idx, feature_cols], train_data.loc[val_idx, target_col]

        trn_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data, categorical_feature=cat_features)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0)
        ]

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=num_boost_round,
            valid_sets=[trn_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1}/{n_folds} - ROC-AUC: {fold_auc:.5f} (Best Iter: {model.best_iteration})")

        feature_importances += model.feature_importance(importance_type="gain") / n_folds

        if test_data is not None:
            test_preds += model.predict(test_data[feature_cols], num_iteration=model.best_iteration) / n_folds

    overall_metrics = calculate_metrics(train_data[target_col], oof_preds)
    print(f"\n---> Overall Out-of-Fold ROC-AUC: {overall_metrics['roc_auc']:.5f}")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return oof_preds, test_preds, overall_metrics, importance_df
