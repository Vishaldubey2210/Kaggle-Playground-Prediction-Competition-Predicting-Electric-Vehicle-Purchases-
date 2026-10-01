import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score


def seed_everything(seed: int = 42):
    """Set seeds for reproducibility across random, numpy, etc."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def load_data(data_dir: str = "data"):
    """
    Load train, test, and sample submission files.
    
    Args:
        data_dir: Relative or absolute path to the data folder.
        
    Returns:
        tuple: (train_df, test_df, sample_sub_df)
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(data_dir, "sample_submission.csv")

    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    sample_sub_df = pd.read_csv(sub_path) if os.path.exists(sub_path) else None

    return train_df, test_df, sample_sub_df


def calculate_metrics(y_true, y_pred_proba, threshold: float = 0.5):
    """
    Calculate primary and secondary classification evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class (1).
        threshold: Decision threshold for discrete classes (default 0.5).
        
    Returns:
        dict: Computed metrics (roc_auc, log_loss, accuracy, f1).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_submission(
    ids,
    predictions,
    id_col: str = "id",
    target_col: str = "Will_Buy_EV",
    output_path: str = "submissions/submission.csv"
):
    """
    Format and save submission predictions to a CSV file.
    
    Args:
        ids: Series or array of test sample IDs.
        predictions: Predicted probabilities for positive class.
        id_col: Name of ID column.
        target_col: Name of target column.
        output_path: Destination CSV path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub = pd.DataFrame({
        id_col: ids,
        target_col: predictions
    })
    sub.to_csv(output_path, index=False)
    print(f"Submission saved successfully to: {output_path} (Shape: {sub.shape})")


def log_experiment(
    log_path: str = "experiments/experiment_log.csv",
    experiment_id: str = None,
    model_name: str = "",
    cv_roc_auc: float = 0.0,
    public_lb_score: float = None,
    features_used: str = "",
    params: str = "",
    notes: str = ""
):
    """
    Append an experiment entry to the experiment tracking CSV.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    if experiment_id is None:
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    entry = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "cv_roc_auc": round(cv_roc_auc, 5) if cv_roc_auc is not None else None,
        "public_lb_score": round(public_lb_score, 5) if public_lb_score is not None else None,
        "features_used": features_used,
        "params": str(params),
        "notes": notes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df_entry = pd.DataFrame([entry])
    if not os.path.exists(log_path):
        df_entry.to_csv(log_path, index=False)
    else:
        df_entry.to_csv(log_path, mode="a", header=False, index=False)
    print(f"Logged experiment {experiment_id} to {log_path}")
