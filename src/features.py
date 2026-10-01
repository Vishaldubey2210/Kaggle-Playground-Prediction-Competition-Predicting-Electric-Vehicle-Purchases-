import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features for Electric Vehicle purchase likelihood.
    
    Features created:
    - Total_Charging_Stations: Sum of home and work charging infrastructure.
    - Charging_Stations_Diff: Difference between work and home charging access.
    - Charging_Accessibility_Index: Weighted score of home vs public charging availability.
    - Income_Per_Car: Annual income divided by number of cars owned (+1).
    - Commute_Intensity: Ratio of daily commute to standard daily commute.
    - Commute_Per_Charging_Station: Commute distance relative to total charging points.
    - EV_Readiness_Index: Composite indicator of subsidy, home charging, and low anxiety.
    """
    df = df.copy()

    # 1. Charging Infrastructure Features
    if "Charging_Stations_Near_Home" in df.columns and "Charging_Stations_Near_Work" in df.columns:
        df["Total_Charging_Stations"] = (
            df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
        )
        df["Charging_Ratio_Work_Home"] = (
            (df["Charging_Stations_Near_Work"] + 1) / (df["Charging_Stations_Near_Home"] + 1)
        )
        df["Charging_Stations_Diff"] = (
            df["Charging_Stations_Near_Work"] - df["Charging_Stations_Near_Home"]
        )

    # 2. Economic & Vehicle Ownership Ratios
    if "Annual_Income_USD" in df.columns and "Number_of_Cars_Owned" in df.columns:
        df["Income_Per_Car"] = df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"] + 1)
        df["Log_Annual_Income"] = np.log1p(np.maximum(0, df["Annual_Income_USD"]))

    # 3. Commute & Charging Balance
    if "Daily_Commute_km" in df.columns:
        df["Log_Daily_Commute"] = np.log1p(np.maximum(0, df["Daily_Commute_km"]))
        if "Total_Charging_Stations" in df.columns:
            df["Commute_Per_Station"] = df["Daily_Commute_km"] / (df["Total_Charging_Stations"] + 1)

    # 4. Binary/Ordinal conversions for composite scores
    home_chg = (df["Home_Charging_Possible"].astype(str).str.lower() == "yes").astype(int) if "Home_Charging_Possible" in df.columns else 0
    subsidy = (df["Subsidy_Available"].astype(str).str.lower() == "yes").astype(int) if "Subsidy_Available" in df.columns else 0
    
    anxiety_map = {"low": 1.0, "medium": 0.5, "high": 0.0}
    anxiety_val = (
        df["Range_Anxiety_Level"].astype(str).str.lower().map(anxiety_map).fillna(0.5)
        if "Range_Anxiety_Level" in df.columns else 0.5
    )

    env_concern = df["Environmental_Concern_Level"] if "Environmental_Concern_Level" in df.columns else 1.0

    # Composite EV Readiness Score
    df["EV_Readiness_Score"] = (
        (home_chg * 2.0) +
        (subsidy * 1.5) +
        (anxiety_val * 2.0) +
        (env_concern * 1.0)
    )

    # 5. Interaction Strings
    if "City_Type" in df.columns and "Current_Car_Type" in df.columns:
        df["City_Car_Interaction"] = df["City_Type"].astype(str) + "_" + df["Current_Car_Type"].astype(str)

    return df


def add_group_aggregations(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    group_cols: List[str] = ["City_Type", "Current_Car_Type"],
    target_num_cols: List[str] = ["Annual_Income_USD", "Daily_Commute_km"]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Calculate group-level aggregate features (mean, std) based on train set to prevent data leakage.
    """
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None

    for grp in group_cols:
        if grp not in train.columns:
            continue
        for num in target_num_cols:
            if num not in train.columns:
                continue

            agg_mean = train.groupby(grp)[num].mean().to_dict()
            agg_std = train.groupby(grp)[num].std().fillna(0).to_dict()

            train[f"{grp}_{num}_mean"] = train[grp].map(agg_mean)
            train[f"{grp}_{num}_diff_mean"] = train[num] - train[f"{grp}_{num}_mean"]

            if test is not None:
                test[f"{grp}_{num}_mean"] = test[grp].map(agg_mean)
                test[f"{grp}_{num}_diff_mean"] = test[num] - test[f"{grp}_{num}_mean"]

    return train, test


def build_feature_pipeline(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """
    Complete end-to-end feature pipeline applying domain features,
    group aggregations, and returning final feature column names.
    """
    train = add_domain_features(train_df)
    test = add_domain_features(test_df) if test_df is not None else None

    train, test = add_group_aggregations(train, test)

    ignore_cols = ["id", "Will_Buy_EV", "fold"]
    feature_cols = [c for c in train.columns if c not in ignore_cols]

    return train, test, feature_cols
