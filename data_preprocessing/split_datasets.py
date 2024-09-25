import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from CONSTANTS import *


def train_val_test_split(X, y, train_size, validation_size, test_size, random_state):
    """
    Method to split a dataframe into a stratified train, valid, and test set based on user defined propotiorns.
    X: input dataframe or numpy array,
    y: input target dataframe or numpy array
    train_size: float (0,1)
    validation_size: float (0,1)
    test_size: float (0,1)
    random_state: integer to fix random seed

    Returns a set of dataframes
    X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        X,
        y,
        test_size=1 - train_size,
        train_size=train_size,
        stratify=y,
        random_state=random_state,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test,
        y_valid_test,
        test_size=(1 - train_size - validation_size) / (1 - train_size),
        train_size=(1 - train_size - test_size) / (1 - train_size),
        stratify=y_valid_test,
        random_state=random_state,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    DATA_PATH_read = (
        "./Data/" + "cure_ckd_egfr_registry_preprocessed_project_preproc_data.csv"
    )

    DATA_PATH_write = "./Data/split_datasets/"

    # read dataset
    df = pd.read_csv(DATA_PATH_read)

    # preprocessing steps
    ## 1. creating temporal binary 40 reduction columns

    reduction_cols = ["year" + str(i) + "_reduction" for i in range(1, 14)] + [
        "year" + str(i) + "_norace_reduction" for i in range(1, 14)
    ]

    reductions_40_ge_cols = [col + "_40_ge" for col in reduction_cols]

    df[reductions_40_ge_cols] = (df[reduction_cols] >= 0.4) * 1

    ## 2. Modifying coverage to include missing data instead of zeros

    # coverage_cols = [col for col in df if "coverage" in col]
    # df[coverage_cols] = df[coverage_cols].replace(0, np.nan)

    # initial target >40% , updating it to >=40% decline
    # at least one instance of >=40% decline
    df["egfr_reduction40_flag"] = (df[reductions_40_ge_cols[:13]].sum(axis=1) > 0) * 1

    ## 2. keeping only specific follow up years from 1-13 year dropping zeros(meaning people not followed)
    cond_follow_up = df["egfr_years_followed"].isin(list(np.arange(1, 14)))
    df = df[cond_follow_up].reset_index(drop=True)

    print("Saving file after applying exlcusion criteria ...")
    df.to_csv(
        DATA_PATH_read.replace(".csv", "_excl_crit_applied.csv"),
        index=False,
    )

    ## some statistics
    # dataset size = 2,250,806
    ## categorical vars stats
    # skipping patient id
    print("Generating some Table 1 statistics before splitting dataset ...")
    (
        df[old_CAT_COLS[1:] + reductions_40_ge_cols].apply(pd.Series.value_counts)
        * 100
        / len(df)
    ).to_csv("Data/reports/categorical_vars_props.csv")

    ## Continuous vars stats

    df[old_TIME_ZERO_COLS + old_CTN_ENTRY_COLS + old_CTN_COLS].describe(
        include="all"
    ).to_csv("Data/reports/continuous_vars_props.csv")

    ## Missing values stats
    (df.isnull().sum() * 100 / len(df)).to_csv("Data/reports/miss_values_perc.csv")

    # features and target
    features = [col for col in df.columns]  # if col not in ["egfr_reduction40_flag"]]
    target = "egfr_reduction40_flag"
    X = df[features]
    y = df[target]

    print("Splitting the whole dataset .....")
    #### combined
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
        X, y, train_size=0.60, validation_size=0.20, test_size=0.20, random_state=3
    )

    X_train.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_combined_train.csv",
        index=False,
    )

    X_valid.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_combined_valid.csv",
        index=False,
    )

    X_test.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_combined_test.csv",
        index=False,
    )

    print("Splitting the UCLA dataset .....")

    #### UCLA
    cond_UCLA = df["site_source_cat"] == 0
    df_UCLA = df[cond_UCLA].copy().reset_index(drop=True)

    X = df_UCLA[features]
    y = df_UCLA[target]

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
        X, y, train_size=0.60, validation_size=0.20, test_size=0.20, random_state=3
    )

    X_train.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_UCLA_train.csv",
        index=False,
    )

    X_valid.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_UCLA_valid.csv",
        index=False,
    )

    X_test.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_UCLA_test.csv",
        index=False,
    )

    print("Splitting the Providence dataset .....")

    #### Providence
    cond_Prov = df["site_source_cat"] == 1
    df_Prov = df[cond_Prov].copy().reset_index(drop=True)

    X = df_Prov[features]
    y = df_Prov[target]

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
        X, y, train_size=0.60, validation_size=0.20, test_size=0.20, random_state=3
    )

    X_train.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_Prov_train.csv",
        index=False,
    )

    X_valid.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_Prov_valid.csv",
        index=False,
    )

    X_test.to_csv(
        DATA_PATH_write
        + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_Prov_test.csv",
        index=False,
    )
