import glob

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from model_tuner import Model
import model_tuner

if __name__ == "__main__":
    print(f"Model Tuner version: {model_tuner.__version__}")
    print(f"Model Tuner authors: {model_tuner.__author__}\n")

    outcomes = [
        "year1_reduction_40_ge",
        "year2_reduction_40_ge",
        "year3_reduction_40_ge",
        "year4_reduction_40_ge",
        "year5_reduction_40_ge",
        "year6_reduction_40_ge",
    ]

    temp_vars = [
        "year1_reduction_40_ge",
        "year1_norace_mean",
        "year1_uacr_mean",
        "year1_pp_mean",
        "year1_ppi_coverage",
        "year1_dbp_mean",
        "year1_aceiarb_coverage",
        "year1_sbp_mean",
        "year1_hba1c_mean",
        "year1_upcr_mean",
        "year1_map_mean",
        "year1_nsaid_coverage",
    ]

    # 1. Define the mapping
    tz_to_y1_map = {
        "time_zero_norace_mean": "year1_norace_mean",
        "time_zero_uacr_mean": "year1_uacr_mean",
        "time_zero_pp_mean": "year1_pp_mean",
        "time_zero_ppi_coverage": "year1_ppi_coverage",
        "time_zero_aceiarb_coverage": "year1_aceiarb_coverage",
        "time_zero_sbp_mean": "year1_sbp_mean",
        "time_zero_dbp_mean": "year1_dbp_mean",
        "time_zero_hba1c_mean": "year1_hba1c_mean",
        "time_zero_map_mean": "year1_map_mean",
        "time_zero_upcr_mean": "year1_upcr_mean",
        "time_zero_nsaid_coverage": "year1_nsaid_coverage",
    }

    # 2. Your original list
    original_vars = [
        "time_zero_norace_mean",
        "study_entry_period_egfrckd_norace_flag",
        "study_entry_period_dxckd_flag",
        "study_entry_age",
        "time_zero_uacr_mean",
        "study_entry_DM_flag",
        "time_zero_pp_mean",
        "time_zero_aceiarb_coverage",
        "time_zero_sbp_mean",
        "time_zero_ppi_coverage",
        "study_entry_aceiarb_flag",
        "study_entry_period_albprockd_flag",
        "time_zero_dbp_mean",
        "time_zero_hba1c_mean",
        "time_zero_map_mean",
        "demo_race_ethnicity_cat",
        "study_entry_ppi_flag",
        "time_zero_upcr_mean",
        "study_entry_HTN_flag",
        "time_zero_nsaid_coverage",
        "ruca_4_class",
        "study_entry_nsaid_flag",
        "demo_sex",
        "study_entry_PDM_flag",
    ]

    # 3. Replace the variables
    updated_vars = [tz_to_y1_map.get(var, var) for var in original_vars]
    ### importing data
    sites = ["UCLA", "Prov", "comb"]
    dataset = "test"
    site = sites[0]
    split_datasets_path = "./Data/split_datasets/"

    file_path = "Data/cure_ckd_egfr_registry_preprocessed_project_preproc_data_excl_crit_applied.csv"
    print(file_path)

    df = pd.read_csv(file_path)  # , nrows=10000)  #  usecols=original_vars + outcomes,

    X = df[original_vars]
    y = df[outcomes[0]]

    # Define Feature Groups
    categorical_features = [
        "demo_race_ethnicity_cat",
        "ruca_4_class",
        "demo_sex",
    ]
    numerical_features = X.select_dtypes(np.number).columns.to_list()

    # 2. Define Preprocessing Pipeline
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # 3. Comprehensive Definition Dictionary
    model_definitions = {
        "xgb": {
            "clc": XGBClassifier(objective="binary:logistic", random_state=222),
            "estimator_name": "xgb",
            "tuned_parameters": {
                "xgb__max_depth": [3, 10, 20],
                "xgb__learning_rate": [0.01, 0.1],
                "xgb__n_estimators": [1000],
                "xgb__early_stopping_rounds": [50],
                "xgb__eval_metric": ["logloss"],
            },
            "randomized_grid": True,
            "n_iter": 5,
            "early": True,
        },
        "cat": {
            "clc": CatBoostClassifier(silent=True, random_state=222),
            "estimator_name": "cat",
            "tuned_parameters": {
                "cat__depth": [4, 6, 10],
                "cat__n_estimators": [1000],
                "cat__early_stopping_rounds": [50],
            },
            "randomized_grid": True,
            "n_iter": 5,
            "early": True,
        },
        "rf": {
            "clc": RandomForestClassifier(random_state=222),
            "estimator_name": "rf",
            "tuned_parameters": {
                "rf__n_estimators": [100, 300],
                "rf__max_depth": [10, 20, None],
            },
            "randomized_grid": True,
            "n_iter": 5,
            "early": False,
        },
        "lr": {
            "clc": LogisticRegression(max_iter=1000, random_state=222),
            "estimator_name": "lr",
            "tuned_parameters": {
                "lr__C": [0.1, 1.0, 10.0],
            },
            "randomized_grid": False,
            "n_iter": 5,
            "early": False,
        },
    }

    # 4. Multi-Model Execution Loop
    # final_comparison = {}
    # final_comparison_CIs = {}
    temporal_results = []

    for m_id, m_def in model_definitions.items():
        print("-" * 80)
        print(f"STARTING: {m_id.upper()}")
        print("-" * 80)

        # Initialize the Model Tuner instance
        mt_instance = Model(
            name=f"Adult_Comparison_{m_id}",
            estimator_name=m_def["estimator_name"],
            calibrate=True,
            estimator=m_def["clc"],
            model_type="classification",
            kfold=False,
            pipeline_steps=[("ColumnTransformer", preprocessor)],
            stratify_y=True,
            stratify_cols=None,
            n_iter=m_def["n_iter"],
            grid=m_def["tuned_parameters"],
            randomized_grid=m_def["randomized_grid"],
            boost_early=m_def["early"],
            scoring=["roc_auc"],
            random_state=3,
            n_jobs=1,
        )

        # Step 1: Tune
        mt_instance.grid_search_param_tuning(X, y, f1_beta_tune=True)

        # Step 2: Retrieve Splits
        X_train, y_train = mt_instance.get_train_data(X, y)
        X_test, y_test = mt_instance.get_test_data(X, y)
        X_valid, y_valid = mt_instance.get_valid_data(X, y)

        # Step 3: Fit (With or without validation data for early stopping)
        if m_def["early"]:
            mt_instance.fit(X_train, y_train, validation_data=[X_valid, y_valid])
        else:
            mt_instance.fit(X_train, y_train)

        # Step 4: Calibrate
        mt_instance.calibrateModel(X, y, f1_beta_tune=True)

        # Step 5: Output and Store Metrics
        # metrics = mt_instance.return_metrics(X_test, y_test, model_metrics=True)
        # metrics_CI = mt_instance.return_bootstrap_metrics(
        #     X_test,
        #     y_test,
        #     metrics=["roc_auc", "average_precision"],
        #     n_samples=1000,
        # )
        # final_comparison[m_id] = metrics
        # final_comparison_CIs[m_id] = metrics_CI

        # --- TEMPORAL SUBSEQUENT YEAR TESTING ---
        print("\n" + "=" * 80)
        print("RUNNING TEMPORAL SUBSEQUENT YEAR TESTING")
        print("=" * 80)

        # We iterate through year pairs: (Year 1 vars -> Year 2 outcome), (Year 2 vars -> Year 3 outcome), etc.
        # Note: You'll need to ensure your 'df' contains the year2, year3... versions of the vars.
        for i in range(0, len(outcomes)):
            current_outcome = outcomes[i]  # e.g., year2_reduction_40_ge
            prev_outcome_idx = i - 1  # e.g., year1_reduction_40_ge

            print(f"Testing Model on: {current_outcome} using Year {i} data...")

            if i > 0:
                # Create a mapping for the specific year shift
                # This shifts time_zero -> year(i)
                year_shift_map = {
                    tz: tz.replace("time_zero", f"year{i}")
                    for tz in original_vars
                    if "time_zero" in tz
                }

                # Define the shifted feature list for this specific test
                shifted_X_vars = [year_shift_map.get(var, var) for var in original_vars]
            else:
                shifted_X_vars = original_vars
            # Important: Ensure the shifted columns exist in your dataframe
            # If they don't follow a strict 'yearN' naming convention, use your provided 'tz_to_y1_map' logic

            # Extract only the test rows from the original dataframe to avoid leakage
            # (Assuming you saved the test indices during training)
            X_temporal = df.loc[X_test.index, shifted_X_vars]
            y_temporal = df.loc[X_test.index, current_outcome]

            # Rename columns back to original_vars so the model pipeline recognizes them
            X_temporal.columns = original_vars

            # Generate Bootstrap Metrics for the future year
            # mt_instance refers to the specific model from the previous loop
            temp_metrics_CI = mt_instance.return_bootstrap_metrics(
                X_temporal,
                y_temporal,
                metrics=["roc_auc", "average_precision"],
                n_samples=1000,
            )

            temp_metrics_CI["Model"] = m_id
            temp_metrics_CI["Target_Year"] = current_outcome
            temporal_results.append(temp_metrics_CI)

    # Combine Temporal Results
    if temporal_results:
        temporal_summary_df = pd.concat(temporal_results).reset_index(drop=True)
        print(temporal_summary_df)
        temporal_summary_df.to_csv(
            "Data/ml_models_results/temporal_comp.csv", index=False
        )

    # # Final Summary Table
    # print("\n" + "=" * 80)
    # print("FINAL RESULTS SUMMARY")
    # print("=" * 80)
    # summary_df = pd.DataFrame(final_comparison).T
    # # 1. Combine the dictionary of DataFrames
    # summary_df_CIs = pd.concat(
    #     final_comparison_CIs, names=["Model", "Index"]
    # ).reset_index(level=0)

    # # 2. Cleanup: Remove the old numeric index if it's not needed
    # summary_df_CIs = summary_df_CIs.reset_index(drop=True)
    # print(summary_df)
    # print(summary_df_CIs)
