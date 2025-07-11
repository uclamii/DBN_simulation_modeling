import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import average_precision_score, get_scorer, roc_auc_score
from tqdm import tqdm

tqdm.pandas()

from sklearn.impute import KNNImputer, SimpleImputer

from random import seed, randint
import scipy.stats as st
from sklearn.metrics import get_scorer, recall_score
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)



def check_input_type(x):
    """Method to check input type pandas Series or numpy.
    Sort index if pandas for sampling efficiency.

    code from: https://github.com/uclamii/model_tuner
    """
    # if y is a numpy array cast it to a dataframe
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    elif isinstance(x, pd.Series):
        x = x.reset_index(drop=True)  # have to reset index
    elif isinstance(x, pd.DataFrame):
        x = x.reset_index(drop=True)  # have to reset index
    else:
        raise ValueError("Only numpy or panndas types supported.")
    return x


def sampling_method(
    y,
    n_samples,
    stratify,
    balance,
):
    """Method to resample a dataframe balanced, stratified, or none.

    Returns resampled y.
    code from: https://github.com/uclamii/model_tuner
    """
    if balance:
        # Perform balanced resampling by downsampling the majority classes
        # resampling the same number of samples as the minority class
        class_counts = y.value_counts()
        num_classes = len(class_counts)
        y_resample = pd.DataFrame()

        # append each sample to y_resample
        for class_label in class_counts.index:
            class_samples = y[y.values == class_label]
            resampled_class_samples = resample(
                class_samples,
                replace=True,
                n_samples=int(
                    n_samples / num_classes
                ),  # same number of samples per class always same fraction
                random_state=randint(0, 1000000),
            )
            y_resample = pd.concat([y_resample, resampled_class_samples])

        y_resample = y_resample.sort_index()  # to set indx to original shuffled state
    else:
        # Resample the target variable
        y_resample = resample(
            y,
            replace=True,
            n_samples=n_samples,
            stratify=stratify,
            random_state=randint(
                0,
                1000000,
            ),
        )
    return y_resample


def evaluate_bootstrap_metrics(
    model=None,
    X=None,
    y=None,
    y_pred_prob=None,
    n_samples=500,
    num_resamples=1000,
    metrics=["roc_auc", "f1_weighted", "average_precision"],
    random_state=42,
    threshold=0.5,
    model_type="classification",
    stratify=None,
    balance=False,
):
    """
    Evaluate various classification metrics on bootstrap samples using a
    pre-trained model or pre-computed predicted probabilities.

    Parameters:
    - model (optional): A pre-trained classifier that has a predict_proba method.
      Not required if y_pred_prob is provided.
    - X (array-like, optional): Input features. Not required if y_pred_prob is provided.
    - y (array-like): Labels.
    - y_pred_prob (array-like, optional): Pre-computed predicted probabilities.
    - n_samples (int): The number of samples in each bootstrap sample.
    - num_resamples (int): The number of resamples to generate.
    - metrics (list): List of metric names to evaluate.
    - random_state (int, optional): Random state used as the seed for each random number
      in the loop
    - threshold (float, optional): Threshold used to turn probability estimates into predictions.

    Returns:
    - DataFrame: Confidence intervals for various metrics.
    code from: https://github.com/uclamii/model_tuner
    """

    regression_metrics = [
        "explained_variance",
        "max_error",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "neg_mean_poisson_deviance",
        "neg_mean_gamma_deviance",
    ]

    # if y is a numpy array cast it to a dataframe
    y = check_input_type(y)
    if y_pred_prob is not None:
        y_pred_prob = check_input_type(y_pred_prob)

    # Set the random seed for reproducibility
    seed(random_state)

    # Ensure either model and X or y_pred_prob are provided
    if y_pred_prob is None and (model is None or X is None):
        raise ValueError("Either model and X or y_pred_prob must be provided.")

    if model_type != "regression" and any(
        metric in metrics for metric in regression_metrics
    ):
        raise ValueError(
            "If using regression metrics please specify model_type='regression'"
        )

    # Initialize a dictionary to store scores for each metric
    scores = {metric: [] for metric in metrics}

    # Perform bootstrap resampling
    for _ in tqdm(range(num_resamples)):

        y_resample = sampling_method(
            y=y,
            n_samples=n_samples,
            stratify=stratify,
            balance=balance,
        )

        # If pre-computed predicted probabilities are provided
        if y_pred_prob is not None:
            resampled_indicies = y_resample.index
            y_pred_prob_resample = y_pred_prob.iloc[resampled_indicies]

            if model_type != "regression":
                y_pred_resample = (y_pred_prob_resample >= threshold).astype(int)
            else:
                y_pred_resample = y_pred_prob_resample
        else:
            X = check_input_type(X)
            # Resample the input features and compute predictions
            resampled_indicies = y_resample.index
            X_resample = X.iloc[resampled_indicies]

            # X_resample = X_resample.values  # numpy array
            if model_type != "regression":
                y_pred_prob_resample = model.predict_proba(X_resample)[:, 1]
            else:
                y_pred_prob_resample = None
            y_pred_resample = model.predict(X_resample)

        # Calculate and store metric scores
        for metric in metrics:
            if metric == "specificity":
                # Compute specificity using recall_score with pos_label=0
                scores[metric].append(
                    recall_score(
                        y_resample,
                        y_pred_resample,
                        pos_label=0,
                    )
                )
                continue
            # Get the scorer function for the given metric
            scorer = get_scorer(metric)
            if metric in ["roc_auc", "average_precision", "brier_score"]:
                # Metrics that use probability predictions
                scores[metric].append(
                    scorer._score_func(y_resample, y_pred_prob_resample)
                )
            elif metric == "precision":
                # Precision with zero division handling
                scores[metric].append(
                    scorer._score_func(
                        y_resample,
                        y_pred_resample,
                        zero_division=0,
                    )
                )
            else:
                # Other metrics
                scores[metric].append(
                    scorer._score_func(
                        y_resample,
                        y_pred_resample,
                    )
                )
    # Initialize a dictionary to store results
    metrics_results = {
        "Metric": [],
        "Mean": [],
        "95% CI Lower": [],
        "95% CI Upper": [],
    }

    # Calculate mean and confidence intervals for each metric
    for metric in metrics:
        metric_scores = scores[metric]
        mean_score = np.mean(metric_scores)
        ci_lower, ci_upper = st.t.interval(
            0.95,
            len(metric_scores) - 1,
            loc=mean_score,
            scale=st.sem(
                metric_scores,
            ),
        )
        metrics_results["Metric"].append(metric)
        metrics_results["Mean"].append(mean_score)
        metrics_results["95% CI Lower"].append(ci_lower)
        metrics_results["95% CI Upper"].append(ci_upper)

    # Convert results to a DataFrame and return
    metrics_df = pd.DataFrame(metrics_results)
    return metrics_df

# Function to calculate transformed variables
def transform_variables(age, male, egfr, acr):
    """
    Transform the input variables as per the equation.
    age: Age of the individual in years
    male: 1 if male, 0 if female
    egfr: eGFR value
    acr: Albumin-to-creatinine ratio (ACR)
    """
    # Transformation
    age_trans = (age - 60) / 10
    male_trans = male - 0.5
    E = 85 if egfr >= 60 else 45  # eGFR threshold
    egfr_trans = (egfr - E) / 5
    ln_acr_trans = np.log(acr / 10)
    
    return age_trans, male_trans, egfr_trans, ln_acr_trans

# Risk equation function
def calculate_risk(age, male, egfr, acr, diabetes=False):
    """
    Calculate the risk using the logistic regression equation.
    diabetes: True if the patient has diabetes, False otherwise.
    """
    # Transform variables
    age_trans, male_trans, egfr_trans, ln_acr_trans = transform_variables(age, male, egfr, acr)
    
    # Coefficients for non-diabetic and diabetic patients
    coefficients_non_diabetic = {
        "age": 0.97,
        "male": 1.11,
        "egfr": 0.83,
        "ln_acr": 1.51
    }
    coefficients_diabetic = {
        "age": 0.86,
        "male": 0.89,
        "egfr": 0.91,
        "ln_acr": 1.67
    }
    
    # Select coefficients based on diabetes status
    coefficients = coefficients_diabetic if diabetes else coefficients_non_diabetic
    constant = 0  # Assumption, adjust if there's a specific constant provided.
    
    # Logistic regression equation
    log_odds = (
        constant +
        age_trans * coefficients["age"] +
        male_trans * coefficients["male"] +
        egfr_trans * coefficients["egfr"] +
        ln_acr_trans * coefficients["ln_acr"]
    )
    
    risk = np.exp(log_odds) / (1 + np.exp(log_odds))
    return risk

if __name__ == "__main__":
    cols = ["study_entry_age","demo_sex","time_zero_norace_mean","time_zero_uacr_mean","study_entry_DM_flag",]
    sites = ["UCLA","Prov","comb"]
    dataset = "test"
    missing_values_options = [False,True]

    for site in sites:
        for missing_values in missing_values_options:

            print(site)

            # Dictionary for renaming columns
            rename_columns = {
                "study_entry_age": "Age",
                "demo_sex": "Male",
                "time_zero_norace_mean": "eGFR",
                "time_zero_uacr_mean": "ACR", 
                "study_entry_DM_flag": "Diabetes",
            }
            outcome = "year3_norace_reduction_40_ge"

            ### importing data
            split_datasets_path = "./Data/split_datasets/"
            # Use glob to find all CSV files in the directory
            csv_files = glob.glob(os.path.join(split_datasets_path, "*.csv"))
            # keep only test sets
            csv_files = [csv_file for csv_file in csv_files 
                        if dataset in csv_file
                        and site in csv_file
                ]
            file_path = csv_files[0]
            print(file_path)

            df = pd.read_csv(file_path,usecols=cols+[outcome])#,nrows=1000)
            print(df.shape)
            print(df[outcome].value_counts())

            if missing_values:
                # Create a mask for rows with missing values
                mask = df.isna().any(axis=1)  # True for rows with any NaN

                # dropna
                df = df.dropna().reset_index(drop=True)
            print(df.shape)
            print(df[outcome].value_counts())
            
            # rename columns
            df.rename(columns=rename_columns,inplace=True)

            # create male
            df["Male"] = (df["Male"] == 0)*1
            df["Diabetes"] = df["Diabetes"] == 1 # boolean to choose diab and non diab model
            
            # Ensure the necessary columns are present
            required_columns = ["Age", "Male", "eGFR", "ACR", "Diabetes"]
            numerical_columns = ["Age", "eGFR", "ACR"]
            categorical_columns = ["Male", "Diabetes"]
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"The input DataFrame must contain the following columns: {required_columns}")
            
            # Impute numerical columns with mean
            # num_imputer = SimpleImputer(strategy="mean")
            num_imputer = KNNImputer(n_neighbors=2, weights="uniform")
            df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

            # Impute categorical columns with most frequent value
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

            # Apply the model and add predictions
            df["Risk"] = df.progress_apply(
                lambda row: calculate_risk(
                    age=row["Age"],
                    male=row["Male"],
                    egfr=row["eGFR"],
                    acr=row["ACR"],
                    diabetes=row["Diabetes"]
                ),
                axis=1
            )

            y_true = df[outcome].values
            y_pred = df["Risk"].values

            auc_roc = roc_auc_score(y_true=y_true, y_score=y_pred)
            avg_precision = average_precision_score(y_true=y_true, y_score=y_pred)
            print(f"Overall AUC-ROC: {auc_roc:.4f}")
            print(f"Overall Average Precision: {avg_precision:.4f}")

            # define metrics
            metrics_to_compute = ["roc_auc", 
                                    "average_precision",
                                    # "neg_brier_score",
                                    # "precision",
                                    # "recall",
                                    # "specificity",
                                    # "f1",
                                    ]
            
            # call evaluate metrics method
            df_metrics = evaluate_bootstrap_metrics(
                                    model=None,
                                    X=None,
                                    y=y_true,
                                    y_pred_prob=y_pred,
                                    n_samples=1000,
                                    num_resamples=1000,
                                    metrics=metrics_to_compute,
                                    random_state=42,
                                    threshold=0.5,
                                    model_type="classification",
                                    stratify=y_true,
                                    balance=True,
                                )
            
            print(df_metrics)

            # print(df["Risk"])
            
            # # Save or display results
            # output_file = "predictions.csv"
            # df.to_csv(output_file, index=False)
            # print(f"Predictions saved to {output_file}")

            # DBNs comparison
            
            ### importing data
            split_datasets_path = "./Data/genie_datasets/DBN_predictions/all_var_DBN_model"
            # Use glob to find all CSV files in the directory
            csv_files = glob.glob(os.path.join(split_datasets_path, "*.csv"))
            # keep only test sets
            csv_files = [
                csv_file for csv_file in csv_files 
                if f"{dataset}_" in csv_file 
                and "no_race_" in csv_file 
                and "counts" not in csv_file
                and site in csv_file
                ]
            file_path = csv_files[0]
            print(file_path)

            df = pd.read_csv(file_path,)#,nrows=1000)
            df["year3_reduction_40_ge"] = (df["year3_reduction_40_ge"] == "S_1")*1

            if missing_values:
                # use mask to match rows
                df = df[mask].reset_index(drop=True)

            y_true = df["year3_reduction_40_ge"].values
            y_pred = df["predictions_year1"].values

            auc_roc = roc_auc_score(y_true=y_true, y_score=y_pred)
            avg_precision = average_precision_score(y_true=y_true, y_score=y_pred)
            print(f"Overall AUC-ROC: {auc_roc:.4f}")
            print(f"Overall Average Precision: {avg_precision:.4f}")

            # define metrics
            metrics_to_compute = ["roc_auc", 
                                    "average_precision",
                                    # "neg_brier_score",
                                    # "precision",
                                    # "recall",
                                    # "specificity",
                                    # "f1",
                                    ]
            
            # call evaluate metrics method
            df_metrics = evaluate_bootstrap_metrics(
                                    model=None,
                                    X=None,
                                    y=y_true,
                                    y_pred_prob=y_pred,
                                    n_samples=1000,
                                    num_resamples=1000,
                                    metrics=metrics_to_compute,
                                    random_state=42,
                                    threshold=0.5,
                                    model_type="classification",
                                    stratify=y_true,
                                    balance=True,
                                )
            
            print(df_metrics)
