import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

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


if __name__ == "__main__":
    # defining paths
    structures_path = "./RAUS/FullNetwork/"

    # model_site = "Combined"
    models_struct_directories = [
        model_name
        for model_name in glob.glob(structures_path + "*")
        if "no_race" in model_name
        and "count" not in model_name
        # and model_site in model_name
        # and "UCLA" in model_name
        # and "PSJH" in model_name
        # and "Combined" in model_name
    ]

    # loop model directories
    for models_struct_directory in tqdm(models_struct_directories):
        print(models_struct_directory)

        model_name = models_struct_directory.replace(structures_path, "")

        if "UCLA" in model_name:
            train_filename = "./Data/genie_datasets/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_train_undersampled_DBN.csv"
            filenames = [
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_train.csv",
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_valid.csv",
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_test.csv",
                "split_discetized_datasets_using_UCLA_discritizer/Prov_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_using_UCLA_discritizer.csv",
            ]
        elif "PSJH" in model_name:
            train_filename = "./Data/genie_datasets/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_Prov_train_undersampled_DBN.csv"
            filenames = [
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_Prov_train.csv",
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_Prov_valid.csv",
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_Prov_test.csv",
                "split_discetized_datasets_using_Prov_discritizer/UCLA_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_using_Prov_discritizer.csv",
            ]
        elif "Combined" in model_name:
            train_filename = "./Data/genie_datasets/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_combined_train_undersampled_DBN.csv"
            filenames = [
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_combined_train.csv",
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_combined_valid.csv",
                "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_combined_test.csv",
            ]
        else:
            pass

        filename = filenames[1]  # only using the validation set

        df_valid = pd.read_csv(
            "./Data/genie_datasets/DBN_predictions/all_var_DBN_model/"
            + filename.replace("split_discetized_datasets/", "")
            .replace("split_discetized_datasets_using_UCLA_discritizer/", "")
            .replace("split_discetized_datasets_using_Prov_discritizer/", "")
            .replace(".csv", "")
            + "_tested_on_"
            + model_name
            + "_DBN"
            + "_with_DBN_predictions.csv",
        )

        print(
            "./Data/genie_datasets/DBN_predictions/all_var_DBN_model/"
            + filename.replace("split_discetized_datasets/", "")
            .replace("split_discetized_datasets_using_UCLA_discritizer/", "")
            .replace("split_discetized_datasets_using_Prov_discritizer/", "")
            .replace(".csv", "")
            + "_tested_on_"
            + model_name
            + "_DBN"
        )

        # ## Estimate metrics over varying threshold

        # Find optimal thresholds using youden statistic (ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2749250/)

        print("Estimating optimal thresholds over time")

        thresholds = np.linspace(0, 1, 100)
        num_of_epochs = 6

        epoch_stats = dict()
        optimal_thresholds = []

        # looping over epochs
        for epoch_num in range(0, num_of_epochs):
            # epoch metrics dict
            epoch_cm_metrics, epoch_metrics = dict(), dict()

            # target variable
            target = "year" + str(1 + epoch_num) + "_reduction_40_ge"
            truth = df_valid[target].str.replace("S_", "").astype(int).values
            predictions = df_valid["predictions_year" + str(epoch_num + 1)]
            tns, fps, fns, tps = [], [], [], []
            precisions, recalls, specificities, f1_scores = [], [], [], []
            J = []  # youden statistic or any statistic to finding the optimal threshold

            # looping over thresholds
            for threshold in thresholds:
                preds = (predictions > threshold) * 1
                tn, fp, fn, tp = confusion_matrix(truth, preds).ravel()

                # confusion matrices
                tns.append(tn)
                fps.append(fp)
                fns.append(fn)
                tps.append(tp)

                precision = precision_score(truth, preds, zero_division=0)
                recall = recall_score(truth, preds, zero_division=0)
                specificity = float(tn) / float(tn + fp)
                f1Score = f1_score(truth, preds)

                # metrics
                precisions.append(precision)
                recalls.append(recall)
                specificities.append(specificity)
                f1_scores.append(f1Score)
                # Youden on ROC curve
                J.append(specificity + recall - 1)
                # # Matthews correlation coefficient
                # MCC = float(tp * tn - fp * fn) / float(
                #     np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 0.0000001)
                # )
                # J.append(MCC)

            optimal_threshold = np.round(thresholds[np.argmax(J)], 2)
            optimal_thresholds.append(optimal_threshold)
            print("Optimal threshold for epoch ", epoch_num, " is: ", optimal_threshold)

            epoch_cm_metrics["TNs"] = tns
            epoch_cm_metrics["FPs"] = fps
            epoch_cm_metrics["FNs"] = fns
            epoch_cm_metrics["TPs"] = tps

            epoch_metrics["Precision/PPV"] = precisions
            epoch_metrics["Recall/Sensitivity"] = recalls
            epoch_metrics["Specificity"] = specificities
            epoch_metrics["F1 score"] = f1_scores

            epoch_stats["epoch_" + str(epoch_num + 1)] = [epoch_cm_metrics] + [
                epoch_metrics
            ]

        # Creating a dataframe to store all results

        df_all_results = pd.DataFrame()

        for filename in filenames:
            filename = (
                "./Data/genie_datasets/DBN_predictions/all_var_DBN_model/"
                + filename.replace("split_discetized_datasets/", "")
                .replace("split_discetized_datasets_using_UCLA_discritizer/", "")
                .replace("split_discetized_datasets_using_Prov_discritizer/", "")
                .replace(".csv", "")
                + "_tested_on_"
                + model_name
                + "_DBN"
                + "_with_DBN_predictions.csv"
            )

            df_to_test = pd.read_csv(filename, low_memory=False)

            # TO-DO: evaluate by CKD flag
            # "study_entry_period_egfrckd_flag": {1: "CKD", 0: "At-Risk CKD"}
            # "study_entry_period_dxckd_flag": {1: "CKD", 0: "At-Risk CKD"},
            # "study_entry_period_albprockd_flag": {1: "CKD", 0: "At-Risk CKD"}

            # stratify metric by race and ethnicity
            # black Vs non black
            # hispnic vs non-hispanicSS
            #     "": all race groups
            #     "1B": "White Latino",
            #     "2": "Black",
            #     "Non 1B" Non-Latino
            #     "Non 2" Non Black
            #     "4": American Indian/Alaskan Native
            ckd_cats = [
                "egfrckd",
                "dxckd",
                "albprockd",
                "ckd",
                "at-risk-ckd",
            ]
            race_ethn_cats = [
                "all races",
                "S_1A",
                "Non_S_1A",
                "S_1B",
                "Non_S_1B",
                "S_2",
                "Non_S_2",
                "S_4",
                "Non_S4",
            ]

            filters = race_ethn_cats + ckd_cats

            #     print(df_to_test["demo_race_ethnicity_cat"].unique())
            #     print(df_to_test["demo_race_ethnicity_cat"].value_counts())

            for filter_cat in filters:
                print(filter_cat)
                if filter_cat == "all races":
                    df_valid = df_to_test
                elif filter_cat == "S_1A":
                    cond = df_to_test["demo_race_ethnicity_cat"] == filter_cat
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "Non_S_1A":
                    cond = df_to_test["demo_race_ethnicity_cat"] != "S_1A"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "S_1B":
                    cond = df_to_test["demo_race_ethnicity_cat"] == filter_cat
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "Non_S_1B":
                    cond = df_to_test["demo_race_ethnicity_cat"] != "S_1B"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "S_2":
                    cond = df_to_test["demo_race_ethnicity_cat"] == filter_cat
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "Non_S_2":
                    cond = df_to_test["demo_race_ethnicity_cat"] != "S_2"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "S_4":
                    cond = df_to_test["demo_race_ethnicity_cat"] == filter_cat
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "Non_S4":
                    cond = df_to_test["demo_race_ethnicity_cat"] != "S_4"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "egfrckd":
                    # when no race adjusted equation varible name is different so finding it
                    # by string match
                    col = [col for col in df_to_test.columns if "egfrckd" in col][0]
                    cond = df_to_test[col] == "S_1"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "dxckd":
                    cond = df_to_test["study_entry_period_dxckd_flag"] == "S_1"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "albprockd":
                    cond = df_to_test["study_entry_period_albprockd_flag"] == "S_1"
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "ckd":
                    # when no race adjusted equation varible name is different so finding it
                    # by string match
                    col = [col for col in df_to_test.columns if "egfrckd" in col][0]
                    cond = (
                        (df_to_test["study_entry_period_albprockd_flag"] == "S_1")
                        | (df_to_test["study_entry_period_dxckd_flag"] == "S_1")
                        | (df_to_test[col] == "S_1")
                    )
                    df_valid = df_to_test[cond].reset_index(drop=True)
                elif filter_cat == "at-risk-ckd":
                    # when no race adjusted equation varible name is different so finding it
                    # by string match
                    col = [col for col in df_to_test.columns if "egfrckd" in col][0]
                    cond = (
                        (df_to_test["study_entry_period_albprockd_flag"] != "S_1")
                        & (df_to_test["study_entry_period_dxckd_flag"] != "S_1")
                        & (df_to_test[col] != "S_1")
                    )
                    df_valid = df_to_test[cond].reset_index(drop=True)
                else:
                    pass

                cols = ["Dataset", "Race Ethnicity category", "Metric"] + [
                    "Prediction Year "
                    + str(epoch_num + 1)
                    + ", target year "
                    + str(epoch_num_targ + 1)
                    for epoch_num in range(num_of_epochs)
                    for epoch_num_targ in range(num_of_epochs)
                    if epoch_num <= epoch_num_targ
                ]

                auc_rocs, ap_aucs = [filename, filter_cat, "AUC ROC"], [
                    filename,
                    filter_cat,
                    "AP",
                ]

                auc_rocs_CIs, ap_aucs_CIs = [filename, filter_cat, "AUC ROC_CIs"], [
                    filename,
                    filter_cat,
                    "AP_CIs",
                ]

                # tns, fps, fns, tps = (
                #     [filename, filter_cat, "TNs"],
                #     [filename, filter_cat, "FPs"],
                #     [filename, filter_cat, "FNs"],
                #     [filename, filter_cat, "TPs"],
                # )
                # tns_rate, fps_rate, fns_rate, tps_rate = (
                #     [filename, filter_cat, "TNs Rate"],
                #     [filename, filter_cat, "FPs Rate"],
                #     [filename, filter_cat, "FNs Rate"],
                #     [filename, filter_cat, "TPs Rate"],
                # )
                precisions, recalls = [filename, filter_cat, "Precision/PPV"], [
                    filename,
                    filter_cat,
                    "Recall/Sensitivity",
                ]
                precisions_CIs, recalls_CIs = [filename, filter_cat, "Precision/PPV_CIs"], [
                    filename,
                    filter_cat,
                    "Recall/Sensitivity_CIs",
                ]
                specificities, f1Scores = [filename, filter_cat, "Specificity"], [
                    filename,
                    filter_cat,
                    "F1 score",
                ]
                specificities_CIs, f1Scores_CIs = [filename, filter_cat, "Specificity_CIs"], [
                    filename,
                    filter_cat,
                    "F1 score_CIs",
                ]
                brier_score_losses = [filename, filter_cat, "Brier Score loss"]
                brier_score_losses_CIs = [filename, filter_cat, "Brier Score loss_CIs"]
                opt_epoch_thresholds = [filename, filter_cat, "Optimal Threshold"]
                prevalences = [filename, filter_cat, "Outcome Prevalence"]

                filenames_epochs = []

                # looping over epochs
                for epoch_num in range(0, num_of_epochs):
                    for epoch_num_targ in range(0, num_of_epochs):
                        if epoch_num <= epoch_num_targ:
                            print(filename)
                            print(
                                "Prediction Year "
                                + str(epoch_num + 1)
                                + ", target year "
                                + str(epoch_num_targ + 1)
                            )

                            # target variable
                            target = (
                                "year" + str(1 + epoch_num_targ) + "_reduction_40_ge"
                            )

                            truth = (
                                df_valid[target]
                                .str.replace("S_", "")
                                .astype(int)
                                .values
                            )

                            # Assessing if filtering by race provides enough samples for both classes
                            if len(np.unique(truth)) > 1:
                                predictions = df_valid[
                                    "predictions_year" + str(epoch_num + 1)
                                ]

                                # define metrics
                                metrics_to_compute = ["roc_auc", 
                                                      "average_precision",
                                                      "neg_brier_score",
                                                      "precision",
                                                      "recall",
                                                      "specificity",
                                                      "f1",
                                                      ]
                                
                                # testing optimal threshold
                                threshold = optimal_thresholds[epoch_num]
                                opt_epoch_thresholds.append(threshold)
                                preds = (predictions > threshold) * 1
                                
                                # call evaluate metrics method
                                df_metrics = evaluate_bootstrap_metrics(
                                                        model=None,
                                                        X=None,
                                                        y=truth,
                                                        y_pred_prob=predictions,
                                                        n_samples=1000,
                                                        num_resamples=1000,
                                                        metrics=metrics_to_compute,
                                                        random_state=42,
                                                        threshold=threshold,
                                                        model_type="classification",
                                                        stratify=truth,
                                                        balance=False,
                                                    )

                                # get values from df add to vars, extend to add CIs

                                auc_roc = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[0],"Mean"].values[0]
                                ap_score = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[1],"Mean"].values[0]
                                auc_roc_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[0],["95% CI Lower","95% CI Upper"]].values.ravel())
                                ap_score_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[1],["95% CI Lower","95% CI Upper"]].values.ravel())
                                auc_rocs.append(auc_roc)
                                ap_aucs.append(ap_score)
                                auc_rocs_CIs.append(auc_roc_CI)
                                ap_aucs_CIs.append(ap_score_CI)

                                brier_scLoss = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[2],"Mean"].values[0]
                                brier_scLoss_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[2],["95% CI Lower","95% CI Upper"]].values.ravel())
                                brier_score_losses.append(brier_scLoss)
                                brier_score_losses_CIs.append(brier_scLoss_CI)

                                # prevalence of outcome
                                prevalence = np.round(
                                    np.sum(truth) * 100 / len(truth), 2
                                )
                                prevalences.append(prevalence)


                                precision = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[3],"Mean"].values[0]
                                recall = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[4],"Mean"].values[0]
                                specificity = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[5],"Mean"].values[0]
                                f1Score = df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[6],"Mean"].values[0]

                                precision_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[3],["95% CI Lower","95% CI Upper"]].values.ravel())
                                recall_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[4],["95% CI Lower","95% CI Upper"]].values.ravel())
                                specificity_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[5],["95% CI Lower","95% CI Upper"]].values.ravel())
                                f1Score_CI = tuple(df_metrics.loc[df_metrics["Metric"] == metrics_to_compute[6],["95% CI Lower","95% CI Upper"]].values.ravel())

                                precisions.append(precision)
                                recalls.append(recall)
                                specificities.append(specificity)
                                f1Scores.append(f1Score)

                                precisions_CIs.append(precision_CI)
                                recalls_CIs.append(recall_CI)
                                specificities_CIs.append(specificity_CI)
                                f1Scores_CIs.append(f1Score_CI)

                            else:
                                auc_rocs.append("N/A")
                                ap_aucs.append("N/A")

                                brier_score_losses.append("N/A")

                                auc_rocs_CIs.append("N/A")
                                ap_aucs_CIs.append("N/A")

                                brier_score_losses_CIs.append("N/A")

                                # testing optimal threshold
                                opt_epoch_thresholds.append("N/A")

                                prevalences.append("N/A")

                                # tns.append("N/A")
                                # fps.append("N/A")
                                # fns.append("N/A")
                                # tps.append("N/A")

                                # tns_rate.append("N/A")
                                # fps_rate.append("N/A")
                                # fns_rate.append("N/A")
                                # tps_rate.append("N/A")

                                precisions.append("N/A")
                                recalls.append("N/A")
                                specificities.append("N/A")
                                f1Scores.append("N/A")
                                precisions_CIs.append("N/A")
                                recalls_CIs.append("N/A")
                                specificities_CIs.append("N/A")
                                f1Scores_CIs.append("N/A")

                        else:
                            pass

                metrics = [
                    auc_rocs,
                    auc_rocs_CIs,
                    ap_aucs,
                    ap_aucs_CIs,
                    brier_score_losses,
                    brier_score_losses_CIs,
                    precisions,
                    precisions_CIs,
                    recalls,
                    recalls_CIs,
                    specificities,
                    specificities_CIs,
                    f1Scores,
                    f1Scores_CIs,
                    opt_epoch_thresholds,
                    prevalences,
                ]

                df_results = pd.DataFrame(data=metrics, columns=cols)

                df_all_results = df_all_results.append(df_results)

        # save results file
        df_all_results.to_csv(
            "./Data/genie_datasets/DBN_predictions/Results/CIs_results/stratified/"
            + model_name
            + "_all_datasets_results.csv",
            index=False,
        )
