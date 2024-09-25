import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


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

                tns, fps, fns, tps = (
                    [filename, filter_cat, "TNs"],
                    [filename, filter_cat, "FPs"],
                    [filename, filter_cat, "FNs"],
                    [filename, filter_cat, "TPs"],
                )
                tns_rate, fps_rate, fns_rate, tps_rate = (
                    [filename, filter_cat, "TNs Rate"],
                    [filename, filter_cat, "FPs Rate"],
                    [filename, filter_cat, "FNs Rate"],
                    [filename, filter_cat, "TPs Rate"],
                )
                precisions, recalls = [filename, filter_cat, "Precision/PPV"], [
                    filename,
                    filter_cat,
                    "Recall/Sensitivity",
                ]
                specificities, f1Scores = [filename, filter_cat, "Specificity"], [
                    filename,
                    filter_cat,
                    "F1 score",
                ]
                brier_score_losses = [filename, filter_cat, "Brier Score loss"]
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

                                auc_roc = roc_auc_score(
                                    truth, predictions, labels=[0, 1]
                                )
                                ap_score = average_precision_score(
                                    truth, predictions, pos_label=1
                                )
                                auc_rocs.append(auc_roc)
                                ap_aucs.append(ap_score)

                                brier_scLoss = brier_score_loss(truth, predictions)
                                brier_score_losses.append(brier_scLoss)

                                # testing optimal threshold
                                threshold = optimal_thresholds[epoch_num]
                                opt_epoch_thresholds.append(threshold)
                                preds = (predictions > threshold) * 1

                                # prevalence of outcome
                                prevalence = np.round(
                                    np.sum(truth) * 100 / len(truth), 2
                                )
                                prevalences.append(prevalence)

                                tn, fp, fn, tp = confusion_matrix(truth, preds).ravel()
                                tns.append(tn)
                                fps.append(fp)
                                fns.append(fn)
                                tps.append(tp)

                                tns_rate.append(tn / (tn + fp))
                                fps_rate.append(fp / (tn + fp))
                                fns_rate.append(fn / (fn + tp))
                                tps_rate.append(tp / (fn + tp))

                                precision = precision_score(truth, preds)
                                recall = recall_score(truth, preds)
                                specificity = float(tn) / float(tn + fp)
                                f1Score = f1_score(truth, preds)

                                precisions.append(precision)
                                recalls.append(recall)
                                specificities.append(specificity)
                                f1Scores.append(f1Score)

                            else:
                                auc_rocs.append("N/A")
                                ap_aucs.append("N/A")

                                brier_score_losses.append("N/A")

                                # testing optimal threshold
                                opt_epoch_thresholds.append("N/A")

                                prevalences.append("N/A")

                                tns.append("N/A")
                                fps.append("N/A")
                                fns.append("N/A")
                                tps.append("N/A")

                                tns_rate.append("N/A")
                                fps_rate.append("N/A")
                                fns_rate.append("N/A")
                                tps_rate.append("N/A")

                                precisions.append("N/A")
                                recalls.append("N/A")
                                specificities.append("N/A")
                                f1Scores.append("N/A")

                        else:
                            pass

                metrics = [
                    auc_rocs,
                    ap_aucs,
                    brier_score_losses,
                    tns,
                    fps,
                    fns,
                    tps,
                    tns_rate,
                    fps_rate,
                    fns_rate,
                    tps_rate,
                    precisions,
                    recalls,
                    specificities,
                    f1Scores,
                    opt_epoch_thresholds,
                    prevalences,
                ]

                df_results = pd.DataFrame(data=metrics, columns=cols)

                df_all_results = df_all_results.append(df_results)

        # save results file
        df_all_results.to_csv(
            "./Data/genie_datasets/DBN_predictions/Results/"
            + model_name
            + "_all_datasets_results.csv",
            index=False,
        )
