import numpy as np
import pandas as pd
import pickle

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def bootstrapped_roc_auc(y_true, y_scores):
    """
    function that returns bootstrapped rocaucs

    """

    n_bootstraps = 1000
    rng_seed = 42

    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:

            continue

        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    return bootstrapped_scores


def ci_bootstrapped_roc_auc(bootstrapped_scores):
    """
    function that returns the confidence interval for the bootstrapped rocaucs

    """

    confidence_interval = []
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_interval.append(confidence_lower)
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    confidence_interval.append(confidence_upper)

    return confidence_interval


def dbn_performance_metrics_tvt(
    df,
    df2,
    df3,
    dataTrainValid,
    sequence_length_dbn,
    method,
    outcome_name,
    site,
    adjusted,
    ncases,
    targets,
    track,
    onlyValid=1,
):

    """

    function that calculates and stores the performance metrics for the DBNs with train/validation/test (tvt) split

    """
    print("Evaluating Model...")
    if onlyValid == 1:
        dataValid = dataTrainValid
    else:
        dataTrain = dataTrainValid[: len(df), :]
        c = len(df) + len(df2)
        dataValid = dataTrainValid[len(df) : c, :]
        dataTest = dataTrainValid[c:, :]

    print("Saving Output...")
    save_path = "./Performance"
    valid = "/Valid"
    test = "/Test"
    roc_aucs_valid = []
    mean_bootstrapped_rocaucs_valid = []
    ci_bootstrapped_rocaucs_valid = []
    ap_scores_valid = []
    fprs_valid = []
    tprs_valid = []
    thresholds_valid = []
    roc_aucs_test = []
    mean_bootstrapped_rocaucs_test = []
    ci_bootstrapped_rocaucs_test = []
    ap_scores_test = []
    fprs_test = []
    tprs_test = []
    thresholds_test = []
    count = 0

    truths = targets
    # print(truths)

    if onlyValid == 1:
        df_truths_valid = df2[truths]
    else:
        df_truths_train = df[truths]
        df_truths_valid = df2[truths]
        df_truths_test = df3[truths]

    if onlyValid == 1:
        df_truths_valid.to_csv(
            save_path
            + valid
            + "/"
            + method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "y_true_screen_valid.csv",
            index=False,
        )
    else:
        df_truths_valid.to_csv(
            save_path
            + valid
            + "/"
            + method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "y_true_screen_valid.csv",
            index=False,
        )
        df_truths_test.to_csv(
            save_path
            + test
            + "/"
            + method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "y_true_screen_test.csv",
            index=False,
        )

    for i in range(sequence_length_dbn):
        if onlyValid == 1:
            y_true_valid = (df_truths_valid.iloc[:, i].values > 1) * 1
            y_scores_valid = dataValid[:, i]
            roc_auc_valid = roc_auc_score(y_true_valid, y_scores_valid)
            bootstrapped_rocaucs_valid = bootstrapped_roc_auc(
                y_true_valid, y_scores_valid
            )
            mean_bootstrapped_rocauc_valid = np.mean(bootstrapped_rocaucs_valid)
            ci_bootstrapped_rocauc_valid = ci_bootstrapped_roc_auc(
                bootstrapped_rocaucs_valid
            )
            ap_score_valid = average_precision_score(y_true_valid, y_scores_valid)
            fpr_valid, tpr_valid, threshold_valid = roc_curve(
                y_true_valid, y_scores_valid
            )
            count += 1
            roc_aucs_valid.append(roc_auc_valid)
            file_name_rocs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "roc_aucs_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_rocs_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % roc_aucs_valid)
            mean_bootstrapped_rocaucs_valid.append(mean_bootstrapped_rocauc_valid)
            file_name_mean_bootstrapped_rocaucs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "mean_bootstrapped_roc_aucs_valid.txt"
            )
            completeName = os.path.join(
                save_path + valid, file_name_mean_bootstrapped_rocaucs_valid
            )
            file1 = open(completeName, "w")
            file1.write("%s\n" % mean_bootstrapped_rocaucs_valid)
            ci_bootstrapped_rocaucs_valid.append(ci_bootstrapped_rocauc_valid)
            file_name_ci_bootstrapped_rocaucs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "ci_bootstrapped_roc_aucs_valid.txt"
            )
            completeName = os.path.join(
                save_path + valid, file_name_ci_bootstrapped_rocaucs_valid
            )
            file1 = open(completeName, "w")
            file1.write("%s\n" % ci_bootstrapped_rocaucs_valid)
            ap_scores_valid.append(ap_score_valid)
            file_name_ap_scores_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "ap_scores_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_ap_scores_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % ap_scores_valid)
            fprs_valid.append(fpr_valid)
            file_name_fprs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "fprs_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_fprs_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % fprs_valid)
            tprs_valid.append(tpr_valid)
            file_name_tprs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "tprs_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_tprs_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % tprs_valid)
            thresholds_valid.append(threshold_valid)
            file_name_thresholds_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "thresholds_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_thresholds_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % thresholds_valid)
        else:
            y_true_valid = (df_truths_valid.iloc[:, i].values > 1) * 1
            y_scores_valid = dataValid[:, i]
            roc_auc_valid = roc_auc_score(y_true_valid, y_scores_valid)
            bootstrapped_rocaucs_valid = bootstrapped_roc_auc(
                y_true_valid, y_scores_valid
            )
            mean_bootstrapped_rocauc_valid = np.mean(bootstrapped_rocaucs_valid)
            ci_bootstrapped_rocauc_valid = ci_bootstrapped_roc_auc(
                bootstrapped_rocaucs_valid
            )
            ap_score_valid = average_precision_score(y_true_valid, y_scores_valid)
            fpr_valid, tpr_valid, threshold_valid = roc_curve(
                y_true_valid, y_scores_valid
            )
            count += 1
            roc_aucs_valid.append(roc_auc_valid)
            file_name_rocs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "roc_aucs_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_rocs_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % roc_aucs_valid)
            mean_bootstrapped_rocaucs_valid.append(mean_bootstrapped_rocauc_valid)
            file_name_mean_bootstrapped_rocaucs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "mean_bootstrapped_roc_aucs_valid.txt"
            )
            completeName = os.path.join(
                save_path + valid, file_name_mean_bootstrapped_rocaucs_valid
            )
            file1 = open(completeName, "w")
            file1.write("%s\n" % mean_bootstrapped_rocaucs_valid)
            ci_bootstrapped_rocaucs_valid.append(ci_bootstrapped_rocauc_valid)
            file_name_ci_bootstrapped_rocaucs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "ci_bootstrapped_roc_aucs_valid.txt"
            )
            completeName = os.path.join(
                save_path + valid, file_name_ci_bootstrapped_rocaucs_valid
            )
            file1 = open(completeName, "w")
            file1.write("%s\n" % ci_bootstrapped_rocaucs_valid)
            ap_scores_valid.append(ap_score_valid)
            file_name_ap_scores_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "ap_scores_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_ap_scores_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % ap_scores_valid)
            fprs_valid.append(fpr_valid)
            file_name_fprs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "fprs_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_fprs_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % fprs_valid)
            tprs_valid.append(tpr_valid)
            file_name_tprs_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "tprs_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_tprs_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % tprs_valid)
            thresholds_valid.append(threshold_valid)
            file_name_thresholds_valid = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "thresholds_valid.txt"
            )
            completeName = os.path.join(save_path + valid, file_name_thresholds_valid)
            file1 = open(completeName, "w")
            file1.write("%s\n" % thresholds_valid)
            y_true_test = (df_truths_test.iloc[:, i].values > 1) * 1
            y_scores_test = dataTest[:, i]
            roc_auc_test = roc_auc_score(y_true_test, y_scores_test)
            bootstrapped_rocaucs_test = bootstrapped_roc_auc(y_true_test, y_scores_test)
            mean_bootstrapped_rocauc_test = np.mean(bootstrapped_rocaucs_test)
            ci_bootstrapped_rocauc_test = ci_bootstrapped_roc_auc(
                bootstrapped_rocaucs_test
            )
            ap_score_test = average_precision_score(y_true_test, y_scores_test)
            fpr_test, tpr_test, threshold_test = roc_curve(y_true_test, y_scores_test)
            count += 1
            roc_aucs_test.append(roc_auc_test)
            file_name_rocs_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "roc_aucs_test.txt"
            )
            completeName = os.path.join(save_path + test, file_name_rocs_test)
            file1 = open(completeName, "w")
            file1.write("%s\n" % roc_aucs_test)
            mean_bootstrapped_rocaucs_test.append(mean_bootstrapped_rocauc_test)
            file_name_mean_bootstrapped_rocaucs_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "mean_bootstrapped_roc_aucs_test.txt"
            )
            completeName = os.path.join(
                save_path + test, file_name_mean_bootstrapped_rocaucs_test
            )
            file1 = open(completeName, "w")
            file1.write("%s\n" % mean_bootstrapped_rocaucs_test)
            ci_bootstrapped_rocaucs_test.append(ci_bootstrapped_rocauc_test)
            file_name_ci_bootstrapped_rocaucs_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "ci_bootstrapped_roc_aucs_test.txt"
            )
            completeName = os.path.join(
                save_path + test, file_name_ci_bootstrapped_rocaucs_test
            )
            file1 = open(completeName, "w")
            file1.write("%s\n" % ci_bootstrapped_rocaucs_test)
            ap_scores_test.append(ap_score_test)
            file_name_ap_scores_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "ap_scores_test.txt"
            )
            completeName = os.path.join(save_path + test, file_name_ap_scores_test)
            file1 = open(completeName, "w")
            file1.write("%s\n" % ap_scores_test)
            fprs_test.append(fpr_test)
            file_name_fprs_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "fprs_test.txt"
            )
            completeName = os.path.join(save_path + test, file_name_fprs_test)
            file1 = open(completeName, "w")
            file1.write("%s\n" % fprs_test)
            tprs_test.append(tpr_test)
            file_name_tprs_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "tprs_test.txt"
            )
            completeName = os.path.join(save_path + test, file_name_tprs_test)
            file1 = open(completeName, "w")
            file1.write("%s\n" % tprs_test)
            thresholds_test.append(threshold_test)
            file_name_thresholds_test = (
                method
                + "_"
                + outcome_name
                + "_"
                + site
                + adjusted
                + "_"
                + track
                + "_"
                + "DBN_Model"
                + "_"
                + "thresholds_test.txt"
            )
            completeName = os.path.join(save_path + test, file_name_thresholds_test)
            file1 = open(completeName, "w")
            file1.write("%s\n" % thresholds_test)
            file1.close()

    return (
        roc_aucs_valid,
        ap_scores_valid,
        fprs_valid,
        tprs_valid,
        thresholds_valid,
        roc_aucs_test,
        ap_scores_test,
        fprs_test,
        tprs_test,
        thresholds_test,
    )


def dbn_performance_metrics_tt(
    df,
    df2,
    dataTrainValid,
    sequence_length_dbn,
    method,
    outcome_name,
    site,
    adjusted,
    ncases,
    targets,
    track,
):

    """

    function that calculates and stores the performance metrics for the DBNs with train/test (tt) split

    """
    print("Evaluating Model...")

    dataTrain = dataTrainValid[: len(df), :]
    c = len(df) + len(df2)
    dataValid = dataTrainValid[len(df) : c, :]

    print("Saving Output...")
    save_path = "./Performance"
    valid = "/Valid"

    roc_aucs_valid = []
    mean_bootstrapped_rocaucs_valid = []
    ci_bootstrapped_rocaucs_valid = []
    ap_scores_valid = []
    fprs_valid = []
    tprs_valid = []
    thresholds_valid = []

    count = 0

    truths = targets
    # print(truths)

    df_truths_train = df[truths]
    df_truths_valid = df2[truths]

    df_truths_valid.to_csv(
        save_path
        + valid
        + "/"
        + method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "DBN_Model"
        + "_"
        + "y_true_screen_valid.csv",
        index=False,
    )

    for i in range(sequence_length_dbn):
        y_true_valid = (df_truths_valid.iloc[:, i].values > 1) * 1
        y_scores_valid = dataValid[:, i]
        roc_auc_valid = roc_auc_score(y_true_valid, y_scores_valid)
        bootstrapped_rocaucs_valid = bootstrapped_roc_auc(y_true_valid, y_scores_valid)
        mean_bootstrapped_rocauc_valid = np.mean(bootstrapped_rocaucs_valid)
        ci_bootstrapped_rocauc_valid = ci_bootstrapped_roc_auc(
            bootstrapped_rocaucs_valid
        )
        ap_score_valid = average_precision_score(y_true_valid, y_scores_valid)
        fpr_valid, tpr_valid, threshold_valid = roc_curve(y_true_valid, y_scores_valid)
        count += 1
        roc_aucs_valid.append(roc_auc_valid)
        file_name_rocs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "roc_aucs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_rocs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % roc_aucs_valid)
        mean_bootstrapped_rocaucs_valid.append(mean_bootstrapped_rocauc_valid)
        file_name_mean_bootstrapped_rocaucs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "mean_bootstrapped_roc_aucs_valid.txt"
        )
        completeName = os.path.join(
            save_path + valid, file_name_mean_bootstrapped_rocaucs_valid
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % mean_bootstrapped_rocaucs_valid)
        ci_bootstrapped_rocaucs_valid.append(ci_bootstrapped_rocauc_valid)
        file_name_ci_bootstrapped_rocaucs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "ci_bootstrapped_roc_aucs_valid.txt"
        )
        completeName = os.path.join(
            save_path + valid, file_name_ci_bootstrapped_rocaucs_valid
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % ci_bootstrapped_rocaucs_valid)
        ap_scores_valid.append(ap_score_valid)
        file_name_ap_scores_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "ap_scores_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_ap_scores_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % ap_scores_valid)
        fprs_valid.append(fpr_valid)
        file_name_fprs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "fprs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_fprs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % fprs_valid)
        tprs_valid.append(tpr_valid)
        file_name_tprs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "tprs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_tprs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % tprs_valid)
        thresholds_valid.append(threshold_valid)
        file_name_thresholds_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "DBN_Model"
            + "_"
            + "thresholds_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_thresholds_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % thresholds_valid)
        file1.close()

    return roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid


def bn_performance_metrics_tvt(
    df,
    df2,
    df3,
    dataTrainValid,
    target,
    sequence_length_bn,
    method,
    outcome_name,
    site,
    adjusted,
    ncases,
    track,
    onlyValid=1,
):

    """

    function that calculates and stores the performance metrics for the BNs with train/validation/test (tvt) split

    """
    print("Evaluating Model...")
    if onlyValid == 1:
        dataValid = dataTrainValid
    else:
        dataTrain = dataTrainValid[: len(df), :]
        c = len(df) + len(df2)
        dataValid = dataTrainValid[len(df) : c, :]
        dataTest = dataTrainValid[c:, :]

    print("Saving Output...")
    save_path = "./Performance"
    valid = "/Valid"
    test = "/Test"
    roc_aucs_valid = []
    mean_bootstrapped_rocaucs_valid = []
    ci_bootstrapped_rocaucs_valid = []
    ap_scores_valid = []
    fprs_valid = []
    tprs_valid = []
    thresholds_valid = []
    roc_aucs_test = []
    mean_bootstrapped_rocaucs_test = []
    ci_bootstrapped_rocaucs_test = []
    ap_scores_test = []
    fprs_test = []
    tprs_test = []
    thresholds_test = []
    count = 0

    truths = target[0]

    if onlyValid == 1:
        df_truths_valid = df2[truths]
    else:
        df_truths_train = df[truths]
        df_truths_valid = df2[truths]
        df_truths_test = df3[truths]

    if onlyValid == 1:
        df_truths_valid.to_csv(
            save_path
            + valid
            + "/"
            + method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "y_true_screen_valid.csv",
            index=False,
        )
    else:
        df_truths_valid.to_csv(
            save_path
            + valid
            + "/"
            + method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "y_true_screen_valid.csv",
            index=False,
        )

        df_truths_test.to_csv(
            save_path
            + test
            + "/"
            + method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "y_true_screen_test.csv",
            index=False,
        )

    if onlyValid == 1:
        y_true_valid = (df_truths_valid.values > 1) * 1
        y_scores_valid = dataValid
        roc_auc_valid = roc_auc_score(y_true_valid, y_scores_valid)
        bootstrapped_rocaucs_valid = bootstrapped_roc_auc(y_true_valid, y_scores_valid)
        mean_bootstrapped_rocauc_valid = np.mean(bootstrapped_rocaucs_valid)
        ci_bootstrapped_rocauc_valid = ci_bootstrapped_roc_auc(
            bootstrapped_rocaucs_valid
        )
        ap_score_valid = average_precision_score(y_true_valid, y_scores_valid)
        fpr_valid, tpr_valid, threshold_valid = roc_curve(y_true_valid, y_scores_valid)
        count += 1
        roc_aucs_valid.append(roc_auc_valid)
        file_name_rocs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "roc_aucs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_rocs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % roc_aucs_valid)
        mean_bootstrapped_rocaucs_valid.append(mean_bootstrapped_rocauc_valid)
        file_name_mean_bootstrapped_rocaucs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "mean_bootstrapped_roc_aucs_valid.txt"
        )
        completeName = os.path.join(
            save_path + valid, file_name_mean_bootstrapped_rocaucs_valid
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % mean_bootstrapped_rocaucs_valid)
        ci_bootstrapped_rocaucs_valid.append(ci_bootstrapped_rocauc_valid)
        file_name_ci_bootstrapped_rocaucs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "ci_bootstrapped_roc_aucs_valid.txt"
        )
        completeName = os.path.join(
            save_path + valid, file_name_ci_bootstrapped_rocaucs_valid
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % ci_bootstrapped_rocaucs_valid)
        ap_scores_valid.append(ap_score_valid)
        file_name_ap_scores_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "ap_scores_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_ap_scores_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % ap_scores_valid)
        fprs_valid.append(fpr_valid)
        file_name_fprs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "fprs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_fprs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % fprs_valid)
        tprs_valid.append(tpr_valid)
        file_name_tprs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "tprs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_tprs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % tprs_valid)
        thresholds_valid.append(threshold_valid)
        file_name_thresholds_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "thresholds_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_thresholds_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % thresholds_valid)
    else:
        y_true_valid = (df_truths_valid.values > 1) * 1
        y_scores_valid = dataValid
        roc_auc_valid = roc_auc_score(y_true_valid, y_scores_valid)
        bootstrapped_rocaucs_valid = bootstrapped_roc_auc(y_true_valid, y_scores_valid)
        mean_bootstrapped_rocauc_valid = np.mean(bootstrapped_rocaucs_valid)
        ci_bootstrapped_rocauc_valid = ci_bootstrapped_roc_auc(
            bootstrapped_rocaucs_valid
        )
        ap_score_valid = average_precision_score(y_true_valid, y_scores_valid)
        fpr_valid, tpr_valid, threshold_valid = roc_curve(y_true_valid, y_scores_valid)
        count += 1
        roc_aucs_valid.append(roc_auc_valid)
        file_name_rocs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "roc_aucs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_rocs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % roc_aucs_valid)
        mean_bootstrapped_rocaucs_valid.append(mean_bootstrapped_rocauc_valid)
        file_name_mean_bootstrapped_rocaucs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "mean_bootstrapped_roc_aucs_valid.txt"
        )
        completeName = os.path.join(
            save_path + valid, file_name_mean_bootstrapped_rocaucs_valid
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % mean_bootstrapped_rocaucs_valid)
        ci_bootstrapped_rocaucs_valid.append(ci_bootstrapped_rocauc_valid)
        file_name_ci_bootstrapped_rocaucs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "ci_bootstrapped_roc_aucs_valid.txt"
        )
        completeName = os.path.join(
            save_path + valid, file_name_ci_bootstrapped_rocaucs_valid
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % ci_bootstrapped_rocaucs_valid)
        ap_scores_valid.append(ap_score_valid)
        file_name_ap_scores_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "ap_scores_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_ap_scores_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % ap_scores_valid)
        fprs_valid.append(fpr_valid)
        file_name_fprs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "fprs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_fprs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % fprs_valid)
        tprs_valid.append(tpr_valid)
        file_name_tprs_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "tprs_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_tprs_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % tprs_valid)
        thresholds_valid.append(threshold_valid)
        file_name_thresholds_valid = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "thresholds_valid.txt"
        )
        completeName = os.path.join(save_path + valid, file_name_thresholds_valid)
        file1 = open(completeName, "w")
        file1.write("%s\n" % thresholds_valid)
        y_true_test = (df_truths_test.values > 1) * 1
        y_scores_test = dataTest
        roc_auc_test = roc_auc_score(y_true_test, y_scores_test)
        bootstrapped_rocaucs_test = bootstrapped_roc_auc(y_true_test, y_scores_test)
        mean_bootstrapped_rocauc_test = np.mean(bootstrapped_rocaucs_test)
        ci_bootstrapped_rocauc_test = ci_bootstrapped_roc_auc(bootstrapped_rocaucs_test)
        ap_score_test = average_precision_score(y_true_test, y_scores_test)
        fpr_test, tpr_test, threshold_test = roc_curve(y_true_test, y_scores_test)
        count += 1
        roc_aucs_test.append(roc_auc_test)
        file_name_rocs_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "roc_aucs_test.txt"
        )
        completeName = os.path.join(save_path + test, file_name_rocs_test)
        file1 = open(completeName, "w")
        file1.write("%s\n" % roc_aucs_test)
        mean_bootstrapped_rocaucs_test.append(mean_bootstrapped_rocauc_test)
        file_name_mean_bootstrapped_rocaucs_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "mean_bootstrapped_roc_aucs_test.txt"
        )
        completeName = os.path.join(
            save_path + test, file_name_mean_bootstrapped_rocaucs_test
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % mean_bootstrapped_rocaucs_test)
        ci_bootstrapped_rocaucs_test.append(ci_bootstrapped_rocauc_test)
        file_name_ci_bootstrapped_rocaucs_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "ci_bootstrapped_roc_aucs_test.txt"
        )
        completeName = os.path.join(
            save_path + test, file_name_ci_bootstrapped_rocaucs_test
        )
        file1 = open(completeName, "w")
        file1.write("%s\n" % ci_bootstrapped_rocaucs_test)
        ap_scores_test.append(ap_score_test)
        file_name_ap_scores_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "ap_scores_test.txt"
        )
        completeName = os.path.join(save_path + test, file_name_ap_scores_test)
        file1 = open(completeName, "w")
        file1.write("%s\n" % ap_scores_test)
        fprs_test.append(fpr_test)
        file_name_fprs_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "fprs_test.txt"
        )
        completeName = os.path.join(save_path + test, file_name_fprs_test)
        file1 = open(completeName, "w")
        file1.write("%s\n" % fprs_test)
        tprs_test.append(tpr_test)
        file_name_tprs_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "tprs_test.txt"
        )
        completeName = os.path.join(save_path + test, file_name_tprs_test)
        file1 = open(completeName, "w")
        file1.write("%s\n" % tprs_test)
        thresholds_test.append(threshold_test)
        file_name_thresholds_test = (
            method
            + "_"
            + outcome_name
            + "_"
            + site
            + adjusted
            + "_"
            + track
            + "_"
            + "BN_Model"
            + "_"
            + "thresholds_test.txt"
        )
        completeName = os.path.join(save_path + test, file_name_thresholds_test)
        file1 = open(completeName, "w")
        file1.write("%s\n" % thresholds_test)
        file1.close()

    return (
        roc_aucs_valid,
        ap_scores_valid,
        fprs_valid,
        tprs_valid,
        thresholds_valid,
        roc_aucs_test,
        ap_scores_test,
        fprs_test,
        tprs_test,
        thresholds_test,
    )


def bn_performance_metrics_tt(
    df,
    df2,
    dataTrainValid,
    target,
    sequence_length_bn,
    method,
    outcome_name,
    site,
    adjusted,
    ncases,
    track,
):

    """

    function that calculates and stores the performance metrics for the BNs with train/test (tt) split

    """
    print("Evaluating Model...")

    dataTrain = dataTrainValid[: len(df), :]
    c = len(df) + len(df2)
    dataValid = dataTrainValid[len(df) : c, :]

    print("Saving Output...")
    save_path = "./Performance"
    valid = "/Valid"

    roc_aucs_valid = []
    mean_bootstrapped_rocaucs_valid = []
    ci_bootstrapped_rocaucs_valid = []
    ap_scores_valid = []
    fprs_valid = []
    tprs_valid = []
    thresholds_valid = []

    count = 0

    truths = target[0]

    df_truths_train = df[truths]
    df_truths_valid = df2[truths]

    df_truths_valid.to_csv(
        save_path
        + valid
        + "/"
        + method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "y_true_screen_valid.csv",
        index=False,
    )

    y_true_valid = (df_truths_valid.values > 1) * 1
    y_scores_valid = dataValid
    roc_auc_valid = roc_auc_score(y_true_valid, y_scores_valid)
    bootstrapped_rocaucs_valid = bootstrapped_roc_auc(y_true_valid, y_scores_valid)
    mean_bootstrapped_rocauc_valid = np.mean(bootstrapped_rocaucs_valid)
    ci_bootstrapped_rocauc_valid = ci_bootstrapped_roc_auc(bootstrapped_rocaucs_valid)
    ap_score_valid = average_precision_score(y_true_valid, y_scores_valid)
    fpr_valid, tpr_valid, threshold_valid = roc_curve(y_true_valid, y_scores_valid)
    count += 1
    roc_aucs_valid.append(roc_auc_valid)
    file_name_rocs_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "roc_aucs_valid.txt"
    )
    completeName = os.path.join(save_path + valid, file_name_rocs_valid)
    file1 = open(completeName, "w")
    file1.write("%s\n" % roc_aucs_valid)
    mean_bootstrapped_rocaucs_valid.append(mean_bootstrapped_rocauc_valid)
    file_name_mean_bootstrapped_rocaucs_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "mean_bootstrapped_roc_aucs_valid.txt"
    )
    completeName = os.path.join(
        save_path + valid, file_name_mean_bootstrapped_rocaucs_valid
    )
    file1 = open(completeName, "w")
    file1.write("%s\n" % mean_bootstrapped_rocaucs_valid)
    ci_bootstrapped_rocaucs_valid.append(ci_bootstrapped_rocauc_valid)
    file_name_ci_bootstrapped_rocaucs_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "ci_bootstrapped_roc_aucs_valid.txt"
    )
    completeName = os.path.join(
        save_path + valid, file_name_ci_bootstrapped_rocaucs_valid
    )
    file1 = open(completeName, "w")
    file1.write("%s\n" % ci_bootstrapped_rocaucs_valid)
    ap_scores_valid.append(ap_score_valid)
    file_name_ap_scores_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "ap_scores_valid.txt"
    )
    completeName = os.path.join(save_path + valid, file_name_ap_scores_valid)
    file1 = open(completeName, "w")
    file1.write("%s\n" % ap_scores_valid)
    fprs_valid.append(fpr_valid)
    file_name_fprs_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "fprs_valid.txt"
    )
    completeName = os.path.join(save_path + valid, file_name_fprs_valid)
    file1 = open(completeName, "w")
    file1.write("%s\n" % fprs_valid)
    tprs_valid.append(tpr_valid)
    file_name_tprs_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "tprs_valid.txt"
    )
    completeName = os.path.join(save_path + valid, file_name_tprs_valid)
    file1 = open(completeName, "w")
    file1.write("%s\n" % tprs_valid)
    thresholds_valid.append(threshold_valid)
    file_name_thresholds_valid = (
        method
        + "_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + "_"
        + "BN_Model"
        + "_"
        + "thresholds_valid.txt"
    )
    completeName = os.path.join(save_path + valid, file_name_thresholds_valid)
    file1 = open(completeName, "w")
    file1.write("%s\n" % thresholds_valid)
    file1.close()

    return roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid


def Pickledump(dbfile, dbfilename):
    db = open(dbfilename, "wb")
    pickle.dump(dbfile, db)
    db.close()


def Pickleload(dbfilename):
    db = open(dbfilename, "rb")
    dbfile = pickle.load(db)
    return dbfile
