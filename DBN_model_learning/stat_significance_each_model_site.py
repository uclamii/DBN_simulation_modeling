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

from scipy import stats
import scipy.stats as st


def getOverlap(a, b):
    """Method to test overlap between 2 tuples.
    if > 0 overlap
    else no overlap
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


if __name__ == "__main__":

    # defining paths
    structures_path = "./RAUS/FullNetwork/"

    models_struct_directories = glob.glob(structures_path + "*")

    models = dict()
    models_names = []

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

        filename = filenames[2]  # only on test set to assess difference between models

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

        print("Saving ground truth and predictions ...")

        num_of_epochs = 6

        epoch_stats = dict()
        optimal_thresholds = []

        # looping over epochs
        for epoch_num in range(0, num_of_epochs):

            # epoch metrics dict
            epoch_stats["year_" + str(epoch_num + 1)] = dict()

            # target variable
            target = "year" + str(1 + epoch_num) + "_reduction_40_ge"
            truth = df_valid[target].str.replace("S_", "").astype(int).values
            predictions = df_valid["predictions_year" + str(epoch_num + 1)]

            epoch_stats["year_" + str(epoch_num + 1)]["dataset_name"] = filename
            epoch_stats["year_" + str(epoch_num + 1)]["truth"] = truth
            epoch_stats["year_" + str(epoch_num + 1)]["predictions"] = predictions

        models[model_name] = epoch_stats
        models_names.append(model_name)

    rows_models = []

    UCLA_model_names = [
        model_name for model_name in models_names if "UCLA" in model_name
    ]
    PSJH_model_names = [
        model_name for model_name in models_names if "PSJH" in model_name
    ]
    Combined_model_names = [
        model_name for model_name in models_names if "Combined" in model_name
    ]

    #################################### PSJH models

    for models_name1 in tqdm(PSJH_model_names[:-1], desc="PSJH_model_names"):
        for models_name2 in PSJH_model_names[1:]:
            if models_name1 != models_name2:
                for epoch_num in range(0, num_of_epochs):
                    x = models[models_name1]["year_" + str(epoch_num + 1)][
                        "predictions"
                    ]
                    y = models[models_name2]["year_" + str(epoch_num + 1)][
                        "predictions"
                    ]
                    res = stats.wilcoxon(
                        np.round(x, 9),
                        np.round(y, 9),
                        # zero_method="zsplit",
                    )

                    truth = models[models_name1]["year_" + str(epoch_num + 1)]["truth"]

                    ######## bootstrap stratified based on class
                    AUCs_x, AUCs_y = [], []
                    APs_x, APs_y = [], []

                    boot_iterations = 100
                    sample_frac = 0.5

                    for boot_iter in tqdm(range(boot_iterations)):
                        # for 70% random
                        df_x = pd.DataFrame(
                            data=np.array([x, truth]).T, columns=["probs", "truth"]
                        )
                        df_y = pd.DataFrame(
                            data=np.array([y, truth]).T, columns=["probs", "truth"]
                        )
                        df_x_sample = df_x.groupby("truth", group_keys=False).apply(
                            lambda x: x.sample(
                                frac=sample_frac, replace=False, random_state=boot_iter
                            )
                        )
                        df_y_sample = df_y.groupby("truth", group_keys=False).apply(
                            lambda x: x.sample(
                                frac=sample_frac,
                                replace=False,
                                random_state=boot_iter + 1,
                            )
                        )

                        sample_x, sample_y = (
                            df_x_sample["probs"].values,
                            df_y_sample["probs"].values,
                        )
                        truth_x, truth_y = (
                            df_x_sample["truth"].values,
                            df_y_sample["truth"].values,
                        )
                        AUCs_x.append(roc_auc_score(truth_x, sample_x))
                        AUCs_y.append(roc_auc_score(truth_y, sample_y))
                        APs_x.append(average_precision_score(truth_x, sample_x))
                        APs_y.append(average_precision_score(truth_y, sample_y))

                    CI_AUC_x = st.t.interval(
                        alpha=0.95,
                        df=len(AUCs_x) - 1,
                        loc=np.mean(AUCs_x),
                        scale=st.sem(AUCs_x),
                    )
                    CI_AUC_y = st.t.interval(
                        alpha=0.95,
                        df=len(AUCs_y) - 1,
                        loc=np.mean(AUCs_y),
                        scale=st.sem(AUCs_y),
                    )

                    CI_AP_x = st.t.interval(
                        alpha=0.95,
                        df=len(APs_x) - 1,
                        loc=np.mean(APs_x),
                        scale=st.sem(APs_x),
                    )
                    CI_AP_y = st.t.interval(
                        alpha=0.95,
                        df=len(APs_y) - 1,
                        loc=np.mean(APs_y),
                        scale=st.sem(APs_y),
                    )

                    overlap_AUCs_CI_xy = getOverlap(CI_AUC_x, CI_AUC_y) > 0
                    overlap_APs_CI_xy = getOverlap(CI_AP_x, CI_AP_y) > 0

                    values = [
                        models_name1,
                        models_name2,
                        models[models_name1]["year_" + str(epoch_num + 1)][
                            "dataset_name"
                        ],
                        "year_" + str(epoch_num + 1),
                        res.statistic,
                        res.pvalue,
                        res.pvalue < 0.05,
                        roc_auc_score(
                            truth,
                            x,
                        ),
                        roc_auc_score(
                            truth,
                            y,
                        ),
                        average_precision_score(
                            truth,
                            x,
                        ),
                        average_precision_score(
                            truth,
                            y,
                        ),
                        CI_AUC_x,
                        CI_AUC_y,
                        CI_AP_x,
                        CI_AP_y,
                        overlap_AUCs_CI_xy,
                        overlap_APs_CI_xy,
                    ]
                    rows_models.append(values)
            else:
                pass

    #################################### UCLA models

    for models_name1 in tqdm(UCLA_model_names[:-1], desc="UCLA_model_names"):
        for models_name2 in UCLA_model_names[1:]:
            if models_name1 != models_name2:
                for epoch_num in range(0, num_of_epochs):
                    x = models[models_name1]["year_" + str(epoch_num + 1)][
                        "predictions"
                    ]
                    y = models[models_name2]["year_" + str(epoch_num + 1)][
                        "predictions"
                    ]
                    res = stats.wilcoxon(
                        np.round(x, 9),
                        np.round(y, 9),
                        # zero_method="zsplit",
                    )

                    truth = models[models_name1]["year_" + str(epoch_num + 1)]["truth"]

                    ######## bootstrap stratified based on class
                    AUCs_x, AUCs_y = [], []
                    APs_x, APs_y = [], []

                    boot_iterations = 100
                    sample_frac = 0.5

                    for boot_iter in tqdm(range(boot_iterations)):
                        # for 70% random
                        df_x = pd.DataFrame(
                            data=np.array([x, truth]).T, columns=["probs", "truth"]
                        )
                        df_y = pd.DataFrame(
                            data=np.array([y, truth]).T, columns=["probs", "truth"]
                        )
                        df_x_sample = df_x.groupby("truth", group_keys=False).apply(
                            lambda x: x.sample(
                                frac=sample_frac, replace=False, random_state=boot_iter
                            )
                        )
                        df_y_sample = df_y.groupby("truth", group_keys=False).apply(
                            lambda x: x.sample(
                                frac=sample_frac,
                                replace=False,
                                random_state=boot_iter + 1,
                            )
                        )

                        sample_x, sample_y = (
                            df_x_sample["probs"].values,
                            df_y_sample["probs"].values,
                        )
                        truth_x, truth_y = (
                            df_x_sample["truth"].values,
                            df_y_sample["truth"].values,
                        )
                        AUCs_x.append(roc_auc_score(truth_x, sample_x))
                        AUCs_y.append(roc_auc_score(truth_y, sample_y))
                        APs_x.append(average_precision_score(truth_x, sample_x))
                        APs_y.append(average_precision_score(truth_y, sample_y))

                    CI_AUC_x = st.t.interval(
                        alpha=0.95,
                        df=len(AUCs_x) - 1,
                        loc=np.mean(AUCs_x),
                        scale=st.sem(AUCs_x),
                    )
                    CI_AUC_y = st.t.interval(
                        alpha=0.95,
                        df=len(AUCs_y) - 1,
                        loc=np.mean(AUCs_y),
                        scale=st.sem(AUCs_y),
                    )

                    CI_AP_x = st.t.interval(
                        alpha=0.95,
                        df=len(APs_x) - 1,
                        loc=np.mean(APs_x),
                        scale=st.sem(APs_x),
                    )
                    CI_AP_y = st.t.interval(
                        alpha=0.95,
                        df=len(APs_y) - 1,
                        loc=np.mean(APs_y),
                        scale=st.sem(APs_y),
                    )

                    overlap_AUCs_CI_xy = getOverlap(CI_AUC_x, CI_AUC_y) > 0
                    overlap_APs_CI_xy = getOverlap(CI_AP_x, CI_AP_y) > 0

                    values = [
                        models_name1,
                        models_name2,
                        models[models_name1]["year_" + str(epoch_num + 1)][
                            "dataset_name"
                        ],
                        "year_" + str(epoch_num + 1),
                        res.statistic,
                        res.pvalue,
                        res.pvalue < 0.05,
                        roc_auc_score(
                            truth,
                            x,
                        ),
                        roc_auc_score(
                            truth,
                            y,
                        ),
                        average_precision_score(
                            truth,
                            x,
                        ),
                        average_precision_score(
                            truth,
                            y,
                        ),
                        CI_AUC_x,
                        CI_AUC_y,
                        CI_AP_x,
                        CI_AP_y,
                        overlap_AUCs_CI_xy,
                        overlap_APs_CI_xy,
                    ]
                    rows_models.append(values)
            else:
                pass

    #################################### Combined models

    for models_name1 in tqdm(Combined_model_names[:-1], desc="Combined_model_names"):
        for models_name2 in Combined_model_names[1:]:
            if models_name1 != models_name2:
                for epoch_num in range(0, num_of_epochs):
                    x = models[models_name1]["year_" + str(epoch_num + 1)][
                        "predictions"
                    ]
                    y = models[models_name2]["year_" + str(epoch_num + 1)][
                        "predictions"
                    ]
                    res = stats.wilcoxon(
                        np.round(x, 9),
                        np.round(y, 9),
                        # zero_method="zsplit",
                    )

                    truth = models[models_name1]["year_" + str(epoch_num + 1)]["truth"]

                    ######## bootstrap stratified based on class
                    AUCs_x, AUCs_y = [], []
                    APs_x, APs_y = [], []

                    boot_iterations = 100
                    sample_frac = 0.5

                    for boot_iter in tqdm(range(boot_iterations)):
                        # for 70% random
                        df_x = pd.DataFrame(
                            data=np.array([x, truth]).T, columns=["probs", "truth"]
                        )
                        df_y = pd.DataFrame(
                            data=np.array([y, truth]).T, columns=["probs", "truth"]
                        )
                        df_x_sample = df_x.groupby("truth", group_keys=False).apply(
                            lambda x: x.sample(
                                frac=sample_frac, replace=False, random_state=boot_iter
                            )
                        )
                        df_y_sample = df_y.groupby("truth", group_keys=False).apply(
                            lambda x: x.sample(
                                frac=sample_frac,
                                replace=False,
                                random_state=boot_iter + 1,  # change samples
                            )
                        )

                        sample_x, sample_y = (
                            df_x_sample["probs"].values,
                            df_y_sample["probs"].values,
                        )
                        truth_x, truth_y = (
                            df_x_sample["truth"].values,
                            df_y_sample["truth"].values,
                        )
                        AUCs_x.append(roc_auc_score(truth_x, sample_x))
                        AUCs_y.append(roc_auc_score(truth_y, sample_y))
                        APs_x.append(average_precision_score(truth_x, sample_x))
                        APs_y.append(average_precision_score(truth_y, sample_y))

                    CI_AUC_x = st.t.interval(
                        alpha=0.95,
                        df=len(AUCs_x) - 1,
                        loc=np.mean(AUCs_x),
                        scale=st.sem(AUCs_x),
                    )
                    CI_AUC_y = st.t.interval(
                        alpha=0.95,
                        df=len(AUCs_y) - 1,
                        loc=np.mean(AUCs_y),
                        scale=st.sem(AUCs_y),
                    )

                    CI_AP_x = st.t.interval(
                        alpha=0.95,
                        df=len(APs_x) - 1,
                        loc=np.mean(APs_x),
                        scale=st.sem(APs_x),
                    )
                    CI_AP_y = st.t.interval(
                        alpha=0.95,
                        df=len(APs_y) - 1,
                        loc=np.mean(APs_y),
                        scale=st.sem(APs_y),
                    )

                    overlap_AUCs_CI_xy = getOverlap(CI_AUC_x, CI_AUC_y) > 0
                    overlap_APs_CI_xy = getOverlap(CI_AP_x, CI_AP_y) > 0

                    values = [
                        models_name1,
                        models_name2,
                        models[models_name1]["year_" + str(epoch_num + 1)][
                            "dataset_name"
                        ],
                        "year_" + str(epoch_num + 1),
                        res.statistic,
                        res.pvalue,
                        res.pvalue < 0.05,
                        roc_auc_score(
                            truth,
                            x,
                        ),
                        roc_auc_score(
                            truth,
                            y,
                        ),
                        average_precision_score(
                            truth,
                            x,
                        ),
                        average_precision_score(
                            truth,
                            y,
                        ),
                        CI_AUC_x,
                        CI_AUC_y,
                        CI_AP_x,
                        CI_AP_y,
                        overlap_AUCs_CI_xy,
                        overlap_APs_CI_xy,
                    ]
                    rows_models.append(values)
            else:
                pass

    cols = [
        "Model 1",
        "Model 2",
        "dataset name",
        "year",
        "statistic",
        "p-value",
        "p-value < 0.05",
        "Model 1 AUCROC",
        "Model 2 AUCROC",
        "Model 1 AP",
        "Model 2 AP",
        "CI_AUC_model_1",
        "CI_AUC_model_2",
        "CI_AP_model_1",
        "CI_AP_model_2",
        "CIs_AUC_overlap",
        "CIs_AP_overlap",
    ]
    df_wilc = pd.DataFrame(data=rows_models, columns=cols)

    df_wilc.to_csv(
        "Data/genie_datasets/DBN_predictions/Results/Wilcoxon_test/all_models_sign_diff.csv"
    )
