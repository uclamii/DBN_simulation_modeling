import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from BN_pysmile import *
from scipy import stats

import itertools


if __name__ == "__main__":
    ## READ results

    ##########################################################

    # Selecting
    #  each years predictions to find the best model
    # for each site and filter

    num_epochs = 6
    metrics = ["AP", "AUC ROC"]
    results = []

    sites = ["UCLA", "PSJH", "Combined"]

    site = sites[1]

    # saving best model csv file
    df_best_models = pd.read_csv(
        "./Data/genie_datasets/DBN_predictions/Results/best_models_per_filter_"
        + site
        + ".csv"
    )

    ##########################################################

    # Selecting
    #  each years predictions to find the best model
    # for each site and filter

    results_path = "./Data/genie_datasets/DBN_predictions/Results/"

    sites_models_results_filepaths = [
        model_path for model_path in glob.glob(results_path + "*") if site in model_path
    ]

    print(sites_models_results_filepaths)

    model_names_filter_groups = df_best_models[df_best_models["Metric"] == "AP"][
        ["Model_site", "Filter"]
    ].reset_index(drop=True)

    print(model_names_filter_groups)

    ##########################################################

    # assigning to another variables
    original_sites_models_results_filepaths = sites_models_results_filepaths

    # looping each model site, and filter combination

    for model_comb_indx in tqdm(range(model_names_filter_groups.shape[0])):
        filter_groups = [model_names_filter_groups.iloc[model_comb_indx, 1]]
        model_names = [model_names_filter_groups.iloc[model_comb_indx, 0]]

        model_name = model_names[0]

        print("##########################################################")

        print("For model " + model_name + " ... generating results ...")

        # focusing on "all" race results
        sites_models_results_filepaths = [
            model_path
            for model_path in original_sites_models_results_filepaths
            if model_name + "all" in model_path
        ]

        # looping for each model the results, and keeping only the results
        # on test data
        for models_results_filepath in sites_models_results_filepaths:
            model_name = models_results_filepath.replace(results_path, "").replace(
                "all_datasets_results.csv", ""
            )

            df_results = pd.read_csv(models_results_filepath)
            df_results = df_results[
                df_results["Dataset"].str.contains("_test_tested_on")
            ].reset_index(drop=True)

        ###################################### loading threshold

        threshold = df_results.loc[
            (df_results["Metric"].str.contains("Optimal"))
            & (df_results["Race Ethnicity category"] == filter_groups[0]),
            "Prediction Year 1, target year 1",
        ].values[0]

        ###################################### loading model

        #### Creating network and saving as xdsl file

        cure_ckd_bn = CURE_CKD_BayesianNetwork()
        # load file
        cure_ckd_bn.load_BN(
            filename="./DBN_model_learning/models/DBNs/"
            + model_name
            + "CURE_CKD_DBN.xdsl"
        )

        ###################################### loading patient predictions

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

        filename = filenames[1]

        df_preds = pd.read_csv(
            "./Data/genie_datasets/DBN_predictions/all_var_DBN_model/"
            + filename.replace("split_discetized_datasets/", "")
            .replace("split_discetized_datasets_using_UCLA_discritizer/", "")
            .replace("split_discetized_datasets_using_Prov_discritizer/", "")
            .replace(".csv", "")
            + "_tested_on_"
            + model_name
            + "DBN"
            + "_with_DBN_predictions.csv",
            # nrows=20000,
        )

        print(
            "./Data/genie_datasets/DBN_predictions/all_var_DBN_model/"
            + filename.replace("split_discetized_datasets/", "")
            .replace("split_discetized_datasets_using_UCLA_discritizer/", "")
            .replace("split_discetized_datasets_using_Prov_discritizer/", "")
            .replace(".csv", "")
            + "_tested_on_"
            + model_name
            + "DBN"
        )

        print(df_preds.columns)
        ###################################### find N closest patients with predictions to the optimal threshold
        # N = 5500
        FNs, TPs, FPs, TNs = [], [], [], []
        Power = 0
        Powers = []
        ranges = np.arange(10, 100000, 100)

        for N in tqdm(ranges):

            df_preds_N_closest = df_preds.iloc[
                (df_preds["predictions_year1"] - threshold).abs().argsort()[:N]
            ].reset_index(drop=True)

            FN = np.sum(
                (df_preds_N_closest["predictions_year1"] <= threshold)
                & (df_preds_N_closest["year1_reduction_40_ge"] == "S_1")
            )
            FNs.append(FN)

            TP = np.sum(
                (df_preds_N_closest["predictions_year1"] > threshold)
                & (df_preds_N_closest["year1_reduction_40_ge"] == "S_1")
            )
            TPs.append(TP)

            TN = np.sum(
                (df_preds_N_closest["predictions_year1"] <= threshold)
                & (df_preds_N_closest["year1_reduction_40_ge"] == "S_0")
            )
            TNs.append(TN)

            FP = np.sum(
                (df_preds_N_closest["predictions_year1"] > threshold)
                & (df_preds_N_closest["year1_reduction_40_ge"] == "S_0")
            )
            FPs.append(FP)

            Power = 1 - FN / (FN + TP)
            Powers.append(Power)

        print(
            "FN",
            FN,
        )

        print(
            "TP",
            TP,
        )

        print(
            "TN",
            TN,
        )

        print(
            "FP",
            FP,
        )

        print("Power: ", np.nanmax(Powers))
        print("Ranges index: ", np.nanargmax(Powers))
        print(
            "FN, TP, FP, TN: ",
            FNs[np.nanargmax(Powers)],
            TPs[np.nanargmax(Powers)],
            FPs[np.nanargmax(Powers)],
            TNs[np.nanargmax(Powers)],
        )
        print("Sample size: ", ranges[np.nanargmax(Powers)])

        N = ranges[np.nanargmax(Powers)]

        df_preds_N_closest = df_preds.iloc[
            (df_preds["predictions_year1"] - threshold).abs().argsort()[:N]
        ].reset_index(drop=True)

        ###################################### create sensitivity analysis module

        structures_path = "RAUS/FullNetwork/"

        models_struct_directories = glob.glob(structures_path + "*")

        if "race" in model_name and "counts" not in model_name:
            models_struct_directory = [
                directory
                for directory in models_struct_directories
                if model_name[:-1] in directory
            ]
        elif "race" not in model_name and "counts" not in model_name:
            models_struct_directory = [
                directory
                for directory in models_struct_directories
                if model_name[:-1] in directory
                and "race" not in directory
                and "counts" not in directory
            ]
        elif "race" not in model_name and "counts" in model_name:
            models_struct_directory = [
                directory
                for directory in models_struct_directories
                if model_name[:-1] in directory
                and "race" not in directory
                and "counts" in directory
            ]
        elif "race" in model_name and "counts" in model_name:
            models_struct_directory = [
                directory
                for directory in models_struct_directories
                if model_name[:-1] in directory
                and "race" in directory
                and "counts" in directory
            ]

        assert len(models_struct_directory) == 1
        models_struct_directory = models_struct_directory[0]

        models_struct_directory_structs = glob.glob(models_struct_directory + "/*.csv")
        model_inter_struct = [
            struct for struct in models_struct_directory_structs if "Inter" in struct
        ][0]
        model_IC_struct = [
            struct
            for struct in models_struct_directory_structs
            if "Initial_Condition" in struct
        ][0]
        model_intraDBN_struct = [
            struct
            for struct in models_struct_directory_structs
            if "Intra_DBN" in struct
        ][0]
        model_Contemporals_struct = [
            struct
            for struct in models_struct_directory_structs
            if "Contemporals" in struct
        ][0]

        #################### inter structure
        df_inter = pd.read_csv(
            model_inter_struct,
            index_col=0,
        )

        inter_indexes = [val.replace("year0", "time_zero") for val in df_inter.index]
        df_inter.index = inter_indexes

        cols_df_inter = [col.replace("year0", "time_zero") for col in df_inter.columns]
        df_inter.columns = cols_df_inter

        # all temporal columns
        cols_inter = [
            col[:4] + str(int(col[4]) + num) + col[5:]
            for col in list(df_inter.index)
            for num in range(num_epochs)
        ]

        #################### time zero to year 1 intra structure
        # initial conditions

        df_intra_year0_1 = pd.read_csv(
            model_IC_struct,
            index_col=0,
        )

        # removing year1 variables from index as we are only interested
        # between connections of year 0 and year 1 vars
        # keeping year1_reduction_40_ge however not present in year 0 to year 1
        drop_indeces = [
            indx
            for indx in df_intra_year0_1.index
            if ("year1" in indx and "reduction_40" not in indx) or "year2" in indx
        ]
        df_intra_year0_1.drop(drop_indeces, inplace=True)

        # removing year0 variables from columns as we only need forward edges from year 0 to year 1
        # not year 2 for year 0 vars to year 2
        keep_cols = [col for col in df_intra_year0_1.columns if "year1" in col]
        df_intra_year0_1 = df_intra_year0_1[keep_cols]

        intra_year0_1_indexes = [
            val.replace("year0", "time_zero") for val in df_intra_year0_1.index
        ]
        df_intra_year0_1.index = intra_year0_1_indexes

        cols_df_intra_year0_1 = [
            col.replace("year0", "time_zero") for col in df_intra_year0_1.columns
        ]
        df_intra_year0_1.columns = cols_df_intra_year0_1

        cols_df_intra_year0_1 = list(df_intra_year0_1.columns)

        # it must not equal since we removed time edges that should not be there
        assert list(df_intra_year0_1.index) != list(df_intra_year0_1.columns)

        #################### year 1 intra structure
        # intraDBN

        df_intra_year1 = pd.read_csv(
            model_intraDBN_struct,
            index_col=0,
        )

        intra_year1_indexes = [
            val.replace("year0", "time_zero") for val in df_intra_year1.index
        ]
        df_intra_year1.index = intra_year1_indexes

        cols_df_intra_year1 = [
            col.replace("year0", "time_zero") for col in df_intra_year1.columns
        ]
        df_intra_year1.columns = cols_df_intra_year1

        cols_df_intra_year1 = list(df_intra_year1.columns)

        assert list(df_intra_year1.index) == list(df_intra_year1.columns)

        cols_df_intra_year_1_4 = [
            col[:4] + str(int(col[4]) + num) + col[5:]
            for col in list(df_intra_year1.index)
            for num in range(num_epochs)
        ]

        #################### static to time zero intra structure
        # contemporals

        df_intra_static_year0 = pd.read_csv(
            model_Contemporals_struct,
            index_col=0,
        )

        intra_static_year0_indexes = [
            val.replace("year0", "time_zero") for val in df_intra_static_year0.index
        ]
        df_intra_static_year0.index = intra_static_year0_indexes

        cols_intra_static_year0 = [
            col.replace("year0", "time_zero") for col in df_intra_static_year0.columns
        ]
        df_intra_static_year0.columns = cols_intra_static_year0

        # filtering out count variables
        intra_static_year0_indexes_bool = [
            True if "site" not in val else False for val in df_intra_static_year0.index
        ]
        cols_intra_static_year0_filt = [
            col.replace("year0", "time_zero")
            for col in df_intra_static_year0.columns
            if "site" not in col
        ]

        df_intra_static_year0 = df_intra_static_year0.loc[
            intra_static_year0_indexes_bool, cols_intra_static_year0_filt
        ]

        cols_df_intra_static_year0 = list(df_intra_static_year0.columns)

        assert list(df_intra_static_year0.index) == list(df_intra_static_year0.columns)

        ################################################################################################

        epoch_num = 0
        # target variable
        target = "year" + str(1 + epoch_num) + "_reduction_40_ge"
        print(target)

        if epoch_num == 0:
            all_cols_to_be_tested = cols_df_intra_static_year0
        else:
            all_cols_to_be_tested = cols_df_intra_static_year0 + [
                col[:4] + str(int(col[4]) + num) + col[5:]
                for col in list(df_intra_year1.index)
                for num in range(0, epoch_num)
            ]
        # testing columns
        cols_to_test = [col for col in all_cols_to_be_tested if target != col]

        # variables to assess
        vars_to_assess = [
            "study_entry_DM_flag",
            "study_entry_period_egfrckd_flag",
            "time_zero_ipv_count",
            "study_entry_period_dxckd_flag",
            "time_zero_mean",
            "study_entry_age",
            "demo_race_ethnicity_cat",
        ]

        cure_ckd_bn.net.update_beliefs()
        beliefs = cure_ckd_bn.net.get_node_value(vars_to_assess[0])
        categories = [
            cure_ckd_bn.net.get_outcome_id(vars_to_assess[0], i)
            for i in range(len(beliefs))
        ]
        print(categories)

        # steps:
        # get distinct races
        # in df set all race to each category
        # save the predictions each time
        # KS test to identify significant differences of all combination of categories
        # report statistical significance and average with std of predictions

        predictions_list = []

        for category in categories:
            df_preds_N_closest_cat = df_preds_N_closest.copy(deep=True)
            df_preds_N_closest_cat[vars_to_assess[0]] = category

            predictions = cure_ckd_bn.testBN(
                df_preds_N_closest_cat,
                cols_to_test,
                target,
                ignore_warnings=False,
            ).values
            predictions_list.append(np.round(predictions, 9))

        sens_results = {}

        sens_results["categories"] = categories
        sens_results["predictions_list"] = predictions_list

        combinations = []

        for i in range(len(predictions_list) - 1):
            for j in range(1, len(predictions_list)):
                combination = {}
                combination["name"] = [categories[i], categories[j]]
                combination["predictions_list_indices"] = [i, j]
                combination["categories_avg_pred"] = [
                    np.mean(predictions_list[i]),
                    np.mean(predictions_list[j]),
                ]
                combination["categories_std_pred"] = [
                    np.std(predictions_list[i]),
                    np.std(predictions_list[j]),
                ]
                combinations.append(combination)

                for index, (first, second) in enumerate(
                    zip(predictions_list[i], predictions_list[j])
                ):
                    if first != second:
                        print(index, first, second)

                print(categories[i], categories[j])
                print(
                    stats.ks_2samp(predictions_list[i], predictions_list[j]).pvalue
                    < 0.05  # / len(cols_to_test)  # bonferoni correction
                )
                print(np.mean(predictions_list[i]), np.mean(predictions_list[j]))

        sens_results["combinations"] = combinations

        print(len(predictions_list))
