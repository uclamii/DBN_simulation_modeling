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
from pickleObjects import *

import sys


def chi2_var_test(s1, s2, bonferroni_corr, alpha=0.05):
    """Method to test if categorical variables are different if True, dignificantly different"""

    # s1 = df1[col1].fillna('nan')
    # s2 = df2[col2].fillna('nan')

    alpha = alpha / bonferroni_corr

    uniq_vals = list(np.unique(list(s1) + list(s2)))

    s1_val_counts = np.array([np.sum(s1 == x) for x in uniq_vals]) + 1  # avoid zeros
    s2_val_counts = np.array([np.sum(s2 == x) for x in uniq_vals]) + 1  # avoid zeros

    p = stats.chi2_contingency([s1_val_counts + 1, s2_val_counts + 1])[
        1
    ]  # p-value is index 1

    return p < alpha


def confusion_matrix(df, epoch_num, threshold):
    """Method to compute FNs,FPs,TNs,TPs."""

    FN = np.sum(
        (df["predictions_year" + str(1 + epoch_num)] <= threshold)
        & (df["year" + str(1 + epoch_num) + "_reduction_40_ge"] == "S_1")
    )

    TP = np.sum(
        (df["predictions_year" + str(1 + epoch_num)] > threshold)
        & (df["year" + str(1 + epoch_num) + "_reduction_40_ge"] == "S_1")
    )

    TN = np.sum(
        (df["predictions_year" + str(1 + epoch_num)] <= threshold)
        & (df["year" + str(1 + epoch_num) + "_reduction_40_ge"] == "S_0")
    )

    FP = np.sum(
        (df["predictions_year" + str(1 + epoch_num)] > threshold)
        & (df["year" + str(1 + epoch_num) + "_reduction_40_ge"] == "S_0")
    )
    return FN, TP, TN, FP


if __name__ == "__main__":
    num_epochs = 6

    # defining paths
    structures_path = "./RAUS/FullNetwork/"

    model_site = sys.argv[1]
    testing_site = sys.argv[2]
    med_keyword = sys.argv[4]
    if sys.argv[3] == "on":
        on_medication = True
    elif sys.argv[3] == "off":
        on_medication = False
    else:
        raise Exception("Is it on or off? ... specify please")

    models_struct_directories = [
        model_name
        for model_name in glob.glob(structures_path + "*")
        if "no_race" in model_name
        and "count" not in model_name
        and model_site in model_name
        # and "UCLA" in model_name
        # and "PSJH" in model_name
        # and "Combined" in model_name
    ]

    models = dict()
    models_names = []

    # TODO: loop model directories
    for models_struct_directory in tqdm(models_struct_directories):

        # defining paths
        structures_path = "./RAUS/FullNetwork/"
        print(models_struct_directory)

        model_name = models_struct_directory.replace(structures_path, "")

        ##########################################################

        # Selecting
        #  each years predictions to find the best model
        # for each site and filter

        results_path = "./Data/genie_datasets/DBN_predictions/Results/"

        print("##########################################################")

        print("For model " + model_name + " ... generating results ...")

        # focusing on "all" race results
        sites_models_results_filepaths = [
            model_path
            for model_path in glob.glob(results_path + "*")
            if model_name in model_path and "all" in model_path
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

        ###################################### loading thresholds

        thresholds = [
            df_results.loc[
                (df_results["Metric"].str.contains("Optimal"))
                & (df_results["Race Ethnicity category"] == "all races"),
                "Prediction Year "
                + str(epoch_num + 1)
                + ", target year "
                + str(epoch_num + 1),
            ].values[0]
            for epoch_num in range(num_epochs)
        ]

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

        # picking testing site dataset
        if model_site == "UCLA" and testing_site == "UCLA":
            filename = filenames[2]
        elif model_site == "UCLA" and testing_site == "PSJH":
            filename = filenames[3]
        elif model_site == "UCLA" and testing_site == "Combined":
            raise ValueError(
                "Testing site spelling must be worng when UCLA is the model site..."
            )
        elif model_site == "PSJH" and testing_site == "PSJH":
            filename = filenames[2]
        elif model_site == "PSJH" and testing_site == "UCLA":
            filename = filenames[3]
        elif model_site == "PSJH" and testing_site == "Combined":
            raise ValueError(
                "Testing site spelling must be worng when PSJH is the model site..."
            )
        elif model_site == "Combined" and testing_site == "Combined":
            filename = filenames[2]
        else:
            raise ValueError("Testing site spelling must be worng...")

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

        ###################################### create sensitivity analysis module

        ###################################### loading model's structure

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
        #################### sensitivity analysis Regression discontinuity finding optimal random set
        sens_res_epochs = {}
        epoch_powers = []
        for epoch_num in range(num_epochs):
            threshold = thresholds[epoch_num]
            ###################################### find N closest patients with predictions to the optimal threshold
            FNs, TPs, FPs, TNs = [], [], [], []
            Power = 0
            Powers = []
            ranges = np.arange(100, 10000, 1)

            for N in tqdm(ranges):

                # getting the N closest points based on distance using absolute value
                # pandas default order is ascending
                df_preds_N_closest = df_preds.iloc[
                    (df_preds["predictions_year" + str(1 + epoch_num)] - threshold)
                    .abs()
                    .argsort()[:N]
                ].reset_index(drop=True)

                ################# testing balanced covariates
                ## condition for Regression Discontinuity validity
                # splitting to left and right side of the threshold
                # subsets to test if distributions are the same

                cond_rhs = (
                    df_preds_N_closest["predictions_year" + str(1 + epoch_num)]
                    > threshold
                )
                cond_lhs = (
                    df_preds_N_closest["predictions_year" + str(1 + epoch_num)]
                    < threshold
                )
                df_preds_N_closest_rhs = df_preds_N_closest.loc[cond_rhs].reset_index(
                    drop=True
                )
                df_preds_N_closest_lhs = df_preds_N_closest.loc[cond_lhs].reset_index(
                    drop=True
                )

                ## testing columns
                cols_to_test = [
                    "study_entry_DM_flag",
                    "study_entry_period_egfrckd_norace_flag",
                    "study_entry_period_dxckd_flag",
                    "time_zero_norace_mean",
                    "study_entry_age",
                    "demo_race_ethnicity_cat",
                ]

                ####################################################
                # # target variable
                # target = "year" + str(1 + epoch_num) + "_reduction_40_ge"

                # if epoch_num == 0:
                #     all_cols_to_be_tested = cols_df_intra_static_year0
                # else:
                #     all_cols_to_be_tested = cols_df_intra_static_year0 + [
                #         col[:4] + str(int(col[4]) + num) + col[5:]
                #         for col in list(df_intra_year1.index)
                #         for num in range(0, epoch_num)
                #     ]
                # # testing columns
                # cols_to_test = [col for col in all_cols_to_be_tested if target != col]

                # variables are all categorical do using chi-square test
                covariates_p_vals = []
                for col in cols_to_test:
                    s1, s2 = df_preds_N_closest_lhs[col].fillna(
                        "nan"
                    ), df_preds_N_closest_rhs[col].fillna("nan")
                    covariates_p_vals.append(
                        chi2_var_test(s1, s2, bonferroni_corr=len(cols_to_test))
                    )
                #################

                FN, TP, TN, FP = confusion_matrix(
                    df_preds_N_closest, epoch_num, threshold
                )
                FNs.append(FN)
                TPs.append(TP)
                TNs.append(TN)
                FPs.append(FP)

                Power = 1 - FN / (FN + TP)

                # Used to test if covariates balanced
                if np.any(covariates_p_vals):
                    Powers.append(-1)  # invalid RD design if not balanced
                else:
                    Powers.append(Power)
                # Powers.append(Power)

            epoch_powers.append(np.nanmax(Powers))
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

            # selecting optimal subset of N points

            df_preds_N_closest = df_preds.iloc[
                (df_preds["predictions_year" + str(1 + epoch_num)] - threshold)
                .abs()
                .argsort()[:N]
            ].reset_index(drop=True)

            ####################################################
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
            # vars_to_assess = [
            #     "study_entry_DM_flag",
            #     "study_entry_period_egfrckd_norace_flag",
            #     "time_zero_ipv_count",
            #     "study_entry_period_dxckd_flag",
            #     "time_zero_mean",
            #     "study_entry_age",
            #     "demo_race_ethnicity_cat",
            # ]

            cols_to_profile = [
                "study_entry_age",
                "demo_sex",
                "demo_race_ethnicity_cat",
                "ruca_4_class",
                "study_entry_period_egfrckd_norace_flag",
                "study_entry_period_dxckd_flag",
                "study_entry_period_albprockd_flag",
                "study_entry_DM_flag",
                "study_entry_PDM_flag",
                "study_entry_HTN_flag",
                "study_entry_aceiarb_flag",
                "study_entry_sglt2_flag",
                "study_entry_glp1_flag",
                "study_entry_nsaid_flag",
                "study_entry_ppi_flag",
                "study_entry_mra_flag",
                "time_zero_hba1c_mean",
                "time_zero_uacr_mean",
                "time_zero_upcr_mean",
                "time_zero_sbp_mean",
                "time_zero_dbp_mean",
                "time_zero_pp_mean",
                "time_zero_map_mean",
                "time_zero_aceiarb_coverage",
                "time_zero_sglt2_coverage",
                "time_zero_glp1_coverage",
                "time_zero_nsaid_coverage",
                "time_zero_ppi_coverage",
                "time_zero_mra_coverage",
                "time_zero_norace_mean",
            ]

            vars_to_assess = cols_to_profile

            sens_results_vars = {}

            for var_to_assess in vars_to_assess:

                #################################
                # re-loading net file
                cure_ckd_bn.load_BN(
                    filename="./DBN_model_learning/models/DBNs/"
                    + model_name
                    + "CURE_CKD_DBN.xdsl"
                )

                cure_ckd_bn.net.update_beliefs()
                beliefs = cure_ckd_bn.net.get_node_value(var_to_assess)
                categories = [
                    cure_ckd_bn.net.get_outcome_id(var_to_assess, i)
                    for i in range(len(beliefs))
                ]
                print("Variable: ", var_to_assess)
                print("Variable categories: ", categories)

                # applying do queries on medications
                med_vars_to_assess = [col for col in cols_to_test if med_keyword in col]
                ################################# do queries on med vars
                for med_var_to_assess in med_vars_to_assess:
                    med_parent_vars = [
                        cure_ckd_bn.net.get_node_name(var_id)
                        for var_id in cure_ckd_bn.net.get_parents(med_var_to_assess)
                    ]
                    # deleting incoming edges of var
                    for med_parent_var in med_parent_vars:
                        cure_ckd_bn.net.delete_arc(med_parent_var, med_var_to_assess)

                # for profiling no need to do DO queries
                # ################################# do queries
                # parent_vars = [
                #     cure_ckd_bn.net.get_node_name(var_id)
                #     for var_id in cure_ckd_bn.net.get_parents(var_to_assess)
                # ]
                # # if no childern no need to test DO query
                # children_vars = cure_ckd_bn.net.get_children(var_to_assess)
                # if len(children_vars) == 0:
                #     continue
                # # deleting incoming edges of var
                # for parent_var in parent_vars:
                #     cure_ckd_bn.net.delete_arc(parent_var, var_to_assess)

                cure_ckd_bn.net.update_beliefs()
                ############################################

                ###### med vars continuous on/off Do medication

                med_vars = []
                med_categories = []

                ## year zero use study entry and time zero
                med_var = "study_entry_" + med_keyword + "_flag"
                med_vars.append(med_var)
                med_var_beliefs = cure_ckd_bn.net.get_node_value(med_var)
                med_var_categories = [
                    cure_ckd_bn.net.get_outcome_id(med_var, i)
                    for i in range(len(med_var_beliefs))
                ]
                if on_medication:
                    med_category = [
                        element for element in med_var_categories if "S_1" in element
                    ]
                    med_categories.append(med_category[0])
                else:
                    med_category = [
                        element for element in med_var_categories if "S_0" in element
                    ]
                    med_categories.append(med_category[0])
                # time zero coverage
                med_var = "time_zero_" + med_keyword + "_coverage"
                med_vars.append(med_var)
                med_var_beliefs = cure_ckd_bn.net.get_node_value(med_var)
                med_var_categories = [
                    cure_ckd_bn.net.get_outcome_id(med_var, i)
                    for i in range(len(med_var_beliefs))
                ]
                if on_medication:
                    med_category = [
                        element for element in med_var_categories if "_le_" in element
                    ]
                    med_categories.append(med_category[0])
                else:
                    med_category = [
                        element for element in med_var_categories if "_s_" in element
                    ]
                    med_categories.append(med_category[0])
                if epoch_num > 0:  # other years use year1-5
                    for year_num in range(1, epoch_num + 1):
                        med_var = (
                            "year" + str(year_num) + "_" + med_keyword + "_coverage"
                        )
                        med_vars.append(med_var)
                        med_var_beliefs = cure_ckd_bn.net.get_node_value(med_var)
                        med_var_categories = [
                            cure_ckd_bn.net.get_outcome_id(med_var, i)
                            for i in range(len(med_var_beliefs))
                        ]
                        if on_medication:
                            med_category = [
                                element
                                for element in med_var_categories
                                if "_le_" in element
                            ]
                            med_categories.append(med_category[0])
                        else:
                            med_category = [
                                element
                                for element in med_var_categories
                                if "_s_" in element
                            ]
                            med_categories.append(med_category[0])

                ############################################

                # steps:
                # get distinct races
                # in df set all race to each category
                # save the predictions each time
                # KS test to identify significant differences of all combination of categories
                # report statistical significance and average with std of predictions

                predictions_list = []

                for category in categories:
                    df_preds_N_closest_cat = df_preds_N_closest.copy(deep=True)
                    df_preds_N_closest_cat[
                        var_to_assess
                    ] = category  # cast same category value
                    for med_category, med_var in zip(med_categories, med_vars):
                        df_preds_N_closest_cat[
                            med_var
                        ] = med_category  # cast same category value

                    # compute P(outcome|var,Do(med_vars),z)
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
                        if i != j:  # same med category
                            combination = {}
                            combination["name"] = [
                                sens_results["categories"][i],
                                sens_results["categories"][j],
                            ]
                            combination["predictions_list_indices"] = [i, j]
                            combination["categories_avg_pred"] = [
                                np.mean(predictions_list[i]),
                                np.mean(predictions_list[j]),
                            ]
                            combination["categories_std_pred"] = [
                                np.std(predictions_list[i]),
                                np.std(predictions_list[j]),
                            ]
                            combination["KS_test_sign"] = (
                                stats.ks_2samp(
                                    predictions_list[i], predictions_list[j]
                                ).pvalue
                                < 0.05
                            )  # / len(cols_to_test)  # bonferoni correction
                            combinations.append(combination)

                            print("\n")
                            print("Variable: ", var_to_assess)
                            print(
                                "Categories: ",
                                sens_results["categories"][i],
                                sens_results["categories"][j],
                            )
                            print(
                                stats.ks_2samp(
                                    predictions_list[i], predictions_list[j]
                                ).pvalue
                                < 0.05  # / len(cols_to_test)  # bonferoni correction
                            )
                            print(
                                np.mean(predictions_list[i]),
                                np.mean(predictions_list[j]),
                            )
                        else:
                            pass

                sens_results["combinations"] = combinations
                sens_results["Power"] = epoch_powers[epoch_num]
                sens_results["threshold"] = threshold
                sens_results_vars[var_to_assess] = sens_results
            sens_res_epochs["Year " + str(epoch_num + 1)] = sens_results_vars
        if on_medication:
            dumpObjects(
                sens_res_epochs,
                "Data/genie_datasets/DBN_predictions/Results/sens_analysis_results/"
                + med_keyword
                + "_continuous_with_med_"
                + model_name
                + "_sens_analysis_results_profiling_applied_on_"
                + testing_site,
            )
        else:
            dumpObjects(
                sens_res_epochs,
                "Data/genie_datasets/DBN_predictions/Results/sens_analysis_results/"
                + med_keyword
                + "_continuous_without_med_"
                + model_name
                + "_sens_analysis_results_profiling_applied_on_"
                + testing_site,
            )
