import pandas as pd
import numpy as np
from tqdm import tqdm

from BN_pysmile import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_score, recall_score


if __name__ == "__main__":

    ### importing structures

    best_rank_metric = "chisquare"

    #################### inter structure
    df_inter = pd.read_csv(
        "../Data/simulation_structures/full_model/four_structures_for_simulation_model_csv/chi2_UCLA_inter_struct_yr1_4.csv",
        index_col=0,
    )

    inter_indexes = [
        val.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for val in df_inter.index
    ]
    df_inter.index = inter_indexes

    cols_df_inter = [
        col.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for col in df_inter.columns
    ]
    df_inter.columns = cols_df_inter

    # all temporal columns
    cols_inter = [
        col[:4] + str(int(col[4]) + num) + col[5:]
        for col in list(df_inter.index)
        for num in range(4)
    ]

    #################### time zero to year 1 intra structure

    df_intra_year0_1 = pd.read_csv(
        "../Data/simulation_structures/full_model/four_structures_for_simulation_model_csv/chi2_UCLA_intra_struct_yr0_1.csv",
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
        val.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for val in df_intra_year0_1.index
    ]
    df_intra_year0_1.index = intra_year0_1_indexes

    cols_df_intra_year0_1 = [
        col.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for col in df_intra_year0_1.columns
    ]
    df_intra_year0_1.columns = cols_df_intra_year0_1

    cols_df_intra_year0_1 = list(df_intra_year0_1.columns)

    # it must not equal since we removed time edges that should not be there
    assert list(df_intra_year0_1.index) != list(df_intra_year0_1.columns)

    #################### year 1 intra structure

    df_intra_year1 = pd.read_csv(
        "../Data/simulation_structures/full_model/four_structures_for_simulation_model_csv/chi2_UCLA_intra_struct_yr1.csv",
        index_col=0,
    )

    intra_year1_indexes = [
        val.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for val in df_intra_year1.index
    ]
    df_intra_year1.index = intra_year1_indexes

    cols_df_intra_year1 = [
        col.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for col in df_intra_year1.columns
    ]
    df_intra_year1.columns = cols_df_intra_year1

    cols_df_intra_year1 = list(df_intra_year1.columns)

    assert list(df_intra_year1.index) == list(df_intra_year1.columns)

    cols_df_intra_year_1_4 = [
        col[:4] + str(int(col[4]) + num) + col[5:]
        for col in list(df_intra_year1.index)
        for num in range(4)
    ]

    #################### static to time zero intra structure

    df_intra_static_year0 = pd.read_csv(
        "../Data/simulation_structures/full_model/four_structures_for_simulation_model_csv/chi2_UCLA_intra_struct_yr0_s.csv",
        index_col=0,
    )

    intra_static_year0_indexes = [
        val.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for val in df_intra_static_year0.index
    ]
    df_intra_static_year0.index = intra_static_year0_indexes

    cols_intra_static_year0 = [
        col.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for col in df_intra_static_year0.columns
    ]
    df_intra_static_year0.columns = cols_intra_static_year0

    # filtering out count variables
    intra_static_year0_indexes_bool = [
        True if "site" not in val
        # and "count" not in val
        # and "ruca_7" not in val
        # and "coverage" not in val
        # and "uacr" not in val
        # and "upcr" not in val
        else False
        for val in df_intra_static_year0.index
    ]
    cols_intra_static_year0_filt = [
        col.replace("year0", "time_zero").replace("reduction_40_wr", "reduction_40_ge")
        for col in df_intra_static_year0.columns
        if "site" not in col
        # and "count" not in col
        # and "ruca_7" not in col
        # and "coverage" not in col
        # and "uacr" not in col
        # and "upcr" not in col
    ]

    df_intra_static_year0 = df_intra_static_year0.loc[
        intra_static_year0_indexes_bool, cols_intra_static_year0_filt
    ]

    cols_df_intra_static_year0 = list(df_intra_static_year0.columns)

    assert list(df_intra_static_year0.index) == list(df_intra_static_year0.columns)

    ### importing data

    df_train = pd.read_csv(
        "../Data/genie_datasets/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_train_undersampled_DBN.csv",
        low_memory=False,
        # nrows=20000,
    )

    cols_to_keep = list(
        np.unique(
            cols_inter
            + cols_df_intra_year_1_4
            + cols_df_intra_year0_1
            + cols_df_intra_static_year0
        )
    )

    # print(df_train[cols_to_keep].isnull().sum()[30:40])
    # print(cols_to_keep)

    df_train = df_train[cols_to_keep]  # .dropna()

    # print(len(df_train))

    #### Creating network and saving as xdsl file

    cure_ckd_bn = CURE_CKD_BayesianNetwork()

    # To create the BN model only from static to time zero
    # comment out in the BN_smile.py the part of the create structure method that creates the
    # remaining epochs.

    cure_ckd_bn.create_BN(
        df_intra_static_time_zero=df_intra_static_year0,
        df_intra_time_zero_year1=df_intra_year0_1,
        df_intra_struct_year1=df_intra_year1,
        df_inter=df_inter,
        df=df_train[cols_to_keep],
        BN_static_t0=True,
    )

    # train network
    cure_ckd_bn.trainBN(df_train=df_train)

    # # learn network
    # cure_ckd_bn.learnBN_structure(df_train=df_train, method="ABN")

    # save file
    cure_ckd_bn.save_BN(filename="CURE_CKD_BN_static_time0.xdsl")

    # load file
    cure_ckd_bn.load_BN(filename="CURE_CKD_BN_static_time0.xdsl")

    filenames = [
        "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_train.csv",
        "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_valid.csv",
        "split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_test.csv",
        "split_discetized_datasets_using_UCLA_discritizer/Prov_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_using_UCLA_discritizer.csv",
    ]

    for filename in filenames:

        df_valid = pd.read_csv(
            "../Data/genie_datasets/" + filename,
            low_memory=False,
            # nrows=5000,
        )

        df_valid = df_valid[cols_to_keep]

        # test network

        num_of_epochs = 1

        for epoch_num in range(0, num_of_epochs):
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

            predictions = cure_ckd_bn.testBN(
                df_valid,
                cols_to_test,
                target,
                ignore_warnings=False,
            ).values
            truth = df_valid[target].str.replace("S_", "").astype(int).values

            df_valid["predictions_year" + str(epoch_num + 1)] = predictions

            preds = predictions > 0.5

            print(precision_score(truth, preds))
            print(recall_score(truth, preds))

            print(np.bincount(truth))
            print(roc_auc_score(truth, predictions))
            print(average_precision_score(truth, predictions))

        # save predictions
        df_valid.to_csv(
            "../Data/genie_datasets/DBN_predictions/all_var_BN_model/"
            + filename.replace("split_discetized_datasets/", "")
            .replace("split_discetized_datasets_using_UCLA_discritizer/", "")
            .replace(".csv", "")
            + "_with_DBN_predictions.csv",
            index=False,
        )
