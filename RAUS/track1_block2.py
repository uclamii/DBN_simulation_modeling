"""
    This directory contains implementations of RAUS for unknown structure learning algorithms using an AKI dataset.

    To run the pipeline and return the RAUS track 1 block 2, run:

```shell
$ python track1_block2.py

"""

import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import os
from os import system
from sklearn.metrics import roc_curve
from chi_squared_ranking import ChiSquareRank
from barplot_raus import RAUSPlot
from select_percentile import SelectPercentile
from sampling_strategy import temporal_undersampling
from learn_structures import (
    dbn_cols_intra,
    NodeSize,
    Inter_NodeSize,
    DBN_NodeSize,
    intra_struct,
    BNModel_TT,
    BNModel_TVT,
    dbn_cols_inter,
    inter_struct,
    DBNModel_TT,
    DBNModel_TVT,
    draw_intra_structure,
    draw_inter_structure,
    draw_compact_structure,
)
from performance_metrics import (
    dbn_performance_metrics_tvt,
    dbn_performance_metrics_tt,
    Pickledump,
    Pickleload,
)


def main(args):

    """
    Input:
    - Cramer's V Ranking DBN
    - ChiSquared Ranking DBN
    - InformationGain DBN

    Output:
    - Saves Ranking Figures
    - Intrastructures
    - Interstructures

    """

    # Load Dataframes
    print("Starting RAUS Framework")
    print("Loading Datasets")
    df_Train = pd.read_csv(args.file_name_train, low_memory=False)
    if (
        args.outcome_name == "AKI_BOS24"
        or args.outcome_name == "AKI_BOS48"
        or args.outcome_name == "AKI_BOS72"
    ):
        column_names = ["aki_progression_" + str(m) + "days" for m in range(1, 8)]
        df_Train[column_names] = df_Train[column_names] + 1  # Octave format starts at 1
    df_Valid = pd.read_csv(args.file_name_valid, low_memory=False)
    if (
        args.outcome_name == "AKI_BOS24"
        or args.outcome_name == "AKI_BOS48"
        or args.outcome_name == "AKI_BOS72"
    ):
        column_names = ["aki_progression_" + str(m) + "days" for m in range(1, 8)]
        df_Valid[column_names] = df_Valid[column_names] + 1  # Octave format starts at 1
    if args.outcome_name == "egfr_reduction40_ge":
        df_Test = pd.read_csv(args.file_name_test, low_memory=False)
        # print(df_Test.head(10))

    ##############################################################################################
    #################################### Track 1 #################################################
    ##############################################################################################

    ##############
    ## BLOCK 2 ##
    #############

    print("Starting Track 1: Chi2-DBN")
    chisquarerank, chi2_variable_order, target = ChiSquareRank(
        df_Train, args.COLS, args.TARGET
    )
    # print('ChiSquared Variable Order:', chi2_variable_order) # to check learned variable order
    # print('Target Variable:', target) # to check learned target variable
    save_figure1, save_figure2 = RAUSPlot(
        chisquarerank,
        "Chi_Square",
        args.clipback,
        args.clipfront,
        chi2_variable_order,
        args.site,
        args.outcome_name,
        args.adjusted,
        "Chi-Squared",
        args.track,
    )

    if args.rank_filter == 1:
        chi2_variable_order = [
            x for x in chi2_variable_order if x not in args.chi2_rank_filter2
        ]
    if args.select_best_k == 1:
        chi2_variable_order = chi2_variable_order[: args.chi2_top_features]
    if args.select_percentile == 1:
        selectpct, chi2_variable_order = SelectPercentile(
            chisquarerank, "Chi_Square", chi2_variable_order, args.percentile
        )
    df = temporal_undersampling(
        df_Train,
        args.TARGET,
        args.outcome_name,
        "chisquarerank",
        args.site,
        args.adjusted,
        args.track,
    )
    chi2_intra_cols = dbn_cols_intra(chi2_variable_order, target)
    # print('Intra_Cols:', chi2_intra_cols) #to check intra columns order
    chi2_ns_list = NodeSize(df, chi2_intra_cols)

    chi2_dag = intra_struct(
        df,
        chi2_intra_cols,
        chi2_ns_list,
        "chisquarerank",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.max_fan_in,
    )
    # graph visualization using networkx
    figure1, figure2 = draw_intra_structure(
        chi2_dag,
        chi2_intra_cols,
        "chisquarerank",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.clipback,
        args.clipfront,
    )

    df2 = df_Valid.copy()
    if args.outcome_name == "egfr_reduction40_ge":
        df3 = df_Test.copy()

    # DBN Component
    chi2_inter_cols, targets = dbn_cols_inter(
        chi2_intra_cols,
        args.sequence_length_dbn,
        args.cols_start,
        "",
        args.cols_end,
        "_",
        args.outcome_name,
    )
    # print('Inter_Cols:', chi2_inter_cols) #to check inter columns

    if args.outcome_name == "egfr_reduction40_ge":
        chi2_inter_ns_list = Inter_NodeSize(df, chi2_intra_cols, chi2_inter_cols)
        chi2_inter_structure = inter_struct(
            df,
            chi2_inter_cols,
            chi2_inter_ns_list,
            "chisquarerank",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.sequence_length_dbn,
            args.track,
            args.max_fan_in,
        )
    else:
        chi2_inter_structure = inter_struct(
            df,
            chi2_inter_cols,
            chi2_ns_list,
            "chisquarerank",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.sequence_length_dbn,
            args.track,
            args.max_fan_in,
        )

    # graph visualization
    figure1, figure2 = draw_inter_structure(
        chi2_inter_structure,
        chi2_intra_cols,
        "chisquarerank",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.clipback,
        args.clipfront,
    )
    # graph visualization
    figure1, figure2 = draw_compact_structure(
        chi2_dag,
        chi2_inter_structure,
        chi2_intra_cols,
        "chisquarerank",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.clipback,
        args.clipfront,
    )

    if args.outcome_name == "egfr_reduction40_ge":
        chi2_dbn_ns_list = DBN_NodeSize(df, chi2_intra_cols, chi2_inter_cols)
        chi2_dataTrainValid = DBNModel_TVT(
            df,
            df2,
            df3,
            chi2_inter_cols,
            chi2_inter_structure,
            args.sequence_length_dbn,
            args.max_iter,
            chi2_dag,
            chi2_dbn_ns_list,
            "chisquarerank",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.track,
            args.ncases,
        )

    elif (
        args.outcome_name == "AKI_BOS24"
        or args.outcome_name == "AKI_BOS48"
        or args.outcome_name == "AKI_BOS72"
    ):
        chi2_dataTrainValid = DBNModel_TT(
            df,
            df2,
            chi2_inter_cols,
            chi2_inter_structure,
            args.sequence_length_dbn,
            args.max_iter,
            chi2_dag,
            chi2_ns_list,
            "chisquarerank",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.track,
            args.ncases,
        )

    # Evaluation Component
    if args.outcome_name == "egfr_reduction40_ge":
        (
            chi2_roc_aucs_valid,
            chi2_ap_scores_valid,
            chi2_fprs_valid,
            chi2_tprs_valid,
            chi2_thresholds_valid,
            chi2_roc_aucs_test,
            chi2_ap_scores_test,
            chi2_fprs_test,
            chi2_tprs_test,
            chi2_thresholds_test,
        ) = dbn_performance_metrics_tvt(
            df,
            df2,
            df3,
            chi2_dataTrainValid,
            args.sequence_length_dbn,
            "chisquarerank",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.ncases,
            targets,
            args.track,
        )

    else:
        (
            chi2_roc_aucs_valid,
            chi2_ap_scores_valid,
            chi2_fprs_valid,
            chi2_tprs_valid,
            chi2_thresholds_valid,
        ) = dbn_performance_metrics_tt(
            df,
            df2,
            chi2_dataTrainValid,
            args.sequence_length_dbn,
            "chisquarerank",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.ncases,
            targets,
            args.track,
        )

    # Pickle dump output
    Pickledump(
        chi2_ap_scores_valid,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "chi2_ap_scores_valid",
    )
    Pickledump(
        chi2_inter_cols,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "chi2_inter_cols",
    )
    Pickledump(
        chi2_inter_structure,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "chi2_inter_structure",
    )
    Pickledump(
        chi2_ns_list,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "chi2_ns_list",
    )

    output = {"save_RAUS_figure": save_figure1}

    return output


##
if __name__ == "__main__":

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name_train",
        # default='/path/to/train/dataset',
        type=str,
    )
    parser.add_argument(
        "--file_name_valid",
        # default='/path/to/valid/dataset',
        type=str,
    )
    parser.add_argument(
        "--file_name_test",
        # default='/path/to/test/dataset',
        type=str,
    )
    parser.add_argument(
        "--cols_start",
        # default= '24hourperiod_0',
        type=str,
    )
    parser.add_argument(
        "--cols_end",
        # default= '_2days',
        type=str,
    )
    parser.add_argument("--track", default="Track1", type=str)
    parser.add_argument(
        "--rank_filter",
        # default= 0,
        type=int,
    )
    parser.add_argument("--chi2_rank_filter2", nargs="+", default=[])
    parser.add_argument(
        "--outcome_name",
        # default= 'AKI_BOS24',
        type=str,
    )
    parser.add_argument(
        "--select_best_k",
        # default= 1,
        type=int,
    )
    parser.add_argument(
        "--chi2_top_features",
        # default= 8,
        type=int,
    )
    parser.add_argument(
        "--select_percentile",
        # default= 0,
        type=int,
    )
    parser.add_argument(
        "--percentile",
        # default= 80,
        type=int,
    )
    parser.add_argument(
        "--ncases",
        # default= 100,
        type=int,
    )
    parser.add_argument(
        "--max_fan_in",
        # default= 2,
        type=int,
    )
    parser.add_argument(
        "--sequence_length_dbn",
        # default= 4,
        type=int,
    )
    parser.add_argument(
        "--max_iter",
        # default= 10,
        type=int,
    )
    parser.add_argument("--site", default="UCLA", type=str)
    parser.add_argument(
        "--adjusted",
        # default= '_w_Race_Adjusted_Variables',
        type=str,
    )
    parser.add_argument("--clipback", default="", type=str)
    parser.add_argument("--clipfront", default="", type=str)
    parser.add_argument("--COLS", nargs="+", default=[])
    parser.add_argument("--TARGET", nargs="+", default=[])

    args = parser.parse_args()

    # Call main function
    output = main(args)
