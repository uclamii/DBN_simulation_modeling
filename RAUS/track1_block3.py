"""
    This directory contains implementations of RAUS for unknown structure learning algorithms using an AKI dataset.

    To run the pipeline and return the RAUS track 1 block 3, run:

```shell
$ python track1_block3.py

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
from information_gain_ranking import IG_MI_Ranking
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
    ## BLOCK 3 ##
    #############

    print("Starting Track 1: IG-DBN")
    ig_mi_ranking, ig_variable_order, target = IG_MI_Ranking(
        df_Train, args.TARGET, args.COLS
    )
    # print('IG_MI Variable Order:', ig_variable_order) # to check learned variable order
    # print('Target Variable:', target) # to check target variable used
    save_figure1, save_figure2 = RAUSPlot(
        ig_mi_ranking,
        "Information_Gain",
        args.clipback,
        args.clipfront,
        ig_variable_order,
        args.site,
        args.outcome_name,
        args.adjusted,
        "Information Gain",
        args.track,
    )

    if args.rank_filter == 1:
        ig_variable_order = [
            x for x in ig_variable_order if x not in args.ig_rank_filter2
        ]
    if args.select_best_k == 1:
        ig_variable_order = ig_variable_order[: args.ig_top_features]
    if args.select_percentile == 1:
        selectpct, ig_variable_order = SelectPercentile(
            ig_mi_ranking, "Information_Gain", ig_variable_order, args.percentile
        )
    df = temporal_undersampling(
        df_Train,
        args.TARGET,
        args.outcome_name,
        "ig_mi_ranking",
        args.site,
        args.adjusted,
        args.track,
    )

    ig_intra_cols = dbn_cols_intra(ig_variable_order, target)
    # print('Intra_Cols:', ig_intra_cols) #to check the intra columns
    ig_ns_list = NodeSize(df, ig_intra_cols)

    ig_dag = intra_struct(
        df,
        ig_intra_cols,
        ig_ns_list,
        "ig_mi_ranking",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.max_fan_in,
    )
    # graph visualization using networkx
    figure1, figure1 = draw_intra_structure(
        ig_dag,
        ig_intra_cols,
        "ig_mi_ranking",
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
    ig_inter_cols, targets = dbn_cols_inter(
        ig_intra_cols,
        args.sequence_length_dbn,
        args.cols_start,
        "",
        args.cols_end,
        "_",
        args.outcome_name,
    )
    # print('Inter_Cols:', ig_inter_cols) #to check inter columns

    if args.outcome_name == "egfr_reduction40_ge":
        ig_inter_ns_list = Inter_NodeSize(df, ig_intra_cols, ig_inter_cols)
        ig_inter_structure = inter_struct(
            df,
            ig_inter_cols,
            ig_inter_ns_list,
            "ig_mi_ranking",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.sequence_length_dbn,
            args.track,
            args.max_fan_in,
        )
    else:
        ig_inter_structure = inter_struct(
            df,
            ig_inter_cols,
            ig_ns_list,
            "ig_mi_ranking",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.sequence_length_dbn,
            args.track,
            args.max_fan_in,
        )

    # graph visualization
    figure1, figure2 = draw_inter_structure(
        ig_inter_structure,
        ig_intra_cols,
        "ig_mi_ranking",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.clipback,
        args.clipfront,
    )
    # graph visualization
    figure1, figure2 = draw_compact_structure(
        ig_dag,
        ig_inter_structure,
        ig_intra_cols,
        "ig_mi_ranking",
        args.outcome_name,
        args.site,
        args.adjusted,
        args.track,
        args.clipback,
        args.clipfront,
    )

    if args.outcome_name == "egfr_reduction40_ge":
        ig_dbn_ns_list = DBN_NodeSize(df, ig_intra_cols, ig_inter_cols)
        ig_dataTrainValid = DBNModel_TVT(
            df,
            df2,
            df3,
            ig_inter_cols,
            ig_inter_structure,
            args.sequence_length_dbn,
            args.max_iter,
            ig_dag,
            ig_dbn_ns_list,
            "ig_mi_ranking",
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
        ig_dataTrainValid = DBNModel_TT(
            df,
            df2,
            ig_inter_cols,
            ig_inter_structure,
            args.sequence_length_dbn,
            args.max_iter,
            ig_dag,
            ig_ns_list,
            "ig_mi_ranking",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.track,
            args.ncases,
        )

    # Evaluation Component
    if args.outcome_name == "egfr_reduction40_ge":
        (
            ig_roc_aucs_valid,
            ig_ap_scores_valid,
            ig_fprs_valid,
            ig_tprs_valid,
            ig_thresholds_valid,
            ig_roc_aucs_test,
            ig_ap_scores_test,
            ig_fprs_test,
            ig_tprs_test,
            ig_thresholds_test,
        ) = dbn_performance_metrics_tvt(
            df,
            df2,
            df3,
            ig_dataTrainValid,
            args.sequence_length_dbn,
            "ig_mi_ranking",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.ncases,
            targets,
            args.track,
        )

    else:
        (
            ig_roc_aucs_valid,
            ig_ap_scores_valid,
            ig_fprs_valid,
            ig_tprs_valid,
            ig_thresholds_valid,
        ) = dbn_performance_metrics_tt(
            df,
            df2,
            ig_dataTrainValid,
            args.sequence_length_dbn,
            "ig_mi_ranking",
            args.outcome_name,
            args.site,
            args.adjusted,
            args.ncases,
            targets,
            args.track,
        )

    # Pickle dump output
    Pickledump(
        ig_ap_scores_valid,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "ig_ap_scores_valid",
    )
    Pickledump(
        ig_inter_cols,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "ig_inter_cols",
    )
    Pickledump(
        ig_inter_structure,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "ig_inter_structure",
    )
    Pickledump(
        ig_ns_list,
        "./Track_Inputs/"
        + args.outcome_name
        + "_"
        + args.site
        + "_"
        + args.track
        + "_"
        + args.adjusted
        + "_"
        + "ig_ns_list",
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
    parser.add_argument("--ig_rank_filter2", nargs="+", default=[])
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
        "--ig_top_features",
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
