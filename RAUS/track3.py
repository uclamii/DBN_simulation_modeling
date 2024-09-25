"""
    This directory contains implementations of RAUS for unknown structure learning algorithms using an AKI dataset.

    To run the pipeline and return the RAUS track 3, run:

```shell
$ python track3.py

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
from cramers_v_ranking import CramersVRank
from chi_squared_ranking import ChiSquareRank
from information_gain_ranking import IG_MI_Ranking
from barplot_raus import RAUSPlot
from learn_structures import dbn_cols_intra,NodeSize,intra_struct,BNModel_TT,BNModel_TVT,dbn_cols_inter,inter_struct,DBNModel_TVT, DBNModel_TT, draw_intra_structure, draw_inter_structure, draw_compact_structure
from performance_metrics import bn_performance_metrics_tvt,bn_performance_metrics_tt,dbn_performance_metrics_tvt,dbn_performance_metrics_tt, Pickledump, Pickleload

def main (args):

    """
    Input:
    - Track2 Output

    Output:
    - Interstructure
    - DBN Performance

    """


    # Load Track 2 Outputs
    print('Starting Track 3: Learning DBN Using Track 2 Top Ranking Approach')
    print('Loading Output from Track 2')
    cv_ap_scores_valid = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "cv_ap_scores_valid")
    cv_intra_cols = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "cv_intra_cols")
    cv_ns_list = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "cv_ns_list")
    cv_dag = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "cv_dag")
    df = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "df")
    df2 = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "df2")
    if args.outcome_name == "egfr_reduction40_ge":
        df3 = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "df3")
    chi2_ap_scores_valid = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "chi2_ap_scores_valid")
    chi2_intra_cols = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "chi2_intra_cols")
    chi2_ns_list = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "chi2_ns_list")
    chi2_dag = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "chi2_dag")
    ig_ap_scores_valid = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "ig_ap_scores_valid")
    ig_intra_cols = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "ig_intra_cols")
    ig_ns_list = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "ig_ns_list")
    ig_dag = Pickleload("./Track3_Inputs/" + args.outcome_name + "_" + "ig_dag")



        ###################################################################
        ## Input Track 2 Top Ranking Approach into Track 3 DBN Component ##
        ###################################################################

        # Best Ranking from BN Used as Input to DBN Component (i.e., Ranking Approaches Compete)
    if cv_ap_scores_valid >= chi2_ap_scores_valid and cv_ap_scores_valid >= ig_ap_scores_valid:
        print('Top Ranking Approach is CV')
        inter_cols,targets = dbn_cols_inter(cv_intra_cols,args.sequence_length_dbn,args.cols_start,"",args.cols_end,"_",args.outcome_name)

        inter_structure = inter_struct(df,inter_cols,cv_ns_list,'cramersv',args.outcome_name,args.site,args.adjusted,args.sequence_length_dbn,args.track,args.max_fan_in)
        #graph visualization
        figure1,figure2 = draw_inter_structure(inter_structure,cv_intra_cols,'cramersv',args.outcome_name,args.site,args.adjusted,args.track,args.clipback,args.clipfront)
        #graph visualization
        figure1,figure2 =   draw_compact_structure(cv_dag,inter_structure,cv_intra_cols,'cramersv',args.outcome_name,args.site,args.adjusted,args.track,args.clipback,args.clipfront)

        if args.outcome_name == 'egfr_reduction40_ge':

            dataTrainValid = DBNModel_TVT(df,df2,df3,inter_cols,inter_structure,args.sequence_length_dbn,args.max_iter,cv_dag,cv_ns_list,'cramersv',args.outcome_name,args.site,args.adjusted,args.track,args.ncases)

        else:
            dataTrainValid = DBNModel_TT(df,df2,inter_cols,inter_structure,args.sequence_length_dbn,args.max_iter,cv_dag,cv_ns_list,'cramersv',args.outcome_name,args.site,args.adjusted,args.track,args.ncases)

    # Evaluate DBN from best ranking approach BN
        if args.outcome_name == 'egfr_reduction40_ge':
            roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid, roc_aucs_test, ap_scores_test, fprs_test, tprs_test, thresholds_test = dbn_performance_metrics_tvt(df,df2,df3,dataTrainValid,args.sequence_length_dbn,'cramersv',args.outcome_name,args.site,args.adjusted,args.ncases,targets,args.track)

        else:
            roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid = dbn_performance_metrics_tt(df,df2,dataTrainValid,args.sequence_length_dbn,'cramersv',args.outcome_name,args.site,args.adjusted,args.ncases,targets,args.track)


    elif chi2_ap_scores_valid >= cv_ap_scores_valid and chi2_ap_scores_valid >= ig_ap_scores_valid:
        print('Top Ranking Approach is Chi2')
        inter_cols,targets = dbn_cols_inter(chi2_intra_cols,args.sequence_length_dbn,args.cols_start,"",args.cols_end,"_",args.outcome_name)

        inter_structure = inter_struct(df,inter_cols,chi2_ns_list,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.sequence_length_dbn,args.track,args.max_fan_in)
        #graph visualization
        figure1,figure2 = draw_inter_structure(inter_structure,chi2_intra_cols,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.track,args.clipback,args.clipfront)
        #graph visualization
        figure1,figure2 =   draw_compact_structure(chi2_dag,inter_structure,chi2_intra_cols,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.track,args.clipback,args.clipfront)

        if args.outcome_name == 'egfr_reduction40_ge':

            dataTrainValid = DBNModel_TVT(df,df2,df3,inter_cols,inter_structure,args.sequence_length_dbn,args.max_iter,chi2_dag,chi2_ns_list,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.track,args.ncases)

        else:
            dataTrainValid = DBNModel_TT(df,df2,inter_cols,inter_structure,args.sequence_length_dbn,args.max_iter,chi2_dag,chi2_ns_list,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.track,args.ncases)


    # Evaluate DBN from best ranking approach BN

        if args.outcome_name == 'egfr_reduction40_ge':
            roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid, roc_aucs_test, ap_scores_test, fprs_test, tprs_test, thresholds_test = dbn_performance_metrics_tvt(df,df2,df3,dataTrainValid,args.sequence_length_dbn,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.ncases,targets,args.track)
        else:
            roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid = dbn_performance_metrics_tt(df,df2,dataTrainValid,args.sequence_length_dbn,'chisquarerank',args.outcome_name,args.site,args.adjusted,args.ncases,targets,args.track)




    elif ig_ap_scores_valid >= cv_ap_scores_valid and ig_ap_scores_valid >= chi2_ap_scores_valid:
        print('Top Ranking Approach is IG')
        inter_cols,targets = dbn_cols_inter(ig_intra_cols,args.sequence_length_dbn,args.cols_start,"",args.cols_end,"_",args.outcome_name)

        inter_structure = inter_struct(df,inter_cols,ig_ns_list,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.sequence_length_dbn,args.track,args.max_fan_in)
        #graph visualization
        figure1,figure2 = draw_inter_structure(inter_structure,ig_intra_cols,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.track,args.clipback,args.clipfront)
        #graph visualization
        figure1,figure2 =   draw_compact_structure(ig_dag,inter_structure,ig_intra_cols,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.track,args.clipback,args.clipfront)

        if args.outcome_name == 'egfr_reduction40_ge':

            dataTrainValid = DBNModel_TVT(df,df2,df3,inter_cols,inter_structure,args.sequence_length_dbn,args.max_iter,ig_dag,ig_ns_list,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.track,args.ncases)

        else:
            dataTrainValid = DBNModel_TT(df,df2,inter_cols,inter_structure,args.sequence_length_dbn,args.max_iter,ig_dag,ig_ns_list,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.track,args.ncases)


    # Evaluate DBN from best ranking approach BN
        if args.outcome_name == 'egfr_reduction40_ge':
            roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid, roc_aucs_test, ap_scores_test, fprs_test, tprs_test, thresholds_test = dbn_performance_metrics_tvt(df,df2,df3,dataTrainValid,args.sequence_length_dbn,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.ncases,targets,args.track)
        else:
            roc_aucs_valid, ap_scores_valid, fprs_valid, tprs_valid, thresholds_valid  = dbn_performance_metrics_tt(df,df2,dataTrainValid,args.sequence_length_dbn,'ig_mi_ranking',args.outcome_name,args.site,args.adjusted,args.ncases,targets,args.track)



    output = {'ap_scores': ap_scores}

    return output

##
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--cols_start',
      #default= '24hourperiod_0',
      type=str)
  parser.add_argument(
      '--cols_end',
      #default= '_2days',
      type=str)
  parser.add_argument(
      '--track',
      default= 'Track2:3',
      type=str)
  parser.add_argument(
        '--outcome_name',
        #default= 'AKI_BOS24',
        type=str)
  parser.add_argument(
        '--ncases',
        #default= 100,
        type=int)
  parser.add_argument(
        '--max_fan_in',
        #default= 2,
        type=int)
  parser.add_argument(
          '--sequence_length_dbn',
          #default= 4,
          type=int)
  parser.add_argument(
          '--max_iter',
          #default= 10,
          type=int)
  parser.add_argument(
        '--site',
        default= 'UCLA',
        type=str)
  parser.add_argument(
        '--adjusted',
        #default= '_w_Race_Adjusted_Variables',
        type=str)
  parser.add_argument(
       '--clipback',
       default= '',
       type=str)
  parser.add_argument(
       '--clipfront',
       default='',
       type=str)
  parser.add_argument(
      '--COLS',
      nargs="+",
      default=[])
  parser.add_argument(
      '--TARGET',
      nargs="+",
      default=[])

  args = parser.parse_args()

  # Call main function
  output = main(args)
