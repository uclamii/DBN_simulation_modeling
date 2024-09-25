"""
    This directory contains implementations of RAUS for unknown structure learning algorithms using an AKI dataset.

    To run the pipeline and return the RAUS top rank per track 1 or 2, run:

```shell
$ python track_compete_fullnetwork.py

"""
import glob
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
    - Track1 Output

    Output:
    - Top rank approach adjacency matrices

    """


    # Load Track 2 Outputs
    print('Starting Competing Tracks for Input to Full Network')
    print('Loading Output from Tracks')
    if args.track == 'Track2:3':
        cv_ap_scores_valid = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_ap_scores_valid")
        cv_intra_cols = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_intra_cols")
        cv_ns_list = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_ns_list")
        cv_dag = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_dag")
        df = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "df")
        df2 = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "df2")
        if args.outcome_name == "egfr_reduction40_ge":
            df3 = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "df3")
        chi2_ap_scores_valid = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_ap_scores_valid")
        chi2_intra_cols = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_intra_cols")
        chi2_ns_list = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_ns_list")
        chi2_dag = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_dag")
        ig_ap_scores_valid = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_ap_scores_valid")
        ig_intra_cols = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_intra_cols")
        ig_ns_list = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_ns_list")
        ig_dag = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_dag")

    #Load Track 1 Outputs
    if args.track == 'Track1':
        cv_ap_scores_valid = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_ap_scores_valid")
        cv_inter_cols = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_inter_cols")
        cv_inter_structure = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_inter_structure")
        chi2_ap_scores_valid = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_ap_scores_valid")
        chi2_inter_cols = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_inter_cols")
        chi2_inter_structure = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_inter_structure")
        ig_ap_scores_valid = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_ap_scores_valid")
        ig_inter_cols = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_inter_cols")
        ig_inter_structure = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_inter_structure")
        cv_ns_list = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "cv_ns_list")
        chi2_ns_list = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "chi2_ns_list")
        ig_ns_list = Pickleload("./Track_Inputs/" + args.outcome_name + "_" + args.site + "_" + args.track + "_" + args.adjusted + "_" + "ig_ns_list")


        ###################################################################
        ## Input Track 2 Top Ranking Approach into Track 3 DBN Component ##
        ###################################################################

        # Best Ranking from BN Used as Input to DBN Component (i.e., Ranking Approaches Compete)
    if args.track == 'Track2:3':
        if cv_ap_scores_valid >= chi2_ap_scores_valid and cv_ap_scores_valid >= ig_ap_scores_valid:
            cv_ap_path = "./Track_Inputs"
            cv_pattern = "*" + args.adjusted + "_" + "cv_ap_scores_valid"
            cv_path_list = glob.glob(f'{cv_ap_path}/{cv_pattern}')
            if cv_path_list:
                print(f'Found:{cv_path_list}')
                print('Top Ranking Approach is CV')
                print('Saving to FullNetwork Directory')

                dag2 = pd.DataFrame(cv_dag, columns=cv_intra_cols) # for easier review to redraw structures in another software
                dag2.index = cv_intra_cols # for easier review to redraw structures in another software
                path = './FullNetwork/' + args.site

                dag2.to_csv(path + '/' + 'cv_intra_structure_w_column_names_' + args.outcome_name + '_' + args.site + args.adjusted + '_' + args.track + '.csv') # for easier review to redraw structures in another software

                np.save(path + '/' + 'cv_intra_structure_' + args.outcome_name + '_' + args.site + args.adjusted+'_' + args.track, dag2)
            else:
                print(f'Did not find:{cv_pattern}')

        elif chi2_ap_scores_valid >= cv_ap_scores_valid and chi2_ap_scores_valid >= ig_ap_scores_valid:
            chi2_ap_path = "./Track_Inputs"
            chi2_pattern = "*" + args.adjusted + "_" + "chi2_ap_scores_valid"
            chi2_path_list = glob.glob(f'{chi2_ap_path}/{chi2_pattern}')
            if chi2_path_list:
                print(f'Found:{chi2_path_list}')
                print('Top Ranking Approach is Chi2')
                print('Saving to FullNetwork Directory')

                dag2 = pd.DataFrame(chi2_dag,   columns=chi2_intra_cols) # for easier review to redraw structures in another software
                dag2.index = chi2_intra_cols # for easier review to redraw structures in another software
                path = './FullNetwork/' + args.site

                dag2.to_csv(path + '/' + 'chi2_intra_structure_w_column_names_' + args.outcome_name + '_' + args.site + args.adjusted + '_' + args.track + '.csv') # for easier review to redraw structures in another software

                np.save(path + '/' + 'chi2_intra_structure_' + args.outcome_name + '_' + args.site + args.adjusted+'_' + args.track, dag2)
            else:
                print(f'Did not find:{chi2_pattern}')


        elif ig_ap_scores_valid >= cv_ap_scores_valid and ig_ap_scores_valid >= chi2_ap_scores_valid:
            ig_ap_path = "./Track_Inputs"
            ig_pattern = "*" + args.adjusted + "_" + "ig_ap_scores_valid"
            ig_path_list = glob.glob(f'{ig_ap_path}/{ig_pattern}')
            if ig_path_list:
                print(f'Found:{ig_path_list}')
                print('Top Ranking Approach is IG')
                print('Saving to FullNetwork Directory')

                dag2 = pd.DataFrame(ig_dag, columns=ig_intra_cols) # for easier review to redraw structures in another software
                dag2.index = ig_intra_cols # for easier review to redraw structures in another software
                path = './FullNetwork/' + args.site

                dag2.to_csv(path + '/' + 'ig_intra_structure_w_column_names_' + args.outcome_name + '_' + args.site + args.adjusted + '_' + args.track + '.csv') # for easier review to redraw structures in another software

                np.save(path + '/' + 'ig_intra_structure_' + args.outcome_name + '_' + args.site + args.adjusted+'_' + args.track, dag2)
            else:
                print(f'Did not find:{ig_pattern}')

    if args.track == 'Track1':
        if cv_ap_scores_valid[0] >= chi2_ap_scores_valid[0] and cv_ap_scores_valid[0] >= ig_ap_scores_valid[0]:
            cv_ap_path = "./Track_Inputs"
            cv_pattern = "*" + args.adjusted + "_" + "cv_ap_scores_valid"
            cv_path_list = glob.glob(f'{cv_ap_path}/{cv_pattern}')
            if cv_path_list:
                print(f'Found:{cv_path_list}')
                print('Top Ranking Approach is CV')
                print('Saving to FullNetwork Directory')
                intraLength = len(cv_ns_list)

                dag2 = pd.DataFrame(cv_inter_structure, columns=cv_inter_cols[intraLength:intraLength*2]) # for easier review to redraw structures in another software
                dag2.index = cv_inter_cols[:intraLength] # for easier review to redraw structures in another software
                path = './FullNetwork/' + args.site

                dag2.to_csv(path + '/' + 'cv_inter_structure_w_column_names_' + args.outcome_name + '_' + args.site + args.adjusted + '_' + args.track + '.csv') # for easier review to redraw structures in another software

                np.save(path + '/' + 'cv_inter_structure_' + args.outcome_name + '_' + args.site + args.adjusted+'_' + args.track, dag2)
            else:
                print(f'Did not find:{cv_pattern}')


        elif chi2_ap_scores_valid[0] >= cv_ap_scores_valid[0] and chi2_ap_scores_valid[0] >= ig_ap_scores_valid[0]:
            chi2_ap_path = "./Track_Inputs"
            chi2_pattern = "*" + args.adjusted + "_" + "chi2_ap_scores_valid"
            chi2_path_list = glob.glob(f'{chi2_ap_path}/{chi2_pattern}')
            if chi2_path_list:
                print(f'Found:{chi2_path_list}')
                print('Top Ranking Approach is Chi2')
                print('Saving to FullNetwork Directory')
                intraLength = len(chi2_ns_list)

                dag2 = pd.DataFrame(chi2_inter_structure, columns=chi2_inter_cols[intraLength:intraLength*2]) # for easier review to redraw structures in another software
                dag2.index = chi2_inter_cols[:intraLength] # for easier review to redraw structures in another software
                path = './FullNetwork/' + args.site

                dag2.to_csv(path + '/' + 'chi2_inter_structure_w_column_names_' + args.outcome_name + '_' + args.site + args.adjusted + '_' + args.track + '.csv') # for easier review to redraw structures in another software

                np.save(path + '/' + 'chi2_inter_structure_' + args.outcome_name + '_' + args.site + args.adjusted+'_' + args.track, dag2)
            else:
                print(f'Did not find:{chi2_pattern}')


        elif ig_ap_scores_valid[0] >= cv_ap_scores_valid[0] and ig_ap_scores_valid[0] >= chi2_ap_scores_valid[0]:
            ig_ap_path = "./Track_Inputs"
            ig_pattern = "*" + args.adjusted + "_" + "ig_ap_scores_valid"
            ig_path_list = glob.glob(f'{ig_ap_path}/{ig_pattern}')
            if ig_path_list:
                print(f'Found:{ig_path_list}')
                print('Top Ranking Approach is IG')
                print('Saving to FullNetwork Directory')
                intraLength = len(ig_ns_list)

                dag2 = pd.DataFrame(ig_inter_structure, columns=ig_inter_cols[intraLength:intraLength*2]) # for easier review to redraw structures in another software
                dag2.index = ig_inter_cols[:intraLength] # for easier review to redraw structures in another software
                path = './FullNetwork/' + args.site

                dag2.to_csv(path + '/' + 'ig_inter_structure_w_column_names_' + args.outcome_name + '_' + args.site + args.adjusted + '_' + args.track + '.csv') # for easier review to redraw structures in another software

                np.save(path + '/' + 'ig_inter_structure_' + args.outcome_name + '_' + args.site + args.adjusted+'_' + args.track, dag2)
            else:
                print(f'Did not find:{ig_pattern}')


    output = {'dag2': dag2}

    return output

##
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--track',
      default= 'Track2:3',
      type=str)
  parser.add_argument(
        '--outcome_name',
        #default= 'AKI_BOS24',
        type=str)
  parser.add_argument(
        '--site',
        default= 'UCLA',
        type=str)
  parser.add_argument(
        '--adjusted',
        #default= '_w_Race_Adjusted_Variables',
        type=str)

  args = parser.parse_args()

  # Call main function
  output = main(args)
