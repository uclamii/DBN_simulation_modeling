import pandas as pd

## paths to adjacency matrics (as dataframes) with row (the index) and column names
#track2
path_1 = './RAUS/intraStructures/matrix_form/chisquarerank_intra_structure_w_column_names_egfr_reduction40_ge_UCLAw_yr0-1observations_yr1outasobs_w_Race_Adjusted_Variables_Track2:3.csv'
path_2 = './RAUS/intraStructures/matrix_form/cramersv_intra_structure_w_column_names_egfr_reduction40_ge_UCLAw_yr0-1observations_yr1outasobs_w_Race_Adjusted_Variables_Track2:3.csv'
path_3 = './RAUS/intraStructures/matrix_form/ig_mi_ranking_intra_structure_w_column_names_egfr_reduction40_ge_UCLAw_yr0-1observations_yr1outasobs_w_Race_Adjusted_Variables_Track2:3.csv'
path_4 = './RAUS/intraStructures/matrix_form/chisquarerank_intra_structure_w_column_names_egfr_reduction40_ge_UCLAwstatic_yr0observations_w_Race_Adjusted_Variables_Track2:3.csv'
path_5 = './RAUS/intraStructures/matrix_form/cramersv_intra_structure_w_column_names_egfr_reduction40_ge_UCLAwstatic_yr0observations_w_Race_Adjusted_Variables_Track2:3.csv'
path_6 = './RAUS/intraStructures/matrix_form/ig_mi_ranking_intra_structure_w_column_names_egfr_reduction40_ge_UCLAwstatic_yr0observations_w_Race_Adjusted_Variables_Track2:3.csv'
path_7 = './RAUS/intraStructures/matrix_form/chisquarerank_intra_structure_w_column_names_egfr_reduction40_ge_UCLAw_yr1observations_w_Race_Adjusted_Variables_Track2:3.csv'
path_8 = './RAUS/intraStructures/matrix_form/cramersv_intra_structure_w_column_names_egfr_reduction40_ge_UCLAw_yr1observations_w_Race_Adjusted_Variables_Track2:3.csv'
path_9 = './RAUS/intraStructures/matrix_form/ig_mi_ranking_intra_structure_w_column_names_egfr_reduction40_ge_UCLAw_yr1observations_w_Race_Adjusted_Variables_Track2:3.csv'

#track1
path_10 = './RAUS/interStructures/matrix_form/chisquarerank_inter_structure_w_column_names_egfr_reduction40_ge_UCLA_w_Race_Adjusted_Variables_Track1.csv'
path_11 = './RAUS/interStructures/matrix_form/cramersv_inter_structure_w_column_names_egfr_reduction40_ge_UCLA_w_Race_Adjusted_Variables_Track1.csv'
path_12 = './RAUS/interStructures/matrix_form/ig_mi_ranking_inter_structure_w_column_names_egfr_reduction40_ge_UCLA_w_Race_Adjusted_Variables_Track1.csv'

## Track2 ##
# year0 and year1 observations (intra-structure)
chi2_UCLA_intra_struct_yr0_1 = pd.read_csv(path_1,index_col='Unnamed: 0')
cv_UCLA_intra_struct_yr0_1 = pd.read_csv(path_2,index_col='Unnamed: 0')
ig_UCLA_intra_struct_yr0_1 = pd.read_csv(path_3,index_col='Unnamed: 0')

# year0 and static observations (intra-structure)
chi2_UCLA_intra_struct_yr0_s = pd.read_csv(path_4,index_col='Unnamed: 0')
cv_UCLA_intra_struct_yr0_s = pd.read_csv(path_5,index_col='Unnamed: 0')
ig_UCLA_intra_struct_yr0_s = pd.read_csv(path_6,index_col='Unnamed: 0')

# year1 observations (intra-structure)
chi2_UCLA_intra_struct_yr1 = pd.read_csv(path_7,index_col='Unnamed: 0')
cv_UCLA_intra_struct_yr1 = pd.read_csv(path_8,index_col='Unnamed: 0')
ig_UCLA_intra_struct_yr1 = pd.read_csv(path_9,index_col='Unnamed: 0')

## Track 1 ##
# year1-4 observations (inter-structure)
chi2_UCLA_inter_struct_yr1_4 = pd.read_csv(path_10,index_col='Unnamed: 0')
cv_UCLA_inter_struct_yr1_4 = pd.read_csv(path_11,index_col='Unnamed: 0')
ig_UCLA_inter_struct_yr1_4 = pd.read_csv(path_12,index_col='Unnamed: 0')
