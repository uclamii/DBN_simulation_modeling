###### Global Variables ######

# # VM dataset paths
# UCLA_Train = "/home/ssm-user/data/egfr_reduction_data_ucla_train.csv"
# UCLA_Valid = "/home/ssm-user/data/egfr_reduction_data_ucla_valid.csv"
# UCLA_Test = "/home/ssm-user/data/egfr_reduction_data_ucla_test.csv"

# PSJH_Train = "/home/ssm-user/data/egfr_reduction_data_psjh_train.csv"
# PSJH_Valid = "/home/ssm-user/data/egfr_reduction_data_psjh_valid.csv"
# PSJH_Test = "/home/ssm-user/data/egfr_reduction_data_psjh_test.csv"

# Combined_Train = "/home/ssm-user/data/egfr_reduction_data_combined_train.csv"
# Combined_Valid = "/home/ssm-user/data/egfr_reduction_data_combined_valid.csv"
# Combined_Test = "/home/ssm-user/data/egfr_reduction_data_combined_test.csv"

# Local machine dataset paths
UCLA_Train = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_ucla_train.csv"
UCLA_Valid = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_ucla_valid.csv"
UCLA_Test = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_ucla_test.csv"

PSJH_Train = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_psjh_train.csv"
PSJH_Valid = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_psjh_valid.csv"
PSJH_Test = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_psjh_test.csv"

Combined_Train = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_combined_train.csv"
Combined_Valid = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_combined_valid.csv"
Combined_Test = "../Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_combined_test.csv"

### Substructure variables inputs ###
# Note, 2 sets of variable inputs for each substructure (race (scenarios 1 & 3) and no race (scenarios 2 & 4) structures)

# Contemporals (Scenario 1 and Scenario 3)
Contemporals_COLS = (
    "year0_mean"
    + " year0_scr_count"
    + " year0_hba1c_count"
    + " year0_hba1c_mean"
    + " year0_uacr_count"
    + " year0_uacr_mean"
    + " year0_upcr_count"
    + " year0_upcr_mean"
    + " year0_bp_count"
    + " year0_sbp_mean"
    + " year0_dbp_mean"
    + " year0_pp_mean"
    + " year0_map_mean"
    + " year0_av_count"
    + " year0_ipv_count"
    + " year0_aceiarb_coverage"
    #    + " year0_sglt2_coverage"
    #    + " year0_glp1_coverage"
    + " year0_nsaid_coverage"
    + " year0_ppi_coverage"
    #    + " year0_mra_coverage"
    + " demo_sex"
    + " demo_race_ethnicity_cat"
    + " ruca_4_class"
    + " ruca_7_class"
    + " study_entry_period_egfrckd_flag"
    + " study_entry_period_dxckd_flag"
    + " study_entry_period_albprockd_flag"
    + " study_entry_DM_flag"
    + " study_entry_PDM_flag"
    + " study_entry_HTN_flag"
    + " study_entry_aceiarb_flag"
    #    + " study_entry_sglt2_flag"
    #    + " study_entry_glp1_flag"
    + " study_entry_nsaid_flag"
    + " study_entry_ppi_flag"
    #    + " study_entry_mra_flag"
    + " study_entry_age"
)
# Contemporals no race (Scenario 2 and Scenario 4)
Contemporals_COLS_norace = (
    "year0_norace_mean"
    + " year0_scr_count"
    + " year0_hba1c_count"
    + " year0_hba1c_mean"
    + " year0_uacr_count"
    + " year0_uacr_mean"
    + " year0_upcr_count"
    + " year0_upcr_mean"
    + " year0_bp_count"
    + " year0_sbp_mean"
    + " year0_dbp_mean"
    + " year0_pp_mean"
    + " year0_map_mean"
    + " year0_av_count"
    + " year0_ipv_count"
    + " year0_aceiarb_coverage"
    #    + " year0_sglt2_coverage"
    #    + " year0_glp1_coverage"
    + " year0_nsaid_coverage"
    + " year0_ppi_coverage"
    #    + " year0_mra_coverage"
    + " demo_sex"
    + " demo_race_ethnicity_cat"
    + " ruca_4_class"
    + " ruca_7_class"
    + " study_entry_period_egfrckd_norace_flag"
    + " study_entry_period_dxckd_flag"
    + " study_entry_period_albprockd_flag"
    + " study_entry_DM_flag"
    + " study_entry_PDM_flag"
    + " study_entry_HTN_flag"
    + " study_entry_aceiarb_flag"
    #    + " study_entry_sglt2_flag"
    #    + " study_entry_glp1_flag"
    + " study_entry_nsaid_flag"
    + " study_entry_ppi_flag"
    #    + " study_entry_mra_flag"
    + " study_entry_age"
)
# Initial conditions (Scenario 1 and Scenario 3)
Initial_Condition_COLS = (
    "year0_mean"
    + " year0_scr_count"
    + " year0_hba1c_count"
    + " year0_hba1c_mean"
    + " year0_uacr_count"
    + " year0_uacr_mean"
    + " year0_upcr_count"
    + " year0_upcr_mean"
    + " year0_bp_count"
    + " year0_sbp_mean"
    + " year0_dbp_mean"
    + " year0_pp_mean"
    + " year0_map_mean"
    + " year0_av_count"
    + " year0_ipv_count"
    + " year0_aceiarb_coverage"
    #    + " year0_sglt2_coverage"
    #    + " year0_glp1_coverage"
    + " year0_nsaid_coverage"
    + " year0_ppi_coverage"
    #    + " year0_mra_coverage"
    + " year1_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_hba1c_mean"
    + " year1_uacr_count"
    + " year1_uacr_mean"
    + " year1_upcr_count"
    + " year1_upcr_mean"
    + " year1_bp_count"
    + " year1_sbp_mean"
    + " year1_dbp_mean"
    + " year1_pp_mean"
    + " year1_map_mean"
    + " year1_av_count"
    + " year1_ipv_count"
    + " year1_aceiarb_coverage"
    #    + " year1_sglt2_coverage"
    #    + " year1_glp1_coverage"
    + " year1_nsaid_coverage"
    + " year1_ppi_coverage"
    #    + " year1_mra_coverage"
    + " year1_reduction_40_ge"
)
# Initial conditions no race (Scenario 2 and Scenario 4)
Initial_Condition_COLS_norace = (
    "year0_norace_mean"
    + " year0_scr_count"
    + " year0_hba1c_count"
    + " year0_hba1c_mean"
    + " year0_uacr_count"
    + " year0_uacr_mean"
    + " year0_upcr_count"
    + " year0_upcr_mean"
    + " year0_bp_count"
    + " year0_sbp_mean"
    + " year0_dbp_mean"
    + " year0_pp_mean"
    + " year0_map_mean"
    + " year0_av_count"
    + " year0_ipv_count"
    + " year0_aceiarb_coverage"
    #    + " year0_sglt2_coverage"
    #    + " year0_glp1_coverage"
    + " year0_nsaid_coverage"
    + " year0_ppi_coverage"
    #    + " year0_mra_coverage"
    + " year1_norace_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_hba1c_mean"
    + " year1_uacr_count"
    + " year1_uacr_mean"
    + " year1_upcr_count"
    + " year1_upcr_mean"
    + " year1_bp_count"
    + " year1_sbp_mean"
    + " year1_dbp_mean"
    + " year1_pp_mean"
    + " year1_map_mean"
    + " year1_av_count"
    + " year1_ipv_count"
    + " year1_aceiarb_coverage"
    #    + " year1_sglt2_coverage"
    #    + " year1_glp1_coverage"
    + " year1_nsaid_coverage"
    + " year1_ppi_coverage"
    #    + " year1_mra_coverage"
    + " year1_reduction_40_ge"
)
# Intra for full DBN (Scenario 1 and Scenario 3)
Intra_DBN_COLS = (
    "year1_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_hba1c_mean"
    + " year1_uacr_count"
    + " year1_uacr_mean"
    + " year1_upcr_count"
    + " year1_upcr_mean"
    + " year1_bp_count"
    + " year1_sbp_mean"
    + " year1_dbp_mean"
    + " year1_pp_mean"
    + " year1_map_mean"
    + " year1_av_count"
    + " year1_ipv_count"
    + " year1_aceiarb_coverage"
    #    + " year1_sglt2_coverage"
    #    + " year1_glp1_coverage"
    + " year1_nsaid_coverage"
    + " year1_ppi_coverage"
    #    + " year1_mra_coverage"
)
# Intra for full DBN no race (Scenario 2 and Scenario 4)
Intra_DBN_COLS_norace = (
    "year1_norace_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_hba1c_mean"
    + " year1_uacr_count"
    + " year1_uacr_mean"
    + " year1_upcr_count"
    + " year1_upcr_mean"
    + " year1_bp_count"
    + " year1_sbp_mean"
    + " year1_dbp_mean"
    + " year1_pp_mean"
    + " year1_map_mean"
    + " year1_av_count"
    + " year1_ipv_count"
    + " year1_aceiarb_coverage"
    #    + " year1_sglt2_coverage"
    #    + " year1_glp1_coverage"
    + " year1_nsaid_coverage"
    + " year1_ppi_coverage"
    #    + " year1_mra_coverage"
)
# Inter for full DBN (Scenario 1 and Scenario 3)
Inter_DBN_COLS = (
    "year1_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_hba1c_mean"
    + " year1_uacr_count"
    + " year1_uacr_mean"
    + " year1_upcr_count"
    + " year1_upcr_mean"
    + " year1_bp_count"
    + " year1_sbp_mean"
    + " year1_dbp_mean"
    + " year1_pp_mean"
    + " year1_map_mean"
    + " year1_av_count"
    + " year1_ipv_count"
    + " year1_aceiarb_coverage"
    #    + " year1_sglt2_coverage"
    #    + " year1_glp1_coverage"
    + " year1_nsaid_coverage"
    + " year1_ppi_coverage"
    #    + " year1_mra_coverage"
)
# Inter for full DBN no race (Scenario 2 and Scenario 4)
Inter_DBN_COLS_norace = (
    "year1_norace_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_hba1c_mean"
    + " year1_uacr_count"
    + " year1_uacr_mean"
    + " year1_upcr_count"
    + " year1_upcr_mean"
    + " year1_bp_count"
    + " year1_sbp_mean"
    + " year1_dbp_mean"
    + " year1_pp_mean"
    + " year1_map_mean"
    + " year1_av_count"
    + " year1_ipv_count"
    + " year1_aceiarb_coverage"
    #    + " year1_sglt2_coverage"
    #    + " year1_glp1_coverage"
    + " year1_nsaid_coverage"
    + " year1_ppi_coverage"
    #    + " year1_mra_coverage"
)

## Substructure variable filters ##
# Note, 2 sets of substructure variable filters (counts (scenarios 1 & 2) and no counts (scenarios 3 & 4) structures)

# contemporals filter drop counts (Scenario 1 and Scenario 2)
Contemporals_Filter = (
    "year0_scr_count"
    + " year0_hba1c_count"
    + " year0_uacr_count"
    + " year0_upcr_count"
    + " year0_bp_count"
    + " year0_av_count"
    + " year0_ipv_count"
    + " ruca_7_class"
)
# contemporals filter keep counts (Scenario 3 and Scenario 4)
Contemporals_Filter_keep_counts = "ruca_7_class"
# initial condition filter drop counts (Scenario 1 and Scenario 2)
Initial_Condition_Filter = (
    "year0_scr_count"
    + " year0_hba1c_count"
    + " year0_uacr_count"
    + " year0_upcr_count"
    + " year0_bp_count"
    + " year0_av_count"
    + " year0_ipv_count"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_uacr_count"
    + " year1_upcr_count"
    + " year1_bp_count"
    + " year1_av_count"
    + " year1_ipv_count"
)
# intial conditions filter keep counts (Scenario 3 and Scenario 4)
Initial_Condition_Filter_keep_counts = "None"
# intra for full DBN filter drop counts (Scenario 1 and Scenario 2)
Intra_DBN_Filter = (
    "year1_scr_count"
    + " year1_hba1c_count"
    + " year1_uacr_count"
    + " year1_upcr_count"
    + " year1_bp_count"
    + " year1_av_count"
    + " year1_ipv_count"
)
# intra for full DBN filter keep counts (Scenario 3 and Scenario 4)
Intra_DBN_Filter_keep_counts = "None"
# inter for full DBN filter drop counts (Scenario 1 and Scenario 2)
Inter_DBN_Filter = (
    "year1_upcr_mean"
    + " year1_uacr_mean"
    + " year1_pp_mean"
    + " year1_dbp_mean"
    + " year1_sbp_mean"
    + " year1_map_mean"
    + " year1_scr_count"
    + " year1_hba1c_count"
    + " year1_uacr_count"
    + " year1_upcr_count"
    + " year1_bp_count"
    + " year1_av_count"
    + " year1_ipv_count"
)
# inter for full DBN filter keep counts (Scenario 3 and Scenario 4)
Inter_DBN_Filter_keep_counts = (
    "year1_upcr_mean"
    + " year1_uacr_mean"
    + " year1_pp_mean"
    + " year1_dbp_mean"
    + " year1_sbp_mean"
    + " year1_map_mean"
)


# Substructure identifiers
Contemporals_Substructure = "Contemporals_Substructure"
Initial_Condition_Substructure = "Initial_Condition_Substructure"
Intra_DBN_Substructure = "Intra_DBN_Substructure"
Inter_DBN_Substructure = "Inter_DBN_Substructure"

# Substructure targets
TARGET_Contemporals = "year1_reduction_40_ge"
TARGET_InitialCondition_Intra_Inter = "year2_reduction_40_ge"  # note for the inter substructure the target will adjust and go up to the sequence length

# what the temporal cols start with
Contemporals_InitialCondition_COLS_Start = "year0"
Intra_Inter_COLS_Start = "year1"

# what the temporal outcome col ends with
COLS_End = "_ge"

# outcome identifier (if 'egfr_reduction40_ge' loads train, valid, and test (TVT) sets and appends outcome identifier to output files)
Outcome = "egfr_reduction40_ge"

# Site identifiers
UCLA = "UCLA"
UCLA_counts = "UCLA_counts"
UCLA_counts_no_race = "UCLA_counts_no_race"
UCLA_no_race = "UCLA_no_race"
PSJH = "PSJH"
PSJH_counts = "PSJH_counts"
PSJH_counts_no_race = "PSJH_counts_no_race"
PSJH_no_race = "PSJH_no_race"
Combined = "Combined"
Combined_counts = "Combined_counts"
Combined_counts_no_race = "Combined_counts_no_race"
Combined_no_race = "Combined_no_race"

# Clipfront/Clipback identifiers
# Clipfront = "None"
# Clipback = "None"

# sequence lengths
seq_len_bn = "1"
seq_len_dbn = "4"

# iterations
max_iterations = "50"

# maximum in-degree edges per node
max_fan_in = "2"

# rank filter
rank_filter = "1"

# track
Track1 = "Track1"
Track2 = "Track2:3"

# rapid deploy on onlyvalid
# TODO: flags to optimize speed
# Onlyvalid = "1"
