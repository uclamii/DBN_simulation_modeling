#######################
#     Global Vars     #
#######################
DATA_PATH = (
    "/Users/davina/Documents/Data/"
    "Rapiddecline project data20191220103345/CURE CKD January 2019/"
)

# Categorical columns
CAT_COLS = [
    "patient_id",
    "site_source_cat",
    "sex_code",
    "race_ethnicity_cat",
    "ruca_4_class",
    "ruca_7_class",
    # "patient_state",
    # "patient_country"
    # "vital_status_code",
    "egfr_entry_period_egfrckd_flag",
    "egfr_entry_period_egfrckd_norace_flag",
    "egfr_entry_period_dxckd_flag",
    "egfr_entry_period_albprockd_flag",
    "egfr_entry_dm_flag",
    "egfr_entry_pdm_flag",
    "egfr_entry_htn_flag",
    "egfr_entry_aceiarb_flag",
    # "egfr_entry_sglt2_flag",
    # "egfr_entry_glp1_flag",
    "egfr_entry_nsaid_flag",
    "egfr_entry_ppi_flag",
    # "egfr_entry_mra_flag",
    "egfr_reduction30_flag",
    "egfr_reduction40_flag",
    "egfr_reduction50_flag",
    "egfr_reduction57_flag",
    "egfr_reduction30_norace_flag",
    "egfr_reduction40_norace_flag",
    "egfr_reduction50_norace_flag",
    "egfr_reduction57_norace_flag",
]

TIME_ZERO_COLS = [
    # Note: hba1c is the same as the _a1c for the following years
    "egfr_entry_period_hba1c_count",
    "egfr_entry_period_hba1c_mean",
    "egfr_entry_period_uacr_count",
    "egfr_entry_period_uacr_mean",
    "egfr_entry_period_upcr_count",
    "egfr_entry_period_upcr_mean",
    "egfr_entry_period_bp_count",
    "egfr_entry_period_sbp_mean",
    "egfr_entry_period_dbp_mean",
    "egfr_entry_period_pp_mean",
    "egfr_entry_period_map_mean",
    "egfr_entry_period_av_count",
    "egfr_entry_period_ipv_count",
    "egfr_entry_period_aceiarb_coverage",
    # "egfr_entry_period_sglt2_coverage",
    # "egfr_entry_period_glp1_coverage",
    "egfr_entry_period_nsaid_coverage",
    "egfr_entry_period_ppi_coverage",
    # "egfr_entry_period_mra_coverage",
    "egfr_entry_period_scr_count",
    "egfr_entry_period_mean",
    "egfr_entry_period_norace_mean",
]

# Continuous columns at entry of study
CTN_ENTRY_COLS = [
    "egfr_entry_age",
    "egfr_years_followed",
    "egfr_slope_year",
    # "study_entry_egfr", #not present in the updated registry
    # "study_entry_a1c", #not present in the updated registry
    # "study_entry_sbp", #not present in the updated registry
]

# Continuous columns
CTN_COLS = (
    ["egfr_year" + str(i) + "_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_norace_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_reduction" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_norace_reduction" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_scr_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_hba1c_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_hba1c_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_uacr_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_uacr_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_upcr_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_upcr_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_bp_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_sbp_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_dbp_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_pp_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_map_mean" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_av_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_ipv_count" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_aceiarb_coverage" for i in range(1, 14)]
    # + ["egfr_year" + str(i) + "_sglt2_coverage" for i in range(1, 14)]
    # + ["egfr_year" + str(i) + "_glp1_coverage" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_nsaid_coverage" for i in range(1, 14)]
    + ["egfr_year" + str(i) + "_ppi_coverage" for i in range(1, 14)]
    # + ["egfr_year" + str(i) + "_mra_coverage" for i in range(1, 14)]
)


RACE_COLS = [
    "ethnicity_Not Categorized",
    "ethnicity_White Non-Latino",
    "ethnicity_White Latino",
    "ethnicity_Black",
    "ethnicity_Asian",
    "ethnicity_American Indian",
    "ethnicity_Hawaiian",
    "ethnicity_Other",
]

# Used by load_features_and_labels
DEFAULT_TARGET = "decl_base2any_ge_40_bin"


# Categorical columns
# Categorical columns
old_CAT_COLS = [
    "patient_id",
    "site_source_cat",
    "demo_sex",
    "demo_race_ethnicity_cat",
    "ruca_4_class",
    "ruca_7_class",
    "study_entry_period_egfrckd_flag",
    "study_entry_period_egfrckd_norace_flag",
    "study_entry_period_dxckd_flag",
    "study_entry_period_albprockd_flag",
    "study_entry_DM_flag",
    "study_entry_PDM_flag",
    "study_entry_HTN_flag",
    "study_entry_aceiarb_flag",
    # "study_entry_sglt2_flag",
    # "study_entry_glp1_flag",
    "study_entry_nsaid_flag",
    "study_entry_ppi_flag",
    # "study_entry_mra_flag",
    "egfr_reduction30_flag",
    "egfr_reduction40_flag",
    "egfr_reduction50_flag",
    "egfr_reduction57_flag",
    "egfr_reduction30_norace_flag",
    "egfr_reduction40_norace_flag",
    "egfr_reduction50_norace_flag",
    "egfr_reduction57_norace_flag",
]

old_TIME_ZERO_COLS = [
    # Note: hba1c is the same as the _a1c for the following years
    "time_zero_hba1c_count",
    "time_zero_hba1c_mean",
    "time_zero_uacr_count",
    "time_zero_uacr_mean",
    "time_zero_upcr_count",
    "time_zero_upcr_mean",
    "time_zero_bp_count",
    "time_zero_sbp_mean",
    "time_zero_dbp_mean",
    "time_zero_pp_mean",
    "time_zero_map_mean",
    "time_zero_av_count",
    "time_zero_ipv_count",
    "time_zero_aceiarb_coverage",
    # "time_zero_sglt2_coverage",
    # "time_zero_glp1_coverage",
    "time_zero_nsaid_coverage",
    "time_zero_ppi_coverage",
    # "time_zero_mra_coverage",
    "time_zero_scr_count",
    "time_zero_mean",
    "time_zero_norace_mean",
]

# Continuous columns at entry of study
old_CTN_ENTRY_COLS = [
    "study_entry_age",
    "egfr_years_followed",
    "egfr_slope_year",
    # "study_entry_egfr", #not present in the updated registry
    # "study_entry_a1c", #not present in the updated registry
    # "study_entry_sbp", #not present in the updated registry
]

# Continuous columns
old_CTN_COLS = (
    ["year" + str(i) + "_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_norace_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_reduction" for i in range(1, 14)]
    + ["year" + str(i) + "_norace_reduction" for i in range(1, 14)]
    + ["year" + str(i) + "_scr_count" for i in range(1, 14)]
    + ["year" + str(i) + "_hba1c_count" for i in range(1, 14)]
    + ["year" + str(i) + "_hba1c_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_uacr_count" for i in range(1, 14)]
    + ["year" + str(i) + "_uacr_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_upcr_count" for i in range(1, 14)]
    + ["year" + str(i) + "_upcr_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_bp_count" for i in range(1, 14)]
    + ["year" + str(i) + "_sbp_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_dbp_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_pp_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_map_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_av_count" for i in range(1, 14)]
    + ["year" + str(i) + "_ipv_count" for i in range(1, 14)]
    + ["year" + str(i) + "_aceiarb_coverage" for i in range(1, 14)]
    # + ["year" + str(i) + "_sglt2_coverage" for i in range(1, 14)]
    # + ["year" + str(i) + "_glp1_coverage" for i in range(1, 14)]
    + ["year" + str(i) + "_nsaid_coverage" for i in range(1, 14)]
    + ["year" + str(i) + "_ppi_coverage" for i in range(1, 14)]
    # + ["year" + str(i) + "_mra_coverage" for i in range(1, 14)]
)

old_RACE_COLS = [
    "Not Otherwise Categorized",
    "White Non-Latino",
    "White Latino",
    "Black",
    "Asian",
    "American Indian/Alaskan Native",
    "Native Hawaiian or Other Pacific Islander",
    "Other",
]


######## remapping new to old columns
assert len(CAT_COLS) == len(old_CAT_COLS)
assert len(TIME_ZERO_COLS) == len(old_TIME_ZERO_COLS)
assert len(CTN_ENTRY_COLS) == len(old_CTN_ENTRY_COLS)
assert len(CTN_COLS) == len(old_CTN_COLS)
assert len(RACE_COLS) == len(old_RACE_COLS)


new_cols = CAT_COLS + TIME_ZERO_COLS + CTN_ENTRY_COLS + CTN_COLS + RACE_COLS
old_cols = (
    old_CAT_COLS
    + old_TIME_ZERO_COLS
    + old_CTN_ENTRY_COLS
    + old_CTN_COLS
    + old_RACE_COLS
)

columnsMapping = {val1: val2 for val1, val2 in zip(new_cols, old_cols)}


####### FIRST STRUCTURE

STATIC_BASELINE_VARS_DBN = [
    # "patient_id",
    "site_source_cat",
    "demo_sex",
    "demo_race_ethnicity_cat",
    "ruca_4_class",
    "ruca_7_class",
    "study_entry_period_egfrckd_flag",
    "study_entry_period_egfrckd_norace_flag",
    "study_entry_period_dxckd_flag",
    "study_entry_period_albprockd_flag",
    "study_entry_DM_flag",
    "study_entry_PDM_flag",
    "study_entry_HTN_flag",
    "study_entry_aceiarb_flag",
    # "study_entry_sglt2_flag",
    # "study_entry_glp1_flag",
    "study_entry_nsaid_flag",
    "study_entry_ppi_flag",
    # "study_entry_mra_flag",
    "study_entry_age",
    # "egfr_reduction30_flag",
    # "egfr_reduction40_flag",
    # "egfr_reduction50_flag",
    # "egfr_reduction57_flag",
    # "egfr_reduction30_norace_flag",
    # "egfr_reduction40_norace_flag",
    # "egfr_reduction50_norace_flag",
    # "egfr_reduction57_norace_flag",
]


TIME_ZERO_DBN_VARS = [
    # Note: hba1c is the same as the _a1c for the following years
    "time_zero_hba1c_count",
    "time_zero_hba1c_mean",
    "time_zero_uacr_count",
    "time_zero_uacr_mean",
    "time_zero_upcr_count",
    "time_zero_upcr_mean",
    "time_zero_bp_count",
    "time_zero_sbp_mean",
    "time_zero_dbp_mean",
    "time_zero_pp_mean",
    "time_zero_map_mean",
    "time_zero_av_count",
    "time_zero_ipv_count",
    "time_zero_aceiarb_coverage",
    # "time_zero_sglt2_coverage",
    # "time_zero_glp1_coverage",
    "time_zero_nsaid_coverage",
    "time_zero_ppi_coverage",
    # "time_zero_mra_coverage",
    "time_zero_scr_count",
    "time_zero_mean",
    "time_zero_norace_mean",
]


####### SECOND STRUCTURE
# Structure between the static + time zero vars and the year 1 vars
# Continuous columns
TEMPORAL_DBN_VARS = (
    ["year" + str(i) + "_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_norace_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_reduction" for i in range(1, 14)]
    + ["year" + str(i) + "_norace_reduction" for i in range(1, 14)]
    + ["year" + str(i) + "_scr_count" for i in range(1, 14)]
    + ["year" + str(i) + "_hba1c_count" for i in range(1, 14)]
    + ["year" + str(i) + "_hba1c_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_uacr_count" for i in range(1, 14)]
    + ["year" + str(i) + "_uacr_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_upcr_count" for i in range(1, 14)]
    + ["year" + str(i) + "_upcr_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_bp_count" for i in range(1, 14)]
    + ["year" + str(i) + "_sbp_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_dbp_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_pp_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_map_mean" for i in range(1, 14)]
    + ["year" + str(i) + "_av_count" for i in range(1, 14)]
    + ["year" + str(i) + "_ipv_count" for i in range(1, 14)]
    + ["year" + str(i) + "_aceiarb_coverage" for i in range(1, 14)]
    # + ["year" + str(i) + "_sglt2_coverage" for i in range(1, 14)]
    # + ["year" + str(i) + "_glp1_coverage" for i in range(1, 14)]
    + ["year" + str(i) + "_nsaid_coverage" for i in range(1, 14)]
    + ["year" + str(i) + "_ppi_coverage" for i in range(1, 14)]
    # + ["year" + str(i) + "_mra_coverage" for i in range(1, 14)]
)


###### THIRD STRUCTURE
### temporal variables
# inta-slice structure
# inter-rslice structure
