import pandas as pd

### UCLA
# Get ANNUAL eGFR Reduction40 Flags... (This is version 3 of the egfr_reduction40 label)
df_train_ucla_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_UCLA_train.csv",
    low_memory=False,
)
df_valid_ucla_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_UCLA_valid.csv",
    low_memory=False,
)
df_test_ucla_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_UCLA_test.csv",
    low_memory=False,
)

### create yr0 observations from time_zero observations
# train set
df_train_ucla_v2["year0_hba1c_count"] = df_train_ucla_v2.time_zero_hba1c_count
df_train_ucla_v2["year0_hba1c_mean"] = df_train_ucla_v2.time_zero_hba1c_mean
df_train_ucla_v2["year0_uacr_count"] = df_train_ucla_v2.time_zero_uacr_count
df_train_ucla_v2["year0_upcr_count"] = df_train_ucla_v2.time_zero_upcr_count
df_train_ucla_v2["year0_bp_count"] = df_train_ucla_v2.time_zero_bp_count
df_train_ucla_v2["year0_av_count"] = df_train_ucla_v2.time_zero_av_count
df_train_ucla_v2["year0_ipv_count"] = df_train_ucla_v2.time_zero_ipv_count
df_train_ucla_v2["year0_aceiarb_coverage"] = df_train_ucla_v2.time_zero_aceiarb_coverage
# df_train_ucla_v2["year0_sglt2_coverage"] = df_train_ucla_v2.time_zero_sglt2_coverage
# df_train_ucla_v2["year0_glp1_coverage"] = df_train_ucla_v2.time_zero_glp1_coverage
df_train_ucla_v2["year0_nsaid_coverage"] = df_train_ucla_v2.time_zero_nsaid_coverage
df_train_ucla_v2["year0_ppi_coverage"] = df_train_ucla_v2.time_zero_ppi_coverage
# df_train_ucla_v2["year0_mra_coverage"] = df_train_ucla_v2.time_zero_mra_coverage
df_train_ucla_v2["year0_scr_count"] = df_train_ucla_v2.time_zero_scr_count
df_train_ucla_v2["year0_norace_mean"] = df_train_ucla_v2.time_zero_norace_mean
df_train_ucla_v2["year0_mean"] = df_train_ucla_v2.time_zero_mean
df_train_ucla_v2["year0_uacr_mean"] = df_train_ucla_v2.time_zero_uacr_mean
df_train_ucla_v2["year0_upcr_mean"] = df_train_ucla_v2.time_zero_upcr_mean
df_train_ucla_v2["year0_sbp_mean"] = df_train_ucla_v2.time_zero_sbp_mean
df_train_ucla_v2["year0_dbp_mean"] = df_train_ucla_v2.time_zero_dbp_mean
df_train_ucla_v2["year0_pp_mean"] = df_train_ucla_v2.time_zero_pp_mean
df_train_ucla_v2["year0_map_mean"] = df_train_ucla_v2.time_zero_map_mean
# valid set
df_valid_ucla_v2["year0_hba1c_count"] = df_valid_ucla_v2.time_zero_hba1c_count
df_valid_ucla_v2["year0_hba1c_mean"] = df_valid_ucla_v2.time_zero_hba1c_mean
df_valid_ucla_v2["year0_uacr_count"] = df_valid_ucla_v2.time_zero_uacr_count
df_valid_ucla_v2["year0_upcr_count"] = df_valid_ucla_v2.time_zero_upcr_count
df_valid_ucla_v2["year0_bp_count"] = df_valid_ucla_v2.time_zero_bp_count
df_valid_ucla_v2["year0_av_count"] = df_valid_ucla_v2.time_zero_av_count
df_valid_ucla_v2["year0_ipv_count"] = df_valid_ucla_v2.time_zero_ipv_count
df_valid_ucla_v2["year0_aceiarb_coverage"] = df_valid_ucla_v2.time_zero_aceiarb_coverage
# df_valid_ucla_v2["year0_sglt2_coverage"] = df_valid_ucla_v2.time_zero_sglt2_coverage
# df_valid_ucla_v2["year0_glp1_coverage"] = df_valid_ucla_v2.time_zero_glp1_coverage
df_valid_ucla_v2["year0_nsaid_coverage"] = df_valid_ucla_v2.time_zero_nsaid_coverage
df_valid_ucla_v2["year0_ppi_coverage"] = df_valid_ucla_v2.time_zero_ppi_coverage
# df_valid_ucla_v2["year0_mra_coverage"] = df_valid_ucla_v2.time_zero_mra_coverage
df_valid_ucla_v2["year0_scr_count"] = df_valid_ucla_v2.time_zero_scr_count
df_valid_ucla_v2["year0_mean"] = df_valid_ucla_v2.time_zero_mean
df_valid_ucla_v2["year0_norace_mean"] = df_valid_ucla_v2.time_zero_norace_mean
df_valid_ucla_v2["year0_uacr_mean"] = df_valid_ucla_v2.time_zero_uacr_mean
df_valid_ucla_v2["year0_upcr_mean"] = df_valid_ucla_v2.time_zero_upcr_mean
df_valid_ucla_v2["year0_sbp_mean"] = df_valid_ucla_v2.time_zero_sbp_mean
df_valid_ucla_v2["year0_dbp_mean"] = df_valid_ucla_v2.time_zero_dbp_mean
df_valid_ucla_v2["year0_pp_mean"] = df_valid_ucla_v2.time_zero_pp_mean
df_valid_ucla_v2["year0_map_mean"] = df_valid_ucla_v2.time_zero_map_mean
# test set
df_test_ucla_v2["year0_hba1c_count"] = df_test_ucla_v2.time_zero_hba1c_count
df_test_ucla_v2["year0_hba1c_mean"] = df_test_ucla_v2.time_zero_hba1c_mean
df_test_ucla_v2["year0_uacr_count"] = df_test_ucla_v2.time_zero_uacr_count
df_test_ucla_v2["year0_upcr_count"] = df_test_ucla_v2.time_zero_upcr_count
df_test_ucla_v2["year0_bp_count"] = df_test_ucla_v2.time_zero_bp_count
df_test_ucla_v2["year0_av_count"] = df_test_ucla_v2.time_zero_av_count
df_test_ucla_v2["year0_ipv_count"] = df_test_ucla_v2.time_zero_ipv_count
df_test_ucla_v2["year0_aceiarb_coverage"] = df_test_ucla_v2.time_zero_aceiarb_coverage
# df_test_ucla_v2["year0_sglt2_coverage"] = df_test_ucla_v2.time_zero_sglt2_coverage
# df_test_ucla_v2["year0_glp1_coverage"] = df_test_ucla_v2.time_zero_glp1_coverage
df_test_ucla_v2["year0_nsaid_coverage"] = df_test_ucla_v2.time_zero_nsaid_coverage
df_test_ucla_v2["year0_ppi_coverage"] = df_test_ucla_v2.time_zero_ppi_coverage
# df_test_ucla_v2["year0_mra_coverage"] = df_test_ucla_v2.time_zero_mra_coverage
df_test_ucla_v2["year0_scr_count"] = df_test_ucla_v2.time_zero_scr_count
df_test_ucla_v2["year0_mean"] = df_test_ucla_v2.time_zero_mean
df_test_ucla_v2["year0_norace_mean"] = df_test_ucla_v2.time_zero_norace_mean
df_test_ucla_v2["year0_uacr_mean"] = df_test_ucla_v2.time_zero_uacr_mean
df_test_ucla_v2["year0_upcr_mean"] = df_test_ucla_v2.time_zero_upcr_mean
df_test_ucla_v2["year0_sbp_mean"] = df_test_ucla_v2.time_zero_sbp_mean
df_test_ucla_v2["year0_dbp_mean"] = df_test_ucla_v2.time_zero_dbp_mean
df_test_ucla_v2["year0_pp_mean"] = df_test_ucla_v2.time_zero_pp_mean
df_test_ucla_v2["year0_map_mean"] = df_test_ucla_v2.time_zero_map_mean

# train set
df_train_ucla_v2["year1_reduction_40_wr"] = df_train_ucla_v2.year1_reduction_40_ge
df_train_ucla_v2["year2_reduction_40_wr"] = df_train_ucla_v2.year2_reduction_40_ge
df_train_ucla_v2["year3_reduction_40_wr"] = df_train_ucla_v2.year3_reduction_40_ge
df_train_ucla_v2["year4_reduction_40_wr"] = df_train_ucla_v2.year4_reduction_40_ge
df_train_ucla_v2["year5_reduction_40_wr"] = df_train_ucla_v2.year5_reduction_40_ge

df_train_ucla_v2.year1_reduction_40_wr.value_counts()
df_train_ucla_v2.year2_reduction_40_wr.value_counts()
df_train_ucla_v2.year3_reduction_40_wr.value_counts()
df_train_ucla_v2.year4_reduction_40_wr.value_counts()
df_train_ucla_v2.year5_reduction_40_wr.value_counts()


df_train_ucla_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_ucla_train.csv"
)


# valid set
df_valid_ucla_v2["year1_reduction_40_wr"] = df_valid_ucla_v2.year1_reduction_40_ge
df_valid_ucla_v2["year2_reduction_40_wr"] = df_valid_ucla_v2.year2_reduction_40_ge
df_valid_ucla_v2["year3_reduction_40_wr"] = df_valid_ucla_v2.year3_reduction_40_ge
df_valid_ucla_v2["year4_reduction_40_wr"] = df_valid_ucla_v2.year4_reduction_40_ge
df_valid_ucla_v2["year5_reduction_40_wr"] = df_valid_ucla_v2.year5_reduction_40_ge

df_valid_ucla_v2.year1_reduction_40_wr.value_counts()
df_valid_ucla_v2.year2_reduction_40_wr.value_counts()
df_valid_ucla_v2.year3_reduction_40_wr.value_counts()
df_valid_ucla_v2.year4_reduction_40_wr.value_counts()
df_valid_ucla_v2.year5_reduction_40_wr.value_counts()

df_valid_ucla_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_ucla_valid.csv"
)

# test set
df_test_ucla_v2["year1_reduction_40_wr"] = df_test_ucla_v2.year1_reduction_40_ge
df_test_ucla_v2["year2_reduction_40_wr"] = df_test_ucla_v2.year2_reduction_40_ge
df_test_ucla_v2["year3_reduction_40_wr"] = df_test_ucla_v2.year3_reduction_40_ge
df_test_ucla_v2["year4_reduction_40_wr"] = df_test_ucla_v2.year4_reduction_40_ge
df_test_ucla_v2["year5_reduction_40_wr"] = df_test_ucla_v2.year5_reduction_40_ge

df_test_ucla_v2.year1_reduction_40_wr.value_counts()
df_test_ucla_v2.year2_reduction_40_wr.value_counts()
df_test_ucla_v2.year3_reduction_40_wr.value_counts()
df_test_ucla_v2.year4_reduction_40_wr.value_counts()
df_test_ucla_v2.year5_reduction_40_wr.value_counts()

df_test_ucla_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_ucla_test.csv"
)

#######################

import pandas as pd

### Prov
# Get ANNUAL eGFR Reduction40 Flags... (This is version 3 of the egfr_reduction40 label)
df_train_Prov_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_Prov_train.csv",
    low_memory=False,
)
df_valid_Prov_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_Prov_valid.csv",
    low_memory=False,
)
df_test_Prov_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_Prov_test.csv",
    low_memory=False,
)

### create yr0 observations from time_zero observations
# train set
df_train_Prov_v2["year0_hba1c_count"] = df_train_Prov_v2.time_zero_hba1c_count
df_train_Prov_v2["year0_hba1c_mean"] = df_train_Prov_v2.time_zero_hba1c_mean
df_train_Prov_v2["year0_uacr_count"] = df_train_Prov_v2.time_zero_uacr_count
df_train_Prov_v2["year0_upcr_count"] = df_train_Prov_v2.time_zero_upcr_count
df_train_Prov_v2["year0_bp_count"] = df_train_Prov_v2.time_zero_bp_count
df_train_Prov_v2["year0_av_count"] = df_train_Prov_v2.time_zero_av_count
df_train_Prov_v2["year0_ipv_count"] = df_train_Prov_v2.time_zero_ipv_count
df_train_Prov_v2["year0_aceiarb_coverage"] = df_train_Prov_v2.time_zero_aceiarb_coverage
# df_train_Prov_v2["year0_sglt2_coverage"] = df_train_Prov_v2.time_zero_sglt2_coverage
# df_train_Prov_v2["year0_glp1_coverage"] = df_train_Prov_v2.time_zero_glp1_coverage
df_train_Prov_v2["year0_nsaid_coverage"] = df_train_Prov_v2.time_zero_nsaid_coverage
df_train_Prov_v2["year0_ppi_coverage"] = df_train_Prov_v2.time_zero_ppi_coverage
# df_train_Prov_v2["year0_mra_coverage"] = df_train_Prov_v2.time_zero_mra_coverage
df_train_Prov_v2["year0_scr_count"] = df_train_Prov_v2.time_zero_scr_count
df_train_Prov_v2["year0_mean"] = df_train_Prov_v2.time_zero_mean
df_train_Prov_v2["year0_norace_mean"] = df_train_Prov_v2.time_zero_norace_mean
df_train_Prov_v2["year0_uacr_mean"] = df_train_Prov_v2.time_zero_uacr_mean
df_train_Prov_v2["year0_upcr_mean"] = df_train_Prov_v2.time_zero_upcr_mean
df_train_Prov_v2["year0_sbp_mean"] = df_train_Prov_v2.time_zero_sbp_mean
df_train_Prov_v2["year0_dbp_mean"] = df_train_Prov_v2.time_zero_dbp_mean
df_train_Prov_v2["year0_pp_mean"] = df_train_Prov_v2.time_zero_pp_mean
df_train_Prov_v2["year0_map_mean"] = df_train_Prov_v2.time_zero_map_mean
# valid set
df_valid_Prov_v2["year0_hba1c_count"] = df_valid_Prov_v2.time_zero_hba1c_count
df_valid_Prov_v2["year0_hba1c_mean"] = df_valid_Prov_v2.time_zero_hba1c_mean
df_valid_Prov_v2["year0_uacr_count"] = df_valid_Prov_v2.time_zero_uacr_count
df_valid_Prov_v2["year0_upcr_count"] = df_valid_Prov_v2.time_zero_upcr_count
df_valid_Prov_v2["year0_bp_count"] = df_valid_Prov_v2.time_zero_bp_count
df_valid_Prov_v2["year0_av_count"] = df_valid_Prov_v2.time_zero_av_count
df_valid_Prov_v2["year0_ipv_count"] = df_valid_Prov_v2.time_zero_ipv_count
df_valid_Prov_v2["year0_aceiarb_coverage"] = df_valid_Prov_v2.time_zero_aceiarb_coverage
# df_valid_Prov_v2["year0_sglt2_coverage"] = df_valid_Prov_v2.time_zero_sglt2_coverage
# df_valid_Prov_v2["year0_glp1_coverage"] = df_valid_Prov_v2.time_zero_glp1_coverage
df_valid_Prov_v2["year0_nsaid_coverage"] = df_valid_Prov_v2.time_zero_nsaid_coverage
df_valid_Prov_v2["year0_ppi_coverage"] = df_valid_Prov_v2.time_zero_ppi_coverage
# df_valid_Prov_v2["year0_mra_coverage"] = df_valid_Prov_v2.time_zero_mra_coverage
df_valid_Prov_v2["year0_scr_count"] = df_valid_Prov_v2.time_zero_scr_count
df_valid_Prov_v2["year0_mean"] = df_valid_Prov_v2.time_zero_mean
df_valid_Prov_v2["year0_norace_mean"] = df_valid_Prov_v2.time_zero_norace_mean
df_valid_Prov_v2["year0_uacr_mean"] = df_valid_Prov_v2.time_zero_uacr_mean
df_valid_Prov_v2["year0_upcr_mean"] = df_valid_Prov_v2.time_zero_upcr_mean
df_valid_Prov_v2["year0_sbp_mean"] = df_valid_Prov_v2.time_zero_sbp_mean
df_valid_Prov_v2["year0_dbp_mean"] = df_valid_Prov_v2.time_zero_dbp_mean
df_valid_Prov_v2["year0_pp_mean"] = df_valid_Prov_v2.time_zero_pp_mean
df_valid_Prov_v2["year0_map_mean"] = df_valid_Prov_v2.time_zero_map_mean
# test set
df_test_Prov_v2["year0_hba1c_count"] = df_test_Prov_v2.time_zero_hba1c_count
df_test_Prov_v2["year0_hba1c_mean"] = df_test_Prov_v2.time_zero_hba1c_mean
df_test_Prov_v2["year0_uacr_count"] = df_test_Prov_v2.time_zero_uacr_count
df_test_Prov_v2["year0_upcr_count"] = df_test_Prov_v2.time_zero_upcr_count
df_test_Prov_v2["year0_bp_count"] = df_test_Prov_v2.time_zero_bp_count
df_test_Prov_v2["year0_av_count"] = df_test_Prov_v2.time_zero_av_count
df_test_Prov_v2["year0_ipv_count"] = df_test_Prov_v2.time_zero_ipv_count
df_test_Prov_v2["year0_aceiarb_coverage"] = df_test_Prov_v2.time_zero_aceiarb_coverage
# df_test_Prov_v2["year0_sglt2_coverage"] = df_test_Prov_v2.time_zero_sglt2_coverage
# df_test_Prov_v2["year0_glp1_coverage"] = df_test_Prov_v2.time_zero_glp1_coverage
df_test_Prov_v2["year0_nsaid_coverage"] = df_test_Prov_v2.time_zero_nsaid_coverage
df_test_Prov_v2["year0_ppi_coverage"] = df_test_Prov_v2.time_zero_ppi_coverage
# df_test_Prov_v2["year0_mra_coverage"] = df_test_Prov_v2.time_zero_mra_coverage
df_test_Prov_v2["year0_scr_count"] = df_test_Prov_v2.time_zero_scr_count
df_test_Prov_v2["year0_mean"] = df_test_Prov_v2.time_zero_mean
df_test_Prov_v2["year0_norace_mean"] = df_test_Prov_v2.time_zero_norace_mean
df_test_Prov_v2["year0_uacr_mean"] = df_test_Prov_v2.time_zero_uacr_mean
df_test_Prov_v2["year0_upcr_mean"] = df_test_Prov_v2.time_zero_upcr_mean
df_test_Prov_v2["year0_sbp_mean"] = df_test_Prov_v2.time_zero_sbp_mean
df_test_Prov_v2["year0_dbp_mean"] = df_test_Prov_v2.time_zero_dbp_mean
df_test_Prov_v2["year0_pp_mean"] = df_test_Prov_v2.time_zero_pp_mean
df_test_Prov_v2["year0_map_mean"] = df_test_Prov_v2.time_zero_map_mean

# train set
df_train_Prov_v2["year1_reduction_40_wr"] = df_train_Prov_v2.year1_reduction_40_ge
df_train_Prov_v2["year2_reduction_40_wr"] = df_train_Prov_v2.year2_reduction_40_ge
df_train_Prov_v2["year3_reduction_40_wr"] = df_train_Prov_v2.year3_reduction_40_ge
df_train_Prov_v2["year4_reduction_40_wr"] = df_train_Prov_v2.year4_reduction_40_ge
df_train_Prov_v2["year5_reduction_40_wr"] = df_train_Prov_v2.year5_reduction_40_ge

df_train_Prov_v2.year1_reduction_40_wr.value_counts()
df_train_Prov_v2.year2_reduction_40_wr.value_counts()
df_train_Prov_v2.year3_reduction_40_wr.value_counts()
df_train_Prov_v2.year4_reduction_40_wr.value_counts()
df_train_Prov_v2.year5_reduction_40_wr.value_counts()


df_train_Prov_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_psjh_train.csv"
)


# valid set
df_valid_Prov_v2["year1_reduction_40_wr"] = df_valid_Prov_v2.year1_reduction_40_ge
df_valid_Prov_v2["year2_reduction_40_wr"] = df_valid_Prov_v2.year2_reduction_40_ge
df_valid_Prov_v2["year3_reduction_40_wr"] = df_valid_Prov_v2.year3_reduction_40_ge
df_valid_Prov_v2["year4_reduction_40_wr"] = df_valid_Prov_v2.year4_reduction_40_ge
df_valid_Prov_v2["year5_reduction_40_wr"] = df_valid_Prov_v2.year5_reduction_40_ge

df_valid_Prov_v2.year1_reduction_40_wr.value_counts()
df_valid_Prov_v2.year2_reduction_40_wr.value_counts()
df_valid_Prov_v2.year3_reduction_40_wr.value_counts()
df_valid_Prov_v2.year4_reduction_40_wr.value_counts()
df_valid_Prov_v2.year5_reduction_40_wr.value_counts()

df_valid_Prov_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_psjh_valid.csv"
)

# test set
df_test_Prov_v2["year1_reduction_40_wr"] = df_test_Prov_v2.year1_reduction_40_ge
df_test_Prov_v2["year2_reduction_40_wr"] = df_test_Prov_v2.year2_reduction_40_ge
df_test_Prov_v2["year3_reduction_40_wr"] = df_test_Prov_v2.year3_reduction_40_ge
df_test_Prov_v2["year4_reduction_40_wr"] = df_test_Prov_v2.year4_reduction_40_ge
df_test_Prov_v2["year5_reduction_40_wr"] = df_test_Prov_v2.year5_reduction_40_ge

df_test_Prov_v2.year1_reduction_40_wr.value_counts()
df_test_Prov_v2.year2_reduction_40_wr.value_counts()
df_test_Prov_v2.year3_reduction_40_wr.value_counts()
df_test_Prov_v2.year4_reduction_40_wr.value_counts()
df_test_Prov_v2.year5_reduction_40_wr.value_counts()

df_test_Prov_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_psjh_test.csv"
)

#######################

import pandas as pd

### combined
# Get ANNUAL eGFR Reduction40 Flags... (This is version 3 of the egfr_reduction40 label)
df_train_combined_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_combined_train.csv",
    low_memory=False,
)
df_valid_combined_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_combined_valid.csv",
    low_memory=False,
)
df_test_combined_v2 = pd.read_csv(
    "./Data/split_label_encoded_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_combined_test.csv",
    low_memory=False,
)


### create yr0 observations from time_zero observations
# train set
df_train_combined_v2["year0_hba1c_count"] = df_train_combined_v2.time_zero_hba1c_count
df_train_combined_v2["year0_hba1c_mean"] = df_train_combined_v2.time_zero_hba1c_mean
df_train_combined_v2["year0_uacr_count"] = df_train_combined_v2.time_zero_uacr_count
df_train_combined_v2["year0_upcr_count"] = df_train_combined_v2.time_zero_upcr_count
df_train_combined_v2["year0_bp_count"] = df_train_combined_v2.time_zero_bp_count
df_train_combined_v2["year0_av_count"] = df_train_combined_v2.time_zero_av_count
df_train_combined_v2["year0_ipv_count"] = df_train_combined_v2.time_zero_ipv_count
df_train_combined_v2[
    "year0_aceiarb_coverage"
] = df_train_combined_v2.time_zero_aceiarb_coverage
df_train_combined_v2[
    "year0_sglt2_coverage"
] = df_train_combined_v2.time_zero_sglt2_coverage
df_train_combined_v2[
    "year0_glp1_coverage"
] = df_train_combined_v2.time_zero_glp1_coverage
df_train_combined_v2[
    "year0_nsaid_coverage"
] = df_train_combined_v2.time_zero_nsaid_coverage
df_train_combined_v2["year0_ppi_coverage"] = df_train_combined_v2.time_zero_ppi_coverage
df_train_combined_v2["year0_mra_coverage"] = df_train_combined_v2.time_zero_mra_coverage
df_train_combined_v2["year0_scr_count"] = df_train_combined_v2.time_zero_scr_count
df_train_combined_v2["year0_mean"] = df_train_combined_v2.time_zero_mean
df_train_combined_v2["year0_norace_mean"] = df_train_combined_v2.time_zero_norace_mean
df_train_combined_v2["year0_uacr_mean"] = df_train_combined_v2.time_zero_uacr_mean
df_train_combined_v2["year0_upcr_mean"] = df_train_combined_v2.time_zero_upcr_mean
df_train_combined_v2["year0_sbp_mean"] = df_train_combined_v2.time_zero_sbp_mean
df_train_combined_v2["year0_dbp_mean"] = df_train_combined_v2.time_zero_dbp_mean
df_train_combined_v2["year0_pp_mean"] = df_train_combined_v2.time_zero_pp_mean
df_train_combined_v2["year0_map_mean"] = df_train_combined_v2.time_zero_map_mean
# valid set
df_valid_combined_v2["year0_hba1c_count"] = df_valid_combined_v2.time_zero_hba1c_count
df_valid_combined_v2["year0_hba1c_mean"] = df_valid_combined_v2.time_zero_hba1c_mean
df_valid_combined_v2["year0_uacr_count"] = df_valid_combined_v2.time_zero_uacr_count
df_valid_combined_v2["year0_upcr_count"] = df_valid_combined_v2.time_zero_upcr_count
df_valid_combined_v2["year0_bp_count"] = df_valid_combined_v2.time_zero_bp_count
df_valid_combined_v2["year0_av_count"] = df_valid_combined_v2.time_zero_av_count
df_valid_combined_v2["year0_ipv_count"] = df_valid_combined_v2.time_zero_ipv_count
df_valid_combined_v2[
    "year0_aceiarb_coverage"
] = df_valid_combined_v2.time_zero_aceiarb_coverage
# df_valid_combined_v2[
#     "year0_sglt2_coverage"
# ] = df_valid_combined_v2.time_zero_sglt2_coverage
# df_valid_combined_v2[
#     "year0_glp1_coverage"
# ] = df_valid_combined_v2.time_zero_glp1_coverage
df_valid_combined_v2[
    "year0_nsaid_coverage"
] = df_valid_combined_v2.time_zero_nsaid_coverage
df_valid_combined_v2["year0_ppi_coverage"] = df_valid_combined_v2.time_zero_ppi_coverage
# df_valid_combined_v2["year0_mra_coverage"] = df_valid_combined_v2.time_zero_mra_coverage
df_valid_combined_v2["year0_scr_count"] = df_valid_combined_v2.time_zero_scr_count
df_valid_combined_v2["year0_mean"] = df_valid_combined_v2.time_zero_mean
df_valid_combined_v2["year0_norace_mean"] = df_valid_combined_v2.time_zero_norace_mean
df_valid_combined_v2["year0_uacr_mean"] = df_valid_combined_v2.time_zero_uacr_mean
df_valid_combined_v2["year0_upcr_mean"] = df_valid_combined_v2.time_zero_upcr_mean
df_valid_combined_v2["year0_sbp_mean"] = df_valid_combined_v2.time_zero_sbp_mean
df_valid_combined_v2["year0_dbp_mean"] = df_valid_combined_v2.time_zero_dbp_mean
df_valid_combined_v2["year0_pp_mean"] = df_valid_combined_v2.time_zero_pp_mean
df_valid_combined_v2["year0_map_mean"] = df_valid_combined_v2.time_zero_map_mean
# test set
df_test_combined_v2["year0_hba1c_count"] = df_test_combined_v2.time_zero_hba1c_count
df_test_combined_v2["year0_hba1c_mean"] = df_test_combined_v2.time_zero_hba1c_mean
df_test_combined_v2["year0_uacr_count"] = df_test_combined_v2.time_zero_uacr_count
df_test_combined_v2["year0_upcr_count"] = df_test_combined_v2.time_zero_upcr_count
df_test_combined_v2["year0_bp_count"] = df_test_combined_v2.time_zero_bp_count
df_test_combined_v2["year0_av_count"] = df_test_combined_v2.time_zero_av_count
df_test_combined_v2["year0_ipv_count"] = df_test_combined_v2.time_zero_ipv_count
df_test_combined_v2[
    "year0_aceiarb_coverage"
] = df_test_combined_v2.time_zero_aceiarb_coverage
# df_test_combined_v2[
#     "year0_sglt2_coverage"
# ] = df_test_combined_v2.time_zero_sglt2_coverage
# df_test_combined_v2["year0_glp1_coverage"] = df_test_combined_v2.time_zero_glp1_coverage
df_test_combined_v2[
    "year0_nsaid_coverage"
] = df_test_combined_v2.time_zero_nsaid_coverage
df_test_combined_v2["year0_ppi_coverage"] = df_test_combined_v2.time_zero_ppi_coverage
# df_test_combined_v2["year0_mra_coverage"] = df_test_combined_v2.time_zero_mra_coverage
df_test_combined_v2["year0_scr_count"] = df_test_combined_v2.time_zero_scr_count
df_test_combined_v2["year0_mean"] = df_test_combined_v2.time_zero_mean
df_test_combined_v2["year0_norace_mean"] = df_test_combined_v2.time_zero_norace_mean
df_test_combined_v2["year0_uacr_mean"] = df_test_combined_v2.time_zero_uacr_mean
df_test_combined_v2["year0_upcr_mean"] = df_test_combined_v2.time_zero_upcr_mean
df_test_combined_v2["year0_sbp_mean"] = df_test_combined_v2.time_zero_sbp_mean
df_test_combined_v2["year0_dbp_mean"] = df_test_combined_v2.time_zero_dbp_mean
df_test_combined_v2["year0_pp_mean"] = df_test_combined_v2.time_zero_pp_mean
df_test_combined_v2["year0_map_mean"] = df_test_combined_v2.time_zero_map_mean

# train set
df_train_combined_v2[
    "year1_reduction_40_wr"
] = df_train_combined_v2.year1_reduction_40_ge
df_train_combined_v2[
    "year2_reduction_40_wr"
] = df_train_combined_v2.year2_reduction_40_ge
df_train_combined_v2[
    "year3_reduction_40_wr"
] = df_train_combined_v2.year3_reduction_40_ge
df_train_combined_v2[
    "year4_reduction_40_wr"
] = df_train_combined_v2.year4_reduction_40_ge
df_train_combined_v2[
    "year5_reduction_40_wr"
] = df_train_combined_v2.year5_reduction_40_ge

df_train_combined_v2.year1_reduction_40_wr.value_counts()
df_train_combined_v2.year2_reduction_40_wr.value_counts()
df_train_combined_v2.year3_reduction_40_wr.value_counts()
df_train_combined_v2.year4_reduction_40_wr.value_counts()
df_train_combined_v2.year5_reduction_40_wr.value_counts()


df_train_combined_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_combined_train.csv"
)


# valid set
df_valid_combined_v2[
    "year1_reduction_40_wr"
] = df_valid_combined_v2.year1_reduction_40_ge
df_valid_combined_v2[
    "year2_reduction_40_wr"
] = df_valid_combined_v2.year2_reduction_40_ge
df_valid_combined_v2[
    "year3_reduction_40_wr"
] = df_valid_combined_v2.year3_reduction_40_ge
df_valid_combined_v2[
    "year4_reduction_40_wr"
] = df_valid_combined_v2.year4_reduction_40_ge
df_valid_combined_v2[
    "year5_reduction_40_wr"
] = df_valid_combined_v2.year5_reduction_40_ge

df_valid_combined_v2.year1_reduction_40_wr.value_counts()
df_valid_combined_v2.year2_reduction_40_wr.value_counts()
df_valid_combined_v2.year3_reduction_40_wr.value_counts()
df_valid_combined_v2.year4_reduction_40_wr.value_counts()
df_valid_combined_v2.year5_reduction_40_wr.value_counts()

df_valid_combined_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_combined_valid.csv"
)

# test set
df_test_combined_v2["year1_reduction_40_wr"] = df_test_combined_v2.year1_reduction_40_ge
df_test_combined_v2["year2_reduction_40_wr"] = df_test_combined_v2.year2_reduction_40_ge
df_test_combined_v2["year3_reduction_40_wr"] = df_test_combined_v2.year3_reduction_40_ge
df_test_combined_v2["year4_reduction_40_wr"] = df_test_combined_v2.year4_reduction_40_ge
df_test_combined_v2["year5_reduction_40_wr"] = df_test_combined_v2.year5_reduction_40_ge

df_test_combined_v2.year1_reduction_40_wr.value_counts()
df_test_combined_v2.year2_reduction_40_wr.value_counts()
df_test_combined_v2.year3_reduction_40_wr.value_counts()
df_test_combined_v2.year4_reduction_40_wr.value_counts()
df_test_combined_v2.year5_reduction_40_wr.value_counts()

df_test_combined_v2.to_csv(
    "./Data/DBN_PAPER_DATA/split_label_encoded_datasets/aws_path/egfr_reduction_data_combined_test.csv"
)

#######################
