.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = CKD_progression
PYTHON_INTERPRETER = python3


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

date_str = $(shell date +%Y-%m-%d -d "1 days ago")

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Save current requirements
freeze:
	pip freeze | grep -v "pkg-resources" > requirements.txt

## Preprocess data for dashboard
preprocess_data:
	$(PYTHON_INTERPRETER) ./data_preprocessing/preprocess.py

########################################################################
######### Split data to UCLA, Providence, and combined #################
# training, validation, and testing 60%, 20%, and 20%

## 1. Split datasets for combined, UCLA, providence to train,val,test
split_data:
	$(PYTHON_INTERPRETER) ./data_preprocessing/split_datasets.py

## 2.a Discretize Split datasets for combined, UCLA, providence to train,val,test
## 2.b. Label encode (1,2 ..) Discretized Split datasets for combined, UCLA,
##    providence to train,val,test
discretize_data:
	$(PYTHON_INTERPRETER) ./data_preprocessing/discretize_datasets.py

## 3.a Discretize Providence using UCLA discritizer
discretize_data_using_UCLA:
	$(PYTHON_INTERPRETER) ./data_preprocessing/discretize_Prov_using_UCLA.py

## 4.a Discretize UCLA using Providence discritizer
discretize_data_using_Prov:
	$(PYTHON_INTERPRETER) ./data_preprocessing/discretize_UCLA_using_Prov.py

########################################################################
########################################################################

## 5. Convert all data to required format for RAUS
raus_format_data:
	$(PYTHON_INTERPRETER) ./data_preprocessing/raus_format_data.py

########################################################################
########################################################################

########################################################################
## 6. RAUS struct learning ################################

# use raus environment: conda activate raus
## without count variables and with and without race adjusted variables (1 site's full DBN = contemporals, initial conditions, intra_for_full_DBN, inter_for_full_DBN) ##
#Scenario 1.1:
# to run them in parallel "make -j 4 scenario_1_1"
scenario_1_1: learn_UCLA_contemporals learn_UCLA_initial_conditions learn_UCLA_intra_for_full_DBN learn_UCLA_inter_for_full_DBN

learn_UCLA_contemporals:
	screen -d -m -S learn_UCLA_contemporals $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_nocounts_ucla_track2:3.py
learn_UCLA_initial_conditions:
	screen -d -m -S learn_UCLA_initial_conditions $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_ucla_track2:3.py
learn_UCLA_intra_for_full_DBN:
	screen -d -m -S learn_UCLA_intra_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_nocounts_ucla_track2:3.py
learn_UCLA_inter_for_full_DBN:
	screen -d -m -S learn_UCLA_inter_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_nocounts_ucla_track1.py

#Scenario 2.1:
# to run them in parallel "make -j 4 scenario_2_1"
scenario_2_1: learn_UCLA_contemporals_norace learn_UCLA_initial_conditions_norace learn_UCLA_intra_for_full_DBN_norace learn_UCLA_inter_for_full_DBN_norace

learn_UCLA_contemporals_norace:
	screen -d -m -S learn_UCLA_contemporals_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_nocounts_norace_ucla_track2:3.py
learn_UCLA_initial_conditions_norace:
	screen -d -m -S learn_UCLA_initial_conditions_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_norace_ucla_track2:3.py
learn_UCLA_intra_for_full_DBN_norace:
	screen -d -m -S learn_UCLA_intra_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_nocounts_norace_ucla_track2:3.py
learn_UCLA_inter_for_full_DBN_norace:
	screen -d -m -S learn_UCLA_inter_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_nocounts_norace_ucla_track1.py

## PSJH
#Scenario 1.2:
# to run them in parallel "make -j 4 scenario_1_2"
scenario_1_2: learn_PSJH_contemporals learn_PSJH_initial_conditions learn_PSJH_intra_for_full_DBN learn_PSJH_inter_for_full_DBN

learn_PSJH_contemporals:
	screen -d -m -S learn_PSJH_contemporals $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_nocounts_psjh_track2:3.py
learn_PSJH_initial_conditions:
	screen -d -m -S learn_PSJH_initial_conditions $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_psjh_track2:3.py
learn_PSJH_intra_for_full_DBN:# not run
	screen -d -m -S learn_PSJH_intra_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_nocounts_psjh_track2:3.py
learn_PSJH_inter_for_full_DBN:
	screen -d -m -S learn_PSJH_inter_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_nocounts_psjh_track1.py

#Scenario 2.2:
scenario_2_2: learn_PSJH_contemporals_norace learn_PSJH_initial_conditions_norace learn_PSJH_intra_for_full_DBN_norace learn_PSJH_inter_for_full_DBN_norace

learn_PSJH_contemporals_norace:
	screen -d -m -S learn_PSJH_contemporals_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_nocounts_norace_psjh_track2:3.py
learn_PSJH_initial_conditions_norace:
	screen -d -m -S learn_PSJH_initial_conditions_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_norace_psjh_track2:3.py
learn_PSJH_intra_for_full_DBN_norace:
	screen -d -m -S learn_PSJH_intra_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_nocounts_norace_psjh_track2:3.py
learn_PSJH_inter_for_full_DBN_norace:
	screen -d -m -S learn_PSJH_inter_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_nocounts_norace_psjh_track1.py

## Combined
#Scenario 1.3:
scenario_1_3: learn_Combined_contemporals learn_Combined_initial_conditions learn_Combined_intra_for_full_DBN learn_Combined_inter_for_full_DBN

learn_Combined_contemporals:
	screen -d -m -S learn_Combined_contemporals $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_nocounts_combined_track2:3.py
learn_Combined_initial_conditions:
	screen -d -m -S learn_Combined_initial_conditions $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_combined_track2:3.py
learn_Combined_intra_for_full_DBN:
	screen -d -m -S learn_Combined_intra_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_nocounts_combined_track2:3.py
learn_Combined_inter_for_full_DBN:
	screen -d -m -S learn_Combined_inter_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_nocounts_combined_track1.py

#Scenario 2.3:
scenario_2_3: learn_Combined_contemporals_norace learn_Combined_initial_conditions_norace learn_Combined_intra_for_full_DBN_norace learn_Combined_inter_for_full_DBN_norace

learn_Combined_contemporals_norace:
	screen -d -m -S learn_Combined_contemporals_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_nocounts_norace_combined_track2:3.py
learn_Combined_initial_conditions_norace:
	screen -d -m -S learn_Combined_initial_conditions_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_norace_combined_track2:3.py
learn_Combined_intra_for_full_DBN_norace:
	screen -d -m -S learn_Combined_intra_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_nocounts_norace_combined_track2:3.py
learn_Combined_inter_for_full_DBN_norace:
	screen -d -m -S learn_Combined_inter_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_nocounts_norace_combined_track1.py


## with count variables and with and without race adjusted variables (1 site's full DBN = contemporals, initial conditions, intra_for_full_DBN, inter_for_full_DBN) ##
#Scenario 3.1:
scenario_3_1: learn_UCLA_counts_contemporals learn_UCLA_counts_initial_conditions learn_UCLA_counts_intra_for_full_DBN learn_UCLA_counts_inter_for_full_DBN


learn_UCLA_counts_contemporals:
	screen -d -m -S learn_UCLA_counts_contemporals $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_counts_ucla_track2:3.py
learn_UCLA_counts_initial_conditions:
	screen -d -m -S learn_UCLA_counts_initial_conditions $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_counts_ucla_track2:3.py
learn_UCLA_counts_intra_for_full_DBN:
	screen -d -m -S learn_UCLA_counts_intra_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_counts_ucla_track2:3.py
learn_UCLA_counts_inter_for_full_DBN:
	screen -d -m -S learn_UCLA_counts_inter_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_counts_ucla_track1.py

#Scenario 4.1:
scenario_4_1: learn_UCLA_counts_contemporals_norace learn_UCLA_counts_initial_conditions_norace learn_UCLA_counts_intra_for_full_DBN_norace learn_UCLA_counts_inter_for_full_DBN_norace

learn_UCLA_counts_contemporals_norace:
	screen -d -m -S learn_UCLA_counts_contemporals_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_counts_norace_ucla_track2:3.py
learn_UCLA_counts_initial_conditions_norace:
	screen -d -m -S learn_UCLA_counts_initial_conditions_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_counts_norace_ucla_track2:3.py
learn_UCLA_counts_intra_for_full_DBN_norace:
	screen -d -m -S learn_UCLA_counts_intra_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_counts_norace_ucla_track2:3.py
learn_UCLA_counts_inter_for_full_DBN_norace:
	screen -d -m -S learn_UCLA_counts_inter_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_counts_norace_ucla_track1.py

## PSJH
#Scenario 3.2:
scenario_3_2: learn_PSJH_counts_contemporals learn_PSJH_counts_initial_conditions learn_PSJH_counts_intra_for_full_DBN learn_PSJH_counts_inter_for_full_DBN

learn_PSJH_counts_contemporals:
	screen -d -m -S learn_PSJH_counts_contemporals $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_counts_psjh_track2:3.py
learn_PSJH_counts_initial_conditions:
	screen -d -m -S learn_PSJH_counts_initial_conditions $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_counts_psjh_track2:3.py
learn_PSJH_counts_intra_for_full_DBN:
	screen -d -m -S learn_PSJH_counts_intra_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_counts_psjh_track2:3.py
learn_PSJH_counts_inter_for_full_DBN:
	screen -d -m -S learn_PSJH_counts_inter_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_counts_psjh_track1.py

#Scenario 4.2:
scenario_4_2: learn_PSJH_counts_contemporals_norace learn_PSJH_counts_initial_conditions_norace learn_PSJH_counts_intra_for_full_DBN_norace learn_PSJH_counts_inter_for_full_DBN_norace

learn_PSJH_counts_contemporals_norace:
	screen -d -m -S learn_PSJH_counts_contemporals_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_counts_norace_psjh_track2:3.py
learn_PSJH_counts_initial_conditions_norace:
	screen -d -m -S learn_PSJH_counts_initial_conditions_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_counts_norace_psjh_track2:3.py
learn_PSJH_counts_intra_for_full_DBN_norace:
	screen -d -m -S learn_PSJH_counts_intra_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_counts_norace_psjh_track2:3.py
learn_PSJH_counts_inter_for_full_DBN_norace:
	screen -d -m -S learn_PSJH_counts_inter_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_counts_norace_psjh_track1.py

## Combined
#Scenario 3.3:
scenario_3_3: learn_Combined_counts_contemporals learn_Combined_counts_initial_conditions learn_Combined_counts_intra_for_full_DBN learn_Combined_counts_inter_for_full_DBN

learn_Combined_counts_contemporals:
	screen -d -m -S learn_Combined_counts_contemporals $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_counts_combined_track2:3.py
learn_Combined_counts_initial_conditions:
	screen -d -m -S learn_Combined_counts_initial_conditions $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_counts_combined_track2:3.py
learn_Combined_counts_intra_for_full_DBN:
	screen -d -m -S learn_Combined_counts_intra_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_counts_combined_track2:3.py
learn_Combined_counts_inter_for_full_DBN:
	screen -d -m -S learn_Combined_counts_inter_for_full_DBN $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_counts_combined_track1.py

#Scenario 4.3:
scenario_4_3: learn_Combined_counts_contemporals_norace learn_Combined_counts_initial_conditions_norace learn_Combined_counts_intra_for_full_DBN_norace learn_Combined_counts_inter_for_full_DBN_norace

learn_Combined_counts_contemporals_norace:
	screen -d -m -S learn_Combined_counts_contemporals_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_wstatic_yr0observations_counts_norace_combined_track2:3.py
learn_Combined_counts_initial_conditions_norace:
	screen -d -m -S learn_Combined_counts_initial_conditions_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr0-1observations_yr1outasobs_counts_norace_combined_track2:3.py
learn_Combined_counts_intra_for_full_DBN_norace:
	screen -d -m -S learn_Combined_counts_intra_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1observations_counts_norace_combined_track2:3.py
learn_Combined_counts_inter_for_full_DBN_norace:
	screen -d -m -S learn_Combined_counts_inter_for_full_DBN_norace $(PYTHON_INTERPRETER) ./RAUS/egfr_reduction40_ge_yr1-4observations_counts_norace_combined_track1.py

# deactivate raus environment: conda deactivate

########################################################################
########################################################################

## 7. Convert all data to required format by GeNie
genie_format_data:
	$(PYTHON_INTERPRETER) ./data_preprocessing/genie_format_data.py

########################################################################
######################## Train data undersampling ######################

## 8. Training data undersampling, # run it on RAUS virtual environment
data_undersampling:
	$(PYTHON_INTERPRETER) ./data_preprocessing/data_undersampling.py


########################################################################
########################################################################

########################################################################
################## Model evaluation ####################################

## 9. Running Full model BNs
full_BNs_models_eval:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/full_models_BNs_vars_dataset.py

## 10. Running Full model DBNs
full_DBNs_models_eval:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/full_models_DBNs_vars_dataset.py

## 11. Evaluate Full model DBNs generate results
full_DBNs_models_results:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/dbn_model_evaluation_stratified_by_filter.py

## 11 a. Evaluate Full model DBNs generate results with bootstrap CIs
# Boostrap in stratified proportions
full_DBNs_models_results_with_CIs_stratified:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/dbn_model_evaluation_stratified_by_filter_with_CIs_stratified.py

## 11 b. Evaluate Full model DBNs generate results with bootstrap CIs
# Boostrap in balanced proportions
full_DBNs_models_results_with_CIs_balanced:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/dbn_model_evaluation_stratified_by_filter_with_CIs_balanced.py

#################### optional in case you compare all scenarios
## 12. Statistical significance between models AUCs
full_DBNs_models_stat_sign:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/stat_significance_each_model_site.py

## 13. Selecting the best models
get_best_DBNs_models_results:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/best_dbn_models_results_visualization.py

####################

## 14.a. All models visulization of performance
models_perf_visualization:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/models_results_visualization.py

## 14.a.i. All models visulization of performance
models_perf_visualization_stratified:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/models_results_visualization_with_CIs_stratified.py

## 14.a.ii. All models visulization of performance
models_perf_visualization_balanced:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/models_results_visualization_with_CIs_balanced.py

## 14.b. All models visulization of performance on external datasets
models_perf_ext_datasets_visualization:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/models_results_ext_datasets_visualization.py

## 14.b.i. All models visulization of performance on external datasets stratified bootstrap
models_perf_ext_datasets_visualization_stratified:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/models_results_ext_datasets_visualization_with_CIs_stratified.py

## 14.b. All models visulization of performance on external datasets
models_perf_ext_datasets_visualization_balanced:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/models_results_ext_datasets_visualization_with_CIs_balanced.py

########################################################################
########################################################################

########################################################################
################## Model sensitivity analysis ##########################

## 15. Sensitivity analysis using Do queries on every variables
# model site and testing site
sens_analysis_of_outcome: sens_analysis_of_outcome_UCLA_UCLA sens_analysis_of_outcome_UCLA_PSJH sens_analysis_of_outcome_PSJH_PSJH sens_analysis_of_outcome_PSJH_UCLA sens_analysis_of_outcome_Combined_Combined

sens_analysis_of_outcome_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model.py UCLA UCLA
sens_analysis_of_outcome_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model.py UCLA PSJH
sens_analysis_of_outcome_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model.py PSJH PSJH
sens_analysis_of_outcome_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model.py PSJH UCLA
sens_analysis_of_outcome_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model.py Combined Combined
	
## 16. Sensitivity analysis using Do queries on every variables using continuous medication or not using continuous medication
# sens_analysis_of_outcome_medication: sens_analysis_of_outcome_on_medication sens_analysis_of_outcome_off_medication

# sens_analysis_of_outcome_on_aceiarb
# model site and testing site
sens_analysis_of_outcome_on_aceiarb: sens_analysis_of_outcome_on_aceiarb_UCLA_UCLA sens_analysis_of_outcome_on_aceiarb_UCLA_PSJH sens_analysis_of_outcome_on_aceiarb_PSJH_PSJH sens_analysis_of_outcome_on_aceiarb_PSJH_UCLA sens_analysis_of_outcome_on_aceiarb_Combined_Combined

sens_analysis_of_outcome_on_aceiarb_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA UCLA on aceiarb
sens_analysis_of_outcome_on_aceiarb_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA PSJH on aceiarb
sens_analysis_of_outcome_on_aceiarb_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH PSJH on aceiarb
sens_analysis_of_outcome_on_aceiarb_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH UCLA on aceiarb
sens_analysis_of_outcome_on_aceiarb_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py Combined Combined on aceiarb

# sens_analysis_of_outcome_off_aceiarb
sens_analysis_of_outcome_off_aceiarb: sens_analysis_of_outcome_off_aceiarb_UCLA_UCLA sens_analysis_of_outcome_off_aceiarb_UCLA_PSJH sens_analysis_of_outcome_off_aceiarb_PSJH_PSJH sens_analysis_of_outcome_off_aceiarb_PSJH_UCLA sens_analysis_of_outcome_off_aceiarb_Combined_Combined

sens_analysis_of_outcome_off_aceiarb_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA UCLA off aceiarb
sens_analysis_of_outcome_off_aceiarb_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA PSJH off aceiarb
sens_analysis_of_outcome_off_aceiarb_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH PSJH off aceiarb
sens_analysis_of_outcome_off_aceiarb_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH UCLA off aceiarb
sens_analysis_of_outcome_off_aceiarb_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py Combined Combined off aceiarb

# sens_analysis_of_outcome_on_nsaid
# model site and testing site
sens_analysis_of_outcome_on_nsaid: sens_analysis_of_outcome_on_nsaid_UCLA_UCLA sens_analysis_of_outcome_on_nsaid_UCLA_PSJH sens_analysis_of_outcome_on_nsaid_PSJH_PSJH sens_analysis_of_outcome_on_nsaid_PSJH_UCLA sens_analysis_of_outcome_on_nsaid_Combined_Combined

sens_analysis_of_outcome_on_nsaid_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA UCLA on nsaid
sens_analysis_of_outcome_on_nsaid_UCLA_PSJH:sens_analysis_of_outcome_on_nsaid_UCLA_UCLA
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA PSJH on nsaid
sens_analysis_of_outcome_on_nsaid_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH PSJH on nsaid
sens_analysis_of_outcome_on_nsaid_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH UCLA on nsaid
sens_analysis_of_outcome_on_nsaid_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py Combined Combined on nsaid

# sens_analysis_of_outcome_off_nsaid
sens_analysis_of_outcome_off_nsaid: sens_analysis_of_outcome_off_nsaid_UCLA_UCLA sens_analysis_of_outcome_off_nsaid_UCLA_PSJH sens_analysis_of_outcome_off_nsaid_PSJH_PSJH sens_analysis_of_outcome_off_nsaid_PSJH_UCLA sens_analysis_of_outcome_off_nsaid_Combined_Combined

sens_analysis_of_outcome_off_nsaid_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA UCLA off nsaid
sens_analysis_of_outcome_off_nsaid_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py UCLA PSJH off nsaid
sens_analysis_of_outcome_off_nsaid_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH PSJH off nsaid
sens_analysis_of_outcome_off_nsaid_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py PSJH UCLA off nsaid
sens_analysis_of_outcome_off_nsaid_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication.py Combined Combined off nsaid

## 17. Sensitivity analysis using Do queries on a set of variables to profile the population
# sens_analysis_of_outcome_on_aceiarb population profiling
# model site and testing site
sens_analysis_of_outcome_on_aceiarb_pop_profiling: sens_analysis_of_outcome_on_aceiarb_pop_profiling_UCLA_UCLA sens_analysis_of_outcome_on_aceiarb_pop_profiling_UCLA_PSJH sens_analysis_of_outcome_on_aceiarb_pop_profiling_PSJH_PSJH sens_analysis_of_outcome_on_aceiarb_pop_profiling_PSJH_UCLA sens_analysis_of_outcome_on_aceiarb_pop_profiling_Combined_Combined

sens_analysis_of_outcome_on_aceiarb_pop_profiling_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA UCLA on aceiarb 
sens_analysis_of_outcome_on_aceiarb_pop_profiling_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA PSJH on aceiarb
sens_analysis_of_outcome_on_aceiarb_pop_profiling_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH PSJH on aceiarb
sens_analysis_of_outcome_on_aceiarb_pop_profiling_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH UCLA on aceiarb
sens_analysis_of_outcome_on_aceiarb_pop_profiling_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py Combined Combined on aceiarb

# sens_analysis_of_outcome_on_nsaid population profiling
# model site and testing site
sens_analysis_of_outcome_on_nsaid_pop_profiling: sens_analysis_of_outcome_on_nsaid_pop_profiling_UCLA_UCLA sens_analysis_of_outcome_on_nsaid_pop_profiling_UCLA_PSJH sens_analysis_of_outcome_on_nsaid_pop_profiling_PSJH_PSJH sens_analysis_of_outcome_on_nsaid_pop_profiling_PSJH_UCLA sens_analysis_of_outcome_on_nsaid_pop_profiling_Combined_Combined

sens_analysis_of_outcome_on_nsaid_pop_profiling_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA UCLA on nsaid 
sens_analysis_of_outcome_on_nsaid_pop_profiling_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA PSJH on nsaid
sens_analysis_of_outcome_on_nsaid_pop_profiling_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH PSJH on nsaid
sens_analysis_of_outcome_on_nsaid_pop_profiling_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH UCLA on nsaid
sens_analysis_of_outcome_on_nsaid_pop_profiling_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py Combined Combined on nsaid

# sens_analysis_of_outcome_off_aceiarb population profiling
# model site and testing site
sens_analysis_of_outcome_off_aceiarb_pop_profiling: sens_analysis_of_outcome_off_aceiarb_pop_profiling_UCLA_UCLA sens_analysis_of_outcome_off_aceiarb_pop_profiling_UCLA_PSJH sens_analysis_of_outcome_off_aceiarb_pop_profiling_PSJH_PSJH sens_analysis_of_outcome_off_aceiarb_pop_profiling_PSJH_UCLA sens_analysis_of_outcome_off_aceiarb_pop_profiling_Combined_Combined

sens_analysis_of_outcome_off_aceiarb_pop_profiling_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA UCLA off aceiarb 
sens_analysis_of_outcome_off_aceiarb_pop_profiling_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA PSJH off aceiarb
sens_analysis_of_outcome_off_aceiarb_pop_profiling_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH PSJH off aceiarb
sens_analysis_of_outcome_off_aceiarb_pop_profiling_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH UCLA off aceiarb
sens_analysis_of_outcome_off_aceiarb_pop_profiling_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py Combined Combined off aceiarb

# sens_analysis_of_outcome_off_nsaid population profiling
# model site and testing site
sens_analysis_of_outcome_off_nsaid_pop_profiling: sens_analysis_of_outcome_off_nsaid_pop_profiling_UCLA_UCLA sens_analysis_of_outcome_off_nsaid_pop_profiling_UCLA_PSJH sens_analysis_of_outcome_off_nsaid_pop_profiling_PSJH_PSJH sens_analysis_of_outcome_off_nsaid_pop_profiling_PSJH_UCLA sens_analysis_of_outcome_off_nsaid_pop_profiling_Combined_Combined

sens_analysis_of_outcome_off_nsaid_pop_profiling_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA UCLA off nsaid 
sens_analysis_of_outcome_off_nsaid_pop_profiling_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py UCLA PSJH off nsaid
sens_analysis_of_outcome_off_nsaid_pop_profiling_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH PSJH off nsaid
sens_analysis_of_outcome_off_nsaid_pop_profiling_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py PSJH UCLA off nsaid
sens_analysis_of_outcome_off_nsaid_pop_profiling_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling.py Combined Combined off nsaid

## 18. Sensitivity analysis using Do queries on a set all variables to profile the population
# sens_analysis_of_outcome_on_aceiarb population profiling
# model site and testing site
sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars: sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_UCLA_UCLA sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_UCLA_PSJH sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_PSJH_PSJH sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_PSJH_UCLA sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_Combined_Combined

sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA UCLA on aceiarb 
sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA PSJH on aceiarb
sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH PSJH on aceiarb
sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH UCLA on aceiarb
sens_analysis_of_outcome_on_aceiarb_pop_profiling_all_vars_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py Combined Combined on aceiarb

# sens_analysis_of_outcome_on_nsaid population profiling
# model site and testing site
sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars: sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_UCLA_UCLA sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_UCLA_PSJH sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_PSJH_PSJH sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_PSJH_UCLA sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_Combined_Combined

sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA UCLA on nsaid 
sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA PSJH on nsaid
sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH PSJH on nsaid
sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH UCLA on nsaid
sens_analysis_of_outcome_on_nsaid_pop_profiling_all_vars_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py Combined Combined on nsaid

# sens_analysis_of_outcome_off_aceiarb population profiling
# model site and testing site
sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars: sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_UCLA_UCLA sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_UCLA_PSJH sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_PSJH_PSJH sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_PSJH_UCLA sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_Combined_Combined

sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA UCLA off aceiarb 
sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA PSJH off aceiarb
sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH PSJH off aceiarb
sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH UCLA off aceiarb
sens_analysis_of_outcome_off_aceiarb_pop_profiling_all_vars_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py Combined Combined off aceiarb

# sens_analysis_of_outcome_off_nsaid population profiling
# model site and testing site
sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars: sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_UCLA_UCLA sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_UCLA_PSJH sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_PSJH_PSJH sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_PSJH_UCLA sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_Combined_Combined

sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA UCLA off nsaid 
sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py UCLA PSJH off nsaid
sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH PSJH off nsaid
sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py PSJH UCLA off nsaid
sens_analysis_of_outcome_off_nsaid_pop_profiling_all_vars_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_medication_pat_profiling_all_vars.py Combined Combined off nsaid


## 19. Sensitivity analysis without using Do queries/just association on a set all variables to profile the population
# sens_analysis_of_outcome_ population profiling all variables just association
# model site and testing site
sens_analysis_of_outcome_pop_profiling_all_vars: sens_analysis_of_outcome_pop_profiling_all_vars_UCLA_UCLA sens_analysis_of_outcome_pop_profiling_all_vars_UCLA_PSJH sens_analysis_of_outcome_pop_profiling_all_vars_PSJH_PSJH sens_analysis_of_outcome_pop_profiling_all_vars_PSJH_UCLA sens_analysis_of_outcome_pop_profiling_all_vars_Combined_Combined

sens_analysis_of_outcome_pop_profiling_all_vars_UCLA_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_pat_profiling_all_vars.py UCLA UCLA  
sens_analysis_of_outcome_pop_profiling_all_vars_UCLA_PSJH:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_pat_profiling_all_vars.py UCLA PSJH 
sens_analysis_of_outcome_pop_profiling_all_vars_PSJH_PSJH: 
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_pat_profiling_all_vars.py PSJH PSJH 
sens_analysis_of_outcome_pop_profiling_all_vars_PSJH_UCLA:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_pat_profiling_all_vars.py PSJH UCLA 
sens_analysis_of_outcome_pop_profiling_all_vars_Combined_Combined:
	$(PYTHON_INTERPRETER) ./DBN_model_learning/sensitivity_analysis_simplest_model_pat_profiling_all_vars.py Combined Combined 


# ## 8. Running simple model
# simple_model_eval:
# 	$(PYTHON_INTERPRETER) ./DBN_model_learning/simple_model_vars_dataset.py

## Preprocess data for dashboard
image:
	docker build -t cure_ckd_dashboard .
