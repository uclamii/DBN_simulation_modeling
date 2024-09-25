# This is the Simulated interVention (SimVene) software for predicting 40% eGFR decline.

- SimVene reduces the complexity for implementing and managing simulated interventions to support clinical decision making.

- This library automates preprocessing, unknown structure learning (via RAUS), model evaluation, and sensitivity analysis to include sharp regression discontinuity and do-calculus.

- We introduce an "orchestration card" to help manage the RAUS implementation across multiple virtual machines, scenarios, sites, and substructures.

- We introduce a sharp regression discontinuity design implementation to randomize patients into control and treatment groups.

- We introduce a do-calculus implementation to delete incoming edges to desired interventions from parents.

- SimVene is built on top of Ranking Approaches for Unknown Structures (RAUS) by [Gordon et al.](https://github.com/dgrdn08/RAUS) and Bayes Net Toolbox (BNT) by [Murphy et al.](https://github.com/bayesnet/bnt)

# Cite

- Please cite the SimVene software if you use the SimVene software in your work.

INSERT ZENODO DOI HERE

## The SimVene software is first implemented in the papers: 
1) "Predicting ≥40% eGFR Decline Using Dynamic Bayesian Networks in Persons With or At-Risk of Chronic Kidney Disease" by Panayiotis Petousis*, David Gordon*, Susanne B. Nicholas, and Alex A.T. Bui on behalf of CURE-CKD
2) "Testing Causal Explanations: A Case Study for Understanding Chronic Kidney Disease" by Panayiotis Petousis*, David Gordon, Susanne B. Nicholas, and Alex A.T. Bui on behalf of CURE-CKD

3) "Using Bayesian Networks to Simulate eGFR Trajectories" published at the 34th meeting of the European Renal Cell Study Group (ERCSG) Abstract.
##### *First authors had equal contributions.

# How to use SimVene

To install the requirements, run: (Note use Python 3.7.4)

```
$ make requirements
```

# Example Commands

## Preprocessing

To preprocess the data, run:

```
$ make preprocess_data
```

To split the data (Train: 60%, Validation: 20%, Test: 20%), run:

```
$ make split_data
```

To discretize the data for all sites (Combined, UCLA, Providence), run:

```
$ make discretize_data
$ make discretize_data_using_UCLA
$ make discretize_data_using_Prov
```

## Unknown Structure Learning via RAUS

To format data for input to RAUS, run:

```
$ make raus_format_data
```

#### Setup RAUS environment requiremnts by following environment setup [instrunctions](https://github.com/dgrdn08/RAUS)

**Note: Each scenario represents a different set of variables used for structure learning. For the above Papers scenario 2 was used, as it represented the most clinically interpretable model.**

To perform unknown structure learning (via RAUS) for Scenario 1 (36 CPUs in parallel), run:

```
$ make -j 4 scenario_1_1
$ make -j 4 scenario_1_2
$ make -j 4 scenario_1_3
```
or, run:

```
$ make learn_UCLA_contemporals & make learn_UCLA_initial_conditions & make learn_UCLA_intra_for_full_DBN & make learn_UCLA_inter_for_full_DBN
$ make learn_PSJH_contemporals & make learn_PSJH_initial_conditions & make learn_PSJH_intra_for_full_DBN & make learn_PSJH_inter_for_full_DBN
$ make learn_Combined_contemporals & make learn_Combined_initial_conditions & make learn_Combined_intra_for_full_DBN & make learn_Combined_inter_for_full_DBN
```

To perform unknown structure learning (via RAUS) for Scenario 2 (36 CPUs in parallel), run:

```
$ make -j 4 scenario_2_1
$ make -j 4 scenario_2_2
$ make -j 4 scenario_2_3
```
or, run:

```
$ make learn_UCLA_contemporals_norace & make learn_UCLA_initial_conditions_norace & make learn_UCLA_intra_for_full_DBN_norace & make learn_UCLA_inter_for_full_DBN_norace
$ make learn_PSJH_contemporals_norace & make learn_PSJH_initial_conditions_norace & make learn_PSJH_intra_for_full_DBN_norace & make learn_PSJH_inter_for_full_DBN_norace
$ make learn_Combined_contemporals_norace & make learn_Combined_initial_conditions_norace & make learn_Combined_intra_for_full_DBN_norace & make learn_Combined_inter_for_full_DBN_norace
```

To perform unknown structure learning (via RAUS) for Scenario 3 (36 CPUs in parallel), run:

```
$ make -j 4 scenario_3_1
$ make -j 4 scenario_3_2
$ make -j 4 scenario_3_3
```
or, run:

```
$ make learn_UCLA_counts_contemporals & make learn_UCLA_counts_initial_conditions & make learn_UCLA_counts_intra_for_full_DBN & make learn_UCLA_counts_inter_for_full_DBN
$ make learn_PSJH_counts_contemporals & make learn_PSJH_counts_initial_conditions & make learn_PSJH_counts_intra_for_full_DBN & make learn_PSJH_counts_inter_for_full_DBN
$ make learn_Combined_counts_contemporals & make learn_Combined_counts_initial_conditions & make learn_Combined_counts_intra_for_full_DBN & make learn_Combined_counts_inter_for_full_DBN
```

To perform unknown structure learning (via RAUS) for Scenario 4 (36 CPUs in parallel), run:

```
$ make -j 4 scenario_4_1
$ make -j 4 scenario_4_2
$ make -j 4 scenario_4_3
```
or, run:

```
$ make learn_UCLA_counts_contemporals_norace & make learn_UCLA_counts_initial_conditions_norace & make learn_UCLA_counts_intra_for_full_DBN_norace & make learn_UCLA_counts_inter_for_full_DBN_norace
$ make learn_PSJH_counts_contemporals_norace & make learn_PSJH_counts_initial_conditions_norace & make learn_PSJH_counts_intra_for_full_DBN_norace & make learn_PSJH_counts_inter_for_full_DBN_norace
$ make learn_Combined_counts_contemporals_norace & make learn_Combined_counts_initial_conditions_norace & make learn_Combined_counts_intra_for_full_DBN_norace & make learn_Combined_counts_inter_for_full_DBN_norace
```

### Just for Paper 3 - use the following script (scenario 4 only UCLA) - for other papers continue to next steps

```
python DBN_model_learning/ERSA_model/full_model_BN_vars_dataset.py
```

## Model Parameter Learning and Evaluation

Note: to use SMILE the API for Bayesian Networks by Bayesfusion you need to obtain a free academic license and replace the existing license file DBN_model_learning/pysmile_license.py. (https://www.bayesfusion.com/downloads/)

To format data for GeNIe, run:

```
$ make genie_format_data
```

To undersample the training set, run:

```
$ make data_undersampling
```

To train the full Bayesian networks, run:

```
$ make full_BNs_models_eval
```

To train the full dynamic Bayesian networks, run:

```
$ make full_DBNs_models_eval
```

To get the results for all the full dynamic Bayesian networks, run:

```
$ make full_DBNs_models_results
```

To get the results for all the full DBNs with bootstrap (stratified to maintain outcome proportions in sampling) and Confidence Intervals (CIs):

```
$ make full_DBNs_models_results_with_CIs_stratified
```

To get the results for all the full DBNs with bootstrap (balanced to maintain 50:50 outcome proportions in sampling) and Confidence Intervals (CIs):
```
$ make full_DBNs_models_results_with_CIs_balanced
```

To evaluate the statistical significance between the full dynamic Bayesian networks, run (not necessary for papers results - can be skipped):

```
$ make full_DBNs_models_stat_sign
```

To get the results for the best full dynamic Bayesian networks, run (not necessary for papers results - can be skipped):

```
$ make get_best_DBNs_models_results
```

To visualize the performance of all models, run:

```
$ make models_perf_visualization
```

To visualize the performance of all models on stratified bootstrap and CIs, run:
```
$ make models_perf_visualization_stratified
```

To visualize the performance of all models on balanced bootstrap and CIs, run:
```
$ make models_perf_visualization_balanced
```

To visualize the performance of all models on external datasets, run:

```
$ make models_perf_ext_datasets_visualization
```

To visualize the performance of all models on stratified bootstrap and CIs on external datasets, run:
```
$ make models_perf_ext_datasets_visualization_stratified
```

To visualize the performance of all models on balanced bootstrap and CIs on external datasets, run:
```
$ make models_perf_ext_datasets_visualization_balanced
```

# Paper 2

## Associational Analysis and Causal analysis

To perform sensitivity/causal analysis using Do queries on every variables model site and testing site, run:

```
$ make sens_analysis_of_outcome
```
or by site, run:

```
$ make sens_analysis_of_outcome_UCLA_UCLA
$ make sens_analysis_of_outcome_UCLA_PSJH
$ make sens_analysis_of_outcome_PSJH_PSJH
$ make sens_analysis_of_outcome_PSJH_UCLA
$ make sens_analysis_of_outcome_Combined_Combined
```

To perform sensitivity analysis alone (associational) without using Do queries on every variables model site and testing site, run:

```
$ make sens_analysis_of_outcome_pop_profiling_all_vars
```
or by site, run:

```
$ make sens_analysis_of_outcome_pop_profiling_all_vars_UCLA_UCLA
$ make sens_analysis_of_outcome_pop_profiling_all_vars_UCLA_PSJH
$ make sens_analysis_of_outcome_pop_profiling_all_vars_PSJH_PSJH
$ make sens_analysis_of_outcome_pop_profiling_all_vars_PSJH_UCLA
$ make sens_analysis_of_outcome_pop_profiling_all_vars_Combined_Combined
```


## To visualize the sensitivity analysis via Jupyter Notebook, run:

Note: change following variables to create combinations of sites.
Example:
model_site = "UCLA"
testing_site = "PSJH"

```
$ jupyter notebook ./notebooks/Sensitivity analysis.ipynb
$ jupyter notebook ./notebooks/Sensitivity analysis association.ipynb
```

One notebook for causal analysis and one notebook for the associational analysis
