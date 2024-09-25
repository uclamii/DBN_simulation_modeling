import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    sites = [
        "UCLA",
        "Prov",
        "combined",
    ]

    for site in tqdm(sites):

        dfTrain = pd.read_csv(
            "./Data/genie_datasets/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_"
            + site
            + "_train.csv",
            # nrows=20000,
        )

        train_frames = []

        dfTrain_subset_1 = dfTrain.loc[dfTrain["year1_reduction_40_ge"] == "S_1"]
        dfTrain_subset_2 = dfTrain.loc[
            dfTrain["year1_reduction_40_ge"] == "S_0"
        ].sample(n=len(dfTrain_subset_1), random_state=123)
        dfTrain_subsets1 = dfTrain_subset_1.append(dfTrain_subset_2)
        train_frames.append(dfTrain_subsets1)
        dfTrain_subset_3 = dfTrain.loc[dfTrain["year2_reduction_40_ge"] == "S_1"]
        dfTrain_subset_4 = dfTrain.loc[
            dfTrain["year2_reduction_40_ge"] == "S_0"
        ].sample(n=len(dfTrain_subset_3), random_state=123)
        dfTrain_subsets2 = dfTrain_subset_3.append(dfTrain_subset_4)
        train_frames.append(dfTrain_subsets2)
        dfTrain_subset_5 = dfTrain.loc[dfTrain["year3_reduction_40_ge"] == "S_1"]
        dfTrain_subset_6 = dfTrain.loc[
            dfTrain["year3_reduction_40_ge"] == "S_0"
        ].sample(n=len(dfTrain_subset_5), random_state=123)
        dfTrain_subsets3 = dfTrain_subset_5.append(dfTrain_subset_6)
        train_frames.append(dfTrain_subsets3)
        dfTrain_subset_7 = dfTrain.loc[dfTrain["year4_reduction_40_ge"] == "S_1"]
        dfTrain_subset_8 = dfTrain.loc[
            dfTrain["year4_reduction_40_ge"] == "S_0"
        ].sample(n=len(dfTrain_subset_7), random_state=123)
        dfTrain_subsets4 = dfTrain_subset_7.append(dfTrain_subset_8)
        train_frames.append(dfTrain_subsets4)
        dfDBNTrain = pd.concat(train_frames, axis=0, ignore_index=True)
        dfTrain_subset_9 = dfTrain.loc[dfTrain["year5_reduction_40_ge"] == "S_1"]
        dfTrain_subset_10 = dfTrain.loc[
            dfTrain["year5_reduction_40_ge"] == "S_0"
        ].sample(n=len(dfTrain_subset_9), random_state=123)
        dfTrain_subsets5 = dfTrain_subset_9.append(dfTrain_subset_10)
        train_frames.append(dfTrain_subsets5)
        dfDBNTrain = pd.concat(train_frames, axis=0, ignore_index=True)

        dfDBNTrain.to_csv(
            "./Data/genie_datasets/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_"
            + site
            + "_train_undersampled_DBN.csv",
            index=False,
        )
