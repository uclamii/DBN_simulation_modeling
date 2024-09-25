import pandas as pd


def temporal_undersampling(dfTrain,TARGET,outcome_name,method,site,adjusted,track):
    """
    function that undersamples the majority class at each timestep
    Note: if using an outcome_name that is not preconfigured in the framework then update the temporal outcomes

    """
    train_frames = []
    if outcome_name == 'egfr_reduction40_ge':
        dfTrain_subset_1 = dfTrain.loc[dfTrain["year1_reduction_40_ge"] == 2]
        dfTrain_subset_2 = dfTrain.loc[dfTrain["year1_reduction_40_ge"] ==1].sample(n=len(dfTrain_subset_1),random_state=123)
        dfTrain_subsets1 = dfTrain_subset_1.append(dfTrain_subset_2)
        train_frames.append(dfTrain_subsets1)
        dfTrain_subset_3 = dfTrain.loc[dfTrain['year2_reduction_40_ge'] == 2]
        dfTrain_subset_4 = dfTrain.loc[dfTrain['year2_reduction_40_ge'] ==1 ].sample(n=len(dfTrain_subset_3),random_state=123)
        dfTrain_subsets2 = dfTrain_subset_3.append(dfTrain_subset_4)
        train_frames.append(dfTrain_subsets2)
        dfTrain_subset_5 = dfTrain.loc[dfTrain['year3_reduction_40_ge'] == 2]
        dfTrain_subset_6 = dfTrain.loc[dfTrain['year3_reduction_40_ge'] ==1 ].sample(n=len(dfTrain_subset_5),random_state=123)
        dfTrain_subsets3 = dfTrain_subset_5.append(dfTrain_subset_6)
        train_frames.append(dfTrain_subsets3)
        dfTrain_subset_7 = dfTrain.loc[dfTrain['year4_reduction_40_ge'] == 2]
        dfTrain_subset_8 = dfTrain.loc[dfTrain['year4_reduction_40_ge'] ==1 ].sample(n=len(dfTrain_subset_7),random_state=123)
        dfTrain_subsets4 = dfTrain_subset_7.append(dfTrain_subset_8)
        train_frames.append(dfTrain_subsets4)
        dfTrain_subset_9 = dfTrain.loc[dfTrain['year5_reduction_40_ge'] == 2]
        dfTrain_subset_10 = dfTrain.loc[dfTrain['year5_reduction_40_ge'] ==1 ].sample(n=len(dfTrain_subset_9),random_state=123)
        dfTrain_subsets5 = dfTrain_subset_9.append(dfTrain_subset_10)
        train_frames.append(dfTrain_subsets5)
        dfDBNTrain = pd.concat(train_frames,axis=0,ignore_index=True)
        path = './USTrain'
        dfDBNTrain.to_csv(path + '/' +  method + '_USTrain_' + outcome_name + '_' + site + adjusted+ '_' + track + '.csv')
    elif outcome_name == 'AKI_BOS24' or outcome_name == 'AKI_BOS48' or outcome_name == 'AKI_BOS72':
        dfTrain_subset_1 = dfTrain.loc[dfTrain["aki_progression_1days"] == 2]
        dfTrain_subset_2 = dfTrain.loc[dfTrain["aki_progression_1days"] == 1].sample(n=len(dfTrain_subset_1),random_state=123)
        dfTrain_subsets1 = dfTrain_subset_1.append(dfTrain_subset_2)
        train_frames.append(dfTrain_subsets1)
        dfTrain_subset_3 = dfTrain.loc[dfTrain["aki_progression_2days"] == 2]
        dfTrain_subset_4 = dfTrain.loc[dfTrain["aki_progression_2days"] == 1].sample(n=len(dfTrain_subset_3),random_state=123)
        dfTrain_subsets2 = dfTrain_subset_3.append(dfTrain_subset_4)
        train_frames.append(dfTrain_subsets2)
        dfTrain_subset_5 = dfTrain.loc[dfTrain['aki_progression_3days'] == 2]
        dfTrain_subset_6 = dfTrain.loc[dfTrain['aki_progression_3days'] == 1 ].sample(n=len(dfTrain_subset_5),random_state=123)
        dfTrain_subsets3 = dfTrain_subset_5.append(dfTrain_subset_6)
        train_frames.append(dfTrain_subsets3)
        dfTrain_subset_7 = dfTrain.loc[dfTrain['aki_progression_4days'] == 2]
        dfTrain_subset_8 = dfTrain.loc[dfTrain['aki_progression_4days'] ==1 ].sample(n=len(dfTrain_subset_7),random_state=123)
        dfTrain_subsets4 = dfTrain_subset_7.append(dfTrain_subset_8)
        train_frames.append(dfTrain_subsets4)
        dfTrain_subset_9 = dfTrain.loc[dfTrain['aki_progression_5days'] == 2]
        dfTrain_subset_10 = dfTrain.loc[dfTrain['aki_progression_5days'] ==1 ].sample(n=len(dfTrain_subset_9),random_state=123)
        dfTrain_subsets5 = dfTrain_subset_9.append(dfTrain_subset_10)
        train_frames.append(dfTrain_subsets5)
        dfTrain_subset_11 = dfTrain.loc[dfTrain['aki_progression_6days'] == 2]
        dfTrain_subset_12 = dfTrain.loc[dfTrain['aki_progression_6days'] ==1 ].sample(n=len(dfTrain_subset_11),random_state=123)
        dfTrain_subsets6 = dfTrain_subset_11.append(dfTrain_subset_12)
        train_frames.append(dfTrain_subsets6)
        dfTrain_subset_13 = dfTrain.loc[dfTrain['aki_progression_7days'] == 2]
        dfTrain_subset_14 = dfTrain.loc[dfTrain['aki_progression_7days'] ==1 ].sample(n=len(dfTrain_subset_13),random_state=123)
        dfTrain_subsets7 = dfTrain_subset_13.append(dfTrain_subset_14)
        train_frames.append(dfTrain_subsets7)
        dfDBNTrain = pd.concat(train_frames,axis=0,ignore_index=True)
        path = './USTrain'
        dfDBNTrain.to_csv(path + '/' +  method + '_USTrain_' + outcome_name + '_' + site + adjusted+ '_' + track + '.csv')

    #print(dfDBNTrain.shape)

    return dfDBNTrain
