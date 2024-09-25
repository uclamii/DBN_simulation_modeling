import pandas as pd
import numpy as np

def compute_impurity(feature):
    """
    This function calculates impurity of a feature.
    Supported impurity criteria: 'entropy'
    input: feature (this needs to be a Pandas series)
    output: feature impurity
    """
    probs = feature.value_counts(normalize=True)


    impurity = -1 * np.sum(np.log2(probs) * probs)

    return(round(impurity, 3))

def comp_feature_information_gain(df, t, descriptive_feature):
    """
    This function calculates information gain for splitting on
    a particular descriptive feature for a given dataset
    and a given impurity criteria.
    Supported split criterion: 'entropy', 'gini'
    """
    target_entropy = compute_impurity(df[t])

    entropy_list = list()
    weight_list = list()

    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level[t])
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))

    information_gain = target_entropy - feature_remaining_impurity

    return information_gain
