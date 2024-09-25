import numpy as np

def SelectPercentile(scores_df,score,predictors,percentile):
    """
    Takes as input the scores from the respective methods (e.g., chi2, cramersv, and information gain)
    and the variable names and selects features in descending order up to the set percentile.
    Assumes scores and variable names are in descending order.

    Input:
    - scores_df: contains the score
    - score: chi2, cramersv, or information gain in descending order
    - predictors: variable names in descending order
    - percentile: feature information threshold (e.g., 80)

    Output:
    - selectpct: reduced set of feature scores. If summed get value closest to the set percentile without going over
    - variables: the reduced set of features that gets input to the structure learning algorithms

    """
    scores = scores_df[score]
    norm = 1 / float(sum(scores))
    score_norm = [x * norm for x in scores]
    score_np = np.array(score_norm)
    selectpct = score_np[score_np.cumsum() <= percentile/100].tolist()
    variables = predictors[:len(selectpct)]

    return selectpct, variables
