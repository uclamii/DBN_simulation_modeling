import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats import contingency_tables

def ChiSquared(predictors, target, margins= True, correction = None):

    crosstabulation = pd.crosstab(predictors, target)
    crosstabulation2 = pd.crosstab(predictors, target, margins = True)
    num_row = crosstabulation2.shape[0] - 1
    num_col = crosstabulation2.shape[1] - 1
    n = crosstabulation2.iloc[-1, -1]

    if correction == None:
        correction = False
    elif correction == True:
        correction = True

    chi_square, pvalue, dof, expected = scipy.stats.chi2_contingency(crosstabulation,correction = correction)

    return chi_square
