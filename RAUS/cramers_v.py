import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats import contingency_tables

def CramersV(predictors, target, margins= True, correction = None, cramer_correction = None):

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

    phi = np.sqrt(chi_square / n)

    if cramer_correction == True:
        phi_corrected = (chi_square / n) - ((num_row - 1) * (num_col - 1) / (n - 1))
        phi_corrected = max(0, phi_corrected)

        row_corrected = num_row - np.square(num_row - 1) / (n - 1)
        col_corrected = num_col - np.square(num_col - 1) / (n - 1)

        Cramers_V = np.sqrt(phi_corrected / min((num_row -1), (num_col - 1)))

    else:
        Cramers_V = np.sqrt(chi_square / (n * min((num_row - 1), (num_col - 1))))


    return Cramers_V, pvalue
