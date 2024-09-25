from cramers_v import *
import pandas as pd

def CramersVRank(df, COLS, TARGET):

    """

    Efficiently learns variable ordering for unknown structure learning algorithms.

    Input arguments:
    - df: the dataframe
    - COLS: the list of predictor names
    - TARGET: the list of outcome name(s)

    Output/Returns:
    - raus: dataframe with the variables rank/order, effect size (Cramer's V), and p-value
    - variable_order: list of the variable order (descending effect size) to use as input to the unknown structure learning algorithm (e.g., K2 algorithm)
    - target: list of the target variable(s) to use as input to the unknown structure learning algorithm (e.g., K2 algorithm)

    """

    Cramers_Vs = []
    pvalues = []
    cols = []
    tar = []
    count = 0
    for i in COLS:
        for j in TARGET:
            Cramers_V, pvalue = CramersV(df[i],df[j])
            Cramers_Vs.append(Cramers_V), pvalues.append(pvalue), cols.append(i), tar.append(j)
            count +=1
            cramersv = pd.DataFrame({'Effect_Size': Cramers_Vs,'P_Value': pvalues,'Variables': cols})
            cramersv = cramersv.set_index('Variables')
            cramersv = cramersv.sort_values(by='Effect_Size', ascending=False)
            cramersv['Rank'] = range(0,len(cramersv))
            cramersv['Rank'] = cramersv.Rank + 1
            cramersv = cramersv[['Rank', 'Effect_Size', 'P_Value']]
            variable_order = cramersv.index.tolist()
            target = tar

    return cramersv, variable_order, target
