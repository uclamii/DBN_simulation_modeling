from chi_squared import *
import pandas as pd

def ChiSquareRank(df, COLS, TARGET):

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

    chisquare = []
    cols = []
    tar = []
    count = 0
    for i in COLS:
        for j in TARGET:
            Chi_squared = ChiSquared(df[i],df[j])
            chisquare.append(Chi_squared), cols.append(i), tar.append(j)
            count +=1
            chisquarerank = pd.DataFrame({'Chi_Square': chisquare,'Variables': cols})
            chisquarerank = chisquarerank.set_index('Variables')
            chisquarerank = chisquarerank.sort_values(by='Chi_Square', ascending=False)
            chisquarerank['Rank'] = range(0,len(chisquarerank))
            chisquarerank['Rank'] = chisquarerank.Rank + 1
            chisquarerank = chisquarerank[['Rank', 'Chi_Square']]
            variable_order = chisquarerank.index.tolist()
            target = tar

    return chisquarerank, variable_order, target
