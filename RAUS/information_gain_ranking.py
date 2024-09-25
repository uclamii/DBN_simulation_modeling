from information_gain import *
import pandas as pd
import numpy as np

def IG_MI_Ranking(df,TARGET,COLS):

    """

    Efficiently learns variable ordering for unknown structure learning algorithms.


    """

    ig_mis = []
    cols = []
    tar = []
    for i in df[COLS].columns:
        ig_mi = comp_feature_information_gain(df, TARGET, i)
        ig_mis.append(ig_mi),cols.append(i)
        ig_mi_ranking = pd.DataFrame({'Information_Gain': ig_mis,'Variables': cols})
        ig_mi_ranking = ig_mi_ranking.set_index('Variables')
        ig_mi_ranking = ig_mi_ranking.sort_values(by='Information_Gain', ascending=False)
        ig_mi_ranking['Rank'] = range(0,len(ig_mi_ranking))
        ig_mi_ranking['Rank'] = ig_mi_ranking.Rank + 1
        ig_mi_ranking = ig_mi_ranking[['Rank', 'Information_Gain']]
        variable_order = ig_mi_ranking.index.tolist()
        for j in TARGET:
            tar.append(j)
            target = tar

    return ig_mi_ranking, variable_order, target
