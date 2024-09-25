from cramers_v_ranking import CramersVRank
from chi_squared_ranking import ChiSquareRank
from information_gain_ranking import IG_MI_Ranking
import pandas as pd
import matplotlib.pyplot as plt

def RAUSPlot(df,col1,clipback,clipfront,variableorder,site,outcome_name,adjusted,YLABEL,track):
    """
    Plot of the discrete variable Rank ordering Approach for Uknown Structure (RAUS) learning algorithms

    """
    rank_order = df[col1]
    plt.bar([x for x in range(len(rank_order))],rank_order)
    _ = plt.xlabel('Variable Order')
    _ = plt.ylabel(YLABEL)
    _ = plt.title('Rank for ' + site + adjusted)
    variableorder_short = [s.replace(clipback, "") for s in variableorder]
    variableorder_short = [s.replace(clipfront, "") for s in variableorder_short]

    plt.xticks(range(len(variableorder_short)), variableorder_short,size='small',rotation='vertical',fontsize=5)
    if col1 == 'Effect_Size':
        save_figure1 = plt.savefig('./rank_figures/cramersv/' + 'Rank_for_' + site + '_' + col1 + '_DBN_' + outcome_name + '_' + adjusted+ '_' + track + '.svg', dpi=1200, bbox_inches='tight', format='svg')
        save_figure2 = plt.savefig('./rank_figures/cramersv/' + 'Rank_for_' + site + '_' + col1 + '_DBN_' + outcome_name + '_' + adjusted+ '_' + track + '.png', dpi=1200, bbox_inches='tight', format='png')
    elif col1 == 'Chi_Square':
        save_figure1 = plt.savefig('./rank_figures/chisquared/' + 'Rank_for_' + site + '_' + col1 + '_DBN_' + outcome_name + '_' + adjusted+ '_' + track + '.svg', dpi=1200, bbox_inches='tight', format='svg')
        save_figure2 = plt.savefig('./rank_figures/chisquared/' + 'Rank_for_' + site + '_' + col1 + '_DBN_' + outcome_name + '_' + adjusted+ '_' + track + '.png', dpi=1200, bbox_inches='tight', format='png')
    elif col1 == 'Information_Gain':
        save_figure1 = plt.savefig('./rank_figures/ig/' + 'Rank_for_' + site + '_' + col1 + '_DBN_' + outcome_name + '_' + adjusted+ '_' + track + '.svg', dpi=1200, bbox_inches='tight', format='svg')
        save_figure2 = plt.savefig('./rank_figures/ig/' + 'Rank_for_' + site + '_' + col1 + '_DBN_' + outcome_name + '_' + adjusted+ '_' + track + '.png', dpi=1200, bbox_inches='tight', format='png')

    return save_figure1,save_figure2
