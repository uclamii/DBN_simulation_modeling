import numpy as np
from cramers_v_ranking import CramersVRank
from chi_squared_ranking import ChiSquareRank
from information_gain_ranking import IG_MI_Ranking
import pandas as pd
import pickle

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
from oct2py import octave
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import networkx as nx


def dbn_cols_intra(COLS, TARGET):
    """

    function that assigns the target to the second position

    """
    insert_position = 1
    COLS.insert(insert_position, TARGET[0])
    intra_cols = COLS

    # print(intra_cols)

    return intra_cols


def NodeSize(df, intra_cols):
    """

    function that gets the node sizes for input to the intra-structure and/or inter-structure functions

    """
    print("Learning the node sizes for the intra-structure and/or inter-structure...")
    ns_list = []
    count = 0
    for i in intra_cols:
        ns = len(df[i].value_counts())
        ns_list.append(ns)
        count += 1

    # print(ns_list)

    return ns_list


def Inter_NodeSize(df, intra_cols, inter_cols):
    """

    MDL discretization generates varying node sizes for same variables across timesteps.
    This function gets the node size for the second timestep for input to the learn inter-structure function.

    """
    print("Learning the node sizes for the inter-structure...")
    ns_list1 = []
    count = 0
    len_t1 = len(intra_cols)
    t1 = inter_cols[:len_t1]
    t2 = inter_cols[len_t1 : len_t1 * 2]
    for j in t2:
        ns = len(df[j].value_counts())
        ns_list1.append(ns)
        count += 1

    return ns_list1


def DBN_NodeSize(df, intra_cols, inter_cols):
    """

    MDL discretization generates varying node sizes for same variables across timesteps. This function gets the maximum node size for a variable across the timesteps for input to the create DBN function.

    """
    print("Learning the node sizes to create DBN...")
    ns_list1 = []
    ns_list2 = []
    ns_list3 = []
    ns_list4 = []
    count = 0
    len_t1 = len(intra_cols)
    t1 = inter_cols[:len_t1]
    t2 = inter_cols[len_t1 : len_t1 * 2]
    t3 = inter_cols[len_t1 * 2 : len_t1 * 3]
    t4 = inter_cols[len_t1 * 3 : len_t1 * 4]
    for i in t1:
        ns = len(df[i].value_counts())
        ns_list1.append(ns)
        count += 1
    for j in t2:
        ns = len(df[j].value_counts())
        ns_list2.append(ns)
        count += 1
    for k in t3:
        ns = len(df[k].value_counts())
        ns_list3.append(ns)
        count += 1
    for l in t4:
        ns = len(df[l].value_counts())
        ns_list4.append(ns)
        count += 1
    ns_list5 = [
        max((w, x, y, z), key=abs)
        for w, x, y, z in zip(ns_list1, ns_list2, ns_list3, ns_list4)
    ]

    return ns_list5


def dbn_cols_inter(
    intra_cols,
    sequence_length_dbn,
    cols_start,
    new_cols_start,
    cols_end,
    newer_cols_start,
    outcome_name,
):
    """
    function that keeps the learned variable ranking (list comprehension methods may reorder the columns)
    and produces the temporal columns for input to the inter-structure and DBN functions as well as produces the targets for
    input to the performance metrics
    Note: if using an outcome_name that is not preconfigured in the framework then
    update the targets variable (i.e., it should match your temporal outcomes, where m is the range of your temporal outcomes)
    and the inter_colss variable (i.e., should match your temporal observations, where m is the range of your temporal observations)

    """

    liss = []
    targets = []
    inter_cols = intra_cols * sequence_length_dbn
    # print(inter_cols)
    for i in range(len(inter_cols)):
        if outcome_name == "egfr_reduction40_ge":
            if inter_cols[i].startswith(cols_start):
                inter_cols[i] = inter_cols[i].replace(cols_start, new_cols_start)
                liss.append(inter_cols[i])
            elif outcome_name == "egfr_reduction40_ge":
                targets = [
                    "year" + str(m) + "_" + "reduction_40_ge"
                    for m in range(2, sequence_length_dbn + 2)
                ]
        elif outcome_name == "AKI_BOS24":
            if inter_cols[i].endswith(cols_start):
                inter_cols[i] = inter_cols[i].replace(cols_start, new_cols_start)
                liss.append(inter_cols[i])
            else:
                targets = [
                    "aki_progression_" + str(m) + "days"
                    for m in range(2, sequence_length_dbn + 2)
                ]
        elif outcome_name == "AKI_BOS48":
            if inter_cols[i].endswith(cols_start):
                inter_cols[i] = inter_cols[i].replace(cols_start, new_cols_start)
                liss.append(inter_cols[i])
            else:
                targets = [
                    "aki_progression_" + str(m) + "days"
                    for m in range(3, sequence_length_dbn + 3)
                ]
        elif outcome_name == "AKI_BOS72":
            if inter_cols[i].endswith(cols_start):
                inter_cols[i] = inter_cols[i].replace(cols_start, new_cols_start)
                liss.append(inter_cols[i])
            else:
                targets = [
                    "aki_progression_" + str(m) + "days"
                    for m in range(4, sequence_length_dbn + 4)
                ]

    for j in range(0, len(liss)):
        if outcome_name == "egfr_reduction40_ge":
            if liss[j].startswith(newer_cols_start):
                K = len(intra_cols) - 1
                inter_colss = [
                    "year" + str(m) for m in range(1, sequence_length_dbn + 1)
                ]  # add 1 to include final timepoint in cols
                res = [ele for ele in inter_colss for y in range(K)]
                inter_colss = [o + p for o, p in zip(res, liss)]
                n = 1  # outcome node position
                while n < len(inter_colss):
                    for p in targets:
                        inter_colss.insert(n, p)
                        n += len(intra_cols)
        elif outcome_name == "AKI_BOS24":
            if liss[j].endswith(newer_cols_start):
                K = len(intra_cols) - 1
                inter_colss = [
                    "24hourperiod_" + str(m) for m in range(0, sequence_length_dbn + 1)
                ]  # add 1 to include final timepoint in cols
                res = [ele for ele in inter_colss for y in range(K)]
                inter_colss = [o + p for o, p in zip(liss, res)]
                n = 1  # outcome node position
                while n < len(inter_colss):
                    for p in targets:
                        inter_colss.insert(n, p)
                        n += len(intra_cols)
            # print(inter_colss)
        elif outcome_name == "AKI_BOS48":
            if liss[j].endswith(newer_cols_start):
                K = len(intra_cols) - 1
                inter_colss = [
                    "24hourperiod_" + str(m) for m in range(0, sequence_length_dbn + 1)
                ]  # add 1 to include final timepoint in cols
                res = [ele for ele in inter_colss for y in range(K)]
                inter_colss = [o + p for o, p in zip(liss, res)]
                n = 1  # outcome node position
                while n < len(inter_colss):
                    for p in targets:
                        inter_colss.insert(n, p)
                        n += len(intra_cols)
        elif outcome_name == "AKI_BOS72":
            if liss[j].endswith(newer_cols_start):
                K = len(intra_cols) - 1
                inter_colss = [
                    "24hourperiod_" + str(m) for m in range(0, sequence_length_dbn + 1)
                ]  # add 1 to include final timepoint in cols
                res = [ele for ele in inter_colss for y in range(K)]
                inter_colss = [o + p for o, p in zip(liss, res)]
                n = 1  # outcome node position
                while n < len(inter_colss):
                    for p in targets:
                        inter_colss.insert(n, p)
                        n += len(intra_cols)

    return inter_colss, targets


def intra_struct(
    df, intra_cols, ns_list, method, outcome_name, site, adjusted, track, max_fan_in
):
    """
    function that connects to the matlab code and BNT package to learn the intra structure

    """
    print("Learning the intra-structure...")
    intraLength = len(ns_list)

    max_fan_in = (
        max_fan_in - 1
    )  # intra-structure in BNT returns a structure where the maximum number of parents is (max_fan_in value) + 1. Therefore subtract 1 to return structure with hdesired number of parents.

    df = df[intra_cols].copy()
    df = df.values
    dag = octave.intraStructLearn(df.T, intraLength, max_fan_in, ns_list)

    dag2 = pd.DataFrame(
        dag, columns=intra_cols
    )  # for easier review to redraw structures in another software
    dag2.index = (
        intra_cols  # for easier review to redraw structures in another software
    )
    path = "./intraStructures/matrix_form"

    dag2.to_csv(
        path
        + "/"
        + method
        + "_intra_structure_w_column_names_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".csv"
    )  # for easier review to redraw structures in another software

    np.save(
        path
        + "/"
        + method
        + "_intra_structure_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track,
        dag,
    )

    # print(dag)

    return dag


def BNModel_TT(
    df,
    df2,
    intra_cols,
    max_iter,
    dag,
    ns_list,
    method,
    outcome_name,
    site,
    adjusted,
    sequence_length_bn,
    track,
    ncases,
):
    """
    function that learns the BN model, parameterizes the network, and produces the marginal probabilities
    Note: use this function when you have train/test (TT) split

    """
    print("Learning the BN Model for TT...")

    intraLength = len(ns_list)
    numNodes = intraLength
    horizon = sequence_length_bn

    df = df[intra_cols].copy()
    # df = df.dropna() #comment this out when not beta testing
    dataTrain = df.values
    df2 = df2[intra_cols].copy()
    # df2 = df2.dropna() #comment this out when not beta testing
    dataValid = df2.values

    print("Starting BNT Function")
    dataTrainValid = octave.BNModelLearn_TT(
        dag,
        intraLength,
        numNodes,
        max_iter,
        dataTrain,
        dataValid,
        ns_list,
        horizon,
        ncases,
    )
    path = "./BNModel"

    np.save(
        path
        + "/"
        + method
        + "_BNModel_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track,
        dataTrainValid,
    )

    # print(dag)

    return dataTrainValid


def BNModel_TVT(
    df,
    df2,
    df3,
    intra_cols,
    max_iter,
    dag,
    ns_list,
    method,
    outcome_name,
    site,
    adjusted,
    sequence_length_bn,
    track,
    ncases,
    onlyValid=1,
):
    """
    function that learns the BN model, parameterizes the network, and produces the marginal probabilities
    Note: use this function when you have train/valid/test (TVT) split

    """
    print("Learning the BN Model for TVT...")

    intraLength = len(ns_list)
    numNodes = intraLength
    horizon = sequence_length_bn

    df = df[intra_cols].copy()
    # df = df.dropna() #comment this out when not beta testing
    dataTrain = df.values
    df2 = df2[intra_cols].copy()
    # df2 = df2.dropna() #comment this out when not beta testing
    dataValid = df2.values
    df3 = df3[intra_cols].copy()
    # df3 = df3.dropna() #comment this out when not beta testing
    dataTest = df3.values
    print("Starting BNT Function")
    dataTrainValid = octave.BNModelLearn_TVT(
        dag,
        intraLength,
        numNodes,
        max_iter,
        dataTrain,
        dataValid,
        dataTest,
        ns_list,
        horizon,
        ncases,
        onlyValid,
    )
    path = "./BNModel"

    np.save(
        path
        + "/"
        + method
        + "_BNModel_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track,
        dataTrainValid,
    )

    # print(dag)

    return dataTrainValid


def inter_struct(
    df,
    inter_cols,
    ns_list,
    method,
    outcome_name,
    site,
    adjusted,
    sequence_length_dbn,
    track,
    max_fan_in,
):
    """

    function that connects to the Matlab code and BNT package and learns the inter-structure

    """
    print("Learning the inter-structure...")
    intraLength = len(ns_list)

    df = df[inter_cols].copy()
    # print(df.shape)
    df_train_fullyobserved = df.dropna()
    # print(df_train_fullyobserved.shape)
    df_train_fullyobserved = df_train_fullyobserved.values

    horizon = sequence_length_dbn
    inter_structure = octave.interStructLearn(
        df_train_fullyobserved, ns_list, max_fan_in, intraLength, horizon
    )

    inter_structure2 = pd.DataFrame(
        inter_structure, columns=inter_cols[intraLength : intraLength * 2]
    )  # for easier review to redraw structures in another software
    inter_structure2.index = inter_cols[
        :intraLength
    ]  # for easier review to redraw structures in another software
    path = "./interStructures/matrix_form"

    inter_structure2.to_csv(
        path
        + "/"
        + method
        + "_inter_structure_w_column_names_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".csv"
    )  # for easier review to redraw structures in another software

    np.save(
        path
        + "/"
        + method
        + "_inter_structure_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track,
        inter_structure,
    )

    # print(inter_structure)

    return inter_structure


def DBNModel_TT(
    df,
    df2,
    inter_cols,
    inter_structure,
    sequence_length_dbn,
    max_iter,
    dag,
    ns_list,
    method,
    outcome_name,
    site,
    adjusted,
    track,
    ncases,
):
    """
    function that learns the DBN model, parameterizes the network, and produces the marginal probabilities
    Note: use this function when you have train/test (TT) split

    """
    print("Learning the DBN Model for TT...")

    intraLength = len(ns_list)
    numNodes = intraLength * 2
    horizon = sequence_length_dbn

    df = df[inter_cols].copy()
    # df = df.dropna() #comment this out when not beta testing
    dataTrain = df.values
    dataTrainMiss = df.values
    df2 = df2[inter_cols].copy()
    # df2 = df2.dropna() #comment this out when not beta testing
    dataValid = df2.values

    print("Starting BNT Function")
    dataTrainValid = octave.DBNModel_TT(
        inter_structure,
        dataTrain,
        dataTrainMiss,
        dataValid,
        ns_list,
        dag,
        max_iter,
        intraLength,
        horizon,
        numNodes,
        ncases,
    )
    path = "./DBNModel"

    np.save(
        path
        + "/"
        + method
        + "_DBNModel_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track,
        dataTrainValid,
    )

    # print(dag)

    return dataTrainValid


def DBNModel_TVT(
    df,
    df2,
    df3,
    inter_cols,
    inter_structure,
    sequence_length_dbn,
    max_iter,
    dag,
    ns_list,
    method,
    outcome_name,
    site,
    adjusted,
    track,
    ncases,
    onlyValid=1,
):
    """
    function that learns the DBN model, parameterizes the network, and produces the marginal probabilities
    Note: use this function when you have train/valid/test (TVT) split

    """
    print("Learning the DBN Model for TVT...")

    intraLength = len(ns_list)
    numNodes = intraLength * 2
    horizon = sequence_length_dbn

    df = df[inter_cols].copy()
    # df = df.dropna() #comment this out when not beta testing
    # print(df.shape)
    dataTrain = df.values
    dataTrainMiss = df.values
    df2 = df2[inter_cols].copy()
    # df2 = df2.dropna() #comment this out when not beta testing
    dataValid = df2.values
    df3 = df3[inter_cols].copy()  # df3 should be dataTest
    # df3 = df3.dropna() #comment this out when not beta testing
    dataTest = df3.values  # df3 should be dataTest
    print("Starting BNT Function")
    dataTrainValid = octave.DBNModel_TVT(
        inter_structure,
        dataTrain,
        dataTrainMiss,
        dataValid,
        dataTest,
        ns_list,
        dag,
        max_iter,
        intraLength,
        horizon,
        numNodes,
        ncases,
        onlyValid,
    )
    path = "./DBNModel"

    np.save(
        path
        + "/"
        + method
        + "_DBNModel_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track,
        dataTrainValid,
    )

    # print(dag)

    return dataTrainValid


def draw_intra_structure(
    adjacency_matrix,
    mylabels,
    method,
    outcome_name,
    site,
    adjusted,
    track,
    clipback,
    clipfront,
):
    """
    function that generates and saves the learned structures in graph visualization form

    """
    A2 = pd.DataFrame(adjacency_matrix, index=mylabels, columns=mylabels)
    gr = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())

    layout = nx.circular_layout(gr)

    plt.figure()
    nx.draw_networkx(gr, layout, node_size=50, font_size=3, edge_color="red")
    plt.axis("off")
    path = "./intraStructures/visualizations"

    figure1 = plt.savefig(
        path
        + "/"
        + method
        + "_intra_structure_visualization_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".svg",
        dpi=1200,
        format="svg",
    )
    figure2 = plt.savefig(
        path
        + "/"
        + method
        + "_intra_structure_visualization_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".png",
        dpi=1200,
        format="png",
    )

    return figure1, figure2


def draw_inter_structure(
    adjacency_matrix_2,
    mylabels,
    method,
    outcome_name,
    site,
    adjusted,
    track,
    clipback,
    clipfront,
):
    """
    function that generates and saves the learned structures in graph visualization form

    """
    A2 = pd.DataFrame(adjacency_matrix_2, index=mylabels, columns=mylabels)
    gr = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())

    layout = nx.circular_layout(gr)

    plt.figure()
    nx.draw_networkx(
        gr,
        layout,
        node_size=1400,
        font_size=5,
        edge_color="black",
        connectionstyle="arc3, rad = 0.1",
        style="dashed",
    )
    nx.draw(
        gr,
        layout,
        node_size=1400,
        node_color="teal",
        alpha=1,
        connectionstyle="arc3, rad = 0.1",
        style="dashed",
    )
    plt.axis("off")
    path = "./interStructures/visualizations"

    figure1 = plt.savefig(
        path
        + "/"
        + method
        + "_inter_structure_visualization_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".svg",
        dpi=1200,
        format="svg",
    )
    figure2 = plt.savefig(
        path
        + "/"
        + method
        + "_inter_structure_visualization_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".png",
        dpi=1200,
        format="png",
    )

    return figure1, figure2


def draw_compact_structure(
    adjacency_matrix_1,
    adjacency_matrix_2,
    mylabels,
    method,
    outcome_name,
    site,
    adjusted,
    track,
    clipback,
    clipfront,
):
    """
    function that generates and saves the learned structures in graph visualization form

    """
    A2 = pd.DataFrame(adjacency_matrix_1, index=mylabels, columns=mylabels)
    gr = nx.from_pandas_adjacency(A2, create_using=nx.MultiDiGraph())
    A3 = pd.DataFrame(adjacency_matrix_2, index=mylabels, columns=mylabels)
    gr2 = nx.from_pandas_adjacency(A3, create_using=nx.MultiDiGraph())

    layout = nx.spiral_layout(gr, equidistant=True, resolution=0.95)
    layout2 = nx.spiral_layout(gr2, equidistant=True, resolution=0.95)

    plt.figure()
    nx.draw_networkx(
        gr,
        layout,
        node_size=1200,
        node_color="teal",
        alpha=1,
        connectionstyle="arc3, rad = -0.1",
        font_size=5,
        font_color="black",
    )
    nx.draw_networkx(
        gr2,
        layout2,
        node_size=1200,
        node_color="teal",
        alpha=1,
        style="dashed",
        connectionstyle="arc3, rad = -0.2",
        font_size=5,
        font_color="black",
    )

    plt.axis("off")
    path = "./compactStructure/visualizations"

    figure1 = plt.savefig(
        path
        + "/"
        + method
        + "_compact_structure_visualization_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".svg",
        dpi=1200,
        format="svg",
    )
    figure2 = plt.savefig(
        path
        + "/"
        + method
        + "_compact_structure_visualization_"
        + outcome_name
        + "_"
        + site
        + adjusted
        + "_"
        + track
        + ".png",
        dpi=1200,
        format="png",
    )

    return figure1, figure2
