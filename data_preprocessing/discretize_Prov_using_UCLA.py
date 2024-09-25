import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from pickleObjects import *
from MDLDiscretization import *
from CONSTANTS import *

reduction_cols = ["year" + str(i) + "_reduction" for i in range(1, 14)] + [
    "year" + str(i) + "_norace_reduction" for i in range(1, 14)
]

reductions_40_ge_cols = [col + "_40_ge" for col in reduction_cols]

cont_cols = old_CTN_ENTRY_COLS + old_TIME_ZERO_COLS + old_CTN_COLS

cat_cols = old_CAT_COLS + reductions_40_ge_cols  # + old_RACE_COLS

target = "egfr_reduction40_flag"

if __name__ == "__main__":
    DATA_PATH_read = (
        "./Data/" + "cure_ckd_egfr_registry_preprocessed_project_preproc_data.csv"
    )

    DATA_PATH_write = "./Data/split_discetized_datasets_using_UCLA_discritizer/"

    # read dataset
    df = pd.read_csv(DATA_PATH_read.replace(".csv", "_excl_crit_applied.csv"))

    ##

    dataset_names = [
        "UCLA",
        "combined",
        "Prov",
    ]

    print("Discretizing " + " Providence using UCLA discritizer" + " .... ")

    #### Providence
    cond_Prov = df["site_source_cat"] == 1
    df_Prov = df[cond_Prov].copy().reset_index(drop=True)

    ################## creating dataset of only continuous columns and the label
    df_Prov_cont = pd.DataFrame(
        data=np.c_[
            df_Prov[cont_cols],
            df_Prov[target],
        ],
        columns=cont_cols + ["target"],
    )

    #########################################################
    ################### Loading UCLA discritizer

    discritizer = loadObjects("Data/discritizers/" + dataset_names[0] + "_discritizer")

    #########################################################
    print("\n")
    print("Transforming the data ...... ")

    # transform data
    df_Prov_cont_discr = discritizer.transform(df=df_Prov_cont)
    df_Prov_cont_discr_label_enc = discritizer.transform(df=df_Prov_cont, mapped=False)

    #######################################################
    ######################## Discritized data

    ### Adding discretized data to original columns
    df_Prov[cont_cols] = df_Prov_cont_discr[cont_cols]

    ################### saving discretized data
    df_Prov.to_csv(
        DATA_PATH_write
        + dataset_names[2]
        + "_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_using_"
        + dataset_names[0]
        + "_discritizer.csv",
        index=False,
    )

    print(df_Prov[:3])

    #######################################################
    ######################## Label encoded data

    ### Adding discretized data to original columns
    df_Prov[cont_cols] = df_Prov_cont_discr_label_enc[cont_cols]

    # target
    # df_Prov[target] = df_Prov[target] + 1  # +1 octave format

    ############# label encoder for categorical data
    ############# only used for structure learning
    dfTrain_UCLA = pd.read_csv(
        "Data/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_UCLA_train.csv"
    )
    for cat_col in tqdm(cat_cols):
        if cat_col not in ["patient_id", "site_source_cat"]:
            le = preprocessing.LabelEncoder()

            # train
            null_mask_train = dfTrain_UCLA[cat_col].isnull()
            dfTrain_UCLA.loc[null_mask_train, cat_col] = "NaN"
            categories = dfTrain_UCLA[cat_col].dropna().unique().tolist()
            le.fit(categories)

            null_mask_df_Prov = df_Prov[cat_col].isnull()
            df_Prov.loc[null_mask_df_Prov, cat_col] = "NaN"
            # ruca starts at 1 no need to add 1
            if cat_col not in ["ruca_7_class", "ruca_4_class"]:
                df_Prov[cat_col] = (
                    le.transform(df_Prov[cat_col]) + 1  # +1 octave format
                )  # +1 octave format
            else:
                df_Prov[cat_col] = le.transform(df_Prov[cat_col])
            df_Prov.loc[null_mask_df_Prov, cat_col] = np.nan
        else:
            pass

    ################### saving discretized data
    df_Prov.to_csv(
        DATA_PATH_write
        + dataset_names[2]
        + "_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_using_"
        + dataset_names[0]
        + "_discritizer.csv",
        index=False,
    )

    print("Transformed data: ")
    print(df_Prov_cont_discr_label_enc[:3])
