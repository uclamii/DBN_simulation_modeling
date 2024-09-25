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

    DATA_PATH_write = "./Data/split_discetized_datasets_using_Prov_discritizer/"

    # read dataset
    df = pd.read_csv(DATA_PATH_read.replace(".csv", "_excl_crit_applied.csv"))

    ##

    dataset_names = [
        "UCLA",
        "combined",
        "Prov",
    ]

    print("Discretizing " + " UCLA using Providence discritizer" + " .... ")

    #### Providence
    cond_UCLA = df["site_source_cat"] == 0
    df_UCLA = df[cond_UCLA].copy().reset_index(drop=True)

    ################## creating dataset of only continuous columns and the label
    df_UCLA_cont = pd.DataFrame(
        data=np.c_[
            df_UCLA[cont_cols],
            df_UCLA[target],
        ],
        columns=cont_cols + ["target"],
    )

    #########################################################
    ################### Loading Prov discritizer

    discritizer = loadObjects("Data/discritizers/" + dataset_names[2] + "_discritizer")

    #########################################################
    print("\n")
    print("Transforming the data ...... ")

    # transform data
    df_UCLA_cont_discr = discritizer.transform(df=df_UCLA_cont)
    df_UCLA_cont_discr_label_enc = discritizer.transform(df=df_UCLA_cont, mapped=False)

    #######################################################
    ######################## Discritized data

    ### Adding discretized data to original columns
    df_UCLA[cont_cols] = df_UCLA_cont_discr[cont_cols]

    ################### saving discretized data
    df_UCLA.to_csv(
        DATA_PATH_write
        + dataset_names[0]
        + "_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_using_"
        + dataset_names[2]
        + "_discritizer.csv",
        index=False,
    )

    print(df_UCLA[:3])

    #######################################################
    ######################## Label encoded data

    ### Adding discretized data to original columns
    df_UCLA[cont_cols] = df_UCLA_cont_discr_label_enc[cont_cols]

    # target
    # df_UCLA[target] = df_UCLA[target] + 1  # +1 octave format

    ############# label encoder for categorical data
    ############# only used for structure learning
    dfTrain_Prov = pd.read_csv(
        "Data/split_discetized_datasets/cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_Prov_train.csv"
    )
    for cat_col in tqdm(cat_cols):
        if cat_col not in ["patient_id", "site_source_cat"]:
            le = preprocessing.LabelEncoder()

            # train
            null_mask_train = dfTrain_Prov[cat_col].isnull()
            dfTrain_Prov.loc[null_mask_train, cat_col] = "NaN"
            categories = dfTrain_Prov[cat_col].dropna().unique().tolist()
            le.fit(categories)

            null_mask_df_UCLA = df_UCLA[cat_col].isnull()
            df_UCLA.loc[null_mask_df_UCLA, cat_col] = "NaN"
            # ruca starts at 1 no need to add 1
            if cat_col not in ["ruca_7_class", "ruca_4_class"]:
                df_UCLA[cat_col] = (
                    le.transform(df_UCLA[cat_col]) + 1  # +1 octave format
                )  # +1 octave format
            else:
                df_UCLA[cat_col] = le.transform(df_UCLA[cat_col])
            df_UCLA.loc[null_mask_df_UCLA, cat_col] = np.nan
        else:
            pass

    ################### saving discretized data
    df_UCLA.to_csv(
        DATA_PATH_write
        + dataset_names[0]
        + "_cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_using_"
        + dataset_names[2]
        + "_discritizer.csv",
        index=False,
    )

    print("Transformed data: ")
    print(df_UCLA_cont_discr_label_enc[:3])
