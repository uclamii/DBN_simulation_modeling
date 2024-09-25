from MDLDiscretization import *
from CONSTANTS import *
from tqdm import tqdm
from pickleObjects import *
from sklearn import preprocessing

reduction_cols = ["year" + str(i) + "_reduction" for i in range(1, 14)] + [
    "year" + str(i) + "_norace_reduction" for i in range(1, 14)
]

reductions_40_ge_cols = [col + "_40_ge" for col in reduction_cols]

cont_cols = old_CTN_ENTRY_COLS + old_TIME_ZERO_COLS + old_CTN_COLS

cat_cols = old_CAT_COLS + reductions_40_ge_cols  # + old_RACE_COLS

target = "egfr_reduction40_flag"


if __name__ == "__main__":
    DATA_PATH_read = "./Data/split_datasets/"
    DATA_PATH_write = "./Data/split_discetized_datasets/"
    DATA_PATH_write_label_enc = "./Data/split_label_encoded_datasets/"

    dataset_names = [
        "UCLA",
        "combined",
        "Prov",
    ]

    for dataset_name in tqdm(dataset_names):
        print("Discretizing " + dataset_name + " .... ")

        ###################### reading train, val, test of each dataset
        dfTrain = pd.read_csv(
            DATA_PATH_read
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_"
            + dataset_name
            + "_train.csv"
        )
        dfValid = pd.read_csv(
            DATA_PATH_read
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_"
            + dataset_name
            + "_valid.csv"
        )
        dfTest = pd.read_csv(
            DATA_PATH_read
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_"
            + dataset_name
            + "_test.csv"
        )

        ################## creating dataset of only continuous columns and the label
        dfTrain_cont = pd.DataFrame(
            data=np.c_[
                dfTrain[cont_cols],
                dfTrain[target],
            ],
            columns=cont_cols + ["target"],
        )
        dfValid_cont = pd.DataFrame(
            data=np.c_[
                dfValid[cont_cols],
                dfValid[target],
            ],
            columns=cont_cols + ["target"],
        )
        dfTest_cont = pd.DataFrame(
            data=np.c_[dfTest[cont_cols], dfTest[target]],
            columns=cont_cols + ["target"],
        )

        ###################### converting DFs to orange tables
        df_orangeTrain = df2Orange(dfTrain_cont, class_name="target")
        df_orangeValid = df2Orange(dfValid_cont, class_name="target")
        df_orangeTest = df2Orange(dfTest_cont, class_name="target")

        # print("Original data: ")
        # print(df_orangeTrain[:3])
        print("\n")
        print("Fitting data ...")
        print("\n")
        discritizer = MDLDescritizer()
        discritizer.fit(cont_data=df_orangeTrain)

        #########################################################
        ################### Saving discritizer

        dumpObjects(discritizer, "Data/discritizers/" + dataset_name + "_discritizer")

        #########################################################

        # print('List of discretizations: ')
        # print(discritizer.dicts)
        print("\n")
        print("Transforming the data ...... ")

        # train
        dfTrain_cont_discr = discritizer.transform(df=dfTrain_cont)
        dfTrain_cont_discr_label_enc = discritizer.transform(
            df=dfTrain_cont, mapped=False
        )

        # valid
        dfValid_cont_discr = discritizer.transform(df=dfValid_cont)
        dfValid_cont_discr_label_enc = discritizer.transform(
            df=dfValid_cont, mapped=False
        )

        # test
        dfTest_cont_discr = discritizer.transform(df=dfTest_cont)
        dfTest_cont_discr_label_enc = discritizer.transform(
            df=dfTest_cont, mapped=False
        )

        #######################################################
        ######################## Discritized data

        ### Adding discretized data to original columns
        dfTrain[cont_cols] = dfTrain_cont_discr[cont_cols]
        dfValid[cont_cols] = dfValid_cont_discr[cont_cols]
        dfTest[cont_cols] = dfTest_cont_discr[cont_cols]

        ################### saving discretized data
        dfTrain.to_csv(
            DATA_PATH_write
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_"
            + dataset_name
            + "_train.csv",
            index=False,
        )
        dfValid.to_csv(
            DATA_PATH_write
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_"
            + dataset_name
            + "_valid.csv",
            index=False,
        )
        dfTest.to_csv(
            DATA_PATH_write
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_"
            + dataset_name
            + "_test.csv",
            index=False,
        )

        print("Transformed data: ")
        print(dfTrain_cont_discr[:3])
        print(dfValid_cont_discr[:3])
        print(dfTest_cont_discr[:3])

        #######################################################
        ######################## Label encoded data

        ### Adding discretized data to original columns
        dfTrain[cont_cols] = dfTrain_cont_discr_label_enc[cont_cols]
        dfValid[cont_cols] = dfValid_cont_discr_label_enc[cont_cols]
        dfTest[cont_cols] = dfTest_cont_discr_label_enc[cont_cols]

        # target
        dfTrain[target] = dfTrain[target] + 1  # +1 octave format
        dfValid[target] = dfValid[target] + 1  # +1 octave format
        dfTest[target] = dfTest[target] + 1  # +1 octave format

        ############# label encoder for categorical data
        ############# only used for structure learning
        for cat_col in cat_cols:
            if cat_col not in ["patient_id"]:
                le = preprocessing.LabelEncoder()

                # train
                null_mask_train = dfTrain[cat_col].isnull()
                dfTrain.loc[null_mask_train, cat_col] = "NaN"
                categories = dfTrain[cat_col].dropna().unique().tolist()
                le.fit(categories)
                # ruca starts at 1 no need to add 1
                if cat_col not in ["ruca_7_class", "ruca_4_class"]:
                    dfTrain[cat_col] = (
                        le.transform(dfTrain[cat_col]) + 1  # +1 octave format
                    )  # +1 octave format
                else:
                    dfTrain[cat_col] = le.transform(dfTrain[cat_col])
                dfTrain.loc[null_mask_train, cat_col] = np.nan
                # valid
                null_mask_valid = dfValid[cat_col].isnull()
                dfValid.loc[null_mask_valid, cat_col] = "NaN"
                # ruca starts at 1 no need to add 1
                if cat_col not in ["ruca_7_class", "ruca_4_class"]:
                    dfValid[cat_col] = (
                        le.transform(dfValid[cat_col]) + 1  # +1 octave format
                    )  # +1 octave format
                else:
                    dfValid[cat_col] = le.transform(dfValid[cat_col])
                dfValid.loc[null_mask_valid, cat_col] = np.nan
                # test
                null_mask_test = dfTest[cat_col].isnull()
                dfTest.loc[null_mask_test, cat_col] = "NaN"
                # ruca starts at 1 no need to add 1
                if cat_col not in ["ruca_7_class", "ruca_4_class"]:
                    dfTest[cat_col] = (
                        le.transform(dfTest[cat_col]) + 1  # +1 octave format
                    )  # +1 octave format
                else:
                    dfTest[cat_col] = le.transform(dfTest[cat_col])
                dfTest.loc[null_mask_test, cat_col] = np.nan
            else:
                pass

        ################### saving discretized data
        dfTrain.to_csv(
            DATA_PATH_write_label_enc
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_"
            + dataset_name
            + "_train.csv",
            index=False,
        )
        dfValid.to_csv(
            DATA_PATH_write_label_enc
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_"
            + dataset_name
            + "_valid.csv",
            index=False,
        )
        dfTest.to_csv(
            DATA_PATH_write_label_enc
            + "cure_ckd_egfr_registry_preprocessed_project_preproc_data_discretized_label_enc_"
            + dataset_name
            + "_test.csv",
            index=False,
        )

        print("Transformed data: ")
        print(dfTrain_cont_discr_label_enc[:3])
        print(dfValid_cont_discr_label_enc[:3])
        print(dfTest_cont_discr_label_enc[:3])
