from MDLDiscretization import *
from CONSTANTS import *
from tqdm import tqdm
from pickleObjects import *
from sklearn import preprocessing

# Importing Library
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.inference import VariableElimination


def predict(self, data):
    """
    https://programtalk.com/python-examples/pgmpy.inference.VariableElimination/
    Predicts states of all the missing variables.
 
    Parameters
    ----------
    data : pandas DataFrame object
        A DataFrame object with column names same as the variables in the model.
 
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import BayesianModel
    >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> train_data = values[:800]
    >>> predict_data = values[800:]
    >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> model.fit(values)
    >>> predict_data = predict_data.copy()
    >>> predict_data.drop('E', axis=1, inplace=True)
    >>> y_pred = model.predict(predict_data)
    >>> y_pred
    array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
           1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
           1, 1, 1, 0, 0, 0, 1, 0])
    """

    if set(data.columns) == set(self.nodes()):
        raise ValueError("No variable missing in data. Nothing to predict")

    elif set(data.columns) - set(self.nodes()):
        raise ValueError("data has variables which are not in the model")

    missing_variables = set(self.nodes()) - set(data.columns)
    pred_values = defaultdict(list)

    model_inference = VariableElimination(self)
    for index, data_point in data.iterrows():
        states_dict = model_inference.map_query(
            variables=missing_variables, evidence=data_point.to_dict()
        )
        for k, v in states_dict.items():
            pred_values[k].append(v)
    return pd.DataFrame(pred_values, index=data.index)


cont_cols = old_CTN_ENTRY_COLS + old_TIME_ZERO_COLS + old_CTN_COLS

cat_cols = old_CAT_COLS  # + old_RACE_COLS

target = "egfr_reduction40_flag"


if __name__ == "__main__":

    from pgmpy.estimators import ExpectationMaximization as EM

    # Defining network structure

    alarm_model = BayesianNetwork(
        [
            ("Burglary", "Alarm"),
            ("Earthquake", "Alarm"),
            ("Alarm", "JohnCalls"),
            ("Alarm", "MaryCalls"),
        ]
    )

    # Define a model structure with latent variables
    model_latent = BayesianNetwork(ebunch=alarm_model.edges())

    # Dataset for latent model which doesn't have values for the latent variables
    samples_latent = samples.drop(model_latent.latents, axis=1)

    model_latent.fit(samples_latent, estimator=EM)

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
            data=np.c_[dfTrain[cont_cols], dfTrain[target],],
            columns=cont_cols + ["target"],
        )
        dfValid_cont = pd.DataFrame(
            data=np.c_[dfValid[cont_cols], dfValid[target],],
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
                dfTrain[cat_col] = (
                    le.transform(dfTrain[cat_col]) + 1  # +1 octave format
                )  # +1 octave format
                dfTrain.loc[null_mask_train, cat_col] = np.nan
                # valid
                null_mask_valid = dfValid[cat_col].isnull()
                dfValid.loc[null_mask_valid, cat_col] = "NaN"
                dfValid[cat_col] = (
                    le.transform(dfValid[cat_col]) + 1  # +1 octave format
                )  # +1 octave format
                dfValid.loc[null_mask_valid, cat_col] = np.nan
                # test
                null_mask_test = dfTest[cat_col].isnull()
                dfTest.loc[null_mask_test, cat_col] = "NaN"
                dfTest[cat_col] = le.transform(dfTest[cat_col]) + 1  # +1 octave format
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
