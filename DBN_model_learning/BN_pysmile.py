import pysmile
import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
from joblib import Parallel, delayed

import pysmile_license

from tqdm import tqdm

tqdm.pandas()


# Tutorial1 creates a simple network with three nodes,

# then writes its content as XDSL file to disk.


def write_pandas_df(df, temp_filename="temp.txt"):
    """Method to temporary save a pandas dataframe."""
    df.to_csv(temp_filename, index=False)
    return


def remove_temp_file(temp_filename="temp.txt"):
    """Method to remove a temporary file."""
    os.remove(temp_filename)
    return


def get_circle_coordinates(
    num_node_points,
    radius=400,
    x_pos_offset=160,
    y_pos_offset=40,
):
    ### making a circle
    angle = np.linspace(0, 2 * np.pi, num_node_points)

    x_pos = radius * np.cos(angle) + x_pos_offset
    y_pos = radius * np.sin(angle) + y_pos_offset
    return x_pos, y_pos


class CURE_CKD_BayesianNetwork:
    def __init__(
        self,
    ):
        self.net = None

    def save_BN(self, filename="CURE_CKD_BN.xdsl"):
        """Method to save BN as an xdsl file."""
        ########## Saving network
        self.net.write_file(filename)

        print("CURE_CKD NETWORK complete: Network written to " + filename)
        return

    def load_BN(self, filename="CURE_CKD_BN.xdsl"):
        """Method to load a BN."""

        net = pysmile.Network()
        net.read_file(filename)

        self.net = net
        print("Model " + filename + " loading complete.")
        return

    def create_BN(
        self,
        df_intra_static_time_zero,
        df_intra_time_zero_year1,
        df_intra_struct_year1,
        df_inter,
        df,
        BN_static_t0=False,
        num_epochs=4,
    ):
        """Method to create a BN structure."""

        net = pysmile.Network()

        ###### static and time zero intra struct
        net = self.create_struct(
            net,
            df_intra_static_time_zero,
            df,
            radius=800,
            x_pos_offset=160,
            y_pos_offset=40,
        )

        # only create up to this point
        if BN_static_t0:
            pass
        else:
            ###### year1-4 intra structs
            net = self.create_temporal_intra_structs(
                df_intra_struct_year1,
                net,
                df,
                num_epochs=num_epochs,
                radius=400,
                x_pos_offset=160,
                y_pos_offset=40,
            )

            ##### year1-4 inter structs
            net = self.create_temporal_inter_structs(
                df_inter,
                net,
                num_epochs=num_epochs,
            )

            ###### time_zero to year 1 struct edges

            net = self.create_inter_struct_edges(
                net,
                df_intra_time_zero_year1,
            )

            # net = self.create_struct(
            #     net,
            #     df_struct,
            #     df,
            #     radius=400,
            #     x_pos_offset=160,
            #     y_pos_offset=40,
            # )

        self.net = net
        return

    def create_temporal_inter_structs(
        self,
        df_inter,
        net,
        num_epochs=4,
    ):
        dfs_inter_year_1_4 = []
        for num in range(
            num_epochs - 2
        ):  # -2 as its temporal columns to add up max to year num_epochs in label
            cols = df_inter.columns
            indeces = df_inter.index
            df_inter_copy = df_inter.copy()
            new_cols = [
                col[:4] + str(int(col[4]) + num) + col[5:] for col in list(cols)
            ]
            new_indeces = [
                col[:4] + str(int(col[4]) + num) + col[5:] for col in list(indeces)
            ]
            df_inter_copy.columns = new_cols
            df_inter_copy.index = new_indeces
            dfs_inter_year_1_4.append(df_inter_copy)

        # edges
        for i in range(num_epochs - 2):
            # index reduced by 2 as we are looping through inter structs windows eg.1-2, 2-3,3-4, ..

            net = self.create_inter_struct_edges(
                net,
                dfs_inter_year_1_4[i],
            )

        return net

    def create_temporal_intra_structs(
        self,
        df_intra_struct_year1,
        net,
        df,
        num_epochs=4,
        radius=400,
        x_pos_offset=160,
        y_pos_offset=40,
    ):
        """Method that creates the intra structures of each epoch for a Markov Blanket"""
        dfs_intra_year_1_4 = []
        # has to be reduced by 1 because year columns indices start at 1, so outcome goes outside of index
        for num in range(num_epochs - 1):
            cols = df_intra_struct_year1.columns
            indeces = df_intra_struct_year1.index
            df_intra_struct_year1_copy = df_intra_struct_year1.copy()
            new_cols = [
                col[:4] + str(int(col[4]) + num) + col[5:] for col in list(cols)
            ]
            new_indeces = [
                col[:4] + str(int(col[4]) + num) + col[5:] for col in list(indeces)
            ]
            df_intra_struct_year1_copy.columns = new_cols
            df_intra_struct_year1_copy.index = new_indeces
            dfs_intra_year_1_4.append(df_intra_struct_year1_copy)

        # edges
        for i in range(num_epochs - 1):
            net = self.create_struct(
                net,
                dfs_intra_year_1_4[i],
                df,
                radius=radius,
                x_pos_offset=x_pos_offset * (i + 1) * 10,
                y_pos_offset=y_pos_offset,
            )

        return net

    def create_inter_struct_edges(
        self,
        net,
        df_struct,
    ):
        """Method to create the edges of an inter structure of a Markov Blanket"""
        # print(net.get_all_node_ids())

        for indx in df_struct.index:
            for col in df_struct.columns:
                if df_struct.loc[indx, col] == 1:
                    net.add_arc(indx, col)
                else:
                    pass
        return net

    def create_struct(
        self,
        net,
        df_struct,
        df,
        radius=400,
        x_pos_offset=160,
        y_pos_offset=40,
    ):
        """Method to create a structure of a Markov Blanket"""

        nodes = []

        x_pos, y_pos = get_circle_coordinates(
            len(df_struct.columns),
            radius=radius,
            x_pos_offset=x_pos_offset,
            y_pos_offset=y_pos_offset,
        )

        for i, col in enumerate(df_struct.columns):
            outcomes = [
                str(outcome) if ~isinstance(outcome, str) else outcome
                for outcome in sorted(df[col].dropna().unique().tolist())
            ]

            nodes.append(
                self.create_cpt_node(
                    net, col, col, outcomes, int(x_pos[i]), int(y_pos[i])
                )
            )

        for indx in df_struct.index:
            for col in df_struct.columns:
                if df_struct.loc[indx, col] == 1:
                    net.add_arc(indx, col)
                else:
                    pass
        return net

    def create_cpt_node(self, net, id, name, outcomes, x_pos, y_pos):
        handle = net.add_node(pysmile.NodeType.CPT, id)

        net.set_node_name(handle, name)

        net.set_node_position(handle, x_pos, y_pos, 85, 55)

        initial_outcome_count = net.get_outcome_count(handle)

        for i in range(0, initial_outcome_count):
            net.set_outcome_id(handle, i, outcomes[i])
            # net.set_outcome_id(handle, outcomes[i], outcomes[i])

        for i in range(initial_outcome_count, len(outcomes)):
            net.add_outcome(handle, outcomes[i])

        return handle

    def learnBN_structure(
        self, df_train, train_ds_filepath="temp.txt", method="Bayesian search"
    ):
        """Method to learn beayesian network structure."""

        # saving df to temporary file
        write_pandas_df(df=df_train)

        # reading as a genie file object
        ds = pysmile.learning.DataSet()
        ds.read_file(train_ds_filepath)

        # removing temporary file
        remove_temp_file()

        if method == "Bayesian search":
            baySearch = pysmile.learning.BayesianSearch()
            baySearch.set_iteration_count(10)
            baySearch.set_max_parents(2)
            self.net = baySearch.learn(ds)
            return
        elif method == "TAN":
            target = "year1_reduction_40_ge"
            tan = pysmile.learning.TAN()
            tan.set_class_variable_id(target)
            self.net = tan.learn(ds)
            return

        elif method == "ABN":
            target = "year1_reduction_40_ge"
            abn = pysmile.learning.NaiveBayes()
            abn.set_class_variable_id(target)
            # abn.set_max_parents(2)
            self.net = abn.learn(ds)
            return

    def trainBN(self, df_train, train_ds_filepath="temp.txt", model_name="test_model"):
        """Method to train a Bayesian network from a dataset."""
        # saving df to temporary file
        write_pandas_df(
            df=df_train,
            temp_filename=train_ds_filepath.replace(".txt", "_" + model_name + ".txt"),
        )

        # reading as a genie file object
        ds = pysmile.learning.DataSet()
        ds.read_file(train_ds_filepath.replace(".txt", "_" + model_name + ".txt"))

        # removing temporary file
        remove_temp_file(
            temp_filename=train_ds_filepath.replace(".txt", "_" + model_name + ".txt")
        )

        # clear all evidence
        self.net.clear_all_evidence()

        # load network and data here
        matching = ds.match_network(self.net)
        em = pysmile.learning.EM()

        # blog suggests this is default setting of Genie
        # https://support.bayesfusion.com/forum/viewtopic.php?p=3640&hilit=setEqSampleSize#p3640
        em.set_eq_sample_size(0)

        em.set_seed(0)  # fixing random seed
        em.set_randomize_parameters(False)
        em.set_uniformize_parameters(True)
        em.set_relevance(True)
        em.learn(ds, self.net, matching)

        print("EM log-likelihood: " + str(em.get_last_score()))

        # clear all evidence
        self.net.clear_all_evidence()

        # update the model beliefs
        self.net.update_beliefs()

        return

    def testBN(
        self,
        df_test,
        cols_to_test,
        target,
        target_output_name="predictions",
        ignore_warnings=True,
    ):
        """Method to set evidenece and test a dataframe based on an outcome node."""
        df_test[target_output_name] = df_test.progress_apply(
            lambda row: self.testRow(
                row,
                cols_to_test,
                target,
                ignore_warnings,
            ),
            axis=1,
        )

        return df_test[target_output_name]

    def parallelTestBN(
        self,
        df_test,
        cols_to_test,
        target,
        target_output_name="predictions",
        nb_workers=4,
        ignore_warnings=True,
    ):
        """Method to set evidenece and test a dataframe based on an outcome node.
        Method does not work model cannot be pickled.
        """
        dfs = np.array_split(df_test[:100], nb_workers)
        # set_loky_pickler(dill_serializer)

        results = Parallel(n_jobs=2, prefer="threads", backend="loky")(
            (
                delayed(self.testBN)(
                    df,
                    cols_to_test,
                    target,
                )
            )
            for df in tqdm(dfs)
        )

        df_test[target_output_name] = pd.concat(results, axis=0)

        return df_test[target_output_name]

        # predictions = []

        # for i in range(len(df_test)):
        #     row = df_test.loc[i, :]
        #     predictions.append(
        #         self.testRow(
        #             row,
        #             cols_to_test,
        #             target,
        #         ),
        #     )
        # return predictions

    def testRow(
        self,
        row,
        cols_to_test,
        target,
        ignore_warnings,
    ):
        """Method to set evidence of a pandas row  and test it."""

        # clear any initial evidence on the network
        self.net.clear_all_evidence()

        for col in cols_to_test:
            if pd.isnull(row[col]):
                pass
            else:
                # set all the evidence
                try:
                    self.net.set_evidence(col, row[col])
                except Exception as e:
                    if ignore_warnings:
                        pass
                    else:
                        print("An exception occurred")
                        print(e)
                        print(
                            "If Error -2, then its impossible to set the following evidence."
                        )
                        print(col, row[col])
                        print("Skipping evidence ... \n")
                        raise Exception("Breaking ... investigate error!")

        # Updating the network:
        self.net.update_beliefs()

        beliefs = self.net.get_node_value(target)
        # print(self.net.get_outcome_ids(target))
        # for i in range(0, len(beliefs)):
        #     print(self.net.get_outcome_id(target, i) + "=" + str(beliefs[i]))

        return beliefs[1]


def load_test_BN(
    model_name,
    df_test,
    cols_to_test,
    target,
    target_output_name="predictions",
    ignore_warnings=True,
):
    model = CURE_CKD_BayesianNetwork()

    model.load_BN(
        filename="DBN_model_learning/models/DBNs/" + model_name + "_CURE_CKD_DBN.xdsl"
    )

    predictions = model.testBN(
        df_test,
        cols_to_test,
        target,
        target_output_name=target_output_name,
        ignore_warnings=ignore_warnings,
    )

    return predictions


def parallelTestBN(
    model_name,
    df_test,
    cols_to_test,
    target,
    target_output_name="predictions",
    nb_workers=4,
    ignore_warnings=True,
):
    """Method to set evidenece and test a dataframe based on an outcome node.
    Method does not work model cannot be pickled.
    """
    dfs = np.array_split(df_test, nb_workers)

    results = Parallel(n_jobs=nb_workers)(
        (
            delayed(load_test_BN)(
                model_name,
                df,
                cols_to_test,
                target,
                target_output_name=target_output_name,
                ignore_warnings=ignore_warnings,
            )
        )
        for df in tqdm(dfs)
    )

    df_test[target_output_name] = pd.concat(results, axis=0)

    return df_test[target_output_name]


if __name__ == "__main__":
    CURE_CKD_BayesianNetwork()
