import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

if __name__ == "__main__":
    ## READ results

    ##########################################################

    # Selecting
    #  each years predictions to find the best model
    # for each site and filter

    num_epochs = 6
    metrics = ["AP", "AUC ROC"]
    results = []

    sites = ["UCLA", "PSJH", "Combined"]

    for site in sites:

        results_path = "./Data/genie_datasets/DBN_predictions/Results/"

        sites_models_results_filepaths = [
            model_path
            for model_path in glob.glob(results_path + "*")
            if site in model_path and "all_datasets_results" in model_path
        ]

        ##########################################################

        for models_results_filepath in sites_models_results_filepaths:
            model_name = models_results_filepath.replace(results_path, "").replace(
                "all_datasets_results.csv", ""
            )

            df_results = pd.read_csv(models_results_filepath)
            df_results = df_results[
                df_results["Dataset"].str.contains(
                    "_test_tested_on"
                )  # only on test set
            ].reset_index(drop=True)

            filters = (
                df_results["Race Ethnicity category"].unique().tolist()
            )  # all filters

            for fitler_cat in filters:
                for metric in metrics:

                    df_results_filter_group = df_results[
                        df_results["Race Ethnicity category"] == fitler_cat
                    ].reset_index(drop=True)

                    # grouping into a d x d df of time
                    pred_target_res = []

                    for pred_year in range(1, num_epochs + 1):
                        pred_year_res = []
                        for target_year in range(1, num_epochs + 1):
                            if pred_year == target_year:
                                pred_year_res.append(
                                    df_results_filter_group.loc[
                                        df_results_filter_group["Metric"] == metric,
                                        "Prediction Year "
                                        + str(pred_year)
                                        + ", target year "
                                        + str(target_year),
                                    ].values[0]
                                )
                            else:
                                pass
                        pred_target_res.append(pred_year_res)
                    results.append(
                        [model_name, fitler_cat, metric]
                        + list(np.ravel(pred_target_res))
                        + [np.nanmean((np.ravel(pred_target_res)))]
                    )
        # saving into a dataframe
        annual_cols = ["Year " + str(epoch) for epoch in range(1, num_epochs + 1)]
        cols = ["Model_site", "Filter", "Metric"] + annual_cols + ["Average metric"]
        df_models_results = pd.DataFrame(results, columns=cols)

        ##########################################################

        # for each filter and for each metric
        # count how many times a model had the max performance
        # also compute the average performance over the years

        for fitler_cat in filters:
            for metric in metrics:
                # over annual metrics
                # cond
                cond = (df_models_results["Metric"] == metric) & (
                    df_models_results["Filter"] == fitler_cat
                )
                max_values = df_models_results.loc[cond, annual_cols].max().values
                max_value_counts = np.zeros(
                    df_models_results.loc[cond, annual_cols].shape
                )
                for max_value in max_values:
                    # print((df_models_results.loc[df_models_results["Metric"]==metric,annual_cols] == max_value))
                    max_value_counts = max_value_counts + (
                        (df_models_results.loc[cond, annual_cols] == max_value).values
                        * 1
                    )
                #  print(max_value_counts)
                max_value_counts = np.sum(max_value_counts, axis=1)
                #   print(max_value_counts)
                df_models_results.loc[
                    cond, "Counts of years of max performance"
                ] = max_value_counts
                ##########
                # average metric
                avg_max_value = df_models_results.loc[cond, "Average metric"].max()
                avg_max_value_bin = (
                    df_models_results.loc[cond, "Average metric"] == avg_max_value
                ).values * 1

                df_models_results.loc[
                    cond, "Max Average metric flag"
                ] = avg_max_value_bin

        ###########################################################

        # focusing on the best models best on average performance
        # per metric and filter for each site
        # saving best models in a dataframe based on the average metric
        df_best_models = pd.DataFrame()
        for fitler_cat in filters:
            for metric in metrics:
                cond = (df_models_results["Metric"] == metric) & (
                    df_models_results["Filter"] == fitler_cat
                )
                # by average metric
                df_models_results_cond = df_models_results[cond].reset_index(drop=True)
                cond_avg_max = (
                    df_models_results_cond["Max Average metric flag"]
                    == df_models_results_cond["Max Average metric flag"].max()
                )
                df_best_models = df_best_models.append(
                    df_models_results_cond[cond_avg_max]
                )
        df_best_models = df_best_models.reset_index(drop=True)

        # saving best model csv file
        # df_best_models.to_csv(
        #     "./Data/genie_datasets/DBN_predictions/Results/best_models_per_filter_"
        #     + site
        #     + ".csv",
        #     index=False,
        # )

        ##########################################################

        # Selecting
        #  each years predictions to find the best model
        # for each site and filter

        num_epochs = 6
        metrics = ["AP", "AUC ROC"]
        results = []

        sites = ["UCLA", "PSJH", "Combined"]

        results_path = "./Data/genie_datasets/DBN_predictions/Results/"

        sites_models_results_filepaths = [
            model_path
            for model_path in glob.glob(results_path + "*")
            if site in model_path
        ]

        print(sites_models_results_filepaths)

        model_names_filter_groups = df_best_models[df_best_models["Metric"] == "AP"][
            ["Model_site", "Filter"]
        ].reset_index(drop=True)

        print(model_names_filter_groups)

        ##########################################################

        # assigning to another variables
        original_sites_models_results_filepaths = sites_models_results_filepaths

        # looping each model site, and filter combination

        for model_comb_indx in tqdm(range(model_names_filter_groups.shape[0])):
            filter_groups = [model_names_filter_groups.iloc[model_comb_indx, 1]]
            model_names = [model_names_filter_groups.iloc[model_comb_indx, 0]]

            model_name = model_names[0]

            print("##########################################################")

            print("For model " + model_name + " ... generating results ...")

            # focusing on "all" race results
            sites_models_results_filepaths = [
                model_path
                for model_path in original_sites_models_results_filepaths
                if model_name + "all" in model_path
            ]

            # looping for each model the results, and keeping only the results
            # on test data
            for models_results_filepath in sites_models_results_filepaths:
                model_name = models_results_filepath.replace(results_path, "").replace(
                    "all_datasets_results.csv", ""
                )

                df_results = pd.read_csv(models_results_filepath)
                df_results = df_results[
                    df_results["Dataset"].str.contains("_test_tested_on")
                ].reset_index(drop=True)

            df_results["Dataset"].unique()

            # not keeping the optimal threshold row
            metrics = [
                metric
                for metric in df_results["Metric"].unique()
                if "Optimal" not in metric
            ]

            print("Computing results for the following metrics ...")
            print(metrics)

            for filter_group in filter_groups:
                print("For cohort group: " + filter_group)

                for metric in metrics:

                    cond = (df_results["Race Ethnicity category"] == filter_group) & (
                        df_results["Dataset"].str.contains(model_name)
                    )

                    df_results_filter_group = df_results[cond].reset_index(drop=True)

                    # grouping into a d x d df of time
                    pred_target_res = []

                    for pred_year in range(1, num_epochs + 1):
                        pred_year_res = []
                        for target_year in range(1, num_epochs + 1):
                            if pred_year <= target_year:
                                values = df_results_filter_group.loc[
                                    df_results_filter_group["Metric"] == metric,
                                    "Prediction Year "
                                    + str(pred_year)
                                    + ", target year "
                                    + str(target_year),
                                ].reset_index(drop=True)[0]
                                pred_year_res.append(values)
                            else:
                                pred_year_res.append(np.nan)
                        pred_target_res.append(pred_year_res)

                    pred_cols = [
                        "Pred. year " + str(i) for i in range(1, num_epochs + 1)
                    ]
                    targ_cols = [
                        "Targ. year " + str(i) for i in range(1, num_epochs + 1)
                    ]

                    df_plot = pd.DataFrame(
                        data=pred_target_res, columns=targ_cols, index=pred_cols
                    )

                    # plots

                    # Create a mask
                    mask = np.triu(np.ones_like(df_plot, dtype=bool))

                    # set x-axis on top or bottom
                    plt.rcParams["xtick.bottom"] = plt.rcParams[
                        "xtick.labelbottom"
                    ] = False
                    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

                    # Create a custom divergin palette
                    # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
                    cmap = mpl.cm.viridis

                    plt.figure(figsize=(10, 10))
                    plt.title(metric + " over time")
                    sns.heatmap(
                        df_plot,
                        mask=~mask,  # use symbol to reverse mask
                        center=0,
                        annot=True,
                        fmt=".2f",
                        square=True,
                        cmap=cmap,
                    )

                    # labels rotation angle
                    plt.yticks(rotation=0)
                    plt.xticks(rotation=45)

                    path_to_write = (
                        results_path
                        + "plots/"
                        + site
                        + "/"
                        + model_name
                        + filter_group.replace(" ", "_")
                    )
                    print(path_to_write, " metric figure: ", metric)

                    os.system("mkdir -p " + path_to_write)
                    plt.savefig(
                        results_path
                        + "plots/"
                        + site
                        + "/"
                        + model_name
                        + filter_group.replace(" ", "_")
                        + "/"
                        + model_name
                        + "_"
                        + filter_group.replace(" ", "_")
                        + "_"
                        + metric.replace("/", "_")
                        + ".png",
                        pad_inches=1,
                    )
                    plt.close()
                    # plt.show()
        ##########################################################
