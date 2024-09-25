import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def tuple_string_to_numeric(s):
    """Method to convert string tuples to list of floats"""
    if pd.isnull(s):
        return np.nan
    else:
        # Remove parentheses and split by comma
        parts = s.strip('()').split(',')
        # Convert each part to float
        numeric_values = [float(part) for part in parts]
        return numeric_values

if __name__ == "__main__":
    ## READ results

    ##########################################################

    # Selecting
    #  each years predictions to find the best model
    # for each site and filter

    num_epochs = 6
    metrics = ["AP", "AUC ROC"]
    results = []

    # TODO: uncomment combined once struct is available
    sites = ["UCLA" , "PSJH", "Combined"]

    for site in sites:
        results_path = "./Data/genie_datasets/DBN_predictions/Results/CIs_results/balanced/"

        sites_models_results_filepaths = [
            model_path
            for model_path in glob.glob(results_path + "*")
            if site in model_path
            and "no_race" in model_path
            and "count" not in model_path
            and "all_datasets_results" in model_path
        ]

        print(sites_models_results_filepaths)

        for models_results_filepath in sites_models_results_filepaths:
            df_results = pd.read_csv(models_results_filepath)
            ##########################################################

            filter_groups = df_results["Race Ethnicity category"].unique()

            model_name = models_results_filepath.replace(results_path, "").replace(
                "all_datasets_results.csv", ""
            )

            # not keeping the optimal threshold row
            metrics = [
                metric
                for metric in df_results["Metric"].unique()
                if "Optimal" not in metric and "Prev" not in metric
            ]

            print("Computing results for the following metrics ...")
            print(metrics)

            for filter_group in filter_groups:
                print("For cohort group: " + filter_group)

                for metric in metrics:
                    if "CIs" in metric:
                        continue
                    cond = (
                        (df_results["Race Ethnicity category"] == filter_group)
                        & (df_results["Dataset"].str.contains(model_name))
                        & (df_results["Dataset"].str.contains("test_tested_on"))
                    )

                    df_results_filter_group = df_results[cond].reset_index(drop=True)

                    # grouping into a d x d df of time
                    pred_target_res, pred_target_res_CIs = [], []

                    for pred_year in range(1, num_epochs + 1):
                        pred_year_res, pred_year_res_CIs = [], []
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
                                values_CIs = df_results_filter_group.loc[
                                    df_results_filter_group["Metric"] == metric+"_CIs",
                                    "Prediction Year "
                                    + str(pred_year)
                                    + ", target year "
                                    + str(target_year),
                                ].reset_index(drop=True)[0]
                                pred_year_res_CIs.append(values_CIs)
                            else:
                                pred_year_res.append(np.nan)
                                pred_year_res_CIs.append(np.nan)
                        pred_target_res.append(pred_year_res)
                        pred_target_res_CIs.append(pred_year_res_CIs)

                    pred_cols = [
                        "Pred. year " + str(i) for i in range(1, num_epochs + 1)
                    ]
                    targ_cols = [
                        "Targ. year " + str(i) for i in range(1, num_epochs + 1)
                    ]

                    df_plot = pd.DataFrame(
                        data=pred_target_res, columns=targ_cols, index=pred_cols
                    )

                    df_plot_CIs = pd.DataFrame(
                        data=pred_target_res_CIs, columns=targ_cols, index=pred_cols
                    )

                    # prevalence vec
                    pred_targ_cols = [
                        "Prediction Year " + str(i) + ", target year " + str(i)
                        for i in range(1, num_epochs + 1)
                    ]
                    prevalences = df_results_filter_group.loc[
                        df_results_filter_group["Metric"] == "Outcome Prevalence",
                        pred_targ_cols,
                    ].values[0]

                    # plots

                    # Create a mask
                    mask = ~np.triu(np.ones_like(df_plot, dtype=bool))

                    mask = np.concatenate((mask, [[True] * num_epochs]), axis=0)

                    prev_series = pd.Series(
                        {col: val for col, val in zip(df_plot.columns, prevalences)},
                        name="Prev. ≥40% eGFR decl. - %",
                    )
                    # prev_series
                    df_plot = df_plot.append(prev_series)

                    # font
                    mpl.rcParams.update({"font.size": 14})

                    # set x-axis on top or bottom
                    plt.rcParams["xtick.bottom"] = plt.rcParams[
                        "xtick.labelbottom"
                    ] = False
                    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

                    # convert strings to floats
                    df_plot = df_plot.apply(pd.to_numeric, errors='coerce')
                    df_plot_CIs = df_plot_CIs.applymap(tuple_string_to_numeric)


                    # Create a custom divergin palette
                    # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
                    cmap = "gray" # mpl.cm.viridis

                    plt.figure(figsize=(10, 10))
                    plt.title(metric + " over time")
                    ax = sns.heatmap(
                        df_plot,
                        mask=mask,  # use symbol to reverse mask
                        center=0,
                        annot=True,
                        fmt=".2f",
                        square=True,
                        cmap=cmap,
                    )


                    # Manually annotate each cell with additional information
                    CIs_vals = [val for val in df_plot_CIs.values.ravel() if isinstance(val,list)]
                    CIs_vals = [[round(num, 2) for num in sublist] for sublist in CIs_vals]
                    for text,value_CIs in zip(ax.texts,CIs_vals):
                        original_annotation = text.get_text()  # Get original annotation
                        additional_info = f'\n{value_CIs}'  # New line of text
                        text.set_text(original_annotation + additional_info)
                        text.set_fontsize(10)  # Adjust the font size as needed
                        text.set_color('black')  # Set text color to black


                    # # labels rotation angle
                    plt.yticks(rotation=0)
                    plt.xticks(rotation=45)

                    # # add annotations with the values of the data
                    for i in [num_epochs]:
                        for j in range(num_epochs):
                            text = "{:.2f}".format(50) # balanced samples so hardcoded to 50-50 #  df_plot.iloc[i, j])
                            plt.text(
                                j + 0.5,
                                i + 0.5,
                                text,
                                ha="center",
                                va="center",
                                color="black",
                            )

                    plt.tight_layout()

                    path_to_write = (
                        results_path
                        + "plots/CIs/balanced/"
                        + site
                        + "/"
                        + model_name
                        + filter_group.replace(" ", "_")
                    )
                    print(path_to_write, " metric figure: ", metric)

                    os.system("mkdir -p " + path_to_write)
                    plt.savefig(
                        results_path
                        + "plots/CIs/balanced/"
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
                        dpi=300,
                    )
                    plt.close()
                    plt.show()
        ##########################################################
