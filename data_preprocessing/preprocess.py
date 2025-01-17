import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Union, Dict, Any, Optional
import warnings
from CONSTANTS import *

"""
Module includes the following main functionality:
- writing preprocessed df to file
- loading features + targets from file
- processing raw data
- generating labels
- one-hot encoding features
"""


#######################
#     Main Driver     #
#######################
def preprocess_data(data_path: str) -> None:
    """The whole preprocessing pipeline.

    Writes df to file at the end.
    """
    df = process_raw_data(data_path)
    df = construct_targets(df)
    df = one_hot_enc(df)

    #### resetting categorical vars to in or categories
    #### for db purposes
    for col in CAT_COLS:
        if isinstance(df[col][0], str):
            pass
        else:
            df.loc[df[col].notnull(), col] = df.loc[df[col].notnull(), col].apply(int)

    #### rename columns ####
    df.rename(columns=columnsMapping, inplace=True)
    #### Write to file ####
    # Feather pickles pandas DFs so that they load much faster
    # Might take more storage space though
    df.to_feather(data_path.replace(".csv", "") + "_project_preproc_data.feather")
    df.to_csv(data_path.replace(".csv", "") + "_project_preproc_data.csv", index=False)


def load_features_and_labels(
    data_path: str,
    target: str = DEFAULT_TARGET,
    subgroup: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns X and y for use with models.

    - If it cannot find a file with the already preprocessed data,
    it will run the preprocess_data pipeline (create the file).
    - It can optionally take a target as a string.
    - It can optionally take a subgroup of the whole dataset to filter for.
        - For example, if you want to filter for CKD and UCLA.
        - Refer to documentation for filter_subgroup.
    """
    try:
        df = pd.read_feather(data_path + "preproc_data.feather")
    except IOError:
        print("Preprocessed file does not exist! Creating...")
        preprocess_data(data_path)
        df = pd.read_feather(data_path + "preproc_data.feather")

    # Filter for subgroup if requested
    if subgroup:
        df = filter_subgroup(df, subgroup)

    # Could also be: ... + CTN_ENTRY_COLS + TIME_ZERO_COLS + RACE_COLS
    # or ... + CTN_ENTRY_COLS + TIME_ZERO_COLS
    features = (
        [col for col in CAT_COLS if "race" not in col] + CTN_ENTRY_COLS + RACE_COLS
    )
    return (df[features], df[target])


def filter_subgroup(
    df: pd.DataFrame, subgroup: Tuple[Dict[str, Any], Dict[str, Any]]
) -> pd.DataFrame:
    """Filters the features/label for the given subgroup.

    Args:
        - subgroup: tuple of some sort of combo of cohort and site source
            - Each stratification is represented as a dictionary
                - it will have a name, and an encoding (unless you want 'all')
                - excluding encoding means don't filter on this column
    """
    cohort, site_source = subgroup
    cohort_filter = df["cohort_cat"] == cohort["encoding"]
    # "True" will not filter on that column (if it doesn't have encoding)
    site_source_filter = (
        df["site_source_cat"] == site_source["encoding"]
        if "encoding" in site_source
        else True
    )
    # Combine filters
    subgroup_filter = cohort_filter & site_source_filter
    return df[subgroup_filter]


#######################
#   Data Processing   #
#######################
def process_raw_data(data_path: str) -> pd.DataFrame:
    """Processes raw csv data.

    - Calculate percentage change in egfr (2 year chunks over the 10 years)
    - Discretize/bin percentage change for labels
    - Sanitize numerical values that should be NaN
    - Trims down the columns returned to categorical, continuous columns,
        and percentage of change in egfr (discretized)
    """
    # Load raw data/CSV
    # TO:DO: change file path, define path in main, for developemnt only read first 1000 rows
    df = pd.read_csv(data_path)  # , nrows=1000000)

    assert "egfr_entry_period_mean" in df.columns

    # No need to do decline for no race as change is the same in eGFR
    # Grab columns with egfr data
    eGFR_cols = ["egfr_year" + str(i) + "_mean" for i in range(1, 14)]

    #### Calculate percentage change in egfr ###
    perc_delta_eGFR = []
    for ind in range(len(eGFR_cols) - 2):
        # create new column to calculate difference
        col_name = "diff_eGFR" + str(ind)
        perc_delta_eGFR.append(col_name)
        # formula: decrease = (egfr[i+2] - egfr[i]) / egfr[i]
        # Note: decrease is calculated over the range of 2  years
        df[col_name] = (df[eGFR_cols[ind + 2]] - df[eGFR_cols[ind]]) / df[
            eGFR_cols[ind]
        ]

    #### Discretize the change in egfr ####
    perc_delta_eGFR_discr = []

    # create bins/labels: [<-100%, -100, -90, ... 90%, 100%, > 100%]
    bins = np.arange(-1, 1.1, 0.1)
    labels = (
        ["<-100%"]
        + [str(int(np.ceil(i * 100))) + "%" for i in bins if int(np.ceil(i * 100)) != 0]
        + [">100%"]
    )
    # update bins for outer range (<-100%, >100%)
    bins = [-np.inf] + list(bins) + [np.inf]

    # Add discretized change into the df
    for col in perc_delta_eGFR:
        col_name = col + "_discr"
        perc_delta_eGFR_discr.append(col_name)
        df[col_name] = pd.cut(df[col], bins=bins, labels=labels)
        df[col_name] = (
            df[col_name].cat.add_categories(["No_data"]).fillna("No_data")
        )  # replace missing values

    #### Fix Missing Value Representation ####
    # Some missing values are represented as numerical values
    # df["study_entry_a1c"].replace(to_replace=-99.99, value=np.nan, inplace=True)
    # df["study_entry_sbp"].replace(to_replace=-999.99, value=np.nan, inplace=True)
    # df["time_zero_hba1c_mean"].replace(to_replace=-99.99, value=np.nan, inplace=True)
    # df["demo_sex"].replace(to_replace=-9, value=np.nan, inplace=True)
    # df["demo_rural_cat"].replace(to_replace=-9, value=np.nan, inplace=True)

    # Make the cat columns 0/1 instead of 1/2
    # Keep the current ordering, just shift to 0/1
    # cat_shifter = {1: 0, 2: 1}
    # cat_columns_to_shift = ['cohort_cat', 'demo_sex', 'site_source_cat']

    # df[cat_columns_to_shift] = df[cat_columns_to_shift].replace(cat_shifter)

    keep_cols = (
        CAT_COLS + TIME_ZERO_COLS + CTN_ENTRY_COLS + CTN_COLS + perc_delta_eGFR_discr
    )
    return df[keep_cols]


#########################
#  Target Construction  #
#########################
def construct_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Creates possible targets to use in a model.

    - %decline (regression)
    - 40% decline (binary)
    - multiclass (10 groups)
    - multiclass (20 groups)
    """
    df["year_of_ge_40_decl_base2any"] = get_year_of_first_decline_from_baseline(
        df, perc=0.4
    )
    df["decl_base2any_ge_40_bin"] = binarize_year_of_decl_base2any(
        df["year_of_ge_40_decl_base2any"]
    )
    ### ignoring for now not used in current projects
    # df["ten_groups"] = get_ten_groups(df)
    # df["twenty_groups"] = get_twenty_groups(df)

    return df


def get_percentage_change(
    df: pd.DataFrame,
    from_year: Union[str, int] = "base",  # Can either be 'base' or int of year
    to_year: Union[str, int] = "last",  # Can either be 'last' or int of year
) -> pd.Series:
    """Returns the percent decline for each patient.

    Helper method to generating different kinds of labels.
    """
    if from_year == "base":
        # Right now (with this dataset) no one is missing time_zero.
        # Technically baseline = time_zero OR the first existing egfr reading
        assert "egfr_entry_period_mean" in df.columns
        start = df["egfr_entry_period_mean"]
    else:  # from_year must be int within range (assume sane input)
        # TODO: deal with if the year asked for DNE
        start = df["egfr_year" + str(from_year) + "_mean"]

    if to_year == "last":
        eGFR_cols = [
            i
            for i in df.columns
            if "egfr_year" in i and "mean" in i and "zero" not in i
        ]
        end = df[eGFR_cols].ffill(axis=1).iloc[:, -1]
    else:  # to_year must be int within range (assume sane input)
        end = df["egfr_year" + str(to_year) + "_mean"]

    return (end - start) / start


def binarize_year_of_decl_base2any(decline_base2any: pd.Series) -> pd.Series:
    """Binarizes the results of the decline from baseline to any year.

    Before 0 means no decline, otherwise it was the year we first
    see %perc decline. Now the years are just "on" flags.
    """
    bin_label = decline_base2any.copy()
    # Replace all non-zero values (year in which perc% decline first occured)
    bin_label[bin_label != 0] = 1
    return bin_label


def get_year_of_first_decline_from_baseline(
    df: pd.DataFrame, perc: float = 0.4
) -> pd.Series:
    """Returns categorical label according to Dennis' definition of perc% decline.

    Note that baseline is the time-zero or the first existing reading.

    Args:
        - perc: (-inf, 1] .4 corresponds to a >=40% drop in egfr.

    Returns:
        Label = in which year did we first see >=perc% decline? (categorical)
            - Implicitly asks:
                At any timepoint after baseline, is there a >=perc% decline?
            - 0 means that >=%perc decline never occured.
    """
    # get percent change from baseline to year_i for all 10 years
    # and then combine the labels as we go through each year
    per_year_labels = []
    for i in range(1, 14):
        # Invert to make declines positive values
        declines = -1 * get_percentage_change(df, "base", i)

        # answers: did >=perc% decline occur at year_i?
        # if >=perc% decline occurs again, we will keep the first
        label_base2year_i = declines.apply(lambda x: i if x >= perc else 0)
        per_year_labels.append(label_base2year_i)

    # Replace 0 values with nan as they indicate no decline occurred,
    # then backfill to get year of first decline in first column.
    # Access first column and swap nan back to 0.
    label = (
        pd.concat(per_year_labels, axis=1)
        .replace(0, np.nan)
        .bfill(axis=1)
        .iloc[:, 0]
        .fillna(0)
    )
    return label


#### DEPRECATED ####
# We don't want to look at these for official analysis anymore
# We can leave these in for personal experimentation.
def get_decline_from_baseline_to_last(
    df: pd.DataFrame,
    geq: bool = False,  # Greater than or equal to flag
    perc: float = 0.4,
) -> pd.Series:
    """Returns Label: if a decline of some percentage occured.

    Args:
    - perc: (percentage) has a .1/10% slack because it's actually a bin.
        For example: perc = .4 is actually [.3, .4]
    Return:
    - label: Series[int] (of either 0 or 1). 1 if the percentage change in egfr
        is really in that bin defined by perc), else 0.
    """
    warnings.warn(
        "This method is not used for official analysis anymore.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Invert to make declines positive values
    declines = -1 * get_percentage_change(df)

    # Remember the percentages are bins, so 40% is actually [30%, 40%]
    label = (
        declines.apply(lambda x: 1 if x >= perc - 0.1 and x <= perc else 0)
        if not geq
        else declines.apply(lambda x: 1 if x >= perc else 0)
    )
    return label


def get_ten_groups(df: pd.DataFrame) -> pd.Series:
    """Returns 10 groups of declines. (Really is 11)

    Groups: [[0, -10%], [-10%, -20%], ..., [-90%, -100%], [0, inf]]
    """
    # It's really 11 buckets, with one for no change or improvement
    declines = get_percentage_change(df)

    buckets = np.concatenate((np.linspace(-1, 0, 11), [np.inf]))
    return pd.cut(declines, buckets)


def get_twenty_groups(df: pd.DataFrame) -> pd.Series:
    """Returns 20 groupings of declines. (Really is 21)

    Similar to before, but we include a positive effect direction too now.
    Groups: [ [0, 10%], [10%, 20%], ..., [90%, 100%],
            [0, -10%], [-10%, -20%], ..., [-90%, -100%],
            [100%, inf]
            ]
    Note that we don't include [-100%, -inf] because it's not possible
    based on our definition of percent chage: (egfr2 - egfr1)/egfr1 =
    egfr2/egfr1 - 1. For this expression to be less than -1 (or -100%)
    egfr2/egfr has to be negative. As long as egfr is not negative,
    this is not possible.
    """
    # It's really 21 groups, with one for >100% change
    declines = get_percentage_change(df)

    buckets = np.concatenate((np.linspace(-1, 1, 21), [np.inf]))
    return pd.cut(declines, buckets)


"""
TODO: implement this, following the old notebook (#2) if we need it.
If we don't need this anymore, we can get rid of this method.
If we need this method, after implementing it as following the
    old notebook, then we want to add it as a potential target.
    Add it to the construct_targets method.
"""


def get_2yr_decline_binarized_40(df: pd.DataFrame) -> None:
    raise NotImplementedError


########################
#   One Hot Encoding   #
########################
def one_hot_enc(df: pd.DataFrame) -> pd.DataFrame:
    """One hot encodes some of the columns.

    Columns:
        - race
        - 10_groups
        - 20_groups
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(df["race_ethnicity_cat"].values.reshape(-1, 1))

    #### RACE ####
    # one_hot_races = enc.transform(
    #     df["race_ethnicity_cat"].values.reshape(-1, 1)
    # ).toarray()
    # df[RACE_COLS] = pd.DataFrame(data=one_hot_races, columns=RACE_COLS)

    #### 10 Group ####
    # We add prefixes to differentiate between the 10s and 20s
    # There is overlap in column names otherwise (which creates problems)
    # df = pd.concat([df, pd.get_dummies(df["ten_groups"], prefix="10")], axis="columns")
    # df = df.drop("ten_groups", axis="columns")

    # #### 20 Group ####
    # df = pd.concat(
    #     [df, pd.get_dummies(df["twenty_groups"], prefix="20")], axis="columns"
    # )
    # df = df.drop("twenty_groups", axis="columns")

    # The dummies create column names that aren't strings
    # This creates problems for feather, so we will convert here
    df.columns = df.columns.astype(str)

    return df


if __name__ == "__main__":
    DATA_PATH = "./Data/" + "cure_ckd_egfr_registry_preprocessed.csv"
    preprocess_data(DATA_PATH)
