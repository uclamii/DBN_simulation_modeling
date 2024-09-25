from CONSTANTS import *
from tqdm import tqdm
from pickleObjects import *
import glob
import pandas as pd
import numpy as np


if __name__ == "__main__":
    DATA_PATH_read_disc_data = "./Data/split_discetized_datasets/"
    DATA_PATH_read_disc_data_by_Prov = (
        "./Data/split_discetized_datasets_using_Prov_discritizer/"
    )
    DATA_PATH_read_disc_data_by_UCLA = (
        "./Data/split_discetized_datasets_using_UCLA_discritizer/"
    )

    filenames_disc = glob.glob(DATA_PATH_read_disc_data + "*.csv")
    filenames_disc_Prov = glob.glob(DATA_PATH_read_disc_data_by_Prov + "*.csv")
    filenames_disc_UCLA = glob.glob(DATA_PATH_read_disc_data_by_UCLA + "*.csv")

    for filenames_list in tqdm(
        [
            filenames_disc,
            filenames_disc_Prov,
            filenames_disc_UCLA,
        ]
    ):
        for filename in tqdm(filenames_list):
            df = pd.read_csv(filename, low_memory=False)
            ### preprocessing some values
            for col in df.columns:
                df[col] = np.where(
                    pd.isnull(df[col]), df[col], "S_" + df[col].astype(str)
                )
                df[col] = df[col].str.replace(" - ", "___")
                df[col] = df[col].str.replace(".", "_")
                df[col] = df[col].str.replace("≥ ", "le_")
                df[col] = df[col].str.replace("< ", "s_")
                df[col] = df[col].str.replace("-", "minus_")

            df.to_csv(filename.replace("/Data/", "/Data/genie_datasets/"), index=False)
