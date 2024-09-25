import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from tqdm import tqdm
import glob


def getReport(df, reportTitle, explorative=False, minimal=True):
    """Method to get profilign report for a dataset"""
    profile = ProfileReport(
        df,
        title=reportTitle,
        explorative=explorative,
        minimal=minimal,
        # pool_size=2,
    )
    return profile


def saveReportHTML(profile, pathFilename):
    """Method to save report as html file"""
    profile.to_file(pathFilename + ".html")
    return


if __name__ == "__main__":
    path = "../Data/"

    filesNames = [
        path + "cure_ckd_egfr_registry-7-12-2021.csv"
    ]  # glob.glob(path + "/*.csv")  # replace txt with csv for csv's

    for fileName, reprotTitle in tqdm(zip(filesNames, filesNames)):
        print(fileName)
        df = pd.read_csv(fileName, encoding="cp1252")  # , nrows=100)
        col_old = 0
        for col in range(50, df.shape[1] + 50, 50):
            print("Number of rows: ", len(df))
            profile = getReport(
                df.iloc[:, col_old:col],
                reportTitle=reprotTitle + "col_" + str(col),
                minimal=True,
            )
            saveReportHTML(
                profile, path + (fileName + "col_" + str(col)).replace(".csv", "")
            )
            col_old = col
