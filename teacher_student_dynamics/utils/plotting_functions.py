import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from teacher_student_dynamics import constants


def plot_all_scalars(df: pd.DataFrame, path: str, prefix: str):
    """Make a plot for every column in a given dataframe.

    Args:
        df: pandas dataframe containing data to plot.
        path: path to save plots.
        prefix: any prefix to plots (prefixing the column name).
    """
    for column in df.columns:
        fig = plt.figure()
        df[column].dropna().plot()
        plt.xlabel(constants.STEP)
        plt.ylabel(column)
        plt.title(prefix)
        fig.savefig(os.path.join(path, f"{prefix}_{column}.pdf"))
        plt.close()


def plot_all_common_scalars(
    dfs: List[pd.DataFrame],
    path: str,
    prefix: str,
    sub_prefixes: List[str],
    styles: List[Dict],
):
    """Make a plot for every column that exists in all dataframes provided.

    Args:
        dfs: list of pandas dataframes containing data to plot.
        path: path to save plots.
        prefix: any prefix to plot names (prefixing the column name).
        sub_prefixes: matches dimension of dfs; prefix to add to legend for each dataset in plot.
        styles: matches dimension of dfs; specification of formatting for each dataset in plot.
    """
    df_columns = [list(df.columns) for df in dfs]

    intersection_columns = set(df_columns[0])
    for df in df_columns[1:]:
        intersection_columns = set(intersection_columns) & set(df)

    for column in intersection_columns:
        fig = plt.figure()
        for df, sub_prefix, style in zip(dfs, sub_prefixes, styles):
            style[constants.LABEL] = sub_prefix
            df[column].dropna().plot(**style)
        plt.xlabel(constants.STEP)
        plt.ylabel(column)
        plt.title(prefix)
        fig.savefig(os.path.join(path, f"{prefix}_{column}.pdf"))
        plt.close()


def plot_all_common_scalar_diffs(
    dfs: List[pd.DataFrame],
    path: str,
    prefix: str,
):
    """Make a plot for every column that exists in all dataframes provided.

    Args:
        dfs: list of pandas dataframes containing data to plot.
        path: path to save plots.
        prefix: any prefix to plot names (prefixing the column name).
    """
    df_columns = [list(df.columns) for df in dfs]

    intersection_columns = set(df_columns[0])
    for df in df_columns[1:]:
        intersection_columns = set(intersection_columns) & set(df)

    for column in intersection_columns:
        fig = plt.figure()
        diffs = (dfs[0][column].dropna() - dfs[1][column].dropna()).dropna()
        zero_diffs = diffs.where(
            np.isclose(
                diffs,
                0,
                atol=1e-10,
                equal_nan=False,
            )
        )

        diffs.plot()
        zero_diffs.dropna().plot(style="o")

        # (dfs[0][column].dropna() - dfs[1][column].dropna()).plot()
        # (dfs[0][column].dropna() - dfs[1][column].dropna()).where(
        #     np.isclose(
        #         dfs[0][column].dropna(),
        #         dfs[1][column].dropna(),
        #         atol=1e-10,
        #         equal_nan=False,
        #     )
        # ).plot(style="o")

        plt.xlabel(constants.STEP)
        plt.ylabel(column)
        plt.title(prefix)
        fig.savefig(os.path.join(path, f"{prefix}_{column}.pdf"))
        plt.close()
