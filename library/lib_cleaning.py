"""
Contains various helper functions to clean and modify the data
"""
import pandas as pd


def fill_null(df, freq, test_col,well_col='NodeID', time_col='Date'):
    """
    Fills in Null Values for a df where data was not present for a specific frequency.
    :param df: DataFrame
    :param freq: The freq where the data is checked and Null values filled up
    :param test_col: The column which needs to be tested, not all columns will be affected tho
    :return: Filled up DataFrame
    """
    df.set_index(time_col, inplace=True)
    df_gp = df.groupby(well_col).resample(freq).max()  # Gp by well_col and resample it with a freq
    df_gp.drop(columns=[well_col], inplace=True)  # Drop well_col as it will be present in the multiindex
    df_gp.reset_index(inplace=True)

    df_null = df_gp[df_gp[test_col].isnull()]  # All Null values, which need to be added to the df
    df_null.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True)  # Get time_col back as a column
    df_full = pd.concat([df, df_null], axis=0, ignore_index=True)
    df_full.sort_values(by=[well_col, time_col], inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    return df_full