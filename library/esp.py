"""
Functions specific to esp wells
"""
import pandas as pd


def label_esp_data(df, fail_df, col_to_create='Label', forecasting_delta='10 days', gp_col='NodeID',
                   st_dt_col='Start Date', fl_dt_col='Fail Date', fl_col='Reason For Pull', verbose=0):
    """
    Will create a label column in the main dataframe
    :param df: The dataframe on which we need to add the Label column
    :param fail_df: The dataframe with the failure info
                    Should include the st_dt_col, fl_dt_col, fl_col
    :param col_to_create: This column will be created in df. Default: 'Label'
    :param forecasting_delta: The time-delata which creates the forecasting zones. Default: '10 days'
    :param gp_col: The column in the main dataframe which is to be grouped. Default: 'NodeID'
    :param st_dt_col: Start Date Column in fail_df. Default: 'Start Date'
    :param fl_dt_col: Fail Date Column in fail_df. Default: 'Fail Date'
    :param fl_col: Failure Column in fail_df. Default: 'Reason For Pull'
    :param verbose: 0 for not printing anything
    """
    vprint = print if verbose != 0 else lambda *a, **k: None  # add verbose condition

    forecasting_delta = pd.Timedelta(forecasting_delta)
    df[col_to_create] = 'Drop'
    fail_df.reset_index(inplace=True, drop=True)

    for i in fail_df.index:
        vprint('\n----------')
        # extract info
        well, st_dt, fail_dt, reason = fail_df.loc[i, [gp_col, st_dt_col, fl_dt_col, fl_col]].values
        vprint(well)

        # boolean condition
        bool_1 = (df[gp_col] == well) & (df.Date >= st_dt) & (
                    df.Date < (fail_dt - forecasting_delta))  # Condition for Normal
        bool_2 = (df[gp_col] == well) & (df.Date >= (fail_dt - forecasting_delta)) & (
                    df.Date <= fail_dt)  # Conditon for `Reason For Pull`
        bool_3 = (df[gp_col] == well) & (df.Date > fail_dt) & (
                    df.Date <= (fail_dt + pd.Timedelta('2 days')))  # Condtion for marking actual falilures

        # labeling
        df.loc[bool_1, col_to_create] = 'Normal'
        df.loc[bool_2, col_to_create] = reason
        df.loc[bool_3, col_to_create] = 'Actual ' + reason

        # info
        vprint('Label Counts')
        vprint(pd.Series(df[col_to_create].value_counts()))

    return print('Labeling Done')
