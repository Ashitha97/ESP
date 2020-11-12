"""
Functions for Preprocessing the data
"""
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#  Feature Engineering Functions
def node_clean(node_str):
    """
    Function that cleans up NodeID strings
    """
    node_str = " ".join(node_str.split())  # remove empty white spaces
    node_str = node_str.replace('#', "").strip().lower().title()  # remove # character, plus clean characters
    node_str = node_str[0:-2] + node_str[-2:].upper()  # last 2 characters whill alwsy be upper case
    return node_str


def fill_null(dataframe, chk_col='PPRL', gp_col='NodeID', time_col='Date'):
    """
    This function will fill in Null Values on those dates where no data-points are present
    Will drop duplicates based on well_col and time_col
    Note: Will drop Non Numerical DataFrames
    TODO: Add Functionality where non-Numerical columns will work
    :param time_col: The timestamp column. Default='Date'
    :param gp_col: Column with group id. Default='NodeID'
    :param dataframe: The dataframe to work on
    :param chk_col: The column which will be used as the basis for filling NUll
                    This column should be numerical and the one with the least Null data points
    """
    data_temp = dataframe.copy()

    # Set time col as index if it is not
    if time_col in data_temp.columns:
        data_temp.set_index(time_col, inplace=True)

    data_gp = data_temp.groupby(gp_col).resample('1D').max()  # Groupby wellname and resample to Day freq
    data_gp.drop(columns=[gp_col], inplace=True)  # Drop these columns as they are present in the index
    data_gp.reset_index(inplace=True)  # Get Back WellCol from

    data_null = data_gp[
        data_gp.loc[:, chk_col].isnull()]  # Get all null values, which need to be added to the main data file
    data_null.reset_index(inplace=True, drop=True)
    data_temp.reset_index(inplace=True)  # get timestamp back in the column for concating

    data_full = pd.concat([data_temp, data_null], axis=0, ignore_index=True)  # concat null and og files
    data_full.sort_values(by=[gp_col, time_col], inplace=True)
    data_full.drop_duplicates(subset=[gp_col, time_col], inplace=True)
    data_full.reset_index(drop=True, inplace=True)

    return data_full


# noinspection PyTypeChecker
def split_by_well(dataset, no_of_wells, fail_col='Label', well_col='NodeID', zero_label='Normal', random_state=42,
                  verbose=0):
    """
    Split by well
    --------------
    - Splits the dataFrame into and train and test
    - Entire wells are split accorind to the failures in fail Column
    - First we identify well ditribution in each type of failure
    - The we keep some group of wells aside for testing Such that all failuress are represented in both train and test dataframes
    TODO: Maybe add a unique well features

    Parameters
    --------------------
    :param dataset: The dataframe that needs to be split
    :param no_of_wells: Number of wells from each class in the test dataset
    :param fail_col: The Column in the dataset which has the labels/failure labels
    :param well_col: The Column which has well names
    :param zero_label: The label which defines the normal working condition
    :param random_state: Default 42
    :param verbose: 0 for not printing anything
    """
    vprint = print if verbose != 0 else lambda *a, **k: None  # add verbose condition
    random.seed = random_state

    # get failure distribution
    fail_dist = dataset.groupby(fail_col)[well_col].agg(['unique'])

    test_wells = np.array([])  # initialize empty array
    for i in fail_dist.index:
        # skip for normal
        if i == zero_label:
            continue

        # get unique list of wells in specific failure
        temp_list = fail_dist.loc[i]['unique']
        if len(temp_list) <= no_of_wells:
            raise Exception(f'Not enough wells to split for Failure: {i} which has just {len(temp_list)} wells')

        well_names = [x for x in
                      random.sample(set(temp_list), no_of_wells)]  # get n random sample of wells from that list
        test_wells = np.append(test_wells, well_names, axis=0)  # append it to main test array

    test = dataset[dataset[well_col].isin(test_wells)]
    test.reset_index(inplace=True, drop=True)

    train = dataset[~dataset[well_col].isin(test_wells)]
    train.reset_index(inplace=True, drop=True)

    vprint('Label Distribution:\n----------')
    vprint(pd.concat([train.Label.value_counts(), test.Label.value_counts()], keys=['Train', 'Test'], axis=1))

    vprint('Wells Used in Test-Set:\n---------', *test_wells, sep='\n')

    vprint("NAN value Distribution in %:\n-------------")
    vprint(pd.concat([train.isnull().sum(axis=0) / len(train), test.isnull().sum(axis=0) / len(test)],
                     keys=['Train', 'Test'], axis=1).round(4) * 100)

    return train, test


class Segmentation:

    @staticmethod
    def create_segments_clean(dataframe, seg_size, overlap, feature_col, label_col):
        """
        Segment the dataframe which has a constant sampling rate with a segment of seg_size.
        Will only work for single well.
        :param dataframe: The time-series df with constant frequency which needs to be segmented
        :param seg_size: The number of data-points in a segment
        :param overlap: Fraction which says how much % of consecutive segments should be different
        :param feature_col: Which Features to use for creating segments (works for multivariate)
        :param label_col: Column name to use for labelling the segments
        :return segments, segment_labels, time

        Some Assumptions:
        Adding Labels:
        - In a segment even if one datapoint is labelled as a failure that segment is labelled as a failure
        - In a failure seg, if the last data-point is 'Normal' that will be dropped
        - Segments with 'Drop' in the label col will be dropped.
        - This will happen even if the segment has a single drop label.
        - While marking drop in the labels take this into account
        """
        valid_seg = []  # initialize empty valid segments array
        valid_seg_time = []  # initialize empty timestamp
        valid_seg_labels = []  # Initialize empty label segments
        t0_old = 0

        for t0 in dataframe.index:
            # Overlap handling
            # go to next iteration if t0 - t0_old (current index - the previous one) is less than how much timedelta in overlap
            if (t0 != 0) & ((t0 - t0_old) < (seg_size * overlap)):
                continue

            tn = t0 + seg_size  # index + seq_size
            seg = dataframe.loc[t0:tn - 1, feature_col].dropna().values  # get the seg and drop nan values
            labels = dataframe.loc[t0:tn - 1, label_col].values  # Get list of labels
            dates = dataframe.loc[t0:tn - 1, 'Date'].values  # get dates

            # check if length of seg is = seg_size
            # Will take care of dropping missing values
            if len(seg) == seg_size:
                # Label Handling
                # Skip those segments which have 'Drop' in any datapoints
                if 'Drop' in labels:
                    continue
                    # if all labels are normal seg_label will be 'Normal'
                elif all(x == 'Normal' for x in labels):
                    seg_label = 'Normal'
                # contains a label other than 'Normal'
                else:
                    seg_label = [x for x in labels if x != 'Normal']
                    seg_label = np.unique(seg_label)[0]
                # If failure seg and last val is 'Normal' drop the segment
                if (seg_label != 'Normal') & (labels[-1] == 'Normal'):
                    continue
                #             print(f'Length of Segment: {len(seg)}\tLength of Vals {len(vals)}\tWith Label {seg_label}')
                valid_seg.append(seg)  # append the valid segments to the valid seg array
                valid_seg_time.append(dates)  # append dates
                valid_seg_labels.append(seg_label)  # append labels

            t0_old = t0  # will be used for overlap handling

        valid_seg = np.array(valid_seg)
        valid_seg_time = np.array(valid_seg_time)
        valid_seg_labels = np.array(valid_seg_labels)

        return valid_seg, valid_seg_labels, valid_seg_time

    @staticmethod
    def nested_dataframe_from_3d_array(X):
        """
        Convert numpy array with shape (n_instance, n_timepoints, n_features)
        into a pandas DataFrame (with timeseries as pandas Series in cells)
        :param X: NumPy ndarray, input
        :returns pandas DataFrame
        """
        dataframe = pd.DataFrame()
        n_instances = X.shape[0]
        n_variables = X.shape[2]

        for variable in range(n_variables):
            dataframe['var_' + str(variable)] = [pd.Series(X[instance, :, variable]) for instance in range(n_instances)]

        return dataframe


class Normalization:

    @staticmethod
    def well_specific(dataset, cols_norm, cols_keep, well_col='NodeID'):
        """
        Normalize the dataset in a well specific basis. Will use a MinMaxScaler
        :param dataset: The dataframe to normalize
        :param cols_norm: Numerical columns to scale
        :param cols_keep: Non-numerical Columns which we need in the dataset
        :param well_col: Group Column. Default: 'NodeID'
        :return: None
        """

        frames = []
        for well in dataset.NodeID.unique():
            temp_scaler = MinMaxScaler()  # initialize scaler for every well
            temp_df = dataset[dataset[well_col] == well].copy()  # get well df
            temp_df.dropna(how='any', subset=cols_norm, inplace=True)  # drop nans only where norm col rows are nan
            temp_df.reset_index(inplace=True, drop=True)
            try:
                scaled_arr = temp_scaler.fit_transform(temp_df[cols_norm])  # scale the data points
            except ValueError:  # will catch if well has no data-points
                print(f'Values Dropped in well: {well}')
                continue

            scaled_temp_df = pd.DataFrame(data=scaled_arr, columns=cols_norm)
            scaled_temp_df = pd.concat([temp_df[cols_keep], scaled_temp_df], axis=1)
            frames.append(scaled_temp_df)
            del temp_df, scaled_arr, scaled_temp_df, temp_scaler

        data_scaled = pd.concat(frames)
        data_scaled.reset_index(inplace=True, drop=True)

        return data_scaled

    @staticmethod
    def full_scaling(dataset, cols_norm, cols_keep, scaler=None):
        """
        Scale the entire dataset
        :param dataset:
        :param cols_norm:
        :param cols_keep:
        :param scaler:
        :return: scaled_df, trained_scaler
        """
        if scaler is None:
            scaler = MinMaxScaler()

        data_temp = dataset.copy()
        data_temp.dropna(how='any', subset=cols_norm, inplace=True)
        data_temp.reset_index(inplace=True, drop=True)
        scaled_arr = scaler.fit_transform(data_temp[cols_norm])

        scaled_df = pd.DataFrame(data=scaled_arr, columns=cols_norm)
        scaled_df = pd.concat([data_temp[cols_keep], scaled_df], axis=1)

        return scaled_df, scaler
