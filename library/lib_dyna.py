"""
Contains various classes and functions to work with
Dynomometer Cards and Multi-Labels
"""

import struct

# Imports
import numpy as np
import pandas as pd
from geoalchemy2.shape import from_shape
from pyefd import elliptic_fourier_descriptors
from shapely.geometry import Polygon
from shapely.wkb import loads
# ml
from sklearn.preprocessing import MultiLabelBinarizer


class CardFunctions:
    """
    This Class contains various static methods which help with
    dyna card data and the cards
    Thus class has inherited the class CardTransformation
    """

    @staticmethod
    def get_dyna(card_arr):
        """
        Transforms Hexadecimal Dyna Card Value into Position and Load value
        :param card_arr: Hexadecimal Array
        :return: Position, Load 2D array
        """
        if pd.isnull(card_arr):
            pos = [0, 0, 0]
            load = [0, 0, 0]

        else:
            test_card = card_arr.strip()
            mid = len(test_card) / 2
            mid = int(mid)

            load = []
            pos = []

            for i in range(0, mid, 8):
                load_temp = test_card[i:i + 8]
                load_int = struct.unpack('f', bytes.fromhex(load_temp))[0]
                load.append(load_int)

                pos_temp = test_card[mid + i:mid + i + 8]
                pos_int = struct.unpack('f', bytes.fromhex(pos_temp))[0]
                pos.append(pos_int)

        return np.column_stack(([pos, load]))

    @staticmethod
    def hex_to_wkb(card_arr):
        """
        Transforms the Hexadecimal based card into a WKB element
        Helps store the data in a postgis db
        :param card_arr: Hexadecimal Card Value
        :return: WKB card value
        """
        xy = CardFunctions.get_dyna(card_arr)

        try:
            polygon = Polygon(xy)
            wkb_element = from_shape(polygon)
        except Exception as e:
            print(e)
            wkb_element = np.nan

        return wkb_element

    @staticmethod
    def hex_to_poly(hex_card):
        """
        Transforms a dyna card in a WKB format to a shapely Polygon Object
        :param hex_card: DynaCard in a WKB format
        :return: DynaCard as a Shapely Polygon
        """
        poly_card = loads(hex_card, hex=True)
        return poly_card

    def __init__(self, df, card_col, well_col):
        self.df = df.copy()  # The Dataset that we are working on
        self.card_col = card_col  # Column Name of card values
        self.well_col = well_col  # Column Name of well values

    def remove_errors(self):
        """
        The function eliminates those df points where FD is NAN
        Note: For now we aren't looking at the error data points
        """

        if self.card_col is None:
            return print("No Changes done as card_col not provided")

        error_data = []

        if isinstance(self.df.loc[0, self.card_col], str):  # Checking if the card col is a shapely object
            self.df.loc[:, self.card_col] = self.df.loc[:, self.card_col].apply(
                lambda x: loads(x, hex=True))  # Convert to Polygon

        for i in self.df.index:
            poly = self.df.loc[i, self.card_col]
            xy = np.asarray(poly.exterior.coords)

            # Ignore zero division error
            try:
                fd = elliptic_fourier_descriptors(xy, order=3)  # get fourier descriptors
            except RuntimeWarning:
                fd = np.nan

            if np.isnan(fd).any():
                error_data.append(i)

        print("Total errors found in {} datapoints".format(len(error_data)))
        # error = self.df.loc[error_data]
        self.df.drop(error_data, inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df.fillna(value=np.nan, inplace=True)

    def well_bounds(self, df_well):
        """
        This function finds the pos-load limits for a specific well
        Note: This can also find the pos-load limits for the entire dataset
        :df_well: DataFrame for which the pos-load limits should be found
        :return: well_limits for a specific well
        """

        if self.card_col is None:
            return print("No Changes done as card_col not provided")

        min_max_arr = np.zeros([df_well.shape[0], 4])  # Initialize the empty array

        for i in df_well.index:
            poly = df_well.loc[i, self.card_col]
            min_max_arr[i, :] = list(poly.bounds)

        pos_min = np.min(min_max_arr[:, 0])
        pos_max = np.max(min_max_arr[:, 2])
        load_min = np.min(min_max_arr[:, 1])
        load_max = np.max(min_max_arr[:, 3])

        well_limits = [pos_min, pos_max, load_min, load_max]
        return well_limits

    def all_bounds(self):
        """
        This function gets the data frame which has the pos load limits for all wells in the data
        Note: In production use another function which imports it from a database
        :return: Data Frame with well value limits
        """

        if self.well_col is None:
            return print("No Changes done as well_col not provided")

        wells = self.df[self.well_col].unique()  # Unique wells in df
        min_max_dict = dict.fromkeys(wells)  # Empty dict with wells as keys

        for well in wells:
            df_well = self.df[self.df[self.well_col] == well]
            df_well.reset_index(drop=True, inplace=True)
            min_max_dict[well] = self.well_bounds(df_well)  # Call well_bounds method

        bounds_df = pd.DataFrame(min_max_dict)
        bounds_df.index = ['pos_min', 'pos_max', 'load_min', 'load_max']

        return bounds_df


class MultiLabels(CardFunctions):
    """
    Class which works on Multi-Labels.
    For Initialization either a df can be provided (df) Or labels as  an array (labels)
    If both are provided an exception will be raised
    """

    def __init__(self, df=None, card_col=None, well_col=None, label_cols=None, labels=None):
        super().__init__(df, card_col, well_col)  # Inherit from CardFunctions
        self.label_cols = label_cols
        self.merged = labels
        if not self.df.empty and self.merged is not None:
            raise Exception("Initialize either df or labels, not both")
        elif self.df.empty and self.merged is None:
            raise Exception("Gotta provide either df or labels")

    def merge_labels(self):
        """
        This orders the labels correctly and merges them
        If labels have been used for initialization, they will be ordered
        and a df will be created with these labels
        Whenever this runs it will update the data as well
        For ordering we use MultiLabelBinarizer
        """
        if self.df is not None:
            merged = self.df[self.label_cols].apply(tuple, axis=1)  # Merges the labels into a list of tuples
            try:
                merged = merged.apply(lambda lbl: tuple(x for x in lbl if x == x))  # Removes None values from tuple
            except:
                merged = merged.apply(
                    lambda lbl: tuple(x for x in lbl if x is not None))  # Removes nan values from tuple
        else:
            merged = self.merged
            self.df = pd.DataFrame(columns=self.label_cols)

        # This block has to be only be run once
        # For the future implement a counter and it only runs when we run it the first time
        # Or something else
        mlb = MultiLabelBinarizer()
        binarized_labels = mlb.fit_transform(merged)
        unique_labels = mlb.inverse_transform(binarized_labels)
        self.merged = unique_labels

        # Updating the data
        ordered_df = pd.DataFrame(self.merged, columns=self.label_cols)  # Make the labels into a df
        self.df.loc[:, self.label_cols] = ordered_df.loc[:, self.label_cols]
        self.df.fillna(np.nan, inplace=True)  # Replace None with NaN
        # print("Merged and Ordered Labels stored in class variable merged and Data updated")
        return None

    def get_group_counts(self):
        """
        Get Counts for gps of labels present in the dataset
        :return: DataFrame of Label Gps and thier Counts
        """
        if self.merged is None:  # Will run the merging function
            self.merge_labels()

        # This will only work if we have multi labels
        # or if we have label_cols matching the actual labels
        # causes errors after removing labels
        # Improve the remove_label func
        ordered_df = pd.DataFrame(self.merged, columns=self.label_cols).fillna('NA')  # Make the labels into a df

        labelComb = ordered_df.groupby(self.label_cols).size().reset_index().rename(columns={0: 'totalVal'})
        labelComb['pctVal'] = np.round(labelComb['totalVal'] / self.df.shape[0] * 100, 2)
        labelComb.sort_values(by='totalVal', ascending=False, inplace=True)
        labelComb.reset_index(drop=True, inplace=True)

        return labelComb

    def get_label_counts(self):
        """
        Get counts for all Labels Present
        :return: Singular Labels DataFrame with their Counts
        """

        frames = []

        for i in self.label_cols:
            frames.append(self.df[i])

            label_counts = pd.concat(frames, axis=0).value_counts()
            label_pct = np.round(label_counts / self.df.shape[0] * 100, 3)  # Pct Value Counts

            counts_df = pd.DataFrame(data={
                "totalVal": label_counts,
                "pctVal": label_pct
            })

        return counts_df

    def remove_labels(self, thresh):
        """
        Removes those gp of labels which are below a threshold.
        Note: This threshold is specified as a pct vale
        :param thresh: Threshold below which label gps are removed
        :return: Updates the Df with the labels removed
        """
        label_comb_df = self.get_group_counts()  # Get the Grouped label info

        self.df.fillna('NA', inplace=True)  # This is done because the label_comb_df will have "NA" instead of nan
        labels_to_remove = label_comb_df.loc[label_comb_df.pctVal < thresh, self.label_cols].apply(tuple,
                                                                                                   axis=1).to_list()
        bool_ = self.df[self.label_cols].apply(tuple, axis=1).isin(labels_to_remove)
        # Updating the df
        self.df = self.df.loc[~bool_, :]  # Dropping all combinations from the "labels_to_remove"
        self.df.replace('NA', np.nan, inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.merge_labels()  # Update the merged Labels as well

        return print("Data has been updated by removing labels below a threshold of {}".format(thresh))


class Features(MultiLabels):

    def __init__(self, df=None, card_col=None, well_col=None, label_cols=None, labels=None):
        super().__init__(df, card_col, well_col, label_cols, labels)

    @staticmethod
    def norm_static(xy, static_bounds):
        """
        Using a static pos-lod limits to normalize the cards
        :param xy: The Actual card pos and load values as a 2D array
        :param static_bounds: An array of [pos_min, pos_max, load_min, load_max]
        :return: Normalized Card (output is in the form of a 2D array of position and load)
        """
        [pos_min, pos_max, load_min, load_max] = static_bounds

        norm_pos = [((i - pos_min) / (pos_max - pos_min)) for i in xy[:, 0]]
        norm_load = [((i - load_min) / (load_max - load_min)) for i in xy[:, 1]]
        norm_xy = np.column_stack(([norm_pos, norm_load]))
        return norm_xy

    @staticmethod
    def norm_card(xy):
        """
        Just use the card values to normalize that specific card
        :param xy: The True Card pos-load values as a 2D-array
        :return: Normalized Card (As a 2D array of position and Load)
        """
        x_mx = np.max(xy[:, 0])
        x_mn = np.min(xy[:, 0])
        y_mx = np.max(xy[:, 1])
        y_mn = np.min(xy[:, 1])

        norm_x = [((i - x_mn) / (x_mx - x_mn)) for i in xy[:, 0]]
        norm_y = [((i - y_mn) / (y_mx - y_mn)) for i in xy[:, 1]]
        norm_xy = np.column_stack(([norm_x, norm_y]))
        return norm_xy

    @staticmethod
    def norm_df(xy, well, bounds_df):
        """
        This function will normalize the card depending on the limits for that well.
        Therefore we need a "bounds_df" which will include the pos-load limits for that particular well
        :param xy: True card pos-load values as an array
        :param well: The well this card belongs to
        :param bounds_df: A dataframe with pos-load limits as the index and Well Names as the column
        :return: Normalized Card (As a 2D array of position and Load)
        """
        pos_max = bounds_df.loc['pos_max', well]
        pos_min = bounds_df.loc['pos_min', well]

        load_max = bounds_df.loc['load_max', well]
        load_min = bounds_df.loc['load_min', well]

        norm_pos = [((i - pos_min) / (pos_max - pos_min)) for i in xy[:, 0]]
        norm_load = [((i - load_min) / (load_max - load_min)) for i in xy[:, 1]]
        norm_xy = np.column_stack(([norm_pos, norm_load]))
        return norm_xy

    def get_X(self, fd_order=5, area=False, centroid=False, normalize_fn=None, norm_arg=None):
        """
        Get the Feature matrix of the multi label dyna card Data Frame
        :param fd_order: Order of Fourier descriptors(Default=5)
        :param area: Area as a feature(Boolean, Default: False)
        :param centroid: Centroids as a feature(Boolean, Default: False)
        :param normalize_fn: 'df', 'static', 'card', None(Default)
        :param norm_arg: Data needed for the normalize_df
                        'df' --> bounds_df
                        'static' --> static_bounds (List of pos-load limits)
        :return:
        """

        if {self.well_col, self.card_col}.issubset(set(self.df.columns)):
            pass
        else:
            print("well_col and/or card_col not present in DataFrame provided")

        n = self.df.shape[0]  # No of rows in feature vector
        m = 0  # Initialize no of columns in feature vector as 0

        if fd_order:
            m += fd_order * 4

        if area:
            m += 1

        if centroid:
            m += 2

        x = np.zeros([n, m])  # initialize the entire feature matrix

        for i in self.df.index:
            i_features = np.zeros([m, ])  # Features in the ith row
            poly = self.df.loc[i, self.card_col]
            well = self.df.loc[i, self.well_col]

            try:
                xy = np.asarray(poly.exterior.coords)
            except Exception as e:
                print(i)
                return print(
                    "Polygon column not transforming to xy, check that it is not in hex or wkb format(Run remove_errors()")

            # Normalizing
            if normalize_fn == 'df':
                norm_xy = self.norm_df(xy, well, norm_arg)

            elif normalize_fn == 'static':
                # Normalizing using the pos-load limits of the current dataset
                try:
                    norm_xy = self.norm_static(xy, norm_arg)
                except ValueError:
                    return print("Give the correct normalize_fn and the corresponding norm_arg")

            elif normalize_fn == 'card':
                # Normalizing the card wrt the card itself
                norm_xy = self.norm_card(xy)

            else:
                # Not Normalized
                norm_xy = xy

            if fd_order:  # Add FD to feature matrix
                fd = elliptic_fourier_descriptors(norm_xy, order=fd_order, normalize=False)
                i_features[: fd_order * 4] = fd.flatten()

            if centroid:  # Centroid Added as i_features
                cent = list(Polygon(norm_xy).centroid.coords)
                i_features[fd_order * 4: (fd_order * 4 + 2)] = cent[0]

            if area:  # Area as a feature
                ar = Polygon(norm_xy).area
                i_features[-1] = ar

            x[i, :] = i_features  # Added to the main feature matrix

        return x

    def get_Y(self):
        """
        Get the label vector in a binarized form.
        MultiLabelBinarizer is used. This is the second out.
        :return: Y, MultiLabelBinarizer object
        """

        if self.merged is None:
            self.merge_labels()  # Update the df with the latest merged labels generated

        mlb = MultiLabelBinarizer()  # Define a mlb object
        y = mlb.fit_transform(self.merged)

        return y, mlb


class Predictions_MultiLabel:
    """
    Class which makes predictions for a multi-label classification problem
    """

    def __init__(self, model, x, mlb):
        """
        Initialization Params
        :param model: ML model
        :param x: Feature Matrix
        :param bin: MultiLabel Binarizer
        """
        self.model = model
        self.x = x
        self.mlb = mlb

    def get_pred_df(self, n):
        """
        Method which gives returns a prediction df
        :param n: No of labels to predict
        :return: a prediction DataFrame
        """
        class_dict = dict(zip(list(range(self.mlb.classes_.shape[0])), self.mlb.classes_))
        y_prob = self.model.predict_proba(self.x)

        pred_df = pd.DataFrame()  # Intialize an empty DataFrame

        for i in range(n):
            i_ = i + 1  # As we are using the desc order 0 in asc will correspond to -1 in desc
            label = np.argsort(y_prob)[:, -i_]  # Get the ith numerical label
            label = np.vectorize(class_dict.get)(label)  # Get the corresponding Class name

            prob = np.sort(y_prob)[:, -i_] * 100  # Get the ith probability

            pred_df['Label' + str(i_)] = label  # Append it to the
            pred_df['Prob' + str(i_)] = prob

        return pred_df.round(2)
