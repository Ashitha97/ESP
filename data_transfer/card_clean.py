"""
Will clean specific card files and add it to the database
"""
# imports
import os
import numpy as np
import pandas as pd

# Card Conv imports
import struct
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

from library.lib_aws import AddData

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


def hex_to_wkb(card_arr):
    """
    Transforms the Hexadecimal based card into a WKB element
    Helps store the data in a postgis db
    :param card_arr: Hexadecimal Card Value
    :return: WKB card value
    """
    xy = get_dyna(card_arr)

    try:
        polygon = Polygon(xy)
        wkb_element = from_shape(polygon)
    except Exception as e:
        print(e)
        wkb_element = np.nan

    return wkb_element


def import_data(file_path, file_name, drop_cols):
    max_time = pd.Timestamp(file_name.split('.')[2] + file_name.split('.')[3])
    data = pd.read_csv(os.path.join(file_path, file_name), parse_dates=['Date', 'AnalysisDate'])
    data.drop(columns=drop_cols, inplace=True)
    data = data[data.Date <= max_time]
    data.sort_values(by=["NodeID", "Date"], inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data


def convert_card_columns(data, card_columns):
    for col in card_columns:
        print(col)
        data.loc[:, col] = data.loc[:, col].apply(hex_to_wkb)

    return data


def main():
    cols_to_drop = [
        'SurfaceCard',
        'DownholeCard',
        'PredictedCard',
        'PocDHCard',
        'CorrectedCard',
        'TorquePlotMinEnergy',
        'TorquePlotMinTorque',
        'TorquePlotCurrent',
        'POCDownholeCard',
        'ElectrogramCardB'
    ]
    card_cols = [
        'SurfaceCardB',
        # 'DownholeCardB',
        # 'PredictedCardB',
        # 'TorquePlotMinEnergyB',
        # 'TorquePlotMinTorqueB',
        # 'TorquePlotCurrentB',
        'POCDownholeCardB'
        # 'PermissibleLoadUpB',
        # 'PermissibleLoadDownB'
    ]


    local_file_path = r'C:\Users\rai_v\OneDrive\Python Coursera\local-data\oasis'
    file= 'CardData.E2E.20200728.1100.csv'

    df = import_data(local_file_path, file, cols_to_drop)
    print("import done")
    df = convert_card_columns(df, card_cols)

    print(df.head())
    print(df.shape[0])

    # Add Data to Database
    df.set_index("Date", inplace=True)
    AddData.add_data(df=df, db='oasis-dev', schema='stream', table='card',
                     merge_type='append', card_col=card_cols)


if __name__ == "__main__":
    main()

