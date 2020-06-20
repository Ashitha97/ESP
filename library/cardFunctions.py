"""
Includes classes and functions which help clean and work with card data
"""
import pandas as pd
import numpy as np
import struct
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon


class Cleaning:

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
            test_card = card_arr
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
        xy = Cleaning.get_dyna(card_arr)

        try:
            polygon = Polygon(xy)
            wkb_element = from_shape(polygon, srid=4326)
        except Exception as e:
            print(e)
            wkb_element = np.nan

        return wkb_element
