"""
Transfers specific column and wells from the 'tblXDiagResults' in the postgres DB
to the PostgreSQL db in an ec2 instance.
This is for visualizing said data in grafana
"""

from library.dbconnection import PostgresRDS
from library.sql_functions import run_query, AddData
from library.card_functions import Cleaning

issue_well = 'Spratley 5494 14-13 15T'

def import_card(well_list, card_cols):
    """
    Import card data from tblCardData from the main DB
    :param card_cols: List of column with dyna card data
    :param well_list: List of wells
    :return: Pandas DataFrame with card cols in WKB format
    """
    query = """
    SELECT 
    "NodeID",
    "Date",
    encode("tblCardData"."POCDownholeCardB", 'hex') as pocdowncard,
    encode("tblCardData"."SurfaceCardB", 'hex') as surfcard,
    encode("tblCardData"."DownholeCardB", 'hex') as downcard,
    encode("tblCardData"."PredictedCardB", 'hex') as "PredictedCardB",
    encode("tblCardData"."TorquePlotMinEnergyB", 'hex') as "TorquePlotMinEnergyB",
    encode("tblCardData"."TorquePlotMinTorqueB", 'hex') as "TorquePlotMinTorqueB",
    encode("tblCardData"."TorquePlotCurrentB", 'hex') as "TorquePlotCurrentB",
    encode("tblCardData"."PermissibleLoadUpB", 'hex') as "PermissibleLoadUpB",
    encode("tblCardData"."PermissibleLoadDownB", 'hex') as "PermissibleLoadDownB",
    "SPM",
    "StrokeLength",
    "Runtime",
    "LoadLimit",
    "HiLoadLimit",
    "LoLoadLimit", 
    "Fillage", 
    "FillBasePct", 
    "CauseID",
    "AnalysisDate"
    FROM xspoc_dbo."tblCardData"
    WHERE "NodeID" in {}
    ORDER BY "NodeID" , "Date";
    """.format(tuple(well_list))

    with PostgresRDS(db='oasis-data') as engine:
        data = run_query(query, engine)

    if set(card_cols).issubset(data.columns):
        pass
    else:
        raise ValueError("Card Columns were not specified properly. Check the query in import_card function")

    for col in card_cols:  # Converting the hex columns to a wkb format
        data.loc[:, col] = data.loc[:, col].apply(Cleaning.hex_to_wkb)

    return data


def main():
    # Add list of wells that we wont to visualize in grafana
    well_list = [
        'Bonner 9-12H',
        'Bonner 9X-12HA',
        'Bonner 9X-12HB',
        'Cade 12-19HA',
        'Cade 12-19HB',
        'Cade 12X-19H',
        'Cook 12-13 6B',
        'Cook 12-13 7T',
        'Cook 12-13 9T',
        'Cook 41-12 11T',
        'Hanover Federal 5300 41-11 10B',
        'Hanover Federal 5300 41-11 11T',
        'Hanover Federal 5300 41-11 12B',
        'Hanover Federal 5300 41-11 13TX',
        'Helling Trust 43-22 10T',
        'Helling Trust 43-22 16T3',
        'Helling Trust 43-22 4B',
        'Helling Trust 44-22 5B',
        'Helling Trust 44-22 6B',
        'Helling Trust 44-22 7B',
        'Johnsrud 5198 14-18 11T',
        'Johnsrud 5198 14-18 13T',
        'Johnsrud 5198 14-18 15TX',
        'Lite 5393 31-11 9B',
        'Lite 5393 41-11 11B',
        'Lite 5393 41-11 12T',
        'Rolfson N 5198 12-17 5T',
        'Rolfson N 5198 12-17 7T',
        'Rolfson S 5198 11-29 2TX',
        'Rolfson S 5198 11-29 4T',
        'Rolfson S 5198 12-29 6T',
        'Rolfson S 5198 12-29 8T',
        'Rolfson S 5198 14-29 11T',
        'Rolfson S 5198 14-29 13T',
        'Stenehjem 14-9H',
        'Spratley 5494 14-13 12B',
        'Spratley 5494 14-13 13T',
        'Stenehjem 14X-9HA',
        'Stenehjem 14X-9HB',
        'Stenehjem 15-9HA',
        'Stenehjem 15-9HB',
        'Stenehjem 15X-9H'
    ]
    card_cols = [
        'downcard',
        'surfcard',
        'pocdowncard',
        "PredictedCardB",
        "TorquePlotMinEnergyB",
        "TorquePlotMinTorqueB",
        "TorquePlotCurrentB",
        "PermissibleLoadUpB",
        "PermissibleLoadDownB"
    ]

    # tblCardData
    card_data = import_card(well_list, card_cols)
    card_data.set_index(['NodeID', "Date"])
    AddData.add_data(df=card_data, db='oasis-dev', schema='xspoc', table='card',
                     merge_type='append', card_col=card_cols)


if __name__ == '__main__':
    main()
