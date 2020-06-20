"""
Transfers specific column and wells from the 'tblXDiagResults' in the postgres DB
to the PostgreSQL db in an ec2 instance.
This is for visualizing said data in grafana
"""

from library.dbconnection import PostgresRDS
from library.sqlFunctions import run_query, AddData
from library.cardFunctions import Cleaning

def import_xdiag(well_list):
    """
    Function will import xdiag data from the main db
    Modify the query to include necessary columns
    :param well_list: List of wells
    :return: Pandas DataFrame
    """
    query = """
    SELECT
    "NodeID", "Date", "FillagePct", "MotorLoad", 
    "TubingPressure", "CasingPressure", "Friction",
    "TubingLeak", "FluidLevelXDiag"
    FROM xspoc_dbo."tblXDiagResults"
    WHERE "NodeID" in {}
    ORDER BY "NodeID", "Date"
    """.format(tuple(well_list))

    with PostgresRDS(db='oasis-data') as engine:
        data = run_query(query, engine)

    return data


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
    encode("tblCardData"."POCDownholeCardB", 'hex') as downcard,
    encode("tblCardData"."SurfaceCardB", 'hex') as surfcard,  -- Add more column if need be
    "SPM", "CardArea", "StrokeLength"
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
        'Cook 41-12 11T'
    ]
    card_cols = ['downcard', 'surfcard']

    #XDiag
    xdiag = import_xdiag(well_list)
    print(xdiag.head())
    AddData.add_data(xdiag, db='oasis-dev', table='xdiag',
                     merge_type='replace', card_col=None, index_col='Date')

    #tblCardData
    card_data = import_card(well_list, card_cols)
    print(card_data.head())
    AddData.add_data(df=card_data, db='oasis-dev', table='card',
                     merge_type='replace', card_col=card_cols, index_col='Date')



if __name__ == '__main__':
    main()
