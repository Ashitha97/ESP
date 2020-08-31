# imports
from library import lib_aws  # Using PostgresRDS and AddData
from library import lib_dyna  # Using CardFunctions

# Basic
import pandas as pd
import numpy as np
import time


def transfer_backup_card(well_name):
    start = time.time()
    # Max time supposed to be in data
    backup_max_time = pd.Timestamp('2020-08-28 07:00:00')

    # query for well
    query = """
    select 
    "NodeID",
    "Date",
    "AnalysisDate",
    "SPM",
    "StrokeLength",
    "Runtime",
    "FillBasePct",
    "Fillage",
    "SecondaryPumpFillage",
    encode("tblCardData"."POCDownholeCardB", 'hex') as "POCDownholeCardB",
    encode("tblCardData"."SurfaceCardB", 'hex') as "SurfaceCardB",
    encode("tblCardData"."DownholeCardB", 'hex') as "DownholeCardB",
    encode("tblCardData"."PredictedCardB", 'hex') as "PredictedCardB",
    encode("tblCardData"."TorquePlotMinEnergyB", 'hex') as "TorquePlotMinEnergyB",
    encode("tblCardData"."TorquePlotMinTorqueB", 'hex') as "TorquePlotMinTorqueB",
    encode("tblCardData"."PermissibleLoadUpB", 'hex') as "PermissibleLoadUpB"
    from oasis_dbo."tblCardData"
    where "NodeID" = '{}'
    order by "Date";
    """.format(well_name)

    # Import from backup
    with lib_aws.PostgresRDS(db="oasis-data") as engine:
        test_data = pd.read_sql(query, engine)

    # Basic Cleaning
    test_data.fillna(np.nan, inplace=True)
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    test_data['AnalysisDate'] = pd.to_datetime(test_data['AnalysisDate'])
    # Cleaning Well Names
    test_data["NodeID"] = (test_data["NodeID"].str.replace("#", "")  # remove #
                           .str.replace('\s+', ' ', regex=True)  # remove multiple spaces if present
                           .str.strip()  # Remove trailing whitespaces
                           .str.lower()  # lower all character
                           .str.title()  # Uppercase first letter of each word
                           .map(lambda x: x[0:-2] + x[-2:].upper()))  # last 2 characters should always be upper case

    # Dropping Wrong Dates
    test_data = test_data[test_data.Date <= backup_max_time]
    test_data.drop_duplicates(subset=['NodeID', 'Date'], inplace=True)
    test_data.reset_index(inplace=True, drop=True)

    # Converting Card cols from hex to wkb
    card_cols = [
        'POCDownholeCardB',
        'SurfaceCardB',
        'DownholeCardB',
        'PredictedCardB',
        'TorquePlotMinEnergyB',
        'TorquePlotMinTorqueB',
        'PermissibleLoadUpB'
    ]

    try:
        for col in card_cols:  # Converting the hex columns to a wkb format
            test_data.loc[:, col] = test_data.loc[:, col].apply(lib_dyna.CardFunctions.hex_to_wkb)
    except Exception as e:
        print(e)
        print(well_name)
        return print("Issue with card col")

    end = time.time()
    print('Time taken for import and cleaning: {:.2f}s'.format(end - start))
    print('NodeID changed from {} to {}'.format(well_name, test_data.loc[0, "NodeID"]))
    # print("Total data points {}".format(test_data.shape[0]))
    # print(f'Max time: {test_data.Date.max()}')

    # Add data to production
    lib_aws.AddData.add_data(df=test_data,
                             db='oasis-prod',
                             schema='xspoc',
                             table='card',
                             merge_type='append',
                             card_col=card_cols,
                             index_col='Date')
    return None


def main():
    # Query all wells
    well_query = """select distinct("NodeID") from oasis_dbo."tblCardData" order by "NodeID";"""

    with lib_aws.PostgresRDS(db="oasis-data") as engine:
        wells = pd.read_sql(well_query, engine)

    # transfer
    for i in wells.index:
        well = wells.loc[i, 'NodeID']
        print(f'\n--------\n{i}: {well}')
        transfer_backup_card(well)


if __name__ == "__main__":
    main()
