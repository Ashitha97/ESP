"""
Script Adds labeled Dynamometer Data to the DB
Note: the Labeled data should be in a .csv format

Functionality to Add:
Get label Info and save it to a log file
Automate the Process of labeling
"""
import pandas as pd
import sys
# Local imports
from library import lib_aws


def main():
    # Using s3fs library
    # If using just boto3 implementations will change
    data_path = 's3://et-oasis/labeledData/dynaLabels.csv'  # currently saved in an s3 bucket
    cols = ['Date', 'NodeID', 'TrueLabel1', 'TrueLabel2', 'downcard'] # Columns that need to match

    # Block to check validity of columns
    try:
        data = pd.read_csv(data_path, usecols=cols, parse_dates=['time'])
    except KeyError as e:
        return print("Columns Dont Match")
    except Exception as e:
        print(e)
        sys.exit()

    # Adding data to the DB
    # Use the class AddData from library.sqlFunc
    lib_aws.AddData.add_data(data, db='oasis-dev', table='dynalabel', schema='clean',
                     merge_type='replace',card_col=['downcard'], index_col='time')

    # Update Index
    with lib_aws.PostgresRDS(db='bonanza-data') as engine:
        with engine.begin() as connection:
            connection.execute("""CREATE UNIQUE INDEX dyna_idx ON clean.dynalabel ("NodeID", "Date");""")


if __name__ == '__main__':
    main()