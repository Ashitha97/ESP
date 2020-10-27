from library import lib_aws
import pandas as pd
import time
import numpy as np


# Helper functions
# Clean up strings
def node_clean(node_str):
    """
    Function that cleans up NodeID strings
    """
    node_str = " ".join(node_str.split())  # remove empty white spaces
    node_str = node_str.replace('#', "").strip().lower().title()  # remove # character, plus clean characters
    node_str = node_str[0:-2] + node_str[-2:].upper()  # last 2 characters will always be upper case
    return node_str


def get_address():
    # Get the addresses we need to pull
    query = """
    select * from espaddr;
    """

    with lib_aws.PostgresRDS(db='esp-data') as engine:
        esp_addr = pd.read_sql(query, engine)

    address = esp_addr.dropna().Address.astype(int).unique()

    return address


def import_data(addr):
    cut_off_dt = pd.Timestamp('2020-08-18')
    query = """
    select * from dbo."tblDataHistory"
    where "Address" = {};
    -- order by "NodeID", "Date";
    """.format(addr)

    try:
        with lib_aws.PostgresRDS(db='oasis-data') as engine:
            sample = pd.read_sql(query, engine, parse_dates=['Date'])

        # data modification
        sample.NodeID = sample.NodeID.apply(node_clean)
        sample = sample[sample.Date < cut_off_dt]
        sample.drop_duplicates(subset=['NodeID', 'Date', 'Address'], inplace=True)
        sample.reset_index(inplace=True, drop=True)
        return sample
    except Exception as e:
        print(e)
        return pd.DataFrame()


def main():
    complete_old = [32176,
                32166,
                32141,
                32145,
                32140,
                32137,
                32125,
                32126,
                40760,
                42150,
                42104,
                40003,
                40007,
                30234,
                30127,
                30170,
                30174,
                30218,
                30222,
                30272,
                32138,
                32142,
                32171,
                32175,
                40001,
                40002,
                40004,
                40005,
                40006,
                40025,
                40030,
                40031,
                40033]

    complete = [32176,
            32166,
            32141,
            32145,
            32140,
            32137,
            32125,
            32126,
            40760,
            42150,
            42104,
            40003,
            40007,
            30234,
            30127,
            30170,
            30174,
            30218,
            30222,
            30272,
            32138,
            32142,
            32171,
            32175,
            40001,
            40002,
            40004,
            40005,
            40006,
            40025,
            40030,
            40031,
            40033,
            40171,
            42106,
            42109,
            42156,
            42158]

    all_addresses = get_address()
    addr_to_use = np.setdiff1d(all_addresses, complete)
    print('stop')
    for a in addr_to_use:
        t0 = time.time()
        print(f"------------\nAddress Working on: {a}")

        # Importing data
        data = import_data(a)

        if data.empty:
            print("No Data Queried")

        else:
            t2 = time.time()
            print("Data Imported in time {:.2f}".format(t2 - t0))

            # Add data to the db
            lib_aws.AddData.add_data(df=data,
                                     db='esp-data',
                                     table='data',
                                     merge_type='append',  # Only use replace if you know what you are doing
                                     index_col='NodeID')

            t3 = time.time()
            print("Full process time {:.2f}".format(t3 - t0))
            complete.append(a)

    print(complete)


if __name__ == '__main__':
    main()
