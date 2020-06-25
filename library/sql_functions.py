"""
Script contains some basic sql based function's
"""
import pandas as pd
from io import StringIO
import csv
import time
from geoalchemy2 import Geometry
from library.dbconnection import PostgresRDS


def run_query(sql, engine):
    """
    Runs SQL queries on the database connected via engine
    :param sql: Sql Query
    :param engine: sql-alchemy engine
    :return: A DataFrame with the results
    """
    data = pd.read_sql(sql, con=engine)
    return data


class AddData:
    """
    Class which has methods to add data into a postgres db
    """

    @staticmethod
    def psql_insert_copy(table, conn, keys, data_iter):
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        table : pandas.io.sql.SQLTable
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
            Column names
        data_iter : Iterable that iterates the values to be inserted
        """
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ', '.join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = '{}.{}'.format(table.schema, table.name)
            else:
                table_name = table.name

            sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
                table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)

    @staticmethod
    def add_data(df, db, table, schema=None, merge_type='append', card_col=None, index_col=None):
        """
        Method to add data to a postgres db
        :param df: Data in the form of a pandas DataFrame
        :param db: Database Name (str)
        :param table: Table Name (str)
        :param merge_type: How to add data. Either 'append' or 'replace'. Default: 'append'
        :param card_col: If Card Columns are present. Default('None')
        :param index_col: If an index column is needed. Default('None')
        :return:
        """
        t0 = time.time()
        if card_col is not None:
            dtype_dict = {i: Geometry("POLYGON") for i in card_col}
        else:
            dtype_dict = None

        if index_col is not None:
            try:
                df.set_index(index_col, inplace=True)
            except KeyError:  # Index Column is already set
                pass

        with PostgresRDS(db=db) as engine:
            try:
                df.to_sql(table, con=engine, schema=schema,if_exists=merge_type, method=AddData.psql_insert_copy,
                          dtype=dtype_dict)
            except Exception as e:
                print(e)
                return print("Data Not Added")

        t1 = time.time()
        return print("Data {}ed on Table {} in time {:.2f}s".format(merge_type, table, t1 - t0))
