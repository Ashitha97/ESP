"""
Class to connect to Databases
"""

import sqlalchemy
from sqlalchemy.orm import Session
from config import username, password, endpoint
import boto3
import joblib
import tempfile
from io import StringIO
import csv
import time
from geoalchemy2 import Geometry


# Functions to save and read models from the s3 bucket
class S3:
    """
    Class with methods to r/w data from an s3 bucket
    Should have access to the bucket being used.
    Set up the credentials and config files in .aws/
    """

    def __init__(self, bucket):
        """
        Initialize
        :param bucket: S3 Bucket name
        """
        self.bucket = bucket

    def save_model(self, obj, name):
        """
        Saves the model to an s3 bucket.
        Use the variables 'bucket_name' and 'location' to specify the path
        :param obj: Objects to be dumped
        :param name: Name of the file, Add the path if need be (/path/filename.pkl)
        :return: Log
        """
        s3 = boto3.resource('s3')
        with tempfile.TemporaryFile() as fp:
            joblib.dump(obj, fp)
            fp.seek(0)
            s3.Bucket(self.bucket).put_object(Key=name, Body=fp.read())
        print("Model Updated successfully")

    def import_model(self, name):
        """
        Imports the model from s3
        Specify the bucket and the location in config variables
        'bucket_name'
        :param name: Name of the model to import, include the pathname if needed('/path/filename.pkl')
        :param s3_bucket: s3 Bucket name.(Should hve access to the bucket, set up config and credentials in .aws/)
        :return: model objs
        """
        s3 = boto3.resource('s3')
        with tempfile.TemporaryFile() as fp:
            s3.Bucket(self.bucket).download_fileobj(Key=name, Fileobj=fp)
            fp.seek(0)
            model_objs = joblib.load(fp)

        return model_objs


# DataBase Classes
class PostgresRDS(object):
    """
    Class Connects to a PostgreSQL DB with password access
    Need to input the database that needs to be connected to
    Note Set the username, password and endpoint in the config file via env variables
    """

    def __init__(self, db):
        self.engine = None
        self.Session = None
        self.db = db

    def connect(self):
        """
        Connects to the db and gives us the engine
        :return: engine
        """
        engine_config = {
            'sqlalchemy.url': 'postgresql+psycopg2://{user}:{pw}@{host}/{db}'.format(
                user=username,
                pw=password,
                host=endpoint,
                db=self.db
            ),
            'sqlalchemy.pool_pre_ping': True,
            'sqlalchemy.pool_recycle': 3600
        }

        engine = sqlalchemy.engine_from_config(engine_config, prefix='sqlalchemy.')
        self.Session = Session(engine)

        return engine

    def __enter__(self):
        self.engine = self.connect()
        print("Connected to {} DataBase".format(self.db))
        return self.engine

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Session.close()
        self.engine.dispose()
        print("Connection Closed")


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
        :param schema: Schema Name (Default is None, in this case will add to the public schema)
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
                df.to_sql(table, con=engine, schema=schema, if_exists=merge_type, method=AddData.psql_insert_copy,
                          dtype=dtype_dict)
            except Exception as e:
                print(e)
                return print("Data Not Added")

        t1 = time.time()
        return print("Data {}ed on Table {} in time {:.2f}s".format(merge_type, table, t1 - t0))
