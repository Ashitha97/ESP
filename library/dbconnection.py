"""
Contains Class which will connect to a postgres DB
Note: This is for an RDS Connection with a password
"""
import sqlalchemy
from sqlalchemy.orm import Session
from config import username, password, endpoint

# Configurations
db = 'oasis-data'

class PostgresRDS(object):
    """
    Class Connects to an RDS instance of Postgresql
    """

    def __init__(self):
        self.engine = None
        self.Session = None

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
                db=db
            ),
            'sqlalchemy.pool_pre_ping': True,
            'sqlalchemy.pool_recycle': 3600
        }

        engine = sqlalchemy.engine_from_config(engine_config, prefix='sqlalchemy.')
        self.Session = Session(engine)

        return engine

    def __enter__(self):
        self.engine = self.connect()
        print("Connected to {} DataBase".format(db))
        return self.engine

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Session.close()
        self.engine.dispose()
        print("Connection Closed")
