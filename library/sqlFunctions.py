"""
Script contains some basic sql based funcitons
"""
import pandas as pd

def run_query(sql, engine):
    """
    Runs SQL queries on the database connected via engine
    :param sql: Sql Query
    :param engine: sql-alchemy engine
    :return: A DataFrame with the results
    """
    data = pd.read_sql(sql, con=engine)
    return data

