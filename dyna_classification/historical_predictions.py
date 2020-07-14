"""
Makes predictions on the entire dataset
Using the model stored in the s3 bucket
"""
from library import lib_aws, lib_dyna
import pandas as pd
import numpy as np

def main():
    # Import the data
    query = """select "NodeID", "Date", pocdowncard from xspoc.card order by "NodeID", "Date";"""
    with lib_aws.PostgresRDS(db='oasis-dev') as engine:
        data = pd.read_sql(query, engine, parse_dates=['Date'])

    # Transform the data
    fea = lib_dyna.Features(df=data,
                            card_col='pocdowncard',
                            well_col='NodeID')
    fea.remove_errors()
    full_bounds = fea.all_bounds()
    X = fea.get_X(fd_order=5, area=True, centroid=True, normalize_fn='df', norm_arg=full_bounds)

    # Get the trained Model
    s3 = lib_aws.S3(bucket='et-oasis')
    model, mlb = s3.import_model('algo/rfcDynaClassification.pkl')  # Importing the model from an s3 bucket

    # Make Predictions
    pred = lib_dyna.Predictions_MultiLabel(model=model,
                                           x=X,
                                           mlb=mlb)
    pred_data = pred.get_pred_df(n=2)  # Get 2 labels as predictions
    pred_data.loc[pred_data.Prob2 <= 35, 'Label2'] = np.nan
    full_pred_data = pd.concat([fea.df[['NodeID', 'Date']], pred_data], axis=1)  # Get the prediction DataFrame

    # Replace the data in the db
    lib_aws.AddData.add_data(df=full_pred_data, db='oasis-dev', table='dynapred', schema='clean',
                             merge_type='replace', card_col=None, index_col='Date')

    # Replace the full bounds df
    lib_aws.AddData.add_data(df=full_bounds, db='oasis-dev', table='dynabounds', schema='clean',
                             merge_type='replace', card_col=None, index_col=None)

    # Update index on pred table in database
    with lib_aws.PostgresRDS(db='oasis-dev') as engine:
        with engine.begin() as connection:
            connection.execute("""CREATE UNIQUE INDEX dynapred_idx ON clean.dynapred ("NodeID", "Date");""")

    return None

if __name__ == '__main__':
    main()
