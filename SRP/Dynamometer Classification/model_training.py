"""
Trains a classification model for dynamometer cards
This is a Multi-label model

Functionality to add:
- To show model metrics (Maybe Use argparse to make it conditional)
- Append these metrics to a log file
- Get bounds from db than from the Features class
"""

import pandas as pd

# ML
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
# Local
from library import lib_aws, lib_dyna

def train_rfc(x, y):
    """
    Fits a RFC model Wrapped with a OneVRest Classifier
    :param x: Feature Matrix
    :param y: Label Matrix
    :return: Trained Model
    """
    rfc_params = {
        'n_estimators': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'max_depth': None,
        'class_weight': 'balanced',
        'bootstrap': False
    }

    rfc = RandomForestClassifier(**rfc_params)
    model = OneVsRestClassifier(rfc)
    print("Training the model")
    model.fit(x, y)
    print("Training done")

    return model

def main():
    # Import labeled data from db
    query = """SELECT * FROM clean.dynalabel ORDER BY "NodeID", "Date";"""
    query_bounds = """SELECT * FROM clean.dynabounds;"""

    with lib_aws.PostgresRDS(db='oasis-dev') as engine:
        data = pd.read_sql(query, engine, parse_dates=['Date'])
        bounds_df = pd.read_sql(query_bounds, engine)
    bounds_df.set_index('index', inplace=True)

    # Use the Features Class (library.dynaFunc.Features)to process the data and get X, Y
    fea = lib_dyna.Features(df=data,
                   card_col='pocdowncard',
                   well_col='NodeID',
                   label_cols=['TrueLabel1', 'TrueLabel2'])
    fea.remove_errors()  # will remove errors
    fea.merge_labels()  # Merge Multi-Labels
    # fea.remove_labels(thresh=0.1) # Remove Labels below  a threshold

    X = fea.get_X(fd_order=5, area=True, centroid=True, normalize_fn='df', norm_arg=bounds_df)
    Y, binarizer = fea.get_Y()  # Get y and binarizer


    # Train and Fit the model
    trained_model = train_rfc(X, Y)
    # Saving the model
    model_name = "algo/rfcDynaClassification.pkl"

    # Setup S3 Connection
    s3 = lib_aws.S3(bucket='et-oasis')  # Bucket being used
    s3.save_model(obj=(trained_model, binarizer), name=model_name)


if __name__ == '__main__':
    main()

