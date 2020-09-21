# Problem Statement and Overview



# Data

The database set up and connecting to it has been discussed in the main README file.

**Dataset for Features**
The main table that is being used for training the model is `xspoc.xdiag`. From this we are using the following features
```
NodeID : Well-Names
Date: Timestamp
PPRL: 
MPRL: 
FluidLoadonPump: 
PumpintakePressure: 
```

**Dataset for Failures**
For information regarding failures (The Labels) we are using `analysis.failure_info`.
// TODO: Go more in-depth about this

In general this table will contain the following columns (Names may change):
```
NodeID
Failure Start Date
Failure End Date
Failure
```



# Helper Functions
In the root directory of this repository the folder `library` container scripts which will be used throughout this project 
for various tasks. A brief description of these helper scripts are as follows:

- `library.lib_aws.py`:

This script contains a few classes which help interact with AWS services:
```
Class S3: This class `pickles` and saves trained models to an s3 bucket.
Class PostgresRDS: Run queries on a Postgres DB. Mostly used for querying data.
Class AddData: Add data into a Postgres DB.
```

- `library.lib_metrics`: 

The `class MultiClassMetrics` will be the one to get metrics for a specific ML model. 


# Algo Idea 1:

From the problem statement we need to start predicting a specific failure `x days` in advance. For now lets consider it to be 
`15 days`. The features and labels are then transformed as follows:

- The window of `15 days` from where the failure occurs is also marked as that specific failure. 
- For example `well a` fails due to `Label 1` on `Aug 17th 2020`.
- Then the labels for data-points from `Aug 2nd 2020` to `Aug 17th 2020`  is also labeled as `Fail Label 1`.
- This is done for all the failure zones that we have.

The failure information is present in `analysis.failure_info` in our database. In addition the notebook `Window Forecasting Notebook.ipynb`
has a method `create_prediction_zones(*args)` which will do that.

Once the labels have been transformed in this fashion. We can imagine this to be a classification problem. Whenever this classifier outs a 
specific class (Label 1 for example) we can assume that the specific well will fail in the next `15 days` due to `Label 1`. 

This algo has been implemented in `Window Forecasting Notebook.ipynb`. 

**TODO: Tasks to work on:**

- [ ] Check Different Feature Engg methods (For now using a rolling window of `7 days`)
- [ ] Try Different ML Algorithms 

**Some Issues With this Methodology**



# Algo Idea 2

- Using LSTM based window forecasting. 
- Idea still needs to be fleshed out.
- May face issues with NAN values


 