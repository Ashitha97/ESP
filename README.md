# OASIS DEVELOPMENT

Development repo for Oasis. 

## Project Structure

Broadly the project is divided into the following sub-categories:

- `Initial Analysis` : Basic Analysis of the tables and data provided by the client.
- `Failure Forecasting`: Notebooks and scripts for predicting failures.
- `Dynamometer Classification`: Classifying dynamometer cards
- `Dynamometer Forecasting`: Forecasting dynamometer card shapes.
- `Data Transfer`: Real time data transfer techniques and scripts.


In addition, the `library` folder contains python scripts which act as helper functions throughout the project. 
Of these scripts the ones of note are:
- `lib_aws`: Classes and methods to interact with AWS services, most notably RDS (Postgres Server) and S3
- `lib_dyna`: Helps in working with dynamometer card data.


## Setup

### RDS Postgres Server

Before running the scripts few environment variables need to be added. These variables will be needed for 
connecting to the postgres database. Following are the environment variables;

```
OASIS_USER
OASIS_PASS
OASIS_HOST
```
 
*Note: For the variable values please contact the project admin*

**Resources:**
- [Setting Env Variables in Windows](https://www.youtube.com/watch?v=IolxqkL7cD8&list=LLLuzKtlkPVRLC83uTqb8suw&index=2&t=219s)
- [Setting Env Variables in MacOS/Linux](https://www.youtube.com/watch?v=5iWhQWVXosU) 


### S3

To interact with s3, `boto3` and `s3fs` libraries will be used. 

In addition the `~/.aws/config` files needs to be set up with the IAM access details.
 

## Database Info  (*Update required)

For the project an PostgreSQL Database Server is being used. The structure of the database is as follows:

//TODO NEED TO UPDATE THIS
```
RDS PostgreSQL Server
|                                                
├── oasis-dev                               # DataBase name
│   ├── clean                               # Schema with processed data from the xspoc schema
│   │   ├── tables                          
│   |   |   ├── dynabounds                  # Well Bounds for pocdowncard
│   |   |   ├── dynalabel                   # Labeled data for dyna classification
│   |   |   ├── dynapred                    # Predictions from dyna classification model
│   |   |   ├── xpred                       # Predictions from window forecasting
│   |   |   └── xspoc                       # Clean and Merged xdiagresults and card tables from xspoc schema
│   │   ├── views                           
│   |   |   └── winpred                     # View generated from xpred and xspoc to visualize window forecasting

│   ├── xspoc                               # Schema with all tables from oasis converted from MS-SQL Server
│   │   ├── tables                          
│   |   |   ├── card                        # Full Card Data
│   |   |   ├── xdiagresults                
│   |   |   ├── xdiagrodresults 
.   .   .   .                               # Other tables havent yet been used
.   .   .   .
│   |   |   ├── ...                       
│   |   |   └── ...                    
│   │   ├── views                         
│   |   |   ├── card_xdiag                  # 
│   |   |   └── merged                      # 


```