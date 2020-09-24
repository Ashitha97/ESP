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
The primary use case will be for saving trained model and for importing data files. 
Before interacting with `s3` the following files need to be set up with your IAM Access details:
- `~/.aws/credentials`
- `~/.aws/config`

Get the acess details from the project admin. Use the following link to see how to set it up: 
- [AWS Credentials setup](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)


## Database Info  (*Update required)

For the project an PostgreSQL Database Server is being used. The structure of the database is as follows:

//TODO NEED TO UPDATE THIS
```
RDS PostgreSQL Server
|                                                
├── oasis-prod                              # DataBase name
│   ├── analysis                            # Schema which contains info for various analysis run
│   │   ├── tables                          
│   |   |   ├── failure_info                # failure_info for failure forecasting
│   |   |   └── dyna_labels                 # labeled dynamometer cards

│   ├── xspoc                               # Schema with all tables from oasis converted from MS-SQL Server
│   │   ├── tables                          
│   |   |   ├── card                        # Dynamometer card data
│   |   |   ├── only_fails                  # Just the failure info, used only for visualization
│   |   |   ├── sample_predictions          # Preditions for failure_forecasts , will be moved
│   |   |   ├── well_info                   # List of wells and groupings                       
│   |   |   ├── welltest                    # Info about production                       
│   |   |   ├── xdiag                       # Main feature dataset                       
│   |   |   └── xdiagrod                    # Rod Specific info


```