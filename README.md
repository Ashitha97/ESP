# OASIS DEVELOPMENT

Data Analysis and product development for Oasis Petroleum

## Setup

Before running the scripts few environment variables need to be added. These variables will be needed for 
connecting to the postgres database. Following are the environment variables;

- OASIS_USER
- OASIS_PASS
- OASIS_HOST

**Resources:**

- [Setting Env Variables in Windows](https://www.youtube.com/watch?v=IolxqkL7cD8&list=LLLuzKtlkPVRLC83uTqb8suw&index=2&t=219s)
- [Setting Env Variables in MacOS/Linux](https://www.youtube.com/watch?v=5iWhQWVXosU) 

*For the variable values please contact the project admin*

## Database Info

For the project an PostgreSQL Database is being. The structure of the database is as follows:

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
.   .   .   .                               # Other tables havent yet been used
.   .   .   .
│   |   |   ├── ...                       
│   |   |   └── ...                    
│   │   ├── views                         
│   |   |   ├── card_xdiag                  # 
│   |   |   └── merged                      # 


```