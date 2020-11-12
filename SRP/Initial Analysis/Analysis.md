# Analysis Overview

## Card Data

Card data is stored in table `tblCarddata`. Once Env Variables have been setup, connection to the database can be established
using the class `PostgresRDS` in file `/dbconnection/postgresql.py`. Use the jupyter notebook `Initial Analysis Card Data.ipynb`.

### Query Examples

To run basic queries use the method `run_query(query, con)` as follows;
```python
from library.lib_aws import PostgresRDS  # Get the connection class
from library.sqlFunctions import run_query

sql_query = """
SELECT DISTINCT("NodeID") 
FROM xspoc_dbo."tblCardData"
ORDER BY "NodeID"
"""

with PostgresRDS() as engine:
    result = run_query(sql_query, engine)  # Result will be in the form of a pandas dataframe
```
The above query will give a list of wells present in the table. Some other examples are given below

<u>**Get Card Data**</u>

To get any sort of Card data (bytea dtype) use the following mapping `encode("tblCardData"."CardColumn", 'hex') as cardcolumn`.

An Example for importing `DownholeCardB` and `SurfaceCardB` from well `K2 Holdings` will be:

```
SELECT 
"NodeID",
"Date",
encode("tblCardData"."DownholeCardB", 'hex') as downcard,
encode("tblCardData"."SurfaceCardB", 'hex') as surfcard
FROM xspoc_dbo."tblCardData"
WHERE "NodeID" = 'K2 Holdings'
ORDER BY "NodeID" , "Date" 
```
