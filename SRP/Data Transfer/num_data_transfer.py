"""
Transfers Numerical data from main db to dev db for
specific wells
"""
from library import lib_aws
import pandas as pd

def query_gen(table_name, well_list=None):
    """
    Generates a query for importing the full data
    from a specific table.
    If well list is None, all rows are imported using this query
    :param table_name: Singular table name (str)
    :param well_list: List of wells (Default: None)
    :return: returns the query
    """
    if well_list is None:
        query = """
        SELECT * FROM xspoc_dbo."{}"
        """.format(table_name)

        return query

    else:
        query = """
        SELECT * FROM xspoc_dbo."{}"
        WHERE "NodeID" IN {}
        ORDER BY "NodeID"
        """.format(table_name, tuple(well_list))

        return query


def main():
    # Basic parameters
    source_db = 'oasis-data'
    end_db = 'oasis-dev'

    well_list = [
        'Bonner 9-12H',
        'Bonner 9X-12HA'
        'Bonner 9X-12HB',
        'Cade 12-19HA',
        'Cade 12-19HB',
        'Cade 12X-19H',
        'Cook 12-13 6B',
        'Cook 12-13 7T',
        'Cook 12-13 9T',
        'Cook 41-12 11T',
        'Hanover Federal 5300 41-11 10B',
        'Hanover Federal 5300 41-11 11T',
        'Hanover Federal 5300 41-11 12B',
        'Hanover Federal 5300 41-11 13TX',
        'Helling Trust 43-22 10T',
        'Helling Trust 43-22 16T3',
        'Helling Trust 43-22 4B',
        'Helling Trust 44-22 5B',
        'Helling Trust 44-22 6B',
        'Helling Trust 44-22 7B',
        'Johnsrud 5198 14-18 11T',
        'Johnsrud 5198 14-18 13T',
        'Johnsrud 5198 14-18 15TX',
        'Lite 5393 31-11 9B',
        'Lite 5393 41-11 11B',
        'Lite 5393 41-11 12T',
        'Rolfson N 5198 12-17 5T',
        'Rolfson N 5198 12-17 7T',
        'Rolfson S 5198 11-29 2TX',
        'Rolfson S 5198 11-29 4T',
        'Rolfson S 5198 12-29 6T',
        'Rolfson S 5198 12-29 8T',
        'Rolfson S 5198 14-29 11T',
        'Rolfson S 5198 14-29 13T',
        'Stenehjem 14-9H',
        'Spratley 5494 14-13 12B',
        'Spratley 5494 14-13 13T',
        'Spratley 5494 14-13 15T',
        'Stenehjem 14X-9HA',
        'Stenehjem 14X-9HB',
        'Stenehjem 15-9HA',
        'Stenehjem 15-9HB',
        'Stenehjem 15X-9H'
    ]
    tables_w_node = [
        'tblRods',
        'tblXDiagRodResults',
         'tblTubings',
         'tblXDiagResults',
         'tblCasings',
         'tblDeviations',
         'tblWellTests',
         'tblXDiagResultsLast',
         'tblIPRAnalysisResults',
         'tblWellTest',
         'tblXDiagFlags',
         'tblProductionStatistics',
         'tblXDiagScores',
         'tblPerforations',
         'tblWellDetails',
         'tblSavedParameters'
    ]
    tables_wo_node = [
        'tblCasingSizes',
        'tblPOCTypeActions',
        'tblPOCTypes',
        'tblPUData',
        'tblParamStandardTypes',
        'tblParameters',
        'tblPumpingUnits',
        'tblRodGrades',
        'tblRodGuides',
        'tblRodMaterials',
        'tblRodSizeGroups',
        'tblRodSizes',
        'tblSAMPumpingUnits',
        'tblSetpointGroups',
        'tblSetpointOptimization',
        'tblSetpointOptimizationAdvisories',
        'tblStates',
        'tblTubingSizes',
        'tblWVRods',
        'tblWellFailureCodes'
    ]

    # Transferring with NodeID
    for i in tables_w_node:
        query1 = query_gen(i, well_list)
        print("Working on Table {}".format(i))
        with lib_aws.PostgresRDS(db=source_db) as engine:
            data = pd.read_sql(query1, engine)

        print("Data Imported with shape {}".format(data.shape))

        lib_aws.AddData.add_data(data, db=end_db, schema='xspoc', table=i[3:].lower(),
                                       merge_type='replace', card_col=None, index_col="NodeID")

        del query1
        del data
        print('----------------')

    # Transferring W/O NodeID
    for i in tables_wo_node:
        query1 = query_gen(i)
        print("Working on Table {}".format(i))
        with lib_aws.PostgresRDS(db=source_db) as engine:
            data = pd.read_sql(query1, engine)

        data.set_index(data.columns[0], inplace=True)

        print("Data Imported with shape {}".format(data.shape))

        lib_aws.AddData.add_data(data, db=end_db, schema='xspoc', table=i[3:].lower(),
                                       merge_type='replace', card_col=None, index_col=None)

        del query1
        del data
        print('----------------')

    return None

if __name__ == '__main__':
    main()