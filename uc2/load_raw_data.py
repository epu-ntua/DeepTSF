"""
Downloads the RDN dataset and saves it as an artifact. ALso need to include interaction with weather apis here.
"""
import requests
import tempfile
import os
import mlflow
import click
import sys
sys.path.append('..')
from pandas.tseries.frequencies import to_offset
from utils import ConfigParser
import logging
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from utils import download_online_file, multiple_ts_file_to_dfs, multiple_dfs_to_ts_file, allow_empty_series_fun, to_seconds, to_standard_form
import shutil
import pretty_errors
import uuid
from exceptions import WrongIDs, EmptyDataframe, DifferentComponentDimensions, WrongColumnNames, DatetimesNotInOrder, WrongIndexFormat
from utils import truth_checker, none_checker
import tempfile
from math import ceil
# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

MONGO_URL = os.environ.get("MONGO_URL")

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)



def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: bool = True,
                            multiple: bool = False,
                            from_database: bool = False,
                            covariates: str = "series",
                            allow_empty_series=False,
                            format="long",
                            log_to_mlflow=True):
    """
    Validates the input after read_csv is called and throws apropriate exception if it detects an error.
    
    The checks that are performed are the following:

    For all timeseries:
        - The dataframe can not be empty
        - All the dates must be sorted

    For non multiple timeseries:
        - Column Datetime must be used as index 
        - If the timeseries is the main dataset, Load must be the only other column in the dataframe
        - If the timeseries is a covariates timeseries, there must be only one column in the dataframe
          named arbitrarily

    For multiple timeseries:
        - Columns 'Date', 'ID', and time columns exist in any order
        - Only the permitted column names exist in the dataframe (see Multiple timeseries file format bellow)
        - All timeseries in the dataframe have the same number of components

    For all timeseries, their resolution is also infered, and stored in mlflow as a tag. Furthermore, we support 2
    formats for multiple time series files; short and long. These formats are presented bellow:

    Multiple timeseries file format (along with example values):

    Long format:

    Index |Datetime            | ID | Timeseries ID | Value 
    0     |2015-04-09 00:00:00 | PT | PT            | 5248  
    1     |2015-04-09 00:00:00 | ES | ES            | 25497
    .
    .

    Short format:

    Index | Date         | ID | Timeseries ID | 00:00:00 | 00:00:00 + resolution | ... | 24:00:00 - resolution
    0     | 2015-04-09   | PT | PT            | 5248     | 5109                  | ... | 5345
    1     | 2015-04-09   | ES | ES            | 25497    | 23492                 | ... | 25487
    .
    .
The columns that can be present in the short format csv have the following meaning:
        - Index: Simply a monotonic integer range
        - Date: The Date each row is referring to
        - ID: Each ID corresponds to a component of a timeseries in the file. 
              This ID must be unique for each timeseries component in the file.
        - Timeseries ID (Optional): Timeseries ID column is not compulsory, and shows the 
              timeseries to which each component belongs. If Timeseries ID is not present, it is 
              assumed that each component represents one separate series (the column is set to ID).
        - Time columns: Columns that store the Value of each component.

    The columns that can be present in the long format csv have the following meaning:
        - Index: Simply a monotonic integer range
        - Datetime: The Datetime each value is referring to
        - ID: Each ID corresponds to a component of a timeseries in the file. 
              This ID must be unique for each timeseries component in the file.
        - Timeseries ID (Optional): Timeseries ID column is not compulsory, and shows the 
              timeseries to which each component belongs. If Timeseries ID is not present, it is 
              assumed that each component represents one seperate series (the column is set to ID).
        - Value: The value of this component in a particular Datetime.

        
    Columns can be in any order and the file will be considered valid.

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
    day_first
        Whether to read the csv assuming day comes before the month
    multiple
        Whether to train on multiple timeseries
    from_database
        Whether the dataset was from MongoDB
    covariates
        If the function is called for the main dataset, then this equal to "series".
        If it is called for the past / future covariate series, then it is equal to
        "past" / "future" respectively. 

    Returns
    -------
    (pandas.DataFrame, int)
        A tuple consisting of the resulting dataframe from series_csv as well as the resolution
    """
    if format == "long":
        ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=(["Datetime"] if multiple else True),
                     dayfirst=day_first,
                     engine='python', 
                     date_format='mixed')
        if multiple:
            ts["Datetime"] = pd.to_datetime(ts["Datetime"])
        else:
            ts.index = pd.to_datetime(ts.index)
    else:
        ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=(["Date"] if multiple else True),
                     dayfirst=day_first,
                     engine='python',
                     date_format='mixed')
        if multiple:
            ts["Date"] = pd.to_datetime(ts["Date"])
        else:
            ts.index = pd.to_datetime(ts.index)

    #Dataframe can not be empty
    if ts.empty:
        raise EmptyDataframe(from_database)
        
    if not multiple:
        #Check that dates are in order. If month is used before day and day_first is set to True, this is not the case.
        if not ts.index.sort_values().equals(ts.index):
            raise DatetimesNotInOrder()
        if type(ts.index[0]) != pd.Timestamp:
            raise WrongIndexFormat()
        #Check that column Datetime is used as index, and that Load is the only other column in the csv for the series csv
        elif covariates == "series" and not (len(ts.columns) == 1 and ts.columns[0] == "Value" and ts.index.name == 'Datetime'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Datetime', "Value"])
        #Check that column Datetime is used as index, and that there is only other column in the csv for the covariates csvs
        elif covariates != "series" and not (len(ts.columns) == 1 and ts.index.name == 'Datetime'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Datetime', '<Value Column Name>'])

        #Infering resolution for single timeseries
        resolution = to_standard_form(pd.to_timedelta(np.diff(ts.index).min()))

    else:
        #If columns don't exist set defaults
        if "Timeseries ID" not in ts.columns and "ID" in ts.columns:
            ts["Timeseries ID"] = ts["ID"]

        if format == "short":
            des_columns = list(map(str, ['Date', 'ID', 'Timeseries ID']))
            #Check that all columns 'Date', 'ID', 'Timeseries ID' and only time columns exist in any order.
            if not set(des_columns).issubset(set(list(ts.columns))) and any(not isinstance(e, pd.Timestamp) for e in (set(list(ts.columns))).difference(set(des_columns))):
                raise WrongColumnNames(list(ts.columns), len(des_columns) + 1, des_columns + ['and the rest should all be time columns'], "short")
            #Check that all dates for each component are sorted
            for ts_id in np.unique(ts["Timeseries ID"]):
                for id in np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"]):
                    if not ts.loc[(ts["ID"] == id) & (ts["Timeseries ID"] == ts_id)]["Date"].sort_values().equals(ts.loc[(ts["ID"] == id) & (ts["Timeseries ID"] == ts_id)]["Date"]):
                        raise DatetimesNotInOrder(id)
        else:
            des_columns = list(map(str, ['Datetime', 'ID', 'Timeseries ID', 'Value']))
            #Check that only columns 'Datetime', 'ID', 'Timeseries ID', 'Value' exist in any order.
            if not set(des_columns) == set(list(ts.columns)):
                raise WrongColumnNames(list(ts.columns), len(des_columns), des_columns, "long")
            #Check that all dates for each component are sorted
            for ts_id in np.unique(ts["Timeseries ID"]):
                for id in np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"]):
                    if not ts.loc[(ts["ID"] == id) & (ts["Timeseries ID"] == ts_id)]["Datetime"].sort_values().equals(ts.loc[(ts["ID"] == id) & (ts["Timeseries ID"] == ts_id)]["Datetime"]):
                        raise DatetimesNotInOrder(id)
                    
        #Check that all timeseries in a multiple timeseries file have the same number of components
        if len(set(len(np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"])) for ts_id in np.unique(ts["Timeseries ID"]))) != 1:
            raise DifferentComponentDimensions()
        
        #Infering resolution for multiple ts
        ts_l, id_l, ts_id_l, resolution = multiple_ts_file_to_dfs(series_csv, day_first, None, format=format)

        if allow_empty_series:
            ts_list_ret, id_l_ret, ts_id_l_ret = allow_empty_series_fun(ts_l, id_l, ts_id_l, allow_empty_series=allow_empty_series)
            ts = multiple_dfs_to_ts_file(ts_list_ret, id_l_ret, ts_id_l_ret, "", save=False, format=format)

    if log_to_mlflow:
        mlflow.set_tag(f'infered_resolution_{covariates}', resolution)
            
    return ts, resolution

def make_multiple(ts_covs, series_csv, day_first, inf_resolution, format):
    """
    In case covariates.

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
    day_first
        Whether to read the csv assuming day comes before the month
    multiple
        Whether to train on multiple timeseries
    resolution
        The resolution of the dataset
    from_database
        Whether the dataset was from MongoDB
    covariates
        If the function is called for the main dataset, then this equal to "series".
        If it is called for the past / future covariate series, then it is equal to
        "past" / "future" respectively. 

    Returns
    -------
    (pandas.DataFrame, int)
        A tuple consisting of the resulting dataframe from series_csv as well as the resolution
    """


    if series_csv != None:
        ts_list, _, _, _, ts_id_l = multiple_ts_file_to_dfs(series_csv, day_first, inf_resolution, format=format)

        ts_list_covs = [[ts_covs] for _ in range(len(ts_list))]
        id_l_covs = [[str(list(ts_covs.columns)[0]) + "_" + ts_id_l[i]] for i in range(len(ts_list))]
    else:
        ts_list_covs = [[ts_covs]]
        id_l_covs = [[list(ts_covs.columns)[0]]]
    return multiple_dfs_to_ts_file(ts_list_covs, id_l_covs, id_l_covs, id_l_covs, id_l_covs, "", save=False, format=format)

from pymongo import MongoClient
import pandas as pd

client = MongoClient(MONGO_URL)


def unfold_timeseries(lds):
    new_loads = {'Date': [], "Value": []}
    prev_date = ''
    for l in reversed(list(lds)):
        if prev_date != l['date']:
            for key in l:
                if key != '_id' and key != 'date':
                    new_date = l['date'] + ' ' + key
                    new_loads['Date'].append(new_date)
                    new_loads["Value"].append(l[key])
        prev_date = l['date']
    return new_loads


def load_data_to_csv(tmpdir, database_name):
    db = client['inergy_prod_db']
    collection = db[database_name]
    df = pd.DataFrame(collection.find()).drop(columns={'_id', ''}, errors='ignore')
    df = unfold_timeseries(collection.find().sort('_id', -1))
    df = pd.DataFrame.from_dict(df)
    df.to_csv(f'{tmpdir}/load.csv', index=False)
    client.close()
    return

@click.command(
    help="Downloads the time series and saves it as an mlflow artifact. Also runs some validation checks."
    )
@click.option("--series-csv",
    type=str,
    default="None",
    help="Local time series csv file"
    )
@click.option("--series-uri",
    default="None",
    help="Remote time series csv file. If set, it overwrites the local value."
    )
@click.option("--past-covs-csv",
    type=str,
    default="None",
    help="Local past covaraites csv file"
    )
@click.option("--past-covs-uri",
    default="None",
    help="Remote past covariates csv file. If set, it overwrites the local value."
    )
@click.option("--future-covs-csv",
    type=str,
    default="None",
    help="Local future covaraites csv file"
    )
@click.option("--future-covs-uri",
    default="None",
    help="Remote future covariates csv file. If set, it overwrites the local value."
    )
@click.option("--day-first",
    type=str,
    default="true",
    help="Whether the date has the day before the month")
@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")
@click.option("--resolution",
    default="None",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--from-database",
    default="false",
    type=str,
    help="Whether to read the dataset from the database."
)
@click.option("--database-name",
    default="rdn_load_data",
    type=str,
    help="Which database file to read."
)
@click.option("--format",
    default="long",
    type=str,
    help="Which file format to use. Only for multiple time series"
)

def load_raw_data(series_csv, series_uri, past_covs_csv, past_covs_uri, future_covs_csv, future_covs_uri, day_first, multiple, resolution, from_database, database_name, format):
    from_database = truth_checker(from_database)
    tmpdir = tempfile.mkdtemp()

    past_covs_csv = none_checker(past_covs_csv)
    past_covs_uri = none_checker(past_covs_uri)
    future_covs_csv = none_checker(future_covs_csv)
    future_covs_uri = none_checker(future_covs_uri)

    if from_database:
        load_data_to_csv(tmpdir, database_name)
        series_csv = f'{tmpdir}/load.csv'

    elif series_uri != "None":
        download_file_path = download_online_file(series_uri, dst_filename="series.csv")
        series_csv = download_file_path

    if past_covs_uri != None:
        download_file_path = download_online_file(past_covs_uri, dst_filename="past_covs.csv")
        past_covs_csv = download_file_path

    if future_covs_uri != None:
        download_file_path = download_online_file(future_covs_uri, dst_filename="future_covs.csv")
        future_covs_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]

    day_first = truth_checker(day_first)

    multiple = truth_checker(multiple)

    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:

        ts, _ = read_and_validate_input(series_csv, day_first, multiple=multiple, from_database=from_database, format=format)

        print(f'Validating timeseries on local file: {series_csv}')
        logging.info(f'Validating timeseries on local file: {series_csv}')

        local_path = local_path.replace("'", "") if "'" in local_path else local_path
        series_filename = os.path.join(*local_path, fname)
        # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
        # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
        print(f'\nUploading timeseries to MLflow server: {series_filename}')
        logging.info(f'\nUploading timeseries to MLflow server: {series_filename}')

        ts_filename = os.path.join(tmpdir, fname)
        ts.to_csv(ts_filename, index=True)
        mlflow.log_artifact(ts_filename, "raw_data")

        if past_covs_csv != None:
            past_covs_csv = past_covs_csv.replace('/', os.path.sep)
            past_covs_fname = past_covs_csv.split(os.path.sep)[-1]
            local_path_past_covs = past_covs_csv.split(os.path.sep)[:-1]

            #try:
            ts_past_covs, _ = read_and_validate_input(past_covs_csv,
                                                              day_first,
                                                              multiple=True,
                                                              from_database=from_database,
                                                              covariates="past",
                                                              format=format)
            # except:
            #     ts_past_covs, inf_resolution = read_and_validate_input(past_covs_csv,
            #                                                                day_first,
            #                                                                multiple=False,
            #                                                                from_database=from_database,
            #                                                                covariates="past")
            #     ts_past_covs = make_multiple(ts_past_covs,
            #                                      series_csv,
            #                                      day_first,
            #                                      str(inf_resolution))
                
            local_path_past_covs = local_path_past_covs.replace("'", "") if "'" in local_path_past_covs else local_path_past_covs
            past_covs_filename = os.path.join(*local_path_past_covs, past_covs_fname)

            print(f'\nUploading past covariates timeseries to MLflow server: {past_covs_filename}')
            logging.info(f'\nUploading past covariates timeseries to MLflow server: {past_covs_filename}')


            ts_past_covs_filename = os.path.join(tmpdir, past_covs_fname)
            ts_past_covs.to_csv(ts_past_covs_filename, index=True)
            mlflow.log_artifact(ts_past_covs_filename, "past_covs_data")
            mlflow.set_tag('past_covs_uri', f'{mlrun.info.artifact_uri}/past_covs_data/{past_covs_fname}')
        else:
            mlflow.set_tag(f'infered_resolution_past', "None")
            mlflow.set_tag('past_covs_uri', "None")

        if future_covs_csv != None:
            future_covs_csv = future_covs_csv.replace('/', os.path.sep)
            future_covs_fname = future_covs_csv.split(os.path.sep)[-1]
            local_path_future_covs = future_covs_csv.split(os.path.sep)[:-1]

            try:
                ts_future_covs, _ = read_and_validate_input(future_covs_csv,
                                                              day_first,
                                                              multiple=True,
                                                              from_database=from_database,
                                                              covariates="future",
                                                              format=format)
            #TODO Catch this exception more robustly
            except:
                ts_future_covs, inf_resolution = read_and_validate_input(future_covs_csv,
                                                                           day_first,
                                                                           multiple=False,
                                                                           from_database=from_database,
                                                                           covariates="future",
                                                                           format=format)
                ts_future_covs = make_multiple(ts_future_covs,
                                                 series_csv,
                                                 day_first,
                                                 inf_resolution,
                                                 format=format)
                                    
            local_path_future_covs = local_path_future_covs.replace("'", "") if "'" in local_path_future_covs else local_path_future_covs
            future_covs_filename = os.path.join(*local_path_future_covs, future_covs_fname)

            print(f'\nUploading future covariates timeseries to MLflow server: {future_covs_filename}')
            logging.info(f'\nUploading future covariates timeseries to MLflow server: {future_covs_filename}')


            ts_future_covs_filename = os.path.join(tmpdir, future_covs_fname)
            ts_future_covs.to_csv(ts_future_covs_filename, index=True)
            mlflow.log_artifact(ts_future_covs_filename, "future_covs_data")
            mlflow.set_tag('future_covs_uri', f'{mlrun.info.artifact_uri}/future_covs_data/{future_covs_fname}')
        else:
            mlflow.set_tag(f'infered_resolution_future', "None")
            mlflow.set_tag('future_covs_uri', "None")

        ## TODO: Read from APi

        # set mlflow tags for next steps
        if multiple:
            #TODO See where in use, possible implications with different duration tss
            if format == "short":
                mlflow.set_tag("dataset_start", datetime.strftime(ts["Date"].iloc[0], "%Y%m%d"))
                mlflow.set_tag("dataset_end", datetime.strftime(ts["Date"].iloc[-1], "%Y%m%d"))
            else:
                mlflow.set_tag("dataset_start", datetime.strftime(ts["Datetime"].iloc[0], "%Y%m%d"))
                mlflow.set_tag("dataset_end", datetime.strftime(ts["Datetime"].iloc[-1], "%Y%m%d"))
        else:
            mlflow.set_tag("dataset_start", datetime.strftime(ts.index[0], "%Y%m%d"))
            mlflow.set_tag("dataset_end", datetime.strftime(ts.index[-1], "%Y%m%d"))
        mlflow.set_tag("run_id", mlrun.info.run_id)

        mlflow.set_tag("stage", "load_raw_data")
        mlflow.set_tag('dataset_uri', f'{mlrun.info.artifact_uri}/raw_data/{fname}')

        return


# check for stream to csv: https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/load_raw_data.py

if __name__ == "__main__":
    print("\n=========== LOAD DATA =============")
    logging.info("\n=========== LOAD DATA =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    logging.info("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
    load_raw_data()