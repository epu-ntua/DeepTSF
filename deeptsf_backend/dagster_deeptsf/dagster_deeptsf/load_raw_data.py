"""
Downloads the RDN dataset and saves it as an artifact. ALso need to include interaction with weather apis here.
"""
import re
import requests
import tempfile
import os
import mlflow
import click
import sys
# import celery
from pandas.tseries.frequencies import to_offset
import logging
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import shutil
import pretty_errors
import uuid
from dagster import multi_asset, AssetIn, AssetOut, MetadataValue, Output, graph_multi_asset 
import tempfile
from math import ceil
from minio import Minio
# get environment variables
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append('..')
from utils import ConfigParser
from utils import none_checker
from utils import download_online_file, multiple_ts_file_to_dfs, multiple_dfs_to_ts_file, allow_empty_series_fun, to_seconds, to_standard_form, truth_checker
from exceptions import WrongIDs, EmptyDataframe, DifferentComponentDimensions, WrongColumnNames, DatetimesNotInOrder, WrongDateFormat, DuplicateDateError, MissingMultipleIndexError, NonIntegerMultipleIndexError, ComponentTooShortError

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

MONGO_URL = os.environ.get("MONGO_URL")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)


from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

def check_datetime_format(datetime_iterable, ids, file_format=None):
    #Check date index column has correct format. All dates must have the same format
    if file_format == "short":
        format_to_check = '%Y-%m-%d'
    else:
        format_to_check = '%Y-%m-%d %H:%M:%S'

    first_date = True
    for date_str, id_row in zip(datetime_iterable, ids):
        try:
            # Attempt to parse with the first format
            pd.to_datetime(date_str, format=format_to_check, errors='raise')
            first_date = False
        except ValueError:
            try:
                # Attempt to parse with the format 'YYYY-MM-DD'
                assert first_date
                format_to_check = '%Y-%m-%d'
                pd.to_datetime(date_str, format=format_to_check, errors='raise')
                first_date = False
            except:
                raise WrongDateFormat(date_str, id_row)

def check_and_convert_column_types(df, intended_types):
    """
    Checks the types of all columns in a DataFrame against a list of intended types.
    Attempts to convert columns to the intended types if they don't match.

    Parameters:
    df (pd.DataFrame): The DataFrame to check and convert.
    intended_types (list): A list of intended types for the columns in the same order as the DataFrame columns.

    Raises:
    ValueError: If any column cannot be converted to the intended type.
    """
    if len(df.columns) != len(intended_types):
        raise ValueError("The number of columns in the DataFrame does not match the number of intended types provided.")
    
    for i, (column, intended_type) in enumerate(zip(df.columns, intended_types)):
        actual_type = df[column].dtype
        if intended_type == str:
            df[column] = df[column].astype(intended_type)

            if df[column].astype(str).str.contains("/", regex=False).any():
                print("WARNING: Replacing / with _")
                df[column] = df[column].str.replace("/", "_", regex=False)

            float_pattern = re.compile(r'^\d+\.\d+$')
            for row_id, row in df.iterrows():
                value = row[column]
                # Check if the string matches the float pattern
                if float_pattern.match(value):
                     raise ValueError(f"Column '{column}' must strictly be str or int, and not float. First value to be float: {value} in row with id {row_id}")

        if actual_type != intended_type:
            try:
                df[column] = df[column].astype(intended_type)
                print(f"Column '{column}' successfully converted to {intended_type}.")
            except Exception as e:
                raise ValueError(f"Column '{column}' could not be converted to {intended_type}. Error: {e}")

    return df

def update_task(current, total, status, task):
    if task:
        task.update_state(state='PROGRESS', meta={'current': current, 'total': total, 'status': status}) 

def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            multiple: bool = False,
                            from_database: bool = False,
                            covariates: str = "series",
                            allow_empty_series=False,
                            format="long",
                            log_to_mlflow=True,
                            task=None,
                            skip_raw_series_validation=False):
    """
    Validates the input after read_csv is called and throws apropriate exception if it detects an error.
    
    The checks that are performed are the following:

    Checks for non-multiple timeseries (single timeseries):
        - The dataframe cannot be empty or have just one row (that would make it impossible to infer frequency).
        - The 'Datetime' column must be used as the index.
        - If the timeseries is the main dataset, 'Load' must be the only other column in the dataframe.
        - If the timeseries is a covariate timeseries, there must be only one column in the dataframe named arbitrarily.
        - The 'Datetime' index must have the correct format ('YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD', all indeces must have the same format) and be of type pd.Timestamp.
        - The 'Value' column must be of type float.
        - There should be no duplicate dates in the index.
        - All dates must be sorted in chronological order.

    Checks for multiple timeseries:
        - The dataframe cannot be empty
        - Timeseries ID column is set equal to ID if it doesn't exist
        - The dataframe index must be of integer type and an increasing sequence starting from 0 with no missing values.
        - Only the permitted column names exist in the dataframe:
            - For the short format, 'Date', 'ID', (optionally 'Timeseries ID') must exist in any order, and the rest should 
                be time columns (they must have names convertible to pd.Timestamp. They are not checked for chronological order.)
            - For the long format, 'Datetime', 'ID', 'Timeseries ID', 'Value' must exist in any order.
            If the columns are not in the order presented above, the will be internally converted in the correct order. Time columns will
            be left as is.
        - The 'Date' or 'Datetime' column must have the correct format
            - For the long format 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD', all indeces must have the same format and be convertible to type pd.Timestamp.
            - For the short format only 'YYYY-MM-DD' is allowed
        - All columns must have the correct types or be able to be converted to them ('datetime64[ns]' for date columns, str for ID columns, and float for value and time columns).
            Specifically for ID columns, if the have any numerical value that is not strictly an integer, an error will be raised. 
        - Then, the following checks are performed for each timeseries component individually:
            - There should be no duplicate dates.
            - All dates must be sorted in chronological order.
            - Every component must have at least 2 samples, for its frequency to be able to be inferred 
        - All components must have the same resolution.
        - All timeseries in the dataframe must have the same number of components.

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

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
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
    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     engine='python')

    if skip_raw_series_validation:
        if not multiple:
            ts.index = pd.to_datetime(ts.index)
            print("Infering resolution for single timeseries...")
            #Infering resolution for single timeseries
            resolution = to_standard_form(pd.to_timedelta(np.diff(ts.index).min()))
        if multiple:
            if format == "long":
                date_col = "Datetime"
                des_columns = ['Datetime', 'ID', 'Timeseries ID', 'Value']
                rest_cols = [col for col in list(ts.columns) if col not in des_columns]
                intended_col_types = ['datetime64[ns]', str, str, float]
            else:
                date_col = "Date"
                des_columns = ['Date', 'ID', 'Timeseries ID']
                rest_cols = [col for col in list(ts.columns) if col not in des_columns]
                try:
                    intended_col_types = ['datetime64[ns]', str, str] + [float for _ in range(len(ts.columns) - 3)]
                except:
                    pass
            ts[date_col] = pd.to_datetime(ts[date_col])
            print("Infering resolution for multiple ts and checking if all ts have the same one...")
            ts_l, id_l, ts_id_l, resolution = multiple_ts_file_to_dfs(series_csv, None, format=format)
        if log_to_mlflow:
            mlflow.set_tag(f'infered_resolution_{covariates}', resolution)
        return ts, resolution


    ######## NON MULTIPLE ########
    if not multiple:
        #Dataframe can not be empty or have just one row
        print("Check 1: Dataframe can not be empty or have just one row...")
        update_task(0, 7, "Check 1: Dataframe can not be empty or have just one row...", task)
        if len(ts) <= 1:
            raise EmptyDataframe(from_database)
        
        #CORRECT COLUMNS PRESENT
        print("Check 2: Correct columns present...")
        update_task(1, 7, "Check 2: Correct columns present...", task)
        #Check that column Datetime is used as index, and that Value is the only other column in the csv for the series csv
        if covariates == "series" and not (len(ts.columns) == 1 and ts.columns[0] == "Value" and ts.index.name == 'Datetime'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Datetime', "Value"])
        #Check that column Datetime is used as index, and that there is only other column in the csv for the covariates csvs
        elif covariates != "series" and not (len(ts.columns) == 1 and ts.index.name == 'Datetime'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Datetime', '<Value Column Name>'])

        #TYPE CHECKS

        print("Check 3: Check date index column has correct format and type...")
        update_task(2, 7, "Check 3: Check date index column has correct format and type...", task)
        #Check date index column has correct format
        check_datetime_format(ts.index, ts.index)
        ts.index = pd.to_datetime(ts.index)

        #Check index type
        index_type = type(ts.index[0])
        if index_type != pd.Timestamp:
            raise TypeError(f"The index column Datetime must be of type pd.Timestamp.")

        print("Check 4: Check Value column type...")
        update_task(3, 7, "Check 4: Check Value column type...", task)
        #Check Value column type
        ts = check_and_convert_column_types(ts, [float])

        print("Check 5: Check for duplicates...")
        update_task(4, 7, "Check 5: Check for duplicates...", task)
        #Check for duplicates
        duplicates = ts.index[ts.index.duplicated()]
        if not duplicates.empty:
            # Raise the custom exception if duplicates are found
            raise DuplicateDateError(duplicates[0], duplicates[0])

        print("Check 6: Check that dates are in order...")
        update_task(5, 7, "Check 6: Check that dates are in order...", task)
        #Check that dates are in order.
        dates_not_in_order = ts[ts.index.sort_values() != ts.index]
        if not dates_not_in_order.empty:
            for i in range(1, len(ts.index)):
                if ts.index[i] < ts.index[i - 1]:
                    first_wrong_date = ts.index[i]
            raise DatetimesNotInOrder(first_wrong_date=first_wrong_date, row_id=first_wrong_date)

        print("Check 7: Infering resolution for single timeseries...")
        update_task(6, 7, "Check 7: Infering resolution for single timeseries...", task)
        #Infering resolution for single timeseries
        resolution = to_standard_form(pd.to_timedelta(np.diff(ts.index).min()))

        update_task(7, 7, "Validation completed!", task)


    ######## MULTIPLE ########
    else:
        
        print("Check 1: Dataframe can not be empty...")
        update_task(0, 9, "Check 1: Dataframe can not be empty...", task)
        #Dataframe can not be empty
        if len(ts) == 0:
            raise EmptyDataframe(from_database)

        #CORRECT COLUMNS PRESENT
        
        #If columns don't exist set defaults
        if "Timeseries ID" not in ts.columns and "ID" in ts.columns:
            ts["Timeseries ID"] = ts["ID"]

        #Setting format dependant names
        if format == "long":
            date_col = "Datetime"
            des_columns = ['Datetime', 'ID', 'Timeseries ID', 'Value']
            rest_cols = [col for col in list(ts.columns) if col not in des_columns]
            intended_col_types = ['datetime64[ns]', str, str, float]
        else:
            date_col = "Date"
            des_columns = ['Date', 'ID', 'Timeseries ID']
            rest_cols = [col for col in list(ts.columns) if col not in des_columns]
            try:
                intended_col_types = ['datetime64[ns]', str, str] + [float for _ in range(len(ts.columns) - 3)]
            except:
                pass
        
        print("Check 2: Check if the index is of integer type...")
        update_task(1, 9, "Check 2: Check if the index is of integer type...", task)
        # Check if the index is of integer type
        if not pd.api.types.is_integer_dtype(ts.index):
            raise NonIntegerMultipleIndexError(ts.index.dtype)

        print("Check 3: Check present columns according to format...")
        update_task(2, 9, "Check 3: Check present columns according to format...", task)
        #Check present columns according to format
        if format == "short":
            if set(des_columns).issubset(set(list(ts.columns))):
                try:
                    ts = ts[des_columns + rest_cols]
                except:
                    pass
            
            #Check that all columns 'Date', 'ID', 'Timeseries ID' and only time columns exist in that order.
            if not (des_columns == list(ts.columns)[:3] and all(bool(re.match(r'^\d{2}:\d{2}:\d{2}$', e)) for e in (set(list(ts.columns))).difference(set(des_columns)))):
                raise WrongColumnNames(list(ts.columns), len(des_columns) + 1, des_columns + ['and the rest should all be time columns'], "short")
        else:
            if set(des_columns).issubset(set(list(ts.columns))):
                try:
                    ts = ts[des_columns + rest_cols]
                except:
                    pass

            #Check that only columns 'Datetime', 'ID', 'Timeseries ID', 'Value' exist in that order.
            if not des_columns == list(ts.columns):
                raise WrongColumnNames(list(ts.columns), len(des_columns), des_columns, "long")

        #TYPE CHECKS

        print("Check 4: Check index range and missing values...")
        update_task(3, 9, "Check 4: Check index range and missing values...", task)
        # Expected complete index range
        expected_index = pd.Index(range(len(ts)))
    
        # Check for missing values in the index
        missing_index = expected_index.difference(ts.index)
        if not missing_index.empty:
            raise MissingMultipleIndexError(missing_index[0])

        print("Check 5: Check date column has correct format...")
        update_task(4, 9, "Check 5: Check date column has correct format...", task)
        #Check date column has correct format
        check_datetime_format(ts[date_col], ts.index, format)
        ts[date_col] = pd.to_datetime(ts[date_col])

        print("Check 6: Check column types and if needed, convert them...")
        update_task(5, 9, "Check 6: Check column types and if needed, convert them...", task)
        #Check column types and if needed, convert them
        ts = check_and_convert_column_types(ts, intended_col_types)

        #Check each component individualy
        print("Check 7: each component individualy for duplicates and out of order dates...")
        update_task(6, 9, "Check 7: each component individualy for duplicates and out of order dates...", task)
        for ts_id in np.unique(ts["Timeseries ID"]):
            for id in np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"]):
                dates = ts[(ts["ID"] == id) & (ts["Timeseries ID"] == ts_id)].copy()
                dates["row_id"] = dates.index
                dates.index = dates[date_col]
                dates = dates[[date_col, "row_id"]]

                #Check for duplicates
                duplicates = dates[dates.index.duplicated()]
                if not duplicates.empty:
                    # Raise the custom exception if duplicates are found
                    raise DuplicateDateError(duplicates.index[0], duplicates["row_id"][0], ts_id, id)
            
                #Check that dates are in order.
                dates_not_in_order = dates[dates.index.sort_values() != dates.index]
                if not dates_not_in_order.empty:
                    for i in range(1, len(dates.index)):
                        if dates.index[i] < dates.index[i - 1]:
                            first_wrong_date = dates.index[i]
                            first_wrong_date_index = dates.iloc[i]["row_id"]
                    raise DatetimesNotInOrder(first_wrong_date, first_wrong_date_index, ts_id, id)
    
                        
        #Check that all timeseries in a multiple timeseries file have the same number of components
        print("Check 8: Check that all timeseries in a multiple timeseries file have the same number of components...")
        update_task(7, 9, "Check 8: Check that all timeseries in a multiple timeseries file have the same number of components...", task)
        comp_dict = {ts_id: len(np.unique(ts.loc[ts["Timeseries ID"] == ts_id]["ID"])) for ts_id in np.unique(ts["Timeseries ID"])}
        if len(set(comp_dict.values())) != 1:
            raise DifferentComponentDimensions(comp_dict)
        
        #Infering resolution for multiple ts
        print("Check 9: Infering resolution for multiple ts and checking if all ts have the same one...")
        update_task(8, 9, "Check 9: Infering resolution for multiple ts and checking if all ts have the same one...", task)
        ts_l, id_l, ts_id_l, resolution = multiple_ts_file_to_dfs(series_csv, None, format=format)

        if allow_empty_series:
            ts_list_ret, id_l_ret, ts_id_l_ret = allow_empty_series_fun(ts_l, id_l, ts_id_l, allow_empty_series=allow_empty_series)
            ts = multiple_dfs_to_ts_file(ts_list_ret, id_l_ret, ts_id_l_ret, "", save=False, format=format)
                
        update_task(9, 9, "Validation completed!", task)
        
    if log_to_mlflow:
        mlflow.set_tag(f'infered_resolution_{covariates}', resolution)
            
    return ts, resolution

# Depricated #

# def make_multiple(ts_covs, series_csv, inf_resolution, format):
#     """
#     In case covariates.

#     Parameters
#     ----------
#     series_csv
#         The path to the local file of the series to be validated
#     multiple
#         Whether to train on multiple timeseries
#     resolution
#         The resolution of the dataset
#     from_database
#         Whether the dataset was from MongoDB
#     covariates
#         If the function is called for the main dataset, then this equal to "series".
#         If it is called for the past / future covariate series, then it is equal to
#         "past" / "future" respectively. 

#     Returns
#     -------
#     (pandas.DataFrame, int)
#         A tuple consisting of the resulting dataframe from series_csv as well as the resolution
#     """


#     if series_csv != None:
#         ts_list, _, _, _, ts_id_l = multiple_ts_file_to_dfs(series_csv, inf_resolution, format=format)

#         ts_list_covs = [[ts_covs] for _ in range(len(ts_list))]
#         id_l_covs = [[str(list(ts_covs.columns)[0]) + "_" + ts_id_l[i]] for i in range(len(ts_list))]
#     else:
#         ts_list_covs = [[ts_covs]]
#         id_l_covs = [[list(ts_covs.columns)[0]]]
#     return multiple_dfs_to_ts_file(ts_list_covs, id_l_covs, id_l_covs, id_l_covs, id_l_covs, "", save=False, format=format)

from pymongo import MongoClient
import pandas as pd

mongo_client = MongoClient(MONGO_URL)


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
    db = mongo_client['inergy_prod_db']
    collection = db[database_name]
    df = pd.DataFrame(collection.find()).drop(columns={'_id', ''}, errors='ignore')
    df = unfold_timeseries(collection.find().sort('_id', -1))
    df = pd.DataFrame.from_dict(df)
    df.to_csv(f'{tmpdir}/load.csv', index=False)
    mongo_client.close()
    return

@multi_asset(
    name="load_raw_data_asset",
    description="For loading the raw dataset to MLflow",
    group_name='deepTSF_pipeline',
    required_resource_keys={"config"},
    ins={'start_pipeline_run': AssetIn(key='start_pipeline_run', dagster_type=str)},
    outs={"load_raw_data_out": AssetOut(dagster_type=dict)})

def load_raw_data_asset(context, start_pipeline_run):
    config = context.resources.config
    # Use config parameters in the asset logic
    tenant = config.tenant
    if none_checker(tenant) is not None:
        mlflow_uri = f"http://{tenant}-mlflow:5000"
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        tenant = "mlflow-bucket"

    series_csv=config.series_csv
    series_uri=config.series_uri
    past_covs_csv=config.past_covs_csv
    past_covs_uri=config.past_covs_uri
    future_covs_csv=config.future_covs_csv
    future_covs_uri=config.future_covs_uri
    multiple=config.multiple
    resolution=config.resolution
    from_database=config.from_database
    database_name=config.database_name
    format=config.format
    experiment_name = config.experiment_name
    darts_model = config.darts_model
    skip_raw_series_validation = config.skip_raw_series_validation
    parent_run_name = config.parent_run_name if none_checker(config.parent_run_name) != None else darts_model + '_pipeline'
    
    tmpdir = tempfile.mkdtemp()

    past_covs_csv = none_checker(past_covs_csv)
    past_covs_uri = none_checker(past_covs_uri)
    future_covs_csv = none_checker(future_covs_csv)
    future_covs_uri = none_checker(future_covs_uri)

    series_uri = none_checker(series_uri)

    if from_database:
        load_data_to_csv(tmpdir, database_name)
        series_csv = f'{tmpdir}/load.csv'

    elif series_uri != None:
        download_file_path = download_online_file(client, series_uri, dst_filename="series.csv", bucket_name=tenant)
        series_csv = download_file_path

    if past_covs_uri != None:
        download_file_path = download_online_file(client, past_covs_uri, dst_filename="past_covs.csv", bucket_name=tenant)
        past_covs_csv = download_file_path

    if future_covs_uri != None:
        download_file_path = download_online_file(client, future_covs_uri, dst_filename="future_covs.csv", bucket_name=tenant)
        future_covs_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"mlflow.runName": parent_run_name}, run_id=start_pipeline_run) as parent_run:
        with mlflow.start_run(tags={"mlflow.runName": "load_data"}, nested=True) as mlrun:

            ts, _ = read_and_validate_input(series_csv, multiple=multiple, from_database=from_database, format=format, skip_raw_series_validation=skip_raw_series_validation)

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

                ts_past_covs, _ = read_and_validate_input(past_covs_csv,
                                                                multiple=True,
                                                                from_database=from_database,
                                                                covariates="past",
                                                                format=format,
                                                                skip_raw_series_validation=skip_raw_series_validation)
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

                ts_future_covs, _ = read_and_validate_input(future_covs_csv,
                                                                multiple=True,
                                                                from_database=from_database,
                                                                covariates="future",
                                                                format=format,
                                                                skip_raw_series_validation=skip_raw_series_validation)
                                        
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
            curr_run_id = mlrun.info.run_id
            
        completed_run = mlflow.tracking.MlflowClient().get_run(curr_run_id)

        load_data_series_uri = completed_run.data.tags['dataset_uri'].replace("mlflow-artifacts:", S3_ENDPOINT_URL + "/" + tenant)
        infered_resolution_series = completed_run.data.tags['infered_resolution_series']

        load_data_past_covs_uri = completed_run.data.tags['past_covs_uri'].replace("mlflow-artifacts:", S3_ENDPOINT_URL + "/" + tenant)
        infered_resolution_past = completed_run.data.tags['infered_resolution_past']

        load_data_future_covs_uri = completed_run.data.tags['future_covs_uri'].replace("mlflow-artifacts:", S3_ENDPOINT_URL + "/" + tenant)
        infered_resolution_future = completed_run.data.tags['infered_resolution_future']

        return Output({"infered_resolution_series": infered_resolution_series,
                                                             "series_uri": load_data_series_uri,
                                                             "past_covs_uri": load_data_past_covs_uri,
                                                             "infered_resolution_past": infered_resolution_past,
                                                             "future_covs_uri": load_data_future_covs_uri,
                                                             "infered_resolution_future": infered_resolution_future})
