"""
Downloads the RDN dataset and saves it as an artifact. ALso need to include interaction with weather apis here.
"""
import requests
import tempfile
import os
import mlflow
import click
from utils import ConfigParser
import logging
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from utils import download_online_file
import shutil
import pretty_errors
import uuid
from exceptions import DatesNotInOrder
from exceptions import WrongColumnNames
from exceptions import WrongIDs
from utils import truth_checker
import tempfile

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

MONGO_URL = os.environ.get("MONGO_URL")


def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: bool = True,
                            multiple: bool = False,
                            resolution: int = 15):
    """
    Validates the input after read_csv is called and
    throws apropriate exception if it detects an error

    Multiple timeseries file format (along with example values):

    Index | Day         | ID | Country | Country Code | 00:00:00 | 00:00:00 + resolution | ... | 24:00:00 - resolution
    0     | 2015-04-09  | 0  | Portugal| PT           | 5248     | 5109                  | ... | 5345
    1     | 2015-04-09  | 1  | Spain   | ES           | 25497    | 23492                 | ... | 25487
    .
    .

    Columns can be in any order and ID must alwaws be convertible to an int, and consequtive. Also, all the above
    column names must be present in the file, and the hour columns must be consequtive and separated by resolution 
    minutes. The lines can be at any order as long as the Day column is increasing for each country.

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

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe from series_csv
    """
    ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     dayfirst=day_first,
                     engine='python')
    if not multiple:
        if not ts.index.sort_values().equals(ts.index):
            raise DatesNotInOrder()
        elif not (len(ts.columns) == 1 and ts.columns[0] == 'Load' and ts.index.name == 'Date'):
            raise WrongColumnNames([ts.index.name] + list(ts.columns), 2, ['Load', 'Date'])
    else:
        des_columns = list(map(str, ['Day', 'ID', 'Country', 'Country Code'] + [(pd.Timestamp("00:00:00") + i*pd.DateOffset(minutes=resolution)).time() for i in range(60*24//resolution)]))
        print(ts["ID"].dtype == [np.int64, np.int32])
        print(set(des_columns) == set(ts.columns))
        try:
            ts["ID"].apply(int)
        except:
            raise WrongIDs(np.unique(ts["ID"]))
        if not(len(des_columns) == len(ts.columns) and set(des_columns) == set(ts.columns)):
            raise WrongColumnNames(list(ts.columns), len(des_columns), des_columns)
#        elif not (np.unique(ts["ID"]) == list(range(max(ts["ID"])))):
#            raise WrongIDs(np.unique(ts["ID"]))
#        for id in np.unique(ts["ID"]):
#            if not ts.loc[ts["ID"] = id]["Day"].sort_values().equals(ts.loc[ts["ID"] = id]["Day"]):
#                raise DatesNotInOrder(id)
        
    return ts

from pymongo import MongoClient
import pandas as pd

client = MongoClient(MONGO_URL)

def load_data_to_csv(tmpdir, mongo_name):
    lds = get_loads_from_db(mongo_name)
    new_loads = unfold_timeseries(lds)
    loads_to_csv(new_loads, tmpdir)
    client.close()


def loads_to_csv(new_loads, tmpdir):
    df = pd.DataFrame.from_dict(new_loads)
    df.to_csv(f'{tmpdir}/load.csv', index=False)


def unfold_timeseries(lds):
    new_loads = {'Date': [], 'Load': []}
    prev_date = ''
    for l in reversed(list(lds)):
        if prev_date != l['date']:
            for key in l:
                if key != '_id' and key != 'date':
                    new_date = l['date'] + ' ' + key
                    new_loads['Date'].append(new_date)
                    new_loads['Load'].append(l[key])
        prev_date = l['date']
    return new_loads


def get_loads_from_db(mongo_name):
    db = client['inergy_prod_db']
    collection = db[mongo_name]
    loads = collection
    lds = loads.find().sort('_id', -1)
    return lds


@click.command(
    help="Downloads the RDN series and saves it as an mlflow artifact "
    "called 'load_x_y.csv'."
    )
# TODO: Update that to accept url as input instead of local file
@click.option("--series-csv",
    type=str,
    default="../../RDN/Load_Data/2009-2019-global-load.csv",
    help="Local time series csv file"
    )
@click.option("--series-uri",
    default="online_artifact",
    help="Remote time series csv file. If set, it overwrites the local value."
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
    default="15",
    type=str,
    help="The resolution of the dataset in minutes."
)
@click.option("--from-mongo",
    default="false",
    type=str,
    help="Whether to read the dataset from mongodb."
)
@click.option("--mongo-name",
    default="rdn_load_data",
    type=str,
    help="Which mongo file to read."
)

def load_raw_data(series_csv, series_uri, day_first, multiple, resolution, from_mongo, mongo_name):

    from_mongo = truth_checker(from_mongo)
    tmpdir = tempfile.mkdtemp()

    if from_mongo:
        load_data_to_csv(tmpdir, mongo_name)
        series_csv = f'{tmpdir}/load.csv'

    elif series_uri != "online_artifact":
        download_file_path = download_online_file(series_uri, dst_filename="series.csv")
        series_csv = download_file_path

    series_csv = series_csv.replace('/', os.path.sep)
    fname = series_csv.split(os.path.sep)[-1]
    local_path = series_csv.split(os.path.sep)[:-1]

    day_first = truth_checker(day_first)

    multiple = truth_checker(multiple)

    resolution = int(resolution)

    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:

        print(f'Validating timeseries on local file: {series_csv}')
        logging.info(f'Validating timeseries on local file: {series_csv}')
        ts = read_and_validate_input(series_csv, day_first, multiple=multiple, resolution=resolution)


        local_path = local_path.replace("'", "") if "'" in local_path else local_path
        series_filename = os.path.join(*local_path, fname)
        # series = pd.read_csv(series_filename,  index_col=0, parse_dates=True, squeeze=True)
        # darts_series = darts.TimeSeries.from_series(series, freq=f'{timestep}min')
        print(f'\nUploading timeseries to MLflow server: {series_filename}')
        logging.info(f'\nUploading timeseries to MLflow server: {series_filename}')

        ts_filename = os.path.join(tmpdir, fname)
        ts.to_csv(ts_filename, index=True)
        mlflow.log_artifact(ts_filename, "raw_data")

        ## TODO: Read from APi

        # set mlflow tags for next steps
        ##TODO fix this
        if multiple:
#            mlflow.set_tag("dataset_start", datetime.strftime(ts["Day"][0], "%Y%m%d"))
#            mlflow.set_tag("dataset_end", datetime.strftime(ts["Day"][-1], "%Y%m%d"))
            pass
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
