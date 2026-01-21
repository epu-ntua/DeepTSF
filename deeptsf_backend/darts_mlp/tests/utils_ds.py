# from dotenv import load_dotenv
# load_dotenv()
import tempfile
# import pretty_errors
import os
# import mlflow
import pandas as pd
import yaml
import darts
from pandas.tseries.frequencies import to_offset
from math import ceil
cur_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from tqdm import tqdm
import logging
from .exceptions import MandatoryArgNotSet, NotValidConfig, EmptySeries, DifferentFrequenciesMultipleTS, ComponentTooShortError, TSIDNotFoundInferenceError
# import json
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)
# import requests
from datetime import date
# import pvlib
import pandas as pd
import matplotlib.pyplot as plt
# from pvlib.pvsystem import PVSystem
# from pvlib.location import Location
# from pvlib.modelchain import ModelChain
# from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pi
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries
import math
from datetime import timezone
from darts.dataprocessing.transformers import MissingValuesFiller
import tempfile
import holidays
from pytz import timezone
import pytz
from datetime import datetime
from typing import Union, List, Tuple
# from minio import S3Error
# from minio.commonconfig import CopySource


def multiple_ts_file_to_dfs(series_csv: Union[str, pd.DataFrame] = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            resolution: str = "15min",
                            value_name="Value",
                            format="long"):
    """
    Reads the input multiple time series file or DataFrame, and returns a tuple containing 
    a list of the time series it consists of, along with their ids and timeseries ids.

    Parameters
    ----------
    series_csv : str or pandas.DataFrame, optional
        The file name of the CSV to be read, or a pandas DataFrame containing the data. 
        If a string is provided, it must point to a valid CSV file in the multiple time series 
        format described in the documentation. If a DataFrame is provided, it should already 
        conform to the expected format.
    resolution : str, optional
        The resolution of the dataset (default is "15min").
    value_name : str, optional
        The name of the value column of the returned DataFrames (default is "Value").
    format : str, optional
        The format in which the data is returned (default is "long").

    Returns
    -------
    Tuple[List[List[pandas.DataFrame]], List[List[str]], List[List[str]]]
        A tuple with the list of lists of DataFrames, the ids of their components, 
        and the time series ids. For example, if the function reads a file or DataFrame 
        with 2 time series (with ids ts_1 and ts_2), and each one consists of 3 components 
        (with ids ts_1_1, ts_1_2, ts_1_3, ts_2_1, ts_2_2, ts_2_3), then the function will return:
        
        (res, id_l, ts_id_l), where:
        - res = [[ts_1_comp_1, ts_1_comp_2, ts_1_comp_3], [ts_2_comp_1, ts_2_comp_2, ts_2_comp_3]]
        - id_l = [[ts_1_1, ts_1_2, ts_1_3], [ts_2_1, ts_2_2, ts_2_3]]
        - ts_id_l = [[ts_1, ts_1, ts_1], [ts_2, ts_2, ts_2]]
        
        All of the above lists of lists have the same number of lists and each sublist 
        contains the same number of elements across the corresponding location in the 
        other lists. Each sublist corresponds to a time series, and each element within 
        a sublist corresponds to a component of that time series.
    """

    if type(series_csv) == str:
        ts = pd.read_csv(series_csv,
                     sep=None,
                     header=0,
                     index_col=0,
                     engine='python')
    else:
        ts = series_csv
        
    if format == "long":
        ts["Datetime"] = pd.to_datetime(ts["Datetime"])
    else:
        ts["Date"] = pd.to_datetime(ts["Date"])


    res = []
    id_l = []
    ts_id_l = []
    ts_ids = list(np.unique(ts["Timeseries ID"]))
    first = True
    print("\nTurning multiple ts file to dataframe list...")
    logging.info("\nTurning multiple ts file to dataframe list...")
    for ts_id in tqdm(ts_ids):
        curr_ts = ts[ts["Timeseries ID"] == ts_id]
        ids = list(np.unique(curr_ts["ID"]))
        res.append([])
        id_l.append([])
        ts_id_l.append([])
        for id in ids:
            curr_comp = curr_ts[curr_ts["ID"] == id]
            if format == 'short':
                curr_comp = pd.melt(curr_comp, id_vars=['Date', 'ID', 'Timeseries ID'], var_name='Time', value_name=value_name)
                curr_comp["Datetime"] = pd.to_datetime(curr_comp['Date'].dt.strftime("%Y-%m-%d") + curr_comp['Time'], format='%Y-%m-%d%H:%M:%S')
            else:
                curr_comp["Datetime"] = pd.to_datetime(curr_comp["Datetime"])
            curr_comp = curr_comp.set_index("Datetime")
            series = curr_comp[value_name].sort_index().dropna()

            #Check if the length of a component is less than one
            if len(series) <= 1:
                raise ComponentTooShortError(len(series), ts_id, id)
            
            if resolution!=None:
                series = series.asfreq(resolution)
            elif first:
                infered_resolution = to_standard_form(pd.to_timedelta(np.diff(series.index).min()))
                series = series.asfreq(infered_resolution)
                first = False
                first_id = id
                first_ts_id = ts_id
            else:
                temp = to_standard_form(pd.to_timedelta(np.diff(series.index).min()))
                if temp != infered_resolution:
                    raise DifferentFrequenciesMultipleTS(temp, id, ts_id, infered_resolution, first_id, first_ts_id)
                else:
                    series = series.asfreq(temp)
                    infered_resolution = temp

            res[-1].append(pd.DataFrame({value_name : series}))
            id_l[-1].append(id)
            ts_id_l[-1].append(ts_id)
    if resolution != None:
        return res, id_l, ts_id_l
    else:
        return res, id_l, ts_id_l, infered_resolution
    
def multiple_dfs_to_ts_file(res_l, id_l, ts_id_l, save_dir, save=True, format="long", value_name="Value"):
    ts_list = []
    print("\nTurning dataframe list to multiple ts file...")
    logging.info("\nTurning dataframe list to multiple ts file...")
    for ts_num, (ts, id_ts, ts_id_ts) in tqdm(list(enumerate(zip(res_l, id_l, ts_id_l)))):
        if type(ts) == darts.timeseries.TimeSeries:
            ts = [ts.univariate_component(i).pd_dataframe() for i in range(ts.n_components)]
        for comp_num, (comp, id, ts_id) in enumerate(zip(ts, id_ts, ts_id_ts)):
            comp.columns.values[0] = value_name
            if format == "short":
                comp["Date"] = comp.index.date
                comp["Time"] = comp.index.time
                comp = pd.pivot_table(comp, index=["Date"], columns=["Time"])
                comp = comp[value_name]
            comp["ID"] = id
            comp["Timeseries ID"] = ts_id
            ts_list.append(comp)
    if format == "long":
        res = pd.concat(ts_list).reset_index()
        res.rename(columns={'index': 'Datetime'}, inplace=True)
    else:
        res = pd.concat(ts_list).sort_values(by=["Date", "Timeseries ID", "ID"])
        res = res.reindex(columns=sorted(res.columns, key=lambda x : 0 if isinstance(x, str) else int(datetime.combine(datetime.today().date(), x).timestamp())))
        res = res.reset_index()
    if save:
        res.to_csv(save_dir)
    return res

def load_local_csv_or_df_as_darts_timeseries(local_path_or_df, 
                                       name='Time Series', 
                                       time_col='Datetime', 
                                       last_date=None, 
                                       multiple = False, 
                                       resolution="15min", 
                                       format="long"):

    try:
        if multiple:
            #TODO Fix this too (
            #file_name, format)
            ts_list, id_l, ts_id_l = multiple_ts_file_to_dfs(series_csv=local_path_or_df, resolution=resolution, format=format)
            covariate_l  = []

            print("\nTurning dataframes to timeseries...")
            logging.info("\nTurning dataframes to timeseries...")
            for comps in tqdm(ts_list):
                first = True
                for df in comps:
                    covariates = darts.TimeSeries.from_dataframe(
                                df,
                                fill_missing_dates=True,
                                freq=None)
                    covariates = covariates.astype(np.float32)
                    if last_date is not None:
                        try:
                            covariates.drop_after(pd.Timestamp(last_date))
                        except:
                            pass
                    if first:
                        first = False
                        covariate_l.append(covariates)
                    else:
                        covariate_l[-1] = covariate_l[-1].stack(covariates)
            covariates = covariate_l
        else:
            id_l, ts_id_l = [[]], [[]]
            if type(local_path_or_df) == pd.DataFrame:
                covariates = darts.TimeSeries.from_dataframe(
                    local_path_or_df,
                    fill_missing_dates=True,
                    freq=None)
            else:
                covariates = darts.TimeSeries.from_csv(
                    local_path_or_df, time_col=time_col,
                    fill_missing_dates=True,
                    freq=None)
            covariates = covariates.astype(np.float32)
            if last_date is not None:
                try:
                    covariates.drop_after(pd.Timestamp(last_date))
                except:
                    pass
    except (FileNotFoundError, PermissionError) as e:
        print(
            f"\nBad {name} file.  The model won't include {name}...")
        logging.info(
            f"\nBad {name} file. The model won't include {name}...")
        covariates, id_l, ts_id_l = None, None, None
    return covariates, id_l, ts_id_l