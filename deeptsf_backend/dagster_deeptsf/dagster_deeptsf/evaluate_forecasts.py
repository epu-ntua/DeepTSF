import pretty_errors
from darts.utils.missing_values import extract_subseries
from functools import reduce
from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.models import (
    NaiveSeasonal,
)
from darts import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import logging
import click
import mlflow
import shutil
from .preprocessing import split_dataset
import tempfile
import random
import shap
from typing import Union
from typing import List
import darts
import json
import statistics
from minio import Minio
from dagster import multi_asset, AssetIn, AssetOut, MetadataValue, Output, graph_multi_asset 
import sys
sys.path.append('..')
from utils import truth_checker, load_yaml_as_dict
from utils import none_checker, truth_checker, download_online_file, load_local_csv_or_df_as_darts_timeseries, load_model, load_scaler, multiple_dfs_to_ts_file, get_pv_forecast, plot_series, to_seconds
from exceptions import EvalSeriesNotFound

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)


# DARTS

# def eval_model(model, train, val, n_steps, future_covariates=None, past_covariates=None):
#     pred = model.predict(n=n_steps,95

#                          future_covariates=future_covariates,
#                          past_covariates=past_covariates)
#     series = train.append(val)
#     plt.figure(figsize=(20, 10))
#     series.drop_before(pd.Timestamp(
#         pred.time_index[0] - datetime.timedelta(days=7))) \
#         .drop_after(pred.time_index[-1]).plot(label='actual')
#     pred.plot(label='forecast')
#     plt.legend()
#     mape_error = darts.metrics.mape(val, pred)
#     print('MAPE = {:.2f}%'.format(mape_error))
#     return mape_error

def append(x, y):
    """
    Append `y` to `x`, discarding any part of `y` whose timestamps already appear in `x`.

    Parameters
    ----------
    x : TimeSeries
        Leading series.
    y : TimeSeries
        Series to append.

    Returns
    -------
    TimeSeries
        `x` followed by the non‑overlapping portion of `y`.
    """
    last_ts = x.end_time()              # final timestamp in x
    if last_ts > y.start_time():
        y_tail  = y.drop_before(last_ts)    # keeps strictly after last_ts
    else:
        y_tail  = y

    # If y had nothing new, just return x (copy so caller can mutate safely)
    if y_tail.n_timesteps == 0:
        return x.copy()

    return x.append(y_tail)


def backtester(model,
               series_transformed,
               test_start_date,
               forecast_horizon,
               stride=None,
               series=None,
               transformer_ts=None,
               retrain=False,
               future_covariates=None,
               past_covariates=None,
               path_to_save_backtest=None,
               m_mase=1,
               num_samples=1,
               pv_ensemble=False,
               resolution="60min",
               id_l=None):
    """ Does the same job with advanced forecast but much more quickly using the darts
    bult-in historical_forecasts method. Use this for evaluation. The other only
    provides pure inference. Provide a unified timeseries test set point based
    on test_start_date. series_transformed does not need to be adjacent to
    training series. if transformer_ts=None then no inverse transform is applied
    to the model predictions.
    Parameters
    ----------
    Returns
    ----------
    """
    # produce the fewest forecasts possible.
    if stride is None:
        stride = forecast_horizon


    #keep last non nan values
    #must be sufficient for historical_forecasts and mase calculation
    #TODO Add check for that in the beggining
    # series = extract_subseries(series, min_gap_size=1, mode='any')[-1]
    # series_transformed = extract_subseries(series_transformed, min_gap_size=1, mode='any')[-1]

    test_start_date = series_transformed.pd_dataframe()[series_transformed.pd_dataframe().index >= pd.Timestamp(test_start_date + " 00:00:00")].index[0]
    # plot_series(df_list=[series_transformed], 
    #                 ts_name_list=["series_transformed"], 
    #                 save_dir=os.path.join(path_to_save_backtest,
    #                                     f'series_transformed.html'))

    # produce list of forecasts
    backtest_series_transformed = model.historical_forecasts(series_transformed,
                                                             future_covariates=future_covariates,
                                                             past_covariates=past_covariates,
                                                             start=test_start_date,
                                                             forecast_horizon=forecast_horizon,
                                                             stride=stride,
                                                             retrain=retrain,
                                                             last_points_only=False,
                                                             verbose=False,
                                                             num_samples=num_samples)

    # with open("/app/dagster_deeptsf/all_series.txt", "w", encoding="utf-8") as fh:
    #     for i, ts in enumerate(backtest_series_transformed, start=1):
    #         fh.write(f"# ---- series_{i} ----\n")
    #         fh.write(ts.pd_dataframe().to_csv(index=True))  # keep timestamps
    #         fh.write("\n")  # blank line between blocks

    overlap = forecast_horizon - stride          # positive ⇒ overlapping windows

    if overlap > 0:
        print(
            f"\nDetected overlap of {overlap} time steps "
            f"(stride={stride} < forecast_horizon={forecast_horizon}).\n"
            f"→ This will be treated as forecasting "
            f"{1+ forecast_horizon - stride} to {forecast_horizon} "
            f"steps ahead instead of the usual 1 to {forecast_horizon}.\n"
            f"→ Dropping the first {overlap} time steps of the forecasted time series."
        )

    elif overlap == 0:
        # stride and forecast_horizon align perfectly – nothing to do
        pass
    else:
        raise ValueError(
            f"Stride ({stride}) exceeds forecast_horizon ({forecast_horizon})."
            "This is not supported for now."
        )

    # flatten lists of forecasts due to last_points_only=False
    if isinstance(backtest_series_transformed, list):
        backtest_series_transformed = reduce(
            append, backtest_series_transformed)
        
    # Drop the first timesteps equal to the overlap
    backtest_series_transformed = backtest_series_transformed[overlap:]

    # backtest_series_transformed.pd_dataframe().to_csv("/app/dagster_deeptsf/combined_series.csv", index=True)

    # inverse scaling
    if transformer_ts is not None and series is not None:
        backtest_series = transformer_ts.inverse_transform(
            backtest_series_transformed)
    else:
        backtest_series = backtest_series_transformed
        print("\nWarning: Scaler not provided. Ensure model provides normal scale predictions")
        logging.info(
            "\n Warning: Scaler not provided. Ensure model provides normal scale predictions")
        
    # plot_series(df_list=[backtest_series], 
    #                 ts_name_list=["backtest_series"], 
    #                 save_dir=os.path.join(path_to_save_backtest,
    #                                     f'backtest_series_no_pv.html'))

        

    if pv_ensemble:
        print("\nAdding pv forecast to prediction")
        logging.info("\nAdding pv forecast to prediction")

        backtest_series = backtest_series - get_pv_forecast([], 
                                                            start=backtest_series.pd_dataframe().index[0], 
                                                            end=backtest_series.pd_dataframe().index[-1], 
                                                            inference=False, 
                                                            kW=60, 
                                                            use_saved=True)
        
        # plot_series(df_list=[backtest_series], 
        #             ts_name_list=["backtest_series"], 
        #             save_dir=os.path.join(path_to_save_backtest,
        #                                 f'backtest_series.html'))


    # # plot all test
    # fig1 = plt.figure(figsize=(15, 8))
    # ax1 = fig1.add_subplot(111)
    # backtest_series.plot(label='forecast')
    # #try except in case of nans before start
    # try:
    #     series \
    #     .drop_before(pd.Timestamp(pd.Timestamp(test_start_date) - datetime.timedelta(days=7))) \
    #     .drop_after(backtest_series.time_index[-1]) \
    #     .plot(label='actual')
    # except:
    #     series \
    #     .drop_before(pd.Timestamp(pd.Timestamp(test_start_date) - datetime.timedelta(days=1))) \
    #     .drop_after(backtest_series.time_index[-1]) \
    #     .plot(label='actual')
    # ax1.legend()
    # ax1.set_title(
    #     f'Backtest, starting {test_start_date}, {forecast_horizon}-steps horizon')
    # # plt.show()

    # try:
    #     # plot one week (better visibility)
    #     forecast_start_date = pd.Timestamp(
    #         test_start_date + datetime.timedelta(days=7))

    #     fig2 = plt.figure(figsize=(15, 8))
    #     ax2 = fig2.add_subplot(111)
    #     backtest_series \
    #         .drop_before(pd.Timestamp(forecast_start_date)) \
    #         .drop_after(forecast_start_date + datetime.timedelta(days=7)) \
    #         .plot(label='Forecast')
    #     series \
    #         .drop_before(pd.Timestamp(forecast_start_date)) \
    #         .drop_after(forecast_start_date + datetime.timedelta(days=7)) \
    #         .plot(label='Actual')
    #     ax2.legend()
    #     ax2.set_title(
    #     f'Weekly forecast, Start date: {forecast_start_date}, Forecast horizon (timesteps): {forecast_horizon}, Forecast extended with backtesting...')
    # except:
    #     pass
    # Metrics

    #Making the test series (ground truth) start at the same time as backtest_series (the result produced by the model)    
    test_series = series.drop_before(pd.Timestamp(test_start_date) + (overlap - 1) * pd.Timedelta(resolution))
    # print("test series start", test_series.pd_dataframe().index[0])
    # print("test series end", test_series.pd_dataframe().index[-1])
    # print("backtest series start", backtest_series.pd_dataframe().index[0])
    # print("backtest series end", backtest_series.pd_dataframe().index[-1])
    # print("insample start", series.drop_after(pd.Timestamp(test_start_date) + overlap * pd.Timedelta(resolution)).pd_dataframe().index[0])
    # print("insample end", series.drop_after(pd.Timestamp(test_start_date) + overlap * pd.Timedelta(resolution)).pd_dataframe().index[-1])
    metrics = {
        "mae": mae_darts(
            test_series,
            backtest_series),
        "rmse": rmse_darts(
            test_series,
            backtest_series),
        "nrmse_min_max": rmse_darts(
            test_series,
            backtest_series) / (
            test_series.pd_dataframe().max().iloc[0]- 
            test_series.pd_dataframe().min().iloc[0]),
        "nrmse_mean": rmse_darts(
            test_series,
            backtest_series) / (
            test_series.pd_dataframe().mean().iloc[0])
    }
    if min(test_series.min(axis=1).values()) > 0 and min(backtest_series.min(axis=1).values()) > 0:
        metrics["mape"] = mape_darts(
            test_series,
            backtest_series)
    else:
        print("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        logging.info("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        metrics["mape"] = np.nan
    #note: insample series for mase begins at the start of the training set and ends at the end of the val set + some values in case of overlap
    try:
        metrics["mase"] = mase_darts(
            test_series,
            backtest_series,
            insample=series.drop_after(pd.Timestamp(test_start_date) + overlap * pd.Timedelta(resolution)),
            m = m_mase)
    except:
        print("\nSeries is periodical. Setting mase to NaN...")
        logging.info("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        metrics["mase"] = np.nan

    try:
        metrics["smape"] = smape_darts(
            test_series,
            backtest_series)
    except:
        print("\nSeries not strictly positive. Setting smape to NaN...")
        logging.info("\nSeries not strictly positive. Setting smape to NaN...")
        metrics["smape"] = np.nan

    for key, value in metrics.items():
        print(key, ': ', value)


    # save plot
    if path_to_save_backtest is not None:
        os.makedirs(path_to_save_backtest, exist_ok=True)
        mape = metrics['mape']
        # fig1.savefig(os.path.join(path_to_save_backtest,
        #     f'test_start_date_{test_start_date.date()}_forecast_horizon_{forecast_horizon}_mape_{mape:.2f}.png'))
        # fig2.savefig(os.path.join(path_to_save_backtest,
        #     f' week2_forecast_start_date_{test_start_date.date()}_forecast_horizon_{forecast_horizon}.png'))

        plot_series(df_list=[series, backtest_series], 
                    ts_name_list=["actual", "prediction"], 
                    id_list=id_l,
                    save_dir=os.path.join(path_to_save_backtest,
                                        f'Actual_vs_Predicted.html'))
        try:
            backtest_series \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions.csv'))
        except:
            backtest_series.quantile_df() \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions.csv'))

        try:
            backtest_series_transformed \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions_transformed.csv'))
        except:
            backtest_series_transformed.quantile_df() \
            .to_csv(os.path.join(path_to_save_backtest, 'predictions_transformed.csv'))

        series_transformed.drop_before(pd.Timestamp(test_start_date) - pd.Timedelta(resolution)) \
        .to_csv(os.path.join(path_to_save_backtest, 'test_transformed.csv'))

        series.drop_before(pd.Timestamp(test_start_date) - pd.Timedelta(resolution)) \
        .to_csv(os.path.join(path_to_save_backtest, 'original_series.csv'))

    return {"metrics": metrics, "eval_plot": plt, "backtest_series": backtest_series}

def build_shap_dataset(size: Union[int, float],
                       train: darts.TimeSeries,
                       test: darts.TimeSeries,
                       shap_input_length: int,
                       shap_output_length: int,
                       past_covs: darts.TimeSeries = None,
                       future_covs: darts.TimeSeries = None,
                       id_l = None,
                       id_l_past_covs = None,
                       id_l_future_covs = None):
    """
    Produces the dataset to be fed to SHAP's explainer. It chooses a subset of
    the validation dataset and it returns a dataframe of these samples along
    with their corresponding covariates if needed by the model. Naive model not supported
    Parameters
    ----------
    size
        The number of samples to be produced. If it is a float, it represents
        the proportion of possible samples of the validation dataset to be
        chosen. If it is an int, it represents the absolute numbe of samples to
        be produced.
    train
        The training dataset of the model. It is needed to set the background samples
        of the explainer.
    test
        The validation dataset of the model.
    shap_input_length
        The length of each sample of the dataset. Also, the input_chunk_length of the model if 
        it exists as a parameter. It can also be set by the user
    shap_output_length
        The length of each sample of the result. Also, the forecast horizon used for the model
    past_covs
        Whether the model has been trained with past covariates
    future_covs
        Whether the model has been trained with future covariates
    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        -First position of tuple:
            A dataframe consisting of the samples of the validation dataset that
            were chosen, along with their corresponding covariates. Its exact form
            is as follows:
            0 timestep  1 timestep  ... <shap_input_length - 1> timestep \
            Step 0 of past covariate 0 ... Step <shap_input_length - 1> of past covariate 0 \
            Step 0 of past covariate 1 ... Step <shap_input_length - 1> of past covariate <past_covs.n_components> \
            Step 0 of future covariate 0 ... Step <shap_input_length + shap_output_length - 1> of future covariate <future_covs.n_components>
        -Second position of tuple:
            A dataframe containing the sample providing the values that replace the data's values that are simulated to be
            missing. Each feature's value is the median of the TimeSeries it originates from. So, if it's a covariate feature,
            its value will be the median of this covariate, and if it is a feature of the dataset, its value will be the median
            of the training dataset.
    """
    #data: The dataset to be given to SHAP
    data = []
    #background: Dataframe containing the sample providing the values that replace the data's values that are simulated to be missing
    background = []
    #columns: The name of the columns of the dataframes
    columns = []
    #Whether it is the first time the for loop is run
    first_iter = True
    samples = set()

    #Choosing the samples of val we will use randomly
    if(type(size) == float):
        size = int(size*(len(test) - shap_input_length + 1))
    if size == len(test) - shap_input_length + 1:
        samples = set(range(size))
    else:
        for i in range(size):
            while(True):
                r = random.randint(0, len(test) - shap_input_length + 1)
                if r not in samples:
                    break
            samples.add(r)

    for i in samples:
        try:
            curr = test[i:i + shap_input_length]
            curr_date = int(curr.time_index[0].timestamp())
            curr_values = curr.random_component_values(copy=False)
            data.append(curr_values.flatten("F"))
            if first_iter:
                for ii in range(test.n_components):
                    columns.extend(["Step " + str(i) + " of comp. " + str(id_l[ii]) for i in range(shap_input_length)])
                    median_of_train = statistics.median(map(lambda x : x.median(axis=0).values()[0,0], train))
                    background.extend([median_of_train for _ in range(shap_input_length)])
            if past_covs != None:
                for ii in range(past_covs.n_components):
                    #TODO: Currently assuming worst case scenario (autoregression for all points)
                    #Fix this to only give the model exactly what it needs
                    # print("PAST")
                    # print(i)
                    # print(test.time_index[i])
                    # print(i + shap_input_length)
                    # print(test.time_index[i + shap_input_length])
                    # print(past_covs.univariate_component(ii)[test.time_index[i]:test.time_index[i + shap_input_length]])
                    # print(past_covs.univariate_component(ii)[test.time_index[i]:test.time_index[i + shap_input_length]].random_component_values(copy=False).flatten())
                    data[-1] = np.concatenate([data[-1], past_covs.univariate_component(ii)[test.time_index[i]:test.time_index[i + shap_input_length + shap_output_length - 1]].random_component_values(copy=False).flatten()])
                    if first_iter:
                        columns.extend(["Step " + str(iii) + " of past cov " + str(id_l_past_covs[ii]) for iii in range(shap_input_length + shap_output_length)])
                        background.extend([past_covs.univariate_component(ii).median(axis=0).values()[0,0] for _ in range(shap_input_length  + shap_output_length)])
            if future_covs != None:
                for ii in range(future_covs.n_components):
                    # print("FUTURE")
                    # print(i)
                    # print(test.time_index[i])
                    # print(i + shap_input_length + shap_output_length)
                    # print(test.time_index[i + shap_input_length + shap_output_length])
                    data[-1] = np.concatenate([data[-1], future_covs.univariate_component(ii)[test.time_index[i]:test.time_index[i + shap_input_length + 2*shap_output_length - 1]].random_component_values(copy=False).flatten()])
                    if first_iter:
                        columns.extend(["Step " + str(iii) + " of fut. cov " + str(id_l_future_covs[ii]) for iii in range(shap_input_length + 2*shap_output_length)])
                        background.extend([future_covs.univariate_component(ii).median(axis=0).values()[0,0] for _ in range(shap_input_length + 2*shap_output_length)])
            data[-1] = np.concatenate([data[-1], [curr_date]])
            if first_iter:
                columns.extend(["Datetime"])
                background.extend([curr_date])
            first_iter = False
        except:
            curr_date = str(test.time_index[i].timestamp())
            print(f"Failed generating sample with date {curr_date}. Skipping to next sample...")
    # print(columns)
    data = pd.DataFrame(data, columns=columns)
    background = pd.DataFrame([background], columns=columns)
    return data, background

def predict(x: darts.TimeSeries,
            n_past_covs: int,
            n_future_covs: int,
            n_comp_series: int, 
            shap_input_length: int,
            shap_output_length: int,
            model,
            scaler_list: List[darts.dataprocessing.transformers.Scaler],
            scaler_past_covariates,
            scaler_future_covariates,
            scale: bool = True,
            num_samples: int = 1):
    """
    The function to be given to KernelExplainer, in order for it to produce predictions
    for all samples of x. These samples have the format described in the above function. Also,
    this function scales the data if necessary and is called multiple times by the SHAP explainer
    Parameters
    ----------
    x
        The dataset of samples to be predicted.
    n_past_covs
        The number of past covariates the model was trained on. This is needed to produce the
        timeseries to be given to the models predict method.
    n_future_covs
        The number of future covariates the model was trained on.
    shap_input_length
        The length of each sample of the dataset. Also, the input_chunk_length of the model if 
        it exists as a parameter. It can also be set by the user
    shap_output_length
        The length of each sample of the result. Also, the forecast horizon used for the model.
    model
        The model to be explained.
    scaler_list
        The list of scalers that were used. First, the training data scaler is listed. Then, all
        the covariate scalers are listed in the order in which they appear in x.
    scale
        Whether to scale the data and the covariates
    Returns
    -------
    numpy.array
        The numpy array of the model's results. Its line number is equal to the number of samples
        provided, and its column number is equal to shap_output_length.
    """

    series_list = []
    past_covs_list = []
    future_covs_list = []
    for sample in x:
        index = [pd.to_datetime(sample[-1], unit='s').tz_localize(None) + pd.offsets.DateOffset(hours=i) for i in range(shap_input_length)]
        index_past = [pd.to_datetime(sample[-1], unit='s').tz_localize(None) + pd.offsets.DateOffset(hours=i) for i in range(shap_input_length + shap_output_length)]
        index_future = [pd.to_datetime(sample[-1], unit='s').tz_localize(None) + pd.offsets.DateOffset(hours=i) for i in range(shap_input_length + 2*shap_output_length)]
        sample = np.array(sample, dtype=np.float32)
        for i in range(n_comp_series):
            data = sample[i*shap_input_length:(i+1)*shap_input_length]
            if i == 0:
                ts = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index, columns=["Value"]))
            else:
                ts = ts.stack(TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index, columns=["Value"])))
        if scale:
            ts = scaler_list[0].transform(ts)
        past_covs = None
        future_covs = None
        for i in range(shap_input_length*n_comp_series, shap_input_length*n_comp_series + (shap_output_length + shap_input_length)*(n_past_covs), shap_output_length + shap_input_length):
            data = sample[i:i+shap_output_length + shap_input_length]
            if i == shap_input_length*n_comp_series:
                past_covs = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index_past, columns=["Covariate"]))
            else:
                temp = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index_past, columns=["Covariate"]))
                past_covs = past_covs.stack(temp)
        if scaler_past_covariates[0] != None:
            past_covs = scaler_past_covariates[0].transform(past_covs)

        if past_covs == None: 
            past_covs_list = None 
        else:
            past_covs_list.append(past_covs)
        for i in range(shap_input_length*n_comp_series + (shap_output_length + shap_input_length)*(n_past_covs), shap_input_length*n_comp_series + (shap_output_length + shap_input_length)*(n_past_covs) + (shap_input_length + 2*shap_output_length)*n_future_covs, shap_input_length + 2*shap_output_length):
            data = sample[i:i+shap_input_length+2*shap_output_length]
            if i == shap_input_length*n_comp_series + (shap_output_length + shap_input_length)*(n_past_covs):
                future_covs = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index_future, columns=["Covariate"]))
            else:
                temp = TimeSeries.from_dataframe(pd.DataFrame(data=data, index=index_future, columns=["Covariate"]))
                future_covs = future_covs.stack(temp)
            
        if scaler_future_covariates[0] != None:
            future_covs = scaler_future_covariates[0].transform(future_covs)
            
        if future_covs == None:
            future_covs_list = None
        else:
            future_covs_list.append(future_covs)
        series_list.append(ts)
    try:
        res = model.predict(shap_output_length, series_list, past_covariates=past_covs_list, future_covariates=future_covs_list, num_samples=num_samples)
    except:
        res = model.predict(shap_output_length, series_list, num_samples=num_samples)
    if scale:
        res = list(map(lambda elem : scaler_list[0].inverse_transform(elem), res))

    res = list(map(lambda elem: elem.random_component_values(copy=False).flatten(), res))
    return np.array(res)
#lambda x: model_rnn.predict(TimeSeries.from_dataframe(pd.DataFrame(index=(x[-1] + pd.offsets.DateOffset(hours=i) for i in range(96)), data=x[:-1])))

def bar_plot_store_json(shap_values, data, filename):
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(20, len(feature_order)):]
    feature_inds = feature_order[:20]
    feature_inds = reversed(feature_inds)
    feature_names = data.columns
    global_shap_values = np.abs(shap_values).mean(0)
    bar_plot_dict = {}
    for i in feature_inds:
        bar_plot_dict[feature_names[i]] = global_shap_values[i]
    with open(filename, "w") as out:
        json.dump(bar_plot_dict, out)

def call_shap(n_past_covs: int,
              n_future_covs: int,
              n_comp_series: int, 
              shap_input_length: int,
              shap_output_length: int,
              model,
              scaler_list: List[darts.dataprocessing.transformers.Scaler],
              scaler_past_covariates,
              scaler_future_covariates,
              background: darts.TimeSeries,
              data: darts.TimeSeries,
              scale: bool = True,
              num_samples: int = 1,
              id_l = None,
              id_l_past_covs = None,
              id_l_future_covs = None):
    """
    The function that calls KernelExplainer, and stores to the MLflow server
    some plots explaining the output of the model. More specifficaly, ...
    Parameters
    ----------
    n_past_covs
        The number of past covariates the model was trained on. This is needed to produce the
        timeseries to be given to the models predict method.
    n_future_covs
        The number of future covariates the model was trained on.
    shap_input_length
        The length of each sample of the dataset. Also, the input_chunk_length of the model if 
        it exists as a parameter. It can also be set by the user
    shap_output_length
        The length of each sample of the result. Also, the forecast horizon used for the model. 
    model
        The model to be explained.
    scaler_list
        The list of scalers that were used. First, the training data scaler is listed. Then, all
        the covariate scalers are listed in the order in which they appear in x.
    scale
        Whether to scale the data and the covariates
    background
        The sample that provides the values that replace the data's values that are simulated to be
        missing
    data
        The samples to be tested
    """

    shap.initjs()
    explainer = shap.KernelExplainer(lambda x : predict(x, 
                                                        n_past_covs, 
                                                        n_future_covs,
                                                        n_comp_series,
                                                        shap_input_length, 
                                                        shap_output_length, 
                                                        model, 
                                                        scaler_list,
                                                        scaler_past_covariates,
                                                        scaler_future_covariates,
                                                        scale), background, num_samples=num_samples)

    shap_values = explainer.shap_values(data, nsamples="auto", normalize=False)
    plt.close()
    interprtmpdir = tempfile.mkdtemp()
    sample = random.randint(0, len(data) - 1)
    for no_comp in range(n_comp_series):
        for out in [0, shap_output_length//2, shap_output_length-1]:
            shap.summary_plot(shap_values[:, : , no_comp*shap_output_length + out], data, show=False)
            plt.savefig(f"{interprtmpdir}/summary_plot_all_samples_out_{out}_comp_{id_l[no_comp]}.png")
            plt.close()
            os.remove(f"{interprtmpdir}/summary_plot_all_samples_out_{out}_comp_{id_l[no_comp]}.png")
            shap.summary_plot(shap_values[:, :, no_comp*shap_output_length + out], data, show=False)
            plt.savefig(f"{interprtmpdir}/summary_plot_all_samples_out_{out}_comp_{id_l[no_comp]}.png")
            plt.close()
            shap.summary_plot(shap_values[:, :, no_comp*shap_output_length + out], data, plot_type='bar', show=False)
            plt.savefig(f"{interprtmpdir}/summary_plot_bar_graph_out_{out}_comp_{id_l[no_comp]}.png")
            plt.close()
            bar_plot_store_json(shap_values[:, :, no_comp*shap_output_length + out], data, f"{interprtmpdir}/summary_plot_bar_data_out_{out}_comp_{id_l[no_comp]}.json")
            shap.force_plot(explainer.expected_value[no_comp*shap_output_length + out],shap_values[:, :, no_comp*shap_output_length + out][sample,:], data.iloc[sample,:],  matplotlib = True, show = False)
            str_ = f"{interprtmpdir}/force_plot_of_{sample}_sample_starting_at_{str(pd.to_datetime(data.iloc[sample].iloc[-1], unit='s').tz_localize(None)).replace(":", "_")}_{out}_output_comp_{id_l[no_comp]}.png"
            plt.savefig(str_)
            plt.close()

    print("\nUploading SHAP interpretation results to MLflow server...")
    logging.info("\nUploading SHAP interpretation results to MLflow server...")

    mlflow.log_artifacts(interprtmpdir, "interpretation")

@multi_asset(
    name="evaluation_asset",
    description="For evaluation of the results",
    group_name='deepTSF_pipeline',
    required_resource_keys={"config"},
    ins={'start_pipeline_run': AssetIn(key='start_pipeline_run', dagster_type=str),
         'training_and_hyperparameter_tuning_out': AssetIn(key='training_and_hyperparameter_tuning_out', dagster_type=dict)},
    outs={"evaluation_out": AssetOut(dagster_type=dict)})

def evaluation_asset(context, start_pipeline_run, training_and_hyperparameter_tuning_out):
    # TODO: Validate evaluation step for all models. It is mainly tailored for the RNNModel for now.

    model_uri = training_and_hyperparameter_tuning_out["model_uri"]
    if none_checker(model_uri) == None:
        print(f'\nNo model in input. Skipping Evaluation')
        logging.info(f'\nNo model in input. Skipping Evaluation')
        return Output({"run_completed": True,})
    
    model_type = training_and_hyperparameter_tuning_out["model_type"]
    series_uri = training_and_hyperparameter_tuning_out["series_uri"]
    future_covs_uri = training_and_hyperparameter_tuning_out["future_covariates_uri"]
    past_covs_uri = training_and_hyperparameter_tuning_out["past_covariates_uri"]
    scaler_uri = training_and_hyperparameter_tuning_out["scaler_uri"]
    scaler_past_covariates_uri = training_and_hyperparameter_tuning_out["scaler_past_covariates_uri"]
    scaler_future_covariates_uri = training_and_hyperparameter_tuning_out["scaler_future_covariates_uri"]
    setup_uri = training_and_hyperparameter_tuning_out["setup_uri"]
    shap_input_length = training_and_hyperparameter_tuning_out["shap_input_length"]
    retrain = training_and_hyperparameter_tuning_out["retrain"]
    cut_date_test = training_and_hyperparameter_tuning_out["cut_date_test"]
    test_end_date = training_and_hyperparameter_tuning_out["test_end_date"]

    config = context.resources.config
    #TODO Maybe remove mode?
    mode = 'remote'
    forecast_horizon = config.forecast_horizon
    stride = config.stride
    shap_output_length = forecast_horizon
    size = config.shap_data_size
    analyze_with_shap = config.analyze_with_shap
    multiple = config.multiple
    eval_series = config.eval_series
    cut_date_val = config.cut_date_val
    resolution = config.resolution
    eval_method = config.eval_method
    evaluate_all_ts = config.evaluate_all_ts
    m_mase = config.m_mase
    num_samples = config.num_samples
    pv_ensemble = config.pv_ensemble
    format = config.format
    experiment_name = config.experiment_name
    darts_model = config.darts_model
    parent_run_name = config.parent_run_name if none_checker(config.parent_run_name) != None else darts_model + '_pipeline'


    evaltmpdir = tempfile.mkdtemp()

    # Argument processing
    test_end_date = none_checker(test_end_date)
    stride = none_checker(stride)
    forecast_horizon = int(forecast_horizon)
    m_mase = int(m_mase)
    num_samples = int(num_samples)
    stride = int(forecast_horizon) if stride is None or stride == -1 else int(stride)
    retrain = retrain
    pv_ensemble = pv_ensemble
    analyze_with_shap = analyze_with_shap
    multiple = multiple
    future_covariates_uri = none_checker(future_covs_uri)
    past_covariates_uri = none_checker(past_covs_uri)
    evaluate_all_ts = evaluate_all_ts
    shap_input_length = none_checker(shap_input_length)
    try:
        size = int(size)
    except:
        size = float(size)
    try:
        shap_input_length = int(shap_input_length)
    except:
        pass
    shap_output_length = int(shap_output_length)

    # Load model / datasets / scalers from Mlflow server

    ## load series from MLflow
    series_path = download_online_file(
        client, series_uri, "series.csv") if mode == 'remote' else series_uri
    series, id_l, ts_id_l = load_local_csv_or_df_as_darts_timeseries(
        local_path_or_df=series_path,
        last_date=test_end_date,
        multiple=multiple,
        resolution=resolution,
        format=format)
    
    series_transformed = series.copy()
    # plot_series(df_list=[series_transformed[0]], 
    #                 ts_name_list=["series_transformed"], 
    #                 save_dir=os.path.join(f"{evaltmpdir}",
    #                                     f'series_transformed_start.html'))
    
    if pv_ensemble:
        print("\nSubtracting pv forecast from series to be fed to model")
        logging.info("\nSubtracting pv forecast from series to be fed to model")

        for i in range(len(series_transformed)):
            print(get_pv_forecast([], 
                                                            start=series_transformed[i].pd_dataframe().index[0], 
                                                            end=series_transformed[i].pd_dataframe().index[-1], 
                                                            inference=False, 
                                                            kW=60, 
                                                            use_saved=True))
            print(series_transformed[i])
            series_transformed[i] = series_transformed[i] + get_pv_forecast([], 
                                                            start=series_transformed[i].pd_dataframe().index[0], 
                                                            end=series_transformed[i].pd_dataframe().index[-1], 
                                                            inference=False, 
                                                            kW=60, 
                                                            use_saved=True)
        
            # plot_series(df_list=[series_transformed[0]], 
            #         ts_name_list=["series_transformed"], 
            #         save_dir=os.path.join(f"{evaltmpdir}",
            #                             f'series_transformed_no_pv.html'))



    if future_covariates_uri is not None:
        future_covs_path = download_online_file(
            client, future_covariates_uri, "future_covariates.csv") if mode == 'remote' else future_covariates_uri
        future_covariates, id_l_future_covs, ts_id_l_future_covs = load_local_csv_or_df_as_darts_timeseries(
            local_path_or_df=future_covs_path,
            last_date=test_end_date,
            multiple=True,
            resolution=resolution,
            format=format)
    else:
        future_covariates = None

    if past_covariates_uri is not None:
        past_covs_path = download_online_file(
            client, past_covariates_uri, "past_covariates.csv") if mode == 'remote' else past_covariates_uri
        past_covariates, id_l_past_covs, ts_id_l_past_covs = load_local_csv_or_df_as_darts_timeseries(
            local_path_or_df=past_covs_path,
            last_date=test_end_date,
            multiple=True,
            resolution=resolution,
            format=format)
    else:
        past_covariates = None

    # TODO: Also implement for local files -> Done?
    ## load model from MLflow
    model = load_model(client, model_uri, mode)
    scaler = load_scaler(scaler_uri=none_checker(scaler_uri), mode=mode)
    scaler_future_covariates = load_scaler(scaler_uri=none_checker(scaler_future_covariates_uri), mode=mode)
    scaler_past_covariates = load_scaler(scaler_uri=none_checker(scaler_past_covariates_uri), mode=mode)

    if scaler is not None:
        if not multiple:
            series_transformed = scaler.transform(series_transformed)
        else:
            series_transformed = [scaler[i].transform(series_transformed[i]) for i in range(len(series_transformed))]
    elif not pv_ensemble:
        series_transformed = series

    # Split in the same way as in training
    ## series
    series_split = split_dataset(
            series,
            val_start_date_str=cut_date_test,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format)

    series_transformed_split = split_dataset(
            series_transformed,
            val_start_date_str=cut_date_test,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format)


    if multiple:
        eval_i = -1
        if eval_method == "ts_ID":
            for i, comps in enumerate(ts_id_l):
                for comp in comps:
                    if eval_series == str(comp):
                        eval_i = i
        else:
            for i, comps in enumerate(id_l):
                for comp in comps:
                    if eval_series == str(comp):
                        eval_i = i
    else:
        eval_i = 0

    if eval_i == -1 and evaluate_all_ts==False:
        raise EvalSeriesNotFound(eval_series)
    # Evaluate Model
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"mlflow.runName": parent_run_name}, run_id=start_pipeline_run) as parent_run:
        with mlflow.start_run(tags={"mlflow.runName": "eval"}, nested=True) as mlrun:
            mlflow.set_tag("run_id", mlrun.info.run_id)
            mlflow.set_tag("stage", "evaluation")
            if evaluate_all_ts and multiple:
                eval_results = {}
                ts_n = len(ts_id_l)
                for eval_i in range(ts_n):
                    backtest_series_transformed = series_transformed_split['all'] if not multiple else series_transformed_split['all'][eval_i]
                    print(f"Testing timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
                    logging.info(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
                    print(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")
                    logging.info(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")
                    print("")
                    evaluation_results = backtester(model=model,
                                                series_transformed=backtest_series_transformed,
                                                series=series_split['all'] if not multiple else series_split['all'][eval_i],
                                                transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                                test_start_date=cut_date_test,
                                                forecast_horizon=forecast_horizon,
                                                stride=stride,
                                                retrain=retrain,
                                                future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                                past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                                path_to_save_backtest=f"{evaltmpdir}/{ts_id_l[eval_i][0]}",
                                                m_mase=m_mase,
                                                num_samples=num_samples,
                                                pv_ensemble=pv_ensemble,
                                                resolution=resolution,
                                                id_l=None if not multiple else id_l[eval_i])
                    eval_results[eval_i] = [str(ts_id_l[eval_i][0])] + [evaluation_results["metrics"]["smape"],
                                                                                evaluation_results["metrics"]["mase"],
                                                                                evaluation_results["metrics"]["mae"],
                                                                                evaluation_results["metrics"]["rmse"],
                                                                                evaluation_results["metrics"]["mape"],
                                                                                evaluation_results["metrics"]["nrmse_min_max"],
                                                                                evaluation_results["metrics"]["nrmse_mean"]]

                eval_results = pd.DataFrame.from_dict(eval_results, orient='index', columns=["Timeseries ID", "smape", "mase", "mae", "rmse", "mape", "nrmse_min_max", "nrmse_mean"])
                save_path = f"{evaltmpdir}/evaluation_results_all_ts.csv"
                eval_results.to_csv(save_path)
                evaluation_results["metrics"] = eval_results.mean(axis=0, numeric_only=True).to_dict()

                if analyze_with_shap:
                    print(f"Can not analye with SHAP as user chose to evaluate all time series. SHAP only supports analysis of a single test time series")
                    logging.info(f"Can not analye with SHAP as user chose to evaluate all time series. SHAP only supports analysis of a single test time series")

            else:
                if multiple:
                    print(f"Testing timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}")
                    logging.info(f"Testing timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}")

                backtest_series_transformed = series_transformed_split['all'] if not multiple else series_transformed_split['all'][eval_i]
                print(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")
                logging.info(f"Testing from {pd.Timestamp(cut_date_test)} to {backtest_series_transformed.time_index[-1]}...")

                evaluation_results = backtester(model=model,
                                                series_transformed=backtest_series_transformed,
                                                series=series_split['all'] if not multiple else series_split['all'][eval_i],
                                                transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                                test_start_date=cut_date_test,
                                                forecast_horizon=forecast_horizon,
                                                stride=stride,
                                                retrain=retrain,
                                                future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                                past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                                path_to_save_backtest=evaltmpdir,
                                                m_mase=m_mase,
                                                num_samples=num_samples,
                                                pv_ensemble=pv_ensemble,
                                                resolution=resolution,
                                                id_l=None if not multiple else id_l[eval_i])
                if analyze_with_shap:

                    if shap_input_length == None or shap_input_length == -1:
                        raise ValueError(f"The model that was chosen does not support parameter input_chunk_length, and therefore needs shap_input_length to be defined explicitelly")
                    data, background = build_shap_dataset(size=size,
                                                    train=series_split['train'],
                                                    test=series_split['test']\
                                                        if not multiple else series_split['test'][eval_i],
                                                    shap_input_length=shap_input_length,
                                                    shap_output_length=shap_output_length,
                                                    future_covs=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                                    past_covs=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                                    id_l=[0] if not multiple else id_l[eval_i],
                                                    id_l_past_covs=None if past_covariates == None else (id_l_past_covs[0] if not multiple else id_l_past_covs[eval_i]),
                                                    id_l_future_covs=None if future_covariates == None else (id_l_future_covs[0] if not multiple else id_l_future_covs[eval_i]))


                    #TODO check SHAP with covariates
                    call_shap(n_past_covs=0 if past_covariates == None else past_covariates[eval_i].n_components,
                        n_future_covs=0 if future_covariates == None else future_covariates[eval_i].n_components,
                        n_comp_series=series_split['test'][eval_i].n_components if multiple else series_split['test'].n_components,
                        shap_input_length=shap_input_length,
                        shap_output_length=shap_output_length,
                        model=model,
                        scaler_list=[scaler if (not multiple or (scaler == None)) else scaler[eval_i]],
                        scaler_future_covariates=[None if scaler_future_covariates == None else scaler_future_covariates[0] if not multiple else scaler_future_covariates[eval_i]],
                        scaler_past_covariates=[None if scaler_past_covariates == None else scaler_past_covariates[0] if not multiple else scaler_past_covariates[eval_i]],
                        background=background,
                        data=data,
                        scale=(scaler != None),
                        num_samples=num_samples,
                        id_l=[0] if not multiple else id_l[eval_i],
                        id_l_past_covs=None if past_covariates == None else (id_l_past_covs[0] if not multiple else id_l_past_covs[eval_i]),
                        id_l_future_covs=None if future_covariates == None else (id_l_future_covs[0] if not multiple else id_l_future_covs[eval_i]))
            # if not multiple:
            #     series_split['test'].to_csv(
            #             os.path.join(evaltmpdir, "test.csv"))
            # else:
            #     multiple_dfs_to_ts_file(series_split['test'], id_l, ts_id_l, os.path.join(evaltmpdir, "test.csv"))

            print("\nUploading evaluation results to MLflow server...")
            logging.info("\nUploading evaluation results to MLflow server...")

            mlflow.log_metrics(evaluation_results["metrics"])
            mlflow.log_artifacts(evaltmpdir, "eval_results")

            print("\nArtifacts uploaded. Deleting local copies...")
            logging.info("\nArtifacts uploaded. Deleting local copies...")

            print("\nEvaluation succesful.\n")
            logging.info("\nEvaluation succesful.\n")

            # Set tags
            mlflow.set_tag("run_id", mlrun.info.run_id)

            curr_run_id = mlrun.info.run_id
            
        completed_run = mlflow.tracking.MlflowClient().get_run(curr_run_id)
        mlflow.log_metrics(completed_run.data.metrics)

        return Output({"run_completed": True,})