
from dotenv import load_dotenv
import tempfile
import pretty_errors
import os
import mlflow
import pandas as pd
import yaml
import darts
from pandas.tseries.frequencies import to_offset
from math import ceil
cur_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
load_dotenv()
from tqdm import tqdm
import logging
import json
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)
import requests
from datetime import date
import pvlib
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
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
from minio import S3Error
from minio.commonconfig import CopySource

def move_object(minio_client, source_bucket, source_object, dest_bucket, dest_object):
    try:
        # Copy the object from the source to the destination.
        # The copy source format is "/<source_bucket>/<source_object>".
        copy_result = minio_client.copy_object(
            dest_bucket,
            dest_object,
            CopySource(source_bucket, source_object)
        )
        print(f"Copied {source_bucket}/{source_object} to {dest_bucket}/{dest_object}")
    except S3Error as err:
        print(f"Error during copy operation: {err}")
        return

    try:
        # Delete the original object after the copy succeeds.
        minio_client.remove_object(source_bucket, source_object)
        print(f"Deleted original object: {source_bucket}/{source_object}")
    except S3Error as err:
        print(f"Error during delete operation: {err}")


def download_file_from_s3_bucket(object_name, dst_filename, dst_dir=None, bucketName='mlflow-bucket'):
    import boto3
    import tempfile
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    s3_resource = boto3.resource(
        's3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(bucketName)
    local_path = os.path.join(dst_dir, dst_filename)
    bucket.download_file(object_name, local_path)
    return local_path

def upload_file_to_minio(bucket_name, file_path, csv_name, client):
    try:
        # Ensure the bucket exists
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
        
        # Upload the file
        client.fput_object(bucket_name, csv_name, file_path)
        print(f"File '{csv_name}' uploaded successfully to bucket '{bucket_name}'.")
    except S3Error as e:
        print(f"Error uploading file: {e}")


def download_online_file(client, url, dst_filename=None, dst_dir=None, bucket_name='mlflow-bucket'):
    import sys
    import tempfile
    import requests
    print("Donwloading_online_file")
    print(url)
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    if dst_filename is None:
        dst_filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, dst_filename)
    url = url.split(bucket_name)[-1]
    client.fget_object(bucket_name, url, filepath)
    # print(req)
    # if req.status_code != 200:
    #     raise Exception(f"\nResponse is not 200\nProblem downloading: {url}")
    #     sys.exit()
    # url_content = req.content
    # filepath = os.path.join(dst_dir, dst_filename)
    # file = open(filepath, 'wb')
    # file.write(url_content)
    # file.close()
    return filepath

def load_local_pkl_as_object(local_path):
    import pickle
    pkl_object = pickle.load(open(local_path, "rb"))
    return pkl_object

def truth_checker(argument):
    """ Returns True if string has specific truth values else False"""
    return argument.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'on']


def none_checker(argument):
    """ Returns True if string has specific truth values else False"""
    if argument is None:
        return None
    elif type(argument) == str and argument.lower() in ['none', 'nope', 'nan', 'na', 'null', 'nope', 'n/a', 'mlflow_artifact_uri']:
        return None
    elif type(argument) == dict and argument == {"insert key": "insert value"}:
        return None
    else:
        return argument


def to_seconds(resolution):
    return ceil(pd.to_timedelta(to_offset(resolution)).total_seconds())

def to_standard_form(freq):

    total_seconds = int(freq.total_seconds())

    if total_seconds % 86400 == 0:
        if total_seconds // 86400 == 1:
            return '1d'  # Daily frequency
        else:
            return f'{total_seconds // 86400}d'
    elif total_seconds % 3600 == 0:
        if total_seconds // 3600 == 1:
            return '1h'  # Hourly frequency
        else:
            return f'{total_seconds // 3600}h'
    elif total_seconds % 60 == 0:
        if total_seconds // 60 == 1:
            return '1min'  # Minutely frequency
        else:
            return f'{total_seconds // 60}min'
    else:
        return f'{total_seconds}s'  # Secondly frequency


def change_form(freq, change_format_to="pandas_form"):
    import re

    # Dictionary to map time units from short to long forms and vice versa
    time_units = {
        "s": "second",
        "min": "minute",
        "h": "hour",
        "d": "day"
    }
    
    # Identify the number and unit from the frequency
    match = re.match(r"(\d+)?(\w+)", freq)
    if not match:
        raise ValueError("Invalid frequency format.")
    
    number, unit = match.groups()

    if not number:
      number = 1
    
    # Convert to the desired format
    if change_format_to == "print_form":
        # From pandas form (e.g., '1h') to human-readable form (e.g., '1 hour')
        full_unit = time_units.get(unit, "unknown")  # Default to 'unknown' if unit not found
        if int(number) > 1:
            full_unit += 's'  # Make plural if more than one
        return f"{number} {full_unit}"
    elif change_format_to == "pandas_form":
        # From human-readable form (e.g., '1 hour') to pandas form (e.g., '1h')
        for short, long in time_units.items():
            if long in freq:
                # Check if the unit matches and convert it
                if ' ' in freq:
                    num, _ = freq.split()
                return f"{num}{short}"
    else:
        raise ValueError("Invalid change_format_to value. Use 'pandas_form' or 'print_form'.")

def make_time_list(resolution):
    import re

    # List of all supported resolutions in increasing order
    all_resolutions = ["1s", "2s", "5s", "15s", "30s", "1min", "2min", "5min", "15min", "30min", "1h", "2h", "6h", "1d", "2d", "5d", "10d"]

    input_seconds = to_seconds(resolution)

    # Filter and convert resolutions
    resolutions = [{"value": change_form(resolution, change_format_to="print_form"), "default": True}]
    for res in all_resolutions:
        if to_seconds(res) > input_seconds:
            resolutions.append({"value": change_form(res, "print_form"), "default": False})
    return resolutions