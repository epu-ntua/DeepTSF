# tasks.py
import os
import tempfile
import datetime
import sys
from celery_DeepTSF.worker import celery_app
from datetime import timedelta
from fastapi import HTTPException
from uc2.load_raw_data import read_and_validate_input
from utils import make_time_list
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

def csv_validator(fname: str, multiple: bool, allow_empty_series=False, format='long', task=None):

    fileExtension = fname.split(".")[-1].lower() == "csv"
    if not fileExtension:
        print("Unsupported file type provided. Please upload CSV file")
        raise HTTPException(status_code=415, detail="Unsupported file type provided. Please upload CSV file")
    try:
        ts, resolution = read_and_validate_input(series_csv=fname, 
                                                 multiple=multiple, allow_empty_series=allow_empty_series, 
                                                 format=format, log_to_mlflow=False, task=task)
    except Exception as e:
        print(f"There was an error validating the file: {e}")
        raise HTTPException(status_code=415, detail=f"There was an error validating the file: {e}")
    
    resolutions = make_time_list(resolution=resolution)    
    return ts, resolutions


@celery_app.task(bind=True, track_started=True)
def upload_and_validate_csv(self, contents: str, filename: str, multiple: bool, format: str):

    # Store uploaded dataset to worker
    try:
        # write locally
        local_dir = tempfile.mkdtemp()
        fname = os.path.join(local_dir, filename)
        with open(fname, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=415, detail="There was an error uploading the file to worker")
    finally:
        print(f'\n{fname}\n')

    print("Validating file...") 
    ts, resolutions = csv_validator(fname, multiple, format=format, task=self)

    if multiple:
        if format == "long":
            dataset_start_multiple = ts.iloc[0]['Datetime']
            dataset_end_multiple = ts.iloc[-1]['Datetime']
        else:
            dataset_start_multiple = ts.iloc[0]['Date']
            dataset_end_multiple = ts.iloc[-1]['Date']
    
    return {"message": "Validation successful", 
            "fname": filename,
            "dataset_start": datetime.datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else str(dataset_start_multiple),
            "allowed_validation_start": datetime.datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else str(dataset_start_multiple + timedelta(days=10)),
            "dataset_end": datetime.datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else str(dataset_end_multiple),
            "allowed_resolutions": resolutions,
            "ts_used_id": None,
            "evaluate_all_ts": True if multiple else None
            }
