# tasks.py
import os
import tempfile
import datetime
import sys
from celery_DeepTSF.worker import celery_app
from datetime import timedelta
from fastapi import HTTPException
from uc2.load_raw_data import read_and_validate_input
#change utils to dagster utils
from utils import make_time_list, download_online_file, truth_checker, move_object
from celery import Celery
import os
from dotenv import load_dotenv
from dagster_celery.tasks import create_task
from minio import Minio

load_dotenv()
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)

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


@celery_app.task(bind=True, track_started=True, queue="dagster")
def upload_and_validate_csv(self, filename: str, multiple: bool, format: str):

    # Download uploaded dataset from s3
    try:
        local_dir = tempfile.mkdtemp()
        download_online_file(client, f'unvalidated/{filename}', dst_dir=local_dir, bucket_name='dataset-storage')
    except Exception:
        raise HTTPException(status_code=415, detail="There was an error uploading the file to worker")
    

    print("Validating file...") 
    ts, resolutions = csv_validator(local_dir + "/" + filename, multiple, format=format, task=self)

    if multiple:
        if format == "long":
            dataset_start_multiple = ts.iloc[0]['Datetime']
            dataset_end_multiple = ts.iloc[-1]['Datetime']
        else:
            dataset_start_multiple = ts.iloc[0]['Date']
            dataset_end_multiple = ts.iloc[-1]['Date']

    move_object(client, 'dataset-storage', f'unvalidated/{filename}', 'dataset-storage', filename)
    
    return {"message": "Validation successful", 
            "fname": filename,
            "dataset_start": datetime.datetime.strftime(ts.index[0], "%Y-%m-%d") if multiple==False else str(dataset_start_multiple),
            "allowed_validation_start": datetime.datetime.strftime(ts.index[0] + timedelta(days=10), "%Y-%m-%d") if multiple==False else str(dataset_start_multiple + timedelta(days=10)),
            "dataset_end": datetime.datetime.strftime(ts.index[-1], "%Y-%m-%d") if multiple==False else str(dataset_end_multiple),
            "allowed_resolutions": resolutions,
            "ts_used_id": None,
            "evaluate_all_ts": True if multiple else None
            }

execute_plan = create_task(celery_app)

if __name__ == '__main__':
    celery_app.worker_main()