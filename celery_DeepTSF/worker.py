from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL= os.environ.get("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND")
# Configure Celery to use Redis as the broker
celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,  # Update this URL with your Redis instance
    backend=CELERY_RESULT_BACKEND  # Using Redis as the result backend for tracking
)