from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL= os.environ.get("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND")
# Configure Celery to use Redis as the broker
celery_app = Celery(
    "dagster",
    broker=CELERY_BROKER_URL,  # Update this URL with your Redis instance
    backend=CELERY_RESULT_BACKEND,  # Using Redis as the result backend for tracking
    worker_redirect_stdouts=False
)

# Keep the worker consuming across Redis restarts / dropped connections. Without
# these, a worker can reconnect after a Redis restart yet never resume polling
# the queue, leaving tasks stuck forever (see the liveness probe in the helm chart).
celery_app.conf.update(
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_transport_options={
        "socket_keepalive": True,
        "socket_timeout": 30,
        "socket_connect_timeout": 30,
        "health_check_interval": 30,
        "retry_on_timeout": True,
    },
    worker_cancel_long_running_tasks_on_connection_loss=True,
)