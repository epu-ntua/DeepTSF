FROM conda-base:latest

SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

RUN pip install dagster==1.8.13 dagster-webserver==1.8.13 dagster-shell==0.24.13 dagster-mlflow==0.24.13 dagster-celery==0.24.13 dagster-postgres==0.24.13 dagster-aws==0.24.13

COPY . .

WORKDIR /app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "celery", "-A", "celery_DeepTSF.tasks", "worker", "--loglevel=debug", "--queues=dagster"]