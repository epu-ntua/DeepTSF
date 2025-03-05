FROM conda-base:latest

# Activate the conda environment
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

# Install additional dependencies
RUN pip install dagster==1.8.13 dagster-webserver==1.8.13 dagster-shell==0.24.13 dagster-mlflow==0.24.13 dagster-celery==0.24.13 dagster-postgres==0.24.13 dagster-aws==0.24.13
RUN pip install protobuf==3.20.*
RUN pip install minio

EXPOSE 8006

COPY . .

WORKDIR /app/dagster_deeptsf
# Set the DAGSTER_HOME environment variable
ENV DAGSTER_HOME /app/dagster_deeptsf

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "dagster", "dev", "-h", "0.0.0.0", "-p", "8006"]
