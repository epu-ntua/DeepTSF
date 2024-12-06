# miniconda base image
FROM continuumio/miniconda3:23.5.2-0

# Set the working directory inside the container
WORKDIR /app

# Copy environment setup file
COPY ../../conda.yaml .

# Create the conda environment from the env conda.yaml file
RUN conda env create -f conda.yaml

# Activate the conda environment
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

# Install additional dependencies
RUN pip install dagster==1.8.13 dagster-webserver==1.8.13 dagster-shell==0.24.13 dagster-mlflow==0.24.13
RUN pip install protobuf==3.20.*
RUN pip install minio

EXPOSE 8006

COPY . .

WORKDIR /app/dagster-deeptsf
# Set the DAGSTER_HOME environment variable
ENV DAGSTER_HOME /app/dagster-deeptsf

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "dagster", "dev", "-h", "0.0.0.0", "-p", "8006"]
