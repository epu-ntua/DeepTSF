FROM python:3.12

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update

RUN apt install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

WORKDIR /app 

# COPY ./celery_DeepTSF/conda_worker.yaml ./conda_worker.yaml
COPY ./conda.yaml ./conda_worker.yaml

RUN conda update conda

RUN conda env create -f "./conda_worker.yaml"

RUN conda init bash
# Activate the conda environment
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

RUN pip install dagster==1.8.13 dagster-webserver==1.8.13 dagster-shell==0.24.13 dagster-mlflow==0.24.13 dagster-celery==0.24.13 dagster-postgres==0.24.13 dagster-aws==0.24.13

COPY . .

WORKDIR /app/dagster-deeptsf/dagster_deeptsf

CMD ["conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "dagster-celery", "worker", "start", "-A", "dagster_celery.app", "-y" ,"../../celery_DeepTSF/celery_confing.yaml", "--loglevel=DEBUG"]
