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

COPY ./celery_DeepTSF/conda_worker.yaml ./conda_worker.yaml

RUN conda update conda

RUN conda env create -f "./conda_worker.yaml"

RUN conda init bash

COPY . .

CMD ["conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "celery", "-A", "celery_DeepTSF.tasks", "worker", "--loglevel=DEBUG"]
