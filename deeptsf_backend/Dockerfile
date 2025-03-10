FROM conda-base:latest

COPY . .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

RUN pip install dagster-celery==0.24.13 pydantic-settings==2.8.1 uvicorn[standard] cryptography

RUN conda env export > environment.yml
 
# RUN echo "conda activate DeepTSF_env" > ~/.bashrc

# RUN python -c "import torch: print(torch.cuda.is_available())"

# RUN /root/miniconda3/envs/DeepTSF_env/bin/uvicorn api:app --host 0.0.0.0 --port 8080

# RUN conda list uvicorn

# RUN /root/miniconda3/envs/DeepTSF_env/uvicorn api:app --host 0.0.0.0 --port 8080

# RUN export MLFLOW_TRACKING_URI="http://localhost:5000" # maybe redundant as I use the .env file in docker-compose

# RUN pip install python-multipart

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

# ENTRYPOINT ["tail", "-f", "/dev/null"]