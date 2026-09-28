# DeepTSF test suite

End-to-end tests of the DeepTSF backend that run on a plain machine: no docker,
no MinIO, no MLflow server, no Dagster daemon and no Celery worker are needed.

For about 145 pipeline configurations (every model, four resolutions, single and
multiple series, covariates, calendar features, SHAP, hyperparameter search,
irregular timestamps, resampling, weather covariates), the tests:

1. **run the real Dagster job** (`deepTSF_pipeline`: load data, ETL, training,
   evaluation) and check what it logged to MLflow (`test_pipeline.py`);
2. **check the model it produced**: the MLflow pyfunc model has all its files
   and loads as the expected darts model (`test_inference.py::test_model_artifacts`);
3. **serve that model through the API**: `POST /serving/get_result`, the
   endpoint the frontend uses, and check the forecast
   (`test_inference.py::test_inference_endpoint`).

`test_resolution.py` adds fast unit tests of resolution inference and of the
regularization of irregular timestamps.

## Quick start

```bash
cd deeptsf_backend/tests

# once: build the Python environment (~4 GB: Python 3.12, CPU torch, the pins of conda.yaml)
DEEPTSF_TEST_HOME=/some/big/disk/deeptsf-tests ./setup_test_env.sh

# run everything (about 20-30 minutes on 4 cores)
DEEPTSF_TEST_HOME=/some/big/disk/deeptsf-tests ./run_tests.sh

# or a subset; arguments go straight to pytest
./run_tests.sh -k "LightGBM and 1h"
./run_tests.sh -k "shap or weather"
./run_tests.sh test_resolution.py
DEEPTSF_TEST_WORKERS=2 ./run_tests.sh -k NBEATS
```

Everything the tests write (the virtualenv, pip/uv caches, the MLflow database,
S3 objects' temp files, darts checkpoints) goes below `$DEEPTSF_TEST_HOME`,
default `tests/.work` (gitignored). Only the files of the last three runs are
kept, in `$DEEPTSF_TEST_HOME/runs/<run id>/<worker>/`; set
`DEEPTSF_KEEP_TEST_FILES=1` to also keep their temp files and checkpoints. A
JUnit report of each run is written to `runs/<run id>/junit.xml`.

| Variable | Default | Meaning |
|---|---|---|
| `DEEPTSF_TEST_HOME` | `tests/.work` | where the environment and all test files live |
| `DEEPTSF_TEST_WORKERS` | `4` | parallel pytest-xdist workers (each needs ~3 GB RAM) |
| `DEEPTSF_KEEP_TEST_FILES` | `0` | `1` keeps temp files and checkpoints of the run |

## What is real and what is replaced

The code under test runs unchanged. Around it:

| In production | In the tests | Why |
|---|---|---|
| Dagster with the celery executor | `dagster.materialize` of the same graph asset, in-process | no daemon / broker needed |
| S3 IO manager between the steps | in-memory IO manager | step outputs are small and stay in one process |
| MinIO | [moto](https://github.com/getmoto/moto)'s S3 server on 127.0.0.1 (`support/servers.py`) | same S3 API; a small middleware drops leading slashes from object names like MinIO does, which `download_online_file` relies on |
| per-tenant MLflow servers | one real `mlflow server` subprocess per worker, artifacts proxied to the S3 server | artifact URIs are `mlflow-artifacts:/...` as in production |
| Keycloak auth | `USE_AUTH=None` | the API mounts its routers without auth dependencies |
| open-meteo (weather covariates) | `support/inference.fake_open_meteo` | answers with the same JSON layout: 10 days of hourly values from today |
| `deeptsf_backend/.env` | never read (`load_dotenv` is disabled) | the tests must not touch the deployment's MinIO / MLflow / Keycloak |

Each xdist worker has its own servers and working directory, so workers do
not share any state. The worker's working directory holds links to the files
the training step packages into every model (`exceptions.py`, `utils.py`,
`inference.py`, `darts_flavor.py`), like `/app` in the worker container.

## The cases

`support/cases.py` builds the matrix; each `Case` is one pipeline run, and its
id (the pytest parameter id) names it, e.g. `TFT-1h-multiple-pastcov-futurecov`.

| Group | What it covers | Cases |
|---|---|---|
| base | all 12 models x resolutions 15min / 1h / 1d / 7d x single / multiple series (ARIMA on multiple series must fail with its error message) | 96 |
| covariates | each model with every covariate kind it supports (past, future, both), plus a few at other resolutions / series types | 24 |
| time_covariates | calendar features created by the ETL (`time_covs`) | 4 |
| shap | SHAP analysis of the evaluation (one test series) | 10 |
| optuna | hyperparameter search (`opt_test`, 2 trials) | 2 |
| irregular / resampling | jittered timestamps; data resampled to a coarser resolution | 6 |
| weather | models trained on a weather variable, served with `weather_covariates` | 3 |

Datasets (`support/data.py`) are synthetic, deterministic and small (160-770
points per series) and models are tiny (one epoch, a few units): the tests check
that every path through the pipeline and the serving works, not forecast quality.

## Layout

```
tests/
  README.md              this file
  setup_test_env.sh      builds $DEEPTSF_TEST_HOME/venv
  requirements-test.txt  test-only packages installed on top of conda.yaml's pins
  run_tests.sh           runs pytest with the right environment and workers
  pytest.ini
  conftest.py            process isolation, local servers, case grouping
  test_pipeline.py       end-to-end pipeline runs
  test_inference.py      model checks and serving through the API
  test_resolution.py     unit tests of resolution inference / regularization
  support/
    environment.py       env vars, sys.path, working directory
    servers.py           local S3 and MLflow servers
    data.py              synthetic datasets
    cases.py             the case matrix and per-model hyperparameters
    pipeline.py          runs a case through the Dagster job, collects its MLflow runs
    inference.py         API client, request building, fake open-meteo
```

## Adding a case

Append a `Case(...)` in `all_cases()` in `support/cases.py`; it is picked up by
the pipeline test and (if it is expected to succeed) by both inference tests.
A new model also needs an entry in `MODELS`, in `hyperparameters()` and in
`DARTS_CLASS` of `test_inference.py`. For a configuration that must fail, set
`expect_error` to (part of) the error message.
