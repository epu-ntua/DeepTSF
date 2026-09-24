<p align="center">
  <a href="https://doi.org/10.1016/j.softx.2024.101758">
    <img alt="DeepTSF" src="https://raw.githubusercontent.com/epu-ntua/DeepTSF/master/deeptsf_backend/docs/version1all.png" width="150" />
  </a>
</p>
<p align="center">
    ✨ DeepTSF is designed to enable codeless machine learning operations for time series forecasting ✨
</p>

<p align="center">
    🙌 Refer to <b><a href="https://github.com/epu-ntua/DeepTSF/wiki/DeepTSF-documentation">https://github.com/epu-ntua/DeepTSF/wiki/DeepTSF-documentation</a></b> for the documentation 📖
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/epu-ntua/DeepTSF/blob/dev/LICENSE.txt) [![DOI](https://img.shields.io/badge/Cite%20this%20paper-Google%20Scholar-blue])](https://doi.org/10.1016/j.softx.2024.101758)

## Installation

To set up DeepTSF on your local system, you need clone the main branch of this repository:

```git clone https://github.com/epu-ntua/DeepTSF.git```

Alternatively you can use the dedicated Github release instead of cloning the main branch.

After that you need to navigate to the root directory of DeepTSF:

```cd /path/to/repo/of/DeepTSF```

Το enable the communication of the client with the logging servers (MLflow, Minio, Postgres), a .env file is needed. 
An example (.env.example) is provided, with default environment variables.

After that, you can set up a full deployment of DeepTSF using Docker.

### Set up locally using Docker

To set up locally using docker first go to DeepTSF's root directory (inside deeptsf_backend) and rename .env.example to .env. Then run the following command in DeepTSF's root directory:

```docker-compose up```

DeepTSF is up and running. Navigate to [http://localhost:3000](http://localhost:3000) and start your experiments!

## Dagster UI for advanced users

For users that require advanced pipeline parameterization and functionalities such as hyperparameter tuning,
a dagster based pipeline is provided. By modifying the config of deeptsf_dagster_job, the user can set all parameters 
described in the extensive documentation of DeepTSF. An example config file is given below:

```
resources:
  config:
    config:
      a: 0.3
      analyze_with_shap: false
      convert_to_local_tz: true
      country: PT
      cut_date_test: "20210101"
      cut_date_val: "20200101"
      darts_model: LightGBM
      database_name: rdn_load_data
      device: gpu
      eval_method: ts_ID
      eval_series: eval_series
      evaluate_all_ts: true
      experiment_name: dagster_test
      forecast_horizon: 24
      format: long
      from_database: false
      future_covs_csv: None
      future_covs_uri: None
      grid_search: false
      hyperparams_entrypoint:
        lags: [-1, -2, -14]
      ignore_previous_runs: true
      imputation_method: linear
      loss_function: mape
      m_mase: 1
      max_thr: -1
      min_non_nan_interval: 24
      multiple: false
      n_trials: 100
      num_samples: 1
      num_workers: 4
      opt_test: false
      order: 1
      parent_run_name: dagster_test
      past_covs_csv: None
      past_covs_uri: None
      pv_ensemble: false
      resampling_agg_method: averaging
      resolution: 1h
      retrain: false
      rmv_outliers: true
      scale: true
      scale_covs: true
      series_csv: dataset-storage/Italy.csv
      series_uri: None
      shap_data_size: 100
      shap_input_length: -1
      std_dev: 4.5
      stride: -1
      test_end_date: None
      time_covs: false
      ts_used_id: None
      wncutoff: 0.000694
      ycutoff: 3
      ydcutoff: 30
      year_range: None
```

For a more complete guide check the extensive documentation.

This application can also be deployed in a kubernetes enviroment. 

#### Set up mlflow tracking server

To run DeepTSF on your system you first have to install the mlflow tracking and minio server.

```git clone https://github.com/epu-ntua/mlflow-tracking-server.git```

```cd mlflow-server```

After that, you need to get the server to run

```docker-compose up```

The MLflow server and client may run on different computers. In this case, remember to change
the addresses on the .env file.

For the extensive DeepTSF documentation please navigate to our [Wiki](https://github.com/epu-ntua/DeepTSF/wiki/DeepTSF-documentation). 

#### 📺 DeepTSF — Video Demonstration

Also, a video demonstration of DeepTSF is avaialble on Youtube.

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=hJbnvXummTI) 

## References
[1] S. Pelekis et al., “DeepTSF: Codeless machine learning operations for time series forecasting,” SoftwareX, vol. 27, p. 101758, Sep. 2024, doi: [10.1016/J.SOFTX.2024.101758](https://doi.org/10.1016/j.softx.2024.101758). <br>
