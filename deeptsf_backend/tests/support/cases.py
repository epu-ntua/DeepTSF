"""The matrix of pipeline configurations the end-to-end tests run.

A ``Case`` describes one pipeline run (model, resolution, series type, which
covariates, SHAP, ...). ``all_cases()`` builds the full matrix; see README.md
for the groups and why each exists. Hyperparameters are deliberately tiny
(one epoch, a handful of units) since the tests check that the pipeline and the
models work end to end, not forecast quality.
"""
from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd

from .data import RESOLUTIONS, ResolutionSpec

# Every model the training step supports (dagster_deeptsf/training.py).
MODELS = ["Naive", "LightGBM", "RandomForest", "ARIMA", "NBEATS", "NHiTS",
          "RNN", "BlockRNN", "TCN", "Transformer", "TFT", "MLP"]

# Covariates each model can use; training.py (and inference) drop the other kind.
PAST_COV_MODELS = ["LightGBM", "RandomForest", "NBEATS", "NHiTS", "BlockRNN", "TCN", "Transformer", "TFT", "MLP"]
FUTURE_COV_MODELS = ["LightGBM", "RandomForest", "RNN", "ARIMA", "TFT"]
REGRESSION_MODELS = ["LightGBM", "RandomForest"]


@dataclass(frozen=True)
class Case:
    model: str
    resolution: str                 # key of data.RESOLUTIONS: resolution of the input data
    kind: str = "multiple"          # "single" or "multiple"
    past_covs: bool = False
    future_covs: bool = False
    time_covs: bool = False         # ETL adds calendar features as future covariates
    shap: bool = False
    optuna: bool = False            # opt_test: optuna search over list-valued hyperparameters
    irregular: bool = False         # input timestamps jittered off the regular grid
    train_resolution: Optional[str] = None   # resample to this (coarser) resolution in the ETL
    expect_error: Optional[str] = None       # the run must fail with this message
    group: str = field(default="base", compare=False)

    @property
    def spec(self) -> ResolutionSpec:
        return RESOLUTIONS[self.train_resolution or self.resolution]

    @property
    def data_spec(self) -> ResolutionSpec:
        """The raw data. When it is resampled to train_resolution, it covers the
        same time span as the training resolution's own dataset."""
        raw = RESOLUTIONS[self.resolution]
        if not self.train_resolution:
            return raw
        ratio = pd.Timedelta(self.spec.freq) / pd.Timedelta(raw.freq)
        return replace(raw, steps=int(self.spec.steps * ratio))

    @property
    def id(self) -> str:
        parts = [self.model, self.resolution + (f"-to-{self.train_resolution}" if self.train_resolution else ""), self.kind]
        for flag in ("past_covs", "future_covs", "time_covs", "shap", "optuna", "irregular"):
            if getattr(self, flag):
                parts.append(flag.replace("_covs", "cov"))
        return "-".join(parts)


def hyperparameters(case: Case) -> dict:
    spec = case.spec
    L, H = spec.input_length, spec.horizon
    torch_common = {"n_epochs": 1, "batch_size": 64, "random_state": 0}
    m = case.model
    if m == "Naive":
        return {"days_seasonality": 1}
    if m in REGRESSION_MODELS:
        hp = {"lags": L, "n_estimators": 10}
        if m == "LightGBM":
            hp["verbose"] = -1
        if case.past_covs:
            hp["lags_past_covariates"] = [-1, -2]
        if case.future_covs or case.time_covs:
            hp["lags_future_covariates"] = [0, 1]
        return hp
    if m == "ARIMA":
        return {"p": 1, "d": 0, "q": 0}
    hp = {"input_chunk_length": L, "output_chunk_length": H, **torch_common}
    if m in ("NBEATS", "NHiTS"):
        hp.update(num_stacks=2, num_blocks=1, num_layers=1, layer_widths=16)
        if m == "NBEATS":
            hp["generic_architecture"] = True
    elif m == "RNN":
        del hp["output_chunk_length"]
        hp.update(model="LSTM", training_length=L + H, hidden_dim=8, n_rnn_layers=1)
    elif m == "BlockRNN":
        hp.update(model="LSTM", hidden_dim=8, n_rnn_layers=1)
    elif m == "TCN":
        hp.update(kernel_size=2, num_filters=4, dilation_base=2)
    elif m == "Transformer":
        hp.update(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16)
    elif m == "TFT":
        hp.update(hidden_size=8, lstm_layers=1, num_attention_heads=1, add_relative_index=True)
    elif m == "MLP":
        hp.update(num_layers=1, layer_width=16)
    return hp


def optuna_hyperparameters(case: Case) -> dict:
    """Same model, but with two values to search over (DeepTSF's ["list", ...] syntax)."""
    hp = hyperparameters(case)
    key = "lags" if case.model in REGRESSION_MODELS else "input_chunk_length"
    hp[key] = ["list", case.spec.input_length // 2, case.spec.input_length]
    return hp


def all_cases() -> list:
    cases = []

    # 1. Every model at every resolution, for a single and a multiple series file.
    for model in MODELS:
        for res in RESOLUTIONS:
            for kind in ("single", "multiple"):
                c = Case(model, res, kind, group="base")
                if model == "ARIMA" and kind == "multiple":
                    c = replace(c, expect_error="ARIMA does not support multiple time series")
                cases.append(c)

    # 2. Covariates: every model with each kind of covariate it supports.
    for model in PAST_COV_MODELS:
        cases.append(Case(model, "1h", "multiple", past_covs=True, group="covariates"))
    for model in FUTURE_COV_MODELS:
        kind = "single" if model == "ARIMA" else "multiple"
        cases.append(Case(model, "1h", kind, future_covs=True, group="covariates"))
    for model in sorted(set(PAST_COV_MODELS) & set(FUTURE_COV_MODELS)):
        cases.append(Case(model, "1h", "multiple", past_covs=True, future_covs=True, group="covariates"))
    cases += [
        Case("LightGBM", "1h", "single", past_covs=True, future_covs=True, group="covariates"),
        Case("TFT", "1d", "single", past_covs=True, future_covs=True, group="covariates"),
        Case("NBEATS", "1d", "multiple", past_covs=True, group="covariates"),
        Case("RNN", "15min", "single", future_covs=True, group="covariates"),
        Case("LightGBM", "7d", "multiple", past_covs=True, future_covs=True, group="covariates"),
        # a covariate kind the model can not use is ignored, in training and at inference
        Case("MLP", "1h", "multiple", past_covs=True, future_covs=True, group="covariates"),
    ]

    # 3. Calendar features generated by the ETL (time_covs).
    for model in ("LightGBM", "NBEATS", "RNN", "TFT"):
        cases.append(Case(model, "1h", "multiple", time_covs=True, group="time_covariates"))

    # 4. SHAP analysis of the evaluation (single test series).
    for model in ("LightGBM", "RandomForest", "NBEATS", "NHiTS", "TCN", "BlockRNN", "Transformer", "MLP"):
        cases.append(Case(model, "1h", "single", shap=True, group="shap"))
    cases += [
        Case("LightGBM", "1h", "multiple", past_covs=True, future_covs=True, shap=True, group="shap"),
        Case("NBEATS", "1d", "multiple", past_covs=True, shap=True, group="shap"),
    ]

    # 5. Hyperparameter search (opt_test with optuna).
    for model in ("LightGBM", "NBEATS"):
        cases.append(Case(model, "1h", "multiple", optuna=True, group="optuna"))

    # 6. Irregular timestamps and resampling to a coarser resolution in the ETL.
    cases += [
        Case("LightGBM", "1h", "single", irregular=True, group="irregular"),
        Case("NBEATS", "1h", "multiple", irregular=True, group="irregular"),
        Case("LightGBM", "7d", "multiple", irregular=True, group="irregular"),
        Case("LightGBM", "15min", "multiple", train_resolution="1h", group="resampling"),
        Case("NBEATS", "15min", "single", train_resolution="1h", group="resampling"),
        Case("Naive", "1h", "single", train_resolution="1d", group="resampling"),
    ]

    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    return cases
