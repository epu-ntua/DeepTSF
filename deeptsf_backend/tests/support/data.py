"""Small synthetic datasets in the CSV formats DeepTSF accepts.

Every dataset is tiny on purpose: just long enough for the train / validation /
test split and the models' input windows, so a full pipeline run takes seconds.

Formats (see docs of load_raw_data.read_and_validate_input):
* single series:   ``Datetime,Value``
* multiple series: ``Index,Datetime,ID,Timeseries ID,Value`` (long format)
* covariates:      always the long multiple format; the i-th Timeseries ID of a
                   covariates file belongs to the i-th Timeseries ID of the series.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResolutionSpec:
    """How much data to generate for a resolution and how to split it."""
    freq: str            # pandas / DeepTSF resolution string
    steps: int           # length of each series
    horizon: int         # forecast horizon used by the cases
    input_length: int    # input_chunk_length / lags used by the cases
    season: int          # period of the generated seasonality, in steps


RESOLUTIONS = {
    "15min": ResolutionSpec("15min", steps=96 * 8, horizon=8, input_length=16, season=96),
    "1h": ResolutionSpec("1h", steps=24 * 32, horizon=6, input_length=24, season=24),
    "1d": ResolutionSpec("1d", steps=400, horizon=3, input_length=14, season=7),
    "7d": ResolutionSpec("7d", steps=160, horizon=2, input_length=8, season=52),
}

START = pd.Timestamp("2023-01-02")          # a Monday, so weekly data is Monday-aligned
SERIES_IDS = {"single": ["Timeseries"], "multiple": ["A", "B"]}
COVARIATE_EXTRA_STEPS = 64                  # future covariates must reach past the series end


def _signal(n: int, season: int, seed: int, level: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (level + 0.02 * t + 10 * np.sin(2 * np.pi * t / season)
            + rng.normal(0, 1.0, n)).round(3)


def index_for(spec: ResolutionSpec, steps: int = None) -> pd.DatetimeIndex:
    return pd.date_range(START, periods=steps or spec.steps, freq=spec.freq)


def split_dates(spec: ResolutionSpec) -> dict:
    """Validation / test start dates at 70% / 85% of the series, whole days
    (the frontend sends YYYYMMDD dates)."""
    idx = index_for(spec)
    val = idx[int(len(idx) * 0.70)].normalize()
    test = idx[int(len(idx) * 0.85)].normalize()
    if val == test:                                       # short sub-daily series
        test = val + pd.Timedelta("1D")
    return {"cut_date_val": val.strftime("%Y%m%d"), "cut_date_test": test.strftime("%Y%m%d"),
            "test_end_date": "None"}


def series_frame(spec: ResolutionSpec, kind: str, index: pd.DatetimeIndex = None) -> pd.DataFrame:
    """The target series, as the pipeline reads it (long format for 'multiple')."""
    idx = index if index is not None else index_for(spec)
    frames = []
    for i, sid in enumerate(SERIES_IDS[kind]):
        frames.append(pd.DataFrame({"Datetime": idx, "ID": sid, "Timeseries ID": sid,
                                    "Value": _signal(len(idx), spec.season, seed=i, level=100 + 50 * i)}))
    return pd.concat(frames, ignore_index=True)


def covariates_frame(spec: ResolutionSpec, kind: str, names=("cov_1", "cov_2"), seed: int = 10) -> pd.DataFrame:
    """Covariates for every series of a dataset, running COVARIATE_EXTRA_STEPS past its end."""
    idx = index_for(spec, spec.steps + COVARIATE_EXTRA_STEPS)
    frames = []
    for i, sid in enumerate(SERIES_IDS[kind]):
        for j, name in enumerate(names):
            frames.append(pd.DataFrame({"Datetime": idx, "ID": f"{sid}_{name}", "Timeseries ID": sid,
                                        "Value": _signal(len(idx), spec.season, seed=seed + 7 * i + j, level=20)}))
    return pd.concat(frames, ignore_index=True)


def write_series_csv(df: pd.DataFrame, kind: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "single":
        df[["Datetime", "Value"]].to_csv(path, index=False)
    else:
        df.sort_values(["Datetime", "Timeseries ID", "ID"]).reset_index(drop=True).to_csv(path, index_label="Index")
    return path


def write_covariates_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["Datetime", "Timeseries ID", "ID"]).reset_index(drop=True).to_csv(path, index_label="Index")
    return path


def jitter(df: pd.DataFrame, spec: ResolutionSpec, seed: int = 3) -> pd.DataFrame:
    """Irregular timestamps: every reading shifted by up to +-10% of a step."""
    rng = np.random.default_rng(seed)
    step = pd.to_timedelta(spec.freq)
    out = df.copy()
    shift = rng.uniform(-0.1, 0.1, len(out)) * step.total_seconds()
    out["Datetime"] = out["Datetime"] + pd.to_timedelta(shift.round(), unit="s")
    return out
