"""End-to-end tests of the DeepTSF pipeline (load data -> ETL -> training -> evaluation).

Each parameter is a ``Case`` from support/cases.py: one model, resolution and
series type, with or without covariates, SHAP, hyperparameter search, ... The
real Dagster job runs in-process against local S3 and MLflow servers; see
README.md for what is (and is not) replaced.
"""
import math

import pytest

from support.cases import FUTURE_COV_MODELS, PAST_COV_MODELS, all_cases
from support.pipeline import get_run

CASES = all_cases()


def _check_run_structure(run):
    case = run.case
    assert run.parent_run is not None, "no parent MLflow run (stage=main) was created"
    names = set(run.children)
    assert {"load_data", "etl", "eval"} <= names, f"missing stage runs, got {sorted(names)}"
    train = run.train_run
    assert train is not None, f"no training run among {sorted(names)}"
    tags = train.data.tags
    for tag in ("model_uri", "pyfunc_model_folder", "series_uri", "model_type"):
        assert tag in tags, f"training run has no {tag} tag"
    # training.py drops the covariate kinds a model can not use (e.g. NBEATS ignores time_covs)
    if case.past_covs and case.model in PAST_COV_MODELS:
        assert tags.get("past_covariates_uri", "None") != "None", "past covariates were not used"
    if (case.future_covs or case.time_covs) and case.model in FUTURE_COV_MODELS:
        assert tags.get("future_covariates_uri", "None") != "None", "future covariates were not used"
    # every child hangs off this case's own parent run
    for child in run.children.values():
        assert child.data.tags["mlflow.parentRunId"] == run.parent_run.info.run_id


def _check_evaluation(run):
    metrics = run.eval_run.data.metrics
    assert metrics, "evaluation logged no metrics"
    mape = [v for k, v in metrics.items() if k.startswith("mape")]
    assert mape and all(math.isfinite(v) for v in mape), f"no finite MAPE in {metrics}"


def _check_shap(run):
    from mlflow.tracking import MlflowClient

    artifacts = [a.path for a in MlflowClient().list_artifacts(run.eval_run.info.run_id)]
    assert any(a.startswith("interpretation") for a in artifacts), f"no SHAP artifacts in {artifacts}"


@pytest.mark.pipeline
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_pipeline(case, services, workdir):
    run = get_run(case, workdir)
    if case.expect_error:
        assert not run.success, f"expected the run to fail with {case.expect_error!r}"
        assert case.expect_error in run.error, run.error
        return
    assert run.success, run.error
    _check_run_structure(run)
    _check_evaluation(run)
    if case.shap:
        _check_shap(run)
