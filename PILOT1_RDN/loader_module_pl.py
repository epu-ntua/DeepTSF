from sys import version_info
import os
import cloudpickle
from utils import load_model, load_scaler
import darts, mlflow, torch



PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

mlflow_serve_conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'darts=={}'.format(darts.__version__),
                'torch=={}'.format(torch.__version__),
                'mlflow=={}'.format(mlflow.__version__)
            ],
        },
    ],
    'name': 'darts_infer_pl_env'
}


class _MLflowPLDartsModelWrapper:
    def __init__(self, darts_model, transformer=None):
        self.model = darts_model
        self.transformer = transformer

    def predict(self, model_input):
        """ 
        :param model_input: Dict
        {"n": int, "history": darts.TimeSeries, "past_covariates": darts.TimeSeries, "future_covariates": darts.TimeSeries, "roll_size":int, "batch_size": int}
        """
        # Does it work?
        n = model_input["n"] if model_input["n"] is not None else self.model.output_chunk_length
        roll_size = model_input["roll_size"] if model_input["roll_size"] is not None else self.model.output_chunk_length
        history = model_input["history"]
        past_covariates = model_input["past_covariates"]
        future_covariates = model_input["future_covariates"]
        batch_size = model_input["batch_size"] if model_input["batch_size"] is not None else 1

        # Transform to darts
        history = history
        if past_covariates is not None:
            past_covariates = past_covariates
        if future_covariates is not None:
            future_covariates = future_covariates

        # Transform
        if self.transformer is not None:
            history = self.transformer.transform(history)

        # Predict
        predictions = self.model.predict(
            n=n,
            roll_size=roll_size,
            series=history,
            future_covariates=future_covariates,
            past_covariates=past_covariates,
            batch_size=batch_size,
            num_loader_workers=2)

        # Untransform
        if self.transformer is not None:
            predictions = self.transformer.inverse_transform(predictions)

        # Return as DataFrame
            return predictions.pd_dataframe()

def _load_pyfunc(model_file):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    # load model from MLflow or local folder
    print(f"Local path inside _load_pyfunc: {model_file}")
    model_file = model_file.replace('/', os.path.sep)
    model_file = model_file.replace('\\', os.path.sep)
    print(f"Local path altered for loading: {model_file}")
    
    model = load_model(model_uri=model_file, mode="local", model_type="pl")
    scaler = load_scaler(scaler_uri=f"{model_file}/scaler_series.pkl", mode="local")

    return _MLflowPLDartsModelWrapper(model, scaler)


