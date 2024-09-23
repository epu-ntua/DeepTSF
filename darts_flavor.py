import os
from utils import load_model, load_scaler, load_ts_id, parse_uri_prediction_input, load_local_model_info, to_seconds
import pretty_errors
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from minio import Minio
from utils import truth_checker 

disable_warnings(InsecureRequestWarning)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)


class _MLflowPLDartsModelWrapper:
    """
    A wrapper class for Darts models that integrates with MLflow's PyFunc format. 
    This wrapper allows the model to be used with MLflow for predictions, while 
    handling input parsing, transformations, and inverse transformations of the 
    time series data.
    
    Attributes:
        model: A Darts time series model used for predictions.
        transformer: A list of transformers or a single transformer applied to the input time series data.
        transformer_past_covs: A list of transformers for past covariates or a single transformer.
        transformer_future_covs: A list of transformers for future covariates or a single transformer.
        ts_id_l: A list of time series IDs used to identify the position of the time series in the training dataset that the user
         wants to apply inference to.
    """


    def __init__(self, darts_model, transformer=None, transformer_past_covs=None, transformer_future_covs=None, ts_id_l=[[]]):
        """
        Initializes the _MLflowPLDartsModelWrapper class.

        Args:
            darts_model: A trained Darts model to be used for predictions.
            transformer (optional): A transformer (or list of transformers) for transforming input time series data.
            transformer_past_covs (optional): A transformer (or list of transformers) for transforming past covariates.
            transformer_future_covs (optional): A transformer (or list of transformers) for transforming future covariates.
            ts_id_l (optional): A list of time series IDs that were used during model training.
        """

        self.model = darts_model
        #TODO Ask if this is right
        self.transformer = transformer if type(transformer) == list or transformer==None else [transformer]
        self.transformer_past_covs = transformer_past_covs if type(transformer_past_covs) == list or transformer_past_covs==None else [transformer_past_covs]
        self.transformer_future_covs = transformer_future_covs if type(transformer_future_covs) == list or transformer_future_covs==None else [transformer_future_covs]
        self.ts_id_l=ts_id_l

    def predict(self, model_input):
        """
        Predict future values based on the input data using the Darts model.

        Args:
            model_input (Dict): A dictionary containing the following keys:
                - "timesteps_ahead": int, The number of timesteps to predict ahead.
                - "series_uri": str, URI for the time series data.
                - "multiple_file_type": Boolean indicating whether the input inference dataset is of multiple file input.
                    It doesn't matter if the model was trained using multiple time series or not, this is only for the
                    input of the inference function
                - "weather_covariates": Boolean indicating whether weather covariates are used.
                - "resolution": str, The resolution of the time series and the covariates.
                - "ts_id_pred": str, Time series ID for prediction. The user must provide this if multiple_file_type=True
                - "series": JSON file containing the series data (or None). One of series_uri or series must not be None
                - "past_covariates": JSON file containing past covariates (or None).
                - "past_covariates_uri": str, URI for the past covariates data (or None).
                - "future_covariates": JSON file containing future covariates (or None).
                - "future_covariates_uri": str, URI for the future covariates data (or None).
                - "roll_size": int specifying the rolling size for predictions.
                - "batch_size": int specifying the batch size for predictions.
                - "format": int specifying the format of the input data (long or short).

        Returns:
            pandas.DataFrame: A dataframe containing the prediction results.
        """

        # Accomodate bentoml
        try:
            model_input = model_input[0]
            batched = True
        except:
            batched = False

        # Parse
        model_input_parsed = parse_uri_prediction_input(client, model_input, self.model, self.ts_id_l)

        # Transform
        if self.transformer is not None:
            print('\nTransforming series...')
            model_input_parsed['series'] = self.transformer[model_input_parsed["idx_in_train_dataset"]].transform(
                model_input_parsed['series'])

        if self.transformer_past_covs is not None:
            print('\nTransforming past covariates...')
            model_input_parsed['past_covariates'] = self.transformer_past_covs[model_input_parsed["idx_in_train_dataset"]].transform(
                model_input_parsed['past_covariates'])
            
        if self.transformer_future_covs is not None:
            print('\nTransforming future covariates...')
            model_input_parsed['future_covariates'] = self.transformer_future_covs[model_input_parsed["idx_in_train_dataset"]].transform(
                model_input_parsed['future_covariates'])

        # Predict 
        predictions = self.model.predict(
            n=model_input_parsed['timesteps_ahead'],
            roll_size=model_input_parsed['roll_size'],
            series=model_input_parsed['series'],
            future_covariates=model_input_parsed['future_covariates'],
            past_covariates=model_input_parsed['past_covariates'],
            batch_size=model_input_parsed['batch_size'])

        ## Untransform
        if self.transformer is not None:
            print('\nInverse transforming series...')
            predictions = self.transformer[model_input_parsed["predict_series_idx"]].inverse_transform(predictions)

        # Return as DataFrame
        if batched:
            return [predictions[0].pd_dataframe()]
        else:
            return predictions[0].pd_dataframe()

def _load_pyfunc(model_folder):
    """
    Load PyFunc implementation. Called by `pyfunc.load_pyfunc` for model loading in MLflow.

    Args:
        model_folder (str): The folder path where the model is stored.

    Returns:
        _MLflowPLDartsModelWrapper: A wrapper object containing the loaded Darts model.
    """
    # load model from MLflow or local folder
    print(f"Local path inside _load_pyfunc: {model_folder}")
    model_folder = model_folder.replace('/', os.path.sep)
    model_folder = model_folder.replace('\\', os.path.sep)
    print(f"Local path altered for loading: {model_folder}")
    
    # TODO: Create a class for these functions instead of bringing them from utils.py
    print(model_folder)
    # Different behaviours for pl and pkl models are defined in load_model

    model = load_model(client, model_root_dir=model_folder, mode="local")
    model_info = load_local_model_info(model_root_dir=model_folder)
    #Loading scalers
    if bool(model_info["scaler"]):
        scaler = load_scaler(scaler_uri=f"{model_folder}/scaler_series.pkl", mode="local")
    else:
        scaler = None

    if bool(model_info["scale_covs"]) and bool(model_info["past_covs"]):
        scaler_past_covs = load_scaler(scaler_uri=f"{model_folder}/scaler_past_covariates.pkl", mode="local")
    else:
        scaler_past_covs = None

    if bool(model_info["scale_covs"]) and bool(model_info["future_covs"]):
        scaler_future_covs = load_scaler(scaler_uri=f"{model_folder}/scaler_future_covariates.pkl", mode="local")
    else:
        scaler_future_covs = None

    #Loading ts_id_l that was used to train the model
    ts_id_l = load_ts_id(load_ts_id_uri=f"{model_folder}/ts_id_l.pkl", mode="local")

    return _MLflowPLDartsModelWrapper(model, scaler, scaler_past_covs, scaler_future_covs, ts_id_l)
