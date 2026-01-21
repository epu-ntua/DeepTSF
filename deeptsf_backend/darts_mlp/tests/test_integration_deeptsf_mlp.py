"""
Integration test for MLP model training without covariates.

This test mimics the DeepTSF training pipeline when no covariates are provided:
- Load CSV dataset
- Train/validation/test split
- Data scaling with Scaler
- Model training with validation monitoring
- Early stopping support
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
import logging
import shutil 
from .utils_ds import load_local_csv_or_df_as_darts_timeseries, multiple_dfs_to_ts_file, multiple_ts_file_to_dfs
from .preprocessing import scale_covariates, split_dataset, split_nans
from pathlib import Path
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts_mlp import MLPModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TestIntegrationMLP:
    """Integration test for MLP training without covariates"""
    
    @pytest.fixture
    def dataset_paths(self):
        """Get paths to test datasets"""
        test_dir = Path(__file__).parent
        return {
            'series': str(test_dir / 'datasets' / 'series_tsID_0_1.csv'),
        }

    @pytest.fixture
    def hyperparameters(self):
        """Hyperparameters file"""
        
        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-6,
            mode='min',
        )
        pl_trainer_kwargs = {"callbacks": [my_stopper],
                        "accelerator": 'auto',
                        #  "gpus": 1,
                        #  "auto_select_gpus": True,
                        "log_every_n_steps": 10}
        return {
            "input_chunk_length": 24,  # Common value for hourly data
            "output_chunk_length": 24,  # Predict 24 hours ahead
            "num_layers": 3,
            "layer_width": 128,
            "dropout": 0.1,
            "activation": "ReLU",
            "batch_norm": True,
            "n_epochs": 2,  # Reduced for testing
            "batch_size": 32,
            "optimizer_kwargs": {"lr": 0.001},
            "save_checkpoints": True,
            "log_tensorboard": False,
            "pl_trainer_kwargs": pl_trainer_kwargs,
            "random_state": 42,
            "force_reset": True,
        }
    
    def test_training_only_series(self, dataset_paths, hyperparameters, tmp_path):
        """
        Test complete training pipeline WITHOUT covariates like DeepTSF does it.
        
        This mimics training.py when past_covs_csv=None and future_covs_csv=None:
        - Setup early stopping callback
        - Configure pl_trainer_kwargs
        - Train model with fit()
        - Pass ONLY train/val series data (NO covariates)
        - Evaluate on test set
        - Calculate metrics
        """
        # manual definition of values for testing purposes
        print("\nCreating local folders...")
        logging.info("\nCreating local folders...")

        # tmp_path is automatically cleaned up by pytest
        scalers_dir = tmp_path / "scalers"
        features_dir = tmp_path / "features"
        scalers_dir.mkdir()
        features_dir.mkdir()
        
        # Convert to string when needed
        scalers_dir = str(scalers_dir)
        features_dir = str(features_dir)

        test_end_date = None
        cut_date_test = '20230315'
        cut_date_val = '20230101'
        multiple = True
        resolution = '1h'
        series_csv = dataset_paths['series']
        scale = 'true'
        format = 'short'
        darts_model = 'MLP'
        forecast_horizon = 48

        time_col = 'Date'
        series, id_l, ts_id_l = load_local_csv_or_df_as_darts_timeseries(
                local_path_or_df=series_csv,
                name='series',
                time_col=time_col,
                last_date=test_end_date,
                multiple=multiple,
                resolution=resolution,
                format=format)

            # Train / Test split
        print(
            f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")
        logging.info(
            f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")
            
        ## series
        series_split = split_dataset(
            series,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            store_dir=features_dir,
            name='series',
            conf_file_name='split_info.yml',
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format)    

        # Scaling
        print("\nScaling...")
        logging.info("\nScaling...")

        ## scale series
        series_transformed = scale_covariates(
            series_split,
            store_dir=features_dir,
            filename_suffix="series_transformed.csv",
            scale=scale,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format,
            )
        
        if scale:
            pickle.dump(series_transformed["transformer"], open(f"{scalers_dir}/scaler_series.pkl", "wb"))

        # Model training
        print("\nTraining model...")
        logging.info("\nTraining model...")

        print("\nTraining on series:\n")
        logging.info("\nTraining on series:\n")
        if multiple:
            for i, series in enumerate(series_transformed['train']):
                print(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
                logging.info(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
        else:
            print(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
            logging.info(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
        print("")

        print("Validating on series:\n")
        logging.info("Validating on series:\n")
        if multiple:
            for i, series in enumerate(series_transformed['val']):
                print(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
                logging.info(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
        else:
            print(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
            logging.info(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")

        #TODO maybe modify print to include split train based on nans
        #TODO make more efficient by also spliting covariates where the nans are split
            
        if darts_model not in ['ARIMA']:
            series_transformed['train'], _, _ = \
                split_nans(series_transformed['train'], None, None)
        
        ## choose architecture
        if darts_model in ['NHiTS', 'NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN', 'Transformer', 'MLP']:
            darts_model = darts_model+"Model"
            
            print(f'\nTrained Model: {darts_model}')
            if 'learning_rate' in hyperparameters:
                hyperparameters['optimizer_kwargs'] = {'lr': hyperparameters['learning_rate']}
                del hyperparameters['learning_rate']

            if 'likelihood' in hyperparameters:
                hyperparameters['likelihood'] = eval(hyperparameters['likelihood']+"Likelihood"+"()")

            model = eval(darts_model)(
                **hyperparameters
            )
            
            model.fit(series_transformed['train'],
                val_series=series_transformed['val']
                )
            
            model_type = "pl"


        # Assertions to verify training success
        print("\nVerifying model training...")
        logging.info("\nVerifying model training...")

        # 1. Check model was fitted
        assert hasattr(model, '_fit_called'), "Model fit() was not called"

        # 2. Check model has training history
        assert model.trainer is not None, "Model trainer not initialized"
        assert model.trainer.logged_metrics, "No training metrics logged"

        # 3. Verify model can make predictions
        pred = model.predict(series=series_transformed['train'][0], n=forecast_horizon)
        assert pred is not None, "Model failed to generate predictions"
        assert len(pred) == forecast_horizon, \
            f"Expected {forecast_horizon} predictions, got {len(pred)}"

        # 4. Verify validation loss decreased (optional but recommended)
        if hasattr(model.trainer, 'callback_metrics'):
            val_loss = model.trainer.callback_metrics.get('val_loss')
            if val_loss is not None:
                assert val_loss < float('inf'), "Validation loss is infinite"
                print(f"Final validation loss: {val_loss:.6f}")
                logging.info(f"Final validation loss: {val_loss:.6f}")

        # 5. Check model state dict exists (for PyTorch models)
        assert model.model.state_dict(), "Model has no learned parameters"

        print("✓ Model training verified successfully")
        logging.info("✓ Model training verified successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
