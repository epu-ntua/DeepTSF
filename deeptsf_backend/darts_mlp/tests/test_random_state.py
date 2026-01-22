import pytest
import torch
from darts_mlp.models.mlp_model import MLPModel
from darts import TimeSeries
import numpy as np


class TestRandomStateCompatibility:
    """Test random_state parameter compatibility with u8darts v0.28.0"""
    
    @pytest.fixture
    def simple_series(self):
        """Create a simple sine wave series"""
        values = np.sin(np.linspace(0, 10, 100))
        return TimeSeries.from_values(values)
    
    def test_random_state_parameter_ignored(self, simple_series):
        """Test that random_state parameter is accepted but ignored (or raises error)"""
        try:
            model = MLPModel(
                input_chunk_length=20,
                output_chunk_length=5,
                num_layers=2,
                layer_width=32,
                n_epochs=1,
                random_state=42,  # This may cause issues
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                }
            )
            # If it doesn't raise an error, it's being silently ignored
            print("random_state parameter accepted (may be ignored)")
        except TypeError as e:
            # If it raises TypeError, random_state is not supported
            pytest.fail(f"random_state parameter not supported: {e}")
    
    def test_without_random_state(self, simple_series):
        """Test that model works without random_state parameter"""
        model = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=1,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        
        model.fit(simple_series)
        pred = model.predict(n=5)
        
        assert len(pred) == 5
        assert pred.values().shape == (5, 1)
    
    def test_reproducibility_with_torch_seed(self, simple_series):
        """Test reproducibility using torch.manual_seed instead of random_state"""
        # Set seed manually
        torch.manual_seed(42)
        np.random.seed(42)
        
        model1 = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=1,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        model1.fit(simple_series)
        pred1 = model1.predict(n=5)
        
        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        model2 = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=1,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        model2.fit(simple_series)
        pred2 = model2.predict(n=5)
        
        # Check if predictions are similar (not exact due to floating point)
        np.testing.assert_allclose(
            pred1.values(), 
            pred2.values(), 
            rtol=1e-5,
            err_msg="Predictions should be reproducible with manual seed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])