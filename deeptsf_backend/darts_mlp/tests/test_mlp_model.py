import pytest
import torch
import numpy as np
from darts import TimeSeries
from darts_mlp.models.mlp_model import MLPModel, _MLPModule


class TestMLPModelInit:
    """Test MLPModel initialization"""
    
    def test_init_valid_parameters(self):
        """Test model initializes with valid parameters"""
        model = MLPModel(
            input_chunk_length=10,
            output_chunk_length=5,
            num_layers=3,
            layer_width=128
        )
        assert model.num_layers == 3
        assert model.layer_width == 128
        assert model.dropout == 0.0
        assert model.activation == "ReLU"
    
    def test_init_default_values(self):
        """Test default values are set correctly"""
        model = MLPModel(
            input_chunk_length=10,
            output_chunk_length=5
        )
        assert model.num_layers == 4
        assert model.layer_width == 256
        assert model.dropout == 0.0
        assert model.batch_norm == False
    
    def test_init_custom_activation(self):
        """Test custom activation function"""
        model = MLPModel(
            input_chunk_length=10,
            output_chunk_length=5,
            activation="GELU"
        )
        assert model.activation == "GELU"


class TestMLPModuleArchitecture:
    """Test _MLPModule network architecture"""
    
    @pytest.fixture
    def basic_module(self):
        """Create a basic module for testing"""
        return _MLPModule(
            input_dim=1,
            output_dim=1,
            nr_params=1,
            num_layers=2,
            layer_width=64,
            dropout=0.0,
            activation="ReLU",
            batch_norm=False,
            input_chunk_length=10,
            output_chunk_length=5,
        )
    
    def test_input_output_size_calculation(self, basic_module):
        """Test input and output size calculations"""
        assert basic_module.input_size == 10  # input_dim * input_chunk_length
        assert basic_module.output_size == 5  # output_dim * output_chunk_length * nr_params
    
    def test_mlp_network_exists(self, basic_module):
        """Test MLP network is created"""
        assert hasattr(basic_module, 'mlp')
        assert isinstance(basic_module.mlp, torch.nn.Module)


class TestMLPModuleForward:
    """Test _MLPModule forward pass"""
    
    def test_forward_shape_univariate(self):
        """Test output shape for univariate input"""
        module = _MLPModule(
            input_dim=1,
            output_dim=1,
            nr_params=1,
            num_layers=2,
            layer_width=64,
            dropout=0.0,
            activation="ReLU",
            batch_norm=False,
            input_chunk_length=10,
            output_chunk_length=5,
        )
        
        # Create input tensor: (batch_size, seq_len, features)
        x = torch.randn(32, 10, 1)
        y = module.forward((x, None))
        
        # Expected output: (batch_size, output_chunk_length, output_dim, nr_params)
        assert y.shape == (32, 5, 1, 1)
    
    def test_forward_shape_multivariate(self):
        """Test output shape for multivariate input"""
        module = _MLPModule(
            input_dim=3,
            output_dim=3,
            nr_params=1,
            num_layers=2,
            layer_width=64,
            dropout=0.0,
            activation="ReLU",
            batch_norm=False,
            input_chunk_length=10,
            output_chunk_length=5,
        )
        
        x = torch.randn(16, 10, 3)
        y = module.forward((x, None))
        
        assert y.shape == (16, 5, 3, 1)
    
    def test_forward_with_covariates(self):
        """Test forward pass with past covariates"""
        module = _MLPModule(
            input_dim=3,  # 1 target + 2 covariates
            output_dim=1,
            nr_params=1,
            num_layers=2,
            layer_width=64,
            dropout=0.0,
            activation="ReLU",
            batch_norm=False,
            input_chunk_length=10,
            output_chunk_length=5,
        )
        
        target = torch.randn(8, 10, 1)
        covariates = torch.randn(8, 10, 2)
        
        y = module.forward((target, covariates))
        
        assert y.shape == (8, 5, 1, 1)
    
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes"""
        module = _MLPModule(
            input_dim=1,
            output_dim=1,
            nr_params=1,
            num_layers=2,
            layer_width=64,
            dropout=0.0,
            activation="ReLU",
            batch_norm=False,
            input_chunk_length=10,
            output_chunk_length=5,
        )
        
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 10, 1)
            y = module.forward((x, None))
            assert y.shape == (batch_size, 5, 1, 1)


class TestMLPModelIntegration:
    """Integration tests with real data"""
    
    @pytest.fixture
    def simple_series(self):
        """Create a simple sine wave series"""
        values = np.sin(np.linspace(0, 10, 100))
        return TimeSeries.from_values(values)
    
    @pytest.fixture
    def multivariate_series(self):
        """Create a multivariate series"""
        values = np.column_stack([
            np.sin(np.linspace(0, 10, 100)),
            np.cos(np.linspace(0, 10, 100)),
        ])
        return TimeSeries.from_values(values)
    
    def test_fit_univariate(self, simple_series):
        """Test fitting on univariate series"""
        model = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=2,  # Quick test
            random_state=42
        )
        
        model.fit(simple_series)
        assert model.model is not None
    
    def test_predict_after_fit(self, simple_series):
        """Test prediction after fitting"""
        model = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=2,
            random_state=42
        )
        
        model.fit(simple_series)
        pred = model.predict(n=5)
        
        assert len(pred) == 5
        assert pred.values().shape == (5, 1)
    
    def test_fit_multivariate(self, multivariate_series):
        """Test fitting on multivariate series"""
        model = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=2,
            random_state=42
        )
        
        model.fit(multivariate_series)
        pred = model.predict(n=5)
        
        assert pred.values().shape == (5, 2)
    
    def test_fit_with_validation(self, simple_series):
        """Test fitting with validation set"""
        train = simple_series[:70]
        val = simple_series[70:]
        
        model = MLPModel(
            input_chunk_length=20,
            output_chunk_length=5,
            num_layers=2,
            layer_width=32,
            n_epochs=2,
            random_state=42
        )
        
        model.fit(train, val_series=val)
        assert model.model is not None


class TestMLPModelEdgeCases:
    """Test edge cases"""
    
    def test_single_batch_size(self):
        """Test with batch_size=1"""
        values = np.sin(np.linspace(0, 10, 50))
        series = TimeSeries.from_values(values)
        
        model = MLPModel(
            input_chunk_length=10,
            output_chunk_length=3,
            num_layers=2,
            layer_width=16,
            batch_size=1,
            n_epochs=1,
            random_state=42
        )
        
        model.fit(series)
        pred = model.predict(n=3)
        assert len(pred) == 3
    
    def test_minimal_architecture(self):
        """Test with minimal architecture"""
        values = np.sin(np.linspace(0, 10, 50))
        series = TimeSeries.from_values(values)
        
        model = MLPModel(
            input_chunk_length=5,
            output_chunk_length=1,
            num_layers=1,
            layer_width=8,
            n_epochs=1,
            random_state=42
        )
        
        model.fit(series)
        pred = model.predict(n=1)
        assert len(pred) == 1


# Optional: Add markers for slow tests
@pytest.mark.slow
class TestMLPModelPerformance:
    """Performance tests (marked as slow)"""
    
    def test_large_series(self):
        """Test with large time series"""
        values = np.sin(np.linspace(0, 100, 10000))
        series = TimeSeries.from_values(values)
        
        model = MLPModel(
            input_chunk_length=50,
            output_chunk_length=10,
            num_layers=3,
            layer_width=128,
            n_epochs=1,
            random_state=42
        )
        
        model.fit(series)
        pred = model.predict(n=10)
        assert len(pred) == 10


if __name__ == "__main__":
    # This allows running: python test_mlp_model.py
    pytest.main([__file__, "-v"])