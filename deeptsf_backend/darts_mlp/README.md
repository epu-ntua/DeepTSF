# Darts-MLP

A Multi-Layer Perceptron (MLP) model implementation for [Darts](https://github.com/unit8co/darts) time series forecasting library.

## Features

- ✅ Fully compatible with Darts API
- ✅ Supports univariate and multivariate time series
- ✅ Supports past covariates
- ✅ Configurable architecture (layers, width, activation)
- ✅ Dropout and batch normalization support
- ✅ PyTorch Lightning based training

## Installation

### Install in Editable Mode (for development)

```bash
cd /home/nibra/darts-mlp
pip install -e .
```

With development dependencies:
```bash
pip install -e .[dev]
```

### Install from Git (once you push to GitHub)

```bash
pip install git+https://github.com/yourusername/darts-mlp.git
```

## Quick Start

```python
from darts_mlp import MLPModel
from darts.datasets import AirPassengersDataset

# Load data
series = AirPassengersDataset().load()

# Create model
model = MLPModel(
    input_chunk_length=12,
    output_chunk_length=6,
    num_layers=3,
    layer_width=128,
    n_epochs=100,
    random_state=42
)

# Train
model.fit(series)

# Predict
forecast = model.predict(n=6)
```

## Parameters

- **input_chunk_length** (int): Number of past time steps used as input
- **output_chunk_length** (int): Number of future time steps to predict
- **num_layers** (int, default=4): Number of hidden layers
- **layer_width** (int, default=256): Width of each hidden layer
- **dropout** (float, default=0.0): Dropout probability
- **activation** (str, default="ReLU"): Activation function
- **batch_norm** (bool, default=False): Whether to use batch normalization
- **n_epochs** (int, default=100): Number of training epochs
- **batch_size** (int, default=32): Training batch size

### Supported Activations
- ReLU, RReLU, PReLU, ELU, Softplus, Tanh, SELU, LeakyReLU, Sigmoid, GELU

## Examples

### Univariate Forecasting
```python
from darts_mlp import MLPModel
import numpy as np
from darts import TimeSeries

# Create simple series
values = np.sin(np.linspace(0, 10, 100))
series = TimeSeries.from_values(values)

# Train and predict
model = MLPModel(input_chunk_length=20, output_chunk_length=5, n_epochs=50)
model.fit(series)
forecast = model.predict(n=10)
```

### Multivariate Forecasting
```python
# Create multivariate series
values = np.column_stack([
    np.sin(np.linspace(0, 10, 100)),
    np.cos(np.linspace(0, 10, 100))
])
series = TimeSeries.from_values(values)

model = MLPModel(input_chunk_length=20, output_chunk_length=5, n_epochs=50)
model.fit(series)
forecast = model.predict(n=10)
```

### With Past Covariates
```python
from darts.datasets import WeatherDataset

series = WeatherDataset().load()
target = series['p (mbar)'][:100]
past_cov = series['rain (mm)'][:100]

model = MLPModel(input_chunk_length=12, output_chunk_length=6, n_epochs=50)
model.fit(target, past_covariates=past_cov)
forecast = model.predict(n=6, past_covariates=past_cov)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=darts_mlp --cov-report=html

# Run specific test class
pytest tests/test_mlp_model.py::TestMLPModelIntegration -v
```

## Architecture

The model consists of:
- Input layer: Flattens `input_chunk_length` × features
- Hidden layers: Configurable number and width with optional batch norm and dropout
- Output layer: Produces `output_chunk_length` × features predictions

## Requirements

- Python >= 3.8
- darts >= 0.24.0
- torch >= 1.9.0
- numpy >= 1.19.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Your Name

## Acknowledgments

- Built on top of [Darts](https://github.com/unit8co/darts) by Unit8
- Uses PyTorch Lightning for training
