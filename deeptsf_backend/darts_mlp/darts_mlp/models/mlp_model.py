import torch
import torch.nn as nn
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule

class MLPModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
        activation: str = "ReLU",
        batch_norm: bool = False,
        **kwargs,
    ):
        """Multi-Layer Perceptron (MLP) for Time Series Forecasting.

            A simple feedforward neural network that maps input sequences directly to output forecasts.
            This model supports past covariates (known for `input_chunk_length` points before prediction time).

            Parameters
            ----------
            input_chunk_length
                Number of time steps in the past to take as a model input (per chunk). Applies to the target
                series and past covariates (if provided).
            output_chunk_length
                Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
                from future covariates to use as a model input (if the model supports future covariates). It is not the same
                as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
                using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
                auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
                the model from using future values of past and / or future covariates for prediction (depending on the
                model's covariate support).
            output_chunk_shift
                Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
                input chunk end). This will create a gap between the input and output. If the model supports
                `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
                `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
                cannot generate autoregressive predictions (`n > output_chunk_length`).
            num_layers
                The number of fully connected layers preceding the final output layer.
            layer_width
                The number of neurons that make up each fully connected layer.
            dropout
                The dropout probability to be used in fully connected layers. This is compatible with Monte Carlo dropout
                at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
                prediction time).
            activation
                The activation function of intermediate layers (default='ReLU').
                Supported activations: ['ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid',
                'GELU']
            batch_norm
                Whether to use batch normalization after each hidden layer.
            **kwargs
                Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
                Darts' :class:`TorchForecastingModel`.

            loss_fn
                PyTorch loss function used for training.
                This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
                Default: ``torch.nn.MSELoss()``.
            likelihood
                One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` models to be used for
                probabilistic forecasts. Default: ``None``.
            torch_metrics
                A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
                at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
            optimizer_cls
                The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
            optimizer_kwargs
                Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
                for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``
                will be used. Default: ``None``.
            lr_scheduler_cls
                Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
                to using a constant learning rate. Default: ``None``.
            lr_scheduler_kwargs
                Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
            use_reversible_instance_norm
                Whether to use reversible instance normalization `RINorm` against distribution shift.
                It is only applied to the features of the target series and not the covariates.
            batch_size
                Number of time series (input and output sequences) used in each training pass. Default: ``32``.
            n_epochs
                Number of epochs over which to train the model. Default: ``100``.
            model_name
                Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
                defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
                of the name is formatted with the local date and time, while PID is the processed ID (preventing models
                spawned at the same time by different processes to share the same model_name). E.g.,
                ``"2021-06-14_09_53_32_torch_model_run_44607"``.
            work_dir
                Path of the working directory, where to save checkpoints and Tensorboard summaries.
                Default: current working directory.
            log_tensorboard
                If set, use Tensorboard to log the different parameters. The logs will be located in:
                ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
            nr_epochs_val_period
                Number of epochs to wait before evaluating the validation loss (if a validation
                ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
            force_reset
                If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
                be discarded). Default: ``False``.
            save_checkpoints
                Whether to automatically save the untrained model and checkpoints from training.
                To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
                :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
                :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
                :func:`save()` and loaded using :func:`load()`. Default: ``False``.
            add_encoders
                A large number of past and future covariates can be automatically generated with `add_encoders`.
                This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
                will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
                transform the generated covariates. This happens all under one hood and only needs to be specified at
                model creation.
                Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
                ``add_encoders``. Default: ``None``.
            random_state
                Controls the randomness of the weights initialization and reproducible forecasting.
            pl_trainer_kwargs
                By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
                that performs the training, validation and prediction processes. These presets include automatic
                checkpointing, tensorboard logging, setting the torch device and more.
                With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
                object. Check the `PL Trainer documentation
                <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
                supported kwargs. Default: ``None``.
            show_warnings
                whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
                your forecasting use case. Default: ``False``.

            Examples
            --------
            >>> from darts.datasets import WeatherDataset
            >>> from darts.models import MLPModel
            >>> series = WeatherDataset().load()
            >>> # predicting atmospheric pressure
            >>> target = series['p (mbar)'][:100]
            >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
            >>> past_cov = series['rain (mm)'][:100]
            >>> model = MLPModel(
            >>>     input_chunk_length=6,
            >>>     output_chunk_length=6,
            >>>     n_epochs=5,
            >>>     num_layers=3,
            >>>     layer_width=128
            >>> )
            >>> model.fit(target, past_covariates=past_cov)
            >>> pred = model.predict(6)
            """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

    @property
    def supports_multivariate(self) -> bool:
        """MLPModel supports multivariate time series."""
        return True

    def _supports_static_covariates(self) -> bool:
        """MLPModel does not support static covariates."""
        return False
    
    def _create_model(self, train_sample) -> torch.nn.Module:
        # samples are made of (past target, past cov, historic future cov, future cov, static cov, future_target)
        (past_target, past_covariates, _, _) = train_sample
        
        # Calculate input dimension (target + past covariates if present)
        input_dim = past_target.shape[1] + (
            past_covariates.shape[1] if past_covariates is not None else 0
        )
        output_dim = past_target.shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            **self.pl_module_params,
        )

class _MLPModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        num_layers: int,
        layer_width: int,
        dropout: float,
        activation: str,
        batch_norm: bool,
        **kwargs,
    ):
        """PyTorch Lightning Module for the MLP architecture.

        Parameters
        ----------
        input_dim
            Number of input features (target dimensions + covariate dimensions).
        output_dim
            Number of output dimensions (target dimensions).
        nr_params
            Number of parameters for the likelihood (1 for deterministic forecasts).
        num_layers
            Number of hidden layers.
        layer_width
            Width of each hidden layer.
        dropout
            Dropout probability.
        activation
            Activation function name.
        batch_norm
            Whether to use batch normalization.
        **kwargs
            Additional PyTorch Lightning module parameters.
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        # Calculate total input size: input_dim * input_chunk_length
        self.input_size = input_dim * self.input_chunk_length
        # Calculate total output size: output_dim * output_chunk_length * nr_params
        self.output_size = output_dim * self.output_chunk_length * nr_params

        # Build the MLP
        self.mlp = self._build_mlp()

    def _build_mlp(self) -> nn.Module:
        """Build the MLP network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, self.layer_width))
        layers.append(getattr(nn, self.activation)())
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.layer_width, self.layer_width))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.layer_width))
            
            layers.append(getattr(nn, self.activation)())
            
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
        
        # Output layer
        layers.append(nn.Linear(self.layer_width, self.output_size))
        
        return nn.Sequential(*layers)

    def forward(self, x_in):
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x_in : tuple
            Tuple of (past_target, past_covariates, future_covariates) tensors.
            Each tensor has shape (batch_size, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, output_chunk_length, output_dim, nr_params).
        """
        # Extract past target and past covariates
        past_target, past_covariates = x_in
        
        # Concatenate target and covariates if covariates exist
        if past_covariates is not None:
            x = torch.cat([past_target, past_covariates], dim=2)
        else:
            x = past_target
        
        # Flatten to (batch_size, input_size)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Forward pass through MLP
        y = self.mlp(x)
        
        # Reshape to (batch_size, output_chunk_length, output_dim, nr_params)
        y = y.reshape(batch_size, self.output_chunk_length, self.output_dim, self.nr_params)
        
        return y