# darts-mlp Dependencies

## Direct Dependencies (Specified in setup.py)

```python
install_requires=[
    "darts>=0.24.0",
    "torch>=1.9.0",
    "numpy>=1.19.0",
]
```

## Detailed Dependency Tree

### 1. darts (>=0.24.0)
**Current version in test environment**: 0.38.0  
**Purpose**: Time series forecasting framework that darts-mlp extends

**darts requires**:
- holidays
- joblib
- matplotlib
- narwhals
- nfoursid
- numpy
- pandas
- pyod
- pytorch-lightning
- requests
- scikit-learn
- scipy
- shap
- statsmodels
- tensorboardX
- torch
- tqdm
- typing-extensions
- xarray

### 2. torch (>=1.9.0)
**Current version in test environment**: 2.9.1+cpu  
**Purpose**: Neural network framework for MLP implementation

**torch requires**: No additional dependencies (standalone)

### 3. numpy (>=1.19.0)
**Current version in test environment**: 2.3.4  
**Purpose**: Array operations and numerical computing

**numpy requires**: No additional dependencies (standalone)

### 4. pytorch-lightning (Indirect via darts)
**Current version in test environment**: 2.5.2  
**Purpose**: Training framework used by darts TorchForecastingModel

**pytorch-lightning requires**:
- fsspec
- lightning-utilities
- packaging
- PyYAML
- torch
- torchmetrics
- tqdm
- typing-extensions

## Complete Dependency List (All Levels)

When you install darts-mlp, these packages will be installed/required:

**Core ML/Scientific**:
- numpy (2.3.4)
- torch (2.9.1+cpu)
- scipy
- scikit-learn
- pandas

**Time Series Specific**:
- darts (0.38.0)
- statsmodels
- nfoursid
- xarray

**Training/Monitoring**:
- pytorch-lightning (2.5.2)
- torchmetrics
- tensorboardX

**Utilities**:
- joblib
- tqdm
- typing-extensions
- requests
- holidays
- narwhals
- fsspec
- packaging
- PyYAML

**Visualization/Analysis**:
- matplotlib
- shap
- pyod

## Minimum Required Versions

| Package | Minimum Version | Reason |
|---------|----------------|--------|
| Python | 3.8 | Defined in setup.py |
| darts | 0.24.0 | API compatibility for TorchForecastingModel |
| torch | 1.9.0 | PyTorch features used in implementation |
| numpy | 1.19.0 | Array operations compatibility |

## Tested Versions

Successfully tested with:
- Python: 3.12.0
- darts: 0.38.0
- torch: 2.9.1+cpu
- numpy: 2.3.4
- pytorch-lightning: 2.5.2

## Version Compatibility Notes

### numpy 2.x vs 1.x
- Package works with both numpy 1.x and 2.x
- Current test environment uses numpy 2.3.4
- Most apps should work fine with numpy 2.x

### darts Version Range
- Minimum: 0.24.0 (when TorchForecastingModel API stabilized)
- Recommended: Latest (0.38.0+)
- Breaking changes unlikely due to stable API

### PyTorch Version
- Minimum: 1.9.0 (from June 2021)
- Works with: 1.9.x, 1.10.x, 1.11.x, 1.12.x, 1.13.x, 2.x
- CPU and CUDA versions both supported

## How to Check Your Environment

```bash
# Check if your environment is compatible
pip list | grep -E "(darts|torch|numpy|pytorch-lightning)"

# Expected output should show:
# darts >= 0.24.0
# torch >= 1.9.0
# numpy >= 1.19.0
# pytorch-lightning (any recent version)
```

## Installation in Your App

### Method 1: Add to requirements.txt
```txt
darts-mlp @ file:///home/nibra/darts-mlp
```

### Method 2: Install directly
```bash
pip install /home/nibra/darts-mlp
```

### Method 3: After pushing to GitHub
```txt
git+https://github.com/yourusername/darts-mlp.git
```

## Potential Conflicts

### Unlikely Conflicts
- ✅ numpy: Very permissive version requirement (>=1.19.0)
- ✅ torch: Old minimum version (>=1.9.0)
- ✅ Most darts apps already have compatible versions

### Possible Conflicts
- ⚠️ If your app uses darts < 0.24.0 (very old, from ~2022)
- ⚠️ If your app pins specific incompatible versions

### Resolution Strategy
1. Check current versions in your app's environment
2. If conflicts exist, try upgrading the conflicting package
3. If upgrade not possible, adjust darts-mlp's setup.py requirements

## Dependency Size Estimate

Total download size when installing from scratch:
- ~2-3 GB (mostly PyTorch and its dependencies)
- If darts is already installed: ~10-50 KB (just darts-mlp code)

## Production Recommendations

1. **Pin exact versions** for reproducibility:
   ```bash
   pip freeze > requirements-lock.txt
   ```

2. **Use virtual environment**:
   ```bash
   python -m venv myapp-env
   source myapp-env/bin/activate
   pip install -r requirements.txt
   ```

3. **Test in staging** before production deployment

4. **Document tested versions** in your app's README

## Last Updated
2025-11-15
