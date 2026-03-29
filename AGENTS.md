## Project Overview

SHAP (SHapley Additive exPlanations) is a Python library for explaining machine learning model predictions using game-theoretic Shapley values. It provides model-agnostic and model-specific explainers for various ML frameworks.

## Build and Development Commands

### Installation from source
```bash
pip install --editable '.[test,plots,docs]'
```
Requires a C compiler (gcc on Linux, mingw64 on Windows). CUDA extensions are optional.

### Running tests
```bash
pytest                           # Run all tests (excludes xslow tests)
pytest tests/explainers/test_tree.py  # Run single test file
pytest -k "test_name"            # Run tests matching pattern
pytest -m xslow                  # Run extremely slow tests (normally skipped)
```

### Plot baseline regeneration
When plot output changes, regenerate baselines before tests will pass:
```bash
pytest tests/plots --mpl-generate-path=tests/plots/baseline
```

### Linting and formatting
```bash
pre-commit install               # Setup pre-commit hooks
pre-commit run --all-files       # Run all checks
ruff check .                     # Lint only
ruff format .                    # Format only
```

### Type checking
```bash
mypy shap                        # Type check (some modules excluded in pyproject.toml)
```

### Building documentation
```bash
cd docs && make html             # Requires pandoc for nbsphinx
```

## Architecture

### Core Components

**Explanation** (`shap/_explanation.py`): The central data structure returned by all explainers. A sliceable container holding SHAP values, base values, data, and feature names. Supports numpy-style operations (sum, mean, abs, etc.) and hierarchical clustering.

**Explainer** (`shap/explainers/_explainer.py`): Base class and smart dispatcher. When called as `shap.Explainer(model)`, it auto-selects the appropriate specialized explainer based on model type.

### Explainer Hierarchy

Model-specific (fast, exact):
- `TreeExplainer`: XGBoost, LightGBM, CatBoost, sklearn trees, pyspark
- `LinearExplainer`: Linear models with optional feature correlation
- `DeepExplainer`: TensorFlow/Keras (based on DeepLIFT)
- `GradientExplainer`: TensorFlow/Keras/PyTorch (expected gradients)
- `GPUTreeExplainer`: CUDA-accelerated tree explanations

Model-agnostic (slower, approximate):
- `KernelExplainer`: Works with any model via weighted linear regression
- `SamplingExplainer`: Monte Carlo sampling approach
- `PermutationExplainer`: Permutation-based SHAP values
- `PartitionExplainer`: Hierarchical feature grouping
- `ExactExplainer`: Exact Shapley values (exponential complexity)

### Maskers (`shap/maskers/`)

Define how features are "masked" (replaced with background values) during explanation:
- `Independent`/`Partition`: Tabular data with independent/grouped features
- `Impute`: Tabular with imputation strategies
- `Text`: Token-level masking for NLP
- `Image`: Superpixel-based masking for images

### Plots (`shap/plots/`)

Visualization functions that operate on Explanation objects:
- `waterfall`, `force`: Individual prediction explanations
- `beeswarm`, `bar`: Global feature importance
- `scatter`, `dependence`: Feature effect plots
- `heatmap`, `decision`: Multi-sample views
- `text`, `image`: Domain-specific visualizations

### C Extensions (`shap/cext/`)

Performance-critical code in Cython for tree algorithms and GPU acceleration.

## Code Conventions

- Uses `ruff` for linting/formatting (numpy docstring convention)
- Type hints are being incrementally added (see `mypy.overrides` in pyproject.toml for current status)
- PR titles should be prefixed: `FIX:`, `ENH:`, `DOCS:`, etc.
- Minimum Python 3.11, NumPy 2.0

## Testing Notes

- Tests use `pytest-mpl` for plot comparison
- Some tests require optional dependencies (tensorflow, torch, xgboost, etc.)
- Tests skip gracefully when optional dependencies are missing
- `test-core` extra provides minimal test dependencies, `test` provides full suite
