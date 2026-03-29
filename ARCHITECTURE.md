# SHAP (SHapley Additive exPlanations)

## Project Overview

SHAP is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.

The project is primarily a Python library with:
*   **C++ and CUDA extensions** for high-performance computation (exact algorithms for tree ensembles).
*   **JavaScript components** for interactive visualizations (React/D3).
*   **Deep Learning support** for TensorFlow, Keras, and PyTorch.

## Architecture & Structure

*   `shap/`: Core Python source code.
    *   `cext/`: C++ and CUDA extension sources.
    *   `explainers/`: Implementation of various explainers (Tree, Deep, Kernel, etc.).
    *   `plots/`: Visualization tools.
*   `javascript/`: React/D3 based visualization library (bundled into the Python package).
*   `tests/`: `pytest` suite.
*   `docs/`: Sphinx documentation.
*   `notebooks/`: Extensive Jupyter notebook examples.

## Building and Running

### Python Environment

1.  **Installation (Editable):**
    ```bash
    pip install -e '.[test,plots,docs]'
    ```
    *   This compiles C++ extensions automatically.
    *   If CUDA is available, GPU extensions are also built.

2.  **Dependencies:**
    *   Core: `numpy`, `scipy`, `scikit-learn`, `pandas`.
    *   Build: `setuptools`, `numpy>=2.0` (for building extensions).
    *   See `pyproject.toml` for full lists of optional dependencies (plots, deep learning frameworks, etc.).

### JavaScript (Visualizations)

The JS code is bundled and included in the Python package. If modifying visualizations:

1.  Navigate to `javascript/`:
    ```bash
    cd javascript
    ```
2.  Install & Build:
    ```bash
    npm install
    npm run build
    ```
3.  **Note:** The build artifact `bundle.js` must be copied to `shap/plots/resources/bundle.js` (CI does this automatically, but local dev might need a manual copy if not using the webpack dev server).

## Testing & Quality

*   **Test Runner:** `pytest`
    ```bash
    pytest
    ```
    *   Configuration in `pyproject.toml` sets `addopts = "--mpl -m 'not xslow'"`.
    *   Use `--import-mode=append` if running against installed package vs local source.

*   **Linting & Formatting:** `ruff`
    ```bash
    ruff check .
    ```
    *   Pre-commit hooks are available: `pre-commit run --all-files`.

*   **Type Checking:** `mypy`
    ```bash
    mypy shap tests
    ```

## Documentation

*   Built with Sphinx.
*   Source in `docs/`.
*   Build command:
    ```bash
    cd docs
    make html
    ```

## Key Development Notes

*   **Numpy Compatibility:** The project builds against `numpy>=2.0` but supports older versions at runtime (via ABI handling).
*   **Extensions:** `setup.py` handles C++/CUDA compilation. It attempts to find CUDA/NVCC automatically.
*   **CI:** GitHub Actions workflows (`.github/workflows/`) run tests across multiple OSs and Python versions.
