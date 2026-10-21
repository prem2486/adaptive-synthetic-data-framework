# Adaptive Synthetic Dataset Generation and Evaluation Framework

A Python-based framework for generating, evaluating, comparing, and selecting synthetic tabular datasets using multiple generation algorithms.

## Overview

Synthetic tabular data can support software testing, experimentation, data analysis, and machine learning when access to original datasets is limited. However, synthetic data must be evaluated carefully because a dataset that looks realistic may still fail to preserve statistical relationships or provide useful predictive performance.

This project provides a framework for comparing multiple synthetic data generation approaches through a common workflow. It includes dataset loading and profiling, synthetic data generation, statistical evaluation, constraint checking, and comparative results presented through a Streamlit interface.

## Key Features

- Load and inspect tabular datasets in CSV format.
- Profile columns and identify numerical and categorical variables.
- Compare multiple synthetic data generation approaches.
- Evaluate statistical similarity between reference and synthetic datasets.
- Assess correlation preservation and categorical distributions.
- Check configured data constraints and numerical ranges.
- Evaluate downstream predictive utility using the Train on Synthetic, Test on Real (TSTR) approach, where supported.
- Compare candidate generators using a composite evaluation score.
- View evaluation results through an interactive Streamlit dashboard.
- Export generated datasets and evaluation results where supported by the application.

## Supported Generation Approaches

The framework is designed to evaluate the following approaches, subject to the installed dependencies and the implementation available in the current version:

- Gaussian Copula
- Gaussian Mixture Model (GMM)
- Conditional Tabular GAN (CTGAN)
- CopulaGAN
- Tabular Variational Autoencoder (TVAE)
- Tabular Denoising Diffusion Probabilistic Model (TabDDPM)
- Large language model-assisted synthetic data generation workflow

The LLM-assisted workflow may require manual interaction with an external browser-based service. It should not be interpreted as a direct API integration unless one is explicitly implemented.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Kolmogorov–Smirnov (KS) statistic | Measures differences between numerical distributions |
| Wasserstein distance | Measures distributional differences |
| Mean/standard deviation gap | Compares numerical summary statistics |
| Correlation MAE | Measures differences between correlation matrices |
| Categorical total variation (TV) | Compares categorical distributions |
| Constraint violation rate | Measures violations of configured data rules |
| TSTR weighted F1 | Evaluates predictive utility using synthetic training data and real test data |
| Composite score | Combines selected metrics into a comparative score |

**Note:** Metric definitions, normalization, weights, and missing-value handling depend on the implementation. A higher composite score should be interpreted according to the scoring formula used by the application.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Statistical and synthetic data generation libraries
- PyTorch and other model-specific dependencies, where required

The exact dependencies and supported Python versions should be confirmed against `requirements.txt`.

## Project Structure

The following is a suggested structure. Adjust it to match the actual repository and Python imports.

```text
adaptive-synthetic-data-framework/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── engine/
│   ├── __init__.py
│   ├── engine.py
│   ├── profiling.py
│   ├── statistics.py
│   ├── arbitration.py
│   ├── copula.py
│   ├── gmm.py
│   ├── ctgan.py
│   ├── tvae.py
│   └── tabddpm.py
├── tests/
└── data/
    └── README.md
```

Some modules may be optional or located elsewhere in the current implementation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the example repository address with the actual GitHub URL.

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some generation algorithms may require additional dependencies, compatible package versions, or suitable hardware. Follow the requirements and instructions included with the repository.

## Run the Application

From the project root, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Typical Workflow

1. Load a reference dataset.
2. Inspect its schema, data types, distributions, and missing values.
3. Select an available synthetic data generation method.
4. Generate one or more candidate datasets.
5. Evaluate candidates using the available statistical, constraint, and predictive metrics.
6. Compare the evaluation results.
7. Select a candidate based on the chosen evaluation criteria.
8. Export the generated dataset and results, where supported.

## Experimental Dataset

The project has been explored using a simulated port-operations dataset inspired by the Visakhapatnam port context.

The dataset represents simulated observations rather than verified individual port-call records. Any comparison with official annual port statistics must document the mapping between row-level variables and published indicators, including aggregation rules and units.

Do not interpret agreement with annual summary statistics alone as proof that synthetic records reproduce real operational behaviour.

## Reproducibility

For reproducible experiments, record:

- The reference dataset and its source.
- The dataset version, dimensions, and schema.
- Random seeds and generation parameters.
- Python and dependency versions.
- Evaluation metric definitions and composite-score weights.
- The number of generated rows.
- Any preprocessing, post-processing, or constraint enforcement.
- The date and configuration of each experiment.

Use only datasets that you are permitted to publish. Remove credentials, API keys, private records, and confidential company information before uploading files to a public repository.

## Testing

If the repository contains automated tests, run them from the project root:

```bash
pytest
```

Install the test dependencies first if they are not included in `requirements.txt`. Test coverage and passing status should be reported only after running the tests in the target environment.

## Limitations

- Results depend on dataset size, feature types, model configuration, and random initialization.
- Not every generator is suitable for every dataset.
- Statistical similarity does not guarantee privacy or prevent disclosure of sensitive information.
- Constraint checks only verify rules explicitly implemented by the framework.
- Predictive utility depends on the selected prediction task, target variable, data split, and evaluation procedure.
- Composite scores are comparative summaries, not universal measures of synthetic-data quality.
- External annual benchmarks may use definitions and aggregation levels that differ from the synthetic dataset.

## Citation

If you use this software in academic work, cite the repository's versioned release or archived DOI when one is available.

A citation can be added here after the authors, release version, publication details, and repository URL have been finalized.

## License

This project is intended to be distributed under the **GNU General Public License v3.0 (GPL-3.0)**, subject to confirmation that you own or have permission to license all included code and assets.

The repository must contain the complete `LICENSE` file, not just this notice. Review third-party code and dependency licenses before publishing.

## Author

**Project:** Adaptive Synthetic Dataset Generation and Evaluation Framework  
**Language:** Python  
**Interface:** Streamlit

