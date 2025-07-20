<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="faircredit-bias-aware-scoring.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# FAIRCREDIT-BIAS-AWARE-SCORING

<em>Empowering Fairness, Transforming Credit Decisions</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/Guri10/faircredit-bias-aware-scoring?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/Guri10/faircredit-bias-aware-scoring?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/Guri10/faircredit-bias-aware-scoring?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Flask-000000.svg?style=flat&logo=Flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Typer-000000.svg?style=flat&logo=Typer&logoColor=white" alt="Typer">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/OpenSSL-721412.svg?style=flat&logo=OpenSSL&logoColor=white" alt="OpenSSL">
<img src="https://img.shields.io/badge/SQLAlchemy-D71F00.svg?style=flat&logo=SQLAlchemy&logoColor=white" alt="SQLAlchemy">
<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat&logo=TOML&logoColor=white" alt="TOML">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/Babel-F9DC3E.svg?style=flat&logo=Babel&logoColor=black" alt="Babel">
<img src="https://img.shields.io/badge/Rich-FAE742.svg?style=flat&logo=Rich&logoColor=black" alt="Rich">
<img src="https://img.shields.io/badge/Gunicorn-499848.svg?style=flat&logo=Gunicorn&logoColor=white" alt="Gunicorn">
<img src="https://img.shields.io/badge/Celery-37814A.svg?style=flat&logo=Celery&logoColor=white" alt="Celery">
<br>
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/DVC-13ADC7.svg?style=flat&logo=DVC&logoColor=white" alt="DVC">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/SQLite-003B57.svg?style=flat&logo=SQLite&logoColor=white" alt="SQLite">
<img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=flat&logo=MLflow&logoColor=white" alt="MLflow">
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=flat&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
<img src="https://img.shields.io/badge/AIOHTTP-2C5BB4.svg?style=flat&logo=AIOHTTP&logoColor=white" alt="AIOHTTP">
<img src="https://img.shields.io/badge/SemVer-3F4551.svg?style=flat&logo=SemVer&logoColor=white" alt="SemVer">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat&logo=YAML&logoColor=white" alt="YAML">

</div>
<br>

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

---

## Overview

faircredit-bias-aware-scoring is an end-to-end framework designed to develop fair and unbiased credit scoring models with reproducible pipelines. It integrates data processing, model training, fairness evaluation, and deployment automation to support equitable financial decision-making.

**Why faircredit-bias-aware-scoring?**

This project empowers developers to build, evaluate, and deploy bias-mitigated credit scoring systems efficiently and reliably. The core features include:

- 🧪 **Reproducible Environments:** Ensures consistent setup across development and deployment with environment.yml.
- 🔍 **Fairness & Bias Mitigation:** Provides tools and notebooks for assessing and improving model fairness.
- ⚙️ **Structured Data Pipelines:** Manages data preprocessing, feature engineering, and model training workflows with DVC.
- 🚀 **Automated CI/CD:** Streamlines testing, model deployment, and API validation via GitHub workflows.
- 📊 **Model & Data Management:** Facilitates seamless model referencing and structured input data for robust scoring.

---

## Project Structure

```sh
└── faircredit-bias-aware-scoring/
    ├── .github
    │   └── workflows
    ├── README.md
    ├── data
    │   ├── .DS_Store
    │   ├── processed
    │   └── raw
    ├── dvc.lock
    ├── dvc.yaml
    ├── environment.yml
    ├── model_uri.txt
    ├── notebooks
    │   ├── .ipynb_checkpoints
    │   ├── 01_explore.ipynb
    │   ├── 02_preprocess.ipynb
    │   ├── 03_train_baseline.ipynb
    │   └── 04_fairness.ipynb
    ├── payload.csv
    ├── payload_records.json
    ├── payload_split.json
    ├── scripts
    │   ├── .ipynb_checkpoints
    │   └── generate_payload.py
    ├── src
    │   ├── .ipynb_checkpoints
    │   ├── __init__.py
    │   ├── fairness.py
    │   ├── preprocessing.py
    │   ├── run_preprocess.py
    │   └── train_baseline.py
    └── tests
        ├── .ipynb_checkpoints
        └── test_api.py
```

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Conda

### Installation

Build faircredit-bias-aware-scoring from the source and install dependencies:

1. **Clone the repository:**

   ```sh
   ❯ git clone https://github.com/Guri10/faircredit-bias-aware-scoring
   ```

2. **Navigate to the project directory:**

   ```sh
   ❯ cd faircredit-bias-aware-scoring
   ```

3. **Install the dependencies:**

**Using [conda](https://docs.conda.io/):**

```sh
❯ conda env create -f environment.yml
```

### Usage

Run the project with:

**Using [conda](https://docs.conda.io/):**

```sh
conda activate {venv}
python {entrypoint}
```

### Testing

Faircredit-bias-aware-scoring uses the {**test_framework**} test framework. Run the test suite with:

**Using [conda](https://docs.conda.io/):**

```sh
conda activate {venv}
pytest
```

---

<div align="left"><a href="#top">⬆ Return</a></div>

---
