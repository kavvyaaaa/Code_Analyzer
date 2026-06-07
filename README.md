# AI-Powered Code Quality & Defect Analyzer

Analyze, visualize, and predict software defects using PROMISE research benchmarks, Git repository mining, static code metrics, and machine learning.

## Features

- **PROMISE Benchmark Explorer** — Load standard defect-prediction datasets (Ant, Camel, jEdit, Lucene, Tomcat), explore CK metrics, and train classifiers.
- **Git Repository Analyzer** — Mine commit history for code churn and bug-fix labels, extract static metrics from Python/Java files, and predict defect hotspots.
- **Single File Inspector** — Paste or upload source code for real-time static analysis, defect probability scoring, and refactoring recommendations.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data collection | Python, GitPython |
| Static analysis | Radon (Python), regex-based parser (Java) |
| ML | Scikit-learn, XGBoost |
| Data / viz | Pandas, NumPy, Plotly, Matplotlib |
| UI | Streamlit |

## Project Structure

```
.
├── app.py                  # Streamlit dashboard
├── test_analyzer.py        # Integration tests
├── requirements.txt
├── data/                   # Cached PROMISE datasets (auto-downloaded)
└── src/
    ├── dataset_manager.py  # PROMISE dataset loading & preprocessing
    ├── git_miner.py        # Git history & churn mining
    ├── metrics_extractor.py# LOC, complexity, coupling metrics
    └── model_trainer.py    # LR / RF / XGBoost training & evaluation
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests

```bash
python test_analyzer.py
```

### 3. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

### Tab 1 — PROMISE Benchmark Explorer

1. Select a dataset (e.g. Apache Ant 1.7).
2. Explore defect distribution, metric scatter plots, and correlation heatmaps.
3. Choose a model (Logistic Regression, Random Forest, or XGBoost) and click **Train Model Now**.
4. Review ROC-AUC, confusion matrix, and feature importances.

### Tab 2 — Git Repository Analyzer

1. Enter a local repo path or remote Git URL (presets include Flask and Requests).
2. Click **Start Git & Code Analysis** to mine commits and extract static metrics.
3. Optionally train a repo-specific defect predictor.
4. View defect hotspot predictions on a churn vs. complexity chart.

### Tab 3 — Single File Inspector

1. Paste Python or Java code (or select a file mined in Tab 2).
2. Click **Inspect Code Quality** for LOC, complexity, coupling metrics.
3. If a model was trained in Tab 1 or Tab 2, see defect probability and refactoring advice.

## Datasets

PROMISE datasets are downloaded automatically on first use and cached in `data/`:

| Key | Project | Source |
|-----|---------|--------|
| ant | Apache Ant 1.7 | PROMISE-backup |
| camel | Apache Camel 1.6 | PROMISE-backup |
| jedit | jEdit 4.3 | PROMISE-backup |
| lucene | Apache Lucene 2.4 | PROMISE-backup |
| tomcat | Apache Tomcat 6.0 | DefectData |

## Metrics Extracted

**Static (CK-style):** LOC, WMC, DIT, NOC, CBO, cyclomatic complexity, method count

**Git-derived:** commit frequency, code churn, bug-fix count, author count

<!-- ## Research Question

> Which software engineering metrics best predict future defects?

Across the Ant benchmark, **LOC**, **RFC** (response for class), and **coupling (CBO)** consistently rank among the top predictive features in Random Forest and XGBoost models — aligning with published defect-prediction literature. -->

## Requirements

- Python 3.10+
- Git (for repository mining features)

## Author
Made by KAVYA RAJ