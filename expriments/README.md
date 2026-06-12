# Experiments

This directory contains the scripts required to reproduce the benchmark construction process, feature engineering pipeline, labeling framework, model training procedures, and evaluation experiments reported in the paper.

## Dataset Construction

### `collect_bitcoin_mempool.py`

Collects raw Bitcoin blockchain data from the Mempool API and builds the chronological blockchain snapshots used throughout the benchmark.

Main functions:

* Block and transaction collection
* Transaction input/output extraction
* Snapshot generation
* SQLite database construction
* Data integrity verification

Output:

* Snapshot databases
* Consolidated Bitcoin transaction database

---

### `build_scam_cache.py`

Builds the external-intelligence repository used for fraud labeling.

Sources include:

* ChainAbuse
* BTCBlack
* OFAC SDN sanctions list
* CryptoScamDB

Main functions:

* Address collection
* Confidence-tier assignment
* BTCBlack validation
* Temporal report aggregation
* Intelligence-cache generation

Output:

* Frozen external-intelligence cache
* Intelligence summary reports

---

### `build_bitfraud_v21_dataset.py`

Constructs the final BITFRAUD benchmark dataset.

Main functions:

* Leakage-controlled feature engineering
* Temporal address-memory computation
* Strict temporal label validation
* Heuristic fraud detection
* Isolation Forest anomaly detection
* Final label generation
* H/L separation validation

Output:

* Final benchmark dataset
* Learning-feature dataset
* Heuristic-feature dataset
* Labeling reports
* Correlation validation reports

---

## Benchmark Evaluation

### `train_3ml_models.py`

Trains and evaluates the benchmark machine-learning baselines.

Models:

* LightGBM
* XGBoost
* CatBoost

The script performs:

* Hyperparameter optimisation
* Validation-based threshold selection
* Strict forward temporal evaluation
* Performance reporting on fully held-out test periods

---

### `temporal_horizon.py`

Evaluates model performance across multiple temporal prediction horizons.

Protocols include:

* Short-horizon evaluation
* Medium-horizon evaluation
* Long-horizon evaluation

The analysis quantifies the effect of temporal distance between training and testing periods.

---

### `structural_exposure.py`

Measures structural exposure and connectivity effects within the benchmark.

The analysis identifies:

* Fraud-exposed regions
* History-exposed regions
* Structurally isolated transactions

and quantifies how graph connectivity influences fraud-detection performance.

---

### `bootstrap_precision_at_k.py`

Performs statistical validation of reported results using bootstrap resampling.

Outputs include:

* Confidence intervals for AUC-ROC
* Confidence intervals for AUC-PR
* Confidence intervals for F1-score
* Precision@K evaluation
* Stability analysis

---

### `shap_analysis.py`

Generates model explainability analyses for the final LightGBM model.

Outputs include:

* Global feature importance
* SHAP summary plots
* Feature contribution rankings
* Interpretation of fraud-detection behaviour

---

## Experimental Protocol

All experiments follow the strict forward temporal evaluation protocol described in the paper.

Training snapshots:

* D1
* D2
* D3

Validation snapshot:

* D4

Test snapshots:

* D5
* D6

No future transaction, future label, future address history, or future external-intelligence information is used during training or validation.

## Reproducibility

The scripts contained in this directory provide the complete experimental workflow required to reproduce the benchmark dataset, labeling process, feature-engineering pipeline, baseline models, statistical validation procedures, and explainability analyses reported in the paper.

