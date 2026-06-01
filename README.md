# Bitcoin Fraud Benchmark

A leakage-aware benchmark for realistic Bitcoin fraud detection under strict temporal evaluation.

## Overview

This repository accompanies the paper:

**"Beyond Aggregate Metrics: Temporal Leakage, Structural Exposure, and Realistic Evaluation in Bitcoin Fraud Detection"**

The benchmark is built from six chronologically ordered Bitcoin blockchain snapshots spanning January to September 2025 and contains more than 1.68 million transactions.

The framework was designed to address common evaluation issues in blockchain fraud detection, including:

* Temporal leakage
* Retrospective intelligence aggregation
* Unrealistic train-test protocols
* Hidden dependence on historical structural exposure

## Main Contributions

* Six chronological Bitcoin snapshots (D1--D6)
* Strict forward temporal evaluation
* Temporally verified external intelligence labels
* Leakage-controlled feature engineering
* Temporal address-memory features
* Structural exposure analysis
* Operational fraud-detection evaluation

## Main Results

Final LightGBM model evaluated on the fully held-out D5--D6 test period:

| Metric    | Value  |
| --------- | ------ |
| AUC-ROC   | 0.9643 |
| AUC-PR    | 0.7657 |
| F1-score  | 0.7557 |
| Precision | 0.7740 |
| Recall    | 0.7383 |

## Repository Structure

```text
data/           Benchmark metadata
src/            Dataset construction and feature generation
experiments/    Training and evaluation scripts
results/        Tables and figures from the paper
```

## Reproducibility

The repository contains the code, experimental protocols, feature definitions, and evaluation procedures required to reproduce the results reported in the paper.

## License

MIT License.
