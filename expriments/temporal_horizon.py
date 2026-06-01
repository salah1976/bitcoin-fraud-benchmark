"""
Temporal horizon analysis for the leakage-aware Bitcoin fraud benchmark.

This script evaluates the final LightGBM configuration under three forward
chronological settings:

1. Short horizon:  train D1, validate D2, test D3
2. Medium horizon: train D1+D2, validate D3, test D4
3. Long horizon:   train D1+D2+D3, validate D4, test D5+D6

The script expects a processed learning dataset containing the 15 learning
features, snapshot identifiers, and the final label column.

Example
-------
python experiments/temporal_horizon.py \
    --data-path data/processed/dataset_learning.csv \
    --out-dir results/temporal_horizon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
TARGET = "label_final"
SNAPSHOT_COL = "snapshot_id"

FEATURES: List[str] = [
    "input_count",
    "output_count",
    "input_addr_count",
    "coinbase_flag",
    "has_witness",
    "script_type_encoded",
    "input_addr_concentration",
    "io_count_ratio",
    "tx_weight",
    "avg_input_value",
    "total_input_scaled",
    "log_output_value",
    "fee_ratio",
    "prev_addr_seen_ratio",
    "prev_addr_seen_count",
]

EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "short_horizon_D1_to_D3",
        "train": ["D1"],
        "val": ["D2"],
        "test": ["D3"],
        "description": "Train on D1, validate on D2, test on D3",
    },
    {
        "name": "medium_horizon_D1D2_to_D4",
        "train": ["D1", "D2"],
        "val": ["D3"],
        "test": ["D4"],
        "description": "Train on D1-D2, validate on D3, test on D4",
    },
    {
        "name": "long_horizon_D1D2D3_to_D5D6",
        "train": ["D1", "D2", "D3"],
        "val": ["D4"],
        "test": ["D5", "D6"],
        "description": "Train on D1-D3, validate on D4, test on D5-D6",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightGBM temporal horizon analysis."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the processed learning dataset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/temporal_horizon",
        help="Directory where result files will be written.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET,
        help="Binary target column name. Default: label_final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed. Default: 42.",
    )
    return parser.parse_args()


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))

    if best_idx >= len(thresholds):
        return 0.5, float(f1[best_idx])

    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def clean_features(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    missing = [c for c in features + [target, SNAPSHOT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[target] = df[target].astype(int)

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def make_lightgbm(scale_pos_weight: float, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=64,
        max_depth=-1,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_samples=50,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def train_eval_experiment(
    df: pd.DataFrame,
    exp: Dict[str, Any],
    features: List[str],
    target: str,
    seed: int,
) -> Dict[str, Any]:
    train_df = df[df[SNAPSHOT_COL].isin(exp["train"])].copy()
    val_df = df[df[SNAPSHOT_COL].isin(exp["val"])].copy()
    test_df = df[df[SNAPSHOT_COL].isin(exp["test"])].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Empty split detected for experiment: {exp['name']}")

    X_train = train_df[features]
    y_train = train_df[target].astype(int)

    X_val = val_df[features]
    y_val = val_df[target].astype(int)

    X_test = test_df[features]
    y_test = test_df[target].astype(int)

    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    print("\n" + "=" * 90)
    print(exp["name"])
    print(exp["description"])
    print("=" * 90)
    print("Train:", exp["train"], X_train.shape, "positive:", y_train.mean())
    print("Val  :", exp["val"], X_val.shape, "positive:", y_val.mean())
    print("Test :", exp["test"], X_test.shape, "positive:", y_test.mean())
    print("scale_pos_weight:", scale_pos_weight)

    model = make_lightgbm(scale_pos_weight=scale_pos_weight, seed=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
    )

    val_score = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = best_threshold_by_f1(y_val.values, val_score)

    test_score = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.values, test_score, threshold)

    metrics.update(
        {
            "experiment": exp["name"],
            "description": exp["description"],
            "model": "LightGBM",
            "train_snapshots": "+".join(exp["train"]),
            "val_snapshots": "+".join(exp["val"]),
            "test_snapshots": "+".join(exp["test"]),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test_positives": int(y_test.sum()),
            "scale_pos_weight": scale_pos_weight,
            "val_f1": float(val_f1),
        }
    )

    print(
        f"AUC-PR={metrics['auc_pr']:.4f} | "
        f"AUC-ROC={metrics['auc_roc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | "
        f"threshold={metrics['threshold']:.4f}"
    )

    return metrics


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    out_results = out_dir / "temporal_horizon_results_lightgbm.csv"
    out_degradation = out_dir / "temporal_horizon_degradation_lightgbm.csv"
    out_report = out_dir / "temporal_horizon_report_lightgbm.json"

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = clean_features(df, FEATURES, args.target)

    print("Dataset:", df.shape)
    print("\nLabel distribution:")
    print(df.groupby(SNAPSHOT_COL)[args.target].agg(["count", "sum", "mean"]))

    rows = [
        train_eval_experiment(df, exp, FEATURES, args.target, args.seed)
        for exp in EXPERIMENTS
    ]

    res = pd.DataFrame(rows)
    res = res[
        [
            "experiment",
            "model",
            "train_snapshots",
            "val_snapshots",
            "test_snapshots",
            "n_train",
            "n_val",
            "n_test",
            "train_positive_rate",
            "val_positive_rate",
            "test_positive_rate",
            "test_positives",
            "auc_pr",
            "auc_roc",
            "f1",
            "precision",
            "recall",
            "brier",
            "tp",
            "fp",
            "tn",
            "fn",
            "threshold",
            "val_f1",
            "scale_pos_weight",
            "description",
        ]
    ]
    res.to_csv(out_results, index=False)

    baseline = res.iloc[0]
    degradation_rows = []
    for _, row in res.iterrows():
        degradation_rows.append(
            {
                "experiment": row["experiment"],
                "delta_auc_pr_vs_short": float(row["auc_pr"] - baseline["auc_pr"]),
                "delta_f1_vs_short": float(row["f1"] - baseline["f1"]),
                "relative_auc_pr_vs_short": float(
                    row["auc_pr"] / max(baseline["auc_pr"], 1e-12)
                ),
                "relative_f1_vs_short": float(
                    row["f1"] / max(baseline["f1"], 1e-12)
                ),
            }
        )

    degradation = pd.DataFrame(degradation_rows)
    degradation.to_csv(out_degradation, index=False)

    report = {
        "dataset": str(data_path),
        "model": "LightGBM",
        "target": args.target,
        "feature_set": "15_learning_features_without_heuristics",
        "features": FEATURES,
        "experiments": EXPERIMENTS,
        "results_csv": str(out_results),
        "degradation_csv": str(out_degradation),
        "results": res.to_dict(orient="records"),
        "degradation": degradation.to_dict(orient="records"),
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 90)
    print("TEMPORAL HORIZON RESULTS — LIGHTGBM")
    print("=" * 90)
    print(res)

    print("\n" + "=" * 90)
    print("TEMPORAL DEGRADATION SUMMARY — LIGHTGBM")
    print("=" * 90)
    print(degradation)

    print("\nSaved:")
    print(out_results)
    print(out_degradation)
    print(out_report)


if __name__ == "__main__":
    main()

