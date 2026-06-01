#!/usr/bin/env python3
"""
LightGBM SHAP analysis for the leakage-aware Bitcoin fraud benchmark.

This script trains the final LightGBM model using the strict temporal protocol:
    Train: D1 + D2 + D3
    Validation: D4
    Test: D5 + D6

It computes SHAP values on a representative sample of the held-out test set and
exports feature-importance tables and SHAP figures.

Example:
    python experiments/shap_analysis.py \
        --data-path data/processed/dataset_learning.csv \
        --out-dir results/shap_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
TARGET = "label_final"

TRAIN_SNAPS = ["D1", "D2", "D3"]
VAL_SNAPS = ["D4"]
TEST_SNAPS = ["D5", "D6"]

FEATURES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightGBM SHAP analysis under strict temporal evaluation."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the processed learning dataset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/shap_analysis",
        help="Directory where SHAP outputs will be saved.",
    )
    parser.add_argument(
        "--max-shap-samples",
        type=int,
        default=25000,
        help="Maximum number of held-out test transactions used for SHAP analysis.",
    )
    parser.add_argument(
        "--top-n-dependence",
        type=int,
        default=8,
        help="Number of top features for which dependence plots are generated.",
    )
    parser.add_argument(
        "--no-dependence-plots",
        action="store_true",
        help="Disable SHAP dependence plot generation.",
    )
    return parser.parse_args()


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))
    if best_idx >= len(thresholds):
        return 0.5, float(f1[best_idx])
    return float(thresholds[best_idx]), float(f1[best_idx])


def load_dataset(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = FEATURES + [TARGET, "snapshot_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TARGET] = df[TARGET].astype(int)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def split_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "train": df[df["snapshot_id"].isin(TRAIN_SNAPS)].copy(),
        "val": df[df["snapshot_id"].isin(VAL_SNAPS)].copy(),
        "test": df[df["snapshot_id"].isin(TEST_SNAPS)].copy(),
    }


def make_model(scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.1,
        reg_lambda=5.0,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def train_lightgbm(splits: Dict[str, pd.DataFrame]) -> Tuple[LGBMClassifier, Dict[str, float]]:
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET].astype(int)
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    model = make_model(scale_pos_weight)

    print("Training LightGBM final model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision")

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    threshold, val_f1 = best_threshold_by_f1(y_val.values, val_score)
    y_pred = (test_score >= threshold).astype(int)

    metrics = {
        "auc_pr": float(average_precision_score(y_test, test_score)),
        "auc_roc": float(roc_auc_score(y_test, test_score)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "val_f1": float(val_f1),
        "scale_pos_weight": scale_pos_weight,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
    }
    print(
        f"AUC-PR={metrics['auc_pr']:.4f} | AUC-ROC={metrics['auc_roc']:.4f} | "
        f"F1={metrics['f1']:.4f} | P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | threshold={metrics['threshold']:.6f}"
    )
    return model, metrics


def sample_test_set(test_df: pd.DataFrame, max_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    if len(X_test) <= max_samples:
        return X_test.copy(), y_test.copy()

    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(X_test), size=max_samples, replace=False)
    return X_test.iloc[sample_idx].copy(), y_test.iloc[sample_idx].copy()


def compute_shap_values(model: LGBMClassifier, X_shap: pd.DataFrame) -> np.ndarray:
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.asarray(shap_values)


def save_shap_outputs(
    shap_values: np.ndarray,
    X_shap: pd.DataFrame,
    out_dir: Path,
    top_n_dependence: int,
    generate_dependence: bool,
) -> Dict[str, object]:
    figures_dir = out_dir / "figures"
    dependence_dir = figures_dir / "dependence_plots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    dependence_dir.mkdir(parents=True, exist_ok=True)

    importance = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": FEATURES, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance_csv = out_dir / "lightgbm_shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    direction_rows = []
    for i, feat in enumerate(FEATURES):
        x = X_shap[feat].values
        s = shap_values[:, i]
        if np.std(x) == 0 or np.std(s) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, s)[0, 1])
        direction_rows.append(
            {
                "feature": feat,
                "mean_abs_shap": float(importance[i]),
                "correlation_with_shap": corr,
                "dominant_effect": "positive" if corr > 0 else "negative",
            }
        )

    direction_df = (
        pd.DataFrame(direction_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    direction_csv = out_dir / "lightgbm_shap_direction_analysis.csv"
    direction_df.to_csv(direction_csv, index=False)

    summary_png = figures_dir / "lightgbm_shap_summary_plot.png"
    summary_pdf = figures_dir / "lightgbm_shap_summary_plot.pdf"
    bar_png = figures_dir / "lightgbm_shap_bar_plot.png"
    bar_pdf = figures_dir / "lightgbm_shap_bar_plot.pdf"

    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.savefig(summary_pdf, bbox_inches="tight")
    plt.close()

    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.savefig(bar_pdf, bbox_inches="tight")
    plt.close()

    dependence_outputs: List[Dict[str, str]] = []
    if generate_dependence:
        print("Generating SHAP dependence plots...")
        top_features = importance_df["feature"].head(top_n_dependence).tolist()
        for feat in top_features:
            try:
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(feat, shap_values, X_shap, show=False)
                plt.tight_layout()
                out_png = dependence_dir / f"dependence_{feat}.png"
                out_pdf = dependence_dir / f"dependence_{feat}.pdf"
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.savefig(out_pdf, bbox_inches="tight")
                plt.close()
                dependence_outputs.append({"feature": feat, "png": str(out_png), "pdf": str(out_pdf)})
            except Exception as exc:  # pragma: no cover - plot failures are non-critical
                print(f"Dependence plot failed for {feat}: {exc}")
                plt.close()

    return {
        "importance_csv": str(importance_csv),
        "direction_csv": str(direction_csv),
        "summary_plot_png": str(summary_png),
        "summary_plot_pdf": str(summary_pdf),
        "bar_plot_png": str(bar_png),
        "bar_plot_pdf": str(bar_pdf),
        "dependence_dir": str(dependence_dir),
        "dependence_outputs": dependence_outputs,
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_path)
    print("Dataset:", df.shape)
    print(df.groupby("snapshot_id")[TARGET].agg(["count", "sum", "mean"]))

    splits = split_dataset(df)
    print("\nTemporal split:")
    for split_name, split_df in splits.items():
        print(split_name, split_df.shape, "positive:", split_df[TARGET].mean())

    model, metrics = train_lightgbm(splits)

    X_shap, y_shap = sample_test_set(splits["test"], args.max_shap_samples)
    print("SHAP sample:", X_shap.shape, "positive:", float(y_shap.mean()))

    shap_values = compute_shap_values(model, X_shap)
    outputs = save_shap_outputs(
        shap_values=shap_values,
        X_shap=X_shap,
        out_dir=out_dir,
        top_n_dependence=args.top_n_dependence,
        generate_dependence=not args.no_dependence_plots,
    )

    report = {
        "dataset": args.data_path,
        "model": {
            "type": "LightGBM",
            "configuration": "final_15_features_with_memory",
            "n_estimators": 700,
            "learning_rate": 0.025,
            "num_leaves": 64,
            "scale_pos_weight": metrics["scale_pos_weight"],
        },
        "features": FEATURES,
        "strict_temporal_protocol": {
            "train": TRAIN_SNAPS,
            "validation": VAL_SNAPS,
            "test": TEST_SNAPS,
        },
        "performance": metrics,
        "outputs": outputs,
    }

    report_path = out_dir / "lightgbm_shap_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nLIGHTGBM SHAP ANALYSIS COMPLETED")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()

