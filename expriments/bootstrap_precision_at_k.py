#!/usr/bin/env python3
"""
LightGBM final evaluation with bootstrap confidence intervals, Precision@K,
top-percent fraud capture, and operational capture figure.

This script reproduces the final LightGBM evaluation used in the paper.

Expected input dataset:
    A CSV file containing:
      - tx_hash
      - snapshot_id
      - block_height
      - label_final
      - the 13 learning features listed in FEATURES

Example:
    python experiments/bootstrap_precision_at_k.py \
        --data-path data/processed/dataset_learning.csv \
        --out-dir results/bootstrap_precision_at_k
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
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

TRAIN_SNAPS = ["D1", "D2", "D3"]
VAL_SNAPS = ["D4"]
TEST_SNAPS = ["D5", "D6"]

BOOTSTRAP_N = 500

K_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
TOP_PERCENT_VALUES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

FEATURES = [
    "input_count",
    "output_count",
    "input_addr_count",
    "coinbase_flag",
    "has_witness",
    "script_type_encoded",
    "input_addr_concentration",
    "io_count_ratio",
    "avg_input_value",
    "log_output_value",
    "fee_ratio",
    "prev_addr_seen_ratio",
    "prev_addr_seen_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final LightGBM evaluation with bootstrap CI and Precision@K."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the processed learning dataset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/bootstrap_precision_at_k",
        help="Directory where result files will be written.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=BOOTSTRAP_N,
        help="Number of bootstrap resampling iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )
    return parser.parse_args()


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))

    if best_idx >= len(thresholds):
        return 0.5, float(f1[best_idx])

    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def precision_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: list[int],
) -> pd.DataFrame:
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true).astype(int)[order]
    total_pos = max(int(np.sum(y_true)), 1)

    rows = []
    for k in k_values:
        if k > len(y_true):
            continue

        topk = y_sorted[:k]
        tp_k = int(topk.sum())

        rows.append(
            {
                "k": int(k),
                "precision_at_k": float(tp_k / k),
                "recall_at_k": float(tp_k / total_pos),
                "tp_at_k": tp_k,
                "alerts": int(k),
                "total_positives": int(total_pos),
            }
        )

    return pd.DataFrame(rows)


def top_percent_capture(
    y_true: np.ndarray,
    y_score: np.ndarray,
    percent_values: list[float],
) -> pd.DataFrame:
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true).astype(int)[order]
    total_pos = max(int(np.sum(y_true)), 1)
    n = len(y_true)

    rows = []
    for pct in percent_values:
        k = max(1, int(n * pct))
        topk = y_sorted[:k]
        tp_k = int(topk.sum())

        rows.append(
            {
                "top_percent": float(pct),
                "alerts": int(k),
                "precision": float(tp_k / k),
                "recall_capture_rate": float(tp_k / total_pos),
                "tp_captured": tp_k,
                "total_positives": int(total_pos),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n = len(y_true)

    collected = {
        "auc_pr": [],
        "auc_roc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "brier": [],
    }

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        ys = y_score[idx]

        if yt.sum() == 0 or yt.sum() == len(yt):
            continue

        yp = (ys >= threshold).astype(int)

        collected["auc_pr"].append(average_precision_score(yt, ys))
        collected["auc_roc"].append(roc_auc_score(yt, ys))
        collected["f1"].append(f1_score(yt, yp, zero_division=0))
        collected["precision"].append(precision_score(yt, yp, zero_division=0))
        collected["recall"].append(recall_score(yt, yp, zero_division=0))
        collected["brier"].append(brier_score_loss(yt, ys))

    rows = []
    for metric, values in collected.items():
        values = np.asarray(values)

        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower_2_5": float(np.percentile(values, 2.5)),
                "ci_upper_97_5": float(np.percentile(values, 97.5)),
                "n_boot_valid": int(len(values)),
            }
        )

    return pd.DataFrame(rows)


def make_top_percent_figure(
    top_pct_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    x = top_pct_df["top_percent"].values * 100
    y = top_pct_df["recall_capture_rate"].values * 100
    p = top_pct_df["precision"].values * 100

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=2, label="Fraud capture rate")
    plt.plot(x, p, marker="s", linewidth=2, linestyle="--", label="Precision")

    plt.xlabel("Top-ranked transactions inspected (%)")
    plt.ylabel("Percentage (%)")
    plt.title("Operational fraud capture among highest-risk alerts")
    plt.grid(True, alpha=0.3)
    plt.legend()

    for _, row in top_pct_df.iterrows():
        pct = row["top_percent"] * 100
        if np.isclose(pct, 0.5) or np.isclose(pct, 1.0):
            plt.annotate(
                f"{row['recall_capture_rate'] * 100:.1f}% captured",
                xy=(pct, row["recall_capture_rate"] * 100),
                xytext=(pct + 0.2, row["recall_capture_rate"] * 100 - 8),
                arrowprops={"arrowstyle": "->", "lw": 0.8},
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def load_dataset(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = FEATURES + [TARGET, "snapshot_id", "block_height", "tx_hash"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[TARGET] = df[TARGET].astype(int)

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def make_model(scale_pos_weight: float, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=5.0,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_metrics = out_dir / "lightgbm_best_model_metrics.csv"
    out_bootstrap = out_dir / "lightgbm_bootstrap_confidence_intervals.csv"
    out_pk = out_dir / "lightgbm_precision_recall_at_k.csv"
    out_top_percent = out_dir / "lightgbm_top_percent_capture.csv"
    out_alerts = out_dir / "lightgbm_ranked_test_alerts.csv"
    out_report = out_dir / "lightgbm_bootstrap_precision_report.json"

    fig_png = fig_dir / "lightgbm_top_percent_capture.png"
    fig_pdf = fig_dir / "lightgbm_top_percent_capture.pdf"

    df = load_dataset(args.data_path)

    print("Dataset:", df.shape)
    print("\nLabel distribution:")
    print(df.groupby("snapshot_id")[TARGET].agg(["count", "sum", "mean"]))

    train_df = df[df["snapshot_id"].isin(TRAIN_SNAPS)].copy()
    val_df = df[df["snapshot_id"].isin(VAL_SNAPS)].copy()
    test_df = df[df["snapshot_id"].isin(TEST_SNAPS)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One temporal split is empty. Check snapshot_id values.")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_val = val_df[FEATURES]
    y_val = val_df[TARGET].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    print("\nTemporal split:")
    print("Train:", X_train.shape, "positive:", y_train.mean())
    print("Val  :", X_val.shape, "positive:", y_val.mean())
    print("Test :", X_test.shape, "positive:", y_test.mean())
    print("scale_pos_weight:", scale_pos_weight)

    model = make_model(scale_pos_weight=scale_pos_weight, seed=args.seed)

    print("\nTraining LightGBM final configuration...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
    )

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    threshold, val_f1 = best_threshold_by_f1(y_val.values, val_score)
    metrics = evaluate_at_threshold(y_test.values, test_score, threshold)

    metrics["val_f1"] = float(val_f1)
    metrics["n_train"] = int(len(y_train))
    metrics["n_val"] = int(len(y_val))
    metrics["n_test"] = int(len(y_test))
    metrics["train_positive_rate"] = float(y_train.mean())
    metrics["val_positive_rate"] = float(y_val.mean())
    metrics["test_positive_rate"] = float(y_test.mean())
    metrics["scale_pos_weight"] = float(scale_pos_weight)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_metrics, index=False)

    print("\n" + "=" * 80)
    print("LIGHTGBM FINAL MODEL TEST METRICS")
    print("=" * 80)
    print(metrics_df)

    ranked = test_df[["tx_hash", "snapshot_id", "block_height", TARGET]].copy()
    ranked["score"] = test_score
    ranked["rank"] = ranked["score"].rank(method="first", ascending=False).astype(int)
    ranked = ranked.sort_values("score", ascending=False)
    ranked.to_csv(out_alerts, index=False)

    pk_df = precision_recall_at_k(y_test.values, test_score, K_VALUES)
    pk_df.to_csv(out_pk, index=False)

    top_pct_df = top_percent_capture(y_test.values, test_score, TOP_PERCENT_VALUES)
    top_pct_df.to_csv(out_top_percent, index=False)

    print("\n" + "=" * 80)
    print("PRECISION / RECALL @ K")
    print("=" * 80)
    print(pk_df)

    print("\n" + "=" * 80)
    print("TOP PERCENT CAPTURE")
    print("=" * 80)
    print(top_pct_df)

    make_top_percent_figure(top_pct_df, fig_png, fig_pdf)

    print("\nFigure saved:")
    print(fig_png)
    print(fig_pdf)

    print("\nComputing bootstrap confidence intervals...")
    boot_df = bootstrap_ci(
        y_test.values,
        test_score,
        threshold,
        n_boot=args.bootstrap_n,
        seed=args.seed,
    )
    boot_df.to_csv(out_bootstrap, index=False)

    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    print(boot_df)

    report = {
        "dataset": args.data_path,
        "model": {
            "type": "LightGBM",
            "configuration": "final_15_features_with_memory",
            "n_estimators": 700,
            "learning_rate": 0.025,
            "num_leaves": 64,
            "scale_pos_weight": float(scale_pos_weight),
        },
        "features": FEATURES,
        "temporal_protocol": {
            "train": TRAIN_SNAPS,
            "validation": VAL_SNAPS,
            "test": TEST_SNAPS,
        },
        "metrics": metrics,
        "outputs": {
            "metrics": str(out_metrics),
            "bootstrap_ci": str(out_bootstrap),
            "precision_recall_at_k": str(out_pk),
            "top_percent_capture": str(out_top_percent),
            "ranked_alerts": str(out_alerts),
            "figure_png": str(fig_png),
            "figure_pdf": str(fig_pdf),
        },
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("LIGHTGBM BOOTSTRAP + PRECISION@K COMPLETED")
    print("=" * 80)

    print("\nSaved:")
    print(out_metrics)
    print(out_bootstrap)
    print(out_pk)
    print(out_top_percent)
    print(out_alerts)
    print(out_report)


if __name__ == "__main__":
    main()

