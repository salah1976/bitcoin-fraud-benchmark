#!/usr/bin/env python3
"""
Structural exposure analysis for the leakage-aware Bitcoin fraud benchmark.

This script trains the final LightGBM model under the strict temporal protocol:
    Train: D1 + D2 + D3
    Validation: D4
    Test: D5 + D6

It decomposes test performance by structural exposure group:
    - fraud_exposed: connected to historically fraudulent training addresses
    - history_exposed: connected to previously observed training addresses, but not fraudulent ones
    - isolated: not connected to any previously observed training address

Example:
    python experiments/structural_exposure.py \
        --data-path data/processed/dataset_learning.csv \
        --db-path data/raw/all_snapshots_extended.db \
        --out-dir results/structural_exposure
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

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

GROUP_ORDER = [
    "overall",
    "fraud_exposed",
    "history_exposed",
    "isolated",
    "exposed_any",
    "no_address_info",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGBM structural exposure analysis.")
    parser.add_argument("--data-path", required=True, help="Path to the processed learning dataset CSV.")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database containing tx_inputs and tx_outputs.")
    parser.add_argument("--out-dir", default="results/structural_exposure", help="Output directory.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def norm_addr(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def best_threshold_by_f1(y_true, y_score) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))
    if best_idx >= len(thresholds):
        return 0.5, float(f1[best_idx])
    return float(thresholds[best_idx]), float(f1[best_idx])


def safe_metrics(y_true, y_score, threshold) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    out: Dict[str, object] = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
    }

    if len(y_true) == 0:
        out.update({"auc_pr": None, "auc_roc": None, "f1": None, "precision": None, "recall": None, "brier": None, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
        return out

    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        out["auc_pr"] = float(average_precision_score(y_true, y_score))
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc_pr"] = None
        out["auc_roc"] = None

    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["brier"] = float(brier_score_loss(y_true, y_score))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tp"] = int(tp)
    out["fp"] = int(fp)
    out["tn"] = int(tn)
    out["fn"] = int(fn)
    return out


def make_bar_figure(results_df: pd.DataFrame, metric: str, out_png: Path, out_pdf: Path, title: str, ylabel: str) -> None:
    plot_df = results_df.groupby("region", as_index=False).first()
    plot_df = plot_df[plot_df["region"].isin(["fraud_exposed", "history_exposed", "isolated"])].copy()
    plot_df["region_label"] = plot_df["region"].map({
        "fraud_exposed": "Fraud-exposed",
        "history_exposed": "History-exposed",
        "isolated": "Isolated",
    })

    plt.figure(figsize=(7, 4.5))
    plt.bar(plot_df["region_label"], plot_df[metric].fillna(0))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max(0.05, min(1.0, plot_df[metric].fillna(0).max() * 1.2)))
    for i, v in enumerate(plot_df[metric].fillna(0).values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def make_combined_figure(results_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    region_order = ["fraud_exposed", "history_exposed", "isolated"]
    plot_df = results_df.groupby("region", as_index=False).first()
    plot_df = plot_df[plot_df["region"].isin(region_order)].set_index("region").loc[region_order].reset_index()

    labels = ["Fraud-exposed", "History-exposed", "Isolated"]
    x = np.arange(len(labels))
    width = 0.35
    f1_vals = plot_df["f1"].fillna(0).values
    aucpr_vals = plot_df["auc_pr"].fillna(0).values

    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, f1_vals, width, label="F1-score")
    plt.bar(x + width / 2, aucpr_vals, width, label="AUC-PR")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Performance by structural exposure group")
    plt.ylim(0, 1.0)
    plt.legend()
    for i, v in enumerate(f1_vals):
        plt.text(i - width / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(aucpr_vals):
        plt.text(i + width / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def load_learning_dataset(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Learning dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    required = FEATURES + [TARGET, "snapshot_id", "tx_hash", "block_height"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in learning dataset: {missing}")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[TARGET] = df[TARGET].astype(int)
    df["tx_hash"] = df["tx_hash"].astype(str)
    return df


def train_final_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, seed: int):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET].astype(int)
    X_test = test_df[FEATURES]

    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    print("\nTemporal split:")
    print("Train:", X_train.shape, "positive:", y_train.mean())
    print("Val  :", X_val.shape, "positive:", y_val.mean())
    print("Test :", X_test.shape, "positive:", test_df[TARGET].mean())
    print("scale_pos_weight:", scale_pos_weight)

    model = LGBMClassifier(
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
    print("\nTraining LightGBM final configuration...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision")

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    threshold, val_f1 = best_threshold_by_f1(y_val.values, val_score)
    print(f"Threshold: {threshold:.6f}")
    print(f"Validation F1: {val_f1:.6f}")
    return test_score, threshold, val_f1, scale_pos_weight


def build_structural_exposure(db_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    print("\nBuilding structural exposure sets from SQLite...")

    train_fraud_tx = set(train_df.loc[train_df[TARGET] == 1, "tx_hash"].astype(str))
    test_tx = set(test_df["tx_hash"].astype(str))
    conn = sqlite3.connect(db_path)
    train_all_addr = set()
    train_fraud_addr = set()

    for table, addr_col in [("tx_inputs", "input_address"), ("tx_outputs", "output_address")]:
        query = f"""
            SELECT tx_hash, {addr_col}
            FROM {table}
            WHERE snapshot_id IN ('D1','D2','D3')
              AND {addr_col} IS NOT NULL
        """
        cur = conn.execute(query)
        while True:
            rows = cur.fetchmany(100000)
            if not rows:
                break
            for tx_hash, addr in rows:
                a = norm_addr(addr)
                if not a:
                    continue
                train_all_addr.add(a)
                if str(tx_hash) in train_fraud_tx:
                    train_fraud_addr.add(a)

    print("Train all addresses   :", len(train_all_addr))
    print("Train fraud addresses :", len(train_fraud_addr))
    print("\nComputing test exposure...")

    test_addr_map = {
        tx: {"tx_hash": tx, "n_addrs": 0, "n_seen_train": 0, "n_seen_train_fraud": 0, "has_seen_train": 0, "has_seen_train_fraud": 0}
        for tx in test_tx
    }

    for table, addr_col in [("tx_inputs", "input_address"), ("tx_outputs", "output_address")]:
        query = f"""
            SELECT tx_hash, {addr_col}
            FROM {table}
            WHERE snapshot_id IN ('D5','D6')
              AND {addr_col} IS NOT NULL
        """
        cur = conn.execute(query)
        while True:
            rows = cur.fetchmany(100000)
            if not rows:
                break
            for tx_hash, addr in rows:
                tx_hash = str(tx_hash)
                if tx_hash not in test_addr_map:
                    continue
                a = norm_addr(addr)
                if not a:
                    continue
                test_addr_map[tx_hash]["n_addrs"] += 1
                if a in train_all_addr:
                    test_addr_map[tx_hash]["n_seen_train"] += 1
                    test_addr_map[tx_hash]["has_seen_train"] = 1
                if a in train_fraud_addr:
                    test_addr_map[tx_hash]["n_seen_train_fraud"] += 1
                    test_addr_map[tx_hash]["has_seen_train_fraud"] = 1
    conn.close()

    exposure_df = pd.DataFrame(list(test_addr_map.values()))
    exposure_df["seen_train_ratio"] = exposure_df["n_seen_train"] / exposure_df["n_addrs"].clip(lower=1)
    exposure_df["seen_train_fraud_ratio"] = exposure_df["n_seen_train_fraud"] / exposure_df["n_addrs"].clip(lower=1)

    def exposure_group(row):
        if row["has_seen_train_fraud"] == 1:
            return "fraud_exposed"
        if row["has_seen_train"] == 1:
            return "history_exposed"
        return "isolated"

    exposure_df["exposure_group"] = exposure_df.apply(exposure_group, axis=1)
    address_counts = {"all_train_addresses": int(len(train_all_addr)), "train_fraud_addresses": int(len(train_fraud_addr))}
    return exposure_df, address_counts


def evaluate_by_exposure(test_pred: pd.DataFrame, threshold: float):
    rows = []
    m = safe_metrics(test_pred[TARGET].values, test_pred["score"].values, threshold)
    m["region"] = "overall"
    rows.append(m)

    for group, g in test_pred.groupby("exposure_group"):
        m = safe_metrics(g[TARGET].values, g["score"].values, threshold)
        m["region"] = group
        rows.append(m)

    test_pred["binary_exposure"] = np.where(
        test_pred["exposure_group"].isin(["fraud_exposed", "history_exposed"]),
        "exposed_any",
        "isolated",
    )
    for group, g in test_pred.groupby("binary_exposure"):
        m = safe_metrics(g[TARGET].values, g["score"].values, threshold)
        m["region"] = group
        rows.append(m)

    results_df = pd.DataFrame(rows)
    results_df = results_df[["region", "n", "positives", "positive_rate", "auc_pr", "auc_roc", "f1", "precision", "recall", "brier", "tp", "fp", "tn", "fn", "threshold"]]
    region_order = {r: i for i, r in enumerate(GROUP_ORDER)}
    results_df["order"] = results_df["region"].map(region_order).fillna(99)
    results_df = results_df.sort_values("order").drop(columns=["order"])
    return results_df, test_pred


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_exposure_tx = out_dir / "lightgbm_test_transactions_exposure.csv"
    out_results = out_dir / "lightgbm_structural_exposure_results.csv"
    out_summary = out_dir / "lightgbm_structural_exposure_summary.csv"
    out_report = out_dir / "lightgbm_structural_exposure_report.json"
    fig_f1_png = fig_dir / "lightgbm_structural_exposure_f1.png"
    fig_f1_pdf = fig_dir / "lightgbm_structural_exposure_f1.pdf"
    fig_aucpr_png = fig_dir / "lightgbm_structural_exposure_aucpr.png"
    fig_aucpr_pdf = fig_dir / "lightgbm_structural_exposure_aucpr.pdf"
    fig_combined_png = fig_dir / "lightgbm_structural_exposure_combined.png"
    fig_combined_pdf = fig_dir / "lightgbm_structural_exposure_combined.pdf"

    df = load_learning_dataset(args.data_path)
    print("Learning dataset:", df.shape)
    print(df.groupby("snapshot_id")[TARGET].agg(["count", "sum", "mean"]))

    train_df = df[df["snapshot_id"].isin(TRAIN_SNAPS)].copy()
    val_df = df[df["snapshot_id"].isin(VAL_SNAPS)].copy()
    test_df = df[df["snapshot_id"].isin(TEST_SNAPS)].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One temporal split is empty. Check snapshot_id values.")

    test_score, threshold, val_f1, scale_pos_weight = train_final_lightgbm(train_df, val_df, test_df, args.seed)
    exposure_df, address_counts = build_structural_exposure(args.db_path, train_df, test_df)

    test_pred = test_df[["tx_hash", "snapshot_id", "block_height", TARGET]].copy()
    test_pred["score"] = test_score
    test_pred["pred"] = (test_score >= threshold).astype(int)
    test_pred = test_pred.merge(exposure_df, on="tx_hash", how="left")
    test_pred["exposure_group"] = test_pred["exposure_group"].fillna("no_address_info")

    exposure_distribution = test_pred.groupby("exposure_group")[TARGET].agg(["count", "sum", "mean"]).reset_index()
    print("\nExposure group distribution:")
    print(exposure_distribution)

    results_df, test_pred = evaluate_by_exposure(test_pred, threshold)
    print("\n" + "=" * 80)
    print("LIGHTGBM STRUCTURAL EXPOSURE RESULTS")
    print("=" * 80)
    print(results_df)

    summary_df = test_pred.groupby(["snapshot_id", "exposure_group"])[TARGET].agg(["count", "sum", "mean"]).reset_index()
    test_pred.to_csv(out_exposure_tx, index=False)
    results_df.to_csv(out_results, index=False)
    summary_df.to_csv(out_summary, index=False)

    make_bar_figure(results_df, "f1", fig_f1_png, fig_f1_pdf, "F1-score by structural exposure group", "F1-score")
    make_bar_figure(results_df, "auc_pr", fig_aucpr_png, fig_aucpr_pdf, "AUC-PR by structural exposure group", "AUC-PR")
    make_combined_figure(results_df, fig_combined_png, fig_combined_pdf)

    report = {
        "dataset": args.data_path,
        "sqlite_db": args.db_path,
        "model": {
            "type": "LightGBM",
            "configuration": "final_15_features_with_memory",
            "threshold": float(threshold),
            "val_f1": float(val_f1),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "features": FEATURES,
        "temporal_protocol": {"train": TRAIN_SNAPS, "validation": VAL_SNAPS, "test": TEST_SNAPS},
        "train_address_counts": address_counts,
        "exposure_distribution": exposure_distribution.to_dict(orient="records"),
        "results": results_df.to_dict(orient="records"),
        "outputs": {
            "transaction_exposure": str(out_exposure_tx),
            "results": str(out_results),
            "summary": str(out_summary),
            "figure_f1_png": str(fig_f1_png),
            "figure_f1_pdf": str(fig_f1_pdf),
            "figure_aucpr_png": str(fig_aucpr_png),
            "figure_aucpr_pdf": str(fig_aucpr_pdf),
            "figure_combined_png": str(fig_combined_png),
            "figure_combined_pdf": str(fig_combined_pdf),
        },
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nSaved:")
    print(out_exposure_tx)
    print(out_results)
    print(out_summary)
    print(out_report)
    print(fig_combined_pdf)


if __name__ == "__main__":
    main()

