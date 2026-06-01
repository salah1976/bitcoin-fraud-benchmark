#!/usr/bin/env python3
"""
Three-model temporal evaluation for the Bitcoin fraud benchmark.

This script reproduces the homogeneous evaluation used in the paper:
- Train: D1 + D2 + D3
- Validation: D4, used only for threshold selection
- Test: D5 + D6 using all blocks
- Optional structural exposure analysis if a SQLite database is provided

The script is intentionally independent from Google Colab. Paths are provided
through command-line arguments.

Example
-------
python experiments/train_3ml_models.py \
    --data-path data/processed/dataset_learning.csv \
    --db-path data/raw/all_snapshots_extended.db \
    --out-dir results/three_ml_evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TARGET = "label_final"
SNAPSHOT_COL = "snapshot_id"
BLOCK_COL = "block_height"
TX_COL = "tx_hash"

TRAIN_SNAPS = ["D1", "D2", "D3"]
VAL_SNAPS = ["D4"]
TEST_SNAPS = ["D5", "D6"]

K_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

BASE_STRUCTURAL = [
    "input_count",
    "output_count",
    "input_addr_count",
    "coinbase_flag",
    "has_witness",
    "script_type_encoded",
    "input_addr_concentration",
    "io_count_ratio",
    "tx_weight",
]

MONETARY = [
    "avg_input_value",
    "total_input_scaled",
    "log_output_value",
    "fee_ratio",
]

TEMPORAL_MEMORY = [
    "prev_addr_seen_ratio",
    "prev_addr_seen_count",
]

FEE_DYNAMICS_EXCLUDED = [
    "fee_per_byte",
    "fee_log",
    "fee_per_input",
    "fee_per_output",
    "fee_urgency_ratio",
]

FEATURE_GROUPS = {
    "base_structural": BASE_STRUCTURAL,
    "monetary": MONETARY,
    "temporal_memory": TEMPORAL_MEMORY,
    "excluded_fee_dynamics": FEE_DYNAMICS_EXCLUDED,
}

FEATURES = BASE_STRUCTURAL + MONETARY + TEMPORAL_MEMORY
FEATURES_NO_MEMORY = BASE_STRUCTURAL + MONETARY

EXPERIMENT_FEATURE_SETS = {
    "15_features_with_memory": FEATURES,
    "13_features_without_memory": FEATURES_NO_MEMORY,
}


# ============================================================
# METRICS
# ============================================================

def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Select the threshold maximizing F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))

    if best_idx >= len(thresholds):
        return 0.5, float(f1[best_idx])

    return float(thresholds[best_idx]), float(f1[best_idx])


def safe_eval_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    """Compute binary and ranking metrics safely for a region/split."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
    }

    if len(y_true) == 0:
        out.update({
            "auc_pr": None,
            "auc_roc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "brier": None,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        })
        return out

    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        out["auc_pr"] = float(average_precision_score(y_true, y_score))
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
        out["brier"] = float(brier_score_loss(y_true, y_score))
    else:
        out["auc_pr"] = None
        out["auc_roc"] = None
        out["brier"] = float(brier_score_loss(y_true, y_score))

    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tp"] = int(tp)
    out["fp"] = int(fp)
    out["tn"] = int(tn)
    out["fn"] = int(fn)

    return out


def add_region_result(
    rows: List[Dict],
    model_name: str,
    feature_set_name: str,
    region: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    val_f1: float,
) -> None:
    metrics = safe_eval_scores(y_true, y_score, threshold)
    metrics.update({
        "model": model_name,
        "feature_set": feature_set_name,
        "region": region,
        "val_f1_for_threshold": float(val_f1),
    })
    rows.append(metrics)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_values: Iterable[int]) -> List[Dict]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    total_pos = max(int(y_true.sum()), 1)

    out = []
    for k in k_values:
        if k > len(y_true):
            continue
        topk = y_sorted[:k]
        tp_k = int(topk.sum())
        out.append({
            "k": int(k),
            "precision_at_k": float(tp_k / k),
            "recall_at_k": float(tp_k / total_pos),
            "tp_at_k": tp_k,
            "alerts": int(k),
            "total_positives": int(total_pos),
        })

    return out


def threshold_for_min_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_precision: float,
) -> Optional[Dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    best = None
    for i, thr in enumerate(thresholds):
        p = float(precision[i])
        r = float(recall[i])
        if p >= min_precision:
            f1 = 2 * p * r / (p + r + 1e-12)
            candidate = {
                "threshold": float(thr),
                "val_precision": p,
                "val_recall": r,
                "val_f1": float(f1),
            }
            if best is None or candidate["val_f1"] > best["val_f1"]:
                best = candidate

    return best


# ============================================================
# STRUCTURAL EXPOSURE
# ============================================================

def norm_addr(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def build_test_exposure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    db_path: Path,
    out_exposure_path: Path,
) -> pd.DataFrame:
    """Build structural exposure groups from the raw SQLite transaction tables."""
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    print("\nBuilding structural exposure sets from SQLite...")

    train_fraud_tx = set(train_df.loc[train_df[TARGET] == 1, TX_COL].astype(str))
    test_tx = set(test_df[TX_COL].astype(str))

    conn = sqlite3.connect(db_path)

    train_all_addr = set()
    train_fraud_addr = set()

    for table, addr_col in [
        ("tx_inputs", "input_address"),
        ("tx_outputs", "output_address"),
    ]:
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

    test_addr_map = {}
    for tx in test_tx:
        test_addr_map[tx] = {
            "tx_hash": tx,
            "n_addrs": 0,
            "n_seen_train": 0,
            "n_seen_train_fraud": 0,
            "has_seen_train": 0,
            "has_seen_train_fraud": 0,
        }

    print("Computing test exposure...")

    for table, addr_col in [
        ("tx_inputs", "input_address"),
        ("tx_outputs", "output_address"),
    ]:
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

    exposure_df["seen_train_ratio"] = (
        exposure_df["n_seen_train"] / exposure_df["n_addrs"].clip(lower=1)
    )
    exposure_df["seen_train_fraud_ratio"] = (
        exposure_df["n_seen_train_fraud"] / exposure_df["n_addrs"].clip(lower=1)
    )

    def exposure_group(row):
        if row["has_seen_train_fraud"] == 1:
            return "fraud_exposed"
        if row["has_seen_train"] == 1:
            return "history_exposed"
        return "isolated"

    exposure_df["exposure_group"] = exposure_df.apply(exposure_group, axis=1)

    test_meta = test_df[[TX_COL, SNAPSHOT_COL, BLOCK_COL, TARGET]].copy()
    test_meta[TX_COL] = test_meta[TX_COL].astype(str)

    exposure_df = test_meta.merge(exposure_df, on=TX_COL, how="left")
    exposure_df["exposure_group"] = exposure_df["exposure_group"].fillna("no_address_info")

    exposure_df["binary_exposure"] = np.where(
        exposure_df["exposure_group"].isin(["fraud_exposed", "history_exposed"]),
        "exposed_any",
        "isolated",
    )

    exposure_df.to_csv(out_exposure_path, index=False)

    print("\nExposure group distribution:")
    print(exposure_df.groupby("exposure_group")[TARGET].agg(["count", "sum", "mean"]))

    return exposure_df


# ============================================================
# MODELS
# ============================================================

def make_models(scale_pos_weight: float) -> Dict[str, object]:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=700,
            learning_rate=0.025,
            num_leaves=64,
            max_depth=-1,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_samples=50,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=700,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=SEED,
            verbose=False,
        ),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    db_path = Path(args.db_path) if args.db_path else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_results = out_dir / "three_ml_results_by_region.csv"
    out_pk = out_dir / "three_ml_precision_at_k.csv"
    out_hp = out_dir / "three_ml_high_precision_thresholds.csv"
    out_exposure_tx = out_dir / "test_transactions_exposure.csv"
    out_report = out_dir / "three_ml_report.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print("Dataset:", df.shape)
    print("\nSnapshot label distribution:")
    print(df.groupby(SNAPSHOT_COL)[TARGET].agg(["count", "sum", "mean"]))

    required_cols = sorted(
        set(FEATURES + FEATURES_NO_MEMORY + [TARGET, SNAPSHOT_COL, TX_COL, BLOCK_COL])
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df[TARGET] = df[TARGET].astype(int)

    numeric_cols = sorted(set(FEATURES + FEATURES_NO_MEMORY))
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    train_df = df[df[SNAPSHOT_COL].isin(TRAIN_SNAPS)].copy()
    val_df = df[df[SNAPSHOT_COL].isin(VAL_SNAPS)].copy()
    test_full_df = df[df[SNAPSHOT_COL].isin(TEST_SNAPS)].copy()

    if train_df.empty or val_df.empty or test_full_df.empty:
        raise ValueError("One temporal split is empty. Check snapshot_id values.")

    print("\nTemporal split sizes:")
    print("Train     :", train_df.shape, "positive_rate:", round(float(train_df[TARGET].mean()), 6))
    print("Validation:", val_df.shape, "positive_rate:", round(float(val_df[TARGET].mean()), 6))
    print("Test full :", test_full_df.shape, "positive_rate:", round(float(test_full_df[TARGET].mean()), 6))

    exposure_full = None

    if db_path is not None and db_path.exists() and not args.skip_exposure:
        exposure_full = build_test_exposure(
            train_df=train_df,
            test_df=test_full_df,
            db_path=db_path,
            out_exposure_path=out_exposure_tx,
        )
    else:
        print("\nStructural exposure skipped.")
        print("Provide --db-path and do not set --skip-exposure to compute exposure groups.")

    y_train = train_df[TARGET].astype(int)
    y_val = val_df[TARGET].astype(int)
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    print("\nscale_pos_weight:", scale_pos_weight)
    print("15 features used:", len(FEATURES), FEATURES)
    print("13 no-memory features used:", len(FEATURES_NO_MEMORY), FEATURES_NO_MEMORY)

    all_results = {}
    rows = []
    pk_rows = []
    hp_rows = []

    for feature_set_name, feature_list in EXPERIMENT_FEATURE_SETS.items():
        print("\n" + "#" * 90)
        print("FEATURE SET:", feature_set_name, "| n_features:", len(feature_list))
        print("#" * 90)

        X_train = train_df[feature_list]
        X_val = val_df[feature_list]
        X_test_full = test_full_df[feature_list]
        y_test_full = test_full_df[TARGET].astype(int)

        for model_name, model in make_models(scale_pos_weight).items():
            print("\n" + "=" * 80)
            print("Training:", model_name, "|", feature_set_name)
            print("=" * 80)

            if model_name == "CatBoost":
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            elif model_name == "LightGBM":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision")
            else:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            val_score = model.predict_proba(X_val)[:, 1]
            threshold, val_f1 = best_threshold_by_f1(y_val.values, val_score)

            test_full_score = model.predict_proba(X_test_full)[:, 1]
            print(f"{model_name}: threshold={threshold:.6f} | validation F1={val_f1:.4f}")
            print("Direct check full AUC-PR:", average_precision_score(y_test_full.values, test_full_score))

            key = f"{feature_set_name}__{model_name}"
            all_results[key] = {
                "threshold": threshold,
                "val_f1": val_f1,
                "feature_list": feature_list,
            }

            add_region_result(
                rows, model_name, feature_set_name, "D5D6_full",
                y_test_full.values, test_full_score, threshold, val_f1
            )
            if exposure_full is not None:
                exp_full_scored = exposure_full.copy()
                exp_full_scored["score"] = test_full_score
                exp_full_scored["pred"] = (test_full_score >= threshold).astype(int)

                for group, g in exp_full_scored.groupby("exposure_group"):
                    add_region_result(
                        rows, model_name, feature_set_name, f"D5D6_full__{group}",
                        g[TARGET].values, g["score"].values, threshold, val_f1
                    )

                for group, g in exp_full_scored.groupby("binary_exposure"):
                    add_region_result(
                        rows, model_name, feature_set_name, f"D5D6_full__{group}",
                        g[TARGET].values, g["score"].values, threshold, val_f1
                    )

            for split_name, y_true, y_score in [
                ("D5D6_full", y_test_full.values, test_full_score),
            ]:
                for row in precision_at_k(y_true, y_score, K_VALUES):
                    row.update({
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "split": split_name,
                    })
                    pk_rows.append(row)

            for target_precision in [0.80, 0.90, 0.95]:
                hp = threshold_for_min_precision(y_val.values, val_score, target_precision)
                if hp is None:
                    continue
                for split_name, y_true, y_score in [
                    ("D5D6_full", y_test_full.values, test_full_score),
                ]:
                    m = safe_eval_scores(y_true, y_score, hp["threshold"])
                    hp_rows.append({
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "split": split_name,
                        "target_precision_on_validation": target_precision,
                        **hp,
                        **m,
                    })

    res_df = pd.DataFrame(rows)
    res_df = res_df[
        [
            "feature_set", "model", "region", "n", "positives", "positive_rate",
            "auc_pr", "auc_roc", "f1", "precision", "recall", "brier",
            "tp", "fp", "tn", "fn", "threshold", "val_f1_for_threshold",
        ]
    ].sort_values(["feature_set", "region", "auc_pr"], ascending=[True, True, False])

    res_df.to_csv(out_results, index=False)

    print("\n" + "=" * 90)
    print("FINAL RESULTS — 3 ML MODELS")
    print("=" * 90)
    print(res_df)

    pk_df = pd.DataFrame(pk_rows)
    pk_df.to_csv(out_pk, index=False)

    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(out_hp, index=False)

    best_full = (
        res_df[res_df["region"] == "D5D6_full"]
        .sort_values("auc_pr", ascending=False)
        .iloc[0]
    )

    report = {
        "dataset": str(data_path),
        "sqlite_db": str(db_path) if db_path is not None else None,
        "target": TARGET,
        "temporal_split": {
            "train": TRAIN_SNAPS,
            "validation": VAL_SNAPS,
            "test": TEST_SNAPS,
        },
        "models": ["XGBoost", "LightGBM", "CatBoost"],
        "feature_groups": FEATURE_GROUPS,
        "feature_sets": {
            "15_features_with_memory": FEATURES,
            "13_features_without_memory": FEATURES_NO_MEMORY,
        },
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test_full": int(len(test_full_df)),
        },
        "positive_rates": {
            "train": float(train_df[TARGET].mean()),
            "validation": float(val_df[TARGET].mean()),
            "test_full": float(test_full_df[TARGET].mean()),
        },
        "best_full_by_aucpr": best_full.to_dict(),
        "outputs": {
            "results_by_region": str(out_results),
            "precision_at_k": str(out_pk),
            "high_precision_thresholds": str(out_hp),
            "transaction_exposure": str(out_exposure_tx) if exposure_full is not None else None,
            "report": str(out_report),
        },
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nSaved files:")
    print(out_results)
    print(out_pk)
    print(out_hp)
    if exposure_full is not None:
        print(out_exposure_tx)
    print(out_report)

    print("\nBest full by AUC-PR:")
    print(best_full)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost, LightGBM, and CatBoost under strict temporal splits."
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the processed learning dataset CSV.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional path to the SQLite database used for structural exposure analysis.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/three_ml_evaluation",
        help="Directory where output tables and reports will be saved.",
    )
    parser.add_argument(
        "--skip-exposure",
        action="store_true",
        help="Skip structural exposure analysis even if --db-path is provided.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

