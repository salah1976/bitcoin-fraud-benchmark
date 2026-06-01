#!/usr/bin/env python3
"""
Build leakage-controlled Bitcoin fraud labels and learning features.

This script constructs the final benchmark files used in the paper:
1. Loads transaction, input, and output data from the SQLite snapshot database.
2. Computes the 15 leakage-controlled learning features.
3. Computes strict temporal external-intelligence labels.
4. Saves the full learning dataset and an audit report.

Important:
- Final labels are based only on temporally valid external intelligence.
- Raw addresses, intelligence metadata, confidence tiers, and report timestamps
  are not used as learning features.
- Heuristic/anomaly labels are intentionally excluded from the final pipeline.
"""

import argparse
import json
import logging
import math
import os
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


SNAP_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6"]

LEARNING_FEATURES = [
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

METADATA_COLUMNS = [
    "tx_hash",
    "block_height",
    "timestamp",
    "datetime",
    "snapshot_id",
]


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "build_labels_features.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def split_addrs(value):
    """Split a comma-separated address field into normalized addresses."""
    if not isinstance(value, str) or not value:
        return []

    return [
        a.strip().lower()
        for a in value.split(",")
        if a.strip() and a.strip().lower() not in {"none", "nan"}
    ]


def load_database(db_path: Path, log: logging.Logger) -> pd.DataFrame:
    """Load transaction, input-address, and output-address tables from SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    log.info("Loading SQLite database: %s", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")

    df_tx = pd.read_sql_query(
        """
        SELECT
            tx_hash,
            block_height,
            timestamp,
            input_value,
            output_value,
            fee_value,
            input_count,
            output_count,
            coinbase_flag,
            has_witness,
            script_type,
            snapshot_id,
            snapshot_role,
            snapshot_date
        FROM transactions
        ORDER BY timestamp, block_height
        """,
        conn,
    )

    df_in = pd.read_sql_query(
        """
        SELECT
            tx_hash,
            COUNT(DISTINCT input_address) AS input_addr_count_raw,
            GROUP_CONCAT(DISTINCT input_address) AS all_input_addrs
        FROM tx_inputs
        GROUP BY tx_hash
        """,
        conn,
    )

    df_out = pd.read_sql_query(
        """
        SELECT
            tx_hash,
            COUNT(DISTINCT output_address) AS output_addr_count_raw,
            SUM(CASE WHEN output_value < 546 THEN 1 ELSE 0 END) AS dust_output_count_raw,
            GROUP_CONCAT(DISTINCT output_address) AS all_output_addrs,
            GROUP_CONCAT(output_value) AS all_output_values,
            GROUP_CONCAT(script_type) AS all_script_types
        FROM tx_outputs
        GROUP BY tx_hash
        """,
        conn,
    )

    conn.close()

    df = df_tx.merge(df_in, on="tx_hash", how="left")
    df = df.merge(df_out, on="tx_hash", how="left")

    numeric_defaults = [
        "input_addr_count_raw",
        "output_addr_count_raw",
        "dust_output_count_raw",
    ]
    for col in numeric_defaults:
        df[col] = df[col].fillna(0)

    text_defaults = [
        "all_input_addrs",
        "all_output_addrs",
        "all_output_values",
        "all_script_types",
    ]
    for col in text_defaults:
        df[col] = df[col].fillna("")

    if "fee_value" not in df.columns or df["fee_value"].isna().all():
        df["fee_value"] = (df["input_value"] - df["output_value"]).clip(lower=0)

    log.info("Loaded %s transactions", f"{len(df):,}")
    log.info("\n%s", df["snapshot_id"].value_counts().sort_index().to_string())

    return df


def preprocess(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Clean invalid transactions and add datetime column."""
    n_before = len(df)

    df = df.drop_duplicates(subset=["tx_hash"]).copy()
    df = df.dropna(
        subset=[
            "input_value",
            "output_value",
            "block_height",
            "timestamp",
            "snapshot_id",
        ]
    ).copy()

    df["fees"] = df["fee_value"].fillna(df["input_value"] - df["output_value"])

    df = df[~((df["coinbase_flag"] == 0) & (df["input_value"] <= 0))].copy()
    df = df[~((df["coinbase_flag"] == 0) & (df["fees"] < 0))].copy()

    df["fees"] = df["fees"].clip(lower=0)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    df = df.sort_values(["timestamp", "block_height"]).reset_index(drop=True)

    log.info("Preprocess: %s -> %s", f"{n_before:,}", f"{len(df):,}")
    return df


def build_temporal_addr_sets(df: pd.DataFrame, log: logging.Logger):
    """
    Build address-history sets for each snapshot.

    For snapshot Dk, only addresses observed in earlier snapshots are included.
    """
    log.info("Building strict temporal address-history sets...")

    addr_history = {}
    cumulative = set()

    for snap in SNAP_ORDER:
        addr_history[snap] = frozenset(cumulative)

        snap_rows = df[df["snapshot_id"] == snap]
        for _, row in snap_rows.iterrows():
            cumulative.update(
                split_addrs(row["all_input_addrs"])
                + split_addrs(row["all_output_addrs"])
            )

        log.info(
            "%s: historical addresses before snapshot = %s",
            snap,
            f"{len(addr_history[snap]):,}",
        )

    return addr_history


def compute_learning_features(df: pd.DataFrame, addr_history, log: logging.Logger):
    """Compute leakage-controlled learning features."""
    log.info("Computing learning features...")

    L = pd.DataFrame(index=df.index)

    L["tx_hash"] = df["tx_hash"]
    L["block_height"] = df["block_height"]
    L["timestamp"] = df["timestamp"]
    L["datetime"] = df["datetime"]
    L["snapshot_id"] = df["snapshot_id"]

    L["input_count"] = df["input_count"].fillna(0).astype(int)
    L["output_count"] = df["output_count"].fillna(0).astype(int)
    L["input_addr_count"] = df["input_addr_count_raw"].fillna(0).astype(int)
    L["coinbase_flag"] = df["coinbase_flag"].fillna(0).astype(int)
    L["has_witness"] = df["has_witness"].fillna(0).astype(int)

    script_order = {
        "unknown": 0,
        "op_return": 0,
        "p2pkh": 1,
        "p2sh": 2,
        "multisig": 2,
        "v0_p2wpkh": 3,
        "v0_p2wsh": 4,
        "p2tr": 5,
    }

    L["script_type_encoded"] = (
        df["script_type"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .map(script_order)
        .fillna(0)
        .astype(int)
    )

    L["input_addr_concentration"] = (
        L["input_addr_count"] / L["input_count"].clip(lower=1)
    ).clip(0, 1)

    L["avg_input_value"] = np.where(
        df["input_count"] > 0,
        df["input_value"] / df["input_count"].clip(lower=1),
        0,
    )

    L["total_input_scaled"] = np.log1p(df["input_value"].clip(lower=0))

    L["tx_weight"] = (
        148 * df["input_count"].clip(lower=0)
        + 34 * df["output_count"].clip(lower=0)
        + 10
    ).clip(lower=1)

    L["fee_ratio"] = (
        df["fees"] / df["input_value"].clip(lower=1e-8)
    ).clip(0, 1).fillna(0)

    L["log_output_value"] = np.log1p(df["output_value"].clip(lower=0))

    L["io_count_ratio"] = (
        df["output_count"] / df["input_count"].clip(lower=1)
    ).clip(0, 50).fillna(1)

    prev_seen_ratios = []
    prev_seen_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Temporal memory"):
        snap = row["snapshot_id"]
        hist = addr_history.get(snap, frozenset())

        addrs = split_addrs(row["all_input_addrs"]) + split_addrs(
            row["all_output_addrs"]
        )

        n_addrs = len(addrs)

        if n_addrs == 0 or not hist:
            prev_seen_ratios.append(0.0)
            prev_seen_counts.append(0)
            continue

        seen = sum(1 for addr in addrs if addr in hist)
        prev_seen_ratios.append(seen / n_addrs)
        prev_seen_counts.append(seen)

    L["prev_addr_seen_ratio"] = prev_seen_ratios
    L["prev_addr_seen_count"] = prev_seen_counts

    return L


def load_scam_lookup(path: Path, log: logging.Logger):
    """Load frozen external-intelligence cache and keep earliest valid report per address."""
    if not path.exists():
        raise FileNotFoundError(f"Scam cache not found: {path}")

    log.info("Loading external-intelligence cache: %s", path)
    df_s = pd.read_csv(path)

    required = {"address", "created_at", "confidence_tier"}
    missing = required - set(df_s.columns)
    if missing:
        raise ValueError(f"Missing columns in scam cache: {sorted(missing)}")

    df_s["address"] = df_s["address"].astype(str).str.strip().str.lower()
    df_s["report_datetime"] = pd.to_datetime(
        df_s["created_at"], utc=True, errors="coerce"
    )
    df_s = df_s.dropna(subset=["address", "report_datetime"])

    tier_rank = {
        "tier0_ofac_sanctions": 0,
        "tier1_chainabuse_btcblack": 1,
        "tier2_chainabuse_verified": 2,
        "tier3_btcblack_confirmed": 3,
    }

    lookup = {}

    for addr, group in df_s.groupby("address"):
        group = group.copy()
        group["tier_rank"] = group["confidence_tier"].map(tier_rank).fillna(99)
        best = group.sort_values(["report_datetime", "tier_rank"]).iloc[0]

        lookup[addr] = {
            "report_datetime": best["report_datetime"],
            "confidence_tier": best["confidence_tier"],
        }

    log.info("Loaded %s scam addresses", f"{len(lookup):,}")
    return lookup


def compute_external_labels(df: pd.DataFrame, scam_lookup, log: logging.Logger):
    """
    Compute strict temporal external-intelligence labels.

    label_verified = 1 only when at least one transaction address has an
    external report timestamped before or at the transaction time.
    """
    log.info("Computing strict temporal external-intelligence labels...")

    lv = []
    lv_retro = []
    future_flag = []
    best_tiers = []
    matched_counts = []

    tier_rank = {
        "tier0_ofac_sanctions": 0,
        "tier1_chainabuse_btcblack": 1,
        "tier2_chainabuse_verified": 2,
        "tier3_btcblack_confirmed": 3,
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="External labels"):
        tx_dt = row["datetime"]

        addrs = list(
            set(
                split_addrs(row["all_input_addrs"])
                + split_addrs(row["all_output_addrs"])
            )
        )

        matched = [addr for addr in addrs if addr in scam_lookup]

        if not matched:
            lv.append(0)
            lv_retro.append(0)
            future_flag.append(False)
            best_tiers.append("")
            matched_counts.append(0)
            continue

        lv_retro.append(1)
        matched_counts.append(len(matched))

        valid = [
            addr
            for addr in matched
            if scam_lookup[addr]["report_datetime"] <= tx_dt
        ]

        if valid:
            lv.append(1)
            future_flag.append(False)
            candidates = valid
        else:
            lv.append(0)
            future_flag.append(True)
            candidates = matched

        best_tier = sorted(
            [scam_lookup[addr]["confidence_tier"] for addr in candidates],
            key=lambda tier: tier_rank.get(tier, 99),
        )[0]
        best_tiers.append(best_tier)

    return pd.DataFrame(
        {
            "label_verified": lv,
            "label_verified_retro": lv_retro,
            "external_future_info_flag": future_flag,
            "external_best_tier": best_tiers,
            "external_matched_addr_count": matched_counts,
            "label_final": lv,
        },
        index=df.index,
    )


def validate_learning_features(
    df_learning: pd.DataFrame, labels: pd.Series, output_dir: Path, log: logging.Logger
):
    """Audit direct leakage and feature-label association."""
    forbidden_terms = [
        "address",
        "addr",
        "tier",
        "report",
        "external",
        "label",
        "scam",
        "btcblack",
        "chainabuse",
    ]

    # addr appears in legitimate temporal memory feature names, so only metadata is forbidden.
    forbidden_exact = {
        "external_best_tier",
        "external_matched_addr_count",
        "external_future_info_flag",
        "label_verified",
        "label_verified_retro",
        "label_final",
    }

    learning_cols = [c for c in LEARNING_FEATURES if c in df_learning.columns]
    forbidden_present = sorted(set(df_learning.columns) & forbidden_exact)

    rows = []
    for col in learning_cols:
        values = (
            df_learning[col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        r, p = stats.spearmanr(values, labels)

        rows.append(
            {
                "feature": col,
                "spearman_r": None if np.isnan(r) else float(r),
                "abs_r": None if np.isnan(r) else float(abs(r)),
                "p_value": None if np.isnan(p) else float(p),
            }
        )

    corr_df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    corr_path = output_dir / "learning_feature_label_spearman.csv"
    corr_df.to_csv(corr_path, index=False)

    report = {
        "n_learning_features": len(learning_cols),
        "learning_features": learning_cols,
        "forbidden_metadata_present_in_learning_dataset": forbidden_present,
        "spearman_csv": str(corr_path),
        "max_abs_learning_feature_label_corr": float(corr_df["abs_r"].max()),
        "max_corr_feature": str(corr_df.iloc[0]["feature"]),
    }

    log.info("Max |Spearman(feature,label)| = %.4f", report["max_abs_learning_feature_label_corr"])
    return report


def build_dataset(args):
    output_dir = Path(args.output_dir)
    log = setup_logging(output_dir)

    db_path = Path(args.db_path)
    scam_cache = Path(args.scam_cache)

    df_raw = load_database(db_path, log)
    df_clean = preprocess(df_raw, log)

    addr_history = build_temporal_addr_sets(df_clean, log)
    df_learning = compute_learning_features(df_clean, addr_history, log)

    scam_lookup = load_scam_lookup(scam_cache, log)
    labels = compute_external_labels(df_clean, scam_lookup, log)

    # Full output contains labels and audit metadata.
    df_full = pd.concat([df_learning, labels], axis=1)
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]

    # Learning output contains only metadata, 15 features, and final label.
    df_model = df_full[METADATA_COLUMNS + LEARNING_FEATURES + ["label_final"]].copy()

    report = validate_learning_features(
        df_model,
        df_model["label_final"],
        output_dir,
        log,
    )

    report["external_label_summary"] = {
        "strict_verified": int(labels["label_verified"].sum()),
        "retrospective_matched": int(labels["label_verified_retro"].sum()),
        "future_info_excluded": int(labels["external_future_info_flag"].sum()),
    }

    report["label_distribution_by_snapshot"] = (
        df_model.groupby("snapshot_id")["label_final"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .to_dict(orient="records")
    )

    output_full = output_dir / "dataset_full_with_label_audit.csv"
    output_learning = output_dir / "dataset_learning_15features.csv"
    output_report = output_dir / "label_feature_report.json"

    df_full.to_csv(output_full, index=False)
    df_model.to_csv(output_learning, index=False)

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DATASET CREATED")
    print("=" * 80)
    print("\nLabel distribution:")
    print(df_model.groupby("snapshot_id")["label_final"].agg(["count", "sum", "mean"]))
    print("\nExternal label summary:")
    print(report["external_label_summary"])
    print(f"\nFull dataset     : {output_full}")
    print(f"Learning dataset : {output_learning}")
    print(f"Report           : {output_report}")

    return df_model, report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build strict temporal labels and 15 leakage-controlled learning features."
    )

    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the SQLite database containing transactions, tx_inputs, and tx_outputs.",
    )

    parser.add_argument(
        "--scam-cache",
        required=True,
        help="Path to the frozen external-intelligence CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where processed datasets and reports will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())

