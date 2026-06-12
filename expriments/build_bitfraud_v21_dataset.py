# SCRIPT 02 FINAL — BITFRAUD v21
# Compatible with:
# MAIN IMPROVEMENTS:
#   ✓ Compatible with merged v21 database


import os
import math
from pathlib import Path
import json
import sqlite3
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from collections import Counter
from sklearn.ensemble import IsolationForest


DRIVE_ROOT = os.getenv("BITFRAUD_ROOT", str(Path(__file__).resolve().parents[1]))

DB_PATH = (
    f"{DRIVE_ROOT}/data/extended/"
    f"all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
)

SCAM_CACHE = (
    f"{DRIVE_ROOT}/data/raw/scam_addresses_v21_frozen.csv"
)

if not os.path.exists(SCAM_CACHE):
    SCAM_CACHE = (
        f"{DRIVE_ROOT}/data/raw/scam_addresses_v20_frozen.csv"
    )

OUT_DIR = f"{DRIVE_ROOT}/data/processed_v21_final"

os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FULL = f"{OUT_DIR}/dataset_v21_full.csv"
OUTPUT_L = f"{OUT_DIR}/dataset_v21_learning.csv"
OUTPUT_H = f"{OUT_DIR}/dataset_v21_heuristic.csv"

OUTPUT_REPORT = f"{OUT_DIR}/v21_labeling_report.json"
OUTPUT_SPEAR = f"{OUT_DIR}/v21_L_label_spearman.csv"


SEED = 42

BURST_WINDOW = 120
IF_CONTAMINATION = 0.02
HEURISTIC_PCTLE = 99

TRAIN_SNAPS = ["D1", "D2", "D3"]
VAL_SNAPS = ["D4"]
TEST_SNAPS = ["D5", "D6"]

SNAP_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{OUT_DIR}/script02_v21.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


def split_addrs(s):

    if not isinstance(s, str) or not s:
        return []

    return [
        a.strip().lower()
        for a in s.split(",")
        if a.strip()
        and a.strip() not in ("None", "nan", "none")
    ]


def load_db(db_path):

    log.info("Loading SQLite database v21...")

    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)

    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")


    df_tx = pd.read_sql_query("""
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
    """, conn)


    df_in = pd.read_sql_query("""
        SELECT
            tx_hash,
            COUNT(DISTINCT input_address)
                AS input_addr_count_raw,

            GROUP_CONCAT(DISTINCT input_address)
                AS all_input_addrs

        FROM tx_inputs
        GROUP BY tx_hash
    """, conn)


    df_out = pd.read_sql_query("""
        SELECT
            tx_hash,

            COUNT(DISTINCT output_address)
                AS output_addr_count_raw,

            SUM(
                CASE
                    WHEN output_value < 546
                    THEN 1
                    ELSE 0
                END
            ) AS dust_output_count_raw,

            GROUP_CONCAT(DISTINCT output_address)
                AS all_output_addrs,

            GROUP_CONCAT(output_value)
                AS all_output_values,

            GROUP_CONCAT(script_type)
                AS all_script_types

        FROM tx_outputs
        GROUP BY tx_hash
    """, conn)

    conn.close()


    df = df_tx.merge(df_in, on="tx_hash", how="left")
    df = df.merge(df_out, on="tx_hash", how="left")


    for c in [
        "input_addr_count_raw",
        "output_addr_count_raw",
        "dust_output_count_raw"
    ]:
        df[c] = df[c].fillna(0)

    for c in [
        "all_input_addrs",
        "all_output_addrs",
        "all_output_values",
        "all_script_types"
    ]:
        df[c] = df[c].fillna("")

    if (
        "fee_value" not in df.columns
        or df["fee_value"].isna().all()
    ):
        df["fee_value"] = (
            df["input_value"] - df["output_value"]
        ).clip(lower=0)

    log.info(f"Loaded {len(df):,} transactions")

    log.info(
        "\n"
        + df["snapshot_id"]
            .value_counts()
            .sort_index()
            .to_string()
    )

    return df


def preprocess(df):

    n0 = len(df)

    df = df.drop_duplicates(
        subset=["tx_hash"]
    ).copy()

    df = df.dropna(
        subset=[
            "input_value",
            "output_value",
            "block_height",
            "timestamp",
            "snapshot_id"
        ]
    )

    df["fees"] = df["fee_value"].fillna(
        df["input_value"] - df["output_value"]
    )

    df = df[
        ~(
            (df["coinbase_flag"] == 0)
            & (df["input_value"] <= 0)
        )
    ].copy()

    df = df[
        ~(
            (df["coinbase_flag"] == 0)
            & (df["fees"] < 0)
        )
    ].copy()

    df["fees"] = df["fees"].clip(lower=0)

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.sort_values(
        ["timestamp", "block_height"]
    ).reset_index(drop=True)

    log.info(
        f"Preprocess: {n0:,} → {len(df):,}"
    )

    return df


def build_temporal_addr_sets(df):

    log.info(
        "Building strict temporal address sets..."
    )

    addr_history = {}
    cumulative = set()

    for snap in SNAP_ORDER:

        addr_history[snap] = frozenset(cumulative)

        snap_rows = df[
            df["snapshot_id"] == snap
        ]

        for _, row in snap_rows.iterrows():

            cumulative.update(
                split_addrs(row["all_input_addrs"])
                + split_addrs(row["all_output_addrs"])
            )

    log.info("Temporal sets built.")

    return addr_history


def compute_L(df, addr_history):

    log.info("Computing L features...")

    L = pd.DataFrame(index=df.index)

    L["tx_hash"] = df["tx_hash"]
    L["block_height"] = df["block_height"]
    L["timestamp"] = df["timestamp"]
    L["datetime"] = df["datetime"]
    L["snapshot_id"] = df["snapshot_id"]


    L["input_count"] = (
        df["input_count"]
        .fillna(0)
        .astype(int)
    )

    L["input_addr_count"] = (
        df["input_addr_count_raw"]
        .fillna(0)
        .astype(int)
    )

    L["coinbase_flag"] = (
        df["coinbase_flag"]
        .fillna(0)
        .astype(int)
    )

    L["has_witness"] = (
        df["has_witness"]
        .fillna(0)
        .astype(int)
    )

    script_order = {
        "unknown":0,
        "op_return":0,
        "p2pkh":1,
        "p2sh":2,
        "multisig":2,
        "v0_p2wpkh":3,
        "v0_p2wsh":4,
        "p2tr":5
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
        L["input_addr_count"]
        / L["input_count"].clip(lower=1)
    ).clip(0, 1)

    L["avg_input_value"] = np.where(
        df["input_count"] > 0,
        df["input_value"]
        / df["input_count"].clip(lower=1),
        0
    )

    L["total_input_scaled"] = np.log1p(
        df["input_value"].clip(lower=0)
    )

    L["tx_weight"] = (
        148 * df["input_count"].clip(lower=0)
        + 34 * df["output_count"].clip(lower=0)
        + 10
    ).clip(lower=1)

    L["fee_per_byte"] = (
        df["fees"]
        / L["tx_weight"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    L["fee_log"] = np.log1p(
        df["fees"].clip(lower=0)
    )

    L["fee_per_input"] = (
        df["fees"]
        / df["input_count"].clip(lower=1)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    L["fee_per_output"] = (
        df["fees"]
        / df["output_count"].clip(lower=1)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    p75 = L.groupby(
        df["snapshot_id"]
    )["fee_per_byte"].transform(
        lambda x: max(x.quantile(0.75), 1e-8)
    )

    L["fee_urgency_ratio"] = (
        L["fee_per_byte"] / p75
    ).clip(0, 100)


    L["output_count"] = (
        df["output_count"]
        .fillna(0)
        .astype(int)
    )

    L["fee_ratio"] = (
        df["fees"]
        / df["input_value"].clip(lower=1e-8)
    ).clip(0, 1).fillna(0)

    L["log_output_value"] = np.log1p(
        df["output_value"].clip(lower=0)
    )

    L["io_count_ratio"] = (
        df["output_count"]
        / df["input_count"].clip(lower=1)
    ).clip(0, 50).fillna(1)


    log.info(
        "Computing strict temporal address features..."
    )

    prev_seen_ratios = []
    prev_seen_counts = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="L temporal"
    ):

        snap = row["snapshot_id"]

        hist = addr_history.get(
            snap,
            frozenset()
        )

        addrs = (
            split_addrs(row["all_input_addrs"])
            + split_addrs(row["all_output_addrs"])
        )

        n = len(addrs)

        if n == 0 or not hist:
            prev_seen_ratios.append(0.0)
            prev_seen_counts.append(0)
            continue

        seen = sum(
            1 for a in addrs
            if a in hist
        )

        prev_seen_ratios.append(seen / n)
        prev_seen_counts.append(seen)

    L["prev_addr_seen_ratio"] = prev_seen_ratios
    L["prev_addr_seen_count"] = prev_seen_counts

    return L


def compute_H(df):

    log.info("Computing H features...")

    H = pd.DataFrame(index=df.index)

    H["tx_hash"] = df["tx_hash"]
    H["block_height"] = df["block_height"]
    H["timestamp"] = df["timestamp"]
    H["snapshot_id"] = df["snapshot_id"]


    H["dust_output_count"] = (
        df["dust_output_count_raw"]
        .fillna(0)
        .astype(int)
    )

    addr_reuse = []
    entropy_vals = []
    ratio_new = []

    known = set()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="H address features"
    ):

        ins = split_addrs(
            row["all_input_addrs"]
        )

        outs = split_addrs(
            row["all_output_addrs"]
        )

        all_ = ins + outs

        addr_reuse.append(
            len(set(ins) & set(outs))
        )

        if all_:

            cnt = Counter(all_)
            total = len(all_)

            ent = -sum(
                (c / total)
                * math.log2(c / total)
                for c in cnt.values()
            )

            entropy_vals.append(ent)

            ratio_new.append(
                sum(
                    1 for a in all_
                    if a not in known
                ) / total
            )

            known.update(all_)

        else:
            entropy_vals.append(0.0)
            ratio_new.append(0.0)

    H["address_reuse_count"] = addr_reuse
    H["addr_entropy"] = entropy_vals
    H["ratio_new_known_addr"] = ratio_new


    H["burst_activity_normalized"] = 0.0

    for snap, idx in H.groupby(
        "snapshot_id"
    ).groups.items():

        tmp = H.loc[idx].sort_values(
            "timestamp"
        )

        ts = tmp["timestamp"].values

        l = np.searchsorted(
            ts,
            ts - BURST_WINDOW,
            side="left"
        )

        r = np.searchsorted(
            ts,
            ts + BURST_WINDOW,
            side="right"
        )

        burst = r - l - 1

        H.loc[tmp.index,
              "burst_activity_normalized"] = np.clip(
            burst / max(np.percentile(burst, 95), 1),
            0,
            20
        )

    H["time_since_last_tx"] = (
        H.sort_values(
            ["block_height", "timestamp"]
        )
        .groupby("block_height")["timestamp"]
        .diff()
        .fillna(0)
        .clip(lower=0)
    )


    output_val_cv = []
    dominant_out_share = []
    same_out_ratio = []
    script_homogeneity = []

    block_tx_count = (
        df.groupby("block_height")["tx_hash"]
        .transform("count")
    )

    snap_median_block = (
        df.assign(
            block_tx_count_tmp=block_tx_count
        )
        .groupby("snapshot_id")[
            "block_tx_count_tmp"
        ]
        .transform("median")
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="H value/script"
    ):

        raw_vals = [
            v for v in str(
                row.get(
                    "all_output_values",
                    ""
                )
            ).split(",")

            if v.strip()
            .lstrip('-')
            .replace('.', '', 1)
            .isdigit()
        ]

        vals = [
            float(v)
            for v in raw_vals
            if float(v) > 0
        ]

        if len(vals) >= 2:

            mean_v = np.mean(vals)

            cv = (
                np.std(vals)
                / max(mean_v, 1e-8)
            )

            dom = (
                max(vals)
                / max(sum(vals), 1e-8)
            )

            mc = Counter(vals).most_common(1)[0][1]

            same = mc / len(vals)

        else:
            cv = 0.0
            dom = 1.0
            same = 1.0

        raw_scripts = [
            s.strip()
            for s in str(
                row.get(
                    "all_script_types",
                    ""
                )
            ).split(",")
            if s.strip()
        ]

        if raw_scripts:

            mc_s = Counter(
                raw_scripts
            ).most_common(1)[0][1]

            homo = mc_s / len(raw_scripts)

        else:
            homo = 0.0

        output_val_cv.append(cv)
        dominant_out_share.append(dom)
        same_out_ratio.append(same)
        script_homogeneity.append(homo)

    H["output_value_cv"] = output_val_cv
    H["dominant_output_share"] = dominant_out_share
    H["same_output_value_ratio"] = same_out_ratio
    H["script_output_homogeneity"] = script_homogeneity

    H["block_tx_density_norm"] = (
        block_tx_count
        / snap_median_block.clip(lower=1)
    ).clip(0, 20).fillna(1)

    H["addr_fanout_ratio"] = (
        df["output_addr_count_raw"].fillna(0)
        / df["input_addr_count_raw"]
            .clip(lower=1)
            .fillna(1)
    ).clip(0, 50)

    return H


def load_scam_lookup(path):

    log.info(f"Loading scam cache: {path}")

    df_s = pd.read_csv(path)

    df_s["address"] = (
        df_s["address"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df_s["report_datetime"] = pd.to_datetime(
        df_s["created_at"],
        utc=True,
        errors="coerce"
    )

    df_s = df_s.dropna(
        subset=["address", "report_datetime"]
    )

    TIER = {
        "tier0_ofac_sanctions":0,
        "tier1_chainabuse_btcblack":1,
        "tier2_chainabuse_verified":2,
        "tier3_btcblack_confirmed":3
    }

    lookup = {}

    for addr, g in df_s.groupby("address"):

        g = g.copy()

        g["tr"] = (
            g["confidence_tier"]
            .map(TIER)
            .fillna(99)
        )

        best = g.sort_values(
            ["report_datetime", "tr"]
        ).iloc[0]

        lookup[addr] = {
            "report_datetime":
                best["report_datetime"],

            "confidence_tier":
                best["confidence_tier"],

            "btcblack_verified":
                bool(
                    best.get(
                        "btcblack_verified",
                        False
                    )
                ),
        }

    log.info(
        f"Loaded {len(lookup):,} scam addresses"
    )

    return lookup


def compute_external_strict(df, scam_lookup):

    log.info(
        "Computing strict external labels..."
    )

    lv = []
    lv_retro = []
    fut_flag = []
    tiers = []
    counts = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="External strict labels"
    ):

        tx_dt = row["datetime"]

        addrs = list(set(
            split_addrs(row["all_input_addrs"])
            + split_addrs(row["all_output_addrs"])
        ))

        matched = [
            a for a in addrs
            if a in scam_lookup
        ]

        if not matched:

            lv.append(0)
            lv_retro.append(0)
            fut_flag.append(False)
            tiers.append("")
            counts.append(0)

            continue

        lv_retro.append(1)
        counts.append(len(matched))

        valid = [
            a for a in matched
            if scam_lookup[a]["report_datetime"]
            <= tx_dt
        ]

        if valid:
            lv.append(1)
            fut_flag.append(False)
            cand = valid
        else:
            lv.append(0)
            fut_flag.append(True)
            cand = matched

        TIER = {
            "tier0_ofac_sanctions":0,
            "tier1_chainabuse_btcblack":1,
            "tier2_chainabuse_verified":2,
            "tier3_btcblack_confirmed":3
        }

        best = sorted(
            [
                scam_lookup[a]["confidence_tier"]
                for a in cand
            ],
            key=lambda x: TIER.get(x, 99)
        )[0]

        tiers.append(best)

    return pd.DataFrame({
        "label_verified": lv,
        "label_verified_retro": lv_retro,
        "external_future_info_flag": fut_flag,
        "external_best_tier": tiers,
        "external_matched_addr_count": counts,
    }, index=df.index)


def compute_heuristic_label(df_H):

    log.info(
        "Computing heuristic score..."
    )

    H = df_H.copy()

    s = pd.Series(
        np.zeros(len(H)),
        index=H.index
    )

    def p(col, q):
        return H[col].quantile(q)

    s += (
        H["dust_output_count"]
        > p("dust_output_count", 0.90)
    ) * 2

    s += (
        H["address_reuse_count"]
        > p("address_reuse_count", 0.95)
    ) * 2

    s += (
        H["addr_entropy"]
        < p("addr_entropy", 0.05)
    ) * 1

    s += (
        H["burst_activity_normalized"]
        > p("burst_activity_normalized", 0.98)
    ) * 2

    s += (
        H["ratio_new_known_addr"]
        > p("ratio_new_known_addr", 0.90)
    ) * 1

    s += (
        (
            H["time_since_last_tx"]
            < p("time_since_last_tx", 0.02)
        )
        &
        (H["time_since_last_tx"] > 0)
    ) * 1

    s += (
        H["output_value_cv"]
        < p("output_value_cv", 0.05)
    ) * 2

    s += (
        H["same_output_value_ratio"]
        > p("same_output_value_ratio", 0.95)
    ) * 2

    s += (
        H["dominant_output_share"]
        > p("dominant_output_share", 0.95)
    ) * 1

    s += (
        H["script_output_homogeneity"]
        > p("script_output_homogeneity", 0.95)
    ) * 1

    s += (
        H["block_tx_density_norm"]
        > p("block_tx_density_norm", 0.98)
    ) * 2

    s += (
        H["addr_fanout_ratio"]
        > p("addr_fanout_ratio", 0.95)
    ) * 1

    q_thr = s.quantile(
        HEURISTIC_PCTLE / 100
    )

    label = (
        s >= q_thr
    ).astype(int)

    log.info(
        f"label_heuristic: "
        f"{label.sum():,} "
        f"({label.mean()*100:.3f}%)"
    )

    return s, label


def compute_if_label(df_H):

    log.info(
        "Computing Isolation Forest..."
    )

    H_COLS = [
        "dust_output_count",
        "address_reuse_count",
        "addr_entropy",
        "burst_activity_normalized",
        "time_since_last_tx",
        "ratio_new_known_addr",
        "output_value_cv",
        "dominant_output_share",
        "same_output_value_ratio",
        "script_output_homogeneity",
        "block_tx_density_norm",
        "addr_fanout_ratio",
    ]

    X = (
        df_H[H_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    train_mask = df_H["snapshot_id"].isin(
        TRAIN_SNAPS + VAL_SNAPS
    )

    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X[train_mask])

    pred = model.predict(X)

    label = pd.Series(
        (pred == -1).astype(int),
        index=df_H.index
    )

    log.info(
        f"label_anomaly: "
        f"{label.sum():,} "
        f"({label.mean()*100:.3f}%)"
    )

    return label


def compute_final_label(lv, lh, la):

    vote = (
        4 * lv
        + 2 * lh
        + 1 * la
    )

    final = (vote >= 4).astype(int)

    return pd.DataFrame({
        "label_verified": lv.astype(int),
        "label_heuristic": lh.astype(int),
        "label_anomaly": la.astype(int),
        "vote_score": vote.astype(int),
        "label_final": final.astype(int),
    })


def validate_HL(df_H, df_L, labels):

    log.info(
        "Validating H/L separation..."
    )

    L_COLS = [
        c for c in df_L.columns
        if c not in (
            "tx_hash",
            "block_height",
            "timestamp",
            "datetime",
            "snapshot_id"
        )
    ]

    H_COLS = [
        c for c in df_H.columns
        if c not in (
            "tx_hash",
            "block_height",
            "timestamp",
            "snapshot_id",
            "heuristic_score"
        )
    ]

    inter = sorted(
        set(H_COLS) & set(L_COLS)
    )

    assert len(inter) == 0, (
        f"H ∩ L violation: {inter}"
    )

    rows = []

    for c in L_COLS:

        r, p = stats.spearmanr(
            df_L[c]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0),

            labels["label_final"]
        )

        rows.append({
            "feature": c,
            "spearman_r": float(r),
            "abs_r": float(abs(r)),
            "p_value":
                float(p)
                if not np.isnan(p)
                else None
        })

    corr_df = pd.DataFrame(rows)

    corr_df = corr_df.sort_values(
        "abs_r",
        ascending=False
    )

    corr_df.to_csv(
        OUTPUT_SPEAR,
        index=False
    )

    report = {
        "H_intersection_L": inter,
        "max_abs_L_label_corr":
            float(corr_df["abs_r"].max()),

        "max_corr_feature":
            str(corr_df.iloc[0]["feature"]),
    }

    log.info(
        f"H ∩ L = {inter}"
    )

    log.info(
        f"Max |Spearman(L,label)| "
        f"= {report['max_abs_L_label_corr']:.4f}"
    )

    return report


def main():

    log.info("=" * 70)
    log.info("BITFRAUD v21 FINAL")
    log.info("=" * 70)


    df_raw = load_db(DB_PATH)

    df_clean = preprocess(df_raw)


    addr_hist = build_temporal_addr_sets(
        df_clean
    )


    df_L = compute_L(
        df_clean,
        addr_hist
    )

    df_H = compute_H(df_clean)


    scam_lookup = load_scam_lookup(
        SCAM_CACHE
    )

    ext_df = compute_external_strict(
        df_clean,
        scam_lookup
    )


    heuristic_score, lh = compute_heuristic_label(
        df_H
    )

    la = compute_if_label(df_H)


    labels = compute_final_label(
        lv=ext_df["label_verified"],
        lh=lh,
        la=la
    )


    df_H["heuristic_score"] = heuristic_score

    df_H = pd.concat(
        [df_H, ext_df],
        axis=1
    )

    for c in labels.columns:
        df_H[c] = labels[c]

    df_final = pd.concat(
        [df_L, ext_df],
        axis=1
    )

    for c in labels.columns:
        df_final[c] = labels[c]

    df_H = df_H.loc[
        :,
        ~df_H.columns.duplicated()
    ]

    df_final = df_final.loc[
        :,
        ~df_final.columns.duplicated()
    ]


    report = validate_HL(
        df_H,
        df_L,
        labels
    )

    report["external_label_summary"] = {
        "strict_verified":
            int(ext_df["label_verified"].sum()),

        "retrospective_matched":
            int(ext_df["label_verified_retro"].sum()),

        "future_info_excluded":
            int(
                ext_df[
                    "external_future_info_flag"
                ].sum()
            ),
    }

    report["vote_system"] = {
        "weights": {
            "verified": 4,
            "heuristic": 2,
            "anomaly": 1
        },
        "threshold": 4,
        "max_possible": 7,
    }

    report["label_distribution_by_snapshot"] = (
        df_final.groupby("snapshot_id")[
            "label_final"
        ]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .to_dict(orient="records")
    )


    with open(OUTPUT_REPORT, "w") as f:
        json.dump(
            report,
            f,
            indent=2,
            default=str
        )

    df_final.to_csv(
        OUTPUT_FULL,
        index=False
    )

    df_final.to_csv(
        OUTPUT_L,
        index=False
    )

    df_H.to_csv(
        OUTPUT_H,
        index=False
    )


    print("\n" + "=" * 80)
    print("BITFRAUD v21 FINAL DATASET CREATED")
    print("=" * 80)

    print("\nLabel distribution:")

    print(
        df_final.groupby("snapshot_id")[
            "label_final"
        ].agg(["count", "sum", "mean"])
    )

    print(
        f"\nMax |Spearman(L,label)| : "
        f"{report['max_abs_L_label_corr']:.4f}"
    )

    print("\nExternal label summary:")
    print(
        report["external_label_summary"]
    )

    print(f"\nFull dataset : {OUTPUT_FULL}")
    print(f"L dataset    : {OUTPUT_L}")
    print(f"H dataset    : {OUTPUT_H}")
    print(f"Report       : {OUTPUT_REPORT}")

    return df_final, df_H, report



if __name__ == "__main__":
    df_v21, df_H_v21, report_v21 = main()
