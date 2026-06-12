# SCRIPT 01 — Bitcoin Data Collection via Mempool API


import os
import time
import sqlite3
import logging
import requests
from pathlib import Path
from tqdm import tqdm


DRIVE_ROOT = os.getenv("BITFRAUD_ROOT", str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = f"{DRIVE_ROOT}/data/raw"
LOG_DIR = f"{DRIVE_ROOT}/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


SNAPSHOTS = {


    "D1": {
        "date": "2024-12",
        "start_block": 875000,
        "role": "train"
    },

    "D4": {
        "date": "2025-02",
        "start_block": 882200,
        "role": "train"
    },

    "D6": {
        "date": "2025-03",
        "start_block": 888000,
        "role": "train"
    },


    "D7": {
        "date": "2025-05",
        "start_block": 896500,
        "role": "validation"
    },

    "D8": {
        "date": "2025-07",
        "start_block": 905500,
        "role": "test_forward"
    },

    "D9": {
        "date": "2025-09",
        "start_block": 914500,
        "role": "test_final"
    },
}


BLOCKS_PER_SNAPSHOT = 100

API_BASE = "https://mempool.space/api"

DELAY_BETWEEN_REQUESTS = 0.20
MAX_RETRIES = 4


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/collection.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS blocks (
    id_block        INTEGER PRIMARY KEY AUTOINCREMENT,
    block_height    INTEGER NOT NULL UNIQUE,
    block_hash      TEXT NOT NULL,
    timestamp       INTEGER NOT NULL,
    tx_count        INTEGER,
    snapshot_id     TEXT NOT NULL,
    snapshot_role   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id_tx           INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_hash         TEXT NOT NULL UNIQUE,
    block_height    INTEGER NOT NULL,
    timestamp       INTEGER NOT NULL,
    input_value     REAL,
    output_value    REAL,
    input_count     INTEGER,
    output_count    INTEGER,
    coinbase_flag   INTEGER DEFAULT 0,
    has_witness     INTEGER DEFAULT 0,
    script_type     TEXT,
    snapshot_id     TEXT NOT NULL,
    FOREIGN KEY (block_height) REFERENCES blocks(block_height)
);

CREATE TABLE IF NOT EXISTS tx_inputs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_hash         TEXT NOT NULL,
    input_address   TEXT,
    input_value     REAL,
    FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash)
);

CREATE TABLE IF NOT EXISTS tx_outputs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_hash         TEXT NOT NULL,
    output_address  TEXT,
    output_value    REAL,
    script_type     TEXT,
    FOREIGN KEY (tx_hash) REFERENCES transactions(tx_hash)
);

CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_height);
CREATE INDEX IF NOT EXISTS idx_tx_hash  ON transactions(tx_hash);
CREATE INDEX IF NOT EXISTS idx_in_hash  ON tx_inputs(tx_hash);
CREATE INDEX IF NOT EXISTS idx_out_hash ON tx_outputs(tx_hash);
CREATE INDEX IF NOT EXISTS idx_tx_snap  ON transactions(snapshot_id);
"""


def api_get(endpoint):
    url = f"{API_BASE}/{endpoint}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "").lower()

                if "text/plain" in content_type:
                    return r.text.strip()

                try:
                    return r.json()
                except Exception:
                    return r.text.strip()

            elif r.status_code == 404:
                return None

            elif r.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"Rate limit 429. Attente {wait}s")
                time.sleep(wait)

            else:
                log.warning(f"HTTP {r.status_code} pour {url}")
                time.sleep(2)

        except requests.RequestException as e:
            log.warning(f"Erreur réseau tentative {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(2 ** attempt)

    log.error(f"Échec après {MAX_RETRIES} tentatives : {url}")
    return None


def get_block_hash(height):
    result = api_get(f"block-height/{height}")

    if isinstance(result, str) and len(result) > 20:
        return result.strip()

    return None


def get_block_info(height):
    block_hash = get_block_hash(height)

    if not block_hash:
        return None

    time.sleep(DELAY_BETWEEN_REQUESTS)

    block_info = api_get(f"block/{block_hash}")

    if not isinstance(block_info, dict):
        return None

    return {
        "block_height": height,
        "block_hash": block_hash,
        "timestamp": block_info.get("timestamp", 0),
        "tx_count": block_info.get("tx_count", 0),
    }


def get_block_transactions_from_hash(block_hash):
    all_txs = []
    start_index = 0

    while True:
        txs = api_get(f"block/{block_hash}/txs/{start_index}")

        if txs is None:
            break

        if not isinstance(txs, list) or len(txs) == 0:
            break

        all_txs.extend(txs)

        if len(txs) < 25:
            break

        start_index += 25
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return all_txs


def parse_transaction(tx, block_height, block_timestamp, snapshot_id):
    tx_hash = tx.get("txid", "")

    vin = tx.get("vin", [])
    vout = tx.get("vout", [])

    is_coinbase = 0
    if vin and vin[0].get("is_coinbase", False):
        is_coinbase = 1

    has_witness = 0
    for inp in vin:
        if inp.get("witness"):
            has_witness = 1
            break

    input_records = []
    total_input = 0.0

    for inp in vin:
        if is_coinbase:
            continue

        prevout = inp.get("prevout") or {}
        val = prevout.get("value", 0) or 0
        addr = prevout.get("scriptpubkey_address", None)

        total_input += val

        input_records.append({
            "tx_hash": tx_hash,
            "input_address": addr,
            "input_value": val
        })

    output_records = []
    total_output = 0.0
    script_types = []

    for out in vout:
        val = out.get("value", 0) or 0
        addr = out.get("scriptpubkey_address", None)
        stype = out.get("scriptpubkey_type", "unknown")

        total_output += val
        script_types.append(stype)

        output_records.append({
            "tx_hash": tx_hash,
            "output_address": addr,
            "output_value": val,
            "script_type": stype
        })

    dominant_script = "unknown"
    if script_types:
        dominant_script = max(set(script_types), key=script_types.count)

    tx_record = {
        "tx_hash": tx_hash,
        "block_height": block_height,
        "timestamp": block_timestamp,
        "input_value": total_input,
        "output_value": total_output,
        "input_count": len(vin) if not is_coinbase else 0,
        "output_count": len(vout),
        "coinbase_flag": is_coinbase,
        "has_witness": has_witness,
        "script_type": dominant_script,
        "snapshot_id": snapshot_id,
    }

    return {
        "transaction": tx_record,
        "inputs": input_records,
        "outputs": output_records
    }


def init_database(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    return conn


def block_already_collected(conn, height):
    r = conn.execute("""
        SELECT COUNT(*)
        FROM blocks b
        JOIN transactions t ON b.block_height = t.block_height
        WHERE b.block_height = ?
    """, (height,)).fetchone()[0]

    return r > 0


def insert_block(conn, block, snapshot_id, role):
    conn.execute("""
        INSERT OR IGNORE INTO blocks
        (block_height, block_hash, timestamp, tx_count, snapshot_id, snapshot_role)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        block["block_height"],
        block["block_hash"],
        block["timestamp"],
        block["tx_count"],
        snapshot_id,
        role
    ))


def insert_transaction_batch(conn, tx_data_list):
    if not tx_data_list:
        return

    txs = [d["transaction"] for d in tx_data_list]
    inputs = [i for d in tx_data_list for i in d["inputs"]]
    outputs = [o for d in tx_data_list for o in d["outputs"]]

    conn.executemany("""
        INSERT OR IGNORE INTO transactions
        (tx_hash, block_height, timestamp, input_value, output_value,
         input_count, output_count, coinbase_flag, has_witness,
         script_type, snapshot_id)
        VALUES (:tx_hash, :block_height, :timestamp, :input_value,
                :output_value, :input_count, :output_count,
                :coinbase_flag, :has_witness, :script_type, :snapshot_id)
    """, txs)

    if inputs:
        conn.executemany("""
            INSERT INTO tx_inputs
            (tx_hash, input_address, input_value)
            VALUES (:tx_hash, :input_address, :input_value)
        """, inputs)

    if outputs:
        conn.executemany("""
            INSERT INTO tx_outputs
            (tx_hash, output_address, output_value, script_type)
            VALUES (:tx_hash, :output_address, :output_value, :script_type)
        """, outputs)


def collect_snapshot(snapshot_name, config, db_path):
    log.info("=" * 60)
    log.info(f"SNAPSHOT {snapshot_name} — {config['date']} — {config['role']}")
    log.info(
        f"Blocs : {config['start_block']} → "
        f"{config['start_block'] + BLOCKS_PER_SNAPSHOT - 1}"
    )
    log.info("=" * 60)

    conn = init_database(db_path)

    stats = {
        "blocks_new": 0,
        "blocks_skipped": 0,
        "transactions_new": 0,
        "errors": 0
    }

    start = config["start_block"]
    end = start + BLOCKS_PER_SNAPSHOT

    for height in tqdm(
        range(start, end),
        desc=f"Snapshot {snapshot_name}",
        unit="bloc"
    ):
        if block_already_collected(conn, height):
            stats["blocks_skipped"] += 1
            continue

        block_info = get_block_info(height)

        if not block_info:
            log.warning(f"Bloc {height} non récupéré — ignoré")
            stats["errors"] += 1
            continue

        raw_txs = get_block_transactions_from_hash(block_info["block_hash"])

        if not raw_txs:
            log.warning(f"Aucune transaction pour bloc {height}")
            stats["errors"] += 1
            continue

        batch = []

        for tx in raw_txs:
            try:
                parsed = parse_transaction(
                    tx=tx,
                    block_height=height,
                    block_timestamp=block_info["timestamp"],
                    snapshot_id=snapshot_name
                )

                if parsed["transaction"]["tx_hash"]:
                    batch.append(parsed)

            except Exception as e:
                log.debug(f"Erreur parsing tx: {e}")
                stats["errors"] += 1

        try:
            insert_block(conn, block_info, snapshot_name, config["role"])
            insert_transaction_batch(conn, batch)
            conn.commit()

            stats["blocks_new"] += 1
            stats["transactions_new"] += len(batch)

        except Exception as e:
            conn.rollback()
            stats["errors"] += 1
            log.error(f"Erreur insertion bloc {height}: {e}")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    total_blocks = conn.execute(
        "SELECT COUNT(*) FROM blocks"
    ).fetchone()[0]

    total_tx = conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

    log.info(f"Snapshot {snapshot_name} terminé")
    log.info(f"Blocs nouveaux          : {stats['blocks_new']}")
    log.info(f"Blocs déjà présents     : {stats['blocks_skipped']}")
    log.info(f"Transactions nouvelles  : {stats['transactions_new']}")
    log.info(f"Total blocs DB          : {total_blocks}")
    log.info(f"Total transactions DB   : {total_tx}")
    log.info(f"Erreurs                 : {stats['errors']}")

    conn.close()

    return stats


def consolidate_snapshots(output_db):
    log.info("Consolidation des snapshots...")

    if os.path.exists(output_db):
        os.remove(output_db)

    conn_out = init_database(output_db)

    for snap_name, config in SNAPSHOTS.items():
        db_path = os.path.join(
            OUTPUT_DIR,
            f"snapshot_{snap_name}_{config['date']}.db"
        )

        if not os.path.exists(db_path):
            log.warning(f"Snapshot absent : {db_path}")
            continue

        conn_in = sqlite3.connect(db_path)
        conn_in.row_factory = sqlite3.Row

        rows = conn_in.execute("SELECT * FROM blocks").fetchall()
        conn_out.executemany("""
            INSERT OR IGNORE INTO blocks
            (block_height, block_hash, timestamp, tx_count, snapshot_id, snapshot_role)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (
                r["block_height"],
                r["block_hash"],
                r["timestamp"],
                r["tx_count"],
                r["snapshot_id"],
                r["snapshot_role"]
            )
            for r in rows
        ])

        rows = conn_in.execute("SELECT * FROM transactions").fetchall()
        conn_out.executemany("""
            INSERT OR IGNORE INTO transactions
            (tx_hash, block_height, timestamp, input_value, output_value,
             input_count, output_count, coinbase_flag, has_witness,
             script_type, snapshot_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                r["tx_hash"],
                r["block_height"],
                r["timestamp"],
                r["input_value"],
                r["output_value"],
                r["input_count"],
                r["output_count"],
                r["coinbase_flag"],
                r["has_witness"],
                r["script_type"],
                r["snapshot_id"]
            )
            for r in rows
        ])

        rows = conn_in.execute("SELECT * FROM tx_inputs").fetchall()
        conn_out.executemany("""
            INSERT INTO tx_inputs
            (tx_hash, input_address, input_value)
            VALUES (?, ?, ?)
        """, [
            (
                r["tx_hash"],
                r["input_address"],
                r["input_value"]
            )
            for r in rows
        ])

        rows = conn_in.execute("SELECT * FROM tx_outputs").fetchall()
        conn_out.executemany("""
            INSERT INTO tx_outputs
            (tx_hash, output_address, output_value, script_type)
            VALUES (?, ?, ?, ?)
        """, [
            (
                r["tx_hash"],
                r["output_address"],
                r["output_value"],
                r["script_type"]
            )
            for r in rows
        ])

        conn_out.commit()
        conn_in.close()

        log.info(f"{snap_name} fusionné")

    b = conn_out.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
    t = conn_out.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    log.info(f"Base consolidée : {b} blocs, {t} transactions")
    log.info(f"Chemin : {output_db}")

    conn_out.close()


def integrity_check(db_path):
    conn = sqlite3.connect(db_path)

    checks = {}

    checks["blocks"] = conn.execute(
        "SELECT COUNT(*) FROM blocks"
    ).fetchone()[0]

    checks["transactions"] = conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

    checks["orphan_transactions"] = conn.execute("""
        SELECT COUNT(*)
        FROM transactions t
        LEFT JOIN blocks b ON t.block_height = b.block_height
        WHERE b.block_height IS NULL
    """).fetchone()[0]

    checks["negative_fees"] = conn.execute("""
        SELECT COUNT(*)
        FROM transactions
        WHERE coinbase_flag = 0
          AND input_value > 0
          AND (input_value - output_value) < 0
    """).fetchone()[0]

    rows = conn.execute("""
        SELECT snapshot_id, COUNT(*) as cnt
        FROM transactions
        GROUP BY snapshot_id
        ORDER BY snapshot_id
    """).fetchall()

    log.info("Distribution par snapshot :")

    for row in rows:
        log.info(f"{row[0]} : {row[1]:,} transactions")

    log.info("Contrôles d'intégrité :")

    for k, v in checks.items():
        log.info(f"{k}: {v}")

    conn.close()

    return checks


def show_saved_files():
    print("\nFichiers sauvegardés dans Drive :")
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"{f:40s} {size_mb:10.2f} MB")


def main():

    log.info("DÉMARRAGE COLLECTE BITCOIN MEMPOOL")
    log.info(f"Drive root : {DRIVE_ROOT}")
    log.info(f"Output dir : {OUTPUT_DIR}")
    log.info(f"Blocs par snapshot : {BLOCKS_PER_SNAPSHOT}")
    log.info(f"API : {API_BASE}")

    all_stats = {}


    TARGET_SNAPSHOTS = ["D7", "D8", "D9"]

    for snap_name in TARGET_SNAPSHOTS:

        config = SNAPSHOTS[snap_name]

        db_path = os.path.join(
            OUTPUT_DIR,
            f"snapshot_{snap_name}_{config['date']}.db"
        )

        stats = collect_snapshot(
            snapshot_name=snap_name,
            config=config,
            db_path=db_path
        )

        all_stats[snap_name] = stats

        log.info("Pause 5s avant snapshot suivant...")
        time.sleep(5)


    consolidated_db = os.path.join(
        OUTPUT_DIR,
        "all_snapshots_extended.db"
    )

    consolidate_snapshots(consolidated_db)

    integrity_check(consolidated_db)

    show_saved_files()

    log.info("=" * 60)
    log.info("COLLECTE TERMINÉE")
    log.info(f"Base consolidée : {consolidated_db}")
    log.info("=" * 60)

    return all_stats



if __name__ == "__main__":
    stats = main()
