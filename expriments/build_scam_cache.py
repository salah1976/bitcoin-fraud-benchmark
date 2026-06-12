
# SCRIPT 11 IMPROVED — SCAM CACHE v21


import os, re, json, time, socket, requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone

DRIVE_ROOT   = os.getenv("BITFRAUD_ROOT", str(Path(__file__).resolve().parents[1]))
RAW_DIR      = f"{DRIVE_ROOT}/data/raw"
EXT_DIR      = f"{DRIVE_ROOT}/data/external_intelligence"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EXT_DIR, exist_ok=True)

GRAPHQL_URL  = "https://chainabuse.com/api/graphql-proxy"
HEADERS      = {"Content-Type": "application/json"}

START_DATE   = "2024-01-01"
END_DATE     = "2026-05-24"

FINAL_OUTPUT = f"{RAW_DIR}/scam_addresses_v21_frozen.csv"
RAW_OUTPUT   = f"{EXT_DIR}/chainabuse_raw_v21.csv"
REPORT_JSON  = f"{EXT_DIR}/scam_addresses_v21_report.json"

MAX_PAGES    = 300
REQ_DELAY    = 0.4
BTC_DELAY    = 0.03

BTC_REGEX = re.compile(r"^(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,90}$")

CATEGORIES = [
    "OTHER_BLACKMAIL", "RANSOMWARE", "SEXTORTION", "IMPERSONATION",
    "FAKE_RETURNS", "PHISHING", "ROMANCE", "FAKE_PROJECT",
    "PIGBUTCHERING", "RUG_PULL", "SIM_SWAP", "CONTRACT_EXPLOIT",
    "OTHER", "HACK", "SCAM", "FRAUD",
]


def is_valid_btc(addr):
    return bool(BTC_REGEX.match(addr.strip())) if isinstance(addr, str) else False

def parse_date(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def get_status(node):
    trusted = (node.get("reportedBy") or {}).get("trusted", False)
    checked = node.get("checked", False)
    return "trusted" if trusted else ("checked" if checked else "not_verified")

def btcblack_lookup(addr, timeout=2):
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        socket.gethostbyname(f"{addr}.bl.btcblack.it")
        return True
    except Exception:
        pass
    finally:
        socket.setdefaulttimeout(old)
    return False


def fetch_chainabuse_category(category, first=100):
    rows, after = [], None
    start_dt = pd.to_datetime(START_DATE, utc=True)

    for _ in range(MAX_PAGES):
        payload = {
            "operationName": "GetReports",
            "variables": {
                "input": {
                    "chains": ["BTC"],
                    "scamCategories": [category],
                    "orderBy": {"field": "CREATED_AT", "direction": "DESC"},
                },
                "first": first,
                "after": after,
            },
            "query": """
            query GetReports($input: ReportsInput, $first: Float, $after: String) {
              reports(input: $input, first: $first, after: $after) {
                pageInfo { hasNextPage endCursor }
                edges {
                  node {
                    checked createdAt
                    reportedBy { trusted }
                    scamCategory
                    addresses { address chain }
                    description
                  }
                }
              }
            }"""
        }
        try:
            r = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=40)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ChainAbuse error {category}: {e}")
            break

        edges  = data.get("data", {}).get("reports", {}).get("edges", [])
        if not edges:
            break

        stop = False
        for edge in edges:
            node = edge.get("node", {})
            dt   = parse_date(node.get("createdAt"))
            if pd.notna(dt) and dt < start_dt:
                stop = True
                continue
            status = get_status(node)
            for addr_obj in node.get("addresses", []):
                if addr_obj.get("chain") != "BTC":
                    continue
                addr = addr_obj.get("address", "").strip()
                if not is_valid_btc(addr):
                    continue
                rows.append({
                    "address": addr, "source": "chainabuse",
                    "category": node.get("scamCategory", category),
                    "status": status, "created_at": node.get("createdAt"),
                })
        if stop:
            break
        pi = data.get("data", {}).get("reports", {}).get("pageInfo", {})
        if pi.get("hasNextPage"):
            after = pi.get("endCursor")
            time.sleep(REQ_DELAY)
        else:
            break
    return rows

def collect_chainabuse():
    print("\n" + "="*60)
    print("SOURCE 1 — CHAINABUSE")
    print("="*60)
    all_rows = []
    for cat in CATEGORIES:
        rows = fetch_chainabuse_category(cat)
        print(f"  {cat:<25} {len(rows):>5} rows")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["address", "category", "created_at", "status"])
    print(f"Total ChainAbuse raw: {len(df):,}")
    return df


def fetch_ofac_btc():
    print("\n" + "="*60)
    print("SOURCE 2 — OFAC SDN (US Treasury Bitcoin)")
    print("="*60)
    rows = []
    url  = "https://www.treasury.gov/ofac/downloads/sdnlist.txt"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"  OFAC HTTP {r.status_code} — skipping")
            return pd.DataFrame()
        text = r.text
        btc_pattern = re.compile(
            r"Digital Currency Address\s*-\s*(?:XBT|BTC)\s+([13][a-zA-HJ-NP-Z0-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59})",
            re.IGNORECASE
        )
        matches = btc_pattern.findall(text)
        for addr in set(matches):
            if is_valid_btc(addr):
                rows.append({
                    "address": addr, "source": "ofac_sdn",
                    "category": "SANCTIONS",
                    "status": "trusted",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
        print(f"  OFAC BTC addresses found: {len(rows)}")
    except Exception as e:
        print(f"  OFAC fetch failed: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_cryptoscamdb():
    print("\n" + "="*60)
    print("SOURCE 3 — CryptoScamDB")
    print("="*60)
    rows = []
    urls = [
        "https://api.cryptoscamdb.org/v1/addresses",
        "https://raw.githubusercontent.com/CryptoScamDB/blacklist/master/data/urls.yaml",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            data = r.json() if "json" in r.headers.get("Content-Type", "") else {}
            entries = data.get("result", data.get("addresses", []))
            if isinstance(entries, dict):
                entries = list(entries.keys())
            for item in entries:
                addr = str(item).strip() if isinstance(item, str) else item.get("address", "")
                if is_valid_btc(addr):
                    rows.append({
                        "address": addr, "source": "cryptoscamdb",
                        "category": "SCAM",
                        "status": "checked",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })
            if rows:
                break
        except Exception as e:
            print(f"  CryptoScamDB {url}: {e}")
            continue
    print(f"  CryptoScamDB addresses found: {len(rows)}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_bitcoin_abuse():
    print("\n" + "="*60)
    print("SOURCE 4 — Bitcoin Abuse fallback (public report sample)")
    print("="*60)
    rows = []
    urls_to_try = [
        "https://www.bitcoinabuse.com/api/reports/check?address=",
    ]
    print("  Bitcoin Abuse requires API key — adding as placeholder")
    print("  → Get key at bitcoinabuse.com and add to BTCABUSE_KEY config")
    return pd.DataFrame()


def run_btcblack(df):
    print("\n" + "="*60)
    print("BTCBLACK VALIDATION")
    print("="*60)
    unique = sorted(df["address"].unique())
    print(f"Addresses to check: {len(unique):,}")
    results = {}
    for addr in tqdm(unique, desc="BTCBlack"):
        try:
            results[addr] = btcblack_lookup(addr)
        except Exception:
            results[addr] = False
        time.sleep(BTC_DELAY)
    df["btcblack_verified"] = df["address"].map(results).fillna(False)
    print(f"BTCBlack verified: {int(df['btcblack_verified'].sum()):,}")
    return df


def assign_tier(row):
    s, b = row["status"], bool(row["btcblack_verified"])
    if row["source"] == "ofac_sdn":
        return "tier0_ofac_sanctions"
    if s in ["trusted", "checked"] and b:
        return "tier1_chainabuse_btcblack"
    if s in ["trusted", "checked"]:
        return "tier2_chainabuse_verified"
    if s == "not_verified" and b:
        return "tier3_btcblack_confirmed"
    return "excluded"

TIER_RANK = {
    "tier0_ofac_sanctions":    0,
    "tier1_chainabuse_btcblack": 1,
    "tier2_chainabuse_verified": 2,
    "tier3_btcblack_confirmed":  3,
}

def best_tier(vals):
    return sorted(vals, key=lambda x: TIER_RANK.get(x, 99))[0]

def build_final_cache(df_all):
    df = df_all[
        (df_all["status"].isin(["trusted", "checked"])) |
        (df_all["source"] == "ofac_sdn") |
        ((df_all["status"] == "not_verified") & (df_all["btcblack_verified"]))
    ].copy()
    df["confidence_tier"] = df.apply(assign_tier, axis=1)
    df = df[df["confidence_tier"] != "excluded"]

    agg = (
        df.groupby("address").agg({
            "source":          lambda x: ";".join(sorted(set(map(str, x)))),
            "category":        lambda x: ";".join(sorted(set(map(str, x)))),
            "status":          lambda x: ";".join(sorted(set(map(str, x)))),
            "created_at":      "min",
            "btcblack_verified": "max",
            "confidence_tier": best_tier,
        }).reset_index()
    )
    agg["verified"] = True
    agg["date"]     = agg["created_at"]
    return agg


def main():
    dfs = []

    df_ca = collect_chainabuse()
    if not df_ca.empty:
        dfs.append(df_ca)

    df_ofac = fetch_ofac_btc()
    if not df_ofac.empty:
        dfs.append(df_ofac)

    df_cs = fetch_cryptoscamdb()
    if not df_cs.empty:
        dfs.append(df_cs)

    fetch_bitcoin_abuse()  # placeholder

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df_all.empty:
        raise RuntimeError("No records collected from any source")

    df_all = df_all.drop_duplicates(subset=["address", "source", "category"])

    df_ca_mask = df_all["source"] == "chainabuse"
    df_all_ca  = df_all[df_ca_mask].copy()
    df_all_ca["created_at_dt"] = pd.to_datetime(df_all_ca["created_at"], utc=True, errors="coerce")
    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt   = pd.to_datetime(END_DATE, utc=True) + pd.Timedelta(days=1)
    df_all_ca = df_all_ca[
        (df_all_ca["created_at_dt"] >= start_dt) &
        (df_all_ca["created_at_dt"] <= end_dt)
    ].drop(columns=["created_at_dt"])

    df_all = pd.concat([df_all_ca, df_all[~df_ca_mask]], ignore_index=True)
    df_all.to_csv(RAW_OUTPUT, index=False)
    print(f"\nRaw records (all sources): {len(df_all):,}")

    df_all = run_btcblack(df_all)
    df_final = build_final_cache(df_all)
    df_final.to_csv(FINAL_OUTPUT, index=False)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period_start": START_DATE, "period_end": END_DATE,
        "sources": df_all["source"].value_counts().to_dict(),
        "unique_addresses": int(len(df_final)),
        "confidence_distribution": df_final["confidence_tier"].value_counts().to_dict(),
        "btcblack_verified": int(df_final["btcblack_verified"].sum()),
        "output_csv": FINAL_OUTPUT,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("SCAM CACHE v21 DONE")
    print("="*60)
    print(f"Unique addresses : {len(df_final):,}")
    print(f"Confidence tiers :\n{df_final['confidence_tier'].value_counts()}")
    print(f"Saved : {FINAL_OUTPUT}")
    return df_final


if __name__ == "__main__":
    df_scam_v21 = main()
