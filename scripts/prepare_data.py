# ============================================================================
# scripts/prepare_data.py
# ============================================================================
"""
Prepare and normalize raw prompt datasets into a single small CSV for the prefilter.
- Preserves existing id/source/meta if present.
- Adds text_norm for rule heuristics.
- Deduplicates, shuffles, caps to MAX_ROWS.
"""
from pathlib import Path
import pandas as pd
import uuid
import json
import sys

# --- CONFIG ---
RAW_CSV_PATH = Path("data/raw/test.csv")
OUTPUT_PATH = Path("data/processed/filtered_data.csv")
SAMPLE_PATH = Path("data/processed/filtered_sample.csv")
MAX_ROWS = 65000
DEFAULT_SOURCE = "huggingface"

# --- HELPERS ---
def fatal(msg: str):
    print("ERROR:", msg)
    sys.exit(1)

# --- MAIN ---
if not RAW_CSV_PATH.exists():
    fatal(f"Raw file not found: {RAW_CSV_PATH.resolve()}")

print(f"Loading raw data from {RAW_CSV_PATH}...")
df = pd.read_csv(RAW_CSV_PATH, encoding="utf-8", low_memory=False)

if "text" not in df.columns or "label" not in df.columns:
    fatal(f"Input CSV must contain at least 'text' and 'label' columns. Found: {list(df.columns)}")

cols_to_keep = []
for c in ["id", "text", "label", "source", "meta"]:
    if c in df.columns:
        cols_to_keep.append(c)
cols_to_keep = ["text", "label"] + [c for c in cols_to_keep if c not in ("text","label")]
df = df[cols_to_keep]

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

try:
    df["label"] = df["label"].astype(int)
except Exception as e:
    fatal(f"Could not convert label column to int: {e}")

if not df["label"].isin([0, 1]).all():
    fatal("Label column must contain only 0 and 1 values")

if "id" not in df.columns:
    df.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(df))])
else:
    df["id"] = df["id"].astype(str)

if "source" not in df.columns:
    df["source"] = DEFAULT_SOURCE

if "meta" not in df.columns:
    df["meta"] = "{}"
else:
    def ensure_json(x):
        if pd.isna(x):
            return "{}"
        if isinstance(x, (dict, list)):
            return json.dumps(x)
        try:
            json.loads(x)
            return x
        except Exception:
            return json.dumps({"raw": str(x)})
    df["meta"] = df["meta"].apply(ensure_json)

before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows")

df["text_norm"] = df["text"].str.lower()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

if len(df) > MAX_ROWS:
    print(f"Capping dataset from {len(df)} -> {MAX_ROWS}")
    df = df.head(MAX_ROWS).copy()

cols_order = ["id", "text", "text_norm", "label", "source", "meta"]
cols_order = [c for c in cols_order if c in df.columns]
df = df[cols_order]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Saved processed dataset ({len(df)} rows) -> {OUTPUT_PATH}")

sample_n = min(200, len(df))
df.sample(sample_n, random_state=42).to_csv(SAMPLE_PATH, index=False, encoding="utf-8")
print(f"Saved a {sample_n}-row sample -> {SAMPLE_PATH}")

print("Label distribution:", df["label"].value_counts().to_dict())
print("Done.")
