# ============================================================================
# src/models/train.py
# ============================================================================
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

import sys
sys.path.append(str(ROOT))
from src.utils.logger import logger

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train():
    emb_path = ROOT / "data" / "processed" / "filtered_embeddings.npy"
    ids_path = ROOT / "data" / "processed" / "filtered_ids.npy"
    meta_path = ROOT / "data" / "processed" / "embedder_meta.json"

    missing = [p for p in [emb_path, ids_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run embedder first! Missing: {missing}")

    X = np.load(emb_path)
    ids = np.load(ids_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df = pd.read_csv(ROOT / "data" / "processed" / "filtered_data.csv")
    df = df.set_index('id').loc[ids].reset_index()

    y = df['label'].values
    logger.info(f"Loaded {len(X)} embeddings | dim={X.shape[1]} | model={meta['model']}")

    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        solver='saga'
    )
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, probas)
    threshold = thresholds[np.where(fpr <= 0.05)[0][-1]]
    logger.info(f"Threshold: {threshold:.4f}")

    # FIXED: Use consistent filename
    joblib.dump(clf, MODEL_DIR / "filtered_clf.joblib")
    (MODEL_DIR / "threshold.txt").write_text(str(float(threshold)))

    metadata = {
        "classifier": "LogisticRegression",
        "embedding_model": meta["model"],
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_test, probas)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "datetime": datetime.now().isoformat(),
    }
    json.dump(metadata, open(MODEL_DIR / "model_metadata.json", "w"), indent=2)

    logger.info(f"Model saved. ROC-AUC: {metadata['roc_auc']:.4f}")
    print(classification_report(y_test, (probas >= threshold).astype(int)))

if __name__ == "__main__":
    train()
