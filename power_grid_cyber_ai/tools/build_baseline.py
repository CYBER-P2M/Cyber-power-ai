"""Build models/baseline_stats.json: per-feature mean/std of the NORMAL class.

    python tools/build_baseline.py
    python tools/build_baseline.py --dataset path/to/final_verified_multiclass.csv --label-col label

Assumption: baseline = all rows labelled Normal in the training dataset
(train/test split indices are not stored, so the split is not reproduced).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--normal-label", default="Normal")
    ap.add_argument("--out", default=str(ROOT / "models" / "baseline_stats.json"))
    a = ap.parse_args()

    meta_path = ROOT / "models" / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    dataset = Path(a.dataset or meta.get("dataset_file", ""))
    label_col = a.label_col or meta.get("target", "label")
    if not dataset.is_file():
        sys.exit(f"Dataset not found: {dataset}\nPass --dataset <csv path>.")
    order = json.loads((ROOT / "models" / "feature_order.json").read_text(encoding="utf-8"))

    df = pd.read_csv(dataset, low_memory=False)
    if label_col not in df.columns:
        sys.exit(f"Label column '{label_col}' not in dataset.")
    missing = [c for c in order if c not in df.columns]
    if missing:
        sys.exit(f"{len(missing)} features missing from dataset, e.g. {missing[:5]}")

    normal = df[df[label_col].astype(str).str.strip().str.lower() == a.normal_label.lower()]
    if normal.empty:
        sys.exit(f"No rows with {label_col} == '{a.normal_label}'. Found: {df[label_col].unique()[:10]}")

    X = normal[order].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    feats = {}
    for c in order:
        col = X[c].dropna()
        feats[c] = {"mean": float(col.mean()) if len(col) else 0.0,
                    "std": float(col.std(ddof=0)) if len(col) else 0.0}
    out = {"baseline_class": a.normal_label, "n_rows": int(len(normal)), "source": str(dataset), "features": feats}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    zero = sum(1 for v in feats.values() if v["std"] == 0)
    print(f"Baseline written: {a.out}\nNormal rows: {len(normal)} | features: {len(feats)} | zero-variance: {zero}")


if __name__ == "__main__":
    main()
